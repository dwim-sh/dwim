// Causal attention for a batch of tokens over 8-bit caches, on the GPU's
// matrix accelerators through Metal 4's tensor operations: what
// `attention_flash.metal` does on the SIMD groups' 8x8 matrices, for a
// GPU with the accelerators, with the same grid, parameters and results.
// A threadgroup takes ROWS rows, a token's query head each, of one
// key/value head, over a chunk of `chunk` positions, which the device
// picks for the depth, in rounds of ROUND. The queries come packed by
// `pack_queries.metal`, half floats scaled for the softmax, the head's
// rows one after another, which the accelerators read as they multiply.
// For each round the threads dequantize its keys into threadgroup memory,
// the accelerators multiply the queries into them for the rows' scores,
// which go to threadgroup memory, where four threads to a row mask the
// scores past the row's token, take their exponentials against the row's
// running maximum, rescaling the row's sum when it grows, and write them
// as half floats; then the threads dequantize the values, and the
// accelerators add the exponentials' products with them to the rows'
// weighted values, which they hold in a cooperative tensor, rescaled first
// by the rows' factors. The keys and values of a round are fetched into
// registers a round ahead, so that their latency hides behind the
// multiplies. Rounds of 16 positions leave threadgroup memory for a
// second threadgroup on a core, which rounds of 32 did not, and the
// multiplies take no fewer.

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

struct Params {
    uint n_heads;
    uint head_dim;
    uint n_kv_heads;
    uint pos;
    uint first;
    uint chunks;
    uint chunk;
    uint tokens;
    uint padded;
};

#ifndef DIM
#define DIM 256
#endif
// What to leave out, to time the rest, as bits: 1 leaves out dequantizing
// the keys and values, 2 the matrix multiplies, and 4 the rescaling.
#ifndef ABLATE
#define ABLATE 0
#endif
// Positions a round takes, rows a threadgroup takes, and its threads:
// four to a row.
#ifndef ROUND
#define ROUND 16
#endif
constant uint ROWS = 64;
constant uint THREADS = 256;
constant uint PER_ROW = THREADS / ROWS;
// Half floats a position of the dequantized keys or values takes, and of
// a row of the exponentials, padded so that the threads writing different
// positions hit different banks.
constant uint STRIDE = DIM + 8;
constant uint PSTRIDE = ROUND + 8;

// The quads of four 8-bit keys or values a thread takes of a round's
// positions, with their blocks' scales: fetched into registers a round
// ahead, so that their latency hides behind the round's multiplies, and
// then dequantized into threadgroup memory as half floats.
constant uint QUADS = DIM / 4;
constant uint FETCHED = ROUND * QUADS / THREADS;

struct Fetched {
    char4 quads[FETCHED];
    half scales[FETCHED];
};

// Fetches the thread's quads of the round's positions from `start`, the
// positions past `len` clamped to the last, which only masked scores meet.
static Fetched fetch(const device char4* quants, const device half* scales, uint start, uint len,
                     uint kv_dim, uint base, uint lid) {
    Fetched f;
    for (uint k = 0; k < FETCHED; k++) {
        uint i = lid + k * THREADS;
        uint p = i / QUADS;
        uint d = (i % QUADS) * 4;
        uint e = min(start + p, len - 1) * kv_dim + base + d;
        f.quads[k] = quants[e / 4];
        f.scales[k] = scales[e / 32];
    }
    return f;
}

// Dequantizes the thread's fetched quads into `kv`.
static void put(threadgroup half* kv, thread const Fetched& f, uint lid) {
    for (uint k = 0; k < FETCHED; k++) {
        uint i = lid + k * THREADS;
        uint p = i / QUADS;
        uint d = (i % QUADS) * 4;
        *(threadgroup half4*)(kv + p * STRIDE + d) = half4(float4(f.quads[k]) * float(f.scales[k]));
    }
}

kernel void attention_mma(
    device float* out [[buffer(0)]],
    device half* q [[buffer(1)]],
    const device char4* k_quants [[buffer(2)]],
    const device half* k_scales [[buffer(3)]],
    const device char4* v_quants [[buffer(4)]],
    const device half* v_scales [[buffer(5)]],
    device float* partials [[buffer(6)]],
    constant Params& p [[buffer(7)]],
    uint2 wg [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]])
{
    threadgroup half kv[ROUND * STRIDE];
    threadgroup float sm[ROWS * ROUND];
    threadgroup half pm[ROWS * PSTRIDE];
    // The factor each row's weighted values are rescaled by this round,
    // and at the end the sum of its exponentials.
    threadgroup float row_f[ROWS];
    threadgroup float row_l[ROWS];

    uint kvh = wg.y % p.n_kv_heads;
    uint c = wg.y / p.n_kv_heads;
    uint group = p.n_heads / p.n_kv_heads;
    uint kv_dim = p.n_kv_heads * DIM;
    uint base = kvh * DIM;
    uint rows = p.tokens * group;
    uint r0 = wg.x * ROWS;
    // The positions any row of the threadgroup attends to in the chunk.
    uint last_row = min(r0 + ROWS, rows) - 1;
    uint tg_len = p.pos + p.first + last_row / group + 1;
    uint start = c * p.chunk;
    if (start >= tg_len) {
        return;
    }
    uint end = min(start + p.chunk, tg_len);

    // The thread's row and its positions of each round, and the row's
    // running maximum and sum, which its four threads all keep.
    uint my = lid / PER_ROW;
    uint c0 = (lid % PER_ROW) * (ROUND / PER_ROW);
    uint len = p.pos + p.first + min(r0 + my, rows - 1) / group + 1;
    float m = -1e30f;
    float l = 0.0f;

    // Scores: the queries (ROWS x DIM) by the keys (ROUND x DIM)
    // transposed; weighted values: the exponentials (ROWS x ROUND) by the
    // values (ROUND x DIM), summed over the rounds.
    constexpr auto qk_desc = matmul2d_descriptor(
        ROWS, ROUND, DIM, false, true, false, matmul2d_descriptor::mode::multiply);
    constexpr auto pv_desc = matmul2d_descriptor(
        ROWS, DIM, ROUND, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<qk_desc, execution_simdgroups<THREADS / 32>> qk;
    matmul2d<pv_desc, execution_simdgroups<THREADS / 32>> pv;
    // The head's queries, the dispatch's rows from `first` on: a row past
    // the batch is zeros, and its results are not kept.
    auto qa = tensor<device half, dextents<int32_t, 2>, tensor_inline>(
        q + kvh * p.padded * DIM, dextents<int32_t, 2>(DIM, p.padded));
    auto qs = qa.slice<DIM, ROWS>(0, p.first * group + r0);
    auto ks = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        kv, dextents<int32_t, 2>(DIM, ROUND), array<int32_t, 2>{1, int32_t(STRIDE)});
    auto vs = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        kv, dextents<int32_t, 2>(DIM, ROUND), array<int32_t, 2>{1, int32_t(STRIDE)});
    auto ss = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(
        sm, dextents<int32_t, 2>(ROUND, ROWS));
    auto ps = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        pm, dextents<int32_t, 2>(ROUND, ROWS), array<int32_t, 2>{1, int32_t(PSTRIDE)});
    auto st = qk.get_destination_cooperative_tensor<decltype(qs), decltype(ks), float>();
    auto ot = pv.get_destination_cooperative_tensor<decltype(ps), decltype(vs), float>();
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < ot.get_capacity(); i++) {
        if (ot.is_valid_element(i)) {
            ot[i] = 0.0f;
        }
    }

    // The round's keys are fetched during the round before, and its values
    // during its scores.
    Fetched keys = fetch(k_quants, k_scales, start, tg_len, kv_dim, base, lid);
    for (uint r = start; r < end; r += ROUND) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
#if !(ABLATE & 1)
        put(kv, keys, lid);
#endif
        Fetched values = fetch(v_quants, v_scales, r, tg_len, kv_dim, base, lid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
#if !(ABLATE & 2)
        qk.run(qs, ks, st);
#endif
        st.store(ss);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // The thread's positions of its row, masked past its token,
        // against the row's new maximum.
        float sv[ROUND / PER_ROW];
        float mx = -1e30f;
        for (uint i = 0; i < ROUND / PER_ROW; i++) {
            bool valid = r + c0 + i < len;
            sv[i] = valid ? sm[my * ROUND + c0 + i] : -1e30f;
            mx = max(mx, sv[i]);
        }
        mx = max(mx, simd_shuffle_xor(mx, 1));
        mx = max(mx, simd_shuffle_xor(mx, 2));
        float m_new = max(m, mx);
        float factor = exp(m - m_new);
        float sum = 0.0f;
        for (uint i = 0; i < ROUND / PER_ROW; i++) {
            float e = r + c0 + i < len ? exp(sv[i] - m_new) : 0.0f;
            sum += e;
            pm[my * PSTRIDE + c0 + i] = half(e);
        }
        sum += simd_shuffle_xor(sum, 1);
        sum += simd_shuffle_xor(sum, 2);
        l = l * factor + sum;
        m = m_new;
        if (lid % PER_ROW == 0) {
            row_f[my] = factor;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // What the rows have summed, rescaled to the new maxima.
#if !(ABLATE & 4)
        #pragma clang loop unroll(full)
        for (uint16_t i = 0; i < ot.get_capacity(); i++) {
            if (ot.is_valid_element(i)) {
                ot[i] *= row_f[ot.get_multidimensional_index(i)[1]];
            }
        }
#endif
#if !(ABLATE & 1)
        put(kv, values, lid);
#endif
        if (r + ROUND < end) {
            keys = fetch(k_quants, k_scales, r + ROUND, tg_len, kv_dim, base, lid);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
#if !(ABLATE & 2)
        pv.run(ps, vs, ot);
#endif
    }
    if (lid % PER_ROW == 0) {
        row_l[my] = l;
        row_f[my] = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // The rows' weighted values, divided by the exponentials' sums where
    // this is the only chunk, and otherwise as partials for
    // `attention_combine.metal` with the row's maximum and sum.
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < ot.get_capacity(); i++) {
        if (ot.is_valid_element(i)) {
            auto ids = ot.get_multidimensional_index(i);
            uint rr = ids[1];
            uint d = ids[0];
            uint orow = r0 + rr;
            if (orow < rows) {
                uint t = p.first + orow / group;
                uint h = kvh * group + orow % group;
                if (p.chunks == 1) {
                    out[(t * p.n_heads + h) * DIM + d] = ot[i] / row_l[rr];
                } else {
                    uint pb = (((t - p.first) * p.n_heads + h) * p.chunks + c) * (DIM + 2);
                    if (d == 0) {
                        partials[pb] = row_f[rr];
                        partials[pb + 1] = row_l[rr];
                    }
                    partials[pb + 2 + d] = ot[i];
                }
            }
        }
    }
}
