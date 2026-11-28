// Causal attention with a bias by distance and a sink for a batch of
// tokens, over 8-bit caches that hold a ring of the latest positions, on
// the GPU's matrix accelerators: `attention_mma.metal`, with what
// Inkling's attention adds to it, as `attention_rel.metal` adds it to the
// kernel a token at a time takes. A token attends over the positions no
// more than `window - 1` back, and the caches hold position `p` in row
// `p % ring`. The chunks of a dispatch start at `base`, the first position
// its first token attends over. A key `d` positions back has added to its
// score the dot product of the head's row of `rel` with row `d` of `bias`,
// if `bias` has one, by the threads that take the scores' exponentials.
// The softmax is taken over the keys and a sink that has no value, whose
// score is the head's of `sinks`: here where the dispatch has one chunk,
// and in `attention_rel_combine.metal` where it has more.
//
// As in `attention_mma.metal`: a threadgroup takes ROWS rows, a token's
// query head each, of one key/value head, over a chunk of `chunk`
// positions in rounds of ROUND. The queries come packed by
// `pack_queries.metal`. For each round the threads dequantize its keys
// into threadgroup memory, the accelerators multiply the queries into them
// for the rows' scores, four threads to a row mask the scores outside the
// row's window, add the bias, take their exponentials against the row's
// running maximum, and write them as half floats; then the threads
// dequantize the values, and the accelerators add the exponentials'
// products with them to the rows' weighted values.

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
    uint window;
    uint ring;
    uint n_rel;
    uint extent;
    uint base;
    uint batched;
    uint bidir;
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
#ifndef ROWS
#define ROWS 64
#endif
#ifndef THREADS
#define THREADS 256
#endif
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
// positions past `len` clamped to the last, which only masked scores meet,
// each from its row of the ring. When `ring >= ROUND`, row wrapping uses a
// compare-and-subtract from `ring_r` instead of integer modulo.
static Fetched fetch(const device char4* quants, const device half* scales, uint start, uint len,
                     uint kv_dim, uint base, uint lid, uint ring, uint ring_r, uint ring_last) {
    Fetched f;
    bool simple = ring >= ROUND;
    for (uint k = 0; k < FETCHED; k++) {
        uint i = lid + k * THREADS;
        uint p = i / QUADS;
        uint d = (i % QUADS) * 4;
        uint row;
        if (simple) {
            if (start + p < len) {
                row = ring_r + p;
                if (row >= ring) {
                    row -= ring;
                }
            } else {
                row = ring_last;
            }
        } else {
            row = min(start + p, len - 1) % ring;
        }
        uint e = row * kv_dim + base + d;
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
        *(threadgroup half4*)(kv + p * STRIDE + d) = half4(f.quads[k]) * f.scales[k];
    }
}

kernel void attention_rel_mma(
    device float* out [[buffer(0)]],
    device half* q [[buffer(1)]],
    const device char4* k_quants [[buffer(2)]],
    const device half* k_scales [[buffer(3)]],
    const device char4* v_quants [[buffer(4)]],
    const device half* v_scales [[buffer(5)]],
    device float* partials [[buffer(6)]],
    const device float4* rel [[buffer(7)]],
    const device float4* bias [[buffer(8)]],
    const device float* sinks [[buffer(9)]],
    constant Params& p [[buffer(10)]],
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
    // What each row's weighted values are divided by, with its sink, where
    // the dispatch has one chunk.
    threadgroup float row_o[ROWS];

    uint kvh = wg.y % p.n_kv_heads;
    uint c = wg.y / p.n_kv_heads;
    uint group = p.n_heads / p.n_kv_heads;
    uint kv_dim = p.n_kv_heads * DIM;
    uint base = kvh * DIM;
    uint rows = p.tokens * group;
    uint r0 = wg.x * ROWS;
    // The positions any row of the threadgroup attends to in the chunk.
    uint last_row = min(r0 + ROWS, rows) - 1;
    uint tg_len = p.bidir != 0 ? p.pos + 1 : p.pos + p.first + last_row / group + 1;
    // The first position its first row does.
    uint first_len = p.bidir != 0 ? p.pos + 1 : p.pos + p.first + r0 / group + 1;
    uint tg_lo = first_len > p.window ? first_len - p.window : 0;
    uint start = p.base + c * p.chunk;
    if (start >= tg_len || start + p.chunk <= tg_lo) {
        return;
    }
    uint end = min(start + p.chunk, tg_len);

    // The thread's row and its positions of each round, and the row's
    // running maximum and sum, which its four threads all keep.
    uint my = lid / PER_ROW;
    uint c0 = (lid % PER_ROW) * (ROUND / PER_ROW);
    uint mine_row = min(r0 + my, rows - 1);
    uint len = p.bidir != 0 ? p.pos + 1 : p.pos + p.first + mine_row / group + 1;
    uint lo = len > p.window ? len - p.window : 0;
    // The row's projection for the bias, up to sixteen wide.
    uint quads = p.n_rel / 4;
    uint head = (p.first + mine_row / group) * p.n_heads + kvh * group + mine_row % group;
    float4 projected[4];
    for (uint j = 0; j < 4; j++) {
        projected[j] = p.extent != 0 && j < quads ? rel[head * quads + j] : float4(0.0f);
    }
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
    // The head's queries, the dispatch's rows: a row past them is zeros,
    // and its results are not kept.
    auto qa = tensor<device half, dextents<int32_t, 2>, tensor_inline>(
        q + kvh * p.padded * DIM, dextents<int32_t, 2>(DIM, p.padded));
    auto qs = qa.slice<DIM, ROWS>(0, r0);
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
    uint ring_r = start % p.ring;
    uint ring_last = (tg_len - 1) % p.ring;
    Fetched keys = fetch(k_quants, k_scales, start, tg_len, kv_dim, base, lid, p.ring, ring_r, ring_last);
    for (uint r = start; r < end; r += ROUND) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
#if !(ABLATE & 1)
        put(kv, keys, lid);
#endif
        Fetched values = fetch(v_quants, v_scales, r, tg_len, kv_dim, base, lid, p.ring, ring_r, ring_last);
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
            uint at = r + c0 + i;
            bool valid = at < len && at >= lo;
            float score = -1e30f;
            if (valid) {
                score = sm[my * ROUND + c0 + i];
                // The bias for how far back the key is.
                uint distance = len - 1 - at;
                if (distance < p.extent) {
                    for (uint j = 0; j < quads; j++) {
                        score += dot(projected[j], bias[distance * quads + j]);
                    }
                }
            }
            sv[i] = score;
            mx = max(mx, sv[i]);
        }
        for (ushort s = 1; s < PER_ROW; s <<= 1) {
            mx = max(mx, simd_shuffle_xor(mx, s));
        }
        float m_new = max(m, mx);
        float factor = exp(m - m_new);
        float sum = 0.0f;
        for (uint i = 0; i < ROUND / PER_ROW; i++) {
            float e = sv[i] > -1e29f ? exp(sv[i] - m_new) : 0.0f;
            sum += e;
            pm[my * PSTRIDE + c0 + i] = half(e);
        }
        for (ushort s = 1; s < PER_ROW; s <<= 1) {
            sum += simd_shuffle_xor(sum, s);
        }
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
        uint next_ring_r = ring_r + ROUND;
        if (p.ring >= ROUND) {
            if (next_ring_r >= p.ring) {
                next_ring_r -= p.ring;
            }
        } else {
            next_ring_r = (r + ROUND) % p.ring;
        }
        if (r + ROUND < end) {
            keys = fetch(k_quants, k_scales, r + ROUND, tg_len, kv_dim, base, lid, p.ring, next_ring_r, ring_last);
        }
        ring_r = next_ring_r;
        threadgroup_barrier(mem_flags::mem_threadgroup);
#if !(ABLATE & 2)
        pv.run(ps, vs, ot);
#endif
    }
    if (lid % PER_ROW == 0) {
        row_l[my] = l;
        row_f[my] = m;
        if (p.bidir != 0) {
            row_o[my] = 1.0f / l;
        } else {
            // The sink is one more to the softmax: the sums are rescaled to
            // whichever is greater, the keys' maximum or the sink.
            float sink = sinks[head];
            float most = max(m, sink);
            float keys_scale = exp(m - most);
            row_o[my] = keys_scale / (l * keys_scale + exp(sink - most));
        }
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
                    out[(t * p.n_heads + h) * DIM + d] = ot[i] * row_o[rr];
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
