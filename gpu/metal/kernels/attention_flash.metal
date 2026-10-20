// Causal attention for a batch of tokens over 8-bit caches, with a
// half-precision scale per block of 32, on the SIMD groups' 8x8 matrix
// units. The query heads that share a key/value head attend over the same
// keys and values, so a key/value head's queries for every token of the
// batch are taken as rows, the token's heads one after another, and a
// threadgroup takes 32 of them, eight to each of its four SIMD groups, over
// a chunk of CHUNK positions. Each SIMD group keeps its rows' queries in
// registers as half floats, scaled for the softmax. For each 32 positions
// of the chunk, the threads dequantize their keys into threadgroup memory
// as half floats, and each SIMD group multiplies its queries into them for
// the rows' scores, masks the positions past each row's token, and takes
// the scores' exponentials against the rows' running maxima, rescaling
// what the rows have summed when a maximum grows; then the threads
// dequantize the values over the keys, and each SIMD group adds the
// exponentials' products with them to its rows' sums. The chunk's maximum,
// sum of exponentials, and weighted values of each row go to the partials
// buffer for `attention_combine.metal` to merge, or, when there is only one
// chunk, straight to the output. A dispatch covers the tokens from `first`
// on, `tokens` of them, with `chunks` chunks for each, the rows of 32 across
// its grid and the key/value heads of each chunk down it.

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint n_heads;
    uint head_dim;
    uint n_kv_heads;
    uint pos;
    uint first;
    uint chunks;
    uint chunk;
    uint tokens;
};

#ifndef DIM
#define DIM 256
#endif
// Positions a threadgroup takes, and a round of them.
constant uint CHUNK = 512;
constant uint ROUND = 32;
// Rows a SIMD group takes, and a threadgroup.
constant uint ROWS = 8;
constant uint THREADS = 128;
// 8x8 matrices across the head.
constant uint DB = DIM / 8;
// Half floats a position of the dequantized keys or values takes, padded
// so that the threads writing different positions hit different banks.
constant uint STRIDE = DIM + 8;

// Dequantizes the keys or values of the round's positions from `start`
// into `kv`, the positions past `len` clamped to the last, which only
// masked scores meet.
static void dequantize(threadgroup half* kv, const device char4* quants, const device half* scales,
                       uint start, uint len, uint kv_dim, uint base, uint lid) {
    constexpr uint quads = DIM / 4;
    for (uint i = lid; i < ROUND * quads; i += THREADS) {
        uint p = i / quads;
        uint d = (i % quads) * 4;
        uint e = min(start + p, len - 1) * kv_dim + base + d;
        *(threadgroup half4*)(kv + p * STRIDE + d) = half4(float4(quants[e / 4]) * float(scales[e / 32]));
    }
}

kernel void attention_flash(
    device float* out [[buffer(0)]],
    const device float* q [[buffer(1)]],
    const device char4* k_quants [[buffer(2)]],
    const device half* k_scales [[buffer(3)]],
    const device char4* v_quants [[buffer(4)]],
    const device half* v_scales [[buffer(5)]],
    device float* partials [[buffer(6)]],
    constant Params& p [[buffer(7)]],
    uint2 wg [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup half kv[ROUND * STRIDE];
    // Each SIMD group's scores for the round, then their exponentials as
    // half floats; and the factors its rows' sums are rescaled by, as a
    // diagonal matrix.
    threadgroup float s_scratch[4][ROWS * ROUND];
    threadgroup half p_scratch[4][ROWS * ROUND];
    threadgroup float diag[4][64];

    uint kvh = wg.y % p.n_kv_heads;
    uint c = wg.y / p.n_kv_heads;
    uint group = p.n_heads / p.n_kv_heads;
    uint kv_dim = p.n_kv_heads * DIM;
    uint base = kvh * DIM;
    uint rows = p.tokens * group;

    // The lane's row of the SIMD group's eight, and its token and head;
    // four lanes share a row in the softmax, eight positions each. A row
    // past the batch takes the last one's place, and its results are not
    // kept.
    uint r0 = wg.x * THREADS / 32 * ROWS + sg * ROWS;
    uint my = lane / 4;
    uint row = min(r0 + my, rows - 1);
    uint t = p.first + row / group;
    uint h = kvh * group + row % group;
    uint len = p.pos + t + 1;
    // The positions any row of the threadgroup attends to in the chunk.
    uint last_row = min(wg.x * THREADS / 32 * ROWS + THREADS / 32 * ROWS, rows) - 1;
    uint tg_len = p.pos + p.first + last_row / group + 1;
    uint start = c * CHUNK;
    if (start >= tg_len) {
        return;
    }
    uint end = min(start + CHUNK, tg_len);

    // The SIMD group's queries, scaled, through threadgroup memory: each
    // lane copies a quarter of its row.
    threadgroup half* qs = kv + sg * ROWS * STRIDE;
    {
        const device float4* src = (const device float4*)(q + (t * p.n_heads + h) * DIM);
        float scale = rsqrt(float(DIM));
        for (uint d = (lane % 4); d < DIM / 4; d += 4) {
            *(threadgroup half4*)(qs + my * STRIDE + 4 * d) = half4(src[d] * scale);
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_half8x8 qm[DB];
    for (uint j = 0; j < DB; j++) {
        simdgroup_load(qm[j], qs + 8 * j, STRIDE);
    }
    for (uint i = lane; i < 64; i += 32) {
        diag[sg][i] = 0.0f;
    }
    simdgroup_float8x8 o[DB];
    for (uint j = 0; j < DB; j++) {
        o[j] = simdgroup_float8x8(0.0f);
    }
    float m = -1e30f;
    float l = 0.0f;

    for (uint r = start; r < end; r += ROUND) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        dequantize(kv, k_quants, k_scales, r, tg_len, kv_dim, base, lid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // The rows' scores for the round's positions, eight at a time.
        for (uint b = 0; b < ROUND / 8; b++) {
            simdgroup_float8x8 s = simdgroup_float8x8(0.0f);
            for (uint j = 0; j < DB; j++) {
                simdgroup_half8x8 k;
                simdgroup_load(k, kv + 8 * b * STRIDE + 8 * j, STRIDE, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(s, qm[j], k, s);
            }
            simdgroup_store(s, s_scratch[sg] + 8 * b, ROUND);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        // The lane's eight positions of its row, masked past its token,
        // against the row's new maximum.
        uint c0 = (lane % 4) * 8;
        float sv[8];
        float mx = -1e30f;
        for (uint i = 0; i < 8; i++) {
            bool valid = r + c0 + i < len;
            sv[i] = valid ? s_scratch[sg][my * ROUND + c0 + i] : -1e30f;
            mx = max(mx, sv[i]);
        }
        mx = max(mx, simd_shuffle_xor(mx, 1));
        mx = max(mx, simd_shuffle_xor(mx, 2));
        float m_new = max(m, mx);
        float factor = exp(m - m_new);
        float sum = 0.0f;
        for (uint i = 0; i < 8; i++) {
            float e = r + c0 + i < len ? exp(sv[i] - m_new) : 0.0f;
            sum += e;
            p_scratch[sg][my * ROUND + c0 + i] = half(e);
        }
        sum += simd_shuffle_xor(sum, 1);
        sum += simd_shuffle_xor(sum, 2);
        l = l * factor + sum;
        m = m_new;
        if (lane % 4 == 0) {
            diag[sg][my * 9] = factor;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        dequantize(kv, v_quants, v_scales, r, tg_len, kv_dim, base, lid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // What the rows have summed, rescaled to the new maxima, plus the
        // exponentials' products with the values.
        simdgroup_float8x8 f;
        simdgroup_load(f, diag[sg], 8);
        simdgroup_half8x8 pm[ROUND / 8];
        for (uint b = 0; b < ROUND / 8; b++) {
            simdgroup_load(pm[b], p_scratch[sg] + 8 * b, ROUND);
        }
        for (uint j = 0; j < DB; j++) {
            simdgroup_multiply(o[j], f, o[j]);
            for (uint b = 0; b < ROUND / 8; b++) {
                simdgroup_half8x8 v;
                simdgroup_load(v, kv + 8 * b * STRIDE + 8 * j, STRIDE);
                simdgroup_multiply_accumulate(o[j], pm[b], v, o[j]);
            }
        }
    }

    // The rows' sums, through the scores' scratch an 8x8 at a time, a lane
    // taking two elements of each.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane % 4 == 0) {
        s_scratch[sg][ROWS * 8 + my * 2] = m;
        s_scratch[sg][ROWS * 8 + my * 2 + 1] = l;
    }
    for (uint j = 0; j < DB; j++) {
        simdgroup_store(o[j], s_scratch[sg], 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = lane; e < 64; e += 32) {
            uint rr = e / 8;
            uint orow = r0 + rr;
            if (orow < rows) {
                uint ot = p.first + orow / group;
                uint oh = kvh * group + orow % group;
                float ol = s_scratch[sg][ROWS * 8 + rr * 2 + 1];
                float v = s_scratch[sg][e];
                if (p.chunks == 1) {
                    out[(ot * p.n_heads + oh) * DIM + 8 * j + e % 8] = v / ol;
                } else {
                    uint pb = (((ot - p.first) * p.n_heads + oh) * p.chunks + c) * (DIM + 2);
                    if (j == 0 && e % 8 == 0) {
                        partials[pb] = s_scratch[sg][ROWS * 8 + rr * 2];
                        partials[pb + 1] = ol;
                    }
                    partials[pb + 2 + 8 * j + e % 8] = v;
                }
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}
