// Causal attention over 8-bit caches, with a half-precision scale per block
// of 32: one threadgroup per (token, key/value head, chunk of 128
// positions), for every query head that shares the key/value head, so that
// the caches are read once for the group and a long context spreads over
// the GPU. A SIMD group scores one position at a time, its lanes each taking
// four elements of the key and adding their products up across the group,
// so that a key is read as one contiguous run, in as many rounds as the
// head is wider than the SIMD group takes at once, the heads' sums added up
// across the lanes together; the scores are softmaxed against the chunk's
// own maximum; then each thread takes four elements of the head at every
// position of its lane and sums the values the scores weigh, and the lanes'
// sums are added together across the SIMD group. The chunk's maximum, sum
// of exponentials, and weighted values go to the partials buffer for
// `attention_combine.metal` to merge, or, when there is only one chunk,
// straight to the output. A dispatch covers the tokens from `first` on,
// with `chunks` chunks for each.

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
};

constant uint CHUNK = 128;
constant uint THREADS = 256;
// Most query heads to a key/value head, and rounds of four elements to a
// lane the widest head takes.
constant uint GROUP = 8;
constant uint ROUNDS = 2;

kernel void attention(
    device float* out [[buffer(0)]],
    const device float4* q [[buffer(1)]],
    const device char4* k_quants [[buffer(2)]],
    const device half* k_scales [[buffer(3)]],
    const device char4* v_quants [[buffer(4)]],
    const device half* v_scales [[buffer(5)]],
    device float* partials [[buffer(6)]],
    constant Params& p [[buffer(7)]],
    uint wg [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint sid [[simdgroup_index_in_threadgroup]],
    uint nsg [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float partial[THREADS / 8][GROUP];
    threadgroup float scores[GROUP][CHUNK];

    uint c = wg % p.chunks;
    uint tk = wg / p.chunks;
    uint t = p.first + tk / p.n_kv_heads;
    uint kvh = tk % p.n_kv_heads;
    uint len = p.pos + t + 1;
    uint start = c * CHUNK;
    if (start >= len) {
        return;
    }
    uint count = min(CHUNK, len - start);
    uint group = p.n_heads / p.n_kv_heads;
    uint kv = kvh * p.head_dim;
    uint kv_dim = p.n_kv_heads * p.head_dim;
    float scale = rsqrt(float(p.head_dim));
    uint quads = p.head_dim / 4;
    // The first query head of the group, for this token, in vectors.
    uint qbase = (t * p.n_heads + kvh * group) * quads;

    // Each lane of a SIMD group takes four elements of a key in each round
    // of as many as the SIMD group takes at once, the lanes side by side,
    // and keeps the queries' matching elements, the heads past the group
    // zero. It sums its products for every head before the lanes add them
    // up, which takes halves of the heads' sums between lanes 16, 8 and 4
    // apart, each lane keeping the half its bit says and adding the other
    // lane's, and then adds the one head's sum it is left with across the
    // last four lanes.
    float4 query[GROUP][ROUNDS];
    for (uint g = 0; g < GROUP; g++) {
        for (uint r = 0; r < ROUNDS; r++) {
            uint d = r * 32 + lane;
            query[g][r] = g < group && d < quads ? q[qbase + g * quads + d] * scale : float4(0.0f);
        }
    }
    uint head = (lane & 16 ? 4 : 0) + (lane & 8 ? 2 : 0) + (lane & 4 ? 1 : 0);
    for (uint i = sid; i < count; i += nsg) {
        float v[GROUP] = {};
        for (uint r = 0; r < ROUNDS; r++) {
            uint d = r * 32 + lane;
            if (d < quads) {
                uint e = (start + i) * kv_dim + kv + 4 * d;
                float4 key = float4(k_quants[e / 4]) * float(k_scales[e / 32]);
                for (uint g = 0; g < GROUP; g++) {
                    v[g] += dot(query[g][r], key);
                }
            }
        }
        float a[4];
        for (uint k = 0; k < 4; k++) {
            bool hi = lane & 16;
            a[k] = (hi ? v[k + 4] : v[k]) + simd_shuffle_xor(hi ? v[k] : v[k + 4], 16);
        }
        float b[2];
        for (uint k = 0; k < 2; k++) {
            bool hi = lane & 8;
            b[k] = (hi ? a[k + 2] : a[k]) + simd_shuffle_xor(hi ? a[k] : a[k + 2], 8);
        }
        bool hi = lane & 4;
        float sum = (hi ? b[1] : b[0]) + simd_shuffle_xor(hi ? b[0] : b[1], 4);
        sum += simd_shuffle_xor(sum, 2);
        sum += simd_shuffle_xor(sum, 1);
        if ((lane & 3) == 0 && head < group) {
            scores[head][i] = sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each head's maximum over the chunk, for a stable softmax, and the
    // sum of its exponentials.
    float m[GROUP];
    float total[GROUP];
    for (uint g = 0; g < group; g++) {
        float s = lid < count ? scores[g][lid] : -1e30f;
        float sm = simd_max(s);
        if (lane == 0) {
            partial[sid][g] = sm;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float e[GROUP];
    for (uint g = 0; g < group; g++) {
        m[g] = -1e30f;
        for (uint i = 0; i < nsg; i++) {
            m[g] = max(m[g], partial[i][g]);
        }
        e[g] = lid < count ? exp(scores[g][lid] - m[g]) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // The rows past the group zero, so that the values' sums can take all
    // of them.
    for (uint g = 0; g < GROUP; g++) {
        if (g >= group) {
            e[g] = 0.0f;
        }
        if (lid < CHUNK) {
            scores[g][lid] = e[g];
        }
        float sum = simd_sum(e[g]);
        if (lane == 0) {
            partial[sid][g] = sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint g = 0; g < group; g++) {
        total[g] = 0.0f;
        for (uint i = 0; i < nsg; i++) {
            total[g] += partial[i][g];
        }
    }

    // Each thread takes four elements of the head at every position of its
    // lane, the lanes striding through the positions together, so that a
    // position's values are read in runs and weighed for every head. A SIMD
    // group's lanes take a power of two of positions at once, the lanes of
    // one position side by side, so that each element's sums over the
    // positions are added up across the SIMD group.
    uint lanes = min(32u, 1u << (31 - clz(THREADS / quads)));
    uint d = lid / lanes;
    uint vlane = lid % lanes;
    float4 acc[GROUP];
    for (uint g = 0; g < GROUP; g++) {
        acc[g] = 0.0f;
    }
    if (d < quads) {
        for (uint i = vlane; i < count; i += lanes) {
            uint e = (start + i) * kv_dim + kv + 4 * d;
            float4 v = float4(v_quants[e / 4]) * float(v_scales[e / 32]);
            for (uint g = 0; g < GROUP; g++) {
                acc[g] += scores[g][i] * v;
            }
        }
    }
    for (uint g = 0; g < GROUP; g++) {
        for (ushort s = 1; s < lanes; s <<= 1) {
            acc[g] += simd_shuffle_xor(acc[g], s);
        }
    }
    if (vlane != 0 || d >= quads) {
        return;
    }
    for (uint g = 0; g < group; g++) {
        uint h = kvh * group + g;
        if (p.chunks == 1) {
            device float4* dst = (device float4*)(out + (t * p.n_heads + h) * p.head_dim);
            dst[d] = acc[g] / total[g];
        } else {
            uint base = (((t - p.first) * p.n_heads + h) * p.chunks + c) * (p.head_dim + 2);
            if (d == 0) {
                partials[base] = m[g];
                partials[base + 1] = total[g];
            }
            for (uint j = 0; j < 4; j++) {
                partials[base + 2 + 4 * d + j] = acc[g][j];
            }
        }
    }
}
