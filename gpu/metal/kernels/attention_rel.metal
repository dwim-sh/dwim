// Causal attention with a bias by distance and a sink, over 8-bit caches
// that hold a ring of the latest positions: `attention.metal`, with what
// Inkling's attention adds to it. One threadgroup per (token, key/value
// head, chunk of CHUNK positions), for every query head that shares the
// key/value head. A token attends over the positions no more than
// `window - 1` back, and the caches hold position `p` in row `p % ring`.
// A key `d` positions back has added to its score the dot product of the
// head's row of `rel` with row `d` of `bias`, if `bias` has one. The
// softmax is taken over the keys and a sink that has no value, whose score
// is the head's of `sinks`: here where a token has one chunk, and in
// `attention_rel_combine.metal` where it has more, from the chunks'
// maxima, sums of exponentials, and weighted values in the partials.

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
    uint padded;
    uint window;
    uint ring;
    uint n_rel;
    uint extent;
    uint base;
    uint batched;
};

#ifndef CHUNK
#define CHUNK 128
#endif
#ifndef ROUNDS
#define ROUNDS 2
#endif
constant uint THREADS = 256;
// Most query heads to a key/value head.
constant uint GROUP = 8;

kernel void attention_rel(
    device float* out [[buffer(0)]],
    const device float4* q [[buffer(1)]],
    const device char4* k_quants [[buffer(2)]],
    const device half* k_scales [[buffer(3)]],
    const device char4* v_quants [[buffer(4)]],
    const device half* v_scales [[buffer(5)]],
    device float* partials [[buffer(6)]],
    const device float* rel [[buffer(7)]],
    const device float* bias [[buffer(8)]],
    const device float* sinks [[buffer(9)]],
    constant Params& p [[buffer(10)]],
    uint wg [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint sid [[simdgroup_index_in_threadgroup]],
    uint nsg [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float partial_m[THREADS / 8][GROUP];
    threadgroup float partial_s[THREADS / 8][GROUP];
    threadgroup float scores[GROUP][CHUNK];

    uint c = wg % p.chunks;
    uint tk = wg / p.chunks;
    uint t = p.first + tk / p.n_kv_heads;
    uint kvh = tk % p.n_kv_heads;
    // The token's own position, the first it attends over, and how many
    // it does.
    uint at = p.pos + t;
    uint lo = at + 1 > p.window ? at + 1 - p.window : 0;
    uint len = at + 1 - lo;
    if (c * CHUNK >= len) {
        return;
    }
    uint start = lo + c * CHUNK;
    uint count = min(uint(CHUNK), len - c * CHUNK);
    uint group = p.n_heads / p.n_kv_heads;
    uint kv = kvh * p.head_dim;
    uint kv_dim = p.n_kv_heads * p.head_dim;
    float scale = rsqrt(float(p.head_dim));
    uint quads = p.head_dim / 4;
    uint qbase = (t * p.n_heads + kvh * group) * quads;

    // Each lane of a SIMD group takes four elements of a key in each round
    // of as many as the SIMD group takes at once, the lanes side by side,
    // and keeps the queries' matching elements.
    float4 query[GROUP][ROUNDS];
    for (uint g = 0; g < group; g++) {
        #pragma clang loop unroll(full)
        for (uint r = 0; r < ROUNDS; r++) {
            uint d = r * 32 + lane;
            query[g][r] = d < quads ? q[qbase + g * quads + d] * scale : float4(0.0f);
        }
    }

    // Preload this lane's head projection for the relative-position bias once
    // before walking the chunk's keys.
    uint head = (lane & 16 ? 4 : 0) + (lane & 8 ? 2 : 0) + (lane & 4 ? 1 : 0);
    bool vec_rel = (p.n_rel & 3u) == 0 && p.n_rel <= 16;
    uint rel_quads = p.n_rel / 4;
    float4 rel_v[4] = {};
    const device float* rel_head = rel + (t * p.n_heads + kvh * group + head) * p.n_rel;
    if ((lane & 3) == 0 && head < group && vec_rel) {
        const device float4* r4 = (const device float4*)rel_head;
        for (uint j = 0; j < rel_quads; j++) {
            rel_v[j] = r4[j];
        }
    }

    uint ring_row = (start + sid) % p.ring;
    uint ring_step = nsg % p.ring;
    for (uint i = sid; i < count; i += nsg) {
        float v[GROUP] = {};
        #pragma clang loop unroll(full)
        for (uint r = 0; r < ROUNDS; r++) {
            uint d = r * 32 + lane;
            if (d < quads) {
                uint e = ring_row * kv_dim + kv + 4 * d;
                float4 key = float4(k_quants[e / 4]) * float(k_scales[e / 32]);
                for (uint g = 0; g < group; g++) {
                    v[g] += dot(query[g][r], key);
                }
            }
        }
        float a[4];
        #pragma clang loop unroll(full)
        for (uint k = 0; k < 4; k++) {
            bool hi = lane & 16;
            a[k] = (hi ? v[k + 4] : v[k]) + simd_shuffle_xor(hi ? v[k] : v[k + 4], 16);
        }
        float b[2];
        #pragma clang loop unroll(full)
        for (uint k = 0; k < 2; k++) {
            bool hi = lane & 8;
            b[k] = (hi ? a[k + 2] : a[k]) + simd_shuffle_xor(hi ? a[k] : a[k + 2], 8);
        }
        bool hi = lane & 4;
        float sum = (hi ? b[1] : b[0]) + simd_shuffle_xor(hi ? b[0] : b[1], 4);
        sum += simd_shuffle_xor(sum, 2);
        sum += simd_shuffle_xor(sum, 1);
        if ((lane & 3) == 0 && head < group) {
            uint distance = at - (start + i);
            if (distance < p.extent) {
                if (vec_rel) {
                    const device float4* b4 = (const device float4*)(bias + distance * p.n_rel);
                    for (uint j = 0; j < rel_quads; j++) {
                        sum += dot(rel_v[j], b4[j]);
                    }
                } else {
                    const device float* bp = bias + distance * p.n_rel;
                    for (uint j = 0; j < p.n_rel; j++) {
                        sum += rel_head[j] * bp[j];
                    }
                }
            }
            scores[head][i] = sum;
        }
        ring_row += ring_step;
        if (ring_row >= p.ring) {
            ring_row -= p.ring;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each head's maximum over the chunk, for a stable softmax, and the
    // sum of its exponentials. Separate max/sum partial tables save a barrier.
    uint active_sg = min(nsg, (count + 31) / 32);
    float m[GROUP];
    float total[GROUP];
    if (sid < active_sg) {
        for (uint g = 0; g < group; g++) {
            float s = lid < count ? scores[g][lid] : -1e30f;
            float sm = simd_max(s);
            if (lane == 0) {
                partial_m[sid][g] = sm;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sid < active_sg) {
        for (uint g = 0; g < group; g++) {
            m[g] = -1e30f;
            for (uint i = 0; i < active_sg; i++) {
                m[g] = max(m[g], partial_m[i][g]);
            }
            float e = lid < count ? exp(scores[g][lid] - m[g]) : 0.0f;
            if (lid < CHUNK) {
                scores[g][lid] = e;
            }
            float sum = simd_sum(e);
            if (lane == 0) {
                partial_s[sid][g] = sum;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint g = 0; g < group; g++) {
        m[g] = -1e30f;
        total[g] = 0.0f;
        for (uint i = 0; i < active_sg; i++) {
            m[g] = max(m[g], partial_m[i][g]);
            total[g] += partial_s[i][g];
        }
    }

    // Each thread takes four elements of the head at every position of its
    // lane, the lanes striding through the positions together.
    uint lanes = min(32u, 1u << (31 - clz(THREADS / quads)));
    uint d = lid / lanes;
    uint vlane = lid % lanes;
    float4 acc[GROUP];
    for (uint g = 0; g < group; g++) {
        acc[g] = 0.0f;
    }
    if (d < quads) {
        uint v_ring = (start + vlane) % p.ring;
        uint v_step = lanes % p.ring;
        for (uint i = vlane; i < count; i += lanes) {
            uint e = v_ring * kv_dim + kv + 4 * d;
            float4 v = float4(v_quants[e / 4]) * float(v_scales[e / 32]);
            for (uint g = 0; g < group; g++) {
                acc[g] += scores[g][i] * v;
            }
            v_ring += v_step;
            if (v_ring >= p.ring) {
                v_ring -= p.ring;
            }
        }
    }
    for (uint g = 0; g < group; g++) {
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
            float sink = sinks[t * p.n_heads + h];
            float most = max(m[g], sink);
            float keys = exp(m[g] - most);
            device float4* dst = (device float4*)(out + (t * p.n_heads + h) * p.head_dim);
            dst[d] = acc[g] * keys / (total[g] * keys + exp(sink - most));
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
