// Causal attention: one threadgroup per (token, head). The threads first
// score the positions the token attends to, strided, into threadgroup
// memory; then softmax the scores; then each thread sums one element of the
// head over the values, reading them coalesced.

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint n_heads;
    uint head_dim;
    uint n_kv_heads;
    uint pos;
};

constant uint MAX_LEN = 4096;

kernel void attention(
    device float* out [[buffer(0)]],
    const device float4* q [[buffer(1)]],
    const device float4* k_cache [[buffer(2)]],
    const device float* v_cache [[buffer(3)]],
    constant Params& p [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint sid [[simdgroup_index_in_threadgroup]],
    uint nsg [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float scores[MAX_LEN];
    threadgroup float partial[32];

    uint t = group / p.n_heads;
    uint h = group % p.n_heads;
    uint kv = (h / (p.n_heads / p.n_kv_heads)) * p.head_dim;
    uint kv_dim = p.n_kv_heads * p.head_dim;
    uint len = p.pos + t + 1;
    float scale = rsqrt(float(p.head_dim));
    uint quads = p.head_dim / 4;
    const device float4* qh = q + group * quads;

    // Scores, and their maximum for a stable softmax.
    float m = -FLT_MAX;
    for (uint pos = lid; pos < len; pos += threads) {
        const device float4* k = k_cache + (pos * kv_dim + kv) / 4;
        float s = 0.0f;
        for (uint d = 0; d < quads; d++) {
            s += dot(qh[d], k[d]);
        }
        s *= scale;
        scores[pos] = s;
        m = max(m, s);
    }
    m = simd_max(m);
    if (lane == 0) {
        partial[sid] = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < nsg; i++) {
        m = max(m, partial[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sum = 0.0f;
    for (uint pos = lid; pos < len; pos += threads) {
        float e = exp(scores[pos] - m);
        scores[pos] = e;
        sum += e;
    }
    sum = simd_sum(sum);
    if (lane == 0) {
        partial[sid] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint i = 0; i < nsg; i++) {
        total += partial[i];
    }

    // Each thread owns one element of the head.
    if (lid < p.head_dim) {
        float o = 0.0f;
        for (uint pos = 0; pos < len; pos++) {
            o += scores[pos] * v_cache[pos * kv_dim + kv + lid];
        }
        out[group * p.head_dim + lid] = o / total;
    }
}
