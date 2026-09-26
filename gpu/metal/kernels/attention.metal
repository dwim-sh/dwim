// Causal attention over 8-bit caches, with a half-precision scale per block
// of 32: one threadgroup per (token, head). The
// threads first score the positions the token attends to, strided, into the
// threadgroup's row of the scores buffer; then softmax the scores; then each
// thread sums one element of the head over the values, reading them
// coalesced. A dispatch covers the tokens from `first` on, and each
// threadgroup's row of scores is `stride` long.

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint n_heads;
    uint head_dim;
    uint n_kv_heads;
    uint pos;
    uint first;
    uint stride;
};

kernel void attention(
    device float* out [[buffer(0)]],
    const device float4* q [[buffer(1)]],
    const device char4* k_quants [[buffer(2)]],
    const device half* k_scales [[buffer(3)]],
    const device char* v_quants [[buffer(4)]],
    const device half* v_scales [[buffer(5)]],
    device float* scores [[buffer(6)]],
    constant Params& p [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint sid [[simdgroup_index_in_threadgroup]],
    uint nsg [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float partial[32];

    uint head = p.first * p.n_heads + group;
    uint t = head / p.n_heads;
    uint h = head % p.n_heads;
    uint kv = (h / (p.n_heads / p.n_kv_heads)) * p.head_dim;
    uint kv_dim = p.n_kv_heads * p.head_dim;
    uint len = p.pos + t + 1;
    float scale = rsqrt(float(p.head_dim));
    uint quads = p.head_dim / 4;
    const device float4* qh = q + head * quads;
    device float* row = scores + group * p.stride;

    // Scores, and their maximum for a stable softmax.
    float m = -FLT_MAX;
    for (uint pos = lid; pos < len; pos += threads) {
        uint base = pos * kv_dim + kv;
        const device char4* k = k_quants + base / 4;
        const device half* ks = k_scales + base / 32;
        float s = 0.0f;
        for (uint d = 0; d < quads; d++) {
            s += dot(qh[d], float4(k[d])) * float(ks[d / 8]);
        }
        s *= scale;
        row[pos] = s;
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
        float e = exp(row[pos] - m);
        row[pos] = e;
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
            uint i = pos * kv_dim + kv + lid;
            o += row[pos] * float(v_quants[i]) * float(v_scales[i / 32]);
        }
        out[head * p.head_dim + lid] = o / total;
    }
}
