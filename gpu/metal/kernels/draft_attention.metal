// Attention of the draft's block of rows over the context before it, in a
// ring of slots per head, and the block itself: one threadgroup per (row,
// query head). The threads score the keys between them, a whole key each,
// softmax the scores against their maximum, then take an element of the
// head each at every key of their share and sum the values the scores
// weigh, the shares added up through threadgroup memory. The ring holds
// `window` slots a head, and a row sees the `window - 1` positions before
// its own and every row of the block.

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint rows;
    uint pos;
    uint window;
    uint n_heads;
    uint head_dim;
    uint n_kv_heads;
};

#ifndef MAX_KEYS
#define MAX_KEYS 2064
#endif
constant uint THREADS = 256;

kernel void draft_attention(
    device float* out [[buffer(0)]],
    const device float* q [[buffer(1)]],
    const device float* k_ring [[buffer(2)]],
    const device float* v_ring [[buffer(3)]],
    const device float* k_block [[buffer(4)]],
    const device float* v_block [[buffer(5)]],
    constant Params& p [[buffer(6)]],
    uint wg [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint sid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float scores[MAX_KEYS];
    threadgroup float maxima[THREADS / 32];
    threadgroup float totals[THREADS / 32];
    threadgroup float sums[THREADS];

    uint row = wg / p.n_heads;
    uint head = wg % p.n_heads;
    uint group = p.n_heads / p.n_kv_heads;
    uint kvh = head / group;
    uint kv_dim = p.n_kv_heads * p.head_dim;
    uint position = p.pos + row;
    uint start = position + 1 > p.window ? position + 1 - p.window : 0;
    uint context = p.pos - start;
    uint keys = context + p.rows;
    const device float* qh = q + (row * p.n_heads + head) * p.head_dim;
    float scale = rsqrt(float(p.head_dim));

    for (uint i = lid; i < keys; i += THREADS) {
        const device float* key = i < context
            ? k_ring + (kvh * p.window + (start + i) % p.window) * p.head_dim
            : k_block + (i - context) * kv_dim + kvh * p.head_dim;
        float s = 0.0f;
        for (uint d = 0; d < p.head_dim; d++) {
            s = fma(qh[d], key[d], s);
        }
        scores[i] = s * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m = -INFINITY;
    for (uint i = lid; i < keys; i += THREADS) {
        m = max(m, scores[i]);
    }
    m = simd_max(m);
    if (lane == 0) {
        maxima[sid] = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = maxima[0];
    for (uint i = 1; i < THREADS / 32; i++) {
        m = max(m, maxima[i]);
    }
    float sum = 0.0f;
    for (uint i = lid; i < keys; i += THREADS) {
        float e = exp(scores[i] - m);
        scores[i] = e;
        sum += e;
    }
    sum = simd_sum(sum);
    if (lane == 0) {
        totals[sid] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint i = 0; i < THREADS / 32; i++) {
        total += totals[i];
    }

    // Each thread takes element d of the head at every key of its share.
    uint parts = THREADS / p.head_dim;
    uint d = lid % p.head_dim;
    uint part = lid / p.head_dim;
    float acc = 0.0f;
    for (uint i = part; i < keys; i += parts) {
        const device float* value = i < context
            ? v_ring + (kvh * p.window + (start + i) % p.window) * p.head_dim
            : v_block + (i - context) * kv_dim + kvh * p.head_dim;
        acc = fma(scores[i], value[d], acc);
    }
    sums[lid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (part == 0) {
        for (uint j = 1; j < parts; j++) {
            acc += sums[j * p.head_dim + d];
        }
        out[(row * p.n_heads + head) * p.head_dim + d] = acc / total;
    }
}
