// Merges the chunks of `attention.metal`: one threadgroup per (token,
// head), each thread taking one element of the head. The chunks' softmaxes
// were taken against their own maxima, so each is rescaled to the maximum
// over the chunks the token attends to before the sums are added.

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint n_heads;
    uint head_dim;
    uint n_kv_heads;
    uint pos;
    uint first;
    uint chunks;
    // Positions to a chunk.
    uint chunk;
};

kernel void attention_combine(
    device float* out [[buffer(0)]],
    const device float* partials [[buffer(1)]],
    constant Params& p [[buffer(2)]],
    uint wg [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]])
{
    uint t = p.first + wg / p.n_heads;
    uint h = wg % p.n_heads;
    if (lid >= p.head_dim) {
        return;
    }
    uint len = p.pos + t + 1;
    uint chunks = (len + p.chunk - 1) / p.chunk;
    uint stride = p.head_dim + 2;
    const device float* base = partials + ((t - p.first) * p.n_heads + h) * p.chunks * stride;
    float m = -1e30f;
    for (uint c = 0; c < chunks; c++) {
        m = max(m, base[c * stride]);
    }
    float total = 0.0f;
    float o = 0.0f;
    for (uint c = 0; c < chunks; c++) {
        float scale = exp(base[c * stride] - m);
        total += scale * base[c * stride + 1];
        o += scale * base[c * stride + 2 + lid];
    }
    out[(t * p.n_heads + h) * p.head_dim + lid] = o / total;
}
