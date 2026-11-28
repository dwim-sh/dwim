// Merges the chunks of `attention_rel.metal`: one threadgroup per (token,
// head), each thread taking one element of the head. The chunks' softmaxes
// were taken against their own maxima, so each is rescaled to the greatest
// of them and the head's sink before the sums are added, the sink's among
// them, which has no value.

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
    uint tokens;
    uint padded;
    uint window;
    uint ring;
    uint n_rel;
    uint extent;
    // For a batch, whose chunks are the dispatch's: its first position.
    uint base;
    uint batched;
    uint bidir;
};

kernel void attention_rel_combine(
    device float* out [[buffer(0)]],
    const device float* partials [[buffer(1)]],
    const device float* sinks [[buffer(2)]],
    constant Params& p [[buffer(3)]],
    uint wg [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]])
{
    uint t = p.first + wg / p.n_heads;
    uint h = wg % p.n_heads;
    if (lid >= p.head_dim) {
        return;
    }
    // The chunks the token attends over: its own, counted from the first
    // position it attends over, or for a batch those of the dispatch's
    // that its positions are in.
    uint at = p.bidir != 0 ? p.pos : p.pos + t;
    uint len = at + 1 > p.window ? p.window : at + 1;
    uint from = 0;
    uint chunks = (len + p.chunk - 1) / p.chunk;
    if (p.batched != 0) {
        from = (at + 1 - len - p.base) / p.chunk;
        chunks = (at - p.base) / p.chunk + 1;
    }
    uint stride = p.head_dim + 2;
    const device float* base = partials + ((t - p.first) * p.n_heads + h) * p.chunks * stride;
    float sink = p.bidir != 0 ? -1e30f : sinks[t * p.n_heads + h];
    float m = sink;
    for (uint c = from; c < chunks; c++) {
        m = max(m, base[c * stride]);
    }
    float total = p.bidir != 0 ? 0.0f : exp(sink - m);
    float o = 0.0f;
    for (uint c = from; c < chunks; c++) {
        float scale = exp(base[c * stride] - m);
        total += scale * base[c * stride + 1];
        o += scale * base[c * stride + 2 + lid];
    }
    out[(t * p.n_heads + h) * p.head_dim + lid] = o / total;
}
