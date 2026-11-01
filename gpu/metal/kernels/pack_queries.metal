// The queries of a batch of tokens as half floats by key/value head, for
// `attention_mma.metal`: out[kvh][row][d] = half(q[t][h][d] / sqrt(dim))
// for the rows of the head, a token's query head each, the heads that
// share a key/value head one after another, and the rows padded to
// `padded` with zeros.

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint n_heads;
    uint n_kv_heads;
    uint dim;
    uint n;
    // The rows of a key/value head with the padding.
    uint padded;
};

kernel void pack_queries(
    device half4* out [[buffer(0)]],
    const device float4* q [[buffer(1)]],
    constant Params& p [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    uint quads = p.dim / 4;
    if (i >= p.n_kv_heads * p.padded * quads) {
        return;
    }
    uint d = i % quads;
    uint row = (i / quads) % p.padded;
    uint kvh = i / quads / p.padded;
    uint group = p.n_heads / p.n_kv_heads;
    if (row >= p.n * group) {
        out[i] = half4(0.0h);
        return;
    }
    uint t = row / group;
    uint h = kvh * group + row % group;
    out[i] = half4(q[(t * p.n_heads + h) * quads + d] * rsqrt(float(p.dim)));
}
