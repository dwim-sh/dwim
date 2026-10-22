// The activations of a batch of tokens as half floats, for
// `matmul_ternary_tile.metal`: out[t][k] = half(x[t][k]), rounded to the
// nearest, with the tokens padded to whole tiles with zeros.

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint n;
    uint cols;
    // The tokens with the padding.
    uint padded;
};

kernel void pack_halves(
    device half4* out [[buffer(0)]],
    const device float4* x [[buffer(1)]],
    constant Params& p [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    uint quads = p.cols / 4;
    if (i >= p.padded * quads) {
        return;
    }
    out[i] = i / quads < p.n ? half4(x[i]) : half4(0.0h);
}
