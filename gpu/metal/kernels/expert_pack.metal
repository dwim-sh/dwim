// The inputs of the experts' tiles as half floats, in the order of the
// tiles: out[j] is the input of the pair `order[j]`, which is its token's
// row where the experts of a token share one, and the pair's own
// otherwise; the rows past the pairs, which the last tiles reach into, are
// zeros.

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint pairs;
    uint cols;
    // The rows with the padding.
    uint padded;
    // Experts per token where they share an input, and one otherwise.
    uint share;
};

kernel void expert_pack(
    device half4* out [[buffer(0)]],
    const device float4* x [[buffer(1)]],
    const device uint* order [[buffer(2)]],
    constant Params& p [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    uint quads = p.cols / 4;
    if (i >= p.padded * quads) {
        return;
    }
    uint j = i / quads;
    out[i] = j < p.pairs ? half4(x[order[j] / p.share * quads + i % quads]) : half4(0.0h);
}
