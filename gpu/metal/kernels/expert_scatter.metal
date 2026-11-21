// Puts the experts' results back by pair: sorted[j] is the result of the
// pair `order[j]`, a row of `rows` each.

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint pairs;
    uint rows;
};

kernel void expert_scatter(
    device float4* out [[buffer(0)]],
    const device float4* sorted [[buffer(1)]],
    const device uint* order [[buffer(2)]],
    constant Params& p [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    uint quads = p.rows / 4;
    if (i >= p.pairs * quads) {
        return;
    }
    out[order[i / quads] * quads + i % quads] = sorted[i];
}
