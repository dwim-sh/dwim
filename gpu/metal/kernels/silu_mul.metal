// gate = silu(gate) * up

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint len;
};

kernel void silu_mul(
    device float* gate [[buffer(0)]],
    const device float* up [[buffer(1)]],
    constant Params& p [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= p.len) {
        return;
    }
    float g = gate[i];
    gate[i] = g / (1.0f + exp(-g)) * up[i];
}
