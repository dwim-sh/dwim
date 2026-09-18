// x *= sigmoid(gate)

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint len;
};

kernel void sigmoid_mul(
    device float* x [[buffer(0)]],
    const device float* gate [[buffer(1)]],
    constant Params& p [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= p.len) {
        return;
    }
    x[i] *= 1.0f / (1.0f + exp(-gate[i]));
}
