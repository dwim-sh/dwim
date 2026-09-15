// x += y

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint len;
};

kernel void add(
    device float* x [[buffer(0)]],
    const device float* y [[buffer(1)]],
    constant Params& p [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= p.len) {
        return;
    }
    x[i] += y[i];
}
