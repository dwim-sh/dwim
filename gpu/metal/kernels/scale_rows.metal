// Scales each row of x, `width` long, by its factor.

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint len;
    uint width;
};

kernel void scale_rows(
    device float* x [[buffer(0)]],
    const device float* factors [[buffer(1)]],
    constant Params& p [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i < p.len) {
        x[i] *= factors[i / p.width];
    }
}
