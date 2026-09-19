// cache[p.offset..][..p.len] = half(src): activations rounded to the nearest
// half-precision float, clamped to the largest finite one.

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint offset;
    uint len;
};

kernel void store_halves(
    device half* cache [[buffer(0)]],
    const device float* src [[buffer(1)]],
    constant Params& p [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= p.len) {
        return;
    }
    cache[p.offset + i] = half(clamp(src[i], -65504.0f, 65504.0f));
}
