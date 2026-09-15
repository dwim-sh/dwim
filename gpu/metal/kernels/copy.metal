// dst[dst_offset..][..len] = src[src_offset..][..len], as a kernel so that
// copies share the compute encoder with everything else.

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint dst_offset;
    uint src_offset;
    uint len;
};

kernel void copy(
    device float* dst [[buffer(0)]],
    const device float* src [[buffer(1)]],
    constant Params& p [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= p.len) {
        return;
    }
    dst[p.dst_offset + i] = src[p.src_offset + i];
}
