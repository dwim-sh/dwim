// Rotates each 1024-element block of every row of x by the normalized
// Walsh-Hadamard transform, multiplying by the signs before it (the forward
// rotation) or after it (the inverse): one threadgroup of 256 per block,
// the transform and the block's constants those of `butterfly.metal`,
// which the device puts ahead of this.

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint width;
    uint inverse;
};

kernel void hadamard(
    device float4* x [[buffer(0)]],
    const device float4* signs [[buffer(1)]],
    constant Params& p [[buffer(2)]],
    uint group [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float4 v[2 * THREADS];
    uint blocks = p.width / BLOCK;
    uint base = ((group / blocks) * p.width + (group % blocks) * BLOCK) / 4 + lid;
    uint sbase = (group % blocks) * BLOCK / 4 + lid;
    float4 e = x[base];
    if (p.inverse == 0) {
        e *= signs[sbase];
    }
    e = butterfly(e, lid, lane, v) * (1.0f / 32.0f);
    if (p.inverse != 0) {
        e *= signs[sbase];
    }
    x[base] = e;
}
