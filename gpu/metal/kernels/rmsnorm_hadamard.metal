// out = rotate(rmsnorm(x) * weight): one threadgroup of 256 per
// 1024-element block of every row of x, which sums the squares of the whole
// row, then normalizes, scales, and signs its block and transforms it as
// `hadamard.metal` does, by `butterfly.metal`, which the device puts ahead
// of this. The row is read as vectors of four, a fixed number per thread
// with the index clamped rather than the load skipped, so that the loads go
// out together instead of one round trip after another.

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint width;
    float eps;
};

// Vectors a thread reads of the row: rows of up to 8192 elements.
constant uint PER_THREAD = 8;

kernel void rmsnorm_hadamard(
    device float4* out [[buffer(0)]],
    const device float4* x [[buffer(1)]],
    const device float4* weight [[buffer(2)]],
    const device float4* signs [[buffer(3)]],
    constant Params& p [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint sid [[simdgroup_index_in_threadgroup]],
    uint nsg [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float partial[32];
    threadgroup float4 v[2 * THREADS];
    uint blocks = p.width / BLOCK;
    uint vectors = p.width / 4;
    uint row = (group / blocks) * vectors;
    uint block = (group % blocks) * (BLOCK / 4);
    // The thread's vector of the block, its weight and its signs, loaded
    // along with the row for the sum, so that nothing waits on the sum but
    // the arithmetic.
    float4 scaled = x[row + block + lid] * weight[block + lid] * signs[block + lid];
    float ss = 0.0f;
    for (uint k = 0; k < PER_THREAD; k++) {
        uint i = lid + k * THREADS;
        float4 e = x[row + min(i, vectors - 1)];
        ss += i < vectors ? dot(e, e) : 0.0f;
    }
    ss = simd_sum(ss);
    if (lane == 0) {
        partial[sid] = ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint i = 0; i < nsg; i++) {
        total += partial[i];
    }
    float scale = rsqrt(total / float(p.width) + p.eps);
    out[row + block + lid] = butterfly(scaled * scale, lid, lane, v) * (1.0f / 32.0f);
}
