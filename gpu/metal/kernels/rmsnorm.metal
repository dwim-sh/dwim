// Normalizes each row of x by its root mean square and scales it by the
// weight: one threadgroup per row.

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint dim;
    float eps;
};

kernel void rmsnorm(
    device float* x [[buffer(0)]],
    const device ushort* weight [[buffer(1)]],
    constant Params& p [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint sid [[simdgroup_index_in_threadgroup]],
    uint nsg [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float partial[32];
    device float* v = x + row * p.dim;
    float ss = 0.0f;
    for (uint i = lid; i < p.dim; i += threads) {
        ss += v[i] * v[i];
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
    float scale = rsqrt(total / float(p.dim) + p.eps);
    for (uint i = lid; i < p.dim; i += threads) {
        v[i] *= scale * as_type<float>(uint(weight[i]) << 16);
    }
}
