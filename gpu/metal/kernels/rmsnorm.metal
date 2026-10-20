// Normalizes each row of x by its root mean square and scales it by the
// weight: one threadgroup per row. The row is read as vectors of four, a
// fixed number per thread with the index clamped rather than the load
// skipped, so that the loads go out together instead of one round trip
// after another.

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint dim;
    float eps;
};

constant uint THREADS = 256;
// Vectors a thread reads: rows of up to 8192 elements.
constant uint PER_THREAD = 8;

kernel void rmsnorm(
    device float4* x [[buffer(0)]],
    const device float4* weight [[buffer(1)]],
    constant Params& p [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint sid [[simdgroup_index_in_threadgroup]],
    uint nsg [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float partial[32];
    uint vectors = p.dim / 4;
    device float4* v = x + row * vectors;
    float4 vs[PER_THREAD];
    float ss = 0.0f;
    for (uint k = 0; k < PER_THREAD; k++) {
        uint i = lid + k * THREADS;
        vs[k] = v[min(i, vectors - 1)];
        ss += i < vectors ? dot(vs[k], vs[k]) : 0.0f;
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
    for (uint k = 0; k < PER_THREAD; k++) {
        uint i = lid + k * THREADS;
        if (i < vectors) {
            v[i] = vs[k] * scale * weight[i];
        }
    }
}
