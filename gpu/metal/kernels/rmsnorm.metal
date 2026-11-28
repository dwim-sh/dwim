// Normalizes each row of x by its root mean square and scales it by the
// weight: one threadgroup per row. Rows up to 256 elements in fours run in
// a single SIMD group (up to two float4s per lane, no barrier); rows up to
// 4096 in fours cache float4 vectors in registers; other rows use a
// two-pass loop.

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint dim;
    float eps;
};

constant uint THREADS = 256;

kernel void rmsnorm(
    device float* x [[buffer(0)]],
    const device float* weight [[buffer(1)]],
    constant Params& p [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint sid [[simdgroup_index_in_threadgroup]],
    uint nsg [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    if ((p.dim & 3u) == 0 && p.dim <= 256) {
        uint vectors = p.dim / 4;
        device float4* v = (device float4*)x + row * vectors;
        const device float4* w = (const device float4*)weight;
        float4 a0 = lane < vectors ? v[lane] : float4(0.0f);
        float4 a1 = lane + 32 < vectors ? v[lane + 32] : float4(0.0f);
        float total = simd_sum(dot(a0, a0) + dot(a1, a1));
        float scale = rsqrt(total / float(p.dim) + p.eps);
        if (lane < vectors) {
            v[lane] = a0 * (scale * w[lane]);
        }
        if (lane + 32 < vectors) {
            v[lane + 32] = a1 * (scale * w[lane + 32]);
        }
        return;
    }

    threadgroup float partial[32];
    if ((p.dim & 3u) == 0 && p.dim <= 4096) {
        uint vectors = p.dim / 4;
        device float4* v = (device float4*)x + row * vectors;
        const device float4* w = (const device float4*)weight;
        float4 cached[4];
        float ss = 0.0f;
        #pragma clang loop unroll(full)
        for (uint k = 0; k < 4; k++) {
            uint i = lid + k * THREADS;
            cached[k] = i < vectors ? v[i] : float4(0.0f);
            ss += dot(cached[k], cached[k]);
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
        #pragma clang loop unroll(full)
        for (uint k = 0; k < 4; k++) {
            uint i = lid + k * THREADS;
            if (i < vectors) {
                v[i] = cached[k] * (scale * w[i]);
            }
        }
        return;
    }

    device float* v = x + row * p.dim;
    float ss = 0.0f;
    for (uint i = lid; i < p.dim; i += THREADS) {
        float a = v[i];
        ss += a * a;
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
    for (uint i = lid; i < p.dim; i += THREADS) {
        v[i] *= scale * weight[i];
    }
}
