// Out-of-place RMS normalization for up to JOBS buffers in one grid:
// out[j][row] = rmsnorm(x[j][row]) * weight[j]. One threadgroup per
// (row, job), eliminating the copy dispatch before rmsnorm.

#include <metal_stdlib>
using namespace metal;

constant uint THREADS = 256;
constant uint JOBS = 4;

struct Params {
    uint dim;
    float eps;
};

static device float* pick_mut(
    device float* a, device float* b, device float* c, device float* d, uint j)
{
    switch (j) {
        case 0: return a;
        case 1: return b;
        case 2: return c;
        default: return d;
    }
}

static const device float* pick(
    const device float* a, const device float* b, const device float* c, const device float* d, uint j)
{
    switch (j) {
        case 0: return a;
        case 1: return b;
        case 2: return c;
        default: return d;
    }
}

kernel void normalize(
    device float* out0 [[buffer(0)]],
    device float* out1 [[buffer(1)]],
    device float* out2 [[buffer(2)]],
    device float* out3 [[buffer(3)]],
    const device float* x0 [[buffer(4)]],
    const device float* x1 [[buffer(5)]],
    const device float* x2 [[buffer(6)]],
    const device float* x3 [[buffer(7)]],
    const device float* w0 [[buffer(8)]],
    const device float* w1 [[buffer(9)]],
    const device float* w2 [[buffer(10)]],
    const device float* w3 [[buffer(11)]],
    constant Params& p [[buffer(12)]],
    uint2 wg [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint sid [[simdgroup_index_in_threadgroup]],
    uint nsg [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float partial[32];
    uint row = wg.x;
    uint job = wg.y;
    device float* out = pick_mut(out0, out1, out2, out3, job);
    const device float* x = pick(x0, x1, x2, x3, job);
    const device float* weight = pick(w0, w1, w2, w3, job);

    if ((p.dim & 3u) == 0 && p.dim <= 4096) {
        uint vectors = p.dim / 4;
        device float4* dst = (device float4*)out + row * vectors;
        const device float4* src = (const device float4*)x + row * vectors;
        const device float4* w = (const device float4*)weight;
        float4 cached[4];
        float ss = 0.0f;
        #pragma clang loop unroll(full)
        for (uint k = 0; k < 4; k++) {
            uint i = lid + k * THREADS;
            cached[k] = i < vectors ? src[i] : float4(0.0f);
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
                dst[i] = cached[k] * (scale * w[i]);
            }
        }
        return;
    }

    device float* dst = out + row * p.dim;
    const device float* src = x + row * p.dim;
    float ss = 0.0f;
    for (uint i = lid; i < p.dim; i += THREADS) {
        float a = src[i];
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
        dst[i] = src[i] * (scale * weight[i]);
    }
}
