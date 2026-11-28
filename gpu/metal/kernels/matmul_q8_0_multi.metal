// Several Q8_0 matrix multiplications in one grid, for single-token decode
// (VEC != 0) or small batches of up to TOKENS tokens (VEC == 0). Each SIMD
// group takes one row of whichever job owns its threadgroup, eliminating
// serial dispatch barriers between independent projections (such as Q,
// gate, K, V, and rel).

#include <metal_stdlib>
using namespace metal;

#ifndef VEC
#define VEC 0
#endif

constant uint JOBS = 6;
constant uint TOKENS = 4;

struct Params {
    uint first[JOBS];
    uint rows[JOBS];
    uint cols[JOBS];
    uint n[JOBS];
};

static device float* pick_out(
    device float* a, device float* b, device float* c,
    device float* d, device float* e, device float* f, uint j)
{
    switch (j) {
        case 0: return a;
        case 1: return b;
        case 2: return c;
        case 3: return d;
        case 4: return e;
        default: return f;
    }
}

static const device uint4* pick_w(
    const device uint4* a, const device uint4* b, const device uint4* c,
    const device uint4* d, const device uint4* e, const device uint4* f, uint j)
{
    switch (j) {
        case 0: return a;
        case 1: return b;
        case 2: return c;
        case 3: return d;
        case 4: return e;
        default: return f;
    }
}

static const device half* pick_s(
    const device half* a, const device half* b, const device half* c,
    const device half* d, const device half* e, const device half* f, uint j)
{
    switch (j) {
        case 0: return a;
        case 1: return b;
        case 2: return c;
        case 3: return d;
        case 4: return e;
        default: return f;
    }
}

static const device float4* pick_x(
    const device float4* a, const device float4* b, const device float4* c,
    const device float4* d, const device float4* e, const device float4* f, uint j)
{
    switch (j) {
        case 0: return a;
        case 1: return b;
        case 2: return c;
        case 3: return d;
        case 4: return e;
        default: return f;
    }
}

static float4 unpack(uint word) {
    return float4(as_type<char4>(word));
}

kernel void matmul_q8_0_multi(
    device float* o0 [[buffer(0)]],
    device float* o1 [[buffer(1)]],
    device float* o2 [[buffer(2)]],
    device float* o3 [[buffer(3)]],
    device float* o4 [[buffer(4)]],
    device float* o5 [[buffer(5)]],
    const device uint4* w0 [[buffer(6)]],
    const device uint4* w1 [[buffer(7)]],
    const device uint4* w2 [[buffer(8)]],
    const device uint4* w3 [[buffer(9)]],
    const device uint4* w4 [[buffer(10)]],
    const device uint4* w5 [[buffer(11)]],
    const device half* s0 [[buffer(12)]],
    const device half* s1 [[buffer(13)]],
    const device half* s2 [[buffer(14)]],
    const device half* s3 [[buffer(15)]],
    const device half* s4 [[buffer(16)]],
    const device half* s5 [[buffer(17)]],
    const device float4* x0 [[buffer(18)]],
    const device float4* x1 [[buffer(19)]],
    const device float4* x2 [[buffer(20)]],
    const device float4* x3 [[buffer(21)]],
    const device float4* x4 [[buffer(22)]],
    const device float4* x5 [[buffer(23)]],
    constant Params& p [[buffer(24)]],
    uint group [[threadgroup_position_in_grid]],
    uint sid [[simdgroup_index_in_threadgroup]],
    uint nsg [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint width [[threads_per_simdgroup]])
{
    uint j = 0;
    #pragma clang loop unroll(full)
    for (uint i = 1; i < JOBS; i++) {
        j += group >= p.first[i];
    }
    uint r = (group - p.first[j]) * nsg + sid;
    uint rows = p.rows[j];
    if (r >= rows) {
        return;
    }
    uint cols = p.cols[j];
    uint blocks = cols / 32;
    device float* out = pick_out(o0, o1, o2, o3, o4, o5, j);
    const device uint4* w = pick_w(w0, w1, w2, w3, w4, w5, j);
    const device half* scales = pick_s(s0, s1, s2, s3, s4, s5, j);
    const device float4* x = pick_x(x0, x1, x2, x3, x4, x5, j);

#if VEC
    float acc = 0.0f;
    for (uint b = lane; b < blocks; b += width) {
        uint4 lo = w[(r * blocks + b) * 2];
        uint4 hi = w[(r * blocks + b) * 2 + 1];
        float d = float(scales[r * blocks + b]);
        const device float4* xb = x + b * 8;
        float sum = dot(unpack(lo.x), xb[0]) + dot(unpack(lo.y), xb[1])
            + dot(unpack(lo.z), xb[2]) + dot(unpack(lo.w), xb[3])
            + dot(unpack(hi.x), xb[4]) + dot(unpack(hi.y), xb[5])
            + dot(unpack(hi.z), xb[6]) + dot(unpack(hi.w), xb[7]);
        acc += d * sum;
    }
    float total = simd_sum(acc);
    if (lane == 0) {
        out[r] = total;
    }
#else
    uint n = p.n[j];
    float acc[TOKENS] = {};
    for (uint b = lane; b < blocks; b += width) {
        uint4 lo = w[(r * blocks + b) * 2];
        uint4 hi = w[(r * blocks + b) * 2 + 1];
        float d = float(scales[r * blocks + b]);
        float4 q[8] = {
            unpack(lo.x), unpack(lo.y), unpack(lo.z), unpack(lo.w),
            unpack(hi.x), unpack(hi.y), unpack(hi.z), unpack(hi.w),
        };
        for (uint k = 0; k < n; k++) {
            const device float4* xb = x + (k * cols + b * 32) / 4;
            float sum = 0.0f;
            for (uint i = 0; i < 8; i++) {
                sum += dot(q[i], xb[i]);
            }
            acc[k] += d * sum;
        }
    }
    for (uint k = 0; k < n; k++) {
        float total = simd_sum(acc[k]);
        if (lane == 0) {
            out[k * rows + r] = total;
        }
    }
#endif
}
