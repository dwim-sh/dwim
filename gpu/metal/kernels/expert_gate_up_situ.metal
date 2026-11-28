// Fused decode expert gate_up matmul + SiTU activation: each sub-group
// computes both row r (gate) and row r + width (up) of one (token, expert)
// pair in a single pass over the token's input vector x, then applies SiTU
// directly in registers and writes act[pair][r]. Eliminates the intermediate
// [pairs, 2 * width] buffer, halves input loads, and removes the situ dispatch.

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint width;
    uint cols;
    uint k;
    float beta;
    float linear_beta;
};

static float safe_tanh(float x) {
    return tanh(clamp(x, -15.0f, 15.0f));
}

#if BITS == 4

constant uint BLOCK = 16;
constant uint LANES = 8;

static float fp4(uint n) {
    float v = float((0xc8643210u >> ((n & 7u) * 4u)) & 0xfu);
    return (n & 8u) != 0 ? -v : v;
}

static float4 low(uint word) {
    return float4(fp4(word), fp4(word >> 8), fp4(word >> 16), fp4(word >> 24));
}

static float4 high(uint word) {
    return float4(fp4(word >> 4), fp4(word >> 12), fp4(word >> 20), fp4(word >> 28));
}

static float scale_of(uint bits) {
    uint e = (bits >> 3) & 0xfu;
    uint m = bits & 7u;
    float raw = e != 0 ? as_type<float>(((e + 119u) << 23) | (m << 20))
                       : float(m) * (1.0f / 1024.0f);
    return (bits == 0 || bits == 0x7fu) ? 0.0f : raw;
}

static float2 block_dot2(
    const device uint2* w, const device uchar* scales, const device float4* x,
    uint row_g, uint row_u, uint blocks, uint b)
{
    const device float4* xb = x + b * 4;
    float4 x0 = xb[0], x1 = xb[1], x2 = xb[2], x3 = xb[3];
    uint2 vg = w[row_g * blocks + b];
    uint2 vu = w[row_u * blocks + b];
    float sg = dot(low(vg.x), x0) + dot(low(vg.y), x1) + dot(high(vg.x), x2) + dot(high(vg.y), x3);
    float su = dot(low(vu.x), x0) + dot(low(vu.y), x1) + dot(high(vu.x), x2) + dot(high(vu.y), x3);
    return float2(scale_of(scales[row_g * blocks + b]) * sg, scale_of(scales[row_u * blocks + b]) * su);
}

#define WORDS uint2
#define SCALE uchar

#else

constant uint BLOCK = 32;
constant uint LANES = 32;

static float4 unpack(uint word) {
    return float4(as_type<char4>(word));
}

static float2 block_dot2(
    const device uint4* w, const device half* scales, const device float4* x,
    uint row_g, uint row_u, uint blocks, uint b)
{
    const device float4* xb = x + b * 8;
    float4 x0 = xb[0], x1 = xb[1], x2 = xb[2], x3 = xb[3];
    float4 x4 = xb[4], x5 = xb[5], x6 = xb[6], x7 = xb[7];
    uint4 g_lo = w[(row_g * blocks + b) * 2];
    uint4 g_hi = w[(row_g * blocks + b) * 2 + 1];
    uint4 u_lo = w[(row_u * blocks + b) * 2];
    uint4 u_hi = w[(row_u * blocks + b) * 2 + 1];
    float sg = dot(unpack(g_lo.x), x0) + dot(unpack(g_lo.y), x1) + dot(unpack(g_lo.z), x2) + dot(unpack(g_lo.w), x3)
        + dot(unpack(g_hi.x), x4) + dot(unpack(g_hi.y), x5) + dot(unpack(g_hi.z), x6) + dot(unpack(g_hi.w), x7);
    float su = dot(unpack(u_lo.x), x0) + dot(unpack(u_lo.y), x1) + dot(unpack(u_lo.z), x2) + dot(unpack(u_lo.w), x3)
        + dot(unpack(u_hi.x), x4) + dot(unpack(u_hi.y), x5) + dot(unpack(u_hi.z), x6) + dot(unpack(u_hi.w), x7);
    return float2(float(scales[row_g * blocks + b]) * sg, float(scales[row_u * blocks + b]) * su);
}

#define WORDS uint4
#define SCALE half

#endif

constant uint ROWS_PER_SIMD = 32 / LANES;

kernel void expert_gate_up_situ(
    device float* act [[buffer(0)]],
    const device WORDS* w [[buffer(1)]],
    const device SCALE* scales [[buffer(2)]],
    const device float4* x [[buffer(3)]],
    const device float* ids [[buffer(4)]],
    const device float* scale [[buffer(5)]],
    constant Params& p [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint sid [[simdgroup_index_in_threadgroup]],
    uint nsg [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint sub = lane / LANES;
    uint sublane = lane % LANES;
    uint r = (group.x * nsg + sid) * ROWS_PER_SIMD + sub;
    uint pair = group.y;
    if (r >= p.width) {
        return;
    }
    uint expert = uint(ids[pair]);
    uint row_g = expert * (2 * p.width) + r;
    uint row_u = row_g + p.width;
    const device float4* xr = x + (pair / p.k) * (p.cols / 4);
    uint blocks = p.cols / BLOCK;
    float2 acc = float2(0.0f);
    for (uint b = sublane; b < blocks; b += LANES) {
        acc += block_dot2(w, scales, xr, row_g, row_u, blocks, b);
    }
    #pragma clang loop unroll(full)
    for (ushort s = LANES / 2; s > 0; s >>= 1) {
        acc += simd_shuffle_xor(acc, s);
    }
    if (sublane == 0) {
        float sc = scale[expert];
        float g = sc * acc.x;
        float u = sc * acc.y;
        act[pair * p.width + r] = (p.beta * p.linear_beta) * safe_tanh(g / p.beta) / (1.0f + exp(-g))
            * safe_tanh(u / p.linear_beta);
    }
}
