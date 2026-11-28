// Fused decode expert down matmul + weighted_sum: each sub-group computes
// row r of token t across all k picked experts, accumulating each expert's
// weighted output directly in a register and writing out[t][r] once.
// Eliminates the intermediate [n * k, rows] buffer and weighted_sum dispatch.

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint rows;
    uint cols;
    uint k;
};

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

static float block_dot(const device uint2* w, const device uchar* scales, const device float4* x, uint row, uint blocks, uint b) {
    uint2 v = w[row * blocks + b];
    const device float4* xb = x + b * 4;
    float sum = dot(low(v.x), xb[0]) + dot(low(v.y), xb[1]) + dot(high(v.x), xb[2]) + dot(high(v.y), xb[3]);
    return scale_of(scales[row * blocks + b]) * sum;
}

#define WORDS uint2
#define SCALE uchar

#else

constant uint BLOCK = 32;
constant uint LANES = 32;

static float4 unpack(uint word) {
    return float4(as_type<char4>(word));
}

static float block_dot(const device uint4* w, const device half* scales, const device float4* x, uint row, uint blocks, uint b) {
    uint4 lo = w[(row * blocks + b) * 2];
    uint4 hi = w[(row * blocks + b) * 2 + 1];
    const device float4* xb = x + b * 8;
    float sum = dot(unpack(lo.x), xb[0]) + dot(unpack(lo.y), xb[1]) + dot(unpack(lo.z), xb[2]) + dot(unpack(lo.w), xb[3])
        + dot(unpack(hi.x), xb[4]) + dot(unpack(hi.y), xb[5]) + dot(unpack(hi.z), xb[6]) + dot(unpack(hi.w), xb[7]);
    return float(scales[row * blocks + b]) * sum;
}

#define WORDS uint4
#define SCALE half

#endif

constant uint ROWS_PER_SIMD = 32 / LANES;

kernel void expert_down_sum(
    device float* out [[buffer(0)]],
    const device WORDS* w [[buffer(1)]],
    const device SCALE* scales [[buffer(2)]],
    const device float4* x [[buffer(3)]],
    const device float* ids [[buffer(4)]],
    const device float* scale [[buffer(5)]],
    const device float* shares [[buffer(6)]],
    constant Params& p [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint sid [[simdgroup_index_in_threadgroup]],
    uint nsg [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint sub = lane / LANES;
    uint sublane = lane % LANES;
    uint r = (group.x * nsg + sid) * ROWS_PER_SIMD + sub;
    uint t = group.y;
    if (r >= p.rows) {
        return;
    }
    uint blocks = p.cols / BLOCK;
    float total = 0.0f;
    for (uint e = 0; e < p.k; e++) {
        uint pair = t * p.k + e;
        uint expert = uint(ids[pair]);
        uint row = expert * p.rows + r;
        const device float4* xr = x + pair * (p.cols / 4);
        float acc = 0.0f;
        for (uint b = sublane; b < blocks; b += LANES) {
            acc += block_dot(w, scales, xr, row, blocks, b);
        }
        #pragma clang loop unroll(full)
        for (ushort s = LANES / 2; s > 0; s >>= 1) {
            acc += simd_shuffle_xor(acc, s);
        }
        if (sublane == 0) {
            total += shares[pair] * scale[expert] * acc;
        }
    }
    if (sublane == 0) {
        out[t * p.rows + r] = total;
    }
}
