// out[t][r] = w[r] · x[t] for Q8_0 weights: each SIMD group takes one row,
// its lanes a block of 32 weights each in turn, so that the lanes read
// adjacent blocks. For a single token (VEC != 0), each lane dots its block
// straight into one accumulator without holding a multi-token array in
// registers; for a batch of up to eight tokens, each unpacked block is
// reused across all tokens. The quants and the scales are laid out as
// `quant.rs` says.

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint rows;
    uint cols;
    uint n;
    uint add;
};

#ifndef VEC
#define VEC 0
#endif

// The four signed bytes of a word, in memory order.
static float4 unpack(uint word) {
    return float4(as_type<char4>(word));
}

#if VEC

kernel void matmul_q8_0(
    device float* out [[buffer(0)]],
    const device uint4* w [[buffer(1)]],
    const device half* scales [[buffer(2)]],
    const device float4* x [[buffer(3)]],
    constant Params& p [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],
    uint sid [[simdgroup_index_in_threadgroup]],
    uint nsg [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint width [[threads_per_simdgroup]])
{
    uint r = group * nsg + sid;
    if (r >= p.rows) {
        return;
    }
    uint blocks = p.cols / 32;
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
        out[r] = p.add != 0 ? out[r] + total : total;
    }
}

#else

// Tokens taken at once.
constant uint TOKENS = 8;

kernel void matmul_q8_0(
    device float* out [[buffer(0)]],
    const device uint4* w [[buffer(1)]],
    const device half* scales [[buffer(2)]],
    const device float4* x [[buffer(3)]],
    constant Params& p [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],
    uint sid [[simdgroup_index_in_threadgroup]],
    uint nsg [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint width [[threads_per_simdgroup]])
{
    uint r = group * nsg + sid;
    if (r >= p.rows) {
        return;
    }
    uint blocks = p.cols / 32;
    for (uint t0 = 0; t0 < p.n; t0 += TOKENS) {
        float acc[TOKENS] = {};
        for (uint b = lane; b < blocks; b += width) {
            uint4 lo = w[(r * blocks + b) * 2];
            uint4 hi = w[(r * blocks + b) * 2 + 1];
            float d = float(scales[r * blocks + b]);
            float4 q[8] = {
                unpack(lo.x), unpack(lo.y), unpack(lo.z), unpack(lo.w),
                unpack(hi.x), unpack(hi.y), unpack(hi.z), unpack(hi.w),
            };
            for (uint k = 0; k < TOKENS; k++) {
                if (t0 + k >= p.n) {
                    break;
                }
                const device float4* xb = x + ((t0 + k) * p.cols + b * 32) / 4;
                float sum = 0.0f;
                for (uint i = 0; i < 8; i++) {
                    sum += dot(q[i], xb[i]);
                }
                acc[k] += d * sum;
            }
        }
        for (uint k = 0; k < TOKENS; k++) {
            float total = simd_sum(acc[k]);
            if (lane == 0 && t0 + k < p.n) {
                uint at = (t0 + k) * p.rows + r;
                out[at] = p.add != 0 ? out[at] + total : total;
            }
        }
    }
}

#endif
