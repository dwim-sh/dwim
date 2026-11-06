// out[t][r] = w[r] · x[t] for Q4_0 weights: as `matmul_q8_0.metal`, each
// SIMD group taking one row and its lanes a block of 32 weights each in
// turn, up to eight tokens at once. A block's quants are one 16-byte word:
// byte j holds element j in its low nibble and element j + 16 in its high
// one, each 8 more than the weight. The quants and the scales are laid out
// as `quant.rs` says.

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint rows;
    uint cols;
    uint n;
};

// Tokens taken at once.
constant uint TOKENS = 8;

// The low nibbles of a word's four bytes, less eight, in memory order.
static float4 low(uint word) {
    return float4(float(word & 0xfu), float((word >> 8) & 0xfu), float((word >> 16) & 0xfu), float((word >> 24) & 0xfu)) - 8.0f;
}

// The high nibbles, likewise.
static float4 high(uint word) {
    return float4(float((word >> 4) & 0xfu), float((word >> 12) & 0xfu), float((word >> 20) & 0xfu), float(word >> 28)) - 8.0f;
}

kernel void matmul_q4_0(
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
            uint4 v = w[r * blocks + b];
            float d = float(scales[r * blocks + b]);
            // Elements 0 to 15 from the low nibbles, 16 to 31 from the high.
            float4 q[8] = {
                low(v.x), low(v.y), low(v.z), low(v.w),
                high(v.x), high(v.y), high(v.z), high(v.w),
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
                out[(t0 + k) * p.rows + r] = total;
            }
        }
    }
}
