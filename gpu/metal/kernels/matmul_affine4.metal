// out[t][r] = w[r] · x[t] for weights four bits each, in groups of 64 with
// a scale and a bias, as `affine4.rs` packs them: each SIMD group takes one
// row, its lanes reading the row's codes as 16-byte chunks of 32, and
// taking each chunk's dot product with eight tokens' activations at once,
// as `matmul_bf16.metal` does, a group's weights entering as its scale
// times the sum of the codes' products plus its bias times the sum of the
// activations. Each eight tokens are a row of the grid of threadgroups.

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint rows;
    uint cols;
    uint n;
};

// Tokens a SIMD group takes at once.
constant uint TOKENS = 8;

// The eight codes of a word, in memory order, as floats.
static void codes(uint word, thread float4& lo, thread float4& hi) {
    lo = float4(word & 15, (word >> 4) & 15, (word >> 8) & 15, (word >> 12) & 15);
    hi = float4((word >> 16) & 15, (word >> 20) & 15, (word >> 24) & 15, word >> 28);
}

kernel void matmul_affine4(
    device float* out [[buffer(0)]],
    const device uchar* w [[buffer(1)]],
    const device float4* x [[buffer(2)]],
    constant Params& p [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint sid [[simdgroup_index_in_threadgroup]],
    uint nsg [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint width [[threads_per_simdgroup]])
{
    uint r = group.x * nsg + sid;
    uint t0 = group.y * TOKENS;
    if (r >= p.rows) {
        return;
    }
    uint row_bytes = p.cols / 2 + p.cols / 16;
    const device uint4* chunks = (const device uint4*)(w + r * row_bytes);
    const device half2* groups = (const device half2*)(w + r * row_bytes + p.cols / 2);
    uint count = min(TOKENS, p.n - t0);
    float acc[TOKENS] = {};
    for (uint c = lane; c < p.cols / 32; c += width) {
        uint4 q = chunks[c];
        half2 sb = groups[c / 2];
        float scale = sb.x;
        float bias = sb.y;
        for (uint t = 0; t < count; t++) {
            const device float4* xt = x + (t0 + t) * (p.cols / 4) + c * 8;
            float products = 0.0f;
            float sum = 0.0f;
            for (uint k = 0; k < 4; k++) {
                float4 lo, hi;
                codes(q[k], lo, hi);
                float4 x0 = xt[2 * k];
                float4 x1 = xt[2 * k + 1];
                products += dot(lo, x0) + dot(hi, x1);
                sum += dot(x0, float4(1.0f)) + dot(x1, float4(1.0f));
            }
            acc[t] += scale * products + bias * sum;
        }
    }
    for (uint t = 0; t < count; t++) {
        float total = simd_sum(acc[t]);
        if (lane == 0) {
            out[(t0 + t) * p.rows + r] = total;
        }
    }
}
