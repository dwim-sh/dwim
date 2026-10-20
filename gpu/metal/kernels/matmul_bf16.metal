// out[t][r] = w[r] · x[t]: each SIMD group takes one row of the bf16 weight
// matrix, reading it as 16-byte chunks, eight weights at a time, and taking
// each chunk's dot product with eight tokens' activations at once, so that
// a batch of tokens reads the row once per eight. Each eight tokens are a
// row of the grid of threadgroups.

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint rows;
    uint cols;
    uint n;
};

// Tokens a SIMD group takes at once.
constant uint TOKENS = 8;

// The four bf16 weights in two words, in memory order: the low half of a
// word first, as little-endian does.
static float4 weights(uint2 pairs) {
    return float4(as_type<float>(pairs.x << 16), as_type<float>(pairs.x & 0xffff0000u),
                  as_type<float>(pairs.y << 16), as_type<float>(pairs.y & 0xffff0000u));
}

kernel void matmul_bf16(
    device float* out [[buffer(0)]],
    const device uint4* w [[buffer(1)]],
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
    uint chunks = p.cols / 8;
    const device uint4* row = w + r * chunks;
    uint count = min(TOKENS, p.n - t0);
    float acc[TOKENS] = {};
    for (uint c = lane; c < chunks; c += width) {
        uint4 wv = row[c];
        float4 lo = weights(wv.xy);
        float4 hi = weights(wv.zw);
        for (uint t = 0; t < count; t++) {
            const device float4* xt = x + (t0 + t) * (p.cols / 4);
            acc[t] += dot(lo, xt[2 * c]) + dot(hi, xt[2 * c + 1]);
        }
    }
    for (uint t = 0; t < count; t++) {
        float total = simd_sum(acc[t]);
        if (lane == 0) {
            out[(t0 + t) * p.rows + r] = total;
        }
    }
}
