// out[t][r] = w[r] · x[t]: each SIMD group takes one row of the bf16 weight
// matrix, reading it as 16-byte chunks, eight weights at a time, and taking
// its dot product with each token's activations in turn.

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint rows;
    uint cols;
    uint n;
};

// The four bf16 weights in two words, in memory order: the low half of a
// word first, as little-endian does.
static float4 weights(uint2 pairs) {
    return float4(as_type<float>(pairs.x << 16), as_type<float>(pairs.x & 0xffff0000u),
                  as_type<float>(pairs.y << 16), as_type<float>(pairs.y & 0xffff0000u));
}

kernel void matmul(
    device float* out [[buffer(0)]],
    const device uint4* w [[buffer(1)]],
    const device float4* x [[buffer(2)]],
    constant Params& p [[buffer(3)]],
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
    uint chunks = p.cols / 8;
    const device uint4* row = w + r * chunks;
    for (uint t = 0; t < p.n; t++) {
        const device float4* xt = x + t * (p.cols / 4);
        float acc = 0.0f;
        for (uint c = lane; c < chunks; c += width) {
            uint4 wv = row[c];
            acc += dot(weights(wv.xy), xt[2 * c]) + dot(weights(wv.zw), xt[2 * c + 1]);
        }
        acc = simd_sum(acc);
        if (lane == 0) {
            out[t * p.rows + r] = acc;
        }
    }
}
