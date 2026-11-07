// out[t][r] = w[r] · x[t] for Q8_0 weights: a warp per row and group of
// eight tokens, eight rows to a block. The warp takes four blocks of 32
// weights at a time, each lane four weights of them, so that the lanes
// read the quants, the scales, and the activations as contiguous runs: a
// lane reads its four quants as one word and its block's scale as a half,
// and multiplies them into the four activations of each of the eight
// tokens, so that a word is read and unpacked once per eight tokens rather
// than once per token; the lanes' sums are added across the warp. The
// quants and the scales are laid out as `quant.rs` says.

// Tokens a warp takes, and rows a block does.
#define Q_TOKENS 8
#define Q_ROWS 8

// The four signed bytes of a word, in memory order.
__device__ __forceinline__ float4 q8_unpack(u32 word) {
    return make_float4(
        (float)((i32)(word << 24) >> 24), (float)((i32)(word << 16) >> 24),
        (float)((i32)(word << 8) >> 24), (float)((i32)word >> 24));
}

// Adds the dot products of the four weights `q`, of a block whose scale is
// `d`, with the activations of the tokens from `t0`, `count` of them, at
// column `c` of `p.cols`, into `acc`.
__device__ __forceinline__ void accumulate(
    float acc[Q_TOKENS], float4 q, float d, const float4* x, u32 t0, u32 count, u32 c, u32 cols) {
#pragma unroll
    for (u32 t = 0; t < Q_TOKENS; t++) {
        if (t < count) {
            acc[t] += d * dot4(q, x[((t0 + t) * cols + c) / 4]);
        }
    }
}

// Writes the warp's sums for its row `r` of `rows` and its tokens from
// `t0`, `count` of them, which every lane holds after `warp_sum`.
__device__ __forceinline__ void store_row(float* out, const float acc[Q_TOKENS], u32 r, u32 t0, u32 count, u32 rows) {
    const u32 lane = threadIdx.x & 31;
    if (lane < count && r < rows) {
        float v = 0.0f;
#pragma unroll
        for (u32 t = 0; t < Q_TOKENS; t++) {
            if (t == lane) {
                v = acc[t];
            }
        }
        out[(t0 + lane) * rows + r] = v;
    }
}

extern "C" __global__ void __launch_bounds__(256) matmul_q8_0(
    float* out, const u32* w, const u16* scales, const float4* x, MatmulParams p) {
    const u32 lane = threadIdx.x & 31;
    const u32 warp = threadIdx.x >> 5;
    u32 token_groups = (p.n + Q_TOKENS - 1) / Q_TOKENS;
    u32 r = ((blockIdx.y / token_groups) * p.stride + blockIdx.x) * Q_ROWS + warp;
    u32 t0 = (blockIdx.y % token_groups) * Q_TOKENS;
    u32 count = min((u32)Q_TOKENS, p.n - t0);
    u32 blocks = p.cols / 32;
    float acc[Q_TOKENS];
#pragma unroll
    for (u32 t = 0; t < Q_TOKENS; t++) {
        acc[t] = 0.0f;
    }
    if (r < p.rows) {
        // The lane's block of each four, and its word of the block's eight.
        for (u32 b = lane >> 3; b < blocks; b += 4) {
            u32 i = r * blocks + b;
            float4 q = q8_unpack(w[i * 8 + (lane & 7)]);
            float d = f16_to_f32(scales[i]);
            accumulate(acc, q, d, x, t0, count, b * 32 + 4 * (lane & 7), p.cols);
        }
    }
#pragma unroll
    for (u32 t = 0; t < Q_TOKENS; t++) {
        acc[t] = warp_sum(acc[t]);
    }
    store_row(out, acc, r, t0, count, p.rows);
}
