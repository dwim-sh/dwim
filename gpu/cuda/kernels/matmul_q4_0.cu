// out[t][r] = w[r] · x[t] for Q4_0 weights: as `matmul_q8_0.cu`, a warp
// per row and group of eight tokens, eight rows to a block, the warp
// taking four blocks of 32 weights at a time and each lane four of them.
// A block's quants are one 16-byte word: byte j holds element j in its
// low nibble and element j + 16 in its high one, each 8 more than the
// weight, so a lane's four weights are the low or the high nibbles of one
// 32-bit word of the four. The quants and the scales are laid out as
// `quant.rs` says.

// The low nibbles of a word's four bytes, less eight, in memory order.
__device__ __forceinline__ float4 q4_low(u32 word) {
    return make_float4(
        (float)(word & 0xfu) - 8.0f, (float)((word >> 8) & 0xfu) - 8.0f,
        (float)((word >> 16) & 0xfu) - 8.0f, (float)((word >> 24) & 0xfu) - 8.0f);
}

extern "C" __global__ void __launch_bounds__(256) matmul_q4_0(
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
        // The lane's block of each four, and of the block's four words its
        // own, whose low nibbles are elements 0 to 15 and high nibbles 16
        // to 31: the lane's four elements are at four times its place in
        // the eight lanes of the block.
        u32 shift = (lane & 4) ? 4 : 0;
        for (u32 b = lane >> 3; b < blocks; b += 4) {
            u32 i = r * blocks + b;
            float4 q = q4_low(w[i * 4 + (lane & 3)] >> shift);
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
