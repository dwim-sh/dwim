// A matrix multiplication by each token's experts in single precision,
// on a GPU without the tensor cores the experts' tile kernels need: what
// they compute, a block to a pair of a token and an expert and a run of
// 128 of the expert's rows. Each warp walks every eighth tile of the
// rows, a lane taking four of them, and the warps' sums are added through
// shared memory. The weights are tiled as `mod.rs` lays them out, in
// NVFP4 or `Q8_0`, all the experts' rows as one matrix: an expert's rows
// need not start where a group of the tiles does, so each row is found
// in its own.

// The weight an NVFP4 nibble of a word holds, doubled.
__device__ __forceinline__ float fp4_at(u32 word, u32 shift) {
    u32 n = word >> shift;
    float v = (float)((0xc8643210u >> ((n & 7u) * 4)) & 0xfu);
    return n & 8u ? -v : v;
}

// The scale of an NVFP4 block from its bits, halved.
__device__ __forceinline__ float fp4_scale_of(u32 bits) {
    u32 e = (bits >> 3) & 0xfu;
    float m = (float)(bits & 7u);
    float raw = e != 0 ? (1.0f + m * 0.125f) * exp2f((float)e - 8.0f) : m * 0.0009765625f;
    return bits == 0 || bits == 0x7fu ? 0.0f : raw;
}

template <bool Q8>
__device__ __forceinline__ void expert_rows_body(
    float* out, const uint4* w, const unsigned char* scales, const float* x, const float* ids,
    const float* scale, ExpertParams p, float* sums) {
    const u32 ROW_WORDS = Q8 ? 4 : 2;
    const u32 tid = threadIdx.x;
    const u32 lane = tid & 31;
    const u32 warp = tid >> 5;
    u32 pair = blockIdx.x;
    u32 group = blockIdx.y;
    u32 expert = (u32)ids[pair];
    const float* px = x + (size_t)(pair / p.share) * p.cols;
    float acc[LANE_ROWS];
#pragma unroll
    for (u32 r = 0; r < LANE_ROWS; r++) {
        acc[r] = 0.0f;
    }
    for (u32 kt = warp; kt < p.tiles; kt += ROWS_WARPS) {
        // The tile's activations, which every row of the lane's takes.
        float xs[TILE_K];
#pragma unroll
        for (u32 e = 0; e < TILE_K; e += 4) {
            float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (kt * TILE_K + e < p.cols) {
                v = *(const float4*)(px + kt * TILE_K + e);
            }
            xs[e] = v.x;
            xs[e + 1] = v.y;
            xs[e + 2] = v.z;
            xs[e + 3] = v.w;
        }
#pragma unroll
        for (u32 r = 0; r < LANE_ROWS; r++) {
            // The row among the expert's, and among all the experts'.
            u32 own = group * TILE_N + lane * LANE_ROWS + r;
            if (own >= p.rows) {
                continue;
            }
            size_t whole = (size_t)expert * p.rows + own;
            u32 row = whole % TILE_N;
            size_t tile = whole / TILE_N * p.tiles + kt;
            const uint4* pw = w + (tile * TILE_N + row) * ROW_WORDS;
            const unsigned char* ps = scales + tile * (4 * TILE_N);
            if (Q8) {
                // Word (i, c) of a row holds elements 2i, 2i + 1, 2i + 8
                // and 2i + 9 of the tile's c-th run of 16 columns.
                const u16* halves = (const u16*)ps;
#pragma unroll
                for (u32 c = 0; c < 4; c++) {
                    float sum = 0.0f;
#pragma unroll
                    for (u32 i = 0; i < 4; i++) {
                        u32 word = word_of(pw[i], c);
                        const float* at = xs + c * 16 + 2 * i;
                        sum += q8_at(word, 0) * at[0] + q8_at(word, 8) * at[1]
                            + q8_at(word, 16) * at[8] + q8_at(word, 24) * at[9];
                    }
                    acc[r] += f16_to_f32(halves[(c / 2) * TILE_N + row]) * sum;
                }
            } else {
                // Word (i, v) of a row holds elements 2i, 2i + 8, 2i + 1
                // and 2i + 9 of the tile's 32 columns v in the low
                // nibbles of its bytes, and the ones 16 after them in the
                // high nibbles: a scale to each 16.
#pragma unroll
                for (u32 v = 0; v < 2; v++) {
                    float lo = 0.0f, hi = 0.0f;
#pragma unroll
                    for (u32 i = 0; i < 4; i++) {
                        u32 word = word_of(pw[i / 2], (i & 1) * 2 + v);
                        const float* at = xs + v * 32 + 2 * i;
                        lo += fp4_at(word, 0) * at[0] + fp4_at(word, 8) * at[8]
                            + fp4_at(word, 16) * at[1] + fp4_at(word, 24) * at[9];
                        hi += fp4_at(word, 4) * at[16] + fp4_at(word, 12) * at[24]
                            + fp4_at(word, 20) * at[17] + fp4_at(word, 28) * at[25];
                    }
                    acc[r] += fp4_scale_of(ps[(2 * v) * TILE_N + row]) * lo
                        + fp4_scale_of(ps[(2 * v + 1) * TILE_N + row]) * hi;
                }
            }
        }
    }
#pragma unroll
    for (u32 r = 0; r < LANE_ROWS; r++) {
        sums[warp * TILE_N + lane * LANE_ROWS + r] = acc[r];
    }
    __syncthreads();
    if (tid < TILE_N && group * TILE_N + tid < p.rows) {
        float v = 0.0f;
#pragma unroll
        for (u32 i = 0; i < ROWS_WARPS; i++) {
            v += sums[i * TILE_N + tid];
        }
        out[(size_t)pair * p.rows + group * TILE_N + tid] = v * scale[expert];
    }
}

#define EXPERT_ROWS_KERNEL(name, Q8) \
    extern "C" __global__ void __launch_bounds__(ROWS_WARPS * 32) name( \
        float* out, const uint4* w, const unsigned char* scales, const float* x, const float* ids, \
        const float* scale, ExpertParams p) { \
        __shared__ float sums[ROWS_WARPS * TILE_N]; \
        expert_rows_body<Q8>(out, w, scales, x, ids, scale, p, sums); \
    }

EXPERT_ROWS_KERNEL(expert_rows_nvfp4, false)
EXPERT_ROWS_KERNEL(expert_rows_q8_0, true)
