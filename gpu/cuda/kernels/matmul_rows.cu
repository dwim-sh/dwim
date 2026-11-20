// out[t][r] = w[r] · x[t] for block-quantized weights, in single
// precision, on a GPU without the tensor cores `matmul_tile.cu` needs. The weights are tiled as `mod.rs`
// lays them out, 128 rows by 64 columns to a tile. A block takes a group
// of 128 rows and up to `T` tokens: each warp walks every eighth tile of
// the group, a lane taking four rows of it, so that a warp reads a tile as
// one contiguous run and an activation is read once for four rows; the
// warps' sums are added through shared memory.

#define TILE_N 128
#define TILE_K 64
#define ROWS_WARPS 8
// Rows a lane takes.
#define LANE_ROWS 4

// The weight a nibble of a word holds: 8 less than its value.
__device__ __forceinline__ float q4_at(u32 word, u32 shift) {
    return (float)((i32)((word >> shift) & 0xfu) - 8);
}

// The weight a byte of a word holds.
__device__ __forceinline__ float q8_at(u32 word, u32 shift) {
    return (float)((i32)(word << (24 - shift)) >> 24);
}

__device__ __forceinline__ u32 word_of(uint4 v, u32 i) {
    return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

template <int T, bool Q8>
__device__ __forceinline__ void rows_body(
    float* out, const uint4* w, const u16* scales, const float* x, TileParams p, float* sums) {
    // 16-byte words to a row of a tile.
    const u32 ROW_WORDS = Q8 ? 4 : 2;
    const u32 tid = threadIdx.x;
    const u32 lane = tid & 31;
    const u32 warp = tid >> 5;
    u32 group = blockIdx.x;
    u32 t0 = p.first + blockIdx.y * T;
    u32 count = min((u32)T, p.first + p.count - t0);
    // A token past the batch's last is taken as the last and not written.
    const float* px[T];
#pragma unroll
    for (u32 t = 0; t < T; t++) {
        px[t] = x + (size_t)(t0 + min(t, count - 1)) * p.cols;
    }
    float acc[LANE_ROWS][T];
#pragma unroll
    for (u32 r = 0; r < LANE_ROWS; r++) {
#pragma unroll
        for (u32 t = 0; t < T; t++) {
            acc[r][t] = 0.0f;
        }
    }
    for (u32 kt = warp; kt < p.tiles; kt += ROWS_WARPS) {
        size_t tile = (size_t)group * p.tiles + kt;
        const uint4* pw = w + (tile * TILE_N + lane * LANE_ROWS) * ROW_WORDS;
        const u16* ps = scales + tile * (2 * TILE_N) + lane * LANE_ROWS;
        uint4 words[LANE_ROWS][Q8 ? 4 : 2];
#pragma unroll
        for (u32 r = 0; r < LANE_ROWS; r++) {
#pragma unroll
            for (u32 i = 0; i < ROW_WORDS; i++) {
                words[r][i] = pw[r * ROW_WORDS + i];
            }
        }
#pragma unroll
        for (u32 v = 0; v < 2; v++) {
            // The second block of a row's last tile may be padding, past
            // the activations.
            u32 col = kt * TILE_K + v * 32;
            if (col >= p.cols) {
                break;
            }
            uint2 sbits = *(const uint2*)(ps + v * TILE_N);
            float d[LANE_ROWS] = {
                f16_to_f32(sbits.x), f16_to_f32(sbits.x >> 16),
                f16_to_f32(sbits.y), f16_to_f32(sbits.y >> 16),
            };
            float sum[LANE_ROWS][T];
#pragma unroll
            for (u32 r = 0; r < LANE_ROWS; r++) {
#pragma unroll
                for (u32 t = 0; t < T; t++) {
                    sum[r][t] = 0.0f;
                }
            }
            if (Q8) {
                // Word (i, c) of a row holds elements 2i, 2i + 1, 2i + 8
                // and 2i + 9 of the tile's c-th run of 16 columns.
#pragma unroll
                for (u32 c = 2 * v; c < 2 * v + 2; c++) {
#pragma unroll
                    for (u32 i = 0; i < 4; i++) {
                        float2 x0[T], x1[T];
#pragma unroll
                        for (u32 t = 0; t < T; t++) {
                            const float* at = px[t] + kt * TILE_K + c * 16 + 2 * i;
                            x0[t] = *(const float2*)at;
                            x1[t] = *(const float2*)(at + 8);
                        }
#pragma unroll
                        for (u32 r = 0; r < LANE_ROWS; r++) {
                            u32 word = word_of(words[r][i], c);
                            float q0 = q8_at(word, 0), q1 = q8_at(word, 8);
                            float q2 = q8_at(word, 16), q3 = q8_at(word, 24);
#pragma unroll
                            for (u32 t = 0; t < T; t++) {
                                sum[r][t] += q0 * x0[t].x + q1 * x0[t].y + q2 * x1[t].x + q3 * x1[t].y;
                            }
                        }
                    }
                }
            } else {
                // Word (i, v) of a row holds elements 2i, 2i + 8, 2i + 1
                // and 2i + 9 of block v in the low nibbles of its bytes,
                // and the ones 16 after them in the high nibbles.
#pragma unroll
                for (u32 i = 0; i < 4; i++) {
                    float2 xs[T][4];
#pragma unroll
                    for (u32 t = 0; t < T; t++) {
                        const float* at = px[t] + col + 2 * i;
#pragma unroll
                        for (u32 e = 0; e < 4; e++) {
                            xs[t][e] = *(const float2*)(at + 8 * e);
                        }
                    }
#pragma unroll
                    for (u32 r = 0; r < LANE_ROWS; r++) {
                        u32 word = word_of(words[r][i / 2], (i & 1) * 2 + v);
                        float q0 = q4_at(word, 0), q1 = q4_at(word, 16);
                        float q2 = q4_at(word, 8), q3 = q4_at(word, 24);
                        float q4 = q4_at(word, 4), q5 = q4_at(word, 20);
                        float q6 = q4_at(word, 12), q7 = q4_at(word, 28);
#pragma unroll
                        for (u32 t = 0; t < T; t++) {
                            sum[r][t] += q0 * xs[t][0].x + q1 * xs[t][0].y + q2 * xs[t][1].x + q3 * xs[t][1].y
                                + q4 * xs[t][2].x + q5 * xs[t][2].y + q6 * xs[t][3].x + q7 * xs[t][3].y;
                        }
                    }
                }
            }
#pragma unroll
            for (u32 r = 0; r < LANE_ROWS; r++) {
#pragma unroll
                for (u32 t = 0; t < T; t++) {
                    acc[r][t] += d[r] * sum[r][t];
                }
            }
        }
    }
#pragma unroll
    for (u32 r = 0; r < LANE_ROWS; r++) {
#pragma unroll
        for (u32 t = 0; t < T; t++) {
            sums[(warp * TILE_N + lane * LANE_ROWS + r) * T + t] = acc[r][t];
        }
    }
    __syncthreads();
    for (u32 o = tid; o < TILE_N * T; o += ROWS_WARPS * 32) {
        u32 row = group * TILE_N + o / T;
        u32 t = o % T;
        if (row < p.rows && t < count) {
            float v = 0.0f;
#pragma unroll
            for (u32 i = 0; i < ROWS_WARPS; i++) {
                v += sums[i * TILE_N * T + o];
            }
            out[(size_t)(t0 + t) * p.rows + row] = v;
        }
    }
}

#define ROWS_KERNEL(name, T, Q8) \
    extern "C" __global__ void __launch_bounds__(ROWS_WARPS * 32) name( \
        float* out, const uint4* w, const u16* scales, const float* x, TileParams p) { \
        __shared__ float sums[ROWS_WARPS * TILE_N * T]; \
        rows_body<T, Q8>(out, w, scales, x, p, sums); \
    }

ROWS_KERNEL(matmul_rows_q4_0, 1, false)
ROWS_KERNEL(matmul_rows_q8_0, 1, true)
ROWS_KERNEL(matmul_rows8_q4_0, 8, false)
ROWS_KERNEL(matmul_rows8_q8_0, 8, true)
