// out[t][r] = w[r] · x[t] for a batch of tokens and block-quantized
// weights, on the tensor cores: the activations as half-precision floats,
// the weights each times its block's scale and rounded to a half, and the
// sums in single precision. What is written is the sums, as floats or as
// halves, or the sums added to what `out` holds, or `silu(sum) * other`
// as halves, by the parameters' mode: the feed-forward network's
// activation and the residual stream's addition without a pass of their
// own over the activations. The
// weights are tiled as `mod.rs` lays them out, 128 rows by 64 columns to
// a tile, each lane's words of a tile side by side. Weights in NVFP4 are
// multiplied the same way, by the experts' kernels at the end.
//
// A block of four warps takes a group of 128 rows and `16 * MT` tokens,
// and walks the row's tiles: the threads copy a tile's weights and scales
// and the tokens' activations for its columns into shared memory, a few
// tiles ahead of the one being multiplied, without waiting for them; then
// each warp multiplies 32 rows by the tokens of it, 16 tokens by 8 rows
// by 16 columns at a time. Four warps to a block rather than eight, and
// stages small enough that several blocks share a multiprocessor, keep
// the tensor cores busiest.
// The blocks of a group of rows come one after another in the grid, so
// that a group's weights are read from memory once for the whole batch.
//
// The scheme is TensorFold's (`licenses/tensorfold.md`).

#if __CUDA_ARCH__ >= 800

// Bytes of a token's activations for a tile's columns.
#define TILE_X (TILE_K * 2)

__device__ __forceinline__ u32 shared_address(const void* p) {
    return (u32)__cvta_generic_to_shared(p);
}

// Starts a copy of 16 bytes into shared memory, or of zeros unless `read`.
__device__ __forceinline__ void copy16(void* dst, const void* src, bool read) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
        :: "r"(shared_address(dst)), "l"(src), "r"(read ? 16 : 0) : "memory");
}

// Marks the copies started since the last mark as a group.
__device__ __forceinline__ void commit_copies() {
    asm volatile("cp.async.commit_group;" ::: "memory");
}

// Waits until all but the last `N` groups of copies have landed.
template <int N>
__device__ __forceinline__ void wait_copies() {
    asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory");
}

// Four 8 by 8 matrices of halves from shared memory, as the tensor cores
// take a 16 by 16 one: lanes 8m to 8m + 7 give the addresses of matrix m's
// rows.
__device__ __forceinline__ void load_matrix(u32 (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(shared_address(p)));
}

// d += a · b: 16 by 16 halves times 16 by 8, summed in single precision.
__device__ __forceinline__ void multiply_add(float (&d)[4], const u32 (&a)[4], u32 b0, u32 b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// The pair of halves `bits`, which hold 1024 plus an integer each in the
// integer's place, less `less`, times the pair `scale`: the integers as
// halves, exactly, and then each product rounded once.
__device__ __forceinline__ u32 scaled_pair(u32 bits, u32 less, u32 scale) {
    u32 r;
    asm("{ .reg .b32 t; sub.rn.f16x2 t, %1, %2; mul.rn.f16x2 %0, t, %3; }"
        : "=r"(r) : "r"(bits | 0x64006400u), "r"(less), "r"(scale));
    return r;
}

// Two Q4_0 weights, the nibbles at `shift` of a word's halves, scaled.
__device__ __forceinline__ u32 q4_pair(u32 word, u32 shift, u32 scale) {
    return scaled_pair((word >> shift) & 0x000f000fu, 0x64086408u, scale);
}

// Two Q8_0 weights, the bytes at `shift` and after it, scaled.
__device__ __forceinline__ u32 q8_pair(u32 word, u32 shift, u32 scale) {
    u32 w = word >> shift;
    return scaled_pair(((w & 0xffu) | ((w & 0xff00u) << 8)) ^ 0x00800080u, 0x64806480u, scale);
}

// The pair of halves an NVFP4 scale's bits stand for, halved as the
// weights' values are doubled: four bits of exponent, biased by 7, over
// three of mantissa, which a half holds exactly; zero and the bits of a
// NaN scale a block to nothing.
__device__ __forceinline__ u32 fp4_scale(u32 bits) {
    u32 e = (bits >> 3) & 0xfu;
    u32 m = bits & 7u;
    u32 half = e != 0 ? ((e + 7) << 10) | (m << 7) : f32_to_f16((float)m * 0.0009765625f);
    return bits == 0 || bits == 0x7fu ? 0 : half * 0x10001u;
}

// Two NVFP4 weights, the nibbles at `shift` of a word's halves, scaled:
// a nibble's low three bits pick its magnitude, doubled, of the eight
// there are, a nibble of the constant each, and its high bit is its sign.
// A weight times its scale is a half exactly.
__device__ __forceinline__ u32 fp4_pair(u32 word, u32 shift, u32 scale) {
    u32 n = word >> shift;
    u32 lo = (0xc8643210u >> ((n & 7u) * 4)) & 0xfu;
    u32 hi = (0xc8643210u >> (((n >> 16) & 7u) * 4)) & 0xfu;
    u32 sign = (n & 0x00080008u) << 12;
    u32 r;
    asm("{ .reg .b32 t; sub.rn.f16x2 t, %1, %2; xor.b32 t, t, %3; mul.rn.f16x2 %0, t, %4; }"
        : "=r"(r) : "r"(lo | (hi << 16) | 0x64006400u), "r"(0x64006400u), "r"(sign), "r"(scale));
    return r;
}

// The formats of the tiled weights.
#define TILE_Q4_0 0
#define TILE_Q8_0 1
#define TILE_NVFP4 2
// Halves as they are, a row's pairs of a tile in the order a lane takes
// them, with no scales: what weights that are not quantized are tiled as.
#define TILE_F16 3

// The place in a token's row of a stage of its `chunk` of 16 bytes: the
// chunks are turned by the row, so that the eight rows `load_matrix` reads
// at once are in different banks of shared memory.
__device__ __forceinline__ u32 swizzle(u32 row, u32 chunk) {
    return row * TILE_X + ((chunk ^ (row & 7)) * 16);
}

// The body of the tile kernels, of weights in `FMT`, for the block's
// `group` of rows. With `EXPERT`, the
// matrix is one of many, the experts', one after another, each `groups`
// groups of rows; the block's tokens are pairs of a token and the expert,
// `count` of them, numbered at `pairs` in shared memory with the row of
// `x` that is each one's input, whose products, times the expert's scale,
// go to row `pair` of `out`, as floats.
struct ExpertBlock {
    u32 expert;
    u32 count;
    u32 groups;
    float scale;
    // The SiTU activation's two betas, where the kernel applies it.
    float beta;
    float linear_beta;
    // Each pair's number, and after them each one's row of the input.
    const u32* pairs;
};

//
// With `EXPERT` as `TILE_SITU`, an expert's matrix is its gate projection
// over its up projection, each half the expert's groups of rows: the block
// multiplies its group of both, one after the other, and what goes to row
// `pair` of `out` is the SiTU activation of the two, as halves, for the
// matmul through the expert's down projection: neither projection's
// products leave the block.
#define TILE_EXPERT 1
#define TILE_SITU 2

template <int MT, int FMT, int STAGES, int EXPERT>
__device__ __forceinline__ void tile_body(
    void* out, const unsigned char* w, const unsigned char* scales, const u16* x, const u16* other, TileParams p,
    ExpertBlock expert, unsigned char* buf, u32 group) {
    const bool Q8 = FMT == TILE_Q8_0;
    const bool F16 = FMT == TILE_F16;
    // Words of a tile a lane takes for each run of eight rows.
    const u32 WORDS = F16 ? 8 : Q8 ? 4 : 2;
    const u32 THREADS = 4 * 32;
    const u32 BM = MT * 16;
    // A stage: the activations, the weights, the scales.
    const u32 X = BM * TILE_X;
    const u32 W = TILE_N * 16 * WORDS;
    const u32 S = F16 ? 0 : 2 * TILE_N * 2;
    const u32 STAGE = X + W + S;
    const u32 tid = threadIdx.x;
    const u32 lane = tid & 31;
    const u32 wn = tid >> 5;
    const u32 m0 = EXPERT ? 0 : blockIdx.x * BM;
    // The gate's products, while the up projection is multiplied.
    float gate[EXPERT == TILE_SITU ? MT : 1][4][4];
    float acc[MT][4][4];
#pragma unroll 1
    for (u32 half = 0; half < (EXPERT == TILE_SITU ? 2 : 1); half++) {
        const size_t tiles =
            ((size_t)(EXPERT ? expert.expert * expert.groups : 0) + group + half * (expert.groups / 2)) * p.tiles;
        if (half != 0) {
            // The stages are the first half's until its last tile is done.
            __syncthreads();
        }

        auto load = [&](u32 s, u32 kt) {
            unsigned char* stage = buf + s * STAGE;
            for (u32 c = tid; c < BM * 8; c += THREADS) {
                u32 r = c / 8;
                u32 ch = c % 8;
                u32 col = kt * TILE_K + ch * 8;
                bool read = (EXPERT ? r < expert.count : m0 + r < p.count) && col < p.cols;
                size_t from = EXPERT ? (read ? expert.pairs[BM + r] : 0) : p.first + m0 + r;
                const u16* src = read ? x + from * p.cols + col : x;
                copy16(stage + swizzle(r, ch), src, read);
            }
            for (u32 c = tid; c < W / 16; c += THREADS) {
                copy16(stage + X + c * 16, w + (tiles + kt) * W + c * 16, true);
            }
            if (tid < S / 16) {
                copy16(stage + X + W + tid * 16, scales + (tiles + kt) * S + tid * 16, true);
            }
        };

#pragma unroll
        for (u32 i = 0; i < MT; i++) {
#pragma unroll
            for (u32 j = 0; j < 4; j++) {
#pragma unroll
                for (u32 e = 0; e < 4; e++) {
                    if (EXPERT == TILE_SITU && half != 0) {
                        gate[i][j][e] = acc[i][j][e];
                    }
                    acc[i][j][e] = 0.0f;
                }
            }
        }
#pragma unroll
        for (u32 s = 0; s < STAGES - 1; s++) {
            if (s < p.tiles) {
                load(s, s);
            }
            commit_copies();
        }
        for (u32 kt = 0; kt < p.tiles; kt++) {
            wait_copies<STAGES - 2>();
            __syncthreads();
            u32 next = kt + STAGES - 1;
            if (next < p.tiles) {
                load(next % STAGES, next);
            }
            commit_copies();
            const unsigned char* stage = buf + (kt % STAGES) * STAGE;
            const u32* pw = (const u32*)(stage + X);
            const u16* ps = (const u16*)(stage + X + W);
            // The lane's words of each of its warp's four runs of eight rows,
            // and its row's scales, twice over as a pair: one for each 32
            // columns, or for each 16 of NVFP4.
            u32 words[4][WORDS];
            u32 sv[4][FMT == TILE_NVFP4 ? 4 : 2];
#pragma unroll
            for (u32 j = 0; j < 4; j++) {
                u32 row = wn * 32 + j * 8 + (lane >> 2);
#pragma unroll
                for (u32 v = 0; v < WORDS; v++) {
                    words[j][v] = pw[((wn * 32 + j * 8) * 4 + lane) * WORDS + v];
                }
                if (F16) {
                    sv[j][0] = 0;
                } else if (FMT == TILE_NVFP4) {
#pragma unroll
                    for (u32 v = 0; v < 4; v++) {
                        sv[j][v] = fp4_scale(((const unsigned char*)ps)[v * TILE_N + row]);
                    }
                } else {
                    sv[j][0] = ps[row] * 0x10001u;
                    sv[j][1] = ps[TILE_N + row] * 0x10001u;
                }
            }
#pragma unroll
            for (u32 k = 0; k < 4; k++) {
                u32 a[MT][4];
#pragma unroll
                for (u32 i = 0; i < MT; i++) {
                    u32 r = i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                    load_matrix(a[i], stage + swizzle(r, k * 2 + (lane >> 4)));
                }
#pragma unroll
                for (u32 j = 0; j < 4; j++) {
                    u32 b0, b1;
                    if (F16) {
                        b0 = words[j][2 * k];
                        b1 = words[j][2 * k + 1];
                    } else if (Q8) {
                        b0 = q8_pair(words[j][k], 0, sv[j][k / 2]);
                        b1 = q8_pair(words[j][k], 16, sv[j][k / 2]);
                    } else if (FMT == TILE_NVFP4) {
                        b0 = fp4_pair(words[j][k / 2], (k & 1) * 4, sv[j][k]);
                        b1 = fp4_pair(words[j][k / 2], (k & 1) * 4 + 8, sv[j][k]);
                    } else {
                        b0 = q4_pair(words[j][k / 2], (k & 1) * 4, sv[j][k / 2]);
                        b1 = q4_pair(words[j][k / 2], (k & 1) * 4 + 8, sv[j][k / 2]);
                    }
#pragma unroll
                    for (u32 i = 0; i < MT; i++) {
                        multiply_add(acc[i][j], a[i], b0, b1);
                    }
                }
            }
        }
    }
#pragma unroll
    for (u32 i = 0; i < MT; i++) {
#pragma unroll
        for (u32 j = 0; j < 4; j++) {
            u32 row = group * TILE_N + wn * 32 + j * 8 + (lane & 3) * 2;
#pragma unroll
            for (u32 h = 0; h < 2; h++) {
                u32 t = m0 + i * 16 + (lane >> 2) + h * 8;
                bool pair = row + 1 < p.rows;
                float v0 = acc[i][j][2 * h];
                float v1 = acc[i][j][2 * h + 1];
                if (EXPERT == TILE_SITU) {
                    // A group's rows are whole, and a pair of them a word.
                    if (t < expert.count) {
                        v0 = situ(gate[i][j][2 * h] * expert.scale, v0 * expert.scale, expert.beta, expert.linear_beta);
                        v1 = situ(
                            gate[i][j][2 * h + 1] * expert.scale, v1 * expert.scale, expert.beta, expert.linear_beta);
                        u32* o = (u32*)((u16*)out + (size_t)expert.pairs[t] * p.rows + row);
                        *o = f32_to_finite_f16(v0) | f32_to_finite_f16(v1) << 16;
                    }
                    continue;
                }
                if (EXPERT) {
                    if (t < expert.count && row < p.rows) {
                        float* o = (float*)out + (size_t)expert.pairs[t] * p.rows + row;
                        o[0] = v0 * expert.scale;
                        if (pair) {
                            o[1] = v1 * expert.scale;
                        }
                    }
                    continue;
                }
                if (t >= p.count || row >= p.rows) {
                    continue;
                }
                size_t at = (size_t)(p.first + t) * p.rows + row;
                if (p.mode == TILE_FLOATS || p.mode == TILE_ADD) {
                    float* o = (float*)out + at;
                    if (p.mode == TILE_ADD) {
                        v0 += o[0];
                        v1 += pair ? o[1] : 0.0f;
                    }
                    o[0] = v0;
                    if (pair) {
                        o[1] = v1;
                    }
                    continue;
                }
                // Two rows of a token side by side, which are a word of
                // halves where a token's rows start on one.
                bool word = pair && (p.rows & 1) == 0;
                u16* o = (u16*)out + at;
                if (p.mode == TILE_SWIGLU) {
                    u32 u = word ? *(const u32*)(other + at) : other[at] | (pair ? other[at + 1] << 16 : 0);
                    v0 = v0 * sigmoid(v0) * f16_to_f32(u);
                    v1 = v1 * sigmoid(v1) * f16_to_f32(u >> 16);
                }
                u32 halves = f32_to_finite_f16(v0) | f32_to_finite_f16(v1) << 16;
                if (word) {
                    *(u32*)o = halves;
                } else {
                    o[0] = (u16)halves;
                    if (pair) {
                        o[1] = (u16)(halves >> 16);
                    }
                }
            }
        }
    }
}

#define TILE_KERNEL(name, MT, FMT, STAGES) \
    extern "C" __global__ void __launch_bounds__(4 * 32) name( \
        void* out, const unsigned char* w, const unsigned char* scales, const u16* x, const u16* other, \
        TileParams p) { \
        extern __shared__ __align__(16) unsigned char buf[]; \
        ExpertBlock none = {0, 0, 0, 0.0f, 0.0f, 0.0f, 0}; \
        tile_body<MT, FMT, STAGES, 0>(out, w, scales, x, other, p, none, buf, blockIdx.y); \
    }

TILE_KERNEL(matmul_tile16_q4_0, 1, TILE_Q4_0, TILE_STAGES)
TILE_KERNEL(matmul_tile16_q8_0, 1, TILE_Q8_0, TILE_STAGES)
TILE_KERNEL(matmul_tile64_q4_0, 4, TILE_Q4_0, TILE_STAGES)
TILE_KERNEL(matmul_tile64_q8_0, 4, TILE_Q8_0, TILE_STAGES)
TILE_KERNEL(matmul_tile16_f16, 1, TILE_F16, TILE_STAGES)
TILE_KERNEL(matmul_tile64_f16, 4, TILE_F16, TILE_STAGES)

// Several matrix multiplications of up to 16 tokens in one grid, the
// groups of rows of one after another's: a token alone goes through
// matrices too small to fill the GPU, each of which would otherwise wait
// for the one before to drain.
#define TILE_JOBS_KERNEL(name, FMT, STAGES) \
    extern "C" __global__ void __launch_bounds__(4 * 32) name(TileJobs p) { \
        extern __shared__ __align__(16) unsigned char buf[]; \
        u32 group = blockIdx.y; \
        u32 j = 0; \
        while (j + 1 < p.jobs && group >= p.job[j].groups) { \
            group -= p.job[j].groups; \
            j++; \
        } \
        ExpertBlock none = {0, 0, 0, 0.0f, 0.0f, 0.0f, 0}; \
        TileParams tile = {p.job[j].rows, p.job[j].cols, p.job[j].tiles, 0, p.count, TILE_FLOATS}; \
        tile_body<1, FMT, STAGES, 0>( \
            p.job[j].out, p.job[j].w, p.job[j].scales, p.job[j].x, 0, tile, none, buf, group); \
    }

TILE_JOBS_KERNEL(matmul_tiles16_q4_0, TILE_Q4_0, TILE_STAGES)
TILE_JOBS_KERNEL(matmul_tiles16_q8_0, TILE_Q8_0, TILE_STAGES)

// A matrix multiplication by each token's experts: a block takes a tile
// of up to `16 * MT` pairs of a token and an expert, all of one expert,
// and a group of 128 of the expert's rows. The blocks of a tile come one
// after another in the grid, and so do the tiles of an expert: the tile's
// inputs are read from memory once, and an expert's weights once for all
// its tiles. The tiles are the three words each of `plan`, an expert, the
// first of its pairs in `order` and how many there are, where a batch's
// pairs are sorted by expert; or, with `p.direct`, a tile to a pair, in
// the pairs' own order, the expert `ids` has for it. The tile's pairs,
// and the rows of `x` that are their inputs, are kept in shared memory,
// past the stages.
#define EXPERT_KERNEL(name, MT, FMT, STAGES, EXPERT) \
    extern "C" __global__ void __launch_bounds__(4 * 32) name( \
        void* out, const unsigned char* w, const unsigned char* scales, const u16* x, const u32* order, \
        const u32* plan, const float* ids, const float* scale, ExpertParams p) { \
        extern __shared__ __align__(16) unsigned char buf[]; \
        const u32 STAGE = MT * 16 * TILE_X + TILE_N * (FMT == TILE_Q8_0 ? 64 : 32) + 2 * TILE_N * 2; \
        u32* pairs = (u32*)(buf + STAGES * STAGE); \
        ExpertBlock block; \
        u32 first; \
        if (p.direct) { \
            block.expert = (u32)ids[blockIdx.y]; \
            first = blockIdx.y; \
            block.count = 1; \
        } else { \
            block.expert = plan[3 * blockIdx.y]; \
            first = plan[3 * blockIdx.y + 1]; \
            block.count = plan[3 * blockIdx.y + 2]; \
        } \
        block.groups = p.groups; \
        block.scale = scale[block.expert]; \
        block.beta = p.beta; \
        block.linear_beta = p.linear_beta; \
        block.pairs = pairs; \
        if (threadIdx.x < block.count) { \
            u32 pair = p.direct ? first : order[first + threadIdx.x]; \
            pairs[threadIdx.x] = pair; \
            pairs[MT * 16 + threadIdx.x] = pair / p.share; \
        } \
        __syncthreads(); \
        TileParams tile = {p.rows, p.cols, p.tiles, 0, 0, 0}; \
        tile_body<MT, FMT, STAGES, EXPERT>(out, w, scales, x, 0, tile, block, buf, blockIdx.x); \
    }

EXPERT_KERNEL(expert_tile16_nvfp4, 1, TILE_NVFP4, TILE_STAGES, TILE_EXPERT)
EXPERT_KERNEL(expert_tile16_q8_0, 1, TILE_Q8_0, TILE_STAGES, TILE_EXPERT)
EXPERT_KERNEL(expert_tile64_nvfp4, 4, TILE_NVFP4, TILE_STAGES, TILE_EXPERT)
EXPERT_KERNEL(expert_tile64_q8_0, 4, TILE_Q8_0, TILE_STAGES, TILE_EXPERT)
EXPERT_KERNEL(expert_situ_tile16_nvfp4, 1, TILE_NVFP4, TILE_STAGES, TILE_SITU)
EXPERT_KERNEL(expert_situ_tile16_q8_0, 1, TILE_Q8_0, TILE_STAGES, TILE_SITU)
EXPERT_KERNEL(expert_situ_tile64_nvfp4, 4, TILE_NVFP4, TILE_STAGES, TILE_SITU)
EXPERT_KERNEL(expert_situ_tile64_q8_0, 4, TILE_Q8_0, TILE_STAGES, TILE_SITU)

#endif
