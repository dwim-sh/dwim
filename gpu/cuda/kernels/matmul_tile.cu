// out[t][r] = w[r] · x[t] for a batch of tokens and block-quantized
// weights, on the tensor cores: the activations as half-precision floats,
// which `to_f16.cu` makes of them, the weights each times its block's
// scale and rounded to a half, and the sums in single precision. The
// weights are tiled as `mod.rs` lays them out, 128 rows by 64 columns to
// a tile, each lane's words of a tile side by side.
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

// The place in a token's row of a stage of its `chunk` of 16 bytes: the
// chunks are turned by the row, so that the eight rows `load_matrix` reads
// at once are in different banks of shared memory.
__device__ __forceinline__ u32 swizzle(u32 row, u32 chunk) {
    return row * TILE_X + ((chunk ^ (row & 7)) * 16);
}

template <int MT, bool Q8, int STAGES>
__device__ __forceinline__ void tile_body(
    float* out, const unsigned char* w, const u16* scales, const u16* x, TileParams p, unsigned char* buf) {
    const u32 THREADS = 4 * 32;
    const u32 BM = MT * 16;
    // A stage: the activations, the weights, the scales.
    const u32 X = BM * TILE_X;
    const u32 W = TILE_N * (Q8 ? 64 : 32);
    const u32 S = 2 * TILE_N * 2;
    const u32 STAGE = X + W + S;
    const u32 tid = threadIdx.x;
    const u32 lane = tid & 31;
    const u32 wn = tid >> 5;
    const u32 group = blockIdx.y;
    const u32 m0 = blockIdx.x * BM;
    const size_t tiles = (size_t)group * p.tiles;

    auto load = [&](u32 s, u32 kt) {
        unsigned char* stage = buf + s * STAGE;
        for (u32 c = tid; c < BM * 8; c += THREADS) {
            u32 r = c / 8;
            u32 ch = c % 8;
            u32 col = kt * TILE_K + ch * 8;
            bool read = m0 + r < p.count && col < p.cols;
            const u16* src = read ? x + (size_t)(p.first + m0 + r) * p.cols + col : x;
            copy16(stage + swizzle(r, ch), src, read);
        }
        for (u32 c = tid; c < W / 16; c += THREADS) {
            copy16(stage + X + c * 16, w + (tiles + kt) * W + c * 16, true);
        }
        if (tid < S / 16) {
            copy16(stage + X + W + tid * 16, scales + (tiles + kt) * (S / 2) + tid * 8, true);
        }
    };

    float acc[MT][4][4];
#pragma unroll
    for (u32 i = 0; i < MT; i++) {
#pragma unroll
        for (u32 j = 0; j < 4; j++) {
#pragma unroll
            for (u32 e = 0; e < 4; e++) {
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
        // and its row's scales, twice over as a pair.
        u32 words[4][Q8 ? 4 : 2];
        u32 sv[4][2];
#pragma unroll
        for (u32 j = 0; j < 4; j++) {
            u32 row = wn * 32 + j * 8 + (lane >> 2);
#pragma unroll
            for (u32 v = 0; v < (Q8 ? 4 : 2); v++) {
                words[j][v] = pw[((wn * 32 + j * 8) * 4 + lane) * (Q8 ? 4 : 2) + v];
            }
            sv[j][0] = ps[row] * 0x10001u;
            sv[j][1] = ps[TILE_N + row] * 0x10001u;
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
                if (Q8) {
                    b0 = q8_pair(words[j][k], 0, sv[j][k / 2]);
                    b1 = q8_pair(words[j][k], 16, sv[j][k / 2]);
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
#pragma unroll
    for (u32 i = 0; i < MT; i++) {
#pragma unroll
        for (u32 j = 0; j < 4; j++) {
            u32 row = group * TILE_N + wn * 32 + j * 8 + (lane & 3) * 2;
#pragma unroll
            for (u32 h = 0; h < 2; h++) {
                u32 t = m0 + i * 16 + (lane >> 2) + h * 8;
                if (t < p.count) {
                    float* at = out + (size_t)(p.first + t) * p.rows + row;
                    if (row < p.rows) {
                        at[0] = acc[i][j][2 * h];
                    }
                    if (row + 1 < p.rows) {
                        at[1] = acc[i][j][2 * h + 1];
                    }
                }
            }
        }
    }
}

#define TILE_KERNEL(name, MT, Q8, STAGES) \
    extern "C" __global__ void __launch_bounds__(4 * 32) name( \
        float* out, const unsigned char* w, const u16* scales, const u16* x, TileParams p) { \
        extern __shared__ __align__(16) unsigned char buf[]; \
        tile_body<MT, Q8, STAGES>(out, w, scales, x, p, buf); \
    }

TILE_KERNEL(matmul_tile16_q4_0, 1, false, TILE_STAGES)
TILE_KERNEL(matmul_tile16_q8_0, 1, true, TILE_STAGES)
TILE_KERNEL(matmul_tile64_q4_0, 4, false, TILE_STAGES)
TILE_KERNEL(matmul_tile64_q8_0, 4, true, TILE_STAGES)

#endif
