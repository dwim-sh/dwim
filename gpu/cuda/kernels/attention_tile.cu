// Causal attention over the 8-bit caches on the tensor cores, for heads of
// 256: what `attention.cu` computes, with the queries, the keys, the
// softmax's weights and the values as half-precision floats and the sums
// in single precision. The scheme is TensorFold's (`licenses/tensorfold.md`).
//
// A block takes a key/value head, a chunk of positions, and as many tokens
// as have `16 * MT` query heads of the group between them, a line each. It
// walks the chunk 32 positions at a time. The threads read a step's keys
// and values into registers while the step before is worked, and then
// widen them into shared memory, each block of a key or a value times its
// scale. The warps score the lines against the keys; each line's scores
// are softmaxed against the largest it has had, what it has summed so far
// scaled down to match; then the warps add the values the weights weigh,
// each warp for its own elements of the head. A chunk's maximum, sum, and
// weighted values go to the partials buffer for `attention_tile_combine`
// to merge, or, when there is one chunk, straight to the output.

#if __CUDA_ARCH__ >= 800

// Positions to a step, and elements to a head.
#define ATT_KEYS 32
#define ATT_DIM 256
// What the score of a position a line does not attend to is, and the
// largest score of a line that has attended to none.
#define ATT_NONE (-1e30f)

// 2 to the `x`, nearly.
__device__ __forceinline__ float exp2_fast(float x) {
    float y;
    asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ u32 half_pair(float a, float b) {
    return f32_to_f16(fminf(fmaxf(a, -65504.0f), 65504.0f)) | f32_to_f16(fminf(fmaxf(b, -65504.0f), 65504.0f)) << 16;
}

// Two 8 by 8 matrices of halves from shared memory, lanes 8m to 8m + 7
// giving the addresses of matrix m's rows; and four, each transposed.
__device__ __forceinline__ void load_matrix2(u32 (&r)[2], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
        : "=r"(r[0]), "=r"(r[1]) : "r"(shared_address(p)));
}

__device__ __forceinline__ void load_matrix_transposed(u32 (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(shared_address(p)));
}

// The place of a row's chunk of 16 bytes of halves, 32 to a row, of the
// queries, the keys or the values, and of the weights, four to a line:
// turned by the row so that the eight rows read as a matrix at once are
// in different banks.
__device__ __forceinline__ u32 query_at(u32 line, u32 chunk) {
    return line * (ATT_DIM * 2) + ((chunk ^ (line & 7)) * 16);
}

__device__ __forceinline__ u32 weight_at(u32 line, u32 chunk) {
    return line * (ATT_KEYS * 2) + ((chunk ^ ((line >> 1) & 3)) * 16);
}

// Four 8-bit integers of a cache as two pairs of halves, each times the
// pair `scale`: a byte less 128 beside the high byte of 1024 is 1024 plus
// the byte plus 128 as a half, exactly.
__device__ __forceinline__ uint2 widen4(u32 bytes, u32 scale) {
    u32 flipped = bytes ^ 0x80808080u;
    u32 lo, hi;
    asm("{ .reg .b32 t; prmt.b32 t, %2, %3, 0x4140; sub.rn.f16x2 t, t, %4; mul.rn.f16x2 %0, t, %5;"
        " prmt.b32 t, %2, %3, 0x4342; sub.rn.f16x2 t, t, %4; mul.rn.f16x2 %1, t, %5; }"
        : "=r"(lo), "=r"(hi) : "r"(flipped), "r"(0x64646464u), "r"(0x64806480u), "r"(scale));
    return make_uint2(lo, hi);
}

// Sixteen of them as two chunks of a row of halves at `to`.
__device__ __forceinline__ void widen16(unsigned char* row, u32 key, u32 chunk, uint4 bytes, u32 scale) {
    scale *= 0x10001u;
    uint2 a = widen4(bytes.x, scale), b = widen4(bytes.y, scale);
    uint2 c = widen4(bytes.z, scale), d = widen4(bytes.w, scale);
    *(uint4*)(row + query_at(key, 2 * chunk)) = make_uint4(a.x, a.y, b.x, b.y);
    *(uint4*)(row + query_at(key, 2 * chunk + 1)) = make_uint4(c.x, c.y, d.x, d.y);
}

template <int MT, int WM>
__device__ __forceinline__ void attention_tile_body(
    float* out, const float* q, const unsigned char* k_quants, const u16* k_scales,
    const unsigned char* v_quants, const u16* v_scales, float* partials, AttentionTileParams p,
    unsigned char* buf) {
    const u32 LINES = 16 * MT;
    const u32 WARPS = 4 * WM;
    const u32 THREADS = WARPS * 32;
    // Threads to a line of the softmax and keys to each, runs of eight
    // elements a warp sums values for, and chunks of 16 bytes of a step's
    // keys, and of its values, a thread reads.
    const u32 LINE_THREADS = THREADS / LINES;
    const u32 THREAD_KEYS = ATT_KEYS / LINE_THREADS;
    const u32 WARP_RUNS = ATT_DIM / 8 / WARPS;
    const u32 READS = ATT_KEYS * (ATT_DIM / 16) / THREADS;
    unsigned char* queries = buf;
    float* scores = (float*)(queries + LINES * ATT_DIM * 2);
    unsigned char* weights = (unsigned char*)(scores + LINES * ATT_KEYS);
    float* alpha = (float*)(weights + LINES * ATT_KEYS * 2);
    float* line_max = alpha + LINES;
    float* line_sum = line_max + LINES;
    unsigned char* keys = (unsigned char*)(line_sum + LINES);
    unsigned char* values = keys + ATT_KEYS * ATT_DIM * 2;

    const u32 tid = threadIdx.x;
    const u32 lane = tid & 31;
    const u32 warp = tid >> 5;
    const u32 group = p.n_heads / p.n_kv_heads;
    const u32 tokens = LINES / group;
    const u32 c = blockIdx.x % p.chunks;
    const u32 kvh = (blockIdx.x / p.chunks) % p.n_kv_heads;
    const u32 t0 = p.first + (blockIdx.x / p.chunks / p.n_kv_heads) * tokens;
    const u32 count = min(tokens, p.first + p.count - t0);
    const u32 lines = count * group;
    const u32 kv_dim = p.n_kv_heads * ATT_DIM;
    // The chunk's positions, up to the last the block's last token
    // attends to.
    const u32 k0 = c * p.chunk;
    const u32 k1 = min(k0 + p.chunk, p.pos + t0 + count);
    const u32 stride = ATT_DIM + 2;
    // Where a line's partials for this chunk go.
    auto partial = [&](u32 line) {
        u32 h = kvh * group + line % group;
        return partials + ((size_t)((t0 + line / group - p.first) * p.n_heads + h) * p.chunks + c) * stride;
    };
    if (k0 >= k1) {
        if (tid < lines) {
            partial(tid)[0] = ATT_NONE;
            partial(tid)[1] = 0.0f;
        }
        return;
    }

    // The queries as halves, each already over the square root of the
    // head's length.
    for (u32 i = tid; i < LINES * (ATT_DIM / 8); i += THREADS) {
        u32 line = i / (ATT_DIM / 8);
        u32 chunk = i % (ATT_DIM / 8);
        uint4 halves = make_uint4(0, 0, 0, 0);
        if (line < lines) {
            const float4* at = (const float4*)(q + ((size_t)(t0 + line / group) * p.n_heads + kvh * group + line % group) * ATT_DIM + chunk * 8);
            float4 a = scale4(at[0], 0.0625f);
            float4 b = scale4(at[1], 0.0625f);
            halves = make_uint4(half_pair(a.x, a.y), half_pair(a.z, a.w), half_pair(b.x, b.y), half_pair(b.z, b.w));
        }
        *(uint4*)(queries + query_at(line, chunk)) = halves;
    }

    // A step's keys and values from `first` on, the thread's chunks of
    // them and each chunk's block's scale, into registers: a key past the
    // chunk's last is read as the first of the caches and not attended to.
    uint4 key_bytes[READS], value_bytes[READS];
    u32 key_scale[READS], value_scale[READS];
    auto fetch = [&](u32 first) {
#pragma unroll
        for (u32 r = 0; r < READS; r++) {
            u32 i = tid + r * THREADS;
            u32 key = first + i / (ATT_DIM / 16);
            size_t at = key < k1 ? (size_t)key * kv_dim + kvh * ATT_DIM + (i % (ATT_DIM / 16)) * 16 : 0;
            key_bytes[r] = *(const uint4*)(k_quants + at);
            value_bytes[r] = *(const uint4*)(v_quants + at);
            key_scale[r] = k_scales[at / CACHE_BLOCK];
            value_scale[r] = v_scales[at / CACHE_BLOCK];
        }
    };
    // And from the registers into shared memory, as halves.
    auto put = [&]() {
#pragma unroll
        for (u32 r = 0; r < READS; r++) {
            u32 i = tid + r * THREADS;
            widen16(keys, i / (ATT_DIM / 16), i % (ATT_DIM / 16), key_bytes[r], key_scale[r]);
            widen16(values, i / (ATT_DIM / 16), i % (ATT_DIM / 16), value_bytes[r], value_scale[r]);
        }
    };

    float o[MT][WARP_RUNS][4];
#pragma unroll
    for (u32 i = 0; i < MT; i++) {
#pragma unroll
        for (u32 n = 0; n < WARP_RUNS; n++) {
#pragma unroll
            for (u32 e = 0; e < 4; e++) {
                o[i][n][e] = 0.0f;
            }
        }
    }
    // The thread's line's largest score and sum so far, which every thread
    // of the line holds.
    const u32 own = tid / LINE_THREADS;
    float largest = ATT_NONE;
    float sum = 0.0f;

    fetch(k0);
    for (u32 first = k0; first < k1; first += ATT_KEYS) {
        __syncthreads();
        put();
        __syncthreads();
        if (first + ATT_KEYS < k1) {
            fetch(first + ATT_KEYS);
        }

        // The scores: a warp takes eight keys and its share of the lines.
        {
            const u32 m0 = (warp / 4) * (MT / WM);
            float sc[MT / WM][4];
#pragma unroll
            for (u32 i = 0; i < MT / WM; i++) {
#pragma unroll
                for (u32 e = 0; e < 4; e++) {
                    sc[i][e] = 0.0f;
                }
            }
#pragma unroll
            for (u32 k = 0; k < ATT_DIM / 16; k++) {
                u32 b[2];
                load_matrix2(b, keys + query_at((warp % 4) * 8 + (lane & 7), k * 2 + ((lane >> 3) & 1)));
#pragma unroll
                for (u32 i = 0; i < MT / WM; i++) {
                    u32 a[4];
                    u32 line = (m0 + i) * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                    load_matrix(a, queries + query_at(line, k * 2 + (lane >> 4)));
                    multiply_add(sc[i], a, b[0], b[1]);
                }
            }
            const u32 at = (warp % 4) * 8 + 2 * (lane & 3);
#pragma unroll
            for (u32 i = 0; i < MT / WM; i++) {
#pragma unroll
                for (u32 h = 0; h < 2; h++) {
                    u32 line = (m0 + i) * 16 + (lane >> 2) + h * 8;
                    // A line attends up to its token's own position.
                    u32 len = line < lines ? p.pos + t0 + line / group + 1 : 0;
                    float2 pair = make_float2(
                        first + at < len ? sc[i][2 * h] : ATT_NONE,
                        first + at + 1 < len ? sc[i][2 * h + 1] : ATT_NONE);
                    *(float2*)(scores + line * ATT_KEYS + at) = pair;
                }
            }
        }
        __syncthreads();

        // The softmax: the threads of a line take its keys between them.
        {
            const u32 at = (tid % LINE_THREADS) * THREAD_KEYS;
            float score[THREAD_KEYS];
            float top = ATT_NONE;
#pragma unroll
            for (u32 i = 0; i < THREAD_KEYS; i++) {
                score[i] = scores[own * ATT_KEYS + at + i];
                top = fmaxf(top, score[i]);
            }
#pragma unroll
            for (u32 x = 1; x < LINE_THREADS; x *= 2) {
                top = fmaxf(top, __shfl_xor_sync(FULL, top, x));
            }
            top = fmaxf(top, largest);
            // What the line has summed is scaled down to its new maximum.
            float a = largest > 0.5f * ATT_NONE ? exp2_fast((largest - top) * 1.4426950408889634f) : 1.0f;
            float w[THREAD_KEYS];
            float total = 0.0f;
#pragma unroll
            for (u32 i = 0; i < THREAD_KEYS; i++) {
                w[i] = score[i] > 0.5f * ATT_NONE ? exp2_fast((score[i] - top) * 1.4426950408889634f) : 0.0f;
                total += w[i];
            }
#pragma unroll
            for (u32 x = 1; x < LINE_THREADS; x *= 2) {
                total += __shfl_xor_sync(FULL, total, x);
            }
            sum = sum * a + total;
            largest = top;
            u16* to = (u16*)(weights + weight_at(own, at / 8)) + at % 8;
#pragma unroll
            for (u32 i = 0; i < THREAD_KEYS; i++) {
                to[i] = (u16)f32_to_f16(w[i]);
            }
            if (tid % LINE_THREADS == 0) {
                alpha[own] = a;
            }
        }
        __syncthreads();

        // The values: a warp takes its share of the head's elements, for
        // every line.
        {
#pragma unroll
            for (u32 i = 0; i < MT; i++) {
                float a0 = alpha[i * 16 + (lane >> 2)];
                float a1 = alpha[i * 16 + (lane >> 2) + 8];
                // Most steps leave a line's maximum as it was.
                if (a0 != 1.0f || a1 != 1.0f) {
#pragma unroll
                    for (u32 n = 0; n < WARP_RUNS; n++) {
                        o[i][n][0] *= a0;
                        o[i][n][1] *= a0;
                        o[i][n][2] *= a1;
                        o[i][n][3] *= a1;
                    }
                }
            }
#pragma unroll
            for (u32 kk = 0; kk < ATT_KEYS / 16; kk++) {
                u32 pa[MT][4];
#pragma unroll
                for (u32 i = 0; i < MT; i++) {
                    u32 line = i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                    load_matrix(pa[i], weights + weight_at(line, kk * 2 + (lane >> 4)));
                }
#pragma unroll
                for (u32 n = 0; n < WARP_RUNS; n += 2) {
                    // Two runs of eight elements at once: the lane's
                    // matrix is eight keys of one of them.
                    u32 b[4];
                    u32 m = lane >> 3;
                    load_matrix_transposed(
                        b, values + query_at(kk * 16 + 8 * (m % 2) + (lane & 7), warp * WARP_RUNS + n + m / 2));
#pragma unroll
                    for (u32 i = 0; i < MT; i++) {
                        multiply_add(o[i][n], pa[i], b[0], b[1]);
                        multiply_add(o[i][n + 1], pa[i], b[2], b[3]);
                    }
                }
            }
        }
    }
    __syncthreads();
    if (tid % LINE_THREADS == 0) {
        line_max[own] = largest;
        line_sum[own] = sum;
    }
    __syncthreads();
#pragma unroll
    for (u32 i = 0; i < MT; i++) {
#pragma unroll
        for (u32 h = 0; h < 2; h++) {
            u32 line = i * 16 + (lane >> 2) + h * 8;
            if (line >= lines) {
                continue;
            }
            u32 head = kvh * group + line % group;
            if (p.chunks == 1) {
                float* at = out + ((size_t)(t0 + line / group) * p.n_heads + head) * ATT_DIM;
                float scale = 1.0f / line_sum[line];
#pragma unroll
                for (u32 n = 0; n < WARP_RUNS; n++) {
                    u32 dim = (warp * WARP_RUNS + n) * 8 + 2 * (lane & 3);
                    *(float2*)(at + dim) = make_float2(o[i][n][2 * h] * scale, o[i][n][2 * h + 1] * scale);
                }
            } else {
                float* at = partial(line);
                if (tid == (line % 16 % 8) * 4) {
                    at[0] = line_max[line];
                    at[1] = line_sum[line];
                }
#pragma unroll
                for (u32 n = 0; n < WARP_RUNS; n++) {
                    u32 dim = (warp * WARP_RUNS + n) * 8 + 2 * (lane & 3);
                    *(float2*)(at + 2 + dim) = make_float2(o[i][n][2 * h], o[i][n][2 * h + 1]);
                }
            }
        }
    }
}

#define ATTENTION_TILE_KERNEL(name, MT, WM) \
    extern "C" __global__ void __launch_bounds__(WM * 4 * 32) name( \
        float* out, const float* q, const unsigned char* k_quants, const u16* k_scales, \
        const unsigned char* v_quants, const u16* v_scales, float* partials, AttentionTileParams p) { \
        extern __shared__ __align__(16) unsigned char buf[]; \
        attention_tile_body<MT, WM>(out, q, k_quants, k_scales, v_quants, v_scales, partials, p, buf); \
    }

ATTENTION_TILE_KERNEL(attention_tile16, 1, 1)
ATTENTION_TILE_KERNEL(attention_tile64, 4, 2)

// Merges the chunks: one block per (token, head), each thread taking one
// element of the head. The chunks' softmaxes were taken against their own
// maxima, so each is rescaled to the maximum over the chunks before the
// sums are added; a chunk the token attends to nothing of is left out.
extern "C" __global__ void __launch_bounds__(ATT_DIM) attention_tile_combine(
    float* out, const float* partials, AttentionTileParams p) {
    u32 t = p.first + blockIdx.x / p.n_heads;
    u32 h = blockIdx.x % p.n_heads;
    u32 stride = ATT_DIM + 2;
    const float* base = partials + (size_t)((t - p.first) * p.n_heads + h) * p.chunks * stride;
    float m = ATT_NONE;
    for (u32 c = 0; c < p.chunks; c++) {
        m = fmaxf(m, base[c * stride]);
    }
    float total = 0.0f;
    float o = 0.0f;
    for (u32 c = 0; c < p.chunks; c++) {
        if (base[c * stride] > 0.5f * ATT_NONE) {
            float scale = expf(base[c * stride] - m);
            total += scale * base[c * stride + 1];
            o += scale * base[c * stride + 2 + threadIdx.x];
        }
    }
    out[((size_t)t * p.n_heads + h) * ATT_DIM + threadIdx.x] = o / total;
}

#endif
