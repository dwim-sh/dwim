// Causal attention over the 8-bit caches on the tensor cores, for heads of
// 256: what `attention.cu` computes, with the queries, the keys, the
// softmax's weights and the values as half-precision floats and the sums
// in single precision. The scheme is TensorFold's (`licenses/tensorfold.md`).
//
// A block takes a key/value head, a chunk of positions, and as many tokens
// as have `16 * MT` query heads of the group between them, a line each. It
// walks the chunk 32 positions at a time, their keys, values and scales
// copied into shared memory a step ahead: the warps score the lines
// against the keys, each key's block times its scale as it is taken; then
// each line's scores are softmaxed against the largest it has had, what it
// has summed so far scaled down to match; then the warps add the values
// the weights weigh, each warp for its own elements of the head. A
// chunk's maximum, sum, and weighted values go to the partials buffer for
// `attention_tile_combine` to merge, or, when there is one chunk, straight
// to the output.

#if __CUDA_ARCH__ >= 800

// Positions to a step, and elements to a head.
#define ATT_KEYS 32
#define ATT_DIM 256
// A stage: the keys, the values, and the scales of each.
#define ATT_STAGE (2 * ATT_KEYS * ATT_DIM + 2 * ATT_KEYS * 16)
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

// The place of a line's chunk of 16 bytes of the queries, 32 to a line,
// and of the weights, four to a line, turned by the line so that the
// eight lines `load_matrix` reads at once are in different banks.
__device__ __forceinline__ u32 query_at(u32 line, u32 chunk) {
    return line * (ATT_DIM * 2) + ((chunk ^ (line & 7)) * 16);
}

__device__ __forceinline__ u32 weight_at(u32 line, u32 chunk) {
    return line * (ATT_KEYS * 2) + ((chunk ^ ((line >> 1) & 3)) * 16);
}

template <int MT, int WM>
__device__ __forceinline__ void attention_tile_body(
    float* out, const float* q, const unsigned char* k_quants, const u16* k_scales,
    const unsigned char* v_quants, const u16* v_scales, float* partials, AttentionTileParams p,
    unsigned char* buf) {
    const u32 LINES = 16 * MT;
    const u32 WARPS = 4 * WM;
    const u32 THREADS = WARPS * 32;
    // Lines a warp softmaxes, and runs of eight elements it sums values
    // for.
    const u32 WARP_LINES = LINES / WARPS;
    const u32 WARP_RUNS = ATT_DIM / 8 / WARPS;
    unsigned char* queries = buf;
    float* scores = (float*)(queries + LINES * ATT_DIM * 2);
    unsigned char* weights = (unsigned char*)(scores + LINES * ATT_KEYS);
    float* alpha = (float*)(weights + LINES * ATT_KEYS * 2);
    float* line_max = alpha + LINES;
    float* line_sum = line_max + LINES;
    unsigned char* stages = (unsigned char*)(line_sum + LINES);

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

    auto fetch = [&](u32 s, u32 first) {
        unsigned char* stage = stages + s * ATT_STAGE;
        for (u32 i = tid; i < ATT_KEYS * (ATT_DIM / 16); i += THREADS) {
            u32 key = i / (ATT_DIM / 16);
            bool read = first + key < k1;
            size_t at = read ? (size_t)(first + key) * kv_dim + kvh * ATT_DIM + (i % (ATT_DIM / 16)) * 16 : 0;
            copy16(stage + i * 16, k_quants + at, read);
            copy16(stage + ATT_KEYS * ATT_DIM + i * 16, v_quants + at, read);
        }
        if (tid < ATT_KEYS) {
            bool read = first + tid < k1;
            size_t at = read ? ((size_t)(first + tid) * kv_dim + kvh * ATT_DIM) / CACHE_BLOCK : 0;
            copy16(stage + 2 * ATT_KEYS * ATT_DIM + tid * 16, k_scales + at, read);
            copy16(stage + 2 * ATT_KEYS * ATT_DIM + (ATT_KEYS + tid) * 16, v_scales + at, read);
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
    // The warp's lines' largest scores and sums so far, which every lane
    // holds.
    float largest[WARP_LINES];
    float sum[WARP_LINES];
#pragma unroll
    for (u32 j = 0; j < WARP_LINES; j++) {
        largest[j] = ATT_NONE;
        sum[j] = 0.0f;
    }

    fetch(0, k0);
    commit_copies();
    u32 s = 0;
    for (u32 first = k0; first < k1; first += ATT_KEYS, s ^= 1) {
        wait_copies<0>();
        __syncthreads();
        if (first + ATT_KEYS < k1) {
            fetch(s ^ 1, first + ATT_KEYS);
        }
        commit_copies();
        const unsigned char* keys = stages + s * ATT_STAGE;
        const unsigned char* values = keys + ATT_KEYS * ATT_DIM;
        const u16* key_scales = (const u16*)(values + ATT_KEYS * ATT_DIM);
        const u16* value_scales = key_scales + ATT_KEYS * 8;

        // The scores: a warp takes eight keys and its share of the lines.
        {
            const u32 key = (warp % 4) * 8 + (lane >> 2);
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
                const unsigned char* at = keys + key * ATT_DIM + k * 16 + 2 * (lane & 3);
                u32 scale = key_scales[key * 8 + k / 2] * 0x10001u;
                u32 b0 = q8_pair(*(const u16*)at, 0, scale);
                u32 b1 = q8_pair(*(const u16*)(at + 8), 0, scale);
#pragma unroll
                for (u32 i = 0; i < MT / WM; i++) {
                    u32 a[4];
                    u32 line = (m0 + i) * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                    load_matrix(a, queries + query_at(line, k * 2 + (lane >> 4)));
                    multiply_add(sc[i], a, b0, b1);
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
                    scores[line * ATT_KEYS + at] = first + at < len ? sc[i][2 * h] : ATT_NONE;
                    scores[line * ATT_KEYS + at + 1] = first + at + 1 < len ? sc[i][2 * h + 1] : ATT_NONE;
                }
            }
        }
        __syncthreads();

        // The softmax: a warp takes its share of the lines, a lane a key.
#pragma unroll
        for (u32 j = 0; j < WARP_LINES; j++) {
            u32 line = warp * WARP_LINES + j;
            float score = scores[line * ATT_KEYS + lane];
            float top = fmaxf(largest[j], warp_max(score));
            // What the line has summed is scaled down to its new maximum.
            float a = largest[j] > 0.5f * ATT_NONE ? exp2_fast((largest[j] - top) * 1.4426950408889634f) : 0.0f;
            float w = score > 0.5f * ATT_NONE ? exp2_fast((score - top) * 1.4426950408889634f) : 0.0f;
            sum[j] = sum[j] * a + warp_sum(w);
            largest[j] = top;
            *(u16*)(weights + weight_at(line, lane >> 3) + (lane & 7) * 2) = (u16)f32_to_f16(w);
            if (lane == 0) {
                alpha[line] = a;
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
#pragma unroll
                for (u32 n = 0; n < WARP_RUNS; n++) {
                    o[i][n][0] *= a0;
                    o[i][n][1] *= a0;
                    o[i][n][2] *= a1;
                    o[i][n][3] *= a1;
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
                // The lane's four keys of the sixteen.
                const u32 key = kk * 16 + 2 * (lane & 3);
#pragma unroll
                for (u32 n = 0; n < WARP_RUNS; n++) {
                    u32 dim = (warp * WARP_RUNS + n) * 8 + (lane >> 2);
                    const unsigned char* at = values + key * ATT_DIM + dim;
                    const u16* scale = value_scales + key * 8 + dim / CACHE_BLOCK;
                    u32 b0 = q8_pair(at[0] | (u32)at[ATT_DIM] << 8, 0, scale[0] | (u32)scale[8] << 16);
                    u32 b1 = q8_pair(at[8 * ATT_DIM] | (u32)at[9 * ATT_DIM] << 8, 0, scale[64] | (u32)scale[72] << 16);
#pragma unroll
                    for (u32 i = 0; i < MT; i++) {
                        multiply_add(o[i][n], pa[i], b0, b1);
                    }
                }
            }
        }
    }
    __syncthreads();
    if (lane == 0) {
#pragma unroll
        for (u32 j = 0; j < WARP_LINES; j++) {
            line_max[warp * WARP_LINES + j] = largest[j];
            line_sum[warp * WARP_LINES + j] = sum[j];
        }
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
