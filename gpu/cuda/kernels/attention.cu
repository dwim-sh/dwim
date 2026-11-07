// Causal attention over 8-bit caches with a half-precision scale per block
// of 32: one block of 256 threads per (token, key/value head, chunk of 128
// positions), for every query head that shares the key/value head, so that
// the caches are read once for the group and a long context spreads over
// the GPU. A warp scores one position at a time, its lanes each taking
// four elements of the key and adding their products up across the warp,
// so that a key is read as one contiguous run, in as many rounds as the
// head is wider than the warp takes at once; the scores are softmaxed
// against the chunk's own maximum; then each thread takes four elements of
// the head at every position of its lane and sums the values the scores
// weigh, and the lanes' sums are added together. The chunk's maximum, sum
// of exponentials, and weighted values go to the partials buffer for
// `attention_combine.cu` to merge, or, when there is only one chunk,
// straight to the output. A dispatch covers the tokens from `first` on,
// with `chunks` chunks for each.

#define CHUNK 128
// Most query heads to a key/value head.
#define GROUP 8

// The four activations of a cache from `i` on, a multiple of four.
__device__ __forceinline__ float4 load_q8(const signed char* quants, const u16* scales, u32 i) {
    char4 c = *(const char4*)(quants + i);
    float s = f16_to_f32(scales[i / CACHE_BLOCK]);
    return make_float4(c.x * s, c.y * s, c.z * s, c.w * s);
}

extern "C" __global__ void __launch_bounds__(256) attention(
    float* out, const float* q, const signed char* k_quants, const u16* k_scales,
    const signed char* v_quants, const u16* v_scales, float* partials, AttentionParams p) {
    __shared__ float partial[8][GROUP];
    __shared__ float scores[GROUP][CHUNK];
    __shared__ float4 values[256];
    const u32 lid = threadIdx.x;
    const u32 lane = lid & 31;
    const u32 warp = lid >> 5;
    const u32 warps = 8;
    u32 c = blockIdx.x % p.chunks;
    u32 tk = blockIdx.x / p.chunks;
    u32 t = p.first + tk / p.n_kv_heads;
    u32 kvh = tk % p.n_kv_heads;
    u32 len = p.pos + t + 1;
    u32 start = c * CHUNK;
    if (start >= len) {
        return;
    }
    u32 count = min((u32)CHUNK, len - start);
    u32 group = p.n_heads / p.n_kv_heads;
    u32 kv = kvh * p.head_dim;
    u32 kv_dim = p.n_kv_heads * p.head_dim;
    float scale = rsqrtf((float)p.head_dim);
    // The first query head of the group, for this token.
    u32 qbase = (t * p.n_heads + kvh * group) * p.head_dim;

    // Each lane of a warp takes four elements of a key, the lanes side by
    // side, and keeps the queries' matching elements; a head wider than
    // the warp takes more than one round, each adding to the scores. A warp
    // scores the same positions in every round.
    u32 quads = p.head_dim / 4;
    u32 rounds = (quads + 31) / 32;
    for (u32 r = 0; r < rounds; r++) {
        u32 d = r * 32 + lane;
        bool in_head = d < quads;
        float4 query[GROUP];
#pragma unroll
        for (u32 g = 0; g < GROUP; g++) {
            query[g] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (g < group && in_head) {
                query[g] = *(const float4*)(q + qbase + g * p.head_dim + 4 * d);
            }
        }
        for (u32 i = warp; i < count; i += warps) {
            float4 key = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (in_head) {
                key = load_q8(k_quants, k_scales, (start + i) * kv_dim + kv + 4 * d);
            }
#pragma unroll
            for (u32 g = 0; g < GROUP; g++) {
                if (g < group) {
                    float s = warp_sum(dot4(query[g], key)) * scale;
                    if (lane == 0) {
                        scores[g][i] = r == 0 ? s : scores[g][i] + s;
                    }
                }
            }
        }
    }
    __syncthreads();

    // Each head's maximum over the chunk, for a stable softmax, and the
    // sum of its exponentials.
    float m[GROUP];
    float e[GROUP];
#pragma unroll
    for (u32 g = 0; g < GROUP; g++) {
        if (g < group) {
            float s = lid < count ? scores[g][lid] : -1e30f;
            float sm = warp_max(s);
            if (lane == 0) {
                partial[warp][g] = sm;
            }
        }
    }
    __syncthreads();
#pragma unroll
    for (u32 g = 0; g < GROUP; g++) {
        m[g] = -1e30f;
        e[g] = 0.0f;
        if (g < group) {
            for (u32 i = 0; i < warps; i++) {
                m[g] = fmaxf(m[g], partial[i][g]);
            }
            if (lid < count) {
                e[g] = expf(scores[g][lid] - m[g]);
            }
        }
    }
    __syncthreads();
    float total[GROUP];
#pragma unroll
    for (u32 g = 0; g < GROUP; g++) {
        if (g < group) {
            if (lid < CHUNK) {
                scores[g][lid] = e[g];
            }
            float sum = warp_sum(e[g]);
            if (lane == 0) {
                partial[warp][g] = sum;
            }
        }
    }
    __syncthreads();
#pragma unroll
    for (u32 g = 0; g < GROUP; g++) {
        total[g] = 0.0f;
        if (g < group) {
            for (u32 i = 0; i < warps; i++) {
                total[g] += partial[i][g];
            }
        }
    }

    // Each thread takes four elements of the head at every position of its
    // lane, the lanes striding through the positions together, so that a
    // position's values are read as one run and weighed for every head.
    u32 lanes = 256 / quads;
    u32 d = lid % quads;
    u32 vlane = lid / quads;
    float4 acc[GROUP];
#pragma unroll
    for (u32 g = 0; g < GROUP; g++) {
        acc[g] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    if (vlane < lanes) {
        for (u32 i = vlane; i < count; i += lanes) {
            float4 v = load_q8(v_quants, v_scales, (start + i) * kv_dim + kv + 4 * d);
#pragma unroll
            for (u32 g = 0; g < GROUP; g++) {
                if (g < group) {
                    acc[g] = add4(acc[g], scale4(v, scores[g][i]));
                }
            }
        }
    }
#pragma unroll
    for (u32 g = 0; g < GROUP; g++) {
        if (g < group) {
            values[lid] = acc[g];
            __syncthreads();
            if (lid < quads) {
                float4 o = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                for (u32 l = 0; l < lanes; l++) {
                    o = add4(o, values[l * quads + lid]);
                }
                u32 h = kvh * group + g;
                u32 obase = (t * p.n_heads + h) * p.head_dim + 4 * lid;
                if (p.chunks == 1) {
                    *(float4*)(out + obase) = scale4(o, 1.0f / total[g]);
                } else {
                    u32 base = (((t - p.first) * p.n_heads + h) * p.chunks + c) * (p.head_dim + 2);
                    if (lid == 0) {
                        partials[base] = m[g];
                        partials[base + 1] = total[g];
                    }
                    partials[base + 2 + 4 * lid] = o.x;
                    partials[base + 2 + 4 * lid + 1] = o.y;
                    partials[base + 2 + 4 * lid + 2] = o.z;
                    partials[base + 2 + 4 * lid + 3] = o.w;
                }
            }
            __syncthreads();
        }
    }
}
