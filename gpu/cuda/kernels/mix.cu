// Attention over the depth: out[t] is a mix of token t's rows of up to
// six states, each normalized by its root mean square, weighted by the
// softmax over the states of each normalized row's dot product with the
// query, over the square root of the width. One block of 256 threads per
// token: its threads take four elements each of every state, the sums of
// squares and the dot products are added up across the block, and each
// thread writes its elements of the mix.

// Vectors a thread takes: rows of up to 4096 elements.
#define MIX_PER_THREAD 4
#define MIX_STATES 6

// Several mixes of the same states at once, each with a query of its own,
// and each then, with `normed`, normalized by its root mean square and
// scaled by its weight, as `rmsnorm` does: the inputs of a layer's
// queries, keys and values, read from the depth once for the three. A mix
// goes to its floats, to its halves, or to both, whichever it has. A
// kernel for each number of mixes a layer has, so that a thread keeps
// what it has read in registers.
#define MIXES 4

struct MixesParams {
    float4* out[MIXES];
    uint2* halves[MIXES];
    const float4* query[MIXES];
    const float4* norm[MIXES];
    const float4* state[MIX_STATES];
    u32 dim;
    u32 states;
    u32 mixes;
    u32 normed;
    float eps;
    float norm_eps;
};

template <int M>
__device__ __forceinline__ void mixes_body(MixesParams p) {
    __shared__ float partial[8][MIX_STATES * (1 + M)];
    __shared__ float after[8][M];
    const u32 lid = threadIdx.x;
    const u32 lane = lid & 31;
    const u32 warp = lid >> 5;
    u32 vectors = p.dim / 4;
    size_t base = (size_t)blockIdx.x * vectors;
    // The thread's elements of every state and of every query, read
    // before any is used: the index is clamped rather than the load
    // skipped, so that they go out together.
    float4 vs[MIX_STATES][MIX_PER_THREAD];
    float4 qs[M][MIX_PER_THREAD];
#pragma unroll
    for (u32 k = 0; k < MIX_PER_THREAD; k++) {
        u32 i = min(lid + k * 256, vectors - 1);
#pragma unroll
        for (u32 s = 0; s < MIX_STATES; s++) {
            vs[s][k] = p.state[min(s, p.states - 1)][base + i];
        }
#pragma unroll
        for (u32 j = 0; j < M; j++) {
            qs[j][k] = p.query[min(j, p.mixes - 1)][i];
        }
    }
#pragma unroll
    for (u32 s = 0; s < MIX_STATES; s++) {
        float squares = 0.0f;
        float products[M];
#pragma unroll
        for (u32 j = 0; j < M; j++) {
            products[j] = 0.0f;
        }
#pragma unroll
        for (u32 k = 0; k < MIX_PER_THREAD; k++) {
            if (s >= p.states || lid + k * 256 >= vectors) {
                vs[s][k] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
            squares += dot4(vs[s][k], vs[s][k]);
#pragma unroll
            for (u32 j = 0; j < M; j++) {
                products[j] += dot4(vs[s][k], qs[j][k]);
            }
        }
        squares = warp_sum(squares);
#pragma unroll
        for (u32 j = 0; j < M; j++) {
            products[j] = warp_sum(products[j]);
        }
        if (lane == 0) {
            partial[warp][s * (1 + M)] = squares;
#pragma unroll
            for (u32 j = 0; j < M; j++) {
                partial[warp][s * (1 + M) + 1 + j] = products[j];
            }
        }
    }
    __syncthreads();

    // What normalizes each state.
    float norms[MIX_STATES];
#pragma unroll
    for (u32 s = 0; s < MIX_STATES; s++) {
        float squares = 0.0f;
#pragma unroll
        for (u32 i = 0; i < 8; i++) {
            squares += partial[i][s * (1 + M)];
        }
        norms[s] = rsqrtf(squares / (float)p.dim + p.eps);
    }
    // Each mix: the states' scores against its query, and each one's
    // share of the mix, times what normalizes it.
    float shares[M][MIX_STATES];
#pragma unroll
    for (u32 j = 0; j < M; j++) {
        float most = -1e30f;
#pragma unroll
        for (u32 s = 0; s < MIX_STATES; s++) {
            float product = 0.0f;
#pragma unroll
            for (u32 i = 0; i < 8; i++) {
                product += partial[i][s * (1 + M) + 1 + j];
            }
            shares[j][s] = product * norms[s] * rsqrtf((float)p.dim);
            if (s < p.states) {
                most = fmaxf(most, shares[j][s]);
            }
        }
        float sum = 0.0f;
#pragma unroll
        for (u32 s = 0; s < MIX_STATES; s++) {
            shares[j][s] = s < p.states ? expf(shares[j][s] - most) : 0.0f;
            sum += shares[j][s];
        }
#pragma unroll
        for (u32 s = 0; s < MIX_STATES; s++) {
            shares[j][s] = shares[j][s] / sum * norms[s];
        }
    }
    // The thread's elements of mix j.
    auto mix = [&](u32 j, u32 k) {
        float4 o = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
        for (u32 s = 0; s < MIX_STATES; s++) {
            o = add4(o, scale4(vs[s][k], shares[j][s]));
        }
        return o;
    };
    float scales[M];
    if (p.normed) {
#pragma unroll
        for (u32 j = 0; j < M; j++) {
            float squares = 0.0f;
#pragma unroll
            for (u32 k = 0; k < MIX_PER_THREAD; k++) {
                float4 o = mix(j, k);
                squares += dot4(o, o);
            }
            squares = warp_sum(squares);
            if (lane == 0) {
                after[warp][j] = squares;
            }
        }
        __syncthreads();
#pragma unroll
        for (u32 j = 0; j < M; j++) {
            float total = 0.0f;
#pragma unroll
            for (u32 i = 0; i < 8; i++) {
                total += after[i][j];
            }
            scales[j] = rsqrtf(total / (float)p.dim + p.norm_eps);
        }
    }
#pragma unroll
    for (u32 j = 0; j < M; j++) {
        if (j >= p.mixes) {
            continue;
        }
#pragma unroll
        for (u32 k = 0; k < MIX_PER_THREAD; k++) {
            u32 i = lid + k * 256;
            if (i < vectors) {
                float4 v = mix(j, k);
                if (p.normed) {
                    v = mul4(scale4(v, scales[j]), p.norm[j][i]);
                }
                if (p.out[j]) {
                    p.out[j][base + i] = v;
                }
                if (p.halves[j]) {
                    p.halves[j][base + i] = halves4(v);
                }
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__(256) mixes1(MixesParams p) {
    mixes_body<1>(p);
}

extern "C" __global__ void __launch_bounds__(256) mixes3(MixesParams p) {
    mixes_body<3>(p);
}

extern "C" __global__ void __launch_bounds__(256) mixes4(MixesParams p) {
    mixes_body<4>(p);
}
