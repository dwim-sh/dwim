// The gated delta rule of linear attention: one block of 256 threads per
// value head, stepping through the tokens in order. The head's state is a
// matrix of keys by values; each thread keeps a column of half of it in
// registers, so what the state holds for a key is summed over the two
// halves through shared memory, as is the output for the query.

// The head dimension the kernel is written for: 256 threads, each holding
// 64 of the 128 by 128 state.
#define DELTA_DIM 128
#define DELTA_HALF 64

extern "C" __global__ void __launch_bounds__(256) delta_net(
    float* out, const float4* q, const float4* k, const float* v, const float* gates, const float* decay,
    float* state, DeltaNetParams p) {
    __shared__ float kv[256];
    __shared__ float oq[256];
    const u32 lid = threadIdx.x;
    u32 h = blockIdx.x;
    u32 j = lid % DELTA_DIM;
    u32 half = lid / DELTA_DIM;
    u32 kh = h / (p.n_v_heads / p.n_k_heads);
    u32 sbase = h * DELTA_DIM * DELTA_DIM;
    float4 s[16];
#pragma unroll
    for (u32 i = 0; i < 16; i++) {
        u32 row = half * DELTA_HALF + 4 * i;
        s[i] = make_float4(
            state[sbase + row * DELTA_DIM + j],
            state[sbase + (row + 1) * DELTA_DIM + j],
            state[sbase + (row + 2) * DELTA_DIM + j],
            state[sbase + (row + 3) * DELTA_DIM + j]);
    }
    float scale = rsqrtf((float)DELTA_DIM);
    float a = decay[h];
    float dt_bias = decay[p.n_v_heads + h];
    for (u32 t = 0; t < p.n; t++) {
        float alpha = gates[t * 2 * p.n_v_heads + h];
        float beta_raw = gates[t * 2 * p.n_v_heads + p.n_v_heads + h];
        float z = alpha + dt_bias;
        float softplus = z > 20.0f ? z : logf(1.0f + expf(z));
        float g = expf(a * softplus);
        float beta = sigmoid(beta_raw);
        // The state decays, and what it holds for the key is read out.
        u32 kbase = ((t * p.n_k_heads + kh) * DELTA_DIM + half * DELTA_HALF) / 4;
        float mem = 0.0f;
#pragma unroll
        for (u32 i = 0; i < 16; i++) {
            s[i] = scale4(s[i], g);
            mem += dot4(s[i], k[kbase + i]);
        }
        kv[lid] = mem;
        __syncthreads();
        float delta = (v[(t * p.n_v_heads + h) * DELTA_DIM + j] - kv[j] - kv[DELTA_DIM + j]) * beta;
        // The value takes its place, and the query reads the state out.
        u32 qbase = ((t * p.n_k_heads + kh) * DELTA_DIM + half * DELTA_HALF) / 4;
        float o = 0.0f;
#pragma unroll
        for (u32 i = 0; i < 16; i++) {
            s[i] = add4(s[i], scale4(k[kbase + i], delta));
            o += dot4(s[i], q[qbase + i]);
        }
        oq[lid] = o;
        __syncthreads();
        if (half == 0) {
            out[(t * p.n_v_heads + h) * DELTA_DIM + j] = (oq[j] + oq[DELTA_DIM + j]) * scale;
        }
        __syncthreads();
    }
#pragma unroll
    for (u32 i = 0; i < 16; i++) {
        u32 row = half * DELTA_HALF + 4 * i;
        state[sbase + row * DELTA_DIM + j] = s[i].x;
        state[sbase + (row + 1) * DELTA_DIM + j] = s[i].y;
        state[sbase + (row + 2) * DELTA_DIM + j] = s[i].z;
        state[sbase + (row + 3) * DELTA_DIM + j] = s[i].w;
    }
}
