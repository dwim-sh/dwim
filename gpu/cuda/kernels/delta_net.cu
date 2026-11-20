// The gated delta rule of linear attention, stepping through the tokens in
// order. A head's state is a matrix of keys by values, and what happens to
// one value's column of it does not depend on the others': a block of 128
// threads takes 64 values of a head, two threads to a value, each keeping
// half of the value's column in registers, a run of four keys in every
// eight, so that the two are side by side in a warp and add their halves
// up without waiting for the block. The tokens' keys, queries, values and
// gates are read 32 steps at a time into shared memory, each lot while the
// one before is stepped through. The scheme is TensorFold's
// (`licenses/tensorfold.md`).

// The head dimension the kernel is written for, the values a block takes,
// and the steps it reads at once.
#define DELTA_DIM 128
#define DELTA_VALUES 64
#define DELTA_STEPS 32
#define DELTA_THREADS (2 * DELTA_VALUES)
// What a thread reads of a lot: runs of four of the keys and the queries,
// and of the values.
#define DELTA_KEY_RUNS (DELTA_STEPS * (DELTA_DIM / 4) / DELTA_THREADS)
#define DELTA_VALUE_RUNS (DELTA_STEPS * (DELTA_VALUES / 4) / DELTA_THREADS)

extern "C" __global__ void __launch_bounds__(DELTA_THREADS) delta_net(
    float* out, const float4* q, const float4* k, const float4* v, const float* gates, const float* decay,
    float* state, DeltaNetParams p) {
    __shared__ float4 keys[DELTA_STEPS][DELTA_DIM / 4];
    __shared__ float4 queries[DELTA_STEPS][DELTA_DIM / 4];
    __shared__ float4 values[DELTA_STEPS][DELTA_VALUES / 4];
    __shared__ float decays[DELTA_STEPS];
    __shared__ float betas[DELTA_STEPS];
    const u32 tid = threadIdx.x;
    const u32 half = tid & 1;
    u32 h = blockIdx.x / (DELTA_DIM / DELTA_VALUES);
    u32 first = (blockIdx.x % (DELTA_DIM / DELTA_VALUES)) * DELTA_VALUES;
    u32 j = first + (tid >> 1);
    u32 kh = h / (p.n_v_heads / p.n_k_heads);
    float* column = state + h * DELTA_DIM * DELTA_DIM + j;
    // The thread's keys are 8i + 4 * half and the three after, for each i.
    float s[DELTA_DIM / 2];
#pragma unroll
    for (u32 i = 0; i < DELTA_DIM / 2; i++) {
        s[i] = column[(8 * (i / 4) + 4 * half + i % 4) * DELTA_DIM];
    }
    float scale = rsqrtf((float)DELTA_DIM);
    float a = decay[h];
    float dt_bias = decay[p.n_v_heads + h];

    // A lot's inputs, from the tokens' buffers into registers: a token
    // past the last is read as the last and not stepped through.
    float4 key_runs[DELTA_KEY_RUNS], query_runs[DELTA_KEY_RUNS], value_runs[DELTA_VALUE_RUNS];
    float alpha, beta;
    auto fetch = [&](u32 t0) {
#pragma unroll
        for (u32 r = 0; r < DELTA_KEY_RUNS; r++) {
            u32 i = tid + r * DELTA_THREADS;
            u32 t = min(t0 + i / (DELTA_DIM / 4), p.n - 1);
            u32 at = (t * p.n_k_heads + kh) * (DELTA_DIM / 4) + i % (DELTA_DIM / 4);
            key_runs[r] = k[at];
            query_runs[r] = q[at];
        }
#pragma unroll
        for (u32 r = 0; r < DELTA_VALUE_RUNS; r++) {
            u32 i = tid + r * DELTA_THREADS;
            u32 t = min(t0 + i / (DELTA_VALUES / 4), p.n - 1);
            value_runs[r] = v[((t * p.n_v_heads + h) * DELTA_DIM + first) / 4 + i % (DELTA_VALUES / 4)];
        }
        if (tid < DELTA_STEPS) {
            u32 t = min(t0 + tid, p.n - 1);
            alpha = gates[t * 2 * p.n_v_heads + h];
            beta = gates[t * 2 * p.n_v_heads + p.n_v_heads + h];
        }
    };
    // And from the registers into shared memory, the gates as what the
    // state decays by and how far the value is taken.
    auto put = [&]() {
#pragma unroll
        for (u32 r = 0; r < DELTA_KEY_RUNS; r++) {
            u32 i = tid + r * DELTA_THREADS;
            keys[i / (DELTA_DIM / 4)][i % (DELTA_DIM / 4)] = key_runs[r];
            queries[i / (DELTA_DIM / 4)][i % (DELTA_DIM / 4)] = query_runs[r];
        }
#pragma unroll
        for (u32 r = 0; r < DELTA_VALUE_RUNS; r++) {
            u32 i = tid + r * DELTA_THREADS;
            values[i / (DELTA_VALUES / 4)][i % (DELTA_VALUES / 4)] = value_runs[r];
        }
        if (tid < DELTA_STEPS) {
            float z = alpha + dt_bias;
            float softplus = z > 20.0f ? z : logf(1.0f + expf(z));
            decays[tid] = expf(a * softplus);
            betas[tid] = sigmoid(beta);
        }
    };

    fetch(0);
    for (u32 t0 = 0; t0 < p.n; t0 += DELTA_STEPS) {
        u32 steps = min((u32)DELTA_STEPS, p.n - t0);
        __syncthreads();
        put();
        __syncthreads();
        if (t0 + DELTA_STEPS < p.n) {
            fetch(t0 + DELTA_STEPS);
        }
        for (u32 step = 0; step < steps; step++) {
            float g = decays[step];
            float value = ((const float*)values[step])[tid >> 1];
            // The state decays, and what it holds for the key is read out.
            float mem[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (u32 i = 0; i < DELTA_DIM / 8; i++) {
                float4 key = keys[step][2 * i + half];
                s[4 * i] *= g;
                s[4 * i + 1] *= g;
                s[4 * i + 2] *= g;
                s[4 * i + 3] *= g;
                mem[0] += s[4 * i] * key.x;
                mem[1] += s[4 * i + 1] * key.y;
                mem[2] += s[4 * i + 2] * key.z;
                mem[3] += s[4 * i + 3] * key.w;
            }
            float held = (mem[0] + mem[1]) + (mem[2] + mem[3]);
            held += __shfl_xor_sync(FULL, held, 1);
            float delta = (value - held) * betas[step];
            // The value takes its place, and the query reads the state out.
            float o[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (u32 i = 0; i < DELTA_DIM / 8; i++) {
                float4 key = keys[step][2 * i + half];
                float4 query = queries[step][2 * i + half];
                s[4 * i] += key.x * delta;
                s[4 * i + 1] += key.y * delta;
                s[4 * i + 2] += key.z * delta;
                s[4 * i + 3] += key.w * delta;
                o[0] += s[4 * i] * query.x;
                o[1] += s[4 * i + 1] * query.y;
                o[2] += s[4 * i + 2] * query.z;
                o[3] += s[4 * i + 3] * query.w;
            }
            float read = (o[0] + o[1]) + (o[2] + o[3]);
            read += __shfl_xor_sync(FULL, read, 1);
            if (half == 0) {
                out[((t0 + step) * p.n_v_heads + h) * DELTA_DIM + j] = read * scale;
            }
        }
    }
#pragma unroll
    for (u32 i = 0; i < DELTA_DIM / 2; i++) {
        column[(8 * (i / 4) + 4 * half + i % 4) * DELTA_DIM] = s[i];
    }
}
