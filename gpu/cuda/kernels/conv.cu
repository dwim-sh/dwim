// The causal convolution of linear attention: each thread convolves one
// channel of one token over the last four tokens, reading the ones before
// the batch from the state, passes the result through SiLU, and writes it to
// the q, k, or v buffer the channel belongs to. The threads of the first
// token also save the last three tokens' inputs as the next state.

#define CONV_KERNEL 4

// Channel c of the token at `t`, which is before the batch when negative.
__device__ __forceinline__ float conv_input(const float* x, const float* state, u32 channels, i32 t, u32 c) {
    if (t < 0) {
        return state[(u32)(t + CONV_KERNEL - 1) * channels + c];
    }
    return x[(u32)t * channels + c];
}

extern "C" __global__ void conv(
    float* q, float* k, float* v, float* state_out, const float* x, const float* state, const float* weight, ConvParams p) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.n * p.channels) {
        return;
    }
    u32 t = i / p.channels;
    u32 c = i % p.channels;
    float acc = 0.0f;
    for (i32 j = 0; j < CONV_KERNEL; j++) {
        acc += weight[c * CONV_KERNEL + j] * conv_input(x, state, p.channels, (i32)t + j + 1 - CONV_KERNEL, c);
    }
    float y = acc * sigmoid(acc);
    u32 v_dim = p.channels - p.q_dim - p.k_dim;
    if (c < p.q_dim) {
        q[t * p.q_dim + c] = y;
    } else if (c < p.q_dim + p.k_dim) {
        k[t * p.k_dim + c - p.q_dim] = y;
    } else {
        v[t * v_dim + c - p.q_dim - p.k_dim] = y;
    }
    if (t == 0) {
        for (i32 s = 0; s < CONV_KERNEL - 1; s++) {
            state_out[(u32)s * p.channels + c] = conv_input(x, state, p.channels, (i32)p.n + s + 1 - CONV_KERNEL, c);
        }
    }
}
