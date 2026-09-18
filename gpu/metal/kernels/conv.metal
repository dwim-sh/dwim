// The causal convolution of linear attention: each thread convolves one
// channel of one token over the last four tokens, reading the ones before
// the batch from the state, passes the result through SiLU, and writes it to
// the q, k, or v buffer the channel belongs to. The threads of the first
// token also save the last three tokens' inputs as the next state.

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint n;
    uint channels;
    uint q_dim;
    uint k_dim;
};

constant int KERNEL = 4;

// Channel c of the token at t, which is before the batch when negative.
static float input(const device float* x, const device float* state, uint channels, int t, uint c) {
    if (t < 0) {
        return state[uint(t + KERNEL - 1) * channels + c];
    }
    return x[uint(t) * channels + c];
}

kernel void conv(
    device float* q [[buffer(0)]],
    device float* k [[buffer(1)]],
    device float* v [[buffer(2)]],
    device float* state_out [[buffer(3)]],
    const device float* x [[buffer(4)]],
    const device float* state [[buffer(5)]],
    const device float* weight [[buffer(6)]],
    constant Params& p [[buffer(7)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= p.n * p.channels) {
        return;
    }
    uint t = i / p.channels;
    uint c = i % p.channels;
    float acc = 0.0f;
    for (int j = 0; j < KERNEL; j++) {
        acc += weight[c * KERNEL + j] * input(x, state, p.channels, int(t) + j + 1 - KERNEL, c);
    }
    float y = acc / (1.0f + exp(-acc));
    uint v_dim = p.channels - p.q_dim - p.k_dim;
    if (c < p.q_dim) {
        q[t * p.q_dim + c] = y;
    } else if (c < p.q_dim + p.k_dim) {
        k[t * p.k_dim + c - p.q_dim] = y;
    } else {
        v[t * v_dim + c - p.q_dim - p.k_dim] = y;
    }
    if (t == 0) {
        for (int s = 0; s < KERNEL - 1; s++) {
            state_out[uint(s) * p.channels + c] = input(x, state, p.channels, int(p.n) + s + 1 - KERNEL, c);
        }
    }
}
