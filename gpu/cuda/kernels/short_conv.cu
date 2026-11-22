// A short causal convolution with a residual: each thread convolves one
// channel of one token over the token and the `taps - 1` before it,
// reading the ones before the batch from the state, the last tap for the
// token itself. With `heads`, the channels are split evenly between as
// many heads, each of which has `taps` more for the token in `dynamic`:
// the token `lag` back is also taken times the head's tap `lag`. The sum
// is added to the channel, and the whole divided by the square root of
// the taps and one. A token alone has its state moved on here, by the
// thread of each channel, once it has read what it needs of it; for a
// batch, `short_conv_state` moves the state on afterwards. With `add`,
// the output is added to what `out` holds: the residual stream.

// Most taps a convolution has, less one.
#define SHORT_CONV_KEPT 8

// Channel c of the token `back` before token t, which is before the batch
// when further back than t.
__device__ __forceinline__ float short_conv_input(
    const float* x, const float* state, ShortConvParams p, u32 t, u32 back, u32 c) {
    if (back > t) {
        return state[(size_t)(p.taps - 1 + t - back) * p.channels + c];
    }
    return x[(size_t)(t - back) * p.channels + c];
}

// Channel c of token t, convolved, and the state moved on past a token
// alone.
__device__ __forceinline__ float short_conv_channel(
    const float* x, float* state, const float* weight, const float* dynamic, ShortConvParams p, u32 t, u32 c) {
    float acc = x[(size_t)t * p.channels + c];
    for (u32 j = 0; j < p.taps; j++) {
        acc += weight[c * p.taps + j] * short_conv_input(x, state, p, t, p.taps - 1 - j, c);
    }
    if (p.heads != 0) {
        u32 head = c / (p.channels / p.heads);
        for (u32 lag = 0; lag < p.taps; lag++) {
            acc += dynamic[((size_t)t * p.heads + head) * p.taps + lag] * short_conv_input(x, state, p, t, lag, c);
        }
    }
    if (p.n == 1) {
        for (u32 s = 0; s + 2 < p.taps; s++) {
            state[s * p.channels + c] = state[(s + 1) * p.channels + c];
        }
        state[(p.taps - 2) * p.channels + c] = x[c];
    }
    return acc * rsqrtf((float)((p.heads != 0 ? 2 : 1) * p.taps + 1));
}

extern "C" __global__ void short_conv(
    float* out, const float* x, float* state, const float* weight, const float* dynamic, ShortConvParams p) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)p.n * p.channels) {
        return;
    }
    float v = short_conv_channel(x, state, weight, dynamic, p, i / p.channels, i % p.channels);
    out[i] = p.add != 0 ? out[i] + v : v;
}

// Moves the state on past a batch: the rows of the `taps - 1` tokens
// before the next, the last of the batch, or of the state as it was where
// the batch is shorter. A thread per channel reads what it needs of the
// old state before it writes the new.
extern "C" __global__ void short_conv_state(float* state, const float* x, ShortConvParams p) {
    u32 c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= p.channels) {
        return;
    }
    float next[SHORT_CONV_KEPT];
    for (u32 s = 0; s + 1 < p.taps; s++) {
        next[s] = p.n + s + 1 >= p.taps
            ? x[(size_t)(p.n + s + 1 - p.taps) * p.channels + c]
            : state[(p.n + s) * p.channels + c];
    }
    for (u32 s = 0; s + 1 < p.taps; s++) {
        state[s * p.channels + c] = next[s];
    }
}
