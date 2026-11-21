// A short causal convolution with a residual: each thread convolves one
// channel of one token over the token and the `taps - 1` before it,
// reading the ones before the batch from the state, the last tap for the
// token itself. With `heads`, the channels are split evenly between as
// many heads, each of which has `taps` more for the token in `dynamic`:
// the token `lag` back is also taken times the head's tap `lag`. The sum
// is added to the channel, and the whole divided by the square root of
// the taps and one. A token alone has its state moved on here, by the
// thread of each channel, once it has read what it needs of it; for a
// batch, `short_conv_state.metal` moves the state on afterwards.

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint n;
    uint channels;
    uint taps;
    uint heads;
};

// Channel c of the token `back` before token t, which is before the batch
// when further back than t.
static float input(const device float* x, device float* state, uint channels, uint taps, uint t, uint back, uint c) {
    if (back > t) {
        return state[(taps - 1 + t - back) * channels + c];
    }
    return x[(t - back) * channels + c];
}

kernel void short_conv(
    device float* out [[buffer(0)]],
    const device float* x [[buffer(1)]],
    device float* state [[buffer(2)]],
    const device float* weight [[buffer(3)]],
    const device float* dynamic [[buffer(4)]],
    constant Params& p [[buffer(5)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= p.n * p.channels) {
        return;
    }
    uint t = i / p.channels;
    uint c = i % p.channels;
    float acc = x[i];
    for (uint j = 0; j < p.taps; j++) {
        acc += weight[c * p.taps + j] * input(x, state, p.channels, p.taps, t, p.taps - 1 - j, c);
    }
    if (p.heads != 0) {
        uint head = c / (p.channels / p.heads);
        for (uint lag = 0; lag < p.taps; lag++) {
            acc += dynamic[(t * p.heads + head) * p.taps + lag] * input(x, state, p.channels, p.taps, t, lag, c);
        }
    }
    out[i] = acc * rsqrt(float((p.heads != 0 ? 2 : 1) * p.taps + 1));
    if (p.n == 1) {
        for (uint s = 0; s + 2 < p.taps; s++) {
            state[s * p.channels + c] = state[(s + 1) * p.channels + c];
        }
        state[(p.taps - 2) * p.channels + c] = x[c];
    }
}
