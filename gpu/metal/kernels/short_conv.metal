// A short causal convolution with a residual: each thread convolves one
// channel of one token over the token and the `taps - 1` before it,
// reading the ones before the batch from the state, the last tap for the
// token itself. With `heads`, the channels are split evenly between as
// many heads, each of which has `taps` more for the token in `dynamic`:
// the token `lag` back is also taken times the head's tap `lag`. Static
// and dynamic taps share one pass over the inputs. With `add`, the result
// is added into `out` (the residual stream). A token alone has its state
// moved on here; for a batch, `short_conv_state.metal` moves it on after.

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint n;
    uint channels;
    uint taps;
    uint heads;
    uint add;
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
    uint head = p.heads != 0 ? c / (p.channels / p.heads) : 0;
    float acc = x[i];
    for (uint lag = 0; lag < p.taps; lag++) {
        float w = weight[c * p.taps + (p.taps - 1 - lag)];
        if (p.heads != 0) {
            w += dynamic[(t * p.heads + head) * p.taps + lag];
        }
        acc += w * input(x, state, p.channels, p.taps, t, lag, c);
    }
    float v = acc * rsqrt(float((p.heads != 0 ? 2 : 1) * p.taps + 1));
    out[i] = p.add != 0 ? out[i] + v : v;
    // A convolution with one tap has no state to move on.
    if (p.n == 1 && p.taps > 1) {
        for (uint s = 0; s + 2 < p.taps; s++) {
            state[s * p.channels + c] = state[(s + 1) * p.channels + c];
        }
        state[(p.taps - 2) * p.channels + c] = x[c];
    }
}
