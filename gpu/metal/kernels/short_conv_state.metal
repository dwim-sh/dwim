// Moves the state of a short convolution on past a batch: the rows of the
// `taps - 1` tokens before the next, the last of the batch, or of the
// state as it was where the batch is shorter. A thread per channel reads
// what it needs of the old state before it writes the new.

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint n;
    uint channels;
    uint taps;
    uint heads;
    uint add;
};

// Most taps a convolution has, less one.
constant uint KEPT = 8;

kernel void short_conv_state(
    device float* state [[buffer(0)]],
    const device float* x [[buffer(1)]],
    constant Params& p [[buffer(2)]],
    uint c [[thread_position_in_grid]])
{
    if (c >= p.channels) {
        return;
    }
    float next[KEPT];
    for (uint s = 0; s + 1 < p.taps; s++) {
        next[s] = p.n + s + 1 >= p.taps ? x[(p.n + s + 1 - p.taps) * p.channels + c] : state[(p.n + s) * p.channels + c];
    }
    for (uint s = 0; s + 1 < p.taps; s++) {
        state[s * p.channels + c] = next[s];
    }
}
