// A convolution over the rows of a block, with taps of each row's own:
// each thread convolves one channel of one row over the row and the
// `kernel - 1` before it. The tap for a row `back` rows before is the
// channel's of `base`, which holds a row of channels per distance, plus
// the row's own for the channel's group, which `taps` holds from `from`
// on in each of its rows, `width` long: a group's for each distance. The
// rows before the block count as zeros.

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint n;
    uint channels;
    uint kernel_size;
    uint group;
    uint width;
    uint from;
};

kernel void block_conv(
    device float* out [[buffer(0)]],
    const device float* x [[buffer(1)]],
    const device float* taps [[buffer(2)]],
    const device float* base [[buffer(3)]],
    constant Params& p [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= p.n * p.channels) {
        return;
    }
    uint t = i / p.channels;
    uint c = i % p.channels;
    uint groups = p.channels / p.group;
    float acc = 0.0;
    for (uint back = 0; back < p.kernel_size && back <= t; back++) {
        float tap = base[back * p.channels + c]
            + taps[t * p.width + p.from + back * groups + c / p.group];
        acc += tap * x[(t - back) * p.channels + c];
    }
    out[i] = acc;
}
