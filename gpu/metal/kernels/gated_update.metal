// x = x - tanh(forget) * x + tanh(gain) * message: what a gate lets into
// the residual stream of a message, and what it has the stream forget.

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint len;
};

// tanh, which the GPU's own makes not a number of where its exponentials
// overflow: past 15 either way it is one to the last bit.
static float safe_tanh(float x) {
    return tanh(clamp(x, -15.0f, 15.0f));
}

kernel void gated_update(
    device float* x [[buffer(0)]],
    const device float* forget [[buffer(1)]],
    const device float* gain [[buffer(2)]],
    const device float* message [[buffer(3)]],
    constant Params& p [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
    if (i < p.len) {
        x[i] = x[i] - safe_tanh(forget[i]) * x[i] + safe_tanh(gain[i]) * message[i];
    }
}
