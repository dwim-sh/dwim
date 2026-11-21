// The SiTU activation: each row of gate_up is a gate then an up
// projection, each `width` wide, and a row of out gets
// beta * tanh(gate / beta) * sigmoid(gate) * linear_beta * tanh(up / linear_beta).

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint len;
    uint width;
    float beta;
    float linear_beta;
};

// tanh, which the GPU's own makes not a number of where its exponentials
// overflow: past 15 either way it is one to the last bit.
static float safe_tanh(float x) {
    return tanh(clamp(x, -15.0f, 15.0f));
}

kernel void situ(
    device float* out [[buffer(0)]],
    const device float* gate_up [[buffer(1)]],
    constant Params& p [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= p.len) {
        return;
    }
    uint row = i / p.width;
    uint j = i % p.width;
    float gate = gate_up[row * 2 * p.width + j];
    float up = gate_up[row * 2 * p.width + p.width + j];
    out[i] = p.beta * safe_tanh(gate / p.beta) / (1.0f + exp(-gate)) * p.linear_beta * safe_tanh(up / p.linear_beta);
}
