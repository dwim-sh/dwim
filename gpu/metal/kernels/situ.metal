// The SiTU activation: each row of gate_up is a gate then an up
// projection, each `width` wide, and a row of out gets
// beta * tanh(gate / beta) * sigmoid(gate) * linear_beta * tanh(up / linear_beta).
// Rows whose width is a multiple of four are processed four elements per thread.

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

static float4 safe_tanh4(float4 x) {
    return tanh(clamp(x, -15.0f, 15.0f));
}

kernel void situ(
    device float* out [[buffer(0)]],
    const device float* gate_up [[buffer(1)]],
    constant Params& p [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    float inv_beta = 1.0f / p.beta;
    float inv_linear_beta = 1.0f / p.linear_beta;
    float scale = p.beta * p.linear_beta;
    if ((p.width & 3u) == 0) {
        uint quads = p.len / 4;
        if (i >= quads) {
            return;
        }
        uint wq = p.width / 4;
        uint row = i / wq;
        uint j = i % wq;
        const device float4* src = (const device float4*)gate_up;
        float4 gate = src[row * 2 * wq + j];
        float4 up = src[row * 2 * wq + wq + j];
        float4 v = scale * safe_tanh4(gate * inv_beta) / (1.0f + exp(-gate)) * safe_tanh4(up * inv_linear_beta);
        ((device float4*)out)[i] = v;
        return;
    }
    if (i >= p.len) {
        return;
    }
    uint row = i / p.width;
    uint j = i % p.width;
    float gate = gate_up[row * 2 * p.width + j];
    float up = gate_up[row * 2 * p.width + p.width + j];
    out[i] = scale * safe_tanh(gate * inv_beta) / (1.0f + exp(-gate)) * safe_tanh(up * inv_linear_beta);
}
