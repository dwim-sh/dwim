// Applies SiTU directly to the expert-sorted gate_up output and writes half4
// activations into the packed input buffer for the down projection, zeroing
// the padding rows past `pairs`. Eliminates expert_scatter + expert_pack
// between gate_up and down in Device::expert_ffn.

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint pairs;
    uint width;
    uint padded;
    float beta;
    float linear_beta;
};

static float4 safe_tanh4(float4 x) {
    return tanh(clamp(x, -15.0f, 15.0f));
}

kernel void expert_situ_pack(
    device half4* out [[buffer(0)]],
    const device float4* gate_up [[buffer(1)]],
    constant Params& p [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    uint wq = p.width / 4;
    if (i >= p.padded * wq) {
        return;
    }
    uint row = i / wq;
    if (row >= p.pairs) {
        out[i] = half4(0.0h);
        return;
    }
    uint j = i % wq;
    float4 gate = gate_up[row * 2 * wq + j];
    float4 up = gate_up[row * 2 * wq + wq + j];
    float inv_beta = 1.0f / p.beta;
    float inv_linear_beta = 1.0f / p.linear_beta;
    float scale = p.beta * p.linear_beta;
    float4 v = scale * safe_tanh4(gate * inv_beta) / (1.0f + exp(-gate)) * safe_tanh4(up * inv_linear_beta);
    out[i] = half4(v);
}
