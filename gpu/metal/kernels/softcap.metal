// Soft cap on logits in place: x[i] = cap * tanh(x[i] / cap).

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint len;
    float cap;
};

kernel void softcap(
    device float* x [[buffer(0)]],
    constant Params& p [[buffer(1)]],
    uint i [[thread_position_in_grid]])
{
    if (i < p.len) {
        x[i] = p.cap * tanh(clamp(x[i] / p.cap, -15.0f, 15.0f));
    }
}
