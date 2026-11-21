// Scales each row of x, `width` long, by `scale`, then each head of it,
// `dim` long, down to a length of `threshold` if it is longer: a thread
// per head.

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint heads;
    uint width;
    uint dim;
    float threshold;
};

kernel void clip(
    device float* x [[buffer(0)]],
    const device float* scale [[buffer(1)]],
    constant Params& p [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= p.heads) {
        return;
    }
    device float* head = x + i * p.dim;
    const device float* by = scale + (i % (p.width / p.dim)) * p.dim;
    float square = 0.0f;
    for (uint j = 0; j < p.dim; j++) {
        float v = head[j] * by[j];
        square += v * v;
    }
    float factor = min(p.threshold / (sqrt(square) + 1e-6f), 1.0f);
    for (uint j = 0; j < p.dim; j++) {
        head[j] = head[j] * by[j] * factor;
    }
}
