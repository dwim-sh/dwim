// out[t] = sum over e of weights[t][e] * y[t][e]: the outputs of a token's
// k experts, a row of `width` each, summed by what each counts for.
// Rows whose width is a multiple of four are summed four elements per thread.

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint len;
    uint width;
    uint k;
};

kernel void weighted_sum(
    device float* out [[buffer(0)]],
    const device float* y [[buffer(1)]],
    const device float* weights [[buffer(2)]],
    constant Params& p [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    if ((p.width & 3u) == 0) {
        uint quads = p.len / 4;
        if (i >= quads) {
            return;
        }
        uint wq = p.width / 4;
        uint t = i / wq;
        uint j = i % wq;
        const device float4* ys = (const device float4*)y;
        float4 acc = float4(0.0f);
        for (uint e = 0; e < p.k; e++) {
            acc += weights[t * p.k + e] * ys[(t * p.k + e) * wq + j];
        }
        ((device float4*)out)[i] = acc;
        return;
    }
    if (i >= p.len) {
        return;
    }
    uint t = i / p.width;
    uint j = i % p.width;
    float acc = 0.0f;
    for (uint e = 0; e < p.k; e++) {
        acc += weights[t * p.k + e] * y[(t * p.k + e) * p.width + j];
    }
    out[i] = acc;
}
