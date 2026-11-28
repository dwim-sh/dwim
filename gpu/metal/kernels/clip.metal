// Scales each row of x, `width` long, by `scale`, then each head of it,
// `dim` long, down to a length of `threshold` if it is longer: one SIMD
// group per head, its lanes taking the head's elements together, for up to
// JOBS clips in one grid.

#include <metal_stdlib>
using namespace metal;

constant uint JOBS = 4;

struct Params {
    uint first[JOBS];
    uint width[JOBS];
    uint dim[JOBS];
    float threshold[JOBS];
};

static device float* buf_x(device float* x0, device float* x1, device float* x2, device float* x3, uint j) {
    switch (j) {
        case 0: return x0;
        case 1: return x1;
        case 2: return x2;
        default: return x3;
    }
}

static const device float* buf_scale(
    const device float* s0, const device float* s1, const device float* s2, const device float* s3, uint j)
{
    switch (j) {
        case 0: return s0;
        case 1: return s1;
        case 2: return s2;
        default: return s3;
    }
}

kernel void clip(
    device float* x0 [[buffer(0)]],
    device float* x1 [[buffer(1)]],
    device float* x2 [[buffer(2)]],
    device float* x3 [[buffer(3)]],
    const device float* s0 [[buffer(4)]],
    const device float* s1 [[buffer(5)]],
    const device float* s2 [[buffer(6)]],
    const device float* s3 [[buffer(7)]],
    constant Params& p [[buffer(8)]],
    uint wg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint j = 0;
    #pragma clang loop unroll(full)
    for (uint i = 1; i < JOBS; i++) {
        j += wg >= p.first[i];
    }
    uint block = wg - p.first[j];
    uint dim = p.dim[j];
    uint width = p.width[j];
    float threshold = p.threshold[j];
    device float* head = buf_x(x0, x1, x2, x3, j) + block * dim;
    const device float* by = buf_scale(s0, s1, s2, s3, j) + (block % (width / dim)) * dim;

    if (dim <= 128) {
        float v[4];
        float square = 0.0f;
        #pragma clang loop unroll(full)
        for (uint k = 0; k < 4; k++) {
            uint idx = lane + k * 32;
            v[k] = idx < dim ? head[idx] * by[idx] : 0.0f;
            square += v[k] * v[k];
        }
        square = simd_sum(square);
        float factor = min(threshold / (sqrt(square) + 1e-6f), 1.0f);
        #pragma clang loop unroll(full)
        for (uint k = 0; k < 4; k++) {
            uint idx = lane + k * 32;
            if (idx < dim) {
                head[idx] = v[k] * factor;
            }
        }
        return;
    }

    float square = 0.0f;
    for (uint idx = lane; idx < dim; idx += 32) {
        float v = head[idx] * by[idx];
        square += v * v;
    }
    square = simd_sum(square);
    float factor = min(threshold / (sqrt(square) + 1e-6f), 1.0f);
    for (uint idx = lane; idx < dim; idx += 32) {
        head[idx] = head[idx] * by[idx] * factor;
    }
}
