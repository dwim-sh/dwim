// out[i] = the sum over the splits of a ternary matmul's columns of their
// partial results, a thread per output: the batch's tokens by rows.

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint rows;
    uint cols;
    uint n;
    uint splits;
};

kernel void matmul_ternary_reduce(
    device float* out [[buffer(0)]],
    const device float* partial [[buffer(1)]],
    constant Params& p [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    uint size = p.n * p.rows;
    if (i >= size) {
        return;
    }
    float total = 0.0f;
    for (uint s = 0; s < p.splits; s++) {
        total += partial[s * size + i];
    }
    out[i] = total;
}
