// out[t] = table[tokens[t]]: each thread converts one bf16 weight of the
// embedding table to an activation.

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint dim;
    uint n;
};

kernel void embed(
    device float* out [[buffer(0)]],
    const device ushort* table [[buffer(1)]],
    const device uint* tokens [[buffer(2)]],
    constant Params& p [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= p.n * p.dim) {
        return;
    }
    uint t = i / p.dim;
    uint j = i % p.dim;
    out[i] = as_type<float>(uint(table[tokens[t] * p.dim + j]) << 16);
}
