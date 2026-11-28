// Fused expert_scatter + weighted_sum for batched MoE: gathers each token's
// k expert outputs directly from the expert-sorted `sorted` buffer via
// `inv_order` and sums them weighted by `shares` into `out`, four elements
// per thread. Eliminates the intermediate [n * k, rows] scatter buffer.

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint n;
    uint rows;
    uint k;
};

kernel void expert_gather_sum(
    device float4* out [[buffer(0)]],
    const device float4* sorted [[buffer(1)]],
    const device uint* inv_order [[buffer(2)]],
    const device float* shares [[buffer(3)]],
    constant Params& p [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
    uint rq = p.rows / 4;
    if (i >= p.n * rq) {
        return;
    }
    uint t = i / rq;
    uint j = i % rq;
    float4 acc = float4(0.0f);
    for (uint e = 0; e < p.k; e++) {
        uint pair = t * p.k + e;
        uint slot = inv_order[pair];
        acc += shares[pair] * sorted[slot * rq + j];
    }
    out[i] = acc;
}
