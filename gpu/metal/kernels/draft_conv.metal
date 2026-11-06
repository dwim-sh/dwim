// The draft's two-tap dynamic convolution: each thread takes one channel of
// one row, weighs it and the row before by the stage's base weights plus
// the row's dynamic weights for the channel's group, and adds the residual
// when finishing a sublayer's output. The row before the first is zero.

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint rows;
    uint hidden;
    uint group;
    uint stage;
    // Whether the residual is added.
    uint residual;
};

kernel void draft_conv(
    device float* out [[buffer(0)]],
    const device float* x [[buffer(1)]],
    const device float* dynamic [[buffer(2)]],
    const device float* base [[buffer(3)]],
    const device float* residual [[buffer(4)]],
    constant Params& p [[buffer(5)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= p.rows * p.hidden) {
        return;
    }
    uint t = i / p.hidden;
    uint c = i % p.hidden;
    uint groups = p.hidden / p.group;
    uint g = c / p.group;
    uint tap = 2 * p.stage;
    const device float* dyn = dynamic + t * 4 * groups;
    float value = x[i] * (base[tap * p.hidden + c] + dyn[tap * groups + g]);
    if (t > 0) {
        value += x[i - p.hidden] * (base[(tap + 1) * p.hidden + c] + dyn[(tap + 1) * groups + g]);
    }
    if (p.residual) {
        value += residual[i];
    }
    out[i] = value;
}
