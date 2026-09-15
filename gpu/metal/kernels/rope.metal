// Rotary position embeddings: each thread rotates one pair of elements,
// half a head apart, by the angle for the token's position.

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint n_heads;
    uint head_dim;
    uint pos;
    uint n;
};

kernel void rope(
    device float* x [[buffer(0)]],
    const device float2* table [[buffer(1)]],
    constant Params& p [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    uint pairs = p.head_dim / 2;
    if (id >= p.n * p.n_heads * pairs) {
        return;
    }
    uint i = id % pairs;
    // Heads are numbered across tokens: token t's first is t * n_heads.
    uint head = id / pairs;
    uint t = head / p.n_heads;
    device float* v = x + head * p.head_dim;
    float2 angle = table[(p.pos + t) * pairs + i];
    float a = v[i];
    float b = v[i + pairs];
    v[i] = a * angle.x - b * angle.y;
    v[i + pairs] = b * angle.x + a * angle.y;
}
