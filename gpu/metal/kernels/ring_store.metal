// Stores rows of keys or values into a ring of slots per head: each thread
// takes one element of one row, and puts it in the slot of the row's
// position of its head.

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint rows;
    uint pos;
    uint window;
    uint n_kv_heads;
    uint head_dim;
};

kernel void ring_store(
    device float* ring [[buffer(0)]],
    const device float* rows [[buffer(1)]],
    constant Params& p [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    uint kv_dim = p.n_kv_heads * p.head_dim;
    if (i >= p.rows * kv_dim) {
        return;
    }
    uint t = i / kv_dim;
    uint h = (i % kv_dim) / p.head_dim;
    uint d = i % p.head_dim;
    uint slot = (p.pos + t) % p.window;
    ring[(h * p.window + slot) * p.head_dim + d] = rows[i];
}
