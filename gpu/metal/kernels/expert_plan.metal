// Lays out a batch's (token, expert) pairs on the GPU, avoiding a host
// readback and pipeline stall on every MoE layer: sorts pair indices by
// expert into `order`, emits wide (up to 32 pairs) and narrow (up to 16
// pairs) tiles into `wide_tiles` and `narrow_tiles`, and writes two pairs
// of MTLDispatchThreadgroupsIndirectArguments into `indirect` (for two
// row tile counts, e.g. gate_up and down). One threadgroup of 256 threads.

#include <metal_stdlib>
using namespace metal;

constant uint THREADS = 256;
constant uint MAX_EXPERTS = 256;
constant uint TILE_NARROW = 16;
constant uint TILE_WIDE = 32;

struct Params {
    uint pairs;
    uint experts;
    uint grid_x_0;
    uint grid_x_1;
};

kernel void expert_plan(
    device uint* order [[buffer(0)]],
    device uint* wide_tiles [[buffer(1)]],
    device uint* narrow_tiles [[buffer(2)]],
    device uint* indirect [[buffer(3)]],
    const device float* ids [[buffer(4)]],
    constant Params& p [[buffer(5)]],
    uint lid [[thread_index_in_threadgroup]])
{
    threadgroup atomic_uint counts[MAX_EXPERTS];
    threadgroup uint first[MAX_EXPERTS];
    threadgroup atomic_uint next[MAX_EXPERTS];
    threadgroup uint wide_off[MAX_EXPERTS];
    threadgroup uint narrow_off[MAX_EXPERTS];

    atomic_store_explicit(&counts[lid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = lid; i < p.pairs; i += THREADS) {
        uint e = uint(ids[i]);
        atomic_fetch_add_explicit(&counts[e], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid == 0) {
        uint sum = 0;
        uint w_sum = 0;
        uint n_sum = 0;
        for (uint e = 0; e < p.experts; e++) {
            uint c = atomic_load_explicit(&counts[e], memory_order_relaxed);
            first[e] = sum;
            atomic_store_explicit(&next[e], sum, memory_order_relaxed);
            sum += c;
            wide_off[e] = w_sum;
            narrow_off[e] = n_sum;
            uint full = c / TILE_WIDE;
            uint rem = c % TILE_WIDE;
            w_sum += full + (rem > TILE_NARROW ? 1u : 0u);
            n_sum += (rem > 0 && rem <= TILE_NARROW) ? 1u : 0u;
        }
        indirect[0] = p.grid_x_0;
        indirect[1] = w_sum;
        indirect[2] = 1u;
        indirect[3] = p.grid_x_0;
        indirect[4] = n_sum;
        indirect[5] = 1u;
        indirect[6] = p.grid_x_1;
        indirect[7] = w_sum;
        indirect[8] = 1u;
        indirect[9] = p.grid_x_1;
        indirect[10] = n_sum;
        indirect[11] = 1u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = lid; i < p.pairs; i += THREADS) {
        uint e = uint(ids[i]);
        uint slot = atomic_fetch_add_explicit(&next[e], 1u, memory_order_relaxed);
        order[slot] = i;
    }

    if (lid < p.experts) {
        uint c = atomic_load_explicit(&counts[lid], memory_order_relaxed);
        uint pos = first[lid];
        uint wi = wide_off[lid];
        while (c > TILE_NARROW) {
            uint of = min(c, TILE_WIDE);
            wide_tiles[3 * wi] = lid;
            wide_tiles[3 * wi + 1] = pos;
            wide_tiles[3 * wi + 2] = of;
            wi++;
            pos += of;
            c -= of;
        }
        if (c > 0) {
            uint ni = narrow_off[lid];
            narrow_tiles[3 * ni] = lid;
            narrow_tiles[3 * ni + 1] = pos;
            narrow_tiles[3 * ni + 2] = c;
        }
    }
}
