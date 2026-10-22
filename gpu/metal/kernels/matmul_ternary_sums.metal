// out[t][r] = w[r] · x[t] for ternary weights and a token or a few, by
// looking each
// byte of a block up in a table of what its trits' elements of the token
// sum to. A byte holds five trits of five fixed elements of its block, so
// for each of a block's 26 byte positions there are 256 sums the byte can
// stand for, and every row shares them: a threadgroup builds the 26 tables
// of a block in threadgroup memory, then each of its rows costs one table
// read and one add per byte, with no unpacking and no multiplies. The
// tables are built a byte position to a SIMD group, its lanes taking eight
// byte values each, whose trits they unpack once and keep, so that a
// position's five activations are read once for all its sums. A
// threadgroup takes THREADS * PER rows, PER per thread, over a range of
// blocks; the ranges of a matrix with too few rows to fill the GPU are
// split among threadgroups, whose partial sums `matmul_ternary_reduce.metal`
// adds up. The words of a row's block are loaded before the tables are
// built, so the loads are in flight through the building. The block layout
// is that of `ternary.rs`, and the blocks are stored block-major, a block of
// every row before the next block, so that a SIMD group's rows read
// adjacent blocks. Each token is a row of the grid of threadgroups, which
// reads the weights for itself.

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint rows;
    uint cols;
    uint n;
    // How many ranges the blocks are split into.
    uint splits;
};

// Threads, rows per thread, and so rows per threadgroup.
#ifndef THREADS
#define THREADS 256
#endif
#ifndef PER
#define PER 4
#endif
constant uint ROWS = THREADS * PER;
// SIMD groups per threadgroup, of 32 lanes.
constant uint NSG = THREADS / 32;
// Words per block: 28 bytes.
constant uint WORDS = 7;
// Bytes of trits per block.
constant uint BYTES = 26;

// The element the `n`th trit of byte `m` of a block belongs to.
static uint element(uint m, uint n) {
    if (m < 16) {
        return m + 16 * n;
    } else if (m < 24) {
        return 80 + (m - 16) + 8 * n;
    }
    return 120 + (m - 24) + 2 * n;
}

kernel void matmul_ternary_sums(
    device float* out [[buffer(0)]],
    const device uint* w [[buffer(1)]],
    const device float* x [[buffer(2)]],
    device float* partial [[buffer(3)]],
    constant Params& p [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint sid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    // For each byte position, the sum each byte value stands for.
    threadgroup float table[BYTES * 256];

    uint blocks = p.cols / 128;
    uint tiles = (p.rows + ROWS - 1) / ROWS;
    uint tile = group.x % tiles;
    uint split = group.x / tiles;
    uint t = group.y;
    x += t * p.cols;
    out += t * p.rows;
    partial += (split * p.n + t) * p.rows;
    uint per = (blocks + p.splits - 1) / p.splits;
    uint first = split * per;
    uint last = min(blocks, first + per);
    uint r0 = tile * ROWS + lid;

    // The trits of the byte values the lane builds the tables of, most
    // significant first, which come out of tripling them.
    float trits[8][5];
    for (uint k = 0; k < 8; k++) {
        uint q = lane + 32 * k;
        for (uint n = 0; n < 5; n++) {
            trits[k][n] = float((q * 3) >> 8) - 1.0f;
            q = (q * 3) & 0xff;
        }
    }

    float acc[PER] = {};
    for (uint b = first; b < last; b++) {
        // The rows' words, each row's clamped into the matrix so that every
        // load is in bounds; a row past the end sums to nothing that is
        // kept.
        uint words[PER][WORDS];
        for (uint i = 0; i < PER; i++) {
            uint row = min(r0 + i * THREADS, p.rows - 1);
            const device uint* base = w + (b * p.rows + row) * WORDS;
            for (uint k = 0; k < WORDS; k++) {
                words[i][k] = base[k];
            }
        }
        // The last block's readers must be done before the tables change.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Each SIMD group builds the tables of every NSG-th byte position,
        // each lane eight byte values of each, from the position's five
        // activations, which every lane reads alike; the last two positions
        // hold four trits.
        const device float* xb = x + b * 128;
        for (uint m = sid; m < BYTES; m += NSG) {
            float e[5];
            for (uint n = 0; n < 4; n++) {
                e[n] = xb[element(m, n)];
            }
            e[4] = m < 24 ? xb[element(m, 4)] : 0.0f;
            for (uint k = 0; k < 8; k++) {
                float s = 0.0f;
                for (uint n = 0; n < 5; n++) {
                    s = fma(trits[k][n], e[n], s);
                }
                table[m * 256 + lane + 32 * k] = s;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = 0; i < PER; i++) {
            float sum = 0.0f;
            for (uint k = 0; k < 6; k++) {
                uint word = words[i][k];
                sum += table[(4 * k) * 256 + (word & 0xff)];
                sum += table[(4 * k + 1) * 256 + ((word >> 8) & 0xff)];
                sum += table[(4 * k + 2) * 256 + ((word >> 16) & 0xff)];
                sum += table[(4 * k + 3) * 256 + (word >> 24)];
            }
            uint word = words[i][6];
            sum += table[24 * 256 + (word & 0xff)];
            sum += table[25 * 256 + ((word >> 8) & 0xff)];
            float scale = float(as_type<half>(ushort(word >> 16)));
            acc[i] = fma(scale, sum, acc[i]);
        }
    }
    for (uint i = 0; i < PER; i++) {
        uint row = r0 + i * THREADS;
        if (row < p.rows) {
            if (p.splits == 1) {
                out[row] = acc[i];
            } else {
                partial[row] = acc[i];
            }
        }
    }
}
