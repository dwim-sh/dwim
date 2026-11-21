// The experts' matrix multiplication for a batch, as tiled matrix products
// on the GPU's matrix accelerators: `expert_matmul.metal` multiplies each
// pair of a token and one of its experts apart, which reads an expert's
// matrix once for every token that goes through it. Here the pairs are
// taken expert by expert, in tiles of up to TOKENS pairs of one expert
// that the host lays out, and a threadgroup takes 64 rows of the expert's
// matrix for a tile: it unpacks them once, 128 columns at a time, as
// `matmul_quant_mma.metal` does, and multiplies the tile's inputs into
// them. The inputs are half floats in the order of the tiles, as
// `expert_pack.metal` lays them out, and the results come out in the same
// order, for `expert_scatter.metal` to put back. The expert's scale goes
// into the weights as they are unpacked.
//
// The weights are NVFP4 or Q8_0, as BITS says, 4 or 8, laid out as
// `quant.rs` says.

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

struct Params {
    uint rows;
    uint cols;
    // The inputs' rows with the padding past the last tile.
    uint padded;
};

#ifndef BITS
#define BITS 4
#endif

// Rows and pairs of a tile, and the columns a step takes.
constant uint ROWS = 64;
#ifndef TOKENS
#define TOKENS 16
#endif
constant uint COLS = 128;
constant uint THREADS = 128;
// Half floats a row of the unpacked weights takes, padded so that the
// threads writing the same columns of different rows hit different banks.
constant uint STRIDE = 132;
// Threads that unpack a row, and the weights of a step each takes.
constant uint PER_ROW = THREADS / ROWS;
constant uint UNIT = COLS / PER_ROW;

#if BITS == 4

// What a weight's four bits stand for, doubled: the scale is halved.
constant half VALUES[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

static half4 low(uint word) {
    return half4(VALUES[word & 0xfu], VALUES[(word >> 8) & 0xfu], VALUES[(word >> 16) & 0xfu], VALUES[(word >> 24) & 0xfu]);
}

static half4 high(uint word) {
    return half4(VALUES[(word >> 4) & 0xfu], VALUES[(word >> 12) & 0xfu], VALUES[(word >> 20) & 0xfu], VALUES[word >> 28]);
}

// A block's scale from its bits, halved. Zero and the bits of a NaN scale
// the block to nothing.
static float scale_of(uint bits) {
    if (bits == 0 || bits == 0x7f) {
        return 0.0f;
    }
    int e = int((bits >> 3) & 0xf);
    float m = float(bits & 7);
    return e == 0 ? m * (0.5f / 512.0f) : ldexp(0.5f + m / 16.0f, e - 7);
}

#define SCALES uint

// A thread's quants and scales for a step: 64 weights are four blocks of
// 16, their quants two 16-byte loads and their scales a word.
struct Fetch {
    uint4 q[2];
    uint d;
};

// The thread's weights, of which `unit` is the first's number in 64s.
static Fetch fetch(const device uint4* w, const device uint* scales, uint unit) {
    Fetch f;
    f.q[0] = w[unit * 2];
    f.q[1] = w[unit * 2 + 1];
    f.d = scales[unit];
    return f;
}

// Unpacks the thread's weights into its part of its row, scaled.
static void unpack(thread const Fetch& f, float scale, threadgroup half* mine) {
    uint words[8] = { f.q[0].x, f.q[0].y, f.q[0].z, f.q[0].w, f.q[1].x, f.q[1].y, f.q[1].z, f.q[1].w };
    for (uint b = 0; b < 4; b++) {
        half d = half(scale_of((f.d >> (8 * b)) & 0xffu) * scale);
        threadgroup half* out = mine + 16 * b;
        *(threadgroup half4*)(out) = low(words[2 * b]) * d;
        *(threadgroup half4*)(out + 4) = low(words[2 * b + 1]) * d;
        *(threadgroup half4*)(out + 8) = high(words[2 * b]) * d;
        *(threadgroup half4*)(out + 12) = high(words[2 * b + 1]) * d;
    }
}

#else

#define SCALES half

// A thread's quants and scales for a step: 64 weights are two blocks of
// 32, their quants two 16-byte loads each.
struct Fetch {
    uint4 q[4];
    half d[2];
};

static Fetch fetch(const device uint4* w, const device half* scales, uint unit) {
    Fetch f;
    for (uint k = 0; k < 4; k++) {
        f.q[k] = w[unit * 4 + k];
    }
    f.d[0] = scales[unit * 2];
    f.d[1] = scales[unit * 2 + 1];
    return f;
}

static void unpack(thread const Fetch& f, float scale, threadgroup half* mine) {
    for (uint k = 0; k < 2; k++) {
        half d = half(float(f.d[k]) * scale);
        uint words[8] = { f.q[2 * k].x, f.q[2 * k].y, f.q[2 * k].z, f.q[2 * k].w,
                          f.q[2 * k + 1].x, f.q[2 * k + 1].y, f.q[2 * k + 1].z, f.q[2 * k + 1].w };
        for (uint v = 0; v < 8; v++) {
            *(threadgroup half4*)(mine + 32 * k + 4 * v) = half4(as_type<char4>(words[v])) * d;
        }
    }
}

#endif

kernel void expert_matmul_mma(
    device float* out [[buffer(0)]],
    const device uint4* w [[buffer(1)]],
    const device SCALES* scales [[buffer(2)]],
    device half* x [[buffer(3)]],
    const device uint* tiles [[buffer(4)]],
    const device float* scale [[buffer(5)]],
    constant Params& p [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]])
{
    threadgroup half wt[ROWS * STRIDE];

    // The tile: its expert, its first pair in the order of the tiles, and
    // how many pairs it has.
    uint expert = tiles[3 * group.y];
    uint t0 = tiles[3 * group.y + 1];
    uint count = tiles[3 * group.y + 2];
    float by = scale[expert];

    uint r0 = group.x * ROWS;
    uint steps = p.cols / COLS;
    uint units = p.cols / UNIT;
    // The row the thread unpacks, clamped into the expert's matrix, and
    // its share of the row. A row past the end gives results that are not
    // kept.
    uint row = lid / PER_ROW;
    uint share = lid % PER_ROW;
    uint wrow = expert * p.rows + min(r0 + row, p.rows - 1);
    uint first = wrow * units + share;
    threadgroup half* mine = wt + row * STRIDE + share * UNIT;

    // The inputs by pair, padded past the last tile; the weights by row;
    // and the results by pair, up to the tile's last, so that the pairs
    // of the tiles after it are cut off.
    auto xs = tensor<device half, dextents<int32_t, 2>, tensor_inline>(
        x, dextents<int32_t, 2>(p.cols, p.padded));
    auto ws = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        wt, dextents<int32_t, 2>(COLS, ROWS), array<int32_t, 2>{1, int32_t(STRIDE)});
    auto results = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
        out, dextents<int32_t, 2>(p.rows, t0 + count));
    constexpr auto desc = matmul2d_descriptor(
        TOKENS, ROWS, COLS, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<THREADS / 32>> op;
    using Xs = decltype(xs.slice<COLS, TOKENS>(0, 0));
    auto acc = op.get_destination_cooperative_tensor<Xs, decltype(ws), float>();
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < acc.get_capacity(); i++) {
        if (acc.is_valid_element(i)) {
            acc[i] = 0.0f;
        }
    }

    Fetch f = fetch(w, scales, first);
    for (uint s = 0; s < steps; s++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        unpack(f, by, mine);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (s + 1 < steps) {
            f = fetch(w, scales, first + (s + 1) * PER_ROW);
        }
        auto xb = xs.slice<COLS, TOKENS>(s * COLS, t0);
        op.run(xb, ws, acc);
    }
    auto tile = results.slice(r0, t0);
    acc.store(tile);
}
