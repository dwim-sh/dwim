// Unpacking a block of ternary weights a word at a time, for
// `matmul_ternary_tile.metal` and `matmul_ternary_mma.metal`, which the
// device puts ahead of their source. A block is 28 bytes, seven words, as
// `ternary.rs` lays it out: 24 bytes of five trits each, 2 of four, and the
// scale as a half float in the last word's high half.

#include <metal_stdlib>
using namespace metal;

// Words per block.
constant uint WORDS = 7;

// The four bytes of a word as two pairs in 16-bit halves, so that tripling
// them does not carry between bytes.
struct Split {
    uint even;
    uint odd;
};

static Split split(uint word) {
    return Split { word & 0x00ff00ffu, (word >> 8) & 0x00ff00ffu };
}

// Peels the most significant trit off each of the four bytes, as -1, 0, or
// 1, in the bytes' order.
static half4 peel(thread Split& s) {
    uint even = s.even * 3;
    uint odd = s.odd * 3;
    s.even = even & 0x00ff00ffu;
    s.odd = odd & 0x00ff00ffu;
    return half4(half((even >> 8) & 0xff), half((odd >> 8) & 0xff), half(even >> 24), half(odd >> 24)) - 1.0h;
}
