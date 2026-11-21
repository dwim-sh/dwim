// dst[i] = src[i] for i < len, of words of BYTES bytes: a cache's quants
// are copied four bytes at a time, and its scales, half floats, one.

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint len;
};

#if BYTES == 4
typedef uint word;
#else
typedef ushort word;
#endif

kernel void copy_words(
    device word* dst [[buffer(0)]],
    const device word* src [[buffer(1)]],
    constant Params& p [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i < p.len) {
        dst[i] = src[i];
    }
}
