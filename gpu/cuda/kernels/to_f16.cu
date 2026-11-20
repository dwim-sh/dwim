// out[i] = x[i] as the bits of a half-precision float, for the tile
// matmuls: the nearest half, or the largest finite one for an activation
// past it. And back: the halves as single-precision floats. Both take the
// activations four at a time, and a buffer is whole fours long.

extern "C" __global__ void to_f16(uint2* out, const float4* x, LenParams p) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i * 4 < p.len) {
        out[i] = halves4(x[i]);
    }
}

extern "C" __global__ void from_f16(float4* out, const uint2* x, LenParams p) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i * 4 < p.len) {
        out[i] = floats4(x[i]);
    }
}
