// out[i] = x[i] as the bits of a half-precision float, for the tile
// matmuls: the nearest half, or the largest finite one for an activation
// past it, rather than an infinity.

extern "C" __global__ void to_f16(u16* out, const float4* x, LenParams p) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i * 4 >= p.len) {
        return;
    }
    float4 v = x[i];
    u32 lo = f32_to_f16(fminf(fmaxf(v.x, -65504.0f), 65504.0f))
        | f32_to_f16(fminf(fmaxf(v.y, -65504.0f), 65504.0f)) << 16;
    u32 hi = f32_to_f16(fminf(fmaxf(v.z, -65504.0f), 65504.0f))
        | f32_to_f16(fminf(fmaxf(v.w, -65504.0f), 65504.0f)) << 16;
    *(uint2*)(out + i * 4) = make_uint2(lo, hi);
}
