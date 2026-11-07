// What every kernel shares: the parameter structs, which match the Rust
// ones in `mod.rs` field for field; half-precision conversions by PTX, so
// that the kernels need no header, which NVRTC may not have; and sums
// across a warp. The kernels are compiled together as one program when
// the device opens, this first.

typedef unsigned int u32;
typedef unsigned short u16;
typedef int i32;

#define FULL 0xffffffffu

// Activations to a block of a key or value cache, which share a scale.
#define CACHE_BLOCK 32

struct MatmulParams {
    u32 rows;
    u32 cols;
    u32 n;
    // Blocks per row of the grid, for matrices with more rows than one
    // dimension of the grid allows. The grid's second dimension counts the
    // groups of tokens of each such row.
    u32 stride;
    // Column splits, which no kernel takes: the struct is the Vulkan
    // kernels' field for field.
    u32 splits;
};

struct RmsnormParams {
    u32 dim;
    float eps;
};

struct RopeParams {
    u32 n_heads;
    u32 head_dim;
    u32 rot_dim;
    u32 pos;
    u32 n;
};

struct ConvParams {
    u32 n;
    u32 channels;
    u32 q_dim;
    u32 k_dim;
};

struct DeltaNetParams {
    u32 n;
    u32 n_k_heads;
    u32 n_v_heads;
    u32 head_dim;
};

struct AttentionParams {
    u32 n_heads;
    u32 head_dim;
    u32 n_kv_heads;
    u32 pos;
    u32 first;
    u32 chunks;
};

struct StoreParams {
    u32 offset;
    u32 len;
};

struct LenParams {
    u32 len;
};

// The value of the bits of a half-precision float.
__device__ __forceinline__ float f16_to_f32(u32 bits) {
    u16 h = (u16)bits;
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}

// The bits of the half-precision float nearest to `f`, ties to even.
__device__ __forceinline__ u32 f32_to_f16(float f) {
    u16 h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
    return h;
}

// The sum, and the largest, over the warp's lanes, which every lane gets.
// Every lane of the warp must call it.
__device__ __forceinline__ float warp_sum(float v) {
    for (int s = 16; s > 0; s >>= 1) {
        v += __shfl_xor_sync(FULL, v, s);
    }
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
    for (int s = 16; s > 0; s >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(FULL, v, s));
    }
    return v;
}

__device__ __forceinline__ float dot4(float4 a, float4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__device__ __forceinline__ float4 add4(float4 a, float4 b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__device__ __forceinline__ float4 sub4(float4 a, float4 b) {
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

__device__ __forceinline__ float4 mul4(float4 a, float4 b) {
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

__device__ __forceinline__ float4 scale4(float4 a, float s) {
    return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}

__device__ __forceinline__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}
