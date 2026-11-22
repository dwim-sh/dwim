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

// What a tile matmul does with its products: writes them as they are, or
// as halves, or adds them to what is there, or writes
// `silu(product) * other` as halves.
#define TILE_FLOATS 0
#define TILE_HALVES 1
#define TILE_ADD 2
#define TILE_SWIGLU 3

// A matmul over tiled weights: `tiles` steps of 64 columns to a row, and
// the tokens from `first` on, `count` of them; `mode` is what becomes of
// the products.
struct TileParams {
    u32 rows;
    u32 cols;
    u32 tiles;
    u32 first;
    u32 count;
    u32 mode;
};

// One of the matrix multiplications of a grid that has several: where its
// products go, its tiled weights and their scales, its tokens' halves, and
// its shape, with how many groups of rows it is.
struct TileJob {
    void* out;
    const unsigned char* w;
    const unsigned char* scales;
    const u16* x;
    u32 rows;
    u32 cols;
    u32 tiles;
    u32 groups;
};

// Most multiplications to a grid.
#define TILE_JOBS 8

// `jobs` multiplications of `count` tokens each.
struct TileJobs {
    TileJob job[TILE_JOBS];
    u32 jobs;
    u32 count;
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

// Attention over the caches in chunks of `chunk` positions, `chunks` of
// them to a token, for the tokens from `first` on, `count` of them.
struct AttentionTileParams {
    u32 n_heads;
    u32 n_kv_heads;
    u32 pos;
    u32 first;
    u32 count;
    u32 chunks;
    u32 chunk;
};

// Attention with a bias by distance and a sink: `window` is how far back
// a token attends, `ring` the rows the caches hold, `n_rel` the width of
// the bias's projection for a head and `extent` the distances it has rows
// for. The kernels on the tensor cores take chunks of `chunk` positions
// from position `base` on, for `count` tokens.
struct AttentionRelParams {
    u32 n_heads;
    u32 head_dim;
    u32 n_kv_heads;
    u32 pos;
    u32 first;
    u32 chunks;
    u32 chunk;
    u32 count;
    u32 base;
    u32 window;
    u32 ring;
    u32 n_rel;
    u32 extent;
};

struct RouteParams {
    u32 experts;
    u32 k;
    float scale;
};

struct ShortConvParams {
    u32 n;
    u32 channels;
    u32 taps;
    u32 heads;
    u32 add;
};

// A convolution whose heads, `dim` long, are kept in a cache of `rows`
// rows from position `pos` on, clipped or normalized as `kept` says with
// `value` for the threshold or the epsilon; `width` is how long what the
// clipped are scaled by is.
struct ConvStoreParams {
    u32 n;
    u32 channels;
    u32 taps;
    u32 heads;
    u32 dim;
    u32 width;
    u32 kept;
    u32 pos;
    u32 rows;
    float value;
};

struct ClipParams {
    u32 width;
    u32 dim;
    float threshold;
};

// Rows of `width` in `len` activations, and the experts to a token where
// the rows are theirs.
struct RowsParams {
    u32 len;
    u32 width;
    u32 k;
};

struct SituParams {
    u32 len;
    u32 width;
    float beta;
    float linear_beta;
};

// A matmul by the experts' matrices, each `rows` by `cols` in `groups`
// groups of rows of `tiles` tiles: `share` pairs of a token and an expert
// share a row of the input, and with `direct` every pair is a tile of its
// own.
struct ExpertParams {
    u32 rows;
    u32 cols;
    u32 tiles;
    u32 groups;
    u32 share;
    u32 direct;
    float beta;
    float linear_beta;
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

// The bits of the half nearest to `f`, or of the largest finite one for an
// activation past it, rather than an infinity.
__device__ __forceinline__ u32 f32_to_finite_f16(float f) {
    return f32_to_f16(fminf(fmaxf(f, -65504.0f), 65504.0f));
}

// Four activations as halves, two to a word.
__device__ __forceinline__ uint2 halves4(float4 v) {
    return make_uint2(
        f32_to_finite_f16(v.x) | f32_to_finite_f16(v.y) << 16,
        f32_to_finite_f16(v.z) | f32_to_finite_f16(v.w) << 16);
}

// The four activations of two words of halves.
__device__ __forceinline__ float4 floats4(uint2 h) {
    return make_float4(f16_to_f32(h.x), f16_to_f32(h.x >> 16), f16_to_f32(h.y), f16_to_f32(h.y >> 16));
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

// The two bf16 weights of a word, in memory order.
// The SiTU activation of an expert's gate and up projections.
__device__ __forceinline__ float situ(float g, float u, float beta, float linear_beta) {
    return beta * tanhf(g / beta) * sigmoid(g) * linear_beta * tanhf(u / linear_beta);
}

__device__ __forceinline__ float bf16_lo(u32 pair) {
    return __uint_as_float(pair << 16);
}

__device__ __forceinline__ float bf16_hi(u32 pair) {
    return __uint_as_float(pair & 0xffff0000u);
}
