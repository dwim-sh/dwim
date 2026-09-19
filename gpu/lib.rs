//! Devices a model runs on: the operations a transformer's forward pass is
//! built from, and their implementations. [`Cpu`] is the reference; the
//! `metal` and [`vulkan`] devices run the same operations on a GPU with
//! hand-written compute kernels.

#[cfg(test)]
#[macro_use]
mod tests;

pub mod cpu;
#[cfg(target_vendor = "apple")]
pub mod metal;
pub mod ternary;
pub mod vulkan;

pub use cpu::Cpu;
#[cfg(target_vendor = "apple")]
pub use metal::Metal;
pub use vulkan::Vulkan;

/// The GPU device of the platform: Metal on Apple's, Vulkan elsewhere.
#[cfg(target_vendor = "apple")]
pub type Gpu = Metal;
/// The GPU device of the platform: Metal on Apple's, Vulkan elsewhere.
#[cfg(not(target_vendor = "apple"))]
pub type Gpu = Vulkan;

/// Elements the Hadamard rotation transforms at a time.
pub const HADAMARD_BLOCK: usize = 1024;

/// Taps of the causal convolution in linear attention.
pub const CONV_KERNEL: usize = 4;

/// A weight matrix on the host, of shape `[rows, cols]`: either raw bf16
/// bits, or rows of ternary blocks.
pub enum Tensor {
    Bf16 { shape: Vec<usize>, data: Vec<u16> },
    Ternary { shape: Vec<usize>, data: Vec<u8> },
}

impl Tensor {
    pub fn shape(&self) -> &[usize] {
        match self {
            Tensor::Bf16 { shape, .. } | Tensor::Ternary { shape, .. } => shape,
        }
    }
}

/// The operations a transformer's forward pass is built from.
///
/// A device owns the memory the model computes in: weights are uploaded to it
/// once, activations live in buffers allocated on it, and every operation runs
/// on it. Operations may run lazily; reading a buffer back waits for them.
pub trait Device {
    /// A vector of f32 activations in device memory. Operations that work
    /// on rows of activations, one per token, take a buffer's length as a
    /// whole number of rows.
    type Buffer;
    /// A weight matrix in device memory, bf16 or ternary.
    type Weight;
    /// A key or value cache in device memory: activations stored as IEEE
    /// half-precision floats, for half the memory attention reads.
    type Cache;

    /// Copies a weight matrix into device memory.
    fn upload(&self, tensor: Tensor) -> Self::Weight;

    /// Allocates a zeroed buffer of `len` activations.
    fn alloc(&self, len: usize) -> Self::Buffer;

    /// Allocates a zeroed cache of `len` activations.
    fn alloc_cache(&self, len: usize) -> Self::Cache;

    /// Sets a buffer's length, up to the length it was allocated with, so
    /// that it holds a batch of some other number of tokens. The
    /// activations it gains are undefined.
    fn resize(&self, buf: &mut Self::Buffer, len: usize);

    /// Copies a buffer out of device memory.
    fn read(&self, buf: &Self::Buffer) -> Vec<f32>;

    /// Copies `data` into a buffer of the same length.
    fn write(&self, buf: &mut Self::Buffer, data: &[f32]);

    /// `dst[dst_offset..][..len] = src[src_offset..][..len]`
    fn copy(&self, dst: &mut Self::Buffer, dst_offset: usize, src: &Self::Buffer, src_offset: usize, len: usize);

    /// `cache[offset..][..src.len()] = f16(src)`: stores activations in a
    /// cache, rounded to the nearest half-precision float, and clamped to the
    /// largest finite one.
    fn store(&self, cache: &mut Self::Cache, offset: usize, src: &Self::Buffer);

    /// The first `len` activations of a cache, as the bits of the half-
    /// precision floats they are stored as.
    fn read_cache(&self, cache: &Self::Cache, len: usize) -> Vec<u16>;

    /// `cache[..data.len()] = data`: bits as [`read_cache`](Self::read_cache)
    /// gives them.
    fn write_cache(&self, cache: &mut Self::Cache, data: &[u16]);

    /// `out[t] = w · x[t]` for each row `x[t]` of `x`, for a weight matrix
    /// `w` of shape `[rows, cols]`: `x` holds `cols` activations per token,
    /// and `out` `rows` per token.
    fn matmul(&self, out: &mut Self::Buffer, w: &Self::Weight, x: &Self::Buffer);

    /// `x += y`
    fn add(&self, x: &mut Self::Buffer, y: &Self::Buffer);

    /// Normalizes each `weight.len()`-long row of `x` by its root mean square,
    /// then scales it by `weight`.
    fn rmsnorm(&self, x: &mut Self::Buffer, weight: &Self::Buffer, eps: f32);

    /// Normalizes each `dim`-long row of `x` to unit length.
    fn l2norm(&self, x: &mut Self::Buffer, dim: usize, eps: f32);

    /// Rotates the first `rot_dim` elements of each `head_dim`-long head of
    /// `x`, which holds `n_heads` heads per token, to encode the token's
    /// position (rotary position embeddings): `pos` for the first token,
    /// `pos + 1` for the next, and so on. Element `i` of a head at position
    /// `p` pairs with element `i + rot_dim / 2`, rotated by the angle whose
    /// cosine and sine are at `table[(p * rot_dim / 2 + i) * 2..][..2]`.
    #[allow(clippy::too_many_arguments)]
    fn rope(&self, x: &mut Self::Buffer, table: &Self::Buffer, pos: usize, n_heads: usize, head_dim: usize, rot_dim: usize);

    /// Causal self-attention for the tokens in `q`, the first at position
    /// `pos`: each of a token's `n_heads` heads, each `head_dim` long,
    /// attends over the key and value caches up to and including the token's
    /// own position, and writes its result to `out`. The caches hold
    /// `n_kv_heads` heads per position, and must already hold the tokens'
    /// own keys and values.
    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        out: &mut Self::Buffer,
        q: &Self::Buffer,
        k_cache: &Self::Cache,
        v_cache: &Self::Cache,
        pos: usize,
        n_heads: usize,
        head_dim: usize,
        n_kv_heads: usize,
    );

    /// `gate = silu(gate) * up`: the SwiGLU activation.
    fn silu_mul(&self, gate: &mut Self::Buffer, up: &Self::Buffer);

    /// `x *= sigmoid(gate)`
    fn sigmoid_mul(&self, x: &mut Self::Buffer, gate: &Self::Buffer);

    /// Rotates each row of `x`, as wide as `signs`, into the basis the
    /// weights are stored in: every [`HADAMARD_BLOCK`] of the row is
    /// multiplied by the signs, then by the normalized Walsh-Hadamard matrix
    /// of that order. The inverse multiplies by the matrix first and the
    /// signs after, which undoes the rotation.
    fn hadamard(&self, x: &mut Self::Buffer, signs: &Self::Buffer, inverse: bool);

    /// `out = rotate(rmsnorm(x) * weight)`: normalizes each row of `x`, as
    /// wide as `weight` and `signs`, by its root mean square, scales it by
    /// `weight`, and rotates it as [`hadamard`](Self::hadamard) does, into
    /// `out`. What every matrix multiplication's input goes through, in one
    /// pass.
    fn norm_rotate(&self, out: &mut Self::Buffer, x: &Self::Buffer, weight: &Self::Buffer, signs: &Self::Buffer, eps: f32);

    /// The causal convolution of linear attention: each channel of `x`,
    /// which holds a row of channels per token, is convolved over the last
    /// [`CONV_KERNEL`] tokens with its taps from `weight`, which holds that
    /// many per channel, and passed through SiLU. `state` holds the rows of
    /// the `CONV_KERNEL - 1` tokens before the batch, and `state_out` gets
    /// the rows of the last `CONV_KERNEL - 1` tokens of it. The channels of
    /// the result are split by token into `q`, `k`, and `v`, in that order,
    /// as wide as each is per token.
    #[allow(clippy::too_many_arguments)]
    fn conv(
        &self,
        q: &mut Self::Buffer,
        k: &mut Self::Buffer,
        v: &mut Self::Buffer,
        state_out: &mut Self::Buffer,
        x: &Self::Buffer,
        state: &Self::Buffer,
        weight: &Self::Buffer,
    );

    /// The gated delta rule of linear attention, one token after another.
    /// Each of the `n_v_heads` heads keeps a `head_dim` by `head_dim` state
    /// in `state`, which decays by `exp(a * softplus(alpha + dt_bias))` per
    /// token, forgets what it holds for the token's key in proportion to
    /// `sigmoid(beta)`, and stores the token's value in its place; the
    /// head's output is what the state holds for the token's query, scaled
    /// by the inverse square root of `head_dim`. `q` and `k` hold
    /// `n_k_heads` heads per token, each shared by the value heads in turn;
    /// `gates` holds `alpha` for every value head then `beta` for every one
    /// per token; and `decay` holds `a` for every value head then `dt_bias`
    /// for every one.
    #[allow(clippy::too_many_arguments)]
    fn delta_net(
        &self,
        out: &mut Self::Buffer,
        q: &Self::Buffer,
        k: &Self::Buffer,
        v: &Self::Buffer,
        gates: &Self::Buffer,
        decay: &Self::Buffer,
        state: &mut Self::Buffer,
        n_k_heads: usize,
        n_v_heads: usize,
        head_dim: usize,
    );
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// `ln(1 + e^x)`, without overflowing for large `x`.
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

/// Converts bf16 bits to f32: bf16 is the top half of an f32.
pub fn bf16(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// The largest finite half-precision float.
pub const F16_MAX: f32 = 65504.0;

/// Converts an f32 to the bits of the nearest IEEE half-precision float,
/// ties to even, clamped to the largest finite one.
pub fn to_f16(x: f32) -> u16 {
    let sign = ((x.to_bits() >> 16) & 0x8000) as u16;
    let a = x.abs().min(F16_MAX);
    if a < f32::from_bits((127 - 14) << 23) {
        // Subnormal: multiples of 2^-24, which may round up to the smallest
        // normal, whose bits follow on from the largest subnormal's.
        return sign | (a * (1u32 << 24) as f32).round_ties_even() as u16;
    }
    let e = ((a.to_bits() >> 23) & 0xff) as i32 - 127;
    // The mantissa scaled to [1024, 2048), which rounding may carry to 2048.
    let m = (a * f32::from_bits(((127 + 10 - e) as u32) << 23)).round_ties_even() as u32;
    let (e, m) = if m == 2048 { (e + 1, 1024) } else { (e, m) };
    sign | (((e + 15) as u16) << 10) | (m - 1024) as u16
}

/// Converts the bits of a finite IEEE half-precision float to f32.
pub fn from_f16(bits: u16) -> f32 {
    let (bits, sign) = (bits as u32, bits as u32 & 0x8000);
    let exp = (bits >> 10) & 0x1f;
    // A normal float has an implicit leading one; a subnormal float scales
    // like the smallest normal one.
    let mantissa = (bits & 0x3ff) | (((exp != 0) as u32) << 10);
    let scale = f32::from_bits((exp.max(1) + 127 - 25) << 23);
    let magnitude = mantissa as f32 * scale;
    f32::from_bits(magnitude.to_bits() | (sign << 16))
}
