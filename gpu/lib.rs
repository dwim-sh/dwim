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

/// A bf16 tensor on the host: its shape, and its values as raw bf16 bits.
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<u16>,
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
    /// A bf16 weight tensor in device memory.
    type Weight;
    /// A key or value cache in device memory: activations stored as IEEE
    /// half-precision floats, for half the memory attention reads.
    type Cache;

    /// Copies a weight tensor into device memory.
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

    /// `out[t] = table[tokens[t]]`: looks up each token's row of an embedding
    /// table.
    fn embed(&self, out: &mut Self::Buffer, table: &Self::Weight, tokens: &[u32]);

    /// `out[t] = w · x[t]` for each row `x[t]` of `x`, for a weight matrix
    /// `w` of shape `[rows, cols]`: `x` holds `cols` activations per token,
    /// and `out` `rows` per token.
    fn matmul(&self, out: &mut Self::Buffer, w: &Self::Weight, x: &Self::Buffer);

    /// `x += y`
    fn add(&self, x: &mut Self::Buffer, y: &Self::Buffer);

    /// Normalizes each `weight.len()`-long row of `x` by its root mean square,
    /// then scales it by `weight`.
    fn rmsnorm(&self, x: &mut Self::Buffer, weight: &Self::Weight, eps: f32);

    /// Rotates each `head_dim`-long head of `x`, which holds `n_heads` heads
    /// per token, to encode the token's position (rotary position
    /// embeddings): `pos` for the first token, `pos + 1` for the next, and so
    /// on. Element `i` of a head at position `p` pairs with element
    /// `i + head_dim / 2`, rotated by the angle whose cosine and sine are at
    /// `table[(p * head_dim / 2 + i) * 2..][..2]`.
    fn rope(&self, x: &mut Self::Buffer, table: &Self::Buffer, pos: usize, n_heads: usize, head_dim: usize);

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
