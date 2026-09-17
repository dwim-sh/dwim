//! The model on the CPU: the reference implementation of every operation,
//! which the GPU kernels are checked against.

use rayon::prelude::*;

use crate::{Device, Tensor, bf16, from_f16, sigmoid, to_f16};

pub struct Cpu;

impl Device for Cpu {
    type Buffer = Vec<f32>;
    type Weight = Tensor;
    type Cache = Vec<u16>;

    fn upload(&self, tensor: Tensor) -> Tensor {
        tensor
    }

    fn alloc(&self, len: usize) -> Vec<f32> {
        vec![0.0; len]
    }

    fn alloc_cache(&self, len: usize) -> Vec<u16> {
        vec![0; len]
    }

    fn resize(&self, buf: &mut Vec<f32>, len: usize) {
        assert!(len <= buf.capacity(), "a buffer of {} activations can't hold {len}", buf.capacity());
        buf.resize(len, 0.0);
    }

    fn read(&self, buf: &Vec<f32>) -> Vec<f32> {
        buf.clone()
    }

    fn write(&self, buf: &mut Vec<f32>, data: &[f32]) {
        buf.copy_from_slice(data);
    }

    fn copy(&self, dst: &mut Vec<f32>, dst_offset: usize, src: &Vec<f32>, src_offset: usize, len: usize) {
        dst[dst_offset..][..len].copy_from_slice(&src[src_offset..][..len]);
    }

    fn store(&self, cache: &mut Vec<u16>, offset: usize, src: &Vec<f32>) {
        for (c, &x) in cache[offset..][..src.len()].iter_mut().zip(src) {
            *c = to_f16(x);
        }
    }

    fn embed(&self, out: &mut Vec<f32>, table: &Tensor, tokens: &[u32]) {
        let dim = table.shape[1];
        assert_eq!(out.len(), tokens.len() * dim);
        for (out, &token) in out.chunks_exact_mut(dim).zip(tokens) {
            let row = &table.data[token as usize * dim..][..dim];
            for (o, &w) in out.iter_mut().zip(row) {
                *o = bf16(w);
            }
        }
    }

    fn matmul(&self, out: &mut Vec<f32>, w: &Tensor, x: &Vec<f32>) {
        let (rows, cols) = (w.shape[0], w.shape[1]);
        let n = x.len() / cols;
        assert_eq!(x.len(), n * cols);
        assert_eq!(out.len(), n * rows);
        if n == 1 {
            out.par_iter_mut()
                .zip(w.data.par_chunks_exact(cols))
                .for_each(|(o, row)| *o = dot(row, x));
            return;
        }
        // Each row of the weights is read from memory once, for every
        // token: the results come out by row, and are transposed into `out`,
        // which holds them by token.
        let mut by_row = vec![0.0; rows * n];
        by_row
            .par_chunks_exact_mut(n)
            .zip(w.data.par_chunks_exact(cols))
            .for_each(|(results, row)| {
                for (result, x) in results.iter_mut().zip(x.chunks_exact(cols)) {
                    *result = dot(row, x);
                }
            });
        for (r, results) in by_row.chunks_exact(n).enumerate() {
            for (t, &result) in results.iter().enumerate() {
                out[t * rows + r] = result;
            }
        }
    }

    fn add(&self, x: &mut Vec<f32>, y: &Vec<f32>) {
        for (a, b) in x.iter_mut().zip(y) {
            *a += b;
        }
    }

    fn rmsnorm(&self, x: &mut Vec<f32>, weight: &Tensor, eps: f32) {
        let dim = weight.data.len();
        for row in x.chunks_exact_mut(dim) {
            let mean_square = row.iter().map(|v| v * v).sum::<f32>() / dim as f32;
            let scale = 1.0 / (mean_square + eps).sqrt();
            for (v, &w) in row.iter_mut().zip(&weight.data) {
                *v *= scale * bf16(w);
            }
        }
    }

    fn rope(&self, x: &mut Vec<f32>, table: &Vec<f32>, pos: usize, n_heads: usize, head_dim: usize) {
        // Each head is rotated as pairs of elements half a head apart, each
        // pair by its own angle.
        let half = head_dim / 2;
        for (t, token) in x.chunks_exact_mut(n_heads * head_dim).enumerate() {
            let angles = &table[(pos + t) * head_dim..][..head_dim];
            for head in token.chunks_exact_mut(head_dim) {
                for (i, angle) in angles.chunks_exact(2).enumerate() {
                    let (cos, sin) = (angle[0], angle[1]);
                    let (a, b) = (head[i], head[i + half]);
                    head[i] = a * cos - b * sin;
                    head[i + half] = b * cos + a * sin;
                }
            }
        }
    }

    fn attention(
        &self,
        out: &mut Vec<f32>,
        q: &Vec<f32>,
        k_cache: &Vec<u16>,
        v_cache: &Vec<u16>,
        pos: usize,
        n_heads: usize,
        head_dim: usize,
        n_kv_heads: usize,
    ) {
        // With grouped-query attention, consecutive query heads share a key
        // and value head.
        let group = n_heads / n_kv_heads;
        let kv_dim = n_kv_heads * head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();
        out.par_chunks_exact_mut(head_dim)
            .zip(q.par_chunks_exact(head_dim))
            .enumerate()
            .for_each(|(i, (out, q))| {
                let (token, head) = (i / n_heads, i % n_heads);
                // The token attends up to and including its own position.
                let len = pos + token + 1;
                let kv = head / group * head_dim;
                let mut scores: Vec<f32> = (0..len)
                    .map(|t| scale * dot_f16(q, &k_cache[t * kv_dim + kv..][..head_dim]))
                    .collect();
                softmax(&mut scores);
                out.fill(0.0);
                for (t, &score) in scores.iter().enumerate() {
                    let v = &v_cache[t * kv_dim + kv..][..head_dim];
                    add_scaled_f16(out, score, v);
                }
            });
    }

    fn silu_mul(&self, gate: &mut Vec<f32>, up: &Vec<f32>) {
        for (g, &u) in gate.iter_mut().zip(up) {
            *g = *g * sigmoid(*g) * u;
        }
    }
}

/// Dot product of a row of bf16 weights with f32 activations.
///
/// Accumulating into eight independent sums lets the compiler vectorize the
/// loop, which a single running sum would not allow.
pub fn dot(w: &[u16], x: &[f32]) -> f32 {
    let mut acc = [0.0f32; 8];
    for (w, x) in w.chunks_exact(8).zip(x.chunks_exact(8)) {
        for i in 0..8 {
            acc[i] += bf16(w[i]) * x[i];
        }
    }
    let done = w.len() / 8 * 8;
    let rest: f32 = w[done..].iter().zip(&x[done..]).map(|(&w, x)| bf16(w) * x).sum();
    acc.iter().sum::<f32>() + rest
}

/// Dot product of f32 activations with f16 ones, in eight independent sums
/// like [`dot`].
fn dot_f16(a: &[f32], b: &[u16]) -> f32 {
    let mut acc = [0.0f32; 8];
    for (a, b) in a.chunks_exact(8).zip(b.chunks_exact(8)) {
        let b = from_f16_8(b.try_into().unwrap());
        for i in 0..8 {
            acc[i] += a[i] * b[i];
        }
    }
    let done = a.len() / 8 * 8;
    let rest: f32 = a[done..].iter().zip(&b[done..]).map(|(a, &b)| a * from_f16(b)).sum();
    acc.iter().sum::<f32>() + rest
}

/// `out += scale * x` for f16 activations `x`.
fn add_scaled_f16(out: &mut [f32], scale: f32, x: &[u16]) {
    for (out, x) in out.chunks_exact_mut(8).zip(x.chunks_exact(8)) {
        let x = from_f16_8(x.try_into().unwrap());
        for i in 0..8 {
            out[i] += scale * x[i];
        }
    }
    let done = out.len() / 8 * 8;
    for (o, &x) in out[done..].iter_mut().zip(&x[done..]) {
        *o += scale * from_f16(x);
    }
}

/// Converts eight f16 activations to f32 with NEON, whose conversion is many
/// times faster than one that goes a value at a time.
#[cfg(all(target_arch = "aarch64", target_feature = "fp16"))]
fn from_f16_8(bits: &[u16; 8]) -> [f32; 8] {
    use std::arch::aarch64::{float16x4_t, uint16x4_t, vcvt_f32_f16, vld1_u16, vst1q_f32};

    let mut out = [0.0; 8];
    for i in [0, 4] {
        unsafe {
            let half = std::mem::transmute::<uint16x4_t, float16x4_t>(vld1_u16(bits[i..].as_ptr()));
            vst1q_f32(out[i..].as_mut_ptr(), vcvt_f32_f16(half));
        }
    }
    out
}

/// Converts eight f16 activations to f32.
#[cfg(not(all(target_arch = "aarch64", target_feature = "fp16")))]
fn from_f16_8(bits: &[u16; 8]) -> [f32; 8] {
    bits.map(from_f16)
}

fn softmax(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in x.iter_mut() {
        *v /= sum;
    }
}

#[cfg(test)]
mod tests {
    use super::from_f16_8;
    use crate::{F16_MAX, from_f16, to_f16};

    #[test]
    fn f16_round_trips_what_it_can_represent() {
        for x in [0.0, -0.0, 1.0, -2.5, 0.333_251_95, 65504.0, -65504.0, 1e-5, -3e-7, 2.0f32.powi(-14)] {
            let bits = to_f16(x);
            let back = from_f16(bits);
            assert_eq!(to_f16(back), bits, "{x}");
        }
        for bits in 0..=u16::MAX {
            if bits & 0x7c00 != 0x7c00 {
                assert_eq!(to_f16(from_f16(bits)), bits, "{bits:#06x}");
            }
        }
    }

    #[test]
    fn f16_converts_the_same_eight_at_a_time() {
        let finite: Vec<u16> = (0..=u16::MAX).filter(|bits| bits & 0x7c00 != 0x7c00).collect();
        for bits in finite.chunks_exact(8) {
            let bits: &[u16; 8] = bits.try_into().unwrap();
            assert_eq!(from_f16_8(bits).map(f32::to_bits), bits.map(|b| from_f16(b).to_bits()));
        }
    }

    #[test]
    fn f16_rounds_to_nearest_and_clamps() {
        assert_eq!(from_f16(to_f16(1.0 + 2.0f32.powi(-11))), 1.0, "a tie rounds to even");
        assert_eq!(from_f16(to_f16(1.0 + 3.0 * 2.0f32.powi(-11))), 1.0 + 2.0f32.powi(-9));
        assert_eq!(from_f16(to_f16(0.1)), 0.099975586);
        assert_eq!(from_f16(to_f16(1e6)), F16_MAX);
        assert_eq!(from_f16(to_f16(-1e6)), -F16_MAX);
        assert_eq!(from_f16(to_f16(1e-9)), 0.0);
    }
}
