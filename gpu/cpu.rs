//! The model on the CPU: the reference implementation of every operation,
//! which the GPU kernels are checked against.

use rayon::prelude::*;

use crate::{CONV_KERNEL, Device, HADAMARD_BLOCK, Tensor, bf16, from_f16, sigmoid, softplus, ternary, to_f16};

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

    fn read_cache(&self, cache: &Vec<u16>, len: usize) -> Vec<u16> {
        cache[..len].to_vec()
    }

    fn write_cache(&self, cache: &mut Vec<u16>, data: &[u16]) {
        cache[..data.len()].copy_from_slice(data);
    }

    fn matmul(&self, out: &mut Vec<f32>, w: &Tensor, x: &Vec<f32>) {
        let (rows, cols) = (w.shape()[0], w.shape()[1]);
        let n = x.len() / cols;
        assert_eq!(x.len(), n * cols);
        assert_eq!(out.len(), n * rows);
        // Each row of the weights is read from memory once, for every
        // token: the results come out by row, and are transposed into `out`,
        // which holds them by token.
        let mut by_row = vec![0.0; rows * n];
        let row_dot = |results: &mut [f32], dot: &dyn Fn(&[f32]) -> f32| {
            for (result, x) in results.iter_mut().zip(x.chunks_exact(cols)) {
                *result = dot(x);
            }
        };
        match w {
            Tensor::Bf16 { data, .. } => by_row
                .par_chunks_exact_mut(n)
                .zip(data.par_chunks_exact(cols))
                .for_each(|(results, row)| row_dot(results, &|x| dot(row, x))),
            Tensor::Ternary { data, .. } => by_row
                .par_chunks_exact_mut(n)
                .zip(data.par_chunks_exact(ternary::row_bytes(cols)))
                .for_each(|(results, row)| row_dot(results, &|x| ternary::dot(row, x))),
        }
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

    fn rmsnorm(&self, x: &mut Vec<f32>, weight: &Vec<f32>, eps: f32) {
        let dim = weight.len();
        for row in x.chunks_exact_mut(dim) {
            let mean_square = row.iter().map(|v| v * v).sum::<f32>() / dim as f32;
            let scale = 1.0 / (mean_square + eps).sqrt();
            for (v, &w) in row.iter_mut().zip(weight) {
                *v *= scale * w;
            }
        }
    }

    fn l2norm(&self, x: &mut Vec<f32>, dim: usize, eps: f32) {
        for row in x.chunks_exact_mut(dim) {
            let scale = 1.0 / (row.iter().map(|v| v * v).sum::<f32>() + eps).sqrt();
            for v in row {
                *v *= scale;
            }
        }
    }

    fn rope(&self, x: &mut Vec<f32>, table: &Vec<f32>, pos: usize, n_heads: usize, head_dim: usize, rot_dim: usize) {
        // The rotated part of each head is rotated as pairs of elements
        // half of it apart, each pair by its own angle.
        let half = rot_dim / 2;
        for (t, token) in x.chunks_exact_mut(n_heads * head_dim).enumerate() {
            let angles = &table[(pos + t) * rot_dim..][..rot_dim];
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

    fn sigmoid_mul(&self, x: &mut Vec<f32>, gate: &Vec<f32>) {
        for (x, &g) in x.iter_mut().zip(gate) {
            *x *= sigmoid(g);
        }
    }

    fn hadamard(&self, x: &mut Vec<f32>, signs: &Vec<f32>, inverse: bool) {
        let width = signs.len();
        assert!(x.len().is_multiple_of(width) && width.is_multiple_of(HADAMARD_BLOCK));
        for row in x.chunks_exact_mut(width) {
            for (block, signs) in row.chunks_exact_mut(HADAMARD_BLOCK).zip(signs.chunks_exact(HADAMARD_BLOCK)) {
                if !inverse {
                    for (v, &s) in block.iter_mut().zip(signs) {
                        *v *= s;
                    }
                }
                walsh_hadamard(block);
                if inverse {
                    for (v, &s) in block.iter_mut().zip(signs) {
                        *v *= s;
                    }
                }
            }
        }
    }

    fn norm_rotate(&self, out: &mut Vec<f32>, x: &Vec<f32>, weight: &Vec<f32>, signs: &Vec<f32>, eps: f32) {
        assert_eq!(out.len(), x.len());
        out.copy_from_slice(x);
        self.rmsnorm(out, weight, eps);
        self.hadamard(out, signs, false);
    }

    fn conv(
        &self,
        q: &mut Vec<f32>,
        k: &mut Vec<f32>,
        v: &mut Vec<f32>,
        state_out: &mut Vec<f32>,
        x: &Vec<f32>,
        state: &Vec<f32>,
        weight: &Vec<f32>,
    ) {
        let channels = weight.len() / CONV_KERNEL;
        let n = x.len() / channels;
        assert_eq!(x.len(), n * channels);
        assert!(state.len() == (CONV_KERNEL - 1) * channels && state_out.len() == state.len());
        let (q_dim, k_dim, v_dim) = (q.len() / n, k.len() / n, v.len() / n);
        assert_eq!(q_dim + k_dim + v_dim, channels);
        // The rows before the batch, then the batch.
        let input = |t: isize, c: usize| {
            if t < 0 {
                state[(t + CONV_KERNEL as isize - 1) as usize * channels + c]
            } else {
                x[t as usize * channels + c]
            }
        };
        for t in 0..n {
            for c in 0..channels {
                let mut acc = 0.0;
                for j in 0..CONV_KERNEL {
                    acc += weight[c * CONV_KERNEL + j] * input(t as isize + j as isize + 1 - CONV_KERNEL as isize, c);
                }
                let y = acc * sigmoid(acc);
                if c < q_dim {
                    q[t * q_dim + c] = y;
                } else if c < q_dim + k_dim {
                    k[t * k_dim + c - q_dim] = y;
                } else {
                    v[t * v_dim + c - q_dim - k_dim] = y;
                }
            }
        }
        for s in 0..CONV_KERNEL - 1 {
            for c in 0..channels {
                state_out[s * channels + c] = input(n as isize + s as isize + 1 - CONV_KERNEL as isize, c);
            }
        }
    }

    fn delta_net(
        &self,
        out: &mut Vec<f32>,
        q: &Vec<f32>,
        k: &Vec<f32>,
        v: &Vec<f32>,
        gates: &Vec<f32>,
        decay: &Vec<f32>,
        state: &mut Vec<f32>,
        n_k_heads: usize,
        n_v_heads: usize,
        head_dim: usize,
    ) {
        let n = v.len() / (n_v_heads * head_dim);
        assert_eq!(v.len(), n * n_v_heads * head_dim);
        assert!(q.len() == n * n_k_heads * head_dim && k.len() == q.len() && out.len() == v.len());
        assert!(gates.len() == n * 2 * n_v_heads && decay.len() == 2 * n_v_heads);
        assert_eq!(state.len(), n_v_heads * head_dim * head_dim);
        let group = n_v_heads / n_k_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        // One head at a time, its state as `s[key][value]`.
        let outs: Vec<Vec<f32>> = state
            .par_chunks_exact_mut(head_dim * head_dim)
            .enumerate()
            .map(|(h, s)| {
                let kh = h / group;
                let mut out = vec![0.0; n * head_dim];
                let mut kv = vec![0.0; head_dim];
                for t in 0..n {
                    let q = &q[(t * n_k_heads + kh) * head_dim..][..head_dim];
                    let k = &k[(t * n_k_heads + kh) * head_dim..][..head_dim];
                    let v = &v[(t * n_v_heads + h) * head_dim..][..head_dim];
                    let alpha = gates[t * 2 * n_v_heads + h];
                    let beta = sigmoid(gates[t * 2 * n_v_heads + n_v_heads + h]);
                    let g = (decay[h] * softplus(alpha + decay[n_v_heads + h])).exp();
                    for e in s.iter_mut() {
                        *e *= g;
                    }
                    kv.fill(0.0);
                    for (i, &ki) in k.iter().enumerate() {
                        for (j, &sij) in s[i * head_dim..][..head_dim].iter().enumerate() {
                            kv[j] += sij * ki;
                        }
                    }
                    for (i, &ki) in k.iter().enumerate() {
                        for (j, sij) in s[i * head_dim..][..head_dim].iter_mut().enumerate() {
                            *sij += ki * (v[j] - kv[j]) * beta;
                        }
                    }
                    let o = &mut out[t * head_dim..][..head_dim];
                    for (i, &qi) in q.iter().enumerate() {
                        for (j, &sij) in s[i * head_dim..][..head_dim].iter().enumerate() {
                            o[j] += sij * qi * scale;
                        }
                    }
                }
                out
            })
            .collect();
        for (h, o) in outs.iter().enumerate() {
            for t in 0..n {
                out[(t * n_v_heads + h) * head_dim..][..head_dim].copy_from_slice(&o[t * head_dim..][..head_dim]);
            }
        }
    }
}

/// The normalized Walsh-Hadamard transform of a block, in place: the sums
/// and differences of pairs one, two, four, and so on apart, over 1/√n.
fn walsh_hadamard(x: &mut [f32]) {
    let n = x.len();
    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(2 * h) {
            for j in i..i + h {
                let (a, b) = (x[j], x[j + h]);
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
        h *= 2;
    }
    let scale = 1.0 / (n as f32).sqrt();
    for v in x {
        *v *= scale;
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
    use super::{from_f16_8, walsh_hadamard};
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

    #[test]
    fn walsh_hadamard_is_sylvesters_matrix_and_its_own_inverse() {
        // Row i of the matrix is the transform of the ith unit vector, and
        // has the sign of the parity of i & j in column j.
        let n = 16;
        for i in 0..n {
            let mut x = vec![0.0; n];
            x[i] = 1.0;
            walsh_hadamard(&mut x);
            for (j, &v) in x.iter().enumerate() {
                let sign = if (i & j).count_ones() % 2 == 0 { 1.0 } else { -1.0 };
                assert_eq!(v, sign / 4.0, "row {i} column {j}");
            }
            walsh_hadamard(&mut x);
            for (j, &v) in x.iter().enumerate() {
                assert!((v - if i == j { 1.0 } else { 0.0 }).abs() < 1e-6);
            }
        }
    }
}
