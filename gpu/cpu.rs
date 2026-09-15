//! The model on the CPU: the reference implementation of every operation,
//! which the GPU kernels are checked against.

use rayon::prelude::*;

use crate::{Device, Tensor, bf16};

pub struct Cpu;

impl Device for Cpu {
    type Buffer = Vec<f32>;
    type Weight = Tensor;

    fn upload(&self, tensor: Tensor) -> Tensor {
        tensor
    }

    fn alloc(&self, len: usize) -> Vec<f32> {
        vec![0.0; len]
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
        k_cache: &Vec<f32>,
        v_cache: &Vec<f32>,
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
                    .map(|t| scale * dot_f32(q, &k_cache[t * kv_dim + kv..][..head_dim]))
                    .collect();
                softmax(&mut scores);
                out.fill(0.0);
                for (t, &score) in scores.iter().enumerate() {
                    let v = &v_cache[t * kv_dim + kv..][..head_dim];
                    for (o, &v) in out.iter_mut().zip(v) {
                        *o += score * v;
                    }
                }
            });
    }

    fn silu_mul(&self, gate: &mut Vec<f32>, up: &Vec<f32>) {
        for (g, &u) in gate.iter_mut().zip(up) {
            *g = *g / (1.0 + (-*g).exp()) * u;
        }
    }
}

/// Dot product of a row of bf16 weights with f32 activations.
///
/// Accumulating into eight independent sums lets the compiler vectorize the
/// loop, which a single running sum would not allow.
fn dot(w: &[u16], x: &[f32]) -> f32 {
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

fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
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
