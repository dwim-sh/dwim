//! The model on the CPU: the reference implementation of every operation,
//! which the GPU kernels are checked against.

use rayon::prelude::*;

use crate::gpu::{
    CACHE_BLOCK, CONV_KERNEL, Device, Tensor,
    float::{bf16, from_f16},
    quant::{self, quantize},
};

pub struct Cpu;

/// A cache of activations in blocks of [`CACHE_BLOCK`] 8-bit integers, and
/// the blocks' scales as f16 bits.
pub struct Cache {
    quants: Vec<i8>,
    scales: Vec<u16>,
}

impl Device for Cpu {
    type Buffer = Vec<f32>;
    type Weight = Tensor;
    type Cache = Cache;

    fn name(&self) -> String {
        "cpu".to_string()
    }

    fn memory(&self) -> Option<u64> {
        crate::sys::memory::available()
    }

    fn takes_nvfp4(&self) -> bool {
        true
    }

    fn upload(&self, tensor: Tensor) -> Tensor {
        tensor
    }

    fn alloc(&self, len: usize) -> Vec<f32> {
        vec![0.0; len]
    }

    fn alloc_cache(&self, len: usize) -> Cache {
        assert!(len.is_multiple_of(CACHE_BLOCK));
        Cache {
            quants: vec![0; len],
            scales: vec![0; len / CACHE_BLOCK],
        }
    }

    fn resize(&self, buf: &mut Vec<f32>, len: usize) {
        assert!(
            len <= buf.capacity(),
            "a buffer of {} activations can't hold {len}",
            buf.capacity()
        );
        buf.resize(len, 0.0);
    }

    fn read(&self, buf: &Vec<f32>) -> Vec<f32> {
        buf.clone()
    }

    fn write(&self, buf: &mut Vec<f32>, data: &[f32]) {
        buf.copy_from_slice(data);
    }

    fn copy(
        &self,
        dst: &mut Vec<f32>,
        dst_offset: usize,
        src: &Vec<f32>,
        src_offset: usize,
        len: usize,
    ) {
        dst[dst_offset..][..len].copy_from_slice(&src[src_offset..][..len]);
    }

    fn store(&self, cache: &mut Cache, offset: usize, src: &Vec<f32>) {
        assert!(offset.is_multiple_of(CACHE_BLOCK) && src.len().is_multiple_of(CACHE_BLOCK));
        let quants = cache.quants[offset..][..src.len()].chunks_exact_mut(CACHE_BLOCK);
        let scales = &mut cache.scales[offset / CACHE_BLOCK..];
        for ((quants, scale), block) in quants.zip(scales).zip(src.chunks_exact(CACHE_BLOCK)) {
            let (bits, q) = quantize(block);
            quants.copy_from_slice(&q);
            *scale = bits;
        }
    }

    fn read_cache(&self, cache: &Cache, len: usize) -> Vec<u8> {
        assert!(len.is_multiple_of(CACHE_BLOCK));
        let quants = cache.quants[..len].iter().map(|&q| q as u8);
        let scales = cache.scales[..len / CACHE_BLOCK]
            .iter()
            .flat_map(|s| s.to_le_bytes());
        quants.chain(scales).collect()
    }

    fn write_cache(&self, cache: &mut Cache, data: &[u8]) {
        let len = data.len() / (CACHE_BLOCK + 2) * CACHE_BLOCK;
        assert_eq!(data.len(), crate::gpu::cache_bytes(len));
        let (quants, scales) = data.split_at(len);
        for (q, &b) in cache.quants[..len].iter_mut().zip(quants) {
            *q = b as i8;
        }
        for (s, b) in cache.scales.iter_mut().zip(scales.chunks_exact(2)) {
            *s = u16::from_le_bytes([b[0], b[1]]);
        }
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
            Tensor::Q4_0 { scales, quants, .. } => by_row
                .par_chunks_exact_mut(n)
                .zip(scales.par_chunks_exact(quant::blocks(cols)))
                .zip(quants.par_chunks_exact(cols / 2))
                .for_each(|((results, scales), quants)| {
                    row_dot(results, &|x| quant::dot_q4_0(scales, quants, x))
                }),
            Tensor::Q8_0 { scales, quants, .. } => by_row
                .par_chunks_exact_mut(n)
                .zip(scales.par_chunks_exact(quant::blocks(cols)))
                .zip(quants.par_chunks_exact(cols))
                .for_each(|((results, scales), quants)| {
                    row_dot(results, &|x| quant::dot_q8_0(scales, quants, x))
                }),
            Tensor::Nvfp4 { scales, quants, .. } => by_row
                .par_chunks_exact_mut(n)
                .zip(scales.par_chunks_exact(cols / quant::NVFP4_BLOCK))
                .zip(quants.par_chunks_exact(cols / 2))
                .for_each(|((results, scales), quants)| {
                    row_dot(results, &|x| quant::dot_nvfp4(scales, quants, x))
                }),
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

    fn rope(
        &self,
        x: &mut Vec<f32>,
        table: &Vec<f32>,
        pos: usize,
        n_heads: usize,
        head_dim: usize,
        rot_dim: usize,
    ) {
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
        k_cache: &Cache,
        v_cache: &Cache,
        pos: usize,
        n_heads: usize,
        head_dim: usize,
        n_kv_heads: usize,
    ) {
        // With grouped-query attention, consecutive query heads share a key
        // and value head.
        let group = n_heads / n_kv_heads;
        let kv_dim = n_kv_heads * head_dim;
        assert!(head_dim.is_multiple_of(CACHE_BLOCK));
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
                    .map(|t| scale * dot_q8(q, k_cache, t * kv_dim + kv))
                    .collect();
                softmax(&mut scores);
                out.fill(0.0);
                for (t, &score) in scores.iter().enumerate() {
                    add_scaled_q8(out, score, v_cache, t * kv_dim + kv);
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
                    acc += weight[c * CONV_KERNEL + j]
                        * input(t as isize + j as isize + 1 - CONV_KERNEL as isize, c);
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
                state_out[s * channels + c] =
                    input(n as isize + s as isize + 1 - CONV_KERNEL as isize, c);
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
                out[(t * n_v_heads + h) * head_dim..][..head_dim]
                    .copy_from_slice(&o[t * head_dim..][..head_dim]);
            }
        }
    }

    fn mix(&self, out: &mut Vec<f32>, states: &[&Vec<f32>], query: &Vec<f32>, eps: f32) {
        let dim = query.len();
        let scale = 1.0 / (dim as f32).sqrt();
        let mut normed = vec![0.0; states.len() * dim];
        let mut scores = vec![0.0; states.len()];
        for (t, out) in out.chunks_exact_mut(dim).enumerate() {
            for ((state, normed), score) in states
                .iter()
                .zip(normed.chunks_exact_mut(dim))
                .zip(&mut scores)
            {
                let row = &state[t * dim..][..dim];
                let mean_square = row.iter().map(|v| v * v).sum::<f32>() / dim as f32;
                let norm = 1.0 / (mean_square + eps).sqrt();
                let mut dot = 0.0;
                for ((n, &v), &q) in normed.iter_mut().zip(row).zip(query) {
                    *n = v * norm;
                    dot += *n * q;
                }
                *score = dot * scale;
            }
            softmax(&mut scores);
            out.fill(0.0);
            for (normed, &score) in normed.chunks_exact(dim).zip(&scores) {
                for (o, &n) in out.iter_mut().zip(normed) {
                    *o += score * n;
                }
            }
        }
    }

    fn gated_update(
        &self,
        x: &mut Vec<f32>,
        forget: &Vec<f32>,
        gain: &Vec<f32>,
        message: &Vec<f32>,
    ) {
        for (((x, &f), &g), &m) in x.iter_mut().zip(forget).zip(gain).zip(message) {
            *x = *x - f.tanh() * *x + g.tanh() * m;
        }
    }

    fn short_conv(
        &self,
        out: &mut Vec<f32>,
        x: &Vec<f32>,
        state: &mut Vec<f32>,
        weight: &Vec<f32>,
        dynamic: Option<(&Vec<f32>, usize)>,
    ) {
        let channels = weight.len() - state.len();
        let taps = weight.len() / channels;
        let n = x.len() / channels;
        assert!(x.len() == n * channels && out.len() == x.len());
        assert_eq!(state.len(), (taps - 1) * channels);
        // The channel's input `back` tokens before token `t`.
        let input = |t: usize, back: usize, c: usize| {
            if back > t {
                state[(taps - 1 + t - back) * channels + c]
            } else {
                x[(t - back) * channels + c]
            }
        };
        let scale = 1.0 / ((if dynamic.is_some() { 2 } else { 1 } * taps + 1) as f32).sqrt();
        for t in 0..n {
            for c in 0..channels {
                let mut acc = x[t * channels + c];
                for j in 0..taps {
                    acc += weight[c * taps + j] * input(t, taps - 1 - j, c);
                }
                if let Some((dynamic, heads)) = dynamic {
                    let head = c / (channels / heads);
                    for lag in 0..taps {
                        acc += dynamic[(t * heads + head) * taps + lag] * input(t, lag, c);
                    }
                }
                out[t * channels + c] = acc * scale;
            }
        }
        let mut next = vec![0.0; state.len()];
        for s in 0..taps - 1 {
            for c in 0..channels {
                // The state's row `s` is the token `taps - 1 - s` back
                // from the one after the batch.
                next[s * channels + c] = if n + s >= taps - 1 {
                    x[(n + s + 1 - taps) * channels + c]
                } else {
                    state[(n + s) * channels + c]
                };
            }
        }
        *state = next;
    }

    fn clip(&self, x: &mut Vec<f32>, scale: &Vec<f32>, dim: usize, threshold: f32) {
        for row in x.chunks_exact_mut(scale.len()) {
            for (head, scale) in row.chunks_exact_mut(dim).zip(scale.chunks_exact(dim)) {
                let mut square = 0.0;
                for (v, &s) in head.iter_mut().zip(scale) {
                    *v *= s;
                    square += *v * *v;
                }
                let factor = (threshold / (square.sqrt() + 1e-6)).min(1.0);
                for v in head {
                    *v *= factor;
                }
            }
        }
    }

    fn scale_rows(&self, x: &mut Vec<f32>, factors: &Vec<f32>) {
        let width = x.len() / factors.len();
        for (row, &factor) in x.chunks_exact_mut(width).zip(factors) {
            for v in row {
                *v *= factor;
            }
        }
    }

    fn block_conv(
        &self,
        out: &mut Vec<f32>,
        x: &Vec<f32>,
        taps: &Vec<f32>,
        from: usize,
        base: &Vec<f32>,
        kernel: usize,
        group: usize,
    ) {
        let channels = base.len() / kernel;
        let groups = channels / group;
        let n = x.len() / channels;
        assert!(x.len() == n * channels && out.len() == x.len());
        assert!(base.len() == kernel * channels && channels == groups * group);
        let width = taps.len() / n;
        assert!(taps.len() == n * width && from + kernel * groups <= width);
        for t in 0..n {
            let own = &taps[t * width + from..][..kernel * groups];
            for c in 0..channels {
                let mut acc = 0.0;
                for back in 0..kernel.min(t + 1) {
                    let tap = base[back * channels + c] + own[back * groups + c / group];
                    acc += tap * x[(t - back) * channels + c];
                }
                out[t * channels + c] = acc;
            }
        }
    }

    fn attention_rel(
        &self,
        out: &mut Vec<f32>,
        q: &Vec<f32>,
        k_cache: &Cache,
        v_cache: &Cache,
        rel: &Vec<f32>,
        bias: &Vec<f32>,
        sinks: &Vec<f32>,
        pos: usize,
        n_heads: usize,
        head_dim: usize,
        n_kv_heads: usize,
        window: usize,
        rows: usize,
    ) {
        let group = n_heads / n_kv_heads;
        let kv_dim = n_kv_heads * head_dim;
        assert!(head_dim.is_multiple_of(CACHE_BLOCK));
        let n = q.len() / (n_heads * head_dim);
        let n_rel = rel.len() / (n * n_heads);
        let extent = bias.len() / n_rel;
        let scale = 1.0 / (head_dim as f32).sqrt();
        out.par_chunks_exact_mut(head_dim)
            .zip(q.par_chunks_exact(head_dim))
            .enumerate()
            .for_each(|(i, (out, q))| {
                let (token, head) = (i / n_heads, i % n_heads);
                let rel = &rel[i * n_rel..][..n_rel];
                // The token attends up to and including its own position,
                // from as far back as the window reaches.
                let at = pos + token;
                let first = (at + 1).saturating_sub(window);
                let kv = head / group * head_dim;
                let mut scores: Vec<f32> = (first..=at)
                    .map(|t| {
                        let distance = at - t;
                        let bias = if distance < extent {
                            let row = &bias[distance * n_rel..][..n_rel];
                            rel.iter().zip(row).map(|(r, b)| r * b).sum()
                        } else {
                            0.0
                        };
                        scale * dot_q8(q, k_cache, t % rows * kv_dim + kv) + bias
                    })
                    .collect();
                // The sink is one more to the softmax, with no value.
                scores.push(sinks[i]);
                softmax(&mut scores);
                out.fill(0.0);
                for (t, &score) in (first..=at).zip(&scores) {
                    add_scaled_q8(out, score, v_cache, t % rows * kv_dim + kv);
                }
            });
    }

    fn route(
        &self,
        ids: &mut Vec<f32>,
        weights: &mut Vec<f32>,
        logits: &Vec<f32>,
        bias: &Vec<f32>,
        k: usize,
        scale: f32,
    ) {
        let experts = bias.len();
        for ((logits, ids), weights) in logits
            .chunks_exact(experts)
            .zip(ids.chunks_exact_mut(k))
            .zip(weights.chunks_exact_mut(k))
        {
            let probs: Vec<f32> = logits.iter().map(|&l| sigmoid(l)).collect();
            // The highest scores first, the lower number first of two the
            // same.
            let mut order: Vec<usize> = (0..experts).collect();
            order.sort_by(|&a, &b| {
                (probs[b] + bias[b])
                    .total_cmp(&(probs[a] + bias[a]))
                    .then(a.cmp(&b))
            });
            let sum: f32 = order[..k].iter().map(|&e| probs[e]).sum();
            for ((id, weight), &e) in ids.iter_mut().zip(weights.iter_mut()).zip(&order) {
                *id = e as f32;
                *weight = probs[e] / (sum + 1e-20) * scale;
            }
        }
    }

    fn expert_matmul(
        &self,
        out: &mut Vec<f32>,
        w: &Tensor,
        scale: &Vec<f32>,
        x: &Vec<f32>,
        ids: &Vec<f32>,
        k: usize,
    ) {
        let cols = w.shape()[1];
        let rows = w.shape()[0] / scale.len();
        let n = ids.len() / k;
        assert_eq!(out.len(), ids.len() * rows);
        // A row of the input per token, or one per expert of a token.
        let shared = x.len() == n * cols;
        assert!(shared || x.len() == ids.len() * cols);
        out.par_chunks_exact_mut(rows)
            .enumerate()
            .for_each(|(i, out)| {
                let expert = ids[i] as usize;
                let x = &x[if shared { i / k } else { i } * cols..][..cols];
                for (r, out) in out.iter_mut().enumerate() {
                    let row = expert * rows + r;
                    *out = scale[expert]
                        * match w {
                            Tensor::Bf16 { data, .. } => dot(&data[row * cols..][..cols], x),
                            Tensor::Q4_0 { scales, quants, .. } => {
                                let blocks = quant::blocks(cols);
                                quant::dot_q4_0(
                                    &scales[row * blocks..][..blocks],
                                    &quants[row * cols / 2..][..cols / 2],
                                    x,
                                )
                            }
                            Tensor::Q8_0 { scales, quants, .. } => {
                                let blocks = quant::blocks(cols);
                                quant::dot_q8_0(
                                    &scales[row * blocks..][..blocks],
                                    &quants[row * cols..][..cols],
                                    x,
                                )
                            }
                            Tensor::Nvfp4 { scales, quants, .. } => {
                                let blocks = cols / quant::NVFP4_BLOCK;
                                quant::dot_nvfp4(
                                    &scales[row * blocks..][..blocks],
                                    &quants[row * cols / 2..][..cols / 2],
                                    x,
                                )
                            }
                        };
                }
            });
    }

    fn situ(
        &self,
        out: &mut Vec<f32>,
        gate_up: &Vec<f32>,
        width: usize,
        beta: f32,
        linear_beta: f32,
    ) {
        for (out, gate_up) in out
            .chunks_exact_mut(width)
            .zip(gate_up.chunks_exact(2 * width))
        {
            let (gate, up) = gate_up.split_at(width);
            for ((o, &g), &u) in out.iter_mut().zip(gate).zip(up) {
                *o = beta * (g / beta).tanh() * sigmoid(g) * linear_beta * (u / linear_beta).tanh();
            }
        }
    }

    fn weighted_sum(&self, out: &mut Vec<f32>, y: &Vec<f32>, weights: &Vec<f32>) {
        let width = y.len() / weights.len();
        let k = weights.len() / (out.len() / width);
        for ((out, y), weights) in out
            .chunks_exact_mut(width)
            .zip(y.chunks_exact(k * width))
            .zip(weights.chunks_exact(k))
        {
            out.fill(0.0);
            for (y, &weight) in y.chunks_exact(width).zip(weights) {
                for (o, &y) in out.iter_mut().zip(y) {
                    *o += weight * y;
                }
            }
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
    let rest: f32 = w[done..]
        .iter()
        .zip(&x[done..])
        .map(|(&w, x)| bf16(w) * x)
        .sum();
    acc.iter().sum::<f32>() + rest
}

/// Dot product of f32 activations with those of a cache from `offset` on,
/// a block at a time.
fn dot_q8(a: &[f32], cache: &Cache, offset: usize) -> f32 {
    let quants = cache.quants[offset..][..a.len()].chunks_exact(CACHE_BLOCK);
    let scales = &cache.scales[offset / CACHE_BLOCK..];
    a.chunks_exact(CACHE_BLOCK)
        .zip(quants)
        .zip(scales)
        .map(|((a, q), &s)| {
            let dot: f32 = a.iter().zip(q).map(|(a, &q)| a * q as f32).sum();
            dot * from_f16(s)
        })
        .sum()
}

/// `out += scale * x` for the activations `x` of a cache from `offset` on.
fn add_scaled_q8(out: &mut [f32], scale: f32, cache: &Cache, offset: usize) {
    let quants = cache.quants[offset..][..out.len()].chunks_exact(CACHE_BLOCK);
    let scales = &cache.scales[offset / CACHE_BLOCK..];
    for ((out, q), &s) in out.chunks_exact_mut(CACHE_BLOCK).zip(quants).zip(scales) {
        let scale = scale * from_f16(s);
        for (o, &q) in out.iter_mut().zip(q) {
            *o += scale * q as f32;
        }
    }
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

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// `ln(1 + e^x)`, without overflowing for large `x`.
fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

#[cfg(test)]
mod tests {
    use crate::gpu::{
        CACHE_BLOCK, Tensor,
        float::{F16_MAX, from_f16, to_f16},
        quant::{self, quantize},
    };

    /// Values in [-1, 1) from a seed.
    fn values(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn selecting_rows_and_column_groups_reorders_every_format() {
        let (rows, cols) = (3, 256);
        let w = values(rows * cols, 5);
        let mut tensors = vec![Tensor::Bf16 {
            shape: vec![rows, cols],
            data: w.iter().map(|v| (v.to_bits() >> 16) as u16).collect(),
        }];
        let blocks = quant::blocks(cols);
        let (mut scales, mut quants) = (vec![0; rows * blocks], vec![0; rows * cols / 2]);
        for ((scales, quants), w) in scales
            .chunks_exact_mut(blocks)
            .zip(quants.chunks_exact_mut(cols / 2))
            .zip(w.chunks_exact(cols))
        {
            quant::quantize_q4_0_row(w, scales, quants);
        }
        tensors.push(Tensor::Q4_0 {
            shape: vec![rows, cols],
            scales,
            quants,
        });
        let (mut scales, mut quants) = (vec![0; rows * blocks], vec![0; rows * cols]);
        for ((scales, quants), w) in scales
            .chunks_exact_mut(blocks)
            .zip(quants.chunks_exact_mut(cols))
            .zip(w.chunks_exact(cols))
        {
            quant::quantize_q8_0_row(w, scales, quants);
        }
        tensors.push(Tensor::Q8_0 {
            shape: vec![rows, cols],
            scales,
            quants,
        });
        for t in &tensors {
            let whole = t.dequantize();
            // Rows 2, 0 and 2 again; then the second half of the columns
            // before the first.
            let picked = t.select_rows(&[2, 0, 2]);
            assert_eq!(picked.shape(), [3, cols]);
            let got = picked.dequantize();
            for (i, r) in [2, 0, 2].into_iter().enumerate() {
                assert_eq!(got[i * cols..][..cols], whole[r * cols..][..cols]);
            }
            let swapped = t.select_column_groups(128, &[1, 0]);
            assert_eq!(swapped.shape(), [rows, cols]);
            let got = swapped.dequantize();
            for r in 0..rows {
                assert_eq!(got[r * cols..][..128], whole[r * cols + 128..][..128]);
                assert_eq!(got[r * cols + 128..][..128], whole[r * cols..][..128]);
            }
        }
    }

    #[test]
    fn f16_round_trips_what_it_can_represent() {
        for x in [
            0.0,
            -0.0,
            1.0,
            -2.5,
            0.333_251_95,
            65504.0,
            -65504.0,
            1e-5,
            -3e-7,
            2.0f32.powi(-14),
        ] {
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
    fn quantize_rounds_to_the_nearest_multiple_of_the_scale() {
        // A largest magnitude of 127 makes the scale exactly one, and the
        // halves ties, which go away from zero.
        let mut block = [0.0; CACHE_BLOCK];
        block[..8].copy_from_slice(&[127.0, 0.5, 1.5, -2.5, 0.49, -0.51, 126.6, -0.0]);
        let (scale, q) = quantize(&block);
        assert_eq!(from_f16(scale), 1.0);
        assert_eq!(q[..8], [127, 1, 2, -3, 0, -1, 127, 0]);

        // The largest magnitude is 127 multiples of the scale, or as near
        // as the scale rounded to f16 comes, and never more.
        let block: Vec<f32> = (0..CACHE_BLOCK).map(|i| (i as f32 - 13.3) * 0.37).collect();
        let (scale, q) = quantize(&block);
        let d = from_f16(scale);
        for (&x, &q) in block.iter().zip(&q) {
            assert!((x - q as f32 * d).abs() <= d / 2.0, "{x} as {q} of {d}");
        }
        assert_eq!(q.iter().map(|q| q.unsigned_abs()).max(), Some(127));

        assert_eq!(quantize(&[0.0; CACHE_BLOCK]), (0, [0; CACHE_BLOCK]));
        let (scale, q) = quantize(&[-1e8; CACHE_BLOCK]);
        assert_eq!((from_f16(scale), q), (F16_MAX, [-127; CACHE_BLOCK]));
    }

    #[test]
    fn f16_rounds_to_nearest_and_clamps() {
        assert_eq!(
            from_f16(to_f16(1.0 + 2.0f32.powi(-11))),
            1.0,
            "a tie rounds to even"
        );
        assert_eq!(
            from_f16(to_f16(1.0 + 3.0 * 2.0f32.powi(-11))),
            1.0 + 2.0f32.powi(-9)
        );
        assert_eq!(from_f16(to_f16(0.1)), 0.099975586);
        assert_eq!(from_f16(to_f16(1e6)), F16_MAX);
        assert_eq!(from_f16(to_f16(-1e6)), -F16_MAX);
        assert_eq!(from_f16(to_f16(1e-9)), 0.0);
    }
}
