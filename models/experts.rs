//! Mixture-of-experts feed-forward on the CPU: the experts' four-bit weights
//! stay in the pack's memory mapping, and only the few an input is routed to
//! are read, spread across every core.

use std::sync::Arc;

use rayon::prelude::*;

use crate::{
    pack::{Dtype, Pack},
    q4,
};

/// One layer's routed experts.
pub struct Experts {
    pack: Arc<Pack>,
    gate: String,
    up: String,
    down: String,
    pub n_experts: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub top_k: usize,
}

/// Rows of the gate and up projections handled by one task.
const CHUNK: usize = 64;

impl Experts {
    /// The experts of the layer whose tensors have the given prefix, such
    /// as `model.layers.3.mlp.experts`.
    pub fn new(pack: Arc<Pack>, prefix: &str, top_k: usize) -> crate::Result<Self> {
        let gate = format!("{prefix}.gate_proj");
        let up = format!("{prefix}.up_proj");
        let down = format!("{prefix}.down_proj");
        let entry = pack.entry(&gate)?;
        if entry.dtype != Dtype::Q4 || entry.shape.len() != 3 {
            return Err(format!("'{gate}' is not a three-dimensional four-bit tensor").into());
        }
        let [n_experts, intermediate, hidden] = [entry.shape[0], entry.shape[1], entry.shape[2]];
        if pack.entry(&up)?.shape != entry.shape || pack.entry(&down)?.shape != [n_experts, hidden, intermediate] {
            return Err(format!("expert shapes under '{prefix}' disagree").into());
        }
        Ok(Self {
            pack,
            gate,
            up,
            down,
            n_experts,
            hidden,
            intermediate,
            top_k,
        })
    }

    /// Picks each token's experts from its router logits: the `top_k` most
    /// likely, with their probabilities renormalized to sum to one.
    pub fn route(&self, logits: &[f32]) -> Vec<Vec<(usize, f32)>> {
        logits
            .chunks_exact(self.n_experts)
            .map(|logits| {
                let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let probs: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
                let sum: f32 = probs.iter().sum();
                let mut order: Vec<usize> = (0..self.n_experts).collect();
                order.select_nth_unstable_by(self.top_k - 1, |&a, &b| probs[b].total_cmp(&probs[a]));
                order.truncate(self.top_k);
                let kept: f32 = order.iter().map(|&e| probs[e]).sum();
                order.into_iter().map(|e| (e, probs[e] / sum / (kept / sum))).collect()
            })
            .collect()
    }

    /// Runs the tokens in `x`, one `hidden`-long row each, through the
    /// experts their `logits` route them to, adding the weighted results
    /// into `out`.
    pub fn forward(&self, x: &[f32], logits: &[f32], out: &mut [f32]) {
        let n = x.len() / self.hidden;
        assert_eq!(x.len(), n * self.hidden);
        assert_eq!(logits.len(), n * self.n_experts);
        assert_eq!(out.len(), x.len());
        let routes = self.route(logits);

        // Every token an expert serves, so its weights are read once.
        let mut hits: Vec<(usize, Vec<(usize, f32)>)> = Vec::new();
        for (t, route) in routes.iter().enumerate() {
            for &(e, w) in route {
                match hits.iter_mut().find(|(hit, _)| *hit == e) {
                    Some((_, tokens)) => tokens.push((t, w)),
                    None => hits.push((e, vec![(t, w)])),
                }
            }
        }

        let gate = self.pack.bytes(&self.gate).expect("gate weights");
        let up = self.pack.bytes(&self.up).expect("up weights");
        let down = self.pack.bytes(&self.down).expect("down weights");
        let in_row = q4::row_bytes(self.hidden);
        let mid_row = q4::row_bytes(self.intermediate);
        let expert_in = self.intermediate * in_row;
        let expert_mid = self.hidden * mid_row;

        // The hidden activations of every (expert, token): silu(gate x) * up x,
        // in chunks of rows.
        let chunks = self.intermediate.div_ceil(CHUNK);
        let mid: Vec<Vec<f32>> = (0..hits.len() * chunks)
            .into_par_iter()
            .map(|task| {
                let (e, tokens) = &hits[task / chunks];
                let rows = (task % chunks) * CHUNK..((task % chunks + 1) * CHUNK).min(self.intermediate);
                let mut h = vec![0.0; tokens.len() * rows.len()];
                for r in rows.clone() {
                    let g = &gate[e * expert_in + r * in_row..][..in_row];
                    let u = &up[e * expert_in + r * in_row..][..in_row];
                    for (i, &(t, _)) in tokens.iter().enumerate() {
                        let x = &x[t * self.hidden..][..self.hidden];
                        let a = q4::dot(g, x);
                        h[i * rows.len() + r - rows.start] = a / (1.0 + (-a).exp()) * q4::dot(u, x);
                    }
                }
                h
            })
            .collect();
        let mid: Vec<Vec<f32>> = hits
            .iter()
            .enumerate()
            .map(|(hit, (_, tokens))| {
                let mut h = vec![0.0; tokens.len() * self.intermediate];
                for c in 0..chunks {
                    let rows = c * CHUNK..((c + 1) * CHUNK).min(self.intermediate);
                    let part = &mid[hit * chunks + c];
                    for i in 0..tokens.len() {
                        h[i * self.intermediate + rows.start..][..rows.len()]
                            .copy_from_slice(&part[i * rows.len()..][..rows.len()]);
                    }
                }
                h
            })
            .collect();

        // Down projection, in chunks of rows, then the weighted sum.
        let down_chunks = self.hidden.div_ceil(CHUNK);
        let outs: Vec<Vec<f32>> = (0..hits.len() * down_chunks)
            .into_par_iter()
            .map(|task| {
                let hit = task / down_chunks;
                let (e, tokens) = &hits[hit];
                let rows = (task % down_chunks) * CHUNK..((task % down_chunks + 1) * CHUNK).min(self.hidden);
                let mut y = vec![0.0; tokens.len() * rows.len()];
                for r in rows.clone() {
                    let d = &down[e * expert_mid + r * mid_row..][..mid_row];
                    for i in 0..tokens.len() {
                        let h = &mid[hit][i * self.intermediate..][..self.intermediate];
                        y[i * rows.len() + r - rows.start] = q4::dot(d, h);
                    }
                }
                y
            })
            .collect();
        for (hit, (_, tokens)) in hits.iter().enumerate() {
            for c in 0..down_chunks {
                let rows = c * CHUNK..((c + 1) * CHUNK).min(self.hidden);
                let part = &outs[hit * down_chunks + c];
                for (i, &(t, w)) in tokens.iter().enumerate() {
                    let out = &mut out[t * self.hidden + rows.start..][..rows.len()];
                    for (o, &y) in out.iter_mut().zip(&part[i * rows.len()..][..rows.len()]) {
                        *o += w * y;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pack::Writer;

    fn values(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn matches_a_plain_evaluation() {
        let (n_experts, hidden, inter, top_k) = (8, 64, 32, 3);
        let dir = std::env::temp_dir().join(format!("hack-experts-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.hack");
        let layout = vec![
            ("e.gate_proj".to_string(), Dtype::Q4, vec![n_experts, inter, hidden]),
            ("e.up_proj".to_string(), Dtype::Q4, vec![n_experts, inter, hidden]),
            ("e.down_proj".to_string(), Dtype::Q4, vec![n_experts, hidden, inter]),
        ];
        let writer = Writer::create(&path, &layout).unwrap();
        // Quantized, then dequantized, so the reference sees the same weights.
        let mut weights = Vec::new();
        for (i, (name, rows, cols)) in [("e.gate_proj", inter, hidden), ("e.up_proj", inter, hidden), ("e.down_proj", hidden, inter)].into_iter().enumerate() {
            let mut all = Vec::new();
            for r in 0..n_experts * rows {
                let w = values(cols, 100 + (i * 1000 + r) as u64);
                let mut q = vec![0; q4::row_bytes(cols)];
                q4::quantize_row(&w, &mut q);
                writer.write(name, (r * q4::row_bytes(cols)) as u64, &q).unwrap();
                let mut back = vec![0.0; cols];
                q4::dequantize_row(&q, &mut back);
                all.extend(back);
            }
            weights.push(all);
        }
        drop(writer);

        let pack = Arc::new(Pack::open(&path).unwrap());
        let experts = Experts::new(pack, "e", top_k).unwrap();
        let n = 5;
        let x = values(n * hidden, 7);
        let logits = values(n * n_experts, 8);
        let mut out = vec![0.0; n * hidden];
        experts.forward(&x, &logits, &mut out);

        for t in 0..n {
            let mut want = vec![0.0; hidden];
            for (e, w) in &experts.route(&logits)[t] {
                let x = &x[t * hidden..][..hidden];
                let mut h = vec![0.0; inter];
                for r in 0..inter {
                    let g: f32 = weights[0][(e * inter + r) * hidden..][..hidden].iter().zip(x).map(|(a, b)| a * b).sum();
                    let u: f32 = weights[1][(e * inter + r) * hidden..][..hidden].iter().zip(x).map(|(a, b)| a * b).sum();
                    h[r] = g / (1.0 + (-g).exp()) * u;
                }
                for r in 0..hidden {
                    let d: f32 = weights[2][(e * hidden + r) * inter..][..inter].iter().zip(&h).map(|(a, b)| a * b).sum();
                    want[r] += w * d;
                }
            }
            for (a, b) in out[t * hidden..][..hidden].iter().zip(&want) {
                assert!((a - b).abs() < 1e-3, "token {t}: {a} vs {b}");
            }
        }
        let route = &experts.route(&logits)[0];
        assert_eq!(route.len(), top_k);
        assert!((route.iter().map(|(_, w)| w).sum::<f32>() - 1.0).abs() < 1e-5);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
