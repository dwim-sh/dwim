//! Speculative decoding: the target with a draft ahead of it. A cycle has
//! the draft propose the tokens after the last one settled, runs them all
//! through the target in one pass, and keeps the ones the target would
//! have picked itself, so that a pass yields several tokens where it
//! would yield one. What comes out is what the target alone would say:
//! the likeliest token at every step at a temperature of zero, and
//! otherwise tokens drawn from the target's own distribution, by the
//! rejection rule of speculative sampling.

use crate::{
    Device, Drafting, LanguageModel, Policy, Result,
    bonsai::{BATCH, Model},
    draft::{BLOCK, Draft, PROPOSALS},
};

/// The target with its draft.
pub struct Speculator<D: Device> {
    target: Model<D>,
    draft: Draft<D>,
    /// The embeddings of a block: the anchor and the mask tokens.
    block: D::Buffer,
    /// Rows of the last verify pass whose context the draft has not been
    /// given yet: their first position and how many, given once the
    /// conversation has settled on them.
    pending: Option<(usize, usize)>,
    stats: Stats,
}

type Stats = Drafting;

/// The first bytes of a saved state with a draft's context, and its
/// layout's version.
const STATE_MAGIC: &[u8; 8] = b"dwimspec";
const STATE_VERSION: u32 = 1;

impl<D: Device> Speculator<D> {
    /// Puts the draft ahead of the target, which must have been loaded on
    /// the same device, and starts capturing what the draft reads.
    pub fn new(mut target: Model<D>, draft: Draft<D>) -> Result<Self> {
        let c = &draft.config;
        if c.hidden != target.config.hidden || c.vocab != target.config.vocab {
            return Err("the draft is not for this model".into());
        }
        if c.target_layers.iter().any(|&l| l >= target.config.layers) {
            return Err("the draft reads layers the model does not have".into());
        }
        target.capture(&c.target_layers);
        let block = target.device.alloc(BLOCK * c.hidden);
        Ok(Self {
            target,
            draft,
            block,
            pending: None,
            stats: Stats::default(),
        })
    }

    /// The target, to look at.
    pub fn target(&self) -> &Model<D> {
        &self.target
    }

    /// Gives the draft the context of the rows settled before `upto`, of
    /// the last verify pass's.
    fn commit_pending(&mut self, upto: usize) {
        if let Some((start, rows)) = self.pending.take() {
            let rows = rows.min(upto.saturating_sub(start));
            if rows > 0 {
                let captures = self.target.captures().expect("capturing");
                self.draft
                    .commit(&self.target.device, captures, BATCH, rows, start);
            }
        }
    }
}

impl<D: Device> LanguageModel for Speculator<D> {
    /// The target's state, then the draft's context.
    fn save(&self, len: usize) -> Option<Vec<u8>> {
        let target = self.target.save(len)?;
        let mut out = Vec::new();
        out.extend_from_slice(STATE_MAGIC);
        out.extend_from_slice(&STATE_VERSION.to_le_bytes());
        out.extend_from_slice(&(target.len() as u64).to_le_bytes());
        out.extend(target);
        out.extend(self.draft.save(&self.target.device, len));
        Some(out)
    }

    fn restore(&mut self, state: &[u8]) -> Result<usize> {
        if state.len() < 20 || &state[..8] != STATE_MAGIC {
            return Err("not a saved state with a draft's context".into());
        }
        if u32::from_le_bytes(state[8..12].try_into()?) != STATE_VERSION {
            return Err("a saved state of another version".into());
        }
        let target_len = u64::from_le_bytes(state[12..20].try_into()?) as usize;
        let target = state
            .get(20..20 + target_len)
            .ok_or("the saved state is cut short")?;
        let len = self.target.restore(target)?;
        self.draft.reset();
        self.draft
            .restore(&self.target.device, len, &state[20 + target_len..])?;
        self.pending = None;
        Ok(len)
    }

    /// Runs the tokens through the target, and gives the draft their
    /// context as it goes.
    fn forward(&mut self, tokens: &[u32], pos: usize) -> Vec<f32> {
        self.commit_pending(pos);
        let mut logits = Vec::new();
        for (i, batch) in tokens.chunks(BATCH).enumerate() {
            let at = pos + i * BATCH;
            logits = self.target.forward(batch, at);
            let captures = self.target.captures().expect("capturing");
            self.draft
                .commit(&self.target.device, captures, BATCH, batch.len(), at);
        }
        logits
    }

    /// A cycle: the draft proposes the tokens after the anchor, the target
    /// verifies the block in one pass, and the proposals are accepted as
    /// far as the target agrees, with the target's own pick after them.
    fn next_tokens(&mut self, anchor: u32, pos: usize, policy: &mut Policy) -> Vec<u32> {
        self.commit_pending(pos);
        let c = &self.draft.config;
        let (hidden, vocab, mask) = (c.hidden, c.vocab, c.mask_token);
        let temperature = policy.sampler.temperature;
        let mut tokens = vec![anchor];
        tokens.resize(BLOCK, mask);
        let start = std::time::Instant::now();
        self.target.embed(&tokens, &mut self.block);
        let proposal = self.draft.propose(
            &self.target.device,
            &self.block,
            pos,
            anchor,
            self.target.head(),
            self.target.signs(hidden),
            temperature,
        );
        self.stats.draft_seconds += start.elapsed().as_secs_f64();
        let start = std::time::Instant::now();
        let mut rows = vec![anchor];
        rows.extend_from_slice(&proposal.tokens);
        let logits = self.target.verify(&rows, pos);
        self.stats.target_seconds += start.elapsed().as_secs_f64();

        // Each proposal in turn against what the target picks after the
        // row before it: at a temperature of zero, kept if it is the
        // likeliest; otherwise kept with the probability the rejection
        // rule gives, and replaced by a draw from the target's
        // distribution less the draft's where it is not.
        let mut out = Vec::with_capacity(BLOCK);
        let mut accepted = 0;
        for (r, &token) in proposal.tokens.iter().enumerate() {
            let row = &logits[r * vocab..][..vocab];
            let p = policy.sampler.distribution(row, policy.forbidden);
            let q = &proposal.distributions[r];
            let of = |dist: &[(u32, f32)], t: u32| {
                dist.iter().find(|(x, _)| *x == t).map_or(0.0, |(_, v)| *v)
            };
            let kept = if temperature == 0.0 {
                p[0].0 == token
            } else {
                policy.sampler.uniform() * of(q, token) < of(&p, token)
            };
            if kept {
                out.push(token);
                accepted += 1;
                continue;
            }
            let mut residual: Vec<(u32, f32)> = p
                .iter()
                .map(|&(t, pt)| (t, (pt - of(q, t)).max(0.0)))
                .filter(|&(_, v)| v > 0.0)
                .collect();
            let sum: f32 = residual.iter().map(|&(_, v)| v).sum();
            for (_, v) in &mut residual {
                *v /= sum;
            }
            let drawn = if residual.is_empty() {
                policy.sampler.draw(&p)
            } else {
                policy.sampler.draw(&residual)
            };
            policy.logits = Some(row.to_vec());
            out.push(drawn);
            break;
        }
        if accepted == PROPOSALS {
            let row = &logits[PROPOSALS * vocab..][..vocab];
            out.push(policy.pick(row));
        }
        // The target keeps the anchor and the accepted proposals; the
        // draft gets their context once the conversation settles on them.
        if accepted < PROPOSALS {
            self.target
                .rewind_to(pos + 1 + accepted)
                .expect("the pass just ran");
        }
        self.pending = Some((pos, 1 + accepted));
        self.stats.cycles += 1;
        self.stats.proposed += PROPOSALS;
        self.stats.accepted += accepted;
        out
    }

    fn rewind(&mut self, len: usize) -> Result<()> {
        self.target.rewind_to(len)?;
        if let Some((start, rows)) = &mut self.pending {
            *rows = (*rows).min(len.saturating_sub(*start));
        }
        self.draft.rewind(len);
        Ok(())
    }

    fn reset(&mut self) {
        self.target.reset();
        self.draft.reset();
        self.pending = None;
    }

    fn max_len(&self) -> usize {
        self.target.max_len()
    }

    fn drafting(&self) -> Option<Drafting> {
        Some(self.stats)
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::Arc;

    use dwim_gpu::{Cpu, Gpu};

    use super::*;
    use crate::{Gguf, Sampler, bonsai, draft, gguf, safetensors};

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        }

        fn bf16(&mut self, n: usize, scale: f32, offset: f32) -> Vec<u16> {
            (0..n)
                .map(|_| ((self.next() * scale + offset).to_bits() >> 16) as u16)
                .collect()
        }
    }

    /// Writes a tiny random draft for the synthetic model into `dir`: two
    /// layers reading the target's layers 1 and 3, heads of 64, a window
    /// of 32 positions, over the target's vocabulary of 64 with the last
    /// token as the mask.
    fn synthetic_draft(dir: &Path) {
        let (hidden, inter, layers, heads, kv_heads, head_dim, rank, vocab) =
            (1024, 512, 2, 4, 2, 64, 32, 64);
        let mut rng = Rng(9);
        let mut tensors: Vec<(String, Vec<usize>, Vec<u16>)> = Vec::new();
        let mut add = |name: &str, shape: Vec<usize>, scale: f32, offset: f32| {
            let n: usize = shape.iter().product();
            tensors.push((name.to_string(), shape, rng.bf16(n, scale, offset)));
        };
        let w = |fan_in: usize| 1.0 / (fan_in as f32).sqrt();
        for i in 0..layers {
            let p = format!("layers.{i}.");
            let a = format!("{p}self_attn.");
            add(
                &format!("{p}input_layernorm.weight"),
                vec![hidden],
                0.1,
                1.0,
            );
            add(
                &format!("{p}post_attention_layernorm.weight"),
                vec![hidden],
                0.1,
                1.0,
            );
            for conv in ["attention_conv", "mlp_conv"] {
                add(
                    &format!("{p}{conv}.base_kernel"),
                    vec![2, 2, hidden],
                    0.1,
                    0.5,
                );
                add(
                    &format!("{p}{conv}.kernel_projection.weight"),
                    vec![4 * hidden / 16, hidden],
                    0.1 * w(hidden),
                    0.0,
                );
            }
            add(
                &format!("{a}q_proj.weight"),
                vec![heads * head_dim, hidden],
                w(hidden),
                0.0,
            );
            add(
                &format!("{a}k_proj.weight"),
                vec![kv_heads * head_dim, hidden],
                w(hidden),
                0.0,
            );
            add(
                &format!("{a}v_proj.weight"),
                vec![kv_heads * head_dim, hidden],
                w(hidden),
                0.0,
            );
            add(
                &format!("{a}o_proj.weight"),
                vec![hidden, heads * head_dim],
                w(heads * head_dim),
                0.0,
            );
            add(&format!("{a}q_norm.weight"), vec![head_dim], 0.1, 1.0);
            add(&format!("{a}k_norm.weight"), vec![head_dim], 0.1, 1.0);
            add(
                &format!("{p}mlp.gate_proj.weight"),
                vec![inter, hidden],
                w(hidden),
                0.0,
            );
            add(
                &format!("{p}mlp.up_proj.weight"),
                vec![inter, hidden],
                w(hidden),
                0.0,
            );
            add(
                &format!("{p}mlp.down_proj.weight"),
                vec![hidden, inter],
                w(inter),
                0.0,
            );
        }
        add("fc.weight", vec![hidden, 2 * hidden], w(2 * hidden), 0.0);
        add("hidden_norm.weight", vec![hidden], 0.1, 1.0);
        add("norm.weight", vec![hidden], 0.1, 1.0);
        add(
            "candidate_selector.hidden_projection.weight",
            vec![rank, hidden],
            w(hidden),
            0.0,
        );
        add(
            "candidate_selector.predecessor_codebook",
            vec![vocab, rank],
            0.3,
            0.0,
        );
        add(
            "candidate_selector.successor_codebook",
            vec![vocab, rank],
            0.3,
            0.0,
        );
        safetensors::write(&draft::weights_path(dir), &tensors).unwrap();
        let config = serde_json::json!({
            "architectures": ["DFlash2DraftModel"],
            "is_causal": false,
            "dflash_config": {
                "block_size": 8,
                "conv_group_size": 16,
                "conv_kernel_size": 2,
                "mask_token_id": vocab - 1,
                "selector_rank": rank,
                "selector_top_k": 16,
                "target_layer_ids": [1, 3]
            },
            "head_dim": head_dim,
            "hidden_size": hidden,
            "intermediate_size": inter,
            "num_attention_heads": heads,
            "num_hidden_layers": layers,
            "num_key_value_heads": kv_heads,
            "rms_norm_eps": 1e-6,
            "rope_parameters": { "rope_theta": 10000.0 },
            "sliding_window": 32,
            "vocab_size": vocab
        });
        std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    }

    /// A fresh synthetic target and draft on `device`.
    fn speculator<D: Device>(dir: &Path, device: D) -> Speculator<D> {
        let gguf = Arc::new(Gguf::open(&dir.join("model.gguf")).unwrap());
        let target = bonsai::Model::load(gguf, device, 128, |_, _| {}).unwrap();
        let draft = Draft::load(dir, &target.device, 128, |_, _| {}).unwrap();
        Speculator::new(target, draft).unwrap()
    }

    /// Generates `n` tokens greedily after `prompt`, a token at a time or
    /// by the model's own stride, returning them and the cycles it took.
    fn greedy<M: LanguageModel>(model: &mut M, prompt: &[u32], n: usize) -> Vec<u32> {
        let mut sampler = Sampler::new(0.0, 1, 1.0, 0.0, 1);
        let logits = model.forward(prompt, 0);
        let mut policy = Policy {
            sampler: &mut sampler,
            forbidden: &[],
            logits: None,
        };
        let mut out = vec![policy.pick(&logits)];
        let mut pos = prompt.len();
        while out.len() < n {
            let anchor = *out.last().unwrap();
            let picked = model.next_tokens(anchor, pos, &mut policy);
            pos += picked.len();
            out.extend(picked);
        }
        out.truncate(n);
        out
    }

    #[test]
    fn drafting_gives_the_targets_greedy_tokens() {
        let dir = std::env::temp_dir().join(format!("dwim-spec-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        bonsai::tests::synthetic(&dir.join("model.gguf"), gguf::PTQ1_0);
        synthetic_draft(&dir);
        // A prompt longer than a batch, so that its context is committed in
        // batches of many rows and then a few.
        let prompt: Vec<u32> = (0..70).map(|i| (i * 7 % 61) as u32).collect();
        // The target alone, a token at a time.
        let gguf = Arc::new(Gguf::open(&dir.join("model.gguf")).unwrap());
        let mut plain = bonsai::Model::load(gguf, Cpu, 128, |_, _| {}).unwrap();
        let want = greedy(&mut plain, &prompt, 40);
        // With the draft: the same tokens, however many a cycle accepts.
        let mut spec = speculator(&dir, Cpu);
        let got = greedy(&mut spec, &prompt, 40);
        assert_eq!(got, want);
        let stats = spec.drafting().unwrap();
        assert!(stats.cycles > 0 && stats.cycles < 40, "{stats:?}");
        eprintln!(
            "accepted {} of {} proposals in {} cycles",
            stats.accepted, stats.proposed, stats.cycles
        );
        // Rewinding to any point of a cycle leaves the model as if the
        // tokens after it were never run.
        let mut spec = speculator(&dir, Cpu);
        let mut sampler = Sampler::new(0.0, 1, 1.0, 0.0, 1);
        let logits = spec.forward(&prompt, 0);
        let mut policy = Policy {
            sampler: &mut sampler,
            forbidden: &[],
            logits: None,
        };
        let first = policy.pick(&logits);
        assert_eq!(first, want[0]);
        let picked = spec.next_tokens(first, prompt.len(), &mut policy);
        assert_eq!(picked, want[1..1 + picked.len()]);
        let len = prompt.len() + 1;
        spec.rewind(len).unwrap();
        let again = spec.next_tokens(want[1], len, &mut policy);
        assert_eq!(again, want[2..2 + again.len()]);
        // A saved state restores to the same continuation.
        let state = spec.save(len + 1).unwrap();
        let continued = spec.next_tokens(want[2], len + 1, &mut policy);
        let mut restored = speculator(&dir, Cpu);
        assert_eq!(restored.restore(&state).unwrap(), len + 1);
        assert_eq!(
            restored.next_tokens(want[2], len + 1, &mut policy),
            continued
        );
        // The GPU agrees with the CPU.
        if let Ok(gpu) = Gpu::new() {
            let mut spec = speculator(&dir, gpu);
            assert_eq!(greedy(&mut spec, &prompt, 40), want);
        } else {
            eprintln!("skipping the GPU comparison: no GPU");
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
