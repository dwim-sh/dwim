//! The DFlash2 draft: a small transformer trained to guess the tokens the
//! target says next, a block of them in one pass, from the target's own
//! hidden states, so that the target can check them all in one pass of its
//! own (`speculative.rs`).
//!
//! The draft reads the target's residual stream after a few of its layers,
//! at every position: those rows, concatenated, projected by `fc` and
//! normalized, are its context, from which each of its layers takes keys
//! and values. A cycle runs a block of eight rows through it: the anchor,
//! the last token the target settled, as its embedding, and seven mask
//! tokens, at the positions after it; each row attends to the context of
//! the last 2048 positions and to every row of the block. Before and after
//! each attention and MLP, a two-tap convolution mixes each row with the
//! row before it, with weights the row's own activations adjust. The
//! target's head gives each row's logits, and a selector walks a path
//! through the sixteen likeliest tokens of each row after the anchor,
//! scoring each candidate by its logit plus a bilinear match with the
//! token chosen before it, through two codebooks and a gate from the row's
//! activations: the seven tokens of that path are the proposals. The
//! architecture is that of z-lab/dflash and of Splash's `DFlashDraft`.

use std::{
    fs,
    path::{Path, PathBuf},
};

use crate::{Device, Result, Safetensors, Tensor, rope_table};

/// Rows of a block: the anchor and the proposals after it.
pub const BLOCK: usize = 8;

/// Proposals a cycle makes.
pub const PROPOSALS: usize = BLOCK - 1;

/// Candidates the selector keeps at each position.
pub const CANDIDATES: usize = 16;

/// The architecture, read from the checkpoint's `config.json`.
#[derive(Clone, Debug)]
pub struct Config {
    pub layers: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rope_theta: f32,
    pub eps: f32,
    /// Positions of context a row attends to, counting its own.
    pub window: usize,
    /// The target's layers whose outputs the draft reads, ascending.
    pub target_layers: Vec<usize>,
    pub mask_token: u32,
    /// Channels that share one dynamic convolution weight.
    pub group: usize,
    pub selector_rank: usize,
    pub vocab: usize,
}

impl Config {
    /// Reads the configuration beside the weights.
    pub fn load(dir: &Path) -> Result<Self> {
        let text = fs::read_to_string(dir.join("config.json"))?;
        let json: serde_json::Value = serde_json::from_str(&text)?;
        if json["architectures"][0].as_str() != Some("DFlash2DraftModel") {
            return Err("the draft is not a DFlash2 draft".into());
        }
        let number = |key: &str| -> Result<usize> {
            json[key]
                .as_u64()
                .map(|v| v as usize)
                .ok_or_else(|| format!("the draft's config has no {key}").into())
        };
        let dflash = &json["dflash_config"];
        let inner = |key: &str| -> Result<usize> {
            dflash[key]
                .as_u64()
                .map(|v| v as usize)
                .ok_or_else(|| format!("the draft's config has no dflash_config.{key}").into())
        };
        let config = Self {
            layers: number("num_hidden_layers")?,
            hidden: number("hidden_size")?,
            intermediate: number("intermediate_size")?,
            heads: number("num_attention_heads")?,
            kv_heads: number("num_key_value_heads")?,
            head_dim: number("head_dim")?,
            rope_theta: json["rope_parameters"]["rope_theta"]
                .as_f64()
                .or_else(|| json["rope_theta"].as_f64())
                .ok_or("the draft's config has no rope theta")? as f32,
            eps: json["rms_norm_eps"]
                .as_f64()
                .ok_or("the draft's config has no rms_norm_eps")? as f32,
            window: number("sliding_window")?,
            target_layers: dflash["target_layer_ids"]
                .as_array()
                .ok_or("the draft's config has no target layers")?
                .iter()
                .map(|v| v.as_u64().map(|v| v as usize))
                .collect::<Option<Vec<_>>>()
                .ok_or("the draft's target layers are not integers")?,
            mask_token: inner("mask_token_id")? as u32,
            group: inner("conv_group_size")?,
            selector_rank: inner("selector_rank")?,
            vocab: number("vocab_size")?,
        };
        if inner("block_size")? < BLOCK || inner("selector_top_k")? != CANDIDATES {
            return Err("the draft's block or selector is not the size the engine runs".into());
        }
        if inner("conv_kernel_size")? != 2 {
            return Err("the draft's convolution is not two taps wide".into());
        }
        if json["is_causal"].as_bool() != Some(false) {
            return Err("the draft's block is not bidirectional".into());
        }
        if config.target_layers.is_empty()
            || config.target_layers.windows(2).any(|w| w[0] >= w[1])
            || !config.hidden.is_multiple_of(config.group)
            || !config.heads.is_multiple_of(config.kv_heads)
        {
            return Err("the draft's shape is not one the engine runs".into());
        }
        Ok(config)
    }

    /// Width of the dynamic convolution weights per row: two stages of
    /// two taps of a weight per group.
    fn dynamic(&self) -> usize {
        4 * self.hidden / self.group
    }

    fn kv_dim(&self) -> usize {
        self.kv_heads * self.head_dim
    }

    fn q_dim(&self) -> usize {
        self.heads * self.head_dim
    }
}

/// Device memory the draft takes, in bytes: `fixed` for its weights and
/// activations, and `per_token` more for each token of context, for the
/// rotary position embedding table.
#[derive(Clone, Copy, Debug)]
pub struct Memory {
    pub fixed: u64,
    pub per_token: u64,
}

impl Memory {
    /// The memory the draft with `config` takes, its projections quantized
    /// four bits a weight.
    pub fn of(config: &Config) -> Self {
        let c = config;
        let layer = c.dynamic() * c.hidden * 2
            + (c.q_dim() + 2 * c.kv_dim()) * c.hidden
            + c.q_dim() * c.hidden
            + 3 * c.intermediate * c.hidden;
        let weights = c.layers * layer
            + c.hidden * c.hidden * c.target_layers.len()
            + c.selector_rank * c.hidden;
        let rings = c.layers * 2 * c.kv_heads * c.window * c.head_dim + c.window * c.hidden;
        let activations = BLOCK * (4 * c.hidden + c.dynamic() + c.q_dim() + 2 * c.kv_dim())
            + BLOCK * 2 * c.intermediate
            + BLOCK * c.vocab
            + crate::bonsai::BATCH * (c.hidden + 2 * c.kv_dim());
        // The projections are quantized to 4.5 bits a weight.
        Self {
            fixed: (weights * 9 / 16 + 4 * (rings + activations)) as u64,
            per_token: (4 * c.head_dim) as u64,
        }
    }
}

struct Layer<D: Device> {
    input_norm: D::Buffer,
    attn_base: D::Buffer,
    attn_dynamic: D::Weight,
    q: D::Weight,
    k: D::Weight,
    v: D::Weight,
    q_norm: D::Buffer,
    k_norm: D::Buffer,
    o: D::Weight,
    post_norm: D::Buffer,
    mlp_base: D::Buffer,
    mlp_dynamic: D::Weight,
    gate: D::Weight,
    up: D::Weight,
    down: D::Weight,
    /// The context's keys and values, in a ring of `window` slots a head.
    k_ring: D::Buffer,
    v_ring: D::Buffer,
}

/// The draft, with its weights on a device and its context so far.
pub struct Draft<D: Device> {
    pub config: Config,
    layers: Vec<Layer<D>>,
    /// The context projection, a matrix per target layer read.
    fc: Vec<D::Weight>,
    hidden_norm: D::Buffer,
    norm: D::Buffer,
    selector: D::Weight,
    /// The selector's codebooks, on the host: a cycle reads a few rows.
    predecessor: Vec<u16>,
    successor: Vec<u16>,
    rope: D::Buffer,
    /// The context rows, normalized, in a ring of `window` slots: what the
    /// keys and values are computed from, kept for saving the state.
    context: D::Buffer,
    /// Positions of context committed so far.
    committed: usize,
    /// Activations for a block.
    h: D::Buffer,
    h2: D::Buffer,
    xn: D::Buffer,
    xc: D::Buffer,
    dynamic: D::Buffer,
    q: D::Buffer,
    k: D::Buffer,
    v: D::Buffer,
    att: D::Buffer,
    o: D::Buffer,
    up: D::Buffer,
    gate: D::Buffer,
    sel: D::Buffer,
    logits: D::Buffer,
    /// Activations for a batch of context rows.
    ctx: D::Buffer,
    ctx_kv: D::Buffer,
    max_len: usize,
}

/// What a block gives: the proposals, and for each the candidates it was
/// picked from with their probabilities, which acceptance weighs the
/// target's against.
pub struct Proposal {
    pub tokens: Vec<u32>,
    pub distributions: Vec<crate::Distribution>,
}

impl<D: Device> Draft<D> {
    /// Loads the draft from the checkpoint in `dir`, uploading its
    /// weights to `device`, with context for sequences of up to `max_len`
    /// tokens, and reporting how many tensors are loaded, out of how many,
    /// as it goes.
    pub fn load(
        dir: &Path,
        device: &D,
        max_len: usize,
        mut on_progress: impl FnMut(usize, usize),
    ) -> Result<Self> {
        let config = Config::load(dir)?;
        let c = &config;
        let st = Safetensors::open(&weights_path(dir))?;
        let total = c.layers * 15 + 4 + 2 * c.target_layers.len();
        let mut done = 0;
        let mut tick = || {
            done += 1;
            on_progress(done, total);
        };
        let mut layers = Vec::with_capacity(c.layers);
        for i in 0..c.layers {
            let p = format!("layers.{i}.");
            let a = format!("{p}self_attn.");
            layers.push(Layer {
                input_norm: buffer(
                    &st,
                    device,
                    &format!("{p}input_layernorm.weight"),
                    &mut tick,
                )?,
                attn_base: buffer(
                    &st,
                    device,
                    &format!("{p}attention_conv.base_kernel"),
                    &mut tick,
                )?,
                q_norm: buffer(&st, device, &format!("{a}q_norm.weight"), &mut tick)?,
                k_norm: buffer(&st, device, &format!("{a}k_norm.weight"), &mut tick)?,
                post_norm: buffer(
                    &st,
                    device,
                    &format!("{p}post_attention_layernorm.weight"),
                    &mut tick,
                )?,
                mlp_base: buffer(&st, device, &format!("{p}mlp_conv.base_kernel"), &mut tick)?,
                attn_dynamic: upload(
                    &st,
                    device,
                    &format!("{p}attention_conv.kernel_projection.weight"),
                    c.dynamic(),
                    c.hidden,
                    &mut tick,
                )?,
                q: upload(
                    &st,
                    device,
                    &format!("{a}q_proj.weight"),
                    c.q_dim(),
                    c.hidden,
                    &mut tick,
                )?,
                k: upload(
                    &st,
                    device,
                    &format!("{a}k_proj.weight"),
                    c.kv_dim(),
                    c.hidden,
                    &mut tick,
                )?,
                v: upload(
                    &st,
                    device,
                    &format!("{a}v_proj.weight"),
                    c.kv_dim(),
                    c.hidden,
                    &mut tick,
                )?,
                o: upload(
                    &st,
                    device,
                    &format!("{a}o_proj.weight"),
                    c.hidden,
                    c.q_dim(),
                    &mut tick,
                )?,
                mlp_dynamic: upload(
                    &st,
                    device,
                    &format!("{p}mlp_conv.kernel_projection.weight"),
                    c.dynamic(),
                    c.hidden,
                    &mut tick,
                )?,
                gate: upload(
                    &st,
                    device,
                    &format!("{p}mlp.gate_proj.weight"),
                    c.intermediate,
                    c.hidden,
                    &mut tick,
                )?,
                up: upload(
                    &st,
                    device,
                    &format!("{p}mlp.up_proj.weight"),
                    c.intermediate,
                    c.hidden,
                    &mut tick,
                )?,
                down: upload(
                    &st,
                    device,
                    &format!("{p}mlp.down_proj.weight"),
                    c.hidden,
                    c.intermediate,
                    &mut tick,
                )?,
                k_ring: device.alloc(c.kv_heads * c.window * c.head_dim),
                v_ring: device.alloc(c.kv_heads * c.window * c.head_dim),
            });
        }
        // The context projection reads the target layers' rows side by
        // side: a matrix per layer, applied to each layer's rows and summed.
        let fc = match st.bf16("fc.weight")? {
            Tensor::Bf16 { shape, data } => {
                let m = c.target_layers.len();
                if shape != [c.hidden, m * c.hidden] {
                    return Err("the draft's context projection is the wrong shape".into());
                }
                (0..m)
                    .map(|j| {
                        let mut part = Vec::with_capacity(c.hidden * c.hidden);
                        for row in data.chunks_exact(m * c.hidden) {
                            part.extend_from_slice(&row[j * c.hidden..][..c.hidden]);
                        }
                        tick();
                        device.upload(
                            Tensor::Bf16 {
                                shape: vec![c.hidden, c.hidden],
                                data: part,
                            }
                            .affine4(),
                        )
                    })
                    .collect()
            }
            _ => unreachable!(),
        };
        let hidden_norm = buffer(&st, device, "hidden_norm.weight", &mut tick)?;
        let norm = buffer(&st, device, "norm.weight", &mut tick)?;
        let selector = upload(
            &st,
            device,
            "candidate_selector.hidden_projection.weight",
            c.selector_rank,
            c.hidden,
            &mut tick,
        )?;
        let codebook = |name: &str| -> Result<Vec<u16>> {
            let info = st.info(name)?;
            if info.shape != [c.vocab, c.selector_rank] {
                return Err(format!("the draft's {name} is the wrong shape").into());
            }
            st.bf16_bits(name)
        };
        let predecessor = codebook("candidate_selector.predecessor_codebook")?;
        tick();
        let successor = codebook("candidate_selector.successor_codebook")?;
        tick();
        let table = rope_table(max_len, c.head_dim, c.rope_theta);
        let mut rope = device.alloc(table.len());
        device.write(&mut rope, &table);
        let batch = crate::bonsai::BATCH;
        Ok(Self {
            layers,
            fc,
            hidden_norm,
            norm,
            selector,
            predecessor,
            successor,
            rope,
            context: device.alloc(c.window * c.hidden),
            committed: 0,
            h: device.alloc(BLOCK * c.hidden),
            h2: device.alloc(BLOCK * c.hidden),
            // Also the second projection of a batch of context rows.
            xn: device.alloc(batch.max(BLOCK) * c.hidden),
            xc: device.alloc(BLOCK * c.hidden),
            dynamic: device.alloc(BLOCK * c.dynamic()),
            q: device.alloc(BLOCK * c.q_dim()),
            k: device.alloc(BLOCK * c.kv_dim()),
            v: device.alloc(BLOCK * c.kv_dim()),
            att: device.alloc(BLOCK * c.q_dim()),
            o: device.alloc(BLOCK * c.hidden),
            up: device.alloc(BLOCK * c.intermediate),
            gate: device.alloc(BLOCK * c.intermediate),
            sel: device.alloc(BLOCK * c.selector_rank),
            logits: device.alloc(BLOCK * c.vocab),
            ctx: device.alloc(batch * c.hidden),
            ctx_kv: device.alloc(batch * c.kv_dim()),
            max_len,
            config,
        })
    }

    /// Positions of context the draft holds so far.
    pub fn committed(&self) -> usize {
        self.committed
    }

    /// Adds `rows` positions of context from `pos` on, which must follow
    /// the positions committed so far: `captured` holds, for each target
    /// layer the draft reads, the rows of its output, one layer's rows
    /// after another's, at a stride of `stride` rows per layer.
    pub fn commit(
        &mut self,
        device: &D,
        captured: &D::Buffer,
        stride: usize,
        rows: usize,
        pos: usize,
    ) {
        let (hidden, window, kv_dim, eps) = (
            self.config.hidden,
            self.config.window,
            self.config.kv_dim(),
            self.config.eps,
        );
        assert!(rows <= stride && pos + rows <= self.max_len);
        assert!(
            pos <= self.committed,
            "context committed at {pos} with a gap after {}",
            self.committed
        );
        let d = device;
        // The context rows: the layers' rows projected and summed, then
        // normalized.
        d.resize(&mut self.ctx, rows * hidden);
        d.resize(&mut self.xn, rows * hidden);
        d.resize(&mut self.ctx_kv, rows * kv_dim);
        for (j, fc) in self.fc.iter().enumerate() {
            let mut part = d.alloc(rows * hidden);
            d.copy(&mut part, 0, captured, j * stride * hidden, rows * hidden);
            if j == 0 {
                d.matmul(&mut self.ctx, fc, &part);
            } else {
                d.matmul(&mut self.xn, fc, &part);
                d.add(&mut self.ctx, &self.xn);
            }
        }
        d.rmsnorm(&mut self.ctx, &self.hidden_norm, eps);
        d.ring_store(&mut self.context, &self.ctx, pos, window, 1, hidden);
        self.keys_and_values(device, pos);
        self.committed = pos + rows;
    }

    /// Each layer's keys and values of the context rows in `ctx`, the
    /// first at position `pos`, into its rings.
    fn keys_and_values(&mut self, d: &D, pos: usize) {
        let c = self.config.clone();
        for layer in &mut self.layers {
            d.matmul(&mut self.ctx_kv, &layer.k, &self.ctx);
            d.rmsnorm(&mut self.ctx_kv, &layer.k_norm, c.eps);
            d.rope(
                &mut self.ctx_kv,
                &self.rope,
                pos,
                c.kv_heads,
                c.head_dim,
                c.head_dim,
            );
            d.ring_store(
                &mut layer.k_ring,
                &self.ctx_kv,
                pos,
                c.window,
                c.kv_heads,
                c.head_dim,
            );
            d.matmul(&mut self.ctx_kv, &layer.v, &self.ctx);
            d.ring_store(
                &mut layer.v_ring,
                &self.ctx_kv,
                pos,
                c.window,
                c.kv_heads,
                c.head_dim,
            );
        }
    }

    /// Forgets the context from position `len` on.
    pub fn rewind(&mut self, len: usize) {
        self.committed = self.committed.min(len);
    }

    /// Forgets the context.
    pub fn reset(&mut self) {
        self.committed = 0;
    }

    /// The context rows of the last positions committed, up to a window
    /// of them, as the bytes of their f32 values, for saving the state.
    pub fn save(&self, device: &D, len: usize) -> Vec<u8> {
        let c = &self.config;
        let len = len.min(self.committed);
        let rows = len.min(c.window);
        let all = device.read(&self.context);
        let mut out = Vec::with_capacity(rows * c.hidden * 4);
        for pos in len - rows..len {
            for v in &all[(pos % c.window) * c.hidden..][..c.hidden] {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        out
    }

    /// Takes up a context `save` gave of a sequence of `len` positions:
    /// the rows are the positions before `len`, as many as there are.
    pub fn restore(&mut self, device: &D, len: usize, bytes: &[u8]) -> Result<()> {
        let (hidden, window, kv_dim) =
            (self.config.hidden, self.config.window, self.config.kv_dim());
        let rows = bytes.len() / (hidden * 4);
        if bytes.len() != rows * hidden * 4 || rows > len || rows > window {
            return Err("a saved draft context of the wrong size".into());
        }
        let d = device;
        let batch = crate::bonsai::BATCH;
        let first = len - rows;
        for start in (first..len).step_by(batch) {
            let count = batch.min(len - start);
            let floats: Vec<f32> = bytes[(start - first) * hidden * 4..][..count * hidden * 4]
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            d.resize(&mut self.ctx, count * hidden);
            d.resize(&mut self.ctx_kv, count * kv_dim);
            d.write(&mut self.ctx, &floats);
            d.ring_store(&mut self.context, &self.ctx, start, window, 1, hidden);
            self.keys_and_values(device, start);
        }
        self.committed = len;
        Ok(())
    }

    /// Proposes the tokens after `anchor`, the token at position `pos`,
    /// which the context must be committed up to: `block` holds the
    /// embeddings of the anchor and of [`PROPOSALS`] mask tokens, a row
    /// each, `head` is the target's head, which takes rows rotated by
    /// `signs`, and `temperature` is the sampler's, zero for the likeliest
    /// path.
    #[allow(clippy::too_many_arguments)]
    pub fn propose(
        &mut self,
        device: &D,
        block: &D::Buffer,
        pos: usize,
        anchor: u32,
        head: &D::Weight,
        signs: &D::Buffer,
        temperature: f32,
    ) -> Proposal {
        assert_eq!(
            self.committed, pos,
            "the context must be committed up to the anchor"
        );
        let c = self.config.clone();
        let d = device;
        let n = BLOCK;
        d.resize(&mut self.xn, n * c.hidden);
        d.copy(&mut self.h, 0, block, 0, n * c.hidden);
        for layer in &self.layers {
            // Attention, with the convolution before and after it, and the
            // residual added as the convolution finishes.
            d.copy(&mut self.xn, 0, &self.h, 0, n * c.hidden);
            d.rmsnorm(&mut self.xn, &layer.input_norm, c.eps);
            d.matmul(&mut self.dynamic, &layer.attn_dynamic, &self.xn);
            d.draft_conv(
                &mut self.xc,
                &self.xn,
                &self.dynamic,
                &layer.attn_base,
                None,
                0,
                c.group,
            );
            d.matmul(&mut self.q, &layer.q, &self.xc);
            d.matmul(&mut self.k, &layer.k, &self.xc);
            d.matmul(&mut self.v, &layer.v, &self.xc);
            d.rmsnorm(&mut self.q, &layer.q_norm, c.eps);
            d.rmsnorm(&mut self.k, &layer.k_norm, c.eps);
            d.rope(
                &mut self.q,
                &self.rope,
                pos,
                c.heads,
                c.head_dim,
                c.head_dim,
            );
            d.rope(
                &mut self.k,
                &self.rope,
                pos,
                c.kv_heads,
                c.head_dim,
                c.head_dim,
            );
            d.draft_attention(
                &mut self.att,
                &self.q,
                &layer.k_ring,
                &layer.v_ring,
                &self.k,
                &self.v,
                pos,
                c.window,
                c.heads,
                c.head_dim,
                c.kv_heads,
            );
            d.matmul(&mut self.o, &layer.o, &self.att);
            d.draft_conv(
                &mut self.h2,
                &self.o,
                &self.dynamic,
                &layer.attn_base,
                Some(&self.h),
                1,
                c.group,
            );
            // The MLP, likewise.
            d.copy(&mut self.xn, 0, &self.h2, 0, n * c.hidden);
            d.rmsnorm(&mut self.xn, &layer.post_norm, c.eps);
            d.matmul(&mut self.dynamic, &layer.mlp_dynamic, &self.xn);
            d.draft_conv(
                &mut self.xc,
                &self.xn,
                &self.dynamic,
                &layer.mlp_base,
                None,
                0,
                c.group,
            );
            d.matmul(&mut self.gate, &layer.gate, &self.xc);
            d.matmul(&mut self.up, &layer.up, &self.xc);
            d.silu_mul(&mut self.gate, &self.up);
            d.matmul(&mut self.o, &layer.down, &self.gate);
            d.draft_conv(
                &mut self.h,
                &self.o,
                &self.dynamic,
                &layer.mlp_base,
                Some(&self.h2),
                1,
                c.group,
            );
        }
        d.copy(&mut self.xn, 0, &self.h, 0, n * c.hidden);
        d.rmsnorm(&mut self.xn, &self.norm, c.eps);
        d.matmul(&mut self.sel, &self.selector, &self.xn);
        d.hadamard(&mut self.xn, signs, false);
        d.matmul(&mut self.logits, head, &self.xn);
        let selector = d.read(&self.sel);
        let logits = d.read(&self.logits);
        self.select(anchor, &logits, &selector, temperature)
    }

    /// Walks the path of proposals: at each row after the anchor, among
    /// the sixteen likeliest tokens of its logits, the one whose logit
    /// plus its match with the token before it scores best, or one drawn
    /// in proportion at the temperature.
    fn select(&self, anchor: u32, logits: &[f32], selector: &[f32], temperature: f32) -> Proposal {
        let c = &self.config;
        let rank = c.selector_rank;
        let mut previous = anchor;
        let mut tokens = Vec::with_capacity(PROPOSALS);
        let mut distributions = Vec::with_capacity(PROPOSALS);
        for row in 1..BLOCK {
            let row_logits = &logits[row * c.vocab..][..c.vocab];
            let candidates = top(row_logits, CANDIDATES);
            let gate = &selector[row * rank..][..rank];
            let pred = &self.predecessor[previous as usize * rank..][..rank];
            // The gate and the predecessor's code, multiplied once for
            // every candidate.
            let context: Vec<f32> = pred
                .iter()
                .zip(gate)
                .map(|(&p, &g)| dwim_gpu::bf16(p) * g)
                .collect();
            let scores: Vec<f32> = candidates
                .iter()
                .map(|&(token, logit)| {
                    let succ = &self.successor[token as usize * rank..][..rank];
                    logit
                        + context
                            .iter()
                            .zip(succ)
                            .map(|(&x, &s)| x * dwim_gpu::bf16(s))
                            .sum::<f32>()
                })
                .collect();
            let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let distribution: crate::Distribution = if temperature == 0.0 {
                let best = scores.iter().position(|&s| s == max).unwrap();
                vec![(candidates[best].0, 1.0)]
            } else {
                let weights: Vec<f32> = scores
                    .iter()
                    .map(|&s| ((s - max) / temperature).exp())
                    .collect();
                let sum: f32 = weights.iter().sum();
                let mut dist: Vec<(u32, f32)> = candidates
                    .iter()
                    .zip(&weights)
                    .map(|(&(token, _), &w)| (token, w / sum))
                    .collect();
                dist.sort_by(|a, b| b.1.total_cmp(&a.1));
                dist
            };
            // The path follows the likeliest choice; a sampling cycle's
            // draw is made by acceptance, from the distribution.
            previous = distribution[0].0;
            tokens.push(previous);
            distributions.push(distribution);
        }
        Proposal {
            tokens,
            distributions,
        }
    }
}

/// The `k` largest logits, with their tokens, in descending order.
fn top(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut best: Vec<(u32, f32)> = Vec::with_capacity(k + 1);
    for (token, &logit) in logits.iter().enumerate() {
        if best.len() == k && logit <= best[k - 1].1 {
            continue;
        }
        let at = best.partition_point(|&(_, l)| l >= logit);
        best.insert(at, (token as u32, logit));
        best.truncate(k);
    }
    best
}

/// Writes a bf16 tensor to a buffer of f32 values.
fn buffer<D: Device>(
    st: &Safetensors,
    device: &D,
    name: &str,
    tick: &mut impl FnMut(),
) -> Result<D::Buffer> {
    let values = st.f32s(name)?;
    let mut buf = device.alloc(values.len());
    device.write(&mut buf, &values);
    tick();
    Ok(buf)
}

/// Uploads a bf16 matrix of the shape given.
fn upload<D: Device>(
    st: &Safetensors,
    device: &D,
    name: &str,
    rows: usize,
    cols: usize,
    tick: &mut impl FnMut(),
) -> Result<D::Weight> {
    let tensor = st.bf16(name)?;
    if tensor.shape() != [rows, cols] {
        return Err(format!(
            "the draft's {name} is {:?}, not [{rows}, {cols}]",
            tensor.shape()
        )
        .into());
    }
    tick();
    // Four bits a weight, as the draft was trained to stand, for a quarter
    // of the reading a cycle does.
    Ok(device.upload(tensor.affine4()))
}

/// The weights file in a checkpoint directory.
pub fn weights_path(dir: &Path) -> PathBuf {
    dir.join("model.safetensors")
}
