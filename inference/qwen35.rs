//! Qwen3.8-27B, the `qwen35` architecture of llama.cpp's files: a hybrid
//! transformer in which every fourth layer is full attention and the rest
//! are Gated DeltaNet linear attention, written as a sequence of operations
//! on a [`Device`]. It runs the model from a file quantized as llama.cpp
//! does, each matrix in whichever format the file has it.
//!
//! The file stores the value heads of linear attention in the order
//! llama.cpp's kernels want, in which value head `h` shares key head
//! `h % n_k`; they are put back into the checkpoint's order, in which value
//! head `h` shares key head `h / (n_v / n_k)`, as the loader reads them. The
//! file ends with a layer that predicts the token after the next, which the
//! model drafts with.

use std::{collections::HashMap, path::PathBuf, sync::Arc};

use crate::{
    gpu::{CACHE_BLOCK, CONV_KERNEL, Device, Tensor, cache_bytes, quant},
    inference::{Gguf, LanguageModel, Result, gguf, hub, quants, rope_table},
};

/// The model as `dwim` names it.
pub const NAME: &str = "qwen3.8-27b";

/// The file holding the model: a GGUF file, which carries the tokenizer
/// and the sampling settings too.
pub const WEIGHTS: &str = "Qwen3.8-27B-Q4_0.gguf";

/// Where the model's files come from: Qwen ships no GGUF of its own, so
/// bartowski's conversion of `Qwen/Qwen3.8-27B`, quantized with an
/// importance matrix, at a pinned commit, so the weights can't change
/// underneath us and outputs stay reproducible. Its Q4_0 file holds most
/// matrices in 4-bit blocks of 32 with an f16 scale, the first layers' down
/// projections as Q4_1, the linear attention's output projections as Q8_0,
/// and the output and a few query projections as Q6_K; the norms and the
/// small linear attention tensors are f32.
pub const REPO: &str = "bartowski/Qwen3.8-27B-GGUF";
pub const REVISION: &str = "87410e7b9a5c21e6ca70c0c9cc5237de909fe621";

/// A file of the model in its repository.
pub struct File {
    pub name: &'static str,
    /// What the Hugging Face cache names its content by: the SHA-256 of a
    /// file in LFS storage, and the git blob SHA-1 of a small one.
    pub digest: &'static str,
}

/// The files fetched: the weights.
pub const FILES: [File; 1] = [File {
    name: WEIGHTS,
    digest: "5eb1de39e9b42aa7f99f1a4fe149d1952cc8103c698cf380b15a43d47e72b7c7",
}];

/// Where the model's files are kept: its repository's snapshot at the
/// pinned revision in the Hugging Face cache, shared with every other
/// tool that uses it. See [`hub`].
pub fn dir() -> Option<PathBuf> {
    hub::snapshot_dir(REPO, REVISION)
}

/// Most tokens a forward pass runs through the model at once. Running a
/// batch of tokens together reads each weight once for the whole batch,
/// rather than once per token; the activations are allocated for this many.
/// A batch is one submission to the device, which a driver may abandon as
/// hung if it runs for seconds, so batches are kept short.
pub const BATCH: usize = 64;

/// Most tokens the model drafts at once: a batch to be verified holds them
/// and the token before.
pub const MAX_DRAFTS: usize = 8;

/// The architecture, read from the GGUF metadata.
#[derive(Debug)]
pub struct Config {
    pub hidden: usize,
    pub intermediate: usize,
    pub layers: usize,
    /// Full attention: heads, key/value heads, their width, and how much of
    /// each head rotary position embeddings rotate.
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rot_dim: usize,
    pub rope_theta: f32,
    /// Every this many layers, the last is full attention.
    pub full_attention_interval: usize,
    /// Linear attention: key heads, value heads, and the width of both.
    pub k_heads: usize,
    pub v_heads: usize,
    pub state_dim: usize,
    pub eps: f32,
    pub vocab: usize,
    /// Positions the model was trained for.
    pub context_length: usize,
    /// Whether the file holds the layer that predicts the token after the
    /// next, which the model drafts with.
    pub drafter: bool,
}

impl Config {
    pub fn load(gguf: &Gguf) -> Result<Self> {
        let arch = gguf.str("general.architecture")?;
        if arch != "qwen35" {
            return Err(format!("the '{arch}' architecture is not supported").into());
        }
        let key = |name: &str| gguf.u32(&format!("{arch}.{name}"));
        // The last blocks of a file may predict further tokens: the first
        // of them is the drafter, where the file has its weights, and the
        // rest are left out, as llama.cpp leaves them all.
        let nextn = gguf
            .get(&format!("{arch}.nextn_predict_layers"))
            .and_then(|v| v.as_i64())
            .unwrap_or(0) as usize;
        let blocks = key("block_count")? as usize;
        if nextn >= blocks {
            return Err("every layer of the model predicts further tokens".into());
        }
        let layers = blocks - nextn;
        let config = Self {
            hidden: key("embedding_length")? as usize,
            intermediate: key("feed_forward_length")? as usize,
            layers,
            heads: key("attention.head_count")? as usize,
            kv_heads: key("attention.head_count_kv")? as usize,
            head_dim: key("attention.key_length")? as usize,
            rot_dim: key("rope.dimension_count")? as usize,
            rope_theta: gguf.f32(&format!("{arch}.rope.freq_base"))?,
            full_attention_interval: key("full_attention_interval")? as usize,
            k_heads: key("ssm.group_count")? as usize,
            v_heads: key("ssm.time_step_rank")? as usize,
            state_dim: key("ssm.state_size")? as usize,
            eps: gguf.f32(&format!("{arch}.attention.layer_norm_rms_epsilon"))?,
            vocab: gguf.info("token_embd.weight")?.dims[1],
            context_length: key("context_length")? as usize,
            drafter: nextn >= 1
                && gguf
                    .info(&format!("blk.{layers}.nextn.eh_proj.weight"))
                    .is_ok(),
        };
        if key("ssm.conv_kernel")? as usize != CONV_KERNEL {
            return Err("the convolution kernel is not four wide".into());
        }
        if key("ssm.inner_size")? as usize != config.v_heads * config.state_dim {
            return Err("the linear attention sizes disagree".into());
        }
        if !config.v_heads.is_multiple_of(config.k_heads) {
            return Err("value heads are not a multiple of key heads".into());
        }
        if !config.head_dim.is_multiple_of(CACHE_BLOCK) {
            return Err("the attention heads are not whole blocks of the caches".into());
        }
        if key("attention.value_length")? as usize != config.head_dim {
            return Err("the attention values are not as wide as the keys".into());
        }
        // A file may say which layers are linear attention outright: it
        // must agree with the interval. Scaled rotary positions, which
        // llama.cpp reads for other models of the architecture, are not
        // implemented.
        if let Some(recurrent) = gguf.get(&format!("{arch}.attention.recurrent_layers")) {
            let recurrent = match recurrent {
                crate::inference::gguf::Value::Array(values) => values,
                _ => return Err("the list of linear layers is not a list".into()),
            };
            if recurrent.len() < config.layers
                || (0..config.layers)
                    .any(|i| recurrent[i].as_bool() != Some(!config.is_attention(i)))
            {
                return Err("the linear layers are not every layer but each fourth".into());
            }
        }
        if gguf
            .keys()
            .any(|k| k.starts_with(&format!("{arch}.rope.scaling.")))
        {
            return Err("scaled rotary positions are not implemented".into());
        }
        // The embedding table is looked up row by row as the model runs,
        // and the head's rows are the logits: both must be what the model
        // expects, or the first token would find out.
        let embedding = gguf.info("token_embd.weight")?;
        if embedding.dims.len() != 2 || embedding.dims[0] != config.hidden {
            return Err("the embedding table is not a row per token of the hidden width".into());
        }
        gguf::row_bytes(embedding.dtype, config.hidden)
            .map_err(|e| format!("the embedding table cannot be read: {e}"))?;
        let head = gguf.info("output.weight")?;
        if head.dims.len() != 2 || head.dims[1] != config.vocab || head.dims[0] != config.hidden {
            return Err("the output projection is not a row per token of the vocabulary".into());
        }
        Ok(config)
    }

    /// Whether the layer is full attention rather than linear.
    pub fn is_attention(&self, layer: usize) -> bool {
        (layer + 1).is_multiple_of(self.full_attention_interval)
    }

    /// Activations per token of each of the forward pass's per-token
    /// buffers, in the order [`State`] has them.
    fn widths(&self) -> [usize; 11] {
        let q_dim = self.heads * self.head_dim;
        let kv_dim = self.kv_heads * self.head_dim;
        let (k_dim, v_dim) = (self.k_dim(), self.v_dim());
        [
            self.hidden,
            self.hidden,
            self.hidden,
            q_dim.max(k_dim),
            q_dim,
            kv_dim.max(k_dim),
            kv_dim.max(v_dim),
            q_dim.max(v_dim),
            2 * k_dim + v_dim,
            v_dim,
            2 * self.v_heads,
        ]
    }

    /// Activations of a linear layer's convolution state: the tokens before
    /// the batch, a row of channels each.
    fn conv_width(&self) -> usize {
        (CONV_KERNEL - 1) * (2 * self.k_dim() + self.v_dim())
    }

    /// Activations of a linear layer's recurrent state: a matrix per value
    /// head.
    fn ssm_width(&self) -> usize {
        self.v_heads * self.state_dim * self.state_dim
    }

    fn k_dim(&self) -> usize {
        self.k_heads * self.state_dim
    }

    fn v_dim(&self) -> usize {
        self.v_heads * self.state_dim
    }
}

/// Device memory the model takes, in bytes: `fixed` for its weights and
/// the activations, which the sequences share, and for each sequence,
/// `per_sequence` for its recurrent states and `per_token` more for each
/// token of its context, for the key/value caches. The rotary position
/// embedding table, which the sequences share, is counted with each, at a
/// few hundred bytes a token.
#[derive(Clone, Copy, Debug)]
pub struct Memory {
    pub fixed: u64,
    pub per_sequence: u64,
    pub per_token: u64,
}

impl Memory {
    /// The memory the model in `gguf`, configured as `c`, takes on a
    /// device. Fails if the file does not say how big one of its tensors
    /// is.
    pub fn of(c: &Config, gguf: &Gguf) -> Result<Self> {
        // Every tensor goes to the device but the embedding table, which
        // stays in the file on the host, in the format the device holds it
        // in: a matrix of Q4_1 or Q6_K is quantized again as Q8_0 on
        // loading, and one of f32 rounded to bf16.
        let mut weights = 0;
        for name in gguf.tensors().filter(|&name| name != "token_embd.weight") {
            let info = gguf.info(name)?;
            let elements: usize = info.dims.iter().product();
            weights += match info.dtype {
                gguf::Q4_1 | gguf::Q6_K if info.dims.len() == 2 => {
                    elements / quant::BLOCK * quants::Q8_0_BYTES
                }
                gguf::F32 if info.dims.len() == 2 => elements * 2,
                _ => info.size()?,
            };
        }
        let attention = (0..c.layers).filter(|&i| c.is_attention(i)).count();
        let linear = c.layers - attention;
        // The activations of a batch, and the copy a mark keeps of a
        // sequence's linear layers' states and of the row of hidden state
        // carried over for the drafter.
        let mut activations = BATCH * (c.widths().iter().sum::<usize>() + 2 * c.intermediate)
            + c.vocab
            + linear * (c.conv_width() + c.ssm_width())
            + c.hidden;
        // A sequence's two convolution states and one recurrent state per
        // linear layer, and the hidden state after each token of a batch,
        // for the drafter.
        let sequence = linear * (2 * c.conv_width() + c.ssm_width()) + (BATCH + 1) * c.hidden;
        // The drafter has a key/value cache of its own in each sequence,
        // and verifying a batch keeps the linear layers' states run through
        // it apart, their inputs for it, the drafter's output, and every
        // token's logits.
        let caches = attention + c.drafter as usize;
        if c.drafter {
            activations += linear
                * (c.ssm_width() + BATCH * (2 * c.k_dim() + c.v_dim() + 2 * c.v_heads))
                + c.hidden
                + MAX_DRAFTS * c.vocab;
        }
        let kv_dim = c.kv_heads * c.head_dim;
        Ok(Self {
            fixed: (weights + 4 * activations) as u64,
            per_sequence: 4 * sequence as u64,
            per_token: (2 * caches * cache_bytes(kv_dim) + 4 * c.rot_dim) as u64,
        })
    }
}

/// The model, with its weights on a device, and the state of a sequence.
pub struct Model<D: Device> {
    pub config: Config,
    pub device: D,
    /// The file, kept mapped for the embedding table, which is looked up
    /// on the CPU: it is most of a third of a gigabyte, and a token needs
    /// one row of it.
    gguf: Arc<Gguf>,
    layers: Vec<Layer<D>>,
    norm: D::Buffer,
    lm_head: D::Weight,
    /// The layer that predicts the token after the next, if the file has
    /// it.
    draft: Option<Draft<D>>,
    state: State<D>,
    /// The sequences besides the one the model is running, by number, and
    /// how many key/value caches and linear layers' states a sequence has.
    parked: HashMap<usize, Sequence<D>>,
    caches: usize,
    slots: usize,
}

/// The layer that predicts the token after the next, to draft with: a full
/// attention layer with a key/value cache of its own, fed the embedding of
/// the token picked and the hidden state it was picked from, each
/// normalized and taken through its half of one projection; its output,
/// normalized, goes through the model's head.
struct Draft<D: Device> {
    embed_norm: D::Buffer,
    hidden_norm: D::Buffer,
    embed_proj: D::Weight,
    hidden_proj: D::Weight,
    layer: Layer<D>,
    norm: D::Buffer,
}

struct Layer<D: Device> {
    attn_norm: D::Buffer,
    mixer: Mixer<D>,
    mlp_norm: D::Buffer,
    gate: D::Weight,
    up: D::Weight,
    down: D::Weight,
}

/// What mixes the tokens in a layer.
enum Mixer<D: Device> {
    Attention {
        q: D::Weight,
        /// The output gate, projected alongside the query.
        gate: D::Weight,
        k: D::Weight,
        v: D::Weight,
        o: D::Weight,
        q_norm: D::Buffer,
        k_norm: D::Buffer,
        /// Which key/value cache is the layer's.
        cache: usize,
    },
    Linear {
        qkv: D::Weight,
        /// The output gate.
        z: D::Weight,
        /// The decay and forget projections, `alpha` then `beta`, in one
        /// bf16 matrix.
        ab: D::Weight,
        /// The convolution taps, four per channel.
        conv: D::Buffer,
        /// `a` for each value head, then `dt_bias` for each.
        decay: D::Buffer,
        norm: D::Buffer,
        out: D::Weight,
        /// Which recurrent state is the layer's.
        slot: usize,
    },
}

impl<D: Device> Model<D> {
    /// Loads the model from its GGUF file, uploading its weights to
    /// `device`, with state for sequences of up to `max_len` tokens, which
    /// may not be more than the model was trained for, and reporting how
    /// many of its tensors are loaded, out of how many, as it goes.
    pub fn load(
        gguf: Arc<Gguf>,
        device: D,
        max_len: usize,
        on_progress: impl FnMut(usize, usize),
    ) -> Result<Self> {
        let config = Config::load(&gguf)?;
        // Past the positions the model was trained on, its attention
        // degrades.
        if max_len == 0 || max_len > config.context_length {
            return Err(format!(
                "a context of {max_len} tokens is outside the 1 to {} the model was trained for",
                config.context_length
            )
            .into());
        }
        let c = &config;
        let attention_layers = (0..c.layers).filter(|&i| c.is_attention(i)).count();
        let mut l = Loader {
            device: &device,
            gguf: &gguf,
            done: 0,
            total: attention_layers * 11
                + (c.layers - attention_layers) * 12
                + 2
                + if c.drafter { 16 } else { 0 },
            on_progress,
        };
        let g = &*gguf;

        let mut layers = Vec::with_capacity(c.layers);
        let (mut caches, mut slots) = (0, 0);
        for i in 0..c.layers {
            layers.push(load_layer(
                &mut l,
                c,
                i,
                c.is_attention(i),
                &mut caches,
                &mut slots,
            )?);
        }
        let norm = l.buffer(g.f32s("output_norm.weight")?);
        let lm_head = l.weight("output.weight")?;
        let draft = if c.drafter {
            let p = format!("blk.{}.nextn", c.layers);
            // The projection takes the embedding first, then the hidden
            // state: the two halves are kept apart, so that the two can be
            // normalized and multiplied as they are and the results added.
            let name = format!("{p}.eh_proj.weight");
            let both = g.matrix(&name)?;
            if both.shape() != [c.hidden, 2 * c.hidden] {
                return Err(format!("{name} does not take an embedding and a hidden state").into());
            }
            Some(Draft {
                embed_norm: l.buffer(g.f32s(&format!("{p}.enorm.weight"))?),
                hidden_norm: l.buffer(g.f32s(&format!("{p}.hnorm.weight"))?),
                embed_proj: l.upload(both.select_column_groups(c.hidden, &[0])),
                hidden_proj: l.upload(both.select_column_groups(c.hidden, &[1])),
                layer: load_layer(&mut l, c, c.layers, true, &mut caches, &mut slots)?,
                norm: l.buffer(g.f32s(&format!("{p}.shared_head_norm.weight"))?),
            })
        } else {
            None
        };

        let state = State::new(&config, &device, max_len, caches, slots);
        Ok(Self {
            config,
            device,
            gguf,
            layers,
            norm,
            lm_head,
            draft,
            state,
            parked: HashMap::new(),
            caches,
            slots,
        })
    }

    /// Runs a batch of tokens, the first at position `pos`, and returns the
    /// logits after the last; or, verifying, the logits after each, a row
    /// per token, with the tokens not committed: the linear layers' states
    /// are run forward apart from the real ones, and the drafter waits for
    /// `rewind` to say which tokens were kept.
    fn forward_batch(&mut self, tokens: &[u32], pos: usize, verify: bool) -> Vec<f32> {
        let p = Pass {
            c: &self.config,
            d: &self.device,
            gguf: &self.gguf,
        };
        let (c, d) = (p.c, p.d);
        let s = &mut self.state;
        let n = tokens.len();
        s.batch(d, n);
        p.embed(&mut s.x, tokens);
        for layer in &self.layers {
            p.layer(s, layer, n, pos, verify);
        }
        if !verify {
            s.seq.parity = !s.seq.parity;
        }
        let Some(draft) = &self.draft else {
            // Only the last token's logits are wanted.
            d.resize(&mut s.xb, c.hidden);
            d.resize(&mut s.xh, c.hidden);
            d.copy(&mut s.xb, 0, &s.x, (n - 1) * c.hidden, c.hidden);
            p.normalize(&mut s.xh, &s.xb, &self.norm, 1);
            d.matmul(&mut s.logits, &self.lm_head, &s.xh);
            return d.read(&s.logits);
        };
        // The normalized hidden state after every token, for the drafter,
        // in the rows after the one carried over.
        p.normalize(&mut s.xh, &s.x, &self.norm, n);
        d.copy(&mut s.seq.hidden, c.hidden, &s.xh, 0, n * c.hidden);
        if verify {
            s.seq.verified = tokens.to_vec();
            s.seq.verified_at = pos;
            d.resize(&mut s.logits, n * c.vocab);
            d.matmul(&mut s.logits, &self.lm_head, &s.xh);
            return d.read(&s.logits);
        }
        d.resize(&mut s.xb, c.hidden);
        d.copy(&mut s.xb, 0, &s.xh, (n - 1) * c.hidden, c.hidden);
        d.resize(&mut s.logits, c.vocab);
        d.matmul(&mut s.logits, &self.lm_head, &s.xb);
        // The drafter catches up: it runs at every position whose next
        // token is known, from the one carried over if there is one, and
        // the hidden state after the batch is carried over for the next.
        let first = if s.seq.carry { 0 } else { 1 };
        p.drafter(
            s,
            draft,
            n - first,
            pos + first - 1,
            first,
            false,
            &tokens[first..],
        );
        s.carry_over(d, n);
        d.read(&s.logits)
    }
}

/// What a pass through the model has at hand: the configuration, the
/// device, and the file for the embedding table.
struct Pass<'a, D: Device> {
    c: &'a Config,
    d: &'a D,
    gguf: &'a Gguf,
}

impl<'a, D: Device> Pass<'a, D> {
    /// `out = rmsnorm(x) * weight` for `rows` rows of `x`, as wide as
    /// `weight`: what a matrix multiplication's input goes through.
    fn normalize(&self, out: &mut D::Buffer, x: &D::Buffer, weight: &D::Buffer, rows: usize) {
        let (c, d) = (self.c, self.d);
        d.copy(out, 0, x, 0, rows * c.hidden);
        d.rmsnorm(out, weight, c.eps);
    }

    /// Writes the embeddings of `tokens` into `out`, which is their size,
    /// looked up on the CPU.
    fn embed(&self, out: &mut D::Buffer, tokens: &[u32]) {
        let c = self.c;
        let mut x = vec![0.0; tokens.len() * c.hidden];
        for (x, &token) in x.chunks_exact_mut(c.hidden).zip(tokens) {
            self.gguf
                .row("token_embd.weight", token as usize, x)
                .expect("checked at load");
        }
        self.d.write(out, &x);
    }

    /// Runs `n` tokens, the first at position `pos`, through a layer: from
    /// the residual stream in `s.x` and back into it. Verifying, a linear
    /// layer runs its recurrent state forward apart from the real one and
    /// keeps its inputs, for the tokens accepted to be committed from.
    fn layer(&self, s: &mut State<D>, layer: &Layer<D>, n: usize, pos: usize, verify: bool) {
        let (c, d) = (self.c, self.d);
        let eps = c.eps;
        let kv_dim = c.kv_heads * c.head_dim;
        // The token mixer, with its result added back into the residual
        // stream. It sees the normalized activations.
        self.normalize(&mut s.xh, &s.x, &layer.attn_norm, n);
        match &layer.mixer {
            Mixer::Attention {
                q,
                gate,
                k,
                v,
                o,
                q_norm,
                k_norm,
                cache,
            } => {
                let q_dim = c.heads * c.head_dim;
                d.resize(&mut s.q, n * q_dim);
                d.resize(&mut s.gate, n * q_dim);
                d.resize(&mut s.k, n * kv_dim);
                d.resize(&mut s.v, n * kv_dim);
                d.resize(&mut s.att, n * q_dim);
                d.matmul(&mut s.q, q, &s.xh);
                d.matmul(&mut s.gate, gate, &s.xh);
                d.matmul(&mut s.k, k, &s.xh);
                d.matmul(&mut s.v, v, &s.xh);
                d.rmsnorm(&mut s.q, q_norm, eps);
                d.rmsnorm(&mut s.k, k_norm, eps);
                d.rope(&mut s.q, &s.rope, pos, c.heads, c.head_dim, c.rot_dim);
                d.rope(&mut s.k, &s.rope, pos, c.kv_heads, c.head_dim, c.rot_dim);
                d.store(&mut s.seq.k_cache[*cache], pos * kv_dim, &s.k);
                d.store(&mut s.seq.v_cache[*cache], pos * kv_dim, &s.v);
                d.attention(
                    &mut s.att,
                    &s.q,
                    &s.seq.k_cache[*cache],
                    &s.seq.v_cache[*cache],
                    pos,
                    c.heads,
                    c.head_dim,
                    c.kv_heads,
                );
                d.sigmoid_mul(&mut s.att, &s.gate);
                d.matmul(&mut s.xb, o, &s.att);
            }
            Mixer::Linear {
                qkv,
                z,
                ab,
                conv,
                decay,
                norm,
                out,
                slot,
            } => {
                let (k_dim, v_dim) = (c.k_dim(), c.v_dim());
                d.resize(&mut s.q, n * k_dim);
                d.resize(&mut s.k, n * k_dim);
                d.resize(&mut s.v, n * v_dim);
                d.resize(&mut s.att, n * v_dim);
                d.matmul(&mut s.qkv, qkv, &s.xh);
                d.matmul(&mut s.z, z, &s.xh);
                d.matmul(&mut s.ab, ab, &s.xh);
                if verify {
                    let width = n * (2 * k_dim + v_dim);
                    d.resize(&mut s.qkv_saved[*slot], width);
                    d.copy(&mut s.qkv_saved[*slot], 0, &s.qkv, 0, width);
                    d.resize(&mut s.ab_saved[*slot], n * 2 * c.v_heads);
                    d.copy(&mut s.ab_saved[*slot], 0, &s.ab, 0, n * 2 * c.v_heads);
                    d.copy(
                        &mut s.ssm_scratch[*slot],
                        0,
                        &s.seq.ssm_state[*slot],
                        0,
                        c.ssm_width(),
                    );
                }
                // The convolution reads the tokens before the batch from
                // one state buffer and leaves the ones after it in the
                // other.
                let [a, b] = &mut s.seq.conv_state[*slot];
                let (state, state_out) = if s.seq.parity { (&*a, b) } else { (&*b, a) };
                d.conv(&mut s.q, &mut s.k, &mut s.v, state_out, &s.qkv, state, conv);
                d.l2norm(&mut s.q, c.state_dim, eps);
                d.l2norm(&mut s.k, c.state_dim, eps);
                let ssm = if verify {
                    &mut s.ssm_scratch[*slot]
                } else {
                    &mut s.seq.ssm_state[*slot]
                };
                d.delta_net(
                    &mut s.att,
                    &s.q,
                    &s.k,
                    &s.v,
                    &s.ab,
                    decay,
                    ssm,
                    c.k_heads,
                    c.v_heads,
                    c.state_dim,
                );
                d.rmsnorm(&mut s.att, norm, eps);
                d.silu_mul(&mut s.z, &s.att);
                d.matmul(&mut s.xb, out, &s.z);
            }
        }
        d.add(&mut s.x, &s.xb);

        // Feed-forward network, likewise added back.
        self.normalize(&mut s.xh, &s.x, &layer.mlp_norm, n);
        d.matmul(&mut s.up, &layer.gate, &s.xh);
        d.matmul(&mut s.gate_ffn, &layer.up, &s.xh);
        d.silu_mul(&mut s.up, &s.gate_ffn);
        d.matmul(&mut s.xb, &layer.down, &s.up);
        d.add(&mut s.x, &s.xb);
    }

    /// Runs the drafter at `rows` positions from `pos`: each fed the
    /// hidden state in a row of `s.seq.hidden` from `first` on, or, chaining
    /// drafts, the drafter's own last output, and the token that followed
    /// the position. Leaves its output, normalized, in `s.xh`: what it
    /// predicts the token after from, through the model's head.
    #[allow(clippy::too_many_arguments)]
    fn drafter(
        &self,
        s: &mut State<D>,
        draft: &Draft<D>,
        rows: usize,
        pos: usize,
        first: usize,
        chain: bool,
        tokens: &[u32],
    ) {
        let (c, d) = (self.c, self.d);
        debug_assert_eq!(tokens.len(), rows);
        if rows == 0 {
            return;
        }
        s.batch(d, rows);
        self.embed(&mut s.xb, tokens);
        d.rmsnorm(&mut s.xb, &draft.embed_norm, c.eps);
        if chain {
            d.copy(&mut s.xh, 0, &s.drafted, 0, c.hidden);
        } else {
            d.copy(
                &mut s.xh,
                0,
                &s.seq.hidden,
                first * c.hidden,
                rows * c.hidden,
            );
        }
        d.rmsnorm(&mut s.xh, &draft.hidden_norm, c.eps);
        d.matmul(&mut s.x, &draft.embed_proj, &s.xb);
        d.matmul(&mut s.xb, &draft.hidden_proj, &s.xh);
        d.add(&mut s.x, &s.xb);
        self.layer(s, &draft.layer, rows, pos, false);
        self.normalize(&mut s.xh, &s.x, &draft.norm, rows);
    }
}

/// The most likely token.
fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &logit) in logits.iter().enumerate() {
        if logit > logits[best] {
            best = i;
        }
    }
    best as u32
}

/// Uploads the weights, counting them off.
struct Loader<'a, D: Device, F: FnMut(usize, usize)> {
    device: &'a D,
    gguf: &'a Gguf,
    done: usize,
    total: usize,
    on_progress: F,
}

impl<D: Device, F: FnMut(usize, usize)> Loader<'_, D, F> {
    fn tick(&mut self) {
        self.done += 1;
        (self.on_progress)(self.done, self.total);
    }

    /// Uploads the named matrix.
    fn weight(&mut self, name: &str) -> Result<D::Weight> {
        let tensor = self.gguf.matrix(name)?;
        Ok(self.upload(tensor))
    }

    /// Uploads a matrix.
    fn upload(&mut self, tensor: Tensor) -> D::Weight {
        let w = self.device.upload(tensor);
        self.tick();
        w
    }

    fn buffer(&mut self, values: Vec<f32>) -> D::Buffer {
        let mut buf = self.device.alloc(values.len());
        self.device.write(&mut buf, &values);
        self.tick();
        buf
    }
}

/// Loads layer `i`, full attention or linear, giving it the next key/value
/// cache or recurrent state.
fn load_layer<D: Device, F: FnMut(usize, usize)>(
    l: &mut Loader<'_, D, F>,
    c: &Config,
    i: usize,
    attention: bool,
    caches: &mut usize,
    slots: &mut usize,
) -> Result<Layer<D>> {
    let g = l.gguf;
    let p = format!("blk.{i}");
    let attn_norm = l.buffer(g.f32s(&format!("{p}.attn_norm.weight"))?);
    let mixer = if attention {
        // The query projection puts each head's gate after its
        // query: the rows are split into two matrices.
        let name = format!("{p}.attn_q.weight");
        let both = g.matrix(&name)?;
        if both.shape()[0] != 2 * c.heads * c.head_dim {
            return Err(format!("{name} does not hold a gate for each query").into());
        }
        let rows = |first: usize| -> Vec<usize> {
            (0..c.heads)
                .flat_map(|h| (0..c.head_dim).map(move |i| (2 * h + first) * c.head_dim + i))
                .collect()
        };
        let q = l.upload(both.select_rows(&rows(0)));
        let gate = l.device.upload(both.select_rows(&rows(1)));
        *caches += 1;
        Mixer::Attention {
            q,
            gate,
            k: l.weight(&format!("{p}.attn_k.weight"))?,
            v: l.weight(&format!("{p}.attn_v.weight"))?,
            o: l.weight(&format!("{p}.attn_output.weight"))?,
            q_norm: l.buffer(g.f32s(&format!("{p}.attn_q_norm.weight"))?),
            k_norm: l.buffer(g.f32s(&format!("{p}.attn_k_norm.weight"))?),
            cache: *caches - 1,
        }
    } else {
        // The value heads, and everything per value head, are put
        // back into the checkpoint's order: the rows of the value
        // heads' matrices, and of the value part of the combined
        // projection after its queries and keys.
        let heads = untiled(c.k_heads, c.v_heads, c.state_dim);
        let name = format!("{p}.attn_qkv.weight");
        let qkv = g.matrix(&name)?;
        let rows: Vec<usize> = (0..2 * c.k_dim())
            .chain(heads.iter().map(|&r| 2 * c.k_dim() + r))
            .collect();
        if qkv.shape()[0] != rows.len() {
            return Err(format!("{name} is not as tall as the queries, keys and values").into());
        }
        let qkv = l.upload(qkv.select_rows(&rows));
        let name = format!("{p}.attn_gate.weight");
        let z = l.upload(g.matrix(&name)?.select_rows(&heads));
        // The decay and forget projections, a row per value head,
        // as one bf16 matrix.
        let heads = untiled(c.k_heads, c.v_heads, 1);
        let ab = match (
            g.matrix(&format!("{p}.ssm_alpha.weight"))?
                .select_rows(&heads),
            g.matrix(&format!("{p}.ssm_beta.weight"))?
                .select_rows(&heads),
        ) {
            (Tensor::Bf16 { shape, data: alpha }, Tensor::Bf16 { data: beta, .. }) => {
                let mut data = alpha;
                data.extend(beta);
                Tensor::Bf16 {
                    shape: vec![2 * shape[0], shape[1]],
                    data,
                }
            }
            _ => {
                return Err(
                    format!("{p}: the decay and forget projections are not f32 or bf16").into(),
                );
            }
        };
        let ab = l.device.upload(ab);
        l.tick();
        let mut conv = g.f32s(&format!("{p}.ssm_conv1d.weight"))?;
        let v = 2 * c.k_dim() * CONV_KERNEL;
        let grouped = untile(&conv[v..], c.k_heads, c.v_heads, c.state_dim * CONV_KERNEL);
        conv[v..].copy_from_slice(&grouped);
        let conv = l.buffer(conv);
        let mut decay = untile(&g.f32s(&format!("{p}.ssm_a"))?, c.k_heads, c.v_heads, 1);
        decay.extend(untile(
            &g.f32s(&format!("{p}.ssm_dt.bias"))?,
            c.k_heads,
            c.v_heads,
            1,
        ));
        let decay = l.buffer(decay);
        let norm = l.buffer(g.f32s(&format!("{p}.ssm_norm.weight"))?);
        // The output projection takes the value heads' outputs in the
        // tiled order: its columns are put back into the checkpoint's too.
        let out = g.matrix(&format!("{p}.ssm_out.weight"))?;
        let out =
            l.upload(out.select_column_groups(c.state_dim, &untiled(c.k_heads, c.v_heads, 1)));
        *slots += 1;
        Mixer::Linear {
            qkv,
            z,
            ab,
            conv,
            decay,
            norm,
            out,
            slot: *slots - 1,
        }
    };
    let mlp_norm = l.buffer(g.f32s(&format!("{p}.post_attention_norm.weight"))?);
    Ok(Layer {
        attn_norm,
        mixer,
        mlp_norm,
        gate: l.weight(&format!("{p}.ffn_gate.weight"))?,
        up: l.weight(&format!("{p}.ffn_up.weight"))?,
        down: l.weight(&format!("{p}.ffn_down.weight"))?,
    })
}

/// The elements of `n_v` heads of `per_head` elements each, in the
/// checkpoint's grouped order, where head `h` shares key head
/// `h / (n_v / n_k)`, as indices into the tiled order llama.cpp keeps the
/// value heads in, where head `j` shares key head `j % n_k`.
fn untiled(n_k: usize, n_v: usize, per_head: usize) -> Vec<usize> {
    let rep = n_v / n_k;
    (0..n_v)
        .flat_map(|h| {
            let j = (h % rep) * n_k + h / rep;
            (0..per_head).map(move |i| j * per_head + i)
        })
        .collect()
}

/// Reorders `n_v` heads of `per_head` elements each from the tiled order
/// into the checkpoint's grouped order, as [`untiled`] says.
fn untile<T: Clone>(data: &[T], n_k: usize, n_v: usize, per_head: usize) -> Vec<T> {
    assert_eq!(data.len(), n_v * per_head);
    untiled(n_k, n_v, per_head)
        .into_iter()
        .map(|i| data[i].clone())
        .collect()
}

/// The first bytes of a saved state, and its layout's version.
const STATE_MAGIC: &[u8; 8] = b"dwimstat";
const STATE_VERSION: u32 = 3;

impl<D: Device> LanguageModel for Model<D> {
    /// The state after `len` positions: a header of the layout's version and
    /// the model's dimensions, then each attention layer's keys and values
    /// for the positions as the caches hold them, the drafter's among
    /// them, then each linear layer's convolution state and recurrent
    /// state, then, for a drafter, the hidden state carried over for it.
    fn save(&self, len: usize) -> Option<Vec<u8>> {
        let (c, d, s) = (&self.config, &self.device, &self.state);
        if !s.seq.verified.is_empty() || (self.draft.is_some() && !s.seq.carry) {
            return None;
        }
        let kv_dim = c.kv_heads * c.head_dim;
        let mut out = Vec::new();
        out.extend_from_slice(STATE_MAGIC);
        out.extend_from_slice(&STATE_VERSION.to_le_bytes());
        for value in [
            len,
            s.seq.k_cache.len(),
            s.seq.ssm_state.len(),
            kv_dim,
            c.conv_width(),
            c.ssm_width(),
            if self.draft.is_some() { c.hidden } else { 0 },
        ] {
            out.extend_from_slice(&(value as u64).to_le_bytes());
        }
        for cache in s.seq.k_cache.iter().chain(&s.seq.v_cache) {
            out.extend(d.read_cache(cache, len * kv_dim));
        }
        for slot in 0..s.seq.ssm_state.len() {
            // The convolution reads the batch before from one buffer of the
            // pair or the other, by the parity.
            let conv = &s.seq.conv_state[slot][if s.seq.parity { 0 } else { 1 }];
            for buf in [conv, &s.seq.ssm_state[slot]] {
                out.extend(d.read(buf).iter().flat_map(|v| v.to_le_bytes()));
            }
        }
        if self.draft.is_some() {
            out.extend(
                d.read(&s.seq.hidden)[..c.hidden]
                    .iter()
                    .flat_map(|v| v.to_le_bytes()),
            );
        }
        Some(out)
    }

    fn restore(&mut self, state: &[u8]) -> Result<usize> {
        let (c, d, s) = (&self.config, &self.device, &mut self.state);
        let kv_dim = c.kv_heads * c.head_dim;
        let mut at = 0;
        let mut take = |n: usize| -> Result<&[u8]> {
            let bytes = state
                .get(at..at + n)
                .ok_or("the saved state is cut short")?;
            at += n;
            Ok(bytes)
        };
        if take(8)? != STATE_MAGIC {
            return Err("not a saved state".into());
        }
        if u32::from_le_bytes(take(4)?.try_into()?) != STATE_VERSION {
            return Err("a saved state of another version".into());
        }
        let mut header = [0; 7];
        for value in &mut header {
            *value = u64::from_le_bytes(take(8)?.try_into()?) as usize;
        }
        let len = header[0];
        if header[1..]
            != [
                s.seq.k_cache.len(),
                s.seq.ssm_state.len(),
                kv_dim,
                c.conv_width(),
                c.ssm_width(),
                if self.draft.is_some() { c.hidden } else { 0 },
            ]
        {
            return Err("a saved state of another model".into());
        }
        if len > s.seq.max_len {
            return Err(
                format!("a saved state of {len} tokens does not fit in the context").into(),
            );
        }
        for cache in s.seq.k_cache.iter_mut().chain(&mut s.seq.v_cache) {
            d.write_cache(cache, take(cache_bytes(len * kv_dim))?);
        }
        for slot in 0..s.seq.ssm_state.len() {
            let [conv, _] = &mut s.seq.conv_state[slot];
            for (buf, width) in [
                (conv, c.conv_width()),
                (&mut s.seq.ssm_state[slot], c.ssm_width()),
            ] {
                let floats: Vec<f32> = take(width * 4)?
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
                d.write(buf, &floats);
            }
        }
        s.seq.parity = true;
        s.seq.verified.clear();
        if s.marked == Some(s.seq.id) {
            s.marked = None;
        }
        if self.draft.is_some() {
            let floats: Vec<f32> = take(c.hidden * 4)?
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            d.resize(&mut s.xb, c.hidden);
            d.write(&mut s.xb, &floats);
            d.copy(&mut s.seq.hidden, 0, &s.xb, 0, c.hidden);
            s.seq.carry = true;
        }
        Ok(len)
    }

    /// Adds the tokens to the model's state. Tokens run through the model
    /// in batches of up to [`BATCH`].
    fn forward(&mut self, tokens: &[u32], pos: usize) -> Vec<f32> {
        assert!(!tokens.is_empty(), "no tokens to run");
        assert!(
            pos + tokens.len() <= self.state.seq.max_len,
            "tokens past the end of the cache"
        );
        assert!(
            self.state.seq.verified.is_empty(),
            "a batch is being verified"
        );
        let mut logits = Vec::new();
        for (i, batch) in tokens.chunks(BATCH).enumerate() {
            logits = self.forward_batch(batch, pos + i * BATCH, false);
        }
        logits
    }

    /// The drafter runs at the position before `token`, fed the hidden
    /// state the token was picked from, which the latest batch carried
    /// over, and predicts the token after; then at each later position,
    /// fed its own output and the token it drafted, until it has drafted
    /// `k`, as many as the context has room to verify.
    fn draft(&mut self, token: u32, pos: usize, k: usize) -> Vec<u32> {
        let Some(draft) = &self.draft else {
            return Vec::new();
        };
        let p = Pass {
            c: &self.config,
            d: &self.device,
            gguf: &self.gguf,
        };
        let (c, d) = (p.c, p.d);
        let s = &mut self.state;
        let k = k.min(MAX_DRAFTS).min(s.seq.max_len.saturating_sub(pos + 1));
        if !s.seq.carry || !s.seq.verified.is_empty() || pos == 0 || k == 0 {
            return Vec::new();
        }
        let mut drafts = Vec::with_capacity(k);
        let mut next = token;
        for i in 0..k {
            p.drafter(s, draft, 1, pos - 1 + i, 0, i > 0, &[next]);
            d.copy(&mut s.drafted, 0, &s.xh, 0, c.hidden);
            d.resize(&mut s.logits, c.vocab);
            d.matmul(&mut s.logits, &self.lm_head, &s.xh);
            next = argmax(&d.read(&s.logits));
            drafts.push(next);
        }
        drafts
    }

    fn verify(&mut self, tokens: &[u32], pos: usize) -> Vec<f32> {
        assert!(self.draft.is_some(), "the model does not draft");
        assert!(
            self.state.seq.verified.is_empty() && self.state.seq.carry,
            "a batch is being verified, or none was run before"
        );
        assert!(
            !tokens.is_empty() && tokens.len() <= MAX_DRAFTS + 1,
            "a batch of {} tokens is not one to verify",
            tokens.len()
        );
        assert!(
            pos + tokens.len() <= self.state.seq.max_len,
            "tokens past the end of the cache"
        );
        self.forward_batch(tokens, pos, true)
    }

    /// The linear layers' states run through the whole batch become the
    /// real ones if every token is kept; otherwise the real ones are run
    /// through the tokens kept, from the inputs saved. The drafter then
    /// catches up with the tokens kept, from the hidden states the model
    /// computed for them, in place of what it ran while drafting, and the
    /// hidden state after the last is carried over.
    fn rewind(&mut self, keep: usize) {
        let p = Pass {
            c: &self.config,
            d: &self.device,
            gguf: &self.gguf,
        };
        let (c, d) = (p.c, p.d);
        let s = &mut self.state;
        let n = s.seq.verified.len();
        assert!(
            n > 0 && (1..=n).contains(&keep),
            "keeping {keep} tokens of a batch of {n} being verified"
        );
        let tokens = std::mem::take(&mut s.seq.verified);
        if keep == n {
            for slot in 0..s.seq.ssm_state.len() {
                std::mem::swap(&mut s.seq.ssm_state[slot], &mut s.ssm_scratch[slot]);
            }
        } else {
            s.batch(d, keep);
            let (k_dim, v_dim) = (c.k_dim(), c.v_dim());
            for layer in &self.layers {
                let Mixer::Linear {
                    conv, decay, slot, ..
                } = &layer.mixer
                else {
                    continue;
                };
                d.copy(
                    &mut s.qkv,
                    0,
                    &s.qkv_saved[*slot],
                    0,
                    keep * (2 * k_dim + v_dim),
                );
                d.copy(&mut s.ab, 0, &s.ab_saved[*slot], 0, keep * 2 * c.v_heads);
                d.resize(&mut s.q, keep * k_dim);
                d.resize(&mut s.k, keep * k_dim);
                d.resize(&mut s.v, keep * v_dim);
                d.resize(&mut s.att, keep * v_dim);
                let [a, b] = &mut s.seq.conv_state[*slot];
                let (state, state_out) = if s.seq.parity { (&*a, b) } else { (&*b, a) };
                d.conv(&mut s.q, &mut s.k, &mut s.v, state_out, &s.qkv, state, conv);
                d.l2norm(&mut s.q, c.state_dim, c.eps);
                d.l2norm(&mut s.k, c.state_dim, c.eps);
                d.delta_net(
                    &mut s.att,
                    &s.q,
                    &s.k,
                    &s.v,
                    &s.ab,
                    decay,
                    &mut s.seq.ssm_state[*slot],
                    c.k_heads,
                    c.v_heads,
                    c.state_dim,
                );
            }
        }
        s.seq.parity = !s.seq.parity;
        if let Some(draft) = &self.draft {
            p.drafter(
                s,
                draft,
                keep,
                s.seq.verified_at - 1,
                0,
                false,
                &tokens[..keep],
            );
            s.carry_over(d, keep);
        }
    }

    /// Copies the linear layers' states aside, on the device, and the
    /// hidden state carried over for the drafter. The attention layers'
    /// caches need no copy, the drafter's among them: attention reads only
    /// the positions before the token's, so the positions after the mark
    /// are written over as the sequence goes on from it again.
    fn mark(&mut self) -> bool {
        let (c, d, s) = (&self.config, &self.device, &mut self.state);
        assert!(s.seq.verified.is_empty(), "a batch is being verified");
        for slot in 0..s.seq.ssm_state.len() {
            // The convolution reads the batch before from one buffer of the
            // pair or the other, by the parity.
            let conv = &s.seq.conv_state[slot][if s.seq.parity { 0 } else { 1 }];
            d.copy(&mut s.mark_conv[slot], 0, conv, 0, c.conv_width());
            d.copy(
                &mut s.mark_ssm[slot],
                0,
                &s.seq.ssm_state[slot],
                0,
                c.ssm_width(),
            );
        }
        // Carried over only for a drafter.
        if s.seq.carry {
            d.copy(&mut s.mark_hidden, 0, &s.seq.hidden, 0, c.hidden);
        }
        s.mark_carry = s.seq.carry;
        s.marked = Some(s.seq.id);
        true
    }

    /// Copies the linear layers' states back from the mark, and the hidden
    /// state carried over for the drafter.
    fn rewind_to_mark(&mut self) -> Result<()> {
        let (c, d, s) = (&self.config, &self.device, &mut self.state);
        if s.marked != Some(s.seq.id) {
            return Err("no mark to rewind to".into());
        }
        if !s.seq.verified.is_empty() {
            return Err("a batch is being verified".into());
        }
        for slot in 0..s.seq.ssm_state.len() {
            let [conv, _] = &mut s.seq.conv_state[slot];
            d.copy(conv, 0, &s.mark_conv[slot], 0, c.conv_width());
            d.copy(
                &mut s.seq.ssm_state[slot],
                0,
                &s.mark_ssm[slot],
                0,
                c.ssm_width(),
            );
        }
        s.seq.parity = true;
        if s.mark_carry {
            d.copy(&mut s.seq.hidden, 0, &s.mark_hidden, 0, c.hidden);
        }
        s.seq.carry = s.mark_carry;
        Ok(())
    }

    /// Zeroes the linear layers' states, as they were before any token,
    /// and forgets what was carried over for the drafter. The attention
    /// layers' caches need no clearing: attention reads only the positions
    /// before the token's.
    fn reset(&mut self) {
        let (c, d, s) = (&self.config, &self.device, &mut self.state);
        for slot in 0..s.seq.ssm_state.len() {
            for conv in &mut s.seq.conv_state[slot] {
                d.write(conv, &vec![0.0; c.conv_width()]);
            }
            d.write(&mut s.seq.ssm_state[slot], &vec![0.0; c.ssm_width()]);
        }
        s.seq.parity = false;
        s.seq.carry = false;
        s.seq.verified.clear();
        if s.marked == Some(s.seq.id) {
            s.marked = None;
        }
    }

    fn max_len(&self) -> usize {
        self.state.seq.max_len
    }

    /// Allocates another sequence of the length the first has, on the
    /// device, as before any token.
    fn add_sequence(&mut self) -> Result<usize> {
        let id = self.parked.len() + 1;
        let seq = Sequence::new(
            &self.config,
            &self.device,
            id,
            self.state.seq.max_len,
            self.caches,
            self.slots,
        );
        self.parked.insert(id, seq);
        Ok(id)
    }

    /// Swaps the sequence the model runs for another: nothing is copied.
    fn select_sequence(&mut self, id: usize) -> Result<()> {
        if self.state.seq.id == id {
            return Ok(());
        }
        if !self.state.seq.verified.is_empty() {
            return Err("a batch is being verified".into());
        }
        let mut seq = self
            .parked
            .remove(&id)
            .ok_or_else(|| format!("no sequence {id}"))?;
        std::mem::swap(&mut self.state.seq, &mut seq);
        self.parked.insert(seq.id, seq);
        Ok(())
    }
}

/// Buffers the forward pass computes in, with room for a batch of
/// [`BATCH`] tokens; the key/value caches of the attention layers, holding
/// every position seen so far in 8-bit blocks; the convolution and
/// recurrent states of the linear attention layers; and the rotary position
/// embedding table for every position there is room for.
struct State<D: Device> {
    /// Activations per token of each per-token buffer, in field order.
    widths: [usize; 11],
    x: D::Buffer,
    xb: D::Buffer,
    xh: D::Buffer,
    q: D::Buffer,
    gate: D::Buffer,
    k: D::Buffer,
    v: D::Buffer,
    att: D::Buffer,
    qkv: D::Buffer,
    z: D::Buffer,
    ab: D::Buffer,
    /// The feed-forward network's two projections, `intermediate` wide.
    up: D::Buffer,
    gate_ffn: D::Buffer,
    intermediate: usize,
    logits: D::Buffer,
    rope: D::Buffer,
    /// The drafter's own output, from one drafted token to the next.
    drafted: D::Buffer,
    /// For a batch being verified, which the sequence it is of has the
    /// tokens of: the linear layers' recurrent states run through it, kept
    /// apart from the real ones until the accepted tokens are committed;
    /// and each linear layer's convolution inputs and gates for it, to
    /// commit from. A sequence is not changed for another while one is
    /// being verified.
    ssm_scratch: Vec<D::Buffer>,
    qkv_saved: Vec<D::Buffer>,
    ab_saved: Vec<D::Buffer>,
    /// The linear layers' states as they were at the mark, a convolution
    /// state and a recurrent state per layer, and the hidden state carried
    /// over for the drafter then, if one was; and which sequence's they
    /// are, if a mark has been made. A mark is one sequence's at a time:
    /// marking another forgets it.
    mark_conv: Vec<D::Buffer>,
    mark_ssm: Vec<D::Buffer>,
    mark_hidden: D::Buffer,
    mark_carry: bool,
    marked: Option<usize>,
    /// The sequence the model is running.
    seq: Sequence<D>,
}

/// A sequence's own state, which the model keeps one of for each sequence
/// it runs, while the weights and the buffers it computes in are shared:
/// the key/value caches of the attention layers, holding every position
/// seen so far in 8-bit blocks; the convolution and recurrent states of the
/// linear attention layers; and what the drafter carries over.
struct Sequence<D: Device> {
    id: usize,
    k_cache: Vec<D::Cache>,
    v_cache: Vec<D::Cache>,
    /// Two per linear layer, the batch's inputs coming from one and its
    /// outputs going to the other, swapping every batch.
    conv_state: Vec<[D::Buffer; 2]>,
    parity: bool,
    ssm_state: Vec<D::Buffer>,
    max_len: usize,
    /// For the drafter, which runs a position behind the model: the
    /// normalized hidden state after each token of the latest batch, in
    /// the rows after the first, which holds the one after the token
    /// before the batch, carried over; and whether it holds one.
    hidden: D::Buffer,
    carry: bool,
    /// A batch being verified: its tokens and the position of the first.
    verified: Vec<u32>,
    verified_at: usize,
}

impl<D: Device> Sequence<D> {
    /// Allocates the state of sequence `id`, of up to `max_len` tokens,
    /// with `caches` key/value caches and `slots` linear layers' states,
    /// all as before any token.
    fn new(c: &Config, d: &D, id: usize, max_len: usize, caches: usize, slots: usize) -> Self {
        let kv_dim = c.kv_heads * c.head_dim;
        let conv_width = c.conv_width();
        let zeroed = |len: usize| {
            let mut buf = d.alloc(len);
            d.write(&mut buf, &vec![0.0; len]);
            buf
        };
        Self {
            id,
            k_cache: (0..caches)
                .map(|_| d.alloc_cache(max_len * kv_dim))
                .collect(),
            v_cache: (0..caches)
                .map(|_| d.alloc_cache(max_len * kv_dim))
                .collect(),
            conv_state: (0..slots)
                .map(|_| [zeroed(conv_width), zeroed(conv_width)])
                .collect(),
            parity: false,
            ssm_state: (0..slots).map(|_| zeroed(c.ssm_width())).collect(),
            max_len,
            hidden: d.alloc((BATCH + 1) * c.hidden),
            carry: false,
            verified: Vec::new(),
            verified_at: 0,
        }
    }
}

impl<D: Device> State<D> {
    /// Allocates the buffers, and the first sequence, number 0, for
    /// sequences of up to `max_len` tokens.
    fn new(c: &Config, d: &D, max_len: usize, caches: usize, slots: usize) -> Self {
        let widths = c.widths();
        let [x, xb, xh, q, gate, k, v, att, qkv, z, ab] =
            widths.map(|width| d.alloc(BATCH * width));
        let table = rope_table(max_len, c.rot_dim, c.rope_theta);
        let mut rope = d.alloc(table.len());
        d.write(&mut rope, &table);
        let conv_width = c.conv_width();
        Self {
            widths,
            x,
            xb,
            xh,
            q,
            gate,
            k,
            v,
            att,
            qkv,
            z,
            ab,
            up: d.alloc(BATCH * c.intermediate),
            gate_ffn: d.alloc(BATCH * c.intermediate),
            intermediate: c.intermediate,
            logits: d.alloc(if c.drafter { MAX_DRAFTS + 1 } else { 1 } * c.vocab),
            rope,
            drafted: d.alloc(c.hidden),
            ssm_scratch: (0..if c.drafter { slots } else { 0 })
                .map(|_| d.alloc(c.ssm_width()))
                .collect(),
            qkv_saved: (0..if c.drafter { slots } else { 0 })
                .map(|_| d.alloc(BATCH * (2 * c.k_dim() + c.v_dim())))
                .collect(),
            ab_saved: (0..if c.drafter { slots } else { 0 })
                .map(|_| d.alloc(BATCH * 2 * c.v_heads))
                .collect(),
            mark_conv: (0..slots).map(|_| d.alloc(conv_width)).collect(),
            mark_ssm: (0..slots).map(|_| d.alloc(c.ssm_width())).collect(),
            mark_hidden: d.alloc(c.hidden),
            mark_carry: false,
            marked: None,
            seq: Sequence::new(c, d, 0, max_len, caches, slots),
        }
    }

    /// Carries the hidden state after the last of `n` tokens over into
    /// the first row, for the drafter, through `xb`.
    fn carry_over(&mut self, d: &D, n: usize) {
        let hidden = self.widths[0];
        d.resize(&mut self.xb, hidden);
        d.copy(&mut self.xb, 0, &self.seq.hidden, n * hidden, hidden);
        d.copy(&mut self.seq.hidden, 0, &self.xb, 0, hidden);
        self.seq.carry = true;
    }

    /// Sizes the buffers for a batch of `n` tokens.
    fn batch(&mut self, device: &D, n: usize) {
        assert!(n <= BATCH, "a batch of {n} tokens is more than {BATCH}");
        let buffers = [
            &mut self.x,
            &mut self.xb,
            &mut self.xh,
            &mut self.q,
            &mut self.gate,
            &mut self.k,
            &mut self.v,
            &mut self.att,
            &mut self.qkv,
            &mut self.z,
            &mut self.ab,
        ];
        for (buffer, width) in buffers.into_iter().zip(self.widths) {
            device.resize(buffer, n * width);
        }
        device.resize(&mut self.up, n * self.intermediate);
        device.resize(&mut self.gate_ffn, n * self.intermediate);
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::{
        gpu::{Cpu, Gpu},
        inference::gguf::Value,
    };

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        }

        fn floats(&mut self, n: usize, scale: f32, offset: f32) -> Vec<u8> {
            (0..n)
                .flat_map(|_| (self.next() * scale + offset).to_le_bytes())
                .collect()
        }

        /// A matrix in a block format as the file lays it out.
        fn packed(&mut self, rows: usize, cols: usize, scale: f32, dtype: u32) -> Vec<u8> {
            let mut data = Vec::new();
            for _ in 0..rows {
                let values: Vec<f32> = (0..cols).map(|_| self.next() * scale).collect();
                data.extend(match dtype {
                    gguf::Q4_0 => quants::pack_q4_0(&values),
                    gguf::Q8_0 => quants::pack_q8_0(&values),
                    _ => unreachable!(),
                });
            }
            data
        }
    }

    /// Writes a tiny random model in the file format as Qwen's file is:
    /// four layers of which the last is full attention, Q4_0 and Q8_0
    /// weights, f32 decay projections, and a block after the layers that
    /// predicts the token after the next.
    fn synthetic(path: &Path) {
        let (hidden, inter, layers, heads, kv_heads, head_dim, rot_dim) =
            (1024, 1024, 4, 4, 2, 256, 64);
        let (k_heads, v_heads, state_dim, vocab) = (2, 8, 128, 64);
        let (k_dim, v_dim) = (k_heads * state_dim, v_heads * state_dim);
        let mut rng = Rng(42);
        let mut tensors: Vec<(String, Vec<usize>, u32, Vec<u8>)> = Vec::new();
        // Weights are small so that the activations stay sane through the
        // layers. The down and output projections are Q8_0, the rest Q4_0.
        let mut quant = |name: &str, rows: usize, cols: usize| {
            let scale = 1.0 / (cols as f32).sqrt();
            let (dtype, data) = if name.contains("ffn_down") || name.contains("ssm_out") {
                (gguf::Q8_0, rng.packed(rows, cols, scale, gguf::Q8_0))
            } else {
                (gguf::Q4_0, rng.packed(rows, cols, scale, gguf::Q4_0))
            };
            tensors.push((name.to_string(), vec![cols, rows], dtype, data));
        };
        quant("output.weight", vocab, hidden);
        // The block after the layers predicts the token after the next: a
        // full attention layer, with the projection of an embedding and a
        // hidden state into its input.
        let blocks = layers + 1;
        for i in 0..blocks {
            let p = format!("blk.{i}");
            if i == layers {
                quant(&format!("{p}.nextn.eh_proj.weight"), hidden, 2 * hidden);
            }
            if (i + 1) % 4 == 0 || i == layers {
                quant(&format!("{p}.attn_q.weight"), heads * head_dim * 2, hidden);
                quant(&format!("{p}.attn_k.weight"), kv_heads * head_dim, hidden);
                quant(&format!("{p}.attn_v.weight"), kv_heads * head_dim, hidden);
                quant(&format!("{p}.attn_output.weight"), hidden, heads * head_dim);
            } else {
                quant(&format!("{p}.attn_qkv.weight"), 2 * k_dim + v_dim, hidden);
                quant(&format!("{p}.attn_gate.weight"), v_dim, hidden);
                quant(&format!("{p}.ssm_out.weight"), hidden, v_dim);
            }
            quant(&format!("{p}.ffn_gate.weight"), inter, hidden);
            quant(&format!("{p}.ffn_up.weight"), inter, hidden);
            quant(&format!("{p}.ffn_down.weight"), hidden, inter);
        }
        let mut rng = Rng(7);
        tensors.push((
            "token_embd.weight".into(),
            vec![hidden, vocab],
            gguf::Q4_0,
            rng.packed(vocab, hidden, 0.5, gguf::Q4_0),
        ));
        tensors.push((
            "output_norm.weight".into(),
            vec![hidden],
            gguf::F32,
            rng.floats(hidden, 0.1, 1.0),
        ));
        for i in 0..blocks {
            let p = format!("blk.{i}");
            tensors.push((
                format!("{p}.attn_norm.weight"),
                vec![hidden],
                gguf::F32,
                rng.floats(hidden, 0.1, 1.0),
            ));
            tensors.push((
                format!("{p}.post_attention_norm.weight"),
                vec![hidden],
                gguf::F32,
                rng.floats(hidden, 0.1, 1.0),
            ));
            if i == layers {
                for name in ["enorm", "hnorm", "shared_head_norm"] {
                    tensors.push((
                        format!("{p}.nextn.{name}.weight"),
                        vec![hidden],
                        gguf::F32,
                        rng.floats(hidden, 0.1, 1.0),
                    ));
                }
            }
            if (i + 1) % 4 == 0 || i == layers {
                tensors.push((
                    format!("{p}.attn_q_norm.weight"),
                    vec![head_dim],
                    gguf::F32,
                    rng.floats(head_dim, 0.1, 1.0),
                ));
                tensors.push((
                    format!("{p}.attn_k_norm.weight"),
                    vec![head_dim],
                    gguf::F32,
                    rng.floats(head_dim, 0.1, 1.0),
                ));
            } else {
                for name in ["ssm_alpha", "ssm_beta"] {
                    tensors.push((
                        format!("{p}.{name}.weight"),
                        vec![hidden, v_heads],
                        gguf::F32,
                        rng.floats(v_heads * hidden, 0.05, 0.0),
                    ));
                }
                let channels = 2 * k_dim + v_dim;
                tensors.push((
                    format!("{p}.ssm_conv1d.weight"),
                    vec![CONV_KERNEL, channels],
                    gguf::F32,
                    rng.floats(channels * CONV_KERNEL, 0.5, 0.0),
                ));
                tensors.push((
                    format!("{p}.ssm_a"),
                    vec![v_heads],
                    gguf::F32,
                    rng.floats(v_heads, 0.5, -0.6),
                ));
                tensors.push((
                    format!("{p}.ssm_dt.bias"),
                    vec![v_heads],
                    gguf::F32,
                    rng.floats(v_heads, 2.0, 0.0),
                ));
                tensors.push((
                    format!("{p}.ssm_norm.weight"),
                    vec![state_dim],
                    gguf::F32,
                    rng.floats(state_dim, 0.1, 1.0),
                ));
            }
        }
        let n = |v: usize| Value::U32(v as u32);
        let mut meta = vec![
            ("general.architecture", Value::Str("qwen35".into())),
            ("qwen35.block_count", n(blocks)),
            ("qwen35.context_length", n(256)),
            ("qwen35.embedding_length", n(hidden)),
            ("qwen35.feed_forward_length", n(inter)),
            ("qwen35.attention.head_count", n(heads)),
            ("qwen35.attention.head_count_kv", n(kv_heads)),
            ("qwen35.attention.key_length", n(head_dim)),
            ("qwen35.attention.value_length", n(head_dim)),
            ("qwen35.rope.dimension_count", n(rot_dim)),
            ("qwen35.rope.freq_base", Value::F32(10000.0)),
            ("qwen35.attention.layer_norm_rms_epsilon", Value::F32(1e-6)),
            ("qwen35.ssm.conv_kernel", n(CONV_KERNEL)),
            ("qwen35.ssm.state_size", n(state_dim)),
            ("qwen35.ssm.group_count", n(k_heads)),
            ("qwen35.ssm.time_step_rank", n(v_heads)),
            ("qwen35.ssm.inner_size", n(v_dim)),
            ("qwen35.full_attention_interval", n(4)),
        ];
        // The last block predicts the token after the next; the file also
        // lists which layers are linear.
        meta.push(("qwen35.nextn_predict_layers", n(1)));
        let recurrent = (0..blocks).map(|i| Value::Bool((i + 1) % 4 != 0 && i < layers));
        meta.push((
            "qwen35.attention.recurrent_layers",
            Value::Array(recurrent.collect()),
        ));
        gguf::write(path, &meta, &tensors).unwrap();
    }

    fn run<D: Device>(path: &Path, device: D) -> [Vec<f32>; 3] {
        let gguf = Arc::new(Gguf::open(path).unwrap());
        let mut model = Model::load(gguf, device, 64, |_, _| {}).unwrap();
        // A batch of several tokens, then one more, then a batch of many,
        // each carrying the state.
        let first = model.forward(&[3, 17, 42, 7, 9], 0);
        let second = model.forward(&[11], 5);
        let many: Vec<u32> = (0..40).map(|i| (i * 5 % 43) as u32).collect();
        let third = model.forward(&many, 6);
        [first, second, third]
    }

    #[test]
    fn restores_a_saved_state() {
        let dir = std::env::temp_dir().join(format!("dwim-qwen35-state-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.gguf");
        synthetic(&path);
        let gguf = Arc::new(Gguf::open(&path).unwrap());
        // A conversation of two batches, and one that takes up from the
        // first's saved state: the same logits, since the state holds the
        // caches exactly.
        let mut whole = Model::load(gguf.clone(), Cpu, 64, |_, _| {}).unwrap();
        whole.forward(&[3, 17, 42, 7, 9], 0);
        let state = whole.save(5).unwrap();
        let want = whole.forward(&[11, 2], 5);
        let mut resumed = Model::load(gguf.clone(), Cpu, 64, |_, _| {}).unwrap();
        assert_eq!(resumed.restore(&state).unwrap(), 5);
        assert_eq!(resumed.forward(&[11, 2], 5), want);
        assert!(resumed.restore(&state[..100]).is_err());

        // Reset, the model is as it was when loaded: the same logits for
        // the same tokens from the start, after a conversation that left
        // the linear layers' states behind.
        let fresh = Model::load(gguf, Cpu, 64, |_, _| {})
            .unwrap()
            .forward(&[3, 17, 42, 7, 9], 0);
        assert_ne!(whole.forward(&[3, 17, 42, 7, 9], 7), fresh);
        whole.reset();
        assert_eq!(whole.forward(&[3, 17, 42, 7, 9], 0), fresh);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Runs a sequence through the model on `device` three ways, and
    /// returns the logits after it each time, with what the drafter then
    /// drafts: straight through; with a detour of other tokens after the
    /// first five, which stay as if they were the sequence's; and with the
    /// detour taken from a mark that is rewound to before the sequence
    /// goes on.
    fn rewinding<D: Device>(path: &Path, device: D) -> [(Vec<f32>, Vec<u32>); 3] {
        let gguf = Arc::new(Gguf::open(path).unwrap());
        let mut model = Model::load(gguf, device, 64, |_, _| {}).unwrap();
        let rest: Vec<u32> = (0..40).map(|i| (i * 5 % 43) as u32).collect();
        model.forward(&[3, 17, 42, 7, 9], 0);
        let straight = (model.forward(&rest, 5), model.draft(8, 45, 3));
        model.reset();
        model.forward(&[3, 17, 42, 7, 9], 0);
        model.forward(&[11, 2, 30], 5);
        model.forward(&[8], 8);
        let detoured = (model.forward(&rest, 9), model.draft(8, 49, 3));
        model.reset();
        model.forward(&[3, 17, 42, 7, 9], 0);
        assert!(model.mark());
        model.forward(&[11, 2, 30], 5);
        model.forward(&[8], 8);
        model.rewind_to_mark().unwrap();
        let rewound = (model.forward(&rest, 5), model.draft(8, 45, 3));
        [straight, detoured, rewound]
    }

    #[test]
    fn rewinds_to_a_mark() {
        let dir = std::env::temp_dir().join(format!("dwim-qwen35-mark-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.gguf");
        synthetic(&path);
        // The detour leaves the linear layers' states behind, and the
        // rewind takes them back: the same logits as without the detour,
        // and the same drafts after them.
        let [straight, detoured, rewound] = rewinding(&path, Cpu);
        assert_ne!(straight.0, detoured.0);
        assert_eq!(straight.0, rewound.0);
        assert_eq!(straight.1, rewound.1);
        if let Ok(gpu) = Gpu::new() {
            let [straight, _, rewound] = rewinding(&path, gpu);
            close(&rewound.0, &straight.0, 2e-3, "rewound on the GPU");
            assert_eq!(straight.1, rewound.1);
        } else {
            eprintln!("skipping the GPU rewind: no GPU");
        }
        // There is no mark to rewind to before one is made, nor after a
        // reset.
        let gguf = Arc::new(Gguf::open(&path).unwrap());
        let mut model = Model::load(gguf, Cpu, 64, |_, _| {}).unwrap();
        assert!(model.rewind_to_mark().is_err());
        model.forward(&[3, 17], 0);
        assert!(model.mark());
        model.reset();
        assert!(model.rewind_to_mark().is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn gpu_agrees_with_cpu() {
        let dir = std::env::temp_dir().join(format!("dwim-qwen35-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.gguf");
        synthetic(&path);
        let cpu = run(&path, Cpu);
        assert!(cpu[0].iter().all(|v| v.is_finite()));
        assert!(cpu[0].iter().any(|&v| v != 0.0));
        assert_ne!(cpu[0], cpu[1]);
        if let Ok(gpu) = Gpu::new() {
            let gpu = run(&path, gpu);
            // The batch of many tokens may go through the GPU's tiled
            // matmul, which sums in half precision.
            for ((cpu, gpu), tolerance) in cpu.iter().zip(&gpu).zip([2e-3, 2e-3, 1e-2]) {
                for (a, b) in cpu.iter().zip(gpu) {
                    assert!((a - b).abs() <= tolerance * (1.0 + a.abs()), "{a} vs {b}");
                }
            }
        } else {
            eprintln!("skipping the GPU comparison: no GPU");
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Checks that every logit of `got` is within `tolerance` of `want`'s,
    /// relative to its size.
    fn close(got: &[f32], want: &[f32], tolerance: f32, what: &str) {
        assert_eq!(got.len(), want.len(), "{what}");
        for (a, b) in got.iter().zip(want) {
            assert!(
                (a - b).abs() <= tolerance * (1.0 + b.abs()),
                "{what}: {a} vs {b}"
            );
        }
    }

    /// Verifies a batch after a prefix, keeps `keep` tokens of it and runs
    /// one more; and runs the same tokens plainly: the logits of the batch
    /// must be those after each of its tokens, and the one more must come
    /// out the same either way, the linear layers' states having been
    /// committed through the tokens kept alone. Then the drafter must
    /// draft the same after either.
    fn verify_and_rewind<D: Device>(
        path: &Path,
        open: impl Fn() -> D,
        keep: usize,
        tolerance: f32,
    ) {
        let gguf = Arc::new(Gguf::open(path).unwrap());
        let prefix = [3, 17, 42, 7, 9];
        let batch = [11, 2, 30, 5];
        let mut spec = Model::load(gguf.clone(), open(), 64, |_, _| {}).unwrap();
        let mut plain = Model::load(gguf, open(), 64, |_, _| {}).unwrap();
        spec.forward(&prefix, 0);
        plain.forward(&prefix, 0);
        let rows = spec.verify(&batch, 5);
        let vocab = spec.config.vocab;
        assert_eq!(rows.len(), batch.len() * vocab);
        for (i, &token) in batch.iter().enumerate() {
            let want = plain.forward(&[token], 5 + i);
            close(
                &rows[i * vocab..][..vocab],
                &want,
                tolerance,
                &format!("row {i}"),
            );
            if i + 1 == keep {
                break;
            }
        }
        spec.rewind(keep);
        let at = 5 + keep;
        close(
            &spec.forward(&[23], at),
            &plain.forward(&[23], at),
            tolerance,
            "after the rewind",
        );
        assert_eq!(spec.draft(8, at + 1, 3), plain.draft(8, at + 1, 3));
    }

    #[test]
    fn verifying_and_rewinding_runs_the_tokens_kept() {
        let dir = std::env::temp_dir().join(format!("dwim-qwen35-verify-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.gguf");
        synthetic(&path);
        for keep in 1..=4 {
            verify_and_rewind(&path, || Cpu, keep, 1e-4);
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn verifying_and_rewinding_on_the_gpu_agrees_with_cpu() {
        let dir =
            std::env::temp_dir().join(format!("dwim-qwen35-verify-gpu-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.gguf");
        synthetic(&path);
        match Gpu::new() {
            Ok(_) => {
                // The batch may go through the tiled matmuls, which sum in
                // half precision.
                for keep in [1, 3, 4] {
                    verify_and_rewind(&path, || Gpu::new().unwrap(), keep, 2e-2);
                }
            }
            Err(e) => eprintln!("skipping: {e}"),
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn drafts_are_the_drafters_most_likely_tokens_and_a_state_carries_them() {
        let dir = std::env::temp_dir().join(format!("dwim-qwen35-draft-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.gguf");
        synthetic(&path);
        let gguf = Arc::new(Gguf::open(&path).unwrap());
        let mut model = Model::load(gguf.clone(), Cpu, 64, |_, _| {}).unwrap();
        // Nothing to draft from before any token.
        assert!(model.draft(3, 0, 4).is_empty());
        model.forward(&[3, 17, 42, 7, 9], 0);
        let drafts = model.draft(11, 5, 4);
        assert_eq!(drafts.len(), 4);
        assert!(drafts.iter().all(|&t| (t as usize) < model.config.vocab));
        // Drafting is a function of the state: the same again.
        assert_eq!(model.draft(11, 5, 4), drafts);
        // Only as many as the context has room to verify.
        assert_eq!(model.draft(11, 5, 100).len(), MAX_DRAFTS);
        let mut small = Model::load(gguf.clone(), Cpu, 8, |_, _| {}).unwrap();
        small.forward(&[3, 17, 42, 7, 9], 0);
        assert_eq!(small.draft(11, 5, 4).len(), 2);
        // A saved state carries what the drafter needs: the same drafts and
        // the same logits after it is restored.
        let state = model.save(5).unwrap();
        let mut resumed = Model::load(gguf, Cpu, 64, |_, _| {}).unwrap();
        assert_eq!(resumed.restore(&state).unwrap(), 5);
        assert_eq!(resumed.draft(11, 5, 4), drafts);
        assert_eq!(resumed.forward(&[11], 5), model.forward(&[11], 5));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Runs two sequences on the model on `device`, interleaved batch by
    /// batch, with a verified draft and a mark rewound to in one while the
    /// other waits, and returns each one's logits and drafts at the end.
    fn interleaving<D: Device>(path: &Path, device: D) -> [(Vec<f32>, Vec<u32>); 2] {
        let gguf = Arc::new(Gguf::open(path).unwrap());
        let mut model = Model::load(gguf, device, 64, |_, _| {}).unwrap();
        let other = model.add_sequence().unwrap();
        let rest: Vec<u32> = (0..40).map(|i| (i * 5 % 43) as u32).collect();
        model.forward(&[3, 17, 42, 7, 9], 0);
        model.select_sequence(other).unwrap();
        model.forward(&[1, 2, 3], 0);
        model.select_sequence(0).unwrap();
        assert!(model.mark());
        model.forward(&[11, 2, 30], 5);
        model.select_sequence(other).unwrap();
        // The mark is the first sequence's: this one has none.
        assert!(model.rewind_to_mark().is_err());
        let drafts = model.draft(4, 3, 2);
        let mut batch = vec![4];
        batch.extend(&drafts);
        model.verify(&batch, 3);
        // Not while the batch is being verified.
        assert!(model.select_sequence(0).is_err());
        model.rewind(1);
        model.select_sequence(0).unwrap();
        model.rewind_to_mark().unwrap();
        let first = (model.forward(&rest, 5), model.draft(8, 45, 3));
        model.select_sequence(other).unwrap();
        let second = (model.forward(&[5, 6], 4), model.draft(7, 6, 3));
        [first, second]
    }

    #[test]
    fn sequences_keep_their_own_state() {
        let dir = std::env::temp_dir().join(format!("dwim-qwen35-seq-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.gguf");
        synthetic(&path);
        // Each sequence alone, on a model of its own.
        let gguf = Arc::new(Gguf::open(&path).unwrap());
        let rest: Vec<u32> = (0..40).map(|i| (i * 5 % 43) as u32).collect();
        let mut alone = Model::load(gguf.clone(), Cpu, 64, |_, _| {}).unwrap();
        alone.forward(&[3, 17, 42, 7, 9], 0);
        let first = (alone.forward(&rest, 5), alone.draft(8, 45, 3));
        alone.reset();
        alone.forward(&[1, 2, 3, 4], 0);
        let second = (alone.forward(&[5, 6], 4), alone.draft(7, 6, 3));
        // The same, interleaved on one model: nothing of one sequence
        // reaches the other.
        let [got_first, got_second] = interleaving(&path, Cpu);
        assert_eq!(got_first, first);
        assert_eq!(got_second, second);
        if let Ok(gpu) = Gpu::new() {
            let [got_first, got_second] = interleaving(&path, gpu);
            close(
                &got_first.0,
                &first.0,
                2e-3,
                "the first sequence on the GPU",
            );
            close(
                &got_second.0,
                &second.0,
                2e-3,
                "the second sequence on the GPU",
            );
            assert_eq!(got_first.1, first.1);
            assert_eq!(got_second.1, second.1);
        } else {
            eprintln!("skipping the GPU sequences: no GPU");
        }
        // A sequence it does not have.
        let mut model = Model::load(gguf, Cpu, 64, |_, _| {}).unwrap();
        assert!(model.select_sequence(1).is_err());
        assert_eq!(model.add_sequence().unwrap(), 1);
        assert!(model.select_sequence(1).is_ok());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Greedy completion of a prompt by the real model, if its file is in
    /// the cache: run with `--ignored --nocapture` and compare with what
    /// llama.cpp says.
    #[test]
    #[ignore]
    fn real_model_completes_a_prompt() {
        let path = dir().unwrap().join(WEIGHTS);
        if !path.exists() {
            eprintln!("skipping: no model at {}", path.display());
            return;
        }
        real_model(&path);
    }

    fn real_model(path: &Path) {
        let gguf = Arc::new(Gguf::open(path).unwrap());
        let tokenizer = crate::inference::Tokenizer::from_gguf(&gguf).unwrap();
        let device = Gpu::new().unwrap();
        let start = std::time::Instant::now();
        let mut model = Model::load(gguf, device, 256, |_, _| {}).unwrap();
        eprintln!("loaded in {:.1}s", start.elapsed().as_secs_f32());
        let mut tokens = tokenizer.encode("The capital of France is").unwrap();
        let prompt = tokens.clone();
        // The prompt first, then the tokens one at a time, each timed on
        // its own.
        let decode = 16;
        let mut text = Vec::new();
        let start = std::time::Instant::now();
        let mut logits = model.forward(&tokens, 0);
        eprintln!(
            "prompt: {:.1} tok/s",
            tokens.len() as f32 / start.elapsed().as_secs_f32()
        );
        let start = std::time::Instant::now();
        for _ in 0..decode {
            let next = (0..logits.len())
                .max_by(|&a, &b| logits[a].total_cmp(&logits[b]))
                .unwrap() as u32;
            text.extend_from_slice(tokenizer.decode(next));
            tokens.push(next);
            logits = model.forward(&tokens[tokens.len() - 1..], tokens.len() - 1);
        }
        eprintln!(
            "decode: {:.1} tok/s over {decode} tokens",
            decode as f32 / start.elapsed().as_secs_f32()
        );
        let text = String::from_utf8_lossy(&text);
        eprintln!("completion: {text:?}");
        assert!(text.starts_with(" Paris."), "{text:?}");
        // The same completion with the drafter: greedy, so that it must
        // come out token for token the same, and faster.
        if model.config.drafter {
            let plain = tokens[prompt.len()..].to_vec();
            model.reset();
            let mut toks = prompt.clone();
            let mut logits = model.forward(&toks, 0);
            let (mut out, mut drafted, mut accepted) = (Vec::new(), 0, 0);
            let start = std::time::Instant::now();
            while out.len() < decode {
                let token = argmax(&logits);
                out.push(token);
                let pos = toks.len();
                let drafts = model.draft(token, pos, 4);
                if drafts.is_empty() {
                    logits = model.forward(&[token], pos);
                    toks.push(token);
                    continue;
                }
                let mut batch = vec![token];
                batch.extend(&drafts);
                let rows = model.verify(&batch, pos);
                let vocab = rows.len() / batch.len();
                let mut keep = 1;
                for (i, &draft) in drafts.iter().enumerate() {
                    if argmax(&rows[i * vocab..][..vocab]) != draft || out.len() >= decode {
                        break;
                    }
                    out.push(draft);
                    keep += 1;
                    accepted += 1;
                }
                drafted += drafts.len();
                model.rewind(keep);
                toks.extend(&batch[..keep]);
                logits = rows[(keep - 1) * vocab..][..vocab].to_vec();
            }
            eprintln!(
                "speculative decode: {:.1} tok/s over {} tokens, {accepted} of {drafted} drafts accepted",
                out.len() as f32 / start.elapsed().as_secs_f32(),
                out.len()
            );
            let got = String::from_utf8_lossy(
                &out.iter()
                    .flat_map(|&t| tokenizer.decode(t).to_vec())
                    .collect::<Vec<u8>>(),
            )
            .to_string();
            eprintln!("speculative completion: {got:?}");
            assert_eq!(out, plain, "{got:?}");
            tokens = toks;
        }
        // A long prompt, carrying on from the completion, for the speed of
        // the batch kernels at their size.
        let passage =
            "The quick brown fox jumps over the lazy dog while the river runs to the sea. "
                .repeat(20);
        let mut long = tokenizer.encode(&passage).unwrap();
        long.truncate(model.max_len() - tokens.len());
        let start = std::time::Instant::now();
        model.forward(&long, tokens.len());
        eprintln!(
            "long prompt: {:.1} tok/s over {} tokens",
            long.len() as f32 / start.elapsed().as_secs_f32(),
            long.len()
        );
    }

    #[test]
    fn untiling_puts_value_heads_back_by_key_head() {
        // Tiled: head j shares key head j % 2; grouped: head h shares h / 3.
        let tiled: Vec<u32> = vec![0, 1, 0, 1, 0, 1];
        let grouped = untile(&tiled, 2, 6, 1);
        assert_eq!(grouped, [0, 0, 0, 1, 1, 1]);
        let tiled: Vec<u32> = (0..6).collect();
        assert_eq!(untile(&tiled, 2, 6, 1), [0, 2, 4, 1, 3, 5]);
    }
}
