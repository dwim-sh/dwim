//! Qwen3.8-27B, the `qwen35` architecture of llama.cpp's files: a hybrid
//! transformer in which every fourth layer is full attention and the rest
//! are Gated DeltaNet linear attention, written as a sequence of operations
//! on a [`Device`]. It runs the model as Qwen released it, from a file
//! quantized as llama.cpp does, and Bonsai 2 27B, PrismML's ternary
//! compression of it; the two differ only in their weights, which the
//! loader takes in whichever format the file has them.
//!
//! Bonsai's matrices are ternary in a rotated basis: the activations a
//! matrix multiplies are first rotated by a blockwise Hadamard transform
//! with fixed signs, whose inverse was folded into the weights, and the
//! embedding table holds rotated rows that are rotated back after lookup.
//! The small recurrent-state tensors of linear attention and the norms are
//! kept in higher precision and take unrotated activations. A file without
//! the rotation's metadata is run without it.
//!
//! The file stores the value heads of linear attention in the order
//! llama.cpp's kernels want, in which value head `h` shares key head
//! `h % n_k`; they are put back into the checkpoint's order, in which value
//! head `h` shares key head `h / (n_v / n_k)`, as the loader reads them. The
//! output projection was rotated in the checkpoint's order and is left as
//! it is. A file may end with a layer that predicts a further token, which
//! llama.cpp leaves out too.

use std::sync::Arc;

use dwim_gpu::{CACHE_BLOCK, CONV_KERNEL, HADAMARD_BLOCK, cache_bytes};

use crate::{Device, Gguf, LanguageModel, Result, Tensor, rope_table};

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
    /// The rotation of the activations into the basis the weights are
    /// stored in, if the weights are rotated.
    pub rotation: Option<Rotation>,
    /// Whether the linear attention output projection's columns are in the
    /// checkpoint's grouped order of value heads, as a rotated file keeps
    /// them, rather than the tiled order llama.cpp's converter puts them
    /// in otherwise.
    pub out_grouped: bool,
    /// Whether the file holds the layer that predicts the token after the
    /// next, which the model drafts with.
    pub drafter: bool,
}

/// The blockwise Hadamard rotation PrismML folds into Bonsai's weights:
/// the signs it multiplies activations of each width by, and which weights
/// take rotated activations.
#[derive(Debug)]
pub struct Rotation {
    /// The signs, by the width of the activations they apply to.
    pub signs: Vec<(usize, Vec<f32>)>,
    /// The rotated weights, by name, sorted.
    pub weights: Vec<String>,
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
            rotation: Rotation::load(gguf)?,
            out_grouped: gguf
                .get("prism.hadamard.gdn_v_grouped")
                .and_then(|v| v.as_bool())
                == Some(true),
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
                crate::gguf::Value::Array(values) => values,
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

    /// Width of the linear attention's queries and keys per token, and of
    /// its values.
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

/// Device memory the model takes, in bytes: `fixed` for its weights,
/// activations, and recurrent states, and `per_token` more for each token
/// of context, for the key/value caches and the rotary position embedding
/// table.
#[derive(Clone, Copy, Debug)]
pub struct Memory {
    pub fixed: u64,
    pub per_token: u64,
}

impl Memory {
    /// The memory the model in `gguf` takes on a device.
    pub fn of(gguf: &Gguf) -> Result<Self> {
        let c = Config::load(gguf)?;
        // Every tensor goes to the device but the embedding table, which
        // stays in the file on the host.
        let mut weights = 0;
        for name in gguf.tensors().filter(|&name| name != "token_embd.weight") {
            weights += gguf.info(name)?.size()?;
        }
        let attention = (0..c.layers).filter(|&i| c.is_attention(i)).count();
        let linear = c.layers - attention;
        let mut activations =
            BATCH * (c.widths().iter().sum::<usize>() + 2 * c.intermediate) + c.vocab;
        let states = linear * (2 * c.conv_width() + c.ssm_width());
        // The drafter has a key/value cache of its own, and verifying a
        // batch keeps the linear layers' states run through it apart, their
        // inputs for it, the hidden state after every token, and every
        // token's logits.
        let caches = attention + c.drafter as usize;
        if c.drafter {
            activations += linear
                * (c.ssm_width() + BATCH * (2 * c.k_dim() + c.v_dim() + 2 * c.v_heads))
                + (BATCH + 2) * c.hidden
                + MAX_DRAFTS * c.vocab;
        }
        let kv_dim = c.kv_heads * c.head_dim;
        Ok(Self {
            fixed: (weights + 4 * (activations + states)) as u64,
            per_token: (2 * caches * cache_bytes(kv_dim) + 4 * c.rot_dim) as u64,
        })
    }
}

impl Rotation {
    /// Reads the rotation from the file's metadata, if the weights are
    /// rotated, refusing one the model does not implement.
    fn load(gguf: &Gguf) -> Result<Option<Self>> {
        if gguf.get("prism.hadamard.version").is_none() {
            if gguf.keys().any(|k| k.starts_with("prism.hadamard.")) {
                return Err("the weights are rotated, but the rotation has no version".into());
            }
            return Ok(None);
        }
        for (key, want) in [
            (
                "prism.hadamard.transform",
                "normalized-sylvester-walsh-hadamard",
            ),
            ("prism.hadamard.axis", "input-last-dimension"),
            ("prism.hadamard.sign_mode", "explicit"),
        ] {
            let got = gguf.str(key)?;
            if got != want {
                return Err(format!("{key} is '{got}', not '{want}'").into());
            }
        }
        if gguf.u32("prism.hadamard.version")? != 1
            || gguf.u32("prism.hadamard.block_size")? as usize != HADAMARD_BLOCK
        {
            return Err("the weights are rotated in a way the model does not implement".into());
        }
        if gguf
            .get("prism.hadamard.gdn_v_grouped")
            .and_then(|v| v.as_bool())
            != Some(true)
        {
            return Err(
                "the linear attention output projection is not in the checkpoint's head order"
                    .into(),
            );
        }
        let mut weights: Vec<String> = gguf
            .array("prism.hadamard.weight_names")?
            .iter()
            .map(|v| v.as_str().map(str::to_string).ok_or("invalid weight name"))
            .collect::<std::result::Result<_, _>>()?;
        weights.sort();
        let inverse: Vec<&str> = gguf
            .array("prism.hadamard.inverse_weight_names")?
            .iter()
            .filter_map(|v| v.as_str())
            .collect();
        if inverse != ["token_embd.weight"] {
            return Err("the embedding table is not rotated as the model expects".into());
        }
        let widths = gguf.array("prism.hadamard.sign_widths")?;
        let values = gguf.array("prism.hadamard.sign_values")?;
        let mut signs: Vec<(usize, Vec<f32>)> = Vec::new();
        let mut at = 0;
        for width in widths {
            let width = width.as_i64().ok_or("invalid sign width")? as usize;
            if signs.iter().any(|(w, _)| *w == width) {
                return Err(format!("two sets of rotation signs {width} wide").into());
            }
            let values: Vec<f32> = values
                .get(at..at + width)
                .ok_or("too few rotation signs")?
                .iter()
                .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                .collect();
            if !values.iter().all(|&s| s == 1.0 || s == -1.0) {
                return Err("rotation signs must be 1 or -1".into());
            }
            signs.push((width, values));
            at += width;
        }
        if at != values.len() {
            return Err("more rotation signs than their widths take".into());
        }
        Ok(Some(Self { signs, weights }))
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
    /// The signs of the rotation, by the width of the activations they
    /// apply to, if the weights are rotated.
    signs: Option<Vec<(usize, D::Buffer)>>,
    state: State<D>,
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
            rotated: Vec::new(),
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
            if c.rotation.is_some() {
                return Err("a drafter in rotated weights is not implemented".into());
            }
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

        // A rotated file says which weights were rotated: they must be
        // exactly the ones the forward pass rotates activations for, and
        // the signs must cover every width it rotates.
        let signs = match &config.rotation {
            Some(rotation) => {
                l.rotated.sort();
                if rotation.weights != l.rotated {
                    return Err("the rotated weights are not the ones the model expects".into());
                }
                for width in [c.hidden, c.v_dim(), c.heads * c.head_dim, c.intermediate] {
                    if !rotation.signs.iter().any(|(w, _)| *w == width) {
                        return Err(
                            format!("no rotation signs for activations {width} wide").into()
                        );
                    }
                }
                Some(
                    rotation
                        .signs
                        .iter()
                        .map(|(width, values)| {
                            let mut buf = device.alloc(*width);
                            device.write(&mut buf, values);
                            (*width, buf)
                        })
                        .collect(),
                )
            }
            None => None,
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
            signs,
            state,
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
            signs: self.signs.as_deref(),
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
            s.parity = !s.parity;
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
        d.copy(&mut s.hidden, c.hidden, &s.xh, 0, n * c.hidden);
        if verify {
            s.verified = tokens.to_vec();
            s.verified_at = pos;
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
        let first = if s.carry { 0 } else { 1 };
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
/// device, the signs of the rotation if the weights are rotated, and the
/// file for the embedding table.
struct Pass<'a, D: Device> {
    c: &'a Config,
    d: &'a D,
    signs: Option<&'a [(usize, D::Buffer)]>,
    gguf: &'a Gguf,
}

impl<'a, D: Device> Pass<'a, D> {
    /// The signs of the rotation for activations of `width`, if the
    /// weights are rotated.
    fn signs(&self, width: usize) -> Option<&'a D::Buffer> {
        self.signs.map(|signs| {
            &signs
                .iter()
                .find(|(w, _)| *w == width)
                .expect("checked at load")
                .1
        })
    }

    /// `out = rmsnorm(x) * weight` for `rows` rows of `x`, as wide as
    /// `weight`, rotated if the weights are: what a matrix multiplication's
    /// input goes through.
    fn normalize(&self, out: &mut D::Buffer, x: &D::Buffer, weight: &D::Buffer, rows: usize) {
        let (c, d) = (self.c, self.d);
        match self.signs(c.hidden) {
            Some(signs) => d.rmsnorm_hadamard(out, x, weight, signs, c.eps),
            None => {
                d.copy(out, 0, x, 0, rows * c.hidden);
                d.rmsnorm(out, weight, c.eps);
            }
        }
    }

    /// Rotates activations of `width` on their way out of a mixer or a
    /// network, if the weights are.
    fn rotate(&self, x: &mut D::Buffer, width: usize) {
        if let Some(signs) = self.signs(width) {
            self.d.hadamard(x, signs, false);
        }
    }

    /// Writes the embeddings of `tokens` into `out`, which is their size:
    /// looked up on the CPU and, if the table is rotated, rotated back on
    /// the device.
    fn embed(&self, out: &mut D::Buffer, tokens: &[u32]) {
        let c = self.c;
        let mut x = vec![0.0; tokens.len() * c.hidden];
        for (x, &token) in x.chunks_exact_mut(c.hidden).zip(tokens) {
            self.gguf
                .row("token_embd.weight", token as usize, x)
                .expect("checked at load");
        }
        self.d.write(out, &x);
        if let Some(signs) = self.signs(c.hidden) {
            self.d.hadamard(out, signs, true);
        }
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
        // stream. It sees the normalized activations, rotated if the
        // weights are, and the recurrent-state projections see them as
        // they are.
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
                d.store(&mut s.k_cache[*cache], pos * kv_dim, &s.k);
                d.store(&mut s.v_cache[*cache], pos * kv_dim, &s.v);
                d.attention(
                    &mut s.att,
                    &s.q,
                    &s.k_cache[*cache],
                    &s.v_cache[*cache],
                    pos,
                    c.heads,
                    c.head_dim,
                    c.kv_heads,
                );
                d.sigmoid_mul(&mut s.att, &s.gate);
                self.rotate(&mut s.att, q_dim);
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
                d.copy(&mut s.xb, 0, &s.x, 0, n * c.hidden);
                d.rmsnorm(&mut s.xb, &layer.attn_norm, eps);
                d.matmul(&mut s.ab, ab, &s.xb);
                if verify {
                    let width = n * (2 * k_dim + v_dim);
                    d.resize(&mut s.qkv_saved[*slot], width);
                    d.copy(&mut s.qkv_saved[*slot], 0, &s.qkv, 0, width);
                    d.resize(&mut s.ab_saved[*slot], n * 2 * c.v_heads);
                    d.copy(&mut s.ab_saved[*slot], 0, &s.ab, 0, n * 2 * c.v_heads);
                    d.copy(
                        &mut s.ssm_scratch[*slot],
                        0,
                        &s.ssm_state[*slot],
                        0,
                        c.ssm_width(),
                    );
                }
                // The convolution reads the tokens before the batch from
                // one state buffer and leaves the ones after it in the
                // other.
                let [a, b] = &mut s.conv_state[*slot];
                let (state, state_out) = if s.parity { (&*a, b) } else { (&*b, a) };
                d.conv(&mut s.q, &mut s.k, &mut s.v, state_out, &s.qkv, state, conv);
                d.l2norm(&mut s.q, c.state_dim, eps);
                d.l2norm(&mut s.k, c.state_dim, eps);
                let ssm = if verify {
                    &mut s.ssm_scratch[*slot]
                } else {
                    &mut s.ssm_state[*slot]
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
                self.rotate(&mut s.z, v_dim);
                d.matmul(&mut s.xb, out, &s.z);
            }
        }
        d.add(&mut s.x, &s.xb);

        // Feed-forward network, likewise added back.
        self.normalize(&mut s.xh, &s.x, &layer.mlp_norm, n);
        d.matmul(&mut s.up, &layer.gate, &s.xh);
        d.matmul(&mut s.gate_ffn, &layer.up, &s.xh);
        d.silu_mul(&mut s.up, &s.gate_ffn);
        self.rotate(&mut s.up, c.intermediate);
        d.matmul(&mut s.xb, &layer.down, &s.up);
        d.add(&mut s.x, &s.xb);
    }

    /// Runs the drafter at `rows` positions from `pos`: each fed the
    /// hidden state in a row of `s.hidden` from `first` on, or, chaining
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
            d.copy(&mut s.xh, 0, &s.hidden, first * c.hidden, rows * c.hidden);
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
    /// Every weight that takes rotated activations.
    rotated: Vec<String>,
}

impl<D: Device, F: FnMut(usize, usize)> Loader<'_, D, F> {
    fn tick(&mut self) {
        self.done += 1;
        (self.on_progress)(self.done, self.total);
    }

    /// Uploads a matrix that takes rotated activations, if the weights are
    /// rotated.
    fn weight(&mut self, name: &str) -> Result<D::Weight> {
        let tensor = self.gguf.matrix(name)?;
        Ok(self.rotated(name, tensor))
    }

    /// Uploads a matrix, read from the named tensor, that takes rotated
    /// activations if the weights are rotated.
    fn rotated(&mut self, name: &str, tensor: Tensor) -> D::Weight {
        self.rotated.push(name.to_string());
        let w = self.device.upload(tensor);
        self.tick();
        w
    }

    /// Uploads a matrix that takes activations as they are.
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
        let q = l.rotated(&name, both.select_rows(&rows(0)));
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
        let qkv = l.rotated(&name, qkv.select_rows(&rows));
        let name = format!("{p}.attn_gate.weight");
        let z = l.rotated(&name, g.matrix(&name)?.select_rows(&heads));
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
        // The output projection takes the value heads' outputs in
        // the checkpoint's order, or in the tiled order, in which
        // case its columns are put back into the checkpoint's too.
        let name = format!("{p}.ssm_out.weight");
        let out = g.matrix(&name)?;
        let out = if c.out_grouped {
            out
        } else {
            out.select_column_groups(c.state_dim, &untiled(c.k_heads, c.v_heads, 1))
        };
        let out = l.rotated(&name, out);
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
        if !s.verified.is_empty() || (self.draft.is_some() && !s.carry) {
            return None;
        }
        let kv_dim = c.kv_heads * c.head_dim;
        let mut out = Vec::new();
        out.extend_from_slice(STATE_MAGIC);
        out.extend_from_slice(&STATE_VERSION.to_le_bytes());
        for value in [
            len,
            s.k_cache.len(),
            s.ssm_state.len(),
            kv_dim,
            c.conv_width(),
            c.ssm_width(),
            if self.draft.is_some() { c.hidden } else { 0 },
        ] {
            out.extend_from_slice(&(value as u64).to_le_bytes());
        }
        for cache in s.k_cache.iter().chain(&s.v_cache) {
            out.extend(d.read_cache(cache, len * kv_dim));
        }
        for slot in 0..s.ssm_state.len() {
            // The convolution reads the batch before from one buffer of the
            // pair or the other, by the parity.
            let conv = &s.conv_state[slot][if s.parity { 0 } else { 1 }];
            for buf in [conv, &s.ssm_state[slot]] {
                out.extend(d.read(buf).iter().flat_map(|v| v.to_le_bytes()));
            }
        }
        if self.draft.is_some() {
            out.extend(
                d.read(&s.hidden)[..c.hidden]
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
                s.k_cache.len(),
                s.ssm_state.len(),
                kv_dim,
                c.conv_width(),
                c.ssm_width(),
                if self.draft.is_some() { c.hidden } else { 0 },
            ]
        {
            return Err("a saved state of another model".into());
        }
        if len > s.max_len {
            return Err(
                format!("a saved state of {len} tokens does not fit in the context").into(),
            );
        }
        for cache in s.k_cache.iter_mut().chain(&mut s.v_cache) {
            d.write_cache(cache, take(cache_bytes(len * kv_dim))?);
        }
        for slot in 0..s.ssm_state.len() {
            let [conv, _] = &mut s.conv_state[slot];
            for (buf, width) in [
                (conv, c.conv_width()),
                (&mut s.ssm_state[slot], c.ssm_width()),
            ] {
                let floats: Vec<f32> = take(width * 4)?
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
                d.write(buf, &floats);
            }
        }
        s.parity = true;
        s.verified.clear();
        if self.draft.is_some() {
            let floats: Vec<f32> = take(c.hidden * 4)?
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            d.resize(&mut s.xb, c.hidden);
            d.write(&mut s.xb, &floats);
            d.copy(&mut s.hidden, 0, &s.xb, 0, c.hidden);
            s.carry = true;
        }
        Ok(len)
    }

    /// Adds the tokens to the model's state. Tokens run through the model
    /// in batches of up to [`BATCH`].
    fn forward(&mut self, tokens: &[u32], pos: usize) -> Vec<f32> {
        assert!(!tokens.is_empty(), "no tokens to run");
        assert!(
            pos + tokens.len() <= self.state.max_len,
            "tokens past the end of the cache"
        );
        assert!(self.state.verified.is_empty(), "a batch is being verified");
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
            signs: self.signs.as_deref(),
            gguf: &self.gguf,
        };
        let (c, d) = (p.c, p.d);
        let s = &mut self.state;
        let k = k.min(MAX_DRAFTS).min(s.max_len.saturating_sub(pos + 1));
        if !s.carry || !s.verified.is_empty() || pos == 0 || k == 0 {
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
            self.state.verified.is_empty() && self.state.carry,
            "a batch is being verified, or none was run before"
        );
        assert!(
            !tokens.is_empty() && tokens.len() <= MAX_DRAFTS + 1,
            "a batch of {} tokens is not one to verify",
            tokens.len()
        );
        assert!(
            pos + tokens.len() <= self.state.max_len,
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
            signs: self.signs.as_deref(),
            gguf: &self.gguf,
        };
        let (c, d) = (p.c, p.d);
        let s = &mut self.state;
        let n = s.verified.len();
        assert!(
            n > 0 && (1..=n).contains(&keep),
            "keeping {keep} tokens of a batch of {n} being verified"
        );
        let tokens = std::mem::take(&mut s.verified);
        if keep == n {
            for slot in 0..s.ssm_state.len() {
                std::mem::swap(&mut s.ssm_state[slot], &mut s.ssm_scratch[slot]);
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
                let [a, b] = &mut s.conv_state[*slot];
                let (state, state_out) = if s.parity { (&*a, b) } else { (&*b, a) };
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
                    &mut s.ssm_state[*slot],
                    c.k_heads,
                    c.v_heads,
                    c.state_dim,
                );
            }
        }
        s.parity = !s.parity;
        if let Some(draft) = &self.draft {
            p.drafter(s, draft, keep, s.verified_at - 1, 0, false, &tokens[..keep]);
            s.carry_over(d, keep);
        }
    }

    /// Zeroes the linear layers' states, as they were before any token,
    /// and forgets what was carried over for the drafter. The attention
    /// layers' caches need no clearing: attention reads only the positions
    /// before the token's.
    fn reset(&mut self) {
        let (c, d, s) = (&self.config, &self.device, &mut self.state);
        for slot in 0..s.ssm_state.len() {
            for conv in &mut s.conv_state[slot] {
                d.write(conv, &vec![0.0; c.conv_width()]);
            }
            d.write(&mut s.ssm_state[slot], &vec![0.0; c.ssm_width()]);
        }
        s.parity = false;
        s.carry = false;
        s.verified.clear();
    }

    fn max_len(&self) -> usize {
        self.state.max_len
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
    k_cache: Vec<D::Cache>,
    v_cache: Vec<D::Cache>,
    /// Two per linear layer, the batch's inputs coming from one and its
    /// outputs going to the other, swapping every batch.
    conv_state: Vec<[D::Buffer; 2]>,
    parity: bool,
    ssm_state: Vec<D::Buffer>,
    rope: D::Buffer,
    max_len: usize,
    /// For the drafter, which runs a position behind the model: the
    /// normalized hidden state after each token of the latest batch, in
    /// the rows after the first, which holds the one after the token
    /// before the batch, carried over; whether it holds one; and the
    /// drafter's own output, from one drafted token to the next.
    hidden: D::Buffer,
    carry: bool,
    drafted: D::Buffer,
    /// A batch being verified: its tokens and the position of the first;
    /// the linear layers' recurrent states run through it, kept apart from
    /// the real ones until the accepted tokens are committed; and each
    /// linear layer's convolution inputs and gates for it, to commit from.
    verified: Vec<u32>,
    verified_at: usize,
    ssm_scratch: Vec<D::Buffer>,
    qkv_saved: Vec<D::Buffer>,
    ab_saved: Vec<D::Buffer>,
}

impl<D: Device> State<D> {
    /// Allocates state for sequences of up to `max_len` tokens.
    fn new(c: &Config, d: &D, max_len: usize, caches: usize, slots: usize) -> Self {
        let kv_dim = c.kv_heads * c.head_dim;
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
            k_cache: (0..caches)
                .map(|_| d.alloc_cache(max_len * kv_dim))
                .collect(),
            v_cache: (0..caches)
                .map(|_| d.alloc_cache(max_len * kv_dim))
                .collect(),
            conv_state: (0..slots)
                .map(|_| [d.alloc(conv_width), d.alloc(conv_width)])
                .collect(),
            parity: false,
            ssm_state: (0..slots)
                .map(|_| d.alloc(c.v_heads * c.state_dim * c.state_dim))
                .collect(),
            rope,
            max_len,
            hidden: d.alloc((BATCH + 1) * c.hidden),
            carry: false,
            drafted: d.alloc(c.hidden),
            verified: Vec::new(),
            verified_at: 0,
            ssm_scratch: (0..if c.drafter { slots } else { 0 })
                .map(|_| d.alloc(c.ssm_width()))
                .collect(),
            qkv_saved: (0..if c.drafter { slots } else { 0 })
                .map(|_| d.alloc(BATCH * (2 * c.k_dim() + c.v_dim())))
                .collect(),
            ab_saved: (0..if c.drafter { slots } else { 0 })
                .map(|_| d.alloc(BATCH * 2 * c.v_heads))
                .collect(),
        }
    }

    /// Carries the hidden state after the last of `n` tokens over into
    /// the first row, for the drafter, through `xb`.
    fn carry_over(&mut self, d: &D, n: usize) {
        let hidden = self.widths[0];
        d.resize(&mut self.xb, hidden);
        d.copy(&mut self.xb, 0, &self.hidden, n * hidden, hidden);
        d.copy(&mut self.hidden, 0, &self.xb, 0, hidden);
        self.carry = true;
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
    use super::*;
    use crate::{
        gguf::{self, Value},
        quants, ternary,
    };
    use dwim_gpu::{Cpu, Gpu};
    use std::path::Path;

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

        fn bf16(&mut self, n: usize, scale: f32) -> Vec<u8> {
            (0..n)
                .flat_map(|_| (((self.next() * scale).to_bits() >> 16) as u16).to_le_bytes())
                .collect()
        }

        fn ternary(&mut self, rows: usize, cols: usize, scale: f32) -> Vec<u8> {
            let mut data = vec![0; rows * ternary::row_bytes(cols)];
            for row in data.chunks_exact_mut(ternary::row_bytes(cols)) {
                let values: Vec<f32> = (0..cols).map(|_| self.next() * scale).collect();
                ternary::quantize_row(&values, row);
            }
            data
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

    /// Writes a tiny random model in the file format, four layers of which
    /// the last is full attention: with ternary weights in a rotated basis
    /// as Bonsai's file has them, or `quantized` as Qwen's file is, with
    /// Q4_0 and Q8_0 weights, f32 decay projections, and a further layer
    /// that predicts the next token.
    fn synthetic(path: &Path, quantized: bool) {
        let (hidden, inter, layers, heads, kv_heads, head_dim, rot_dim) =
            (1024, 1024, 4, 4, 2, 256, 64);
        let (k_heads, v_heads, state_dim, vocab) = (2, 8, 128, 64);
        let (k_dim, v_dim) = (k_heads * state_dim, v_heads * state_dim);
        let mut rng = Rng(42);
        let mut signs = Vec::new();
        for _ in 0..hidden {
            signs.push(Value::I32(if rng.next() < 0.0 { -1 } else { 1 }));
        }
        let mut names = Vec::new();
        let mut tensors: Vec<(String, Vec<usize>, u32, Vec<u8>)> = Vec::new();
        // Weights are small so that the activations stay sane through the
        // layers; ternary scales are their largest magnitude. The down and
        // output projections of a quantized file are Q8_0, the rest Q4_0.
        let mut tern = |name: &str, rows: usize, cols: usize, names: &mut Vec<Value>| {
            names.push(Value::Str(name.to_string()));
            let scale = 1.0 / (cols as f32).sqrt();
            let (dtype, data) = if !quantized {
                (gguf::PTQ1_0, rng.ternary(rows, cols, scale))
            } else if name.contains("ffn_down") || name.contains("ssm_out") {
                (gguf::Q8_0, rng.packed(rows, cols, scale, gguf::Q8_0))
            } else {
                (gguf::Q4_0, rng.packed(rows, cols, scale, gguf::Q4_0))
            };
            tensors.push((name.to_string(), vec![cols, rows], dtype, data));
        };
        tern("output.weight", vocab, hidden, &mut names);
        // A quantized file has a block after the layers that predicts the
        // token after the next: a full attention layer, with the projection
        // of an embedding and a hidden state into its input.
        let blocks = if quantized { layers + 1 } else { layers };
        for i in 0..blocks {
            let p = format!("blk.{i}");
            if i == layers {
                tern(
                    &format!("{p}.nextn.eh_proj.weight"),
                    hidden,
                    2 * hidden,
                    &mut names,
                );
            }
            if (i + 1) % 4 == 0 || i == layers {
                tern(
                    &format!("{p}.attn_q.weight"),
                    heads * head_dim * 2,
                    hidden,
                    &mut names,
                );
                tern(
                    &format!("{p}.attn_k.weight"),
                    kv_heads * head_dim,
                    hidden,
                    &mut names,
                );
                tern(
                    &format!("{p}.attn_v.weight"),
                    kv_heads * head_dim,
                    hidden,
                    &mut names,
                );
                tern(
                    &format!("{p}.attn_output.weight"),
                    hidden,
                    heads * head_dim,
                    &mut names,
                );
            } else {
                tern(
                    &format!("{p}.attn_qkv.weight"),
                    2 * k_dim + v_dim,
                    hidden,
                    &mut names,
                );
                tern(&format!("{p}.attn_gate.weight"), v_dim, hidden, &mut names);
                tern(&format!("{p}.ssm_out.weight"), hidden, v_dim, &mut names);
            }
            tern(&format!("{p}.ffn_gate.weight"), inter, hidden, &mut names);
            tern(&format!("{p}.ffn_up.weight"), inter, hidden, &mut names);
            tern(&format!("{p}.ffn_down.weight"), hidden, inter, &mut names);
        }
        let mut rng = Rng(7);
        tensors.push(if quantized {
            (
                "token_embd.weight".into(),
                vec![hidden, vocab],
                gguf::Q4_0,
                rng.packed(vocab, hidden, 0.5, gguf::Q4_0),
            )
        } else {
            (
                "token_embd.weight".into(),
                vec![hidden, vocab],
                gguf::PTQ1_0,
                rng.ternary(vocab, hidden, 0.5),
            )
        });
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
                    tensors.push(if quantized {
                        (
                            format!("{p}.{name}.weight"),
                            vec![hidden, v_heads],
                            gguf::F32,
                            rng.floats(v_heads * hidden, 0.05, 0.0),
                        )
                    } else {
                        (
                            format!("{p}.{name}.weight"),
                            vec![hidden, v_heads],
                            gguf::BF16,
                            rng.bf16(v_heads * hidden, 0.05),
                        )
                    });
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
            (
                "qwen35.block_count",
                n(if quantized { layers + 1 } else { layers }),
            ),
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
        if quantized {
            // The last block predicts the token after the next; the file
            // also lists which layers are linear.
            meta.push(("qwen35.nextn_predict_layers", n(1)));
            let recurrent = (0..layers + 1).map(|i| Value::Bool((i + 1) % 4 != 0 && i < layers));
            meta.push((
                "qwen35.attention.recurrent_layers",
                Value::Array(recurrent.collect()),
            ));
        } else {
            meta.extend([
                ("prism.hadamard.version", n(1)),
                ("prism.hadamard.block_size", n(HADAMARD_BLOCK)),
                (
                    "prism.hadamard.transform",
                    Value::Str("normalized-sylvester-walsh-hadamard".into()),
                ),
                (
                    "prism.hadamard.axis",
                    Value::Str("input-last-dimension".into()),
                ),
                ("prism.hadamard.sign_mode", Value::Str("explicit".into())),
                ("prism.hadamard.weight_names", Value::Array(names)),
                ("prism.hadamard.sign_widths", Value::Array(vec![n(hidden)])),
                ("prism.hadamard.sign_values", Value::Array(signs)),
                (
                    "prism.hadamard.inverse_weight_names",
                    Value::Array(vec![Value::Str("token_embd.weight".into())]),
                ),
                ("prism.hadamard.gdn_v_grouped", Value::Bool(true)),
            ]);
        }
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
        synthetic(&path, false);
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

    #[test]
    fn gpu_agrees_with_cpu() {
        for quantized in [false, true] {
            let dir = std::env::temp_dir().join(format!(
                "dwim-qwen35-{}-{}",
                if quantized { "quantized" } else { "ternary" },
                std::process::id()
            ));
            std::fs::create_dir_all(&dir).unwrap();
            let path = dir.join("model.gguf");
            synthetic(&path, quantized);
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
                        assert!(
                            (a - b).abs() <= tolerance * (1.0 + a.abs()),
                            "quantized {quantized}: {a} vs {b}"
                        );
                    }
                }
            } else {
                eprintln!("skipping the GPU comparison: no GPU");
            }
            std::fs::remove_dir_all(&dir).unwrap();
        }
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
        synthetic(&path, true);
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
        synthetic(&path, true);
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
        synthetic(&path, true);
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

    /// Greedy completion of a prompt by each real model whose file is in
    /// the cache: run with `--ignored --nocapture` and compare with what
    /// llama.cpp says.
    #[test]
    #[ignore]
    fn real_model_completes_a_prompt() {
        for file in [
            "bonsai-2-27b/Ternary-Bonsai-2-27B-PTQ1_0.gguf",
            "qwen3.8-27b/Qwen3.8-27B-Q4_0.gguf",
        ] {
            let path = dirs::cache_dir().unwrap().join("dwim/models").join(file);
            if !path.exists() {
                eprintln!("skipping: no model at {}", path.display());
                continue;
            }
            eprintln!("{file}");
            real_model(&path);
        }
    }

    fn real_model(path: &Path) {
        let gguf = Arc::new(Gguf::open(path).unwrap());
        let tokenizer = crate::Tokenizer::from_gguf(&gguf).unwrap();
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
        // The same completion with the drafter, where the model has one:
        // greedy, so that it must come out token for token the same, and
        // faster.
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
