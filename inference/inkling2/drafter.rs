//! Inkling 2 Mini's drafter: a network of five layers that drafts the
//! tokens after the last one picked, a block of them in one pass, for the
//! model to verify as a batch. It is a file of its own, DFlash 2 of Inco
//! AI's, of the `dflash` architecture as llama.cpp names it, and borrows
//! the model's embeddings and its output projection. What it computes is
//! what llama.cpp's graph for it does (ggml-org/llama.cpp#33073).
//!
//! The drafter reads the model, not the text: as the model runs a token,
//! its residual stream on the way into a few of its layers and its
//! normalized output are kept, the token's *features*. One projection
//! fuses them, and each of the drafter's layers projects the fusion to a
//! key and a value with the matrices it has for its own, which join the
//! layer's caches at the token's position: the *context*.
//!
//! To draft after a token, the drafter runs a block of eight rows at the
//! positions from the token's on: the token's embedding, and seven of the
//! mask token's. Each layer is attention and a SwiGLU feed-forward
//! network, both on a normalized input. The rows attend to each other in
//! both directions, and to the context before them, over a window of the
//! latest positions. Queries and keys are normalized a head at a time and
//! rotated by their position. The rows are drafted at once, so nothing
//! of a row's token reaches the next but through a convolution: the input
//! and the output of attention and of the network each go through one
//! over the row and the row before, whose taps the row's own input sets.
//!
//! A row's output through the model's output projection is its logits,
//! and its sixteen likeliest tokens its candidates. A selector then goes
//! down the rows and picks one of each row's: a candidate's score is its
//! logit and a correction for the token picked before it, the dot product
//! of a vector of the candidate's with one of that token's, weighed by a
//! projection of the row's output. The scores, capped softly as the
//! model's logits are, are the selector's logits for the row, and their
//! softmax says how sure it is of its pick.
//!
//! A block costs a pass over the drafter and one over the output
//! projection, a gigabyte in all, however many of its rows are used, and
//! a token verified costs the weights of its own experts: the model asks
//! for the drafts the selector is sure enough of.

use std::sync::Arc;

use crate::{
    gpu::{CACHE_BLOCK, Device, cache_bytes},
    inference::{Gguf, Memory, Result},
};

/// What a drafter's file names its architecture.
pub const ARCHITECTURE: &str = "dflash";

/// The tensors that stay in the file on the host: the selector's vectors
/// of each token, a row of which is looked up as it is needed, and the
/// norm of the model's embeddings, which the model has.
const HOST: [&str; 3] = [
    "selector_predecessor.weight",
    "selector_successor.weight",
    "token_embd_norm.weight",
];

/// Width of the bias by distance the attention is given, which the
/// drafter has none of: zeros.
const REL: usize = 16;

/// A sink's score that takes nothing from the keys.
const NO_SINK: f32 = -1e9;

/// The drafter's architecture, read from its file.
#[derive(Debug)]
pub struct Config {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    /// Width of a feed-forward network inside.
    pub ff: usize,
    /// Rows of a block: the token drafted after, and the drafts.
    pub block: usize,
    /// The convolutions: how many rows one reaches over, and how many
    /// channels share a tap of a row's own.
    pub kernel: usize,
    pub group: usize,
    /// The selector: how wide its vectors are, and how many candidates a
    /// row has.
    pub rank: usize,
    pub candidates: usize,
    /// What the selector's scores are capped to, softly.
    pub softcap: f32,
    /// Which layers attend over a window of the latest positions, and how
    /// far back it reaches.
    pub local: Vec<bool>,
    pub window: usize,
    /// The layers of the model whose input is a feature, counted as the
    /// model runs them, the last being its output.
    pub targets: Vec<usize>,
    pub eps: f32,
    pub rope_base: f32,
    /// The token a row not yet drafted holds.
    pub mask: u32,
    /// Tokens of the vocabulary.
    pub vocab: usize,
}

impl Config {
    pub fn load(gguf: &Gguf) -> Result<Self> {
        let arch = gguf.str("general.architecture")?;
        if arch != ARCHITECTURE {
            return Err(format!("the '{arch}' architecture is not a drafter's").into());
        }
        let key = |name: &str| gguf.u32(&format!("{arch}.{name}")).map(|v| v as usize);
        let local = gguf
            .array(&format!("{arch}.attention.sliding_window_pattern"))?
            .iter()
            .map(|v| v.as_bool().ok_or("the window pattern is not of booleans"))
            .collect::<std::result::Result<Vec<bool>, _>>()?;
        let targets = gguf
            .array(&format!("{arch}.target_layers"))?
            .iter()
            .map(|v| {
                v.as_i64()
                    .map(|v| v as usize)
                    .ok_or("a target layer is not a number")
            })
            .collect::<std::result::Result<Vec<usize>, _>>()?;
        // The selector has a vector for each token of the vocabulary.
        let vocab = match gguf.info("selector_successor.weight") {
            Ok(info) if info.dims.len() == 2 => info.dims[1],
            _ => return Err("the drafter has no selector: not one dwim runs".into()),
        };
        let config = Self {
            hidden: key("embedding_length")?,
            layers: key("block_count")?,
            heads: key("attention.head_count")?,
            kv_heads: key("attention.head_count_kv")?,
            head_dim: key("attention.key_length")?,
            ff: key("feed_forward_length")?,
            block: key("block_size")?,
            kernel: key("conv_kernel_size")?,
            group: key("conv_group_size")?,
            rank: key("selector_rank")?,
            candidates: key("selector_top_k")?,
            softcap: gguf.f32(&format!("{arch}.final_logit_softcapping"))?,
            local,
            window: key("attention.sliding_window")?,
            targets,
            eps: gguf.f32(&format!("{arch}.attention.layer_norm_rms_epsilon"))?,
            rope_base: gguf.f32(&format!("{arch}.rope.freq_base"))?,
            mask: gguf.u32("tokenizer.ggml.mask_token_id")?,
            vocab,
        };
        let c = &config;
        // The scores are capped once the correction is added, as Inkling's
        // drafter was trained: one that caps the logits before it is of
        // another model.
        let capped_after = gguf
            .get(&format!("{arch}.selector_softcap_after_correction"))
            .and_then(|v| v.as_bool())
            == Some(true);
        if c.local.len() != c.layers
            || c.targets.len() < 2
            || !c.targets.is_sorted()
            || c.block < 2
            || c.kernel == 0
            || c.group == 0
            || !c.hidden.is_multiple_of(c.group)
            || c.candidates == 0
            || c.candidates > c.vocab
            || c.mask as usize >= c.vocab
            || c.softcap <= 0.0
            || !capped_after
            || c.kv_heads == 0
            || !c.heads.is_multiple_of(c.kv_heads)
            || !c.head_dim.is_multiple_of(CACHE_BLOCK)
        {
            return Err("the drafter's dimensions are not ones dwim runs".into());
        }
        Ok(config)
    }

    fn q_dim(&self) -> usize {
        self.heads * self.head_dim
    }

    fn kv_dim(&self) -> usize {
        self.kv_heads * self.head_dim
    }

    /// Taps a row has of its own for a convolution's input, and as many
    /// again for its output.
    fn taps(&self) -> usize {
        self.kernel * (self.hidden / self.group)
    }

    /// The device memory the drafter in `gguf`, configured as this, takes
    /// beside the model, for batches of `most` tokens: `per_sequence` for
    /// the windowed layers' keys and values, and `per_token` for those of
    /// the layers that attend over everything.
    pub fn memory(&self, gguf: &Gguf, most: usize) -> Result<Memory> {
        let c = self;
        let mut weights = 0;
        for name in gguf.tensors().filter(|name| !HOST.contains(name)) {
            weights += gguf.info(name)?.size()?;
        }
        let kv = 2 * cache_bytes(c.kv_dim());
        let local = c.local.iter().filter(|&&local| local).count();
        let rings = local * c.ring(most, usize::MAX) * kv;
        let batch = most * (3 * c.hidden + 3 * c.kv_dim() + c.head_dim);
        let block = c.block * (6 * c.hidden + 2 * c.taps() + 2 * c.q_dim() + 2 * c.ff + c.rank)
            + 2 * c.q_dim();
        Ok(Memory {
            // And the copy a mark keeps of a sequence's rings.
            fixed: (weights + 4 * (batch + block) + rings) as u64,
            per_sequence: rings as u64,
            per_token: ((c.layers - local) * kv) as u64,
        })
    }

    /// Positions a windowed layer's caches hold: the window and a block,
    /// and a batch more of context stored at once. No more than a
    /// sequence is long.
    fn ring(&self, batch: usize, max_len: usize) -> usize {
        (self.window + self.block + batch).min(max_len)
    }
}

/// The drafter on a device, with the buffers it computes in.
pub struct Drafter<D: Device> {
    pub config: Config,
    /// The file, kept mapped for the selector's vectors of each token,
    /// which are looked up on the CPU as the model's embeddings are.
    gguf: Arc<Gguf>,
    /// The fusion of the features: its columns for each feature, a matrix
    /// of their own, and the norm of what they add up to.
    fuse: Vec<D::Weight>,
    fuse_norm: D::Buffer,
    layers: Vec<Layer<D>>,
    out_norm: D::Buffer,
    /// The selector's projection of a row's output.
    selector: D::Weight,
    /// The embedding of the mask token, normalized as the model's input
    /// is.
    mask: Vec<f32>,
    s: Scratch<D>,
}

struct Layer<D: Device> {
    attn_norm: D::Buffer,
    attn_conv: Conv<D>,
    q: D::Weight,
    k: D::Weight,
    v: D::Weight,
    o: D::Weight,
    q_norm: D::Buffer,
    k_norm: D::Buffer,
    ffn_norm: D::Buffer,
    ffn_conv: Conv<D>,
    gate: D::Weight,
    up: D::Weight,
    down: D::Weight,
}

/// The convolutions around attention or a feed-forward network: the
/// projection of the normalized input to the taps a row has of its own,
/// for the input and then for the output, and the taps every row has, a
/// row of channels for each row a convolution reaches back.
struct Conv<D: Device> {
    taps: D::Weight,
    before: D::Buffer,
    after: D::Buffer,
}

/// What the drafter computes in: for a batch of the model's tokens whose
/// features become context, and for a block.
struct Scratch<D: Device> {
    /// Most tokens of context stored at once.
    most: usize,
    fused: D::Buffer,
    part: D::Buffer,
    normed: D::Buffer,
    k: D::Buffer,
    v: D::Buffer,
    wrapped: D::Buffer,
    /// The cosines and sines that rotate the positions being run.
    table: D::Buffer,
    /// The block: its stream, the same normalized, a row's own taps, what
    /// a convolution gives, and a layer's output.
    x: D::Buffer,
    xn: D::Buffer,
    taps: D::Buffer,
    conv: D::Buffer,
    out: D::Buffer,
    q: D::Buffer,
    att: D::Buffer,
    /// A row's queries and what it attends to, for attention a row at a
    /// time.
    q_row: D::Buffer,
    att_row: D::Buffer,
    h: D::Buffer,
    h_scratch: D::Buffer,
    /// The outputs of the rows drafted, and the selector's projection of
    /// them.
    rows: D::Buffer,
    weighed: D::Buffer,
    /// No bias by distance, and no sink.
    rel: D::Buffer,
    bias: D::Buffer,
    sinks: D::Buffer,
}

/// A sequence's context: each layer's keys and values, a windowed layer's
/// in a ring, and how many positions there are context for.
pub struct Context<D: Device> {
    k: Vec<D::Cache>,
    v: Vec<D::Cache>,
    rows: Vec<usize>,
    /// Positions there is context for, or [`BROKEN`].
    pub len: usize,
}

/// A length no sequence has: of a context that is not whole, which the
/// drafter does not draft from.
pub const BROKEN: usize = usize::MAX;

/// What a mark keeps of a context: the windowed layers' rings, which are
/// written over the positions before the mark as the sequence goes on, and
/// how long the context was.
pub struct Mark<D: Device> {
    rings: Vec<D::Cache>,
    len: usize,
}

/// A token drafted, and how sure the selector is of it: the probability
/// its scores give the token among the row's candidates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Draft {
    pub token: u32,
    pub sure: f32,
}

impl<D: Device> Drafter<D> {
    /// Loads the drafter from its GGUF file onto `device`, with buffers
    /// for `most` tokens of context at once. `mask` is the embedding of
    /// the file's mask token, normalized as the model's input is.
    pub fn load(gguf: Arc<Gguf>, device: &D, most: usize, mask: Vec<f32>) -> Result<Self> {
        let config = Config::load(&gguf)?;
        let c = &config;
        let buffer = |values: &[f32]| {
            let mut buf = device.alloc(values.len());
            device.write(&mut buf, values);
            buf
        };
        let floats = |name: &str, len: usize| -> Result<Vec<f32>> {
            let values = gguf.f32s(name)?;
            if values.len() != len {
                return Err(format!("{name} is not {len} values").into());
            }
            Ok(values)
        };
        let vector =
            |name: &str, len: usize| -> Result<D::Buffer> { Ok(buffer(&floats(name, len)?)) };
        // A matrix of `rows` outputs of `cols` inputs.
        let weight = |name: &str, rows: usize, cols: usize| -> Result<D::Weight> {
            let matrix = gguf.matrix(name)?;
            if matrix.shape() != [rows, cols] {
                return Err(format!("{name} is not {rows} rows of {cols}").into());
            }
            Ok(device.upload(matrix))
        };
        // The taps every row has, which the file holds for the input and
        // then for the output: a row of channels for each row back.
        let conv = |p: &str, name: &str| -> Result<Conv<D>> {
            let base = floats(&format!("{p}.{name}_conv_base"), 2 * c.kernel * c.hidden)?;
            let (before, after) = base.split_at(c.kernel * c.hidden);
            Ok(Conv {
                taps: weight(
                    &format!("{p}.{name}_conv_proj.weight"),
                    2 * c.taps(),
                    c.hidden,
                )?,
                before: buffer(before),
                after: buffer(after),
            })
        };
        let (hidden, q_dim, kv_dim) = (c.hidden, c.q_dim(), c.kv_dim());
        let mut layers = Vec::with_capacity(c.layers);
        for i in 0..c.layers {
            let p = format!("blk.{i}");
            layers.push(Layer {
                attn_norm: vector(&format!("{p}.attn_norm.weight"), hidden)?,
                attn_conv: conv(&p, "attn")?,
                q: weight(&format!("{p}.attn_q.weight"), q_dim, hidden)?,
                k: weight(&format!("{p}.attn_k.weight"), kv_dim, hidden)?,
                v: weight(&format!("{p}.attn_v.weight"), kv_dim, hidden)?,
                o: weight(&format!("{p}.attn_output.weight"), hidden, q_dim)?,
                q_norm: vector(&format!("{p}.attn_q_norm.weight"), c.head_dim)?,
                k_norm: vector(&format!("{p}.attn_k_norm.weight"), c.head_dim)?,
                ffn_norm: vector(&format!("{p}.ffn_norm.weight"), hidden)?,
                ffn_conv: conv(&p, "ffn")?,
                gate: weight(&format!("{p}.ffn_gate.weight"), c.ff, hidden)?,
                up: weight(&format!("{p}.ffn_up.weight"), c.ff, hidden)?,
                down: weight(&format!("{p}.ffn_down.weight"), hidden, c.ff)?,
            });
        }
        // The fusion takes the features one after another in a row: its
        // columns for each are a matrix of that feature alone.
        let fusion = gguf.matrix("fc.weight")?;
        if fusion.shape() != [hidden, c.targets.len() * hidden] {
            return Err("the drafter's fusion is not of a feature for each target".into());
        }
        let fuse = (0..c.targets.len())
            .map(|feature| device.upload(fusion.select_column_groups(hidden, &[feature])))
            .collect();
        for name in ["selector_predecessor.weight", "selector_successor.weight"] {
            if gguf.info(name)?.dims != [c.rank, c.vocab] {
                return Err(format!("{name} is not a vector of the selector's a token").into());
            }
        }
        if mask.len() != hidden {
            return Err("the mask token's embedding is not as wide as the drafter".into());
        }
        let block = c.block;
        let s = Scratch {
            most,
            fused: device.alloc(most * hidden),
            part: device.alloc(most * hidden),
            normed: device.alloc(most * hidden),
            k: device.alloc(most.max(block) * kv_dim),
            v: device.alloc(most.max(block) * kv_dim),
            wrapped: device.alloc(most.max(block) * kv_dim),
            table: device.alloc(most.max(block) * c.head_dim),
            x: device.alloc(block * hidden),
            xn: device.alloc(block * hidden),
            taps: device.alloc(block * 2 * c.taps()),
            conv: device.alloc(block * hidden),
            out: device.alloc(block * hidden),
            q: device.alloc(block * q_dim),
            att: device.alloc(block * q_dim),
            q_row: device.alloc(q_dim),
            att_row: device.alloc(q_dim),
            h: device.alloc(block * c.ff),
            h_scratch: device.alloc(block * c.ff),
            rows: device.alloc(block * hidden),
            weighed: device.alloc(block * c.rank),
            rel: buffer(&vec![0.0; c.heads * REL]),
            bias: buffer(&[0.0; REL]),
            sinks: buffer(&vec![NO_SINK; c.heads]),
        };
        Ok(Self {
            fuse,
            fuse_norm: vector("enc.output_norm.weight", hidden)?,
            out_norm: vector("output_norm.weight", hidden)?,
            selector: weight("selector_hidden.weight", c.rank, hidden)?,
            layers,
            mask,
            s,
            gguf,
            config,
        })
    }

    /// A sequence's context, empty, with room for `max_len` positions.
    pub fn context(&self, device: &D, max_len: usize) -> Context<D> {
        let c = &self.config;
        let rows: Vec<usize> = c
            .local
            .iter()
            .map(|&local| match local {
                true => c.ring(self.s.most, max_len),
                false => max_len,
            })
            .collect();
        let caches = || -> Vec<D::Cache> {
            rows.iter()
                .map(|rows| device.alloc_cache(rows * c.kv_dim()))
                .collect()
        };
        Context {
            k: caches(),
            v: caches(),
            rows,
            len: 0,
        }
    }

    /// Room for what a mark keeps of a context.
    pub fn mark(&self, device: &D, context: &Context<D>) -> Mark<D> {
        let c = &self.config;
        let local = context
            .rows
            .iter()
            .zip(&c.local)
            .filter(|(_, local)| **local);
        Mark {
            rings: local
                .flat_map(|(rows, _)| [rows, rows])
                .map(|rows| device.alloc_cache(rows * c.kv_dim()))
                .collect(),
            len: BROKEN,
        }
    }

    /// Copies what a mark keeps of `context` into `mark`.
    pub fn keep(&self, d: &D, context: &Context<D>, mark: &mut Mark<D>) {
        let c = &self.config;
        let mut rings = mark.rings.iter_mut();
        for i in (0..c.layers).filter(|&i| c.local[i]) {
            let len = context.rows[i] * c.kv_dim();
            for cache in [&context.k[i], &context.v[i]] {
                d.copy_cache(rings.next().expect("a copy for each ring"), cache, len);
            }
        }
        mark.len = context.len;
    }

    /// Puts `context` back as `mark` kept it.
    pub fn put_back(&self, d: &D, context: &mut Context<D>, mark: &Mark<D>) {
        let c = &self.config;
        let mut rings = mark.rings.iter();
        for i in (0..c.layers).filter(|&i| c.local[i]) {
            let len = context.rows[i] * c.kv_dim();
            for cache in [&mut context.k[i], &mut context.v[i]] {
                d.copy_cache(cache, rings.next().expect("a copy for each ring"), len);
            }
        }
        context.len = mark.len;
    }

    /// Appends the context of the first `len` positions to `out`: whether
    /// it is whole, which it is if the context ends there, then each
    /// layer's keys and values as the caches hold them.
    pub fn save(&self, d: &D, context: &Context<D>, len: usize, out: &mut Vec<u8>) {
        let c = &self.config;
        out.push((context.len == len) as u8);
        if context.len != len {
            return;
        }
        for i in 0..c.layers {
            let rows = len.min(context.rows[i]);
            out.extend_from_slice(&(rows as u64).to_le_bytes());
            for cache in [&context.k[i], &context.v[i]] {
                out.extend(d.read_cache(cache, rows * c.kv_dim()));
            }
        }
    }

    /// Reads the context [`Self::save`] wrote for `len` positions back
    /// from `take`, which gives the next bytes of a saved state. A context
    /// saved from caches of another size is left broken.
    pub fn restore<'a>(
        &self,
        d: &D,
        context: &mut Context<D>,
        len: usize,
        mut take: impl FnMut(usize) -> Result<&'a [u8]>,
    ) -> Result<()> {
        let c = &self.config;
        context.len = BROKEN;
        if take(1)? != [1] {
            return Ok(());
        }
        let mut whole = true;
        for i in 0..c.layers {
            let rows = u64::from_le_bytes(take(8)?.try_into()?) as usize;
            let bytes = cache_bytes(rows * c.kv_dim());
            let (k, v) = (take(bytes)?, take(bytes)?);
            if rows == len.min(context.rows[i]) {
                d.write_cache(&mut context.k[i], k);
                d.write_cache(&mut context.v[i], v);
            } else {
                whole = false;
            }
        }
        if whole {
            context.len = len;
        }
        Ok(())
    }

    /// Whether the drafter can draft after a token at `pos` of a sequence
    /// of up to `max_len`: the positions before it are all in the context,
    /// and there is room for a block.
    pub fn ready(&self, context: &Context<D>, pos: usize, max_len: usize) -> bool {
        context.len == pos && pos > 0 && pos + self.config.block <= max_len
    }

    /// Makes context of the features of `n` tokens the model ran, the
    /// first at `pos`: `features` holds a buffer for each of the model's
    /// layers the drafter reads, a row a token.
    pub fn inject(
        &mut self,
        d: &D,
        context: &mut Context<D>,
        features: &[D::Buffer],
        n: usize,
        pos: usize,
    ) {
        let c = &self.config;
        let s = &mut self.s;
        assert!(n <= s.most && features.len() == c.targets.len());
        let (hidden, kv_dim) = (c.hidden, c.kv_dim());
        for buf in [&mut s.fused, &mut s.part, &mut s.normed] {
            d.resize(buf, n * hidden);
        }
        d.resize(&mut s.k, n * kv_dim);
        d.resize(&mut s.v, n * kv_dim);
        write_table(d, &mut s.table, c, pos, n);
        // The fusion of the features, a feature's part after another.
        d.matmul(&mut s.fused, &self.fuse[0], &features[0]);
        for (fuse, feature) in self.fuse.iter().zip(features).skip(1) {
            d.matmul_add(&mut s.fused, fuse, feature, &mut s.part);
        }
        d.normalize(&mut s.normed, &s.fused, &self.fuse_norm, n * hidden, c.eps);
        for (i, layer) in self.layers.iter().enumerate() {
            d.matmuls(&mut [
                (&mut s.k, &layer.k, &s.normed),
                (&mut s.v, &layer.v, &s.normed),
            ]);
            d.rmsnorm(&mut s.k, &layer.k_norm, c.eps);
            d.rope(&mut s.k, &s.table, 0, c.kv_heads, c.head_dim, c.head_dim);
            let rows = context.rows[i];
            store(
                d,
                &mut context.k[i],
                pos,
                rows,
                n,
                kv_dim,
                &mut s.k,
                &mut s.wrapped,
            );
            store(
                d,
                &mut context.v[i],
                pos,
                rows,
                n,
                kv_dim,
                &mut s.v,
                &mut s.wrapped,
            );
        }
        context.len = pos + n;
    }

    /// Drafts `k` tokens to follow `token` at `pos`, whose embedding,
    /// normalized as the model's input is, is `anchor`: the block's rows
    /// through the layers, the logits of the rows drafted through `head`,
    /// the model's output projection, which has `padded` rows, into
    /// `logits`, which has room for the logits of a block's drafts, and
    /// then the selector down the rows.
    #[allow(clippy::too_many_arguments)]
    pub fn draft(
        &mut self,
        d: &D,
        context: &mut Context<D>,
        anchor: &[f32],
        token: u32,
        pos: usize,
        k: usize,
        head: &D::Weight,
        padded: usize,
        logits: &mut D::Buffer,
    ) -> Result<Vec<Draft>> {
        let c = &self.config;
        let s = &mut self.s;
        let (hidden, block, kv_dim, q_dim) = (c.hidden, c.block, c.kv_dim(), c.q_dim());
        let k = k.min(block - 1);
        // The token's row, and the mask's for the rest.
        let mut rows = anchor.to_vec();
        for _ in 1..block {
            rows.extend_from_slice(&self.mask);
        }
        d.resize(&mut s.k, block * kv_dim);
        d.resize(&mut s.v, block * kv_dim);
        d.write(&mut s.x, &rows);
        write_table(d, &mut s.table, c, pos, block);
        let taps = c.taps();
        for (i, layer) in self.layers.iter().enumerate() {
            // Attention, between the convolutions of its input and of its
            // output.
            d.normalize(&mut s.xn, &s.x, &layer.attn_norm, block * hidden, c.eps);
            let conv = &layer.attn_conv;
            d.matmul(&mut s.taps, &conv.taps, &s.xn);
            d.block_conv(
                &mut s.conv,
                &s.xn,
                &s.taps,
                0,
                &conv.before,
                c.kernel,
                c.group,
            );
            d.matmuls(&mut [
                (&mut s.q, &layer.q, &s.conv),
                (&mut s.k, &layer.k, &s.conv),
                (&mut s.v, &layer.v, &s.conv),
            ]);
            d.rmsnorm(&mut s.q, &layer.q_norm, c.eps);
            d.rmsnorm(&mut s.k, &layer.k_norm, c.eps);
            d.rope(&mut s.q, &s.table, 0, c.heads, c.head_dim, c.head_dim);
            d.rope(&mut s.k, &s.table, 0, c.kv_heads, c.head_dim, c.head_dim);
            let cache_rows = context.rows[i];
            store(
                d,
                &mut context.k[i],
                pos,
                cache_rows,
                block,
                kv_dim,
                &mut s.k,
                &mut s.wrapped,
            );
            store(
                d,
                &mut context.v[i],
                pos,
                cache_rows,
                block,
                kv_dim,
                &mut s.v,
                &mut s.wrapped,
            );
            // Every row attends from the block's last position, which
            // has all the block's keys before it; a windowed layer as far
            // back as the window reaches of the row's own position.
            for row in 0..block {
                let window = match c.local[i] {
                    true => c.window + block - 1 - row,
                    false => usize::MAX,
                };
                d.copy(&mut s.q_row, 0, &s.q, row * q_dim, q_dim);
                d.attention_rel(
                    &mut s.att_row,
                    &s.q_row,
                    &context.k[i],
                    &context.v[i],
                    &s.rel,
                    &s.bias,
                    &s.sinks,
                    pos + block - 1,
                    c.heads,
                    c.head_dim,
                    c.kv_heads,
                    window,
                    cache_rows,
                );
                d.copy(&mut s.att, row * q_dim, &s.att_row, 0, q_dim);
            }
            d.matmul(&mut s.out, &layer.o, &s.att);
            d.block_conv(
                &mut s.conv,
                &s.out,
                &s.taps,
                taps,
                &conv.after,
                c.kernel,
                c.group,
            );
            d.add(&mut s.x, &s.conv);
            // And the feed-forward network, between its own.
            d.normalize(&mut s.xn, &s.x, &layer.ffn_norm, block * hidden, c.eps);
            let conv = &layer.ffn_conv;
            d.matmul(&mut s.taps, &conv.taps, &s.xn);
            d.block_conv(
                &mut s.conv,
                &s.xn,
                &s.taps,
                0,
                &conv.before,
                c.kernel,
                c.group,
            );
            d.swiglu(&mut s.h, &layer.gate, &layer.up, &s.conv, &mut s.h_scratch);
            d.matmul(&mut s.out, &layer.down, &s.h);
            d.block_conv(
                &mut s.conv,
                &s.out,
                &s.taps,
                taps,
                &conv.after,
                c.kernel,
                c.group,
            );
            d.add(&mut s.x, &s.conv);
        }
        d.rmsnorm(&mut s.x, &self.out_norm, c.eps);

        // The rows drafted: their logits, and the selector's projection.
        d.resize(&mut s.rows, k * hidden);
        d.copy(&mut s.rows, 0, &s.x, hidden, k * hidden);
        d.resize(&mut s.weighed, k * c.rank);
        d.matmul(&mut s.weighed, &self.selector, &s.rows);
        d.resize(logits, k * padded);
        d.matmul(logits, head, &s.rows);
        let logits = d.read(logits);
        let weighed = d.read(&s.weighed);

        // The selector, a row at a time: a candidate's score is its logit
        // and its vector's dot product with the vector of the token
        // before, weighed by the row's.
        let rank = c.rank;
        let mut before = token;
        let mut drafts = Vec::with_capacity(k);
        let mut condition = vec![0.0f32; rank];
        let mut vector = vec![0.0f32; rank];
        for (row, weights) in logits.chunks_exact(padded).zip(weighed.chunks_exact(rank)) {
            self.gguf.row(
                "selector_predecessor.weight",
                before as usize,
                &mut condition,
            )?;
            for (condition, weight) in condition.iter_mut().zip(weights) {
                *condition *= weight;
            }
            let mut scores = Vec::with_capacity(c.candidates);
            for (candidate, logit) in likeliest(&row[..c.vocab], c.candidates) {
                self.gguf
                    .row("selector_successor.weight", candidate as usize, &mut vector)?;
                let correction: f32 = vector.iter().zip(&condition).map(|(a, b)| a * b).sum();
                scores.push((
                    candidate,
                    c.softcap * ((logit + correction) / c.softcap).tanh(),
                ));
            }
            let (token, best) = scores
                .iter()
                .copied()
                .reduce(|best, score| if score.1 > best.1 { score } else { best })
                .expect("a row has candidates");
            let total: f32 = scores.iter().map(|(_, score)| (score - best).exp()).sum();
            drafts.push(Draft {
                token,
                sure: 1.0 / total,
            });
            before = token;
        }
        Ok(drafts)
    }
}

/// Writes the cosines and sines that rotate `n` positions from `pos` on,
/// as [`Device::rope`] takes them for its first position.
fn write_table<D: Device>(d: &D, table: &mut D::Buffer, c: &Config, pos: usize, n: usize) {
    let half = c.head_dim / 2;
    let mut values = Vec::with_capacity(n * c.head_dim);
    for p in pos..pos + n {
        for i in 0..half {
            let freq = 1.0 / c.rope_base.powf((2 * i) as f32 / c.head_dim as f32);
            let (sin, cos) = (p as f32 * freq).sin_cos();
            values.extend([cos, sin]);
        }
    }
    d.resize(table, values.len());
    d.write(table, &values);
}

/// Stores the `n` rows of `src`, each `width` wide, in a cache of `rows`
/// rows, the first at position `pos`: in two parts where they run past
/// the end of the cache's rows, the second from its start.
#[allow(clippy::too_many_arguments)]
fn store<D: Device>(
    d: &D,
    cache: &mut D::Cache,
    pos: usize,
    rows: usize,
    n: usize,
    width: usize,
    src: &mut D::Buffer,
    wrapped: &mut D::Buffer,
) {
    let at = pos % rows;
    let first = n.min(rows - at);
    if first < n {
        let rest = (n - first) * width;
        d.resize(wrapped, rest);
        d.copy(wrapped, 0, src, first * width, rest);
        d.store(cache, 0, wrapped);
        d.resize(src, first * width);
    }
    d.store(cache, at * width, src);
    d.resize(src, n * width);
}

/// The `k` likeliest tokens of `logits`, each with its logit, the
/// likeliest first, and of two as likely the first: in one pass over the
/// logits, a vocabulary's worth, for a handful.
fn likeliest(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
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
