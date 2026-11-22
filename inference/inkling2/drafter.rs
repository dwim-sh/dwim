//! Inkling 2 Mini's drafter: a network of three layers that drafts the
//! tokens after the last one picked, a block of them in one pass, for the
//! model to verify as a batch. It is a file of its own, of the `dflash`
//! architecture as llama.cpp names it, and borrows the model's embeddings
//! and its output projection. What it computes is what llama.cpp's graph
//! for it does (ggml-org/llama.cpp#32218).
//!
//! The drafter reads the model, not the text: as the model runs a token,
//! its residual stream on the way into a few of its layers and its
//! normalized output are kept, the token's *features*. Each of the
//! drafter's layers mixes them in its own proportions and projects the mix
//! to a key and a value, which join the layer's caches at the token's
//! position: the *context*. One more mix of them, of the last token run,
//! is carried over for the head.
//!
//! To draft after a token, the drafter runs a block of eight rows at the
//! positions from the token's on: the token's embedding, and seven of a
//! mask's. Each layer is attention and a SwiGLU feed-forward network, both
//! on a normalized input. The rows attend to each other in both
//! directions, and to the context before them: over everything in one
//! layer, and over a window of the latest positions in the others. Queries
//! and keys are normalized a head at a time and rotated by their position.
//!
//! A row's output is not yet its token's logits. A small recurrent head
//! goes down the block: from its state, the token before, the row's output
//! and the mix carried over, it makes a correction to the row's output,
//! and the model's output projection of the two is the row's logits, whose
//! likeliest token is the draft and the token before of the next row.
//!
//! A draft costs a pass over the output projection, half a gigabyte, and
//! a token verified costs the weights of its own experts, so drafting
//! pays where the model keeps most of what is drafted: the model asks for
//! one token more after a batch it kept whole, and for no more than it
//! kept after another.

use std::sync::Arc;

use rayon::prelude::*;

use crate::{
    gpu::{CACHE_BLOCK, Device, cache_bytes},
    inference::{Gguf, Memory, Result},
};

/// What a drafter's file names its architecture.
pub const ARCHITECTURE: &str = "dflash";

/// The tensors of the head, which runs on the host.
const HOST: [&str; 7] = [
    "markov_w1.weight",
    "joint_proj.weight",
    "joint_proj.bias",
    "dspark_hidden_out.weight",
    "dspark_hidden_out.bias",
    "mask_embd.weight",
    "dspark_fuse.weight",
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
    /// Width of the head's state.
    pub rank: usize,
    /// Which layers attend over a window of the latest positions, and how
    /// far back of the block's first it reaches.
    pub local: Vec<bool>,
    pub window: usize,
    /// The layers of the model whose input is a feature, counted as the
    /// model runs them, the last being its output.
    pub targets: Vec<usize>,
    pub eps: f32,
    pub rope_base: f32,
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
        let config = Self {
            hidden: key("embedding_length")?,
            layers: key("block_count")?,
            heads: key("attention.head_count")?,
            kv_heads: key("attention.head_count_kv")?,
            head_dim: key("attention.key_length")?,
            ff: key("feed_forward_length")?,
            block: key("block_size")?,
            rank: key("dspark.markov_rank")?,
            local,
            window: key("attention.sliding_window")?,
            targets,
            eps: gguf.f32(&format!("{arch}.attention.layer_norm_rms_epsilon"))?,
            rope_base: gguf.f32(&format!("{arch}.rope.freq_base"))?,
            vocab: key("vocab_size_unpadded")?,
        };
        let c = &config;
        if c.local.len() != c.layers
            || c.targets.len() < 2
            || !c.targets.is_sorted()
            || c.block < 2
            || c.kv_heads == 0
            || !c.heads.is_multiple_of(c.kv_heads)
            || !c.head_dim.is_multiple_of(CACHE_BLOCK)
            || key("rope.dimension_count")? != c.head_dim
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

    /// The device memory the drafter in `gguf`, configured as this, takes
    /// beside the model, for batches of `most` tokens: `per_sequence` for
    /// the windowed layers' keys and values, and `per_token` for those of
    /// the layers that attend over everything.
    pub fn memory(&self, gguf: &Gguf, most: usize) -> Result<Memory> {
        let c = self;
        // The head's tensors stay on the host.
        let mut weights = 0;
        for name in gguf.tensors().filter(|name| !HOST.contains(name)) {
            weights += gguf.info(name)?.size()?;
        }
        let kv = 2 * cache_bytes(c.kv_dim());
        let local = c.local.iter().filter(|&&local| local).count();
        let rings = local * c.ring(most, usize::MAX) * kv;
        let batch = most * (3 * c.hidden + 3 * c.kv_dim() + c.head_dim);
        let block = c.block * (3 * c.hidden + 2 * c.q_dim() + 2 * c.ff) + 2 * c.q_dim();
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
    /// The file, kept mapped for the head's table of tokens, which is
    /// looked up on the CPU as the model's embeddings are.
    gguf: Arc<Gguf>,
    layers: Vec<Layer<D>>,
    /// Each layer's shares of the features, and then the head's: one
    /// number to a buffer, for scaling a whole buffer by.
    shares: Vec<Vec<D::Buffer>>,
    /// The norm of the mix carried over for the head, and of the output.
    rnn_norm: D::Buffer,
    out_norm: D::Buffer,
    /// The embedding of a row not yet drafted.
    mask: Vec<f32>,
    /// The head, which runs on the host: the projection of its inputs to
    /// its gate, its candidate state and its latent, and of the latent to
    /// a correction, a row of each matrix after another.
    joint: Vec<f32>,
    joint_bias: Vec<f32>,
    hidden_out: Vec<f32>,
    hidden_out_bias: Vec<f32>,
    s: Scratch<D>,
}

struct Layer<D: Device> {
    /// The context: the norm of the layer's mix of the features, its
    /// projections to keys and values, and the norm of a key's head.
    ctx_norm: D::Buffer,
    k_ctx: D::Weight,
    v_ctx: D::Weight,
    k_ctx_norm: D::Buffer,
    attn_norm: D::Buffer,
    q: D::Weight,
    k: D::Weight,
    v: D::Weight,
    o: D::Weight,
    q_norm: D::Buffer,
    k_norm: D::Buffer,
    ffn_norm: D::Buffer,
    gate: D::Weight,
    up: D::Weight,
    down: D::Weight,
}

/// What the drafter computes in: for a batch of the model's tokens whose
/// features become context, and for a block.
struct Scratch<D: Device> {
    /// Most tokens of context stored at once.
    most: usize,
    mixed: D::Buffer,
    part: D::Buffer,
    normed: D::Buffer,
    k: D::Buffer,
    v: D::Buffer,
    wrapped: D::Buffer,
    /// The cosines and sines that rotate the positions being run.
    table: D::Buffer,
    /// A row of the block's output with the head's correction to it.
    row: D::Buffer,
    /// The block: its stream, the same normalized, a layer's output, and
    /// the output normalized.
    x: D::Buffer,
    xn: D::Buffer,
    out: D::Buffer,
    q: D::Buffer,
    att: D::Buffer,
    /// A row's queries and what it attends to, for attention a row at a
    /// time.
    q_row: D::Buffer,
    att_row: D::Buffer,
    h: D::Buffer,
    h_scratch: D::Buffer,
    /// No bias by distance, and no sink.
    rel: D::Buffer,
    bias: D::Buffer,
    sinks: D::Buffer,
}

/// A sequence's context: each layer's keys and values, a windowed layer's
/// in a ring, the mix carried over from the last position, and how many
/// positions there are context for.
pub struct Context<D: Device> {
    k: Vec<D::Cache>,
    v: Vec<D::Cache>,
    rows: Vec<usize>,
    carried: D::Buffer,
    /// Positions there is context for, or [`BROKEN`].
    pub len: usize,
}

/// A length no sequence has: of a context that is not whole, which the
/// drafter does not draft from.
pub const BROKEN: usize = usize::MAX;

/// What a mark keeps of a context: the windowed layers' rings, which are
/// written over the positions before the mark as the sequence goes on, the
/// mix carried over, and how long the context was.
pub struct Mark<D: Device> {
    rings: Vec<D::Cache>,
    carried: D::Buffer,
    len: usize,
}

impl<D: Device> Drafter<D> {
    /// Loads the drafter from its GGUF file onto `device`, with buffers
    /// for `most` tokens of context at once.
    pub fn load(gguf: Arc<Gguf>, device: &D, most: usize) -> Result<Self> {
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
        let weight = |name: &str| -> Result<D::Weight> { Ok(device.upload(gguf.matrix(name)?)) };
        let mut layers = Vec::with_capacity(c.layers);
        for i in 0..c.layers {
            let p = format!("blk.{i}");
            layers.push(Layer {
                ctx_norm: vector(&format!("{p}.dflare_norm.weight"), c.hidden)?,
                k_ctx: weight(&format!("{p}.attn_k_ctx.weight"))?,
                v_ctx: weight(&format!("{p}.attn_v_ctx.weight"))?,
                k_ctx_norm: vector(&format!("{p}.attn_k_ctx_norm.weight"), c.head_dim)?,
                attn_norm: vector(&format!("{p}.attn_norm.weight"), c.hidden)?,
                q: weight(&format!("{p}.attn_q.weight"))?,
                k: weight(&format!("{p}.attn_k.weight"))?,
                v: weight(&format!("{p}.attn_v.weight"))?,
                o: weight(&format!("{p}.attn_output.weight"))?,
                q_norm: vector(&format!("{p}.attn_q_norm.weight"), c.head_dim)?,
                k_norm: vector(&format!("{p}.attn_k_norm.weight"), c.head_dim)?,
                ffn_norm: vector(&format!("{p}.ffn_norm.weight"), c.hidden)?,
                gate: weight(&format!("{p}.ffn_gate.weight"))?,
                up: weight(&format!("{p}.ffn_up.weight"))?,
                down: weight(&format!("{p}.ffn_down.weight"))?,
            });
        }
        // A row of the fusion's weights for each layer and one for the
        // head, each row's softmax its shares of the features.
        let fuse = gguf.f32s("dspark_fuse.weight")?;
        let targets = c.targets.len();
        if fuse.len() != (c.layers + 1) * targets {
            return Err("the drafter's fusion is not a row for each layer and the head".into());
        }
        let shares = fuse
            .chunks_exact(targets)
            .map(|row| {
                let most = row.iter().cloned().fold(f32::MIN, f32::max);
                let sum: f32 = row.iter().map(|w| (w - most).exp()).sum();
                row.iter()
                    .map(|w| buffer(&[(w - most).exp() / sum]))
                    .collect()
            })
            .collect();
        // A matrix as rows on the host, each row an output's weights.
        let rows = |name: &str, cols: usize, rows: usize| -> Result<Vec<f32>> {
            let mut out = vec![0.0; rows * cols];
            for (i, row) in out.chunks_exact_mut(cols).enumerate() {
                gguf.row(name, i, row)?;
            }
            Ok(out)
        };
        let joint = rows("joint_proj.weight", 2 * c.rank + 2 * c.hidden, 3 * c.rank)?;
        let hidden_out = rows("dspark_hidden_out.weight", c.rank, c.hidden)?;
        // The head's table of tokens stays in the file: a row a token.
        let table = gguf.info("markov_w1.weight")?.shape();
        if table.len() != 2 || table[1] != c.rank || table[0] < c.vocab {
            return Err("the drafter's table is not a row of its head's for each token".into());
        }
        let (block, hidden) = (c.block, c.hidden);
        let s = Scratch {
            most,
            mixed: device.alloc(most * hidden),
            part: device.alloc(most * hidden),
            normed: device.alloc(most * hidden),
            k: device.alloc(most * c.kv_dim()),
            v: device.alloc(most * c.kv_dim()),
            wrapped: device.alloc(most * c.kv_dim()),
            table: device.alloc(most.max(block) * c.head_dim),
            row: device.alloc(hidden),
            x: device.alloc(block * hidden),
            xn: device.alloc(block * hidden),
            out: device.alloc(block * hidden),
            q: device.alloc(block * c.q_dim()),
            att: device.alloc(block * c.q_dim()),
            q_row: device.alloc(c.q_dim()),
            att_row: device.alloc(c.q_dim()),
            h: device.alloc(block * c.ff),
            h_scratch: device.alloc(block * c.ff),
            rel: buffer(&vec![0.0; c.heads * REL]),
            bias: buffer(&[0.0; REL]),
            sinks: buffer(&vec![NO_SINK; c.heads]),
        };
        Ok(Self {
            rnn_norm: vector("dspark_norm.weight", hidden)?,
            out_norm: vector("output_norm.weight", hidden)?,
            mask: floats("mask_embd.weight", hidden)?,
            joint,
            joint_bias: floats("joint_proj.bias", 3 * c.rank)?,
            hidden_out,
            hidden_out_bias: floats("dspark_hidden_out.bias", hidden)?,
            layers,
            shares,
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
            carried: device.alloc(c.hidden),
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
            carried: device.alloc(c.hidden),
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
        d.copy(&mut mark.carried, 0, &context.carried, 0, c.hidden);
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
        d.copy(&mut context.carried, 0, &mark.carried, 0, c.hidden);
        context.len = mark.len;
    }

    /// Appends the context of the first `len` positions to `out`: whether
    /// it is whole, which it is if the context ends there, then each
    /// layer's keys and values as the caches hold them, and the mix
    /// carried over.
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
        out.extend(
            d.read(&context.carried)
                .iter()
                .flat_map(|v| v.to_le_bytes()),
        );
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
        let carried: Vec<f32> = take(c.hidden * 4)?
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        d.write(&mut context.carried, &carried);
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
        for buf in [&mut s.mixed, &mut s.part, &mut s.normed] {
            d.resize(buf, n * hidden);
        }
        d.resize(&mut s.k, n * kv_dim);
        d.resize(&mut s.v, n * kv_dim);
        write_table(d, &mut s.table, c, pos, n);
        for (i, layer) in self.layers.iter().enumerate() {
            mix(
                d,
                &mut s.mixed,
                &mut s.part,
                features,
                &self.shares[i],
                0,
                n * hidden,
            );
            d.normalize(&mut s.normed, &s.mixed, &layer.ctx_norm, n * hidden, c.eps);
            d.matmuls(&mut [
                (&mut s.k, &layer.k_ctx, &s.normed),
                (&mut s.v, &layer.v_ctx, &s.normed),
            ]);
            d.rmsnorm(&mut s.k, &layer.k_ctx_norm, c.eps);
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
        // The head's mix, of the last token.
        d.resize(&mut s.mixed, hidden);
        d.resize(&mut s.part, hidden);
        mix(
            d,
            &mut s.mixed,
            &mut s.part,
            features,
            &self.shares[c.layers],
            (n - 1) * hidden,
            hidden,
        );
        d.copy(&mut context.carried, 0, &s.mixed, 0, hidden);
        d.rmsnorm(&mut context.carried, &self.rnn_norm, c.eps);
        context.len = pos + n;
    }

    /// Drafts `k` tokens to follow `token` at `pos`, whose embedding,
    /// normalized as the model's input is, is `anchor`: the block's rows
    /// through the layers, and then the head down them, each row's logits
    /// through `head`, the model's output projection, into `logits`.
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
        logits: &mut D::Buffer,
    ) -> Result<Vec<u32>> {
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
        for (i, layer) in self.layers.iter().enumerate() {
            d.normalize(&mut s.xn, &s.x, &layer.attn_norm, block * hidden, c.eps);
            d.matmuls(&mut [
                (&mut s.q, &layer.q, &s.xn),
                (&mut s.k, &layer.k, &s.xn),
                (&mut s.v, &layer.v, &s.xn),
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
            // Every row attends as the block's last position does, which
            // has all the block's keys before it; a windowed layer from
            // as far back as its window reaches of the block's first.
            let window = match c.local[i] {
                true => c.window + block,
                false => usize::MAX,
            };
            for row in 0..block {
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
            d.matmul_add(&mut s.x, &layer.o, &s.att, &mut s.out);
            d.normalize(&mut s.xn, &s.x, &layer.ffn_norm, block * hidden, c.eps);
            d.swiglu(&mut s.h, &layer.gate, &layer.up, &s.xn, &mut s.h_scratch);
            d.matmul_add(&mut s.x, &layer.down, &s.h, &mut s.out);
        }
        d.rmsnorm(&mut s.x, &self.out_norm, c.eps);
        let outputs = d.read(&s.x);
        let carried = d.read(&context.carried);

        // The head, a row at a time: its state starts at nothing.
        let rank = c.rank;
        let mut state = vec![0.0f32; rank];
        let mut before = token;
        let mut drafts = Vec::with_capacity(k);
        let mut joint = vec![0.0f32; 2 * rank + 2 * hidden];
        let mut projected = vec![0.0f32; 3 * rank];
        let mut row = vec![0.0f32; hidden];
        for i in 1..=k {
            let output = &outputs[i * hidden..][..hidden];
            joint[..rank].copy_from_slice(&state);
            self.gguf.row(
                "markov_w1.weight",
                before as usize,
                &mut joint[rank..2 * rank],
            )?;
            joint[2 * rank..][..hidden].copy_from_slice(output);
            joint[2 * rank + hidden..].copy_from_slice(&carried);
            project(&mut projected, &self.joint, &self.joint_bias, &joint);
            let (gate, rest) = projected.split_at(rank);
            let (candidate, latent) = rest.split_at(rank);
            for ((state, gate), candidate) in state.iter_mut().zip(gate).zip(candidate) {
                let (gate, candidate) = (sigmoid(*gate), candidate.tanh());
                *state = candidate + gate * (*state - candidate);
            }
            let latent: Vec<f32> = latent.iter().map(|v| v.tanh()).collect();
            // The correction is to the row's output: the projection of
            // the two is the row's logits.
            project(&mut row, &self.hidden_out, &self.hidden_out_bias, &latent);
            for (row, output) in row.iter_mut().zip(output) {
                *row += output;
            }
            d.write(&mut s.row, &row);
            d.matmul(logits, head, &s.row);
            before = likeliest(&d.read(logits)[..c.vocab]);
            drafts.push(before);
        }
        Ok(drafts)
    }
}

/// `out = Σ shares[j] * features[j][from..][..len]`, with `part` as long
/// for the terms.
fn mix<D: Device>(
    d: &D,
    out: &mut D::Buffer,
    part: &mut D::Buffer,
    features: &[D::Buffer],
    shares: &[D::Buffer],
    from: usize,
    len: usize,
) {
    d.copy(out, 0, &features[0], from, len);
    d.scale_rows(out, &shares[0]);
    for (feature, share) in features.iter().zip(shares).skip(1) {
        d.copy(part, 0, feature, from, len);
        d.scale_rows(part, share);
        d.add(out, part);
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

/// `out = matrix · x + bias`, the matrix a row of weights for each output.
fn project(out: &mut [f32], matrix: &[f32], bias: &[f32], x: &[f32]) {
    out.par_iter_mut()
        .zip(matrix.par_chunks_exact(x.len()))
        .zip(bias)
        .for_each(|((out, row), bias)| {
            *out = bias + row.iter().zip(x).map(|(w, x)| w * x).sum::<f32>();
        });
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// The first of the likeliest tokens.
fn likeliest(logits: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &logit) in logits.iter().enumerate() {
        if logit > logits[best] {
            best = i;
        }
    }
    best as u32
}
