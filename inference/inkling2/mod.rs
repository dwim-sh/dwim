//! The `inkling2_loop` architecture of llama.cpp's files, which Inkling 2
//! Mini is of: a looped transformer with a mixture of experts for its
//! feed-forward networks, written as a sequence of operations on a
//! [`Device`]. It runs the model from a file in the layout mainline
//! llama.cpp reads, whose experts are in NVFP4, the format they were
//! trained in, or `Q8_0`.
//!
//! The layers come in blocks of six. A token goes through the first block
//! once, the prelude; through the blocks of the core several times over,
//! each pass starting from the prelude's output; and through the rest
//! once. A pass after the first reads messages from the one before: a mix
//! of what the pass before left, which a learned gate lets into the inputs
//! of each core layer's queries, keys, values and feed-forward network.
//! The passes share the core's weights, and each keeps keys, values and
//! convolution states of its own: a slot for every time a layer is run.
//!
//! The first layer of a block starts a residual stream anew, from its
//! attention's output alone, and what the stream held joins the blocks
//! kept. A layer reads its inputs not from the stream but from a mix of
//! the stream and the blocks kept, weighted by attention over the depth
//! with queries of the layer's own.
//!
//! Attention has no rotary positions: a key's score gets a bias by how far
//! back it is, the dot product of a projection of the query's input with
//! a learned row per distance, and the softmax includes a sink that takes
//! what no key should get. All but the last layer of a block attend over
//! a window of the latest positions, and keep only those. Keys, values,
//! and the outputs of attention and of the feed-forward network each go
//! through a short causal convolution.

use std::{collections::HashMap, mem, sync::Arc};

use crate::{
    gpu::{CACHE_BLOCK, Device, Kept, Tensor, cache_bytes},
    inference::{Gguf, LanguageModel, Memory, Result, gguf},
};

/// What a file of the architecture calls it.
pub const ARCHITECTURE: &str = "inkling2_loop";

/// What the root mean square of a state is steadied by where the depth is
/// mixed and where a message is gated.
const MIX_EPS: f32 = 1e-6;

/// Batches of the device's that the model runs as one. A device reads an
/// expert's weights once for a batch, however many of its tokens go
/// through the expert, and a short batch leaves most experts a token or
/// two: a longer one costs little more, and reads a prompt that much
/// faster.
const BATCHES: usize = 4;

/// The inputs a pass's messages go into: a layer's queries, keys and
/// values, and its feed-forward network.
const CONTROLS: usize = 4;
const FFN: usize = 3;

/// The convolutions of a layer, in the order a slot keeps their states:
/// over the keys, the values, attention's output and the feed-forward
/// network's.
const CONVS: usize = 4;

/// The architecture, read from the GGUF metadata.
#[derive(Debug)]
pub struct Config {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    /// Which layers attend over a window of the latest positions, and how
    /// many it holds.
    pub local: Vec<bool>,
    pub window: usize,
    /// The relative-position bias: how wide the projection it is taken
    /// with is for a head, and how many distances have a row.
    pub rel: usize,
    pub rel_extent: usize,
    /// The lengths queries and keys, and the bias's projection, are
    /// clipped to.
    pub qk_clip: f32,
    pub rel_clip: f32,
    /// Past `log_floor` positions, queries in the layers that attend over
    /// everything grow by `log_alpha` times the logarithm of how far
    /// past.
    pub log_floor: usize,
    pub log_alpha: f32,
    /// Taps of the short convolutions.
    pub taps: usize,
    /// Layers to a block of the residual stream.
    pub block: usize,
    /// The mixture of experts: how many there are, how many a token goes
    /// through, how wide each is inside, and what their weights sum to.
    pub experts: usize,
    pub experts_used: usize,
    pub expert_width: usize,
    pub expert_scale: f32,
    pub situ_beta: f32,
    pub situ_linear_beta: f32,
    /// The core: the layers that run `passes` times, and the rank of the
    /// gates its messages go through.
    pub core_start: usize,
    pub core_end: usize,
    pub passes: usize,
    pub gate_rank: usize,
    pub eps: f32,
    /// What the logits are capped to, softly, if anything.
    pub softcap: f32,
    /// The tokens of the vocabulary, and the rows the embedding table and
    /// the output projection have, which are padded past it.
    pub vocab: usize,
    pub vocab_padded: usize,
    /// Positions the model was trained for.
    pub context_length: usize,
}

impl Config {
    pub fn load(gguf: &Gguf) -> Result<Self> {
        let arch = gguf.str("general.architecture")?;
        if arch != ARCHITECTURE {
            return Err(format!("the '{arch}' architecture is not supported").into());
        }
        let key = |name: &str| gguf.u32(&format!("{arch}.{name}")).map(|v| v as usize);
        let float = |name: &str| gguf.f32(&format!("{arch}.{name}"));
        let layers = key("block_count")?;
        // A file says how many heads each layer has: they must all have
        // as many.
        let per_layer = |name: &str| -> Result<usize> {
            let values = gguf.array(&format!("{arch}.{name}"))?;
            let first = values.first().and_then(|v| v.as_i64());
            match first {
                Some(first)
                    if values.len() == layers
                        && values.iter().all(|v| v.as_i64() == Some(first)) =>
                {
                    Ok(first as usize)
                }
                _ => Err(format!("the layers differ in their {name}").into()),
            }
        };
        let local = gguf
            .array(&format!("{arch}.attention.sliding_window_pattern"))?
            .iter()
            .map(|v| v.as_bool().ok_or("the window pattern is not of booleans"))
            .collect::<std::result::Result<Vec<bool>, _>>()?;
        let config = Self {
            hidden: key("embedding_length")?,
            layers,
            heads: per_layer("attention.head_count")?,
            kv_heads: per_layer("attention.head_count_kv")?,
            head_dim: key("attention.key_length")?,
            local,
            window: key("attention.sliding_window")?,
            rel: key("attention.relative_length")?,
            rel_extent: key("attention.relative_extent")?,
            qk_clip: float("attention.qk_clip")?,
            rel_clip: float("attention.relative_clip")?,
            log_floor: key("attention.log_scale_floor")?,
            log_alpha: float("attention.log_scale_alpha")?,
            taps: key("shortconv.l_cache")?,
            block: key("attn_res.block_size")?,
            experts: key("expert_count")?,
            experts_used: key("expert_used_count")?,
            expert_width: key("expert_feed_forward_length")?,
            expert_scale: float("expert_weights_scale")?,
            situ_beta: float("activation.situ_beta")?,
            situ_linear_beta: float("activation.situ_linear_beta")?,
            core_start: key("loop.start")?,
            core_end: key("loop.end")?,
            passes: key("loop.count")?,
            gate_rank: key("loop.gate_rank")?,
            eps: float("attention.layer_norm_rms_epsilon")?,
            softcap: float("final_logit_softcapping").unwrap_or(0.0),
            vocab: key("vocab_size_unpadded")?,
            vocab_padded: key("vocab_size")?,
            context_length: key("context_length")?,
        };
        let c = &config;
        if c.local.len() != c.layers {
            return Err("the window pattern is not a layer long".into());
        }
        if key("attention.value_length")? != c.head_dim
            || key("attention.key_length_swa")? != c.head_dim
            || key("attention.value_length_swa")? != c.head_dim
        {
            return Err("the attention heads are not all as wide".into());
        }
        if !c.head_dim.is_multiple_of(CACHE_BLOCK) {
            return Err("the attention heads are not whole blocks of the caches".into());
        }
        if c.kv_heads == 0 || !c.heads.is_multiple_of(c.kv_heads) {
            return Err("the heads are not a multiple of the key heads".into());
        }
        if c.block == 0
            || !c.layers.is_multiple_of(c.block)
            || !c.core_start.is_multiple_of(c.block)
            || !c.core_end.is_multiple_of(c.block)
            || c.core_start == 0
            || c.core_start >= c.core_end
            || c.core_end > c.layers
            || c.passes == 0
        {
            return Err("the core is not whole blocks of the layers, after the first".into());
        }
        if gguf
            .get(&format!("{arch}.loop.varying"))
            .and_then(|v| v.as_bool())
            == Some(true)
        {
            return Err("a core that runs a varying number of times is not implemented".into());
        }
        if c.taps < 2 || c.window == 0 || c.rel == 0 || c.rel_extent == 0 || c.log_floor == 0 {
            return Err("the convolutions, the window or the bias have no size".into());
        }
        if c.experts_used == 0 || c.experts_used > c.experts {
            return Err("a token goes through more experts than there are".into());
        }
        if c.vocab == 0 || c.vocab > c.vocab_padded {
            return Err("the vocabulary is larger than the tables for it".into());
        }
        // The embedding table is looked up row by row as the model runs,
        // and the head's rows are the logits: both must be what the model
        // expects, or the first token would find out.
        for name in ["token_embd.weight", "output.weight"] {
            let info = gguf.info(name)?;
            if info.dims != [c.hidden, c.vocab_padded] {
                return Err(format!("{name} is not a row per token of the hidden width").into());
            }
            gguf::row_bytes(info.dtype, c.hidden)
                .map_err(|e| format!("{name} cannot be read: {e}"))?;
        }
        Ok(config)
    }

    /// How many times a layer is run for a token, each with keys, values
    /// and convolution states of its own: the layers once, and the core's
    /// again for each pass after the first.
    pub fn slots(&self) -> usize {
        self.layers + (self.passes - 1) * (self.core_end - self.core_start)
    }

    /// The layer whose weights slot `slot` runs.
    fn layer_of(&self, slot: usize) -> usize {
        if slot < self.layers {
            slot
        } else {
            self.core_start + (slot - self.layers) % (self.core_end - self.core_start)
        }
    }

    /// The slot of core layer `layer` in pass `pass`.
    fn slot_of(&self, layer: usize, pass: usize) -> usize {
        if pass == 0 {
            layer
        } else {
            self.layers + (pass - 1) * (self.core_end - self.core_start) + (layer - self.core_start)
        }
    }

    /// Activations of a token's keys or values.
    fn kv_dim(&self) -> usize {
        self.kv_heads * self.head_dim
    }

    /// Activations of a token's queries.
    fn q_dim(&self) -> usize {
        self.heads * self.head_dim
    }

    /// Channels of each of a layer's convolutions, in the order a slot
    /// keeps their states.
    fn conv_channels(&self) -> [usize; CONVS] {
        [self.kv_dim(), self.kv_dim(), self.hidden, self.hidden]
    }

    /// Positions a windowed layer's caches hold: the window, and a batch
    /// more, since a batch's keys are all stored before its first token
    /// attends, which still reaches a window back. No more than a sequence
    /// is long.
    fn ring(&self, batch: usize, max_len: usize) -> usize {
        (self.window + batch).min(max_len)
    }

    /// Activations per token of the forward pass's per-token buffers, all
    /// told.
    fn activations(&self) -> usize {
        let c = self;
        let k = c.experts_used;
        // The stream, the inputs mixed and normalized, the blocks kept
        // and the messages, and the rest as wide as the stream; then what
        // attention and the experts work in.
        let streams = 19 + c.layers / c.block + (c.core_end - c.core_start) / c.block + CONTROLS;
        streams * c.hidden
            + 3 * c.q_dim()
            + 6 * c.kv_dim()
            + c.heads * (c.rel + 1)
            + 2 * c.kv_heads * c.taps
            + c.experts
            + 2 * k
            + k * (3 * c.expert_width + c.hidden)
            + 1
    }

    /// The device memory the model in `gguf`, configured as this, takes:
    /// `per_sequence` is for the windowed layers' keys and values and the
    /// convolutions' states, and `per_token` for the keys and values of
    /// the layers that attend over everything. Fails if the file does not
    /// say how big one of its tensors is.
    pub fn memory(&self, gguf: &Gguf, batch: usize) -> Result<Memory> {
        let c = self;
        let batch = batch * BATCHES;
        // Every tensor goes to the device but the embedding table, which
        // stays in the file on the host, in the format the file has it
        // in, but for a matrix of f32, which is rounded to bf16.
        let mut weights = 0;
        for name in gguf.tensors().filter(|&name| name != "token_embd.weight") {
            let info = gguf.info(name)?;
            weights += match info.dtype {
                gguf::F32 if info.dims.len() == 2 && is_matrix(name) => info.size()? / 2,
                _ => info.size()?,
            };
        }
        let local = (0..c.slots()).filter(|&s| c.local[c.layer_of(s)]).count();
        let global = c.slots() - local;
        let conv = c.slots() * (c.taps - 1) * c.conv_channels().iter().sum::<usize>();
        let kv = 2 * cache_bytes(c.kv_dim());
        // A windowed layer's ring is no longer than a sequence, which is
        // not known yet: counted in full.
        let rings = local * c.ring(batch, usize::MAX) * kv;
        Ok(Memory {
            // The activations of a batch, the logits, and the copy a mark
            // keeps of a sequence's rings and convolution states.
            fixed: (weights + 4 * (batch * c.activations() + c.vocab_padded + conv) + rings) as u64,
            per_sequence: (rings + 4 * conv) as u64,
            // And what the queries at the position grow by, which the
            // sequences share, counted with each.
            per_token: (global * kv + 4) as u64,
        })
    }
}

/// Whether the f32 tensor `name` of two dimensions is a matrix, which goes
/// to the device as bf16, rather than rows of vectors the device keeps as
/// they are.
fn is_matrix(name: &str) -> bool {
    name.ends_with("ffn_gate_inp.weight") || name.contains(".loop_") && name.contains("_ctrl_")
}

/// The model, with its weights on a device, and the state of a sequence.
pub struct Model<D: Device> {
    config: Config,
    device: D,
    /// The file, kept mapped for the embedding table, which is looked up
    /// on the CPU: it is half a gigabyte, and a token needs one row of it.
    gguf: Arc<Gguf>,
    embed_norm: D::Buffer,
    layers: Vec<Layer<D>>,
    /// What the core's layers read the pass before with, a layer each.
    loops: Vec<Loop<D>>,
    out_query: D::Buffer,
    out_norm: D::Buffer,
    head: D::Weight,
    /// Ones, for the operations that scale by a weight where the model
    /// has none: as wide as a head, and as a token's keys.
    ones_head: D::Buffer,
    ones_kv: D::Buffer,
    state: State<D>,
    /// The sequences besides the one the model is running, by number.
    parked: HashMap<usize, Sequence<D>>,
}

struct Layer<D: Device> {
    /// Whether the layer attends over a window of the latest positions,
    /// rather than over everything.
    local: bool,
    /// The queries the layer's attention reads the depth with: none for
    /// the first layer, which reads the stream; one for a windowed layer;
    /// and one each for the queries, keys and values of the others.
    attn_query: Vec<D::Buffer>,
    /// The norms of attention's inputs: one for a windowed layer, and one
    /// each for the queries, keys and values of the others.
    attn_norm: Vec<D::Buffer>,
    q: D::Weight,
    k: D::Weight,
    v: D::Weight,
    o: D::Weight,
    /// The output gate.
    gate: D::Weight,
    /// The projection the relative-position bias is taken with, the rows
    /// it is taken against, one per distance, and each head's sink.
    rel: D::Weight,
    rel_bias: D::Buffer,
    sink: D::Weight,
    /// What queries and the bias's projection are scaled by before they
    /// are clipped.
    q_clip: D::Buffer,
    rel_clip: D::Buffer,
    /// The taps of the convolutions, in a slot's order, and the
    /// projections of the taps the keys and values of a layer that
    /// attends over everything get from their inputs.
    conv: [D::Buffer; CONVS],
    k_taps: Option<D::Weight>,
    v_taps: Option<D::Weight>,
    ffn_query: D::Buffer,
    ffn_norm: D::Buffer,
    router: D::Weight,
    router_bias: D::Buffer,
    /// The experts' gate and up projections, a matrix of both per expert,
    /// and their down projections, with each expert's scale for each.
    gate_up: D::Weight,
    gate_up_scale: D::Buffer,
    down: D::Weight,
    down_scale: D::Buffer,
}

/// What a core layer reads the pass before with: a query over what that
/// pass left for each of the layer's inputs, and the gate each message
/// goes through.
struct Loop<D: Device> {
    queries: [D::Buffer; CONTROLS],
    gates: [Gate<D>; CONTROLS],
}

/// A gate on a message: a projection of the input and of the message, each
/// normalized, down to a few dimensions, and from those what the input
/// gains of the message and what it forgets of itself.
struct Gate<D: Device> {
    down_input: D::Weight,
    down_message: D::Weight,
    gain: D::Weight,
    forget: D::Weight,
}

impl<D: Device> Gate<D> {
    /// The gate's weights, as [`Device::gates`] takes them.
    fn weights(&self) -> [&D::Weight; 4] {
        [
            &self.down_input,
            &self.down_message,
            &self.gain,
            &self.forget,
        ]
    }
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
        if max_len == 0 || max_len > config.context_length {
            return Err(format!(
                "a context of {max_len} tokens is outside the 1 to {} the model was trained for",
                config.context_length
            )
            .into());
        }
        let c = &config;
        let mut l = Loader {
            device: &device,
            gguf: &gguf,
            done: 0,
            // Every tensor but the embedding table.
            total: gguf.tensors().count() - 1,
            on_progress,
        };
        let embed_norm = l.vector("token_embd_norm.weight", c.hidden)?;
        let mut layers = Vec::with_capacity(c.layers);
        for i in 0..c.layers {
            layers.push(load_layer(&mut l, c, i)?);
        }
        let mut loops = Vec::with_capacity(c.core_end - c.core_start);
        for i in c.core_start..c.core_end {
            loops.push(load_loop(&mut l, c, i)?);
        }
        let out_query = l.vector("output_res_query.weight", c.hidden)?;
        let out_norm = l.vector("output_norm.weight", c.hidden)?;
        let head = l.weight("output.weight")?;
        let ones = |len: usize| {
            let mut buf = device.alloc(len);
            device.write(&mut buf, &vec![1.0; len]);
            buf
        };
        let (ones_head, ones_kv) = (ones(c.head_dim), ones(c.kv_dim()));
        let state = State::new(c, &device, max_len);
        Ok(Self {
            config,
            device,
            gguf,
            embed_norm,
            layers,
            loops,
            out_query,
            out_norm,
            head,
            ones_head,
            ones_kv,
            state,
            parked: HashMap::new(),
        })
    }

    /// Runs a batch of tokens, the first at position `pos`, and returns the
    /// logits after the last, if they are `wanted`: a batch that another
    /// follows has no use for them.
    fn forward_batch(&mut self, tokens: &[u32], pos: usize, wanted: bool) -> Vec<f32> {
        let (c, d) = (&self.config, &self.device);
        let p = Pass {
            c,
            d,
            ones_head: &self.ones_head,
            ones_kv: &self.ones_kv,
        };
        let s = &mut self.state;
        let n = tokens.len();
        let width = n * c.hidden;
        s.batch(c, d, n);

        // The embeddings, looked up on the CPU, and normalized.
        let mut x = vec![0.0; width];
        for (x, &token) in x.chunks_exact_mut(c.hidden).zip(tokens) {
            self.gguf
                .row("token_embd.weight", token as usize, x)
                .expect("checked at load");
        }
        d.write(&mut s.x, &x);
        d.rmsnorm(&mut s.x, &self.embed_norm, c.eps);

        s.kept = 0;
        for i in 0..c.core_start {
            p.layer(s, &self.layers[i], None, i, i, n, pos);
        }
        // Every pass through the core starts from the prelude's output,
        // with the blocks the prelude kept.
        d.copy(&mut s.prelude, 0, &s.x, 0, width);
        let prelude_kept = s.kept;
        for pass in 0..c.passes {
            if pass > 0 {
                // What the pass before left for this one to read: the
                // blocks it kept after its first, and its output, which
                // change places with what this pass will write over.
                s.banked = s.kept - prelude_kept - 1;
                for b in 0..s.banked {
                    mem::swap(&mut s.bank[b], &mut s.blocks[prelude_kept + 1 + b]);
                }
                mem::swap(&mut s.bank_out, &mut s.x);
                d.copy(&mut s.x, 0, &s.prelude, 0, width);
                s.kept = prelude_kept;
            }
            for i in c.core_start..c.core_end {
                let looped = (pass > 0).then(|| &self.loops[i - c.core_start]);
                if let Some(looped) = looped {
                    let mut states: Vec<&D::Buffer> = s.bank[..s.banked].iter().collect();
                    states.push(&s.bank_out);
                    let mut messages: Vec<&mut D::Buffer> = s.messages.iter_mut().collect();
                    let queries: Vec<&D::Buffer> = looped.queries.iter().collect();
                    d.mixes(&mut messages, &states, &queries, MIX_EPS, None);
                }
                p.layer(s, &self.layers[i], looped, c.slot_of(i, pass), i, n, pos);
            }
        }
        for i in c.core_end..c.layers {
            p.layer(s, &self.layers[i], None, i, i, n, pos);
        }
        if !wanted {
            // Waits for the batch, by reading an activation back: a batch
            // is a submission of its own to the device, not the start of
            // the next's.
            d.resize(&mut s.growth, 1);
            d.read(&s.growth);
            return Vec::new();
        }

        // Only the last token's logits are wanted: its row of the stream
        // and of each block kept, mixed, normalized, and through the head.
        // A token alone is its states' only row.
        let last = (n - 1) * c.hidden;
        let mut rows: Vec<&mut D::Buffer> = s.inputs.iter_mut().chain(&mut s.normed).collect();
        assert!(s.kept < rows.len(), "more blocks than there are rows for");
        let kept = s.blocks[..s.kept].iter().chain([&s.x]);
        let states: Vec<&D::Buffer> = if n == 1 {
            kept.collect()
        } else {
            for (row, state) in rows.iter_mut().zip(kept) {
                d.resize(row, c.hidden);
                d.copy(row, 0, state, last, c.hidden);
            }
            rows[..s.kept + 1].iter().map(|row| &**row).collect()
        };
        d.resize(&mut s.ffn_normed, c.hidden);
        d.mixes(
            &mut [&mut s.ffn_normed],
            &states,
            &[&self.out_query],
            MIX_EPS,
            Some((&[&self.out_norm], c.eps)),
        );
        d.matmul(&mut s.logits, &self.head, &s.ffn_normed);
        let mut logits = match c.softcap != 0.0 {
            true => d.read_capped(&mut s.logits, c.softcap),
            false => d.read(&s.logits),
        };
        // The rows past the vocabulary are padding.
        logits.truncate(c.vocab);
        logits
    }
}

/// What a pass through the model has at hand: the configuration, the
/// device, and the ones.
struct Pass<'a, D: Device> {
    c: &'a Config,
    d: &'a D,
    ones_head: &'a D::Buffer,
    ones_kv: &'a D::Buffer,
}

impl<D: Device> Pass<'_, D> {
    /// Runs `n` tokens, the first at position `pos`, through `layer`, the
    /// model's layer `index`, with the keys, values and convolution states
    /// of `slot`: from the residual stream in `s.x` and the blocks kept,
    /// and back into them. `looped` is what the layer reads the pass
    /// before with, in a pass after the first, with the messages for it
    /// in `s.messages`.
    #[allow(clippy::too_many_arguments)]
    fn layer(
        &self,
        s: &mut State<D>,
        layer: &Layer<D>,
        looped: Option<&Loop<D>>,
        slot: usize,
        index: usize,
        n: usize,
        pos: usize,
    ) {
        let (c, d) = (self.c, self.d);
        let width = n * c.hidden;
        let kv_dim = c.kv_dim();

        // Attention's inputs: the stream itself in the first layer, and a
        // mix of it and the blocks kept in the others. The queries, keys
        // and values read the same input in a windowed layer, unless a
        // message makes them differ, and one each in the others.
        let apart = !layer.local || looped.is_some();
        let used = if apart { 3 } else { 1 };
        let norm = |j: usize| &layer.attn_norm[j.min(layer.attn_norm.len() - 1)];
        let queries: Vec<&D::Buffer> = layer.attn_query.iter().take(used).collect();
        let mixed = queries.len();
        if looped.is_none() && (mixed == 0 || mixed == used) {
            // Nothing comes between an input and its norm: the stream is
            // normalized as it is, or the inputs are as they are mixed.
            if mixed == 0 {
                let mut outs: Vec<&mut D::Buffer> = s.normed.iter_mut().take(used).collect();
                let norms: Vec<&D::Buffer> = (0..used).map(norm).collect();
                d.normalizes(&mut outs, &vec![&s.x; used], &norms, width, c.eps);
            } else {
                let mut states: Vec<&D::Buffer> = s.blocks[..s.kept].iter().collect();
                states.push(&s.x);
                let mut outs: Vec<&mut D::Buffer> = s.normed.iter_mut().take(used).collect();
                let norms: Vec<&D::Buffer> = (0..used).map(norm).collect();
                d.mixes(&mut outs, &states, &queries, MIX_EPS, Some((&norms, c.eps)));
            }
        } else {
            if mixed > 0 {
                let mut states: Vec<&D::Buffer> = s.blocks[..s.kept].iter().collect();
                states.push(&s.x);
                let mut outs: Vec<&mut D::Buffer> = s.inputs.iter_mut().take(mixed).collect();
                d.mixes(&mut outs, &states, &queries, MIX_EPS, None);
            } else {
                d.copy(&mut s.inputs[0], 0, &s.x, 0, width);
            }
            let [q_in, rest @ ..] = &mut s.inputs;
            for input in rest.iter_mut().take(used - 1).skip(mixed.max(1) - 1) {
                d.copy(input, 0, q_in, 0, width);
            }
            if let Some(looped) = looped {
                // The messages for the three, each through its gate.
                let mut inputs: Vec<&mut D::Buffer> = s.inputs.iter_mut().collect();
                let messages: Vec<&D::Buffer> = s.messages[..FFN].iter().collect();
                let weights: Vec<_> = looped.gates[..FFN].iter().map(Gate::weights).collect();
                d.gates(
                    &mut inputs,
                    &messages,
                    &weights,
                    c.hidden,
                    c.gate_rank,
                    MIX_EPS,
                );
            }
            let mut outs: Vec<&mut D::Buffer> = s.normed.iter_mut().take(used).collect();
            let inputs: Vec<&D::Buffer> = s.inputs.iter().take(used).collect();
            let norms: Vec<&D::Buffer> = (0..used).map(norm).collect();
            d.normalizes(&mut outs, &inputs, &norms, width, c.eps);
        }
        let [q_n, k_n, v_n] = &s.normed;
        let (k_n, v_n) = if apart { (k_n, v_n) } else { (q_n, q_n) };

        // The projections: queries, the output gate, the bias's and the
        // sinks from the queries' input, and keys and values from theirs.
        d.matmuls(&mut [
            (&mut s.q, &layer.q, q_n),
            (&mut s.gate, &layer.gate, q_n),
            (&mut s.k, &layer.k, k_n),
            (&mut s.v, &layer.v, v_n),
            (&mut s.rel, &layer.rel, q_n),
        ]);
        d.matmul(&mut s.sinks, &layer.sink, q_n);
        d.clips(&mut [
            (&mut s.q, &layer.q_clip, c.head_dim, c.qk_clip),
            (&mut s.rel, &layer.rel_clip, c.rel, c.rel_clip),
        ]);
        // Far enough into a sequence, the queries of a layer that attends
        // over everything grow with the logarithm of the position, and
        // the bias with them.
        if !layer.local && pos + n > c.log_floor {
            d.copy(&mut s.growth, 0, &s.grown, pos, n);
            d.scale_rows(&mut s.q, &s.growth);
            d.scale_rows(&mut s.rel, &s.growth);
        }

        // The keys and values are convolved, the keys clipped and the
        // values normalized, a head each, and join the slot's caches, a
        // windowed layer's in a ring; and the queries attend over them.
        let (window, rows) = if layer.local {
            (c.window, s.seq.ring)
        } else {
            (usize::MAX, s.seq.max_len)
        };
        let taps = match (&layer.k_taps, &layer.v_taps) {
            (Some(k_taps), Some(v_taps)) => {
                d.matmul(&mut s.k_taps, k_taps, k_n);
                d.matmul(&mut s.v_taps, v_taps, v_n);
                [Some((&s.k_taps, c.kv_heads)), Some((&s.v_taps, c.kv_heads))]
            }
            _ => [None, None],
        };
        let [k_state, v_state, ..] = &mut s.seq.conv[slot];
        let clipped = Kept::Clipped {
            dim: c.head_dim,
            threshold: c.qk_clip,
        };
        d.short_conv_store(
            &mut s.seq.k_cache[slot],
            pos,
            rows,
            n,
            kv_dim,
            &s.k,
            k_state,
            &layer.conv[0],
            taps[0],
            self.ones_kv,
            clipped,
            &mut s.k_conv,
            &mut s.wrapped,
        );
        d.short_conv_store(
            &mut s.seq.v_cache[slot],
            pos,
            rows,
            n,
            kv_dim,
            &s.v,
            v_state,
            &layer.conv[1],
            taps[1],
            self.ones_head,
            Kept::Normed { eps: c.eps },
            &mut s.v_conv,
            &mut s.wrapped,
        );
        d.attention_rel(
            &mut s.att,
            &s.q,
            &s.seq.k_cache[slot],
            &s.seq.v_cache[slot],
            &s.rel,
            &layer.rel_bias,
            &s.sinks,
            pos,
            c.heads,
            c.head_dim,
            c.kv_heads,
            window,
            rows,
        );
        d.sigmoid_mul(&mut s.att, &s.gate);
        d.matmul(&mut s.out, &layer.o, &s.att);

        // The first layer of a block starts the stream anew from its
        // attention's output, convolved, and what the stream held joins
        // the blocks kept; the others add to the stream.
        let state = &mut s.seq.conv[slot][2];
        if index.is_multiple_of(c.block) {
            mem::swap(&mut s.blocks[s.kept], &mut s.x);
            s.kept += 1;
            d.short_conv(&mut s.x, &s.out, state, &layer.conv[2], None);
        } else {
            d.short_conv_add(&mut s.x, &s.out, state, &layer.conv[2], &mut s.out_conv);
        }

        // The mixture of experts, on a mix of the stream and the blocks
        // kept: the router picks each token's experts, whose outputs are
        // summed by what the router gives each.
        {
            let mut states: Vec<&D::Buffer> = s.blocks[..s.kept].iter().collect();
            states.push(&s.x);
            let query = [&layer.ffn_query];
            match looped {
                None => d.mixes(
                    &mut [&mut s.ffn_normed],
                    &states,
                    &query,
                    MIX_EPS,
                    Some((&[&layer.ffn_norm], c.eps)),
                ),
                Some(_) => d.mixes(&mut [&mut s.mixed], &states, &query, MIX_EPS, None),
            }
        }
        if let Some(looped) = looped {
            d.gates(
                &mut [&mut s.mixed],
                &[&s.messages[FFN]],
                &[looped.gates[FFN].weights()],
                c.hidden,
                c.gate_rank,
                MIX_EPS,
            );
            d.normalize(&mut s.ffn_normed, &s.mixed, &layer.ffn_norm, width, c.eps);
        }
        d.matmul(&mut s.router, &layer.router, &s.ffn_normed);
        let k = c.experts_used;
        d.route(
            &mut s.ids,
            &mut s.shares,
            &s.router,
            &layer.router_bias,
            k,
            c.expert_scale,
        );
        d.expert_ffn(
            &mut s.down,
            &layer.gate_up,
            &layer.gate_up_scale,
            &layer.down,
            &layer.down_scale,
            &s.ffn_normed,
            &s.ids,
            k,
            c.expert_width,
            c.situ_beta,
            c.situ_linear_beta,
            &mut s.gate_up,
            &mut s.act,
        );
        d.weighted_sum(&mut s.out, &s.down, &s.shares);
        let state = &mut s.seq.conv[slot][3];
        d.short_conv_add(&mut s.x, &s.out, state, &layer.conv[3], &mut s.out_conv);
    }
}

/// Uploads the weights, counting the file's tensors off.
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
        self.tick();
        Ok(self.device.upload(tensor))
    }

    /// The named f32 tensor as rows of `width`, each a buffer.
    fn vectors(&mut self, name: &str, width: usize) -> Result<Vec<D::Buffer>> {
        let values = self.gguf.f32s(name)?;
        if values.is_empty() || !values.len().is_multiple_of(width) {
            return Err(format!("{name} is not rows of {width}").into());
        }
        self.tick();
        Ok(values
            .chunks_exact(width)
            .map(|row| self.buffer(row))
            .collect())
    }

    /// The named f32 tensor, of `len` values, as a buffer.
    fn vector(&mut self, name: &str, len: usize) -> Result<D::Buffer> {
        match <[D::Buffer; 1]>::try_from(self.vectors(name, len)?) {
            Ok([buffer]) => Ok(buffer),
            Err(_) => Err(format!("{name} is not {len} values").into()),
        }
    }

    fn buffer(&self, values: &[f32]) -> D::Buffer {
        let mut buf = self.device.alloc(values.len());
        self.device.write(&mut buf, values);
        buf
    }

    /// The matrices of a layer's experts, a matrix of `rows` by `cols`
    /// each, and each expert's scale, which is one where the file has
    /// none.
    fn experts(
        &mut self,
        name: &str,
        experts: usize,
        rows: usize,
        cols: usize,
    ) -> Result<(D::Weight, D::Buffer)> {
        let weight = format!("{name}.weight");
        let (tensor, count) = self.gguf.experts(&weight)?;
        if count != experts || tensor.shape() != [experts * rows, cols] {
            return Err(format!("{weight} is not {experts} matrices of {rows} by {cols}").into());
        }
        if matches!(tensor, Tensor::Nvfp4 { .. }) && !self.device.takes_nvfp4() {
            return Err(format!(
                "{} does not run weights in NVFP4, which the model's experts are in",
                self.device.name()
            )
            .into());
        }
        self.tick();
        let scale = format!("{name}.scale");
        let scale = if self.gguf.info(&scale).is_ok() {
            self.vector(&scale, experts)?
        } else {
            self.buffer(&vec![1.0; experts])
        };
        Ok((self.device.upload(tensor), scale))
    }
}

/// Loads layer `i`.
fn load_layer<D: Device, F: FnMut(usize, usize)>(
    l: &mut Loader<'_, D, F>,
    c: &Config,
    i: usize,
) -> Result<Layer<D>> {
    let p = format!("blk.{i}");
    let local = c.local[i];
    let attn_query = if i == 0 {
        Vec::new()
    } else {
        l.vectors(&format!("{p}.attn_res_query.weight"), c.hidden)?
    };
    let attn_norm = if local {
        vec![l.vector(&format!("{p}.attn_norm.weight"), c.hidden)?]
    } else {
        vec![
            l.vector(&format!("{p}.attn_q_inp_norm.weight"), c.hidden)?,
            l.vector(&format!("{p}.attn_k_inp_norm.weight"), c.hidden)?,
            l.vector(&format!("{p}.attn_v_inp_norm.weight"), c.hidden)?,
        ]
    };
    if i > 0 && attn_query.len() != attn_norm.len() {
        return Err(format!("{p} does not have a query over the depth for each input").into());
    }
    let extent = if local { c.window } else { c.rel_extent };
    let channels = c.conv_channels();
    let (gate_up, gate_up_scale) = l.experts(
        &format!("{p}.ffn_gate_up_exps"),
        c.experts,
        2 * c.expert_width,
        c.hidden,
    )?;
    let (down, down_scale) = l.experts(
        &format!("{p}.ffn_down_exps"),
        c.experts,
        c.hidden,
        c.expert_width,
    )?;
    let layer = Layer {
        local,
        attn_query,
        attn_norm,
        q: l.weight(&format!("{p}.attn_q.weight"))?,
        k: l.weight(&format!("{p}.attn_k.weight"))?,
        v: l.weight(&format!("{p}.attn_v.weight"))?,
        o: l.weight(&format!("{p}.attn_output.weight"))?,
        gate: l.weight(&format!("{p}.attn_gate.weight"))?,
        rel: l.weight(&format!("{p}.attn_rel.weight"))?,
        rel_bias: l.vector(&format!("{p}.attn_rel_bias.weight"), extent * c.rel)?,
        sink: l.weight(&format!("{p}.attn_sink_proj.weight"))?,
        q_clip: l.vector(&format!("{p}.attn_q_clip.weight"), c.q_dim())?,
        rel_clip: l.vector(&format!("{p}.attn_rel_clip.weight"), c.heads * c.rel)?,
        conv: [
            l.vector(&format!("{p}.attn_k_conv.weight"), c.taps * channels[0])?,
            l.vector(&format!("{p}.attn_v_conv.weight"), c.taps * channels[1])?,
            l.vector(&format!("{p}.attn_post_conv.weight"), c.taps * channels[2])?,
            l.vector(&format!("{p}.ffn_post_conv.weight"), c.taps * channels[3])?,
        ],
        k_taps: match local {
            true => None,
            false => Some(l.weight(&format!("{p}.attn_k_conv_proj.weight"))?),
        },
        v_taps: match local {
            true => None,
            false => Some(l.weight(&format!("{p}.attn_v_conv_proj.weight"))?),
        },
        ffn_query: l.vector(&format!("{p}.ffn_res_query.weight"), c.hidden)?,
        ffn_norm: l.vector(&format!("{p}.ffn_norm.weight"), c.hidden)?,
        router: l.weight(&format!("{p}.ffn_gate_inp.weight"))?,
        router_bias: l.vector(&format!("{p}.exp_probs_b.bias"), c.experts)?,
        gate_up,
        gate_up_scale,
        down,
        down_scale,
    };
    Ok(layer)
}

/// Loads what core layer `i` reads the pass before with.
fn load_loop<D: Device, F: FnMut(usize, usize)>(
    l: &mut Loader<'_, D, F>,
    c: &Config,
    i: usize,
) -> Result<Loop<D>> {
    let p = format!("blk.{i}.loop");
    let name = format!("{p}_msg_query.weight");
    let queries = <[D::Buffer; CONTROLS]>::try_from(l.vectors(&name, c.hidden)?)
        .map_err(|_| format!("{name} is not a query for each message"))?;
    let mut gates = Vec::with_capacity(CONTROLS);
    for control in ["q", "k", "v", "ffn"] {
        // The projection down takes the input first, then the message:
        // the two halves are kept apart, so that the two can be
        // normalized and multiplied as they are and the results added.
        let name = format!("{p}_{control}_ctrl_down.weight");
        let both = l.gguf.matrix(&name)?;
        if both.shape() != [c.gate_rank, 2 * c.hidden] {
            return Err(format!("{name} does not take an input and a message").into());
        }
        l.tick();
        gates.push(Gate {
            down_input: l.device.upload(both.select_column_groups(c.hidden, &[0])),
            down_message: l.device.upload(both.select_column_groups(c.hidden, &[1])),
            gain: l.weight(&format!("{p}_{control}_ctrl_gain.weight"))?,
            forget: l.weight(&format!("{p}_{control}_ctrl_forget.weight"))?,
        });
    }
    let Ok(gates) = <[Gate<D>; CONTROLS]>::try_from(gates) else {
        unreachable!("a gate for each control");
    };
    Ok(Loop { queries, gates })
}

/// The first bytes of a saved state, and its layout's version.
const STATE_MAGIC: &[u8; 8] = b"dwimink2";
const STATE_VERSION: u32 = 1;

impl<D: Device> LanguageModel for Model<D> {
    /// The state after `len` positions: a header of the layout's version
    /// and the model's dimensions, then each slot's keys and values as the
    /// caches hold them, a windowed layer's as its ring has them, then
    /// each slot's convolution states.
    fn save(&self, len: usize) -> Option<Vec<u8>> {
        let (c, d, s) = (&self.config, &self.device, &self.state);
        let kv_dim = c.kv_dim();
        let mut out = Vec::new();
        out.extend_from_slice(STATE_MAGIC);
        out.extend_from_slice(&STATE_VERSION.to_le_bytes());
        for value in [len, c.slots(), kv_dim, s.seq.ring, c.hidden, c.taps] {
            out.extend_from_slice(&(value as u64).to_le_bytes());
        }
        for slot in 0..c.slots() {
            let rows = s.seq.rows(c, slot, len);
            for cache in [&s.seq.k_cache[slot], &s.seq.v_cache[slot]] {
                out.extend(d.read_cache(cache, rows * kv_dim));
            }
        }
        for conv in s.seq.conv.iter().flatten() {
            out.extend(d.read(conv).iter().flat_map(|v| v.to_le_bytes()));
        }
        Some(out)
    }

    fn restore(&mut self, state: &[u8]) -> Result<usize> {
        let (c, d, s) = (&self.config, &self.device, &mut self.state);
        let kv_dim = c.kv_dim();
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
        let mut header = [0; 6];
        for value in &mut header {
            *value = u64::from_le_bytes(take(8)?.try_into()?) as usize;
        }
        let len = header[0];
        if header[1..] != [c.slots(), kv_dim, s.seq.ring, c.hidden, c.taps] {
            return Err("a saved state of another model".into());
        }
        if len > s.seq.max_len {
            return Err(
                format!("a saved state of {len} tokens does not fit in the context").into(),
            );
        }
        for slot in 0..c.slots() {
            let rows = s.seq.rows(c, slot, len);
            for cache in [&mut s.seq.k_cache[slot], &mut s.seq.v_cache[slot]] {
                d.write_cache(cache, take(cache_bytes(rows * kv_dim))?);
            }
        }
        for (conv, channels) in s
            .seq
            .conv
            .iter_mut()
            .flat_map(|slot| slot.iter_mut().zip(c.conv_channels()))
        {
            let floats: Vec<f32> = take((c.taps - 1) * channels * 4)?
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            d.write(conv, &floats);
        }
        if s.marked == Some(s.seq.id) {
            s.marked = None;
        }
        Ok(len)
    }

    /// Adds the tokens to the model's state. Tokens run through the model
    /// in batches of as many as the device takes at once.
    fn forward(&mut self, tokens: &[u32], pos: usize) -> Vec<f32> {
        assert!(!tokens.is_empty(), "no tokens to run");
        assert!(
            pos + tokens.len() <= self.state.seq.max_len,
            "tokens past the end of the cache"
        );
        let mut logits = Vec::new();
        let most = self.state.most;
        let batches = tokens.len().div_ceil(most);
        for (i, batch) in tokens.chunks(most).enumerate() {
            logits = self.forward_batch(batch, pos + i * most, i + 1 == batches);
        }
        logits
    }

    /// The model's drafter is a file of its own, which is not read: it
    /// drafts nothing.
    fn draft(&mut self, _token: u32, _pos: usize, _k: usize) -> Vec<u32> {
        Vec::new()
    }

    fn verify(&mut self, _tokens: &[u32], _pos: usize) -> Vec<f32> {
        panic!("the model does not draft");
    }

    fn rewind(&mut self, _keep: usize) {
        panic!("the model does not draft");
    }

    /// Copies the windowed layers' rings aside, on the device, and the
    /// convolutions' states. The caches of the layers that attend over
    /// everything need no copy: attention reads only the positions before
    /// the token's, so the positions after the mark are written over as
    /// the sequence goes on from it again. A ring's are written over the
    /// positions before the mark.
    fn mark(&mut self) -> bool {
        let (c, d, s) = (&self.config, &self.device, &mut self.state);
        let len = s.seq.ring * c.kv_dim();
        let mut rings = s.mark_rings.iter_mut();
        for slot in (0..c.slots()).filter(|&slot| c.local[c.layer_of(slot)]) {
            for cache in [&s.seq.k_cache[slot], &s.seq.v_cache[slot]] {
                d.copy_cache(rings.next().expect("a copy for each ring"), cache, len);
            }
        }
        for ((mark, conv), channels) in s
            .mark_conv
            .iter_mut()
            .zip(s.seq.conv.iter().flatten())
            .zip(c.conv_channels().into_iter().cycle())
        {
            d.copy(mark, 0, conv, 0, (c.taps - 1) * channels);
        }
        s.marked = Some(s.seq.id);
        true
    }

    /// Copies the rings and the convolutions' states back from the mark.
    fn rewind_to_mark(&mut self) -> Result<()> {
        let (c, d, s) = (&self.config, &self.device, &mut self.state);
        if s.marked != Some(s.seq.id) {
            return Err("no mark to rewind to".into());
        }
        let len = s.seq.ring * c.kv_dim();
        let mut rings = s.mark_rings.iter();
        for slot in (0..c.slots()).filter(|&slot| c.local[c.layer_of(slot)]) {
            for cache in [&mut s.seq.k_cache[slot], &mut s.seq.v_cache[slot]] {
                d.copy_cache(cache, rings.next().expect("a copy for each ring"), len);
            }
        }
        for ((mark, conv), channels) in s
            .mark_conv
            .iter()
            .zip(s.seq.conv.iter_mut().flatten())
            .zip(c.conv_channels().into_iter().cycle())
        {
            d.copy(conv, 0, mark, 0, (c.taps - 1) * channels);
        }
        Ok(())
    }

    /// Zeroes the convolutions' states, as they were before any token.
    /// The caches need no clearing: attention reads only the positions
    /// before the token's.
    fn reset(&mut self) {
        let (c, d, s) = (&self.config, &self.device, &mut self.state);
        for (conv, channels) in s
            .seq
            .conv
            .iter_mut()
            .flat_map(|slot| slot.iter_mut().zip(c.conv_channels()))
        {
            d.write(conv, &vec![0.0; (c.taps - 1) * channels]);
        }
        if s.marked == Some(s.seq.id) {
            s.marked = None;
        }
    }

    fn max_len(&self) -> usize {
        self.state.seq.max_len
    }

    fn batch(&self) -> usize {
        self.state.most
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
            self.state.most,
        );
        self.parked.insert(id, seq);
        Ok(id)
    }

    /// Swaps the sequence the model runs for another: nothing is copied.
    fn select_sequence(&mut self, id: usize) -> Result<()> {
        if self.state.seq.id == id {
            return Ok(());
        }
        let mut seq = self
            .parked
            .remove(&id)
            .ok_or_else(|| format!("no sequence {id}"))?;
        mem::swap(&mut self.state.seq, &mut seq);
        self.parked.insert(seq.id, seq);
        Ok(())
    }
}

/// Buffers the forward pass computes in, with room for a batch of `most`
/// tokens, and the sequence the model is running.
struct State<D: Device> {
    /// Most tokens to a batch, which the per-token buffers have room for.
    most: usize,
    /// The residual stream.
    x: D::Buffer,
    /// What the stream held where each block began, as many as are
    /// `kept`.
    blocks: Vec<D::Buffer>,
    kept: usize,
    /// The prelude's output, which every pass through the core starts
    /// from; what the pass before kept after its first block, as many as
    /// are `banked`, and its output; and the messages read from those.
    prelude: D::Buffer,
    bank: Vec<D::Buffer>,
    banked: usize,
    bank_out: D::Buffer,
    messages: [D::Buffer; CONTROLS],
    /// Attention's inputs for the queries, keys and values, and the same
    /// normalized.
    inputs: [D::Buffer; 3],
    normed: [D::Buffer; 3],
    q: D::Buffer,
    gate: D::Buffer,
    k: D::Buffer,
    v: D::Buffer,
    k_conv: D::Buffer,
    v_conv: D::Buffer,
    /// The part of a batch's keys or values that goes to the start of a
    /// ring.
    wrapped: D::Buffer,
    k_taps: D::Buffer,
    v_taps: D::Buffer,
    rel: D::Buffer,
    sinks: D::Buffer,
    /// What each token's queries grow by, far into a sequence, and what
    /// the queries at every position there is room for do: one, up to
    /// where they start to grow.
    growth: D::Buffer,
    grown: D::Buffer,
    att: D::Buffer,
    /// The output of attention or of the experts, and the same convolved.
    out: D::Buffer,
    out_conv: D::Buffer,
    /// The experts' input, mixed from the depth, and the same normalized.
    mixed: D::Buffer,
    ffn_normed: D::Buffer,
    router: D::Buffer,
    ids: D::Buffer,
    shares: D::Buffer,
    gate_up: D::Buffer,
    act: D::Buffer,
    down: D::Buffer,
    logits: D::Buffer,
    /// The copy a mark keeps of the windowed layers' rings and of the
    /// convolutions' states, and which sequence it is of.
    mark_rings: Vec<D::Cache>,
    mark_conv: Vec<D::Buffer>,
    marked: Option<usize>,
    seq: Sequence<D>,
}

/// The state of a sequence: each slot's keys and values, holding every
/// position seen so far in 8-bit blocks, or the latest in a ring, and the
/// states of its convolutions.
struct Sequence<D: Device> {
    id: usize,
    max_len: usize,
    /// Positions a windowed layer's caches hold.
    ring: usize,
    k_cache: Vec<D::Cache>,
    v_cache: Vec<D::Cache>,
    conv: Vec<[D::Buffer; CONVS]>,
}

impl<D: Device> Sequence<D> {
    fn new(c: &Config, device: &D, id: usize, max_len: usize, batch: usize) -> Self {
        let ring = c.ring(batch, max_len);
        let cache = |slot: usize| {
            let rows = if c.local[c.layer_of(slot)] {
                ring
            } else {
                max_len
            };
            device.alloc_cache(rows * c.kv_dim())
        };
        Self {
            id,
            max_len,
            ring,
            k_cache: (0..c.slots()).map(cache).collect(),
            v_cache: (0..c.slots()).map(cache).collect(),
            conv: (0..c.slots())
                .map(|_| {
                    c.conv_channels()
                        .map(|channels| device.alloc((c.taps - 1) * channels))
                })
                .collect(),
        }
    }

    /// Rows of a slot's caches that hold positions, `len` into the
    /// sequence.
    fn rows(&self, c: &Config, slot: usize, len: usize) -> usize {
        if c.local[c.layer_of(slot)] {
            len.min(self.ring)
        } else {
            len
        }
    }
}

impl<D: Device> State<D> {
    fn new(c: &Config, device: &D, max_len: usize) -> Self {
        let most = device.batch() * BATCHES;
        let k = c.experts_used;
        let buf = |width: usize| device.alloc(most * width);
        let hidden = || buf(c.hidden);
        let seq = Sequence::new(c, device, 0, max_len, most);
        let local = (0..c.slots()).filter(|&s| c.local[c.layer_of(s)]).count();
        Self {
            most,
            x: hidden(),
            blocks: (0..c.layers / c.block).map(|_| hidden()).collect(),
            kept: 0,
            prelude: hidden(),
            bank: (0..(c.core_end - c.core_start) / c.block)
                .map(|_| hidden())
                .collect(),
            banked: 0,
            bank_out: hidden(),
            messages: [hidden(), hidden(), hidden(), hidden()],
            inputs: [hidden(), hidden(), hidden()],
            normed: [hidden(), hidden(), hidden()],
            q: buf(c.q_dim()),
            gate: buf(c.q_dim()),
            k: buf(c.kv_dim()),
            v: buf(c.kv_dim()),
            k_conv: buf(c.kv_dim()),
            v_conv: buf(c.kv_dim()),
            wrapped: buf(c.kv_dim()),
            k_taps: buf(c.kv_heads * c.taps),
            v_taps: buf(c.kv_heads * c.taps),
            rel: buf(c.heads * c.rel),
            sinks: buf(c.heads),
            growth: buf(1),
            grown: {
                let growth: Vec<f32> = (0..max_len)
                    .map(|at| {
                        let past = ((at + 1) as f32 / c.log_floor as f32).max(1.0);
                        1.0 + c.log_alpha * past.ln()
                    })
                    .collect();
                let mut grown = device.alloc(max_len);
                device.write(&mut grown, &growth);
                grown
            },
            att: buf(c.q_dim()),
            out: hidden(),
            out_conv: hidden(),
            mixed: hidden(),
            ffn_normed: hidden(),
            router: buf(c.experts),
            ids: buf(k),
            shares: buf(k),
            gate_up: buf(k * 2 * c.expert_width),
            act: buf(k * c.expert_width),
            down: buf(k * c.hidden),
            logits: device.alloc(c.vocab_padded),
            mark_rings: (0..2 * local)
                .map(|_| device.alloc_cache(seq.ring * c.kv_dim()))
                .collect(),
            mark_conv: (0..c.slots())
                .flat_map(|_| c.conv_channels())
                .map(|channels| device.alloc((c.taps - 1) * channels))
                .collect(),
            marked: None,
            seq,
        }
    }

    /// Sizes the per-token buffers for a batch of `n` tokens.
    fn batch(&mut self, c: &Config, d: &D, n: usize) {
        assert!(n <= self.most, "a batch of {n} tokens is too many");
        let k = c.experts_used;
        let hidden = [
            &mut self.x,
            &mut self.prelude,
            &mut self.bank_out,
            &mut self.out,
            &mut self.out_conv,
            &mut self.mixed,
            &mut self.ffn_normed,
        ]
        .into_iter()
        .chain(&mut self.blocks)
        .chain(&mut self.bank)
        .chain(&mut self.messages)
        .chain(&mut self.inputs)
        .chain(&mut self.normed);
        for buf in hidden {
            d.resize(buf, n * c.hidden);
        }
        for (buf, width) in [
            (&mut self.q, c.q_dim()),
            (&mut self.gate, c.q_dim()),
            (&mut self.att, c.q_dim()),
            (&mut self.k, c.kv_dim()),
            (&mut self.v, c.kv_dim()),
            (&mut self.k_conv, c.kv_dim()),
            (&mut self.v_conv, c.kv_dim()),
            (&mut self.k_taps, c.kv_heads * c.taps),
            (&mut self.v_taps, c.kv_heads * c.taps),
            (&mut self.rel, c.heads * c.rel),
            (&mut self.sinks, c.heads),
            (&mut self.growth, 1),
            (&mut self.router, c.experts),
            (&mut self.ids, k),
            (&mut self.shares, k),
            (&mut self.gate_up, k * 2 * c.expert_width),
            (&mut self.act, k * c.expert_width),
            (&mut self.down, k * c.hidden),
        ] {
            d.resize(buf, n * width);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use super::*;
    use crate::{
        gpu::{Cpu, Gpu, float::to_bf16},
        inference::quants,
        model::gguf::Value,
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

        fn bf16(&mut self, n: usize, scale: f32) -> Vec<u8> {
            (0..n)
                .flat_map(|_| to_bf16(self.next() * scale).to_le_bytes())
                .collect()
        }

        fn q8_0(&mut self, rows: usize, cols: usize, scale: f32) -> Vec<u8> {
            (0..rows)
                .flat_map(|_| {
                    let values: Vec<f32> = (0..cols).map(|_| self.next() * scale).collect();
                    quants::pack_q8_0(&values)
                })
                .collect()
        }

        /// Blocks of NVFP4 as the file lays them out: every value there
        /// is, under scales of a few hundredths.
        fn nvfp4(&mut self, weights: usize) -> Vec<u8> {
            let mut byte = |below: f32| ((self.next() + 1.0) * 0.5 * below) as u8;
            let mut data = Vec::new();
            for _ in 0..weights / quants::NVFP4_FILE_BLOCK {
                data.extend((0..4).map(|_| (2 << 3) + byte(16.0)));
                data.extend((0..32).map(|_| byte(256.0)));
            }
            data
        }
    }

    /// Writes a tiny random model in the file format as Inkling's file is:
    /// six layers in blocks of two, the second block run three times, the
    /// last layer of each block attending over everything and the others
    /// over a window of four; four experts of which a token goes through
    /// two, in NVFP4 with scales in the first four layers and Q8_0 in the
    /// last two.
    fn synthetic(path: &Path, used: usize) {
        let (hidden, layers, heads, kv_heads, head_dim) = (64, 6, 4, 2, 32);
        let (window, rel, extent, taps, rank) = (4, 4, 8, 4, 8);
        let (experts, inner) = (4, 64);
        let (vocab, padded) = (40, 48);
        let (q_dim, kv_dim) = (heads * head_dim, kv_heads * head_dim);
        let mut rng = Rng(2);
        let mut tensors: Vec<(String, Vec<usize>, u32, Vec<u8>)> = Vec::new();
        let mut add = |name: String, dims: &[usize], dtype: u32, data: Vec<u8>| {
            tensors.push((name, dims.to_vec(), dtype, data));
        };
        // Weights are small so that the activations stay sane through the
        // layers.
        let small = |cols: usize| 1.0 / (cols as f32).sqrt();
        add(
            "token_embd.weight".into(),
            &[hidden, padded],
            gguf::Q8_0,
            rng.q8_0(padded, hidden, 0.5),
        );
        add(
            "output.weight".into(),
            &[hidden, padded],
            gguf::Q8_0,
            rng.q8_0(padded, hidden, small(hidden)),
        );
        for name in ["token_embd_norm", "output_norm"] {
            let data = rng.floats(hidden, 0.1, 1.0);
            add(format!("{name}.weight"), &[hidden], gguf::F32, data);
        }
        let data = rng.floats(hidden, 1.0, 0.0);
        add("output_res_query.weight".into(), &[hidden], gguf::F32, data);
        for i in 0..layers {
            let p = format!("blk.{i}");
            let local = i % 2 == 0;
            for (name, rows, cols) in [
                ("attn_q", q_dim, hidden),
                ("attn_k", kv_dim, hidden),
                ("attn_v", kv_dim, hidden),
                ("attn_output", hidden, q_dim),
                ("attn_gate", q_dim, hidden),
                ("attn_rel", heads * rel, hidden),
            ] {
                let data = rng.q8_0(rows, cols, small(cols));
                add(
                    format!("{p}.{name}.weight"),
                    &[cols, rows],
                    gguf::Q8_0,
                    data,
                );
            }
            let data = rng.bf16(heads * hidden, small(hidden));
            add(
                format!("{p}.attn_sink_proj.weight"),
                &[hidden, heads],
                gguf::BF16,
                data,
            );
            let floats: &[(&str, &[usize], f32, f32)] = &[
                ("attn_q_clip", &[head_dim, heads], 0.5, 1.0),
                ("attn_rel_clip", &[rel, heads], 0.5, 1.0),
                (
                    "attn_rel_bias",
                    &[rel, if local { window } else { extent }],
                    1.0,
                    0.0,
                ),
                ("attn_k_conv", &[taps, kv_dim], 0.5, 0.0),
                ("attn_v_conv", &[taps, kv_dim], 0.5, 0.0),
                ("attn_post_conv", &[taps, hidden], 0.5, 0.0),
                ("ffn_post_conv", &[taps, hidden], 0.5, 0.0),
                ("ffn_res_query", &[hidden], 1.0, 0.0),
                ("ffn_norm", &[hidden], 0.1, 1.0),
                ("ffn_gate_inp", &[hidden, experts], 0.5, 0.0),
            ];
            for &(name, dims, scale, offset) in floats {
                let data = rng.floats(dims.iter().product(), scale, offset);
                add(format!("{p}.{name}.weight"), dims, gguf::F32, data);
            }
            let data = rng.floats(experts, 0.2, 0.0);
            add(format!("{p}.exp_probs_b.bias"), &[experts], gguf::F32, data);
            if local {
                let data = rng.floats(hidden, 0.1, 1.0);
                add(format!("{p}.attn_norm.weight"), &[hidden], gguf::F32, data);
            } else {
                for name in ["attn_q_inp_norm", "attn_k_inp_norm", "attn_v_inp_norm"] {
                    let data = rng.floats(hidden, 0.1, 1.0);
                    add(format!("{p}.{name}.weight"), &[hidden], gguf::F32, data);
                }
                for name in ["attn_k_conv_proj", "attn_v_conv_proj"] {
                    let data = rng.bf16(taps * kv_heads * hidden, small(hidden));
                    add(
                        format!("{p}.{name}.weight"),
                        &[hidden, taps * kv_heads],
                        gguf::BF16,
                        data,
                    );
                }
            }
            if i > 0 {
                let queries = if local { 1 } else { 3 };
                let dims: &[usize] = if local { &[hidden] } else { &[hidden, 3] };
                let data = rng.floats(queries * hidden, 1.0, 0.0);
                add(format!("{p}.attn_res_query.weight"), dims, gguf::F32, data);
            }
            for (name, rows, cols) in [
                ("ffn_gate_up_exps", 2 * inner, hidden),
                ("ffn_down_exps", hidden, inner),
            ] {
                let dims = [cols, rows, experts];
                if i < 4 {
                    let data = rng.nvfp4(experts * rows * cols);
                    add(format!("{p}.{name}.weight"), &dims, gguf::NVFP4, data);
                    let data = rng.floats(experts, 0.2, 1.0);
                    add(format!("{p}.{name}.scale"), &[experts], gguf::F32, data);
                } else {
                    let data = rng.q8_0(experts * rows, cols, small(cols));
                    add(format!("{p}.{name}.weight"), &dims, gguf::Q8_0, data);
                }
            }
            if (2..4).contains(&i) {
                let data = rng.floats(CONTROLS * hidden, 1.0, 0.0);
                add(
                    format!("{p}.loop_msg_query.weight"),
                    &[hidden, CONTROLS],
                    gguf::F32,
                    data,
                );
                for control in ["q", "k", "v", "ffn"] {
                    let p = format!("{p}.loop_{control}_ctrl");
                    let data = rng.floats(rank * 2 * hidden, small(hidden), 0.0);
                    add(
                        format!("{p}_down.weight"),
                        &[2 * hidden, rank],
                        gguf::F32,
                        data,
                    );
                    for name in ["gain", "forget"] {
                        let data = rng.floats(hidden * rank, 0.5, 0.0);
                        add(
                            format!("{p}_{name}.weight"),
                            &[rank, hidden],
                            gguf::F32,
                            data,
                        );
                    }
                }
            }
        }
        let n = |v: usize| Value::U32(v as u32);
        let each = |v: usize| Value::Array(vec![n(v); layers]);
        let pattern = (0..layers).map(|i| Value::Bool(i % 2 == 0)).collect();
        let a = ARCHITECTURE;
        let key = |name: &str| -> &'static str { format!("{a}.{name}").leak() };
        let meta = vec![
            ("general.architecture", Value::Str(a.into())),
            (key("block_count"), n(layers)),
            (key("context_length"), n(1024)),
            (key("embedding_length"), n(hidden)),
            (key("attention.head_count"), each(heads)),
            (key("attention.head_count_kv"), each(kv_heads)),
            (key("attention.key_length"), n(head_dim)),
            (key("attention.value_length"), n(head_dim)),
            (key("attention.key_length_swa"), n(head_dim)),
            (key("attention.value_length_swa"), n(head_dim)),
            (key("attention.layer_norm_rms_epsilon"), Value::F32(1e-6)),
            (key("attention.sliding_window"), n(window)),
            (
                key("attention.sliding_window_pattern"),
                Value::Array(pattern),
            ),
            (key("attention.relative_length"), n(rel)),
            (key("attention.relative_extent"), n(extent)),
            (key("attention.qk_clip"), Value::F32(4.0)),
            (key("attention.relative_clip"), Value::F32(2.0)),
            (key("attention.log_scale_floor"), n(16)),
            (key("attention.log_scale_alpha"), Value::F32(0.1)),
            (key("shortconv.l_cache"), n(taps)),
            (key("attn_res.block_size"), n(2)),
            (key("expert_count"), n(experts)),
            (key("expert_used_count"), n(used)),
            (key("expert_feed_forward_length"), n(inner)),
            (key("expert_weights_scale"), Value::F32(2.0)),
            (key("activation.situ_beta"), Value::F32(4.0)),
            (key("activation.situ_linear_beta"), Value::F32(25.0)),
            (key("final_logit_softcapping"), Value::F32(30.0)),
            (key("loop.start"), n(2)),
            (key("loop.end"), n(4)),
            (key("loop.count"), n(3)),
            (key("loop.gate_rank"), n(rank)),
            (key("vocab_size"), n(padded)),
            (key("vocab_size_unpadded"), n(vocab)),
        ];
        gguf::write(path, &meta, &tensors).unwrap();
    }

    /// A tiny model's file in a directory of its own, removed when dropped.
    struct File(PathBuf);

    impl File {
        fn new(name: &str) -> Self {
            Self::routing(name, 2)
        }

        /// The file of a model whose tokens go through `used` of its four
        /// experts each.
        fn routing(name: &str, used: usize) -> Self {
            let dir =
                std::env::temp_dir().join(format!("dwim-inkling2-{name}-{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            synthetic(&dir.join("model.gguf"), used);
            Self(dir)
        }

        fn load<D: Device>(&self, device: D, max_len: usize) -> Model<D> {
            let gguf = Arc::new(Gguf::open(&self.0.join("model.gguf")).unwrap());
            Model::load(gguf, device, max_len, |_, _| {}).unwrap()
        }
    }

    impl Drop for File {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// A sequence of tokens: the tests' are longer than a windowed layer's
    /// ring, which is the window and a batch.
    fn sequence(len: usize) -> Vec<u32> {
        (0..len)
            .map(|i| (i * 7 % 37 + i / 11) as u32 % 40)
            .collect()
    }

    fn close(a: &[f32], b: &[f32], tolerance: f32) {
        assert_eq!(a.len(), b.len());
        for (i, (a, b)) in a.iter().zip(b).enumerate() {
            assert!((a - b).abs() <= tolerance, "logit {i}: {a} vs {b}");
        }
    }

    #[test]
    fn logits_do_not_depend_on_how_tokens_are_batched() {
        let file = File::new("batches");
        let tokens = sequence(600);
        // Whole batches, which wrap the rings mid-batch; a token at a
        // time; and uneven pieces.
        let mut model = file.load(Cpu, 700);
        let whole = model.forward(&tokens, 0);
        assert_eq!(whole.len(), 40);
        assert!(whole.iter().all(|logit| logit.is_finite()));
        model.reset();
        let mut single = Vec::new();
        for (i, token) in tokens.iter().enumerate() {
            single = model.forward(&[*token], i);
        }
        close(&single, &whole, 1e-4);
        model.reset();
        let mut pieces = Vec::new();
        let mut at = 0;
        for size in [3, 256, 1, 130, 210] {
            pieces = model.forward(&tokens[at..at + size], at);
            at += size;
        }
        close(&pieces, &whole, 1e-4);
        // The tokens matter, and so does what came long before the last.
        let mut other = tokens.clone();
        other[20] = (other[20] + 1) % 40;
        model.reset();
        let changed = model.forward(&other, 0);
        assert!(
            changed
                .iter()
                .zip(&whole)
                .any(|(a, b)| (a - b).abs() > 1e-3)
        );
    }

    #[test]
    fn restores_a_saved_state() {
        let file = File::new("state");
        let tokens = sequence(420);
        // A conversation of two parts, and one that takes up from the
        // first's saved state, past where the rings wrap: the same logits.
        let mut whole = file.load(Cpu, 500);
        whole.forward(&tokens[..400], 0);
        let state = whole.save(400).unwrap();
        let want = whole.forward(&tokens[400..], 400);
        let mut resumed = file.load(Cpu, 500);
        assert_eq!(resumed.restore(&state).unwrap(), 400);
        assert_eq!(resumed.forward(&tokens[400..], 400), want);
        assert!(resumed.restore(&state[..100]).is_err());
        // A model with room for less has shorter caches: not its state.
        assert!(file.load(Cpu, 50).restore(&state).is_err());

        // Reset, the model is as it was when loaded.
        let fresh = file.load(Cpu, 500).forward(&tokens[..5], 0);
        assert_ne!(whole.forward(&tokens[..5], 420), fresh);
        whole.reset();
        assert_eq!(whole.forward(&tokens[..5], 0), fresh);
    }

    #[test]
    fn rewinds_to_a_mark() {
        let file = File::new("mark");
        let tokens = sequence(400);
        let mut model = file.load(Cpu, 700);
        model.forward(&tokens[..300], 0);
        let straight = model.forward(&tokens[300..], 300);
        // A detour from a mark, long enough to write over what the rings
        // held before it, then back: the same logits as without it.
        model.reset();
        model.forward(&tokens[..300], 0);
        assert!(model.mark());
        model.forward(&sequence(280), 300);
        model.rewind_to_mark().unwrap();
        assert_eq!(model.forward(&tokens[300..], 300), straight);
        // Without the rewind, the detour stays.
        model.reset();
        model.forward(&tokens[..300], 0);
        model.forward(&sequence(30), 300);
        assert_ne!(model.forward(&tokens[300..], 330), straight);
    }

    #[test]
    fn sequences_keep_their_own_state() {
        let file = File::new("sequences");
        let (a, b) = (sequence(90), sequence(70)[20..].to_vec());
        let mut alone = file.load(Cpu, 100);
        let want_a = alone.forward(&a, 0);
        alone.reset();
        let want_b = alone.forward(&b, 0);
        // Two sequences run in turns, a piece of one and then of the
        // other.
        let mut model = file.load(Cpu, 100);
        let second = model.add_sequence().unwrap();
        model.forward(&a[..40], 0);
        model.select_sequence(second).unwrap();
        model.forward(&b[..10], 0);
        model.select_sequence(0).unwrap();
        let got_a = model.forward(&a[40..], 40);
        model.select_sequence(second).unwrap();
        let got_b = model.forward(&b[10..], 10);
        assert_eq!(got_a, want_a);
        assert_eq!(got_b, want_b);
        assert!(model.select_sequence(7).is_err());
    }

    #[test]
    fn gpu_matches_cpu() {
        let Ok(gpu) = Gpu::new() else {
            eprintln!("skipping: no GPU");
            return;
        };
        if !gpu.takes_nvfp4() {
            eprintln!("skipping: {} does not run NVFP4", gpu.name());
            return;
        }
        // Every token goes through every expert: where the router picks,
        // a sum that comes out the other way in its last bit can tip
        // which expert is picked, and the two devices then run slightly
        // different networks, which is not what is checked here.
        let file = File::routing("gpu", 4);
        let tokens = sequence(450);
        let run = |mut model: Box<dyn LanguageModel>| {
            let prompt = model.forward(&tokens[..400], 0);
            assert!(model.mark());
            model.forward(&sequence(40), 400);
            model.rewind_to_mark().unwrap();
            let mut last = Vec::new();
            for (i, token) in tokens[400..].iter().enumerate() {
                last = model.forward(&[*token], 400 + i);
            }
            (prompt, last)
        };
        let (want_prompt, want_last) = run(Box::new(file.load(Cpu, 500)));
        let (prompt, last) = run(Box::new(file.load(gpu, 500)));
        // CUDA multiplies through the attention matrices in half
        // precision, a token as much as a batch, which over this model's
        // 18 runs of a layer comes to 3e-2; a kernel that is wrong is off
        // by far more.
        close(&prompt, &want_prompt, 5e-2);
        close(&last, &want_last, 5e-2);
    }

    #[test]
    #[ignore]
    fn real_model_completes_a_prompt() {
        let weights = crate::model::Model::InklingMini.weights().unwrap();
        let path = weights.path(&weights.dir().unwrap());
        if !path.exists() {
            eprintln!("skipping: no model at {}", path.display());
            return;
        }
        let gguf = Arc::new(Gguf::open(&path).unwrap());
        let format = crate::model::Model::InklingMini.format(&gguf).unwrap();
        let tokenizer = format.tokenizer();
        let device = Gpu::new().unwrap();
        let start = std::time::Instant::now();
        let mut model = Model::load(gguf, device, 2048, |_, _| {}).unwrap();
        eprintln!("loaded in {:.1}s", start.elapsed().as_secs_f32());
        let mut tokens = tokenizer.encode("The capital of France is").unwrap();
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
        assert!(text.starts_with(" Paris"), "{text:?}");
        // A long prompt, carrying on from the completion, for the speed
        // of a batch.
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
}
