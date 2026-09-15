//! Qwen3: a decoder-only transformer with grouped-query attention, QK-norm,
//! rotary position embeddings, and a SwiGLU feed-forward network, written as
//! a sequence of operations on a [`Device`].

use std::{fs, path::Path};

use serde::Deserialize;

use crate::{Device, Result, Weights};

/// Most tokens a forward pass runs through the model at once. Running a
/// batch of tokens together reads each weight once for the whole batch,
/// rather than once per token; the activations are allocated for this many.
pub const BATCH: usize = 64;

/// The architecture, read from a Hugging Face `config.json`.
#[derive(Debug, Deserialize)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub vocab_size: usize,
    /// Whether the output projection shares the token embedding table.
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        Ok(serde_json::from_str(&fs::read_to_string(path)?)?)
    }
}

/// The model, with its weights on a device.
pub struct Model<D: Device> {
    pub config: Config,
    pub device: D,
    embed: D::Weight,
    layers: Vec<Layer<D>>,
    norm: D::Weight,
    /// Output projection, or `None` when it shares the embedding table.
    lm_head: Option<D::Weight>,
}

struct Layer<D: Device> {
    attn_norm: D::Weight,
    q: D::Weight,
    k: D::Weight,
    v: D::Weight,
    o: D::Weight,
    q_norm: D::Weight,
    k_norm: D::Weight,
    mlp_norm: D::Weight,
    gate: D::Weight,
    up: D::Weight,
    down: D::Weight,
}

impl<D: Device> Model<D> {
    /// Loads a model from a directory holding a Hugging Face `config.json` and
    /// `model.safetensors`, uploading its weights to `device`, and reporting
    /// how many of its tensors are loaded, out of how many, as it goes.
    pub fn load(dir: &Path, device: D, mut on_progress: impl FnMut(usize, usize)) -> Result<Self> {
        let config = Config::load(&dir.join("config.json"))?;
        let mut weights = Weights::open(&dir.join("model.safetensors"))?;
        let w = &mut weights;
        let d = &device;
        let total = config.num_hidden_layers * 11 + if config.tie_word_embeddings { 2 } else { 3 };
        let mut done = 0;
        let mut load = |name: &str| -> Result<D::Weight> {
            let weight = d.upload(w.read(name)?);
            done += 1;
            on_progress(done, total);
            Ok(weight)
        };

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let p = format!("model.layers.{i}");
            layers.push(Layer {
                attn_norm: load(&format!("{p}.input_layernorm.weight"))?,
                q: load(&format!("{p}.self_attn.q_proj.weight"))?,
                k: load(&format!("{p}.self_attn.k_proj.weight"))?,
                v: load(&format!("{p}.self_attn.v_proj.weight"))?,
                o: load(&format!("{p}.self_attn.o_proj.weight"))?,
                q_norm: load(&format!("{p}.self_attn.q_norm.weight"))?,
                k_norm: load(&format!("{p}.self_attn.k_norm.weight"))?,
                mlp_norm: load(&format!("{p}.post_attention_layernorm.weight"))?,
                gate: load(&format!("{p}.mlp.gate_proj.weight"))?,
                up: load(&format!("{p}.mlp.up_proj.weight"))?,
                down: load(&format!("{p}.mlp.down_proj.weight"))?,
            });
        }
        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            Some(load("lm_head.weight")?)
        };
        let embed = load("model.embed_tokens.weight")?;
        let norm = load("model.norm.weight")?;
        Ok(Self {
            embed,
            layers,
            norm,
            lm_head,
            config,
            device,
        })
    }

    /// Runs `tokens`, the first at position `pos`, through the model, adding
    /// their keys and values to the cache in `state`, and returns the logits
    /// for the token that follows the last of them. Tokens run through the
    /// model in batches of up to [`BATCH`].
    pub fn forward(&self, state: &mut State<D>, tokens: &[u32], pos: usize) -> Vec<f32> {
        assert!(!tokens.is_empty(), "no tokens to run");
        assert!(pos + tokens.len() <= state.max_len, "tokens past the end of the cache");
        let mut logits = Vec::new();
        for (i, batch) in tokens.chunks(BATCH).enumerate() {
            logits = self.forward_batch(state, batch, pos + i * BATCH);
        }
        logits
    }

    fn forward_batch(&self, state: &mut State<D>, tokens: &[u32], pos: usize) -> Vec<f32> {
        let c = &self.config;
        let d = &self.device;
        let s = state;
        let n = tokens.len();
        let eps = c.rms_norm_eps;
        let kv_dim = c.num_key_value_heads * c.head_dim;
        s.batch(d, n);

        d.embed(&mut s.x, &self.embed, tokens);
        for (i, layer) in self.layers.iter().enumerate() {
            // Attention, with the result added back into the residual stream.
            d.copy(&mut s.xb, 0, &s.x, 0, n * c.hidden_size);
            d.rmsnorm(&mut s.xb, &layer.attn_norm, eps);
            d.matmul(&mut s.q, &layer.q, &s.xb);
            d.matmul(&mut s.k, &layer.k, &s.xb);
            d.matmul(&mut s.v, &layer.v, &s.xb);
            d.rmsnorm(&mut s.q, &layer.q_norm, eps);
            d.rmsnorm(&mut s.k, &layer.k_norm, eps);
            d.rope(&mut s.q, &s.rope, pos, c.num_attention_heads, c.head_dim);
            d.rope(&mut s.k, &s.rope, pos, c.num_key_value_heads, c.head_dim);
            d.copy(&mut s.k_cache[i], pos * kv_dim, &s.k, 0, n * kv_dim);
            d.copy(&mut s.v_cache[i], pos * kv_dim, &s.v, 0, n * kv_dim);
            d.attention(
                &mut s.att,
                &s.q,
                &s.k_cache[i],
                &s.v_cache[i],
                pos,
                c.num_attention_heads,
                c.head_dim,
                c.num_key_value_heads,
            );
            d.matmul(&mut s.xb, &layer.o, &s.att);
            d.add(&mut s.x, &s.xb);

            // Feed-forward network, likewise added back.
            d.copy(&mut s.xb, 0, &s.x, 0, n * c.hidden_size);
            d.rmsnorm(&mut s.xb, &layer.mlp_norm, eps);
            d.matmul(&mut s.gate, &layer.gate, &s.xb);
            d.matmul(&mut s.up, &layer.up, &s.xb);
            d.silu_mul(&mut s.gate, &s.up);
            d.matmul(&mut s.xb, &layer.down, &s.gate);
            d.add(&mut s.x, &s.xb);
        }
        // Only the last token's logits are wanted.
        d.resize(&mut s.xb, c.hidden_size);
        d.copy(&mut s.xb, 0, &s.x, (n - 1) * c.hidden_size, c.hidden_size);
        d.rmsnorm(&mut s.xb, &self.norm, eps);
        d.matmul(&mut s.logits, self.lm_head.as_ref().unwrap_or(&self.embed), &s.xb);
        d.read(&s.logits)
    }
}

/// Buffers the forward pass computes in, with room for a batch of
/// [`BATCH`] tokens, the key/value cache holding every position seen so far,
/// and the rotary position embedding table for every position there is room
/// for.
pub struct State<D: Device> {
    /// Activations per token of each buffer, in the order of the fields.
    widths: [usize; 8],
    x: D::Buffer,
    xb: D::Buffer,
    q: D::Buffer,
    k: D::Buffer,
    v: D::Buffer,
    att: D::Buffer,
    gate: D::Buffer,
    up: D::Buffer,
    logits: D::Buffer,
    k_cache: Vec<D::Buffer>,
    v_cache: Vec<D::Buffer>,
    rope: D::Buffer,
    max_len: usize,
}

impl<D: Device> State<D> {
    /// Allocates state for sequences of up to `max_len` tokens.
    pub fn new(model: &Model<D>, max_len: usize) -> Self {
        let c = &model.config;
        let d = &model.device;
        let q_dim = c.num_attention_heads * c.head_dim;
        let kv_dim = c.num_key_value_heads * c.head_dim;
        let widths = [
            c.hidden_size,
            c.hidden_size,
            q_dim,
            kv_dim,
            kv_dim,
            q_dim,
            c.intermediate_size,
            c.intermediate_size,
        ];
        let [x, xb, q, k, v, att, gate, up] = widths.map(|width| d.alloc(BATCH * width));
        let table = rope_table(max_len, c.head_dim, c.rope_theta);
        let mut rope = d.alloc(table.len());
        d.write(&mut rope, &table);
        Self {
            widths,
            x,
            xb,
            q,
            k,
            v,
            att,
            gate,
            up,
            logits: d.alloc(c.vocab_size),
            k_cache: (0..c.num_hidden_layers).map(|_| d.alloc(max_len * kv_dim)).collect(),
            v_cache: (0..c.num_hidden_layers).map(|_| d.alloc(max_len * kv_dim)).collect(),
            rope,
            max_len,
        }
    }

    pub fn max_len(&self) -> usize {
        self.max_len
    }

    /// Sizes the buffers for a batch of `n` tokens.
    fn batch(&mut self, device: &D, n: usize) {
        assert!(n <= BATCH, "a batch of {n} tokens is more than {BATCH}");
        let buffers = [
            &mut self.x,
            &mut self.xb,
            &mut self.q,
            &mut self.k,
            &mut self.v,
            &mut self.att,
            &mut self.gate,
            &mut self.up,
        ];
        for (buffer, width) in buffers.into_iter().zip(self.widths) {
            device.resize(buffer, n * width);
        }
    }
}

/// The cosines and sines of the angles rotary position embeddings rotate
/// by, laid out as [`Device::rope`] expects, for positions up to `max_len`.
/// Each of a head's pairs of elements rotates at its own frequency.
pub fn rope_table(max_len: usize, head_dim: usize, theta: f32) -> Vec<f32> {
    let half = head_dim / 2;
    let mut table = Vec::with_capacity(max_len * head_dim);
    for pos in 0..max_len {
        for i in 0..half {
            let freq = 1.0 / theta.powf((2 * i) as f32 / head_dim as f32);
            let (sin, cos) = (pos as f32 * freq).sin_cos();
            table.extend([cos, sin]);
        }
    }
    table
}
