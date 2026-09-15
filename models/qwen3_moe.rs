//! Qwen3-MoE: the Qwen3 transformer with its feed-forward network replaced
//! by a mixture of experts. The attention runs on the device; the routed
//! experts, which are most of the weights but few of the ones any token
//! uses, run four-bit on the CPU straight from the pack's memory mapping.

use std::{fs, path::Path, sync::Arc};

use hack_gpu::bf16;
use serde::Deserialize;

use crate::{Device, LanguageModel, Result, experts::Experts, pack::Pack, rope_table};

/// Most tokens a forward pass runs through the model at once.
pub const BATCH: usize = 64;

/// The architecture, read from a Hugging Face `config.json`.
#[derive(Debug, Deserialize)]
pub struct Config {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub vocab_size: usize,
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        Ok(serde_json::from_str(&fs::read_to_string(path)?)?)
    }
}

/// The model, with its dense weights on a device, its experts in a pack,
/// and the state of a sequence.
pub struct Model<D: Device> {
    pub config: Config,
    pub device: D,
    pack: Arc<Pack>,
    layers: Vec<Layer<D>>,
    norm: D::Weight,
    lm_head: D::Weight,
    state: State<D>,
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
    router: D::Weight,
    experts: Experts,
}

impl<D: Device> Model<D> {
    /// Loads a model from a pack converted from Hugging Face weights, with
    /// `config.json` beside it in `dir`, uploading its dense weights to
    /// `device`, with state for sequences of up to `max_len` tokens, and
    /// reporting how many of its tensors are loaded, out of how many.
    pub fn load(dir: &Path, pack: Arc<Pack>, device: D, max_len: usize, mut on_progress: impl FnMut(usize, usize)) -> Result<Self> {
        let config = Config::load(&dir.join("config.json"))?;
        let c = &config;
        let d = &device;
        let total = c.num_hidden_layers * 9 + 2;
        let mut done = 0;
        let mut load = |name: &str| -> Result<D::Weight> {
            let weight = d.upload(pack.bf16(name)?);
            done += 1;
            on_progress(done, total);
            Ok(weight)
        };

        let mut layers = Vec::with_capacity(c.num_hidden_layers);
        for i in 0..c.num_hidden_layers {
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
                router: load(&format!("{p}.mlp.gate.weight"))?,
                experts: Experts::new(pack.clone(), &format!("{p}.mlp.experts"), c.num_experts_per_tok)?,
            });
        }
        let norm = load("model.norm.weight")?;
        let lm_head = load("lm_head.weight")?;
        let state = State::new(&config, d, max_len);
        Ok(Self {
            config,
            device,
            pack,
            layers,
            norm,
            lm_head,
            state,
        })
    }

    fn forward_batch(&mut self, tokens: &[u32], pos: usize) -> Vec<f32> {
        let c = &self.config;
        let d = &self.device;
        let s = &mut self.state;
        let n = tokens.len();
        let eps = c.rms_norm_eps;
        let kv_dim = c.num_key_value_heads * c.head_dim;
        s.batch(d, n);

        // Embeddings are looked up on the CPU: the table is most of a
        // gigabyte, and a token needs one row of it.
        let table = self.pack.bytes("model.embed_tokens.weight").expect("embeddings");
        let mut x = Vec::with_capacity(n * c.hidden_size);
        for &token in tokens {
            let row = &table[token as usize * c.hidden_size * 2..][..c.hidden_size * 2];
            x.extend(row.chunks_exact(2).map(|b| bf16(u16::from_le_bytes([b[0], b[1]]))));
        }
        d.write(&mut s.x, &x);

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
            d.store(&mut s.k_cache[i], pos * kv_dim, &s.k);
            d.store(&mut s.v_cache[i], pos * kv_dim, &s.v);
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

            // The mixture of experts, likewise added back: the router on the
            // device, the experts it picks on the CPU.
            d.copy(&mut s.xb, 0, &s.x, 0, n * c.hidden_size);
            d.rmsnorm(&mut s.xb, &layer.mlp_norm, eps);
            d.matmul(&mut s.router, &layer.router, &s.xb);
            let xb = d.read(&s.xb);
            let logits = d.read(&s.router);
            let mut moe = vec![0.0; n * c.hidden_size];
            layer.experts.forward(&xb, &logits, &mut moe);
            d.write(&mut s.xb, &moe);
            d.add(&mut s.x, &s.xb);
        }

        // Only the last token's logits are wanted.
        d.resize(&mut s.xb, c.hidden_size);
        d.copy(&mut s.xb, 0, &s.x, (n - 1) * c.hidden_size, c.hidden_size);
        d.rmsnorm(&mut s.xb, &self.norm, eps);
        d.matmul(&mut s.logits, &self.lm_head, &s.xb);
        d.read(&s.logits)
    }
}

impl<D: Device> LanguageModel for Model<D> {
    /// Adds the tokens' keys and values to the cache. Tokens run through the
    /// model in batches of up to [`BATCH`].
    fn forward(&mut self, tokens: &[u32], pos: usize) -> Vec<f32> {
        assert!(!tokens.is_empty(), "no tokens to run");
        assert!(pos + tokens.len() <= self.state.max_len, "tokens past the end of the cache");
        let mut logits = Vec::new();
        for (i, batch) in tokens.chunks(BATCH).enumerate() {
            logits = self.forward_batch(batch, pos + i * BATCH);
        }
        logits
    }

    fn max_len(&self) -> usize {
        self.state.max_len
    }
}

/// Buffers the forward pass computes in, with room for a batch of
/// [`BATCH`] tokens, the key/value cache holding every position seen so far
/// in half precision, and the rotary position embedding table for every
/// position there is room for.
struct State<D: Device> {
    /// Activations per token of each per-token buffer, in field order.
    widths: [usize; 7],
    x: D::Buffer,
    xb: D::Buffer,
    q: D::Buffer,
    k: D::Buffer,
    v: D::Buffer,
    att: D::Buffer,
    router: D::Buffer,
    logits: D::Buffer,
    k_cache: Vec<D::Cache>,
    v_cache: Vec<D::Cache>,
    rope: D::Buffer,
    max_len: usize,
}

impl<D: Device> State<D> {
    /// Allocates state for sequences of up to `max_len` tokens.
    fn new(c: &Config, d: &D, max_len: usize) -> Self {
        let q_dim = c.num_attention_heads * c.head_dim;
        let kv_dim = c.num_key_value_heads * c.head_dim;
        let widths = [c.hidden_size, c.hidden_size, q_dim, kv_dim, kv_dim, q_dim, c.num_experts];
        let [x, xb, q, k, v, att, router] = widths.map(|width| d.alloc(BATCH * width));
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
            router,
            logits: d.alloc(c.vocab_size),
            k_cache: (0..c.num_hidden_layers).map(|_| d.alloc_cache(max_len * kv_dim)).collect(),
            v_cache: (0..c.num_hidden_layers).map(|_| d.alloc_cache(max_len * kv_dim)).collect(),
            rope,
            max_len,
        }
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
            &mut self.router,
        ];
        for (buffer, width) in buffers.into_iter().zip(self.widths) {
            device.resize(buffer, n * width);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        pack::{Dtype, Writer},
        q4,
    };
    use hack_gpu::{Cpu, Vulkan};

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        }
    }

    /// Writes a tiny random model in the pack format, with its config.
    fn synthetic(dir: &Path) -> Config {
        let config = r#"{
            "hidden_size": 64, "num_hidden_layers": 3, "num_attention_heads": 2, "num_key_value_heads": 1,
            "head_dim": 32, "num_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 32,
            "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "vocab_size": 128
        }"#;
        fs::write(dir.join("config.json"), config).unwrap();
        let c: Config = serde_json::from_str(config).unwrap();
        let h = c.hidden_size;
        let mut layout: Vec<(String, Dtype, Vec<usize>)> = vec![
            ("model.embed_tokens.weight".into(), Dtype::Bf16, vec![c.vocab_size, h]),
            ("lm_head.weight".into(), Dtype::Bf16, vec![c.vocab_size, h]),
            ("model.norm.weight".into(), Dtype::Bf16, vec![h]),
        ];
        for i in 0..c.num_hidden_layers {
            let p = format!("model.layers.{i}");
            let mut t = |name: &str, dtype: Dtype, shape: Vec<usize>| layout.push((format!("{p}.{name}"), dtype, shape));
            t("input_layernorm.weight", Dtype::Bf16, vec![h]);
            t("post_attention_layernorm.weight", Dtype::Bf16, vec![h]);
            t("self_attn.q_proj.weight", Dtype::Bf16, vec![c.num_attention_heads * c.head_dim, h]);
            t("self_attn.k_proj.weight", Dtype::Bf16, vec![c.num_key_value_heads * c.head_dim, h]);
            t("self_attn.v_proj.weight", Dtype::Bf16, vec![c.num_key_value_heads * c.head_dim, h]);
            t("self_attn.o_proj.weight", Dtype::Bf16, vec![h, c.num_attention_heads * c.head_dim]);
            t("self_attn.q_norm.weight", Dtype::Bf16, vec![c.head_dim]);
            t("self_attn.k_norm.weight", Dtype::Bf16, vec![c.head_dim]);
            t("mlp.gate.weight", Dtype::Bf16, vec![c.num_experts, h]);
            t("mlp.experts.gate_proj", Dtype::Q4, vec![c.num_experts, c.moe_intermediate_size, h]);
            t("mlp.experts.up_proj", Dtype::Q4, vec![c.num_experts, c.moe_intermediate_size, h]);
            t("mlp.experts.down_proj", Dtype::Q4, vec![c.num_experts, h, c.moe_intermediate_size]);
        }
        let writer = Writer::create(&dir.join("model.hack"), &layout).unwrap();
        let mut rng = Rng(42);
        for (name, dtype, shape) in &layout {
            let cols = *shape.last().unwrap();
            let rows: usize = shape[..shape.len() - 1].iter().product();
            // Weights small enough that activations stay sane through the
            // layers; norms near one.
            let norm = name.ends_with("norm.weight");
            let scale = if norm { 1.0 } else { 0.2 };
            let offset = if norm { 1.0 } else { 0.0 };
            match dtype {
                Dtype::Bf16 => {
                    let bytes: Vec<u8> = (0..rows * cols)
                        .map(|_| ((rng.next() * scale + offset).to_bits() >> 16) as u16)
                        .flat_map(|v| v.to_le_bytes())
                        .collect();
                    writer.write(name, 0, &bytes).unwrap();
                }
                Dtype::Q4 => {
                    let row_bytes = q4::row_bytes(cols);
                    let mut bytes = vec![0; rows * row_bytes];
                    for r in 0..rows {
                        let row: Vec<f32> = (0..cols).map(|_| rng.next() * scale).collect();
                        q4::quantize_row(&row, &mut bytes[r * row_bytes..][..row_bytes]);
                    }
                    writer.write(name, 0, &bytes).unwrap();
                }
            }
        }
        c
    }

    fn run<D: Device>(dir: &Path, device: D) -> (Vec<f32>, Vec<f32>) {
        let pack = Arc::new(Pack::open(&dir.join("model.hack")).unwrap());
        let mut model = Model::load(dir, pack, device, 64, |_, _| {}).unwrap();
        // A batch of several tokens, then one more, carrying the state.
        let first = model.forward(&[3, 17, 42, 7, 99], 0);
        let second = model.forward(&[11], 5);
        (first, second)
    }

    #[test]
    fn gpu_agrees_with_cpu() {
        let dir = std::env::temp_dir().join(format!("hack-qwen3-moe-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        synthetic(&dir);
        let (cpu_first, cpu_second) = run(&dir, Cpu);
        assert!(cpu_first.iter().all(|v| v.is_finite()));
        assert!(cpu_first.iter().any(|&v| v != 0.0));
        if let Ok(gpu) = Vulkan::new() {
            let (gpu_first, gpu_second) = run(&dir, gpu);
            for (a, b) in cpu_first.iter().zip(&gpu_first).chain(cpu_second.iter().zip(&gpu_second)) {
                assert!((a - b).abs() <= 1e-3 * (1.0 + a.abs()), "{a} vs {b}");
            }
        } else {
            eprintln!("skipping the GPU comparison: no Vulkan device");
        }
        fs::remove_dir_all(&dir).unwrap();
    }
}
