use std::{
    error::Error,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::SystemTime,
};

use hack_gpu::Device;
use hack_harness::ToolFormat;
use hack_models::{LanguageModel, Sampler, pack::Pack, qwen3, qwen3_moe};

/// Model used when none is named on the command line.
pub const DEFAULT: &str = "qwen3-0.6b";

/// Tokens of conversation a model has room for unless the command line says
/// otherwise.
pub const DEFAULT_CONTEXT: usize = 32768;

/// Models hack knows how to fetch.
pub const MODELS: &[Model] = &[
    Model {
        name: "qwen3-0.6b",
        repo: "Qwen/Qwen3-0.6B",
        revision: "c1899de289a04d12100db370d81485cdf75e47ca",
        files: &[
            "LICENSE",
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "model.safetensors",
        ],
        sharded: false,
        tools: ToolFormat::Json,
    },
    // The routed experts run on the CPU through x86 code, which has not been
    // made to work on Apple platforms, so the model is not offered there.
    #[cfg(not(target_vendor = "apple"))]
    Model {
        name: "qwen3-coder-30b-a3b",
        repo: "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        revision: "b2cff646eb4bb1d68355c01b18ae02e7cf42d120",
        files: &["LICENSE", "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json"],
        sharded: true,
        tools: ToolFormat::Xml,
    },
];

/// A model hosted on Hugging Face.
///
/// The revision pins a commit, so the weights can't change underneath us and
/// outputs stay reproducible.
pub struct Model {
    pub name: &'static str,
    pub repo: &'static str,
    pub revision: &'static str,
    /// The small files fetched as they are.
    pub files: &'static [&'static str],
    /// Whether the weights come as shards, converted into a pack.
    pub sharded: bool,
    /// How the model writes tool calls.
    pub tools: ToolFormat,
}

impl Model {
    /// Looks up a model by name.
    pub fn find(name: &str) -> Option<&'static Model> {
        MODELS.iter().find(|model| model.name == name)
    }

    /// URL to download one of the model's files from.
    pub fn url(&self, file: &str) -> String {
        format!("https://huggingface.co/{}/resolve/{}/{}", self.repo, self.revision, file)
    }

    /// Local directory the model's files are stored in, under the platform's
    /// cache directory (`~/.cache` on Linux, `~/Library/Caches` on macOS):
    /// the files can always be fetched again.
    pub fn dir(&self) -> Option<PathBuf> {
        dirs::cache_dir().map(|dir| dir.join("hack").join("models").join(self.name))
    }
}

/// Loads the model in `dir`, of whichever architecture its `config.json`
/// names, onto `device`, with room for `context` tokens of conversation,
/// reporting how many of its tensors are loaded, out of how many, as it
/// goes.
pub fn load<D: Device + 'static>(
    dir: &Path,
    device: D,
    context: usize,
    mut on_progress: impl FnMut(usize, usize),
) -> Result<Box<dyn LanguageModel>, Box<dyn Error>> {
    let config: serde_json::Value = serde_json::from_str(&fs::read_to_string(dir.join("config.json"))?)?;
    // Past the positions the model was trained on, its attention degrades.
    let trained = config["max_position_embeddings"].as_u64().unwrap_or(u64::MAX) as usize;
    if context == 0 || context > trained {
        return Err(format!("a context of {context} tokens is outside the 1 to {trained} the model was trained for").into());
    }
    Ok(match config["model_type"].as_str().unwrap_or("") {
        "qwen3" => Box::new(qwen3::Model::load(dir, device, context, on_progress)?),
        "qwen3_moe" => {
            let pack = Arc::new(Pack::open(&dir.join("model.hack"))?);
            Box::new(qwen3_moe::Model::load(dir, pack, device, context, &mut on_progress)?)
        }
        other => return Err(format!("unsupported model type '{other}'").into()),
    })
}

/// The sampler for the model in `dir`, with its recommended settings from
/// its `generation_config.json` where it has them.
pub fn sampler(dir: &Path) -> Sampler {
    let config: serde_json::Value = fs::read_to_string(dir.join("generation_config.json"))
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or_default();
    let number = |key: &str, default: f64| config[key].as_f64().unwrap_or(default);
    Sampler::new(
        number("temperature", 0.7) as f32,
        number("top_k", 20.0) as usize,
        number("top_p", 0.8) as f32,
        seed(),
    )
}

/// Seed for sampling, different on every run.
fn seed() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_nanos() as u64)
        .unwrap_or(0)
}
