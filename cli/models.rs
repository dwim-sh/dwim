use std::{
    error::Error,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::SystemTime,
};

use dwim_gpu::Device;
use dwim_models::{Chat, Gguf, LanguageModel, Sampler};

/// Saved states kept, newest first: a state is a few hundred megabytes,
/// and the system prompt changes with the project's `AGENTS.md`.
const STATES_KEPT: usize = 4;

/// Model used when none is named on the command line.
pub const DEFAULT: &str = "bonsai-2-27b";

/// Tokens of conversation a model has room for unless the command line says
/// otherwise.
pub const DEFAULT_CONTEXT: usize = 16384;

/// Models `dwim` knows how to fetch. The first is the default.
pub const MODELS: &[Model] = &[
    Model {
        name: "bonsai-2-27b",
        about: "Qwen3.8-27B made ternary by PrismML, 6 GB",
        repo: "prism-ml/Ternary-Bonsai-2-27B-gguf",
        revision: "6ed5e12bf84b7a63069882c91dd9e9218647d17b",
        files: &["LICENSE", "NOTICE.txt", "Ternary-Bonsai-2-27B-PTQ1_0.gguf"],
        weights: "Ternary-Bonsai-2-27B-PTQ1_0.gguf",
        memory: 16 << 30,
    },
    // Qwen ships no GGUF of its own; bartowski's is a conversion of
    // `Qwen/Qwen3.8-27B` quantized with an importance matrix, whose Q4_0 file
    // holds most matrices in 4-bit blocks of 32 with an f16 scale, the
    // first layers' down projections as Q4_1, the linear attention's output
    // projections as Q8_0, and the output and a few query projections as
    // Q6_K; the norms and the small linear attention tensors are f32.
    Model {
        name: "qwen3.8-27b",
        about: "Qwen3.8-27B at four bits, 16 GB",
        repo: "bartowski/Qwen3.8-27B-GGUF",
        revision: "87410e7b9a5c21e6ca70c0c9cc5237de909fe621",
        files: &["Qwen3.8-27B-Q4_0.gguf"],
        weights: "Qwen3.8-27B-Q4_0.gguf",
        memory: 32 << 30,
    },
];

/// A model hosted on Hugging Face.
///
/// The revision pins a commit, so the weights can't change underneath us and
/// outputs stay reproducible.
pub struct Model {
    pub name: &'static str,
    /// What the model is, in a few words, for the `/model` listing.
    pub about: &'static str,
    pub repo: &'static str,
    pub revision: &'static str,
    /// The files fetched.
    pub files: &'static [&'static str],
    /// The one among them holding the model: a GGUF file, which carries
    /// the tokenizer and the sampling settings too.
    pub weights: &'static str,
    /// Least memory, in bytes, a machine whose CPU and GPU share it needs
    /// to run the model: the weights, with room left for the caches, the
    /// activations, and everything else running.
    pub memory: u64,
}

impl Model {
    /// Looks up a model by name.
    pub fn find(name: &str) -> Option<&'static Model> {
        MODELS.iter().find(|model| model.name == name)
    }

    /// URL to download one of the model's files from.
    pub fn url(&self, file: &str) -> String {
        format!(
            "https://huggingface.co/{}/resolve/{}/{}",
            self.repo, self.revision, file
        )
    }

    /// Local directory the model's files are stored in, under the platform's
    /// cache directory (`~/.cache` on Linux, `~/Library/Caches` on macOS):
    /// the files can always be fetched again.
    pub fn dir(&self) -> Option<PathBuf> {
        dirs::cache_dir().map(|dir| dir.join("dwim").join("models").join(self.name))
    }

    /// Opens the model's weights in `dir`.
    pub fn open(&self, dir: &Path) -> Result<Arc<Gguf>, Box<dyn Error>> {
        Ok(Arc::new(Gguf::open(&dir.join(self.weights))?))
    }
}

/// Loads the model whose weights are `gguf` onto `device`, with room for
/// `context` tokens of conversation, reporting how many of its tensors are
/// loaded, out of how many, as it goes.
pub fn load<D: Device + 'static>(
    gguf: Arc<Gguf>,
    device: D,
    context: usize,
    on_progress: impl FnMut(usize, usize),
) -> Result<Box<dyn LanguageModel>, Box<dyn Error>> {
    dwim_models::load(gguf, device, context, on_progress)
}

/// The sampler for the model, with the settings its file recommends where
/// it has them.
/// Opens the conversation with the system prompt: from the state saved by
/// an earlier run of the same model with the same prompt, if there is one
/// and it fits, and otherwise by reading the prompt, reporting progress as
/// `Chat::system` does, and saving the state for the next run. Returns
/// whether the state was restored.
pub fn start<M: LanguageModel>(
    chat: &mut Chat<M>,
    model: &Model,
    prompt: &str,
    on_progress: impl FnMut(usize, usize),
) -> Result<bool, Box<dyn Error>> {
    let path = state_path(model, prompt);
    if let Some(path) = &path
        && let Ok(state) = fs::read(path)
        && chat.restore(&state).is_ok()
    {
        return Ok(true);
    }
    chat.system(prompt, on_progress)?;
    if let Some(path) = path
        && let Some(state) = chat.save()
        && let Some(dir) = path.parent()
        && fs::create_dir_all(dir).is_ok()
        && fs::write(&path, state).is_ok()
    {
        prune(dir);
    }
    Ok(false)
}

/// Where the state after `prompt` on `model` is kept: named by a hash of
/// the two, in the cache directory.
fn state_path(model: &Model, prompt: &str) -> Option<PathBuf> {
    // FNV-1a, so that the name is the same from one build to the next.
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in model.name.bytes().chain([0]).chain(prompt.bytes()) {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    Some(
        dirs::cache_dir()?
            .join("dwim")
            .join("states")
            .join(format!("{hash:016x}.bin")),
    )
}

/// Removes all but the newest saved states.
fn prune(dir: &Path) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    let mut states: Vec<_> = entries
        .flatten()
        .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "bin"))
        .filter_map(|entry| Some((entry.metadata().ok()?.modified().ok()?, entry.path())))
        .collect();
    states.sort();
    for (_, path) in states.iter().rev().skip(STATES_KEPT) {
        let _ = fs::remove_file(path);
    }
}

pub fn sampler(gguf: &Gguf) -> Sampler {
    let number = |key: &str, default: f32| {
        gguf.f32(&format!("general.sampling.{key}"))
            .unwrap_or(default)
    };
    Sampler::new(
        number("temp", 1.0),
        number("top_k", 20.0) as usize,
        number("top_p", 0.95),
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
