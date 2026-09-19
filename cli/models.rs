use std::{
    error::Error,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::SystemTime,
};

use dwim_gpu::Device;
use dwim_models::{Chat, Gguf, LanguageModel, Sampler, bonsai};

/// Saved states kept, newest first: a state is a few hundred megabytes,
/// and the system prompt changes with the directory and the day.
const STATES_KEPT: usize = 4;

/// Model used when none is named on the command line.
pub const DEFAULT: &str = "bonsai-2-27b";

/// Tokens of conversation a model has room for unless the command line says
/// otherwise.
pub const DEFAULT_CONTEXT: usize = 32768;

/// Models `dwim` knows how to fetch.
pub const MODELS: &[Model] = &[Model {
    name: "bonsai-2-27b",
    repo: "prism-ml/Ternary-Bonsai-2-27B-gguf",
    revision: "6ed5e12bf84b7a63069882c91dd9e9218647d17b",
    files: &["LICENSE", "NOTICE.txt", "Ternary-Bonsai-2-27B-PTQ1_0.gguf"],
    weights: "Ternary-Bonsai-2-27B-PTQ1_0.gguf",
}];

/// A model hosted on Hugging Face.
///
/// The revision pins a commit, so the weights can't change underneath us and
/// outputs stay reproducible.
pub struct Model {
    pub name: &'static str,
    pub repo: &'static str,
    pub revision: &'static str,
    /// The files fetched.
    pub files: &'static [&'static str],
    /// The one among them holding the model: a GGUF file, which carries
    /// the tokenizer and the sampling settings too.
    pub weights: &'static str,
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
        dirs::cache_dir().map(|dir| dir.join("dwim").join("models").join(self.name))
    }

    /// Opens the model's weights in `dir`.
    pub fn open(&self, dir: &Path) -> Result<Arc<Gguf>, Box<dyn Error>> {
        Ok(Arc::new(Gguf::open(&dir.join(self.weights))?))
    }
}

/// Loads the model onto `device`, with room for `context` tokens of
/// conversation, reporting how many of its tensors are loaded, out of how
/// many, as it goes.
pub fn load<D: Device + 'static>(
    gguf: Arc<Gguf>,
    device: D,
    context: usize,
    on_progress: impl FnMut(usize, usize),
) -> Result<Box<dyn LanguageModel>, Box<dyn Error>> {
    // Past the positions the model was trained on, its attention degrades.
    let trained = bonsai::Config::load(&gguf)?.context_length;
    if context == 0 || context > trained {
        return Err(format!("a context of {context} tokens is outside the 1 to {trained} the model was trained for").into());
    }
    Ok(Box::new(bonsai::Model::load(gguf, device, context, on_progress)?))
}

/// The sampler for the model, with the settings its file recommends where
/// it has them.
/// Opens the conversation with the system prompt: from the state saved by
/// an earlier run of the same model with the same prompt, if there is one
/// and it fits, and otherwise by reading the prompt, reporting progress as
/// `Chat::system` does, and saving the state for the next run. Returns
/// whether the state was restored.
pub fn start<M: LanguageModel>(chat: &mut Chat<M>, model: &Model, prompt: &str, on_progress: impl FnMut(usize, usize)) -> Result<bool, Box<dyn Error>> {
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
    Some(dirs::cache_dir()?.join("dwim").join("states").join(format!("{hash:016x}.bin")))
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
    let number = |key: &str, default: f32| gguf.f32(&format!("general.sampling.{key}")).unwrap_or(default);
    Sampler::new(number("temp", 1.0), number("top_k", 20.0) as usize, number("top_p", 0.95), seed())
}

/// Seed for sampling, different on every run.
fn seed() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_nanos() as u64)
        .unwrap_or(0)
}
