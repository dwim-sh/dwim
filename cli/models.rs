use std::{
    error::Error,
    path::{Path, PathBuf},
    sync::Arc,
    time::SystemTime,
};

use dwim_gpu::Device;
use dwim_models::{Gguf, LanguageModel, Sampler, bonsai};

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
