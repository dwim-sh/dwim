use std::path::PathBuf;

/// Model used when none is named on the command line.
pub const DEFAULT: &str = "qwen3-0.6b";

/// Models hack knows how to fetch.
pub const MODELS: &[Model] = &[Model {
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
}];

/// A model hosted on Hugging Face.
///
/// The revision pins a commit, so the weights can't change underneath us and
/// outputs stay reproducible.
pub struct Model {
    pub name: &'static str,
    pub repo: &'static str,
    pub revision: &'static str,
    pub files: &'static [&'static str],
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
