//! What is known of the model wherever it runs, by the engine that runs
//! it and by whatever reaches one that does.

mod chat;
mod effort;
pub mod format;
pub mod gguf;
pub mod hub;
pub mod qwen;
mod sampling;
mod session;
mod size;
mod tokenizer;
mod weights;

use std::{fmt, str::FromStr, sync::Arc};

pub use chat::{Chat, Generation, State};
pub use effort::Effort;
pub use format::{Format, Turn};
pub use sampling::Sampling;
pub use session::{Answer, Chunk, Reply, Session, Stats, Tally};
pub use size::{Api, Device, Size};
pub use tokenizer::Tokenizer;
pub use weights::{File, OnDisk, Weights};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// The models dwim runs: each a network its makers trained and named.
/// This is the one place that tells their families apart: what is a
/// family's own is in its directory, and is reached through the model.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum Model {
    /// Qwen3.8-27B, which runs on this machine: the model dwim runs unless
    /// told to run another.
    #[default]
    Qwen27b,
    /// Qwen3.8-2.4T-A95B, the open weights of Qwen3.8 Max: 2.4 trillion
    /// parameters, of which 95 billion are active for a token. It takes
    /// more memory than a machine has, and runs where a provider has it.
    QwenMax,
}

impl Model {
    /// The model's name.
    pub const fn name(self) -> &'static str {
        match self {
            Model::Qwen27b => "qwen3.8-27b",
            Model::QwenMax => "qwen3.8-2.4t-a95b",
        }
    }

    /// The size the model is called by.
    pub const fn size(self) -> &'static str {
        match self {
            Model::Qwen27b => "medium",
            Model::QwenMax => "large",
        }
    }

    /// The weights the model runs from on this machine, if it runs here.
    pub fn weights(self) -> Option<&'static Weights> {
        match self {
            Model::Qwen27b => Some(&qwen::QWEN_27B),
            Model::QwenMax => None,
        }
    }

    /// The model's chat format, over the vocabulary a GGUF file of it
    /// carries.
    pub fn format(self, gguf: &gguf::Metadata) -> Result<Arc<dyn Format>> {
        match self {
            Model::Qwen27b | Model::QwenMax => qwen::format(gguf),
        }
    }
}

impl fmt::Display for Model {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for Model {
    type Err = String;

    /// A model by its size, or by its name.
    fn from_str(value: &str) -> std::result::Result<Self, String> {
        [Model::Qwen27b, Model::QwenMax]
            .into_iter()
            .find(|model| value == model.size() || value == model.name())
            .ok_or_else(|| format!("unknown model '{value}' (expected medium or large)"))
    }
}
