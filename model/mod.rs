//! What is known of the model wherever it runs, by the engine that runs
//! it and by whatever reaches one that does.

mod chat;
pub mod format;
pub mod gguf;
pub mod hub;
mod sampling;
mod session;
mod size;
mod tokenizer;

use std::{fmt, str::FromStr};

pub use chat::{Chat, Generation, State};
pub use format::{Format, Turn};
pub use sampling::Sampling;
pub use session::{Answer, Chunk, Reply, Session, Stats, Tally};
pub use size::{Api, Device, Size};
pub use tokenizer::Tokenizer;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// The models dwim runs: each a network its makers trained and named. They
/// are of one family, with one chat format and one way of calling tools,
/// so a conversation with one reads the same to another.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Model {
    /// Qwen3.8-27B, which runs on this machine.
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
