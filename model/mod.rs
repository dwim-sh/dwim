//! What is known of the model wherever it runs, by the engine that runs
//! it and by whatever reaches one that does.

mod chat;
mod effort;
pub mod format;
pub mod gguf;
pub mod hub;
pub mod inkling;
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
    /// Inkling 2 Mini, Thinking Machines' looped model of 26 billion
    /// parameters, of which 3.4 billion are active for a token: it runs on
    /// this machine, where the GPU takes its weights, and at OpenRouter.
    InklingMini,
}

impl Model {
    /// Every model there is.
    pub const ALL: [Model; 3] = [Model::Qwen27b, Model::QwenMax, Model::InklingMini];

    /// The model's name.
    pub const fn name(self) -> &'static str {
        match self {
            Model::Qwen27b => "qwen3.8-27b",
            Model::QwenMax => "qwen3.8-2.4t-a95b",
            Model::InklingMini => "inkling2-mini",
        }
    }

    /// Another word the model goes by, if it has one: the Qwen models go
    /// by their sizes too.
    pub const fn alias(self) -> Option<&'static str> {
        match self {
            Model::Qwen27b => Some("medium"),
            Model::QwenMax => Some("large"),
            Model::InklingMini => None,
        }
    }

    /// The weights the model runs from on this machine, if it runs here.
    pub fn weights(self) -> Option<&'static Weights> {
        match self {
            Model::Qwen27b => Some(&qwen::QWEN_27B),
            Model::QwenMax => None,
            Model::InklingMini => Some(&inkling::INKLING_MINI),
        }
    }

    /// How the model's tokens are sampled where no file of it says how:
    /// at a host.
    pub const fn sampling(self) -> Sampling {
        match self {
            Model::Qwen27b | Model::QwenMax => qwen::SAMPLING,
            Model::InklingMini => inkling::SAMPLING,
        }
    }

    /// How much the model reasons before it replies unless told. Qwen
    /// at `low`, not its default of `xhigh`, at which it thinks at length
    /// about code it could read. Inkling at `high`, its makers' default:
    /// at `low` it thought a median of 1,945 tokens an instance on
    /// SWE-bench Verified Mini, a third of what Qwen did at `low`, and
    /// its thoughts ran to the thinking budget 7 times in 50 sessions to
    /// Qwen's 89, so there is no overthinking to hold it from.
    pub const fn effort(self) -> Effort {
        match self {
            Model::Qwen27b | Model::QwenMax => Effort::Low,
            Model::InklingMini => Effort::High,
        }
    }

    /// The model's chat format, over the vocabulary a GGUF file of it
    /// carries.
    pub fn format(self, gguf: &gguf::Metadata) -> Result<Arc<dyn Format>> {
        match self {
            Model::Qwen27b | Model::QwenMax => qwen::format(gguf),
            Model::InklingMini => inkling::format(gguf),
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

    /// A model by its name, or by the other word it goes by.
    fn from_str(value: &str) -> std::result::Result<Self, String> {
        Model::ALL
            .into_iter()
            .find(|model| value == model.name() || Some(value) == model.alias())
            .ok_or_else(|| {
                let models: Vec<String> = Model::ALL
                    .iter()
                    .map(|model| match model.alias() {
                        Some(alias) => format!("{model} or {alias}"),
                        None => model.to_string(),
                    })
                    .collect();
                format!("unknown model '{value}' (expected {})", models.join(", "))
            })
    }
}
