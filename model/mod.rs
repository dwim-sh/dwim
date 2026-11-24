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
    /// Inkling 2 Mini, Thinking Machines' looped model of 26 billion
    /// parameters, of which 3.4 billion are active for a token: it runs on
    /// this machine, where the GPU takes its weights, and at OpenRouter.
    /// The model dwim runs unless told to run another.
    #[default]
    InklingMini,
    /// Inkling 2 Small, the larger of the Inkling 2 models: it takes more
    /// memory than a machine has, and runs where a provider has it.
    InklingSmall,
    /// Qwen3.8-27B, which runs on this machine.
    Qwen27b,
    /// Qwen3.8-2.4T-A95B, the open weights of Qwen3.8 Max: 2.4 trillion
    /// parameters, of which 95 billion are active for a token. It takes
    /// more memory than a machine has, and runs where a provider has it.
    QwenMax,
}

impl Model {
    /// Every model there is, in the order the shell's pickers show them:
    /// the Inkling 2 models, and the Qwen models, which the pickers do not
    /// show but the command line still names.
    pub const ALL: [Model; 4] = [
        Model::InklingMini,
        Model::InklingSmall,
        Model::Qwen27b,
        Model::QwenMax,
    ];

    /// The models the shell's pickers show, in the order they are shown:
    /// the Inkling 2 ones, which go by their sizes too.
    pub const SHOWN: [Model; 2] = [Model::InklingMini, Model::InklingSmall];

    /// The model's name.
    pub const fn name(self) -> &'static str {
        match self {
            Model::InklingMini => "inkling2-mini",
            Model::InklingSmall => "inkling2-small",
            Model::Qwen27b => "qwen3.8-27b",
            Model::QwenMax => "qwen3.8-2.4t-a95b",
        }
    }

    /// Another word the model goes by, if it has one: the Inkling 2 models
    /// go by their sizes too.
    pub const fn alias(self) -> Option<&'static str> {
        match self {
            Model::InklingMini => Some("medium"),
            Model::InklingSmall => Some("large"),
            Model::Qwen27b => None,
            Model::QwenMax => None,
        }
    }

    /// The weights the model runs from on this machine, if it runs here.
    pub fn weights(self) -> Option<&'static Weights> {
        match self {
            Model::InklingMini => Some(&inkling::INKLING_MINI),
            Model::InklingSmall => None,
            Model::Qwen27b => Some(&qwen::QWEN_27B),
            Model::QwenMax => None,
        }
    }

    /// The effort the model is asked to reason at when none is given:
    /// medium for Inkling 2 Mini, low for the rest.
    pub const fn default_effort(self) -> Effort {
        match self {
            Model::InklingMini => Effort::Medium,
            _ => Effort::Low,
        }
    }

    /// How the model's tokens are sampled where no file of it says how:
    /// at a host.
    pub const fn sampling(self) -> Sampling {
        match self {
            Model::InklingMini | Model::InklingSmall => inkling::SAMPLING,
            Model::Qwen27b | Model::QwenMax => qwen::SAMPLING,
        }
    }

    /// Whether the model has to be steered as it works to keep to the
    /// rules of its system prompt: told them in the user's message, and
    /// reminded of one where it has just gone another way. Inkling 2
    /// Mini's makers say of it that "when using tools, it is less reliable
    /// than larger models at following detailed instructions", and they
    /// run it with a shell alone. Told in the system prompt to lay out a
    /// plan first, or to change files with edit and not sed, it did in
    /// none of 8 trials each, and went on through bash; told the same in
    /// the message, it did in 7 of 8 and 5 of 8. Qwen3.8-27B does both
    /// from the prompt.
    pub const fn needs_steering(self) -> bool {
        match self {
            Model::InklingMini | Model::InklingSmall => true,
            Model::Qwen27b | Model::QwenMax => false,
        }
    }

    /// The model's chat format, over the vocabulary a GGUF file of it
    /// carries.
    pub fn format(self, gguf: &gguf::Metadata) -> Result<Arc<dyn Format>> {
        match self {
            Model::InklingMini | Model::InklingSmall => inkling::format(gguf),
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

    /// A model by its name, or by the other word it goes by.
    fn from_str(value: &str) -> std::result::Result<Self, String> {
        Model::ALL
            .into_iter()
            .find(|model| value == model.name() || Some(value) == model.alias())
            .ok_or_else(|| {
                let models: Vec<String> = Model::SHOWN
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
