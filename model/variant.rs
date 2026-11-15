use std::{fmt, str::FromStr};

/// The models dwim runs, by size. They are of one family, with one chat
/// format and one way of calling tools, so a conversation with one reads
/// the same to another.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Variant {
    /// Qwen3.8-27B, which runs on this machine.
    Medium,
    /// Qwen3.8-2.4T-A95B, the open weights of Qwen3.8 Max: 2.4 trillion
    /// parameters, of which 95 billion are active for a token. It takes
    /// more memory than a machine has, and runs where a provider has it.
    Large,
}

impl Variant {
    /// The model's name.
    pub const fn name(self) -> &'static str {
        match self {
            Variant::Medium => "qwen3.8-27b",
            Variant::Large => "qwen3.8-2.4t-a95b",
        }
    }

    /// The size the model is called by.
    pub const fn size(self) -> &'static str {
        match self {
            Variant::Medium => "medium",
            Variant::Large => "large",
        }
    }
}

impl fmt::Display for Variant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for Variant {
    type Err = String;

    /// A model by its size, or by its name.
    fn from_str(value: &str) -> Result<Self, String> {
        [Variant::Medium, Variant::Large]
            .into_iter()
            .find(|variant| value == variant.size() || value == variant.name())
            .ok_or_else(|| format!("unknown model '{value}' (expected medium or large)"))
    }
}
