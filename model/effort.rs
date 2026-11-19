//! How much the model reasons before it replies.

use std::str::FromStr;

/// How much the model reasons before it replies, as its chat template
/// asks for it: the template opens the system prompt with a line for an
/// effort of `xhigh`, its default, or `low`, and with none for `medium`.
/// Medium unless set: PrismML's known issues for Bonsai, its ternary
/// build of the model, which these figures are from and are carried over
/// unmeasured (huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf,
/// KNOWN_ISSUES.md), advise it for the model's long reasoning, as "about as accurate" as
/// `xhigh` when the output is limited and thinking "noticeably less", and
/// find that `low` "used about as many tokens as the default `xhigh`".
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Effort {
    Xhigh,
    #[default]
    Medium,
    Low,
}

impl Effort {
    /// The effort as the template, and a host, name it.
    pub fn name(self) -> &'static str {
        match self {
            Effort::Xhigh => "xhigh",
            Effort::Medium => "medium",
            Effort::Low => "low",
        }
    }

    /// The line the template opens the system prompt with, if any.
    pub fn instruction(self) -> Option<&'static str> {
        match self {
            Effort::Xhigh => Some(
                "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.",
            ),
            Effort::Medium => None,
            Effort::Low => Some(
                "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.",
            ),
        }
    }
}

impl FromStr for Effort {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "xhigh" => Ok(Effort::Xhigh),
            "medium" => Ok(Effort::Medium),
            "low" => Ok(Effort::Low),
            _ => Err(format!(
                "unknown reasoning effort '{s}' (expected xhigh, medium or low)"
            )),
        }
    }
}
