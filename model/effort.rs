//! How much the model reasons before it replies.

use std::str::FromStr;

/// How much the model reasons before it replies, as its chat template
/// asks for it: Qwen's template opens the system prompt with a line for
/// an effort of `xhigh`, its default, or `low`, and with none for
/// `medium`. It has no `high`, which Inkling's has, and is its default:
/// the line for it here is dwim's own, in the words of the template's
/// others, and what Qwen makes of it is not measured. Unless set, a
/// model reasons at [`crate::model::Model::effort`]. For Qwen that is
/// low, for the briefest thoughts the template asks for, which
/// a thought ended at the thinking budget cuts short the least. PrismML's
/// known issues for Bonsai, its ternary build of the model, advise
/// `medium` instead, as "about as accurate" as `xhigh` when the output is
/// limited and thinking "noticeably less", and find that `low` "used
/// about as many tokens as the default `xhigh`"; those figures are not
/// measured here (huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf,
/// KNOWN_ISSUES.md).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Effort {
    Xhigh,
    High,
    Medium,
    Low,
}

impl Effort {
    /// The effort as the template, and a host, name it.
    pub fn name(self) -> &'static str {
        match self {
            Effort::Xhigh => "xhigh",
            Effort::High => "high",
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
            Effort::High => Some(
                "Reasoning effort is set to high. Please think through the task and check key assumptions before the final answer.",
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
            "high" => Ok(Effort::High),
            "medium" => Ok(Effort::Medium),
            "low" => Ok(Effort::Low),
            _ => Err(format!(
                "unknown reasoning effort '{s}' (expected xhigh, high, medium or low)"
            )),
        }
    }
}
