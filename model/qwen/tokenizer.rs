//! The tokenizer of Qwen's vocabularies: byte-level BPE, with the split
//! into words that is theirs.

use crate::model::{Result, Tokenizer, gguf::Metadata};

/// The regular expression Qwen's tokenizers split text into words with.
const PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// The tokenizer of the vocabulary a GGUF file of a Qwen model carries.
/// Fails if the file says its text is split another way.
pub fn tokenizer(gguf: &Metadata) -> Result<Tokenizer> {
    let pre = gguf.str("tokenizer.ggml.pre")?;
    if !matches!(pre, "qwen2" | "qwen35") {
        return Err(format!("the '{pre}' pre-tokenizer is not supported").into());
    }
    Tokenizer::from_gguf(gguf, PATTERN)
}

impl Tokenizer {
    /// A tokenizer over a tiny vocabulary for tests: a token per byte, one
    /// merge, and the special tokens of Qwen's chat format.
    pub fn tiny() -> Self {
        Self::over(&crate::model::qwen::format::SPECIAL_TOKENS, PATTERN)
    }
}
