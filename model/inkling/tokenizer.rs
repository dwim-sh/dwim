//! The tokenizer of Inkling's vocabulary: byte-level BPE, with the split
//! into words that is its own, and a word that is a token of the
//! vocabulary taken whole.

use crate::model::{Result, Tokenizer, gguf::Metadata};

/// The regular expression Inkling's tokenizer splits text into words
/// with, as its `tokenizer.json` has it.
const PATTERN: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// The tokenizer of the vocabulary a GGUF file of Inkling carries. Fails
/// if the file says its text is split another way.
pub fn tokenizer(gguf: &Metadata) -> Result<Tokenizer> {
    let pre = gguf.str("tokenizer.ggml.pre")?;
    if pre != "inkling2" {
        return Err(format!("the '{pre}' pre-tokenizer is not supported").into());
    }
    Ok(Tokenizer::from_gguf(gguf, PATTERN)?.with_whole_words())
}

impl Tokenizer {
    /// A tokenizer over a tiny vocabulary for tests: a token per byte, one
    /// merge, and the special tokens of Inkling's chat format.
    #[cfg(test)]
    pub fn tiny_inkling() -> Self {
        Self::over(&crate::model::inkling::format::SPECIAL_TOKENS, PATTERN)
    }
}
