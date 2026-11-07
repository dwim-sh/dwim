//! Language models: Qwen3.8-27B, its tokenizer and weights, and a chat
//! around it.

mod chat;
pub mod gguf;
pub mod hub;
pub mod quants;
pub mod qwen35;
mod sampler;
#[cfg(test)]
pub mod testing;
mod tokenizer;

pub use chat::{Answer, Chat, Chunk, DRAFTS, Reply, Stats, Tally, Turn};
pub use gguf::Gguf;
pub use sampler::Sampler;
pub use tokenizer::Tokenizer;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// A model that predicts the next token, keeping the sequence so far in
/// its own state.
pub trait LanguageModel {
    /// The model's state after the first `len` positions, as bytes for
    /// [`restore`](Self::restore) to take up from, if the model can give
    /// it.
    fn save(&self, _len: usize) -> Option<Vec<u8>> {
        None
    }

    /// Puts the model in a state [`save`](Self::save) gave, and returns
    /// how many positions it holds.
    fn restore(&mut self, _state: &[u8]) -> Result<usize> {
        Err("this model cannot restore a state".into())
    }

    /// Runs `tokens`, the first at position `pos`, through the model, and
    /// returns the logits for the token that follows the last of them.
    fn forward(&mut self, tokens: &[u32], pos: usize) -> Vec<f32>;

    /// Drafts up to `k` tokens to follow `token`, which was picked at
    /// position `pos` from the logits the latest run gave: the most likely
    /// token after each by the drafter's reckoning, if the model has a
    /// drafter and nothing is being verified, and none otherwise. The
    /// token and the drafts are then run as one batch by
    /// [`verify`](Self::verify).
    fn draft(&mut self, _token: u32, _pos: usize, _k: usize) -> Vec<u32> {
        Vec::new()
    }

    /// Runs `tokens`, the first at position `pos`, as a batch to be
    /// verified: a token picked and the tokens drafted after it. Returns
    /// the logits after each, a row per token, and commits none of them
    /// until [`rewind`](Self::rewind) says how many to keep. Only for a
    /// model that drafted them.
    fn verify(&mut self, _tokens: &[u32], _pos: usize) -> Vec<f32> {
        unreachable!("the model does not draft")
    }

    /// Commits the first `keep` tokens of the batch being verified, one at
    /// least, and drops the rest.
    fn rewind(&mut self, _keep: usize) {
        unreachable!("the model does not draft")
    }

    /// Marks the sequence so far, keeping what
    /// [`rewind_to_mark`](Self::rewind_to_mark) needs to take it up again
    /// from here, if the model can: returns whether it did. A mark is kept
    /// until the next, or until the model is reset or restored.
    fn mark(&mut self) -> bool {
        false
    }

    /// Puts the model back at the mark, forgetting the positions after
    /// it: the next tokens start where the mark was made.
    fn rewind_to_mark(&mut self) -> Result<()> {
        Err("this model cannot rewind to a mark".into())
    }

    /// Forgets the sequence so far, so that the next tokens start at
    /// position 0.
    fn reset(&mut self);

    /// Longest sequence the state has room for.
    fn max_len(&self) -> usize;
}

impl<M: LanguageModel + ?Sized> LanguageModel for Box<M> {
    fn save(&self, len: usize) -> Option<Vec<u8>> {
        (**self).save(len)
    }

    fn restore(&mut self, state: &[u8]) -> Result<usize> {
        (**self).restore(state)
    }

    fn forward(&mut self, tokens: &[u32], pos: usize) -> Vec<f32> {
        (**self).forward(tokens, pos)
    }

    fn draft(&mut self, token: u32, pos: usize, k: usize) -> Vec<u32> {
        (**self).draft(token, pos, k)
    }

    fn verify(&mut self, tokens: &[u32], pos: usize) -> Vec<f32> {
        (**self).verify(tokens, pos)
    }

    fn rewind(&mut self, keep: usize) {
        (**self).rewind(keep)
    }

    fn mark(&mut self) -> bool {
        (**self).mark()
    }

    fn rewind_to_mark(&mut self) -> Result<()> {
        (**self).rewind_to_mark()
    }

    fn reset(&mut self) {
        (**self).reset()
    }

    fn max_len(&self) -> usize {
        (**self).max_len()
    }
}

/// The cosines and sines of the angles rotary position embeddings rotate
/// by, laid out as [`Device::rope`] expects, for positions up to `max_len`
/// and `rot_dim` rotated elements per head. Each pair of elements rotates at
/// its own frequency.
pub fn rope_table(max_len: usize, rot_dim: usize, theta: f32) -> Vec<f32> {
    let half = rot_dim / 2;
    let mut table = Vec::with_capacity(max_len * rot_dim);
    for pos in 0..max_len {
        for i in 0..half {
            let freq = 1.0 / theta.powf((2 * i) as f32 / rot_dim as f32);
            let (sin, cos) = (pos as f32 * freq).sin_cos();
            table.extend([cos, sin]);
        }
    }
    table
}
