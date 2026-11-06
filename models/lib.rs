//! Language models: Bonsai, its tokenizer and weights, and a chat around
//! it.

pub mod bonsai;
pub mod cache;
mod chat;
pub mod draft;
pub mod gguf;
pub mod safetensors;
mod sampler;
pub mod speculative;
#[cfg(any(test, feature = "testing"))]
pub mod testing;
mod tokenizer;

pub use cache::Source;
pub use chat::{Chat, Chunk, Stats, Tally, Turn};
pub use dwim_gpu::{Device, Tensor, ternary, twobit};
pub use gguf::Gguf;
pub use safetensors::Safetensors;
pub use sampler::{Distribution, Policy, Sampler};
pub use tokenizer::Tokenizer;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// How drafting ahead has gone: the cycles run, the tokens proposed, and
/// the ones the target accepted.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Drafting {
    pub cycles: usize,
    pub proposed: usize,
    pub accepted: usize,
    /// Seconds spent in the draft, and in the target's passes over its
    /// proposals.
    pub draft_seconds: f64,
    pub target_seconds: f64,
}

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

    /// Runs `anchor`, the token at position `pos`, through the model, and
    /// picks the token after it with `policy`: that token, and perhaps
    /// more. A model that drafts ahead returns every drafted token it had
    /// its target accept, each picked as `policy` would pick it, and then
    /// the one picked after them; its state then holds the anchor and the
    /// accepted tokens, and the last token returned is yet to run through
    /// it. The tokens are at the positions after `pos`, in order.
    fn next_tokens(&mut self, anchor: u32, pos: usize, policy: &mut Policy) -> Vec<u32> {
        let logits = self.forward(&[anchor], pos);
        vec![policy.pick(&logits)]
    }

    /// Forgets the tokens from position `len` on, which can only be
    /// those [`next_tokens`](Self::next_tokens) last ran through the model
    /// beyond the anchor.
    fn rewind(&mut self, _len: usize) -> Result<()> {
        Err("the model cannot forget tokens".into())
    }

    /// Forgets the sequence so far, so that the next tokens start at
    /// position 0.
    fn reset(&mut self);

    /// How the model's drafting has gone, if it drafts ahead.
    fn drafting(&self) -> Option<Drafting> {
        None
    }

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

    fn next_tokens(&mut self, anchor: u32, pos: usize, policy: &mut Policy) -> Vec<u32> {
        (**self).next_tokens(anchor, pos, policy)
    }

    fn rewind(&mut self, len: usize) -> Result<()> {
        (**self).rewind(len)
    }

    fn reset(&mut self) {
        (**self).reset()
    }

    fn drafting(&self) -> Option<Drafting> {
        (**self).drafting()
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
