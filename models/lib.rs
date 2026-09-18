//! Language models: Bonsai, its tokenizer and weights, and a chat around
//! it.

pub mod bonsai;
mod chat;
pub mod gguf;
mod sampler;
mod tokenizer;

pub use chat::{Chat, Chunk, ToolCall};
pub use gguf::Gguf;
pub use hack_gpu::{Device, Tensor, ternary};
pub use sampler::Sampler;
pub use tokenizer::Tokenizer;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// A model that predicts the next token, keeping the sequence so far in
/// its own state.
pub trait LanguageModel {
    /// Runs `tokens`, the first at position `pos`, through the model, and
    /// returns the logits for the token that follows the last of them.
    fn forward(&mut self, tokens: &[u32], pos: usize) -> Vec<f32>;

    /// Longest sequence the state has room for.
    fn max_len(&self) -> usize;
}

impl<M: LanguageModel + ?Sized> LanguageModel for Box<M> {
    fn forward(&mut self, tokens: &[u32], pos: usize) -> Vec<f32> {
        (**self).forward(tokens, pos)
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
