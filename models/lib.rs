//! Language models: the transformer, its tokenizer and weights, and a chat
//! around it.

mod chat;
pub mod qwen3;
mod safetensors;
mod sampler;
mod tokenizer;

pub use chat::{Chat, Chunk, ToolCall};
pub use hack_gpu::{Device, Tensor};
pub use safetensors::Weights;
pub use sampler::Sampler;
pub use tokenizer::Tokenizer;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
