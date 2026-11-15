//! What is known of the model wherever it runs, by the engine that runs
//! it and by whatever reaches one that does.

pub mod format;
pub mod gguf;
pub mod hub;
mod sampling;
mod tokenizer;

pub use format::{Format, Turn};
pub use sampling::Sampling;
pub use tokenizer::Tokenizer;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
