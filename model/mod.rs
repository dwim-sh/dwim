//! What is known of the model wherever it runs, by the engine that runs
//! it and by whatever reaches one that does.

pub mod gguf;
pub mod hub;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
