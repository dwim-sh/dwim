//! What is known of the model wherever it runs, by the engine that runs
//! it and by whatever reaches one that does.

mod chat;
pub mod format;
pub mod gguf;
pub mod hub;
mod sampling;
mod session;
mod size;
mod tokenizer;
mod variant;

pub use chat::{Chat, Generation, State};
pub use format::{Format, Turn};
pub use sampling::Sampling;
pub use session::{Answer, Chunk, Reply, Session, Stats, Tally};
pub use size::{Device, Size};
pub use tokenizer::Tokenizer;
pub use variant::Variant;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
