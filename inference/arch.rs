//! The architectures dwim runs: how a model's network computes. A model's
//! weights come as a GGUF file, whose header names the architecture they
//! are for in its `general.architecture` field: `qwen35` in Qwen3.8-27B's,
//! and `inkling2_loop` in Inkling 2 Mini's.
//! Each architecture is a module of its own, and this is the one place
//! that tells them apart; what runs a model after this holds a
//! [`LanguageModel`], whatever its architecture.

use std::sync::Arc;

use crate::{
    gpu::Device,
    inference::{Gguf, LanguageModel, Memory, Result, inkling2, qwen35},
};

/// What a file says of its model before it is loaded: the device memory
/// the model takes, and the positions it was trained for, past which its
/// attention degrades.
pub struct Plan {
    pub memory: Memory,
    pub trained: usize,
}

/// What the model in `gguf` takes on a device that runs `batch` tokens at
/// once. Fails if the file names an architecture dwim does not run.
pub fn plan(gguf: &Gguf, batch: usize) -> Result<Plan> {
    match gguf.str("general.architecture")? {
        qwen35::ARCHITECTURE => {
            let config = qwen35::Config::load(gguf)?;
            Ok(Plan {
                memory: config.memory(gguf, batch)?,
                trained: config.context_length,
            })
        }
        inkling2::ARCHITECTURE => {
            let config = inkling2::Config::load(gguf)?;
            Ok(Plan {
                memory: config.memory(gguf, batch)?,
                trained: config.context_length,
            })
        }
        other => Err(unsupported(other)),
    }
}

/// Loads the model in `gguf` onto `device`, with room for `context` tokens
/// of conversation, reporting how many of its tensors are loaded, out of
/// how many, as it goes.
pub fn load<D: Device + 'static>(
    gguf: Arc<Gguf>,
    device: D,
    context: usize,
    on_progress: impl FnMut(usize, usize),
) -> Result<Box<dyn LanguageModel>> {
    match gguf.str("general.architecture")? {
        qwen35::ARCHITECTURE => Ok(Box::new(qwen35::Model::load(
            gguf,
            device,
            context,
            on_progress,
        )?)),
        inkling2::ARCHITECTURE => Ok(Box::new(inkling2::Model::load(
            gguf,
            device,
            context,
            on_progress,
        )?)),
        other => Err(unsupported(other)),
    }
}

/// What is said of a file of an architecture dwim does not run.
fn unsupported(architecture: &str) -> Box<dyn std::error::Error> {
    format!("the '{architecture}' architecture is not supported").into()
}
