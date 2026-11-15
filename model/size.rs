//! The device a machine runs the model on, and how big the model is there.

use std::fmt;

use serde::{Deserialize, Serialize};

/// Where to run the model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Device {
    Cpu,
    Gpu,
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Device::Cpu => "cpu",
            Device::Gpu => "gpu",
        })
    }
}

/// How many sessions the model has room for at once, the tokens of
/// conversation each has room for, and the device memory they take:
/// `model` bytes for the weights and the activations, which the sessions
/// share, and `conversation` bytes more for each session, for its recurrent
/// states and its tokens of context.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct Size {
    pub sessions: usize,
    pub context: usize,
    pub model: u64,
    pub conversation: u64,
}
