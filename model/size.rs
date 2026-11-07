//! The device a machine runs the model on, and how big the model is there.

use std::{fmt, str::FromStr};

use serde::{Deserialize, Serialize};

/// Where to run the model: the CPU, or the GPU through an API of one's
/// choosing or whichever opens.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Device {
    Cpu,
    Gpu(Option<Api>),
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Device::Cpu => "cpu",
            Device::Gpu(None) => "gpu",
            Device::Gpu(Some(Api::Cuda)) => "cuda",
            Device::Gpu(Some(Api::Vulkan)) => "vulkan",
            Device::Gpu(Some(Api::Metal)) => "metal",
        })
    }
}

/// An API a GPU is driven through.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Api {
    Cuda,
    Vulkan,
    Metal,
}

impl Api {
    pub fn name(self) -> &'static str {
        match self {
            Api::Cuda => "CUDA",
            Api::Vulkan => "Vulkan",
            Api::Metal => "Metal",
        }
    }
}

impl fmt::Display for Api {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for Api {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "cuda" => Ok(Api::Cuda),
            "vulkan" => Ok(Api::Vulkan),
            "metal" => Ok(Api::Metal),
            _ => Err(format!(
                "unknown API '{s}' (expected cuda, vulkan, or metal)"
            )),
        }
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
