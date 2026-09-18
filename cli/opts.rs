use argh::FromArgs;

use crate::models;

/// A coding agent at the command line
#[derive(FromArgs)]
pub struct Opts {
    /// model to run (default: bonsai-2-27b)
    #[argh(option, default = "models::DEFAULT.to_string()")]
    pub model: String,

    /// device to run the model on: gpu (default), or cpu
    #[argh(option, default = "Device::Gpu")]
    pub device: Device,

    /// tokens of conversation the model has room for (default: 32768)
    #[argh(option, default = "models::DEFAULT_CONTEXT")]
    pub context: usize,

    /// prompt to answer without the shell, as the words after the options
    #[argh(positional, greedy)]
    pub prompt: Vec<String>,
}

/// Where to run the model.
#[derive(Clone, Copy)]
pub enum Device {
    Cpu,
    Gpu,
}

impl argh::FromArgValue for Device {
    fn from_arg_value(value: &str) -> Result<Self, String> {
        match value {
            "cpu" => Ok(Device::Cpu),
            "gpu" => Ok(Device::Gpu),
            _ => Err(format!("unknown device '{value}' (expected gpu or cpu)")),
        }
    }
}
