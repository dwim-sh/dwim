use std::path::PathBuf;

use argh::FromArgs;

use crate::{model, session::Settings};

/// A coding agent at the command line
#[derive(FromArgs)]
pub struct Opts {
    /// model to run (default: qwen3.8-27b)
    #[argh(option, default = "model::NAME.to_string()")]
    pub model: String,

    /// device to run the model on: gpu (default), or cpu
    #[argh(option, default = "Device::Gpu")]
    pub device: Device,

    /// tokens of conversation the model has room for (default: as many
    /// as fit in the device's free memory)
    #[argh(option, arg_name = "tokens")]
    pub context: Option<usize>,

    /// download the model if it is not yet, print the path of its weights,
    /// and exit
    #[argh(switch)]
    pub download: bool,

    /// print the version and exit
    #[argh(switch)]
    pub version: bool,

    /// print where the time went when the reply ends
    #[argh(switch)]
    pub stats: bool,

    /// time the model reading a page and generating at several depths of
    /// context, and print tokens a second for each
    #[argh(switch)]
    pub bench: bool,

    /// run the server that has the model, which sessions connect to, in the
    /// foreground; a session starts one itself when none is running
    #[argh(switch)]
    pub serve: bool,

    /// show the model's thinking and the files it reads in the shell,
    /// which ctrl+o toggles
    #[argh(switch)]
    pub thinking: bool,

    /// most tokens the model may think at a time before it is made to act,
    /// or 0 for no limit (default: 2048)
    #[argh(
        option,
        arg_name = "tokens",
        default = "dwim::harness::THINKING_BUDGET.unwrap_or(0)"
    )]
    pub thinking_budget: usize,

    /// how much the model reasons before it replies, as its chat template
    /// asks it to: xhigh, medium or low (default: medium)
    #[argh(option, arg_name = "effort", default = "dwim::harness::Effort::Medium")]
    pub reasoning_effort: dwim::harness::Effort,

    /// sampling temperature (default: what the model file recommends; 0
    /// always picks the likeliest token)
    #[argh(option, arg_name = "t")]
    pub temperature: Option<f32>,

    /// tokens the model drafts ahead and verifies in one pass, or 0 to
    /// decode a token at a time (default: 4)
    #[argh(option, arg_name = "n", default = "dwim::models::DRAFTS")]
    pub speculate: usize,

    /// program to run the model's shell commands through, as `shell -c
    /// command` (default: sh)
    #[argh(option, arg_name = "program")]
    pub shell: Option<PathBuf>,

    /// file to write the session's transcript to, as JSON Lines (default:
    /// a new file under ~/.local/state/dwim/projects/)
    #[argh(option, arg_name = "file")]
    pub transcript: Option<PathBuf>,

    /// prompt to answer without the shell, as the words after the options
    #[argh(positional, greedy)]
    pub prompt: Vec<String>,
}

impl Opts {
    /// How the model runs, as the options set it.
    pub fn settings(&self) -> Settings {
        Settings {
            model: self.model.clone(),
            device: self.device,
            context: self.context,
            thinking_budget: budget(self.thinking_budget),
            effort: self.reasoning_effort,
            temperature: self.temperature,
            speculate: self.speculate,
            shell: self.shell.clone(),
        }
    }
}

/// A thinking budget as the command line and the `/thinking-budget`
/// command give it: a number of tokens, or 0 for no limit.
pub fn budget(tokens: usize) -> Option<usize> {
    (tokens > 0).then_some(tokens)
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
