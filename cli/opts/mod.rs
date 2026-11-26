mod help;

use std::{env, io, path::PathBuf};

use argh::{EarlyExit, FromArgs};
use dwim::{
    model::{Device, Model},
    providers::Kind,
};

use crate::session::{Engine, Settings};

/// A coding agent at the command line
#[derive(FromArgs)]
pub struct Opts {
    /// model to run: inkling2-mini (default), which medium also names, on
    /// this machine, at OpenRouter or at Vercel, or inkling2-small, which
    /// large also names, at OpenRouter or Vercel
    #[argh(option, default = "Model::default()")]
    pub model: Model,

    /// where the model runs: local, on this machine, openrouter, with the
    /// key in OPENROUTER_API_KEY, vercel, Vercel's AI Gateway, with the
    /// key in AI_GATEWAY_API_KEY, or cerebras, with the key in
    /// CEREBRAS_API_KEY (default: local where the model runs here, and a
    /// host there is a key to where it does not)
    #[argh(option)]
    pub provider: Option<Kind>,

    /// device to run the model on: gpu (default), cpu, or the GPU through
    /// one API: cuda, vulkan, or metal
    #[argh(option, default = "Device::Gpu(None)", from_str_fn(device))]
    pub device: Device,

    /// tokens of conversation each session has room for (default: as
    /// many as fit in the device's free memory, shared among the sessions)
    #[argh(option, arg_name = "tokens")]
    pub context: Option<usize>,

    /// most sessions the model has room for at once, each with state and
    /// a context of its own beside the weights they share (default: 2)
    #[argh(option, arg_name = "n", from_str_fn(sessions))]
    pub sessions: Option<usize>,

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
    /// asks it to: xhigh, high, medium or low (default: medium for
    /// inkling2-mini, low otherwise)
    #[argh(option, arg_name = "effort")]
    pub reasoning_effort: Option<dwim::model::Effort>,

    /// sampling temperature (default: what the model file recommends; 0
    /// always picks the likeliest token)
    #[argh(option, arg_name = "t", from_str_fn(temperature))]
    pub temperature: Option<f32>,

    /// tokens the model drafts ahead and verifies in one pass, or 0 to
    /// decode a token at a time (default: 4)
    #[argh(option, arg_name = "n", default = "dwim::inference::DRAFTS")]
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
    /// The options as the command line gives them: a parse error prints
    /// the help, or the error, and exits.
    pub fn parse() -> Opts {
        let args: Vec<String> = env::args().collect();
        let command = args[0].rsplit('/').next().unwrap_or(&args[0]);
        let rest: Vec<&str> = args[1..].iter().map(String::as_str).collect();
        match Opts::from_args(&[command], &rest) {
            Ok(opts) => opts,
            Err(EarlyExit { output, status }) => {
                match status {
                    Ok(()) => print!("{}", help::help(&output, &rest)),
                    Err(()) => eprint!("{output}"),
                }
                std::process::exit(status.map_or(1, |()| 0));
            }
        }
    }

    /// How the agent runs, as the options set it, in the working
    /// directory, with a model that runs on this machine in the server
    /// this program is.
    pub fn settings(&self) -> io::Result<Settings> {
        Ok(Settings {
            dir: env::current_dir()?,
            model: self.model,
            provider: self.provider,
            disabled: Vec::new(),
            engine: Engine::Server {
                program: env::current_exe()?,
            },
            device: self.device,
            context: self.context,
            sessions: self.sessions,
            thinking_budget: budget(self.thinking_budget),
            effort: self
                .reasoning_effort
                .unwrap_or_else(|| self.model.default_effort()),
            temperature: self.temperature,
            speculate: self.speculate,
            shell: self.shell.clone(),
            tools: Vec::new(),
        })
    }

    /// What the options ask dwim to do. A switch that does something
    /// instead of a session wins over a prompt, and `--version` over the
    /// rest.
    pub fn command(&self) -> Command {
        let prompt = self.prompt.join(" ");
        let prompt = prompt.trim();
        if self.version {
            Command::Version
        } else if self.download {
            Command::Download
        } else if self.bench {
            Command::Bench
        } else if self.serve {
            Command::Serve
        } else if prompt.is_empty() {
            Command::Shell
        } else {
            Command::Once(prompt.to_string())
        }
    }
}

/// What dwim is asked to do.
pub enum Command {
    /// Print the version.
    Version,
    /// Download the model and print the path of its weights.
    Download,
    /// Time the model and print tokens a second.
    Bench,
    /// Run the server that has the model.
    Serve,
    /// Start an interactive session.
    Shell,
    /// Answer a prompt without the shell.
    Once(String),
}

/// A thinking budget as the command line and the `/thinking-budget`
/// command give it: a number of tokens, or 0 for no limit.
pub fn budget(tokens: usize) -> Option<usize> {
    (tokens > 0).then_some(tokens)
}

/// A device as `--device` takes it.
fn device(value: &str) -> Result<Device, String> {
    match value {
        "cpu" => Ok(Device::Cpu),
        "gpu" => Ok(Device::Gpu(None)),
        _ => value
            .parse()
            .map(|api| Device::Gpu(Some(api)))
            .map_err(|_| {
                format!("unknown device '{value}' (expected gpu, cpu, cuda, vulkan, or metal)")
            }),
    }
}

/// A number of sessions as `--sessions` takes it: 1 or more.
fn sessions(value: &str) -> Result<usize, String> {
    match value.parse::<usize>() {
        Ok(0) => Err("a server needs room for one session at least".to_string()),
        Ok(n) => Ok(n),
        Err(_) => Err("expected a number of sessions, 1 or more".to_string()),
    }
}

/// A temperature as `--temperature` takes it: a number, 0 or more.
fn temperature(value: &str) -> Result<f32, String> {
    match value.parse::<f32>() {
        Ok(t) if t.is_finite() && t >= 0.0 => Ok(t),
        _ => Err("expected a temperature, a number 0 or more".to_string()),
    }
}
