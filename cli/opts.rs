use std::{collections::BTreeMap, env, path::PathBuf};

use argh::{EarlyExit, FromArgs};
use dwim::inference::Loaded;
use dwim::model::{Device, Model, Size};
use dwim::providers::Kind;

use crate::{model, session::Settings, tui};

/// A coding agent at the command line
#[derive(FromArgs)]
pub struct Opts {
    /// model to run: inkling2-mini (default), which medium also names, on
    /// this machine or at OpenRouter, or inkling2-small, which large also
    /// names, at OpenRouter or Vercel
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
    /// asks it to: xhigh, high, medium or low (default: low)
    #[argh(option, arg_name = "effort", default = "dwim::model::Effort::Low")]
    pub reasoning_effort: dwim::model::Effort,

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
                    Ok(()) => print!("{}", help(&output, &rest)),
                    Err(()) => eprint!("{output}"),
                }
                std::process::exit(status.map_or(1, |()| 0));
            }
        }
    }

    /// How the model runs, as the options set it.
    pub fn settings(&self) -> Settings {
        Settings {
            model: self.model,
            provider: self.provider,
            device: self.device,
            context: self.context,
            sessions: self.sessions,
            thinking_budget: budget(self.thinking_budget),
            effort: self.reasoning_effort,
            temperature: self.temperature,
            speculate: self.speculate,
            shell: self.shell.clone(),
        }
    }

    /// The prompt to answer without the shell: the words after the
    /// options, or none when there are none or they are all blank.
    pub fn prompt(&self) -> Option<String> {
        let prompt = self.prompt.join(" ");
        let prompt = prompt.trim();
        (!prompt.is_empty()).then(|| prompt.to_string())
    }
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

/// The headings `--help` lists the options under, each with its options by
/// name. An option under none of them is left out of the help.
const GROUPS: &[(&str, &[&str])] = &[
    (
        "The model:",
        &[
            "--context",
            "--device",
            "--model",
            "--provider",
            "--sessions",
        ],
    ),
    (
        "Thinking and sampling:",
        &[
            "--reasoning-effort",
            "--speculate",
            "--temperature",
            "--thinking-budget",
        ],
    ),
    (
        "The session:",
        &["--shell", "--stats", "--thinking", "--transcript"],
    ),
    (
        "Instead of a session:",
        &["--bench", "--download", "--help", "--serve", "--version"],
    ),
];

/// The first line of the help text, in place of the one argh writes, which
/// spells out every option.
const USAGE: &str = "Usage: dwim [options] [prompt...]";

/// What dwim is, under the usage line.
const ABOUT: &str = "A coding agent at the command line. With no prompt, it starts an \
    interactive session in the current directory; given one, it answers it without the \
    shell and exits.";

/// The column an option's description starts in, when the option's name
/// ends two or more columns before it; a longer name has its description
/// under it instead.
const COLUMN: usize = 24;

/// The width the help text wraps at.
const WIDTH: usize = 80;

/// The column argh starts an option's description in.
const ARGH_COLUMN: usize = 20;

/// The help text: the options argh describes in `output`, sorted under
/// headings, with the default context worked out for the model and the
/// device the arguments name, if the model is downloaded and the device
/// opens.
pub fn help(output: &str, args: &[&str]) -> String {
    let value = |option: &str| {
        args.iter()
            .position(|&arg| arg == option)
            .and_then(|i| args.get(i + 1))
    };
    let model = value("--model")
        .and_then(|value| value.parse().ok())
        .unwrap_or_default();
    let device = value("--device")
        .and_then(|value| device(value).ok())
        .unwrap_or(Device::Gpu(None));
    layout(output, Loaded::default_size(model, device))
}

/// The help text laid out from argh's `output`, saying the default context
/// is `size` if that is known.
fn layout(output: &str, size: Option<Size>) -> String {
    let mut options = options(output);
    if let (Some(size), Some((_, description))) = (size, options.get_mut("--context")) {
        *description = format!(
            "tokens of conversation each session has room for (default: {} on this machine, \
             taking {} a session beside the model's {}, with room for {} sessions)",
            size.context,
            model::bytes(size.conversation),
            model::bytes(size.model),
            size.sessions
        );
    }
    let mut text = format!("{USAGE}\n\n");
    for line in tui::wrap(ABOUT, WIDTH) {
        text.push_str(line);
        text.push('\n');
    }
    for (heading, names) in GROUPS {
        text.push('\n');
        text.push_str(heading);
        text.push('\n');
        for name in names.iter() {
            let Some((usage, description)) = options.get(name) else {
                continue;
            };
            let label = format!("  {usage}");
            let mut lines = tui::wrap(description, WIDTH - COLUMN).into_iter();
            if label.len() + 2 <= COLUMN {
                let first = lines.next().unwrap_or("");
                text.push_str(&format!("{label:<COLUMN$}{first}\n"));
            } else {
                text.push_str(&label);
                text.push('\n');
            }
            for line in lines {
                text.push_str(&format!("{:COLUMN$}{line}\n", ""));
            }
        }
    }
    text
}

/// The options argh's help `output` describes, by name: how each is given,
/// as the usage line has it, such as `--context <tokens>`, and what it does.
fn options(output: &str) -> BTreeMap<&str, (String, String)> {
    let mut lines = output.lines();
    let usages: BTreeMap<&str, &str> = lines
        .next()
        .unwrap_or("")
        .split('[')
        .filter_map(|part| part.split(']').next())
        .filter(|part| part.starts_with("--"))
        .map(|usage| (usage.split(' ').next().unwrap_or(usage), usage))
        .collect();
    let mut entries: Vec<(&str, String)> = Vec::new();
    for line in lines.skip_while(|line| *line != "Options:").skip(1) {
        if let Some(rest) = line
            .strip_prefix("  ")
            .filter(|rest| !rest.starts_with(' '))
        {
            let name = rest.split([',', ' ']).next().unwrap_or(rest);
            let description = line.get(ARGH_COLUMN..).unwrap_or("").trim();
            entries.push((name, description.to_string()));
        } else if let Some((_, description)) =
            entries.last_mut().filter(|_| line.starts_with("    "))
        {
            if !description.is_empty() {
                description.push(' ');
            }
            description.push_str(line.trim());
        } else {
            break;
        }
    }
    entries
        .into_iter()
        .map(|(name, description)| {
            let usage = usages.get(name).copied().unwrap_or(name).to_string();
            (name, (usage, description))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use argh::FromArgs;

    use super::*;

    fn parse(args: &[&str]) -> Result<Opts, String> {
        Opts::from_args(&["dwim"], args).map_err(|e| e.output)
    }

    #[test]
    fn refuses_options_out_of_range() {
        let err = parse(&["--sessions", "0"]).err().expect("--sessions 0");
        assert!(err.contains("a server needs room for one session at least"));

        assert!(parse(&["--sessions", "x"]).is_err());
        assert_eq!(parse(&["--sessions", "3"]).unwrap().sessions, Some(3));

        let err = parse(&["--temperature", "nan"])
            .err()
            .expect("--temperature nan");
        assert!(err.contains("a number 0 or more"));
        let err = parse(&["--temperature", "inf"])
            .err()
            .expect("--temperature inf");
        assert!(err.contains("a number 0 or more"));
        assert!(parse(&["--temperature", "-1"]).is_err());

        assert_eq!(
            parse(&["--temperature", "0"]).unwrap().temperature,
            Some(0.0)
        );
        assert_eq!(parse(&[]).unwrap().sessions, None);
    }
}
