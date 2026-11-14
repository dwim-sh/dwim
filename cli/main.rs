mod bench;
mod fetch;
mod model;
mod once;
mod opts;
mod run;
mod serve;
mod session;
mod tui;

use std::{
    collections::BTreeMap,
    env,
    error::Error,
    path::{Path, PathBuf},
};

use argh::{EarlyExit, FromArgValue, FromArgs};
use dwim::harness::Log;
use dwim::models::qwen35;
use dwim::sys::memory;

use opts::{Device, Opts};
use session::Stderr;

fn main() {
    let args: Vec<String> = env::args().collect();
    let command = args[0].rsplit('/').next().unwrap_or(&args[0]);
    let rest: Vec<&str> = args[1..].iter().map(String::as_str).collect();
    let opts = match Opts::from_args(&[command], &rest) {
        Ok(opts) => opts,
        Err(EarlyExit { output, status }) => {
            match status {
                Ok(()) => print!("{}", help(&output, &rest)),
                Err(()) => eprint!("{output}"),
            }
            std::process::exit(status.map_or(1, |()| 0));
        }
    };
    if opts.version {
        println!("dwim {}", env!("CARGO_PKG_VERSION"));
        return;
    }
    let settings = opts.settings();
    let prompt = opts.prompt.join(" ");
    let result = memory::check().and_then(|()| {
        if opts.download {
            let mut stderr = Stderr::new();
            let dir = session::fetch_model(&settings, |loading| stderr.report(&loading))?;
            println!("{}", dir.join(qwen35::WEIGHTS).display());
            return Ok(());
        }
        if opts.bench {
            return bench::bench(&settings);
        }
        if opts.serve {
            return serve::serve(&settings);
        }
        let log = log(opts.transcript)?;
        match prompt.trim() {
            "" => run::run(settings, opts.thinking, log),
            prompt => once::once(&settings, prompt, opts.stats, log),
        }
    });
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

/// The session's transcript, opened before anything else so that a file
/// it cannot be written to is found out at once: in `path`, if the
/// options name one, or in a new file in the working directory's folder of
/// the user's state directory (`~/.local/state`, or on macOS, where there
/// is none, `~/Library/Application Support`), out of the agent's sight.
/// Where that cannot be written, the session goes on without one, with a
/// warning.
fn log(path: Option<PathBuf>) -> Result<Log, Box<dyn Error>> {
    if let Some(path) = path {
        return Log::create(&path)
            .map_err(|e| format!("cannot write the transcript to {}: {e}", path.display()).into());
    }
    let Some(dir) = dirs::state_dir().or_else(dirs::data_local_dir) else {
        return Ok(Log::none());
    };
    let path = dir
        .join("dwim")
        .join("projects")
        .join(project(&env::current_dir()?))
        .join(Log::name());
    Ok(Log::create(&path).unwrap_or_else(|e| {
        eprintln!(
            "warning: keeping no transcript: cannot write {}: {e}",
            path.display()
        );
        Log::none()
    }))
}

/// The name of the folder for the transcripts of sessions in `dir`: its
/// path with everything but letters and digits made a dash, such as
/// `-home-penberg-src-dwim`.
fn project(dir: &Path) -> String {
    dir.to_string_lossy()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect()
}

/// The headings `--help` lists the options under, each with its options by
/// name. An option under none of them is left out of the help.
const GROUPS: &[(&str, &[&str])] = &[
    (
        "The model:",
        &["--context", "--device", "--model", "--sessions"],
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
/// headings, with the default context worked out for the device the
/// arguments name, if the model is downloaded and the device opens.
fn help(output: &str, args: &[&str]) -> String {
    let device = args
        .iter()
        .position(|&arg| arg == "--device")
        .and_then(|i| args.get(i + 1))
        .and_then(|value| Device::from_arg_value(value).ok())
        .unwrap_or(Device::Gpu);
    layout(output, model::default_size(device))
}

/// The help text laid out from argh's `output`, saying the default context
/// is `size` if that is known.
fn layout(output: &str, size: Option<model::Size>) -> String {
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
