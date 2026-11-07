mod bench;
mod fetch;
mod model;
mod once;
mod opts;
mod run;
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
    let result = check_memory().and_then(|()| {
        if opts.download {
            let mut stderr = Stderr::new();
            let dir = session::fetch_model(&settings, |loading| stderr.report(&loading))?;
            println!("{}", dir.join(qwen35::WEIGHTS).display());
            return Ok(());
        }
        if opts.bench {
            return bench::bench(&settings);
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
/// name. Every option has to be under one of them, which a test checks.
const GROUPS: &[(&str, &[&str])] = &[
    ("The model:", &["--context", "--device", "--model"]),
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
        &["--bench", "--download", "--help", "--version"],
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
        .unwrap_or(Device::Gpu(None));
    layout(output, model::default_size(device))
}

/// The help text laid out from argh's `output`, saying the default context
/// is `size` if that is known.
fn layout(output: &str, size: Option<model::Size>) -> String {
    let mut options = options(output);
    if let (Some(size), Some((_, description))) = (size, options.get_mut("--context")) {
        *description = format!(
            "tokens of conversation the model has room for (default: {} on this machine, \
             taking {} beside the model's {})",
            size.context,
            model::bytes(size.conversation),
            model::bytes(size.model)
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

/// Memory a Mac needs to run the model: its CPU and GPU share it, and the
/// weights alone take 16 GB of it.
#[cfg(target_vendor = "apple")]
const MIN_MEMORY: u64 = 32 << 30;

/// Refuses to go on, before any weights are downloaded, on a Mac with too
/// little memory to run the model.
#[cfg(target_vendor = "apple")]
fn check_memory() -> Result<(), Box<dyn Error>> {
    let memory = dwim::gpu::host::total().ok_or("cannot determine the memory size")?;
    if memory < MIN_MEMORY {
        return Err(format!(
            "dwim requires {} GiB or more DRAM on a Mac, and this one has {} GiB",
            MIN_MEMORY >> 30,
            memory >> 30
        )
        .into());
    }
    Ok(())
}

#[cfg(not(target_vendor = "apple"))]
fn check_memory() -> Result<(), Box<dyn Error>> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The help text argh writes for the options.
    fn output() -> String {
        match Opts::from_args(&["dwim"], &["--help"]) {
            Err(EarlyExit {
                output,
                status: Ok(()),
            }) => output,
            _ => panic!("--help did not print the help"),
        }
    }

    #[test]
    fn help_lists_every_option_once() {
        let listed: Vec<&str> = GROUPS
            .iter()
            .flat_map(|(_, names)| names.iter().copied())
            .collect();
        let mut sorted = listed.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), listed.len(), "an option is listed twice");
        let output = output();
        let parsed: Vec<&str> = options(&output).into_keys().collect();
        assert_eq!(sorted, parsed, "the groups and the options differ");
    }

    #[test]
    fn help_sorts_each_group_by_name() {
        for (heading, names) in GROUPS {
            assert!(
                names.windows(2).all(|pair| pair[0] < pair[1]),
                "{heading} is out of order"
            );
        }
    }

    #[test]
    fn help_keeps_the_descriptions() {
        let output = output();
        let options = options(&output);
        let (usage, description) = &options["--thinking-budget"];
        assert_eq!(usage, "--thinking-budget <tokens>");
        assert_eq!(
            description,
            "most tokens the model may think at a time before it is made to act, \
             or 0 for no limit (default: 2048)"
        );
        let (usage, description) = &options["--help"];
        assert_eq!(usage, "--help");
        assert_eq!(description, "display usage information");
    }

    #[test]
    fn help_fits_the_width() {
        let size = model::Size {
            context: 65536,
            model: 17_000_000_000,
            conversation: 3_000_000_000,
        };
        let text = layout(&output(), Some(size));
        for line in text.lines() {
            assert!(line.len() <= WIDTH, "too long: {line:?}");
        }
        assert!(text.starts_with(USAGE));
        assert!(text.contains("65536 on this machine"));
        assert!(text.contains("\nThe model:\n  --context <tokens>"));
    }
}
