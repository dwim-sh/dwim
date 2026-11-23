//! The `--help` text: argh's description of the options, sorted under
//! headings and wrapped to the terminal's width.

use std::collections::BTreeMap;

use dwim::{
    inference::Loaded,
    model::{Device, Size},
};

use super::device;
use crate::{tui, units};

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
            units::bytes(size.conversation),
            units::bytes(size.model),
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
