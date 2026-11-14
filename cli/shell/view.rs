//! What the shell says, as lines and words: each of these takes values and
//! returns text, and draws nothing itself.

use std::{fmt::Write, path::Path, time::Duration};

use crossterm::style::Stylize;
use dwim::gpu::BACKEND;
use dwim::models::qwen35;

use crate::{
    model::{self, Size},
    opts::Device,
    shell::cmd::COMMANDS,
    tui::{self, Line, span},
};

/// What the `/help` command says: the commands, and the keys.
pub fn help(columns: usize) -> Vec<Line> {
    let width = COMMANDS
        .iter()
        .map(|(name, _)| name.len())
        .max()
        .unwrap_or(0)
        + 2;
    let mut lines: Vec<String> = COMMANDS
        .iter()
        .map(|(name, what)| format!("{name:<width$}{what}"))
        .collect();
    lines.push("ctrl+o shows or hides the thought · esc interrupts · ctrl+d quits".to_string());
    report(vec![span("● "), span("Commands").bold()], lines, columns)
}

/// The model, for the `/model` command: its source and its weights on
/// disk, the device it runs on and the memory the model takes there, and
/// how much conversation it has room for and the memory that takes, as a
/// table of labels and values.
pub fn model_listing(
    name: &str,
    device: Device,
    device_name: Option<&str>,
    size: Option<&Size>,
    columns: usize,
) -> Vec<Line> {
    let header = vec![span("● "), span(name.to_string()).bold()];
    let weights = match model::locate(name) {
        Ok(dir) => {
            let (files, bytes) = model::on_disk(&dir);
            match files {
                0 => "not fetched".to_string(),
                n if n == qwen35::FILES.len() => {
                    format!("{} in {}", model::bytes(bytes), tilde(&dir))
                }
                n => format!(
                    "{n}/{} files, {} in {}",
                    qwen35::FILES.len(),
                    model::bytes(bytes),
                    tilde(&dir)
                ),
            }
        }
        Err(e) => e.to_string(),
    };
    let mut on = match (device, device_name) {
        (Device::Cpu, _) => "CPU".to_string(),
        (Device::Gpu, None) => "GPU".to_string(),
        (Device::Gpu, Some(gpu)) => format!("GPU · {gpu} via {BACKEND}"),
    };
    if let Some(size) = size {
        on.push_str(&format!(" · {}", model::bytes(size.model)));
    }
    let mut rows = vec![
        (
            "source",
            format!("{} at {}", qwen35::REPO, &qwen35::REVISION[..7]),
        ),
        ("weights", weights),
        ("device", on),
    ];
    if let Some(size) = size {
        rows.push((
            "context",
            format!(
                "{} tokens ({})",
                size.context,
                model::bytes(size.conversation)
            ),
        ));
        rows.push((
            "sessions",
            format!("room for {} at once, sharing the model", size.sessions),
        ));
    }
    report(header, table(rows), columns)
}

/// How a download is going: how much of the file has arrived, out of how
/// much, at what speed, and the time left once the file's size is known.
/// The fields are padded to fixed widths, as the status line's are.
pub fn download_details(done: u64, total: Option<u64>, elapsed: Duration) -> String {
    let mut details = format!("{:>8}", model::bytes(done));
    if let Some(total) = total {
        details.push_str(&format!(
            " / {:>8} ({:>3}%)",
            model::bytes(total),
            done * 100 / total.max(1)
        ));
    }
    let rate = rate(done as f64, elapsed);
    let speed = rate.map_or(String::new(), |rate| {
        format!("{}/s", model::bytes(rate as u64))
    });
    write!(details, " · {speed:>10}").expect("a String takes any write");
    if let Some(total) = total {
        let left = match rate {
            Some(rate) if done < total => {
                let left = ((total - done) as f64 / rate).round() as u64;
                match left {
                    0..60 => format!("{left}s"),
                    60..3600 => format!("{}m {:02}s", left / 60, left % 60),
                    _ => format!("{}h {:02}m", left / 3600, left % 3600 / 60),
                }
            }
            _ => String::new(),
        };
        write!(details, " · {left:>7} left").expect("a String takes any write");
    }
    details
}

/// How the thinking budget reads when asked.
pub fn budget_notice(tokens: Option<usize>) -> String {
    match tokens {
        Some(tokens) => format!("Thinking budget: {tokens} tokens"),
        None => "No thinking budget".to_string(),
    }
}

/// How setting the thinking budget is acknowledged.
pub fn budget_set(tokens: Option<usize>) -> String {
    match tokens {
        Some(tokens) => format!("Thinking budget set to {tokens} tokens"),
        None => "Thinking budget lifted".to_string(),
    }
}

/// What the `/auto` command says of auto mode, and how setting it is
/// acknowledged.
pub fn auto_notice(auto: bool) -> String {
    if auto {
        "Auto mode on: the model goes on by itself until it says it is done".to_string()
    } else {
        "Auto mode off: the model asks when it has something to ask".to_string()
    }
}

/// One line said in place of a reply.
pub fn notice(text: impl Into<String>) -> Vec<Line> {
    vec![vec![span(format!("● {}", text.into()))]]
}

/// A report: its header, then its details under `⎿`, dimmed and wrapped to
/// the terminal's `columns`, the way a tool's output follows its call.
pub fn report(
    header: Line,
    details: impl IntoIterator<Item = String>,
    columns: usize,
) -> Vec<Line> {
    let mut lines = vec![header];
    let mut first = true;
    for detail in details {
        for text in tui::wrap(&detail, columns.saturating_sub(4)) {
            let prefix = if first { "  ⎿ " } else { "    " };
            first = false;
            lines.push(vec![span(format!("{prefix}{text}")).dark_grey()]);
        }
    }
    lines
}

/// Rows of a label and a value as lines, with the values aligned.
fn table(rows: Vec<(&str, String)>) -> Vec<String> {
    let width = rows.iter().map(|(label, _)| label.len()).max().unwrap_or(0) + 2;
    rows.into_iter()
        .map(|(label, value)| format!("{label:<width$}{value}"))
        .collect()
}

/// What part of the reply is being generated.
#[derive(Clone, Copy, PartialEq)]
pub enum Segment {
    /// The model's thought, before it replies.
    Thought,
    /// The reply itself.
    Text,
    /// The note the model goes on from after the conversation is compacted.
    Note,
}

/// A line of a part of the reply, the first one marked with a bullet. A
/// thought is dimmed, so the reply stands out from it, and so is the note
/// the model goes on from after a compaction.
pub fn reply_line(segment: Segment, (i, text): (usize, &str)) -> Line {
    match segment {
        Segment::Thought | Segment::Note => {
            let bullet = if i == 0 { span("✻ ") } else { span("  ") };
            vec![bullet.dark_grey(), span(text).dark_grey().italic()]
        }
        Segment::Text => {
            let bullet = if i == 0 { span("● ") } else { span("  ") };
            vec![bullet, span(text)]
        }
    }
}

/// How fast `done` of something went in `elapsed`, per second; nothing
/// before a second has passed, or before anything was done, since it is
/// too early to tell.
pub fn rate(done: f64, elapsed: Duration) -> Option<f64> {
    (elapsed >= Duration::from_secs(1) && done > 0.0).then(|| done / elapsed.as_secs_f64())
}

/// A path, with the home directory as `~`.
pub fn tilde(path: &Path) -> String {
    match dirs::home_dir().and_then(|home| path.strip_prefix(home).ok()) {
        Some(rest) if rest.as_os_str().is_empty() => "~".to_string(),
        Some(rest) => format!("~/{}", rest.display()),
        None => path.display().to_string(),
    }
}
