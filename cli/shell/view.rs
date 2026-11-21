//! What the shell says, as lines and words: each of these takes values and
//! returns text, and draws nothing itself.

use std::{fmt::Write, path::Path, time::Duration};

use crossterm::style::Stylize;
use dwim::inference::{Loaded, OnDisk};
use dwim::model::{Device, Model, Size};
use dwim::providers::{Kind, hosted::Host};
use dwim::sys;
use dwim::tools::plan::{Plan, Step};

use crate::{
    model,
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
    lines.push("enter while it works changes its course · esc interrupts".to_string());
    lines.push("ctrl+o shows or hides the thought · ctrl+d quits".to_string());
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
    let weights = match Loaded::dir(name) {
        Ok(dir) => {
            let OnDisk { files, of, bytes } = Loaded::on_disk(&dir);
            match files {
                0 => "not fetched".to_string(),
                n if n == of => {
                    format!("{} in {}", model::bytes(bytes), tilde(&dir))
                }
                n => format!("{n}/{of} files, {} in {}", model::bytes(bytes), tilde(&dir)),
            }
        }
        Err(e) => e.to_string(),
    };
    let mut on = match (device, device_name) {
        (Device::Cpu, _) => "CPU".to_string(),
        (Device::Gpu(_), None) => "GPU".to_string(),
        (Device::Gpu(_), Some(name)) => format!("GPU · {name}"),
    };
    if let Some(size) = size {
        on.push_str(&format!(" · {}", model::bytes(size.model)));
    }
    let mut rows = vec![
        (
            "source",
            format!("{} at {}", Loaded::REPO, &Loaded::REVISION[..7]),
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

/// The model at `host`, for the `/model` command: what the host calls
/// it, and how much conversation the session has room for, as a table of
/// labels and values.
pub fn hosted_listing(host: &Host, model: Model, context: usize, columns: usize) -> Vec<Line> {
    let header = vec![span("● "), span(model.name().to_string()).bold()];
    let rows = vec![
        ("provider", host.name.to_string()),
        ("model", (host.model)(model).to_string()),
        ("context", format!("{context} tokens")),
    ];
    report(header, table(rows), columns)
}

/// Where the model runs, for the provider picked that the session is on.
pub fn provider_notice(provider: Kind, model: Model) -> String {
    match provider.host() {
        None => format!(
            "{} runs on this machine, in the dwim server · /server shows it",
            model.size()
        ),
        Some(host) => format!(
            "{} runs at {}, as {}",
            model.size(),
            host.name,
            (host.model)(model)
        ),
    }
}

/// What the shell says on starting where the model has nowhere to run:
/// why, under `reason`, and what to do about it, whether or not there is
/// a key to OpenRouter.
pub fn nowhere(reason: &str, key: bool, columns: usize) -> Vec<Line> {
    let mut reason = reason.to_string();
    if let Some(first) = reason.get(..1) {
        reason.replace_range(..1, &first.to_uppercase());
    }
    report(
        vec![span("● "), span("No model to run yet").bold()],
        vec![reason, connect_hint(key)],
        columns,
    )
}

/// What to do to have a model where none runs, whether or not there is a
/// key to OpenRouter.
pub fn connect_hint(key: bool) -> String {
    if key {
        "/model picks where the model runs · /provider enables more providers".to_string()
    } else {
        "/login signs in to OpenRouter, which runs the model for you".to_string()
    }
}

/// What a command that needs a model says while there is none, whether
/// or not there is a key to OpenRouter.
pub fn no_provider(key: bool) -> String {
    format!("No model to answer yet · {}", connect_hint(key))
}

/// How moving the conversation to another model is acknowledged.
pub fn switched(model: Model, provider: Kind) -> String {
    match provider.host() {
        None => format!("Switched to {model}, on this machine"),
        Some(host) => format!("Switched to {model}, at {}", host.name),
    }
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

/// Most steps of a plan shown over the input box: of a longer one, the
/// steps done are counted in a line, and the last of those to come.
const PLAN_STEPS: usize = 8;

/// Most of the choices a plan made that are shown over the input box.
const PLAN_ASSUMES: usize = 4;

/// Most cells of the bar that says how far along a plan is.
const PLAN_BAR: usize = 16;

/// The plan of the work over the input box: a heading with a bar that
/// says how far along it is, then the steps, a line each, marked done,
/// under way or to come, with what a step changed and whether that was
/// checked at the right, then the choices the model made where the request
/// left them open. Under the step under way goes `doing`, what the model
/// is at. The plan of a model not `working` is paused, for the user to
/// take up or set aside. A `stale` plan is from before the user's last
/// message, which the model has yet to answer: it is greyed out, and its
/// heading says so.
pub fn plan(
    plan: &Plan,
    doing: Option<&str>,
    working: bool,
    stale: bool,
    columns: usize,
) -> Vec<Line> {
    let steps = plan.steps();
    let under_way = plan.under_way();
    let doing = doing.filter(|_| !stale);
    let mut heading = vec![span("● ").cyan(), span("Plan ").bold()];
    heading.extend(bar(plan));
    heading.push(span(format!(" {} of {}", plan.done(), steps.len())).dark_grey());
    if stale {
        heading.push(span(" · from before your message").yellow());
    } else if !working {
        heading.push(span(" · paused · esc sets it aside").dark_grey());
    }
    let mut lines = vec![tui::truncate(&heading, columns)];
    // Of a long plan, the steps done but the last are counted, and the
    // steps to come past the next few.
    let long = steps.len() > PLAN_STEPS;
    let first = if long {
        plan.done().saturating_sub(1)
    } else {
        0
    };
    let end = if long {
        (first + PLAN_STEPS).min(steps.len())
    } else {
        steps.len()
    };
    if first > 0 {
        lines.push(vec![span(format!("  ✓ {first} earlier steps")).dark_grey()]);
    }
    for (i, step) in steps.iter().enumerate().take(end).skip(first) {
        let current = under_way == Some(i);
        // What the step rests on goes at the right, and the step is cut
        // to leave it room.
        let rests = rests_on(step, current);
        let room = columns
            .saturating_sub(4 + tui::width(&rests) + if rests.is_empty() { 0 } else { 2 })
            .max(8);
        let text = tui::fit(&plain(&step.text), room);
        let pad = columns.saturating_sub(4 + tui::width(&vec![span(&text)]) + tui::width(&rests));
        let mut line = match (step.done, current) {
            _ if stale => {
                let mark = if step.done { "  ✓ " } else { "  ○ " };
                vec![span(mark).dark_grey(), span(text).dark_grey()]
            }
            (true, _) => vec![span("  ✓ ").green(), span(text).dark_grey()],
            (false, true) => vec![span("  ◆ ").cyan(), span(text).bold()],
            (false, false) => vec![span("  ○ ").dark_grey(), span(text)],
        };
        if !rests.is_empty() {
            line.push(span(" ".repeat(pad)));
            line.extend(rests);
        }
        lines.push(tui::truncate(&line, columns));
        if current && let Some(doing) = doing {
            let doing = tui::fit(doing, columns.saturating_sub(6));
            lines.push(vec![span(format!("    ⎿ {doing}")).dark_grey()]);
        }
    }
    if end < steps.len() {
        lines.push(vec![
            span(format!("  ○ {} more", steps.len() - end)).dark_grey(),
        ]);
    }
    let assumes = plan.assumes();
    if !assumes.is_empty() {
        lines.push(Line::new());
        lines.push(vec![span("  Assumptions").dark_grey().bold()]);
        for assume in assumes.iter().take(PLAN_ASSUMES) {
            let assume = tui::fit(&plain(assume), columns.saturating_sub(4));
            lines.push(vec![span(format!("  · {assume}")).dark_grey()]);
        }
        if assumes.len() > PLAN_ASSUMES {
            lines.push(vec![
                span(format!("  · {} more", assumes.len() - PLAN_ASSUMES)).dark_grey(),
            ]);
        }
    }
    lines
}

/// What stands over the input box while the model works with no plan
/// laid out yet: word that none is, and `doing`, what the model is at.
pub fn no_plan(doing: Option<&str>, columns: usize) -> Vec<Line> {
    let mut lines = vec![vec![
        span("● ").cyan(),
        span("Plan").bold(),
        span(" · working out the steps").dark_grey(),
    ]];
    if let Some(doing) = doing {
        let doing = tui::fit(doing, columns.saturating_sub(6));
        lines.push(vec![span(format!("    ⎿ {doing}")).dark_grey()]);
    }
    lines
}

/// What goes into the conversation as the plan changes from `old` to
/// `new`: the steps when it is laid out or they change, the choices made
/// when they are new, and each step as it is finished, with what it came
/// to and what it rests on.
pub fn plan_news(old: &Plan, new: &Plan, columns: usize) -> Vec<Line> {
    let mut lines = Vec::new();
    if new.is_empty() {
        return lines;
    }
    let texts = |plan: &Plan| -> Vec<String> {
        plan.steps().iter().map(|step| step.text.clone()).collect()
    };
    // A paragraph under `first`, a prefix, wrapped to the terminal and
    // dimmed.
    let under = |lines: &mut Vec<Line>, first: &str, text: &str| {
        let indent = " ".repeat(tui::width(&vec![span(first)]));
        for (i, text) in tui::wrap(text, columns.saturating_sub(indent.len()))
            .into_iter()
            .enumerate()
        {
            let prefix = if i == 0 { first } else { &indent };
            lines.push(vec![span(format!("{prefix}{text}")).dark_grey()]);
        }
    };
    if texts(old) != texts(new) {
        lines.push(vec![
            span("● ").cyan(),
            span(if old.is_empty() {
                "Plan"
            } else {
                "Plan changed"
            })
            .bold(),
        ]);
        for (i, step) in new
            .steps()
            .iter()
            .enumerate()
            .filter(|(_, step)| !step.done)
        {
            under(&mut lines, &format!("  {}. ", i + 1), &plain(&step.text));
        }
        lines.push(Line::new());
    }
    if old.assumes() != new.assumes() && !new.assumes().is_empty() {
        lines.push(vec![span("● ").cyan(), span("Assumptions").bold()]);
        for assume in new.assumes() {
            under(&mut lines, "  · ", &plain(assume));
        }
        lines.push(Line::new());
    }
    for step in new.steps().iter().take(new.done()).skip(old.done()) {
        for (i, text) in tui::wrap(&plain(&step.text), columns.saturating_sub(2))
            .into_iter()
            .enumerate()
        {
            let mark = if i == 0 {
                span("✓ ").green()
            } else {
                span("  ")
            };
            lines.push(vec![mark, span(text).bold()]);
        }
        if !step.result.is_empty() {
            under(&mut lines, "  ⎿ ", &step.result);
        }
        if !step.changed.is_empty() {
            let files = match step.changed.len() {
                1..=3 => step.changed.join(", "),
                n => format!("{n} files"),
            };
            let mut line = vec![span(format!("  ⎿ {files} · ")).dark_grey()];
            match &step.check {
                Some(check) => {
                    line.push(span(tui::fit(&check.command, 60)).dark_grey());
                    line.push(if check.passed {
                        span(" ✓").green()
                    } else {
                        span(" ✗ failed").red()
                    });
                }
                None => line.push(span("unchecked").yellow()),
            }
            lines.push(tui::truncate(&line, columns));
        }
        lines.push(Line::new());
    }
    lines
}

/// The bar that says how far along `plan` is: a cell a step, or fewer for
/// a long plan, filled for the steps done.
fn bar(plan: &Plan) -> Line {
    let steps = plan.steps().len();
    let cells = steps.min(PLAN_BAR);
    let filled = (plan.done() * cells).checked_div(steps).unwrap_or(0);
    vec![
        span("▰".repeat(filled)).cyan(),
        span("▱".repeat(cells - filled)).dark_grey(),
    ]
}

/// What a step of the plan rests on, in a few words, to go at the right
/// of its line: the file it changed, or how many, and a mark for whether
/// the command run since succeeded. A step done whose change had nothing
/// run since is unchecked; of the step under way, `current`, that is not
/// said, since it is not over. Nothing for a step that changed nothing.
fn rests_on(step: &Step, current: bool) -> Line {
    let files = match step.changed.as_slice() {
        [] => return Line::new(),
        [one] => one.rsplit('/').next().unwrap_or(one).to_string(),
        many => format!("{} files", many.len()),
    };
    let mut line = vec![span(files).dark_grey()];
    match &step.check {
        Some(check) if check.passed => line.push(span(" ✓").green()),
        Some(_) => line.push(span(" ✗").red()),
        None if current => {}
        None => line.push(span(" unchecked").yellow()),
    }
    line
}

/// `text` as it reads on a line of the plan: without the backticks a
/// model puts around names, and on one line.
fn plain(text: &str) -> String {
    text.replace('`', "")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
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
    match sys::dirs::home().and_then(|home| path.strip_prefix(home).ok()) {
        Some(rest) if rest.as_os_str().is_empty() => "~".to_string(),
        Some(rest) => format!("~/{}", rest.display()),
        None => path.display().to_string(),
    }
}
