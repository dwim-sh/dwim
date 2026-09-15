//! The agent around the model: runs the tools it calls and feeds the
//! results back, until it replies with text alone.
//!
//! A [`Harness`] wraps a [`Chat`] and reports each turn's thoughts, text,
//! tool calls, and tool output as [`Event`]s, so a user interface can show
//! them as they happen. The one tool is `bash`, which runs a shell command,
//! and the system prompt pushes the model to use it rather than answer from
//! memory or ask the user for a command. It also tells the model about the
//! project it works in, since a small model won't go looking on its own.

use std::{
    error::Error,
    fs,
    ops::ControlFlow,
    path::Path,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

use hack_models::{Chat, Chunk, Device, ToolCall};

/// Most of a tool's output that goes back to the model, so that a chatty
/// command can't fill the context window.
const MAX_OUTPUT: usize = 2000;

/// Most of a project's `AGENTS.md` that goes into the system prompt.
const MAX_INSTRUCTIONS: usize = 4000;

/// Most of the files in the working directory the system prompt lists.
const MAX_FILES: usize = 50;

/// What the model gets back when it makes the same call twice in a row,
/// instead of running it again: nothing ran in between to change its output,
/// and small models otherwise tend to repeat a call over and over.
const REPEATED: &str = "error: you just ran this, and its output is above. Don't run it again: use that output, run something else, or reply to the user.";

/// How the agent should behave: the start of the system prompt.
const INSTRUCTIONS: &str = r#"You are hack, a coding agent working in the user's project directory at a Unix command line. You have a bash tool that runs shell commands there, and you may use it at any time without asking.

- For anything about the project, its files, its git history, or the system, run commands to find out before you answer. Don't answer from memory when a command can tell you.
- Never say you can't access files or run commands, and never ask the user which command to run: pick one yourself.
- When a request could be a question or a task, treat it as a task and do it.
- To change something, run the commands that change it instead of explaining how.
- If a command fails, read the error and try another way.
- Keep going until the request is done, then reply in a few sentences with what you found or did.

For example, for "review commit abc123", run `git show abc123` and point out bugs and risks in the change; for "what files are here?", run `ls`; for "what time is it?", run `date`."#;

/// The tools, declared in the form Qwen3's chat template puts them: the end
/// of the system prompt.
const TOOLS: &str = r#"# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "bash", "description": "Run a shell command and return its output.", "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "The command to run."}}, "required": ["command"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"#;

/// The system prompt for an agent working in `dir`: how to behave, where it
/// is and what the project looks like, the project's own instructions from
/// its `AGENTS.md` if it has one, and the tools.
pub fn system_prompt(dir: &Path) -> String {
    let mut prompt = format!("{INSTRUCTIONS}\n\n{}", environment(dir));
    if let Ok(instructions) = fs::read_to_string(dir.join("AGENTS.md")) {
        let instructions = truncate(instructions.trim(), MAX_INSTRUCTIONS);
        prompt.push_str(&format!("\n\n# Project instructions\n\nFrom AGENTS.md:\n\n{instructions}"));
    }
    prompt.push_str("\n\n");
    prompt.push_str(TOOLS);
    prompt
}

/// Where the agent is: the working directory and the files in it, whether
/// it is a git repository, the platform, and the date.
fn environment(dir: &Path) -> String {
    let branch = Command::new("git")
        .args(["branch", "--show-current"])
        .current_dir(dir)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string());
    let git = match branch {
        Some(branch) if !branch.is_empty() => format!("yes, on branch {branch}"),
        Some(_) => "yes".to_string(),
        None => "no".to_string(),
    };
    let platform = match std::env::consts::OS {
        "macos" => "macOS",
        "linux" => "Linux",
        os => os,
    };
    let days = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |since| since.as_secs() / 86400);
    format!(
        "# Environment\n\nWorking directory: {}\nFiles: {}\nGit repository: {git}\nPlatform: {platform}\nDate: {}",
        dir.display(),
        files(dir),
        date(days as i64),
    )
}

/// The files and directories in `dir`, directories with a trailing slash,
/// leaving out hidden ones.
fn files(dir: &Path) -> String {
    let Ok(entries) = fs::read_dir(dir) else {
        return String::new();
    };
    let mut names: Vec<String> = entries
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name().into_string().ok()?;
            match entry.file_type().ok()? {
                _ if name.starts_with('.') => None,
                kind if kind.is_dir() => Some(format!("{name}/")),
                _ => Some(name),
            }
        })
        .collect();
    names.sort();
    if names.len() > MAX_FILES {
        names.truncate(MAX_FILES);
        names.push("…".to_string());
    }
    names.join(" ")
}

/// The date `days` after 1970-01-01, as year-month-day.
fn date(days: i64) -> String {
    // Howard Hinnant's civil_from_days: count in 400-year eras of 146097
    // days, from a year that starts in March so that leap days come last.
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let day_of_era = z.rem_euclid(146097);
    let year_of_era = (day_of_era - day_of_era / 1460 + day_of_era / 36524 - day_of_era / 146096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_index = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_index + 2) / 5 + 1;
    let month = if month_index < 10 { month_index + 3 } else { month_index - 9 };
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    format!("{year:04}-{month:02}-{day:02}")
}

/// Cuts `text` down to at most `max` bytes, on a character boundary, marking
/// the cut with an ellipsis.
fn truncate(text: &str, max: usize) -> String {
    if text.len() <= max {
        return text.to_string();
    }
    let end = (0..=max).rev().find(|&i| text.is_char_boundary(i)).unwrap_or(0);
    format!("{}…", &text[..end])
}

/// What happens during a turn, as it happens.
pub enum Event<'a> {
    /// Part of the model's thought, before it replies.
    Thought(&'a str),
    /// Text of the reply.
    Text(&'a str),
    /// A tool is about to run: its name and how it was called.
    Call { name: &'a str, detail: &'a str },
    /// What the tool returned.
    Output(&'a str),
}

/// The loop around a [`Chat`] that runs the tools the model calls and feeds
/// the results back, until the model replies with text alone.
pub struct Harness<D: Device> {
    chat: Chat<D>,
}

impl<D: Device> Harness<D> {
    pub fn new(chat: Chat<D>) -> Self {
        Self { chat }
    }

    /// Number of tokens in the conversation so far.
    pub fn tokens(&self) -> usize {
        self.chat.tokens()
    }

    /// Sends a message from the user, reporting the reply and any tool
    /// calls it makes to `on_event`. The turn ends early if `on_event`
    /// breaks.
    pub fn send(
        &mut self,
        message: &str,
        mut on_event: impl FnMut(Event) -> ControlFlow<()>,
    ) -> Result<(), Box<dyn Error>> {
        let mut calls = self.chat.send(message, |chunk| on_event(event(chunk)))?;
        // The last call run, as its name and arguments.
        let mut last = None;
        while !calls.is_empty() {
            let mut outputs = Vec::new();
            for call in &calls {
                let output = match ToolCall::parse(call) {
                    Ok(call) => {
                        let detail = describe(&call);
                        if on_event(Event::Call {
                            name: &call.name,
                            detail: &detail,
                        })
                        .is_break()
                        {
                            return Ok(());
                        }
                        let this = Some((call.name.clone(), call.arguments.to_string()));
                        if this == last {
                            REPEATED.to_string()
                        } else {
                            last = this;
                            run(&call)
                        }
                    }
                    Err(e) => format!("error: malformed tool call: {e}"),
                };
                if on_event(Event::Output(&output)).is_break() {
                    return Ok(());
                }
                outputs.push(output);
            }
            calls = self.chat.respond(&outputs, |chunk| on_event(event(chunk)))?;
        }
        Ok(())
    }
}

/// The event for a piece of the model's reply.
fn event(chunk: Chunk) -> Event {
    match chunk {
        Chunk::Thought(text) => Event::Thought(text),
        Chunk::Text(text) => Event::Text(text),
    }
}

/// How a call reads on screen: the command for `bash`, and the arguments
/// as written for anything else.
fn describe(call: &ToolCall) -> String {
    match call.arguments.get("command").and_then(|command| command.as_str()) {
        Some(command) if call.name == "bash" => command.to_string(),
        _ => call.arguments.to_string(),
    }
}

/// Runs a tool call, returning what the model should see.
fn run(call: &ToolCall) -> String {
    if call.name != "bash" {
        return format!("error: unknown tool '{}'", call.name);
    }
    let Some(command) = call.arguments.get("command").and_then(|command| command.as_str()) else {
        return "error: bash needs a 'command' string".to_string();
    };
    let output = match Command::new("sh").args(["-c", command]).output() {
        Ok(output) => output,
        Err(e) => return format!("error: {e}"),
    };
    let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&output.stderr));
    if !output.status.success() {
        text.push_str(&format!("({})\n", output.status));
    }
    let text = truncate(text.trim_end(), MAX_OUTPUT);
    if text.is_empty() {
        return "(no output)".to_string();
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(json: &str) -> ToolCall {
        ToolCall::parse(json).unwrap()
    }

    #[test]
    fn runs_bash() {
        let output = run(&call(r#"{"name": "bash", "arguments": {"command": "echo hi; echo err >&2"}}"#));
        assert_eq!(output, "hi\nerr");
        let output = run(&call(r#"{"name": "bash", "arguments": {"command": "exit 3"}}"#));
        assert_eq!(output, "(exit status: 3)");
        let output = run(&call(r#"{"name": "bash", "arguments": {"command": "true"}}"#));
        assert_eq!(output, "(no output)");
    }

    #[test]
    fn caps_output() {
        let output = run(&call(r#"{"name": "bash", "arguments": {"command": "yes ä | head -c 5000"}}"#));
        assert!(output.len() <= MAX_OUTPUT + "…".len());
        assert!(output.ends_with('…'));
    }

    #[test]
    fn refuses_what_it_does_not_know() {
        assert!(run(&call(r#"{"name": "rm", "arguments": {}}"#)).starts_with("error: unknown tool"));
        assert!(run(&call(r#"{"name": "bash", "arguments": {}}"#)).starts_with("error: bash needs"));
        assert!(ToolCall::parse("not json").is_err());
    }

    #[test]
    fn dates() {
        assert_eq!(date(0), "1970-01-01");
        assert_eq!(date(11016), "2000-02-29");
        assert_eq!(date(19782), "2024-02-29");
        assert_eq!(date(20711), "2026-09-15");
    }

    #[test]
    fn describes_the_project() {
        let prompt = system_prompt(Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap());
        assert!(prompt.starts_with(INSTRUCTIONS));
        assert!(prompt.ends_with(TOOLS));
        let files = prompt.lines().find_map(|line| line.strip_prefix("Files: ")).unwrap();
        assert!(files.split(' ').any(|file| file == "Cargo.toml"));
        assert!(files.split(' ').any(|file| file == "harness/"));
        assert!(!files.contains(".git"));
        assert!(prompt.contains("From AGENTS.md:\n\n# Hack"));
    }

    #[test]
    fn describes_calls() {
        assert_eq!(describe(&call(r#"{"name": "bash", "arguments": {"command": "ls -l"}}"#)), "ls -l");
        assert_eq!(describe(&call(r#"{"name": "other", "arguments": {"x": 1}}"#)), r#"{"x":1}"#);
    }
}
