//! The agent around the model: runs the tools it calls and feeds the
//! results back, until it replies with text alone.
//!
//! A [`Harness`] wraps a [`Chat`] and reports each turn's thoughts, text,
//! tool calls, and tool output as [`Event`]s, so a user interface can show
//! them as they happen. The one tool is `bash`, which runs a shell command,
//! and the system prompt pushes the model to use it rather than answer from
//! memory or ask the user for a command.

use std::{error::Error, ops::ControlFlow, process::Command};

use hack_models::{Chat, Chunk, Device, ToolCall};

/// Most of a tool's output that goes back to the model, so that a chatty
/// command can't fill the context window.
const MAX_OUTPUT: usize = 2000;

/// What the model gets back when it makes the same call twice in a row,
/// instead of running it again: nothing ran in between to change its output,
/// and small models otherwise tend to repeat a call over and over.
const REPEATED: &str = "error: you just ran this, and its output is above. Don't run it again: use that output, run something else, or reply to the user.";

/// The system prompt declaring the tools, in the form Qwen3's chat template
/// puts them.
pub const SYSTEM: &str = r#"You are hack, a coding agent working in the user's project directory at a Unix command line. You have a bash tool that runs shell commands there, and you may use it at any time without asking.

- For anything about the project, its files, its git history, or the system, run commands to find out before you answer. Don't answer from memory when a command can tell you.
- Never say you can't access files or run commands, and never ask the user which command to run: pick one yourself.
- When a request could be a question or a task, treat it as a task and do it.
- To change something, run the commands that change it instead of explaining how.
- If a command fails, read the error and try another way.
- Keep going until the request is done, then reply in a few sentences with what you found or did.

For example, for "review commit abc123", run `git show abc123` and point out bugs and risks in the change; for "what files are here?", run `ls`; for "what time is it?", run `date`.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "bash", "description": "Run a shell command and return its output.", "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "The command to run."}}, "required": ["command"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"#;

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
    if text.len() > MAX_OUTPUT {
        let end = (0..=MAX_OUTPUT).rev().find(|&i| text.is_char_boundary(i)).unwrap_or(0);
        text.truncate(end);
        text.push_str("…\n");
    }
    if text.is_empty() {
        text.push_str("(no output)");
    }
    text.trim_end().to_string()
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
    fn describes_calls() {
        assert_eq!(describe(&call(r#"{"name": "bash", "arguments": {"command": "ls -l"}}"#)), "ls -l");
        assert_eq!(describe(&call(r#"{"name": "other", "arguments": {"x": 1}}"#)), r#"{"x":1}"#);
    }
}
