//! The tools the model can call: `bash`, which runs a shell command, and
//! `read`, which gives it a file a page at a time. Each declares itself
//! as the JSON signature the model's chat template lists, and turns a
//! call into the text the model sees.

pub mod bash;
pub mod read;

use dwim_models::ToolCall;
use serde_json::Value;

use crate::bash::Bash;

/// The tools' signatures, one per line, as the system prompt lists them.
pub fn signatures() -> String {
    [bash::SIGNATURE, read::SIGNATURE].join("\n")
}

/// The tools, with what they keep between calls.
pub struct Tools {
    bash: Bash,
}

impl Default for Tools {
    fn default() -> Self {
        Self::new()
    }
}

impl Tools {
    pub fn new() -> Self {
        Self { bash: Bash::new() }
    }

    /// Runs a call, returning what the model should see.
    pub fn run(&mut self, call: &ToolCall) -> String {
        match call.name.as_str() {
            "bash" => self.bash.run(&call.arguments),
            "read" => read::run(&call.arguments),
            name => format!("error: unknown tool '{name}'"),
        }
    }
}

/// How a call reads on screen: the command for `bash`, the file and where
/// in it for `read`, and the arguments as written for anything else.
pub fn describe(call: &ToolCall) -> String {
    match call.name.as_str() {
        "bash" => string(&call.arguments, "command").map(str::to_string),
        "read" => string(&call.arguments, "path").map(|path| match read::start(&call.arguments) {
            Some(start) if start > 1 => format!("{path} from line {start}"),
            _ => path.to_string(),
        }),
        _ => None,
    }
    .unwrap_or_else(|| call.arguments.to_string())
}

/// The string argument `key`, if there is one.
fn string<'a>(arguments: &'a Value, key: &str) -> Option<&'a str> {
    arguments.get(key)?.as_str()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(text: &str) -> ToolCall {
        ToolCall::parse(text).unwrap()
    }

    #[test]
    fn runs_the_tool_named() {
        let mut tools = Tools::new();
        assert_eq!(tools.run(&call(r#"{"name": "bash", "arguments": {"command": "echo hi"}}"#)), "hi\n[exit code 0]");
        let read = tools.run(&call(r#"{"name": "read", "arguments": {"path": "Cargo.toml"}}"#));
        assert!(read.starts_with("1\t[package]\n"), "{read}");
        assert!(tools.run(&call(r#"{"name": "rm", "arguments": {}}"#)).starts_with("error: unknown tool"));
        assert!(tools.run(&call(r#"{"name": "bash", "arguments": {}}"#)).starts_with("error: bash needs"));
        assert!(tools.run(&call(r#"{"name": "read", "arguments": {}}"#)).starts_with("error: read needs"));
    }

    #[test]
    fn describes_calls() {
        assert_eq!(describe(&call(r#"{"name": "bash", "arguments": {"command": "ls -l"}}"#)), "ls -l");
        assert_eq!(describe(&call(r#"{"name": "read", "arguments": {"path": "a.rs"}}"#)), "a.rs");
        assert_eq!(describe(&call(r#"{"name": "read", "arguments": {"path": "a.rs", "start": 201}}"#)), "a.rs from line 201");
        assert_eq!(describe(&call("<function=read>\n<parameter=path>\na.rs\n</parameter>\n<parameter=start>\n201\n</parameter>\n</function>")), "a.rs from line 201");
        assert_eq!(describe(&call(r#"{"name": "other", "arguments": {"x": 1}}"#)), r#"{"x":1}"#);
    }

    #[test]
    fn lists_both_tools() {
        let signatures = signatures();
        assert_eq!(signatures.lines().count(), 2);
        for line in signatures.lines() {
            let signature: Value = serde_json::from_str(line).unwrap();
            assert_eq!(signature["type"], "function");
            assert!(signature["function"]["parameters"]["required"].is_array());
        }
    }
}
