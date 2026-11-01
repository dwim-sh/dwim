//! The tools the model can call: `bash`, which runs a shell command,
//! `read`, which gives it a file a page at a time, and `edit`, which
//! replaces a string in a file with another. Each declares itself
//! as the JSON signature the model's chat template lists, and turns a
//! call into the text the model sees.

pub mod bash;
pub mod call;
pub mod edit;
pub mod read;

use serde_json::Value;

use crate::bash::Bash;
pub use crate::call::ToolCall;

/// The tools' signatures, one per line, as the system prompt lists them.
pub fn signatures() -> String {
    [bash::SIGNATURE, read::SIGNATURE, edit::SIGNATURE].join("\n")
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
            "edit" => edit::run(&call.arguments),
            name => format!("error: unknown tool '{name}'"),
        }
    }
}

/// How a call reads on screen: the command for `bash`, the file and where
/// in it for `read` and `edit`, and the arguments as written for anything
/// else, or for a read the tool will refuse.
pub fn describe(call: &ToolCall) -> String {
    match call.name.as_str() {
        "bash" => string(&call.arguments, "command").map(str::to_string),
        "read" => string(&call.arguments, "path").and_then(|path| {
            let (start, lines) = read::range(&call.arguments).ok()?;
            Some(match (start, lines) {
                (Some(start), Some(1)) => format!("{path} line {start}"),
                (Some(start), Some(lines)) => {
                    let end = start.saturating_add(lines as u64 - 1);
                    format!("{path} lines {start}–{end}")
                }
                (Some(start), None) if start > 1 => format!("{path} from line {start}"),
                _ => path.to_string(),
            })
        }),
        "edit" => {
            string(&call.arguments, "path").map(|path| match number(&call.arguments, "start") {
                Some(start) => format!("{path} at line {start}"),
                None => path.to_string(),
            })
        }
        _ => None,
    }
    .unwrap_or_else(|| call.arguments.to_string())
}

/// The argument `key` as a number, written as a number or, as the
/// `<function=…>` call format writes every argument, as a string.
fn number(arguments: &Value, key: &str) -> Option<u64> {
    match arguments.get(key)? {
        Value::Number(n) => n.as_u64(),
        Value::String(s) => s.trim().parse().ok(),
        _ => None,
    }
}

/// `word`, or its plural if `n` is not one.
fn plural(n: u64, word: &str) -> String {
    if n == 1 {
        word.to_string()
    } else {
        format!("{word}s")
    }
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
        assert_eq!(
            tools.run(&call(
                r#"{"name": "bash", "arguments": {"command": "echo hi"}}"#
            )),
            "hi\n[exit code 0]"
        );
        let read = tools.run(&call(
            r#"{"name": "read", "arguments": {"path": "Cargo.toml"}}"#,
        ));
        assert!(read.starts_with("1\t[package]\n"), "{read}");
        assert!(
            tools
                .run(&call(r#"{"name": "rm", "arguments": {}}"#))
                .starts_with("error: unknown tool")
        );
        assert!(
            tools
                .run(&call(r#"{"name": "bash", "arguments": {}}"#))
                .starts_with("error: bash needs")
        );
        assert!(
            tools
                .run(&call(r#"{"name": "read", "arguments": {}}"#))
                .starts_with("error: read needs")
        );
    }

    #[test]
    fn describes_calls() {
        assert_eq!(
            describe(&call(
                r#"{"name": "bash", "arguments": {"command": "ls -l"}}"#
            )),
            "ls -l"
        );
        assert_eq!(
            describe(&call(r#"{"name": "read", "arguments": {"path": "a.rs"}}"#)),
            "a.rs"
        );
        assert_eq!(
            describe(&call(
                r#"{"name": "read", "arguments": {"path": "a.rs", "start": 201}}"#
            )),
            "a.rs from line 201"
        );
        assert_eq!(
            describe(&call(
                "<function=read>\n<parameter=path>\na.rs\n</parameter>\n<parameter=start>\n201\n</parameter>\n</function>"
            )),
            "a.rs from line 201"
        );
        assert_eq!(
            describe(&call(
                r#"{"name": "read", "arguments": {"path": "a.rs", "start": 440, "lines": 30}}"#
            )),
            "a.rs lines 440–469"
        );
        assert_eq!(
            describe(&call(
                r#"{"name": "read", "arguments": {"path": "a.rs", "lines": "1"}}"#
            )),
            "a.rs line 1"
        );
        // As many lines as the tool shows, and the arguments as written
        // for a read it refuses.
        assert_eq!(
            describe(&call(
                r#"{"name": "read", "arguments": {"path": "a.rs", "start": 440, "lines": 1000}}"#
            )),
            format!("a.rs lines 440–{}", 440 + read::PAGE - 1)
        );
        assert_eq!(
            describe(&call(
                r#"{"name": "read", "arguments": {"path": "a.rs", "start": 18446744073709551615, "lines": 2}}"#
            )),
            "a.rs lines 18446744073709551615–18446744073709551615"
        );
        assert_eq!(
            describe(&call(
                r#"{"name": "read", "arguments": {"path": "a.rs", "lines": 0}}"#
            )),
            r#"{"lines":0,"path":"a.rs"}"#
        );
        assert_eq!(
            describe(&call(
                r#"{"name": "read", "arguments": {"path": "a.rs", "start": "x", "lines": 30}}"#
            )),
            r#"{"lines":30,"path":"a.rs","start":"x"}"#
        );
        assert_eq!(
            describe(&call(
                r#"{"name": "edit", "arguments": {"path": "a.rs", "old_string": "a", "new_string": "b", "start": "3"}}"#
            )),
            "a.rs at line 3"
        );
        assert_eq!(
            describe(&call(r#"{"name": "other", "arguments": {"x": 1}}"#)),
            r#"{"x":1}"#
        );
    }

    #[test]
    fn lists_the_tools() {
        let signatures = signatures();
        assert_eq!(signatures.lines().count(), 3);
        for line in signatures.lines() {
            let signature: Value = serde_json::from_str(line).unwrap();
            assert_eq!(signature["type"], "function");
            assert!(signature["function"]["parameters"]["required"].is_array());
        }
    }
}
