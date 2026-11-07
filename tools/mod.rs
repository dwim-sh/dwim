//! The tools the model can call: `bash`, which runs a shell command,
//! `read`, which gives it a file a page at a time, `edit`, which replaces
//! a string in a file with another, `write`, which writes a file whole,
//! `grep`, which searches files for a pattern, and `glob`, which lists
//! the files whose paths match one. Each is a [`Tool`]: the signature the
//! system prompt declares it with, whether it changes a file, how a call
//! reads on screen, and what it does. The tools work in the directory the
//! agent does: a path a call gives is taken from there.

pub mod bash;
mod call;
mod edit;
mod glob;
mod grep;
pub mod read;
mod write;

use std::{
    fs,
    path::{Path, PathBuf},
};

use serde_json::Value;

use crate::tools::bash::Bash;
pub use crate::tools::call::ToolCall;

/// A tool: what the model is told of it, and what it does.
pub struct Tool {
    name: &'static str,
    /// The signature the system prompt declares it with.
    signature: fn() -> Signature,
    /// Whether a call that succeeds changes a file.
    changes: bool,
    /// How a call reads on screen, if its arguments make sense.
    describe: fn(&Value) -> Option<String>,
    /// Runs a call, returning what the model should see, or what it is
    /// told went wrong.
    run: fn(&mut Tools, &Value) -> Result<String, String>,
}

/// The tools, in the order the system prompt lists them.
static TOOLS: [Tool; 6] = [
    bash::TOOL,
    read::TOOL,
    edit::TOOL,
    write::TOOL,
    grep::TOOL,
    glob::TOOL,
];

/// The tool called `name`.
fn tool(name: &str) -> Option<&'static Tool> {
    TOOLS.iter().find(|tool| tool.name == name)
}

/// The tools' signatures, one per line, as the system prompt lists them.
pub fn signatures() -> String {
    TOOLS
        .iter()
        .map(|tool| (tool.signature)().json(tool.name))
        .collect::<Vec<_>>()
        .join("\n")
}

/// A tool's signature: what it does, and its parameters.
struct Signature {
    description: String,
    parameters: Vec<Parameter>,
}

/// A parameter of a tool: its name, its JSON type, what it is, and
/// whether a call must give it.
struct Parameter {
    name: &'static str,
    kind: &'static str,
    description: String,
    required: bool,
}

impl Parameter {
    /// A string parameter.
    fn string(name: &'static str, description: impl Into<String>) -> Self {
        Self {
            name,
            kind: "string",
            description: description.into(),
            required: true,
        }
    }

    /// An integer parameter.
    fn integer(name: &'static str, description: impl Into<String>) -> Self {
        Self {
            kind: "integer",
            ..Self::string(name, description)
        }
    }

    /// The parameter, which a call may leave out.
    fn optional(self) -> Self {
        Self {
            required: false,
            ..self
        }
    }
}

impl Signature {
    /// The signature of the tool called `name`, as the chat template lists
    /// it: JSON as Python's `json.dumps` writes it, with a space after
    /// each colon and comma, since that is how the template's `tojson`
    /// writes one.
    fn json(&self, name: &str) -> String {
        let quoted = |text: &str| Value::String(text.to_string()).to_string();
        let properties: Vec<String> = self
            .parameters
            .iter()
            .map(|p| {
                format!(
                    "{}: {{\"type\": {}, \"description\": {}}}",
                    quoted(p.name),
                    quoted(p.kind),
                    quoted(&p.description)
                )
            })
            .collect();
        let required: Vec<String> = self
            .parameters
            .iter()
            .filter(|p| p.required)
            .map(|p| quoted(p.name))
            .collect();
        format!(
            "{{\"type\": \"function\", \"function\": {{\"name\": {}, \"description\": {}, \"parameters\": {{\"type\": \"object\", \"properties\": {{{}}}, \"required\": [{}]}}}}}}",
            quoted(name),
            quoted(&self.description),
            properties.join(", "),
            required.join(", ")
        )
    }
}

/// The tools, working in a directory, with what they keep between calls.
pub struct Tools {
    dir: PathBuf,
    bash: Bash,
}

impl Tools {
    /// The tools working in `dir`: a path a call gives is taken from
    /// there, and `bash` runs there.
    pub fn new(dir: &Path) -> Self {
        Self {
            dir: dir.to_path_buf(),
            bash: Bash::default(),
        }
    }

    /// Runs bash commands through `shell`, as `shell -c command`, instead
    /// of `sh`.
    pub fn set_shell(&mut self, shell: PathBuf) {
        self.bash.set_shell(shell);
    }

    /// Runs a call, returning what the model should see.
    pub fn run(&mut self, call: &ToolCall) -> Output {
        match tool(&call.name) {
            Some(tool) => Output::of((tool.run)(self, &call.arguments), tool.changes),
            None => Output::of(Err(format!("unknown tool '{}'", call.name)), false),
        }
    }
}

/// What a call gives the model, and whether it changed a file.
pub struct Output {
    pub text: String,
    pub changed: bool,
}

impl Output {
    /// What the model sees of a call's result, marked `error:` if it went
    /// wrong, which changed a file if the tool `changes` one and it did
    /// not.
    fn of(result: Result<String, String>, changes: bool) -> Self {
        match result {
            Ok(text) => Self {
                text,
                changed: changes,
            },
            Err(error) => Self {
                text: format!("error: {error}"),
                changed: false,
            },
        }
    }
}

/// How a call reads on screen: as its tool describes it, or as the
/// arguments as written, for a call of no tool or one its tool cannot
/// make sense of.
pub fn describe(call: &ToolCall) -> String {
    tool(&call.name)
        .and_then(|tool| (tool.describe)(&call.arguments))
        .unwrap_or_else(|| call.arguments.to_string())
}

/// The string argument `key`, which `tool` needs.
fn required<'a>(arguments: &'a Value, tool: &str, key: &str) -> Result<&'a str, String> {
    string(arguments, key).ok_or_else(|| format!("{tool} needs a '{key}' string"))
}

/// The string argument `key`, if there is one.
fn string<'a>(arguments: &'a Value, key: &str) -> Option<&'a str> {
    arguments.get(key)?.as_str()
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

/// The argument `key` as a number, if the call gives it, or the error
/// for one given that is not a number, saying it is `what`.
fn optional_number(arguments: &Value, key: &str, what: &str) -> Result<Option<u64>, String> {
    match arguments.get(key) {
        None => Ok(None),
        Some(_) => number(arguments, key)
            .map(Some)
            .ok_or_else(|| format!("'{key}' is {what}")),
    }
}

/// The directory argument `path`, if the call gives one that is not
/// empty.
fn directory(arguments: &Value) -> Option<&str> {
    string(arguments, "path").filter(|path| !path.is_empty())
}

/// Checks that `path`, taken from `dir`, is `dir` or below it, symbolic
/// links resolved, as the tools that search require of where they are
/// pointed: the rest of the disk is not the project's. Says why otherwise:
/// `tool` does not search outside, or the path or `dir` cannot be
/// resolved.
fn within(tool: &str, dir: &Path, path: &str) -> Result<(), String> {
    let (target, root) = match (fs::canonicalize(dir.join(path)), fs::canonicalize(dir)) {
        (Ok(target), Ok(root)) => (target, root),
        (Err(error), _) => return Err(format!("{path} cannot be resolved: {error}")),
        (_, Err(error)) => {
            return Err(format!("the working directory cannot be resolved: {error}"));
        }
    };
    if target.starts_with(root) {
        Ok(())
    } else {
        Err(format!(
            "{tool} searches the working directory and the directories under it; {path} is outside it. Use bash for other places."
        ))
    }
}

/// How a search reads on screen: the pattern, and where it is looked for.
fn search(arguments: &Value) -> Option<String> {
    let pattern = string(arguments, "pattern")?;
    Some(match directory(arguments) {
        Some(path) => format!("{pattern} in {path}"),
        None => pattern.to_string(),
    })
}

/// `one` if `n` is one, and otherwise `many`: the word for `n` of
/// something.
fn plural<'a>(n: u64, one: &'a str, many: &'a str) -> &'a str {
    if n == 1 { one } else { many }
}

/// Cuts `text` to at most `max` bytes, on a character boundary, with a
/// mark; returns whether it was cut.
fn cut(text: &mut String, max: usize) -> bool {
    if text.len() <= max {
        return false;
    }
    let end = (0..=max)
        .rev()
        .find(|&i| text.is_char_boundary(i))
        .unwrap_or(0);
    text.truncate(end);
    text.push('…');
    true
}

/// The line `line` without its ending, whether that is `\n` or `\r\n`.
fn without_ending(line: &[u8]) -> &[u8] {
    let content = line.strip_suffix(b"\n").unwrap_or(line);
    content.strip_suffix(b"\r").unwrap_or(content)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(text: &str) -> ToolCall {
        ToolCall::parse(text).unwrap()
    }

    #[test]
    fn runs_the_tool_named() {
        let mut tools = Tools::new(Path::new("."));
        let hi = tools.run(&call(
            r#"{"name": "bash", "arguments": {"command": "echo hi"}}"#,
        ));
        assert_eq!(hi.text, "hi\n[exit code 0]");
        assert!(!hi.changed);
        let read = tools.run(&call(
            r#"{"name": "read", "arguments": {"path": "Cargo.toml"}}"#,
        ));
        assert!(read.text.starts_with("1\t[package]\n"), "{}", read.text);
        assert!(
            tools
                .run(&call(r#"{"name": "rm", "arguments": {}}"#))
                .text
                .starts_with("error: unknown tool")
        );
        assert!(
            tools
                .run(&call(r#"{"name": "bash", "arguments": {}}"#))
                .text
                .starts_with("error: bash needs")
        );
        assert!(
            tools
                .run(&call(r#"{"name": "read", "arguments": {}}"#))
                .text
                .starts_with("error: read needs")
        );
    }

    #[test]
    fn says_whether_a_call_changed_a_file() {
        let dir = std::env::temp_dir().join(format!("dwim-tools-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut tools = Tools::new(&dir);
        let written = tools.run(&call(
            r#"{"name": "write", "arguments": {"path": "a.txt", "content": "x"}}"#,
        ));
        assert!(written.changed, "{}", written.text);
        assert_eq!(std::fs::read_to_string(dir.join("a.txt")).unwrap(), "x\n");
        let refused = tools.run(&call(
            r#"{"name": "edit", "arguments": {"path": "a.txt", "old_string": "y", "new_string": "z"}}"#,
        ));
        assert!(refused.text.starts_with("error: "), "{}", refused.text);
        assert!(!refused.changed);
        let read = tools.run(&call(r#"{"name": "read", "arguments": {"path": "a.txt"}}"#));
        assert!(!read.changed);
        std::fs::remove_dir_all(dir).unwrap();
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
            describe(&call(
                r#"{"name": "write", "arguments": {"path": "a.rs", "content": "x"}}"#
            )),
            "a.rs"
        );
        assert_eq!(
            describe(&call(
                r#"{"name": "grep", "arguments": {"pattern": "fn main", "path": "src"}}"#
            )),
            "fn main in src"
        );
        assert_eq!(
            describe(&call(
                r#"{"name": "glob", "arguments": {"pattern": "*.rs"}}"#
            )),
            "*.rs"
        );
        assert_eq!(
            describe(&call(r#"{"name": "other", "arguments": {"x": 1}}"#)),
            r#"{"x":1}"#
        );
    }

    #[test]
    fn lists_the_tools() {
        let signatures = signatures();
        assert_eq!(signatures.lines().count(), TOOLS.len());
        for (line, tool) in signatures.lines().zip(&TOOLS) {
            let signature: Value = serde_json::from_str(line).unwrap();
            assert_eq!(signature["type"], "function");
            assert_eq!(signature["function"]["name"], tool.name);
            assert!(signature["function"]["parameters"]["required"].is_array());
        }
        // As Python's json.dumps writes it, which the template's tojson
        // does, with the constants the tools work by in the prose.
        let read = Signature {
            description: format!("Read up to {} lines.", read::PAGE),
            parameters: vec![
                Parameter::string("path", "The file."),
                Parameter::integer("start", "The \"first\" line.").optional(),
            ],
        };
        assert_eq!(
            read.json("read"),
            format!(
                r#"{{"type": "function", "function": {{"name": "read", "description": "Read up to {} lines.", "parameters": {{"type": "object", "properties": {{"path": {{"type": "string", "description": "The file."}}, "start": {{"type": "integer", "description": "The \"first\" line."}}}}, "required": ["path"]}}}}}}"#,
                read::PAGE
            )
        );
    }

    #[test]
    fn cuts_text_between_characters() {
        let mut short = "abc".to_string();
        assert!(!cut(&mut short, 3));
        assert_eq!(short, "abc");
        assert_eq!(plural(1, "match", "matches"), "match");
        assert_eq!(plural(2, "match", "matches"), "matches");
        assert_eq!(plural(0, "line", "lines"), "lines");
        let mut long = "ääää".to_string();
        assert!(cut(&mut long, 5));
        assert_eq!(long, "ää…");
        assert_eq!(without_ending(b"a\r\n"), b"a");
        assert_eq!(without_ending(b"a"), b"a");
    }
}
