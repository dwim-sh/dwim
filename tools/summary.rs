//! What a call did, in a line a person reads once it has run: a verb in
//! the past tense, what it was done to, and how it went. A command is
//! named by the programs it runs, not written out with its flags and
//! pipes, and a file changed by the lines added and removed, where the
//! model's output would be dozens of lines.

use crate::tools::{ToolCall, string, tool};

/// What a tool does, said in the past tense once it has.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verb {
    Ran,
    Read,
    Edited,
    Wrote,
    Searched,
    Listed,
}

impl Verb {
    /// The verb, as a line begins with it.
    pub fn word(self) -> &'static str {
        match self {
            Verb::Ran => "Ran",
            Verb::Read => "Read",
            Verb::Edited => "Edited",
            Verb::Wrote => "Wrote",
            Verb::Searched => "Searched",
            Verb::Listed => "Listed",
        }
    }

    /// The verb for a call that went wrong, in place of `word`.
    pub fn failed(self) -> &'static str {
        match self {
            Verb::Ran => "Couldn't run",
            Verb::Read => "Couldn't read",
            Verb::Edited => "Couldn't edit",
            Verb::Wrote => "Couldn't write",
            Verb::Searched => "Couldn't search",
            Verb::Listed => "Couldn't list",
        }
    }

    /// What the verb was done to, for a heading over calls of several
    /// verbs: `one` of them, or `many`, as in "edited a file" and "edited
    /// files".
    pub fn phrase(self, many: bool) -> &'static str {
        match (self, many) {
            (Verb::Ran, false) => "ran a command",
            (Verb::Ran, true) => "ran commands",
            (Verb::Read, false) => "read a file",
            (Verb::Read, true) => "read files",
            (Verb::Edited, false) => "edited a file",
            (Verb::Edited, true) => "edited files",
            (Verb::Wrote, false) => "wrote a file",
            (Verb::Wrote, true) => "wrote files",
            (Verb::Searched, _) => "searched",
            (Verb::Listed, _) => "listed files",
        }
    }

    /// What several calls of the verb are counted as.
    pub fn calls(self) -> &'static str {
        match self {
            Verb::Ran => "commands",
            Verb::Read => "reads",
            Verb::Edited => "edits",
            Verb::Wrote => "writes",
            Verb::Searched => "searches",
            Verb::Listed => "listings",
        }
    }
}

/// What a call did.
#[derive(Clone, Debug, PartialEq)]
pub struct Summary {
    pub verb: Verb,
    /// What it was done to: the programs a command runs, a file, a
    /// pattern.
    pub object: String,
    /// The lines it added to a file and removed from it, if it changed
    /// one and they are known.
    pub lines: Option<(usize, usize)>,
    /// How it went, in a few words, if that needs saying: the part of a
    /// file read, how many matches, or what went wrong.
    pub note: Option<String>,
    /// How it failed, in the words shown in place of the verb, as
    /// "Exited 101", if it did.
    pub failed: Option<String>,
}

/// What `call` did, whose result was `output`, as the model saw it: for a
/// call of a tool that does something to files or runs a command, which
/// `plan` and `done` do not, and whose arguments make sense.
pub fn summary(call: &ToolCall, output: &str) -> Option<Summary> {
    let tool = tool(&call.name)?;
    let verb = tool.verb?;
    let arguments = &call.arguments;
    let described = (tool.describe)(arguments)?;
    // A file is named by its path alone, so that calls on one file are
    // said to be on one: the part of it read is a note.
    let (object, note) = match verb {
        Verb::Ran => (commands(string(arguments, "command")?), None),
        Verb::Searched => (format!("for {described}"), None),
        Verb::Read | Verb::Edited | Verb::Wrote => {
            let path = string(arguments, "path")?;
            let part = described.strip_prefix(path).unwrap_or_default().trim();
            let note = (verb == Verb::Read && !part.is_empty()).then(|| part.to_string());
            (path.to_string(), note)
        }
        Verb::Listed => (described, None),
    };
    let mut summary = Summary {
        verb,
        object,
        lines: None,
        note,
        failed: None,
    };
    if let Some(error) = output.strip_prefix("error: ") {
        summary.failed = Some(verb.failed().to_string());
        summary.note = error.lines().next().map(str::to_string);
        return Some(summary);
    }
    match verb {
        Verb::Ran => {
            if let Some(failed) = failure(output) {
                summary.failed = Some(failed);
                summary.note = complaint(output);
            }
        }
        Verb::Edited => {
            let old = string(arguments, "old_string").unwrap_or_default();
            let new = string(arguments, "new_string").unwrap_or_default();
            summary.lines = Some(changed(old, new));
        }
        Verb::Wrote => {
            let lines = string(arguments, "content")
                .unwrap_or_default()
                .lines()
                .count();
            if output.starts_with("[replaced") {
                summary.note = Some(format!("replaced, {lines} lines"));
            } else {
                summary.lines = Some((lines, 0));
            }
        }
        Verb::Searched => summary.note = Some(found(output, "match", "matches")),
        Verb::Listed => summary.note = Some(found(output, "file", "files")),
        Verb::Read => {}
    }
    Some(summary)
}

/// Programs that are named with the subcommand they run, as `cargo test`
/// and `git commit` are: the program alone says little of what was done.
const SUBCOMMANDS: [&str; 22] = [
    "apt",
    "apt-get",
    "brew",
    "bun",
    "cargo",
    "docker",
    "gh",
    "git",
    "go",
    "just",
    "kubectl",
    "lake",
    "make",
    "npm",
    "npx",
    "pip",
    "pip3",
    "pnpm",
    "rustup",
    "systemctl",
    "uv",
    "yarn",
];

/// Words that come before a program without being it.
const PREFIXES: [&str; 8] = [
    "sudo", "time", "env", "exec", "command", "nohup", "nice", "xargs",
];

/// Commands that set up for the ones that do the work, and keywords that
/// stand for no command of their own.
const SETUP: [&str; 15] = [
    "cd", "pushd", "popd", "export", "set", "unset", "source", ".", ":", "true", "for", "case",
    "esac", "fi", "done",
];

/// Keywords that come before a command.
const KEYWORDS: [&str; 7] = ["if", "then", "elif", "else", "while", "until", "do"];

/// The programs `command` runs, a shell command line, named as a person
/// would say them, each once and in order: `cargo test, git commit` for
/// one that changes to a directory, runs the tests and pipes them to
/// `tail`, then commits. The ends of pipelines, which filter what the
/// program before them printed, the setup of `cd` and `export`, and the
/// bodies of here-documents, are left out. A command of nothing else is
/// named by its first program.
pub fn commands(command: &str) -> String {
    let mut names: Vec<String> = Vec::new();
    let mut first = None;
    for words in programs(command) {
        let Some(name) = name(&words) else {
            continue;
        };
        first.get_or_insert_with(|| name.clone());
        let setup = SETUP.contains(&words[0].as_str()) || words[0] == "echo";
        if !setup && !names.contains(&name) {
            names.push(name);
        }
    }
    if names.is_empty() {
        names.extend(first);
    }
    if names.is_empty() {
        return command.split_whitespace().collect::<Vec<_>>().join(" ");
    }
    names.join(", ")
}

/// The name of the program `words` run: its file name, and the
/// subcommand for a program that has them.
fn name(words: &[String]) -> Option<String> {
    let program = words.first()?;
    let base = program.rsplit('/').next().unwrap_or(program);
    if base.is_empty() {
        return None;
    }
    match words.get(1) {
        Some(sub)
            if SUBCOMMANDS.contains(&base)
                && sub.starts_with(|c: char| c.is_ascii_lowercase())
                && sub
                    .chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || "-_:".contains(c)) =>
        {
            Some(format!("{base} {sub}"))
        }
        _ => Some(base.to_string()),
    }
}

/// The words of each command `command` runs that is not the end of a
/// pipeline, with what comes before its program left out: assignments,
/// keywords, `sudo` and the like.
fn programs(command: &str) -> Vec<Vec<String>> {
    let mut programs = Vec::new();
    for (text, piped) in commands_of(&without_heredocs(command)) {
        if piped {
            continue;
        }
        let mut words: Vec<String> = text
            .split_whitespace()
            .map(|word| word.trim_matches(|c| "(){}'\"".contains(c)).to_string())
            .filter(|word| !word.is_empty())
            .collect();
        if words.first().is_some_and(|word| word.starts_with('#')) {
            continue;
        }
        let skip = words
            .iter()
            .position(|word| {
                !(KEYWORDS.contains(&word.as_str())
                    || PREFIXES.contains(&word.as_str())
                    || assignment(word))
            })
            .unwrap_or(words.len());
        words.drain(..skip);
        if !words.is_empty() {
            programs.push(words);
        }
    }
    programs
}

/// Whether `word` sets a variable, as `RUST_LOG=debug` does.
fn assignment(word: &str) -> bool {
    word.split_once('=').is_some_and(|(name, _)| {
        !name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
    })
}

/// `command` without the bodies of its here-documents, and with its
/// lines continued by a backslash joined.
fn without_heredocs(command: &str) -> String {
    let joined = command.replace("\\\n", " ");
    let mut kept = Vec::new();
    let mut until: Option<String> = None;
    for line in joined.lines() {
        if let Some(end) = &until {
            if line.trim() == end {
                until = None;
            }
            continue;
        }
        kept.push(line);
        if let Some((_, rest)) = line.split_once("<<") {
            let end: String = rest
                .trim_start_matches(['-', '~'])
                .trim_start()
                .chars()
                .take_while(|c| !c.is_whitespace() && !";|&)".contains(*c))
                .filter(|c| !"'\"".contains(*c))
                .collect();
            if !end.is_empty() {
                until = Some(end);
            }
        }
    }
    kept.join("\n")
}

/// The commands of a command line, split where `;`, `&&`, `||`, `&`, `|`
/// or a line's end separates them outside quotes, each with whether it
/// takes what the one before it printed through a pipe.
fn commands_of(line: &str) -> Vec<(String, bool)> {
    let mut commands = Vec::new();
    let mut current = String::new();
    let mut piped = false;
    let mut quote = None;
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        match (quote, c) {
            (Some(q), c) if c == q => {
                quote = None;
                current.push(c);
            }
            (Some(_), c) => current.push(c),
            (None, '\'' | '"') => {
                quote = Some(c);
                current.push(c);
            }
            (None, '\\') => {
                current.push(c);
                current.extend(chars.next());
            }
            (None, ';' | '\n') => {
                commands.push((std::mem::take(&mut current), piped));
                piped = false;
            }
            (None, '&' | '|') => {
                let doubled = chars.next_if_eq(&c).is_some();
                // A redirection, as `2>&1` and `&>` are, separates nothing.
                if c == '&' && (current.ends_with('>') || chars.peek() == Some(&'>')) {
                    current.push(c);
                    continue;
                }
                commands.push((std::mem::take(&mut current), piped));
                piped = c == '|' && !doubled;
            }
            (None, c) => current.push(c),
        }
    }
    commands.push((current, piped));
    commands.retain(|(text, _)| !text.trim().is_empty());
    commands
}

/// How a command whose result is `output` failed, in the words shown in
/// place of `Ran`, if it did: what it exited with, or that it was killed.
fn failure(output: &str) -> Option<String> {
    let last = output.lines().last().unwrap_or_default();
    if let Some(code) = last
        .strip_prefix("[exit code ")
        .and_then(|rest| rest.strip_suffix(']'))
    {
        return (code != "0").then(|| format!("Exited {code}"));
    }
    if output.contains("[killed after") {
        return Some("Timed out".to_string());
    }
    if output.contains("[killed by signal") {
        return Some("Killed".to_string());
    }
    None
}

/// What a command that failed said of why, from its `output`: the first
/// line that reports an error or a failure, or else its last line.
fn complaint(output: &str) -> Option<String> {
    let lines: Vec<&str> = output
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('['))
        .collect();
    let complains = |line: &&&str| {
        let lower = line.to_ascii_lowercase();
        lower.starts_with("error")
            || lower.starts_with("fatal")
            || lower.contains("panicked")
            || line.contains("FAILED")
            || line.contains("Error:")
    };
    lines
        .iter()
        .find(complains)
        .or(lines.last())
        .map(|line| line.to_string())
}

/// How many matches or files a search or a listing whose result is
/// `output` found, said with the words for `one` and `many`: its result
/// is what it found, a line each, then a note in brackets.
fn found(output: &str, one: &str, many: &str) -> String {
    let lines: Vec<&str> = output.lines().collect();
    let note = lines.last().copied().unwrap_or_default();
    if note.starts_with("[no ") {
        return format!("no {many}");
    }
    let n = lines.len().saturating_sub(1);
    if note.starts_with("[the first") {
        return format!("first {n} {many}");
    }
    format!("{n} {}", if n == 1 { one } else { many })
}

/// The lines an edit of `old` to `new` adds and removes: those of each
/// that are not the lines both begin or end with.
fn changed(old: &str, new: &str) -> (usize, usize) {
    let old: Vec<&str> = old.lines().collect();
    let new: Vec<&str> = new.lines().collect();
    let start = old.iter().zip(&new).take_while(|(a, b)| a == b).count();
    let end = old[start..]
        .iter()
        .rev()
        .zip(new[start..].iter().rev())
        .take_while(|(a, b)| a == b)
        .count();
    (new.len() - start - end, old.len() - start - end)
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::*;

    fn call(name: &str, arguments: Value) -> ToolCall {
        ToolCall {
            name: name.to_string(),
            arguments,
        }
    }

    #[test]
    fn names_the_programs_a_command_runs() {
        assert_eq!(
            commands("cd gpu && cargo test --lib metal:: 2>&1 | tail -20"),
            "cargo test"
        );
        assert_eq!(
            commands("cargo clippy --all-targets -- -D warnings && cargo fmt --check"),
            "cargo clippy, cargo fmt"
        );
        assert_eq!(
            commands("RUST_LOG=debug ./target/debug/dwim --once hi; git status"),
            "dwim, git status"
        );
        assert_eq!(
            commands(
                "git add -A && git commit -F - <<'EOF'\nsys: Move libc\n\nBody; with | pipes\nEOF"
            ),
            "git add, git commit"
        );
        assert_eq!(commands("cd src && ls"), "ls");
        assert_eq!(commands("cd src"), "cd");
        assert_eq!(commands("grep -rn 'a|b' src | head"), "grep");
        assert_eq!(commands("sleep 1 & wait"), "sleep, wait");
    }

    #[test]
    fn says_why_a_command_failed() {
        let ran = |output: &str| summary(&call("bash", json!({"command": "cargo clippy"})), output);
        let failed = ran("    Checking dwim\nwarning: unused\nerror: variable does not need to be mutable\n  --> src/lib.rs:3\nerror: could not compile `dwim`\n[exit code 101]").unwrap();
        assert_eq!(failed.failed.as_deref(), Some("Exited 101"));
        assert_eq!(
            failed.note.as_deref(),
            Some("error: variable does not need to be mutable")
        );
        let passed = ran("    Finished\n[exit code 0]").unwrap();
        assert_eq!((passed.failed, passed.note), (None, None));
        let killed = ran("[killed after 300 s, the timeout; give a longer timeout]").unwrap();
        assert_eq!(killed.failed.as_deref(), Some("Timed out"));
    }

    #[test]
    fn counts_the_lines_an_edit_changes() {
        let edit = summary(
            &call(
                "edit",
                json!({"path": "a.rs", "old_string": "a\nb\nc\nd", "new_string": "a\nB\nB2\nB3\nd"}),
            ),
            "[a.rs: lines 1–4 replaced by 5 lines; 9 lines in all]",
        )
        .unwrap();
        assert_eq!(edit.lines, Some((3, 2)));
        let again = summary(
            &call(
                "edit",
                json!({"path": "a.rs", "start": 6, "old_string": "x", "new_string": "y"}),
            ),
            "[a.rs: line 6 replaced]",
        )
        .unwrap();
        assert_eq!(again.object, edit.object);
        let wrong = summary(
            &call(
                "edit",
                json!({"path": "a.rs", "old_string": "x", "new_string": "y"}),
            ),
            "error: old_string is not in a.rs\nthe closest lines are",
        )
        .unwrap();
        assert_eq!(wrong.lines, None);
        assert_eq!(wrong.failed.as_deref(), Some("Couldn't edit"));
        assert_eq!(wrong.note.as_deref(), Some("old_string is not in a.rs"));
    }

    #[test]
    fn counts_what_a_search_found() {
        let grep = |output: &str| {
            summary(&call("grep", json!({"pattern": "alpha"})), output)
                .unwrap()
                .note
        };
        assert_eq!(
            grep("a.rs:1:alpha\nb.rs:4:alpha\n[2 matches for alpha]").as_deref(),
            Some("2 matches")
        );
        assert_eq!(
            grep("[no matches for alpha]").as_deref(),
            Some("no matches")
        );
    }

    #[test]
    fn says_nothing_of_plan_and_done() {
        assert_eq!(
            summary(&call("done", json!({"result": "built"})), "ok"),
            None
        );
    }
}
