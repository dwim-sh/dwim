//! The `grep` tool: the lines of the project's files that match a regular
//! expression, with their file and line number, through `rg` where it is
//! installed and `grep` elsewhere, up to a hundred of them. Left with
//! `bash` alone, the model composed the pipeline itself, with whatever
//! exclusions it remembered: a search that crossed `target/` or
//! `node_modules/` filled its preview with what no one wanted, and a
//! search through `cat` and `head` lost the line numbers it needed next.

use std::{
    io::{BufRead, BufReader},
    process::{Command, Stdio},
    sync::OnceLock,
};

use serde_json::Value;

use crate::string;

/// The tool as the system prompt declares it.
pub const SIGNATURE: &str = r#"{"type": "function", "function": {"name": "grep", "description": "Search files for a regular expression and show the matching lines as file:line:text, up to 100 of them, leaving out files git ignores. To see the code around a match, read the file from its line.", "parameters": {"type": "object", "properties": {"pattern": {"type": "string", "description": "The regular expression to search for."}, "path": {"type": "string", "description": "The file or directory to search (default: the working directory)."}, "glob": {"type": "string", "description": "Only files whose name matches this pattern, such as *.rs. Optional."}}, "required": ["pattern"]}}}"#;

/// Most matches shown.
pub const MOST: usize = 100;

/// Most bytes of a match shown; a longer line is cut there, with a mark.
const LINE: usize = 500;

/// Whether `rg` is installed, looked up once.
fn ripgrep() -> bool {
    static RIPGREP: OnceLock<bool> = OnceLock::new();
    *RIPGREP.get_or_init(|| {
        Command::new("rg")
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok()
    })
}

/// Searches as `arguments` ask, returning what the model should see.
pub fn run(arguments: &Value) -> String {
    let Some(pattern) = string(arguments, "pattern") else {
        return "error: grep needs a 'pattern' string".to_string();
    };
    if pattern.is_empty() {
        return "error: the pattern is empty".to_string();
    }
    let path = string(arguments, "path")
        .filter(|path| !path.is_empty())
        .unwrap_or(".");
    let glob = string(arguments, "glob").filter(|glob| !glob.is_empty());
    let mut command = if ripgrep() {
        let mut command = Command::new("rg");
        command.args([
            "--line-number",
            "--with-filename",
            "--no-heading",
            "--color",
            "never",
            "--sort",
            "path",
        ]);
        if let Some(glob) = glob {
            command.args(["--glob", glob]);
        }
        command
    } else {
        let mut command = Command::new("grep");
        command.args([
            "-r",
            "-n",
            "-H",
            "-I",
            "--exclude-dir=.git",
            "--exclude-dir=target",
            "--exclude-dir=node_modules",
        ]);
        if let Some(glob) = glob {
            command.arg(format!("--include={glob}"));
        }
        command
    };
    command
        .args(["-e", pattern, "--", path])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(e) => return format!("error: could not search: {e}"),
    };
    // The matches up to the most shown, after which the search is ended:
    // it would otherwise run on for nothing.
    let stdout = BufReader::new(child.stdout.take().unwrap());
    let mut matches = Vec::new();
    let mut more = false;
    for line in stdout.split(b'\n') {
        let Ok(line) = line else {
            break;
        };
        if matches.len() == MOST {
            more = true;
            break;
        }
        matches.push(shown(&line));
    }
    if more {
        let _ = child.kill();
    }
    let output = child.wait_with_output();
    let what = match glob {
        Some(glob) => format!("{pattern} in {path} ({glob})"),
        None => format!("{pattern} in {path}"),
    };
    if matches.is_empty() {
        return match output {
            // Both searchers exit with 1 for no match and 2 for an error.
            Ok(output) if output.status.code() == Some(1) => format!("[no matches for {what}]"),
            Ok(output) => format!("error: {}", String::from_utf8_lossy(&output.stderr).trim()),
            Err(e) => format!("error: could not search: {e}"),
        };
    }
    matches.sort_by_key(|line| key(line));
    let note = if more {
        format!(
            "[the first {MOST} matches for {what}; narrow the pattern, the path or the glob for the rest]"
        )
    } else {
        format!(
            "[{} {} for {what}]",
            matches.len(),
            if matches.len() == 1 {
                "match"
            } else {
                "matches"
            }
        )
    };
    format!("{}\n{note}", matches.join("\n"))
}

/// A match as the model sees it: text, without a carriage return at its
/// end, cut at [`LINE`] bytes with a mark.
fn shown(line: &[u8]) -> String {
    let line = line.strip_suffix(b"\r").unwrap_or(line);
    let mut text = String::from_utf8_lossy(line).into_owned();
    if text.len() > LINE {
        let end = (0..=LINE)
            .rev()
            .find(|&i| text.is_char_boundary(i))
            .unwrap_or(0);
        text.truncate(end);
        text.push('…');
    }
    text
}

/// What a match sorts by: its file, then its line as a number.
fn key(line: &str) -> (String, u64) {
    let mut parts = line.splitn(3, ':');
    let file = parts.next().unwrap_or_default().to_string();
    let number = parts
        .next()
        .and_then(|number| number.parse().ok())
        .unwrap_or(0);
    (file, number)
}

#[cfg(test)]
mod tests {
    use std::{fs, path::PathBuf};

    use serde_json::json;

    use super::*;

    /// A directory of its own with a few files in it.
    fn project() -> PathBuf {
        static DIRS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = DIRS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("dwim-grep-{}-{n}", std::process::id()));
        fs::create_dir_all(dir.join("sub")).unwrap();
        fs::write(dir.join("a.rs"), "fn alpha() {}\nlet x = 1;\n").unwrap();
        fs::write(dir.join("sub/b.txt"), "alpha beta\ngamma\n").unwrap();
        dir
    }

    #[test]
    fn finds_lines_with_their_file_and_number() {
        let dir = project();
        let at = dir.to_str().unwrap();
        let found = run(&json!({ "pattern": "alpha", "path": at }));
        assert_eq!(
            found,
            format!(
                "{at}/a.rs:1:fn alpha() {{}}\n{at}/sub/b.txt:1:alpha beta\n[2 matches for alpha in {at}]"
            )
        );
        let found = run(&json!({ "pattern": "alpha", "path": at, "glob": "*.rs" }));
        assert_eq!(
            found,
            format!("{at}/a.rs:1:fn alpha() {{}}\n[1 match for alpha in {at} (*.rs)]")
        );
        let found = run(&json!({ "pattern": "gam+a", "path": format!("{at}/sub/b.txt") }));
        assert_eq!(
            found,
            format!("{at}/sub/b.txt:2:gamma\n[1 match for gam+a in {at}/sub/b.txt]")
        );
        assert_eq!(
            run(&json!({ "pattern": "delta", "path": at })),
            format!("[no matches for delta in {at}]")
        );
        assert!(
            run(&json!({ "pattern": "(", "path": at })).starts_with("error: "),
            "a malformed pattern is an error"
        );
        assert!(run(&json!({ "path": at })).starts_with("error: grep needs"));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn shows_the_first_hundred_matches() {
        let dir = project();
        let at = dir.to_str().unwrap();
        let text: String = (1..=150).map(|i| format!("needle {i}\n")).collect();
        fs::write(dir.join("many.txt"), text).unwrap();
        let found = run(&json!({ "pattern": "needle", "path": at }));
        let lines: Vec<&str> = found.lines().collect();
        assert_eq!(lines.len(), MOST + 1, "{found}");
        assert_eq!(lines[0], format!("{at}/many.txt:1:needle 1"));
        assert_eq!(lines[MOST - 1], format!("{at}/many.txt:100:needle 100"));
        assert_eq!(
            lines[MOST],
            format!(
                "[the first {MOST} matches for needle in {at}; narrow the pattern, the path or the glob for the rest]"
            )
        );
        fs::remove_dir_all(dir).unwrap();
    }
}
