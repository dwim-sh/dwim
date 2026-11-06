//! The `glob` tool: the files whose paths match a pattern, up to a
//! hundred, leaving out what git ignores. `*.rs` names every Rust file
//! under the directory, `src/*.rs` those right in `src/`, and
//! `src/**/*_test.go` those anywhere below it. Left with `bash` alone, the
//! model ran `find` and `ls -R` over `target/` and `node_modules/`, and
//! its preview was full before it reached the project's own files.

use std::{fs, path::Path, process::Command};

use serde_json::Value;

use crate::{plural, string};

/// The tool as the system prompt declares it.
pub const SIGNATURE: &str = r#"{"type": "function", "function": {"name": "glob", "description": "List the files whose paths match a pattern, up to 100, leaving out files git ignores: *.rs for every Rust file under the directory, src/*.rs for those directly in src, src/**/*_test.go for those anywhere below it. A pattern without a slash matches a file name anywhere.", "parameters": {"type": "object", "properties": {"pattern": {"type": "string", "description": "The pattern: * matches within a name, ** across directories, ? one character."}, "path": {"type": "string", "description": "The directory to search (default: the working directory)."}}, "required": ["pattern"]}}}"#;

/// Most files shown.
pub const MOST: usize = 100;

/// Lists the files `arguments` ask for, returning what the model should
/// see.
pub fn run(arguments: &Value) -> String {
    let Some(pattern) = string(arguments, "pattern") else {
        return "error: glob needs a 'pattern' string".to_string();
    };
    let path = string(arguments, "path")
        .filter(|path| !path.is_empty())
        .unwrap_or(".");
    if !Path::new(path).is_dir() {
        return format!("error: {path} is not a directory");
    }
    let segments: Vec<&str> = pattern
        .trim_start_matches("./")
        .split('/')
        .filter(|segment| !segment.is_empty())
        .collect();
    if segments.is_empty() {
        return "error: the pattern is empty".to_string();
    }
    let segments = if segments.len() == 1 {
        vec!["**", segments[0]]
    } else {
        segments
    };
    let files = files(Path::new(path));
    let found: Vec<&String> = files
        .iter()
        .filter(|file| matches(&segments, &file.split('/').collect::<Vec<_>>()))
        .collect();
    let what = match path {
        "." => pattern.to_string(),
        path => format!("{pattern} in {path}"),
    };
    if found.is_empty() {
        return format!("[no files match {what}]");
    }
    let shown: Vec<String> = found
        .iter()
        .take(MOST)
        .map(|file| match path {
            "." => file.to_string(),
            path => format!("{}/{file}", path.trim_end_matches('/')),
        })
        .collect();
    let note = if found.len() > MOST {
        format!(
            "[the first {MOST} of {} files matching {what}; narrow the pattern for the rest]",
            found.len()
        )
    } else {
        format!(
            "[{} {} matching {what}]",
            found.len(),
            plural(found.len() as u64, "file")
        )
    };
    format!("{}\n{note}", shown.join("\n"))
}

/// The files under `dir`, as paths relative to it with `/` between their
/// parts, in order, hidden files and directories left out: what git lists
/// where `dir` is in a repository, which leaves out what it ignores too,
/// and otherwise everything.
fn files(dir: &Path) -> Vec<String> {
    let listed = Command::new("git")
        .args([
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ])
        .current_dir(dir)
        .output()
        .ok()
        .filter(|output| output.status.success());
    let mut files = match listed {
        Some(output) => output
            .stdout
            .split(|&byte| byte == 0)
            .filter(|file| !file.is_empty())
            .map(|file| String::from_utf8_lossy(file).into_owned())
            .filter(|file| !file.split('/').any(|part| part.starts_with('.')))
            // A file deleted but not yet committed is still listed.
            .filter(|file| dir.join(file).is_file())
            .collect(),
        None => {
            let mut files = Vec::new();
            walk(dir, "", &mut files);
            files
        }
    };
    files.sort();
    files.dedup();
    files
}

/// Adds the files under `dir`, whose path relative to the root is `at`, to
/// `files`, leaving out hidden ones and not following symbolic links.
fn walk(dir: &Path, at: &str, files: &mut Vec<String>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        if name.starts_with('.') {
            continue;
        }
        let path = if at.is_empty() {
            name.to_string()
        } else {
            format!("{at}/{name}")
        };
        match entry.file_type() {
            Ok(kind) if kind.is_dir() => walk(&entry.path(), &path, files),
            Ok(kind) if kind.is_file() => files.push(path),
            _ => {}
        }
    }
}

/// Whether the path `parts` matches the pattern `pattern`, both split at
/// their slashes: `**` matches any number of parts, and within a part `*`
/// matches any run of characters and `?` one.
fn matches(pattern: &[&str], parts: &[&str]) -> bool {
    match (pattern.first(), parts.first()) {
        (None, None) => true,
        (None, Some(_)) => false,
        (Some(&"**"), _) => {
            matches(&pattern[1..], parts) || (!parts.is_empty() && matches(pattern, &parts[1..]))
        }
        (Some(part), Some(name)) => segment(part, name) && matches(&pattern[1..], &parts[1..]),
        (Some(_), None) => false,
    }
}

/// Whether the name `name` matches the pattern `pattern`, where `*`
/// matches any run of characters and `?` one.
fn segment(pattern: &str, name: &str) -> bool {
    fn go(pattern: &[char], name: &[char]) -> bool {
        match pattern.first() {
            None => name.is_empty(),
            Some('*') => go(&pattern[1..], name) || (!name.is_empty() && go(pattern, &name[1..])),
            Some('?') => !name.is_empty() && go(&pattern[1..], &name[1..]),
            Some(&c) => name.first() == Some(&c) && go(&pattern[1..], &name[1..]),
        }
    }
    let pattern: Vec<char> = pattern.chars().collect();
    let name: Vec<char> = name.chars().collect();
    go(&pattern, &name)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use serde_json::json;

    use super::*;

    /// A directory of its own with a small tree in it.
    fn tree() -> PathBuf {
        static DIRS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = DIRS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("dwim-glob-{}-{n}", std::process::id()));
        for sub in ["src/deep", "docs", ".hidden"] {
            fs::create_dir_all(dir.join(sub)).unwrap();
        }
        for file in [
            "a.rs",
            "src/b.rs",
            "src/deep/c.rs",
            "docs/x.md",
            ".hidden/h.rs",
        ] {
            fs::write(dir.join(file), "").unwrap();
        }
        dir
    }

    #[test]
    fn lists_the_files_a_pattern_matches() {
        let dir = tree();
        let at = dir.to_str().unwrap();
        let list = |pattern: &str| run(&json!({ "pattern": pattern, "path": at }));
        assert_eq!(
            list("*.rs"),
            format!(
                "{at}/a.rs\n{at}/src/b.rs\n{at}/src/deep/c.rs\n[3 files matching *.rs in {at}]"
            )
        );
        assert_eq!(
            list("src/*.rs"),
            format!("{at}/src/b.rs\n[1 file matching src/*.rs in {at}]")
        );
        assert_eq!(
            list("src/**/*.rs"),
            format!("{at}/src/b.rs\n{at}/src/deep/c.rs\n[2 files matching src/**/*.rs in {at}]")
        );
        assert_eq!(
            list("**/?.md"),
            format!("{at}/docs/x.md\n[1 file matching **/?.md in {at}]")
        );
        assert_eq!(list("*.go"), format!("[no files match *.go in {at}]"));
        assert!(run(&json!({ "pattern": "*.rs", "path": "/nonexistent" })).starts_with("error: "));
        assert!(run(&json!({ "path": at })).starts_with("error: glob needs"));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn leaves_out_what_git_ignores_and_caps_the_list() {
        let dir = tree();
        let at = dir.to_str().unwrap();
        assert!(
            Command::new("git")
                .args(["init", "-q"])
                .current_dir(&dir)
                .status()
                .unwrap()
                .success()
        );
        fs::write(dir.join(".gitignore"), "ignored/\n").unwrap();
        fs::create_dir_all(dir.join("ignored")).unwrap();
        fs::write(dir.join("ignored/i.rs"), "").unwrap();
        let list = |pattern: &str| run(&json!({ "pattern": pattern, "path": at }));
        assert_eq!(
            list("*.rs"),
            format!(
                "{at}/a.rs\n{at}/src/b.rs\n{at}/src/deep/c.rs\n[3 files matching *.rs in {at}]"
            )
        );
        for i in 0..150 {
            fs::write(dir.join(format!("docs/{i:03}.md")), "").unwrap();
        }
        let found = list("docs/*.md");
        let lines: Vec<&str> = found.lines().collect();
        assert_eq!(lines.len(), MOST + 1);
        assert_eq!(lines[0], format!("{at}/docs/000.md"));
        assert_eq!(
            lines[MOST],
            format!(
                "[the first {MOST} of 151 files matching docs/*.md in {at}; narrow the pattern for the rest]"
            )
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn matches_patterns() {
        let m = |pattern: &str, path: &str| {
            let pattern: Vec<&str> = pattern.split('/').collect();
            let path: Vec<&str> = path.split('/').collect();
            matches(&pattern, &path)
        };
        assert!(m("*.rs", "a.rs"));
        assert!(!m("*.rs", "src/a.rs"));
        assert!(m("**/*.rs", "a.rs"));
        assert!(m("**/*.rs", "src/deep/a.rs"));
        assert!(m("src/**", "src/deep/a.rs"));
        assert!(m("src/**/a.rs", "src/a.rs"));
        assert!(!m("src/*/a.rs", "src/a.rs"));
        assert!(m("?.rs", "a.rs") && !m("?.rs", "ab.rs"));
        assert!(segment("a*b*c", "aXbYc") && !segment("a*b", "ac"));
    }
}
