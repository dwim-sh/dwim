//! The `glob` tool: the files whose paths match a pattern, up to a
//! hundred, leaving out what git ignores. `*.rs` names every Rust file
//! under the directory, `src/*.rs` those right in `src/`, and
//! `src/**/*_test.go` those anywhere below it. Left with `bash` alone, the
//! model ran `find` and `ls -R` over `target/` and `node_modules/`, and
//! its preview was full before it reached the project's own files. Where
//! there is no git to ignore those, the tool's own walk leaves them out
//! and enters only the directories the pattern can reach, so that a cap
//! on the walk does not fall before the files the pattern names. The
//! search stays within the working directory: asked for `README*` in `/`,
//! the model once had the tool walk the whole disk for a minute and got
//! back a page of other projects' READMEs.

use std::{fs, path::Path, process::Command};

use serde_json::Value;

use crate::tools::{Parameter, Signature, Tool, directory, plural, required, search, within};

const NAME: &str = "glob";

pub const TOOL: Tool = Tool {
    name: NAME,
    signature,
    changes: false,
    describe: search,
    run: |tools, arguments| run(&tools.dir, arguments),
};

fn signature() -> Signature {
    Signature {
        description: format!(
            "List the files whose paths match a pattern, up to {MOST}, leaving out files git ignores: *.rs for every Rust file under the directory, src/*.rs for those directly in src, src/**/*_test.go for those anywhere below it. A pattern without a slash matches a file name anywhere."
        ),
        parameters: vec![
            Parameter::string(
                "pattern",
                "The pattern: * matches within a name, ** across directories, ? one character.",
            ),
            Parameter::string(
                "path",
                "The directory to search, the working directory or one below it (default: the working directory).",
            )
            .optional(),
        ],
    }
}

/// Most files shown.
const MOST: usize = 100;

/// Most entries the walk visits where there is no git to list the files.
const CAP: usize = 5000;

/// Lists the files `arguments` ask for, under `dir` or a path taken from
/// it, which must be `dir` or below it, returning what the model should
/// see.
fn run(dir: &Path, arguments: &Value) -> Result<String, String> {
    let pattern = required(arguments, NAME, "pattern")?;
    let path = directory(arguments).unwrap_or(".");
    let root = dir.join(path);
    if !root.is_dir() {
        return Err(format!("{path} is not a directory"));
    }
    within(NAME, dir, path)?;
    let segments: Vec<&str> = pattern
        .trim_start_matches("./")
        .split('/')
        .filter(|segment| !segment.is_empty())
        .collect();
    if segments.is_empty() {
        return Err("the pattern is empty".to_string());
    }
    // A name alone is a name anywhere.
    let segments = if segments.len() == 1 {
        vec!["**", segments[0]]
    } else {
        segments
    };
    let listing = files(&root, &segments);
    let found: Vec<&String> = listing
        .files
        .iter()
        .filter(|file| matches(&segments, &file.split('/').collect::<Vec<_>>()))
        .collect();
    let what = match path {
        "." => pattern.to_string(),
        path => format!("{pattern} in {path}"),
    };
    // The note: how many matched, and how to see the rest where the
    // pattern matched more than are shown or the walk stopped short.
    let count = match found.len() {
        0 => "no files match".to_string(),
        n if n > MOST => format!("the first {MOST} of {n} files matching"),
        n => format!("{n} {} matching", plural(n as u64, "file", "files")),
    };
    let among = if listing.complete {
        String::new()
    } else {
        format!(" among the first {CAP} entries walked")
    };
    let narrow = match (found.len() > MOST, listing.complete) {
        (true, false) => Some("the pattern or the path"),
        (true, true) => Some("the pattern"),
        (false, false) => Some("the path"),
        (false, true) => None,
    };
    let advice = match narrow {
        Some(narrow) if found.is_empty() => format!("; narrow {narrow} to search further"),
        Some(narrow) => format!("; narrow {narrow} for the rest"),
        None => String::new(),
    };
    let mut lines: Vec<String> = found
        .iter()
        .take(MOST)
        .map(|file| match path {
            "." => file.to_string(),
            path => format!("{}/{file}", path.trim_end_matches('/')),
        })
        .collect();
    let unread = match listing.unread.as_slice() {
        [] => String::new(),
        [one] => format!("; {one} could not be read"),
        [one, more @ ..] => format!(
            "; {one} and {} more {} could not be read",
            more.len(),
            plural(more.len() as u64, "directory", "directories")
        ),
    };
    lines.push(format!("[{count} {what}{among}{advice}{unread}]"));
    Ok(lines.join("\n"))
}

/// Directories the walk leaves out unless the pattern names them: where
/// there is no git to ignore them, they hold more files than the project.
const BUILT: [&str; 2] = ["node_modules", "target"];

/// What is listed under a directory: the files, as paths relative to it
/// with `/` between their parts, in order; whether the list is complete,
/// which it is not when the walk hit the cap; and the directories the
/// walk could not read, by the same paths.
struct Listing {
    files: Vec<String>,
    complete: bool,
    unread: Vec<String>,
}

/// The files under `dir` the pattern `pattern` could match, hidden files
/// and directories left out: what git lists where `dir` is in a
/// repository, which leaves out what it ignores too, and otherwise what
/// a walk of the directories the pattern reaches finds, up to `CAP`
/// entries.
fn files(dir: &Path, pattern: &[&str]) -> Listing {
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
    let mut listing = match listed {
        Some(output) => Listing {
            files: output
                .stdout
                .split(|&byte| byte == 0)
                .filter(|file| !file.is_empty())
                .map(|file| String::from_utf8_lossy(file).into_owned())
                .filter(|file| !file.split('/').any(|part| part.starts_with('.')))
                // A file deleted but not yet committed is still listed.
                .filter(|file| dir.join(file).is_file())
                .collect(),
            complete: true,
            unread: Vec::new(),
        },
        None => {
            let mut listing = Listing {
                files: Vec::new(),
                complete: true,
                unread: Vec::new(),
            };
            let mut entries = 0;
            listing.complete = walk(dir, "", pattern, &mut listing, &mut entries);
            listing
        }
    };
    listing.files.sort();
    listing.files.dedup();
    listing
}

/// Adds the files under `dir`, whose path relative to the root is `at`, to
/// `listing`, in name order, leaving out hidden ones and not following
/// symbolic links, entering only the directories the pattern `pattern`
/// reaches and none of `BUILT` the pattern does not name, counting the
/// entries visited in `entries` and stopping at `CAP` of them. A directory
/// it cannot read it notes in `listing`. Returns whether it saw all there
/// was up to the cap.
fn walk(
    dir: &Path,
    at: &str,
    pattern: &[&str],
    listing: &mut Listing,
    entries: &mut usize,
) -> bool {
    let Ok(entries_in) = fs::read_dir(dir) else {
        let at = if at.is_empty() { "." } else { at };
        listing.unread.push(at.to_string());
        return true;
    };
    let mut entries_in: Vec<(String, fs::DirEntry)> = entries_in
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name().into_string().ok()?;
            (!name.starts_with('.')).then_some((name, entry))
        })
        .collect();
    // The first names up to the room the cap leaves, and one over it so
    // that the cap is seen hit: a huge directory is not sorted whole.
    let room = CAP - *entries + 1;
    let cut = entries_in.len() > room;
    if cut {
        entries_in.select_nth_unstable_by(room, |a, b| a.0.cmp(&b.0));
        entries_in.truncate(room);
    }
    entries_in.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    for (name, entry) in entries_in {
        let name = name.as_str();
        let path = if at.is_empty() {
            name.to_string()
        } else {
            format!("{at}/{name}")
        };
        let Ok(kind) = entry.file_type() else {
            continue;
        };
        let wanted = if kind.is_dir() {
            reaches(pattern, &path.split('/').collect::<Vec<_>>())
                && (!BUILT.contains(&name) || pattern.contains(&name))
        } else {
            kind.is_file()
        };
        if !wanted {
            continue;
        }
        if *entries == CAP {
            return false;
        }
        *entries += 1;
        if kind.is_dir() {
            if !walk(&entry.path(), &path, pattern, listing, entries) {
                return false;
            }
        } else {
            listing.files.push(path);
        }
    }
    !cut
}

/// Whether a file in the directory `parts`, or in one below it, could
/// match the pattern `pattern`, both split at their slashes: whether a
/// walk need enter it.
fn reaches(pattern: &[&str], parts: &[&str]) -> bool {
    match (pattern.first(), parts.first()) {
        (None, _) => false,
        (Some(_), None) => true,
        (Some(&"**"), Some(_)) => reaches(&pattern[1..], parts) || reaches(pattern, &parts[1..]),
        (Some(part), Some(name)) => segment(part, name) && reaches(&pattern[1..], &parts[1..]),
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
    use std::{os::unix::fs::PermissionsExt, path::PathBuf};

    use serde_json::json;

    use super::*;

    /// Runs the call in `arguments` with `dir` as the working directory,
    /// as the model sees the result.
    fn run_in(dir: &Path, arguments: &Value) -> String {
        crate::tools::Output::of(super::run(dir, arguments), false).text
    }

    /// Runs the call in `arguments` in the working directory.
    fn run(arguments: &Value) -> String {
        run_in(Path::new("."), arguments)
    }

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
        let list = |pattern: &str| run_in(&dir, &json!({ "pattern": pattern, "path": at }));
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
        assert_eq!(
            run_in(&dir, &json!({ "pattern": "*.rs", "path": "src" })),
            "src/b.rs\nsrc/deep/c.rs\n[2 files matching *.rs in src]"
        );
        assert!(
            run_in(&dir, &json!({ "pattern": "*.rs", "path": "/nonexistent" }))
                .starts_with("error: ")
        );
        assert!(run_in(&dir, &json!({ "path": at })).starts_with("error: glob needs"));
        // In the working directory itself: this package's.
        assert!(run(&json!({ "pattern": "glob.rs" })).starts_with("tools/glob.rs\n"));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn refuses_a_path_outside_the_working_directory() {
        let dir = tree();
        let at = dir.to_str().unwrap();
        let refused = |path: &str| {
            format!(
                "error: glob searches the working directory and the directories under it; {path} is outside it. Use bash for other places."
            )
        };
        // Above the working directory, by its absolute path or a relative one.
        let src = dir.join("src");
        assert_eq!(
            run_in(&src, &json!({ "pattern": "*.rs", "path": at })),
            refused(at)
        );
        assert_eq!(
            run_in(&src, &json!({ "pattern": "*.rs", "path": ".." })),
            refused("..")
        );
        assert_eq!(
            run_in(&src, &json!({ "pattern": "*.rs", "path": "/" })),
            refused("/")
        );
        assert_eq!(
            run(&json!({ "pattern": "README*", "path": "/" })),
            refused("/")
        );
        // A symbolic link out of it leads outside too.
        std::os::unix::fs::symlink(&dir, src.join("up")).unwrap();
        assert_eq!(
            run_in(&src, &json!({ "pattern": "*.rs", "path": "up" })),
            refused("up")
        );
        // A path that cannot be resolved is not called outside.
        assert!(
            within(NAME, &src, "gone")
                .unwrap_err()
                .starts_with("gone cannot be resolved: ")
        );
        assert!(
            within(NAME, &dir.join("gone"), "deep")
                .unwrap_err()
                .starts_with("deep cannot be resolved: ")
        );
        // The working directory and the directories below it are fine.
        assert_eq!(
            run_in(&src, &json!({ "pattern": "*.rs", "path": "." })),
            "b.rs\ndeep/c.rs\n[2 files matching *.rs]"
        );
        assert_eq!(
            run_in(&src, &json!({ "pattern": "*.rs", "path": "deep" })),
            "deep/c.rs\n[1 file matching *.rs in deep]"
        );
        assert_eq!(
            run_in(&dir, &json!({ "pattern": "*.rs", "path": "src/deep" })),
            "src/deep/c.rs\n[1 file matching *.rs in src/deep]"
        );
        assert_eq!(
            run_in(&dir, &json!({ "pattern": "*.rs", "path": "src/" })),
            "src/b.rs\nsrc/deep/c.rs\n[2 files matching *.rs in src/]"
        );
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
        let list = |pattern: &str| run_in(&dir, &json!({ "pattern": pattern, "path": at }));
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
    fn walks_the_tree_itself_without_git() {
        let dir = tree();
        // A .git that is no repository keeps git from listing the tree,
        // wherever the temporary directory is, so the walk is used.
        fs::write(dir.join(".git"), "not a repository\n").unwrap();
        let list = |pattern: &str| run_in(&dir, &json!({ "pattern": pattern }));
        // A directory that cannot be read is named, and the rest listed.
        let private = dir.join("private");
        fs::create_dir(&private).unwrap();
        fs::write(private.join("p.rs"), "").unwrap();
        fs::set_permissions(&private, fs::Permissions::from_mode(0o000)).unwrap();
        if fs::read_dir(&private).is_err() {
            assert_eq!(
                list("*.rs"),
                "a.rs\nsrc/b.rs\nsrc/deep/c.rs\n[3 files matching *.rs; private could not be read]"
            );
        }
        fs::set_permissions(&private, fs::Permissions::from_mode(0o755)).unwrap();
        fs::remove_dir_all(&private).unwrap();
        // Built directories are left out, unless the pattern names them.
        for file in ["node_modules/m.js", "target/t.rs"] {
            fs::create_dir_all(dir.join(file).parent().unwrap()).unwrap();
            fs::write(dir.join(file), "").unwrap();
        }
        assert_eq!(
            list("*.rs"),
            "a.rs\nsrc/b.rs\nsrc/deep/c.rs\n[3 files matching *.rs]"
        );
        assert_eq!(list("*.js"), "[no files match *.js]");
        assert_eq!(
            list("node_modules/*.js"),
            "node_modules/m.js\n[1 file matching node_modules/*.js]"
        );
        fs::create_dir(dir.join("many")).unwrap();
        for i in 0..CAP + 1000 {
            fs::write(dir.join(format!("many/{i:05}.txt")), "").unwrap();
        }
        // For a name anywhere, the walk visits a.rs, docs, docs/x.md and
        // many in name order, then as many files in many as the cap
        // leaves, and never src.
        assert_eq!(
            list("*.md"),
            format!(
                "docs/x.md\n[1 file matching *.md among the first {CAP} entries walked; narrow the path for the rest]"
            )
        );
        // For a pattern that names its directories, it enters those alone:
        // a.rs and many, then the files in many; or just src.
        let walked = CAP - 2;
        let found = list("many/*.txt");
        let lines: Vec<&str> = found.lines().collect();
        assert_eq!(lines.len(), MOST + 1);
        assert_eq!(lines[0], "many/00000.txt");
        assert_eq!(
            lines[MOST],
            format!(
                "[the first {MOST} of {walked} files matching many/*.txt among the first {CAP} entries walked; narrow the pattern or the path for the rest]"
            )
        );
        assert_eq!(list("src/*.rs"), "src/b.rs\n[1 file matching src/*.rs]");
        assert_eq!(
            list("src/**/*.rs"),
            "src/b.rs\nsrc/deep/c.rs\n[2 files matching src/**/*.rs]"
        );
        // Under the cap, the list is whole again.
        assert_eq!(
            run_in(&dir, &json!({ "pattern": "*.rs", "path": "src" })),
            "src/b.rs\nsrc/deep/c.rs\n[2 files matching *.rs in src]"
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
        let r = |pattern: &str, dir: &str| {
            let pattern: Vec<&str> = pattern.split('/').collect();
            let dir: Vec<&str> = dir.split('/').collect();
            reaches(&pattern, &dir)
        };
        assert!(r("src/*.rs", "src") && !r("src/*.rs", "docs"));
        assert!(!r("src/*.rs", "src/deep") && r("src/**/*.rs", "src/deep"));
        assert!(r("**/*.rs", "src/deep") && r("src/**", "src/deep"));
        assert!(r("*/deep/*.rs", "src") && !r("*/deep/*.rs", "src/shallow"));
    }
}
