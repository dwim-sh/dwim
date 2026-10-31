//! The `edit` tool: replaces one occurrence of a string in a file with
//! another, and shows the lines as they now stand, numbered as `read`
//! numbers them, with how the lines after them moved, so that the model
//! sees its change made without reading the file again. The string to
//! replace is matched as the model copies it from a page: exactly, or but
//! for the white space at the ends of its lines, or but for how far they
//! are indented. A line a page cut matches by its start, and is kept
//! whole: the model has not seen the rest of it, so an edit that would
//! change or drop it is refused, and one that copies it as the page shows
//! it, cut, writes it back whole. A string that matches more than once is
//! refused, with the lines it matches at, and the line it begins at,
//! given, says which. Left to itself, the model
//! edits by an old string and a new, in Python through bash, and reads
//! the file back after each to see that it took: this is that, as a tool,
//! with the reading back in the result.

use std::{borrow::Cow, fs};

use serde_json::Value;

use crate::{number, plural, read, string};

/// The tool as the system prompt declares it.
pub const SIGNATURE: &str = r#"{"type": "function", "function": {"name": "edit", "description": "Replace one occurrence of old_string in a file with new_string, and show the lines as they now stand, numbered, with how later lines moved. Copy old_string from read exactly, enough lines to match once; if it could match more than once, give start, the line it begins at. Empty new_string removes the lines. A line read cut with … is kept whole: copy it, cut, into new_string too.", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "The file to edit."}, "old_string": {"type": "string", "description": "The lines to replace, as read shows them, without their numbers."}, "new_string": {"type": "string", "description": "The lines to put in their place."}, "start": {"type": "integer", "description": "The line old_string begins at, counting from 1, to say which occurrence when it matches more than once. Optional."}}, "required": ["path", "old_string", "new_string"]}}}"#;

/// Lines around an edit shown on each side of it.
const CONTEXT: usize = 2;

/// Most lines of a range shown: its start and end, half each, past this.
const SHOWN: usize = 40;

/// Makes the edit in `arguments`, returning what the model should see.
pub fn run(arguments: &Value) -> String {
    let Some(path) = string(arguments, "path") else {
        return "error: edit needs a 'path' string".to_string();
    };
    let Some(old) = string(arguments, "old_string") else {
        return "error: edit needs an 'old_string', the lines to replace".to_string();
    };
    let Some(new) = string(arguments, "new_string") else {
        return "error: edit needs a 'new_string', the lines to put in their place".to_string();
    };
    if old.trim().is_empty() {
        return "error: old_string is empty; to add lines, give the line before them as old_string and that line with the new ones as new_string".to_string();
    }
    if old == new {
        return "error: old_string and new_string are the same; nothing changed".to_string();
    }
    let start = match arguments.get("start") {
        None => None,
        Some(_) => match number(arguments, "start") {
            Some(start) => Some(start as usize),
            None => return "error: 'start' is a line number".to_string(),
        },
    };
    let content = match fs::read(path) {
        Ok(content) => content,
        Err(e) => return format!("error: cannot read {path}: {e}"),
    };
    let lines: Vec<&[u8]> = content.split_inclusive(|&byte| byte == b'\n').collect();
    let count = lines.len();
    let texts: Vec<Cow<str>> = lines
        .iter()
        .map(|line| String::from_utf8_lossy(read::content(line)))
        .collect();
    let wanted = split(old);
    let found = find(&texts, &wanted);
    let at = match (found.as_slice(), start) {
        ([], _) => {
            let around = match start {
                Some(start) if start >= 1 && start <= count => format!(
                    ". The lines around {start} stand as:\n{}",
                    numbered(
                        &lines,
                        start.saturating_sub(CONTEXT + 1),
                        (start + wanted.len() + CONTEXT - 1).min(count)
                    )
                ),
                _ => "; read the lines and copy them".to_string(),
            };
            return format!("error: old_string not found in {path}; nothing changed{around}");
        }
        ([at], _) => *at,
        (many, Some(start)) => match many.iter().find(|&&at| at + 1 == start) {
            Some(&at) => at,
            None => {
                return format!(
                    "error: old_string matches {} times in {path}, at lines {}, and none begins at line {start}; nothing changed",
                    many.len(),
                    list(many)
                );
            }
        },
        (many, None) => {
            return format!(
                "error: old_string matches {} times in {path}, at lines {}; give start, the line of the one you mean; nothing changed",
                many.len(),
                list(many)
            );
        }
    };
    let (first, last) = (at + 1, at + wanted.len());
    let new = match whole(&lines[at..last], &wanted, new) {
        Ok(new) => new,
        Err(line) => {
            return format!(
                "error: line {} of {path} is longer than read shows, cut at {} bytes, and old_string has it cut, so you have not seen all of it; to keep it, put it in new_string as read shows it, with its …; to change it, use bash; nothing changed",
                at + line + 1,
                read::LINE
            );
        }
    };

    let updated = replaced(&lines, first, last, &new);
    if let Err(e) = fs::write(path, &updated) {
        return format!("error: cannot write {path}: {e}");
    }
    let now: Vec<&[u8]> = updated.split_inclusive(|&byte| byte == b'\n').collect();
    let added = now.len() + last + 1 - first - count;
    let shown = numbered(
        &now,
        (first - 1).saturating_sub(CONTEXT),
        (first - 1 + added + CONTEXT).min(now.len()),
    );
    let what = match added {
        0 => format!("lines {first}–{last} removed"),
        added => format!(
            "lines {first}–{last} replaced by {added} {}",
            plural(added as u64, "line")
        ),
    };
    let moved = if last < count && first + added != last + 1 {
        format!("; line {} is now line {}", last + 1, first + added)
    } else {
        String::new()
    };
    format!(
        "{shown}\n[{path}: {what}{moved}; {} {} in all]",
        now.len(),
        plural(now.len() as u64, "line")
    )
}

/// The lines of `text`, without their endings.
fn split(text: &str) -> Vec<&str> {
    text.strip_suffix('\n')
        .unwrap_or(text)
        .split('\n')
        .map(|line| line.strip_suffix('\r').unwrap_or(line))
        .collect()
}

/// How closely a wanted line has to match a line of the file: exactly;
/// or but for the white space at the end; or but for the white space at
/// both ends, so that a string copied with its indentation lost still
/// matches.
#[derive(Clone, Copy)]
enum Tier {
    Exact,
    TrailingSpace,
    Trimmed,
}

impl Tier {
    /// `text` with the white space the tier disregards taken off.
    fn trim(self, text: &str) -> &str {
        match self {
            Tier::Exact => text,
            Tier::TrailingSpace => text.trim_end(),
            Tier::Trimmed => text.trim(),
        }
    }
}

/// Where `wanted` begins in `lines`, the lines of the file without their
/// endings, as indexes from zero, at the closest tier that matches it
/// anywhere.
fn find(lines: &[Cow<str>], wanted: &[&str]) -> Vec<usize> {
    if wanted.len() > lines.len() {
        return Vec::new();
    }
    for tier in [Tier::Exact, Tier::TrailingSpace, Tier::Trimmed] {
        let found: Vec<usize> = (0..=lines.len() - wanted.len())
            .filter(|&at| {
                wanted
                    .iter()
                    .enumerate()
                    .all(|(k, want)| matches(want, &lines[at + k], tier))
            })
            .collect();
        if !found.is_empty() {
            return found;
        }
    }
    Vec::new()
}

/// Whether `want`, a line as the model gives it, is `line`, a line of the
/// file without its ending, at `tier`; or its start, if `want` is a line
/// `read` cut.
fn matches(want: &str, line: &str, tier: Tier) -> bool {
    // Whether read cut the line, as it stands, before any trimming.
    let long = line.len() > read::LINE;
    tier.trim(want) == tier.trim(line)
        || want
            .strip_suffix('…')
            .is_some_and(|head| long && tier.trim(line).starts_with(tier.trim(head)))
}

/// The lines of `text`, the new string, as they go into the file in place
/// of `old`, the lines matched by `wanted`. A line of `old` that `read`
/// cut, matched by its start, is one the model has not seen whole: a line
/// of `text` that is it as `read` shows it, cut, is that line whole, and
/// if no line of `text` is, the edit would lose what the model has not
/// seen, so the index of the first such line is the error.
fn whole(old: &[&[u8]], wanted: &[&str], text: &str) -> Result<Vec<Vec<u8>>, usize> {
    let cut: Vec<(usize, String)> = old
        .iter()
        .zip(wanted)
        .enumerate()
        .filter_map(|(k, (line, want))| {
            let (shown, long) = read::shown_line(line);
            (long && want.trim_end().ends_with('…')).then_some((k, shown))
        })
        .collect();
    let mut kept = vec![false; cut.len()];
    let mut new = Vec::new();
    if !text.is_empty() {
        for line in split(text) {
            let same = cut
                .iter()
                .position(|(_, shown)| line.ends_with('…') && shown.trim() == line.trim());
            match same {
                Some(i) => {
                    kept[i] = true;
                    new.push(read::content(old[cut[i].0]).to_vec());
                }
                None => new.push(line.as_bytes().to_vec()),
            }
        }
    }
    match kept.iter().position(|&kept| !kept) {
        Some(i) => Err(cut[i].0),
        None => Ok(new),
    }
}

/// Line numbers from indexes, as a list.
fn list(at: &[usize]) -> String {
    at.iter()
        .map(|at| (at + 1).to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

/// The file of `lines` with lines `start` through `end`, counting from 1,
/// replaced by `new`, lines without their endings. The new lines end as
/// the file's first line does, and the file ends in a newline if it did.
fn replaced(lines: &[&[u8]], start: usize, end: usize, new: &[Vec<u8>]) -> Vec<u8> {
    let eol: &[u8] = match lines.first() {
        Some(line) if line.ends_with(b"\r\n") => b"\r\n",
        _ => b"\n",
    };
    let unterminated = lines.last().is_some_and(|line| !line.ends_with(b"\n"));
    let mut updated = lines[..start - 1].concat();
    for line in new {
        updated.extend_from_slice(line);
        updated.extend_from_slice(eol);
    }
    updated.extend_from_slice(&lines[end..].concat());
    if unterminated && end == lines.len() {
        let without = updated.len() - eol.len().min(updated.len());
        if updated[without..] == *eol {
            updated.truncate(without);
        }
    }
    updated
}

/// The lines of `lines` after the first `from`, up to line `to`, each after
/// its number and a tab, as `read` shows them; only the start and the end
/// of a range longer than [`SHOWN`].
fn numbered(lines: &[&[u8]], from: usize, to: usize) -> String {
    let show = |range: std::ops::Range<usize>| {
        range
            .map(|i| format!("{}\t{}", i + 1, read::shown_line(lines[i]).0))
            .collect::<Vec<_>>()
    };
    let mut shown = if to - from > SHOWN {
        let mut shown = show(from..from + SHOWN / 2);
        shown.push(format!("[{} lines more]", to - from - SHOWN));
        shown.extend(show(to - SHOWN / 2..to));
        shown
    } else {
        show(from..to)
    };
    if shown.is_empty() {
        shown.push("[no lines]".to_string());
    }
    shown.join("\n")
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use serde_json::json;

    use super::*;

    /// A file with `text` that goes when it is dropped.
    struct File(PathBuf);

    impl File {
        fn new(text: &str) -> Self {
            static FILES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let n = FILES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!("dwim-edit-{}-{n}", std::process::id()));
            fs::write(&path, text).unwrap();
            Self(path)
        }

        fn path(&self) -> &str {
            self.0.to_str().unwrap()
        }

        fn text(&self) -> String {
            fs::read_to_string(&self.0).unwrap()
        }

        /// Runs the tool on the file.
        fn edit(&self, old: &str, new: &str) -> String {
            run(&json!({ "path": self.path(), "old_string": old, "new_string": new }))
        }

        /// Runs the tool on the file, saying where the old string begins.
        fn edit_at(&self, old: &str, new: &str, start: u64) -> String {
            run(&json!({
                "path": self.path(),
                "old_string": old,
                "new_string": new,
                "start": start,
            }))
        }
    }

    impl Drop for File {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.0);
        }
    }

    const FIVE: &str = "one\ntwo\nthree\nfour\nfive\n";

    #[test]
    fn replaces_a_string_and_shows_the_lines() {
        let file = File::new(FIVE);
        let result = file.edit("two\nthree\n", "2\n3\n3b\n");
        assert_eq!(file.text(), "one\n2\n3\n3b\nfour\nfive\n");
        assert_eq!(
            result,
            format!(
                "1\tone\n2\t2\n3\t3\n4\t3b\n5\tfour\n6\tfive\n\
                 [{}: lines 2–3 replaced by 3 lines; line 4 is now line 5; 6 lines in all]",
                file.path()
            )
        );
        // Without a trailing newline on either string, the same.
        let file = File::new(FIVE);
        file.edit("two\nthree", "2\n3\n3b");
        assert_eq!(file.text(), "one\n2\n3\n3b\nfour\nfive\n");
    }

    #[test]
    fn says_nothing_of_lines_that_did_not_move() {
        let file = File::new(FIVE);
        let result = file.edit("three", "3");
        assert_eq!(file.text(), "one\ntwo\n3\nfour\nfive\n");
        assert!(
            result.ends_with(&format!(
                "[{}: lines 3–3 replaced by 1 line; 5 lines in all]",
                file.path()
            )),
            "{result}"
        );
    }

    #[test]
    fn adds_and_removes_lines() {
        // Lines are added by giving the line before them and that line
        // with them.
        let file = File::new(FIVE);
        file.edit("two", "two\ntwo and a half");
        assert_eq!(file.text(), "one\ntwo\ntwo and a half\nthree\nfour\nfive\n");
        // And removed by an empty new string.
        let file = File::new(FIVE);
        let result = file.edit("two\nthree\n", "");
        assert_eq!(file.text(), "one\nfour\nfive\n");
        assert!(
            result.ends_with(&format!(
                "[{}: lines 2–3 removed; line 4 is now line 2; 3 lines in all]",
                file.path()
            )),
            "{result}"
        );
        // The last lines, in a file without a final newline.
        let file = File::new("one\ntwo\nthree");
        file.edit("three", "3");
        assert_eq!(file.text(), "one\ntwo\n3");
    }

    #[test]
    fn matches_past_white_space_and_cut_lines() {
        // Trailing white space in the file does not matter.
        let file = File::new("a  \n  b\t\nc\n");
        assert!(!file.edit("a\n  b", "x").starts_with("error"));
        assert_eq!(file.text(), "x\nc\n");
        // Nor the indentation, if nothing matches with it.
        let file = File::new("    fn f() {\n        1\n    }\n");
        assert!(
            !file
                .edit("fn f() {\n1", "    fn g() {\n        2")
                .starts_with("error")
        );
        assert_eq!(file.text(), "    fn g() {\n        2\n    }\n");
    }

    #[test]
    fn matches_a_cut_line_by_its_start_and_keeps_it_whole() {
        let long = "y".repeat(400);
        let file = File::new(&format!("short\n{long}\nend\n"));
        let cut = format!("{}…", "y".repeat(read::LINE));
        // Copied cut into new_string, the line goes back whole.
        let result = file.edit(&format!("short\n{cut}"), &format!("brief\n{cut}"));
        assert!(result.starts_with("1\tbrief\n2\t"), "{result}");
        assert_eq!(file.text(), format!("brief\n{long}\nend\n"));
        // Changed or dropped, the edit would lose what the model has not
        // seen, so it is refused.
        let tweaked = format!("{}…", "z".repeat(read::LINE));
        for new in [tweaked.as_str(), "replaced", ""] {
            let result = file.edit(&cut, new);
            assert_eq!(
                result,
                format!(
                    "error: line 2 of {} is longer than read shows, cut at 300 bytes, and old_string has it cut, so you have not seen all of it; to keep it, put it in new_string as read shows it, with its …; to change it, use bash; nothing changed",
                    file.path()
                )
            );
            assert_eq!(file.text(), format!("brief\n{long}\nend\n"));
        }
        // A cut line with white space at its end matches at every tier.
        let spaced = format!("{}   ", "y".repeat(read::LINE - 1));
        let file = File::new(&format!("a  \n{spaced}\nend\n"));
        let cut = format!("{} …", "y".repeat(read::LINE - 1));
        let result = file.edit(&format!("a\n{cut}"), &format!("b\n{cut}"));
        assert!(result.starts_with("1\tb\n2\t"), "{result}");
        assert_eq!(file.text(), format!("b\n{spaced}\nend\n"));
        // A line that ends in … in the file is just a line.
        let file = File::new("so on…\nend\n");
        file.edit("so on…", "and so on…");
        assert_eq!(file.text(), "and so on…\nend\n");
        // And a long line given whole was seen whole.
        let file = File::new(&format!("short\n{long}\nend\n"));
        file.edit(&long, "replaced");
        assert_eq!(file.text(), "short\nreplaced\nend\n");
    }

    #[test]
    fn refuses_what_it_does_not_find() {
        let file = File::new(FIVE);
        let result = file.edit("six", "6");
        assert_eq!(file.text(), FIVE);
        assert_eq!(
            result,
            format!(
                "error: old_string not found in {}; nothing changed; read the lines and copy them",
                file.path()
            )
        );
        // With a start, the lines there are shown.
        let result = file.edit_at("six", "6", 4);
        assert_eq!(
            result,
            format!(
                "error: old_string not found in {}; nothing changed. The lines around 4 stand as:\n2\ttwo\n3\tthree\n4\tfour\n5\tfive",
                file.path()
            )
        );
        assert_eq!(file.text(), FIVE);
    }

    #[test]
    fn refuses_a_string_that_matches_twice_unless_told_where() {
        let file = File::new("a\nb\na\nb\nc\n");
        let result = file.edit("a\nb", "x");
        assert_eq!(
            result,
            format!(
                "error: old_string matches 2 times in {}, at lines 1, 3; give start, the line of the one you mean; nothing changed",
                file.path()
            )
        );
        assert_eq!(file.text(), "a\nb\na\nb\nc\n");
        let result = file.edit_at("a\nb", "x", 5);
        assert!(result.contains("none begins at line 5"), "{result}");
        assert_eq!(file.text(), "a\nb\na\nb\nc\n");
        let result = file.edit_at("a\nb", "x", 3);
        assert!(result.starts_with("1\ta\n2\tb\n3\tx\n4\tc\n"), "{result}");
        assert_eq!(file.text(), "a\nb\nx\nc\n");
        // A wrong start does not matter when the string matches once.
        let file = File::new(FIVE);
        assert!(!file.edit_at("three", "3", 1).starts_with("error"));
        assert_eq!(file.text(), "one\ntwo\n3\nfour\nfive\n");
    }

    #[test]
    fn keeps_how_the_file_ends_its_lines() {
        let file = File::new("one\r\ntwo\r\nthree\r\n");
        file.edit("two", "2\n2b");
        assert_eq!(file.text(), "one\r\n2\r\n2b\r\nthree\r\n");
    }

    #[test]
    fn needs_its_arguments_and_a_change() {
        let file = File::new(FIVE);
        assert!(run(&json!({ "old_string": "a", "new_string": "b" })).contains("'path'"));
        assert!(run(&json!({ "path": file.path(), "new_string": "b" })).contains("'old_string'"));
        assert!(run(&json!({ "path": file.path(), "old_string": "a" })).contains("'new_string'"));
        assert!(file.edit("", "b").contains("empty"));
        assert!(file.edit("two", "two").contains("the same"));
        assert_eq!(
            run(
                &json!({ "path": file.path(), "old_string": "two", "new_string": "2", "start": "x" })
            ),
            "error: 'start' is a line number"
        );
        assert_eq!(file.text(), FIVE);
        assert!(file.edit("two", "2").starts_with("1\tone"));
        assert!(
            run(&json!({ "path": "/nonexistent/file", "old_string": "a", "new_string": "b" }))
                .starts_with("error: cannot read /nonexistent/file: ")
        );
    }

    #[test]
    fn shows_the_ends_of_a_long_edit() {
        let text: String = (1..=10).map(|i| format!("line {i}\n")).collect();
        let file = File::new(&text);
        let new: String = (1..=100).map(|i| format!("new {i}\n")).collect();
        let result = file.edit("line 5", &new);
        assert!(result.contains("[64 lines more]"), "{result}");
        assert!(
            result.starts_with("3\tline 3\n4\tline 4\n5\tnew 1\n"),
            "{result}"
        );
        assert!(
            result.contains(&format!(
                "[{}: lines 5–5 replaced by 100 lines; line 6 is now line 105; 109 lines in all]",
                file.path()
            )),
            "{result}"
        );
    }
}
