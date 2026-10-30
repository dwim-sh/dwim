//! The `edit` tool: replaces lines of a file by their numbers, the numbers
//! `read` gives, and shows them as they now stand. It takes the line the
//! range starts at as it should stand now, so an edit from a line number
//! that has gone stale is refused, with the lines as they are, rather than
//! made in the wrong place.

use std::fs;

use serde_json::Value;

use crate::{number, plural, read, string};

/// The tool as the system prompt declares it.
pub const SIGNATURE: &str = r#"{"type": "function", "function": {"name": "edit", "description": "Replace lines of a file by their line numbers, as read numbers them. The result shows the lines as they now stand, and says how later lines moved. If the file's line at start is not the first you give, nothing changes and the result shows the lines as they are.", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "The file to edit."}, "start": {"type": "integer", "description": "The first line to replace, counting from 1."}, "end": {"type": "integer", "description": "The last line to replace. To add lines before start without replacing any, start - 1."}, "first": {"type": "string", "description": "The line at start as it stands now, without its number. Empty when adding lines past the end of the file."}, "text": {"type": "string", "description": "The lines to put in their place. Empty to remove them."}}, "required": ["path", "start", "end", "first", "text"]}}}"#;

/// Lines around an edit shown on each side of it.
const CONTEXT: usize = 2;

/// Most lines of a range shown: its start and end, half each, past this.
const SHOWN: usize = 40;

/// Makes the edit in `arguments`, returning what the model should see.
pub fn run(arguments: &Value) -> String {
    let Some(path) = string(arguments, "path") else {
        return "error: edit needs a 'path' string".to_string();
    };
    let (Some(start), Some(end)) = (number(arguments, "start"), number(arguments, "end")) else {
        return "error: edit needs 'start' and 'end' line numbers".to_string();
    };
    let Some(first) = string(arguments, "first") else {
        return "error: edit needs a 'first' string, the line at start as it stands now"
            .to_string();
    };
    let Some(text) = string(arguments, "text") else {
        return "error: edit needs a 'text' string".to_string();
    };
    let content = match fs::read(path) {
        Ok(content) => content,
        Err(e) => return format!("error: cannot read {path}: {e}"),
    };
    let lines: Vec<&[u8]> = content.split_inclusive(|&byte| byte == b'\n').collect();
    let count = lines.len();
    let (start, end) = (start as usize, end as usize);
    if start == 0 || start > count + 1 || end + 1 < start || end > count {
        return format!(
            "error: no lines {start}–{end} in {path}, which has {count} {}; nothing changed",
            plural(count as u64, "line")
        );
    }
    if let Some(line) = lines.get(start - 1)
        && !same(first, line)
    {
        return format!(
            "error: line {start} of {path} is not {first:?}; nothing changed. The lines stand as:\n{}",
            numbered(
                &lines,
                start.saturating_sub(CONTEXT + 1),
                (end + CONTEXT).min(count)
            )
        );
    }

    let updated = replaced(&lines, start, end, text);
    if let Err(e) = fs::write(path, &updated) {
        return format!("error: cannot write {path}: {e}");
    }
    let new: Vec<&[u8]> = updated.split_inclusive(|&byte| byte == b'\n').collect();
    let added = new.len() + end + 1 - start - count;
    let shown = numbered(
        &new,
        (start - 1).saturating_sub(CONTEXT),
        (start - 1 + added + CONTEXT).min(new.len()),
    );
    let what = match (end + 1 - start, added) {
        (0, 0) => "nothing changed".to_string(),
        (0, added) => format!(
            "{added} {} added before line {start}",
            plural(added as u64, "line")
        ),
        (_, 0) => format!("lines {start}–{end} removed"),
        (_, added) => format!(
            "lines {start}–{end} replaced by {added} {}",
            plural(added as u64, "line")
        ),
    };
    let moved = if end < count && start + added != end + 1 {
        format!("; line {} is now line {}", end + 1, start + added)
    } else {
        String::new()
    };
    format!(
        "{shown}\n[{path}: {what}{moved}; {} {} in all]",
        new.len(),
        plural(new.len() as u64, "line")
    )
}

/// Whether `first` is `line`, past the white space around them, or its
/// start if `first` is a line `read` cut.
fn same(first: &str, line: &[u8]) -> bool {
    let line = String::from_utf8_lossy(line);
    let (first, line) = (first.trim(), line.trim());
    first == line
        || first
            .strip_suffix('…')
            .is_some_and(|start| line.len() > read::LINE && line.starts_with(start))
}

/// The file of `lines` with lines `start` through `end`, counting from 1,
/// replaced by those of `text`. The new lines end as the file's first line
/// does, and the file ends in a newline if it did.
fn replaced(lines: &[&[u8]], start: usize, end: usize, text: &str) -> Vec<u8> {
    let eol: &[u8] = match lines.first() {
        Some(line) if line.ends_with(b"\r\n") => b"\r\n",
        _ => b"\n",
    };
    let new: Vec<&str> = if text.is_empty() {
        Vec::new()
    } else {
        text.strip_suffix('\n')
            .unwrap_or(text)
            .split('\n')
            .collect()
    };
    let unterminated = lines.last().is_some_and(|line| !line.ends_with(b"\n"));
    let mut updated = lines[..start - 1].concat();
    if !new.is_empty() && !updated.is_empty() && !updated.ends_with(b"\n") {
        updated.extend_from_slice(eol);
    }
    for line in &new {
        updated.extend_from_slice(line.strip_suffix('\r').unwrap_or(line).as_bytes());
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
        fn edit(&self, start: u64, end: u64, first: &str, text: &str) -> String {
            run(&json!({
                "path": self.path(),
                "start": start,
                "end": end,
                "first": first,
                "text": text,
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
    fn replaces_lines_and_shows_them() {
        let file = File::new(FIVE);
        let result = file.edit(2, 3, "two", "TWO\nTWO AND A HALF\nTHREE");
        assert_eq!(file.text(), "one\nTWO\nTWO AND A HALF\nTHREE\nfour\nfive\n");
        assert_eq!(
            result,
            format!(
                "1\tone\n2\tTWO\n3\tTWO AND A HALF\n4\tTHREE\n5\tfour\n6\tfive\n\
                 [{}: lines 2–3 replaced by 3 lines; line 4 is now line 5; 6 lines in all]",
                file.path()
            )
        );
    }

    #[test]
    fn says_nothing_of_lines_that_did_not_move() {
        let file = File::new(FIVE);
        let result = file.edit(2, 2, "two", "TWO");
        assert!(
            result.ends_with(&format!(
                "[{}: lines 2–2 replaced by 1 line; 5 lines in all]",
                file.path()
            )),
            "{result}"
        );
    }

    #[test]
    fn adds_and_removes_lines() {
        let file = File::new(FIVE);
        let result = file.edit(2, 1, "two", "new\n");
        assert_eq!(file.text(), "one\nnew\ntwo\nthree\nfour\nfive\n");
        assert!(
            result.contains("1 line added before line 2; line 2 is now line 3"),
            "{result}"
        );

        let file = File::new(FIVE);
        let result = file.edit(2, 4, "two", "");
        assert_eq!(file.text(), "one\nfive\n");
        assert!(
            result.contains("lines 2–4 removed; line 5 is now line 2"),
            "{result}"
        );

        // Past the end of the file, and into an empty one.
        let file = File::new(FIVE);
        file.edit(6, 5, "", "six");
        assert_eq!(file.text(), format!("{FIVE}six\n"));
        let file = File::new("");
        file.edit(1, 0, "", "one");
        assert_eq!(file.text(), "one\n");
    }

    #[test]
    fn keeps_how_the_file_ends_its_lines() {
        // A last line without a newline stays without one.
        let file = File::new("one\ntwo");
        file.edit(2, 2, "two", "TWO");
        assert_eq!(file.text(), "one\nTWO");
        // Unless lines follow it.
        let file = File::new("one\ntwo");
        file.edit(3, 2, "", "three");
        assert_eq!(file.text(), "one\ntwo\nthree");
        let file = File::new("one\ntwo\n");
        file.edit(1, 1, "one", "ONE");
        assert_eq!(file.text(), "ONE\ntwo\n");
        // Carriage returns stay.
        let file = File::new("one\r\ntwo\r\n");
        file.edit(1, 1, "one", "ONE\nUNO");
        assert_eq!(file.text(), "ONE\r\nUNO\r\ntwo\r\n");
    }

    #[test]
    fn refuses_a_line_number_gone_stale() {
        let file = File::new(FIVE);
        let result = file.edit(2, 3, "three", "x");
        assert_eq!(file.text(), FIVE);
        assert_eq!(
            result,
            format!(
                "error: line 2 of {} is not \"three\"; nothing changed. The lines stand as:\n\
                 1\tone\n2\ttwo\n3\tthree\n4\tfour\n5\tfive",
                file.path()
            )
        );
        // White space around the line does not matter.
        assert!(!file.edit(2, 2, "  two ", "2").starts_with("error"));
    }

    #[test]
    fn takes_a_long_line_as_read_cut_it() {
        let long = "x".repeat(read::LINE + 10);
        let file = File::new(&format!("{long}\ntwo\n"));
        let cut = format!("{}…", "x".repeat(read::LINE));
        assert!(!file.edit(1, 1, &cut, "short").starts_with("error"));
        assert_eq!(file.text(), "short\ntwo\n");
        // But a line that is not long is not taken by its start.
        let file = File::new("one\n");
        assert!(file.edit(1, 1, "o…", "x").starts_with("error"));
    }

    #[test]
    fn refuses_lines_that_are_not_there() {
        let file = File::new(FIVE);
        for (start, end) in [(0, 1), (7, 7), (3, 1), (2, 6)] {
            let result = file.edit(start, end, "two", "x");
            assert!(result.starts_with("error: no lines"), "{result}");
        }
        assert_eq!(file.text(), FIVE);
        let result = run(&json!({
            "path": "/nonexistent/file", "start": 1, "end": 1, "first": "", "text": "",
        }));
        assert!(
            result.starts_with("error: cannot read /nonexistent/file: "),
            "{result}"
        );
    }

    #[test]
    fn needs_all_its_arguments() {
        let path = "/nonexistent/file";
        assert!(run(&json!({ "start": 1, "end": 1, "first": "", "text": "" })).contains("'path'"));
        assert!(
            run(&json!({ "path": path, "end": 1, "first": "", "text": "" })).contains("'start'")
        );
        assert!(
            run(&json!({ "path": path, "start": 1, "end": 1, "text": "" })).contains("'first'")
        );
        assert!(
            run(&json!({ "path": path, "start": 1, "end": 1, "first": "" })).contains("'text'")
        );
    }

    #[test]
    fn takes_numbers_written_as_strings() {
        // As the `<function=…>` call format writes every argument.
        let file = File::new(FIVE);
        let result = run(&json!({
            "path": file.path(), "start": "2", "end": "2", "first": "two", "text": "2",
        }));
        assert!(!result.starts_with("error"), "{result}");
        assert_eq!(file.text(), "one\n2\nthree\nfour\nfive\n");
    }

    #[test]
    fn shows_the_ends_of_a_long_edit() {
        let file = File::new(FIVE);
        let text: String = (0..100).map(|i| format!("new {i}\n")).collect();
        let result = file.edit(2, 2, "two", &text);
        assert_eq!(file.text().lines().count(), 104);
        assert!(result.contains("\n2\tnew 0\n"), "{result}");
        assert!(result.contains("lines more]"), "{result}");
        assert!(result.contains("\n101\tnew 99\n"), "{result}");
        assert!(!result.contains("new 50"), "{result}");
    }
}
