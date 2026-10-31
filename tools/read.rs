//! The `read` tool: a file, a page of numbered lines at a time, with
//! where the next page starts. The page is fixed and the model is told
//! its size, so it knows what it will get before it asks and never has to
//! guess at a range: it reads the next page, or starts from a line.
//!
//! A file longer than a page, read without a start line, comes as an
//! outline instead of its first page: the lines that define something, a
//! function, a type, a heading, with their numbers, so that the model
//! picks the line it needs and reads from there rather than paging
//! through the file from the top. Given the prompt alone, it paged: on a
//! run it read three files of 1300 lines end to end, a page at a time,
//! before it planned anything, and the outline makes the jump the easy
//! path rather than the next page.

use std::{
    fs::File,
    io::{self, BufRead, BufReader},
};

use serde_json::Value;

use crate::{number, plural, string};

/// Most lines in a page.
pub const PAGE: usize = 200;

/// Most bytes in a page: a page of long lines ends early, so that one
/// page can't take up much of the context.
pub const BYTES: usize = 16 << 10;

/// Most bytes of a line shown; a longer line is cut there, with a mark.
pub const LINE: usize = 300;

/// Most bytes of an outline entry shown; a longer line is cut there.
pub const ENTRY: usize = 120;

/// The tool as the system prompt declares it.
pub const SIGNATURE: &str = r#"{"type": "function", "function": {"name": "read", "description": "Read a file from a line, a page of up to 200 numbered lines (or 16 KB) at a time. The result says how many lines the file has and where the next page starts. Without a start line, a file longer than a page comes as an outline instead: its definitions with their line numbers, to pick the line to read from.", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "The file to read."}, "start": {"type": "integer", "description": "The line to start from, counting from 1. Left out, the outline of a long file, or the whole of a short one."}}, "required": ["path"]}}}"#;

/// Reads the page of the file in `arguments`, returning what the model
/// should see.
pub fn run(arguments: &Value) -> String {
    let Some(path) = string(arguments, "path") else {
        return "error: read needs a 'path' string".to_string();
    };
    let start = match arguments.get("start") {
        None => None,
        Some(_) => match start(arguments) {
            Some(start) => Some(start.max(1)),
            None => return "error: 'start' is a line number".to_string(),
        },
    };
    let open = || File::open(path).map(BufReader::new);
    let file = match open() {
        Ok(file) => file,
        Err(e) => return format!("error: cannot read {path}: {e}"),
    };
    let directory = file
        .get_ref()
        .metadata()
        .is_ok_and(|metadata| metadata.is_dir());
    if directory {
        return format!("error: {path} is a directory; list it with ls");
    }
    let result = match start {
        Some(start) => page(file, path, start),
        // The outline of a long file, or the first page of one that has
        // nothing to outline, read again from the start.
        None => outline(file, path).and_then(|outline| match outline {
            Some(text) => Ok(text),
            None => page(open()?, path, 1),
        }),
    };
    match result {
        Ok(text) => text,
        Err(e) => format!("error: reading {path}: {e}"),
    }
}

/// The `start` argument as a line number, written as a number or, as the
/// `<function=…>` call format writes every argument, as a string.
pub fn start(arguments: &Value) -> Option<u64> {
    number(arguments, "start")
}

/// The line `line` as a page shows it, without its line ending, cut at
/// [`LINE`] bytes with a mark, and whether it was cut.
pub fn shown_line(line: &[u8]) -> (String, bool) {
    let content = line.strip_suffix(b"\n").unwrap_or(line);
    let content = content.strip_suffix(b"\r").unwrap_or(content);
    let mut content = String::from_utf8_lossy(content).into_owned();
    let long = content.len() > LINE;
    if long {
        let end = (0..=LINE)
            .rev()
            .find(|&i| content.is_char_boundary(i))
            .unwrap_or(0);
        content.truncate(end);
        content.push('…');
    }
    (content, long)
}

/// Words that open a definition, after any modifiers: what the outline
/// of a file lists.
const DEFINITIONS: &[&str] = &[
    "fn",
    "struct",
    "enum",
    "trait",
    "impl",
    "mod",
    "const",
    "static",
    "type",
    "macro_rules",
    "def",
    "class",
    "function",
    "func",
    "interface",
    "object",
    "protocol",
    "extension",
    "namespace",
    "module",
    "record",
    "union",
];

/// Words that may come before a definition's own word.
const MODIFIERS: &[&str] = &[
    "pub",
    "export",
    "default",
    "async",
    "unsafe",
    "extern",
    "abstract",
    "final",
    "public",
    "private",
    "protected",
    "internal",
    "override",
    "declare",
    "inline",
];

/// Words that open a statement rather than a definition, so that a line
/// at the margin that ends in a brace is not taken for one.
const STATEMENTS: &[&str] = &[
    "if", "else", "for", "while", "do", "switch", "match", "loop", "try", "catch", "return", "let",
    "var", "with",
];

/// The outline of `file`: the lines that define something, each after its
/// number and a tab, up to [`PAGE`] of them or [`BYTES`] in all, between a
/// line saying what it is and how to read the code, and one saying what was
/// left out if the outline was cut. `None` for a file of a page or less,
/// which is read whole instead, and for one with nothing to outline.
///
/// A line defines something if it sits at the margin or one indent in and,
/// after modifiers such as `pub` or `export`, opens with a definition's
/// word such as `fn`, `class` or `def`; or, at the margin, calls something
/// and ends in an open brace, as a C function does; or, in Markdown, is a
/// heading. One indent in lists the methods of a type and the functions
/// of a test module, but not the statements of a function's body.
fn outline(mut file: impl BufRead, path: &str) -> io::Result<Option<String>> {
    let markdown = path.ends_with(".md") || path.ends_with(".markdown");
    let mut entries = String::new();
    let mut line = Vec::new();
    // Lines read so far, entries shown, the line of the last one shown,
    // and definitions found past the entries that fit.
    let (mut lines, mut shown, mut last, mut more) = (0u64, 0usize, 0u64, 0u64);
    loop {
        line.clear();
        if file.read_until(b'\n', &mut line)? == 0 {
            break;
        }
        lines += 1;
        let content = String::from_utf8_lossy(&line);
        let content = content.trim_end();
        if !defines(content, markdown) {
            continue;
        }
        let mut content = content.to_string();
        if content.len() > ENTRY {
            let end = (0..=ENTRY)
                .rev()
                .find(|&i| content.is_char_boundary(i))
                .unwrap_or(0);
            content.truncate(end);
            content.push('…');
        }
        let entry = format!("{lines}\t{content}\n");
        if shown == PAGE || entries.len() + entry.len() > BYTES {
            more += 1;
            continue;
        }
        entries.push_str(&entry);
        shown += 1;
        last = lines;
    }
    if lines <= PAGE as u64 || shown == 0 {
        return Ok(None);
    }
    let mut text = format!(
        "[{path} has {lines} lines; its definitions follow, with their line numbers. Read {path} from a line to see the code there.]\n{entries}"
    );
    if more > 0 {
        text.push_str(&format!(
            "[… and {more} more past line {last}; read {path} from line {} for them]",
            last + 1
        ));
    } else {
        text.push_str(&format!(
            "[end of outline; read {path} from a line to see the code]"
        ));
    }
    Ok(Some(text))
}

/// Whether `line` defines something, as the outline lists.
fn defines(line: &str, markdown: bool) -> bool {
    let trimmed = line.trim_start();
    if trimmed.is_empty() {
        return false;
    }
    let indent = line.len() - trimmed.len();
    let margin = indent == 0;
    if indent > 4 || (indent > 1 && line.starts_with('\t')) {
        return false;
    }
    if markdown {
        return margin
            && trimmed.starts_with('#')
            && trimmed.trim_start_matches('#').starts_with(' ');
    }
    // A token's word: what it opens with, before any `<`, `(`, `!` or `:`.
    let word = |token: &str| {
        let end = token
            .find(|c: char| !c.is_alphanumeric() && c != '_')
            .unwrap_or(token.len());
        token[..end].to_string()
    };
    let mut tokens = trimmed.split_whitespace().peekable();
    while let Some(&token) = tokens.peek() {
        let modifier = MODIFIERS.contains(&word(token).as_str())
            || (token.starts_with('"') && token.ends_with('"'));
        if !modifier {
            break;
        }
        tokens.next();
    }
    let Some(token) = tokens.next() else {
        return false;
    };
    let word = word(token);
    if DEFINITIONS.contains(&word.as_str()) {
        return true;
    }
    // A call at the margin that opens a block: a C function, or a block
    // in a language without a word for one. Not a line that goes on from
    // the one before, such as the `) -> T {` of a signature.
    margin
        && trimmed.ends_with('{')
        && trimmed.contains('(')
        && trimmed.starts_with(|c: char| c.is_alphabetic() || c == '_')
        && !STATEMENTS.contains(&word.as_str())
}

/// The page of `file` from line `start`: its lines, each after its number
/// and a tab, up to [`PAGE`] of them or [`BYTES`] in all, then a line of
/// where the page sits in the file and where the next page starts.
fn page(mut file: impl BufRead, path: &str, start: u64) -> io::Result<String> {
    let mut text = String::new();
    let mut line = Vec::new();
    // Lines read so far, shown in the page, and cut for length.
    let (mut lines, mut shown, mut cut) = (0, 0, 0);
    loop {
        line.clear();
        if file.read_until(b'\n', &mut line)? == 0 {
            break;
        }
        lines += 1;
        if lines < start {
            continue;
        }
        let (content, long) = shown_line(&line);
        let entry = format!("{lines}\t{content}\n");
        if shown == PAGE || (shown > 0 && text.len() + entry.len() > BYTES) {
            // The file goes on past the page: count the rest.
            lines += count(&mut file)?;
            break;
        }
        text.push_str(&entry);
        shown += 1;
        cut += u64::from(long);
    }
    let end = start + shown as u64 - 1;
    let mut note = if lines == 0 {
        format!("[{path} is empty]")
    } else if shown == 0 {
        format!(
            "[{path} has {lines} {}; nothing from line {start}]",
            plural(lines, "line")
        )
    } else if end < lines {
        format!(
            "[lines {start}–{end} of {lines} in {path}; next: read {path} from line {}]",
            end + 1
        )
    } else {
        format!("[lines {start}–{end} of {lines} in {path}; end of file]")
    };
    if cut > 0 {
        note.insert_str(
            note.len() - 1,
            &format!(
                "; {cut} {} longer than {LINE} bytes cut",
                plural(cut, "line")
            ),
        );
    }
    text.push_str(&note);
    Ok(text)
}

/// Number of lines left in `file`, reading it to the end. A last line
/// without a newline counts.
fn count(file: &mut impl BufRead) -> io::Result<u64> {
    let mut lines = 0;
    let mut last = b'\n';
    loop {
        let buf = file.fill_buf()?;
        if buf.is_empty() {
            break;
        }
        lines += buf.iter().filter(|&&b| b == b'\n').count() as u64;
        last = buf[buf.len() - 1];
        let n = buf.len();
        file.consume(n);
    }
    Ok(lines + u64::from(last != b'\n'))
}

#[cfg(test)]
mod tests {
    use std::{fs, path::PathBuf};

    use super::*;

    /// A file with `text` in a directory of its own, and its path.
    fn file(name: &str, text: &str) -> (PathBuf, String) {
        static FILES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = FILES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("dwim-read-{}-{n}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join(name);
        fs::write(&path, text).unwrap();
        let text = path.to_str().unwrap().to_string();
        (dir, text)
    }

    fn read(path: &str, start: Option<u64>) -> String {
        let mut arguments = serde_json::json!({ "path": path });
        if let Some(start) = start {
            arguments["start"] = start.into();
        }
        run(&arguments)
    }

    #[test]
    fn pages_through_a_file() {
        let text: String = (1..=450).map(|i| format!("line {i}\n")).collect();
        let (dir, path) = file("long.txt", &text);
        let first = read(&path, None);
        assert!(first.starts_with("1\tline 1\n2\tline 2\n"), "{first}");
        assert!(first.contains("\n200\tline 200\n"), "{first}");
        assert!(!first.contains("line 201\n"));
        assert!(
            first.ends_with(&format!(
                "[lines 1–200 of 450 in {path}; next: read {path} from line 201]"
            )),
            "{first}"
        );
        assert_eq!(first.lines().count(), PAGE + 1);

        let second = read(&path, Some(201));
        assert!(second.starts_with("201\tline 201\n"), "{second}");
        assert!(
            second.ends_with(&format!(
                "[lines 201–400 of 450 in {path}; next: read {path} from line 401]"
            )),
            "{second}"
        );

        let last = read(&path, Some(401));
        assert!(last.starts_with("401\tline 401\n"));
        assert!(last.contains("\n450\tline 450\n"));
        assert!(
            last.ends_with(&format!("[lines 401–450 of 450 in {path}; end of file]")),
            "{last}"
        );

        // Following the cursor covers the file exactly once.
        let mut seen = Vec::new();
        for page in [&first, &second, &last] {
            seen.extend(
                page.lines()
                    .filter(|line| !line.starts_with('['))
                    .map(|line| line.split('\t').next().unwrap().parse::<u64>().unwrap()),
            );
        }
        assert_eq!(seen, (1..=450).collect::<Vec<_>>());

        assert_eq!(
            read(&path, Some(451)),
            format!("[{path} has 450 lines; nothing from line 451]")
        );
        assert_eq!(
            read(&path, Some(0)).lines().next().unwrap(),
            "1\tline 1",
            "a start below 1 means the first line"
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn reads_the_edges() {
        let (dir, path) = file("empty.txt", "");
        assert_eq!(read(&path, None), format!("[{path} is empty]"));
        let one = dir.join("one.txt");
        fs::write(&one, "just this").unwrap();
        let one = one.to_str().unwrap();
        assert_eq!(
            read(one, None),
            format!("1\tjust this\n[lines 1–1 of 1 in {one}; end of file]")
        );
        let crlf = dir.join("crlf.txt");
        fs::write(&crlf, "a\r\nb\r\n").unwrap();
        let crlf = crlf.to_str().unwrap();
        assert_eq!(
            read(crlf, None),
            format!("1\ta\n2\tb\n[lines 1–2 of 2 in {crlf}; end of file]")
        );
        let bytes = dir.join("bytes.bin");
        fs::write(
            &bytes,
            [b"x\xc3\xa4\n".as_slice(), &[0xff, 0xfe, b'y']].concat(),
        )
        .unwrap();
        let bytes = bytes.to_str().unwrap();
        assert_eq!(
            read(bytes, None),
            format!("1\tx\u{e4}\n2\t\u{FFFD}\u{FFFD}y\n[lines 1–2 of 2 in {bytes}; end of file]")
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn cuts_long_lines_between_characters() {
        let long = "ä".repeat(400);
        let (dir, path) = file("wide.txt", &format!("short\n{long}\n"));
        let page = read(&path, None);
        let second = page.lines().nth(1).unwrap();
        assert!(second.starts_with("2\t"));
        assert!(second.ends_with('…'));
        assert!(!second.contains('\u{FFFD}'));
        assert!(second.len() <= "2\t".len() + LINE + '…'.len_utf8());
        assert!(
            page.ends_with(&format!(
                "[lines 1–2 of 2 in {path}; end of file; 1 line longer than {LINE} bytes cut]"
            )),
            "{page}"
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn ends_a_page_of_long_lines_early() {
        // 100 lines of 299 bytes: three pages by bytes, half of one by lines.
        let text: String = (1..=100).map(|i| format!("{i:<299}\n")).collect();
        let (dir, path) = file("wide.txt", &text);
        let mut start = 1;
        let mut seen = Vec::new();
        while start <= 100 {
            let page = read(&path, Some(start));
            let body: usize = page
                .lines()
                .filter(|line| !line.starts_with('['))
                .map(|line| line.len() + 1)
                .sum();
            assert!(body <= BYTES, "{body} bytes of lines");
            let shown: Vec<u64> = page
                .lines()
                .filter(|line| !line.starts_with('['))
                .map(|line| line.split('\t').next().unwrap().parse().unwrap())
                .collect();
            assert!(!shown.is_empty());
            let end = *shown.last().unwrap();
            if end < 100 {
                assert!(
                    page.ends_with(&format!(
                        "[lines {start}–{end} of 100 in {path}; next: read {path} from line {}]",
                        end + 1
                    )),
                    "{page}"
                );
            } else {
                assert!(
                    page.ends_with(&format!(
                        "[lines {start}–100 of 100 in {path}; end of file]"
                    )),
                    "{page}"
                );
            }
            seen.extend(shown);
            start = end + 1;
        }
        assert_eq!(seen, (1..=100).collect::<Vec<_>>());
        assert!(seen.len() > 1);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn outlines_a_long_file() {
        let mut text = String::from("//! A module.\n\nuse std::fs;\n\n");
        text.push_str("pub fn alpha() -> u32 {\n    let x = 1;\n    x\n}\n\n");
        text.push_str("pub fn delta(\n    a: u32,\n) -> u32 {\n    a\n}\n\n");
        text.push_str("pub(crate) struct Foo;\n\nimpl Foo {\n    /// Beta.\n    pub fn beta(&self) {\n        for i in 0..3 {\n            dbg!(i);\n        }\n    }\n}\n\n");
        text.push_str("impl<T: Clone> Wrap<T> {\n    fn get(&self) -> T {\n        self.0.clone()\n    }\n}\n\n");
        text.push_str("#[cfg(test)]\nmod tests {\n    #[test]\n    fn gamma() {\n        assert!(true);\n    }\n}\n");
        while text.lines().count() <= PAGE {
            text.push_str("// filler\n");
        }
        let (dir, path) = file("code.rs", &text);
        let lines = text.lines().count();
        let outline = read(&path, None);
        assert!(
            outline.starts_with(&format!(
                "[{path} has {lines} lines; its definitions follow"
            )),
            "{outline}"
        );
        let entries: Vec<&str> = outline.lines().filter(|l| !l.starts_with('[')).collect();
        assert_eq!(
            entries,
            [
                "5\tpub fn alpha() -> u32 {",
                "10\tpub fn delta(",
                "16\tpub(crate) struct Foo;",
                "18\timpl Foo {",
                "20\t    pub fn beta(&self) {",
                "27\timpl<T: Clone> Wrap<T> {",
                "28\t    fn get(&self) -> T {",
                "34\tmod tests {",
                "36\t    fn gamma() {",
            ],
            "{outline}"
        );
        assert!(
            outline.ends_with(&format!(
                "[end of outline; read {path} from a line to see the code]"
            )),
            "{outline}"
        );
        // From a line, the page is as ever.
        let page = read(&path, Some(1));
        assert!(page.starts_with("1\t//! A module.\n"), "{page}");
        assert!(
            page.contains(&format!("next: read {path} from line 201]")),
            "{page}"
        );
        // A file of a page or less comes whole, outline or not.
        let short = dir.join("short.rs");
        fs::write(&short, "fn a() {}\nfn b() {}\n").unwrap();
        let short = short.to_str().unwrap();
        assert!(read(short, None).starts_with("1\tfn a() {}\n2\tfn b() {}\n"));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn outlines_other_languages() {
        for (name, text, want) in [
            (
                "script.py",
                "import os\n\nclass A:\n    def m(self):\n        return 1\n\nasync def main():\n    x = 1\n",
                vec!["3\tclass A:", "4\t    def m(self):", "7\tasync def main():"],
            ),
            (
                "prog.c",
                "#include <stdio.h>\n\nstatic int add(int a, int b) {\n    if (a) {\n        return b;\n    }\n    return a + b;\n}\n\nint main(void) {\n    return add(1, 2);\n}\n",
                vec!["3\tstatic int add(int a, int b) {", "10\tint main(void) {"],
            ),
            (
                "notes.md",
                "# Title\n\nText with #hashtag.\n\n## Part\n\n  # not a heading\n### Sub\n",
                vec!["1\t# Title", "5\t## Part", "8\t### Sub"],
            ),
        ] {
            let mut text = text.to_string();
            while text.lines().count() <= PAGE {
                text.push('\n');
            }
            let (dir, path) = file(name, &text);
            let outline = read(&path, None);
            let entries: Vec<&str> = outline.lines().filter(|l| !l.starts_with('[')).collect();
            assert_eq!(entries, want, "{name}: {outline}");
            fs::remove_dir_all(dir).unwrap();
        }
    }

    #[test]
    fn cuts_a_long_outline() {
        let text: String = (1..=500).map(|i| format!("fn f{i}() {{}}\n")).collect();
        let (dir, path) = file("many.rs", &text);
        let outline = read(&path, None);
        let entries: Vec<&str> = outline.lines().filter(|l| !l.starts_with('[')).collect();
        assert_eq!(entries.len(), PAGE);
        assert_eq!(entries[0], "1\tfn f1() {}");
        assert_eq!(entries[PAGE - 1], "200\tfn f200() {}");
        assert!(
            outline.ends_with(&format!(
                "[… and 300 more past line 200; read {path} from line 201 for them]"
            )),
            "{outline}"
        );
        let long = "fn ".to_string() + &"x".repeat(200) + "() {}\n";
        let text: String = (1..=300).map(|_| long.clone()).collect();
        let (dir2, path) = file("wide.rs", &text);
        let outline = read(&path, None);
        let entry = outline.lines().nth(1).unwrap();
        assert!(entry.ends_with('…'), "{entry}");
        assert!(entry.len() <= "1\t".len() + ENTRY + '…'.len_utf8());
        fs::remove_dir_all(dir).unwrap();
        fs::remove_dir_all(dir2).unwrap();
    }

    #[test]
    fn refuses_what_it_cannot_read() {
        assert!(
            read("/nonexistent/file", None).starts_with("error: cannot read /nonexistent/file: ")
        );
        assert!(read("/", None).starts_with("error: / is a directory"));
        assert_eq!(
            run(&serde_json::json!({ "path": "Cargo.toml", "start": "abc" })),
            "error: 'start' is a line number"
        );
        assert!(
            run(&serde_json::json!({ "path": "Cargo.toml", "start": "1" }))
                .starts_with("1\t[package]")
        );
    }
}
