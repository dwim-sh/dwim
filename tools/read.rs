//! The `read` tool: a file, a page of numbered lines at a time, with
//! where the next page starts. The page is fixed and the model is told
//! its size, so it knows what it will get before it asks and never has to
//! guess at a range: it reads the next page, or starts from a line.
//!
//! A read without a start line is of the file's first page, and a page
//! is long enough that most files come whole. A read may ask for fewer
//! lines than a page, but the model is told to read more at once rather
//! than less, as OpenCode tells it: a file longer than a page used to
//! come as an outline of its definitions, and the advice was to read
//! just the lines an edit changes, and on SWE-bench's django-11797 the
//! model read the 2,300 lines of Django's query.py in 28 slices of 40
//! lines or so, never saw a function whole, and went on reading for half
//! an hour without an edit.

use std::{
    fs::File,
    io::{self, BufRead, BufReader},
    path::Path,
};

use serde_json::Value;

use crate::{Parameter, Signature, Tool, cut, optional_number, plural, required, string};

const NAME: &str = "read";

pub const TOOL: Tool = Tool {
    name: NAME,
    signature,
    changes: false,
    describe,
    run: |tools, arguments| run(&tools.dir, arguments),
};

/// Most lines in a page. A page is what the model gets of a file in one
/// call: a page too small has it read a file in slices, each a round
/// trip, and lose the thread between them, while a file of a thousand
/// lines read whole is a tenth of a small context. Most files come whole.
pub const PAGE: usize = 1000;

/// Most bytes in a page: a page of long lines ends early, so that one
/// page can't take up much of the context.
const BYTES: usize = 48 << 10;

/// Most bytes of a line shown; a longer line is cut there, with a mark.
pub const LINE: usize = 1000;

fn signature() -> Signature {
    Signature {
        description: format!(
            "Read a file from a line, a page of up to {PAGE} numbered lines (or {} KB) at a time, or as many lines as asked for. The result says how many lines the file has and where the next page starts. Prefer one larger read over many small slices, and use grep to find specific content in a large file.",
            BYTES >> 10
        ),
        parameters: vec![
            Parameter::string("path", "The file to read."),
            Parameter::integer(
                "start",
                "The line to start from, counting from 1. Left out, the first line.",
            )
            .optional(),
            Parameter::integer(
                "lines",
                format!(
                    "How many lines to read from start, up to {PAGE}, or from the first line without a start. Left out, a page."
                ),
            )
            .optional(),
        ],
    }
}

/// How a call reads on screen: the file, and where in it; nothing for a
/// read the tool will refuse.
fn describe(arguments: &Value) -> Option<String> {
    let path = string(arguments, "path")?;
    let (start, lines) = range(arguments).ok()?;
    Some(match (start, lines) {
        (Some(start), Some(1)) => format!("{path} line {start}"),
        (Some(start), Some(lines)) => {
            let end = start.saturating_add(lines as u64 - 1);
            format!("{path} lines {start}–{end}")
        }
        (Some(start), None) if start > 1 => format!("{path} from line {start}"),
        _ => path.to_string(),
    })
}

/// Reads the page of the file in `arguments`, a path taken from `dir`,
/// returning what the model should see.
fn run(dir: &Path, arguments: &Value) -> Result<String, String> {
    let path = required(arguments, NAME, "path")?;
    let (start, lines) = range(arguments)?;
    let most = lines.unwrap_or(PAGE);
    let file = File::open(dir.join(path))
        .map(BufReader::new)
        .map_err(|e| format!("cannot read {path}: {e}"))?;
    let directory = file
        .get_ref()
        .metadata()
        .is_ok_and(|metadata| metadata.is_dir());
    if directory {
        return Err(format!("{path} is a directory; list it with glob"));
    }
    page(file, path, start.unwrap_or(1), most).map_err(|e| format!("reading {path}: {e}"))
}

/// What the arguments ask for: the line to start from, and how many lines,
/// each where given, or the error the model is shown for one that is not
/// a number. More lines than a page is a page, and lines asked for without
/// a start are from the first.
fn range(arguments: &Value) -> Result<(Option<u64>, Option<usize>), String> {
    let start = optional_number(arguments, "start", "a line number")?.map(|start| start.max(1));
    let lines = match optional_number(arguments, "lines", "a number of lines, 1 or more")? {
        None => None,
        Some(0) => return Err("'lines' is a number of lines, 1 or more".to_string()),
        Some(lines) => Some((lines as usize).min(PAGE)),
    };
    Ok((start.or(lines.map(|_| 1)), lines))
}

/// The line `line` as a page shows it, without its line ending, cut at
/// [`LINE`] bytes with a mark, and whether it was cut.
pub fn shown_line(line: &[u8]) -> (String, bool) {
    let mut content = String::from_utf8_lossy(crate::without_ending(line)).into_owned();
    let long = cut(&mut content, LINE);
    (content, long)
}

/// The page of `file` from line `start`: its lines, each after its number
/// and a tab, up to `most` of them or [`BYTES`] in all, then a line of
/// where the page sits in the file and where the next page starts.
fn page(mut file: impl BufRead, path: &str, start: u64, most: usize) -> io::Result<String> {
    let mut text = String::new();
    let mut line = Vec::new();
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
        if shown == most || (shown > 0 && text.len() + entry.len() > BYTES) {
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

    /// Runs the call in `arguments`, as the model sees the result.
    fn run(arguments: &Value) -> String {
        crate::Output::of(super::run(Path::new("."), arguments), false).text
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
        // Two pages and a half.
        let (a, b, c, n) = (PAGE, 2 * PAGE, 2 * PAGE + 1, 2 * PAGE + PAGE / 4);
        let text: String = (1..=n).map(|i| format!("line {i}\n")).collect();
        let (dir, path) = file("long.txt", &text);
        let first = read(&path, None);
        assert!(first.starts_with("1\tline 1\n2\tline 2\n"), "{first}");
        assert!(first.contains(&format!("\n{a}\tline {a}\n")), "{first}");
        assert!(!first.contains(&format!("line {}\n", a + 1)));
        assert!(
            first.ends_with(&format!(
                "[lines 1–{a} of {n} in {path}; next: read {path} from line {}]",
                a + 1
            )),
            "{first}"
        );
        assert_eq!(first.lines().count(), PAGE + 1);

        let second = read(&path, Some(a as u64 + 1));
        assert!(
            second.starts_with(&format!("{}\tline {}\n", a + 1, a + 1)),
            "{second}"
        );
        assert!(
            second.ends_with(&format!(
                "[lines {}–{b} of {n} in {path}; next: read {path} from line {c}]",
                a + 1
            )),
            "{second}"
        );

        let last = read(&path, Some(c as u64));
        assert!(last.starts_with(&format!("{c}\tline {c}\n")));
        assert!(last.contains(&format!("\n{n}\tline {n}\n")));
        assert!(
            last.ends_with(&format!("[lines {c}–{n} of {n} in {path}; end of file]")),
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
        assert_eq!(seen, (1..=n as u64).collect::<Vec<_>>());

        assert_eq!(
            read(&path, Some(n as u64 + 1)),
            format!("[{path} has {n} lines; nothing from line {}]", n + 1)
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
        let long = "ä".repeat(LINE);
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
        // 400 lines of 299 bytes: three pages by bytes, half of one by lines.
        let text: String = (1..=400).map(|i| format!("{i:<299}\n")).collect();
        let (dir, path) = file("wide.txt", &text);
        let mut start = 1;
        let mut seen = Vec::new();
        while start <= 400 {
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
            if end < 400 {
                assert!(
                    page.ends_with(&format!(
                        "[lines {start}–{end} of 400 in {path}; next: read {path} from line {}]",
                        end + 1
                    )),
                    "{page}"
                );
            } else {
                assert!(
                    page.ends_with(&format!(
                        "[lines {start}–400 of 400 in {path}; end of file]"
                    )),
                    "{page}"
                );
            }
            seen.extend(shown);
            start = end + 1;
        }
        assert_eq!(seen, (1..=400).collect::<Vec<_>>());
        assert!(seen.len() > 1);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn reads_the_lines_asked_for() {
        let n = PAGE + 50;
        let text: String = (1..=n).map(|i| format!("line {i}\n")).collect();
        let (dir, path) = file("long.txt", &text);
        let some = run(&serde_json::json!({ "path": &path, "start": 440, "lines": 5 }));
        assert_eq!(
            some,
            format!(
                "440\tline 440\n441\tline 441\n442\tline 442\n443\tline 443\n444\tline 444\n[lines 440–444 of {n} in {path}; next: read {path} from line 445]"
            )
        );
        // Lines without a start are from the first, in place of the outline;
        // more than a page is a page.
        let first = run(&serde_json::json!({ "path": &path, "lines": "1" }));
        assert_eq!(
            first,
            format!("1\tline 1\n[lines 1–1 of {n} in {path}; next: read {path} from line 2]")
        );
        let page = run(&serde_json::json!({ "path": &path, "lines": 10 * PAGE }));
        assert_eq!(page.lines().count(), PAGE + 1);
        assert!(
            run(&serde_json::json!({ "path": &path, "lines": 0 })).starts_with("error: 'lines' is"),
        );
        assert!(
            run(&serde_json::json!({ "path": &path, "lines": "x" }))
                .starts_with("error: 'lines' is"),
        );
        fs::remove_dir_all(dir).unwrap();
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
