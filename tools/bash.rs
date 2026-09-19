//! The `bash` tool: runs a shell command and gives the model what it
//! printed as a preview, its standard output apart from its standard
//! error, with a line of how it ended that is always there. A stream
//! that outgrows its preview is shown by its start and its end, and is
//! kept whole in a file of the tool's own, which the result names along
//! with the line to `read` it from, so the model can get the rest with
//! that tool instead of running the command again.

use std::{
    collections::VecDeque,
    env,
    fs::{self, DirBuilder, File},
    io::{self, ErrorKind, Read, Write},
    os::unix::{fs::DirBuilderExt, process::ExitStatusExt},
    path::{Path, PathBuf},
    process::{self, Command, ExitStatus, Stdio},
    sync::atomic::{AtomicUsize, Ordering},
    thread,
};

use serde_json::Value;

use crate::string;

/// The tool as the system prompt declares it.
pub const SIGNATURE: &str = r#"{"type": "function", "function": {"name": "bash", "description": "Run a shell command and return what it printed and its exit code. Long output is cut to its start and end, and kept whole in a file the result names for `read`.", "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "The command to run."}}, "required": ["command"]}}}"#;

/// Most of one stream of a command's output that goes back to the model as
/// it is, so that a chatty command can't fill the context window. A longer
/// stream is shown by its first and last [`HALF`] bytes, cut at line ends
/// where it can be, and kept whole on disk for the model to read from.
const PREVIEW: usize = 2000;

/// How much of each end of a stream too long for its preview is shown.
const HALF: usize = PREVIEW / 2;

/// Most of one stream that is kept on disk. What a command prints past that
/// is discarded, and the model is told how much.
const RETAINED: u64 = 4 << 20;

/// The `bash` tool, and where it keeps a command's output when there is
/// more of it than the model is shown: a directory of this process's own
/// under the system's temporary directory, readable by this user alone,
/// with a file per stream of each command that outgrew its preview, named
/// by the command's number, as `3.stdout`. The directory is made when
/// first needed and removed when the tool is dropped. A process that ends
/// without dropping it, as the shell does while the model is busy, leaves
/// the directory behind, and the next to start removes what earlier
/// processes that are no longer running left.
pub struct Bash {
    dir: PathBuf,
    /// Commands run so far; the next one's files are named by the count.
    calls: usize,
    /// Most bytes of a stream kept in its file.
    retain: u64,
}

/// Tells apart the output directories of the tools of one process.
static OUTPUT_DIRS: AtomicUsize = AtomicUsize::new(0);

impl Default for Bash {
    fn default() -> Self {
        Self::new()
    }
}

impl Bash {
    pub fn new() -> Self {
        let tmp = env::temp_dir();
        sweep(&tmp);
        let n = OUTPUT_DIRS.fetch_add(1, Ordering::Relaxed);
        Self {
            dir: tmp.join(format!("dwim-{}-{n}", process::id())),
            calls: 0,
            retain: RETAINED,
        }
    }

    /// The file for the `stream` of the command being run.
    fn file(&self, stream: &str) -> PathBuf {
        self.dir.join(format!("{}.{stream}", self.calls))
    }

    /// Runs the command in `arguments`, returning what the model should see.
    pub fn run(&mut self, arguments: &Value) -> String {
        let Some(command) = string(arguments, "command") else {
            return "error: bash needs a 'command' string".to_string();
        };
        self.calls += 1;
        let mut child = match Command::new("sh")
            .args(["-c", command])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
        {
            Ok(child) => child,
            Err(e) => return format!("[error: could not run sh: {e}]"),
        };
        let mut stdout = Stream::new("stdout", self);
        let mut stderr = Stream::new("stderr", self);
        let (out, err) = (child.stdout.take().unwrap(), child.stderr.take().unwrap());
        // Each pipe is read on its own thread, so that neither fills up and
        // blocks the command while the other is being read.
        thread::scope(|s| {
            s.spawn(|| stderr.drain(err));
            stdout.drain(out);
        });
        let status = match child.wait() {
            Ok(status) => ended(status),
            Err(e) => format!("[error: could not wait for sh: {e}]"),
        };
        report(&stdout, &stderr, &status)
    }
}

impl Drop for Bash {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.dir);
    }
}

/// Removes the output directories in `tmp` of dwim processes that are no
/// longer running.
fn sweep(tmp: &Path) {
    let Ok(entries) = fs::read_dir(tmp) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let pid = name
            .to_str()
            .and_then(|name| name.strip_prefix("dwim-"))
            .and_then(|rest| rest.split('-').next())
            .and_then(|pid| pid.parse::<u32>().ok());
        if let Some(pid) = pid
            && pid != process::id()
            && !alive(pid)
        {
            let _ = fs::remove_dir_all(entry.path());
        }
    }
}

/// Whether a process with `pid` is running. Signal 0 checks without
/// sending anything, and is refused for another user's process, which is
/// running all the same.
fn alive(pid: u32) -> bool {
    let refused = unsafe { libc::kill(pid as libc::pid_t, 0) } != 0;
    !refused || io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

/// One stream of a command's output as it is read: its first [`PREVIEW`]
/// bytes and its last [`HALF`] in memory, all of it in a file once it
/// outgrows the preview, up to the retention cap, and counts of the rest.
struct Stream {
    name: &'static str,
    head: Vec<u8>,
    tail: VecDeque<u8>,
    /// Bytes read so far.
    total: u64,
    /// Newlines read so far.
    newlines: u64,
    /// Bytes written to the file, from the start of the stream.
    kept: u64,
    file: Option<File>,
    dir: PathBuf,
    path: PathBuf,
    retain: u64,
    /// Why the stream could not be read to the end or kept in its file, if
    /// it could not.
    error: Option<String>,
}

impl Stream {
    fn new(name: &'static str, bash: &Bash) -> Self {
        Self {
            name,
            head: Vec::new(),
            tail: VecDeque::new(),
            total: 0,
            newlines: 0,
            kept: 0,
            file: None,
            dir: bash.dir.clone(),
            path: bash.file(name),
            retain: bash.retain,
            error: None,
        }
    }

    /// Reads `pipe` to its end, or until reading it fails.
    fn drain(&mut self, mut pipe: impl Read) {
        let mut buf = vec![0; 64 << 10];
        loop {
            match pipe.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => self.push(&buf[..n]),
                Err(e) if e.kind() == ErrorKind::Interrupted => {}
                Err(e) => {
                    self.error = Some(format!("could not read the rest: {e}"));
                    break;
                }
            }
        }
    }

    /// Takes in the next `bytes` of the stream.
    fn push(&mut self, bytes: &[u8]) {
        let before = self.total;
        self.total += bytes.len() as u64;
        self.newlines += bytes.iter().filter(|&&b| b == b'\n').count() as u64;
        let room = PREVIEW - self.head.len();
        self.head.extend_from_slice(&bytes[..bytes.len().min(room)]);
        self.tail.extend(bytes.iter().copied());
        if self.tail.len() > HALF {
            self.tail.drain(..self.tail.len() - HALF);
        }
        if self.total > PREVIEW as u64
            && self.error.is_none()
            && let Err(e) = self.keep(before, bytes)
        {
            self.error = Some(format!("could not keep the rest: {e}"));
        }
    }

    /// Writes what the file lacks of the stream up to the retention cap,
    /// opening it with the head of the stream if it is not open yet. The
    /// stream was `before` bytes long before `bytes`.
    fn keep(&mut self, before: u64, bytes: &[u8]) -> io::Result<()> {
        if self.file.is_none() {
            DirBuilder::new()
                .mode(0o700)
                .recursive(true)
                .create(&self.dir)?;
            let mut file = File::create(&self.path)?;
            let kept = self.head.len().min(self.retain as usize);
            file.write_all(&self.head[..kept])?;
            self.kept = kept as u64;
            self.file = Some(file);
        }
        let end = self.total.min(self.retain);
        if end > self.kept {
            // The head covers the stream up to where the file began, so
            // what the file lacks is within `bytes`.
            let range = (self.kept - before) as usize..(end - before) as usize;
            self.file.as_mut().unwrap().write_all(&bytes[range])?;
            self.kept = end;
        }
        Ok(())
    }

    /// Number of lines read: a last line without a newline counts.
    fn lines(&self) -> u64 {
        self.newlines + u64::from(self.tail.back().is_some_and(|&b| b != b'\n'))
    }

    /// The stream as the model sees it: all of it if it fits the preview,
    /// and otherwise its start and its end around a note of what lies
    /// between them.
    fn preview(&self) -> String {
        if self.total <= PREVIEW as u64 {
            return String::from_utf8_lossy(&self.head).trim_end().to_string();
        }
        // Cut the start after a line end if one comes late enough to keep
        // most of the half, and the end before one that comes early
        // enough, and elsewhere at a character boundary.
        let head = &self.head[..HALF];
        let cut = match head.iter().rposition(|&b| b == b'\n') {
            Some(at) if at + 1 >= HALF / 2 => at + 1,
            _ => char_end(head),
        };
        let head = &head[..cut];
        let (a, b) = self.tail.as_slices();
        let window = [a, b].concat();
        let skip = match window.iter().position(|&b| b == b'\n') {
            Some(at) if at < HALF / 2 => at + 1,
            _ => char_start(&window),
        };
        let tail = &window[skip..];
        // The lines with any part not shown: from the one the head ends
        // in, to the one before the tail's first, or the one the tail
        // starts in when it starts partway through a line.
        let first = head.iter().filter(|&&b| b == b'\n').count() as u64 + 1;
        let midline = skip == 0 || window[skip - 1] != b'\n';
        let last = self.newlines - tail.iter().filter(|&&b| b == b'\n').count() as u64
            + u64::from(midline);
        let bytes = self.total - head.len() as u64 - tail.len() as u64;
        let lines = if first == last {
            format!("line {first}")
        } else {
            format!("lines {first}–{last}")
        };
        let next = match self.kept {
            0 => String::new(),
            _ => format!("; next: read {} from line {first}", self.path.display()),
        };
        format!(
            "{}\n[… {lines} ({bytes} bytes) not shown{next} …]\n{}",
            String::from_utf8_lossy(head).trim_end(),
            String::from_utf8_lossy(tail).trim_end(),
        )
    }

    /// What the model is told about the stream beyond its preview, if
    /// there is anything: how much of it there was, and where it is.
    fn note(&self) -> Option<String> {
        if self.total <= PREVIEW as u64 && self.error.is_none() {
            return None;
        }
        let mut parts = vec![format!(
            "{} {}, {} bytes",
            self.lines(),
            plural(self.lines(), "line"),
            self.total
        )];
        let path = self.path.display();
        if self.file.is_some() {
            parts.push(if self.kept == self.total {
                format!("whole in {path}")
            } else {
                format!(
                    "the first {} bytes in {path}, the rest discarded",
                    self.kept
                )
            });
        }
        parts.extend(self.error.clone());
        Some(format!("[{}: {}]", self.name, parts.join(", ")))
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

/// Where a cut at the end of `bytes` leaves no partial UTF-8 character:
/// their length, less an incomplete character at the end.
fn char_end(bytes: &[u8]) -> usize {
    for i in (bytes.len().saturating_sub(3)..bytes.len()).rev() {
        let b = bytes[i];
        if b & 0xC0 == 0x80 {
            continue;
        }
        let need = match b {
            0xF0.. => 4,
            0xE0.. => 3,
            0xC0.. => 2,
            _ => 1,
        };
        return if bytes.len() - i < need {
            i
        } else {
            bytes.len()
        };
    }
    bytes.len()
}

/// Where a cut at the start of `bytes` leaves no partial UTF-8 character:
/// past the continuation bytes of one cut off before them.
fn char_start(bytes: &[u8]) -> usize {
    bytes
        .iter()
        .take(3)
        .position(|&b| b & 0xC0 != 0x80)
        .unwrap_or(bytes.len().min(3))
}

/// How a command ended, as the last line of its result.
fn ended(status: ExitStatus) -> String {
    match (status.code(), status.signal()) {
        (Some(code), _) => format!("[exit code {code}]"),
        (None, Some(signal)) => {
            let name = match signal {
                libc::SIGHUP => " (SIGHUP)",
                libc::SIGINT => " (SIGINT)",
                libc::SIGQUIT => " (SIGQUIT)",
                libc::SIGILL => " (SIGILL)",
                libc::SIGABRT => " (SIGABRT)",
                libc::SIGFPE => " (SIGFPE)",
                libc::SIGKILL => " (SIGKILL)",
                libc::SIGBUS => " (SIGBUS)",
                libc::SIGSEGV => " (SIGSEGV)",
                libc::SIGPIPE => " (SIGPIPE)",
                libc::SIGALRM => " (SIGALRM)",
                libc::SIGTERM => " (SIGTERM)",
                _ => "",
            };
            format!("[killed by signal {signal}{name}]")
        }
        (None, None) => format!("[{status}]"),
    }
}

/// What the model sees of a command: the preview of what it printed to
/// standard output, then to standard error after a `[stderr]` line, then
/// where the rest of either is if there is more, and how it ended. The two
/// streams are kept apart, so the order in which the command wrote to them
/// is not shown; the error on one cannot be crowded out by the other.
fn report(stdout: &Stream, stderr: &Stream, status: &str) -> String {
    let mut text = String::new();
    let preview = stdout.preview();
    if !preview.is_empty() {
        text.push_str(&preview);
        text.push('\n');
    }
    let preview = stderr.preview();
    if !preview.is_empty() {
        text.push_str("[stderr]\n");
        text.push_str(&preview);
        text.push('\n');
    }
    for note in [stdout.note(), stderr.note()].into_iter().flatten() {
        text.push_str(&note);
        text.push('\n');
    }
    text.push_str(status);
    text
}

#[cfg(test)]
mod tests {
    use std::{sync::mpsc, time::Duration};

    use dwim_models::ToolCall;

    use super::*;

    fn call(text: &str) -> ToolCall {
        ToolCall::parse(text).unwrap()
    }

    /// Runs `command` through the tool.
    fn sh(command: &str, bash: &mut Bash) -> String {
        bash.run(&serde_json::json!({ "command": command }))
    }

    #[test]
    fn runs_bash() {
        let mut bash = Bash::new();
        assert_eq!(
            sh("echo hi; echo err >&2", &mut bash),
            "hi\n[stderr]\nerr\n[exit code 0]"
        );
        assert_eq!(sh("exit 3", &mut bash), "[exit code 3]");
        assert_eq!(sh("true", &mut bash), "[exit code 0]");
        assert_eq!(sh("echo; echo", &mut bash), "[exit code 0]");
        assert_eq!(
            sh("kill -9 $$", &mut bash),
            "[killed by signal 9 (SIGKILL)]"
        );
        assert!(!bash.dir.exists(), "nothing outgrew its preview");
    }

    #[test]
    fn keeps_the_error_and_the_status_after_a_flood() {
        let mut bash = Bash::new();
        let output = sh("seq 1 50000; echo 'boom: it broke' >&2; exit 7", &mut bash);
        assert!(output.starts_with("1\n2\n3\n"), "{output}");
        assert!(output.contains("\n[… lines "), "{output}");
        assert!(
            output.contains("\n49999\n50000\n[stderr]\nboom: it broke\n"),
            "{output}"
        );
        assert!(
            output.contains("\n[stdout: 50000 lines, 288894 bytes, whole in "),
            "{output}"
        );
        assert!(output.ends_with("\n[exit code 7]"), "{output}");
        assert!(output.len() < PREVIEW + 400, "{}", output.len());
    }

    #[test]
    fn keeps_long_output_for_reading_later() {
        let mut bash = Bash::new();
        let output = sh("seq 1 50000", &mut bash);
        let path = bash.dir.join("1.stdout");
        let note = output
            .lines()
            .find(|line| line.starts_with("[stdout: "))
            .unwrap();
        assert!(
            note.ends_with(&format!("whole in {}]", path.display())),
            "{note}"
        );
        let expected: String = (1..=50000).map(|i| format!("{i}\n")).collect();
        assert_eq!(fs::read_to_string(&path).unwrap(), expected);

        // The lines the preview says it left out are the ones between what
        // it shows, so a range read gets what is missing.
        let marker = output
            .lines()
            .find(|line| line.starts_with("[… lines "))
            .unwrap();
        assert!(
            marker.contains(&format!("; next: read {} from line ", path.display())),
            "{marker}"
        );
        let range = marker
            .strip_prefix("[… lines ")
            .unwrap()
            .split(' ')
            .next()
            .unwrap();
        let (first, last) = range.split_once('–').unwrap();
        let (first, last): (u64, u64) = (first.parse().unwrap(), last.parse().unwrap());
        let shown: Vec<u64> = output
            .lines()
            .take_while(|line| !line.starts_with('['))
            .chain(
                output
                    .lines()
                    .skip_while(|line| !line.starts_with("[… lines"))
                    .skip(1),
            )
            .filter_map(|line| line.parse().ok())
            .collect();
        assert!(shown.iter().all(|&n| n < first || n > last), "{output}");
        assert!(!shown.contains(&first) && !shown.contains(&last));
        assert_eq!(
            shown.iter().filter(|&&n| n < first).max(),
            Some(&(first - 1))
        );
        assert_eq!(shown.iter().filter(|&&n| n > last).min(), Some(&(last + 1)));
        let read = sh(
            &format!(
                "sed -n '{first},{}p;{},{last}p' {}",
                first + 2,
                last - 2,
                path.display()
            ),
            &mut bash,
        );
        let read: Vec<u64> = read.lines().filter_map(|line| line.parse().ok()).collect();
        assert_eq!(
            read,
            [first, first + 1, first + 2, last - 2, last - 1, last]
        );
        // And the read tool, from the line the marker names, starts on the
        // first line left out.
        let page = crate::read::run(
            &serde_json::json!({ "path": path.to_str().unwrap(), "start": first }),
        );
        assert!(page.starts_with(&format!("{first}\t{first}\n")), "{page}");
        assert_eq!(
            sh(&format!("sed -n '500,502p' {}", path.display()), &mut bash),
            "500\n501\n502\n[exit code 0]"
        );

        drop(bash);
        assert!(!path.exists(), "the harness takes its files with it");
    }

    #[test]
    fn drains_both_streams_at_once() {
        let (done, finished) = mpsc::channel();
        thread::spawn(move || {
            let mut bash = Bash::new();
            let output = sh(
                "(head -c 300000 /dev/zero | tr '\\0' e >&2) & head -c 300000 /dev/zero | tr '\\0' o; wait",
                &mut bash,
            );
            let _ = done.send(output);
        });
        let output = finished
            .recv_timeout(Duration::from_secs(60))
            .expect("the command hung");
        assert!(
            output.contains("\n[stdout: 1 line, 300000 bytes, whole in "),
            "{output}"
        );
        assert!(
            output.contains("\n[stderr: 1 line, 300000 bytes, whole in "),
            "{output}"
        );
        assert!(output.ends_with("[exit code 0]"));
    }

    #[test]
    fn cuts_between_characters() {
        let mut bash = Bash::new();
        let output = sh("yes ä | head -c 5000", &mut bash);
        assert!(!output.contains('\u{FFFD}'), "{output}");
        assert!(output.starts_with("ä\nä\n"));
        assert!(output.contains("\nä\n[… lines "), "{output}");
        assert!(output.contains(" …]\nä\nä\n"), "{output}");
        assert_eq!(fs::metadata(bash.dir.join("1.stdout")).unwrap().len(), 5000);
        let output = sh("printf 'x\\377\\376y'", &mut bash);
        assert_eq!(output, "x\u{FFFD}\u{FFFD}y\n[exit code 0]");
        // A single long line is cut at character boundaries, not line ends.
        let output = sh("yes ä | head -c 5000 | tr -d '\\n'", &mut bash);
        assert!(!output.contains('\u{FFFD}'), "{output}");
        assert!(output.contains("ä\n[… line 1 ("), "{output}");
    }

    #[test]
    fn discards_past_the_cap_and_says_so() {
        let mut bash = Bash::new();
        bash.retain = 3000;
        let output = sh("seq 1 2000", &mut bash);
        let path = bash.dir.join("1.stdout");
        let note = output
            .lines()
            .find(|line| line.starts_with("[stdout: "))
            .unwrap();
        assert_eq!(
            note,
            format!(
                "[stdout: 2000 lines, 8893 bytes, the first 3000 bytes in {}, the rest discarded]",
                path.display()
            )
        );
        let expected: String = (1..=2000).map(|i| format!("{i}\n")).collect();
        assert_eq!(fs::read_to_string(&path).unwrap(), expected[..3000]);
        // The preview still shows the end, which was read even if not kept.
        assert!(output.contains("\n1999\n2000\n[stdout: "), "{output}");
    }

    #[test]
    fn reports_a_file_it_could_not_keep() {
        let mut bash = Bash::new();
        bash.dir = PathBuf::from("/dev/null/nowhere");
        let output = sh("seq 1 2000", &mut bash);
        let note = output
            .lines()
            .find(|line| line.starts_with("[stdout: "))
            .unwrap();
        assert!(
            note.starts_with("[stdout: 2000 lines, 8893 bytes, could not keep the rest: "),
            "{note}"
        );
        assert!(output.contains("\n1999\n2000\n[stdout: "), "{output}");
        assert!(output.ends_with("[exit code 0]"));
    }

    #[test]
    fn sweeps_after_dead_processes() {
        let tmp = env::temp_dir();
        let stale = tmp.join("dwim-4000000-0");
        let mine = tmp.join(format!("dwim-{}-sweep", process::id()));
        let other = tmp.join("dwim-gguf-4000000");
        for dir in [&stale, &mine, &other] {
            fs::create_dir_all(dir).unwrap();
        }
        sweep(&tmp);
        assert!(!stale.exists());
        assert!(mine.exists() && other.exists());
        fs::remove_dir_all(mine).unwrap();
        fs::remove_dir_all(other).unwrap();
    }

    #[test]
    fn needs_a_command() {
        assert!(
            Bash::new()
                .run(&serde_json::json!({}))
                .starts_with("error: bash needs")
        );
    }

    #[test]
    fn parses_coder_calls() {
        let mut bash = Bash::new();
        let ls = call("<function=bash>\n<parameter=command>\nls -l\n</parameter>\n</function>");
        assert_eq!(ls.name, "bash");
        let echo =
            call("<function=bash>\n<parameter=command>\necho a\necho b\n</parameter>\n</function>");
        assert_eq!(bash.run(&echo.arguments), "a\nb\n[exit code 0]");
        assert!(ToolCall::parse("<function=bash>\n<parameter=command>\nls").is_err());
    }
}
