//! The agent around the model: runs the tools it calls and feeds the
//! results back, until it replies with text alone.
//!
//! A [`Harness`] wraps a [`Chat`] and reports each turn's thoughts, text,
//! tool calls, and tool output as [`Event`]s, so a user interface can show
//! them as they happen. The one tool is `bash`, which runs a shell command,
//! and the system prompt pushes the model to use it rather than answer from
//! memory or ask the user for a command. It also tells the model about the
//! project it works in, since a model won't always go looking on its own.
//!
//! What a command prints goes back to the model as a preview, its standard
//! output apart from its standard error, with a line of how it ended that
//! is always there. A stream that outgrows its preview is shown by its
//! start and its end, and is kept whole in a file of the harness's own,
//! which the result names, so the model can read the rest of it with the
//! same tool instead of running the command again.

use std::{
    collections::VecDeque,
    env,
    error::Error,
    fs::{self, DirBuilder, File},
    io::{self, ErrorKind, Read, Write},
    ops::ControlFlow,
    os::unix::{fs::DirBuilderExt, process::ExitStatusExt},
    path::{Path, PathBuf},
    process::{self, Command, ExitStatus, Stdio},
    sync::atomic::{AtomicUsize, Ordering},
    thread,
    time::{SystemTime, UNIX_EPOCH},
};

use dwim_models::{Chat, Chunk, LanguageModel, ToolCall};

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

/// Most of a project's `AGENTS.md` that goes into the system prompt.
const MAX_INSTRUCTIONS: usize = 4000;

/// Most of the files in the working directory the system prompt lists.
const MAX_FILES: usize = 50;

/// What the model gets back when it makes the same call twice in a row,
/// instead of running it again: nothing ran in between to change its output,
/// and small models otherwise tend to repeat a call over and over.
const REPEATED: &str = "error: you just ran this, and its output is above. Don't run it again: use that output, run something else, or reply to the user.";

/// How the agent should behave: the start of the system prompt.
const INSTRUCTIONS: &str = r#"You are `dwim`, a coding agent working in the user's project directory at a Unix command line. You have a bash tool that runs shell commands there, and you may use it at any time without asking.

- For anything about the project, its files, its git history, or the system, run commands to find out before you answer. Don't answer from memory when a command can tell you.
- Never say you can't access files or run commands, and never ask the user which command to run: pick one yourself.
- When a request could be a question or a task, treat it as a task and do it.
- To change something, run the commands that change it instead of explaining how.
- If a command fails, read the error and try another way. A command's result ends with its exit code, and what it printed to standard error comes after a `[stderr]` line.
- When a command prints more than fits, the result shows the start and the end of its output and names a file that holds all of it: read the part you need from the file with `sed -n 'A,Bp'` or `grep -n` instead of running the command again.
- Keep going until the request is done, then reply in a few sentences with what you found or did.

For example, for "review commit abc123", run `git show abc123` and point out bugs and risks in the change; for "what files are here?", run `ls`; for "what time is it?", run `date`."#;

/// The tools, declared as Bonsai's chat template puts them: the start of
/// the system prompt, before what the user's system prompt says.
const TOOLS: &str = r#"# Tools

You have access to the following functions:

<tools>
{"type": "function", "function": {"name": "bash", "description": "Run a shell command and return its output.", "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "The command to run."}}, "required": ["command"]}}}
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>"#;

/// The system prompt for an agent working in `dir`: the tools, then how to
/// behave, where it is and what the project looks like, and the project's
/// own instructions from its `AGENTS.md` if it has one.
pub fn system_prompt(dir: &Path) -> String {
    let mut prompt = format!("{TOOLS}\n\n{INSTRUCTIONS}\n\n{}", environment(dir));
    if let Ok(instructions) = fs::read_to_string(dir.join("AGENTS.md")) {
        let instructions = truncate(instructions.trim(), MAX_INSTRUCTIONS);
        prompt.push_str(&format!("\n\n# Project instructions\n\nFrom AGENTS.md:\n\n{instructions}"));
    }
    prompt
}

/// Where the agent is: the working directory and the files in it, whether
/// it is a git repository, the platform, and the date.
fn environment(dir: &Path) -> String {
    let branch = Command::new("git")
        .args(["branch", "--show-current"])
        .current_dir(dir)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string());
    let git = match branch {
        Some(branch) if !branch.is_empty() => format!("yes, on branch {branch}"),
        Some(_) => "yes".to_string(),
        None => "no".to_string(),
    };
    let platform = match std::env::consts::OS {
        "macos" => "macOS",
        "linux" => "Linux",
        os => os,
    };
    let days = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |since| since.as_secs() / 86400);
    format!(
        "# Environment\n\nWorking directory: {}\nFiles: {}\nGit repository: {git}\nPlatform: {platform}\nDate: {}",
        dir.display(),
        files(dir),
        date(days as i64),
    )
}

/// The files and directories in `dir`, directories with a trailing slash,
/// leaving out hidden ones.
fn files(dir: &Path) -> String {
    let Ok(entries) = fs::read_dir(dir) else {
        return String::new();
    };
    let mut names: Vec<String> = entries
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name().into_string().ok()?;
            match entry.file_type().ok()? {
                _ if name.starts_with('.') => None,
                kind if kind.is_dir() => Some(format!("{name}/")),
                _ => Some(name),
            }
        })
        .collect();
    names.sort();
    if names.len() > MAX_FILES {
        names.truncate(MAX_FILES);
        names.push("…".to_string());
    }
    names.join(" ")
}

/// The date `days` after 1970-01-01, as year-month-day.
fn date(days: i64) -> String {
    // Howard Hinnant's civil_from_days: count in 400-year eras of 146097
    // days, from a year that starts in March so that leap days come last.
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let day_of_era = z.rem_euclid(146097);
    let year_of_era = (day_of_era - day_of_era / 1460 + day_of_era / 36524 - day_of_era / 146096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_index = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_index + 2) / 5 + 1;
    let month = if month_index < 10 { month_index + 3 } else { month_index - 9 };
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    format!("{year:04}-{month:02}-{day:02}")
}

/// Cuts `text` down to at most `max` bytes, on a character boundary, marking
/// the cut with an ellipsis.
fn truncate(text: &str, max: usize) -> String {
    if text.len() <= max {
        return text.to_string();
    }
    let end = (0..=max).rev().find(|&i| text.is_char_boundary(i)).unwrap_or(0);
    format!("{}…", &text[..end])
}

/// What happens during a turn, as it happens.
pub enum Event<'a> {
    /// Part of the model's thought, before it replies.
    Thought(&'a str),
    /// Text of the reply.
    Text(&'a str),
    /// A tool is about to run: its name and how it was called.
    Call { name: &'a str, detail: &'a str },
    /// What the tool returned, as the model sees it.
    Output(&'a str),
}

/// The loop around a [`Chat`] that runs the tools the model calls and feeds
/// the results back, until the model replies with text alone.
pub struct Harness<M: LanguageModel> {
    chat: Chat<M>,
    outputs: Outputs,
}

impl<M: LanguageModel> Harness<M> {
    pub fn new(chat: Chat<M>) -> Self {
        Self {
            chat,
            outputs: Outputs::new(),
        }
    }

    /// Number of tokens in the conversation so far.
    pub fn tokens(&self) -> usize {
        self.chat.tokens()
    }

    /// Sends a message from the user, reporting the reply and any tool
    /// calls it makes to `on_event`. The turn ends early if `on_event`
    /// breaks.
    pub fn send(
        &mut self,
        message: &str,
        mut on_event: impl FnMut(Event) -> ControlFlow<()>,
    ) -> Result<(), Box<dyn Error>> {
        let mut calls = self.chat.send(message, |chunk| on_event(event(chunk)))?;
        // The last call run, as its name and arguments.
        let mut last = None;
        while !calls.is_empty() {
            let mut outputs = Vec::new();
            for call in &calls {
                let output = match ToolCall::parse(call) {
                    Ok(call) => {
                        let detail = describe(&call);
                        if on_event(Event::Call {
                            name: &call.name,
                            detail: &detail,
                        })
                        .is_break()
                        {
                            return Ok(());
                        }
                        let this = Some((call.name.clone(), call.arguments.to_string()));
                        if this == last {
                            REPEATED.to_string()
                        } else {
                            last = this;
                            run(&call, &mut self.outputs)
                        }
                    }
                    Err(e) => format!("error: malformed tool call: {e}"),
                };
                if on_event(Event::Output(&output)).is_break() {
                    return Ok(());
                }
                outputs.push(output);
            }
            calls = self.chat.respond(&outputs, |chunk| on_event(event(chunk)))?;
        }
        Ok(())
    }
}

/// The event for a piece of the model's reply.
fn event(chunk: Chunk) -> Event {
    match chunk {
        Chunk::Thought(text) => Event::Thought(text),
        Chunk::Text(text) => Event::Text(text),
    }
}

/// How a call reads on screen: the command for `bash`, and the arguments
/// as written for anything else.
fn describe(call: &ToolCall) -> String {
    match call.arguments.get("command").and_then(|command| command.as_str()) {
        Some(command) if call.name == "bash" => command.to_string(),
        _ => call.arguments.to_string(),
    }
}

/// Where a command's output is kept when there is more of it than the
/// model is shown: a directory of this process's own under the system's
/// temporary directory, readable by this user alone, with a file per
/// stream of each command that outgrew its preview, named by the command's
/// number, as `3.stdout`. The directory is made when first needed and
/// removed when the harness is dropped. A process that ends without
/// dropping it, as the shell does while the model is busy, leaves the
/// directory behind, and the next harness to start removes what earlier
/// processes that are no longer running left.
struct Outputs {
    dir: PathBuf,
    /// Commands run so far; the next one's files are named by the count.
    calls: usize,
    /// Most bytes of a stream kept in its file.
    retain: u64,
}

/// Tells apart the output directories of the harnesses of one process.
static OUTPUT_DIRS: AtomicUsize = AtomicUsize::new(0);

impl Outputs {
    fn new() -> Self {
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
}

impl Drop for Outputs {
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
    fn new(name: &'static str, outputs: &Outputs) -> Self {
        Self {
            name,
            head: Vec::new(),
            tail: VecDeque::new(),
            total: 0,
            newlines: 0,
            kept: 0,
            file: None,
            dir: outputs.dir.clone(),
            path: outputs.file(name),
            retain: outputs.retain,
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
            DirBuilder::new().mode(0o700).recursive(true).create(&self.dir)?;
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
        let last = self.newlines - tail.iter().filter(|&&b| b == b'\n').count() as u64 + u64::from(midline);
        let bytes = self.total - head.len() as u64 - tail.len() as u64;
        let lines = if first == last { format!("line {first}") } else { format!("lines {first}–{last}") };
        format!(
            "{}\n[… {lines} ({bytes} bytes) not shown …]\n{}",
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
        let mut parts = vec![format!("{} {}, {} bytes", self.lines(), plural(self.lines(), "line"), self.total)];
        let path = self.path.display();
        if self.file.is_some() {
            parts.push(if self.kept == self.total {
                format!("whole in {path}")
            } else {
                format!("the first {} bytes in {path}, the rest discarded", self.kept)
            });
        }
        parts.extend(self.error.clone());
        Some(format!("[{}: {}]", self.name, parts.join(", ")))
    }
}

/// `word`, or its plural if `n` is not one.
fn plural(n: u64, word: &str) -> String {
    if n == 1 { word.to_string() } else { format!("{word}s") }
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
        return if bytes.len() - i < need { i } else { bytes.len() };
    }
    bytes.len()
}

/// Where a cut at the start of `bytes` leaves no partial UTF-8 character:
/// past the continuation bytes of one cut off before them.
fn char_start(bytes: &[u8]) -> usize {
    bytes.iter().take(3).position(|&b| b & 0xC0 != 0x80).unwrap_or(bytes.len().min(3))
}

/// Runs a tool call, returning what the model should see.
fn run(call: &ToolCall, outputs: &mut Outputs) -> String {
    if call.name != "bash" {
        return format!("error: unknown tool '{}'", call.name);
    }
    let Some(command) = call.arguments.get("command").and_then(|command| command.as_str()) else {
        return "error: bash needs a 'command' string".to_string();
    };
    outputs.calls += 1;
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
    let mut stdout = Stream::new("stdout", outputs);
    let mut stderr = Stream::new("stderr", outputs);
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

    use super::*;

    fn call(json: &str) -> ToolCall {
        ToolCall::parse(json).unwrap()
    }

    /// Runs `command` through the bash tool.
    fn bash(command: &str, outputs: &mut Outputs) -> String {
        let arguments = serde_json::json!({ "name": "bash", "arguments": { "command": command } });
        run(&call(&arguments.to_string()), outputs)
    }

    #[test]
    fn runs_bash() {
        let mut outputs = Outputs::new();
        assert_eq!(bash("echo hi; echo err >&2", &mut outputs), "hi\n[stderr]\nerr\n[exit code 0]");
        assert_eq!(bash("exit 3", &mut outputs), "[exit code 3]");
        assert_eq!(bash("true", &mut outputs), "[exit code 0]");
        assert_eq!(bash("echo; echo", &mut outputs), "[exit code 0]");
        assert_eq!(bash("kill -9 $$", &mut outputs), "[killed by signal 9 (SIGKILL)]");
        assert!(!outputs.dir.exists(), "nothing outgrew its preview");
    }

    #[test]
    fn keeps_the_error_and_the_status_after_a_flood() {
        let mut outputs = Outputs::new();
        let output = bash("seq 1 50000; echo 'boom: it broke' >&2; exit 7", &mut outputs);
        assert!(output.starts_with("1\n2\n3\n"), "{output}");
        assert!(output.contains("\n[… lines "), "{output}");
        assert!(output.contains("\n49999\n50000\n[stderr]\nboom: it broke\n"), "{output}");
        assert!(output.contains("\n[stdout: 50000 lines, 288894 bytes, whole in "), "{output}");
        assert!(output.ends_with("\n[exit code 7]"), "{output}");
        assert!(output.len() < PREVIEW + 400, "{}", output.len());
    }

    #[test]
    fn keeps_long_output_for_reading_later() {
        let mut outputs = Outputs::new();
        let output = bash("seq 1 50000", &mut outputs);
        let path = outputs.dir.join("1.stdout");
        let note = output.lines().find(|line| line.starts_with("[stdout: ")).unwrap();
        assert!(note.ends_with(&format!("whole in {}]", path.display())), "{note}");
        let expected: String = (1..=50000).map(|i| format!("{i}\n")).collect();
        assert_eq!(fs::read_to_string(&path).unwrap(), expected);

        // The lines the preview says it left out are the ones between what
        // it shows, so a range read gets what is missing.
        let marker = output.lines().find(|line| line.starts_with("[… lines ")).unwrap();
        let range = marker.strip_prefix("[… lines ").unwrap().split(' ').next().unwrap();
        let (first, last) = range.split_once('–').unwrap();
        let (first, last): (u64, u64) = (first.parse().unwrap(), last.parse().unwrap());
        let shown: Vec<u64> = output
            .lines()
            .take_while(|line| !line.starts_with('['))
            .chain(output.lines().skip_while(|line| !line.starts_with("[… lines")).skip(1))
            .filter_map(|line| line.parse().ok())
            .collect();
        assert!(shown.iter().all(|&n| n < first || n > last), "{output}");
        assert!(!shown.contains(&first) && !shown.contains(&last));
        assert_eq!(shown.iter().filter(|&&n| n < first).max(), Some(&(first - 1)));
        assert_eq!(shown.iter().filter(|&&n| n > last).min(), Some(&(last + 1)));
        let read = bash(&format!("sed -n '{first},{}p;{},{last}p' {}", first + 2, last - 2, path.display()), &mut outputs);
        let read: Vec<u64> = read.lines().filter_map(|line| line.parse().ok()).collect();
        assert_eq!(read, [first, first + 1, first + 2, last - 2, last - 1, last]);
        assert_eq!(bash(&format!("sed -n '500,502p' {}", path.display()), &mut outputs), "500\n501\n502\n[exit code 0]");

        drop(outputs);
        assert!(!path.exists(), "the harness takes its files with it");
    }

    #[test]
    fn drains_both_streams_at_once() {
        let (done, finished) = mpsc::channel();
        thread::spawn(move || {
            let mut outputs = Outputs::new();
            let output = bash(
                "(head -c 300000 /dev/zero | tr '\\0' e >&2) & head -c 300000 /dev/zero | tr '\\0' o; wait",
                &mut outputs,
            );
            let _ = done.send(output);
        });
        let output = finished.recv_timeout(Duration::from_secs(60)).expect("the command hung");
        assert!(output.contains("\n[stdout: 1 line, 300000 bytes, whole in "), "{output}");
        assert!(output.contains("\n[stderr: 1 line, 300000 bytes, whole in "), "{output}");
        assert!(output.ends_with("[exit code 0]"));
    }

    #[test]
    fn cuts_between_characters() {
        let mut outputs = Outputs::new();
        let output = bash("yes ä | head -c 5000", &mut outputs);
        assert!(!output.contains('\u{FFFD}'), "{output}");
        assert!(output.starts_with("ä\nä\n"));
        assert!(output.contains("\nä\n[… lines "), "{output}");
        assert!(output.contains(" not shown …]\nä\nä\n"), "{output}");
        assert_eq!(fs::metadata(outputs.dir.join("1.stdout")).unwrap().len(), 5000);
        let output = bash("printf 'x\\377\\376y'", &mut outputs);
        assert_eq!(output, "x\u{FFFD}\u{FFFD}y\n[exit code 0]");
        // A single long line is cut at character boundaries, not line ends.
        let output = bash("yes ä | head -c 5000 | tr -d '\\n'", &mut outputs);
        assert!(!output.contains('\u{FFFD}'), "{output}");
        assert!(output.contains("ä\n[… line 1 ("), "{output}");
    }

    #[test]
    fn discards_past_the_cap_and_says_so() {
        let mut outputs = Outputs::new();
        outputs.retain = 3000;
        let output = bash("seq 1 2000", &mut outputs);
        let path = outputs.dir.join("1.stdout");
        let note = output.lines().find(|line| line.starts_with("[stdout: ")).unwrap();
        assert_eq!(
            note,
            format!("[stdout: 2000 lines, 8893 bytes, the first 3000 bytes in {}, the rest discarded]", path.display())
        );
        let expected: String = (1..=2000).map(|i| format!("{i}\n")).collect();
        assert_eq!(fs::read_to_string(&path).unwrap(), expected[..3000]);
        // The preview still shows the end, which was read even if not kept.
        assert!(output.contains("\n1999\n2000\n[stdout: "), "{output}");
    }

    #[test]
    fn reports_a_file_it_could_not_keep() {
        let mut outputs = Outputs::new();
        outputs.dir = PathBuf::from("/dev/null/nowhere");
        let output = bash("seq 1 2000", &mut outputs);
        let note = output.lines().find(|line| line.starts_with("[stdout: ")).unwrap();
        assert!(note.starts_with("[stdout: 2000 lines, 8893 bytes, could not keep the rest: "), "{note}");
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
    fn refuses_what_it_does_not_know() {
        let mut outputs = Outputs::new();
        assert!(run(&call(r#"{"name": "rm", "arguments": {}}"#), &mut outputs).starts_with("error: unknown tool"));
        assert!(run(&call(r#"{"name": "bash", "arguments": {}}"#), &mut outputs).starts_with("error: bash needs"));
        assert!(ToolCall::parse("not json").is_err());
    }

    #[test]
    fn dates() {
        assert_eq!(date(0), "1970-01-01");
        assert_eq!(date(11016), "2000-02-29");
        assert_eq!(date(19782), "2024-02-29");
        assert_eq!(date(20711), "2026-09-15");
    }

    #[test]
    fn describes_the_project() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        let prompt = system_prompt(dir);
        assert!(prompt.starts_with(TOOLS));
        assert!(prompt.contains(INSTRUCTIONS));
        let files = prompt.lines().find_map(|line| line.strip_prefix("Files: ")).unwrap();
        assert!(files.split(' ').any(|file| file == "Cargo.toml"));
        assert!(files.split(' ').any(|file| file == "harness/"));
        assert!(!files.contains(".git"));
        assert!(prompt.contains("From AGENTS.md:\n\n# `dwim`"));
    }

    #[test]
    fn parses_coder_calls() {
        let mut outputs = Outputs::new();
        let ls = call("<function=bash>\n<parameter=command>\nls -l\n</parameter>\n</function>");
        assert_eq!(ls.name, "bash");
        assert_eq!(describe(&ls), "ls -l");
        let echo = call("<function=bash>\n<parameter=command>\necho a\necho b\n</parameter>\n</function>");
        assert_eq!(run(&echo, &mut outputs), "a\nb\n[exit code 0]");
        assert!(ToolCall::parse("<function=bash>\n<parameter=command>\nls").is_err());
    }

    #[test]
    fn describes_calls() {
        assert_eq!(describe(&call(r#"{"name": "bash", "arguments": {"command": "ls -l"}}"#)), "ls -l");
        assert_eq!(describe(&call(r#"{"name": "other", "arguments": {"x": 1}}"#)), r#"{"x":1}"#);
    }
}
