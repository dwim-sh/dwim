//! The `bash` tool: runs a shell command, in the working directory or one
//! the call names, for at most a timeout, and gives the model what it
//! printed as a preview, its standard output apart from its standard
//! error, with a line of how it ended that is always there. A command
//! that runs to its timeout is killed, with what it started, and the
//! model is told, so that a test that hangs or a server that was never
//! going to end does not hold the session until the user notices. A stream
//! that outgrows its preview is shown by its start and its end, and is
//! kept whole in a file of the tool's own, which the result names along
//! with the line to `read` it from, so the model can get the rest with
//! that tool instead of running the command again. A pipeline ends with
//! the status of its last command to fail, not of its last command: the
//! model pipes builds and tests through `head` and `tail` to keep their
//! output short, and with the shell's own rule every one of them ended
//! with `head`'s zero, so a failed build read as a success to the model
//! and to the harness alike.
//!
//! A command runs with no terminal: in a session of its own, with the
//! variables set that tell git and the programs it runs to do without an
//! editor, a pager, or a prompt. Without them, `git commit --amend` with a
//! trailer and no `--no-edit` opened `vi` on the terminal the user was
//! looking at dwim in, and the session sat at "Reading…" until the user
//! went looking for why, as a program that asks on the terminal, such as
//! `less`, `sudo`, or `ssh` at a new host, would too. With no terminal to
//! ask on, they fail, and the model sees the error.
//!
//! A command that would take away changes the model has made to the
//! project's tracked files is not run: on SWE-bench's django-11797 the
//! model made the fix, then ran `git stash` to see its new test fail
//! without it, and the run ended with nothing changed.

use std::{
    collections::VecDeque,
    env,
    fs::{self, DirBuilder, File},
    io::{self, ErrorKind, Read, Write},
    os::unix::{
        fs::DirBuilderExt,
        process::{CommandExt, ExitStatusExt},
    },
    path::{Path, PathBuf},
    process::{self, Command, ExitStatus, Stdio},
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc,
    },
    thread,
    time::Duration,
};

use serde_json::Value;

use crate::{
    sys,
    tools::{Parameter, Signature, Tool, Tools, optional_number, plural, required, string},
};

const NAME: &str = "bash";

/// The environment every command runs with, over the user's: no editor,
/// since there is no one to type into one, so that `git commit --amend`,
/// `git rebase` and the like take the message they are given; no pager,
/// so that `git log` and `git diff` print and end; and no prompt for a
/// password or a passphrase, which could only hang.
const ENVIRONMENT: &[(&str, &str)] = &[
    ("GIT_EDITOR", "true"),
    ("EDITOR", "true"),
    ("VISUAL", "true"),
    ("GIT_PAGER", "cat"),
    ("PAGER", "cat"),
    ("GIT_TERMINAL_PROMPT", "0"),
    ("GIT_SEQUENCE_EDITOR", "true"),
];

pub const TOOL: Tool = Tool {
    name: NAME,
    signature,
    changes: false,
    describe,
    run,
};

fn signature() -> Signature {
    Signature {
        description: "Run a shell command in the working directory and return what it printed and its exit code (a pipeline's is that of its last command to fail). Long output is cut to its start and end, and kept whole in a file the result names for `read`. A command still running at its timeout is killed.".to_string(),
        parameters: vec![
            Parameter::string("command", "The command to run."),
            Parameter::integer(
                "timeout",
                format!(
                    "Seconds the command may run (default {TIMEOUT}; 0 for no limit). Give a long build or test run more."
                ),
            )
            .optional(),
            Parameter::string(
                "workdir",
                "The directory to run it in (default: the working directory).",
            )
            .optional(),
        ],
    }
}

/// How a call reads on screen: the command.
fn describe(arguments: &Value) -> Option<String> {
    string(arguments, "command").map(str::to_string)
}

fn run(tools: &mut Tools, arguments: &Value) -> Result<String, String> {
    tools.bash.run(&tools.dir, arguments)
}

/// Seconds a command may run unless the call says otherwise: long enough
/// for a build or a test run, short enough that a command that hangs, or
/// a server that was never going to end, does not hold the session long.
pub const TIMEOUT: u64 = 300;

/// What the model is told for a command that would take away changes it
/// has made to the project's tracked files.
const DISCARDS: &str = "not run: this would take away the changes you have made to the project's files (git stash, checkout or restore of files, reset --hard, or clean), and those changes are the work. To run the code without your change, undo it with edit, run what you want, and then make the change again with edit. git diff shows your changes.";

/// Most of one stream of a command's output that goes back to the model as
/// it is, so that a chatty command can't fill the context window. A longer
/// stream is shown by its first and last [`HALF`] bytes, cut at line ends
/// where it can be, and kept whole on disk for the model to read from. A
/// test run or a diff of a few hundred lines comes whole: a preview of a
/// couple of thousand bytes had the model read the file of nearly every
/// build and test, a round trip each.
const PREVIEW: usize = 32 << 10;

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
    /// The program commands run through, as `shell -c command`: `sh`,
    /// unless another is set.
    shell: PathBuf,
    outputs: PathBuf,
    /// Commands run so far; the next one's files are named by the count.
    calls: usize,
    /// Most bytes of a stream kept in its file.
    retain: u64,
}

/// Tells apart the output directories of the tools of one process.
static OUTPUT_DIRS: AtomicUsize = AtomicUsize::new(0);

impl Default for Bash {
    fn default() -> Self {
        let tmp = env::temp_dir();
        sweep(&tmp);
        let n = OUTPUT_DIRS.fetch_add(1, Ordering::Relaxed);
        Self {
            shell: PathBuf::from("sh"),
            outputs: tmp.join(format!("dwim-{}-{n}", process::id())),
            calls: 0,
            retain: RETAINED,
        }
    }
}

impl Bash {
    /// Runs commands through `shell`, as `shell -c command`, instead of
    /// `sh`: a script that runs them somewhere else, such as in a container
    /// that has what the project needs installed.
    pub fn set_shell(&mut self, shell: PathBuf) {
        self.shell = shell;
    }

    /// The file for the `stream` of the command being run.
    fn file(&self, stream: &str) -> PathBuf {
        self.outputs.join(format!("{}.{stream}", self.calls))
    }

    /// Runs the command in `arguments` in `dir`, or in the directory the
    /// call names, returning what the model should see; not one that would
    /// discard the model's changes to the project at `dir`.
    fn run(&mut self, dir: &Path, arguments: &Value) -> Result<String, String> {
        let command = required(arguments, NAME, "command")?;
        let timeout = match optional_number(
            arguments,
            "timeout",
            "a number of seconds, or 0 for no limit",
        )? {
            None => Some(Duration::from_secs(TIMEOUT)),
            Some(0) => None,
            Some(seconds) => Some(Duration::from_secs(seconds)),
        };
        let workdir = string(arguments, "workdir").map(|workdir| dir.join(workdir));
        if let Some(dir) = &workdir
            && !dir.is_dir()
        {
            return Err(format!("workdir {} is not a directory", dir.display()));
        }
        if discards(command, dir) && changed(dir) {
            return Err(DISCARDS.to_string());
        }
        self.calls += 1;
        // `set -o pipefail` first, by itself, so that a shell without it
        // runs the command all the same.
        let command = format!("set -o pipefail 2>/dev/null\n{command}");
        let mut shell = Command::new(&self.shell);
        shell
            .args(["-c", &command])
            .envs(ENVIRONMENT.iter().copied())
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .current_dir(workdir.as_deref().unwrap_or(dir));
        // In a session of its own, which is a process group of its own
        // with the command's ID, so that the timeout ends what the
        // command started along with it, and with no controlling
        // terminal, so that nothing it runs can open the user's.
        unsafe {
            shell.pre_exec(|| {
                if libc::setsid() == -1 {
                    return Err(io::Error::last_os_error());
                }
                Ok(())
            });
        }
        let mut child = shell
            .spawn()
            .map_err(|e| format!("could not run {}: {e}", self.shell.display()))?;
        let pid = child.id();
        let mut stdout = Stream::new("stdout", self);
        let mut stderr = Stream::new("stderr", self);
        let (out, err) = (child.stdout.take().unwrap(), child.stderr.take().unwrap());
        let (done, finished) = mpsc::channel::<()>();
        let timed_out = AtomicBool::new(false);
        // Each pipe is read on its own thread, so that neither fills up and
        // blocks the command while the other is being read; a third kills
        // the command's process group at the timeout, which closes the
        // pipes too, unless the command ends first.
        let waited = thread::scope(|s| {
            s.spawn(|| stderr.drain(err));
            if let Some(timeout) = timeout {
                let timed_out = &timed_out;
                s.spawn(move || {
                    if finished.recv_timeout(timeout).is_err() {
                        timed_out.store(true, Ordering::Relaxed);
                        unsafe { libc::kill(-(pid as libc::pid_t), libc::SIGKILL) };
                    }
                });
            }
            stdout.drain(out);
            let waited = child.wait();
            let _ = done.send(());
            waited
        });
        let status = match waited {
            Ok(_) if timed_out.load(Ordering::Relaxed) => format!(
                "[killed after {} s, the timeout; give a longer timeout if the command needs it, or 0 for none]",
                timeout.unwrap_or_default().as_secs()
            ),
            Ok(status) => ended(status),
            Err(e) => format!("[error: could not wait for {}: {e}]", self.shell.display()),
        };
        Ok(report(&stdout, &stderr, &status))
    }
}

impl Drop for Bash {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.outputs);
    }
}

/// Whether a shell command would take away changes to the files of the git
/// work tree at `dir`: `git stash` that saves them, `git checkout` of files,
/// named with `--`, as `.`, or by a path that is there, or forced, `git
/// restore` of files, `git reset --hard`, or `git clean` with force, in any
/// of the commands it chains.
fn discards(command: &str, dir: &Path) -> bool {
    command.split(['\n', ';', '&', '|', '(', ')']).any(|part| {
        let words: Vec<&str> = part.split_whitespace().collect();
        let Some(at) = words.iter().position(|word| *word == "git") else {
            return false;
        };
        // The subcommand, past git's own options and their values.
        let mut rest = words[at + 1..].iter().copied();
        let subcommand = loop {
            match rest.next() {
                Some("-C" | "-c" | "--git-dir" | "--work-tree") => {
                    rest.next();
                }
                Some(word) if word.starts_with('-') => {}
                other => break other,
            }
        };
        let args: Vec<&str> = rest.collect();
        match subcommand {
            Some("stash") => args
                .first()
                .is_none_or(|first| ["push", "save"].contains(first) || first.starts_with('-')),
            Some("checkout") => args.iter().any(|arg| {
                ["--", ".", "-f", "--force"].contains(arg)
                    || !arg.starts_with('-') && dir.join(arg).exists()
            }),
            Some("restore") => {
                !args.contains(&"--staged")
                    || args.iter().any(|arg| ["--worktree", "-W"].contains(arg))
            }
            Some("reset") => args.contains(&"--hard"),
            Some("clean") => args.iter().any(|arg| {
                arg.starts_with('-') && !arg.starts_with("--") && arg.contains('f')
                    || *arg == "--force"
            }),
            _ => false,
        }
    })
}

/// Whether the git work tree at `dir` has changes to its tracked files; not
/// if it is not one.
fn changed(dir: &Path) -> bool {
    Command::new("git")
        .args(["status", "--porcelain", "--untracked-files=no"])
        .current_dir(dir)
        .output()
        .is_ok_and(|output| output.status.success() && !output.stdout.is_empty())
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
            && !sys::process::alive(pid)
        {
            let _ = fs::remove_dir_all(entry.path());
        }
    }
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
            dir: bash.outputs.clone(),
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
            plural(self.lines(), "line", "lines"),
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
    use std::{os::unix::fs::PermissionsExt, sync::mpsc, time::Duration};

    use super::*;
    use crate::tools::ToolCall;

    fn call(text: &str) -> ToolCall {
        ToolCall::parse(text).unwrap()
    }

    /// Runs the call in `arguments` in the working directory, as the model
    /// sees the result.
    fn run(bash: &mut Bash, arguments: &Value) -> String {
        crate::tools::Output::of(bash.run(Path::new("."), arguments), false).text
    }

    /// Runs `command` through the tool.
    fn sh(command: &str, bash: &mut Bash) -> String {
        run(bash, &serde_json::json!({ "command": command }))
    }

    #[test]
    fn runs_bash() {
        let mut bash = Bash::default();
        assert_eq!(
            sh("echo hi; echo err >&2", &mut bash),
            "hi\n[stderr]\nerr\n[exit code 0]"
        );
        assert_eq!(sh("exit 3", &mut bash), "[exit code 3]");
        assert_eq!(sh("true", &mut bash), "[exit code 0]");
        assert_eq!(sh("echo; echo", &mut bash), "[exit code 0]");
        // A pipeline fails when any of its commands does.
        assert_eq!(sh("false | true", &mut bash), "[exit code 1]");
        assert_eq!(sh("exit 4 | head -1", &mut bash), "[exit code 4]");
        assert_eq!(sh("echo hi | head -1", &mut bash), "hi\n[exit code 0]");
        assert_eq!(
            sh("kill -9 $$", &mut bash),
            "[killed by signal 9 (SIGKILL)]"
        );
        assert!(!bash.outputs.exists(), "nothing outgrew its preview");
    }

    #[test]
    fn runs_through_the_shell_set() {
        let script = env::temp_dir().join(format!("dwim-shell-{}", process::id()));
        fs::write(&script, "#!/bin/sh\necho via; exec sh \"$@\"\n").unwrap();
        fs::set_permissions(&script, fs::Permissions::from_mode(0o700)).unwrap();
        let mut bash = Bash::default();
        bash.set_shell(script.clone());
        assert_eq!(sh("echo hi", &mut bash), "via\nhi\n[exit code 0]");
        assert_eq!(sh("exit 3", &mut bash), "via\n[exit code 3]");
        fs::remove_file(script).unwrap();
        bash.set_shell(PathBuf::from("/nonexistent/shell"));
        assert!(sh("true", &mut bash).starts_with("error: could not run /nonexistent/shell: "));
    }

    #[test]
    fn kills_a_command_at_its_timeout() {
        // Timed loosely: on a loaded machine the shell may start late, so
        // what matters is that the command is killed, and long before it
        // would have ended by itself.
        let mut bash = Bash::default();
        let start = std::time::Instant::now();
        let output = run(
            &mut bash,
            &serde_json::json!({
                "command": "echo start; sleep 60; echo end",
                "timeout": 2
            }),
        );
        assert!(start.elapsed() < Duration::from_secs(30));
        assert!(!output.contains("end"), "{output}");
        assert!(
            output.ends_with(
                "[killed after 2 s, the timeout; give a longer timeout if the command needs it, or 0 for none]"
            ),
            "{output}"
        );
        // What the command started goes with it, so the pipes close.
        let start = std::time::Instant::now();
        let output = run(
            &mut bash,
            &serde_json::json!({
                "command": "sleep 60 & sleep 60",
                "timeout": "2"
            }),
        );
        assert!(start.elapsed() < Duration::from_secs(30));
        assert!(output.ends_with(", or 0 for none]"), "{output}");
        assert_eq!(
            run(
                &mut bash,
                &serde_json::json!({ "command": "true", "timeout": 0 })
            ),
            "[exit code 0]"
        );
        assert!(
            run(
                &mut bash,
                &serde_json::json!({ "command": "true", "timeout": "soon" })
            )
            .starts_with("error: 'timeout' is")
        );
    }

    #[test]
    fn runs_with_no_terminal() {
        let mut bash = Bash::default();
        // The variables that keep git and its programs from asking.
        assert_eq!(
            sh(
                "echo $GIT_EDITOR $EDITOR $GIT_PAGER $GIT_TERMINAL_PROMPT",
                &mut bash
            ),
            "true true cat 0\n[exit code 0]"
        );
        // No controlling terminal to ask on, even in a dwim run from one.
        assert_eq!(
            sh(
                "(exec < /dev/tty) 2>/dev/null && echo tty || echo none",
                &mut bash
            ),
            "none\n[exit code 0]"
        );
        // An amend with a trailer and no `--no-edit` opens the editor, which
        // takes the message as it is instead of hanging.
        let dir = env::temp_dir().join(format!("dwim-amend-{}", process::id()));
        fs::create_dir_all(&dir).unwrap();
        let git = "git -c user.name=t -c user.email=t@t -c commit.gpgsign=false";
        let out = run(
            &mut bash,
            &serde_json::json!({
                "command": format!(
                    "git init -q . && {git} commit -q --allow-empty -m first && \
                     {git} commit -q --allow-empty --amend --trailer=Generated-by:dwim && \
                     git log -1 --format=%B"
                ),
                "workdir": dir,
            }),
        );
        fs::remove_dir_all(&dir).unwrap();
        assert_eq!(out, "first\n\nGenerated-by: dwim\n[exit code 0]");
    }

    #[test]
    fn runs_in_the_directory_given() {
        let mut bash = Bash::default();
        let dir = env::temp_dir().canonicalize().unwrap();
        assert_eq!(
            run(
                &mut bash,
                &serde_json::json!({ "command": "pwd", "workdir": dir })
            ),
            format!("{}\n[exit code 0]", dir.display())
        );
        assert!(
            run(
                &mut bash,
                &serde_json::json!({ "command": "pwd", "workdir": "/nonexistent/dir" })
            )
            .starts_with("error: workdir /nonexistent/dir is not a directory")
        );
    }

    #[test]
    fn keeps_the_error_and_the_status_after_a_flood() {
        let mut bash = Bash::default();
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
        let mut bash = Bash::default();
        let output = sh("seq 1 50000", &mut bash);
        let path = bash.outputs.join("1.stdout");
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
        let page = Tools::new(Path::new("."))
            .run(&ToolCall {
                name: "read".to_string(),
                arguments: serde_json::json!({ "path": path.to_str().unwrap(), "start": first }),
            })
            .text;
        assert!(page.starts_with(&format!("{first}\t{first}\n")), "{page}");
        assert_eq!(
            sh(&format!("sed -n '500,502p' {}", path.display()), &mut bash),
            "500\n501\n502\n[exit code 0]"
        );

        drop(bash);
        assert!(!path.exists(), "the tool takes its files with it");
    }

    #[test]
    fn drains_both_streams_at_once() {
        let (done, finished) = mpsc::channel();
        thread::spawn(move || {
            let mut bash = Bash::default();
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
        let mut bash = Bash::default();
        let output = sh("yes ä | head -c 50000", &mut bash);
        assert!(!output.contains('\u{FFFD}'), "{output}");
        assert!(output.starts_with("ä\nä\n"));
        assert!(output.contains("\nä\n[… lines "), "{output}");
        assert!(output.contains(" …]\nä\nä\n"), "{output}");
        assert_eq!(
            fs::metadata(bash.outputs.join("1.stdout")).unwrap().len(),
            50000
        );
        let output = sh("printf 'x\\377\\376y'", &mut bash);
        assert_eq!(output, "x\u{FFFD}\u{FFFD}y\n[exit code 0]");
        // A single long line is cut at character boundaries, not line ends.
        let output = sh("yes ä | head -c 50000 | tr -d '\\n'", &mut bash);
        assert!(!output.contains('\u{FFFD}'), "{output}");
        assert!(output.contains("ä\n[… line 1 ("), "{output}");
    }

    #[test]
    fn discards_past_the_cap_and_says_so() {
        let mut bash = Bash::default();
        bash.retain = 3000;
        let output = sh("seq 1 20000", &mut bash);
        let path = bash.outputs.join("1.stdout");
        let note = output
            .lines()
            .find(|line| line.starts_with("[stdout: "))
            .unwrap();
        assert_eq!(
            note,
            format!(
                "[stdout: 20000 lines, 108894 bytes, the first 3000 bytes in {}, the rest discarded]",
                path.display()
            )
        );
        let expected: String = (1..=20000).map(|i| format!("{i}\n")).collect();
        assert_eq!(fs::read_to_string(&path).unwrap(), expected[..3000]);
        // The preview still shows the end, which was read even if not kept.
        assert!(output.contains("\n19999\n20000\n[stdout: "), "{output}");
    }

    #[test]
    fn reports_a_file_it_could_not_keep() {
        let mut bash = Bash::default();
        bash.outputs = PathBuf::from("/dev/null/nowhere");
        let output = sh("seq 1 20000", &mut bash);
        let note = output
            .lines()
            .find(|line| line.starts_with("[stdout: "))
            .unwrap();
        assert!(
            note.starts_with("[stdout: 20000 lines, 108894 bytes, could not keep the rest: "),
            "{note}"
        );
        assert!(output.contains("\n19999\n20000\n[stdout: "), "{output}");
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
        assert!(run(&mut Bash::default(), &serde_json::json!({})).starts_with("error: bash needs"));
    }

    #[test]
    fn knows_the_commands_that_discard_changes() {
        // A checkout of a path that is there discards it, as the model ran
        // it on django-11797; one of a branch, which is not a path, doesn't.
        let dir = env::temp_dir().join(format!("dwim-discards-{}", process::id()));
        fs::create_dir_all(dir.join("django/db/models")).unwrap();
        fs::write(dir.join("django/db/models/lookups.py"), "").unwrap();
        for command in [
            "git stash",
            "cd /x && git stash && python3 tests/runtests.py lookup",
            "git stash push -m wip",
            "git stash -u",
            "git -C /x stash",
            "git checkout -- django/db/models/lookups.py",
            "git checkout .",
            "git checkout -f main",
            "cp lookups.py /tmp/b && git checkout django/db/models/lookups.py && echo reverted",
            "git checkout HEAD django/db/models/lookups.py",
            "git restore lookups.py",
            "git restore --staged --worktree lookups.py",
            "git reset --hard",
            "git reset --hard HEAD~1",
            "git clean -fd",
            "git clean --force",
        ] {
            assert!(discards(command, &dir), "{command}");
        }
        for command in [
            "git status",
            "git diff",
            "git stash list",
            "git stash show -p",
            "git stash pop",
            "git checkout -b fix",
            "git checkout main",
            "git restore --staged lookups.py",
            "git reset HEAD lookups.py",
            "git clean -n",
            "git log --oneline -5 && git show HEAD",
            "echo git stash is not run here > notes.txt",
        ] {
            assert!(!discards(command, &dir), "{command}");
        }
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn refuses_to_discard_changes_to_the_project() {
        // With a tracked file changed, git stash is refused and the change
        // stays; with none changed, it runs.
        let dir = env::temp_dir().join(format!("dwim-refuses-{}", process::id()));
        fs::create_dir_all(&dir).unwrap();
        let git = |args: &[&str]| {
            assert!(
                Command::new("git")
                    .args(["-c", "user.name=t", "-c", "user.email=t@t"])
                    .args(args)
                    .current_dir(&dir)
                    .output()
                    .unwrap()
                    .status
                    .success()
            );
        };
        git(&["init", "-q"]);
        fs::write(dir.join("a.txt"), "old\n").unwrap();
        git(&["add", "a.txt"]);
        git(&["commit", "-qm", "a"]);
        assert!(!changed(&dir));
        fs::write(dir.join("a.txt"), "new\n").unwrap();
        assert!(changed(&dir));
        let mut bash = Bash::default();
        let stash = serde_json::json!({ "command": "git stash" });
        assert_eq!(bash.run(&dir, &stash), Err(DISCARDS.to_string()));
        assert_eq!(fs::read_to_string(dir.join("a.txt")).unwrap(), "new\n");
        fs::write(dir.join("a.txt"), "old\n").unwrap();
        assert_eq!(
            bash.run(&dir, &stash),
            Ok("No local changes to save\n[exit code 0]".to_string())
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn parses_coder_calls() {
        let mut bash = Bash::default();
        let ls = call("<function=bash>\n<parameter=command>\nls -l\n</parameter>\n</function>");
        assert_eq!(ls.name, "bash");
        let echo =
            call("<function=bash>\n<parameter=command>\necho a\necho b\n</parameter>\n</function>");
        assert_eq!(run(&mut bash, &echo.arguments), "a\nb\n[exit code 0]");
        assert!(ToolCall::parse("<function=bash>\n<parameter=command>\nls").is_err());
    }
}
