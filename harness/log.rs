//! The transcript of a session, as JSON Lines: a record for each thing that
//! happens, from what the model is given before the first message to each
//! message, thought, reply, tool call and result, and compaction, with the
//! seconds since the session started and the tokens they took.
//!
//! Each record is written whole as it happens, so a session that is killed
//! leaves all of it up to then. Streamed text, a thought, a reply's text, or
//! a note, is gathered into one record, written when it starts and written
//! over in place as it grows, at least once a second, so a session that is
//! killed while the model is writing loses at most the last second of it.
//! Writing is best effort: once a write fails, the rest of the session goes
//! unrecorded rather than stopping the agent.

use std::{
    fmt::Display,
    fs::{self, File, OpenOptions},
    io::{self, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    process,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use serde::Serialize;
use serde_json::{Value, json};

use crate::date;

/// Longest the streamed text being gathered goes without being written.
const REWRITE: Duration = Duration::from_secs(1);

/// A record as written: what kind it is and when, then its own fields.
#[derive(Serialize)]
struct Record<'a> {
    #[serde(rename = "type")]
    kind: &'a str,
    time: f64,
    #[serde(flatten)]
    fields: Value,
}

/// The transcript of a session, written to a file as it goes; or nowhere.
pub struct Log {
    file: Option<File>,
    path: Option<PathBuf>,
    start: Instant,
    /// The streamed text being gathered.
    pending: Option<Pending>,
}

/// Streamed text being gathered into a record, which is already in the file
/// as of when it was last written.
struct Pending {
    kind: &'static str,
    text: String,
    /// Where in the file the record starts.
    at: u64,
    /// When the record was last written.
    written: Instant,
}

impl Log {
    /// A transcript written to `path`, readable by the user alone, with the
    /// directories it goes in made if they aren't there.
    pub fn create(path: &Path) -> io::Result<Self> {
        if let Some(dir) = path.parent() {
            fs::create_dir_all(dir)?;
        }
        let mut options = OpenOptions::new();
        options.write(true).create(true).truncate(true);
        #[cfg(unix)]
        std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
        Ok(Self {
            file: Some(options.open(path)?),
            path: Some(path.to_path_buf()),
            start: Instant::now(),
            pending: None,
        })
    }

    /// A transcript that goes nowhere.
    pub fn none() -> Self {
        Self {
            file: None,
            path: None,
            start: Instant::now(),
            pending: None,
        }
    }

    /// The file the transcript is written to, if there is one.
    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    /// A file name for the transcript of a session starting now in this
    /// process: the time in UTC and the process ID, such as
    /// `2026-10-01T093012-4242.jsonl`.
    pub fn name() -> String {
        let (date, time) = now();
        format!("{date}T{}-{}.jsonl", time.replace(':', ""), process::id())
    }

    /// Records what the model is given before the first message: which
    /// model, where, how many tokens of context it has, and the system
    /// prompt.
    pub fn start(&mut self, model: &str, dir: &Path, context: usize, system: &str) {
        let (date, time) = now();
        self.write(
            "start",
            json!({
                "date": format!("{date}T{time}Z"),
                "version": env!("CARGO_PKG_VERSION"),
                "model": model,
                "dir": dir.display().to_string(),
                "context": context,
                "system": system,
            }),
        );
    }

    /// Records a message from the user, with what went ahead of it, if
    /// anything, and the tokens in the context before it.
    pub fn user(&mut self, message: &str, ahead: Option<&str>, tokens: usize) {
        self.write(
            "user",
            json!({ "text": message, "ahead": ahead, "tokens": tokens }),
        );
    }

    /// Gathers a piece of streamed text of `kind`: `thought`, `text`, or
    /// `note`.
    pub fn stream(&mut self, kind: &'static str, piece: &str) {
        match &mut self.pending {
            Some(pending) if pending.kind == kind => {
                pending.text.push_str(piece);
                if pending.written.elapsed() < REWRITE {
                    return;
                }
            }
            _ => {
                self.flush();
                let Some(file) = &mut self.file else {
                    return;
                };
                let Ok(at) = file.stream_position() else {
                    self.file = None;
                    return;
                };
                self.pending = Some(Pending {
                    kind,
                    text: piece.to_string(),
                    at,
                    written: Instant::now(),
                });
            }
        }
        self.rewrite();
    }

    /// Records the end of a reply: the calls it made as written, whether it
    /// was stopped short for room, the tokens in the context after it, and
    /// what generating it took, from the model's stats before and after.
    pub fn reply(
        &mut self,
        calls: &[String],
        cut: bool,
        tokens: usize,
        before: &dwim_models::Stats,
        after: &dwim_models::Stats,
    ) {
        let seconds = (after.prompt.seconds + after.thought.seconds + after.answer.seconds)
            - (before.prompt.seconds + before.thought.seconds + before.answer.seconds);
        self.write(
            "reply",
            json!({
                "calls": calls,
                "cut": cut,
                "tokens": tokens,
                "prompt_tokens": after.prompt.tokens - before.prompt.tokens,
                "thought_tokens": after.thought.tokens - before.thought.tokens,
                "answer_tokens": after.answer.tokens - before.answer.tokens,
                "seconds": round(seconds),
            }),
        );
    }

    /// Records a tool call about to run.
    pub fn call(&mut self, name: &str, detail: &str) {
        self.write("call", json!({ "name": name, "detail": detail }));
    }

    /// Records what a tool call returned, as the model sees it, and how long
    /// it ran: nothing if it did not run, as a repeated or malformed call.
    pub fn output(&mut self, output: &str, seconds: Option<f64>) {
        self.write(
            "output",
            json!({ "text": output, "seconds": seconds.map(round) }),
        );
    }

    /// Records that the conversation is being compacted.
    pub fn compacting(&mut self) {
        self.write("compacting", json!({}));
    }

    /// Records that the conversation was compacted: how many tokens it took
    /// before, and how many it takes now.
    pub fn compacted(&mut self, before: usize, after: usize) {
        self.write("compacted", json!({ "before": before, "after": after }));
    }

    /// Records how a turn ended, if not on its own: with an error, or
    /// interrupted.
    pub fn end(&mut self, error: Option<&dyn Display>, interrupted: bool) {
        if let Some(error) = error {
            self.write("error", json!({ "message": error.to_string() }));
        } else if interrupted {
            self.write("interrupted", json!({}));
        } else {
            self.flush();
        }
    }

    /// Writes a record, after the streamed text gathered before it.
    fn write(&mut self, kind: &str, fields: Value) {
        self.flush();
        self.put(kind, fields);
    }

    /// Writes the streamed text gathered so far, if there is any, and
    /// starts the next record after it.
    fn flush(&mut self) {
        self.rewrite();
        self.pending = None;
    }

    /// Writes the record of the streamed text being gathered over the one
    /// written before, which is the last line in the file.
    fn rewrite(&mut self) {
        let Some(pending) = &mut self.pending else {
            return;
        };
        let Some(file) = &mut self.file else {
            return;
        };
        pending.written = Instant::now();
        let line = line(self.start, pending.kind, json!({ "text": pending.text }));
        let at = pending.at;
        let written = file
            .seek(SeekFrom::Start(at))
            .and_then(|_| file.write_all(line.as_bytes()))
            .and_then(|()| file.set_len(at + line.len() as u64));
        if written.is_err() {
            self.file = None;
        }
    }

    /// Writes a record as a line of its own.
    fn put(&mut self, kind: &str, fields: Value) {
        let Some(file) = &mut self.file else {
            return;
        };
        if file
            .write_all(line(self.start, kind, fields).as_bytes())
            .is_err()
        {
            self.file = None;
        }
    }
}

impl Drop for Log {
    /// Writes the streamed text gathered last.
    fn drop(&mut self) {
        self.flush();
    }
}

/// A record as a line of JSON, timed from `start`.
fn line(start: Instant, kind: &str, fields: Value) -> String {
    let record = Record {
        kind,
        time: round(start.elapsed().as_secs_f64()),
        fields,
    };
    let mut line = serde_json::to_string(&record).unwrap_or_default();
    line.push('\n');
    line
}

/// The date and the time of day now, in UTC.
fn now() -> (String, String) {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.as_secs());
    let day = secs % 86400;
    (
        date((secs / 86400) as i64),
        format!("{:02}:{:02}:{:02}", day / 3600, day / 60 % 60, day % 60),
    )
}

/// Seconds to the millisecond.
fn round(seconds: f64) -> f64 {
    (seconds * 1000.0).round() / 1000.0
}

#[cfg(test)]
mod tests {
    use std::{env, mem};

    use super::*;

    fn records(path: &Path) -> Vec<(String, String)> {
        fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|line| {
                let record: Value = serde_json::from_str(line).unwrap();
                let field = |name: &str| record[name].as_str().unwrap_or_default().to_string();
                (field("type"), field("text") + &field("detail"))
            })
            .collect()
    }

    #[test]
    fn keeps_streamed_text_of_a_killed_session() {
        let path = env::temp_dir().join(format!("dwim-log-test-{}.jsonl", process::id()));
        let mut log = Log::create(&path).unwrap();
        log.stream("thought", "Let me ");
        assert_eq!(records(&path), [("thought".into(), "Let me ".into())]);
        log.stream("thought", "look.");
        log.call("bash", "ls");
        log.stream("text", "Here");
        log.pending.as_mut().unwrap().written -= REWRITE;
        log.stream("text", " it is.");
        // Killed: nothing more is written.
        mem::forget(log);
        assert_eq!(
            records(&path),
            [
                ("thought".into(), "Let me look.".into()),
                ("call".into(), "ls".into()),
                ("text".into(), "Here it is.".into()),
            ]
        );
        fs::remove_file(&path).unwrap();
    }
}
