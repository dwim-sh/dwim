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
    env,
    fmt::Display,
    fs::File,
    io::{self, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    process,
    time::{Duration, Instant},
};

use crate::model::{Effort, Sampling};
use serde::Serialize;
use serde_json::{Value, json};

use crate::{
    harness::{Stats, now},
    sys,
};

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
pub struct Transcript {
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

impl Transcript {
    /// The session's transcript, opened before anything else so that a file
    /// it cannot be written to is found out at once: in `path`, if the
    /// options name one, or in a new file in the working directory's folder
    /// of the user's state directory (`~/.local/state`, or on macOS, where
    /// there is none, `~/Library/Application Support`), out of the agent's
    /// sight; and where that cannot be written, in the same folder of the
    /// temporary directory, with a warning. A compaction removes turns on
    /// the grounds that the transcript keeps them, so the session goes on
    /// without one only if neither can be written.
    pub fn open(path: Option<&Path>) -> io::Result<Self> {
        if let Some(path) = path {
            return Self::create(path).map_err(|e| {
                io::Error::other(format!(
                    "cannot write the transcript to {}: {e}",
                    path.display()
                ))
            });
        }
        let file = Path::new("dwim")
            .join("projects")
            .join(project(&env::current_dir()?))
            .join(Self::name());
        let dirs = sys::dirs::state().into_iter().chain([env::temp_dir()]);
        Ok(Self::create_first(dirs.map(|dir| dir.join(&file))))
    }

    /// A transcript written to the first of `paths` that can be, with a
    /// warning for each before it; one that goes nowhere, with a warning,
    /// if none can be.
    fn create_first(paths: impl IntoIterator<Item = PathBuf>) -> Self {
        for path in paths {
            match Self::create(&path) {
                Ok(transcript) => return transcript,
                Err(e) => eprintln!("warning: cannot write {}: {e}", path.display()),
            }
        }
        eprintln!("warning: keeping no transcript");
        Self::none()
    }

    /// A transcript written to `path`, with the directories it goes in made
    /// if they aren't there: the file and its directories are readable by
    /// the user alone.
    pub fn create(path: &Path) -> io::Result<Self> {
        if let Some(dir) = path.parent() {
            sys::fs::create_private_dir(dir)?;
        }
        Ok(Self {
            file: Some(sys::fs::create_private(path)?),
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

    /// A file name for the transcript of a session starting now in this
    /// process: the time in UTC and the process ID, such as
    /// `2026-10-01T093012-4242.jsonl`.
    fn name() -> String {
        let (date, time) = now();
        format!("{date}T{}-{}.jsonl", time.replace(':', ""), process::id())
    }

    /// The file the transcript is written to, if there is one.
    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    /// Records what the model is given before the first message: which
    /// model, on which device, where, how many tokens of context it has,
    /// how its tokens are sampled, the most tokens of a thought, if they
    /// are limited, the effort it is asked to reason at, and the system
    /// prompt.
    #[allow(clippy::too_many_arguments)]
    pub fn start(
        &mut self,
        model: &str,
        device: &str,
        dir: &Path,
        context: usize,
        sampling: Sampling,
        thinking_budget: Option<usize>,
        effort: Effort,
        system: &str,
    ) {
        let (date, time) = now();
        self.write(
            "start",
            json!({
                "date": format!("{date}T{time}Z"),
                "version": env!("CARGO_PKG_VERSION"),
                "model": model,
                "device": device,
                "dir": dir.display().to_string(),
                "context": context,
                "temperature": decimal(sampling.temperature),
                "top_k": sampling.top_k,
                "top_p": decimal(sampling.top_p),
                "min_p": decimal(sampling.min_p),
                "seed": sampling.seed,
                "thinking_budget": thinking_budget,
                "reasoning_effort": effort.name(),
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

    /// Records the end of a reply: the calls it made as written, whether its
    /// thought was ended at the thinking budget, whether it was stopped
    /// short for room, who ran the model and what was asked again, where
    /// there is either to say, the tokens in the context after it, and
    /// what generating it took, from the model's stats before and after.
    pub fn reply(
        &mut self,
        reply: &crate::model::Reply,
        tokens: usize,
        before: &crate::model::Stats,
        after: &crate::model::Stats,
    ) {
        let seconds = (after.prompt.seconds + after.thought.seconds + after.answer.seconds)
            - (before.prompt.seconds + before.thought.seconds + before.answer.seconds);
        let mut record = json!({
            "calls": reply.calls,
            "thought_ended": reply.thought_ended,
            "cut": reply.cut,
            "tokens": tokens,
            "prompt_tokens": after.prompt.tokens - before.prompt.tokens,
            "thought_tokens": after.thought.tokens - before.thought.tokens,
            "answer_tokens": after.answer.tokens - before.answer.tokens,
            "seconds": round(seconds),
        });
        if let Some(provider) = &reply.provider {
            record["provider"] = json!(provider);
        }
        if !reply.retried.is_empty() {
            record["retried"] = json!(reply.retried);
        }
        self.write("reply", record);
    }

    /// Records that the most tokens of a thought are now `tokens`, or that
    /// they are no longer limited.
    pub fn thinking_budget(&mut self, tokens: Option<usize>) {
        self.write("thinking_budget", json!({ "tokens": tokens }));
    }

    /// Records that the conversation goes on with `model`, on `device`,
    /// with `context` tokens of room.
    pub fn model(&mut self, model: &str, device: &str, context: usize) {
        self.write(
            "model",
            json!({ "model": model, "device": device, "context": context }),
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

    /// Records the reminder that follows a call's results once the model
    /// has gone long without changing a file.
    pub fn nudge(&mut self, text: &str) {
        self.write("nudge", json!({ "text": text }));
    }

    /// Records the plan of the work as it now stands, as the model is told
    /// it: empty once it is set aside, or the work is over.
    pub fn plan(&mut self, plan: &str) {
        self.write("plan", json!({ "plan": plan }));
    }

    /// Records that the conversation is being compacted.
    pub fn compacting(&mut self) {
        self.write("compacting", json!({}));
    }

    /// Records that the conversation was compacted: how many tokens it took
    /// before, and how many it takes now; the briefing it starts over
    /// from, with where the agent is, the note, and the user's most recent
    /// messages; and how many of the most recent turns were read back
    /// after it.
    pub fn compacted(&mut self, before: usize, after: usize, briefing: &str, turns: usize) {
        self.write(
            "compacted",
            json!({ "before": before, "after": after, "briefing": briefing, "turns": turns }),
        );
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

    /// Records that the session finished, rather than being killed: the
    /// tokens in the context, and where the session's time went, as the
    /// tokens and seconds of each kind and the tool calls and their
    /// seconds.
    pub fn finish(&mut self, tokens: usize, stats: &Stats) {
        let tally = |tally: &crate::model::Tally| json!({ "tokens": tally.tokens, "seconds": round(tally.seconds) });
        let model = &stats.model;
        self.write(
            "end",
            json!({
                "tokens": tokens,
                "cached": tally(&model.cached),
                "prompt": tally(&model.prompt),
                "thought": tally(&model.thought),
                "answer": tally(&model.answer),
                "calls": stats.calls,
                "tool_seconds": round(stats.tool_seconds),
            }),
        );
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

impl Drop for Transcript {
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

/// `x` as the double with the same shortest decimal, so that it is written
/// as it reads: 0.95 rather than 0.949999988079071.
fn decimal(x: f32) -> f64 {
    x.to_string().parse().unwrap_or(x.into())
}

/// Seconds to the millisecond.
fn round(seconds: f64) -> f64 {
    (seconds * 1000.0).round() / 1000.0
}

/// The name of the folder for the transcripts of sessions in `dir`: its
/// path with everything but letters and digits made a dash, such as
/// `-home-penberg-src-dwim`.
fn project(dir: &Path) -> String {
    dir.to_string_lossy()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::{env, fs, mem};

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
    fn writes_where_it_can() {
        // A folder that cannot be made, under a file, is passed over for
        // the next; with none left, the transcript goes nowhere.
        let dir = env::temp_dir().join(format!("dwim-transcript-test-{}", process::id()));
        fs::create_dir_all(&dir).unwrap();
        let file = dir.join("file");
        fs::write(&file, "").unwrap();
        let blocked = file.join("t.jsonl");
        let open = dir.join("t.jsonl");
        let transcript = Transcript::create_first([blocked.clone(), open.clone()]);
        assert_eq!(transcript.path(), Some(open.as_path()));
        assert!(open.is_file());
        assert_eq!(Transcript::create_first([blocked]).path(), None);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn keeps_streamed_text_of_a_killed_session() {
        let path = env::temp_dir().join(format!("dwim-log-test-{}.jsonl", process::id()));
        let mut log = Transcript::create(&path).unwrap();
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
