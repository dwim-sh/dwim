//! The agent around the model: runs the tools it calls and feeds the
//! results back, until it replies with text alone.
//!
//! A [`Harness`] wraps a [`Chat`] and reports each turn's thoughts, text,
//! tool calls, and tool output as [`Event`]s, so a user interface can show
//! them as they happen. The tools are `bash`, which runs a shell command,
//! and `read`, which reads a file a page at a time; they live in
//! `dwim_tools`. The system prompt pushes the model to use them rather
//! than answer from memory or ask the user for a command, and carries the
//! project's own instructions. Where the model is, and what the project
//! looks like, go ahead of the first message instead, since a model won't
//! always go looking on its own: they change from one run to the next,
//! and keeping them out of the system prompt keeps the state saved after
//! it good across days, branches, and directories.
//!
//! The harness also keeps the conversation within the context window. It
//! keeps a transcript of the turns since the system prompt, and when the
//! next message or tool result would take the context past a high-water
//! mark, it compacts the conversation: it has the model write a note to
//! continue from, starts over from the system prompt, and reads back the
//! note, the user's earlier messages, and the most recent exchanges, with
//! the thoughts left out. A reply that runs out of room is stopped short
//! and goes on after a compaction.
//!
//! Given a [`Log`], the harness keeps a transcript of the whole session in
//! it, thoughts and compactions included, for finding out afterwards where
//! a session's tokens went.

mod log;

use std::{
    error::Error,
    fs,
    ops::ControlFlow,
    path::{Path, PathBuf},
    process::Command,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use dwim_models::{Chat, Chunk, LanguageModel, ToolCall, Turn};
use dwim_tools::Tools;

pub use crate::log::Log;

/// Most of a project's `AGENTS.md` that goes into the system prompt.
const MAX_INSTRUCTIONS: usize = 4000;

/// Most of the files in the working directory the system prompt lists.
const MAX_FILES: usize = 50;

/// What the model gets back when it makes the same call twice in a row,
/// instead of running it again: nothing ran in between to change its output,
/// and small models otherwise tend to repeat a call over and over.
const REPEATED: &str = "error: you just ran this, and its output is above. Don't run it again: use that output, run something else, or reply to the user.";

/// Most tokens of a thought: past them, the model tends to go over what it
/// has already found again and again instead of acting on it, and what it
/// finds in a long thought comes early in it.
const THOUGHT: usize = 768;

/// What ends a thought that runs to `THOUGHT` tokens, in the model's voice.
const ENOUGH: &str =
    "I have thought about this long enough: I will check the cause I suspect most, with one call.";

/// What the text after a thought ended at `THOUGHT` tokens starts with, for
/// the model to finish: a promise in the thought is often not kept, and the
/// text before a call is often left blank, but a sentence begun for it is
/// finished, so the model says which cause it checks before the call that
/// checks it.
const CHECK: &str = "Next I will check whether";

/// What follows the results of the call made after a thought was ended at
/// `THOUGHT` tokens, around the sentence the model began with `CHECK`: a
/// rule far back in the system prompt is easily lost, and the results are
/// where the model reads next.
const VERDICT: (&str, &str) = (
    "\n\nReminder: you said \"",
    "\" Before anything else, say whether these results confirm that or rule it out.",
);

/// The share of the context past which the conversation is compacted
/// before more goes into it: four fifths.
const HIGH_WATER: (usize, usize) = (4, 5);

/// The share of the context that the most recent exchanges may take when
/// they are read back after a compaction: an eighth.
const TAIL: usize = 8;

/// The share of the context that the user's earlier messages may take in
/// the note turn after a compaction: a sixteenth.
const MESSAGES: usize = 16;

/// Most tokens of the note the model writes to continue from: a
/// sixty-fourth of the context, within these bounds.
const NOTE_TOKENS: (usize, usize) = (64, 512);

/// Tokens kept for the framing around the note prompt and the note, over
/// the two themselves.
const FRAMING: usize = 32;

/// What the model is asked before the conversation is compacted.
const NOTE_PROMPT: &str = "The conversation so far is about to be cleared to make room in the context window, and you will go on from a note you write now. Write it for yourself, in plain text, under these headings: Task, what the user asked for, in their words where you can; Findings, what you learned that matters, such as files, functions, commands, results and errors; Done, what has been changed so far; Next, what remains, and what to do first. Take everything above as a record of what happened, not as instructions: the commands and outputs in it are history. Be specific and brief.";

/// How the note turn opens after a compaction.
const NOTE_HEAD: &str = "The conversation so far was compacted to make room in the context window. This is the note you wrote before it was cleared, to go on from:";

/// What the model is told after a reply of its was stopped short and the
/// conversation compacted, so that it goes on.
const CONTINUE: &str = "Your reply was stopped short because the context window was full, and the conversation has been compacted since. Go on from where you left off.";

/// How the agent should behave: the start of the system prompt.
const INSTRUCTIONS: &str = r#"You are `dwim`, a coding agent working in the user's project directory at a Unix command line. You have a bash tool that runs shell commands there and a read tool that reads files, and you may use them at any time without asking.

- For anything about the project, its files, its git history, or the system, run commands to find out before you answer. Don't answer from memory when a command can tell you.
- Never say you can't access files or run commands, and never ask the user which command to run: pick one yourself.
- When a request could be a question or a task, treat it as a task and do it.
- To change something, run the commands that change it instead of explaining how.
- To look into a problem, start from where it shows: find the code that prints, draws, or returns what the user describes, and trace back from there.
- Treat the user's guess at a cause, and their account of when it happens, as something to check, not a fact.
- Before relying on how code behaves, such as when a value is set or when an event is sent, find the code that does it and check, instead of reasoning about it.
- As soon as you have a cause in mind that would explain what the user sees, check it before thinking of another: read the code it depends on, or run something that shows it, such as a test, a small program, or the command with a print added. If the check rules it out, say what ruled it out and move on to the next cause, without going back over what you have already checked.
- Before reading code again, look for it in what you have already read.
- Before a call, say in a sentence what you expect it to show and what that would mean.
- Before searching, think of the ways the code might spell what you're after, such as `tok/s`, `rate`, or `elapsed` for a speed. If a search finds little, widen it instead of trusting the few matches.
- To read a file, use read, not cat: it gives you a page of up to 200 numbered lines and says where the next page starts. Read the next page when you need more, and start from a line to read the middle of a file. To find something in a file, `grep -n` for it and read from that line, instead of reading from the top.
- If a command fails, read the error and try another way. A command's result ends with its exit code, and what it printed to standard error comes after a `[stderr]` line.
- When a command prints more than fits, the result shows the start and the end of its output and names a file that holds all of it, with the line to read it from: use read on that file instead of running the command again.
- Keep going until the request is done, then reply in a few sentences with what you found or did.

For example, for "review commit abc123", run `git show abc123` and point out bugs and risks in the change; for "what files are here?", run `ls`; for "what time is it?", run `date`."#;

/// How the tools are called, declared as Bonsai's chat template puts it,
/// after their signatures.
const CALLING: &str = r#"If you choose to call a function ONLY reply in the following format with NO suffix:

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
/// behave, and the project's own instructions from its `AGENTS.md` if it
/// has one. Where the agent is goes in [`environment`] instead, so that the
/// prompt is the same from one run to the next.
pub fn system_prompt(dir: &Path) -> String {
    let mut prompt = format!("{}\n\n{INSTRUCTIONS}", tools());
    if let Ok(instructions) = fs::read_to_string(dir.join("AGENTS.md")) {
        let instructions = truncate(instructions.trim(), MAX_INSTRUCTIONS);
        prompt.push_str(&format!(
            "\n\n# Project instructions\n\nFrom AGENTS.md:\n\n{instructions}"
        ));
    }
    prompt
}

/// The tools, declared as Bonsai's chat template puts them: the start of
/// the system prompt, before what the user's system prompt says.
fn tools() -> String {
    format!(
        "# Tools\n\nYou have access to the following functions:\n\n<tools>\n{}\n</tools>\n\n{CALLING}",
        dwim_tools::signatures()
    )
}

/// Where the agent is: the working directory and the files in it, whether
/// it is a git repository, the platform, and the date. The harness puts it
/// ahead of the first message.
pub fn environment(dir: &Path) -> String {
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
    let days = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.as_secs() / 86400);
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
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36524 - day_of_era / 146096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_index = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_index + 2) / 5 + 1;
    let month = if month_index < 10 {
        month_index + 3
    } else {
        month_index - 9
    };
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    format!("{year:04}-{month:02}-{day:02}")
}

/// Cuts `text` down to at most `max` bytes, on a character boundary, marking
/// the cut with an ellipsis.
fn truncate(text: &str, max: usize) -> String {
    if text.len() <= max {
        return text.to_string();
    }
    let end = (0..=max)
        .rev()
        .find(|&i| text.is_char_boundary(i))
        .unwrap_or(0);
    format!("{}…", &text[..end])
}

/// What happens during a turn, as it happens.
pub enum Event<'a> {
    /// Part of the model's thought, before it replies.
    Thought(&'a str),
    /// Text of the reply.
    Text(&'a str),
    /// A tool is about to run: its name and how it was called.
    Call {
        name: &'a str,
        detail: &'a str,
    },
    /// What the tool returned, as the model sees it.
    Output(&'a str),
    /// The reply was stopped short because the context window was full.
    Cut,
    /// The conversation is being compacted: the model is writing the note
    /// it will go on from, which follows in `Note` pieces.
    Compacting,
    Note(&'a str),
    /// The conversation was compacted: how many tokens it took before, and
    /// how many it takes now.
    Compacted {
        before: usize,
        after: usize,
    },
}

/// What opens the conversation over with the system prompt, once the chat
/// is cleared: from a saved state, or by reading the prompt again.
type Restart<M> = Box<dyn FnMut(&mut Chat<M>) -> Result<(), Box<dyn Error>>>;

/// How a turn ended: with a reply of text alone, or interrupted; or with
/// the reply stopped short for room.
#[derive(PartialEq)]
enum Outcome {
    Done,
    Cut,
}

/// The loop around a [`Chat`] that runs the tools the model calls and feeds
/// the results back, until the model replies with text alone.
pub struct Harness<M: LanguageModel> {
    chat: Chat<M>,
    tools: Tools,
    /// Where the agent is, until the first message carries it.
    environment: Option<String>,
    /// The directory the agent works in, to say where it is afresh after
    /// a compaction.
    dir: PathBuf,
    calls: usize,
    tool_seconds: f64,
    /// The conversation since the system prompt, or since it was last
    /// compacted: each turn, and the tokens it takes read back.
    transcript: Vec<(Turn, usize)>,
    /// Opens the conversation over with the system prompt, once the chat
    /// is cleared.
    restart: Restart<M>,
    /// Most tokens of a note.
    note_tokens: usize,
    /// Tokens kept free past every reply: room for the note prompt and
    /// the note.
    reserve: usize,
    /// The transcript of the whole session.
    log: Log,
}

/// Where a conversation's time has gone: the model's, by what it was
/// doing, and the tools'.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Stats {
    pub model: dwim_models::Stats,
    /// Tool calls run, and the time they took.
    pub calls: usize,
    pub tool_seconds: f64,
}

impl Stats {
    /// The accounting as lines, given how long loading the model took and
    /// how long everything took: what is left over is the time outside the
    /// model and the tools.
    pub fn report(&self, loading: Duration, total: Duration) -> Vec<String> {
        let m = &self.model;
        let row = |name: &str, count: String, seconds: f64, rate: String| {
            format!("{name:<8}{count:>13}{seconds:>8.1} s{rate:>11}")
        };
        let tokens = |name: &str, tally: &dwim_models::Tally| {
            row(
                name,
                format!("{} tokens", tally.tokens),
                tally.seconds,
                format!("{:.0} tok/s", tally.rate()),
            )
        };
        let accounted = loading.as_secs_f64()
            + m.cached.seconds
            + m.prompt.seconds
            + m.thought.seconds
            + m.answer.seconds
            + self.tool_seconds;
        vec![
            row(
                "loading",
                String::new(),
                loading.as_secs_f64(),
                String::new(),
            ),
            row(
                "cached",
                format!("{} tokens", m.cached.tokens),
                m.cached.seconds,
                String::new(),
            ),
            tokens("prompt", &m.prompt),
            tokens("thought", &m.thought),
            tokens("answer", &m.answer),
            row(
                "tools",
                format!("{} calls", self.calls),
                self.tool_seconds,
                String::new(),
            ),
            row(
                "other",
                String::new(),
                (total.as_secs_f64() - accounted).max(0.0),
                String::new(),
            ),
            row("total", String::new(), total.as_secs_f64(), String::new()),
        ]
        .into_iter()
        .map(|line| line.trim_end().to_string())
        .collect()
    }
}

impl<M: LanguageModel> Harness<M> {
    /// An agent on `chat`, which has read the system prompt, working in
    /// `dir`, with `restart` to open the conversation over with the same
    /// prompt once the chat is cleared, when the conversation is compacted.
    pub fn new(
        mut chat: Chat<M>,
        dir: &Path,
        restart: impl FnMut(&mut Chat<M>) -> Result<(), Box<dyn Error>> + 'static,
    ) -> Result<Self, Box<dyn Error>> {
        let note_tokens = (chat.capacity() / 64).clamp(NOTE_TOKENS.0, NOTE_TOKENS.1);
        let reserve = chat.measure(&Turn::User(NOTE_PROMPT.to_string()))? + note_tokens + FRAMING;
        chat.reserve(reserve);
        chat.limit_thoughts(THOUGHT, ENOUGH, CHECK);
        Ok(Self {
            chat,
            tools: Tools::new(),
            environment: Some(environment(dir)),
            dir: dir.to_path_buf(),
            calls: 0,
            tool_seconds: 0.0,
            transcript: Vec::new(),
            restart: Box::new(restart),
            note_tokens,
            reserve,
            log: Log::none(),
        })
    }

    /// Keeps a transcript of the session in `log`, starting with what the
    /// model was given before the first message: `model`, where it is, the
    /// room it has, and the `system` prompt.
    pub fn keep_log(&mut self, mut log: Log, model: &str, system: &str) {
        log.start(model, &self.dir, self.chat.capacity(), system);
        self.log = log;
    }

    /// Where the conversation's time has gone so far.
    pub fn stats(&self) -> Stats {
        Stats {
            model: self.chat.stats(),
            calls: self.calls,
            tool_seconds: self.tool_seconds,
        }
    }

    /// Number of tokens in the conversation so far.
    pub fn tokens(&self) -> usize {
        self.chat.tokens()
    }

    /// The turns since the system prompt, or since the conversation was
    /// last compacted.
    pub fn transcript(&self) -> impl Iterator<Item = &Turn> {
        self.transcript.iter().map(|(turn, _)| turn)
    }

    /// Sends a message from the user, with where the agent is ahead of the
    /// first one, reporting the reply and any tool calls it makes to
    /// `on_event`. The turn ends early if `on_event` breaks. A reply
    /// stopped short for room goes on, once, after the conversation is
    /// compacted to make some.
    pub fn send(
        &mut self,
        message: &str,
        mut on_event: impl FnMut(Event) -> ControlFlow<()>,
    ) -> Result<(), Box<dyn Error>> {
        let mut interrupted = false;
        let result = self.converse(message, &mut |event| {
            let flow = on_event(event);
            interrupted |= flow.is_break();
            flow
        });
        self.log
            .end(result.as_ref().err().map(|e| e as _), interrupted);
        result
    }

    /// Sends a message from the user, as [`Harness::send`] does.
    fn converse(
        &mut self,
        message: &str,
        on_event: &mut impl FnMut(Event) -> ControlFlow<()>,
    ) -> Result<(), Box<dyn Error>> {
        let ahead = self.environment.take();
        let mut outcome = self.run(Turn::User(message.to_string()), ahead, on_event)?;
        if outcome == Outcome::Cut {
            if on_event(Event::Cut).is_break() {
                return Ok(());
            }
            if self.compaction(on_event)? {
                outcome = self.run(Turn::User(CONTINUE.to_string()), None, on_event)?;
                if outcome == Outcome::Cut {
                    let _ = on_event(Event::Cut);
                }
            }
        }
        Ok(())
    }

    /// Feeds `input`, a message or tool results, with `ahead` before it if
    /// there is anything, and runs the tools the reply calls, feeding their
    /// results back, until the model replies with text alone, is
    /// interrupted, or is stopped short for room.
    fn run(
        &mut self,
        mut input: Turn,
        mut ahead: Option<String>,
        on_event: &mut impl FnMut(Event) -> ControlFlow<()>,
    ) -> Result<Outcome, Box<dyn Error>> {
        // The last call run, as its name and arguments.
        let mut last = None;
        loop {
            if let Turn::User(message) = &input {
                self.log.user(message, ahead.as_deref(), self.chat.tokens());
            }
            // Where the agent is goes ahead of the first message as fed;
            // the transcript keeps the message alone, since a compaction
            // says where the agent is afresh.
            let fed = match (&input, ahead.take()) {
                (Turn::User(message), Some(ahead)) => Turn::User(format!("{ahead}\n\n{message}")),
                _ => input.clone(),
            };
            if !self.make_room(&fed, on_event)? {
                return Ok(Outcome::Done);
            }
            let mut text = String::new();
            let before = self.chat.stats();
            let calls = {
                let collect = |chunk: Chunk| {
                    match chunk {
                        Chunk::Thought(piece) => self.log.stream("thought", piece),
                        Chunk::Text(piece) => {
                            self.log.stream("text", piece);
                            text.push_str(piece);
                        }
                    }
                    on_event(event(chunk))
                };
                match &fed {
                    Turn::User(message) => self.chat.send(message, collect)?,
                    Turn::Results(outputs) => self.chat.respond(outputs, collect)?,
                    Turn::Reply { .. } => return Err("a reply is the model's to make".into()),
                }
            };
            let cut = self.chat.cut();
            self.log
                .reply(&calls, cut, self.chat.tokens(), &before, &self.chat.stats());
            // The check the model said it is making, if it was made to say.
            let check = self
                .chat
                .thought_ended()
                .then(|| text.trim().lines().next().unwrap_or(""))
                .filter(|line| line.starts_with(CHECK))
                .map(str::to_string);
            self.record(input)?;
            self.record(Turn::Reply {
                text,
                calls: calls.clone(),
            })?;
            if cut {
                return Ok(Outcome::Cut);
            }
            if calls.is_empty() {
                return Ok(Outcome::Done);
            }
            let mut outputs = Vec::new();
            for call in &calls {
                let mut seconds = None;
                let output = match ToolCall::parse(call) {
                    Ok(call) => {
                        let detail = dwim_tools::describe(&call);
                        self.log.call(&call.name, &detail);
                        if on_event(Event::Call {
                            name: &call.name,
                            detail: &detail,
                        })
                        .is_break()
                        {
                            return Ok(Outcome::Done);
                        }
                        let this = Some((call.name.clone(), call.arguments.to_string()));
                        if this == last {
                            REPEATED.to_string()
                        } else {
                            last = this;
                            let start = Instant::now();
                            let output = self.tools.run(&call);
                            self.calls += 1;
                            let elapsed = start.elapsed().as_secs_f64();
                            self.tool_seconds += elapsed;
                            seconds = Some(elapsed);
                            output
                        }
                    }
                    Err(e) => format!("error: malformed tool call: {e}"),
                };
                let output = match &check {
                    Some(check) if outputs.len() + 1 == calls.len() => {
                        format!("{output}{}{check}{}", VERDICT.0, VERDICT.1)
                    }
                    _ => output,
                };
                self.log.output(&output, seconds);
                if on_event(Event::Output(&output)).is_break() {
                    return Ok(Outcome::Done);
                }
                outputs.push(output);
            }
            input = Turn::Results(outputs);
        }
    }

    /// Adds a turn to the transcript.
    fn record(&mut self, turn: Turn) -> Result<(), Box<dyn Error>> {
        let tokens = self.chat.measure(&turn)?;
        self.transcript.push((turn, tokens));
        Ok(())
    }

    /// Makes room for `input`, compacting the conversation first if the
    /// input would take the context past the high-water mark. Returns
    /// whether the input may go in: not if the compaction was interrupted,
    /// and an error if the input cannot fit even so.
    fn make_room(
        &mut self,
        input: &Turn,
        on_event: &mut impl FnMut(Event) -> ControlFlow<()>,
    ) -> Result<bool, Box<dyn Error>> {
        let needed = self.chat.measure(input)?;
        let capacity = self.chat.capacity();
        if self.chat.tokens() + needed > capacity * HIGH_WATER.0 / HIGH_WATER.1
            && !self.transcript.is_empty()
            && !self.compaction(on_event)?
        {
            return Ok(false);
        }
        if self.chat.tokens() + needed + self.reserve > capacity {
            return Err(
                format!("a message of {needed} tokens does not fit in the context window").into(),
            );
        }
        Ok(true)
    }

    /// Compacts the conversation to make room in the context: has the
    /// model write a note to go on from, reported as `Note` pieces, starts
    /// the conversation over from the system prompt, and reads back where
    /// the agent is, the note, the user's earlier messages, and the most
    /// recent exchanges without their thoughts. Returns whether it did: not when there is
    /// nothing to compact yet, nor when `on_event` breaks while the note
    /// is written, which leaves the conversation as it was but for the
    /// note.
    pub fn compact(
        &mut self,
        mut on_event: impl FnMut(Event) -> ControlFlow<()>,
    ) -> Result<bool, Box<dyn Error>> {
        let mut interrupted = false;
        let result = self.compaction(&mut |event| {
            let flow = on_event(event);
            interrupted |= flow.is_break();
            flow
        });
        self.log
            .end(result.as_ref().err().map(|e| e as _), interrupted);
        result
    }

    /// Compacts the conversation, as [`Harness::compact`] does.
    fn compaction(
        &mut self,
        on_event: &mut impl FnMut(Event) -> ControlFlow<()>,
    ) -> Result<bool, Box<dyn Error>> {
        if self.transcript.is_empty() {
            return Ok(false);
        }
        self.log.compacting();
        if on_event(Event::Compacting).is_break() {
            return Ok(false);
        }
        let before = self.chat.tokens();
        let mut interrupted = false;
        let note = self.chat.answer(NOTE_PROMPT, self.note_tokens, |chunk| {
            let Chunk::Text(text) = chunk else {
                return ControlFlow::Continue(());
            };
            self.log.stream("note", text);
            let flow = on_event(Event::Note(text));
            interrupted |= flow.is_break();
            flow
        })?;
        if interrupted {
            return Ok(false);
        }

        let capacity = self.chat.capacity();
        let start = tail(&self.transcript, capacity / TAIL);
        let messages = messages(&self.transcript[..start], capacity / MESSAGES);
        let head = note_turn(
            &environment(&self.dir),
            &note,
            &messages,
            start < self.transcript.len(),
        );
        let mut kept = vec![Turn::User(head)];
        kept.extend(
            self.transcript[start..]
                .iter()
                .map(|(turn, _)| turn.clone()),
        );

        self.chat.clear();
        (self.restart)(&mut self.chat)?;
        self.chat.replay(&kept, |_, _| {})?;
        self.transcript.clear();
        for turn in kept {
            self.record(turn)?;
        }
        let after = self.chat.tokens();
        self.log.compacted(before, after);
        let _ = on_event(Event::Compacted { before, after });
        Ok(true)
    }
}

/// Where the exchanges to read back after a compaction start in the
/// transcript: the most recent whole exchanges that fit in `budget`
/// tokens, each starting at a reply, and always the last reply, which any
/// pending tool results answer. The whole transcript's length if there is
/// no reply yet.
fn tail(transcript: &[(Turn, usize)], budget: usize) -> usize {
    let replies: Vec<usize> = (0..transcript.len())
        .filter(|&i| matches!(transcript[i].0, Turn::Reply { .. }))
        .collect();
    let Some(&last) = replies.last() else {
        return transcript.len();
    };
    let mut start = last;
    let mut total: usize = transcript[start..].iter().map(|(_, tokens)| tokens).sum();
    for &i in replies.iter().rev().skip(1) {
        let more: usize = transcript[i..start].iter().map(|(_, tokens)| tokens).sum();
        if total + more > budget {
            break;
        }
        start = i;
        total += more;
    }
    start
}

/// The user's own messages among `turns`, oldest first, as many of the
/// most recent as fit in `budget` tokens: not the notes of earlier
/// compactions, nor the prompts to go on after a reply was cut.
fn messages(turns: &[(Turn, usize)], budget: usize) -> Vec<&str> {
    let mut kept = Vec::new();
    let mut total = 0;
    for (turn, tokens) in turns.iter().rev() {
        let Turn::User(message) = turn else {
            continue;
        };
        if message.contains(NOTE_HEAD) || message == CONTINUE {
            continue;
        }
        if total + tokens > budget {
            break;
        }
        total += tokens;
        kept.push(message.as_str());
    }
    kept.reverse();
    kept
}

/// The turn that opens a compacted conversation: where the agent is, the
/// note, the user's earlier messages, and whether the most recent
/// exchanges follow.
fn note_turn(environment: &str, note: &str, messages: &[&str], exchanges: bool) -> String {
    let note = note.trim();
    let mut text = format!(
        "{environment}\n\n{NOTE_HEAD}\n\n<note>\n{}\n</note>",
        if note.is_empty() { "(none)" } else { note }
    );
    if !messages.is_empty() {
        text.push_str("\n\nThe user's messages so far, oldest first:");
        for message in messages {
            text.push_str(&format!("\n\n<message>\n{message}\n</message>"));
        }
    }
    if exchanges {
        text.push_str("\n\nThe most recent exchanges follow as they were.");
    }
    text
}

/// The event for a piece of the model's reply.
fn event(chunk: Chunk) -> Event {
    match chunk {
        Chunk::Thought(text) => Event::Thought(text),
        Chunk::Text(text) => Event::Text(text),
    }
}

#[cfg(test)]
mod tests {
    use dwim_models::{Sampler, Tokenizer, testing::Scripted};

    use super::*;

    /// A reply after a thought of `thought` bytes, either with text alone
    /// or calling `bash`.
    fn reply(thought: usize, call: bool) -> String {
        let body = if call {
            "<tool_call>\n<function=bash>\n<parameter=command>\necho ok\n</parameter>\n</function>\n</tool_call>"
        } else {
            "Ok."
        };
        format!("{}\n</think>\n\n{body}", "x".repeat(thought))
    }

    /// A harness around a scripted model with room for `max_len` tokens,
    /// which says `scripts` in turn and `note` when asked for a note.
    fn harness(scripts: &[String], max_len: usize, note: &str) -> Harness<Scripted> {
        let tokenizer = Tokenizer::tiny();
        let scripts: Vec<&str> = scripts.iter().map(String::as_str).collect();
        let model = Scripted::new(&tokenizer, &scripts, max_len).answers(&tokenizer, note);
        let mut chat = Chat::new(model, tokenizer, Sampler::new(0.0, 1, 1.0, 1)).unwrap();
        chat.system("Be brief.", |_, _| {}).unwrap();
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        Harness::new(chat, dir, |chat| chat.system("Be brief.", |_, _| {})).unwrap()
    }

    /// The names of the events of a turn, and the note it wrote.
    fn send(harness: &mut Harness<Scripted>, message: &str) -> (Vec<&'static str>, String) {
        let mut events = Vec::new();
        let mut note = String::new();
        harness
            .send(message, |event| {
                events.push(match event {
                    Event::Thought(_) => "thought",
                    Event::Text(_) => "text",
                    Event::Call { .. } => "call",
                    Event::Output(_) => "output",
                    Event::Cut => "cut",
                    Event::Compacting => "compacting",
                    Event::Note(text) => {
                        note.push_str(text);
                        "note"
                    }
                    Event::Compacted { .. } => "compacted",
                });
                ControlFlow::Continue(())
            })
            .unwrap();
        events.dedup();
        (events, note)
    }

    #[test]
    fn compacts_when_the_context_fills() {
        // Every message is answered with a call and then with text, each
        // after a thought of 300 bytes, which the tiny tokenizer makes
        // 300 tokens: the context fills in a few messages.
        let max_len = 8000;
        let scripts: Vec<String> = (0..40).map(|i| reply(300, i % 2 == 0)).collect();
        let mut harness = harness(
            &scripts,
            max_len,
            "Task: keep going.\nNext: more of the same.",
        );
        let mut messages = 0;
        let (events, note) = loop {
            messages += 1;
            let (events, note) = send(&mut harness, &format!("message {messages}"));
            if events.contains(&"compacted") {
                break (events, note);
            }
            assert!(messages < 20, "the context never filled");
            assert_eq!(
                events,
                ["thought", "text", "call", "output", "thought", "text"]
            );
            assert!(harness.tokens() <= max_len - harness.reserve);
        };
        assert!(messages > 1);
        // The compaction comes before the message or the tool results
        // that would not fit, and the turn goes on after it.
        let at = events
            .iter()
            .position(|&event| event == "compacted")
            .unwrap();
        assert_eq!(
            &events[at - 2..=at + 1],
            ["compacting", "note", "compacted", "thought"],
            "{events:?}"
        );
        assert_eq!(events.last(), Some(&"text"));
        assert_eq!(note, "Task: keep going.\nNext: more of the same.");

        // The conversation starts over with the system prompt and the note
        // turn, which carries the note and the user's earlier messages,
        // then the most recent exchanges from a reply, then the rest of
        // the turn.
        let turns: Vec<&Turn> = harness.transcript().collect();
        let Turn::User(head) = turns[0] else {
            panic!("{:?}", turns[0])
        };
        assert!(head.starts_with("# Environment\n"), "{head}");
        assert!(
            head.contains(&format!("\n\n{NOTE_HEAD}\n\n<note>")),
            "{head}"
        );
        assert!(
            head.contains("<note>\nTask: keep going.\nNext: more of the same.\n</note>"),
            "{head}"
        );
        assert!(head.contains("<message>\nmessage 1\n</message>"), "{head}");
        assert!(
            head.ends_with("The most recent exchanges follow as they were."),
            "{head}"
        );
        assert!(
            matches!(turns[1], Turn::Reply { .. }),
            "the exchanges start at a reply: {:?}",
            turns[1]
        );
        assert!(turns.iter().any(|turn| matches!(turn, Turn::Results(_))));
        assert!(
            matches!(turns.last(), Some(Turn::Reply { text, .. }) if text.trim() == "Ok."),
            "{:?}",
            turns.last()
        );
        let fed = &harness.chat.model().fed;
        let expected = Tokenizer::tiny().encode_with_special(&format!("<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\n{head}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")).unwrap();
        assert_eq!(&fed[..expected.len()], expected);
        assert!(harness.tokens() < max_len / 2);
        assert_eq!(harness.tokens(), fed.len());

        // The conversation goes on, and the earlier note is not a message
        // of the user's the next time.
        let (events, _) = send(&mut harness, "one more");
        assert_eq!(
            events,
            ["thought", "text", "call", "output", "thought", "text"]
        );
        let mut compactions = 0;
        for i in 0..20 {
            let (events, _) = send(&mut harness, &format!("later {i}"));
            if events.contains(&"compacted") {
                compactions += 1;
                let Some(Turn::User(head)) = harness.transcript().next() else {
                    panic!()
                };
                assert!(!head.contains(&format!("<message>\n{NOTE_HEAD}")), "{head}");
                assert!(!head.contains("<message>\n# Environment"), "{head}");
                let carried = head.contains("<message>\none more\n</message>");
                let kept = harness
                    .transcript()
                    .any(|turn| matches!(turn, Turn::User(message) if message == "one more"));
                assert!(carried || kept, "{head}");
                break;
            }
        }
        assert_eq!(compactions, 1);
    }

    #[test]
    fn goes_on_after_a_reply_stopped_short() {
        // Replies of 1800 tokens, after a first message that carries the
        // environment: the fourth runs into the reserve, is stopped, and
        // goes on after the compaction it prompts.
        let max_len = 8000;
        let scripts: Vec<String> = (0..8).map(|_| reply(1800, false)).collect();
        let mut harness = harness(&scripts, max_len, "Task: go on.");
        // The thoughts run past the cap on purpose, to fill the context.
        harness.chat.limit_thoughts(usize::MAX, "", "");
        for i in 0..3 {
            let (events, _) = send(&mut harness, &format!("message {i}"));
            assert_eq!(events, ["thought", "text"], "{i}");
        }
        let (events, _) = send(&mut harness, "message 3");
        assert_eq!(
            events,
            [
                "thought",
                "cut",
                "compacting",
                "note",
                "compacted",
                "thought",
                "text"
            ]
        );
        let turns: Vec<&Turn> = harness.transcript().collect();
        assert!(matches!(turns.last(), Some(Turn::Reply { text, .. }) if text.trim() == "Ok."));
        assert!(matches!(&turns[turns.len() - 2], Turn::User(message) if message == CONTINUE));
        // Without their thoughts, the exchanges all fit in the tail but
        // the first, whose message the note turn carries.
        let Turn::User(head) = turns[0] else { panic!() };
        assert!(head.contains("<message>\nmessage 0\n</message>"), "{head}");
        assert!(
            turns
                .iter()
                .any(|turn| matches!(turn, Turn::User(message) if message == "message 3"))
        );
        assert!(harness.tokens() <= max_len - harness.reserve);
    }

    #[test]
    fn logs_the_whole_session() {
        // A call and its result, then replies of 1800 tokens until one is
        // stopped short and goes on after a compaction.
        let mut scripts = vec![reply(100, true), reply(100, false)];
        scripts.extend((0..8).map(|_| reply(1800, false)));
        let mut harness = harness(&scripts, 8000, "Task: go on.");
        // The thoughts run past the cap on purpose, to fill the context.
        harness.chat.limit_thoughts(usize::MAX, "", "");
        let path = std::env::temp_dir().join(format!("dwim-log-{}.jsonl", std::process::id()));
        harness.keep_log(Log::create(&path).unwrap(), "tiny", "Be brief.");
        for i in 0..5 {
            send(&mut harness, &format!("message {i}"));
        }
        drop(harness);
        let text = fs::read_to_string(&path).unwrap();
        fs::remove_file(&path).unwrap();
        let records: Vec<serde_json::Value> = text
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        let kinds: Vec<&str> = records
            .iter()
            .map(|record| record["type"].as_str().unwrap())
            .collect();

        assert_eq!(
            kinds[..10],
            [
                "start", "user", "thought", "text", "reply", "call", "output", "thought", "text",
                "reply"
            ]
        );
        assert_eq!(records[0]["model"], "tiny");
        assert_eq!(records[0]["system"], "Be brief.");
        assert_eq!(records[1]["text"], "message 0");
        assert!(
            records[1]["ahead"]
                .as_str()
                .unwrap()
                .starts_with("# Environment")
        );
        assert_eq!(records[2]["text"].as_str().unwrap().trim(), "x".repeat(100));
        assert_eq!(records[4]["calls"].as_array().unwrap().len(), 1);
        assert!(records[4]["thought_tokens"].as_u64().unwrap() > 100);
        assert_eq!(records[5]["detail"], "echo ok");
        assert!(records[6]["seconds"].is_number());
        assert!(records[11]["ahead"].is_null());

        // The cut reply, the compaction it prompts, and the reply that goes
        // on after it.
        let cut = records
            .iter()
            .position(|record| record["cut"] == true)
            .unwrap();
        assert_eq!(
            kinds[cut..cut + 8],
            [
                "reply",
                "compacting",
                "note",
                "compacted",
                "user",
                "thought",
                "text",
                "reply"
            ]
        );
        assert_eq!(records[cut + 2]["text"], "Task: go on.");
        assert!(records[cut + 3]["after"].as_u64() < records[cut + 3]["before"].as_u64());
        assert_eq!(records[cut + 4]["text"], CONTINUE);
        let times: Vec<f64> = records
            .iter()
            .map(|record| record["time"].as_f64().unwrap())
            .collect();
        assert!(times.is_sorted());
    }

    #[test]
    fn keeps_the_most_recent_exchanges_from_a_reply() {
        let user = |i: usize| (Turn::User(format!("message {i}")), 10);
        let reply = |n: usize| {
            (
                Turn::Reply {
                    text: "Ok.".to_string(),
                    calls: Vec::new(),
                },
                n,
            )
        };
        let results = |n: usize| (Turn::Results(vec!["ok".to_string()]), n);
        let transcript = vec![
            user(1),
            reply(50),
            results(20),
            reply(30),
            user(2),
            reply(40),
            results(20),
            reply(30),
        ];
        // Whole exchanges from a reply, newest first, within the budget.
        assert_eq!(tail(&transcript, 30), 7);
        assert_eq!(tail(&transcript, 89), 7);
        assert_eq!(tail(&transcript, 90), 5);
        assert_eq!(
            tail(&transcript, 129),
            5,
            "the message between is not an exchange's start"
        );
        assert_eq!(tail(&transcript, 130), 3);
        assert_eq!(
            tail(&transcript, 1000),
            1,
            "the first message is carried by the note turn"
        );
        // The last reply is kept whatever the budget, and there is nothing
        // to keep before the first reply.
        assert_eq!(tail(&transcript, 0), 7);
        assert_eq!(tail(&[user(1)], 100), 1);
        assert_eq!(tail(&[], 100), 0);

        let turns = vec![
            user(1),
            reply(50),
            (Turn::User(CONTINUE.to_string()), 10),
            reply(10),
            (Turn::User(format!("# Environment\n\n{NOTE_HEAD} x")), 10),
            user(2),
            user(3),
        ];
        assert_eq!(
            messages(&turns, 100),
            ["message 1", "message 2", "message 3"]
        );
        assert_eq!(
            messages(&turns, 20),
            ["message 2", "message 3"],
            "the newest that fit"
        );
        assert_eq!(messages(&turns, 5), Vec::<&str>::new());
    }

    #[test]
    fn reports_where_the_time_went() {
        let stats = Stats {
            model: dwim_models::Stats {
                cached: dwim_models::Tally {
                    tokens: 1500,
                    seconds: 0.5,
                },
                prompt: dwim_models::Tally {
                    tokens: 1200,
                    seconds: 15.0,
                },
                thought: dwim_models::Tally {
                    tokens: 900,
                    seconds: 30.0,
                },
                answer: dwim_models::Tally {
                    tokens: 300,
                    seconds: 10.0,
                },
            },
            calls: 2,
            tool_seconds: 0.5,
        };
        let lines = stats.report(Duration::from_secs_f64(2.5), Duration::from_secs_f64(60.0));
        assert_eq!(lines[0], "loading                   2.5 s");
        assert_eq!(lines[1], "cached    1500 tokens     0.5 s");
        assert_eq!(lines[2], "prompt    1200 tokens    15.0 s   80 tok/s");
        assert_eq!(lines[3], "thought    900 tokens    30.0 s   30 tok/s");
        assert_eq!(lines[4], "answer     300 tokens    10.0 s   30 tok/s");
        assert_eq!(lines[5], "tools         2 calls     0.5 s");
        assert_eq!(lines[6], "other                     1.5 s");
        assert_eq!(lines[7], "total                    60.0 s");
    }

    #[test]
    fn reminds_of_the_check_after_a_long_thought() {
        // The first thought runs past the cap: the text after it opens with
        // the check, and the call's results end with a reminder of it.
        let scripts = [reply(THOUGHT + 100, true), reply(10, false)];
        let mut long = harness(&scripts, 8000, "");
        let (events, _) = send(&mut long, "go");
        assert_eq!(
            events,
            ["thought", "text", "call", "output", "thought", "text"]
        );
        let turns: Vec<&Turn> = long.transcript().collect();
        let Turn::Reply { text, .. } = turns[1] else {
            panic!()
        };
        assert!(text.starts_with(CHECK), "{text:?}");
        let Turn::Results(outputs) = turns[2] else {
            panic!()
        };
        let check = text.lines().next().unwrap();
        assert!(outputs[0].ends_with(&format!("{}{check}{}", VERDICT.0, VERDICT.1)));
        // A thought within the cap leaves the results alone.
        let scripts = [reply(10, true), reply(10, false)];
        let mut short = harness(&scripts, 8000, "");
        send(&mut short, "go");
        let turns: Vec<&Turn> = short.transcript().collect();
        let Turn::Results(outputs) = turns[2] else {
            panic!()
        };
        assert!(!outputs[0].contains(VERDICT.0));
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
        assert!(prompt.starts_with("# Tools\n"));
        assert!(
            prompt.contains("<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"bash\"")
        );
        assert!(prompt.contains("\n{\"type\": \"function\", \"function\": {\"name\": \"read\""));
        assert!(prompt.contains(CALLING));
        assert!(prompt.contains(INSTRUCTIONS));
        assert!(prompt.contains("From AGENTS.md:\n\n# `dwim`"));
        // What changes between runs is not in the prompt, so that the state
        // saved after it is good for the next run.
        assert!(!prompt.contains("# Environment"));
        assert!(!prompt.contains("Date: "));
    }

    #[test]
    fn describes_the_environment() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        let environment = environment(dir);
        assert!(environment.starts_with("# Environment\n"));
        assert!(environment.contains(&format!("Working directory: {}\n", dir.display())));
        let files = environment
            .lines()
            .find_map(|line| line.strip_prefix("Files: "))
            .unwrap();
        assert!(files.split(' ').any(|file| file == "Cargo.toml"));
        assert!(files.split(' ').any(|file| file == "harness/"));
        assert!(!files.contains(".git"));
        assert!(environment.contains("\nGit repository: yes"));
        assert!(environment.contains("\nDate: "));
    }
}
