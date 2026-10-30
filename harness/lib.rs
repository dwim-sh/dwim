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
//! note, the user's earlier messages, the pages it read of the files the
//! note names, read again, and the most recent exchanges, with the thoughts
//! left out.
//! A reply that runs out of room is stopped short and goes on after a
//! compaction.
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

use dwim_models::{Chat, Chunk, LanguageModel, Turn};
use dwim_tools::{ToolCall, Tools};

pub use crate::log::Log;

/// Most of a project's `AGENTS.md` that goes into the system prompt.
const MAX_INSTRUCTIONS: usize = 4000;

/// Most of the files in the working directory the system prompt lists.
const MAX_FILES: usize = 50;

/// What the model gets back when it makes the same call twice in a row,
/// instead of running it again: nothing ran in between to change its output,
/// and small models otherwise tend to repeat a call over and over.
const REPEATED: &str = "error: you just ran this, and its output is above. Don't run it again: use that output, run something else, or reply to the user.";

/// Most tokens the model may think before it is made to act, unless the
/// budget is set otherwise: the thinking budget.
pub const THINKING_BUDGET: usize = 2048;

/// What ends a thought that runs to the thinking budget, in the model's
/// voice, as Qwen3's own thinking budget ends one: the model then goes on
/// to call a tool or reply from what it has thought so far.
const BUDGET_CLOSING: &str = "Considering the limited time by the user, I have to give the solution based on the thinking directly now.";

/// The share of the context past which the conversation is compacted
/// before more goes into it: four fifths.
const HIGH_WATER: (usize, usize) = (4, 5);

/// The share of the context that the most recent exchanges may take when
/// they are read back after a compaction: an eighth.
const TAIL: usize = 8;

/// The share of the context that the user's earlier messages may take in
/// the note turn after a compaction: a sixteenth.
const MESSAGES: usize = 16;

/// The share of the context that the pages the model read of the files the
/// note names may take when they are read back after a compaction: a
/// sixteenth.
const PAGES: usize = 16;

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
- To read a file, use read, not cat: it gives you a page of up to 200 numbered lines and says where the next page starts. Read the next page when you need more, and start from a line to read the middle of a file.
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
    /// Most tokens of a thought, if they are limited.
    thinking_budget: Option<usize>,
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
        chat.limit_thoughts(THINKING_BUDGET, BUDGET_CLOSING);
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
            thinking_budget: Some(THINKING_BUDGET),
            log: Log::none(),
        })
    }

    /// Limits a thought to `tokens`, or lifts the limit: a thought that
    /// runs to the budget is ended with a closing in the model's voice,
    /// and the model goes on to act on what it has thought so far.
    pub fn set_thinking_budget(&mut self, tokens: Option<usize>) {
        self.chat
            .limit_thoughts(tokens.unwrap_or(usize::MAX), BUDGET_CLOSING);
        self.thinking_budget = tokens;
        self.log.thinking_budget(tokens);
    }

    /// Most tokens of a thought, if they are limited.
    pub fn thinking_budget(&self) -> Option<usize> {
        self.thinking_budget
    }

    /// Keeps a transcript of the session in `log`, starting with what the
    /// model was given before the first message: `model`, the `device` it
    /// runs on, where it is, the room it has, how it samples, its thinking
    /// budget, and the `system` prompt.
    pub fn keep_log(&mut self, mut log: Log, model: &str, device: &str, system: &str) {
        log.start(
            model,
            device,
            &self.dir,
            self.chat.capacity(),
            self.chat.sampler(),
            self.thinking_budget,
            system,
        );
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
            self.log.reply(
                &calls,
                self.chat.thought_ended(),
                cut,
                self.chat.tokens(),
                &before,
                &self.chat.stats(),
            );
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
    /// the agent is, the note, the user's earlier messages, the pages the
    /// model read of the files the note names, and the most recent
    /// exchanges without their thoughts. Returns whether it did: not when
    /// there is nothing to compact yet, nor when `on_event` breaks while
    /// the note is written, which leaves the conversation as it was but
    /// for the note.
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
        let environment = environment(&self.dir);
        let exchanges = start < self.transcript.len();
        let head = |pages: bool| note_turn(&environment, &note, &messages, pages, exchanges);

        self.chat.clear();
        (self.restart)(&mut self.chat)?;
        // The newest pages that fit in their budget, read again so that
        // they are what the files say now, each as an exchange of its own.
        // A file that can no longer be read has no page. They go only as
        // far as leaves the conversation under the high-water mark with the
        // system prompt, the head turn, and the tail: the oldest go first.
        let mut kept_pages = Vec::new();
        let mut total = 0;
        for call in pages(&self.transcript, start, &note) {
            let output = self.tools.run(&ToolCall::parse(call)?);
            if output.starts_with("error:") {
                continue;
            }
            let reply = Turn::Reply {
                text: String::new(),
                calls: vec![call.to_string()],
            };
            let results = Turn::Results(vec![output]);
            let tokens = self.chat.measure(&reply)? + self.chat.measure(&results)?;
            if total + tokens > capacity / PAGES {
                continue;
            }
            total += tokens;
            kept_pages.push((reply, results, tokens));
        }
        let room = (capacity * HIGH_WATER.0 / HIGH_WATER.1).saturating_sub(
            self.chat.tokens()
                + self.chat.measure(&Turn::User(head(true)))?
                + self.transcript[start..]
                    .iter()
                    .map(|(_, tokens)| tokens)
                    .sum::<usize>(),
        );
        while total > room {
            let (_, _, tokens) = kept_pages.pop().expect("the pages add up to the total");
            total -= tokens;
        }
        let head = head(!kept_pages.is_empty());
        let mut kept = vec![Turn::User(head.clone())];
        for (reply, results, _) in kept_pages.into_iter().rev() {
            kept.extend([reply, results]);
        }
        kept.extend(
            self.transcript[start..]
                .iter()
                .map(|(turn, _)| turn.clone()),
        );
        self.chat.replay(&kept, |_, _| {})?;
        self.transcript.clear();
        for turn in kept {
            self.record(turn)?;
        }
        let after = self.chat.tokens();
        self.log
            .compacted(before, after, &head, self.transcript.len() - 1);
        let _ = on_event(Event::Compacted { before, after });
        Ok(true)
    }
}

impl<M: LanguageModel> Drop for Harness<M> {
    /// Ends the transcript: the session finished.
    fn drop(&mut self) {
        let stats = self.stats();
        self.log.finish(self.chat.tokens(), &stats);
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

/// The `read` calls among the turns before `start` in `transcript` that
/// read a page of a file `note` names, newest first: the newest of each
/// page, and none of a page the turns from `start` on read too.
fn pages<'a>(transcript: &'a [(Turn, usize)], start: usize, note: &str) -> Vec<&'a str> {
    let mut seen: Vec<(String, u64)> = transcript[start..]
        .iter()
        .filter_map(|(turn, _)| match turn {
            Turn::Reply { calls, .. } => Some(calls),
            _ => None,
        })
        .flatten()
        .filter_map(|call| page(call))
        .collect();
    let mut kept = Vec::new();
    for i in (0..start).rev() {
        let (Turn::Reply { calls, .. }, _) = &transcript[i] else {
            continue;
        };
        for call in calls.iter().rev() {
            let Some(page) = page(call) else {
                continue;
            };
            if names(note, &page.0) && !seen.contains(&page) {
                seen.push(page);
                kept.push(call.as_str());
            }
        }
    }
    kept
}

/// The file and the line a page starts at, if `call`, the body of a tool
/// call, reads one.
fn page(call: &str) -> Option<(String, u64)> {
    let call = ToolCall::parse(call).ok()?;
    if call.name != "read" {
        return None;
    }
    let path = call.arguments.get("path")?.as_str()?;
    let start = dwim_tools::read::start(&call.arguments).unwrap_or(1).max(1);
    Some((path.to_string(), start))
}

/// Whether `note` names the file at `path`: a word of it, of the characters
/// paths are written in, is the path, or a path that ends in it or that it
/// ends in, a whole component at a time. So `harness/lib.rs` names
/// `lib.rs` and `/src/harness/lib.rs`, but not `tools/lib.rs`, and `data.rs`
/// does not name `a.rs`.
fn names(note: &str, path: &str) -> bool {
    let path = path.strip_prefix("./").unwrap_or(path);
    let ends_in = |long: &str, end: &str| {
        long.strip_suffix(end)
            .is_some_and(|rest| rest.is_empty() || rest.ends_with('/'))
    };
    !path.is_empty()
        && note
            .split(|c: char| !c.is_alphanumeric() && !"/._-+".contains(c))
            .map(|word| word.trim_end_matches('.'))
            .map(|word| word.strip_prefix("./").unwrap_or(word))
            .filter(|word| !word.is_empty())
            .any(|word| ends_in(path, word) || ends_in(word, path))
}

/// The turn that opens a compacted conversation: where the agent is, the
/// note, the user's earlier messages, and whether pages the model read
/// earlier, and the most recent exchanges, follow.
fn note_turn(
    environment: &str,
    note: &str,
    messages: &[&str],
    pages: bool,
    exchanges: bool,
) -> String {
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
    text.push_str(match (pages, exchanges) {
        (true, true) => "\n\nPages you read earlier of files the note names follow, read again just now, and then the most recent exchanges as they were.",
        (true, false) => "\n\nPages you read earlier of files the note names follow, read again just now.",
        (false, true) => "\n\nThe most recent exchanges follow as they were.",
        (false, false) => "",
    });
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

    /// A reply after an empty thought that reads `path`.
    fn read(path: &str) -> String {
        format!(
            "\n</think>\n\n<tool_call>\n<function=read>\n<parameter=path>\n{path}\n</parameter>\n</function>\n</tool_call>"
        )
    }

    /// A directory of files of ten short lines each for the model to read,
    /// removed when dropped.
    struct Files(PathBuf);

    impl Files {
        fn new(names: &[&str]) -> Self {
            static DIRS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let n = DIRS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let dir = std::env::temp_dir().join(format!("dwim-harness-{}-{n}", std::process::id()));
            fs::create_dir_all(&dir).unwrap();
            for name in names {
                let text: String = (1..=10).map(|i| format!("{name} line {i}\n")).collect();
                fs::write(dir.join(name), text).unwrap();
            }
            Self(dir)
        }

        /// The path of the file `name`, as the model reads it.
        fn path(&self, name: &str) -> String {
            self.0.join(name).to_str().unwrap().to_string()
        }
    }

    impl Drop for Files {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
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

    /// The records of the log at `path`, which is removed.
    fn records(path: &Path) -> Vec<serde_json::Value> {
        let text = fs::read_to_string(path).unwrap();
        fs::remove_file(path).unwrap();
        text.lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
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
    fn ends_a_thought_at_the_thinking_budget() {
        // A thought of 1000 tokens, under a budget of 100, is ended there
        // with the closing, and the reply goes on to its call and its text.
        let scripts = [reply(1000, true), reply(1000, false)];
        let mut limited = harness(&scripts, 8000, "");
        assert_eq!(limited.thinking_budget(), Some(THINKING_BUDGET));
        let path = std::env::temp_dir().join(format!("dwim-budget-{}.jsonl", std::process::id()));
        limited.keep_log(Log::create(&path).unwrap(), "tiny", "cpu", "Be brief.");
        limited.set_thinking_budget(Some(100));
        let mut thought = String::new();
        let mut events = Vec::new();
        limited
            .send("go", |event| {
                events.push(match event {
                    Event::Thought(text) => {
                        thought.push_str(text);
                        "thought"
                    }
                    Event::Text(_) => "text",
                    Event::Call { .. } => "call",
                    Event::Output(_) => "output",
                    _ => "other",
                });
                ControlFlow::Continue(())
            })
            .unwrap();
        events.dedup();
        assert_eq!(
            events,
            ["thought", "text", "call", "output", "thought", "text"]
        );
        assert!(thought.contains(BUDGET_CLOSING), "{thought}");
        assert!(thought.len() < 1000, "{}", thought.len());
        let turns: Vec<&Turn> = limited.transcript().collect();
        assert!(
            matches!(turns.last(), Some(Turn::Reply { text, .. }) if text.trim().ends_with("Ok.")),
            "{:?}",
            turns.last()
        );
        drop(limited);
        // The log has the budget, its change, and the thought it ended.
        let records = records(&path);
        assert_eq!(records[0]["thinking_budget"], THINKING_BUDGET);
        assert_eq!(records[1]["type"], "thinking_budget");
        assert_eq!(records[1]["tokens"], 100);
        let replies: Vec<&serde_json::Value> = records
            .iter()
            .filter(|record| record["type"] == "reply")
            .collect();
        assert_eq!(replies.len(), 2);
        assert!(replies.iter().all(|reply| reply["thought_ended"] == true));

        // Lifted, the budget lets a thought run.
        let scripts = [reply(1000, false)];
        let mut free = harness(&scripts, 8000, "");
        free.set_thinking_budget(None);
        assert_eq!(free.thinking_budget(), None);
        let mut thought = String::new();
        free.send("go", |event| {
            if let Event::Thought(text) = event {
                thought.push_str(text);
            }
            ControlFlow::Continue(())
        })
        .unwrap();
        assert!(!thought.contains(BUDGET_CLOSING));
        assert!(thought.len() >= 1000);
    }

    #[test]
    fn logs_the_whole_session() {
        // A call and its result, then replies of 1800 tokens until one is
        // stopped short and goes on after a compaction.
        let mut scripts = vec![reply(100, true), reply(100, false)];
        scripts.extend((0..8).map(|_| reply(1800, false)));
        let mut harness = harness(&scripts, 8000, "Task: go on.");
        let path = std::env::temp_dir().join(format!("dwim-log-{}.jsonl", std::process::id()));
        harness.keep_log(Log::create(&path).unwrap(), "tiny", "cpu", "Be brief.");
        for i in 0..5 {
            send(&mut harness, &format!("message {i}"));
        }
        drop(harness);
        let records = records(&path);
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
        assert_eq!(records[0]["thinking_budget"], THINKING_BUDGET);
        assert_eq!(records[0]["device"], "cpu");
        assert_eq!(records[0]["temperature"], 0.0);
        assert_eq!(records[0]["top_k"], 1);
        assert_eq!(records[0]["top_p"], 1.0);
        assert_eq!(records[0]["seed"], 1);
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
        assert_eq!(records[4]["thought_ended"], false);
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
        let head = records[cut + 3]["head"].as_str().unwrap();
        assert!(head.starts_with("# Environment"), "{head}");
        assert!(head.contains("Task: go on."), "{head}");
        assert!(head.contains("message 0"), "{head}");
        assert!(records[cut + 3]["turns"].as_u64().unwrap() > 0);
        assert_eq!(records[cut + 4]["text"], CONTINUE);
        let times: Vec<f64> = records
            .iter()
            .map(|record| record["time"].as_f64().unwrap())
            .collect();
        assert!(times.is_sorted());
        let end = records.last().unwrap();
        assert_eq!(end["type"], "end");
        assert_eq!(end["calls"], 1);
        assert!(end["thought"]["tokens"].as_u64().unwrap() > 1800);
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

    /// A harness with room for `max_len` tokens that has read the files
    /// `read` in turn, one a message, and then run `filler` bash commands,
    /// one a message, and says `note` when asked for one. Returns it with
    /// its transcript.
    fn after_reading(
        files: &Files,
        read: &[&str],
        filler: usize,
        max_len: usize,
        note: &str,
    ) -> (Harness<Scripted>, Vec<Turn>) {
        let mut scripts = Vec::new();
        for name in read {
            scripts.push(self::read(&files.path(name)));
            scripts.push(reply(0, false));
        }
        for _ in 0..filler {
            scripts.push(reply(0, true));
            scripts.push(reply(0, false));
        }
        let mut harness = harness(&scripts, max_len, note);
        for i in 0..read.len() + filler {
            let (events, _) = send(&mut harness, &format!("message {i}"));
            assert_eq!(
                events,
                ["thought", "text", "call", "output", "thought", "text"]
            );
        }
        let transcript: Vec<Turn> = harness.transcript().cloned().collect();
        (harness, transcript)
    }

    /// The turns of `turns` that hold a page of `name`: results with its
    /// first line.
    fn pages_of<'a>(turns: &[&'a Turn], name: &str) -> Vec<&'a Turn> {
        let line = format!("1\t{name} line 1\n");
        turns
            .iter()
            .filter(|turn| matches!(turn, Turn::Results(outputs) if outputs.iter().any(|output| output.contains(&line))))
            .copied()
            .collect()
    }

    #[test]
    fn keeps_the_pages_the_note_names() {
        // Two files read, then enough commands that neither read is in
        // the tail; the note names one of them.
        let files = Files::new(&["alpha.txt", "beta.txt"]);
        let (mut harness, before) = after_reading(
            &files,
            &["alpha.txt", "beta.txt"],
            20,
            16000,
            "Task: fix alpha.txt.\nNext: edit line 3 of alpha.txt.",
        );
        assert!(harness.compact(|_| ControlFlow::Continue(())).unwrap());
        let turns: Vec<&Turn> = harness.transcript().collect();
        let Turn::User(head) = turns[0] else { panic!() };
        assert!(
            head.ends_with("Pages you read earlier of files the note names follow, read again just now, and then the most recent exchanges as they were."),
            "{head}"
        );
        // The page comes as the exchange it came from: the reply that
        // read it, and the results.
        let Turn::Reply { calls, .. } = turns[1] else {
            panic!("{:?}", turns[1])
        };
        assert_eq!(calls.len(), 1);
        assert!(calls[0].contains(&files.path("alpha.txt")), "{}", calls[0]);
        assert_eq!(pages_of(&turns[2..3], "alpha.txt").len(), 1);
        assert_eq!(pages_of(&turns, "alpha.txt").len(), 1);
        assert_eq!(pages_of(&turns, "beta.txt").len(), 0);
        // The tail follows, as it was.
        let tail = &turns[3..];
        assert!(matches!(tail[0], Turn::Reply { .. }), "{:?}", tail[0]);
        assert!(
            before[before.len() - tail.len()..]
                .iter()
                .zip(tail)
                .all(|(a, &b)| a == b)
        );
        assert_eq!(harness.tokens(), harness.chat.model().fed.len());
        assert!(harness.tokens() <= 16000 * HIGH_WATER.0 / HIGH_WATER.1);
    }

    #[test]
    fn reads_the_pages_again() {
        // The file named has changed since it was read, and the other one
        // the note names is gone: the page is what the file says now, and a
        // file that cannot be read has no page.
        let files = Files::new(&["alpha.txt", "beta.txt"]);
        let (mut harness, _) = after_reading(
            &files,
            &["alpha.txt", "beta.txt"],
            20,
            16000,
            "Next: edit alpha.txt and beta.txt.",
        );
        fs::write(files.path("alpha.txt"), "alpha.txt changed\n").unwrap();
        fs::remove_file(files.path("beta.txt")).unwrap();
        assert!(harness.compact(|_| ControlFlow::Continue(())).unwrap());
        let turns: Vec<&Turn> = harness.transcript().collect();
        let Turn::Results(outputs) = turns[2] else {
            panic!("{:?}", turns[2])
        };
        assert!(
            outputs[0].contains("1\talpha.txt changed"),
            "{}",
            outputs[0]
        );
        assert_eq!(pages_of(&turns, "alpha.txt").len(), 0);
        assert!(!turns.iter().any(|turn| matches!(turn, Turn::Reply { calls, .. } if calls.iter().any(|call| call.contains("beta.txt")))));
        assert!(!turns.iter().any(
            |turn| matches!(turn, Turn::Results(outputs) if outputs[0].contains("cannot read"))
        ));
    }

    #[test]
    fn keeps_a_page_in_the_tail_once() {
        // The file read first is read again last, so that the tail holds
        // its page already.
        let files = Files::new(&["alpha.txt"]);
        let mut scripts = vec![read(&files.path("alpha.txt")), reply(0, false)];
        for _ in 0..20 {
            scripts.push(reply(0, true));
            scripts.push(reply(0, false));
        }
        scripts.push(read(&files.path("alpha.txt")));
        scripts.push(reply(0, false));
        let mut harness = harness(&scripts, 16000, "Next: edit alpha.txt.");
        for i in 0..22 {
            send(&mut harness, &format!("message {i}"));
        }
        let before: Vec<&Turn> = harness.transcript().collect();
        assert_eq!(pages_of(&before, "alpha.txt").len(), 2);
        assert!(harness.compact(|_| ControlFlow::Continue(())).unwrap());
        let turns: Vec<&Turn> = harness.transcript().collect();
        let Turn::User(head) = turns[0] else { panic!() };
        assert!(
            head.ends_with("The most recent exchanges follow as they were."),
            "{head}"
        );
        assert_eq!(pages_of(&turns, "alpha.txt").len(), 1);
        assert!(matches!(turns.last(), Some(Turn::Reply { text, .. }) if text.trim() == "Ok."));
    }

    #[test]
    fn keeps_the_newest_pages_that_fit() {
        // Both files are named, but the pages' budget holds one page
        // and not two: the newer is kept.
        let files = Files::new(&["alpha.txt", "beta.txt"]);
        let max_len = 8000;
        let (mut harness, before) = after_reading(
            &files,
            &["alpha.txt", "beta.txt"],
            10,
            max_len,
            "Next: edit alpha.txt and beta.txt.",
        );
        let page = |name: &str| {
            let i = before
                .iter()
                .position(|turn| matches!(turn, Turn::Reply { calls, .. } if calls.iter().any(|call| call.contains(&files.path(name)))))
                .unwrap();
            harness.chat.measure(&before[i]).unwrap()
                + harness.chat.measure(&before[i + 1]).unwrap()
        };
        let (alpha, beta) = (page("alpha.txt"), page("beta.txt"));
        assert!(
            beta <= max_len / PAGES && alpha + beta > max_len / PAGES,
            "{alpha} {beta}"
        );
        assert!(harness.compact(|_| ControlFlow::Continue(())).unwrap());
        let turns: Vec<&Turn> = harness.transcript().collect();
        assert_eq!(pages_of(&turns, "alpha.txt").len(), 0);
        assert_eq!(pages_of(&turns, "beta.txt").len(), 1);
        assert_eq!(pages_of(&turns[2..3], "beta.txt").len(), 1);
    }

    #[test]
    fn keeps_an_older_page_that_fits_past_a_newer_one_that_does_not() {
        let files = Files::new(&["alpha.txt", "beta.txt"]);
        let long: String = (1..=100).map(|i| format!("beta.txt line {i}\n")).collect();
        fs::write(files.path("beta.txt"), long).unwrap();
        let (mut harness, _) = after_reading(
            &files,
            &["alpha.txt", "beta.txt"],
            10,
            8000,
            "Next: edit alpha.txt and beta.txt.",
        );
        assert!(harness.compact(|_| ControlFlow::Continue(())).unwrap());
        let turns: Vec<&Turn> = harness.transcript().collect();
        assert_eq!(pages_of(&turns, "beta.txt").len(), 0);
        assert_eq!(pages_of(&turns[2..3], "alpha.txt").len(), 1);
    }

    #[test]
    fn reads_back_the_head_turn_then_the_pages_then_the_tail() {
        let files = Files::new(&["alpha.txt", "beta.txt"]);
        let (mut harness, before) = after_reading(
            &files,
            &["alpha.txt", "beta.txt"],
            20,
            16000,
            "Next: edit alpha.txt, then beta.txt.",
        );
        assert!(harness.compact(|_| ControlFlow::Continue(())).unwrap());
        let turns: Vec<&Turn> = harness.transcript().collect();
        assert!(matches!(turns[0], Turn::User(head) if head.starts_with("# Environment\n")));
        // The pages, oldest first, each as its exchange.
        for (i, name) in ["alpha.txt", "beta.txt"].iter().enumerate() {
            let reply = turns[1 + 2 * i];
            assert!(
                matches!(reply, Turn::Reply { calls, .. } if calls.len() == 1 && calls[0].contains(&files.path(name))),
                "{reply:?}"
            );
            assert_eq!(pages_of(&turns[2 + 2 * i..3 + 2 * i], name).len(), 1);
        }
        // Then the tail, whole exchanges from a reply, as it was.
        let tail = &turns[5..];
        assert!(matches!(tail[0], Turn::Reply { .. }));
        let start = before.len() - tail.len();
        assert!(before[start..].iter().zip(tail).all(|(a, &b)| a == b));
        assert_eq!(pages_of(&turns, "alpha.txt").len(), 1);
        assert_eq!(pages_of(&turns, "beta.txt").len(), 1);
    }

    #[test]
    fn picks_the_pages_the_note_names() {
        let user = |i: usize| (Turn::User(format!("message {i}")), 10);
        let reply = |call: &str, n: usize| {
            (
                Turn::Reply {
                    text: String::new(),
                    calls: vec![call.to_string()],
                },
                n,
            )
        };
        let text = || {
            (
                Turn::Reply {
                    text: "Ok.".to_string(),
                    calls: Vec::new(),
                },
                5,
            )
        };
        let results = |output: &str, n: usize| (Turn::Results(vec![output.to_string()]), n);
        let read_a = "<function=read>\n<parameter=path>\na.rs\n</parameter>\n</function>";
        let read_b = r#"{"name": "read", "arguments": {"path": "src/b/lib.rs", "start": 201}}"#;
        let grep_a =
            "<function=bash>\n<parameter=command>\ngrep -n fn a.rs\n</parameter>\n</function>";
        let transcript = vec![
            user(1),
            reply(read_a, 20), // 1
            results("1\tfn a()", 100),
            text(), // 3
            user(2),
            reply(read_b, 20), // 5
            results("201\tfn b()", 100),
            reply(grep_a, 20), // 7
            results("3:fn a()", 50),
            reply(read_a, 20), // 9
            results("1\tfn a()", 100),
            text(), // 11
        ];
        let all = transcript.len();
        let calls = |start: usize, note: &str| pages(&transcript, start, note);
        // The newest page of each file the note names, newest first.
        assert_eq!(calls(all, "edit a.rs and lib.rs"), [read_a, read_b]);
        assert_eq!(calls(all, "edit src/b/lib.rs"), [read_b]);
        assert_eq!(calls(all, "edit a.rs"), [read_a]);
        assert_eq!(calls(all, "edit c.rs"), Vec::<&str>::new());
        assert_eq!(calls(all, ""), Vec::<&str>::new());
        // A page the tail holds is not kept again, nor is a bash command's
        // output a page.
        assert_eq!(calls(9, "a.rs lib.rs"), [read_b]);
        assert_eq!(calls(7, "a.rs lib.rs"), [read_b]);
        assert_eq!(
            calls(5, "a.rs lib.rs"),
            Vec::<&str>::new(),
            "the tail holds a.rs's page already"
        );
        assert_eq!(calls(0, "a.rs lib.rs"), Vec::<&str>::new());
        assert_eq!(page(grep_a), None);
        assert_eq!(page("not a call"), None);
        assert_eq!(page(read_b), Some(("src/b/lib.rs".to_string(), 201)));
    }

    #[test]
    fn keeps_a_page_apart_from_the_calls_around_it() {
        let reply = |calls: &[&str]| {
            (
                Turn::Reply {
                    text: String::new(),
                    calls: calls.iter().map(|call| call.to_string()).collect(),
                },
                10,
            )
        };
        let results = |outputs: &[&str]| {
            (
                Turn::Results(outputs.iter().map(|output| output.to_string()).collect()),
                10,
            )
        };
        let read_a = r#"{"name": "read", "arguments": {"path": "a.rs"}}"#;
        let read_a_on = r#"{"name": "read", "arguments": {"path": "a.rs", "start": "201"}}"#;
        let read_b = r#"{"name": "read", "arguments": {"path": "b.rs"}}"#;
        let test = r#"{"name": "bash", "arguments": {"command": "cargo test a.rs"}}"#;
        let transcript = vec![
            reply(&[read_a, test, read_b]),
            results(&["old a", "a long log", "b"]),
            reply(&[read_a_on]),
            results(&["a on"]),
            reply(&[read_a]),
            results(&["new a"]),
        ];
        // Only the pages of the files named come back, not the calls made
        // with them, and a page read again comes back once.
        assert_eq!(
            pages(&transcript, transcript.len(), "edit a.rs"),
            [read_a, read_a_on]
        );
        assert_eq!(
            pages(&transcript, transcript.len(), "edit a.rs and b.rs"),
            [read_a, read_a_on, read_b]
        );
        // A page read in the tail is not kept from before it, whatever it
        // was read along with.
        assert_eq!(pages(&transcript, 4, "edit a.rs"), [read_a_on]);
    }

    #[test]
    fn names_files_by_whole_path_components() {
        // By the path, its end, or a longer path that ends in it, wherever
        // in the text it is written.
        for note in [
            "Edit harness/lib.rs.",
            "lib.rs:12 is wrong",
            "see `lib.rs`, then",
            "(/src/harness/lib.rs)",
            "./harness/lib.rs",
        ] {
            assert!(names(note, "harness/lib.rs"), "{note}");
        }
        assert!(names("Edit harness/lib.rs.", "./harness/lib.rs"));
        // Not the same name in another directory, nor the end of a name.
        assert!(!names("Edit harness/lib.rs.", "tools/lib.rs"));
        assert!(!names("Edit data.rs.", "a.rs"));
        assert!(!names("Edit a.rs.", "data.rs"));
        assert!(!names("a b c", "x"));
        assert!(!names("", "a.rs"));
        assert!(!names("a.rs", ""));
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
