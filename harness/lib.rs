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
    hash::{DefaultHasher, Hash, Hasher},
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
pub const THINKING_BUDGET: usize = 512;

/// Thoughts in a row ended at the budget, while investigating, after
/// which the model is asked for its plan and set to acting: a thought
/// that runs to the budget twice over the same ground is the model going
/// round in circles, and asking for the plan ends them.
pub const CUTS_BEFORE_PLAN: u32 = 2;

/// Tokens the conversation may grow by, read and generated, without a
/// change made to a file, before the model is asked for its plan, or to
/// bring it up to date, and set to acting: the decision budget. Thoughts
/// cut at their budget are one way to spend without deciding; reading
/// page after page, each after a short thought, is the other, and this
/// bounds both. A change starts the count over, as does a message from
/// the user. A run that read three files whole before its first long
/// thought had grown by about 25k tokens by then.
pub const DECISION_BUDGET: usize = 32 << 10;

/// Most tokens of a plan.
const PLAN_TOKENS: usize = 768;

/// What the model is asked for its plan with, once it has spent the
/// decision budget, or had its thoughts cut [`CUTS_BEFORE_PLAN`] times in
/// a row, without a change made.
const PLAN_PROMPT: &str = "You have looked at enough to decide, so plan now, and then act on the plan rather than look further or think it over again. Write the plan in plain text, without tools: the smallest change that does the job, as a numbered list of the file, the lines, and what goes there, in the order to make them; then what to run to check them. If something must still be found out first, say what, in one line at the end.";

/// What the model is asked with when, acting, it has spent the decision
/// budget again without a change.
const REPLAN_PROMPT: &str = "You have gone on for a while without changing a file. Bring the plan up to date, in plain text, without tools: what is done, and what remains as a numbered list of the file, the lines, and what goes there, in the order to make them; then what to run to check them.";

/// What the model is told once it has written its plan.
const ACT_PROMPT: &str =
    "Carry out the plan, one change at a time, and check the result after each.";

/// How the model samples while acting: Qwen3.8's settings for replies
/// made without thinking, where the file's are for those made with it.
const ACT_SAMPLING: (f32, f32) = (0.7, 0.8);

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
- To read a file, use read, not cat. A long file read without a start line comes as an outline, its definitions with their line numbers: pick the line you need and read from it, a page of up to 200 numbered lines that says where the next page starts. Read a file whole only when you need all of it.
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
    /// The model is writing its plan, having been asked for it, which
    /// follows in `Plan` pieces; it acts on it after.
    Planning,
    Plan(&'a str),
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

/// What the agent is doing: investigating, with thinking, until it makes
/// a change, or its thoughts run to the budget twice, or it has spent the
/// decision budget; or acting, without thinking, on what it found or
/// planned, thinking once more only after a call fails.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Investigate,
    Act,
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
    /// Whether the agent is investigating or acting.
    mode: Mode,
    /// Thoughts in a row ended at the budget while investigating, with no
    /// change made since.
    cuts: u32,
    /// Tokens the conversation has grown by since the last change to a
    /// file, the last plan, or the user's message.
    spent: usize,
    /// Tokens of growth without a change before the plan is asked for.
    decision_budget: usize,
    /// Whether the model has written a plan this turn, so that the next
    /// request is to bring it up to date.
    planned: bool,
    /// Whether the last call failed while acting, so that the next reply
    /// is made with a thought, to find out why.
    recover: bool,
    /// How the model samples while investigating: the file's settings.
    sampling: (f32, f32),
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
            mode: Mode::Investigate,
            cuts: 0,
            spent: 0,
            decision_budget: DECISION_BUDGET,
            planned: false,
            recover: false,
            sampling: (0.0, 0.0),
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

    /// Sets how many tokens the conversation may grow by without a change
    /// made before the model is asked for its plan, [`DECISION_BUDGET`]
    /// unless set.
    pub fn set_decision_budget(&mut self, tokens: usize) {
        self.decision_budget = tokens;
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
        self.set_mode(Mode::Investigate);
        self.spent = 0;
        self.planned = false;
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
            // Acting, the model thinks only to find out why a call failed.
            self.chat
                .set_thinking(self.mode == Mode::Investigate || std::mem::take(&mut self.recover));
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
            if self.mode == Mode::Investigate {
                self.cuts = if self.chat.thought_ended() {
                    self.cuts + 1
                } else {
                    0
                };
            }
            let before = fingerprint(&self.dir);
            let mut changed = false;
            let mut failed = false;
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
                            failed |= call_failed(&call.name, &output);
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
            changed |= before.is_some() && fingerprint(&self.dir) != before;
            if changed {
                // A change is a decision made: the count starts over, and
                // the model acts from here.
                self.spent = 0;
                self.set_mode(Mode::Act);
            }
            // The results count as spent, though they are recorded with the
            // reply to them.
            let pending = self.chat.measure(&Turn::Results(outputs.clone()))?;
            let undecided = self.spent + pending >= self.decision_budget
                || (self.mode == Mode::Investigate && self.cuts >= CUTS_BEFORE_PLAN);
            if !changed && undecided {
                // Spent without deciding: the plan, or the plan brought up
                // to date, and then acting on it. The results go in first,
                // with room made for them as for any input.
                if !self.make_room(&Turn::Results(outputs.clone()), on_event)? {
                    return Ok(Outcome::Done);
                }
                self.chat.feed_results(&outputs)?;
                self.record(Turn::Results(outputs))?;
                self.plan(on_event)?;
                self.set_mode(Mode::Act);
                self.spent = 0;
                input = Turn::User(ACT_PROMPT.to_string());
                continue;
            }
            if self.mode == Mode::Act {
                self.recover = failed;
            }
            input = Turn::Results(outputs);
        }
    }

    /// Sets the agent investigating or acting: with thinking and the
    /// file's sampling, or without thinking and with the sampling for it.
    fn set_mode(&mut self, mode: Mode) {
        if self.sampling == (0.0, 0.0) {
            let sampler = self.chat.sampler();
            self.sampling = (sampler.temperature, sampler.top_p);
        }
        let (temperature, top_p) = match mode {
            Mode::Investigate => self.sampling,
            Mode::Act => ACT_SAMPLING,
        };
        let sampler = self.chat.sampler_mut();
        sampler.temperature = temperature;
        sampler.top_p = top_p;
        self.chat.set_thinking(mode == Mode::Investigate);
        if mode != self.mode {
            self.log.mode(match mode {
                Mode::Investigate => "investigate",
                Mode::Act => "act",
            });
        }
        self.mode = mode;
        self.cuts = 0;
        self.recover = false;
    }

    /// Asks the model for its plan, reported as `Plan` pieces, and records
    /// the exchange, so that the plan stays in the conversation to act on.
    fn plan(
        &mut self,
        on_event: &mut impl FnMut(Event) -> ControlFlow<()>,
    ) -> Result<(), Box<dyn Error>> {
        let prompt = if self.planned {
            REPLAN_PROMPT
        } else {
            PLAN_PROMPT
        };
        self.planned = true;
        self.log.user(prompt, None, self.chat.tokens());
        let _ = on_event(Event::Planning);
        // The plan takes what is free above the reserve, which is the
        // note's should the conversation have to be compacted next.
        let room = self.chat.capacity().saturating_sub(
            self.chat.tokens() + self.chat.measure(&Turn::User(prompt.to_string()))? + self.reserve,
        );
        let before = self.chat.stats();
        let plan = self.chat.answer(prompt, PLAN_TOKENS.min(room), |chunk| {
            let Chunk::Text(text) = chunk else {
                return ControlFlow::Continue(());
            };
            self.log.stream("plan", text);
            on_event(Event::Plan(text))
        })?;
        self.log.reply(
            &[],
            false,
            self.chat.cut(),
            self.chat.tokens(),
            &before,
            &self.chat.stats(),
        );
        self.record(Turn::User(prompt.to_string()))?;
        self.record(Turn::Reply {
            text: plan,
            calls: Vec::new(),
        })
    }

    /// Adds a turn to the transcript.
    fn record(&mut self, turn: Turn) -> Result<(), Box<dyn Error>> {
        let tokens = self.chat.measure(&turn)?;
        self.spent += tokens;
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
        let mut kept = vec![Turn::User(head.clone())];
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
        if message.contains(NOTE_HEAD)
            || [CONTINUE, PLAN_PROMPT, REPLAN_PROMPT, ACT_PROMPT].contains(&message.as_str())
        {
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

/// Whether a call to the tool `name` failed, by its `output`: an error
/// from the tool, or a command that exited with a status other than
/// zero.
fn call_failed(name: &str, output: &str) -> bool {
    output.starts_with("error:")
        || (name == "bash"
            && output
                .rsplit_once("[exit code ")
                .is_some_and(|(_, code)| !code.starts_with("0]")))
}

/// What the files of `dir` are, as a number that changes when any of
/// them does: the changes git sees, against the index and in untracked
/// files, with the sizes and times of the untracked ones. `None` where
/// git cannot say.
fn fingerprint(dir: &Path) -> Option<u64> {
    let git = |args: &[&str]| {
        Command::new("git")
            .args(args)
            .current_dir(dir)
            .output()
            .ok()
            .filter(|output| output.status.success())
            .map(|output| output.stdout)
    };
    let status = git(&["status", "--porcelain=v1", "--untracked-files=all"])?;
    let diff = git(&["diff", "--no-color"])?;
    let mut hasher = DefaultHasher::new();
    status.hash(&mut hasher);
    diff.hash(&mut hasher);
    for line in String::from_utf8_lossy(&status).lines() {
        if let Some(path) = line.strip_prefix("?? ")
            && let Ok(metadata) = fs::metadata(dir.join(path))
        {
            metadata.len().hash(&mut hasher);
            metadata
                .modified()
                .ok()
                .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
                .map(|since| since.as_nanos())
                .hash(&mut hasher);
        }
    }
    Some(hasher.finish())
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
        let mut chat = Chat::new(model, tokenizer, Sampler::new(0.0, 1, 1.0, 0.0, 1)).unwrap();
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
                    Event::Planning => "planning",
                    Event::Plan(_) => "plan",
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
        // The replies are long thoughts: not the budget's business here.
        harness.set_thinking_budget(None);
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

    /// A reply after a thought of `thought` bytes, calling `tool` with
    /// `argument` as its one parameter, `parameter`; with no thought at all
    /// if `thought` is zero, as a reply made without thinking is scripted.
    fn reply_calling(thought: usize, tool: &str, parameter: &str, argument: &str) -> String {
        let call = format!(
            "<tool_call>\n<function={tool}>\n<parameter={parameter}>\n{argument}\n</parameter>\n</function>\n</tool_call>"
        );
        if thought == 0 {
            call
        } else {
            format!("{}\n</think>\n\n{call}", "x".repeat(thought))
        }
    }

    /// A git repository of its own, for a harness to work in. The scripts
    /// that work in it name it, as the model does: bash runs where the
    /// process does.
    fn repo() -> PathBuf {
        static REPOS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = REPOS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("dwim-modes-{}-{n}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        assert!(
            Command::new("git")
                .args(["init", "-q"])
                .current_dir(&dir)
                .status()
                .unwrap()
                .success()
        );
        dir
    }

    /// A harness around a scripted model working in `dir`, under a budget
    /// of 100, keeping a log at `path`. `DIR` in a script stands for the
    /// directory.
    fn harness_in(dir: &Path, scripts: &[String], path: &Path) -> Harness<Scripted> {
        let at = dir.display().to_string();
        let scripts: Vec<String> = scripts.iter().map(|s| s.replace("DIR", &at)).collect();
        let tokenizer = Tokenizer::tiny();
        let scripts: Vec<&str> = scripts.iter().map(String::as_str).collect();
        let model = Scripted::new(&tokenizer, &scripts, 100_000);
        let mut chat = Chat::new(model, tokenizer, Sampler::new(1.0, 1, 0.95, 0.0, 1)).unwrap();
        chat.system("Be brief.", |_, _| {}).unwrap();
        let mut harness =
            Harness::new(chat, dir, |chat| chat.system("Be brief.", |_, _| {})).unwrap();
        harness.set_thinking_budget(Some(100));
        harness.keep_log(Log::create(path).unwrap(), "tiny", "cpu", "Be brief.");
        harness
    }

    /// The replies of the log at `path`, as thought tokens and whether the
    /// thought was ended at the budget, and the modes it records.
    fn replies_and_modes(path: &Path) -> (Vec<(u64, bool)>, Vec<String>) {
        let records = records(path);
        let replies = records
            .iter()
            .filter(|record| record["type"] == "reply")
            .map(|reply| {
                (
                    reply["thought_tokens"].as_u64().unwrap(),
                    reply["thought_ended"].as_bool().unwrap(),
                )
            })
            .collect();
        let modes = records
            .iter()
            .filter(|record| record["type"] == "mode")
            .map(|record| record["mode"].as_str().unwrap().to_string())
            .collect();
        (replies, modes)
    }

    #[test]
    fn plans_after_two_cut_thoughts_and_then_acts() {
        // Two thoughts in a row run over the budget with nothing changed:
        // the model is asked for its plan, which it writes without
        // thinking, and acts from then on without thinking, with the
        // sampling for it; a thought that ran over once, then ended by
        // itself, would not have done it.
        let path = std::env::temp_dir().join(format!("dwim-plans-{}.jsonl", std::process::id()));
        let scripts = [
            reply_calling(1000, "bash", "command", "echo looked"),
            reply_calling(1000, "bash", "command", "echo looked again"),
            "1. a.txt, line 1: write made.".to_string(),
            reply_calling(0, "bash", "command", "echo made > DIR/a.txt"),
            "Done.".to_string(),
        ];
        let dir = repo();
        let mut harness = harness_in(&dir, &scripts, &path);
        let (events, _) = send(&mut harness, "go");
        assert_eq!(
            events,
            [
                "thought", "text", "call", "output", "thought", "text", "call", "output",
                "planning", "plan", "call", "output", "text"
            ],
            "{events:?}"
        );
        let turns: Vec<&Turn> = harness.transcript().collect();
        assert!(
            turns
                .iter()
                .any(|turn| matches!(turn, Turn::User(m) if m == PLAN_PROMPT)),
            "the plan prompt is in the transcript"
        );
        assert!(
            turns.iter().any(|turn| matches!(turn, Turn::Reply { text, calls } if text.contains("1. a.txt") && calls.is_empty())),
            "and so is the plan"
        );
        assert!(
            turns
                .iter()
                .any(|turn| matches!(turn, Turn::User(m) if m == ACT_PROMPT)),
            "and the word to act"
        );
        assert_eq!(fs::read_to_string(dir.join("a.txt")).unwrap(), "made\n");
        assert_eq!(harness.chat.sampler().temperature, ACT_SAMPLING.0);
        drop(harness);
        let (replies, modes) = replies_and_modes(&path);
        assert_eq!(modes, ["act"], "{modes:?}");
        assert_eq!(replies.len(), 5, "{replies:?}");
        assert!(replies[0].1 && replies[1].1, "two cuts: {replies:?}");
        assert_eq!(
            replies[2].0, 0,
            "the plan is written without thinking: {replies:?}"
        );
        assert_eq!(replies[3].0, 0, "and the model acts without: {replies:?}");
        assert_eq!(replies[4].0, 0, "{replies:?}");
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn plans_once_the_decision_budget_is_spent_and_again_while_acting() {
        // Short thoughts, each followed by a read of a long file, spend
        // the decision budget without a change: the plan is asked for and
        // the model acts. Acting, reads that spend it again get the plan
        // brought up to date; a change starts the count over.
        let path = std::env::temp_dir().join(format!("dwim-spent-{}.jsonl", std::process::id()));
        let dir = repo();
        fs::write(dir.join("big.txt"), "y".repeat(1500)).unwrap();
        // Each read differs, or the second would be refused as repeated.
        let scripts = [
            reply_calling(20, "bash", "command", "cat DIR/big.txt; echo 1"),
            reply_calling(20, "bash", "command", "cat DIR/big.txt; echo 2"),
            "1. a.txt, line 1: write made.".to_string(),
            reply_calling(0, "bash", "command", "cat DIR/big.txt; echo 3"),
            reply_calling(0, "bash", "command", "cat DIR/big.txt; echo 4"),
            "Done, what remains: 1. a.txt, line 1: write made.".to_string(),
            reply_calling(0, "bash", "command", "echo made > DIR/a.txt"),
            "Done.".to_string(),
        ];
        let mut harness = harness_in(&dir, &scripts, &path);
        // Two reads of 1500 bytes, under a budget of 2000 tokens of growth.
        harness.set_decision_budget(2000);
        let (events, _) = send(&mut harness, "go");
        assert_eq!(
            events,
            [
                "thought", "text", "call", "output", "thought", "text", "call", "output",
                "planning", "plan", "call", "output", "call", "output", "planning", "plan", "call",
                "output", "text"
            ],
            "{events:?}"
        );
        let turns: Vec<&Turn> = harness.transcript().collect();
        assert_eq!(
            turns
                .iter()
                .filter(|turn| matches!(turn, Turn::User(m) if m == PLAN_PROMPT))
                .count(),
            1
        );
        assert_eq!(
            turns
                .iter()
                .filter(|turn| matches!(turn, Turn::User(m) if m == REPLAN_PROMPT))
                .count(),
            1
        );
        assert_eq!(fs::read_to_string(dir.join("a.txt")).unwrap(), "made\n");
        drop(harness);
        let (replies, modes) = replies_and_modes(&path);
        assert_eq!(modes, ["act"], "{modes:?}");
        assert!(
            replies[0].0 > 0 && !replies[0].1 && replies[1].0 > 0 && !replies[1].1,
            "short thoughts, not cut: {replies:?}"
        );
        assert!(
            replies[2..].iter().all(|&(thought, _)| thought == 0),
            "none after: {replies:?}"
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn acts_once_the_model_changes_a_file_and_thinks_again_after_a_failure() {
        // The model changes a file while investigating: it acts from then
        // on, without thinking, until a command fails, which gets it one
        // thought, and then without again. The next message has it
        // investigate again, with the file's sampling.
        let path = std::env::temp_dir().join(format!("dwim-acts-{}.jsonl", std::process::id()));
        let scripts = [
            reply_calling(20, "bash", "command", "echo made > DIR/a.txt"),
            reply_calling(0, "bash", "command", "false"),
            reply_calling(30, "bash", "command", "cat DIR/a.txt"),
            "Done.".to_string(),
            reply_calling(20, "bash", "command", "echo ok"),
            "Ok.".to_string(),
        ];
        let dir = repo();
        let mut harness = harness_in(&dir, &scripts, &path);
        let investigating = harness.chat.sampler().temperature;
        send(&mut harness, "go");
        assert_eq!(harness.chat.sampler().temperature, ACT_SAMPLING.0);
        send(&mut harness, "again");
        assert_eq!(harness.chat.sampler().temperature, investigating);
        drop(harness);
        let (replies, modes) = replies_and_modes(&path);
        assert_eq!(modes, ["act", "investigate"], "{modes:?}");
        assert_eq!(replies.len(), 6, "{replies:?}");
        assert!(replies[0].0 > 0 && !replies[0].1, "{replies:?}");
        assert_eq!(replies[1].0, 0, "acting, no thought: {replies:?}");
        assert!(replies[2].0 > 0, "a thought after the failure: {replies:?}");
        assert_eq!(replies[3].0, 0, "then none again: {replies:?}");
        assert!(
            replies[4].0 > 0,
            "investigating again on the next message: {replies:?}"
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn logs_the_whole_session() {
        // A call and its result, then replies of 1800 tokens until one is
        // stopped short and goes on after a compaction.
        let mut scripts = vec![reply(100, true), reply(100, false)];
        scripts.extend((0..8).map(|_| reply(1800, false)));
        let mut harness = harness(&scripts, 8000, "Task: go on.");
        // The long thoughts are to be stopped short for room, not ended
        // at the budget.
        harness.set_thinking_budget(None);
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
        assert!(records[0]["thinking_budget"].is_null());
        assert_eq!(records[0]["device"], "cpu");
        assert_eq!(records[0]["temperature"], 0.0);
        assert_eq!(records[0]["top_k"], 1);
        assert_eq!(records[0]["top_p"], 1.0);
        assert_eq!(records[0]["min_p"], 0.0);
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
