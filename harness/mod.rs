//! The agent around the model: runs the tools it calls and feeds the
//! results back, until it replies with text alone.
//!
//! The model thinks before every reply, for as long as it needs to: a
//! thought is ended early only when it repeats itself, or at a thinking
//! budget if one is set. A turn ends when the model replies with text
//! alone, or, after a great many calls without a reply, when it is asked
//! to say what it has done and what remains. The harness refuses a call
//! that repeats the one before it, whose output is already there.
//!
//! A [`Harness`] wraps a [`Conversation`] and reports each turn's thoughts, text,
//! tool calls, and tool output as [`Event`]s, so a user interface can show
//! them as they happen. The tools live in `crate::tools`. The system prompt
//! pushes the model to use them rather than answer from memory or ask the
//! user for a command, and carries the project's own instructions. Where
//! the model is, and what the project looks like, go ahead of the first
//! message instead, since a model won't always go looking on its own: they
//! change from one run to the next, and keeping them out of the system
//! prompt keeps the state saved after it good across days, branches, and
//! directories.
//!
//! The harness also keeps the conversation within the context window: it
//! keeps a transcript of the turns since the system prompt, and when the
//! next message or tool result would take the context past a high-water
//! mark, `compaction` has the model write a note and starts the
//! conversation over from it.
//!
//! Given a [`Log`], the harness keeps a transcript of the whole session in
//! it, thoughts and compactions included, for finding out afterwards where
//! a session's tokens went.

mod compaction;
mod log;

use std::{
    error::Error,
    fs,
    ops::ControlFlow,
    path::{Path, PathBuf},
    process::Command,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use crate::inference::{Chunk, Conversation, Turn};
use crate::tools::{ToolCall, Tools, bash, read};

pub use crate::harness::log::Log;

/// Most of a project's `AGENTS.md` that goes into the system prompt.
const MAX_INSTRUCTIONS: usize = 32 << 10;

/// Most of the files in the working directory the system prompt lists.
const MAX_FILES: usize = 50;

/// What the model gets back when it makes the same call twice in a row,
/// instead of running it again: nothing ran in between to change its output,
/// and small models otherwise tend to repeat a call over and over.
const REPEATED: &str = "error: you just ran this, and its output is above. Don't run it again: use that output, run something else, or reply to the user.";

/// What the model gets back for a call that was not run because the turn
/// was interrupted before it.
const NOT_RUN: &str = "not run: the turn was interrupted";

/// What the model gets back for a call that ran before the turn was
/// interrupted, when there is no room left for what it returned.
const DROPPED: &str =
    "interrupted: this ran, but there was no room in the context window for its output";

/// Most tokens the model may think at a time before it is made to act, if
/// they are limited: the thinking budget. The figures here are from
/// Bonsai, PrismML's ternary build of the model, which dwim ran before,
/// and are carried over to the model itself unmeasured. PrismML's known
/// issues for that build (huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf,
/// KNOWN_ISSUES.md) say it "tends to reason at length, which costs time and tokens even when
/// the answer is correct", and advise a thinking budget. Over the 351
/// thoughts of six runs without one on SWE-bench's django-11797, half were
/// under 50 tokens and 95 in 100 under 1,620; the dozen over 2,048, a
/// seventh of all the thinking, were for the most part recalling how the
/// code works and going back and forth over it, and none of those runs
/// changed the code. Three of the four runs with this budget did, two of
/// them with the fix, one decided in a thought of 1,873 tokens that a
/// budget of 1,024 would have cut.
pub const THINKING_BUDGET: Option<usize> = Some(2048);

/// Most tool calls the model may make in one turn before it is asked to
/// reply with what it has: a turn that runs this long without a reply has
/// lost its way, and the user is better served by an account of it than
/// by another call. A hundred is far past what a task's worth of calls
/// takes.
pub const MAX_CALLS: usize = 100;

/// Tool calls in a row without one that changed a file, after
/// which the model is reminded to make the change it has found: on
/// SWE-bench's django-11797 the model showed the bug and read the code at
/// fault, and then read and searched for the rest of an hour without
/// changing a file, run after run. Twenty calls take a reproduction and
/// the reading around it.
pub const NUDGE_AFTER: usize = 20;

/// What the results of a call are followed by once the model has gone
/// [`NUDGE_AFTER`] calls without a change: a reminder, which it may follow
/// with a change or with what it still needs to find out, and which takes
/// nothing from it, neither its thought nor its tools.
const NUDGE: &str = "<system-reminder>You have made many tool calls since you last changed a file. If you know what to change, make the change now with edit, then run the reproduction or the tests to check it. If you don't, say in a sentence what you still need to find out, and find that out next.</system-reminder>";

/// Most times in a turn the model is told to go on in auto mode: enough
/// for the status reports of a long task, few enough that one that never
/// ends does.
pub const AUTO_ROUNDS: usize = 10;

/// What the model is told when, in auto mode, it replies without calling
/// a tool and without saying it is done, or gives its account at the call
/// cap. On SWE-bench's django-11797 runs ended with the fix found, or made
/// and not yet checked, on a reply saying what remained, with no one to
/// answer it.
const GO_ON: &str = "Go on with the work: no one will read this until you are done, and the tools are on. Do the next thing that remains, such as making the change you have found with edit, or running your reproduction or the tests to check one you have made. Only when nothing remains, with the change made and checked, reply with what you did and end the reply with DONE on a line of its own.";

/// How a reply says, in auto mode, that the work is done: its last line.
const DONE: &str = "DONE";

/// What the model is asked with once it has made [`MAX_CALLS`] calls in a
/// turn: a message it answers without tools, which ends the turn.
const STEPS_PROMPT: &str = "You have made many tool calls in this turn without replying, and the tools are off until the user's next message. Reply now with what you have done, what you found, and what remains, in a few sentences.";

/// What ends a thought that runs to the thinking budget before the model
/// has called a tool in the turn, in the model's voice, as Qwen3's own
/// thinking budget ends one, but to check rather than to answer: Qwen3's
/// "give the solution based on the thinking directly now" has the model
/// answer from a thought that, on SWE-bench's django-11797, was most often
/// recalling how the code works, when a read or a run would have told it.
const RECALL_CLOSING: &str = "I am not certain of this from memory, so I will check it in the code now with a tool, and decide from what it shows.";

/// What ends a thought that runs to the budget once the model has called
/// a tool in the turn: what it is thinking about is then in the results,
/// not recalled, and the thought is to be decided from, not checked over
/// again. Ended with [`RECALL_CLOSING`], a thought that had found a
/// spacing bug in a status line, from the function it had just read,
/// re-read the function, thought the same two thousand tokens over again,
/// and was sent by the second closing to the git history, three calls
/// that bore nothing on the fix.
const DECIDE_CLOSING: &str = "I have thought about this enough, and what I need is in the output above. I will go with what I have worked out rather than go over it again: if there is a change to make, I make it now with edit, and otherwise I reply with it.";

/// Most tokens of the account of a turn that reached [`MAX_CALLS`], in a
/// context with room for it. A small context allows a sixteenth of itself.
const ACCOUNT_TOKENS: usize = 1024;

/// How the agent should behave: the start of the system prompt, with the
/// tools' own limits in it.
fn instructions() -> String {
    format!(
        r#"You are `dwim`, a coding agent working in the user's project directory at a Unix command line. You have a bash tool that runs shell commands there, a read tool that reads files, an edit tool that replaces a string in a file and shows the lines as they now stand, a write tool that writes a file whole, and grep and glob tools that search files for text and for names, and you may use them at any time without asking.

- For anything about the project, its files, its git history, or the system, run commands to find out before you answer. Don't answer from memory when a command can tell you.
- While you think, notice when you are recalling how code works rather than reading it: "I think", "I recall", "I believe", "probably". The code here is one version of the project, and may not be the one you remember. Stop the thought there, and read or run that code instead. The same goes for going back and forth: when you have written "wait" or "actually" about the same question twice, stop and find out with a tool. One read or one run settles what more thinking will not.
- Never say you can't access files or run commands, and never ask the user which command to run: pick one yourself.
- When a request could be a question or a task, treat it as a task and do it.
- To change something, run the commands that change it instead of explaining how.
- To find where something is, use grep for text and glob for file names, not find, grep or ls through bash: they leave out what git ignores and show line numbers to read from.
- To read a file, use read, not cat: a page of up to {page} numbered lines from the line you give, or from the first, that says where the next page starts. Prefer one larger read over many small slices: read the whole function or class you are working on, and the code around it, rather than a few lines at a time, and use grep to find specific content in a large file.
- To create a file, or replace one wholly, use write, not a heredoc. To change lines in a file, use edit, not sed or a script: give the lines that change as read shows them, with a line or two around them to find the place, and the lines to put in their place; a change in several places is several edits. Its result shows the lines as they now stand, so don't read the file back to check.
- If a command fails, read the error and try another way. A command's result ends with its exit code, and what it printed to standard error comes after a `[stderr]` line. A command is killed after {timeout} seconds unless you give it a longer timeout; give a long build or test run one.
- When a command prints more than fits, the result shows the start and the end of its output and names a file that holds all of it, with the line to read it from: use read on that file instead of running the command again.
- Once a command or a read has given you what you needed, use it and take the next step: don't run the same command again, and don't read the same lines again.
- To fix a problem, find the code responsible, make the smallest change that fixes it, then run the code or its tests to check the change, and iterate: if the check fails, read the error, change the code again, and check again.
- To check a change, run the project's own tests the way its documentation or CI runs them, and to show a problem, add a test beside the existing ones that cover the code: they come with what the code needs set up, which a script of your own has to set up again. Find out how the tests are run, from the project's README, CONTRIBUTING or CI files, before writing one.
- When you believe you know the fix, make it with edit and run your reproduction or the tests: that is the check. Don't first prove the fix in your head. An edit can be undone, and running the changed code tells you more than reasoning about it.
- Where what you see differs from what the request describes, the request says what the code should do: check what you see against that, rather than conclude there is nothing to fix.
- A `<system-reminder>` block at the end of a tool's result is from the harness, not the tool: read it and follow it.
- Keep going until the request is done, then reply in a few sentences with what you found or did.

For example, for "review commit abc123", run `git show abc123` and point out bugs and risks in the change; for "what files are here?", run `ls`; for "what time is it?", run `date`."#,
        page = read::PAGE,
        timeout = bash::TIMEOUT,
    )
}

/// How the tools are called, declared as the model's chat template puts it,
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

/// How much the model reasons before it replies, as its chat template
/// asks for it: the template opens the system prompt with a line for an
/// effort of `xhigh`, its default, or `low`, and with none for `medium`.
/// Medium unless set: PrismML's known issues for Bonsai, its ternary
/// build of the model, which these figures are from and are carried over
/// unmeasured (huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf,
/// KNOWN_ISSUES.md), advise it for the model's long reasoning, as "about as accurate" as
/// `xhigh` when the output is limited and thinking "noticeably less", and
/// find that `low` "used about as many tokens as the default `xhigh`".
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Effort {
    Xhigh,
    #[default]
    Medium,
    Low,
}

impl Effort {
    /// The line the template opens the system prompt with, if any.
    pub fn instruction(self) -> Option<&'static str> {
        match self {
            Effort::Xhigh => Some(
                "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.",
            ),
            Effort::Medium => None,
            Effort::Low => Some(
                "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.",
            ),
        }
    }
}

impl std::str::FromStr for Effort {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "xhigh" => Ok(Effort::Xhigh),
            "medium" => Ok(Effort::Medium),
            "low" => Ok(Effort::Low),
            _ => Err(format!(
                "unknown reasoning effort '{s}' (expected xhigh, medium or low)"
            )),
        }
    }
}

/// The system prompt for an agent working in `dir`, reasoning at
/// `effort`: the template's line for the effort, if it has one, the tools,
/// then how to behave, and the project's own instructions from its
/// `AGENTS.md` if it has one. Where the agent is goes in [`environment`]
/// instead, so that the prompt is the same from one run to the next.
pub fn system_prompt(dir: &Path, effort: Effort) -> String {
    let mut prompt = String::new();
    if let Some(line) = effort.instruction() {
        prompt.push_str(line);
        prompt.push_str("\n\n");
    }
    prompt.push_str(&tools());
    prompt.push_str("\n\n");
    prompt.push_str(&instructions());
    if let Ok(instructions) = fs::read_to_string(dir.join("AGENTS.md")) {
        let instructions = truncate(instructions.trim(), MAX_INSTRUCTIONS);
        prompt.push_str("\n\n# Project instructions\n\nFrom AGENTS.md:\n\n");
        prompt.push_str(&instructions);
    }
    prompt
}

/// The tools, declared as the model's chat template puts them: the start of
/// the system prompt, before what the user's system prompt says.
fn tools() -> String {
    format!(
        "# Tools\n\nYou have access to the following functions:\n\n<tools>\n{}\n</tools>\n\n{CALLING}",
        crate::tools::signatures()
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
    format!(
        "# Environment\n\nWorking directory: {}\nEvery command runs in the working directory, and every path is relative to it; do not cd there or spell it out.\nFiles: {}\nGit repository: {git}\nPlatform: {platform}\nDate: {}",
        dir.display(),
        files(dir),
        now().0,
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
#[derive(Debug, PartialEq)]
pub enum Event {
    /// Part of the model's thought, before it replies.
    Thought(String),
    /// Text of the reply.
    Text(String),
    /// A token sampled, as it may be hidden (a tool-call tag or body, a
    /// thought that closes) rather than shown as a thought or text piece.
    Token,
    /// A tool is about to run: its name and how it was called.
    Call {
        name: String,
        detail: String,
    },
    /// What the tool returned, as the model sees it.
    Output(String),
    /// The reply was stopped short because the context window was full.
    Cut,
    /// The conversation is being compacted: the model is writing the note
    /// it will go on from, which follows in `Note` pieces.
    Compacting,
    Note(String),
    /// The conversation was compacted: how many tokens it took before, and
    /// how many it takes now.
    Compacted {
        before: usize,
        after: usize,
    },
}

/// What a turn's events go to, which ends the turn early by breaking.
type OnEvent<'a> = &'a mut dyn FnMut(Event) -> ControlFlow<()>;

/// What opens the conversation over with the system prompt, once the chat
/// is cleared: from a saved state, or by reading the prompt again.
type Restart<C> = Box<dyn FnMut(&mut C) -> Result<(), Box<dyn Error>>>;

/// How a turn ended: with a reply of text alone, or interrupted; or with
/// the reply stopped short for room.
#[derive(PartialEq)]
enum Outcome {
    Done,
    Cut,
}

/// A turn as the transcript keeps it: the turn, the tokens it takes read
/// back, and whether a user turn is the harness's own prompt to the model
/// rather than the user's message.
struct Entry {
    turn: Turn,
    tokens: usize,
    prompt: bool,
}

/// The loop around a [`Conversation`] that runs the tools the model calls
/// and feeds the results back, until the model replies with text alone.
pub struct Harness<C: Conversation> {
    chat: C,
    tools: Tools,
    /// Where the agent is, until the first message carries it.
    environment: Option<String>,
    /// The directory the agent works in, to say where it is afresh after
    /// a compaction.
    dir: PathBuf,
    calls: usize,
    tool_seconds: f64,
    /// The conversation since the system prompt, or since it was last
    /// compacted.
    transcript: Vec<Entry>,
    /// Opens the conversation over with the system prompt, once the chat
    /// is cleared.
    restart: Restart<C>,
    /// Most tokens of a note.
    note_tokens: usize,
    /// Tool calls made in the turn so far, and the most there may be
    /// before the model is asked to reply.
    steps: usize,
    max_calls: usize,
    /// Tool calls in a row without one that changed a file, and how many
    /// before the model is reminded to make its change.
    unchanged: usize,
    nudge_after: usize,
    /// Auto mode: whether a reply with no call that doesn't say it is
    /// done, or the account at the call cap, is answered with word to go
    /// on, as when no one is there to answer it, and how many times it has
    /// been this turn.
    auto: bool,
    went_on: usize,
    /// The transcript of the whole session.
    log: Log,
}

/// Where a conversation's time has gone: the model's, by what it was
/// doing, and the tools'.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Stats {
    pub model: crate::inference::Stats,
    /// Tool calls run, and the time they took.
    pub calls: usize,
    pub tool_seconds: f64,
}

impl<C: Conversation> Harness<C> {
    /// An agent on `chat`, which has read the system prompt, working in
    /// `dir`, with `restart` to open the conversation over with the same
    /// prompt once the chat is cleared, when the conversation is compacted.
    pub fn new(
        mut chat: C,
        dir: &Path,
        restart: impl FnMut(&mut C) -> Result<(), Box<dyn Error>> + 'static,
    ) -> Result<Self, Box<dyn Error>> {
        let note_tokens = compaction::reserve(&mut chat)?;
        let mut harness = Self {
            chat,
            tools: Tools::new(dir),
            environment: Some(environment(dir)),
            dir: dir.to_path_buf(),
            calls: 0,
            tool_seconds: 0.0,
            transcript: Vec::new(),
            restart: Box::new(restart),
            note_tokens,
            steps: 0,
            max_calls: MAX_CALLS,
            unchanged: 0,
            nudge_after: NUDGE_AFTER,
            auto: false,
            went_on: 0,
            log: Log::none(),
        };
        harness.limit_thoughts(THINKING_BUDGET);
        Ok(harness)
    }

    /// Limits a thought to `tokens`, or lifts the limit: a thought that
    /// runs to the budget is ended with a closing in the model's voice,
    /// and the model goes on to act on what it has thought so far.
    pub fn set_thinking_budget(&mut self, tokens: Option<usize>) {
        self.limit_thoughts(tokens);
        self.log.thinking_budget(tokens);
    }

    /// Has the chat end a thought at `tokens`, the thinking budget, if
    /// there is one, with the closing for where the turn is: to check with
    /// a tool before the model has called one in the turn, when it is
    /// thinking from memory, and to decide from the results once it has.
    fn limit_thoughts(&mut self, tokens: Option<usize>) {
        let closing = if self.steps == 0 {
            RECALL_CLOSING
        } else {
            DECIDE_CLOSING
        };
        self.chat.limit_thoughts(tokens, closing);
    }

    /// Runs the model's shell commands through `shell`, as `shell -c
    /// command`, instead of `sh`: a script that runs them somewhere else,
    /// such as in a container that has what the project needs installed,
    /// while the model, and its reading and editing of files, stay here.
    pub fn set_shell(&mut self, shell: PathBuf) {
        self.tools.set_shell(shell);
    }

    /// Sets how many tool calls in a row without one that changed a file
    /// the model may make before it is reminded to make its change,
    /// [`NUDGE_AFTER`] unless set.
    #[cfg(test)]
    fn set_nudge_after(&mut self, calls: usize) {
        self.nudge_after = calls;
    }

    /// Turns auto mode on or off, for when the model runs with no one to
    /// answer it: in it, a reply with no call that doesn't end with DONE,
    /// and the account it gives at the call cap, are answered with word to
    /// go on, with the tools on, up to [`AUTO_ROUNDS`] times a turn. It is
    /// off unless set.
    pub fn set_auto(&mut self, auto: bool) {
        self.auto = auto;
    }

    /// Sets how many tool calls the model may make in one turn before it
    /// is asked to reply with what it has, [`MAX_CALLS`] unless set.
    #[cfg(test)]
    fn set_call_limit(&mut self, calls: usize) {
        self.max_calls = calls;
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
            self.chat.thought_limit(),
            system,
        );
        self.log = log;
    }

    /// The conversation the harness runs on.
    pub fn conversation(&mut self) -> &mut C {
        &mut self.chat
    }

    /// Goes on with the conversation on `chat`, a conversation opened
    /// afresh with the system prompt, in place of the one it was on: reads
    /// the conversation so far back into it, with where the agent is ahead
    /// of the first message if that went ahead of it when it was sent.
    pub fn reopen(&mut self, chat: C) -> Result<(), Box<dyn Error>> {
        self.chat = chat;
        let mut turns: Vec<Turn> = self
            .transcript
            .iter()
            .map(|entry| entry.turn.clone())
            .collect();
        if self.environment.is_none()
            && let (Some(Turn::User(message)), Some(first)) =
                (turns.first_mut(), self.transcript.first())
            && !first.prompt
        {
            *message = format!("{}\n\n{message}", environment(&self.dir));
        }
        self.chat.replay(&turns)?;
        Ok(())
    }

    /// Where the conversation's time has gone so far.
    pub fn stats(&self) -> Stats {
        Stats {
            model: self.chat.stats(),
            calls: self.calls,
            tool_seconds: self.tool_seconds,
        }
    }

    /// The turns since the system prompt, or since the conversation was
    /// last compacted.
    #[cfg(test)]
    fn transcript(&self) -> impl Iterator<Item = &Turn> {
        self.transcript.iter().map(|entry| &entry.turn)
    }

    /// Sends a message from the user, with where the agent is ahead of the
    /// first one, reporting the reply and any tool calls it makes to
    /// `on_event`. The turn ends early if `on_event` breaks. A reply
    /// stopped short for room goes on, once, after the conversation is
    /// compacted to make some.
    pub fn send(
        &mut self,
        message: &str,
        on_event: impl FnMut(Event) -> ControlFlow<()>,
    ) -> Result<(), Box<dyn Error>> {
        self.ended(on_event, |harness, on_event| {
            harness.converse(message, on_event)
        })
    }

    /// Compacts the conversation to make room in the context: has the
    /// model write a note to go on from, reported as `Note` pieces, starts
    /// the conversation over from the system prompt, and reads back where
    /// the agent is, the note, the user's earlier messages, and the most
    /// recent exchanges without their thoughts. Returns whether it did: not
    /// when there is nothing to compact yet, nor when `on_event` breaks
    /// while the note is written, which leaves the conversation as it was
    /// but for the note. Fails as the model does, or if what is read back
    /// does not fit.
    pub fn compact(
        &mut self,
        on_event: impl FnMut(Event) -> ControlFlow<()>,
    ) -> Result<bool, Box<dyn Error>> {
        self.ended(on_event, |harness, on_event| {
            compaction::compact(harness, on_event)
        })
    }

    /// Runs `work`, reporting to `on_event`, and records in the log how it
    /// ended, if not on its own: with an error, or interrupted.
    fn ended<T>(
        &mut self,
        mut on_event: impl FnMut(Event) -> ControlFlow<()>,
        work: impl FnOnce(&mut Self, OnEvent) -> Result<T, Box<dyn Error>>,
    ) -> Result<T, Box<dyn Error>> {
        let mut interrupted = false;
        let result = work(self, &mut |event| {
            let flow = on_event(event);
            interrupted |= flow.is_break();
            flow
        });
        self.log
            .end(result.as_ref().err().map(|e| e as _), interrupted);
        result
    }

    /// Sends a message from the user, as [`Harness::send`] does.
    fn converse(&mut self, message: &str, on_event: OnEvent) -> Result<(), Box<dyn Error>> {
        let ahead = self.environment.take();
        self.steps = 0;
        self.unchanged = 0;
        self.went_on = 0;
        let mut outcome = self.run(Turn::User(message.to_string()), false, ahead, on_event)?;
        if outcome == Outcome::Cut {
            if on_event(Event::Cut).is_break() {
                return Ok(());
            }
            if compaction::compact(self, on_event)? {
                outcome = self.run(
                    Turn::User(compaction::CONTINUE.to_string()),
                    true,
                    None,
                    on_event,
                )?;
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
    /// interrupted, or is stopped short for room; or until it has made
    /// [`MAX_CALLS`] calls in the turn, when it is asked to reply with
    /// what it has. A message that is the harness's own `prompt` rather
    /// than the user's is kept as such.
    fn run(
        &mut self,
        mut input: Turn,
        mut prompt: bool,
        mut ahead: Option<String>,
        on_event: OnEvent,
    ) -> Result<Outcome, Box<dyn Error>> {
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
            // The results that reach the cap need room for the account
            // they get too.
            let capped = self.steps >= self.max_calls;
            let more = if capped { self.account_tokens()? } else { 0 };
            if !compaction::make_room(self, &fed, more, on_event)? {
                // Results that a compaction was to make room for still
                // answer the reply before them.
                if let Turn::Results(outputs) = input {
                    self.close(outputs)?;
                }
                return Ok(Outcome::Done);
            }
            if capped {
                let Turn::Results(outputs) = &fed else {
                    return Err("the cap is reached by tool results".into());
                };
                self.chat.feed_results(outputs)?;
                self.record(input, prompt)?;
                let outcome = self.account(on_event)?;
                if outcome == Outcome::Cut || !self.go_on() {
                    return Ok(outcome);
                }
                self.steps = 0;
                input = Turn::User(GO_ON.to_string());
                prompt = true;
                continue;
            }
            let budget = self.chat.thought_limit();
            self.limit_thoughts(budget);
            let mut text = String::new();
            let before = self.chat.stats();
            let reply = {
                let collect = |chunk: Chunk| {
                    match chunk {
                        Chunk::Thought(piece) => self.log.stream("thought", piece),
                        Chunk::Text(piece) => {
                            self.log.stream("text", piece);
                            text.push_str(piece);
                        }
                        Chunk::Token => {}
                    }
                    on_event(event(chunk))
                };
                match &fed {
                    Turn::User(message) => self.chat.send(message, collect)?,
                    Turn::Results(outputs) => self.chat.respond(outputs, collect)?,
                    Turn::Reply { .. } => return Err("a reply is the model's to make".into()),
                }
            };
            self.log.reply(
                &reply.calls,
                reply.thought_ended,
                reply.cut,
                self.chat.tokens(),
                &before,
                &self.chat.stats(),
            );
            self.record(input, prompt)?;
            self.record(
                Turn::Reply {
                    text,
                    calls: reply.calls.clone(),
                },
                false,
            )?;
            if reply.cut {
                return Ok(Outcome::Cut);
            }
            if reply.calls.is_empty() {
                if !self.go_on() {
                    return Ok(Outcome::Done);
                }
                input = Turn::User(GO_ON.to_string());
                prompt = true;
                continue;
            }
            // Every call of the reply gets a result, as the model expects
            // of a reply that calls: once the turn is interrupted, the
            // calls that remain get word that they were not run.
            let mut outputs = Vec::new();
            let mut interrupted = false;
            for call in &reply.calls {
                let mut seconds = None;
                let mut changed = false;
                self.steps += 1;
                let output = match ToolCall::parse(call) {
                    Ok(call) => {
                        let detail = crate::tools::describe(&call);
                        self.log.call(&call.name, &detail);
                        interrupted = interrupted
                            || on_event(Event::Call {
                                name: call.name.clone(),
                                detail: detail.clone(),
                            })
                            .is_break();
                        let this = Some((call.name.clone(), call.arguments.to_string()));
                        if interrupted {
                            NOT_RUN.to_string()
                        } else if this == last {
                            REPEATED.to_string()
                        } else {
                            last = this;
                            let start = Instant::now();
                            let output = self.tools.run(&call);
                            self.calls += 1;
                            let elapsed = start.elapsed().as_secs_f64();
                            self.tool_seconds += elapsed;
                            seconds = Some(elapsed);
                            changed = output.changed;
                            output.text
                        }
                    }
                    Err(_) if interrupted => NOT_RUN.to_string(),
                    Err(e) => format!("error: malformed tool call: {e}"),
                };
                self.log.output(&output, seconds);
                interrupted = interrupted || on_event(Event::Output(output.clone())).is_break();
                if changed {
                    self.unchanged = 0;
                } else {
                    self.unchanged += 1;
                }
                outputs.push(output);
            }
            if interrupted {
                self.close(outputs)?;
                return Ok(Outcome::Done);
            }
            if self.unchanged >= self.nudge_after
                && let Some(last) = outputs.last_mut()
            {
                last.push_str("\n\n");
                last.push_str(NUDGE);
                self.log.nudge(NUDGE);
                self.unchanged = 0;
            }
            input = Turn::Results(outputs);
            prompt = false;
        }
    }

    /// Feeds `outputs` as the results of the calls the last reply made,
    /// with no reply to them, and records them: how a turn interrupted at
    /// its calls ends, so that the next message follows a whole exchange.
    /// No conversation is compacted to make room for them, since that is
    /// the model's work too: if they do not fit, the calls that ran get
    /// word that their output was dropped instead.
    fn close(&mut self, mut outputs: Vec<String>) -> Result<(), Box<dyn Error>> {
        let mut turn = Turn::Results(outputs.clone());
        if self.chat.measure(&turn)? > self.chat.free() {
            for output in outputs.iter_mut().filter(|output| *output != NOT_RUN) {
                *output = DROPPED.to_string();
            }
            turn = Turn::Results(outputs.clone());
        }
        self.chat.feed_results(&outputs)?;
        self.record(turn, false)
    }

    /// Whether the model, in auto mode, is to be told to go on after the
    /// reply it just gave: not if it said it is done, nor once it has been
    /// told [`AUTO_ROUNDS`] times this turn. Counts it if so.
    fn go_on(&mut self) -> bool {
        let done = matches!(
            self.transcript.last(),
            Some(Entry { turn: Turn::Reply { text, .. }, .. }) if said_done(text)
        );
        if !self.auto || done || self.went_on >= AUTO_ROUNDS {
            return false;
        }
        self.went_on += 1;
        true
    }

    /// Asks the model, once it has made [`MAX_CALLS`] calls in a turn, for
    /// an account of the turn, which it gives without tools, reported as
    /// `Text` pieces, and records the exchange. The turn ends with it.
    fn account(&mut self, on_event: OnEvent) -> Result<Outcome, Box<dyn Error>> {
        self.log.user(STEPS_PROMPT, None, self.chat.tokens());
        let before = self.chat.stats();
        let answer = self
            .chat
            .answer(STEPS_PROMPT, &[], self.account_limit(), |text| {
                self.log.stream("text", text);
                on_event(Event::Text(text.to_string()))
            })?;
        self.log.reply(
            &[],
            false,
            answer.cut,
            self.chat.tokens(),
            &before,
            &self.chat.stats(),
        );
        self.record(Turn::User(STEPS_PROMPT.to_string()), true)?;
        self.record(
            Turn::Reply {
                text: answer.text,
                calls: Vec::new(),
            },
            false,
        )?;
        Ok(if answer.cut {
            Outcome::Cut
        } else {
            Outcome::Done
        })
    }

    /// Most tokens of the account of a turn: [`ACCOUNT_TOKENS`], or a
    /// sixteenth of the context if that is less.
    fn account_limit(&self) -> usize {
        ACCOUNT_TOKENS.min(self.chat.capacity() / 16)
    }

    /// Most tokens the account of a turn takes, with the prompt for it.
    fn account_tokens(&mut self) -> Result<usize, Box<dyn Error>> {
        self.chat.answer_tokens(STEPS_PROMPT, self.account_limit())
    }

    /// Adds a turn to the transcript, a user turn as the harness's own
    /// `prompt` or the user's message.
    fn record(&mut self, turn: Turn, prompt: bool) -> Result<(), Box<dyn Error>> {
        let tokens = self.chat.measure(&turn)?;
        self.transcript.push(Entry {
            turn,
            tokens,
            prompt,
        });
        Ok(())
    }
}

impl<C: Conversation> Drop for Harness<C> {
    /// Ends the transcript: the session finished.
    fn drop(&mut self) {
        let stats = self.stats();
        self.log.finish(self.chat.tokens(), &stats);
    }
}

/// Whether a reply says the work is done: its last line is [`DONE`], with
/// any emphasis or punctuation around the word.
fn said_done(text: &str) -> bool {
    text.lines()
        .rev()
        .find(|line| !line.trim().is_empty())
        .is_some_and(|line| line.trim_matches(|c: char| !c.is_alphanumeric()) == DONE)
}

impl Event {
    /// What the user is told of the event, for the ones that are told
    /// rather than shown: the reply stopped short, and the compaction as
    /// it starts and as it ends.
    pub fn notice(&self) -> Option<String> {
        match self {
            Event::Cut => Some("Stopped short: the context window is full".to_string()),
            Event::Compacting => Some("Compacting the conversation…".to_string()),
            Event::Compacted { before, after } => {
                Some(format!("Compacted from {before} to {after} tokens"))
            }
            _ => None,
        }
    }
}

/// The event for a piece of the model's reply.
fn event(chunk: Chunk) -> Event {
    match chunk {
        Chunk::Thought(text) => Event::Thought(text.to_string()),
        Chunk::Text(text) => Event::Text(text.to_string()),
        Chunk::Token => Event::Token,
    }
}

#[cfg(test)]
mod tests {
    use crate::inference::{Chat, Format, Sampler, Tokenizer, testing::Scripted};

    use super::*;

    /// A thought of `n` bytes that goes somewhere: the counting numbers run
    /// together, which the tiny tokenizer makes `n` tokens, and which never
    /// repeats a window of itself, as a thought of one letter over would.
    fn filler(n: usize) -> String {
        (1..)
            .flat_map(|i: u32| i.to_string().into_bytes())
            .map(char::from)
            .take(n)
            .collect()
    }

    /// A reply after a thought of `thought` bytes, either with text alone
    /// or calling `bash`.
    pub fn reply(thought: usize, call: bool) -> String {
        reply_saying(thought, 0, call)
    }

    /// A reply after a thought of `thought` bytes that says `text` bytes
    /// of text, and then either ends or calls `bash`.
    pub fn reply_saying(thought: usize, text: usize, call: bool) -> String {
        let body = if call {
            "<tool_call>\n<function=bash>\n<parameter=command>\necho ok\n</parameter>\n</function>\n</tool_call>"
        } else {
            "Ok."
        };
        let text = if text > 0 {
            format!("{}\n", filler(text))
        } else {
            String::new()
        };
        format!("{}\n</think>\n\n{text}{body}", filler(thought))
    }

    /// A harness around a scripted model with room for `max_len` tokens,
    /// which says `scripts` in turn and `note` when asked for a note.
    pub fn harness(scripts: &[String], max_len: usize, note: &str) -> Harness<Chat<Scripted>> {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        Harness::new(chat(scripts, max_len, note), dir, |chat| {
            chat.system("Be brief.", |_, _| {})
        })
        .unwrap()
    }

    /// A conversation as `harness` has it, opened with its system prompt.
    fn chat(scripts: &[String], max_len: usize, note: &str) -> Chat<Scripted> {
        let tokenizer = Tokenizer::tiny();
        let scripts: Vec<&str> = scripts.iter().map(String::as_str).collect();
        let model = Scripted::new(&tokenizer, &scripts, max_len).answers(&tokenizer, note);
        let mut chat = Chat::new(
            model,
            Format::new(tokenizer).unwrap(),
            Sampler::new(0.0, 1, 1.0, 0.0, 1),
        )
        .unwrap();
        chat.system("Be brief.", |_, _| {}).unwrap();
        chat
    }

    #[test]
    fn a_reopened_conversation_reads_back_as_it_was() {
        // The conversation read back into a fresh one, as when the server
        // is restarted, takes as many tokens as it did, with where the
        // agent is ahead of the first message as it was sent.
        let scripts = [reply(20, false), reply(20, false)];
        let mut harness = harness(&scripts, 8000, "");
        send(&mut harness, "one");
        send(&mut harness, "two");
        let tokens = harness.chat.tokens();
        harness.reopen(chat(&[], 8000, "")).unwrap();
        assert_eq!(harness.chat.tokens(), tokens);
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
    pub fn send(
        harness: &mut Harness<Chat<Scripted>>,
        message: &str,
    ) -> (Vec<&'static str>, String) {
        let mut events = Vec::new();
        let mut note = String::new();
        harness
            .send(message, |event| {
                let name = match event {
                    Event::Thought(_) => "thought",
                    Event::Text(_) => "text",
                    Event::Token => return ControlFlow::Continue(()),
                    Event::Call { .. } => "call",
                    Event::Output(_) => "output",
                    Event::Cut => "cut",
                    Event::Compacting => "compacting",
                    Event::Note(text) => {
                        note.push_str(&text);
                        "note"
                    }
                    Event::Compacted { .. } => "compacted",
                };
                events.push(name);
                ControlFlow::Continue(())
            })
            .unwrap();
        events.dedup();
        (events, note)
    }

    #[test]
    fn ends_a_thought_at_the_thinking_budget() {
        // A thought of 1000 tokens, under a budget of 100, is ended there
        // with the closing, and the reply goes on to its call and its text.
        let scripts = [reply(1000, true), reply(1000, false)];
        let mut limited = harness(&scripts, 8000, "");
        assert_eq!(limited.chat.thought_limit(), THINKING_BUDGET);
        assert_eq!(THINKING_BUDGET, Some(2048));
        let path = std::env::temp_dir().join(format!("dwim-budget-{}.jsonl", std::process::id()));
        limited.keep_log(Log::create(&path).unwrap(), "tiny", "cpu", "Be brief.");
        limited.set_thinking_budget(Some(100));
        let mut thought = String::new();
        let mut events = Vec::new();
        limited
            .send("go", |event| {
                let name = match event {
                    Event::Thought(text) => {
                        thought.push_str(&text);
                        "thought"
                    }
                    Event::Text(_) => "text",
                    Event::Call { .. } => "call",
                    Event::Output(_) => "output",
                    Event::Token => return ControlFlow::Continue(()),
                    _ => "other",
                };
                events.push(name);
                ControlFlow::Continue(())
            })
            .unwrap();
        events.dedup();
        assert_eq!(
            events,
            ["thought", "text", "call", "output", "thought", "text"]
        );
        // The first thought, before any call in the turn, is ended to
        // check; the second, after the call, to decide.
        let recall = thought.find(RECALL_CLOSING).expect(&thought);
        let decide = thought.find(DECIDE_CLOSING).expect(&thought);
        assert!(recall < decide, "{thought}");
        assert_eq!(thought.matches(RECALL_CLOSING).count(), 1, "{thought}");
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
        assert_eq!(records[0]["thinking_budget"], 2048);
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
        assert_eq!(free.chat.thought_limit(), None);
        let mut thought = String::new();
        free.send("go", |event| {
            if let Event::Thought(text) = event {
                thought.push_str(&text);
            }
            ControlFlow::Continue(())
        })
        .unwrap();
        assert!(!thought.contains(RECALL_CLOSING));
        assert!(!thought.contains(DECIDE_CLOSING));
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
            format!("{}\n</think>\n\n{call}", filler(thought))
        }
    }

    /// A directory of its own, for a harness to work in. The scripts that
    /// work in it name it, as the model does: bash runs where the process
    /// does.
    fn scratch() -> PathBuf {
        static DIRS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = DIRS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("dwim-harness-{}-{n}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A harness around a scripted model working in `dir`, under a budget
    /// of 100, keeping a log at `path`. `DIR` in a script stands for the
    /// directory.
    fn harness_in(dir: &Path, scripts: &[String], path: &Path) -> Harness<Chat<Scripted>> {
        harness_within(dir, scripts, path, 100_000)
    }

    /// A harness as `harness_in` makes one, with room for `max_len`
    /// tokens.
    fn harness_within(
        dir: &Path,
        scripts: &[String],
        path: &Path,
        max_len: usize,
    ) -> Harness<Chat<Scripted>> {
        let at = dir.display().to_string();
        let scripts: Vec<String> = scripts.iter().map(|s| s.replace("DIR", &at)).collect();
        let tokenizer = Tokenizer::tiny();
        let scripts: Vec<&str> = scripts.iter().map(String::as_str).collect();
        let model = Scripted::new(&tokenizer, &scripts, max_len);
        let mut chat = Chat::new(
            model,
            Format::new(tokenizer).unwrap(),
            Sampler::new(1.0, 1, 0.95, 0.0, 1),
        )
        .unwrap();
        chat.system("Be brief.", |_, _| {}).unwrap();
        let mut harness =
            Harness::new(chat, dir, |chat| chat.system("Be brief.", |_, _| {})).unwrap();
        harness.set_thinking_budget(Some(100));
        harness.keep_log(Log::create(path).unwrap(), "tiny", "cpu", "Be brief.");
        harness
    }

    /// The replies of the log at `path`, as thought tokens and whether the
    /// thought was ended at the budget.
    fn replies(path: &Path) -> Vec<(u64, bool)> {
        records(path)
            .iter()
            .filter(|record| record["type"] == "reply")
            .map(|reply| {
                (
                    reply["thought_tokens"].as_u64().unwrap(),
                    reply["thought_ended"].as_bool().unwrap(),
                )
            })
            .collect()
    }

    #[test]
    fn thinks_before_every_reply() {
        // A change made does not turn thinking off, nor does a command
        // that fails: every reply thinks, at the sampling the chat was
        // made with.
        let path = std::env::temp_dir().join(format!("dwim-thinks-{}.jsonl", std::process::id()));
        let scripts = [
            reply_calling(20, "bash", "command", "echo made > DIR/a.txt"),
            reply_calling(30, "bash", "command", "false"),
            reply_calling(40, "bash", "command", "cat DIR/a.txt"),
            reply(50, false),
        ];
        let dir = scratch();
        let mut harness = harness_in(&dir, &scripts, &path);
        let temperature = harness.chat.sampler().temperature;
        let (events, _) = send(&mut harness, "go");
        assert_eq!(
            events,
            [
                "thought", "text", "call", "output", "thought", "text", "call", "output",
                "thought", "text", "call", "output", "thought", "text"
            ],
            "{events:?}"
        );
        assert_eq!(fs::read_to_string(dir.join("a.txt")).unwrap(), "made\n");
        assert_eq!(harness.chat.sampler().temperature, temperature);
        drop(harness);
        let replies = replies(&path);
        assert_eq!(replies.len(), 4, "{replies:?}");
        assert!(
            replies
                .iter()
                .all(|&(thought, ended)| thought > 0 && !ended),
            "{replies:?}"
        );
        fs::remove_dir_all(dir).unwrap();
    }

    /// A reply that calls `bash` twice, to make a file in `dir` and then
    /// another.
    fn reply_making_two(dir: &Path) -> String {
        let call = |file: &str| {
            format!(
                "<tool_call>\n<function=bash>\n<parameter=command>\ntouch {}\n</parameter>\n</function>\n</tool_call>",
                dir.join(file).display()
            )
        };
        format!("{}\n</think>\n\n{}\n{}", filler(10), call("a"), call("b"))
    }

    /// Sends `message`, breaking at the first event that `stop` is true of.
    fn send_until(
        harness: &mut Harness<Chat<Scripted>>,
        message: &str,
        stop: impl Fn(&Event) -> bool,
    ) {
        harness
            .send(message, |event| {
                if stop(&event) {
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                }
            })
            .unwrap();
    }

    /// The transcript's turns as a letter each: a user's, a reply, results.
    fn shape(harness: &Harness<Chat<Scripted>>) -> String {
        harness
            .transcript()
            .map(|turn| match turn {
                Turn::User(_) => 'u',
                Turn::Reply { .. } => 'r',
                Turn::Results(_) => 't',
            })
            .collect()
    }

    #[test]
    fn answers_the_calls_of_a_turn_interrupted_before_they_run() {
        let dir = scratch();
        let scripts = [reply_making_two(&dir), reply(10, false)];
        let mut harness = harness(&scripts, 100_000, "");
        send_until(&mut harness, "go", |event| {
            matches!(event, Event::Call { .. })
        });
        assert!(!dir.join("a").exists() && !dir.join("b").exists());
        send(&mut harness, "what happened?");

        // The reply's two calls are answered before the next message, and
        // the model was fed what the transcript has.
        assert_eq!(shape(&harness), "urtur");
        let turns: Vec<&Turn> = harness.transcript().collect();
        let Turn::Results(outputs) = turns[2] else {
            panic!("{:?}", turns[2])
        };
        assert_eq!(outputs.len(), 2, "{outputs:?}");
        assert!(
            outputs.iter().all(|output| output == NOT_RUN),
            "{outputs:?}"
        );
        let tokenizer = Tokenizer::tiny();
        let fed: Vec<u8> = harness
            .chat
            .model()
            .fed
            .iter()
            .flat_map(|&token| tokenizer.decode(token).to_vec())
            .collect();
        let fed = String::from_utf8(fed).unwrap();
        let results = fed.find("<tool_response>").expect("no results were fed");
        assert_eq!(fed.matches("<tool_response>").count(), 2, "{fed}");
        assert!(results < fed.find("what happened?").unwrap(), "{fed}");
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn answers_the_calls_of_a_turn_interrupted_after_one_ran() {
        let dir = scratch();
        let scripts = [reply_making_two(&dir), reply(10, false)];
        let mut harness = harness(&scripts, 100_000, "");
        send_until(&mut harness, "go", |event| {
            matches!(event, Event::Output(_))
        });
        assert!(dir.join("a").exists() && !dir.join("b").exists());
        send(&mut harness, "what happened?");

        // The call that ran keeps its output, and the other was not run.
        assert_eq!(shape(&harness), "urtur");
        let turns: Vec<&Turn> = harness.transcript().collect();
        let Turn::Results(outputs) = turns[2] else {
            panic!("{:?}", turns[2])
        };
        assert_eq!(outputs.len(), 2, "{outputs:?}");
        assert_ne!(outputs[0], NOT_RUN);
        assert_eq!(outputs[1], NOT_RUN);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn reminds_the_model_to_change_a_file_after_many_calls_without() {
        // Under a nudge of three: the results of the third call without a
        // change carry the reminder, and the count starts over after it;
        // a file written starts it over too.
        let dir = scratch();
        let file = dir.join("a.txt");
        let write = format!(
            "{}\n</think>\n\n<tool_call>\n<function=write>\n<parameter=path>\n{}\n</parameter>\n<parameter=content>\nmade\n</parameter>\n</function>\n</tool_call>",
            filler(10),
            file.display()
        );
        let mut scripts: Vec<String> = (0..3)
            .map(|i| reply_calling(10, "bash", "command", &format!("echo {i}")))
            .collect();
        scripts.push(write);
        scripts.extend((3..5).map(|i| reply_calling(10, "bash", "command", &format!("echo {i}"))));
        scripts.push(reply(10, false));
        let mut harness = harness(&scripts, 100_000, "");
        harness.set_nudge_after(3);
        send(&mut harness, "go");
        let results: Vec<&String> = harness
            .transcript()
            .filter_map(|turn| match turn {
                Turn::Results(outputs) => outputs.last(),
                _ => None,
            })
            .collect();
        assert_eq!(results.len(), 6, "{results:?}");
        let nudged: Vec<bool> = results.iter().map(|r| r.ends_with(NUDGE)).collect();
        assert_eq!(
            nudged,
            [false, false, true, false, false, false],
            "{results:?}"
        );
        assert_eq!(fs::read_to_string(&file).unwrap(), "made\n");
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn goes_on_in_auto_mode_until_the_model_says_it_is_done() {
        // In auto mode, a reply of text alone is answered with word to go
        // on, and the turn goes on to the calls after it, until a reply
        // ends with DONE; out of it, the turn ends at the first reply.
        let scripts = [
            reply(10, false),
            reply_calling(10, "bash", "command", "echo 1"),
            reply(10, false),
            format!(
                "{}\n</think>\n\nChanged it and checked it.\n\n**DONE**",
                filler(10)
            ),
            reply(10, false),
        ];
        let mut auto = harness(&scripts, 100_000, "");
        auto.set_auto(true);
        send(&mut auto, "go");
        let turns: Vec<&Turn> = auto.transcript().collect();
        let told = turns
            .iter()
            .filter(|turn| matches!(turn, Turn::User(m) if m == GO_ON))
            .count();
        assert_eq!(told, 2, "{turns:?}");
        assert!(matches!(turns.last(), Some(Turn::Reply { text, .. }) if said_done(text)));
        assert!(turns.iter().any(|turn| matches!(turn, Turn::Results(_))));

        let mut plain = harness(&scripts, 100_000, "");
        send(&mut plain, "go");
        assert_eq!(plain.transcript().count(), 2);

        // A model that never says it is done is told to go on no more
        // than AUTO_ROUNDS times.
        let scripts: Vec<String> = (0..AUTO_ROUNDS + 3).map(|_| reply(10, false)).collect();
        let mut endless = harness(&scripts, 100_000, "");
        endless.set_auto(true);
        send(&mut endless, "go");
        let told = endless
            .transcript()
            .filter(|turn| matches!(turn, Turn::User(m) if m == GO_ON))
            .count();
        assert_eq!(told, AUTO_ROUNDS);
    }

    #[test]
    fn says_done_on_the_last_line() {
        assert!(said_done("All set.\nDONE"));
        assert!(said_done("All set.\n\n**DONE.**\n"));
        assert!(!said_done("DONE with the reading, now the fix."));
        assert!(!said_done("What remains: run the tests."));
    }

    #[test]
    fn goes_on_after_the_account_at_the_call_cap() {
        // In auto mode, the account at a cap of two is answered with word
        // to go on, the count starts over with the tools on, and the turn
        // ends when a reply says it is done.
        let scripts = [
            reply_calling(10, "bash", "command", "echo 0"),
            reply_calling(10, "bash", "command", "echo 1"),
            reply_calling(10, "bash", "command", "echo 2"),
            format!("{}\n</think>\n\nAll checked.\nDONE", filler(10)),
        ];
        let mut harness = harness(&scripts, 100_000, "I looked twice.");
        harness.set_call_limit(2);
        harness.set_auto(true);
        send(&mut harness, "go");
        let turns: Vec<&Turn> = harness.transcript().collect();
        let at = turns
            .iter()
            .position(|turn| matches!(turn, Turn::User(m) if m == STEPS_PROMPT))
            .expect("asked for an account");
        assert!(
            matches!(turns[at + 2], Turn::User(m) if m == GO_ON),
            "{turns:?}"
        );
        assert!(matches!(turns[at + 3], Turn::Reply { calls, .. } if calls.len() == 1));
        assert!(matches!(turns.last(), Some(Turn::Reply { text, .. }) if said_done(text)));
    }

    #[test]
    fn asks_for_an_account_at_the_call_cap() {
        // Three calls under a cap of three: the results of the third are
        // fed, the model is asked for an account of the turn, which it
        // gives without tools, and the turn ends. The next message starts
        // the count over.
        let scripts: Vec<String> = (0..6)
            .map(|i| reply_calling(10, "bash", "command", &format!("echo {i}")))
            .collect();
        let mut harness = harness(&scripts, 100_000, "I looked three times.");
        harness.set_call_limit(3);
        let (events, _) = send(&mut harness, "go");
        assert_eq!(
            events,
            [
                "thought", "text", "call", "output", "thought", "text", "call", "output",
                "thought", "text", "call", "output", "text"
            ],
            "{events:?}"
        );
        let turns: Vec<&Turn> = harness.transcript().collect();
        assert!(
            matches!(&turns[turns.len() - 2], Turn::User(m) if m == STEPS_PROMPT),
            "{:?}",
            turns[turns.len() - 2]
        );
        assert!(
            matches!(turns.last(), Some(Turn::Reply { text, calls }) if text == "I looked three times." && calls.is_empty()),
            "{:?}",
            turns.last()
        );
        let (events, _) = send(&mut harness, "again");
        assert_eq!(events[..4], ["thought", "text", "call", "output"]);
    }

    #[test]
    fn logs_the_whole_session() {
        // A call and its result, then replies after thoughts of 1800
        // tokens, which leave the context as each reply is made, and one
        // whose thought is stopped short and goes on after a compaction.
        let mut scripts = vec![reply(100, true), reply(100, false)];
        scripts.extend((0..8).map(|i| reply(if i == 2 { 8000 } else { 1800 }, false)));
        let mut harness = harness(&scripts, 8000, " go on.");
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
        assert_eq!(records[2]["text"].as_str().unwrap().trim(), filler(100));
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
        assert_eq!(
            records[cut + 2]["text"],
            "Task: go on.\n\nFindings:\n\nDone:\n\nNext:"
        );
        assert!(records[cut + 3]["after"].as_u64() < records[cut + 3]["before"].as_u64());
        let head = records[cut + 3]["head"].as_str().unwrap();
        assert!(head.starts_with("# Environment"), "{head}");
        assert!(head.contains("Task: go on."), "{head}");
        assert!(head.contains("message 0"), "{head}");
        assert!(records[cut + 3]["turns"].as_u64().unwrap() > 0);
        assert_eq!(records[cut + 4]["text"], compaction::CONTINUE);
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
    fn dates() {
        assert_eq!(date(0), "1970-01-01");
        assert_eq!(date(11016), "2000-02-29");
        assert_eq!(date(19782), "2024-02-29");
        assert_eq!(date(20711), "2026-09-15");
    }

    #[test]
    fn opens_the_prompt_with_the_line_for_the_effort() {
        // As the model's template does: a line for xhigh and for low, ahead of
        // the tools, and none for medium, the default.
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        assert_eq!(Effort::default(), Effort::Medium);
        assert!(system_prompt(dir, Effort::Medium).starts_with("# Tools\n"));
        for (effort, word) in [(Effort::Xhigh, "xhigh"), (Effort::Low, "low")] {
            let prompt = system_prompt(dir, effort);
            let line = format!("Reasoning effort is set to {word}.");
            assert!(prompt.starts_with(&line), "{prompt}");
            assert!(prompt.contains(".\n\n# Tools\n"));
            assert_eq!(word.parse::<Effort>(), Ok(effort));
        }
        assert_eq!("medium".parse::<Effort>(), Ok(Effort::Medium));
        assert!("high".parse::<Effort>().is_err());
    }

    #[test]
    fn describes_the_project() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let prompt = system_prompt(dir, Effort::Medium);
        assert!(prompt.starts_with("# Tools\n"));
        assert!(
            prompt.contains("<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"bash\"")
        );
        assert!(prompt.contains("\n{\"type\": \"function\", \"function\": {\"name\": \"read\""));
        assert!(prompt.contains(CALLING));
        assert!(prompt.contains(&instructions()));
        // The tools' limits as the tools have them.
        assert!(prompt.contains(&format!("a page of up to {} numbered lines", read::PAGE)));
        assert!(prompt.contains(&format!("killed after {} seconds", bash::TIMEOUT)));
        assert!(prompt.contains("From AGENTS.md:\n\n# `dwim`"));
        // What changes between runs is not in the prompt, so that the state
        // saved after it is good for the next run.
        assert!(!prompt.contains("# Environment"));
        assert!(!prompt.contains("Date: "));
    }

    #[test]
    fn describes_the_environment() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let environment = environment(dir);
        assert!(environment.starts_with("# Environment\n"));
        assert!(environment.contains(&format!("Working directory: {}\n", dir.display())));
        assert!(environment.contains("\nEvery command runs in the working directory"));
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
