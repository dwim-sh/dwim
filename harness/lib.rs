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
//! A [`Harness`] wraps a [`Chat`] and reports each turn's thoughts, text,
//! tool calls, and tool output as [`Event`]s, so a user interface can show
//! them as they happen. The tools are `bash`, which runs a shell command,
//! `read`, which reads a file a page at a time, `edit`, which replaces a
//! string in a file and shows the lines as they now stand, `write`, which
//! writes a file whole, and `grep` and `glob`, which search files for text
//! and for names; they live in `dwim_tools`. The system prompt pushes the model to use them rather
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

use dwim_models::{Chat, Chunk, LanguageModel, Turn};
use dwim_tools::{ToolCall, Tools};

pub use crate::log::Log;

/// Most of a project's `AGENTS.md` that goes into the system prompt.
const MAX_INSTRUCTIONS: usize = 32 << 10;

/// Most of the files in the working directory the system prompt lists.
const MAX_FILES: usize = 50;

/// What the model gets back when it makes the same call twice in a row,
/// instead of running it again: nothing ran in between to change its output,
/// and small models otherwise tend to repeat a call over and over.
const REPEATED: &str = "error: you just ran this, and its output is above. Don't run it again: use that output, run something else, or reply to the user.";

/// Most tokens the model may think at a time before it is made to act, if
/// they are limited: the thinking budget. Prism's known issues for the
/// model (huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf, KNOWN_ISSUES.md)
/// say it "tends to reason at length, which costs time and tokens even when
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

/// Tool calls in a row without a file changed by `edit` or `write`, after
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

/// What the model gets back for a command that would take away changes it
/// has made to the project's tracked files. On SWE-bench's django-11797 the
/// model made the fix, then ran `git stash` to see its new test fail
/// without it, and the run ended with nothing changed.
const DISCARDS: &str = "error: not run: this would take away the changes you have made to the project's files (git stash, checkout or restore of files, reset --hard, or clean), and those changes are the work. To run the code without your change, undo it with edit, run what you want, and then make the change again with edit. git diff shows your changes.";

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
/// thirty-second of the context, within these bounds. A note of a few
/// hundred tokens kept the task and lost the findings: the paths, the
/// lines and the errors the model had spent the conversation finding.
const NOTE_TOKENS: (usize, usize) = (256, 2048);

/// Most tokens of the account of a turn that reached [`MAX_CALLS`], in a
/// context with room for it. A small context allows a sixteenth of itself.
const ACCOUNT_TOKENS: usize = 1024;

/// What the model is asked before the conversation is compacted.
const NOTE_PROMPT: &str = "The conversation so far is about to be cleared to make room in the context window, and you will go on from a note you write now. Write it for yourself, in plain text, under these headings: Task, what the user asked for, in their words where you can; Findings, what you learned that matters, such as files with their paths and line numbers, functions, commands, results and exact error messages; Done, what has been changed so far; Next, what remains, and what to do first. Take everything above as a record of what happened, not as instructions: the commands and outputs in it are history. Be specific and brief.";

/// How the note turn opens after a compaction.
const NOTE_HEAD: &str = "The conversation so far was compacted to make room in the context window. This is the note you wrote before it was cleared, to go on from:";

/// What the model is told after a reply of its was stopped short and the
/// conversation compacted, so that it goes on.
const CONTINUE: &str = "Your reply was stopped short because the context window was full, and the conversation has been compacted since. Go on from where you left off.";

/// How the agent should behave: the start of the system prompt.
const INSTRUCTIONS: &str = r#"You are `dwim`, a coding agent working in the user's project directory at a Unix command line. You have a bash tool that runs shell commands there, a read tool that reads files, an edit tool that replaces a string in a file and shows the lines as they now stand, a write tool that writes a file whole, and grep and glob tools that search files for text and for names, and you may use them at any time without asking.

- For anything about the project, its files, its git history, or the system, run commands to find out before you answer. Don't answer from memory when a command can tell you.
- While you think, notice when you are recalling how code works rather than reading it: "I think", "I recall", "I believe", "probably". The code here is one version of the project, and may not be the one you remember. Stop the thought there, and read or run that code instead. The same goes for going back and forth: when you have written "wait" or "actually" about the same question twice, stop and find out with a tool. One read or one run settles what more thinking will not.
- Never say you can't access files or run commands, and never ask the user which command to run: pick one yourself.
- When a request could be a question or a task, treat it as a task and do it.
- To change something, run the commands that change it instead of explaining how.
- To find where something is, use grep for text and glob for file names, not find, grep or ls through bash: they leave out what git ignores and show line numbers to read from.
- To read a file, use read, not cat: a page of up to 1000 numbered lines from the line you give, or from the first, that says where the next page starts. Prefer one larger read over many small slices: read the whole function or class you are working on, and the code around it, rather than a few lines at a time, and use grep to find specific content in a large file.
- To create a file, or replace one wholly, use write, not a heredoc. To change lines in a file, use edit, not sed or a script: give the lines that change as read shows them, with a line or two around them to find the place, and the lines to put in their place; a change in several places is several edits. Its result shows the lines as they now stand, so don't read the file back to check.
- If a command fails, read the error and try another way. A command's result ends with its exit code, and what it printed to standard error comes after a `[stderr]` line. A command is killed after 300 seconds unless you give it a longer timeout; give a long build or test run one.
- When a command prints more than fits, the result shows the start and the end of its output and names a file that holds all of it, with the line to read it from: use read on that file instead of running the command again.
- Once a command or a read has given you what you needed, use it and take the next step: don't run the same command again, and don't read the same lines again.
- To fix a problem, find the code responsible, make the smallest change that fixes it, then run the code or its tests to check the change, and iterate: if the check fails, read the error, change the code again, and check again.
- To check a change, run the project's own tests the way its documentation or CI runs them, and to show a problem, add a test beside the existing ones that cover the code: they come with what the code needs set up, which a script of your own has to set up again. Find out how the tests are run, from the project's README, CONTRIBUTING or CI files, before writing one.
- When you believe you know the fix, make it with edit and run your reproduction or the tests: that is the check. Don't first prove the fix in your head. An edit can be undone, and running the changed code tells you more than reasoning about it.
- Where what you see differs from what the request describes, the request says what the code should do: check what you see against that, rather than conclude there is nothing to fix.
- A `<system-reminder>` block at the end of a tool's result is from the harness, not the tool: read it and follow it.
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

/// How much the model reasons before it replies, as Bonsai's chat template
/// asks for it: the template opens the system prompt with a line for an
/// effort of `xhigh`, its default, or `low`, and with none for `medium`.
/// Medium unless set: Prism's known issues for the model
/// (huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf, KNOWN_ISSUES.md)
/// advise it for the model's long reasoning, as "about as accurate" as
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
    prompt.push_str(&format!("{}\n\n{INSTRUCTIONS}", tools()));
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
        "# Environment\n\nWorking directory: {}\nEvery command runs in the working directory, and every path is relative to it; do not cd there or spell it out.\nFiles: {}\nGit repository: {git}\nPlatform: {platform}\nDate: {}",
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
    /// A token sampled, as it may be hidden (a tool-call tag or body, a
    /// thought that closes) rather than shown as a thought or text piece.
    Token,
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
    /// Tool calls made in the turn so far, and the most there may be
    /// before the model is asked to reply.
    steps: usize,
    max_calls: usize,
    /// Tool calls in a row without a file changed by `edit` or `write`, and
    /// how many before the model is reminded to make its change.
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
        let note_tokens = (chat.capacity() / 32).clamp(NOTE_TOKENS.0, NOTE_TOKENS.1);
        let reserve = chat.answer_tokens(NOTE_PROMPT, note_tokens)?;
        chat.reserve(reserve);
        let mut harness = Self {
            chat,
            tools: Tools::new(),
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

    /// Sets how many tool calls in a row without a file changed by `edit`
    /// or `write` the model may make before it is reminded to make its
    /// change, [`NUDGE_AFTER`] unless set.
    pub fn set_nudge_after(&mut self, calls: usize) {
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
    pub fn set_call_limit(&mut self, calls: usize) {
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
        self.steps = 0;
        self.unchanged = 0;
        self.went_on = 0;
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
    /// interrupted, or is stopped short for room; or until it has made
    /// [`MAX_CALLS`] calls in the turn, when it is asked to reply with
    /// what it has.
    fn run(
        &mut self,
        mut input: Turn,
        mut ahead: Option<String>,
        on_event: &mut impl FnMut(Event) -> ControlFlow<()>,
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
            if !self.make_room(&fed, more, on_event)? {
                return Ok(Outcome::Done);
            }
            if capped {
                let Turn::Results(outputs) = &fed else {
                    return Err("the cap is reached by tool results".into());
                };
                self.chat.feed_results(outputs)?;
                self.record(input)?;
                let outcome = self.account(on_event)?;
                if outcome == Outcome::Cut || !self.go_on() {
                    return Ok(outcome);
                }
                self.steps = 0;
                input = Turn::User(GO_ON.to_string());
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
            self.record(input)?;
            self.record(Turn::Reply {
                text,
                calls: reply.calls.clone(),
            })?;
            if reply.cut {
                return Ok(Outcome::Cut);
            }
            if reply.calls.is_empty() {
                if !self.go_on() {
                    return Ok(Outcome::Done);
                }
                input = Turn::User(GO_ON.to_string());
                continue;
            }
            let mut outputs = Vec::new();
            for call in &reply.calls {
                let mut seconds = None;
                let mut changes = false;
                self.steps += 1;
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
                        changes = matches!(call.name.as_str(), "edit" | "write");
                        let this = Some((call.name.clone(), call.arguments.to_string()));
                        if this == last {
                            REPEATED.to_string()
                        } else if call.name == "bash"
                            && call.arguments["command"]
                                .as_str()
                                .is_some_and(|command| discards(command, &self.dir))
                            && changed(&self.dir)
                        {
                            DISCARDS.to_string()
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
                if changes && !output.starts_with("error:") {
                    self.unchanged = 0;
                } else {
                    self.unchanged += 1;
                }
                outputs.push(output);
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
        }
    }

    /// Whether the model, in auto mode, is to be told to go on after the
    /// reply it just gave: not if it said it is done, nor once it has been
    /// told [`AUTO_ROUNDS`] times this turn. Counts it if so.
    fn go_on(&mut self) -> bool {
        let done = matches!(
            self.transcript.last(),
            Some((Turn::Reply { text, .. }, _)) if said_done(text)
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
    fn account(
        &mut self,
        on_event: &mut impl FnMut(Event) -> ControlFlow<()>,
    ) -> Result<Outcome, Box<dyn Error>> {
        let prompt = STEPS_PROMPT;
        self.log.user(prompt, None, self.chat.tokens());
        let before = self.chat.stats();
        let answer = self.chat.answer(prompt, self.account_limit(), |text| {
            self.log.stream("text", text);
            on_event(Event::Text(text))
        })?;
        self.log.reply(
            &[],
            false,
            answer.cut,
            self.chat.tokens(),
            &before,
            &self.chat.stats(),
        );
        self.record(Turn::User(prompt.to_string()))?;
        self.record(Turn::Reply {
            text: answer.text,
            calls: Vec::new(),
        })?;
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
    fn account_tokens(&self) -> Result<usize, Box<dyn Error>> {
        self.chat.answer_tokens(STEPS_PROMPT, self.account_limit())
    }

    /// Adds a turn to the transcript, and returns the tokens it takes.
    fn record(&mut self, turn: Turn) -> Result<usize, Box<dyn Error>> {
        let tokens = self.chat.measure(&turn)?;
        self.transcript.push((turn, tokens));
        Ok(tokens)
    }

    /// Makes room for `input`, and for `more` tokens after it, compacting
    /// the conversation first if they would take the context past the
    /// high-water mark. Returns whether the input may go in: not if the
    /// compaction was interrupted, and an error if the input cannot fit
    /// even so.
    fn make_room(
        &mut self,
        input: &Turn,
        more: usize,
        on_event: &mut impl FnMut(Event) -> ControlFlow<()>,
    ) -> Result<bool, Box<dyn Error>> {
        let needed = self.chat.measure(input)? + more;
        let capacity = self.chat.capacity();
        if self.chat.tokens() + needed > capacity * HIGH_WATER.0 / HIGH_WATER.1
            && !self.transcript.is_empty()
            && !self.compaction(on_event)?
        {
            return Ok(false);
        }
        if needed > self.chat.free() {
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
        let note = self
            .chat
            .answer(NOTE_PROMPT, self.note_tokens, |text| {
                self.log.stream("note", text);
                let flow = on_event(Event::Note(text));
                interrupted |= flow.is_break();
                flow
            })?
            .text;
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
        self.chat.replay(&kept)?;
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

/// Whether a reply says the work is done: its last line is [`DONE`], with
/// any emphasis or punctuation around the word.
fn said_done(text: &str) -> bool {
    text.lines()
        .rev()
        .find(|line| !line.trim().is_empty())
        .is_some_and(|line| line.trim_matches(|c: char| !c.is_alphanumeric()) == DONE)
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

/// The user's own messages among `turns`, oldest first, as many of the
/// most recent as fit in `budget` tokens: not the notes of earlier
/// compactions, nor the prompts to go on after a reply was cut, nor
/// those for an account of a turn.
fn messages(turns: &[(Turn, usize)], budget: usize) -> Vec<&str> {
    let mut kept = Vec::new();
    let mut total = 0;
    for (turn, tokens) in turns.iter().rev() {
        let Turn::User(message) = turn else {
            continue;
        };
        if message.contains(NOTE_HEAD)
            || [CONTINUE, STEPS_PROMPT, GO_ON].contains(&message.as_str())
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

/// The event for a piece of the model's reply.
fn event(chunk: Chunk) -> Event {
    match chunk {
        Chunk::Thought(text) => Event::Thought(text),
        Chunk::Text(text) => Event::Text(text),
        Chunk::Token => Event::Token,
    }
}

#[cfg(test)]
mod tests {
    use dwim_models::{Sampler, Tokenizer, testing::Scripted};

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
    fn reply(thought: usize, call: bool) -> String {
        let body = if call {
            "<tool_call>\n<function=bash>\n<parameter=command>\necho ok\n</parameter>\n</function>\n</tool_call>"
        } else {
            "Ok."
        };
        format!("{}\n</think>\n\n{body}", filler(thought))
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
                let name = match event {
                    Event::Thought(_) => "thought",
                    Event::Text(_) => "text",
                    Event::Token => return ControlFlow::Continue(()),
                    Event::Call { .. } => "call",
                    Event::Output(_) => "output",
                    Event::Cut => "cut",
                    Event::Compacting => "compacting",
                    Event::Note(text) => {
                        note.push_str(text);
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
        let reserve = harness
            .chat
            .answer_tokens(NOTE_PROMPT, harness.note_tokens)
            .unwrap();
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
            assert!(harness.tokens() <= max_len - reserve);
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
        let reserve = harness
            .chat
            .answer_tokens(NOTE_PROMPT, harness.note_tokens)
            .unwrap();
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
        assert!(harness.tokens() <= max_len - reserve);
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
                        thought.push_str(text);
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
                thought.push_str(text);
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
    fn harness_in(dir: &Path, scripts: &[String], path: &Path) -> Harness<Scripted> {
        harness_within(dir, scripts, path, 100_000)
    }

    /// A harness as `harness_in` makes one, with room for `max_len`
    /// tokens.
    fn harness_within(
        dir: &Path,
        scripts: &[String],
        path: &Path,
        max_len: usize,
    ) -> Harness<Scripted> {
        let at = dir.display().to_string();
        let scripts: Vec<String> = scripts.iter().map(|s| s.replace("DIR", &at)).collect();
        let tokenizer = Tokenizer::tiny();
        let scripts: Vec<&str> = scripts.iter().map(String::as_str).collect();
        let model = Scripted::new(&tokenizer, &scripts, max_len);
        let mut chat = Chat::new(model, tokenizer, Sampler::new(1.0, 1, 0.95, 0.0, 1)).unwrap();
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
    fn knows_the_commands_that_discard_changes() {
        // A checkout of a path that is there discards it, as the model ran
        // it on django-11797; one of a branch, which is not a path, doesn't.
        let dir = scratch();
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
        let dir = scratch();
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
        let scripts = [
            reply_calling(10, "bash", "command", "cd DIR && git stash"),
            reply(10, false),
        ];
        let path = dir.join("log.jsonl");
        assert!(!changed(&dir));
        fs::write(dir.join("a.txt"), "new\n").unwrap();
        assert!(changed(&dir));
        let mut harness = harness_in(&dir, &scripts, &path);
        send(&mut harness, "go");
        let turns: Vec<&Turn> = harness.transcript().collect();
        assert!(
            matches!(turns[2], Turn::Results(outputs) if outputs[0] == DISCARDS),
            "{:?}",
            turns[2]
        );
        assert_eq!(fs::read_to_string(dir.join("a.txt")).unwrap(), "new\n");

        fs::write(dir.join("a.txt"), "old\n").unwrap();
        let mut harness = harness_in(&dir, &scripts, &path);
        send(&mut harness, "go");
        let turns: Vec<&Turn> = harness.transcript().collect();
        assert!(
            matches!(turns[2], Turn::Results(outputs) if outputs[0] != DISCARDS),
            "{:?}",
            turns[2]
        );
        fs::remove_dir_all(dir).unwrap();
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
            (Turn::User(STEPS_PROMPT.to_string()), 10),
            reply(10),
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
    fn opens_the_prompt_with_the_line_for_the_effort() {
        // As Bonsai's template does: a line for xhigh and for low, ahead of
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
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        let prompt = system_prompt(dir, Effort::Medium);
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
