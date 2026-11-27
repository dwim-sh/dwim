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
//! A [`Harness`] wraps a [`Session`] and reports each turn's thoughts, text,
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
//! keeps the conversation since the system prompt, and when the next
//! message or tool result would take the context past a high-water mark,
//! `compaction` has the model write a note and starts the conversation
//! over from a briefing that holds it.
//!
//! The model lays out the work as a plan, with the `plan` tool, and the
//! tools keep it from then on: the harness reports it as it changes, for
//! the user to watch, reads it back to the model with a message that comes
//! while steps are left, until the user sets it aside, asks after the
//! steps left when the model stops short of them, and, once the conversation has grown long, starts each
//! step in a conversation cleared down to the plan.
//!
//! Given a [`Transcript`], the harness keeps the whole session in it,
//! thoughts and compactions included, for finding out afterwards where a
//! session's tokens went.

pub mod compaction;
mod snapshot;
mod transcript;

use std::{
    cmp::Reverse,
    error::Error,
    fs,
    ops::ControlFlow,
    path::{Path, PathBuf},
    process::Command,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use crate::{
    harness::snapshot::Snapshot,
    model::{Chunk, Effort, Model, Session, Turn},
    tools::{ToolCall, Tools, bash, plan::Plan, read},
};

pub use crate::harness::transcript::Transcript;

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
/// takes. Under a plan they are counted from the start of the step under
/// way: a turn of five steps took 94 calls to its fifth, and was stopped
/// in the middle of it.
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
/// a tool and without saying it is done, in a turn in which it has called
/// one, or gives its account at the call cap. On SWE-bench's django-11797 runs ended with the fix found, or made
/// and not yet checked, on a reply saying what remained, with no one to
/// answer it.
const GO_ON: &str = "Go on with the work: no one will read this until you are done, and the tools are on. Do the next thing that remains, such as making the change you have found with edit, or running your reproduction or the tests to check one you have made. Only when nothing remains, with the change made and checked, reply with what you did and end the reply with DONE on a line of its own.";

/// How a reply says, in auto mode, that the work is done: its last line.
const DONE: &str = "DONE";

/// What follows a message from the user in auto mode: that no one is
/// there, and how a reply says the work is done. Told of it only by
/// [`GO_ON`], a model that had finished was sent back to its work once in
/// nearly every session, since its first account of it could not end as
/// asked: on SWE-bench Verified Mini, in 46 of 50 sessions of Inkling 2
/// Mini, which then checked its change over again for up to six calls.
const UNATTENDED: &str = "<system-reminder>No one will read your replies until the work is done, and the tools stay on. Keep going until what the message above asks is done and checked. Then reply with what you found or did, and end that reply with DONE on a line of its own.</system-reminder>";

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

/// What the model is asked when it replies without a call while the plan
/// it has been working has steps left, once a turn: a model that finishes
/// the last step tends to reply without saying the step is done, and the
/// plan the user reads would say the work stopped short of it.
const STEPS_LEFT: &str = "The plan has steps that are not done. If the step under way is finished, call done with what it came to, and go on with the next. If you cannot go on without the user, reply with what you need from them.";

/// The system prompt's rules on planning and on the file tools, in short,
/// for a model that needs steering, which follows them from the user's
/// message. They are
/// flat orders because Inkling 2 Mini skips them otherwise: under a
/// heading, or with "if the message asks for a change" before the plan,
/// it planned in none of three sessions. The cost is a plan for some
/// plain questions.
const RULES: &str = "Before anything else, call plan with the steps you see. Read files with the read tool, not cat, sed or awk through bash. Search with the grep tool, not grep through bash. Change files with the edit tool, not a script, sed or a heredoc through bash. Use bash only to run code and tests.";

/// Most times in a turn a model that needs steering is reminded of one
/// rule: one it has gone against that often it is not going to keep.
const MOST_REMINDERS: usize = 5;

/// A rule a model that needs steering is reminded of in the result of the
/// call that went against it, which is where it is deciding what to do
/// next: the same words in the system prompt are far from there.
#[derive(Clone, Copy)]
enum Rule {
    Read,
    Search,
    Edit,
    Plan,
}

impl Rule {
    /// What the model is told, of a call that changed the files `changed`.
    fn reminder(self, changed: &[String]) -> String {
        match self {
            Rule::Read => {
                "You read a file through bash. Read files with the read tool.".to_string()
            }
            Rule::Search => {
                "You searched through bash. Search with the grep and glob tools.".to_string()
            }
            Rule::Edit => format!(
                "You changed {} through bash. Change files with the edit tool.",
                changed.join(", ")
            ),
            Rule::Plan => {
                "You changed a file with no plan laid out. Call plan with the steps you see."
                    .to_string()
            }
        }
    }
}

/// The share of the context past which a step of the plan starts in a
/// conversation cleared down to the plan: half. A step's end is where the
/// least is lost by clearing, since what the step came to is in the plan,
/// and the compaction that would otherwise come at the high-water mark
/// comes mid-step and costs a note. The share is not measured.
const FRESH: usize = 2;

/// Most tokens of the account of a turn that reached [`MAX_CALLS`], in a
/// context with room for it. A small context allows a sixteenth of itself.
const ACCOUNT_TOKENS: usize = 1024;

/// How the agent should behave, as `model` is told it: each family of
/// models has its own words.
fn instructions(model: Model) -> String {
    match model {
        Model::InklingMini | Model::InklingSmall => INKLING.to_string(),
        Model::Qwen27b | Model::QwenMax => qwen(),
    }
}

/// How the agent should behave, as the Inkling 2 models are told it: a
/// few short orders, with no conditions and no exceptions. Told at the
/// length the Qwen models are, Inkling 2 Mini laid out a plan in 4 trials
/// of 12 and changed the file with edit in none; told so, in 8 and in 4.
/// A bug reported as a question it explained and left in 10 trials of 12,
/// where the longer words said both to treat a question as a task and
/// that a question needs no plan; told to fix it even so, it left none.
const INKLING: &str = r#"You are `dwim`, a coding agent working in the user's project directory at a Unix command line. Your tools are bash, read, edit, write, grep, glob, plan and done. Use them without asking.

- Before you change anything, call plan with the steps you see. Call done as each step is finished.
- Read files with read, not cat.
- Search with grep and glob, not find, grep or ls through bash.
- Change files with edit, not sed or a script. Create files with write, not a heredoc.
- Use bash to run code, tests and other commands.
- Find out with a command before you answer. Don't answer from memory.
- If the user reports a bug, find the cause, fix it with edit, and check it, even when they ask it as a question.
- Check a change by running the project's own tests.
- If a command fails, read the error and try another way.
- Don't ask the user to choose. Make the choice a careful engineer would, and give it in the plan's `assumes`.
- Follow a `<system-reminder>` at the end of a tool's result: it is from the harness.
- Keep going until the request is done, then reply in a few sentences with what you found or did."#;

/// How the agent should behave, as the Qwen models are told it, with the
/// tools' own limits in it.
fn qwen() -> String {
    format!(
        r#"You are `dwim`, a coding agent working in the user's project directory at a Unix command line. You have a bash tool that runs shell commands there, a read tool that reads files, an edit tool that replaces a string in a file and shows the lines as they now stand, a write tool that writes a file whole, grep and glob tools that search files for text and for names, and plan and done tools that lay out the plan of the work for the user and finish a step of it, and you may use them at any time without asking.

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
- For any request that changes something, call plan first: before reading more than a file or two, lay out the steps as you see them now, starting with what you need to find out, such as `find where fetch() retries`. The user watches the plan to see where you are headed, and a plan that is roughly right now is worth more to them than one that is exact later. Call plan again when what you learn changes the steps to come. Then take the steps in order, and call done as each is finished, which tells you the next. Only a question needs no plan.
- Don't stop to ask the user to choose. Where the request leaves a choice open, make the choice a careful engineer would, give it in the plan's `assumes`, such as a default, a limit, or which cases you cover, and go on: the user says if they want otherwise.
- When a message from the user comes while the plan has steps left, the plan as it stands follows the message. If the work goes on, call plan first, with the steps to come as they now should be, the same ones if the message changes none, and go on from there; if it only asks something, answer it and leave the plan as it is.
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

/// The system prompt for an agent on `model` working in `dir`: the
/// template's line for `effort`, if the prompt is to carry one and the
/// effort has one, the tools, then how to behave as the model is told
/// it, and the project's own instructions
/// from its `AGENTS.md` if it has one. Where the agent is goes in [`environment`]
/// instead, so that the prompt is the same from one run to the next.
pub fn system_prompt(dir: &Path, model: Model, effort: Option<Effort>) -> String {
    let mut prompt = String::new();
    if let Some(line) = effort.and_then(Effort::instruction) {
        prompt.push_str(line);
        prompt.push_str("\n\n");
    }
    prompt.push_str(&tools());
    prompt.push_str("\n\n");
    prompt.push_str(&instructions(model));
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

/// The rule a shell command goes against by what it starts with: reading
/// a file with `cat` and its like, or searching with `grep` or `find`,
/// where there is a tool for each. A command that pipes into one of them
/// is running something else.
fn habit(command: &str) -> Option<Rule> {
    let mut words = command.split_whitespace();
    let mut first = words.next()?;
    // Past a `cd` to where the command is to run.
    if first == "cd" {
        first = words.skip_while(|word| *word != "&&").nth(1)?;
    }
    match first {
        "cat" | "head" | "tail" | "nl" | "less" | "more" => Some(Rule::Read),
        "sed" if command.contains(" -n ") => Some(Rule::Read),
        "grep" | "rg" | "ag" | "find" => Some(Rule::Search),
        _ => None,
    }
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
    /// What the command a call of `bash` runs has printed so far, while it
    /// runs: a few times a second while it prints.
    Progress(bash::Progress),
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
    /// The plan of the work, as it now stands: laid out, changed, a step
    /// of it finished, or what is known of a step added to. Empty once the
    /// work it was the plan of is over.
    Plan(Plan),
}

/// What a turn's events go to, which ends the turn early by breaking.
type OnEvent<'a> = &'a mut dyn FnMut(Event) -> ControlFlow<()>;

/// What opens the conversation over with the system prompt, once the session
/// is cleared: from a saved state, or by reading the prompt again.
type Restart<S> = Box<dyn FnMut(&mut S) -> Result<(), Box<dyn Error>>>;

/// How a turn ended: with a reply of text alone, or interrupted; or with
/// the reply stopped short for room.
#[derive(PartialEq)]
enum Outcome {
    Done,
    Cut,
}

/// A turn as the conversation keeps it: the turn, the tokens it takes read
/// back, and whether a user turn is the harness's own prompt to the model
/// rather than the user's message.
#[derive(Clone)]
pub struct Entry {
    pub turn: Turn,
    pub tokens: usize,
    pub prompt: bool,
}

/// The loop around a [`Session`] that runs the tools the model calls
/// and feeds the results back, until the model replies with text alone.
pub struct Harness<S: Session> {
    session: S,
    tools: Tools,
    /// Where the agent is, until the first message carries it.
    environment: Option<String>,
    /// The directory the agent works in, to say where it is afresh after
    /// a compaction.
    dir: PathBuf,
    /// The model the conversation is on, once the harness is told.
    model: Option<Model>,
    calls: usize,
    tool_seconds: f64,
    /// The conversation since the system prompt, or since it was last
    /// compacted.
    conversation: Vec<Entry>,
    /// Opens the conversation over with the system prompt, once the session
    /// is cleared.
    restart: Restart<S>,
    /// Most tokens of a note.
    note_tokens: usize,
    /// The thinking budget: most tokens a thought may run to, if they
    /// are limited.
    thinking_budget: Option<usize>,
    /// Tool calls made in the turn so far, and the most there may be
    /// before the model is asked to reply.
    steps: usize,
    max_calls: usize,
    /// Tool calls in a row without one that changed a file, and how many
    /// before the model is reminded to make its change.
    unchanged: usize,
    nudge_after: usize,
    /// How many times this turn the model has been reminded of each
    /// [`Rule`].
    reminded: [usize; 4],
    /// Auto mode: whether a reply with no call that doesn't say it is
    /// done, or the account at the call cap, is answered with word to go
    /// on, as when no one is there to answer it, and how many times it has
    /// been this turn.
    auto: bool,
    went_on: usize,
    /// Whether the model has called a tool this turn: a turn without a
    /// call has no work to go on with.
    called: bool,
    /// The plan as last reported, to report it when it changes.
    shown: Plan,
    /// Whether the model has laid out the plan or finished a step of it
    /// this turn, and whether it has been asked after the steps left.
    planned: bool,
    asked: bool,
    /// The user's messages of the session, as the conversation has recorded
    /// them, which a compacted conversation reads back.
    said: Vec<Entry>,
    /// The transcript of the whole session.
    transcript: Transcript,
}

/// Where a conversation's time has gone: the model's, by what it was
/// doing, and the tools'.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Stats {
    pub model: crate::model::Stats,
    /// Tool calls run, and the time they took.
    pub calls: usize,
    pub tool_seconds: f64,
}

impl<S: Session> Harness<S> {
    /// An agent on `session`, which has read the system prompt, working in
    /// `dir`, with `restart` to open the conversation over with the same
    /// prompt once the session is cleared, when the conversation is compacted.
    pub fn new(
        mut session: S,
        dir: &Path,
        restart: impl FnMut(&mut S) -> Result<(), Box<dyn Error>> + 'static,
    ) -> Result<Self, Box<dyn Error>> {
        let note_tokens = compaction::reserve(&mut session)?;
        let mut harness = Self {
            session,
            tools: Tools::new(dir),
            environment: Some(environment(dir)),
            dir: dir.to_path_buf(),
            model: None,
            calls: 0,
            tool_seconds: 0.0,
            conversation: Vec::new(),
            restart: Box::new(restart),
            note_tokens,
            thinking_budget: None,
            steps: 0,
            max_calls: MAX_CALLS,
            unchanged: 0,
            reminded: [0; 4],
            nudge_after: NUDGE_AFTER,
            auto: false,
            went_on: 0,
            called: false,
            shown: Plan::default(),
            planned: false,
            asked: false,
            said: Vec::new(),
            transcript: Transcript::none(),
        };
        harness.limit_thoughts(THINKING_BUDGET);
        Ok(harness)
    }

    /// Limits a thought to `tokens`, or lifts the limit: a thought that
    /// runs to the budget is ended with a closing in the model's voice,
    /// and the model goes on to act on what it has thought so far.
    pub fn set_thinking_budget(&mut self, tokens: Option<usize>) {
        self.limit_thoughts(tokens);
        self.transcript.thinking_budget(tokens);
    }

    /// Has the session end a thought at `tokens`, the thinking budget, if
    /// there is one, with the closing for where the turn is: to check with
    /// a tool before the model has called one in the turn, when it is
    /// thinking from memory, and to decide from the results once it has.
    fn limit_thoughts(&mut self, tokens: Option<usize>) {
        let closing = if self.steps == 0 {
            RECALL_CLOSING
        } else {
            DECIDE_CLOSING
        };
        self.thinking_budget = tokens;
        self.session.limit_thoughts(tokens, closing);
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
    /// answer it: in it, a message is followed by word that no one is
    /// there and that the work ends with DONE, and a reply with no call
    /// that doesn't end with DONE,
    /// and the account it gives at the call cap, are answered with word to
    /// go on, with the tools on, up to [`AUTO_ROUNDS`] times a turn. A
    /// turn in which the model has called no tool ends at its reply, as
    /// the answer to a question does. It is off unless set.
    pub fn set_auto(&mut self, auto: bool) {
        self.auto = auto;
    }

    /// Sets the plan of the work aside, as the user does who has moved on
    /// from it with steps left: the model is no longer told of it.
    pub fn set_plan_aside(&mut self) {
        self.tools.clear_plan();
        self.shown = Plan::default();
        self.transcript.plan("");
    }

    /// Sets how many tool calls the model may make in one turn before it
    /// is asked to reply with what it has, [`MAX_CALLS`] unless set.
    #[cfg(test)]
    fn set_call_limit(&mut self, calls: usize) {
        self.max_calls = calls;
    }

    /// Keeps the session in `transcript`, starting with what the
    /// model was given before the first message: `model`, the `device` it
    /// runs on, where it is, the room it has, how it samples, its thinking
    /// budget, the `effort` it is asked to reason at, and the `system`
    /// prompt.
    pub fn keep_transcript(
        &mut self,
        mut transcript: Transcript,
        model: &str,
        device: &str,
        effort: Effort,
        system: &str,
    ) {
        transcript.start(
            model,
            device,
            &self.dir,
            self.session.capacity(),
            self.session.sampling(),
            self.thinking_budget,
            effort,
            system,
        );
        self.transcript = transcript;
    }

    /// Tells the harness which model the conversation is on.
    pub fn set_model(&mut self, model: Model) {
        self.model = Some(model);
    }

    /// What a model that needs steering is reminded of in the result of
    /// `call`: the rules it went against, each at most [`MOST_REMINDERS`]
    /// times a turn. `edited` are the files that were there before the
    /// call and that it changed, and `unplanned` is whether it changed a
    /// file with no plan laid out. A git command puts files back as they
    /// were or brings in another's, which is not a change to make with
    /// edit.
    fn reminder(&mut self, call: &ToolCall, edited: &[String], unplanned: bool) -> Option<String> {
        if !self.model.is_some_and(Model::needs_steering) {
            return None;
        }
        let command = (call.name == "bash")
            .then(|| call.arguments.get("command")?.as_str())
            .flatten();
        let mut rules = Vec::new();
        match command {
            Some(command) if !edited.is_empty() && !command.contains("git ") => {
                rules.push(Rule::Edit);
            }
            Some(command) => rules.extend(habit(command)),
            None => {}
        }
        if unplanned {
            rules.push(Rule::Plan);
        }
        rules.retain(|rule| {
            let count = &mut self.reminded[*rule as usize];
            *count += 1;
            *count <= MOST_REMINDERS
        });
        if rules.is_empty() {
            return None;
        }
        let said: Vec<String> = rules.iter().map(|rule| rule.reminder(edited)).collect();
        Some(format!(
            "<system-reminder>{}</system-reminder>",
            said.join(" ")
        ))
    }

    /// What goes ahead of the user's first message, and again after a
    /// compaction: the environment, then the rules if the model needs
    /// steering.
    fn preamble(&self) -> String {
        let environment = environment(&self.dir);
        if self.model.is_some_and(Model::needs_steering) {
            format!("{environment}\n\n{RULES}")
        } else {
            environment
        }
    }

    /// Records in the transcript that the conversation goes on with
    /// `model`, on `device`, as it does after [`Harness::reopen`] on a
    /// session with another model.
    pub fn note_model(&mut self, model: &str, device: &str) {
        self.transcript
            .model(model, device, self.session.capacity());
    }

    /// The conversation the harness runs on.
    pub fn session(&mut self) -> &mut S {
        &mut self.session
    }

    /// Goes on with the conversation on `session`, a conversation opened
    /// afresh with the system prompt, in place of the one it was on: reads
    /// the conversation so far back into it, with where the agent is ahead
    /// of the first message if that went ahead of it when it was sent.
    pub fn reopen(&mut self, session: S) -> Result<(), Box<dyn Error>> {
        self.session = session;
        self.note_tokens = compaction::reserve(&mut self.session)?;
        let mut turns: Vec<Turn> = self
            .conversation
            .iter()
            .map(|entry| entry.turn.clone())
            .collect();
        if self.environment.is_none()
            && let (Some(Turn::User(message)), Some(first)) =
                (turns.first_mut(), self.conversation.first())
            && !first.prompt
        {
            *message = format!("{}\n\n{message}", self.preamble());
        }
        self.session.replay(&turns)?;
        Ok(())
    }

    /// Where the conversation's time has gone so far.
    pub fn stats(&self) -> Stats {
        Stats {
            model: self.session.stats(),
            calls: self.calls,
            tool_seconds: self.tool_seconds,
        }
    }

    /// The turns since the system prompt, or since the conversation was
    /// last compacted.
    #[cfg(test)]
    fn conversation(&self) -> impl Iterator<Item = &Turn> {
        self.conversation.iter().map(|entry| &entry.turn)
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

    /// Runs `work`, reporting to `on_event`, and records in the transcript
    /// how it ended, if not on its own: with an error, or interrupted.
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
        self.transcript
            .end(result.as_ref().err().map(|e| e as _), interrupted);
        result
    }

    /// Sends a message from the user, as [`Harness::send`] does.
    fn converse(&mut self, message: &str, on_event: OnEvent) -> Result<(), Box<dyn Error>> {
        let ahead = self.environment.take().map(|_| self.preamble());
        self.steps = 0;
        self.unchanged = 0;
        self.reminded = [0; 4];
        self.went_on = 0;
        self.called = false;
        self.planned = false;
        self.asked = false;
        let after = self.plan_so_far(on_event);
        let after = match (after, self.auto) {
            (Some(plan), true) => Some(format!("{plan}\n\n{UNATTENDED}")),
            (None, true) => Some(UNATTENDED.to_string()),
            (after, false) => after,
        };
        let mut outcome = self.run(
            Turn::User(message.to_string()),
            false,
            ahead,
            after,
            on_event,
        )?;
        if outcome == Outcome::Cut {
            if on_event(Event::Cut).is_break() {
                return Ok(());
            }
            if compaction::compact(self, on_event)? {
                outcome = self.run(
                    Turn::User(compaction::CONTINUE.to_string()),
                    true,
                    None,
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

    /// What follows a message from the user of the plan, if one is there:
    /// the plan as it stands, if it has steps left. One with every step
    /// done was the plan of work that is over, and is forgotten.
    fn plan_so_far(&mut self, on_event: OnEvent) -> Option<String> {
        if self.tools.plan().finished() {
            self.tools.clear_plan();
            self.show_plan(on_event);
        }
        let plan = self.tools.plan();
        (!plan.is_empty()).then(|| {
            format!(
                "<system-reminder>The plan of the work has steps left. As it stands:\n\n{}\n\nDo what the message above asks. If the work goes on, call plan first, before any other tool, with the steps to come as they now should be: the same steps if the message changes none of them. Until you do, the user is shown the plan as from before their message. If the message only asks something, such as a question, answer that and leave the plan as it is.</system-reminder>",
                plan.brief()
            )
        })
    }

    /// Reports the plan to `on_event`, if it has changed since it was last
    /// reported, and returns whether it had.
    fn show_plan(&mut self, on_event: OnEvent) -> bool {
        let changed = *self.tools.plan() != self.shown;
        if changed {
            let brief = self.tools.plan().brief();
            if brief != self.shown.brief() {
                self.transcript.plan(&brief);
            }
            self.shown = self.tools.plan().clone();
            let _ = on_event(Event::Plan(self.shown.clone()));
        }
        changed
    }

    /// Feeds `input`, a message or tool results, with `ahead` before it if
    /// there is anything, and runs the tools the reply calls, feeding their
    /// results back, until the model replies with text alone, is
    /// interrupted, or is stopped short for room; or until it has made
    /// [`MAX_CALLS`] calls in the turn, when it is asked to reply with
    /// what it has. A message that is the harness's own `prompt` rather
    /// than the user's is kept as such. A message is followed by `after`,
    /// if there is any, as fed.
    fn run(
        &mut self,
        mut input: Turn,
        mut prompt: bool,
        mut ahead: Option<String>,
        mut after: Option<String>,
        on_event: OnEvent,
    ) -> Result<Outcome, Box<dyn Error>> {
        let mut last = None;
        loop {
            if let Turn::User(message) = &input {
                self.transcript
                    .user(message, ahead.as_deref(), self.session.tokens());
            }
            // Where the agent is goes ahead of the first message as fed,
            // and the plan as it stands after one that comes while steps
            // are left; the conversation keeps the message alone, since a
            // compaction says where the agent is, and the plan, afresh.
            let fed = match (&input, ahead.take(), after.take()) {
                (Turn::User(message), ahead, after) if ahead.is_some() || after.is_some() => {
                    if let Some(after) = &after {
                        self.transcript.nudge(after);
                    }
                    let parts = [ahead.as_deref(), Some(message.as_str()), after.as_deref()];
                    Turn::User(parts.into_iter().flatten().collect::<Vec<_>>().join("\n\n"))
                }
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
            // Results go in whatever they take, since they answer the
            // reply before them: if they do not fit even after the
            // compaction, the largest outputs give way to word of how
            // large they were. A message that does not fit cannot go in.
            let fed = match fed {
                Turn::Results(outputs) => {
                    let outputs = self.fit(outputs, more)?;
                    input = Turn::Results(outputs.clone());
                    Turn::Results(outputs)
                }
                fed => {
                    let needed = self.session.measure(&fed)? + more;
                    if needed > self.session.free() {
                        return Err(format!(
                            "a message of {needed} tokens does not fit in the context window"
                        )
                        .into());
                    }
                    fed
                }
            };
            if capped {
                let Turn::Results(outputs) = &fed else {
                    return Err("the cap is reached by tool results".into());
                };
                self.session.feed_results(outputs)?;
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
            self.limit_thoughts(self.thinking_budget);
            let mut text = String::new();
            let mut stopped = false;
            let before = self.session.stats();
            let reply = {
                let collect = |chunk: Chunk| {
                    match chunk {
                        Chunk::Thought(piece) => self.transcript.stream("thought", piece),
                        Chunk::Text(piece) => {
                            self.transcript.stream("text", piece);
                            text.push_str(piece);
                        }
                        Chunk::Token => {}
                    }
                    let flow = on_event(event(chunk));
                    stopped |= flow.is_break();
                    flow
                };
                match &fed {
                    Turn::User(message) => self.session.send(message, collect)?,
                    Turn::Results(outputs) => self.session.respond(outputs, collect)?,
                    Turn::Reply { .. } => return Err("a reply is the model's to make".into()),
                }
            };
            self.transcript.reply(
                &reply,
                self.session.tokens(),
                &before,
                &self.session.stats(),
            );
            self.record(input, prompt)?;
            self.record(
                Turn::Reply {
                    text,
                    calls: reply.calls.clone(),
                    thought: String::new(),
                },
                false,
            )?;
            if reply.cut {
                return Ok(Outcome::Cut);
            }
            if reply.calls.is_empty() {
                if self.go_on() {
                    input = Turn::User(GO_ON.to_string());
                } else if !stopped
                    && self.planned
                    && !self.asked
                    && self.tools.plan().under_way().is_some()
                {
                    self.asked = true;
                    input = Turn::User(STEPS_LEFT.to_string());
                } else {
                    return Ok(Outcome::Done);
                }
                prompt = true;
                continue;
            }
            // Every call of the reply gets a result, as the model expects
            // of a reply that calls: once the turn is interrupted, the
            // calls that remain get word that they were not run.
            let mut outputs = Vec::new();
            let mut interrupted = false;
            let done = self.tools.plan().done();
            for call in &reply.calls {
                let mut seconds = None;
                let mut changed = false;
                let mut reminder = None;
                self.steps += 1;
                self.called = true;
                let output = match ToolCall::parse(call) {
                    Ok(call) => {
                        let detail = crate::tools::describe(&call);
                        self.transcript.call(&call.name, &detail);
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
                            // What the call changed is what differs in
                            // the files once it has run, whichever tool
                            // it is.
                            let before = self.tools.changes(&call).then(|| Snapshot::of(&self.dir));
                            let unplanned = self.tools.plan().is_empty();
                            // What a command prints is shown as it runs.
                            let output = self.tools.run_with(
                                &call,
                                Some(&mut |progress| {
                                    let _ = on_event(Event::Progress(progress.clone()));
                                }),
                            );
                            if let Some(before) = before {
                                let paths = Snapshot::of(&self.dir).changed(&before);
                                for path in &paths {
                                    self.tools.changed(path);
                                    changed = true;
                                }
                                let edited: Vec<String> =
                                    paths.into_iter().filter(|path| before.has(path)).collect();
                                reminder = self.reminder(&call, &edited, changed && unplanned);
                            }
                            self.calls += 1;
                            let elapsed = start.elapsed().as_secs_f64();
                            self.tool_seconds += elapsed;
                            seconds = Some(elapsed);
                            self.planned |= self.show_plan(on_event);
                            output
                        }
                    }
                    Err(_) if interrupted => NOT_RUN.to_string(),
                    Err(e) => format!("error: malformed tool call: {e}"),
                };
                self.transcript.output(&output, seconds);
                interrupted = interrupted || on_event(Event::Output(output.clone())).is_break();
                let mut output = output;
                if let Some(reminder) = reminder {
                    output.push_str("\n\n");
                    output.push_str(&reminder);
                    self.transcript.nudge(&reminder);
                }
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
            // The step that starts once one is finished starts in a
            // conversation cleared down to the plan, if this one has
            // grown long.
            // A step finished is the sign that the turn has not lost its
            // way, which the cap on a turn's calls is there to catch: the
            // calls are counted from the step's start. The step that
            // starts then starts in a conversation cleared down to the
            // plan, if this one has grown long.
            if self.tools.plan().done() > done {
                self.steps = 0;
                if self.tools.plan().under_way().is_some()
                    && self.session.tokens() > self.session.capacity() / FRESH
                {
                    compaction::start_over(self, on_event)?;
                }
            }
            if self.unchanged >= self.nudge_after
                && let Some(last) = outputs.last_mut()
            {
                last.push_str("\n\n");
                last.push_str(NUDGE);
                self.transcript.nudge(NUDGE);
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
        if self.session.measure(&turn)? > self.session.free() {
            for output in outputs.iter_mut().filter(|output| *output != NOT_RUN) {
                *output = DROPPED.to_string();
            }
            turn = Turn::Results(outputs.clone());
        }
        self.session.feed_results(&outputs)?;
        self.record(turn, false)
    }

    /// `outputs` as they fit in the context with `more` tokens after them:
    /// as they are, if they fit; otherwise with the largest replaced by
    /// word of how many tokens it took, then the next largest, one at a
    /// time, until they fit. Fails if they do not fit even with every
    /// output replaced.
    fn fit(
        &mut self,
        mut outputs: Vec<String>,
        more: usize,
    ) -> Result<Vec<String>, Box<dyn Error>> {
        let mut sizes = Vec::with_capacity(outputs.len());
        for output in &outputs {
            sizes.push(Some(
                self.session.measure(&Turn::Results(vec![output.clone()]))?,
            ));
        }
        loop {
            let needed = self.session.measure(&Turn::Results(outputs.clone()))? + more;
            if needed <= self.session.free() {
                return Ok(outputs);
            }
            // The largest output not yet replaced, the first of those as
            // large.
            let Some(i) = (0..outputs.len())
                .filter(|&i| sizes[i].is_some())
                .max_by_key(|&i| (sizes[i], Reverse(i)))
            else {
                return Err(
                    format!("results of {needed} tokens do not fit in the context window").into(),
                );
            };
            outputs[i] = too_large(sizes[i].take().unwrap_or(0));
            self.transcript.nudge(&outputs[i]);
        }
    }

    /// Whether the model, in auto mode, is to be told to go on after the
    /// reply it just gave: not if it said it is done, nor if it has called
    /// no tool this turn, which is how it answers a question and leaves
    /// no work to go on with, nor once it has been told [`AUTO_ROUNDS`]
    /// times this turn. Counts it if so.
    fn go_on(&mut self) -> bool {
        let done = matches!(
            self.conversation.last(),
            Some(Entry { turn: Turn::Reply { text, .. }, .. }) if said_done(text)
        );
        if !self.auto || done || !self.called || self.went_on >= AUTO_ROUNDS {
            return false;
        }
        self.went_on += 1;
        true
    }

    /// Asks the model, once it has made [`MAX_CALLS`] calls in a turn, for
    /// an account of the turn, which it gives without tools, reported as
    /// `Text` pieces, and records the exchange. The turn ends with it.
    fn account(&mut self, on_event: OnEvent) -> Result<Outcome, Box<dyn Error>> {
        self.transcript
            .user(STEPS_PROMPT, None, self.session.tokens());
        let before = self.session.stats();
        let answer = self
            .session
            .answer(STEPS_PROMPT, &[], self.account_limit(), |text| {
                self.transcript.stream("text", text);
                on_event(Event::Text(text.to_string()))
            })?;
        self.transcript.reply(
            &crate::model::Reply {
                cut: answer.cut,
                ..Default::default()
            },
            self.session.tokens(),
            &before,
            &self.session.stats(),
        );
        self.record(Turn::User(STEPS_PROMPT.to_string()), true)?;
        self.record(
            Turn::Reply {
                text: answer.text,
                calls: Vec::new(),
                thought: String::new(),
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
        ACCOUNT_TOKENS.min(self.session.capacity() / 16)
    }

    /// Most tokens the account of a turn takes, with the prompt for it.
    fn account_tokens(&mut self) -> Result<usize, Box<dyn Error>> {
        self.session
            .answer_tokens(STEPS_PROMPT, self.account_limit())
    }

    /// Adds a turn to the conversation, a user turn as the harness's own
    /// `prompt` or the user's message; the user's are kept among what they
    /// said too.
    fn record(&mut self, turn: Turn, prompt: bool) -> Result<(), Box<dyn Error>> {
        let tokens = self.session.measure(&turn)?;
        let entry = Entry {
            turn,
            tokens,
            prompt,
        };
        if matches!(entry.turn, Turn::User(_)) && !prompt {
            self.said.push(entry.clone());
        }
        self.conversation.push(entry);
        Ok(())
    }
}

impl<S: Session> Drop for Harness<S> {
    /// Ends the transcript: the session finished.
    fn drop(&mut self) {
        let stats = self.stats();
        self.transcript.finish(self.session.tokens(), &stats);
    }
}

/// What the model gets back in place of an output of `tokens` tokens
/// that does not fit in the context window.
fn too_large(tokens: usize) -> String {
    format!(
        "not shown: this output took {tokens} tokens, more than there is room for in the context window. Ask for less of it at a time, such as a range of lines or a filtered command."
    )
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
    use std::{iter, sync::Arc};

    use crate::{
        inference::{Local, Sampler, testing::Scripted},
        model::{Tokenizer, qwen::ChatMl},
    };

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
    pub fn harness(scripts: &[String], max_len: usize, note: &str) -> Harness<Local<Scripted>> {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        Harness::new(session(scripts, max_len, note), dir, |session| {
            session.system("Be brief.", |_, _| {})
        })
        .unwrap()
    }

    /// A conversation as `harness` has it, opened with its system prompt.
    fn session(scripts: &[String], max_len: usize, note: &str) -> Local<Scripted> {
        let tokenizer = Tokenizer::tiny();
        let scripts: Vec<&str> = scripts.iter().map(String::as_str).collect();
        let model = Scripted::new(&tokenizer, &scripts, max_len).answers(&tokenizer, note);
        let mut session = Local::new(
            model,
            Arc::new(ChatMl::new(tokenizer).unwrap()),
            Sampler::new(0.0, 1, 1.0, 0.0, 1),
        )
        .unwrap();
        session.system("Be brief.", |_, _| {}).unwrap();
        session
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
        let tokens = harness.session.tokens();
        harness.reopen(session(&[], 8000, "")).unwrap();
        assert_eq!(harness.session.tokens(), tokens);
    }

    #[test]
    fn a_reopened_conversation_keeps_its_budget_and_its_reserve() {
        // The conversation it is reopened on is new, and knows neither the
        // thinking budget nor the room kept for a compaction's note: a
        // thought of 1000 tokens is ended at the budget of 100 all the
        // same, and as much room is free as was.
        let mut harness = harness(&[], 8000, "");
        harness.set_thinking_budget(Some(100));
        let free = harness.session.free();
        harness
            .reopen(session(&[reply(1000, false)], 8000, ""))
            .unwrap();
        assert_eq!(harness.session.free(), free);
        let mut thought = String::new();
        harness
            .send("go", |event| {
                if let Event::Thought(text) = event {
                    thought.push_str(&text);
                }
                ControlFlow::Continue(())
            })
            .unwrap();
        assert!(thought.contains(RECALL_CLOSING), "{thought}");
        assert!(thought.len() < 1000, "{}", thought.len());
    }

    /// The records of the transcript at `path`, which is removed.
    fn records(path: &Path) -> Vec<serde_json::Value> {
        let text = fs::read_to_string(path).unwrap();
        fs::remove_file(path).unwrap();
        text.lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }

    /// The names of the events of a turn, and the note it wrote.
    pub fn send(
        harness: &mut Harness<Local<Scripted>>,
        message: &str,
    ) -> (Vec<&'static str>, String) {
        let mut events = Vec::new();
        let mut note = String::new();
        harness
            .send(message, |event| {
                let name = match event {
                    Event::Thought(_) => "thought",
                    Event::Text(_) => "text",
                    Event::Token | Event::Progress(_) => return ControlFlow::Continue(()),
                    Event::Call { .. } => "call",
                    Event::Output(_) => "output",
                    Event::Cut => "cut",
                    Event::Compacting => "compacting",
                    Event::Note(text) => {
                        note.push_str(&text);
                        "note"
                    }
                    Event::Compacted { .. } => "compacted",
                    Event::Plan(_) => "plan",
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
        assert_eq!(limited.thinking_budget, THINKING_BUDGET);
        assert_eq!(THINKING_BUDGET, Some(2048));
        let path = std::env::temp_dir().join(format!("dwim-budget-{}.jsonl", std::process::id()));
        limited.keep_transcript(
            Transcript::create(&path).unwrap(),
            "tiny",
            "cpu",
            Effort::Low,
            "Be brief.",
        );
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
                    Event::Token | Event::Progress(_) => return ControlFlow::Continue(()),
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
        let turns: Vec<&Turn> = limited.conversation().collect();
        assert!(
            matches!(turns.last(), Some(Turn::Reply { text, .. }) if text.trim().ends_with("Ok.")),
            "{:?}",
            turns.last()
        );
        drop(limited);
        // The transcript has the budget, its change, and the thought it ended.
        let records = records(&path);
        assert_eq!(records[0]["thinking_budget"], 2048);
        assert_eq!(records[0]["reasoning_effort"], "low");
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
        assert_eq!(free.thinking_budget, None);
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
    /// of 100, keeping a transcript at `path`. `DIR` in a script stands for the
    /// directory.
    fn harness_in(dir: &Path, scripts: &[String], path: &Path) -> Harness<Local<Scripted>> {
        harness_within(dir, scripts, path, 100_000)
    }

    /// A harness as `harness_in` makes one, with room for `max_len`
    /// tokens.
    fn harness_within(
        dir: &Path,
        scripts: &[String],
        path: &Path,
        max_len: usize,
    ) -> Harness<Local<Scripted>> {
        let at = dir.display().to_string();
        let scripts: Vec<String> = scripts.iter().map(|s| s.replace("DIR", &at)).collect();
        let tokenizer = Tokenizer::tiny();
        let scripts: Vec<&str> = scripts.iter().map(String::as_str).collect();
        let model = Scripted::new(&tokenizer, &scripts, max_len);
        let mut session = Local::new(
            model,
            Arc::new(ChatMl::new(tokenizer).unwrap()),
            Sampler::new(1.0, 1, 0.95, 0.0, 1),
        )
        .unwrap();
        session.system("Be brief.", |_, _| {}).unwrap();
        let mut harness = Harness::new(session, dir, |session| {
            session.system("Be brief.", |_, _| {})
        })
        .unwrap();
        harness.set_thinking_budget(Some(100));
        harness.keep_transcript(
            Transcript::create(path).unwrap(),
            "tiny",
            "cpu",
            Effort::Low,
            "Be brief.",
        );
        harness
    }

    /// The replies of the transcript at `path`, as thought tokens and
    /// whether the thought was ended at the budget.
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
        // that fails: every reply thinks, at the sampling the session was
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
        let temperature = harness.session.sampling().temperature;
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
        assert_eq!(harness.session.sampling().temperature, temperature);
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
    /// What the model of `harness` has been fed, as text.
    fn fed(harness: &Harness<Local<Scripted>>) -> String {
        let tokenizer = Tokenizer::tiny();
        let fed: Vec<u8> = harness
            .session
            .model()
            .fed
            .iter()
            .flat_map(|&token| tokenizer.decode(token).to_vec())
            .collect();
        String::from_utf8(fed).unwrap()
    }

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
        harness: &mut Harness<Local<Scripted>>,
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

    /// The conversation's turns as a letter each: a user's, a reply, results.
    fn shape(harness: &Harness<Local<Scripted>>) -> String {
        harness
            .conversation()
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
        // the model was fed what the conversation has.
        assert_eq!(shape(&harness), "urtur");
        let turns: Vec<&Turn> = harness.conversation().collect();
        let Turn::Results(outputs) = turns[2] else {
            panic!("{:?}", turns[2])
        };
        assert_eq!(outputs.len(), 2, "{outputs:?}");
        assert!(
            outputs.iter().all(|output| output == NOT_RUN),
            "{outputs:?}"
        );
        let fed = fed(&harness);
        let results = fed.find("<tool_response>").expect("no results were fed");
        assert_eq!(fed.matches("<tool_response>").count(), 2, "{fed}");
        assert!(results < fed.find("what happened?").unwrap(), "{fed}");
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn answers_a_call_whose_output_is_larger_than_the_context() {
        // A call prints more than the whole context holds. Its result is
        // recorded all the same, with word of how large the output was in
        // its place, and the model replies to that: the call is answered,
        // and the turn goes on.
        let scripts = [
            calling(
                "bash",
                &[("command", "head -c 6000 /dev/zero | tr '\\0' a")],
            ),
            reply(10, false),
        ];
        let mut harness = harness(&scripts, 4000, " go on.");
        send(&mut harness, "go");
        assert_eq!(shape(&harness), "urtr");
        let turns: Vec<&Turn> = harness.conversation().collect();
        let Turn::Results(outputs) = turns[2] else {
            panic!("{:?}", turns[2])
        };
        assert!(
            outputs[0].starts_with("not shown: this output took "),
            "{outputs:?}"
        );
        assert!(matches!(turns[3], Turn::Reply { text, .. } if text.trim() == "Ok."));
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
        let turns: Vec<&Turn> = harness.conversation().collect();
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
        // a file changed starts it over too, whether write changed it or
        // a command bash ran did.
        let dir = scratch();
        let path = dir.join("log.jsonl");
        let echo = |i: usize| calling("bash", &[("command", &format!("echo {i}"))]);
        let mut scripts: Vec<String> = (0..3).map(echo).collect();
        scripts.push(calling("write", &[("path", "a.txt"), ("content", "made")]));
        scripts.extend((3..5).map(echo));
        scripts.push(calling(
            "bash",
            &[("command", "sed -i.bak s/made/done/ DIR/a.txt")],
        ));
        scripts.extend((5..7).map(echo));
        scripts.push(reply(10, false));
        let mut harness = harness_in(&dir, &scripts, &path);
        harness.set_nudge_after(3);
        send(&mut harness, "go");
        let results: Vec<&String> = harness
            .conversation()
            .filter_map(|turn| match turn {
                Turn::Results(outputs) => outputs.last(),
                _ => None,
            })
            .collect();
        let nudged: Vec<bool> = results.iter().map(|r| r.ends_with(NUDGE)).collect();
        assert_eq!(
            nudged,
            [false, false, true, false, false, false, false, false, false],
            "{results:?}"
        );
        assert_eq!(fs::read_to_string(dir.join("a.txt")).unwrap(), "done\n");
        fs::remove_dir_all(dir).unwrap();
    }

    /// A reply after a short thought that calls `tool` with `parameters`.
    fn calling(tool: &str, parameters: &[(&str, &str)]) -> String {
        let parameters: String = parameters
            .iter()
            .map(|(name, value)| format!("<parameter={name}>\n{value}\n</parameter>\n"))
            .collect();
        format!(
            "{}\n</think>\n\n<tool_call>\n<function={tool}>\n{parameters}</function>\n</tool_call>",
            filler(10)
        )
    }

    /// Sends `message`, and returns the plan each time it was reported.
    fn plans(harness: &mut Harness<Local<Scripted>>, message: &str) -> Vec<Plan> {
        let mut plans = Vec::new();
        harness
            .send(message, |event| {
                if let Event::Plan(plan) = event {
                    plans.push(plan);
                }
                ControlFlow::Continue(())
            })
            .unwrap();
        plans
    }

    #[test]
    fn keeps_the_plan_the_model_lays_out_and_what_each_step_rests_on() {
        // The model lays out two steps, changes a file with write and one
        // with bash, runs a command, and says each step is done: the plan
        // is reported as it changes, the first step with the files it
        // changed and the command run after.
        let dir = scratch();
        let path = dir.join("log.jsonl");
        let scripts = [
            calling(
                "plan",
                &[
                    ("steps", "- write the file\n- look it over"),
                    ("assumes", "one line"),
                ],
            ),
            calling("write", &[("path", "a.txt"), ("content", "made")]),
            calling("bash", &[("command", "echo made > DIR/b.txt")]),
            calling("bash", &[("command", "cat DIR/a.txt")]),
            calling("done", &[("result", "a.txt holds made")]),
            calling("done", &[("result", "it reads well")]),
            reply(10, false),
        ];
        let mut harness = harness_in(&dir, &scripts, &path);
        let plans = plans(&mut harness, "go");
        let first = plans.first().unwrap();
        assert_eq!(first.steps().len(), 2);
        assert_eq!(first.under_way(), Some(0));
        assert_eq!(first.assumes(), ["one line"]);
        let last = plans.last().unwrap();
        assert!(last.finished(), "{last:?}");
        let step = &last.steps()[0];
        assert_eq!(step.result, "a.txt holds made");
        assert_eq!(step.changed, ["a.txt", "b.txt"]);
        let check = step.check.as_ref().unwrap();
        assert!(
            check.passed && check.command.starts_with("cat "),
            "{check:?}"
        );
        assert_eq!(last.steps()[1].changed, Vec::<String>::new());
        // The model is told the step under way as each is finished.
        let results: Vec<&String> = harness
            .conversation()
            .filter_map(|turn| match turn {
                Turn::Results(outputs) => outputs.last(),
                _ => None,
            })
            .collect();
        assert!(results[0].contains("Step 1 of 2 is under way: write the file"));
        assert!(results[4].contains("Step 2 of 2 is under way: look it over"));
        assert!(results[5].contains("Every step of the plan is done"));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn asks_after_the_steps_left_when_the_model_stops_short_of_them() {
        // The model finishes the work and replies without saying the last
        // step is done: it is asked once, says so, and replies again.
        let scripts = [
            calling("plan", &[("steps", "one\ntwo")]),
            calling("done", &[("result", "did one")]),
            reply(10, false),
            calling("done", &[("result", "did two")]),
            reply(10, false),
        ];
        let mut harness = harness(&scripts, 100_000, "");
        let plans = plans(&mut harness, "go");
        assert!(plans.last().unwrap().finished());
        let asked = |harness: &Harness<Local<Scripted>>| {
            harness
                .conversation()
                .filter(|turn| matches!(turn, Turn::User(text) if text == STEPS_LEFT))
                .count()
        };
        assert_eq!(asked(&harness), 1);

        // One that stops short again after it is asked has its say: the
        // turn ends there, with the step left.
        let scripts = [
            calling("plan", &[("steps", "one\ntwo")]),
            reply(10, false),
            reply(10, false),
            reply(10, false),
        ];
        let mut harness = self::harness(&scripts, 100_000, "");
        let plans = self::plans(&mut harness, "go");
        assert_eq!(plans.last().unwrap().under_way(), Some(0));
        assert_eq!(asked(&harness), 1);

        // One the user stops mid-reply is not asked: the user has the say.
        let scripts = [
            calling("plan", &[("steps", "one\ntwo")]),
            reply(10, false),
            reply(10, false),
        ];
        let mut harness = self::harness(&scripts, 100_000, "");
        let mut planned = false;
        harness
            .send("go", |event| match event {
                Event::Plan(_) => {
                    planned = true;
                    ControlFlow::Continue(())
                }
                Event::Thought(_) if planned => ControlFlow::Break(()),
                _ => ControlFlow::Continue(()),
            })
            .unwrap();
        assert_eq!(asked(&harness), 0);
    }

    #[test]
    fn reminds_inkling_of_a_rule_in_the_result_of_the_call_that_went_against_it() {
        // Inkling reads a file through bash, then changes it through bash
        // with no plan laid out: each result ends with what to do instead.
        // Qwen, doing the same, is told nothing.
        let results = |model: Model| -> Vec<String> {
            let dir = scratch();
            fs::write(dir.join("a.txt"), "made\n").unwrap();
            let scripts = [
                calling("bash", &[("command", "cat DIR/a.txt")]),
                calling("bash", &[("command", "sed -i.bak s/made/done/ DIR/a.txt")]),
                calling("bash", &[("command", "echo checked")]),
                reply(10, false),
            ];
            let mut harness = harness_in(&dir, &scripts, &dir.join("log.jsonl"));
            harness.set_model(model);
            send(&mut harness, "go");
            let results = harness
                .conversation()
                .filter_map(|turn| match turn {
                    Turn::Results(outputs) => outputs.last().cloned(),
                    _ => None,
                })
                .collect();
            fs::remove_dir_all(dir).unwrap();
            results
        };
        let told = results(Model::InklingMini);
        assert!(
            told[0].starts_with("made\n[exit code 0]\n\n<system-reminder>You read a file"),
            "{told:?}"
        );
        assert!(
            told[1].contains("You changed a.txt through bash. Change files with the edit tool."),
            "{told:?}"
        );
        assert!(
            told[1].contains("Call plan with the steps you see."),
            "{told:?}"
        );
        assert_eq!(told[2], "checked\n[exit code 0]");
        for result in results(Model::Qwen27b) {
            assert!(!result.contains("<system-reminder>"), "{result}");
        }
    }

    #[test]
    fn gives_inkling_the_rules_ahead_of_the_first_message() {
        // With Inkling, the harness says how to work with where the agent
        // is, once, ahead of the first message, and the conversation keeps
        // the messages alone; with Qwen, it says only where the agent is.
        let scripts = [reply(10, false), reply(10, false)];
        let mut told = harness(&scripts, 100_000, "");
        told.set_model(Model::InklingMini);
        send(&mut told, "first message");
        send(&mut told, "on");
        let read = fed(&told);
        assert_eq!(read.matches(RULES).count(), 1);
        assert!(read.find("# Environment").unwrap() < read.find(RULES).unwrap());
        assert!(read.find(RULES).unwrap() < read.find("first message").unwrap());
        assert!(
            matches!(told.conversation().next(), Some(Turn::User(text)) if text == "first message")
        );

        let mut plain = harness(&scripts, 100_000, "");
        plain.set_model(Model::Qwen27b);
        send(&mut plain, "go");
        assert!(fed(&plain).contains("# Environment"));
        assert!(!fed(&plain).contains(RULES));
    }

    #[test]
    fn counts_the_calls_toward_the_cap_from_the_start_of_a_step() {
        // Under a cap of three calls, a plan of two steps takes six: the
        // step finished starts the count over, and the turn ends with the
        // model's reply, not with the account asked for at the cap.
        let scripts = [
            calling("plan", &[("steps", "one\ntwo")]),
            reply_calling(10, "bash", "command", "echo 1"),
            calling("done", &[("result", "did one")]),
            reply_calling(10, "bash", "command", "echo 2"),
            calling("done", &[("result", "did two")]),
            reply(10, false),
        ];
        let mut harness = harness(&scripts, 100_000, "");
        harness.set_call_limit(3);
        let plans = plans(&mut harness, "go");
        assert!(plans.last().unwrap().finished());
        assert!(
            !harness
                .conversation()
                .any(|turn| matches!(turn, Turn::User(text) if text == STEPS_PROMPT))
        );
    }

    #[test]
    fn a_message_while_steps_are_left_is_followed_by_the_plan() {
        // The turn is interrupted with a step left. The next message goes
        // to the model with the plan as it stands after it, and the model
        // changes the steps to come; the step done stays done.
        let dir = scratch();
        let path = dir.join("log.jsonl");
        let scripts = [
            calling("plan", &[("steps", "one\ntwo\nthree")]),
            calling("done", &[("result", "did one")]),
            calling("plan", &[("steps", "two\nfour")]),
            reply(10, false),
            reply(10, false),
            reply(10, false),
        ];
        let mut harness = harness_in(&dir, &scripts, &path);
        let mut calls = 0;
        harness
            .send("go", |event| {
                calls += usize::from(matches!(event, Event::Output(_)));
                if calls == 2 {
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                }
            })
            .unwrap();
        let plans = plans(&mut harness, "not three, four");
        let texts: Vec<(&str, bool)> = plans
            .last()
            .unwrap()
            .steps()
            .iter()
            .map(|step| (step.text.as_str(), step.done))
            .collect();
        assert_eq!(texts, [("one", true), ("two", false), ("four", false)]);
        // A plan set aside is not read back with the next message.
        harness.set_plan_aside();
        assert!(self::plans(&mut harness, "what is the time?").is_empty());
        drop(harness);
        let reminders: Vec<String> = records(&path)
            .iter()
            .filter(|record| record["type"] == "nudge")
            .map(|record| record["text"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(reminders.len(), 1, "{reminders:?}");
        assert!(
            reminders[0].contains("1. [done] one: did one\n2. [under way] two\n3. [to come] three"),
            "{reminders:?}"
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn starts_a_step_in_a_conversation_cleared_down_to_the_plan_once_it_is_long() {
        // The first step ends with the conversation past half the context:
        // the next starts from the plan, the user's message, and the reply
        // that finished the step, with its results.
        let scripts = [
            calling("plan", &[("steps", "find it\nfix it\ncheck it")]),
            reply_saying(10, 4200, true),
            calling("done", &[("result", "it is in send(), stream.rs:120")]),
            calling("done", &[("result", "fixed")]),
            reply(10, false),
            reply(10, false),
        ];
        let mut harness = harness(&scripts, 8000, "");
        let (events, _) = send(&mut harness, "fix the retries");
        assert!(events.contains(&"compacted"), "{events:?}");
        assert!(
            !events.contains(&"note"),
            "the model writes no note: {events:?}"
        );
        assert_eq!(
            events.iter().filter(|&&event| event == "compacted").count(),
            1,
            "{events:?}"
        );
        assert!(
            harness.session.tokens() < 4000,
            "{}",
            harness.session.tokens()
        );
        let turns: Vec<&Turn> = harness.conversation().collect();
        let Turn::User(briefing) = turns[0] else {
            panic!("{:?}", turns[0])
        };
        assert!(briefing.starts_with("# Environment\n"), "{briefing}");
        assert!(
            briefing.contains("<plan>\n1. [done] find it: it is in send(), stream.rs:120\n2. [under way] fix it\n3. [to come] check it\n</plan>"),
            "{briefing}"
        );
        assert!(!briefing.contains(&filler(100)), "{briefing}");
        assert!(
            briefing.contains("<message>\nfix the retries\n</message>"),
            "{briefing}"
        );
        assert!(
            matches!(turns[1], Turn::Reply { calls, .. } if calls.len() == 1),
            "{:?}",
            turns[1]
        );
        assert!(
            matches!(turns[2], Turn::Results(outputs) if outputs[0].contains("Step 2 of 3 is under way: fix it")),
            "{:?}",
            turns[2]
        );
    }

    #[test]
    fn goes_on_in_auto_mode_until_the_model_says_it_is_done() {
        // In auto mode, a reply of text alone after a call is answered
        // with word to go on, and the turn goes on to the calls after it,
        // until a reply ends with DONE; out of it, the turn ends at the
        // first reply of text alone.
        let scripts = [
            reply_calling(10, "bash", "command", "echo 0"),
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
        let turns: Vec<&Turn> = auto.conversation().collect();
        let told = turns
            .iter()
            .filter(|turn| matches!(turn, Turn::User(m) if m == GO_ON))
            .count();
        assert_eq!(told, 2, "{turns:?}");
        assert!(matches!(turns.last(), Some(Turn::Reply { text, .. }) if said_done(text)));
        assert!(turns.iter().any(|turn| matches!(turn, Turn::Results(_))));

        let mut plain = harness(&scripts, 100_000, "");
        send(&mut plain, "go");
        assert_eq!(plain.conversation().count(), 4);
        // Only in auto mode is the model told, with the message, how to
        // say it is done.
        assert!(fed(&auto).contains(UNATTENDED));
        assert!(!fed(&plain).contains(DONE));

        // A model that never says it is done is told to go on no more
        // than AUTO_ROUNDS times.
        let scripts: Vec<String> = iter::once(reply_calling(10, "bash", "command", "echo 0"))
            .chain((0..AUTO_ROUNDS + 3).map(|_| reply(10, false)))
            .collect();
        let mut endless = harness(&scripts, 100_000, "");
        endless.set_auto(true);
        send(&mut endless, "go");
        let told = endless
            .conversation()
            .filter(|turn| matches!(turn, Turn::User(m) if m == GO_ON))
            .count();
        assert_eq!(told, AUTO_ROUNDS);
    }

    #[test]
    fn ends_at_an_answer_without_a_call_in_auto_mode() {
        // A question answered without a tool leaves no work to go on
        // with: the turn ends at the answer, though it doesn't say DONE.
        // The next message starts over, and is gone on with once it has
        // made a call.
        let scripts = [
            reply(10, false),
            reply_calling(10, "bash", "command", "echo 1"),
            reply(10, false),
            format!("{}\n</think>\n\nChecked.\nDONE", filler(10)),
        ];
        let mut harness = harness(&scripts, 100_000, "");
        harness.set_auto(true);
        send(&mut harness, "say hello");
        assert_eq!(harness.conversation().count(), 2);
        send(&mut harness, "go");
        let told = harness
            .conversation()
            .filter(|turn| matches!(turn, Turn::User(m) if m == GO_ON))
            .count();
        assert_eq!(told, 1);
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
        let turns: Vec<&Turn> = harness.conversation().collect();
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
        let turns: Vec<&Turn> = harness.conversation().collect();
        assert!(
            matches!(&turns[turns.len() - 2], Turn::User(m) if m == STEPS_PROMPT),
            "{:?}",
            turns[turns.len() - 2]
        );
        assert!(
            matches!(turns.last(), Some(Turn::Reply { text, calls, .. }) if text == "I looked three times." && calls.is_empty()),
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
        harness.keep_transcript(
            Transcript::create(&path).unwrap(),
            "tiny",
            "cpu",
            Effort::Low,
            "Be brief.",
        );
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
        let briefing = records[cut + 3]["briefing"].as_str().unwrap();
        assert!(briefing.starts_with("# Environment"), "{briefing}");
        assert!(briefing.contains("Task: go on."), "{briefing}");
        assert!(briefing.contains("message 0"), "{briefing}");
        assert!(
            briefing.contains(&format!("transcript, word for word: {}", path.display())),
            "{briefing}"
        );
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
        // As the model's template does: a line for xhigh and for low, and
        // one for high, ahead of the tools, and none for medium.
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        assert!(system_prompt(dir, Model::Qwen27b, Some(Effort::Medium)).starts_with("# Tools\n"));
        // A host writes the line itself, so its prompt is given none.
        assert!(system_prompt(dir, Model::Qwen27b, None).starts_with("# Tools\n"));
        for (effort, word) in [
            (Effort::Xhigh, "xhigh"),
            (Effort::High, "high"),
            (Effort::Low, "low"),
        ] {
            let prompt = system_prompt(dir, Model::Qwen27b, Some(effort));
            let line = format!("Reasoning effort is set to {word}.");
            assert!(prompt.starts_with(&line), "{prompt}");
            assert!(prompt.contains(".\n\n# Tools\n"));
            assert_eq!(word.parse::<Effort>(), Ok(effort));
        }
        assert_eq!("medium".parse::<Effort>(), Ok(Effort::Medium));
        assert!("highest".parse::<Effort>().is_err());
    }

    #[test]
    fn describes_the_project() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let prompt = system_prompt(dir, Model::Qwen27b, Some(Effort::Medium));
        assert!(prompt.starts_with("# Tools\n"));
        assert!(
            prompt.contains("<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"bash\"")
        );
        assert!(prompt.contains("\n{\"type\": \"function\", \"function\": {\"name\": \"read\""));
        assert!(prompt.contains(CALLING));
        assert!(prompt.contains(&instructions(Model::Qwen27b)));
        // The tools' limits as the tools have them.
        assert!(prompt.contains(&format!("a page of up to {} numbered lines", read::PAGE)));
        assert!(prompt.contains(&format!("killed after {} seconds", bash::TIMEOUT)));
        let agents = fs::read_to_string(dir.join("AGENTS.md")).unwrap();
        assert!(prompt.contains(&format!("From AGENTS.md:\n\n{}", agents.trim())));
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
