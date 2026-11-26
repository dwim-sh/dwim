//! The harness's compaction against its Lean spec,
//! `specs/Dwim/Harness/Compaction.lean`.
//!
//! The spec says what a compaction keeps of a conversation: where the tail
//! starts, which of the messages the user has sent the briefing carries,
//! and that the briefing says where the session's transcript is, which
//! holds whatever compaction removes; and what each budget is in a window
//! of a given size. `lake exe compaction-oracle` answers for the spec, and
//! `select`, the briefing and the budgets must answer the same: on conversations of every shape the chat
//! format allows, after earlier compactions that left some of the user's
//! messages out of them, in windows whose budgets fall at and a token
//! either side of what the conversation's suffixes take, where an
//! off-by-one would show. The cases are made from seeds, so a disagreement
//! is the same on every run.

use std::path::Path;

use dwim::{
    harness::{
        Entry,
        compaction::{briefing, high_water, message_budget, note_budget, select, tail_budget},
    },
    model::Turn,
};

use crate::{
    Result,
    spec::{Rng, ask},
};

/// Conversations the code is compared with the spec on.
const CASES: u64 = 10_000;

/// Where the session's transcript is, in every case.
const TRANSCRIPT: &str = "/home/user/.local/state/dwim/projects/p/t.jsonl";

/// Window sizes the budgets are compared on: every size up to this one,
/// which takes the note's budget past both of its bounds.
const WINDOWS: usize = 100_000;

/// Compares the code with the spec on [`CASES`] conversations and on
/// every window up to [`WINDOWS`] tokens, and says whether they agreed on
/// all.
pub fn run(mut args: impl Iterator<Item = String>) -> Result<bool> {
    if let Some(arg) = args.next() {
        return Err(format!("compaction takes no arguments, not '{arg}'").into());
    }
    let cases: Vec<Case> = (0..CASES).map(Case::new).collect();
    let mut input: String = cases.iter().map(Case::written).collect();
    for capacity in 0..=WINDOWS {
        input.push_str(&format!("budgets {capacity}\n"));
    }
    let spec = ask("compaction-oracle", &input)?;
    let mut answers = spec.lines();
    for (seed, case) in cases.iter().enumerate() {
        let spec = answers
            .next()
            .ok_or("the oracle left a conversation unanswered")?;
        let code = case.kept();
        if code != spec {
            eprintln!(
                "difftest: compaction: case {seed} disagrees\n  case: {}  code: {code}\n  spec: {spec}",
                case.written()
            );
            if code.ends_with("transcript=(none)") {
                eprintln!(
                    "  the briefing does not say where the transcript is, so the model cannot read back what compaction removed"
                );
            }
            return Ok(false);
        }
    }
    for capacity in 0..=WINDOWS {
        let spec = answers
            .next()
            .ok_or("the oracle left a window unanswered")?;
        let code = budgets(capacity);
        if code != spec {
            eprintln!(
                "difftest: compaction: the budgets of a window of {capacity} disagree\n  code: {code}\n  spec: {spec}"
            );
            return Ok(false);
        }
    }
    println!(
        "compaction: the code agrees with the spec on {CASES} conversations and {} windows",
        WINDOWS + 1
    );
    Ok(true)
}

/// The budgets of a window of `capacity` tokens, as the oracle writes them.
fn budgets(capacity: usize) -> String {
    format!(
        "high_water={} note={} tail={} messages={}",
        high_water(capacity),
        note_budget(capacity),
        tail_budget(capacity),
        message_budget(capacity)
    )
}

/// A conversation, the messages the user has sent, and the window to
/// compact it in.
struct Case {
    capacity: usize,
    said: Vec<Entry>,
    conversation: Vec<Entry>,
}

impl Case {
    /// The case made from `seed`. The user has sent a few messages that
    /// earlier compactions left out of the conversation, which may open
    /// with the briefing of one of them. Then come exchanges, each opened
    /// by a message of the user's, a prompt of the harness's, or neither,
    /// then replies that call tools, each answered by results, then one
    /// that calls none; and, at times, last, a reply whose calls are still
    /// to be answered. A message of the user's is named by its place in
    /// what the user has sent, so the messages carried say which they were.
    fn new(seed: u64) -> Case {
        let mut rng = Rng::seeded(seed);
        let mut said = Vec::new();
        for _ in 0..rng.below(4) {
            let message = format!("m{}", said.len());
            add(&mut said, Turn::User(message), false, &mut rng);
        }
        let mut conversation = Vec::new();
        if !said.is_empty() && rng.below(2) == 0 {
            let briefing = Turn::User("briefing".to_string());
            add(&mut conversation, briefing, true, &mut rng);
        }
        for _ in 0..rng.below(7) {
            match rng.below(4) {
                0 | 1 => {
                    let message = format!("m{}", said.len());
                    add(&mut conversation, Turn::User(message), false, &mut rng);
                    said.push(conversation[conversation.len() - 1].clone());
                }
                2 => {
                    let prompt = Turn::User("go on".to_string());
                    add(&mut conversation, prompt, true, &mut rng);
                }
                _ => {}
            }
            while rng.below(3) == 0 {
                add(&mut conversation, reply(true), false, &mut rng);
                let results = Turn::Results(vec!["ok".to_string()]);
                add(&mut conversation, results, false, &mut rng);
            }
            add(&mut conversation, reply(false), false, &mut rng);
        }
        if rng.below(4) == 0 {
            add(&mut conversation, reply(true), false, &mut rng);
        }
        // As often as not, the window's eighth falls at the tail's
        // boundaries, and otherwise its sixteenth at the messages'.
        let capacity = if rng.below(2) == 0 {
            rng.around(&suffixes(&conversation)) * 8 + rng.below(8)
        } else {
            rng.around(&suffixes(&said)) * 16 + rng.below(16)
        };
        Case {
            capacity,
            said,
            conversation,
        }
    }

    /// What the code keeps, as the oracle writes what the spec keeps: and
    /// where the briefing the code writes says the transcript is, if it
    /// says.
    fn kept(&self) -> String {
        let (start, carried) = select(&self.said, &self.conversation, self.capacity);
        let exchanges = start < self.conversation.len();
        let transcript = Some(Path::new(TRANSCRIPT));
        let text = briefing("env", "Task: go on.", "", &carried, transcript, exchanges);
        let transcript = if text.contains(TRANSCRIPT) {
            TRANSCRIPT
        } else {
            "(none)"
        };
        format!(
            "tail={start} carried={} transcript={transcript}",
            carried.join(",")
        )
    }

    /// The case as the oracle reads it: the capacity, where the transcript
    /// is, each turn of the conversation, and after a `|` the messages the
    /// user has sent.
    fn written(&self) -> String {
        let mut line = format!("case {} {TRANSCRIPT}", self.capacity);
        for entry in &self.conversation {
            line.push_str(&format!(" {}", notation(entry)));
        }
        line.push_str(" |");
        for entry in &self.said {
            line.push_str(&format!(" {}", notation(entry)));
        }
        line.push('\n');
        line
    }
}

/// `entry` as the oracle reads a turn: its kind and its tokens, and a
/// message of the user's its name too, as `user:m0:12`.
fn notation(entry: &Entry) -> String {
    match &entry.turn {
        Turn::User(_) if entry.prompt => format!("prompt:{}", entry.tokens),
        Turn::User(message) => format!("user:{message}:{}", entry.tokens),
        Turn::Reply { calls, .. } if calls.is_empty() => format!("reply:{}", entry.tokens),
        Turn::Reply { .. } => format!("call:{}", entry.tokens),
        Turn::Results(_) => format!("results:{}", entry.tokens),
    }
}

/// Adds `turn` to `entries`, taking tokens from `rng`.
fn add(entries: &mut Vec<Entry>, turn: Turn, prompt: bool, rng: &mut Rng) {
    entries.push(Entry {
        turn,
        tokens: rng.below(100),
        prompt,
    });
}

/// A reply, which calls a tool if `calls`.
fn reply(calls: bool) -> Turn {
    Turn::Reply {
        text: String::new(),
        calls: if calls {
            vec!["call".to_string()]
        } else {
            Vec::new()
        },
        thought: String::new(),
    }
}

/// The tokens every suffix of `entries` takes, the empty one too.
fn suffixes(entries: &[Entry]) -> Vec<usize> {
    let mut sizes = vec![0];
    for entry in entries.iter().rev() {
        sizes.push(sizes[sizes.len() - 1] + entry.tokens);
    }
    sizes
}
