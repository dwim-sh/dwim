//! The harness's compaction against its Lean spec in `specs/`.
//!
//! The spec says what a compaction keeps of a conversation: where the
//! tail starts, and which of the messages the user has sent the head
//! carries. `lake exe compaction-oracle` says what it keeps of each
//! conversation it reads, and `select` must keep the same: on
//! conversations of every shape the chat grammar allows, after earlier
//! compactions that left some of the user's messages out of them, in
//! windows whose budgets fall at and a token either side of what their
//! suffixes take, where an off-by-one would show. The conversations are
//! made from seeds, so a disagreement is the same on every run.

use dwim::{
    harness::{Entry, compaction::select},
    model::Turn,
};

use crate::{
    Result,
    spec::{Rng, ask},
};

/// Conversations the code is compared with the spec on.
const CASES: u64 = 10_000;

/// Compares the code with the spec on [`CASES`] conversations, and says
/// whether they agreed on all.
pub fn run(mut args: impl Iterator<Item = String>) -> Result<bool> {
    if let Some(arg) = args.next() {
        return Err(format!("compaction takes no arguments, not '{arg}'").into());
    }
    let cases: Vec<Case> = (0..CASES).map(Case::new).collect();
    let spec = ask(
        "compaction-oracle",
        &cases.iter().map(Case::written).collect::<String>(),
    )?;
    let mut answers = spec.lines();
    for (seed, case) in cases.iter().enumerate() {
        let spec = answers
            .next()
            .ok_or("the oracle left a conversation unanswered")?;
        let code = case.kept();
        if code != spec {
            eprintln!(
                "difftest: compaction: case {seed} disagrees\n  conversation: {}  code: {code}\n  spec: {spec}",
                case.written()
            );
            return Ok(false);
        }
    }
    println!("compaction: the code agrees with the spec on {CASES} conversations");
    Ok(true)
}

/// A conversation, the messages the user has sent, and the window to
/// compact it in.
struct Case {
    capacity: usize,
    said: Vec<Entry>,
    entries: Vec<Entry>,
}

impl Case {
    /// The case made from `seed`. The user has sent a few messages that
    /// earlier compactions left out of the conversation, which may open
    /// with the head of one of them. Then come exchanges, each opened by
    /// a user message, a prompt, or neither, then replies that call tools,
    /// each answered by results, then one that calls none; and, at times,
    /// last, a reply whose calls are still to be answered. A user message
    /// says its index in what the user has sent, so that the messages
    /// kept say which they were.
    fn new(seed: u64) -> Case {
        let mut rng = Rng::seeded(seed);
        let mut said = Vec::new();
        for _ in 0..rng.below(4) {
            let message = said.len().to_string();
            add(&mut said, Turn::User(message), false, &mut rng);
        }
        let mut entries = Vec::new();
        if !said.is_empty() && rng.below(2) == 0 {
            add(&mut entries, Turn::User("head".to_string()), true, &mut rng);
        }
        for _ in 0..rng.below(7) {
            match rng.below(4) {
                0 | 1 => {
                    let message = said.len().to_string();
                    add(&mut entries, Turn::User(message.clone()), false, &mut rng);
                    let tokens = entries[entries.len() - 1].tokens;
                    said.push(Entry {
                        turn: Turn::User(message),
                        tokens,
                        prompt: false,
                    });
                }
                2 => add(
                    &mut entries,
                    Turn::User("go on".to_string()),
                    true,
                    &mut rng,
                ),
                _ => {}
            }
            while rng.below(3) == 0 {
                add(&mut entries, reply(true), false, &mut rng);
                add(
                    &mut entries,
                    Turn::Results(vec!["ok".to_string()]),
                    false,
                    &mut rng,
                );
            }
            add(&mut entries, reply(false), false, &mut rng);
        }
        if rng.below(4) == 0 {
            add(&mut entries, reply(true), false, &mut rng);
        }
        // The window is chosen so that, as often as not, its eighth falls
        // at the tail's boundaries, and otherwise its sixteenth at the
        // messages'.
        let capacity = if rng.below(2) == 0 {
            rng.around(&suffixes(entries.iter())) * 8 + rng.below(8)
        } else {
            rng.around(&suffixes(said.iter())) * 16 + rng.below(16)
        };
        Case {
            capacity,
            said,
            entries,
        }
    }

    /// What the code keeps, as the oracle writes what the spec does.
    fn kept(&self) -> String {
        let (start, carried) = select(&self.said, &self.entries, self.capacity);
        format!("tail={start} messages={}", carried.join(","))
    }

    /// The case as the oracle reads it: the capacity, each turn as its
    /// kind and its tokens, in the notation of the spec, and after a `|`
    /// the messages the user has sent.
    fn written(&self) -> String {
        let mut line = self.capacity.to_string();
        for entry in &self.entries {
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

/// `entry` in the notation of the spec: its kind and its tokens.
fn notation(entry: &Entry) -> String {
    let kind = match &entry.turn {
        Turn::User(_) if entry.prompt => "prompt",
        Turn::User(_) => "msg",
        Turn::Reply { calls, .. } if calls.is_empty() => "answer",
        Turn::Reply { .. } => "call",
        Turn::Results(_) => "res",
    };
    format!("{kind} {}", entry.tokens)
}

/// Adds a turn of `tokens` taken from `rng` to `entries`.
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
fn suffixes<'a>(entries: impl DoubleEndedIterator<Item = &'a Entry>) -> Vec<usize> {
    let mut sizes = vec![0];
    for entry in entries.rev() {
        sizes.push(sizes[sizes.len() - 1] + entry.tokens);
    }
    sizes
}
