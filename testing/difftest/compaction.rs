//! The harness's compaction against its Lean spec,
//! `specs/Dwim/Harness/Compaction.lean`.
//!
//! The spec says what a compaction keeps of a conversation: where the tail
//! starts, which of the messages the user has sent the briefing carries,
//! and that the briefing says where the session's transcript is, which
//! holds whatever compaction removes; and what each budget is in a window
//! of a given size. `lake exe compaction-oracle` answers for the spec, and
//! `select`, the briefing and the budgets must answer the same.
//!
//! As Cedar's differential testing does, it compares the two on random
//! cases, many of them, from more than one generator, and says how often
//! the cases reach each branch of compaction, since a generator whose cases
//! all take the same path tests little however many there are. Most cases
//! are conversations of the shapes the chat format allows; some are turns
//! in any order; and some are long, or have turns of thousands of tokens.
//! Windows are chosen so that budgets fall at and a token either side of
//! what the conversation's suffixes take, where an off-by-one would show.
//!
//! A case is made from its seed, so a disagreement is the same on every
//! run. By default the seeds are the same too, for `make check`; `--random`
//! starts from a seed of the time, to find what the default ones do not,
//! and `--cases` runs as many as asked. A disagreement is shrunk to the
//! smallest case that still disagrees, and reported with the command that
//! runs its seed again.

use std::{
    path::Path,
    time::{SystemTime, UNIX_EPOCH},
};

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

/// Conversations the code is compared with the spec on, unless asked for
/// another number.
const CASES: u64 = 10_000;

/// Cases the oracle is asked at a time, so that memory stays the same
/// however many are asked for.
const BATCH: u64 = 100_000;

/// Where the session's transcript is, in every case.
const TRANSCRIPT: &str = "/home/user/.local/state/dwim/projects/p/t.jsonl";

/// Window sizes the budgets are compared on: every size up to this one,
/// which takes the note's budget past both of its bounds.
const WINDOWS: usize = 100_000;

/// What the command line asks for: the seed of the first case, and how
/// many cases.
struct Args {
    seed: u64,
    cases: u64,
}

impl Args {
    fn parse(mut args: impl Iterator<Item = String>) -> Result<Self> {
        let mut parsed = Args {
            seed: 0,
            cases: CASES,
        };
        while let Some(arg) = args.next() {
            let mut number = |name: &str| -> Result<u64> {
                args.next()
                    .and_then(|n| n.parse().ok())
                    .ok_or_else(|| format!("{name} takes a number").into())
            };
            match arg.as_str() {
                "--seed" => parsed.seed = number("--seed")?,
                "--cases" => parsed.cases = number("--cases")?,
                "--random" => {
                    parsed.seed = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos() as u64
                }
                _ => {
                    return Err(format!(
                        "compaction takes --seed, --cases and --random, not '{arg}'"
                    )
                    .into());
                }
            }
        }
        Ok(parsed)
    }
}

/// Compares the code with the spec on the cases asked for and on every
/// window up to [`WINDOWS`] tokens, and says whether they agreed on all.
pub fn run(args: impl Iterator<Item = String>) -> Result<bool> {
    let Args { seed, cases } = Args::parse(args)?;
    let mut reached = Reached::default();
    for first in (0..cases).step_by(BATCH as usize) {
        let batch: Vec<Case> = (first..cases.min(first + BATCH))
            .map(|i| Case::new(seed.wrapping_add(i)))
            .collect();
        let input: String = batch.iter().map(Case::written).collect();
        let spec = ask("compaction-oracle", &input)?;
        let mut answers = spec.lines();
        for (i, case) in batch.iter().enumerate() {
            let spec = answers
                .next()
                .ok_or("the oracle left a conversation unanswered")?;
            if case.kept() != spec {
                let seed = seed.wrapping_add(first + i as u64);
                report(seed, case, spec);
                let small = shrink(case.clone())?;
                if small.written() != case.written() {
                    let spec = ask("compaction-oracle", &small.written())?;
                    eprintln!("  shrunk to {} turns:", small.conversation.len());
                    report_lines(&small, spec.trim_end());
                }
                eprintln!("  again: cargo run --bin difftest compaction --seed {seed} --cases 1");
                return Ok(false);
            }
            reached.count(case);
        }
    }
    let input: String = (0..=WINDOWS)
        .map(|capacity| format!("budgets {capacity}\n"))
        .collect();
    let spec = ask("compaction-oracle", &input)?;
    let mut answers = spec.lines();
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
        "compaction: the code agrees with the spec on {cases} conversations, seeds {seed} on, and {} windows",
        WINDOWS + 1
    );
    reached.print(cases);
    Ok(true)
}

/// Says that the case of `seed` disagrees with the spec, which answered
/// `spec`.
fn report(seed: u64, case: &Case, spec: &str) {
    eprintln!("difftest: compaction: the case of seed {seed} disagrees");
    report_lines(case, spec);
}

/// The case, what the code answers and what the spec answers.
fn report_lines(case: &Case, spec: &str) {
    let code = case.kept();
    eprintln!("  case: {}  code: {code}\n  spec: {spec}", case.written());
    if code.ends_with("transcript=(none)") {
        eprintln!(
            "  the briefing does not say where the transcript is, so the model cannot read back what compaction removed"
        );
    }
}

/// The smallest case found, from `case`, that still disagrees with the
/// spec: one turn fewer, fewer tokens, or a smaller window at a time, for
/// as long as a smaller one disagrees.
fn shrink(mut case: Case) -> Result<Case> {
    loop {
        let smaller = case.smaller();
        if smaller.is_empty() {
            return Ok(case);
        }
        let input: String = smaller.iter().map(Case::written).collect();
        let spec = ask("compaction-oracle", &input)?;
        let next = smaller
            .into_iter()
            .zip(spec.lines())
            .find(|(small, spec)| small.kept() != *spec);
        match next {
            Some((small, _)) => case = small,
            None => return Ok(case),
        }
    }
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

/// How many cases reached each branch of compaction: whether the tail fit
/// its budget, was the last exchange however large, or found no reply;
/// and whether the briefing carried none of the user's messages, some
/// within their budget, or the latest over it.
#[derive(Default)]
struct Reached {
    loose: u64,
    long: u64,
    tail_fits: u64,
    tail_last: u64,
    tail_none: u64,
    carried_none: u64,
    carried_fits: u64,
    carried_latest: u64,
}

impl Reached {
    fn count(&mut self, case: &Case) {
        let (start, carried) = select(&case.said, &case.conversation, case.capacity);
        let tail: usize = case.conversation[start..].iter().map(|e| e.tokens).sum();
        if start == case.conversation.len() {
            self.tail_none += 1;
        } else if tail <= tail_budget(case.capacity) {
            self.tail_fits += 1;
        } else {
            self.tail_last += 1;
        }
        let tokens: usize = case
            .said
            .iter()
            .filter(|e| matches!(&e.turn, Turn::User(m) if carried.contains(&m.as_str())))
            .map(|e| e.tokens)
            .sum();
        if carried.is_empty() {
            self.carried_none += 1;
        } else if tokens <= message_budget(case.capacity) {
            self.carried_fits += 1;
        } else {
            self.carried_latest += 1;
        }
        self.loose += u64::from(case.loose);
        self.long += u64::from(case.long);
    }

    fn print(&self, cases: u64) {
        let share = |n: u64| n * 100 / cases.max(1);
        println!(
            "  cases: {}% in any order, {}% long or of large turns",
            share(self.loose),
            share(self.long)
        );
        println!(
            "  tail: {}% fit the budget, {}% the last exchange over it, {}% no reply",
            share(self.tail_fits),
            share(self.tail_last),
            share(self.tail_none)
        );
        println!(
            "  carried: {}% none, {}% within the budget, {}% the latest over it",
            share(self.carried_none),
            share(self.carried_fits),
            share(self.carried_latest)
        );
    }
}

/// A conversation, the messages the user has sent, and the window to
/// compact it in; and which generator made it.
#[derive(Clone)]
struct Case {
    capacity: usize,
    said: Vec<Entry>,
    conversation: Vec<Entry>,
    loose: bool,
    long: bool,
}

impl Case {
    /// The case made from `seed`: one time in eight long, with more turns,
    /// turns of up to thousands of tokens, and more messages the user sent
    /// before; and one time in four of turns in any order, the rest of the
    /// shapes the chat format allows.
    fn new(seed: u64) -> Case {
        let mut rng = Rng::seeded(seed);
        let long = rng.below(8) == 0;
        let (turns, tokens, before) = if long { (40, 5000, 30) } else { (7, 100, 4) };
        let loose = rng.below(4) == 0;
        let mut case = Case {
            capacity: 0,
            said: Vec::new(),
            conversation: Vec::new(),
            loose,
            long,
        };
        for _ in 0..rng.below(before) {
            case.add_said(&mut rng, tokens);
        }
        if loose {
            case.any_order(&mut rng, turns * 3, tokens);
        } else {
            case.exchanges(&mut rng, turns, tokens);
        }
        // As often as not, the window's eighth falls at the tail's
        // boundaries, and otherwise its sixteenth at the messages'.
        case.capacity = if rng.below(2) == 0 {
            rng.around(&suffixes(&case.conversation)) * 8 + rng.below(8)
        } else {
            rng.around(&suffixes(&case.said)) * 16 + rng.below(16)
        };
        case
    }

    /// Adds exchanges of the shapes the chat format allows, after the
    /// briefing of an earlier compaction at times: each opened by a
    /// message of the user's, a prompt of the harness's, or neither, then
    /// replies that call tools, each answered by results, then one that
    /// calls none; and, at times, last, a reply whose calls are still to be
    /// answered.
    fn exchanges(&mut self, rng: &mut Rng, turns: usize, tokens: usize) {
        if !self.said.is_empty() && rng.below(2) == 0 {
            self.add(rng, prompt("briefing"), tokens);
        }
        for _ in 0..rng.below(turns) {
            match rng.below(4) {
                0 | 1 => self.add_user(rng, tokens),
                2 => self.add(rng, prompt("go on"), tokens),
                _ => {}
            }
            while rng.below(3) == 0 {
                self.add(rng, reply(true), tokens);
                self.add(rng, results(), tokens);
            }
            self.add(rng, reply(false), tokens);
        }
        if rng.below(4) == 0 {
            self.add(rng, reply(true), tokens);
        }
    }

    /// Adds turns of every kind in any order: what the chat format would
    /// not allow, such as results that answer no call, two replies in a
    /// row, or a briefing in the middle.
    fn any_order(&mut self, rng: &mut Rng, turns: usize, tokens: usize) {
        for _ in 0..rng.below(turns) {
            match rng.below(6) {
                0 => self.add_user(rng, tokens),
                1 => self.add(rng, prompt("go on"), tokens),
                2 => self.add(rng, prompt("briefing"), tokens),
                3 => self.add(rng, reply(false), tokens),
                4 => self.add(rng, reply(true), tokens),
                _ => self.add(rng, results(), tokens),
            }
        }
    }

    /// Adds a message the user sent that an earlier compaction left out of
    /// the conversation. A message of the user's is named by its place in
    /// what the user has sent, so the messages carried say which they were.
    fn add_said(&mut self, rng: &mut Rng, tokens: usize) {
        let message = format!("m{}", self.said.len());
        self.said.push(entry(user(&message), rng.below(tokens)));
    }

    /// Adds a message of the user's to the conversation, and to what the
    /// user has sent.
    fn add_user(&mut self, rng: &mut Rng, tokens: usize) {
        self.add_said(rng, tokens);
        self.conversation
            .push(self.said[self.said.len() - 1].clone());
    }

    /// Adds `turn` to the conversation, of up to `tokens` from `rng`.
    fn add(&mut self, rng: &mut Rng, turn: (Turn, bool), tokens: usize) {
        let mut added = entry(turn.0, rng.below(tokens));
        added.prompt = turn.1;
        self.conversation.push(added);
    }

    /// The cases a step smaller than this one: with a turn of the
    /// conversation left out, the user's message with it if it is one;
    /// with a message the user sent before it left out; with a turn of half
    /// its tokens, or none; or in a smaller window.
    fn smaller(&self) -> Vec<Case> {
        let mut smaller = Vec::new();
        for i in 0..self.conversation.len() {
            let mut case = self.clone();
            let removed = case.conversation.remove(i);
            case.said.retain(|e| !same_message(e, &removed));
            smaller.push(case);
        }
        for i in 0..self.said.len() {
            if !self
                .conversation
                .iter()
                .any(|e| same_message(e, &self.said[i]))
            {
                let mut case = self.clone();
                case.said.remove(i);
                smaller.push(case);
            }
        }
        for i in 0..self.conversation.len() {
            let tokens = self.conversation[i].tokens;
            for fewer in [tokens / 2, 0] {
                if fewer < tokens {
                    let mut case = self.clone();
                    case.conversation[i].tokens = fewer;
                    let changed = case.conversation[i].clone();
                    for e in &mut case.said {
                        if same_message(e, &changed) {
                            e.tokens = fewer;
                        }
                    }
                    smaller.push(case);
                }
            }
        }
        for capacity in [self.capacity / 2, self.capacity.saturating_sub(1)] {
            if capacity < self.capacity {
                let mut case = self.clone();
                case.capacity = capacity;
                smaller.push(case);
            }
        }
        smaller
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

/// Whether `a` and `b` are the same message of the user's.
fn same_message(a: &Entry, b: &Entry) -> bool {
    match (&a.turn, &b.turn) {
        (Turn::User(x), Turn::User(y)) => !a.prompt && !b.prompt && x == y,
        _ => false,
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

/// An entry of `turn` taking `tokens`, the user's own if a user turn.
fn entry(turn: Turn, tokens: usize) -> Entry {
    Entry {
        turn,
        tokens,
        prompt: false,
    }
}

/// A message of the user's.
fn user(message: &str) -> Turn {
    Turn::User(message.to_string())
}

/// A user turn the harness writes, such as a briefing, as a turn and that
/// it is the harness's own.
fn prompt(text: &str) -> (Turn, bool) {
    (Turn::User(text.to_string()), true)
}

/// A reply, which calls a tool if `calls`.
fn reply(calls: bool) -> (Turn, bool) {
    let calls = if calls {
        vec!["call".to_string()]
    } else {
        Vec::new()
    };
    let reply = Turn::Reply {
        text: String::new(),
        calls,
        thought: String::new(),
    };
    (reply, false)
}

/// The results of the tools a reply called.
fn results() -> (Turn, bool) {
    (Turn::Results(vec!["ok".to_string()]), false)
}

/// The tokens every suffix of `entries` takes, the empty one too.
fn suffixes(entries: &[Entry]) -> Vec<usize> {
    let mut sizes = vec![0];
    for entry in entries.iter().rev() {
        sizes.push(sizes[sizes.len() - 1] + entry.tokens);
    }
    sizes
}
