//! The harness's turns against the Lean spec of a turn in `specs/`.
//!
//! The spec proves that however a turn ends, the conversation it leaves
//! is well-formed and does not end with a call: every call is answered.
//! Here the real harness, with a scripted model, plays turns made from
//! seeds: replies that read files of every size, some larger than the
//! whole window, one or two at a time; turns the user interrupts while
//! the tools run; windows small enough that the conversation is compacted
//! again and again. After every turn, `lake exe turn-oracle` says whether
//! the conversation the harness left is as the spec says it must be.

use std::{fs, ops::ControlFlow, path::PathBuf};

use dwim::harness::{Event, Harness};
use dwim::inference::{Local, Sampler, testing::Scripted};
use dwim::model::{Format, Tokenizer, Turn};

use crate::{
    Result,
    spec::{Rng, ask},
};

/// Sequences of turns the harness plays.
const CASES: u64 = 400;

/// The files the model reads, by name, with the bytes each holds: from a
/// few lines to more than the largest window here.
const FILES: [(&str, usize); 4] = [
    ("tiny", 40),
    ("small", 400),
    ("large", 4_000),
    ("huge", 20_000),
];

/// Plays every case, and says whether the harness left every conversation
/// as the spec says.
pub fn run(mut args: impl Iterator<Item = String>) -> Result<bool> {
    if let Some(arg) = args.next() {
        return Err(format!("harness takes no arguments, not '{arg}'").into());
    }
    let dir = files()?;
    let mut conversations = Vec::new();
    for seed in 0..CASES {
        let case = Case::new(seed);
        let played = case.play(&dir).map_err(|e| format!("case {seed}: {e}"))?;
        for (turn, conversation) in played.into_iter().enumerate() {
            conversations.push((seed, turn, conversation));
        }
    }
    let input: String = conversations
        .iter()
        .map(|(_, _, conversation)| format!("{conversation}\n"))
        .collect();
    let answers = ask("turn-oracle", &input)?;
    let mut answers = answers.lines();
    for (seed, turn, conversation) in &conversations {
        let answer = answers
            .next()
            .ok_or("the oracle left a conversation unanswered")?;
        if answer != "answered" {
            eprintln!(
                "difftest: harness: case {seed}, after turn {turn}, the conversation is {answer}:\n  {conversation}"
            );
            return Ok(false);
        }
    }
    println!(
        "harness: every call answered after {} turns of {CASES} cases",
        conversations.len()
    );
    Ok(true)
}

/// A window, what the model says, and the turns the user starts.
struct Case {
    max_len: usize,
    scripts: Vec<String>,
    /// For each message the user sends, the call or output event to
    /// interrupt the turn at, if it is interrupted.
    interrupts: Vec<Option<usize>>,
}

impl Case {
    /// The case made from `seed`: a window of 6,000 to 16,000 tokens,
    /// room for the harness's own prompts, which the tiny tokenizer makes
    /// a token a byte, and for a note in place of every output; a
    /// few messages, a turn each, some interrupted at a call or an
    /// output; and replies that read one or two files each, or call
    /// nothing, which the scripted model says in order, turn after turn.
    fn new(seed: u64) -> Case {
        let mut rng = Rng::seeded(seed);
        let max_len = 6_000 + rng.below(10_000);
        let turns = 1 + rng.below(4);
        let interrupts = (0..turns)
            .map(|_| (rng.below(4) == 0).then(|| rng.below(6)))
            .collect();
        let mut scripts = Vec::new();
        for _ in 0..turns {
            for _ in 0..rng.below(4) {
                let reads = (0..1 + rng.below(2))
                    .map(|_| FILES[rng.below(FILES.len())].0)
                    .collect::<Vec<_>>();
                scripts.push(reading(&reads));
            }
            scripts.push(format!("{}\n</think>\n\nOk.", filler(10)));
        }
        Case {
            max_len,
            scripts,
            interrupts,
        }
    }

    /// Plays the turns in a harness working in `dir`, and returns the
    /// conversation after each, its turns by kind.
    fn play(&self, dir: &std::path::Path) -> Result<Vec<String>> {
        let tokenizer = Tokenizer::tiny();
        let scripts: Vec<&str> = self.scripts.iter().map(String::as_str).collect();
        let model =
            Scripted::new(&tokenizer, &scripts, self.max_len).answers(&tokenizer, " go on.");
        let mut session = Local::new(
            model,
            Format::new(tokenizer)?,
            Sampler::new(0.0, 1, 1.0, 0.0, 1),
        )?;
        session.system("Be brief.", |_, _| {})?;
        let mut harness = Harness::new(session, dir, |session| {
            session.system("Be brief.", |_, _| {})?;
            Ok(())
        })?;
        let mut conversations = Vec::new();
        for (turn, interrupt) in self.interrupts.iter().enumerate() {
            let mut events = 0;
            // A turn that fails leaves the conversation as it is, and the
            // next message follows it, as in the shell: what it left is
            // the spec's to judge.
            let _ = harness.send(&format!("message {turn}"), |event| {
                if matches!(event, Event::Call { .. } | Event::Output(_)) {
                    events += 1;
                    if Some(events) == interrupt.map(|at| at + 1) {
                        return ControlFlow::Break(());
                    }
                }
                ControlFlow::Continue(())
            });
            conversations.push(kinds(&harness));
        }
        Ok(conversations)
    }
}

/// The conversation `harness` holds, its turns by kind, as the oracle
/// reads them.
fn kinds(harness: &Harness<Local<Scripted>>) -> String {
    harness
        .transcript()
        .map(|turn| match turn {
            Turn::User(_) => "user",
            Turn::Reply { calls, .. } if calls.is_empty() => "answer",
            Turn::Reply { .. } => "call",
            Turn::Results(_) => "res",
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// A reply that reads the files named `reads`, a call each.
fn reading(reads: &[&str]) -> String {
    let calls: String = reads
        .iter()
        .map(|file| {
            format!(
                "<tool_call>\n<function=read>\n<parameter=path>\n{file}\n</parameter>\n</function>\n</tool_call>"
            )
        })
        .collect();
    format!("{}\n</think>\n\n{calls}", filler(10))
}

/// A thought of `n` bytes, which never repeats a window of itself: the
/// counting numbers run together.
fn filler(n: usize) -> String {
    (1..)
        .flat_map(|i: u32| i.to_string().into_bytes())
        .map(char::from)
        .take(n)
        .collect()
}

/// A directory with the files the model reads, each lines of 79 letters.
fn files() -> Result<PathBuf> {
    let dir = std::env::temp_dir().join(format!("dwim-difftest-harness-{}", std::process::id()));
    fs::create_dir_all(&dir)?;
    for (name, bytes) in FILES {
        let line = format!("{}\n", "a".repeat(79));
        fs::write(dir.join(name), line.repeat(bytes / line.len() + 1))?;
    }
    Ok(dir)
}
