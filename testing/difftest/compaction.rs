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

use std::{
    io::Write,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread,
};

use dwim::harness::{Entry, compaction::select};
use dwim::model::Turn;

use crate::Result;

/// Conversations the code is compared with the spec on.
const CASES: u64 = 10_000;

/// Compares the code with the spec on [`CASES`] conversations, and says
/// whether they agreed on all.
pub fn run(mut args: impl Iterator<Item = String>) -> Result<bool> {
    if let Some(arg) = args.next() {
        return Err(format!("compaction takes no arguments, not '{arg}'").into());
    }
    let cases: Vec<Case> = (0..CASES).map(Case::new).collect();
    let spec = oracle(&cases.iter().map(Case::written).collect::<String>())?;
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
        let mut rng = Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
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

/// What the spec's oracle answers to `input`, a transcript a line.
fn oracle(input: &str) -> Result<String> {
    let specs = Path::new(env!("CARGO_MANIFEST_DIR")).join("specs");
    let mut child = Command::new(lake()?)
        .args(["exe", "compaction-oracle"])
        .current_dir(specs)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    // The oracle answers as it reads, so it is fed from another thread
    // while its answers are read here, or both pipes could fill.
    let mut stdin = child
        .stdin
        .take()
        .ok_or("the oracle's input is not piped")?;
    let input = input.to_string();
    let feeder = thread::spawn(move || stdin.write_all(input.as_bytes()));
    let output = child.wait_with_output()?;
    feeder
        .join()
        .map_err(|_| "the oracle's feeder panicked")??;
    if !output.status.success() {
        return Err(format!(
            "the oracle failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    Ok(String::from_utf8(output.stdout)?)
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

/// Where `lake` is: on the path, or where elan installs it.
fn lake() -> Result<PathBuf> {
    let elan = std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".elan/bin/lake"));
    std::env::var_os("PATH")
        .iter()
        .flat_map(std::env::split_paths)
        .map(|dir| dir.join("lake"))
        .chain(elan)
        .find(|path| path.is_file())
        .ok_or_else(|| "no lake: install a Lean toolchain with elan".into())
}

/// A xorshift generator, so that a case is the same on every run.
struct Rng(u64);

impl Rng {
    /// A number below `n`, or 0 if `n` is.
    fn below(&mut self, n: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 % n.max(1) as u64) as usize
    }

    /// A budget: one of `sizes`, or a token either side of it, two times
    /// in three; any up to past the largest the third time.
    fn around(&mut self, sizes: &[usize]) -> usize {
        let largest = sizes.iter().max().copied().unwrap_or(0);
        if self.below(3) == 0 {
            return self.below(largest + 10);
        }
        let size = sizes[self.below(sizes.len())];
        (size + self.below(3)).saturating_sub(1)
    }
}
