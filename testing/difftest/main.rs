//! Differential testing: code that has to be fast or practical, checked
//! against a reference that is simpler and trusted, on the same inputs,
//! where any disagreement is a bug in the code. It is the differential
//! random testing of "How we built Cedar with automated reasoning and
//! differential testing" (Amazon Science, 2023).
//!
//! Each subject has its own reference:
//!
//! - `difftest model [PROMPT...] [--repeat N]`: the model on the GPU
//!   against the model on the CPU, with the real weights.
//! - `difftest compaction`: the harness's compaction against its Lean
//!   spec in `specs/`, on generated conversations.
//! - `difftest harness`: the harness's turns against the Lean spec of a
//!   turn, which says every call is answered however a turn ends.

mod compaction;
mod harness;
mod model;
mod spec;

use std::{error::Error, process::ExitCode};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

/// How the subjects are asked for.
const USAGE: &str = "usage: difftest model [PROMPT...] [--repeat N]\n       difftest compaction\n       difftest harness";

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let agreed = match args.next().as_deref() {
        Some("model") => model::run(args),
        Some("compaction") => compaction::run(args),
        Some("harness") => harness::run(args),
        _ => Err(USAGE.into()),
    };
    match agreed {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("difftest: {e}");
            ExitCode::FAILURE
        }
    }
}
