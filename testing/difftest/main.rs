//! Differential testing: code that has to be fast or practical, checked
//! against a reference that is simpler and trusted, on the same inputs,
//! where any disagreement is a bug in the code. Each subject has its own
//! reference:
//!
//! - `difftest model [--model MODEL] [PROMPT...] [--repeat N]`: the model
//!   on the GPU against the model on the CPU, with the real weights.
//! - `difftest compaction`: the harness's compaction against its Lean spec
//!   in `specs/`, on generated conversations.

mod compaction;
mod model;
mod spec;

use std::{error::Error, process::ExitCode};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

/// How the subjects are asked for.
const USAGE: &str = "usage: difftest model [--model MODEL] [PROMPT...] [--repeat N]
       difftest compaction";

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let agreed = match args.next().as_deref() {
        Some("model") => model::run(args),
        Some("compaction") => compaction::run(args),
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
