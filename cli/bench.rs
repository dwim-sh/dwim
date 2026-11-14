//! Measuring the model's speed as the context grows: `dwim --bench`.
//!
//! Reading a page and generating a reply both slow down as the conversation
//! deepens, far more than the arithmetic of attention accounts for. This
//! fills the context to each of several depths and, at each, times the
//! model reading a page of tokens and generating a few more, and prints
//! tokens a second for both, so that the slope is a number a change can be
//! measured against.

use std::{error::Error, time::Instant};

use dwim::inference::LanguageModel as _;

use crate::session::{self, Settings, Stderr};

/// The depths of context, in tokens, at which the model is timed: those
/// the context has room for, with the page and the generated tokens after.
const DEPTHS: [usize; 5] = [1 << 10, 8 << 10, 32 << 10, 64 << 10, 128 << 10];

/// Tokens of the page read at each depth: about what a tool's output of
/// two hundred lines takes.
const PAGE: usize = 2048;

/// Tokens generated one at a time at each depth.
const GENERATED: usize = 32;

/// Most tokens fed at once while filling the context, so that the filling
/// can be reported as it goes.
const FILL: usize = 4096;

/// What the context is filled with, over and over: code, as a session's
/// context is mostly.
const FILLER: &str = include_str!("bench.rs");

/// Times the model at each depth of context and prints a table of tokens
/// a second, reading and generating, on standard output.
pub fn bench(settings: &Settings) -> Result<(), Box<dyn Error>> {
    let mut stderr = Stderr::new();
    let mut model = session::load_model(settings, 1, |status| stderr.report(&status))?;
    let depths = depths(model.max_len());
    if depths.is_empty() {
        return Err(format!(
            "a context of {} tokens has no room for a depth of {} with a page of {PAGE} and {GENERATED} generated tokens after it",
            model.max_len(),
            DEPTHS[0]
        )
        .into());
    }
    let filler = model.format().tokenizer.encode(FILLER)?;
    let token = |i: usize| filler[i % filler.len()];
    println!("{:>8}  {:>12}  {:>12}", "context", "read", "generate");
    let mut pos = 0;
    for depth in depths {
        while pos < depth {
            let n = FILL.min(depth - pos);
            let tokens: Vec<u32> = (pos..pos + n).map(token).collect();
            model.forward(&tokens, pos);
            pos += n;
            stderr.stage(
                format!("filling the context to {depth}"),
                &format!("{pos}/{depth}"),
                pos >= depth,
            );
        }
        let page: Vec<u32> = (pos..pos + PAGE).map(token).collect();
        let start = Instant::now();
        let mut logits = model.forward(&page, pos);
        let read = PAGE as f64 / start.elapsed().as_secs_f64();
        pos += PAGE;
        let start = Instant::now();
        for _ in 0..GENERATED {
            logits = model.forward(&[likeliest(&logits)], pos);
            pos += 1;
        }
        let generate = GENERATED as f64 / start.elapsed().as_secs_f64();
        println!("{depth:>8}  {read:>6.1} tok/s  {generate:>6.1} tok/s");
    }
    Ok(())
}

/// The depths to time in a context of `max_len` tokens: those of
/// [`DEPTHS`] with room for the page and the generated tokens after.
fn depths(max_len: usize) -> Vec<usize> {
    DEPTHS
        .into_iter()
        .filter(|depth| depth + PAGE + GENERATED <= max_len)
        .collect()
}

/// The token the logits make likeliest.
fn likeliest(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(i, _)| i as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn times_the_depths_the_context_has_room_for() {
        assert_eq!(depths(2048), Vec::<usize>::new());
        assert_eq!(depths(4096), [1024]);
        assert_eq!(depths(16384), [1024, 8192]);
        assert_eq!(depths(1 << 18), DEPTHS);
    }
}
