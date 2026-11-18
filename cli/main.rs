mod bench;
mod config;
mod model;
mod once;
mod opts;
mod serve;
mod session;
mod shell;
mod tui;

use std::error::Error;

use dwim::{harness::Log, inference::Model, providers::Kind, sys::memory};

use opts::Opts;
use session::Stderr;

fn main() {
    let opts = Opts::parse();
    if opts.version {
        println!("dwim {}", env!("CARGO_PKG_VERSION"));
        return;
    }
    if let Err(e) = run(opts) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

/// What dwim does with its options: download the model, time it, run the
/// server, or run a session, with the prompt if there is one.
fn run(opts: Opts) -> Result<(), Box<dyn Error>> {
    let settings = opts.settings();
    let prompt = opts.prompt.join(" ");
    // Running the model here takes the memory for it; a session at a
    // provider takes none, and one with nowhere to run starts without
    // a model, for the shell's `/login` to sign in.
    let local = opts.download || opts.bench || opts.serve;
    if local || session::provider(settings.provider, settings.model)? == Some(Kind::Local) {
        memory::check()?;
    }
    if opts.download {
        let mut stderr = Stderr::new();
        let dir = session::fetch_model(&settings, |loading| stderr.report(&loading))?;
        println!("{}", Model::weights(&dir).display());
        return Ok(());
    }
    if opts.bench {
        return bench::bench(&settings);
    }
    if opts.serve {
        return serve::serve(&settings);
    }
    let log = Log::open(opts.transcript.as_deref())?;
    match prompt.trim() {
        "" => shell::run(settings, opts.thinking, log),
        prompt => once::once(&settings, prompt, opts.stats, log),
    }
}
