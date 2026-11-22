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

use dwim::harness::Log;

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

fn run(opts: Opts) -> Result<(), Box<dyn Error>> {
    let settings = opts.settings();
    if opts.download {
        let mut stderr = Stderr::new();
        let dir = session::fetch_model(&settings, |loading| stderr.report(&loading))?;
        if let Some(weights) = settings.model.weights() {
            println!("{}", weights.path(&dir).display());
        }
        return Ok(());
    }
    if opts.bench {
        return bench::bench(&settings);
    }
    if opts.serve {
        return serve::serve(&settings);
    }
    let log = Log::open(opts.transcript.as_deref())?;
    match opts.prompt() {
        None => shell::run(settings, opts.thinking, log),
        Some(prompt) => once::once(&settings, &prompt, opts.stats, log),
    }
}
