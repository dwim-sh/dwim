mod bench;
mod config;
mod model;
mod once;
mod opts;
mod serve;
mod session;
mod shell;
mod tui;

use std::env;

use argh::{EarlyExit, FromArgs};
use dwim::harness::Log;
use dwim::inference::Model;
use dwim::providers::Kind;
use dwim::sys::memory;

use opts::Opts;
use session::Stderr;

fn main() {
    let args: Vec<String> = env::args().collect();
    let command = args[0].rsplit('/').next().unwrap_or(&args[0]);
    let rest: Vec<&str> = args[1..].iter().map(String::as_str).collect();
    let opts = match Opts::from_args(&[command], &rest) {
        Ok(opts) => opts,
        Err(EarlyExit { output, status }) => {
            match status {
                Ok(()) => print!("{}", opts::help(&output, &rest)),
                Err(()) => eprint!("{output}"),
            }
            std::process::exit(status.map_or(1, |()| 0));
        }
    };
    if opts.version {
        println!("dwim {}", env!("CARGO_PKG_VERSION"));
        return;
    }
    let settings = opts.settings();
    let prompt = opts.prompt.join(" ");
    let result = (|| {
        // Running the model here takes the memory for it; a session at a
        // provider takes none, and one with nowhere to run starts without
        // a model, for the shell's `/provider` to sign in.
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
    })();
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
