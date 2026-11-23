mod bench;
mod config;
mod download;
mod model;
mod once;
mod opts;
mod serve;
mod session;
mod shell;
mod tui;

use std::error::Error;

use dwim::harness::Log;

use opts::{Command, Opts};

fn main() {
    let opts = Opts::parse();
    if let Err(e) = run(opts) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run(opts: Opts) -> Result<(), Box<dyn Error>> {
    let settings = opts.settings();
    let log = || Log::open(opts.transcript.as_deref());
    match opts.command() {
        Command::Version => {
            println!("dwim {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        Command::Download => download::download(&settings),
        Command::Bench => bench::bench(&settings),
        Command::Serve => serve::serve(&settings),
        Command::Shell => shell::run(settings, opts.thinking, log()?),
        Command::Once(prompt) => once::once(&settings, &prompt, opts.stats, log()?),
    }
}
