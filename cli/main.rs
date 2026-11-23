mod cmd;
mod config;
mod opts;
mod session;
mod stderr;
mod tui;
mod units;

use std::error::Error;

use crate::{
    cmd::{bench, download, once, serve, shell},
    opts::{Command, Opts},
};

fn main() {
    let opts = Opts::parse();
    if let Err(e) = run(opts) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run(opts: Opts) -> Result<(), Box<dyn Error>> {
    let settings = opts.settings();
    match opts.command() {
        Command::Version => {
            println!("dwim {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        Command::Download => download::run(&settings),
        Command::Bench => bench::run(&settings),
        Command::Serve => serve::run(&settings),
        Command::Shell => shell::run(settings, opts.thinking, opts.transcript.as_deref()),
        Command::Once(prompt) => {
            once::run(&settings, &prompt, opts.stats, opts.transcript.as_deref())
        }
    }
}
