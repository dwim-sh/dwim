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
    match opts.command() {
        Command::Version => {
            println!("dwim {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        Command::Download => download::run(&opts.settings()?),
        Command::Bench => bench::run(&opts.settings()?),
        Command::Serve => serve::run(&opts.settings()?),
        Command::Shell => shell::run(opts.settings()?, opts.thinking, opts.transcript.as_deref()),
        Command::Once(prompt) => once::run(
            &opts.settings()?,
            &prompt,
            opts.stats,
            opts.transcript.as_deref(),
        ),
    }
}
