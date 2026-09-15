mod fetch;
mod models;
mod opts;
mod run;
mod tui;

use opts::Opts;

fn main() {
    let opts: Opts = argh::from_env();

    if let Err(e) = run::run(&opts.model, opts.device) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
