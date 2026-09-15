mod fetch;
mod models;
mod opts;
mod run;
mod tui;

use opts::Opts;

fn main() {
    let opts: Opts = argh::from_env();

    let prompt = opts.prompt.join(" ");
    if let Err(e) = run::run(&opts.model, opts.device, &prompt) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
