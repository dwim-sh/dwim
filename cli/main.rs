mod fetch;
mod models;
mod once;
mod opts;
mod run;
mod tui;

use opts::Opts;

fn main() {
    let opts: Opts = argh::from_env();
    if opts.version {
        println!("dwim {}", env!("CARGO_PKG_VERSION"));
        return;
    }

    // A prompt on the command line is a one-off: answer it and exit, with
    // nothing on standard output but the reply. Without one, open the shell.
    let prompt = opts.prompt.join(" ");
    let result = match prompt.trim() {
        "" => run::run(&opts.model, opts.device, opts.context, opts.thinking),
        prompt => once::once(&opts.model, opts.device, opts.context, prompt, opts.stats),
    };
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
