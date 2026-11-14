mod bench;
mod fetch;
mod model;
mod once;
mod opts;
mod serve;
mod session;
mod shell;
mod tui;

use std::{
    env,
    error::Error,
    path::{Path, PathBuf},
};

use argh::{EarlyExit, FromArgs};
use dwim::harness::Log;
use dwim::models::qwen35;
use dwim::sys::{self, memory};

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
    let result = memory::check().and_then(|()| {
        if opts.download {
            let mut stderr = Stderr::new();
            let dir = session::fetch_model(&settings, |loading| stderr.report(&loading))?;
            println!("{}", dir.join(qwen35::WEIGHTS).display());
            return Ok(());
        }
        if opts.bench {
            return bench::bench(&settings);
        }
        if opts.serve {
            return serve::serve(&settings);
        }
        let log = log(opts.transcript)?;
        match prompt.trim() {
            "" => shell::run(settings, opts.thinking, log),
            prompt => once::once(&settings, prompt, opts.stats, log),
        }
    });
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

/// The session's transcript, opened before anything else so that a file
/// it cannot be written to is found out at once: in `path`, if the
/// options name one, or in a new file in the working directory's folder of
/// the user's state directory (`~/.local/state`, or on macOS, where there
/// is none, `~/Library/Application Support`), out of the agent's sight.
/// Where that cannot be written, the session goes on without one, with a
/// warning.
fn log(path: Option<PathBuf>) -> Result<Log, Box<dyn Error>> {
    if let Some(path) = path {
        return Log::create(&path)
            .map_err(|e| format!("cannot write the transcript to {}: {e}", path.display()).into());
    }
    let Some(dir) = sys::dirs::state() else {
        return Ok(Log::none());
    };
    let path = dir
        .join("dwim")
        .join("projects")
        .join(project(&env::current_dir()?))
        .join(Log::name());
    Ok(Log::create(&path).unwrap_or_else(|e| {
        eprintln!(
            "warning: keeping no transcript: cannot write {}: {e}",
            path.display()
        );
        Log::none()
    }))
}

/// The name of the folder for the transcripts of sessions in `dir`: its
/// path with everything but letters and digits made a dash, such as
/// `-home-penberg-src-dwim`.
fn project(dir: &Path) -> String {
    dir.to_string_lossy()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect()
}
