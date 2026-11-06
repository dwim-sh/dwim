//! Answering one prompt and exiting, without the shell.
//!
//! The model's reply goes to standard output and nothing else does, so that
//! a script can read the reply on its own. Getting ready, what the model is
//! thinking, the commands it runs, and what they printed all go to standard
//! error.

use std::{
    error::Error,
    io::{self, Write},
    ops::ControlFlow,
    path::Path,
    time::Instant,
};

use dwim_harness::{Event, Log};

use crate::session::{self, Settings, Stderr};

/// Answers `prompt` with the model, running the tools it calls, and returns
/// once it replies with text alone, keeping a transcript of it in `log`;
/// then reports where the time went, if `stats` asks for it.
pub fn once(
    settings: &Settings,
    prompt: &str,
    stats: bool,
    log: Log,
) -> Result<(), Box<dyn Error>> {
    let started = Instant::now();
    let transcript = log.path().map(Path::to_path_buf);
    let mut stderr = Stderr::new();
    let mut loaded = session::load(settings, log, |loading| stderr.report(&loading))?;
    if let Some(path) = transcript {
        eprintln!("transcript in {}", path.display());
    }
    // No one is here to answer the model, so it goes on by itself.
    loaded.harness.set_auto(true);
    let mut printer = Printer::default();
    loaded.harness.send(prompt, |event| {
        printer.print(event);
        ControlFlow::Continue(())
    })?;
    drop(printer);
    if stats {
        for line in loaded
            .harness
            .stats()
            .report(loaded.loading, started.elapsed())
        {
            eprintln!("{line}");
        }
    }
    Ok(())
}

/// Prints a turn as it happens: the reply to standard output, and the
/// thought behind it, the tools it runs, and the note it writes along the
/// way to standard error.
#[derive(Default)]
struct Printer {
    /// Whether anything of the reply has been printed, so that the blank
    /// lines the model opens with are left out.
    replied: bool,
    /// Whether standard error is part way through a line of thought.
    thinking: bool,
}

impl Printer {
    fn print(&mut self, event: Event) {
        if let Some(notice) = event.notice() {
            self.end_thought();
            eprintln!("[{notice}]");
            return;
        }
        match event {
            Event::Thought(text) | Event::Note(text) => {
                let text = if self.thinking {
                    text.as_str()
                } else {
                    text.trim_start()
                };
                if !text.is_empty() {
                    eprint!("{text}");
                    let _ = io::stderr().flush();
                    self.thinking = true;
                }
            }
            Event::Text(text) => {
                self.end_thought();
                let text = if self.replied {
                    text.as_str()
                } else {
                    text.trim_start()
                };
                if !text.is_empty() {
                    print!("{text}");
                    let _ = io::stdout().flush();
                    self.replied = true;
                }
            }
            Event::Call { name, detail } => {
                self.end_thought();
                eprintln!("{name}({detail})");
            }
            Event::Output(output) => eprintln!("{output}"),
            Event::Token | Event::Cut | Event::Compacting | Event::Compacted { .. } => {}
        }
    }

    /// Ends the line of thought being printed, if there is one.
    fn end_thought(&mut self) {
        if self.thinking {
            eprintln!();
            self.thinking = false;
        }
    }
}

impl Drop for Printer {
    /// Ends the reply with a newline, as a command's output should end.
    fn drop(&mut self) {
        self.end_thought();
        if self.replied {
            println!();
        }
    }
}
