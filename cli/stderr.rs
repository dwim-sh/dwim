//! What the server is doing, told on standard error, for the commands
//! without the shell: `download`, `bench`, `serve`, and a prompt answered
//! once.

use std::io::{self, IsTerminal, Write};

use dwim::server::protocol::Status;

use crate::units;

/// What the server is doing, as it is told on standard error: a line
/// rewritten as it goes when standard error is a terminal, and one line
/// for each stage when it is not.
pub struct Stderr {
    terminal: bool,
    stage: String,
    /// The device the model loads onto, once it is open.
    device: String,
    /// Whether this is a session that started the server, which reports
    /// the server's loading as its starting up.
    starting: bool,
}

impl Stderr {
    pub fn new() -> Self {
        Self {
            terminal: io::stderr().is_terminal(),
            stage: String::new(),
            device: String::new(),
            starting: false,
        }
    }

    /// Tells what the server is doing.
    pub fn report(&mut self, status: &Status) {
        match status {
            Status::Downloading { file, done, total } => self.stage(
                format!("downloading {file}"),
                &format!(
                    "{}/{}",
                    units::bytes(*done),
                    units::bytes(total.unwrap_or(*done))
                ),
                total.is_some_and(|total| *done >= total),
            ),
            Status::Device(name) => self.device.clone_from(name),
            Status::Sized(size) => eprintln!("{}", units::describe(size)),
            Status::Weights { done, total } if self.starting => self.stage(
                "starting dwim server".to_string(),
                &format!("{}%", done * 100 / (*total).max(1)),
                done >= total,
            ),
            Status::Weights { done, total } => self.stage(
                if self.device.is_empty() {
                    "loading the model".to_string()
                } else {
                    format!("loading the model on {}", self.device)
                },
                &format!("{done}/{total}"),
                done >= total,
            ),
            Status::Prompting { read, total } => self.stage(
                "reading the system prompt".to_string(),
                &format!("{read}/{total}"),
                read >= total,
            ),
            Status::Restarting => self.stage("restarting dwim server".to_string(), "", false),
            Status::Starting => {
                self.starting = true;
                self.stage("starting dwim server".to_string(), "", false)
            }
            Status::Opening => {}
            Status::Busy => self.stage(
                "waiting for another session's turn with the model".to_string(),
                "",
                true,
            ),
        }
    }

    /// Tells how far along `stage` is, with `progress` as its measure, and
    /// whether it is `done`.
    pub fn stage(&mut self, stage: String, progress: &str, done: bool) {
        if self.stage != stage {
            self.stage = stage;
            if !self.terminal {
                eprintln!("{}…", self.stage);
            }
        }
        if self.terminal {
            eprint!("\r\x1b[K{}… {progress}", self.stage);
            if done {
                eprintln!();
            }
            let _ = io::stderr().flush();
        }
    }
}
