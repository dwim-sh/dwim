//! The server that has the model: `dwim --serve`.
//!
//! A session starts one itself when none is listening, detached from its
//! terminal with its standard error in a log file, so this is for running
//! the server by hand: in the foreground, with what it reports on standard
//! error. It listens before it loads the model, so that the sessions that
//! connect meanwhile see the loading, and exits once it has gone a while
//! without a session.

use std::error::Error;

use dwim::{
    agent,
    inference::Loaded,
    server::{self, Hello, Server},
};

use crate::{session::Settings, stderr::Stderr};

/// Runs the server with the model as `settings` have it, until it exits.
pub fn run(settings: &Settings) -> Result<(), Box<dyn Error>> {
    let path = server::socket_path();
    let sessions = settings.sessions.unwrap_or(Loaded::SESSIONS);
    if sessions == 0 {
        return Err("a server needs room for one session at least".into());
    }
    let hello = Hello {
        model: settings.model.name().to_string(),
        device: settings.device.to_string(),
        context: settings.context,
        room: sessions,
    };
    let server = Server::bind(&path, hello)?;
    eprintln!("listening on {}", path.display());
    let mut stderr = Stderr::new();
    agent::serve(server, settings, sessions, |status| stderr.report(status))
}
