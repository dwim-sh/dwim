//! The server that has the model: `dwim --serve`.
//!
//! A session starts one itself when none is listening, detached from its
//! terminal with its standard error in a log file, so this is for running
//! the server by hand: in the foreground, with what it reports on standard
//! error. It listens before it loads the model, so that the sessions that
//! connect meanwhile see the loading, and exits once it has gone a while
//! without a session.

use std::error::Error;

use dwim::models::Format;
use dwim::server::{self, Hello, Server};

use crate::{
    model,
    session::{self, Settings, Stderr},
};

/// Runs the server with the model as `settings` have it, until it exits.
pub fn serve(settings: &Settings) -> Result<(), Box<dyn Error>> {
    let path = server::socket_path();
    let hello = Hello {
        model: settings.model.clone(),
        device: settings.device.to_string(),
        context: settings.context,
    };
    let server = Server::bind(&path, hello)?;
    eprintln!("listening on {}", path.display());
    let mut stderr = Stderr::new();
    let loaded = match session::load_model(settings, |status| {
        stderr.report(&status);
        server.status(&status);
    }) {
        Ok(loaded) => loaded,
        Err(e) => {
            server.fail(&e.to_string());
            return Err(e);
        }
    };
    let format = Format::new(loaded.tokenizer)?;
    let gguf = loaded.gguf;
    let temperature = settings.temperature;
    let sampler = move |asked: Option<f32>| model::sampler(&gguf, asked.or(temperature));
    server.run(loaded.model, format, loaded.device, loaded.size, sampler)
}
