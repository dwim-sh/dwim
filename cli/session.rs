//! A session with the model, for the shell and for a one-off prompt alike:
//! the agent's settings with the user's configuration in them, and where a
//! session's time has gone.

use std::{error::Error, io, path::Path, time::Duration};

use dwim::{
    agent,
    harness::{self, Harness, Log},
    model::{Model, Tally},
    providers::Kind,
};

use crate::config::Config;

pub use dwim::agent::{
    Ready, Session, Settings, Status, fetch_model, harness, load_model, ready, runs,
};

/// Where `model` runs, as [`agent::provider`] finds, leaving out the
/// providers the configuration disables. Nowhere, if the command line
/// named no provider and the model can run neither here nor at a host
/// without a key: the shell starts without a model then, for `/login` to
/// sign in.
pub fn provider(provider: Option<Kind>, model: Model) -> Result<Option<Kind>, Box<dyn Error>> {
    agent::provider(provider, model, &Config::load()?.disabled_providers())
}

/// Why `model` has nowhere to run, with the providers `config` disables.
pub fn nowhere(model: Model, config: &Config) -> String {
    agent::nowhere(model, &config.disabled_providers())
}

/// Opens a session and builds the harness around it, as [`agent::load`]
/// does, with the providers the configuration disables left out.
pub fn load(
    settings: &Settings,
    log: Log,
    on_status: impl FnMut(Status) + Send + 'static,
) -> Result<Ready, Box<dyn Error>> {
    agent::load(&configured(settings)?, log, on_status)
}

/// Opens a session where the model runs, as [`agent::open`] does, with
/// the providers the configuration disables left out.
pub fn open(
    settings: &Settings,
    on_status: impl FnMut(Status) + Send + 'static,
) -> Result<(Session, String), Box<dyn Error>> {
    agent::open(&configured(settings)?, on_status)
}

/// Opens the session over, as [`agent::reopen`] does, with the providers
/// the configuration disables left out.
pub fn reopen(
    settings: &Settings,
    harness: &mut Harness<Session>,
    on_status: impl FnMut(Status) + Send + 'static,
) -> Result<(), Box<dyn Error>> {
    agent::reopen(&configured(settings)?, harness, on_status)
}

/// Goes on with the conversation on the model `settings` now name, as
/// [`agent::switch`] does, with the providers the configuration disables
/// left out.
pub fn switch(
    settings: &Settings,
    harness: &mut Harness<Session>,
    on_status: impl FnMut(Status) + Send + 'static,
) -> Result<Kind, Box<dyn Error>> {
    agent::switch(&configured(settings)?, harness, on_status)
}

/// The session's transcript, opened before anything else so that a file
/// it cannot be written to is found out at once: in `path`, if the
/// options name one, or in a new file in the folder for `dir`, the working
/// directory, of the user's state directory. Where that cannot be written,
/// the session goes on without one, with a warning.
pub fn transcript(path: Option<&Path>, dir: &Path) -> io::Result<Log> {
    match path {
        Some(path) => Log::create(path).map_err(|e| {
            io::Error::other(format!(
                "cannot write the transcript to {}: {e}",
                path.display()
            ))
        }),
        None => Ok(Log::in_state(dir).unwrap_or_else(|e| {
            eprintln!("warning: keeping no transcript: {e}");
            Log::none()
        })),
    }
}

/// `settings` with the providers the configuration disables as it is now:
/// the shell changes it while a session runs.
fn configured(settings: &Settings) -> Result<Settings, Box<dyn Error>> {
    Ok(Settings {
        disabled: Config::load()?.disabled_providers(),
        ..settings.clone()
    })
}

/// Where `stats` say a session's time has gone, as lines, given how long loading the model took and
/// how long everything took: what is left over is the time outside the
/// model and the tools.
pub fn accounting(stats: &harness::Stats, loading: Duration, total: Duration) -> Vec<String> {
    let m = &stats.model;
    let row = |name: &str, count: String, seconds: f64, rate: String| {
        format!("{name:<8}{count:>13}{seconds:>8.1} s{rate:>11}")
    };
    let tokens = |name: &str, tally: &Tally| {
        row(
            name,
            format!("{} tokens", tally.tokens),
            tally.seconds,
            format!("{:.0} tok/s", tally.rate()),
        )
    };
    let accounted = loading.as_secs_f64()
        + m.cached.seconds
        + m.prompt.seconds
        + m.thought.seconds
        + m.answer.seconds
        + stats.tool_seconds;
    let mut lines = vec![
        row(
            "loading",
            String::new(),
            loading.as_secs_f64(),
            String::new(),
        ),
        row(
            "cached",
            format!("{} tokens", m.cached.tokens),
            m.cached.seconds,
            String::new(),
        ),
        tokens("prompt", &m.prompt),
        tokens("thought", &m.thought),
        tokens("answer", &m.answer),
        row(
            "tools",
            format!("{} calls", stats.calls),
            stats.tool_seconds,
            String::new(),
        ),
        row(
            "other",
            String::new(),
            (total.as_secs_f64() - accounted).max(0.0),
            String::new(),
        ),
        row("total", String::new(), total.as_secs_f64(), String::new()),
    ];
    // Drafting, after the tokens generated, where the model drafted.
    if m.drafted > 0 {
        lines.insert(
            5,
            format!(
                "{:<8}{:>13}{:>10}{:>11}",
                "drafts",
                format!("{} tokens", m.drafted),
                "",
                format!("{:.0}% kept", 100.0 * m.accepted as f64 / m.drafted as f64)
            ),
        );
    }
    // What a provider charged, where the model runs at one.
    if m.cost > 0.0 {
        lines.push(format!("{:<8}{:>13}", "cost", format!("${:.4}", m.cost)));
    }
    lines
        .into_iter()
        .map(|line| line.trim_end().to_string())
        .collect()
}
