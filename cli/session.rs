//! A session with the model, for the shell and for a one-off prompt alike:
//! the settings it runs with, opening it on the server that has the model,
//! and the harness around it.

use std::{
    env,
    error::Error,
    io::{self, IsTerminal, Write},
    path::PathBuf,
    time::{Duration, Instant},
};

use dwim::harness::{self, Harness, Log};
use dwim::inference::{Model, Progress};
use dwim::model::{Device, Tally};
use dwim::providers::dwim::{self as server, Remote};

use crate::model;

pub use dwim::server::protocol::Status;

/// How the model runs, as the command line sets it.
pub struct Settings {
    /// The model's name, which is the one model's.
    pub model: String,
    pub device: Device,
    /// Tokens of conversation to make room for in each session, or as
    /// many as fit.
    pub context: Option<usize>,
    /// Sessions to make room for at once, or the default.
    pub sessions: Option<usize>,
    /// Most tokens of a thought, if they are limited.
    pub thinking_budget: Option<usize>,
    pub effort: harness::Effort,
    /// The sampling temperature, or the model file's.
    pub temperature: Option<f32>,
    /// Tokens the model drafts after each token picked, or none.
    pub speculate: usize,
    /// The program the model's shell commands run through, or `sh`.
    pub shell: Option<PathBuf>,
}

impl Settings {
    /// The server the settings ask for.
    fn server(&self) -> server::Options {
        server::Options {
            model: self.model.clone(),
            device: self.device,
            context: self.context,
            sessions: self.sessions,
        }
    }
}

/// The harness around the model, ready for a message.
pub struct Loaded {
    pub harness: Harness<Remote>,
    /// How long getting the model ready took: reaching the server, the
    /// loading if it was loading, and the system prompt.
    pub loading: Duration,
}

/// Fetches the model's files if they are not yet there, reporting the
/// download to `on_status`, and returns the directory they are in.
pub fn fetch_model(
    settings: &Settings,
    mut on_status: impl FnMut(Status),
) -> Result<PathBuf, Box<dyn Error>> {
    let dir = Model::dir(&settings.model)?;
    Model::fetch(&dir, |progress| {
        on_status(Status::Downloading {
            file: progress.file.to_string(),
            done: progress.done,
            total: progress.total,
        })
    })?;
    Ok(dir)
}

/// Fetches the model if needed, opens the device, and loads the model
/// onto it in this process, sized for `sessions` sequences, reporting
/// each stage to `on_status`: what the server does, and `bench`.
pub fn load_model(
    settings: &Settings,
    sessions: usize,
    mut on_status: impl FnMut(Status),
) -> Result<Model, Box<dyn Error>> {
    let dir = fetch_model(settings, &mut on_status)?;
    Model::load(
        &dir,
        settings.device,
        settings.context,
        sessions,
        |progress| {
            on_status(match progress {
                Progress::Device(name) => Status::Device(name),
                Progress::Sized(size) => Status::Sized(size),
                Progress::Weights { done, total } => Status::Weights { done, total },
            })
        },
    )
}

/// Reaches the server, starting one if none is running, opens a session
/// with the system prompt for the working directory, and builds the
/// harness around it with `settings`, keeping a transcript in `log`.
/// `on_status` is told what the server is doing meanwhile, and after, when
/// the session waits for another's request.
pub fn load(
    settings: &Settings,
    log: Log,
    on_status: impl FnMut(Status) + 'static,
) -> Result<Loaded, Box<dyn Error>> {
    let loading = Instant::now();
    let (session, system) = open(settings, on_status)?;
    let device = session.device().to_string();
    let cwd = env::current_dir()?;
    let mut harness = Harness::new(session, &cwd, {
        let system = system.clone();
        move |session: &mut Remote| session.start(&system, |_, _| {}).map(|_| ())
    })?;
    harness.set_thinking_budget(settings.thinking_budget);
    if let Some(shell) = settings.shell.clone() {
        harness.set_shell(shell);
    }
    harness.keep_log(log, &settings.model, &device, &system);
    Ok(Loaded {
        harness,
        loading: loading.elapsed(),
    })
}

/// Opens the session over on a server, starting one if none is running,
/// after the one it was on stopped, and reads the conversation so far
/// back into it. `on_status` is told what the server is doing, as with
/// [`load`].
pub fn reopen(
    settings: &Settings,
    harness: &mut Harness<Remote>,
    on_status: impl FnMut(Status) + 'static,
) -> Result<(), Box<dyn Error>> {
    let (session, _) = open(settings, on_status)?;
    harness.reopen(session)
}

/// Reaches the server, starting one if none is running, and opens a
/// session on it with the system prompt for the working directory, which
/// it returns with the session. `on_status` is told what the server is
/// doing meanwhile, and after, when the session waits for another's
/// request.
fn open(
    settings: &Settings,
    mut on_status: impl FnMut(Status) + 'static,
) -> Result<(Remote, String), Box<dyn Error>> {
    let connection = server::connect(&settings.server(), &mut on_status)?;
    on_status(Status::Opening);
    // A session that connects once the model is loaded, or part way
    // through, is told of the device and the size afterwards instead.
    let (mut device_told, mut size_told) = (false, false);
    let mut session = connection.open(settings.temperature, settings.speculate, |status| {
        match status {
            Status::Device(_) => device_told = true,
            Status::Sized(_) => size_told = true,
            _ => {}
        }
        on_status(status)
    })?;
    if !device_told {
        on_status(Status::Device(session.device().to_string()));
    }
    if !size_told {
        on_status(Status::Sized(session.size()));
    }
    let system = harness::system_prompt(&env::current_dir()?, settings.effort);
    session.start(&system, |read, total| {
        on_status(Status::Prompting { read, total })
    })?;
    session.on_status(on_status);
    Ok((session, system))
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
    lines
        .into_iter()
        .map(|line| line.trim_end().to_string())
        .collect()
}

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
                    model::bytes(*done),
                    model::bytes(total.unwrap_or(*done))
                ),
                total.is_some_and(|total| *done >= total),
            ),
            Status::Device(name) => self.device.clone_from(name),
            Status::Sized(size) => eprintln!("{}", model::describe(size)),
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
