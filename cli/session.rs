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
use dwim::model::{Device, Tally, Variant};
use dwim::providers::{
    Any, Conversation, Kind, cerebras,
    dwim::{self as server},
    openrouter, vercel,
};
use dwim::sys::memory;

use crate::{
    model,
    shell::picker::{Choice, Group},
};

pub use dwim::server::protocol::Status;

/// How the model runs, as the command line sets it.
#[derive(Clone)]
pub struct Settings {
    /// The model to run.
    pub model: Variant,
    /// Where it runs, if the command line says.
    pub provider: Option<Kind>,
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
            model: self.model.name().to_string(),
            device: self.device,
            context: self.context,
            sessions: self.sessions,
        }
    }
}

/// The session the harness runs on: a conversation at whichever provider
/// the model runs at.
pub type Session = Conversation<Any>;

/// The harness around the model, ready for a message.
pub struct Loaded {
    pub harness: Harness<Session>,
    /// How long getting the model ready took: reaching the server, the
    /// loading if it was loading, and the system prompt.
    pub loading: Duration,
}

/// Where `model` runs: at `provider` if the command line named one, and
/// otherwise on this machine if it can, or at a host there is a key to:
/// OpenRouter, or else Vercel's AI Gateway, or else Cerebras. The large
/// model takes more memory than a machine has, and runs only at
/// OpenRouter or Vercel's AI Gateway. Nowhere, if the
/// command line named no provider and the model can run neither here nor
/// at a host without a key: the shell starts without a model then, for
/// `/provider` to sign in.
pub fn provider(provider: Option<Kind>, model: Variant) -> Result<Option<Kind>, Box<dyn Error>> {
    match (provider, model) {
        (Some(Kind::Local), Variant::Large) => Err(format!(
            "{} does not run on this machine: it runs with --provider openrouter, or vercel",
            Variant::Large
        )
        .into()),
        (Some(Kind::Cerebras), Variant::Large) => Err(format!(
            "{} does not run at Cerebras: it runs with --provider openrouter, or vercel",
            Variant::Large
        )
        .into()),
        (Some(kind), _) => Ok(Some(kind)),
        (None, Variant::Medium) if memory::check().is_ok() => Ok(Some(Kind::Local)),
        (None, model) => Ok(keyed(model)),
    }
}

/// The host of `model` there is a key to, if there is one: OpenRouter
/// ahead of Vercel's AI Gateway, and both ahead of Cerebras, which runs
/// the medium model only.
fn keyed(model: Variant) -> Option<Kind> {
    if openrouter::key().is_some() {
        Some(Kind::OpenRouter)
    } else if vercel::key().is_some() {
        Some(Kind::Vercel)
    } else if model == Variant::Medium && cerebras::key().is_some() {
        Some(Kind::Cerebras)
    } else {
        None
    }
}

/// Why `model` has nowhere to run, when [`provider`] finds none: why it
/// does not run on this machine, which leaves the hosts, without a key.
pub fn nowhere(model: Variant) -> String {
    match (model, memory::check()) {
        (Variant::Medium, Err(e)) => e.to_string(),
        (Variant::Medium, Ok(())) => {
            format!("{model} runs only at OpenRouter, Vercel AI Gateway or Cerebras")
        }
        _ => format!("{model} runs only at OpenRouter or Vercel AI Gateway"),
    }
}

/// The providers, each with the models it runs, for the model picker: the
/// large model only at OpenRouter and Vercel's AI Gateway. This machine cannot be used without the
/// memory for a model, nor a host without a key to it.
pub fn providers() -> Vec<Group> {
    vec![
        Group {
            provider: Kind::Local,
            models: vec![Variant::Medium],
            unavailable: memory::check().err().map(|e| e.to_string()),
        },
        Group {
            provider: Kind::OpenRouter,
            models: vec![Variant::Medium, Variant::Large],
            unavailable: openrouter::key()
                .is_none()
                .then(|| "/provider signs in".to_string()),
        },
        Group {
            provider: Kind::Vercel,
            models: vec![Variant::Medium, Variant::Large],
            unavailable: vercel::key().is_none().then(no_vercel_key),
        },
        Group {
            provider: Kind::Cerebras,
            models: vec![Variant::Medium],
            unavailable: cerebras::key().is_none().then(no_cerebras_key),
        },
    ]
}

/// The providers, each with what picking it comes to, for the provider
/// picker, where the session's model runs at `current`. This machine
/// cannot be picked without the memory for a model; OpenRouter can be
/// without a key to it, which picking it signs in for, and picking it
/// while on it signs in again. Vercel's AI Gateway and Cerebras have no
/// signing in, and cannot be picked without a key in the environment.
pub fn provider_choices(current: Option<Kind>) -> Vec<Choice> {
    vec![
        Choice {
            provider: Kind::Local,
            name: "This machine",
            note: format!("{} in the dwim server", Variant::Medium),
            unavailable: memory::check().err().map(|e| e.to_string()),
        },
        Choice {
            provider: Kind::OpenRouter,
            name: "OpenRouter",
            note: match openrouter::key() {
                Some(_) if current == Some(Kind::OpenRouter) => {
                    "signed in · pick to sign in again".to_string()
                }
                Some(_) => "signed in".to_string(),
                None => "sign in with your browser to get a key".to_string(),
            },
            unavailable: None,
        },
        Choice {
            provider: Kind::Vercel,
            name: vercel::HOST.name,
            note: format!("key in {}", vercel::KEY),
            unavailable: vercel::key().is_none().then(no_vercel_key),
        },
        Choice {
            provider: Kind::Cerebras,
            name: cerebras::HOST.name,
            note: format!("{} only · key in {}", Variant::Medium, cerebras::KEY),
            unavailable: cerebras::key().is_none().then(no_cerebras_key),
        },
    ]
}

/// What to do to use Vercel's AI Gateway where there is no key to it.
fn no_vercel_key() -> String {
    format!("set {} to a key from the Vercel dashboard", vercel::KEY)
}

/// What to do to use Cerebras where there is no key to it.
fn no_cerebras_key() -> String {
    format!(
        "set {} to a key from the Cerebras cloud console",
        cerebras::KEY
    )
}

/// Fetches the model's files if they are not yet there, reporting the
/// download to `on_status`, and returns the directory they are in.
pub fn fetch_model(
    settings: &Settings,
    mut on_status: impl FnMut(Status),
) -> Result<PathBuf, Box<dyn Error>> {
    let dir = Model::dir(settings.model.name())?;
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
    on_status: impl FnMut(Status) + Send + 'static,
) -> Result<Loaded, Box<dyn Error>> {
    let loading = Instant::now();
    let (session, system) = open(settings, on_status)?;
    let harness = harness(settings, session, system, log)?;
    Ok(Loaded {
        harness,
        loading: loading.elapsed(),
    })
}

/// Builds the harness around `session`, opened with the system prompt
/// `system`, with `settings`, keeping a transcript in `log`.
pub fn harness(
    settings: &Settings,
    mut session: Session,
    system: String,
    log: Log,
) -> Result<Harness<Session>, Box<dyn Error>> {
    let device = device(&mut session);
    let cwd = env::current_dir()?;
    let mut harness = Harness::new(session, &cwd, {
        let system = system.clone();
        move |session: &mut Session| session.start(&system, |_, _| {})
    })?;
    harness.set_thinking_budget(settings.thinking_budget);
    if let Some(shell) = settings.shell.clone() {
        harness.set_shell(shell);
    }
    harness.keep_log(log, settings.model.name(), &device, &system);
    Ok(harness)
}

/// Opens the session over on a server, starting one if none is running,
/// after the one it was on stopped, and reads the conversation so far
/// back into it. `on_status` is told what the server is doing, as with
/// [`load`].
pub fn reopen(
    settings: &Settings,
    harness: &mut Harness<Session>,
    on_status: impl FnMut(Status) + Send + 'static,
) -> Result<(), Box<dyn Error>> {
    let (session, _) = open(settings, on_status)?;
    harness.reopen(session)
}

/// Goes on with the conversation on the model `settings` now name, where
/// that one runs, in place of the model it was on: opens a session there,
/// reads the conversation so far back into it, and notes the change in
/// the transcript. Returns where the model runs. `on_status` is told what
/// the server is doing, as with [`load`].
pub fn switch(
    settings: &Settings,
    harness: &mut Harness<Session>,
    on_status: impl FnMut(Status) + Send + 'static,
) -> Result<Kind, Box<dyn Error>> {
    reopen(settings, harness, on_status)?;
    let device = device(harness.session());
    harness.note_model(settings.model.name(), &device);
    Ok(harness.session().provider().kind())
}

/// What a session's model runs on, as the transcript names it.
fn device(session: &mut Session) -> String {
    match session.provider() {
        Any::Dwim(remote) => remote.device().to_string(),
        Any::Hosted(hosted) => format!("{} · {}", hosted.host().name, hosted.model()),
    }
}

/// Opens a session where the model runs, with the system prompt for the
/// working directory, which it returns with the session: on the server,
/// reached, or started if none is running, with `on_status` told what the
/// server is doing meanwhile, and after, when the session waits for
/// another's request; or at a host, where there is nothing to wait for.
pub fn open(
    settings: &Settings,
    mut on_status: impl FnMut(Status) + Send + 'static,
) -> Result<(Session, String), Box<dyn Error>> {
    let system = harness::system_prompt(&env::current_dir()?, settings.effort);
    let Some(kind) = provider(settings.provider, settings.model)? else {
        return Err(format!(
            "{}: run it at OpenRouter by signing in with /provider in dwim, or by setting {}, or at Vercel AI Gateway by setting {}{}",
            nowhere(settings.model),
            openrouter::KEY,
            vercel::KEY,
            match settings.model {
                Variant::Medium => format!(", or at Cerebras by setting {}", cerebras::KEY),
                Variant::Large => String::new(),
            }
        )
        .into());
    };
    let (model, context, temperature) = (settings.model, settings.context, settings.temperature);
    let hosted = match kind {
        Kind::Local => None,
        Kind::OpenRouter => Some(openrouter::open(model, context, temperature)?),
        Kind::Vercel => Some(vercel::open(model, context, temperature)?),
        Kind::Cerebras => Some(cerebras::open(model, context, temperature)?),
    };
    if let Some(hosted) = hosted {
        let mut session = Conversation::new(Any::Hosted(hosted));
        session.start(&system, |_, _| {})?;
        return Ok((session, system));
    }
    let connection = server::connect(&settings.server(), &mut on_status)?;
    on_status(Status::Opening);
    // A session that connects once the model is loaded, or part way
    // through, is told of the device and the size afterwards instead.
    let (mut device_told, mut size_told) = (false, false);
    let remote = connection.open(settings.temperature, settings.speculate, |status| {
        match status {
            Status::Device(_) => device_told = true,
            Status::Sized(_) => size_told = true,
            _ => {}
        }
        on_status(status)
    })?;
    if !device_told {
        on_status(Status::Device(remote.device().to_string()));
    }
    if !size_told {
        on_status(Status::Sized(remote.size()));
    }
    let mut session = Conversation::new(Any::Dwim(remote));
    session.start(&system, |read, total| {
        on_status(Status::Prompting { read, total })
    })?;
    if let Some(remote) = session.provider().dwim() {
        remote.on_status(on_status);
    }
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
    // What a provider charged, where the model runs at one.
    if m.cost > 0.0 {
        lines.push(format!("{:<8}{:>13}", "cost", format!("${:.4}", m.cost)));
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
