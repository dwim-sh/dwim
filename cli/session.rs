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
use dwim::inference::{self, Loaded, Progress};
use dwim::model::{Device, Effort, Model, Tally};
use dwim::providers::{
    Any, Conversation, Kind, cerebras,
    dwim::{self as server},
    hosted::{self, Host},
    openrouter, vercel,
};
use dwim::sys::memory;

use crate::{
    config::Config,
    model,
    shell::picker::{Account, Choice, Group},
};

pub use dwim::server::protocol::Status;

/// How the model runs, as the command line sets it.
#[derive(Clone)]
pub struct Settings {
    /// The model to run.
    pub model: Model,
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
    pub effort: Effort,
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
pub struct Ready {
    pub harness: Harness<Session>,
    /// How long getting the model ready took: reaching the server, the
    /// loading if it was loading, and the system prompt.
    pub loading: Duration,
}

/// Where `model` runs: at `provider` if the command line named one, and
/// otherwise on this machine if it can, or at a host there is a key to:
/// OpenRouter, or else Vercel's AI Gateway, or else Cerebras, leaving out
/// the providers the configuration disables. The large model takes more
/// memory than a machine has, and runs only at OpenRouter or Vercel's AI
/// Gateway. Nowhere, if the command line named no provider and the model
/// can run neither here nor at a host without a key: the shell starts
/// without a model then, for `/login` to sign in. A provider the command
/// line names is only returned if the model can run there: a local one
/// needs this machine to have the memory for it.
pub fn provider(provider: Option<Kind>, model: Model) -> Result<Option<Kind>, Box<dyn Error>> {
    match (provider, model) {
        (Some(Kind::Local), Model::QwenMax) => Err(format!(
            "{} does not run on this machine: it runs with --provider openrouter, or vercel",
            Model::QwenMax
        )
        .into()),
        (Some(Kind::Cerebras), Model::QwenMax) => Err(format!(
            "{} does not run at Cerebras: it runs with --provider openrouter, or vercel",
            Model::QwenMax
        )
        .into()),
        (Some(Kind::Local), _) => {
            memory::check()?;
            Ok(Some(Kind::Local))
        }
        (Some(kind), _) => Ok(Some(kind)),
        (None, model) => {
            let config = Config::load()?;
            Ok(KINDS
                .into_iter()
                .find(|&kind| !config.disabled(kind) && ready(kind, model)))
        }
    }
}

/// The providers, in the order a model is run at the first that can run
/// it, and in which the pickers show them.
const KINDS: [Kind; 4] = [Kind::Local, Kind::OpenRouter, Kind::Vercel, Kind::Cerebras];

/// Whether `model` can run at `kind` as things are: on this machine with
/// the memory for it, at a host that runs it with a key to the host.
fn ready(kind: Kind, model: Model) -> bool {
    runs(kind, model)
        && match kind {
            Kind::Local => memory::check().is_ok(),
            Kind::OpenRouter => openrouter::key().is_some(),
            Kind::Vercel => vercel::key().is_some(),
            Kind::Cerebras => cerebras::key().is_some(),
        }
}

/// Whether `kind` is a provider of `model`: all are of the medium model,
/// OpenRouter and Vercel's AI Gateway of the large one.
fn runs(kind: Kind, model: Model) -> bool {
    match kind {
        Kind::Local | Kind::Cerebras => model == Model::Qwen27b,
        Kind::OpenRouter | Kind::Vercel => true,
    }
}

/// Why `model` has nowhere to run, when [`provider`] finds none: providers
/// of it are disabled in `config`, or else it does not run on this
/// machine, which leaves the hosts, without a key.
pub fn nowhere(model: Model, config: &Config) -> String {
    let disabled: Vec<&str> = KINDS
        .into_iter()
        .filter(|&kind| runs(kind, model) && config.disabled(kind))
        .map(name)
        .collect();
    if !disabled.is_empty() {
        return format!(
            "{model} has nowhere to run with {} disabled",
            disabled.join(" and ")
        );
    }
    match (model, memory::check()) {
        (Model::Qwen27b, Err(e)) => e.to_string(),
        (Model::Qwen27b, Ok(())) => {
            format!("{model} runs only at OpenRouter, Vercel or Cerebras")
        }
        _ => format!("{model} runs only at OpenRouter or Vercel"),
    }
}

/// What `kind` is called in a sentence.
fn name(kind: Kind) -> &'static str {
    kind.host().map_or("Local", |host| host.name)
}

/// The providers a model can run at, each with the models it runs, for
/// the model picker, and whether there are more that it cannot: one that
/// is disabled in `config`, this machine without the memory for a model,
/// a host without a key to it. They are this machine first, then the
/// hosts by name. Where the session's model runs, `current`, is among
/// them whatever is said of it, since the model does run there.
pub fn providers(config: &Config, current: Option<Kind>) -> (Vec<Group>, bool) {
    let usable = |&kind: &Kind| {
        Some(kind) == current || (!config.disabled(kind) && ready(kind, Model::Qwen27b))
    };
    let groups: Vec<Group> = SHOWN
        .into_iter()
        .filter(usable)
        .map(|kind| Group {
            provider: kind,
            name: label(kind),
            models: [Model::Qwen27b, Model::QwenMax]
                .into_iter()
                .filter(|&model| runs(kind, model))
                .collect(),
        })
        .collect();
    let more = groups.len() < SHOWN.len();
    (groups, more)
}

/// The providers, for the provider picker: this machine first, then the
/// hosts by name, each with whether it is disabled in `config`, and what
/// keeps a model from running there if anything does: this machine wants
/// the memory for a model, and a host a key to it.
pub fn provider_choices(config: &Config) -> Vec<Choice> {
    SHOWN
        .into_iter()
        .map(|kind| Choice {
            provider: kind,
            name: label(kind),
            problem: match kind.host() {
                None => memory::check().err().map(|e| e.to_string()),
                Some(host) => hosted::key::find(host)
                    .is_none()
                    .then(|| "no key · /login signs in".to_string()),
            },
            disabled: config.disabled(kind),
        })
        .collect()
}

/// The providers in the order the pickers show them: this machine, then
/// the hosts by name.
const SHOWN: [Kind; 4] = [Kind::Local, Kind::Cerebras, Kind::OpenRouter, Kind::Vercel];

/// What `kind` is called in a picker: this machine with a house by it.
fn label(kind: Kind) -> &'static str {
    kind.host().map_or("Local 🏠", |host| host.name)
}

/// The hosts, each with what there is of a key to it, for the sign-in
/// picker: the environment's, which is used ahead of one kept and is not
/// the shell's to change, or one kept from signing in, or none.
pub fn accounts() -> Vec<Account> {
    let hosts: [&'static Host; 3] = [&cerebras::HOST, &openrouter::HOST, &vercel::HOST];
    hosts
        .into_iter()
        .map(|host| {
            let kept = hosted::key::kept(host).is_some();
            let browser = host.kind == Kind::OpenRouter;
            let note = if hosted::key::environment(host).is_some() {
                format!("key in {}", host.key)
            } else if kept && browser {
                "signed in · enter signs in again".to_string()
            } else if kept {
                "signed in · enter takes another key".to_string()
            } else if browser {
                "sign in with your browser".to_string()
            } else {
                "paste a key".to_string()
            };
            Account { host, note, kept }
        })
        .collect()
}

/// Fetches the model's files if they are not yet there, reporting the
/// download to `on_status`, and returns the directory they are in. Checks
/// first that this machine has the memory to run the model, so an
/// undersized machine fails before the weights are downloaded: every
/// caller means to run the model here.
pub fn fetch_model(
    settings: &Settings,
    mut on_status: impl FnMut(Status),
) -> Result<PathBuf, Box<dyn Error>> {
    memory::check()?;
    let model = settings.model;
    let weights = model
        .weights()
        .ok_or_else(|| format!("{model} does not run on this machine"))?;
    let dir = weights.dir().ok_or("cannot determine the home directory")?;
    inference::fetch(weights, &dir, |progress| {
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
) -> Result<Loaded, Box<dyn Error>> {
    let dir = fetch_model(settings, &mut on_status)?;
    Loaded::load(
        settings.model,
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
) -> Result<Ready, Box<dyn Error>> {
    let loading = Instant::now();
    let (session, system) = open(settings, on_status)?;
    let harness = harness(settings, session, system, log)?;
    Ok(Ready {
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
    harness.keep_log(
        log,
        settings.model.name(),
        &device,
        settings.effort,
        &system,
    );
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
    let Some(kind) = provider(settings.provider, settings.model)? else {
        return Err(format!(
            "{}: run it at OpenRouter by signing in with /login in dwim, or by setting {}, or at Vercel by setting {}{}",
            nowhere(settings.model, &Config::load()?),
            openrouter::KEY,
            vercel::KEY,
            match settings.model {
                Model::Qwen27b => format!(", or at Cerebras by setting {}", cerebras::KEY),
                Model::QwenMax => String::new(),
            }
        )
        .into());
    };
    // A host writes the effort's line into the prompt itself, as the
    // model's chat template does, so the line is written here only for
    // the model run here.
    let effort = settings.effort;
    let system = harness::system_prompt(
        &env::current_dir()?,
        (kind == Kind::Local).then_some(effort),
    );
    let (model, context, temperature) = (settings.model, settings.context, settings.temperature);
    let hosted = match kind {
        Kind::Local => None,
        Kind::OpenRouter => Some(openrouter::open(model, effort, context, temperature)?),
        Kind::Vercel => Some(vercel::open(model, effort, context, temperature)?),
        Kind::Cerebras => Some(cerebras::open(model, effort, context, temperature)?),
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
