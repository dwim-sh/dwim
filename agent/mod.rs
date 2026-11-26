//! An agent, put together: the settings it runs with, where its model
//! runs, the session opened there with the system prompt, and the harness
//! around it. What the `dwim` command and a program that has the agent in
//! it both start from.
//!
//! An [`Agent`] is opened with [`Settings`] and sent messages, and
//! reports what it does in a turn as the harness's events:
//!
//! ```no_run
//! use std::{ops::ControlFlow, path::Path};
//!
//! use dwim::{
//!     agent::{Agent, Settings},
//!     harness::{Event, Log},
//!     model::Model,
//! };
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let settings = Settings::new(Path::new("."), Model::InklingMini);
//! let mut agent = Agent::open(settings, Log::none(), |_status| {})?;
//! agent.send("What does this project do?", |event| {
//!     if let Event::Text(text) = event {
//!         print!("{text}");
//!     }
//!     ControlFlow::Continue(())
//! })?;
//! # Ok(())
//! # }
//! ```
//!
//! The model runs where the settings say: at a host there is a key to,
//! or on this machine, where [`Engine`] says what loads it. The settings
//! [`Settings::new`] makes load it in the program itself, as an
//! [`Embedded`] engine; `dwim` has it in a server of its own, which its
//! sessions share. The pieces an agent is opened from are here too, for a
//! program that opens the session on a thread of its own, as the shell
//! does: [`open`], [`harness()`], [`reopen`] and [`switch`].

mod engine;

use std::{
    error::Error,
    ops::ControlFlow,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use crate::{
    harness::{self, Event, Harness, Log},
    inference::{self, DRAFTS, Loaded, Progress},
    model::{Device, Effort, Model},
    providers::{
        Any, Conversation, Kind, cerebras,
        dwim::{self as server},
        openrouter, vercel,
    },
    sys::memory,
    tools::{self, Custom},
};

pub use crate::{
    agent::engine::{Embedded, serve},
    server::protocol::Status,
};

/// How an agent runs: where it works, its model, and how the model runs.
#[derive(Clone)]
pub struct Settings {
    /// The directory the agent works in.
    pub dir: PathBuf,
    /// The model to run.
    pub model: Model,
    /// Where it runs, if that is settled.
    pub provider: Option<Kind>,
    /// The providers a model is not run at unless `provider` names one.
    pub disabled: Vec<Kind>,
    /// Where the model is loaded, when it runs on this machine.
    pub engine: Engine,
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
    /// The tools the program gives the agent beside the agent's own.
    pub tools: Vec<Custom>,
}

/// Where a model that runs on this machine is loaded.
#[derive(Clone)]
pub enum Engine {
    /// In a dwim server, a process of its own, which the sessions on this
    /// machine share: `program` is started with `--serve` when none is
    /// listening, and is `dwim`, or a program that serves as it does.
    Server { program: PathBuf },
    /// In this program, on a thread of it, once a session is opened: the
    /// sessions opened with clones of the engine share the model, which
    /// is dropped when the last of them and of the clones is.
    Embedded(Embedded),
}

impl Settings {
    /// The settings of an agent that works in `dir` on `model`, as `dwim`
    /// runs it unless told, but for where a model that runs on this
    /// machine is loaded, which is in this program: it runs where
    /// [`provider`] finds, on the GPU, with the context that fits, the
    /// thinking budget the harness has, and the effort the model reasons
    /// at by default.
    pub fn new(dir: &Path, model: Model) -> Self {
        Self {
            dir: dir.to_path_buf(),
            model,
            provider: None,
            disabled: Vec::new(),
            engine: Engine::Embedded(Embedded::default()),
            device: Device::Gpu(None),
            context: None,
            sessions: None,
            thinking_budget: harness::THINKING_BUDGET,
            effort: model.default_effort(),
            temperature: None,
            speculate: DRAFTS,
            shell: None,
            tools: Vec::new(),
        }
    }

    /// The server the settings ask for, started as `program`.
    fn server(&self, program: &Path) -> server::Options {
        server::Options {
            program: program.to_path_buf(),
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

/// An agent, ready for a message: the harness around a session with the
/// model, opened where the model runs.
pub struct Agent {
    harness: Harness<Session>,
    loading: Duration,
}

impl Agent {
    /// Opens a session where the model `settings` name runs, with the
    /// system prompt for the directory the agent works in, and builds the
    /// harness around it, keeping a transcript in `log`. A model that
    /// runs on this machine is loaded first if it is not, and fetched if
    /// its files are not there. `on_status` is told how that goes, and
    /// after, when the session waits for another's request.
    pub fn open(
        settings: Settings,
        log: Log,
        on_status: impl FnMut(Status) + Send + 'static,
    ) -> Result<Self, Box<dyn Error>> {
        let loading = Instant::now();
        let (session, system) = open(&settings, on_status)?;
        let harness = harness(&settings, session, system, log)?;
        Ok(Self {
            harness,
            loading: loading.elapsed(),
        })
    }

    /// Sends a message from the user and runs the turn it starts, to the
    /// reply that calls no tool, reporting what happens to `on_event`,
    /// which ends the turn early by breaking: what [`Harness::send`]
    /// does.
    pub fn send(
        &mut self,
        message: &str,
        on_event: impl FnMut(Event) -> ControlFlow<()>,
    ) -> Result<(), Box<dyn Error>> {
        self.harness.send(message, on_event)
    }

    /// The harness, for what else is asked of it: to compact the
    /// conversation, to go on with no one to answer, to set the plan
    /// aside.
    pub fn harness(&mut self) -> &mut Harness<Session> {
        &mut self.harness
    }

    /// Where the conversation's time has gone so far.
    pub fn stats(&self) -> harness::Stats {
        self.harness.stats()
    }

    /// How long getting the model ready took: reaching it, the loading
    /// if it was loading, and the system prompt.
    pub fn loading(&self) -> Duration {
        self.loading
    }
}

/// Where `model` runs: at `provider` if one is named, and otherwise on
/// this machine if it can, or at a host there is a key to: OpenRouter, or
/// else Vercel's AI Gateway, or else Cerebras, leaving out the `disabled`
/// providers. Inkling 2 Mini runs on this machine, at OpenRouter or at
/// Vercel's AI Gateway, and the large models, Inkling 2 Small and Qwen
/// Max, only at OpenRouter or Vercel's AI Gateway. Nowhere, if no provider
/// is named and the model can run neither here nor at a host without a
/// key. A provider that is named is only returned if the model can run
/// there: a local one needs this machine to have the memory for it.
pub fn provider(
    provider: Option<Kind>,
    model: Model,
    disabled: &[Kind],
) -> Result<Option<Kind>, Box<dyn Error>> {
    match (provider, model) {
        (Some(Kind::Local), Model::QwenMax | Model::InklingSmall) => Err(format!(
            "{model} does not run on this machine: it runs with --provider openrouter, or vercel"
        )
        .into()),
        (Some(Kind::Cerebras), Model::QwenMax | Model::InklingSmall) => Err(format!(
            "{model} does not run at Cerebras: it runs with --provider openrouter, or vercel"
        )
        .into()),
        (Some(Kind::Cerebras), Model::InklingMini) => Err(format!(
            "{model} is not run at Cerebras: it runs with --provider local, openrouter, or vercel"
        )
        .into()),
        (Some(Kind::Local), _) => {
            memory::check()?;
            Ok(Some(Kind::Local))
        }
        (Some(kind), _) => Ok(Some(kind)),
        (None, model) => Ok(KINDS
            .into_iter()
            .find(|kind| !disabled.contains(kind) && ready(*kind, model))),
    }
}

/// The providers, in the order a model is run at the first that can run
/// it, and in which the pickers show them.
const KINDS: [Kind; 4] = [Kind::Local, Kind::OpenRouter, Kind::Vercel, Kind::Cerebras];

/// Whether `model` can run at `kind` as things are: on this machine with
/// the memory for it, at a host that runs it with a key to the host.
pub fn ready(kind: Kind, model: Model) -> bool {
    runs(kind, model)
        && match kind {
            Kind::Local => memory::check().is_ok(),
            Kind::OpenRouter => openrouter::key().is_some(),
            Kind::Vercel => vercel::key().is_some(),
            Kind::Cerebras => cerebras::key().is_some(),
        }
}

/// Whether `kind` is a provider of `model`: on this machine the ones with
/// weights, at Cerebras the Qwen model, and at OpenRouter and Vercel's AI
/// Gateway all of them.
pub fn runs(kind: Kind, model: Model) -> bool {
    match kind {
        Kind::Local => model.weights().is_some(),
        Kind::Cerebras => model == Model::Qwen27b,
        Kind::OpenRouter | Kind::Vercel => true,
    }
}

/// Why `model` has nowhere to run, when [`provider`] finds none: providers
/// of it are `disabled`, or else it does not run on this machine, which
/// leaves the hosts, without a key.
pub fn nowhere(model: Model, disabled: &[Kind]) -> String {
    let disabled: Vec<&str> = KINDS
        .into_iter()
        .filter(|kind| runs(*kind, model) && disabled.contains(kind))
        .map(name)
        .collect();
    if !disabled.is_empty() {
        return format!(
            "{model} has nowhere to run with {} disabled",
            disabled.join(" and ")
        );
    }
    match (model, memory::check()) {
        (Model::Qwen27b | Model::InklingMini, Err(e)) => e.to_string(),
        (Model::Qwen27b, Ok(())) => {
            format!("{model} runs only at OpenRouter, Vercel or Cerebras")
        }
        (Model::InklingMini, Ok(())) | (Model::QwenMax | Model::InklingSmall, _) => {
            format!("{model} runs only at OpenRouter or Vercel")
        }
    }
}

/// What `kind` is called in a sentence.
fn name(kind: Kind) -> &'static str {
    kind.host().map_or("Local", |host| host.name)
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

/// Builds the harness around `session`, opened with the system prompt
/// `system`, with `settings`, keeping a transcript in `log`.
pub fn harness(
    settings: &Settings,
    mut session: Session,
    system: String,
    log: Log,
) -> Result<Harness<Session>, Box<dyn Error>> {
    let device = device(&mut session);
    let mut harness = Harness::new(session, &settings.dir, {
        let system = system.clone();
        move |session: &mut Session| session.start(&system, |_, _| {})
    })?;
    harness.set_thinking_budget(settings.thinking_budget);
    harness.set_model(settings.model);
    if let Some(shell) = settings.shell.clone() {
        harness.set_shell(shell);
    }
    harness.add_tools(settings.tools.clone());
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
/// [`Agent::open`].
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
/// the server is doing, as with [`Agent::open`].
pub fn switch(
    settings: &Settings,
    harness: &mut Harness<Session>,
    on_status: impl FnMut(Status) + Send + 'static,
) -> Result<Kind, Box<dyn Error>> {
    harness.set_model(settings.model);
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
/// directory the agent works in, which it returns with the session: on
/// the server, reached, or started if none is running, or on the engine
/// in this program, as the settings have it, with `on_status` told what
/// that is doing meanwhile, and after, when the session waits for
/// another's request; or at a host, where there is nothing to wait for.
/// Settings with tools that cannot be told apart are refused.
pub fn open(
    settings: &Settings,
    mut on_status: impl FnMut(Status) + Send + 'static,
) -> Result<(Session, String), Box<dyn Error>> {
    tools::distinct(&settings.tools)?;
    let Some(kind) = provider(settings.provider, settings.model, &settings.disabled)? else {
        let vercel = format!(", or at Vercel by setting {}", vercel::KEY);
        return Err(format!(
            "{}: run it at OpenRouter by signing in with /login in dwim, or by setting {}{}",
            nowhere(settings.model, &settings.disabled),
            openrouter::KEY,
            match settings.model {
                Model::Qwen27b => format!("{vercel}, or at Cerebras by setting {}", cerebras::KEY),
                Model::QwenMax | Model::InklingSmall | Model::InklingMini => vercel,
            }
        )
        .into());
    };
    // A host writes the effort's line into the prompt itself, as the
    // model's chat template does, so the line is written here only for
    // the model run here.
    let effort = settings.effort;
    let system = harness::system_prompt(
        &settings.dir,
        settings.model,
        (kind == Kind::Local).then_some(effort),
        &settings.tools,
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
    let connection = match &settings.engine {
        Engine::Server { program } => server::connect(&settings.server(program), &mut on_status)?,
        Engine::Embedded(embedded) => embedded.connect(settings)?,
    };
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
