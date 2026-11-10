//! A session with the model, for the shell and for a one-off prompt alike:
//! the settings it runs with, loading it, and the harness around it.

use std::{
    env,
    error::Error,
    io::{self, IsTerminal, Write},
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};

use dwim::harness::{self, Harness, Log};
use dwim::models::{Chat, Gguf, LanguageModel, Tokenizer};
use dwim::skills::Skill;

use crate::{
    fetch,
    model::{self, Opened, Size},
    opts::Device,
};

/// How the model runs, as the command line sets it.
pub struct Settings {
    /// The model's name, which is the one model's.
    pub model: String,
    pub device: Device,
    /// Tokens of conversation to make room for, or as many as fit.
    pub context: Option<usize>,
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

/// How far along loading the model is, as it happens.
#[derive(Debug)]
pub enum Loading {
    /// Part of one of the model's files has been downloaded.
    Downloading(fetch::Progress),
    /// The device the model is being loaded onto is open: its name.
    Device(String),
    /// How much room the model has for conversation, and the memory it
    /// takes with it.
    Sized(Size),
    /// How many of the model's tensors are loaded, out of how many.
    Weights { done: usize, total: usize },
    /// The model is reading the system prompt: how many of its tokens so
    /// far, out of how many.
    Prompting { read: usize, total: usize },
}

/// The model loaded onto its device, with the file it came from.
pub struct LoadedModel {
    pub model: Box<dyn LanguageModel>,
    pub tokenizer: Tokenizer,
    pub gguf: Arc<Gguf>,
    /// The name of the device it runs on.
    pub device: String,
    /// How long sizing and loading it took.
    pub loading: Duration,
}

/// The harness around the model, ready for a message.
pub struct Loaded {
    pub harness: Harness<Box<dyn LanguageModel>>,
    /// How long loading the model took.
    pub loading: Duration,
}

/// Fetches the model's files if they are not yet there, reporting the
/// download to `on_loading`, and returns the directory they are in.
pub fn fetch_model(
    settings: &Settings,
    mut on_loading: impl FnMut(Loading),
) -> Result<PathBuf, Box<dyn Error>> {
    let dir = model::locate(&settings.model)?;
    fetch::fetch(&dir, |progress| on_loading(Loading::Downloading(progress)))?;
    Ok(dir)
}

/// Fetches the model if needed, opens the device, and loads the model
/// onto it, reporting each stage to `on_loading`.
pub fn load_model(
    settings: &Settings,
    mut on_loading: impl FnMut(Loading),
) -> Result<LoadedModel, Box<dyn Error>> {
    let dir = fetch_model(settings, &mut on_loading)?;
    let gguf = model::open(&dir)?;
    let device = Opened::open(settings.device)?;
    let name = device.name();
    on_loading(Loading::Device(name.clone()));
    let tokenizer = Tokenizer::from_gguf(&gguf)?;
    let loading = Instant::now();
    let size = model::size(&gguf, &device, settings.context)?;
    on_loading(Loading::Sized(size));
    let model = device.load(gguf.clone(), size.context, |done, total| {
        on_loading(Loading::Weights { done, total })
    })?;
    Ok(LoadedModel {
        model,
        tokenizer,
        gguf,
        device: name,
        loading: loading.elapsed(),
    })
}

/// Loads the model, opens the conversation with the system prompt for the
/// working directory, listing `skills`, and builds the harness around it
/// with `settings`, keeping a transcript in `log`.
pub fn load(
    settings: &Settings,
    skills: Vec<Skill>,
    log: Log,
    mut on_loading: impl FnMut(Loading),
) -> Result<Loaded, Box<dyn Error>> {
    let loaded = load_model(settings, &mut on_loading)?;
    let sampler = model::sampler(&loaded.gguf, settings.temperature);
    let mut chat = Chat::new(loaded.model, loaded.tokenizer, sampler)?;
    chat.set_drafts(settings.speculate);
    let cwd = env::current_dir()?;
    let system = harness::system_prompt(&cwd, settings.effort, &skills);
    model::start(&mut chat, &system, |read, total| {
        on_loading(Loading::Prompting { read, total })
    })?;
    let mut harness = Harness::new(chat, &cwd, {
        let system = system.clone();
        move |chat| model::start(chat, &system, |_, _| {}).map(|_| ())
    })?;
    harness.set_skills(skills);
    harness.set_thinking_budget(settings.thinking_budget);
    if let Some(shell) = settings.shell.clone() {
        harness.set_shell(shell);
    }
    harness.keep_log(log, &settings.model, &loaded.device, &system);
    Ok(Loaded {
        harness,
        loading: loaded.loading,
    })
}

/// Loading as it is told on standard error: a line rewritten as it goes
/// when standard error is a terminal, and one line for each stage when it
/// is not.
pub struct Stderr {
    terminal: bool,
    stage: String,
    /// The device the model loads onto, once it is open.
    device: String,
}

impl Stderr {
    pub fn new() -> Self {
        Self {
            terminal: io::stderr().is_terminal(),
            stage: String::new(),
            device: String::new(),
        }
    }

    /// Tells how far along loading is.
    pub fn report(&mut self, loading: &Loading) {
        match loading {
            Loading::Downloading(progress) => self.stage(
                format!("downloading {}", progress.file),
                &format!(
                    "{}/{}",
                    model::bytes(progress.done),
                    model::bytes(progress.total.unwrap_or(progress.done))
                ),
                progress.total.is_some_and(|total| progress.done >= total),
            ),
            Loading::Device(name) => self.device.clone_from(name),
            Loading::Sized(size) => eprintln!("{}", model::describe(size)),
            Loading::Weights { done, total } => self.stage(
                format!("loading the model on {}", self.device),
                &format!("{done}/{total}"),
                done >= total,
            ),
            Loading::Prompting { read, total } => self.stage(
                "reading the system prompt".to_string(),
                &format!("{read}/{total}"),
                read >= total,
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
