//! A session with the model, for the shell and for a one-off prompt alike:
//! the settings it runs with, reaching the server that has the model,
//! starting one when none is running, and the harness around it.

use std::{
    env,
    error::Error,
    fmt,
    fs::{self, OpenOptions},
    io::{self, IsTerminal, Write},
    os::unix::process::CommandExt,
    path::PathBuf,
    process::{Command, Stdio},
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

use dwim::harness::{self, Harness, Log};
use dwim::models::{Gguf, LanguageModel, Tokenizer};
use dwim::server::{self, Client, Connection, protocol::Welcome};

use crate::{
    fetch,
    model::{self, Opened, Size},
    opts::Device,
};

pub use dwim::server::protocol::Status;

/// Longest a session waits for the server it started to listen, which it
/// does before loading the model.
const STARTING: Duration = Duration::from_secs(10);

/// Longest a session waits for a server it asked to go to exit, before it
/// starts another: two servers loading the model at once take more memory
/// than there is.
const RETIRING: Duration = Duration::from_secs(30);

/// How often a session looks again while it waits for a server.
const POLL: Duration = Duration::from_millis(50);

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

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Device::Cpu => "cpu",
            Device::Gpu => "gpu",
        })
    }
}

/// The model loaded onto its device, with the file it came from.
pub struct LoadedModel {
    pub model: Box<dyn LanguageModel>,
    pub tokenizer: Tokenizer,
    pub gguf: Arc<Gguf>,
    /// The name of the device it runs on.
    pub device: String,
    pub size: Size,
}

/// The harness around the model, ready for a message.
pub struct Loaded {
    pub harness: Harness<Client>,
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
    let dir = model::locate(&settings.model)?;
    fetch::fetch(&dir, |progress| {
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
) -> Result<LoadedModel, Box<dyn Error>> {
    let dir = fetch_model(settings, &mut on_status)?;
    let gguf = model::open(&dir)?;
    let device = Opened::open(settings.device)?;
    let name = device.name();
    on_status(Status::Device(name.clone()));
    let tokenizer = Tokenizer::from_gguf(&gguf)?;
    let size = model::size(&gguf, &device, settings.context, sessions)?;
    on_status(Status::Sized(size));
    let model = device.load(gguf.clone(), size.context, |done, total| {
        on_status(Status::Weights { done, total })
    })?;
    Ok(LoadedModel {
        model,
        tokenizer,
        gguf,
        device: name,
        size,
    })
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
    let (client, system) = open(settings, on_status)?;
    let device = client.device().to_string();
    let cwd = env::current_dir()?;
    let mut harness = Harness::new(client, &cwd, {
        let system = system.clone();
        move |client: &mut Client| client.start(&system, |_, _| {}).map(|_| ())
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
    harness: &mut Harness<Client>,
    on_status: impl FnMut(Status) + 'static,
) -> Result<(), Box<dyn Error>> {
    let (client, _) = open(settings, on_status)?;
    harness.reopen(client)
}

/// Reaches the server, starting one if none is running, and opens a
/// session on it with the system prompt for the working directory, which
/// it returns with the session. `on_status` is told what the server is
/// doing meanwhile, and after, when the session waits for another's
/// request.
fn open(
    settings: &Settings,
    mut on_status: impl FnMut(Status) + 'static,
) -> Result<(Client, String), Box<dyn Error>> {
    let connection = connect(settings, &mut on_status)?;
    on_status(Status::Opening);
    // A session that connects once the model is loaded, or part way
    // through, is told of the device and the size afterwards instead.
    let (mut device_told, mut size_told) = (false, false);
    let mut client = connection.open(settings.temperature, settings.speculate, |status| {
        match status {
            Status::Device(_) => device_told = true,
            Status::Sized(_) => size_told = true,
            _ => {}
        }
        on_status(status)
    })?;
    if !device_told {
        on_status(Status::Device(client.device().to_string()));
    }
    if !size_told {
        on_status(Status::Sized(client.size()));
    }
    let system = harness::system_prompt(&env::current_dir()?, settings.effort);
    client.start(&system, |read, total| {
        on_status(Status::Prompting { read, total })
    })?;
    client.on_status(on_status);
    Ok((client, system))
}

/// The server listening, as it says who it is, if one is.
pub fn running() -> Option<Welcome> {
    Connection::connect(&server::socket_path())
        .ok()?
        .hello()
        .ok()
}

/// Stops the server `client` is a session on, if it has no other: asks it
/// to go, ends the session, and waits for the server to exit.
pub fn stop(client: &mut Client) -> Result<(), Box<dyn Error>> {
    let mut connection =
        Connection::connect(&server::socket_path()).map_err(|_| "no dwim server is running")?;
    let welcome = connection.hello()?;
    let others = welcome.sessions.saturating_sub(1);
    if others > 0 {
        return Err(format!(
            "the dwim server has {others} other {}: quit {} first",
            plural(others, "session"),
            if others == 1 { "it" } else { "them" }
        )
        .into());
    }
    connection.retire()?;
    drop(connection);
    client.close();
    wait_for_exit(welcome.pid);
    Ok(())
}

/// Whether `welcome` is from a server of this build of dwim.
pub fn same_build(welcome: &Welcome) -> bool {
    welcome.build == server::build()
}

/// Connects to the server, starting one when none is listening. A server
/// from another build, or with the model on another device or with
/// another context than asked for, is asked to go and replaced if it has
/// no sessions; with sessions, it is an error, since two servers would not
/// fit in memory.
fn connect(
    settings: &Settings,
    on_status: &mut impl FnMut(Status),
) -> Result<Connection, Box<dyn Error>> {
    let path = server::socket_path();
    let mut started: Option<Instant> = None;
    loop {
        if let Ok(mut connection) = Connection::connect(&path) {
            let welcome = connection.hello()?;
            if let Some(difference) = difference(&welcome, settings) {
                if welcome.sessions > 0 {
                    return Err(format!(
                        "the dwim server {difference}, and has {} other {}: quit {} first, or run with the same options",
                        welcome.sessions,
                        plural(welcome.sessions, "session"),
                        if welcome.sessions == 1 { "it" } else { "them" }
                    )
                    .into());
                }
                on_status(Status::Restarting);
                connection.retire()?;
                drop(connection);
                wait_for_exit(welcome.pid);
                started = None;
            } else {
                return Ok(connection);
            }
        }
        match started {
            None => {
                spawn(settings)?;
                on_status(Status::Starting);
                started = Some(Instant::now());
            }
            Some(since) if since.elapsed() > STARTING => {
                return Err(format!(
                    "the dwim server did not start listening on {}: see {}",
                    path.display(),
                    log_path().map_or("its log".to_string(), |path| path.display().to_string())
                )
                .into());
            }
            Some(_) => {}
        }
        thread::sleep(POLL);
    }
}

/// How the server differs from what `settings` ask for, if it does.
fn difference(welcome: &Welcome, settings: &Settings) -> Option<String> {
    if welcome.build != server::build() {
        return Some("is from another build of dwim".to_string());
    }
    if welcome.model != settings.model {
        return Some(format!("runs {}, not {}", welcome.model, settings.model));
    }
    if welcome.device != settings.device.to_string() {
        return Some(format!(
            "runs the model on the {}, not the {}",
            welcome.device.to_uppercase(),
            settings.device.to_string().to_uppercase()
        ));
    }
    if let Some(sessions) = settings.sessions
        && welcome.room != sessions
    {
        return Some(format!(
            "has room for {} {}, not {sessions}",
            welcome.room,
            plural(welcome.room, "session")
        ));
    }
    if let Some(context) = settings.context
        && welcome.context != Some(context)
    {
        return Some(match welcome.context {
            Some(theirs) => format!("has a context of {theirs} tokens, not {context}"),
            None => format!("was not asked for a context of {context} tokens"),
        });
    }
    None
}

fn plural(n: usize, word: &str) -> String {
    if n == 1 {
        word.to_string()
    } else {
        format!("{word}s")
    }
}

/// Waits for the process `pid` to exit, up to a while.
fn wait_for_exit(pid: u32) {
    let deadline = Instant::now() + RETIRING;
    // SAFETY: a signal of 0 sends nothing, and only asks whether the
    // process is there.
    while unsafe { libc::kill(pid as libc::pid_t, 0) } == 0 && Instant::now() < deadline {
        thread::sleep(POLL);
    }
}

/// Starts the server as `settings` have it, detached: in its own process
/// group, with no terminal, and with its standard error appended to its
/// log. The shell puts it in the background, so that it is not this
/// process's child to wait for.
fn spawn(settings: &Settings) -> Result<(), Box<dyn Error>> {
    let exe = env::current_exe()?;
    let mut args = vec![
        "--serve".to_string(),
        "--model".to_string(),
        settings.model.clone(),
        "--device".to_string(),
        settings.device.to_string(),
    ];
    if let Some(context) = settings.context {
        args.push("--context".to_string());
        args.push(context.to_string());
    }
    if let Some(sessions) = settings.sessions {
        args.push("--sessions".to_string());
        args.push(sessions.to_string());
    }
    let log = match log_path() {
        Some(path) => {
            if let Some(dir) = path.parent() {
                let _ = fs::create_dir_all(dir);
            }
            OpenOptions::new()
                .append(true)
                .create(true)
                .open(&path)
                .map_or(Stdio::null(), Stdio::from)
        }
        None => Stdio::null(),
    };
    let status = Command::new("sh")
        .arg("-c")
        .arg(r#"exec "$0" "$@" &"#)
        .arg(exe)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(log)
        .process_group(0)
        .status()
        .map_err(|e| format!("cannot start the dwim server: {e}"))?;
    if !status.success() {
        return Err("cannot start the dwim server".into());
    }
    Ok(())
}

/// Where the server's log is: in the user's state directory
/// (`~/.local/state`, or on macOS, where there is none, `~/Library/Application
/// Support`).
pub fn log_path() -> Option<PathBuf> {
    Some(
        dirs::state_dir()
            .or_else(dirs::data_local_dir)?
            .join("dwim")
            .join("server.log"),
    )
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
