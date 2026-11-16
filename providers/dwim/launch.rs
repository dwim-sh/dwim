//! Reaching the server: finding the one that is running, starting one
//! when none is, replacing one that is not as asked for, and stopping it.

use std::{
    env,
    fs::OpenOptions,
    path::PathBuf,
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant},
};

use crate::{
    model::{Device, Result},
    providers::dwim::{Connection, Remote},
    server::{
        self,
        protocol::{Status, Welcome},
    },
    sys,
};

/// The server a session asks for: the model it has, and how it runs it.
pub struct Options {
    /// The model's name.
    pub model: String,
    pub device: Device,
    /// Tokens of conversation to make room for in each session, or as
    /// many as fit.
    pub context: Option<usize>,
    /// Sessions to make room for at once, or the default.
    pub sessions: Option<usize>,
}

/// Longest a session waits for the server it started to listen, which it
/// does before loading the model.
const STARTING: Duration = Duration::from_secs(10);

/// Longest a session waits for a server it asked to go to exit, before it
/// starts another: two servers loading the model at once take more memory
/// than there is.
const RETIRING: Duration = Duration::from_secs(30);

/// How often a session looks again while it waits for a server.
const POLL: Duration = Duration::from_millis(50);

/// The server listening, as it says who it is, if one is.
pub fn running() -> Option<Welcome> {
    Connection::connect(&server::socket_path())
        .ok()?
        .hello()
        .ok()
}

/// Stops the server `session` is on, if it has no other: asks it
/// to go, ends the session, and waits for the server to exit.
pub fn stop(session: &mut Remote) -> Result<()> {
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
    session.close();
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
pub fn connect(options: &Options, on_status: &mut impl FnMut(Status)) -> Result<Connection> {
    let path = server::socket_path();
    let mut started: Option<Instant> = None;
    loop {
        if let Ok(mut connection) = Connection::connect(&path) {
            let welcome = connection.hello()?;
            if let Some(difference) = difference(&welcome, options) {
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
                spawn(options)?;
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

/// How the server differs from what `options` ask for, if it does.
fn difference(welcome: &Welcome, options: &Options) -> Option<String> {
    if welcome.build != server::build() {
        return Some("is from another build of dwim".to_string());
    }
    if welcome.model != options.model {
        return Some(format!("runs {}, not {}", welcome.model, options.model));
    }
    if welcome.device != options.device.to_string() {
        return Some(format!(
            "runs the model on the {}, not the {}",
            welcome.device.to_uppercase(),
            options.device.to_string().to_uppercase()
        ));
    }
    if let Some(sessions) = options.sessions
        && welcome.room != sessions
    {
        return Some(format!(
            "has room for {} {}, not {sessions}",
            welcome.room,
            plural(welcome.room, "session")
        ));
    }
    if let Some(context) = options.context
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
    while sys::process::alive(pid) && Instant::now() < deadline {
        thread::sleep(POLL);
    }
}

/// Starts the server as `options` have it, detached: in its own process
/// group, with no terminal, and with its standard error appended to its
/// log. The shell puts it in the background, so that it is not this
/// process's child to wait for.
fn spawn(options: &Options) -> Result<()> {
    let exe = env::current_exe()?;
    let mut args = vec![
        "--serve".to_string(),
        "--model".to_string(),
        options.model.clone(),
        "--device".to_string(),
        options.device.to_string(),
    ];
    if let Some(context) = options.context {
        args.push("--context".to_string());
        args.push(context.to_string());
    }
    if let Some(sessions) = options.sessions {
        args.push("--sessions".to_string());
        args.push(sessions.to_string());
    }
    let log = match log_path() {
        Some(path) => {
            if let Some(dir) = path.parent() {
                let _ = sys::fs::create_private_dir(dir);
            }
            OpenOptions::new()
                .append(true)
                .create(true)
                .open(&path)
                .map_or(Stdio::null(), Stdio::from)
        }
        None => Stdio::null(),
    };
    let mut command = Command::new("sh");
    command
        .arg("-c")
        .arg(r#"exec "$0" "$@" &"#)
        .arg(exe)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(log);
    sys::process::own_group(&mut command);
    let status = command
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
    Some(sys::dirs::state()?.join("dwim").join("server.log"))
}
