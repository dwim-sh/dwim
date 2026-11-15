//! The model's thread: it has the session with the model, does what the
//! shell asks of it, a request at a time, and reports back as it goes.

use std::{
    error::Error,
    ops::ControlFlow,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, RecvTimeoutError, Sender, TryRecvError},
    },
    thread,
    time::{Duration, Instant},
};

use dwim::harness::{self, Event, Harness, Log};
use dwim::model::Variant;
use dwim::providers::{
    Kind,
    dwim::{Remote, log_path, running, same_build, stop},
};

use crate::{
    session::{self, Session, Settings},
    shell::{cmd::ServerAction, view},
};

/// Longest the shell waits, on quitting, for the model's thread to stop
/// the turn in progress and end the transcript.
const FINISH: Duration = Duration::from_secs(2);

/// How often the thread looks for a request while the model loads.
const POLL: Duration = Duration::from_millis(50);

/// The model's thread, as the shell has it: where to send requests, and
/// where the replies come back.
pub struct Worker {
    requests: Sender<Request>,
    replies: Receiver<Reply>,
    /// Set to stop the turn in progress.
    stop: Arc<AtomicBool>,
    /// Disconnected once the thread is done.
    finished: Receiver<()>,
}

impl Worker {
    /// Starts the thread, which loads the model as `settings` have it and
    /// keeps a transcript in `log`.
    pub fn spawn(settings: Settings, log: Log) -> Self {
        let (requests, requests_rx) = mpsc::channel();
        let (replies_tx, replies) = mpsc::channel();
        let stop = Arc::new(AtomicBool::new(false));
        let (done, finished) = mpsc::channel::<()>();
        thread::spawn({
            let stop = stop.clone();
            move || {
                let _done = done;
                if let Err(e) = work(settings, log, requests_rx, &replies_tx, &stop) {
                    let _ = replies_tx.send(Reply::Failed(e.to_string()));
                }
            }
        });
        Self {
            requests,
            replies,
            stop,
            finished,
        }
    }

    /// Asks the thread for something, which it gets to after what it was
    /// asked before.
    pub fn send(&self, request: Request) {
        self.stop.store(false, Ordering::Relaxed);
        let _ = self.requests.send(request);
    }

    /// The next thing the thread has reported, if it has reported anything.
    pub fn try_recv(&self) -> Result<Reply, TryRecvError> {
        self.replies.try_recv()
    }

    /// Stops the turn in progress.
    pub fn interrupt(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }

    /// Hangs up on the thread, which stops the turn in progress and, with
    /// no more requests to answer, ends the transcript: waits a while for
    /// that if `wait` is set.
    pub fn finish(self, wait: bool) {
        let Self {
            requests,
            replies,
            stop,
            finished,
        } = self;
        stop.store(true, Ordering::Relaxed);
        drop((requests, replies));
        if wait {
            let _ = finished.recv_timeout(FINISH);
        }
    }
}

/// What the shell asks of the model thread.
pub enum Request {
    /// A message from the user.
    Message(String),
    /// Compact the conversation now, as the `/compact` command asks.
    Compact,
    /// Limit a thought to this many tokens, or lift the limit, as the
    /// `/thinking-budget` command asks.
    ThinkingBudget(Option<usize>),
    /// Turn auto mode on or off, as the `/auto` command asks.
    Auto(bool),
    /// Show, stop, start or restart the server, as the `/server` command
    /// asks.
    Server(ServerAction),
    /// Go on with the conversation on this model at this provider, as
    /// picked with the `/model` command.
    Model(Kind, Variant),
}

/// What the model thread reports back.
pub enum Reply {
    /// What the server is doing while the session waits on it.
    Status(session::Status),
    /// The model is ready for a message, and how long loading it took.
    Ready(Duration),
    /// The session is open on the server again, and goes on with what was
    /// asked of it.
    Reopened,
    /// Something that happened in the turn.
    Event(Event),
    /// Something to tell the user, in place of a reply.
    Notice(String),
    /// The thinking budget, as now set.
    ThinkingBudget(Option<usize>),
    /// Auto mode, as now set.
    Auto(bool),
    /// The model the conversation is now on, and where it runs.
    Model { variant: Variant, provider: Kind },
    /// The reply is over, and where the conversation's time has gone.
    Done(harness::Stats),
    /// The turn failed, though the model is still there for the next one.
    Error(String),
    /// The session could not be opened where the model was to run, and
    /// why: none is, until another model or provider is picked.
    Unopened(String),
    /// The model's thread is gone.
    Failed(String),
}

/// Loads the model as `settings` have it and answers messages until the UI
/// hangs up, keeping a transcript in `log`. While the model loads, another
/// can be picked, and is loaded instead.
fn work(
    mut settings: Settings,
    log: Log,
    requests: Receiver<Request>,
    replies: &Sender<Reply>,
    stop: &AtomicBool,
) -> Result<(), Box<dyn Error>> {
    let loading = Instant::now();
    let Some((session, system)) = open(&mut settings, &requests, replies)? else {
        return Ok(());
    };
    let mut harness = session::harness(&settings, session, system, log)?;
    let _ = replies.send(Reply::Ready(loading.elapsed()));
    // Whether the session's server was stopped, so the next request that
    // needs the model opens it on another.
    let mut stopped = false;
    let on_status = || {
        let replies = replies.clone();
        move |status| {
            let _ = replies.send(Reply::Status(status));
        }
    };

    for request in requests {
        let on_event = |event: Event| {
            let _ = replies.send(Reply::Event(event));
            if stop.load(Ordering::Relaxed) {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        };
        if stopped && matches!(request, Request::Message(_) | Request::Compact) {
            if let Err(e) = session::reopen(&settings, &mut harness, on_status()) {
                let _ = replies.send(Reply::Error(e.to_string()));
                let _ = replies.send(Reply::Done(harness.stats()));
                continue;
            }
            stopped = false;
            let _ = replies.send(Reply::Reopened);
        }
        let result = match request {
            Request::Message(message) => harness.send(&message, on_event),
            Request::Server(action) => {
                server(&settings, &mut harness, &mut stopped, action, on_status).map(|notice| {
                    let _ = replies.send(Reply::Notice(notice));
                })
            }
            Request::Compact => harness.compact(on_event).map(|compacted| {
                if !compacted && !stop.load(Ordering::Relaxed) {
                    let _ = replies.send(Reply::Notice("Nothing to compact yet".to_string()));
                }
            }),
            Request::ThinkingBudget(tokens) => {
                harness.set_thinking_budget(tokens);
                let _ = replies.send(Reply::ThinkingBudget(tokens));
                Ok(())
            }
            Request::Auto(auto) => {
                harness.set_auto(auto);
                let _ = replies.send(Reply::Auto(auto));
                Ok(())
            }
            Request::Model(provider, variant) => {
                let before = (settings.provider, settings.model);
                (settings.provider, settings.model) = (Some(provider), variant);
                match session::switch(&settings, &mut harness, on_status()) {
                    Ok(provider) => {
                        stopped = false;
                        let _ = replies.send(Reply::Model { variant, provider });
                        Ok(())
                    }
                    // The session may be neither model's by now: the next
                    // message opens it over on the model it was on.
                    Err(e) => {
                        (settings.provider, settings.model) = before;
                        stopped = true;
                        Err(e)
                    }
                }
            }
        };
        if let Err(e) = result {
            let _ = replies.send(Reply::Error(e.to_string()));
        }
        let _ = replies.send(Reply::Done(harness.stats()));
    }
    Ok(())
}

/// Opens the session where `settings` have the model, and its system
/// prompt, without keeping the shell waiting: a model picked meanwhile
/// changes `settings`, and the session is opened there instead, the one
/// being opened left to itself. Where the model has nowhere to run, or
/// the session cannot be opened, nothing is opened until one is picked.
/// Nothing if the shell hangs up first.
fn open(
    settings: &mut Settings,
    requests: &Receiver<Request>,
    replies: &Sender<Reply>,
) -> Result<Option<(Session, String)>, Box<dyn Error>> {
    let mut opening = session::provider(settings.provider, settings.model)?
        .map(|_| Opening::start(settings, replies));
    loop {
        if let Some(open) = &opening {
            match open.opened.try_recv() {
                Ok(Ok(opened)) => return Ok(Some(opened)),
                Ok(Err(e)) => {
                    let _ = replies.send(Reply::Unopened(e));
                    opening = None;
                }
                Err(TryRecvError::Empty) => {}
                Err(TryRecvError::Disconnected) => {
                    return Err("the model stopped loading unexpectedly".into());
                }
            }
        }
        match requests.recv_timeout(POLL) {
            Ok(Request::Model(provider, variant)) => {
                if let Some(open) = &opening {
                    open.left.store(true, Ordering::Relaxed);
                }
                (settings.provider, settings.model) = (Some(provider), variant);
                let _ = replies.send(Reply::Model { variant, provider });
                opening = Some(Opening::start(settings, replies));
            }
            Ok(_) => {
                let _ = replies.send(Reply::Error("the model is still loading".to_string()));
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => {
                if let Some(open) = &opening {
                    open.left.store(true, Ordering::Relaxed);
                }
                return Ok(None);
            }
        }
    }
}

/// A session being opened, on a thread of its own.
struct Opening {
    /// The session and its system prompt, once it is open, or why it is
    /// not.
    opened: Receiver<Result<(Session, String), String>>,
    /// Set when the session is no longer wanted: the thread says no more
    /// of what the server is doing, and the session it opens is dropped.
    left: Arc<AtomicBool>,
}

impl Opening {
    /// Starts opening the session where `settings` have the model, telling
    /// `replies` what the server is doing meanwhile.
    fn start(settings: &Settings, replies: &Sender<Reply>) -> Self {
        let (sender, opened) = mpsc::channel();
        let left = Arc::new(AtomicBool::new(false));
        thread::spawn({
            let settings = settings.clone();
            let replies = replies.clone();
            let left = left.clone();
            move || {
                let on_status = move |status| {
                    if !left.load(Ordering::Relaxed) {
                        let _ = replies.send(Reply::Status(status));
                    }
                };
                let _ = sender.send(session::open(&settings, on_status).map_err(|e| e.to_string()));
            }
        });
        Self { opened, left }
    }
}

/// Does what `/server` asks, `action`, for the session `harness` runs,
/// which `stopped` says is not on a server: returns what to tell of it.
/// `on_status` makes what is told what the server does as it starts.
fn server<F: FnMut(session::Status) + Send + 'static>(
    settings: &Settings,
    harness: &mut Harness<session::Session>,
    stopped: &mut bool,
    action: ServerAction,
    on_status: impl Fn() -> F,
) -> Result<String, Box<dyn Error>> {
    if harness.session().provider().dwim().is_none() {
        return Ok("This session's model runs at OpenRouter · it has no dwim server".to_string());
    }
    match action {
        ServerAction::Show => Ok(match running() {
            None => "No dwim server is running · the next message starts it".to_string(),
            Some(welcome) => {
                let device = if *stopped {
                    welcome.device.to_uppercase()
                } else {
                    remote(harness).device().to_string()
                };
                let mut notice = format!(
                    "dwim server {} · {} on {device} · {} of {} {} open",
                    welcome.pid,
                    welcome.model,
                    welcome.sessions,
                    welcome.room,
                    if welcome.room == 1 {
                        "session"
                    } else {
                        "sessions"
                    },
                );
                if !same_build(&welcome) {
                    notice.push_str(" · from another build of dwim");
                }
                if let Some(path) = log_path() {
                    notice.push_str(&format!(" · log {}", view::tilde(&path)));
                }
                notice
            }
        }),
        ServerAction::Stop if *stopped => {
            Ok("The dwim server is stopped · the next message starts it".to_string())
        }
        ServerAction::Stop => {
            stop(remote(harness))?;
            *stopped = true;
            Ok("Stopped the dwim server · the next message starts it again".to_string())
        }
        ServerAction::Start if !*stopped => {
            Ok("The dwim server is running · /server restart restarts it".to_string())
        }
        ServerAction::Start | ServerAction::Restart => {
            if !*stopped {
                stop(remote(harness))?;
                *stopped = true;
            }
            session::reopen(settings, harness, on_status())?;
            *stopped = false;
            Ok(match action {
                ServerAction::Start => "Started the dwim server",
                _ => "Restarted the dwim server",
            }
            .to_string())
        }
    }
}

/// The session `harness` runs on the dwim server.
fn remote(harness: &mut Harness<session::Session>) -> &mut Remote {
    harness
        .session()
        .provider()
        .dwim()
        .expect("the session is on the dwim server")
}
