//! The model's thread: it has the session with the model, does what the
//! shell asks of it, a request at a time, and reports back as it goes.

use std::{
    error::Error,
    ops::ControlFlow,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, Sender, TryRecvError},
    },
    thread,
    time::Duration,
};

use dwim::harness::{self, Event, Harness, Log};
use dwim::server::Client;

use crate::{
    session::{self, Settings},
    shell::{cmd::ServerAction, view},
};

/// Longest the shell waits, on quitting, for the model's thread to stop
/// the turn in progress and end the transcript.
const FINISH: Duration = Duration::from_secs(2);

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
                if let Err(e) = work(&settings, log, requests_rx, &replies_tx, &stop) {
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
    /// The reply is over, and where the conversation's time has gone.
    Done(harness::Stats),
    /// The turn failed, though the model is still there for the next one.
    Error(String),
    /// The model is gone: it could not be fetched or loaded.
    Failed(String),
}

/// Loads the model and answers messages until the UI hangs up, keeping a
/// transcript in `log`.
fn work(
    settings: &Settings,
    log: Log,
    requests: Receiver<Request>,
    replies: &Sender<Reply>,
    stop: &AtomicBool,
) -> Result<(), Box<dyn Error>> {
    let loaded = session::load(settings, log, {
        let replies = replies.clone();
        move |status| {
            let _ = replies.send(Reply::Status(status));
        }
    })?;
    let mut harness = loaded.harness;
    let _ = replies.send(Reply::Ready(loaded.loading));
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
            if let Err(e) = session::reopen(settings, &mut harness, on_status()) {
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
                server(settings, &mut harness, &mut stopped, action, on_status).map(|notice| {
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
        };
        if let Err(e) = result {
            let _ = replies.send(Reply::Error(e.to_string()));
        }
        let _ = replies.send(Reply::Done(harness.stats()));
    }
    Ok(())
}

/// Does what `/server` asks, `action`, for the session `harness` runs,
/// which `stopped` says is not on a server: returns what to tell of it.
/// `on_status` makes what is told what the server does as it starts.
fn server<F: FnMut(session::Status) + 'static>(
    settings: &Settings,
    harness: &mut Harness<Client>,
    stopped: &mut bool,
    action: ServerAction,
    on_status: impl Fn() -> F,
) -> Result<String, Box<dyn Error>> {
    match action {
        ServerAction::Show => Ok(match session::running() {
            None => "No dwim server is running · the next message starts it".to_string(),
            Some(welcome) => {
                let device = if *stopped {
                    welcome.device.to_uppercase()
                } else {
                    harness.conversation().device().to_string()
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
                if !session::same_build(&welcome) {
                    notice.push_str(" · from another build of dwim");
                }
                if let Some(path) = session::log_path() {
                    notice.push_str(&format!(" · log {}", view::tilde(&path)));
                }
                notice
            }
        }),
        ServerAction::Stop if *stopped => {
            Ok("The dwim server is stopped · the next message starts it".to_string())
        }
        ServerAction::Stop => {
            session::stop(harness.conversation())?;
            *stopped = true;
            Ok("Stopped the dwim server · the next message starts it again".to_string())
        }
        ServerAction::Start if !*stopped => {
            Ok("The dwim server is running · /server restart restarts it".to_string())
        }
        ServerAction::Start | ServerAction::Restart => {
            if !*stopped {
                session::stop(harness.conversation())?;
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
