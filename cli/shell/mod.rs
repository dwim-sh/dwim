//! The shell: an input box at the bottom of the terminal, with the model
//! answering above it. The model runs on a thread of its own so the screen
//! stays responsive while it downloads, loads, and generates.

mod cmd;
pub mod picker;
mod view;
mod worker;

use std::{
    env,
    error::Error,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, TryRecvError},
    },
    thread,
    time::{Duration, Instant},
};

use crossterm::{
    event::{self, Event as Key, KeyCode, KeyEvent, KeyEventKind, KeyModifiers},
    style::Stylize,
};
use dwim::harness::{self, Event, Log};
use dwim::model::{Device, Size, Variant};
use dwim::providers::{
    Kind, hosted,
    openrouter::{self, login::Login},
};
use dwim::sys::browser;
use dwim::tools::plan::Plan;

use crate::{
    session::{self, Settings},
    tui::{self, Input, Line, Screen, span},
};

use cmd::Command;
use picker::{Picker, Providers};
use view::Segment;
use worker::{Reply, Request, Worker};

/// How often the screen is redrawn while the model is busy, to animate the
/// spinner.
const TICK: Duration = Duration::from_millis(80);

/// Runs the shell until the user quits, showing the model's thinking and
/// what it reads if `verbose` is set, or once ctrl+o is pressed, and
/// keeping a transcript of the session in `log`.
pub fn run(settings: Settings, verbose: bool, log: Log) -> Result<(), Box<dyn Error>> {
    let transcript = log.path().map(Path::to_path_buf);
    let variant = settings.model;
    let provider = session::provider(settings.provider, variant)?;
    let context = settings.context.unwrap_or(hosted::CONTEXT);
    let model = variant.name().to_string();
    let device = settings.device;
    let thinking_budget = settings.thinking_budget;
    let worker = Worker::spawn(settings, log);

    let mut app = App {
        screen: Screen::new()?,
        model,
        variant,
        provider,
        context,
        device,
        device_name: None,
        size: None,
        transcript,
        cwd: env::current_dir().unwrap_or_default(),
        input: Input::default(),
        picker: None,
        providers: None,
        signin: None,
        history: Vec::new(),
        recalled: None,
        status: match provider {
            Some(_) => Status::Connecting(Instant::now()),
            None => Status::Idle,
        },
        ready: false,
        started: Instant::now(),
        loading: Duration::ZERO,
        stats: harness::Stats::default(),
        reply: String::new(),
        segment: Segment::Text,
        reply_width: 0,
        committed: 0,
        interrupted: false,
        call: String::new(),
        verbose,
        thinking_budget,
        auto: false,
        plan: Plan::default(),
        doing: None,
        redirect: None,
        unheeded: false,
        lines: Vec::new(),
        blank: false,
        worker,
    };
    if provider.is_none() {
        let (columns, _) = app.screen.size();
        let lines = view::nowhere(
            &session::nowhere(variant),
            openrouter::key().is_some(),
            columns,
        );
        app.say(&lines);
    }
    let result = app.run();
    // Leave the conversation on screen, but not the input box, with where
    // the transcript is as the last word and a blank line between it and
    // whatever the shell prints next.
    if let Some(path) = &app.transcript {
        app.lines.push(vec![
            span(format!("  Transcript in {}", view::tilde(path))).dark_grey(),
        ]);
    }
    app.lines.push(Line::new());
    app.screen.draw(&app.lines, &[], (0, 0))?;
    // With no more messages to answer, and the turn in progress stopped,
    // the model's thread ends the transcript; while the model is still
    // loading, there is no transcript to end.
    let ready = app.ready;
    app.into_worker().finish(ready);
    result
}

/// A sign-in to OpenRouter under way.
struct Signin {
    /// The address that sends the browser to OpenRouter, and whether the
    /// browser was opened at it.
    url: String,
    opened: bool,
    since: Instant,
    /// Set to call the sign-in off, which dropping it does.
    cancelled: Arc<AtomicBool>,
    /// The key, once the user has agreed in the browser, or why there is
    /// none.
    key: Receiver<Result<Option<String>, String>>,
}

impl Signin {
    /// The sign-in as rows, and which of them the cursor is on: what it
    /// waits for, and where to go if the browser is not there.
    fn rows(&self) -> (Vec<Line>, usize) {
        const SPINNER: [&str; 4] = ["◐", "◓", "◑", "◒"];
        let frame = SPINNER[(self.since.elapsed().as_millis() / 150) as usize % SPINNER.len()];
        let go = if self.opened {
            "If the browser did not open, go to "
        } else {
            "Open your browser at "
        };
        let rows = vec![
            vec![span("  Sign in to OpenRouter").bold()],
            Line::new(),
            vec![
                span(format!(
                    "  {frame} Waiting for you to agree in the browser…"
                )),
                span(format!(" {}s", self.since.elapsed().as_secs())).dark_grey(),
            ],
            vec![
                span(format!("    {go}")).dark_grey(),
                span(&self.url).underlined(),
            ],
            Line::new(),
            vec![
                span("    OpenRouter makes a key for dwim, which is kept on this machine")
                    .dark_grey(),
            ],
        ];
        (rows, 2)
    }
}

impl Drop for Signin {
    fn drop(&mut self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }
}

/// What the model is doing.
enum Status {
    /// Downloading one of the model's files, since when: how many bytes so
    /// far, out of how many if that is known.
    Downloading {
        since: Instant,
        file: String,
        done: u64,
        total: Option<u64>,
    },
    /// Reaching the server, since when.
    Connecting(Instant),
    /// Waiting for a server that is not as asked for to go, since when.
    Restarting(Instant),
    /// Waiting for the server to start up, since when: how many of the
    /// model's tensors it has loaded so far, out of how many, once it says.
    Starting {
        since: Instant,
        done: usize,
        total: usize,
    },
    /// Opening a session on the server, since when.
    Opening(Instant),
    /// Reading the system prompt, since when: how many of its tokens so
    /// far, out of how many.
    Prompting {
        since: Instant,
        read: usize,
        total: usize,
    },
    Idle,
    /// Moving the conversation to another model, since when.
    Switching(Instant),
    /// Reading the prompt, before the first token of the reply.
    Reading(Instant),
    /// Running a tool the model called, since when: the model has nothing
    /// to do until the tool's output arrives.
    Running(Instant),
    /// Waiting for the model, which another session has, since when.
    Waiting(Instant),
    Generating {
        start: Instant,
        tokens: usize,
    },
}

struct App {
    screen: Screen,
    model: String,
    /// The model by size, where it runs, if anywhere yet, and the tokens
    /// of conversation a session at a provider has room for.
    variant: Variant,
    provider: Option<Kind>,
    context: usize,
    /// The kind of device the model runs on, and its name once it is open.
    device: Device,
    device_name: Option<String>,
    /// The room the model has for conversation, once it is sized.
    size: Option<Size>,
    /// Where the transcript is kept, if one is.
    transcript: Option<PathBuf>,
    cwd: PathBuf,
    input: Input,
    /// The model picker, while it is open in place of the input box.
    picker: Option<Picker>,
    /// The provider picker, while it is open in place of the input box.
    providers: Option<Providers>,
    /// The sign-in to OpenRouter under way, shown in place of the input
    /// box.
    signin: Option<Signin>,
    /// Messages sent so far, and which one is recalled into the input.
    history: Vec<String>,
    recalled: Option<usize>,
    status: Status,
    /// Whether the model is loaded, and the session ready for a message.
    /// Until it is, the slash commands that need no session still work,
    /// and `/model` picks another model to load instead.
    ready: bool,
    /// When the shell started, how long loading the model took, and where
    /// the conversation's time has gone, as of the last reply.
    started: Instant,
    loading: Duration,
    stats: harness::Stats,
    /// The part of the reply being generated, which part it is, the width
    /// it is wrapped at, and how many of its lines have been printed.
    reply: String,
    segment: Segment,
    reply_width: usize,
    committed: usize,
    interrupted: bool,
    /// The name of the tool last called, whose output comes next.
    call: String,
    /// Whether the model's thinking and the files it reads are shown.
    /// While the thought is hidden, it is kept but none of it is printed,
    /// so showing it prints all of it so far; a read hidden shows only
    /// its last line, which says what was read.
    verbose: bool,
    /// Most tokens of a thought, if they are limited, as the harness last
    /// said.
    thinking_budget: Option<usize>,
    /// Whether auto mode is on, as the harness last said.
    auto: bool,
    /// The plan of the work, as the harness last reported it, and the call
    /// the model last made, which is what it is doing in the step under
    /// way.
    plan: Plan,
    doing: Option<String>,
    /// What the user wrote while the model worked: the model is stopped
    /// for it, and it is sent once the turn has ended.
    redirect: Option<String>,
    /// Whether the model has yet to answer the user's last message: until
    /// it calls plan, which it is told to do first, or the turn ends, the
    /// plan shown is from before the message. Another call says nothing of
    /// whether the message was taken in: the model may go on with the step
    /// it was at as if there were none.
    unheeded: bool,
    /// Lines waiting to be printed above the live region, and whether the
    /// last line printed there was blank.
    lines: Vec<Line>,
    blank: bool,
    worker: Worker,
}

impl App {
    /// Closes the screen, leaving the model's thread.
    fn into_worker(self) -> Worker {
        self.worker
    }

    fn run(&mut self) -> Result<(), Box<dyn Error>> {
        let mut dirty = true;
        loop {
            if dirty || !matches!(self.status, Status::Idle) || self.signin.is_some() {
                self.draw()?;
                dirty = false;
            }
            if event::poll(TICK)? {
                match event::read()? {
                    Key::Key(key) if key.kind != KeyEventKind::Release => {
                        if self.key(key) {
                            return Ok(());
                        }
                    }
                    Key::Paste(text) => self.input.insert(&tui::sanitize(&text)),
                    _ => {}
                }
                dirty = true;
            }
            loop {
                match self.worker.try_recv() {
                    Ok(reply) => self.reply(reply)?,
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        return Err("the model stopped unexpectedly".into());
                    }
                }
                dirty = true;
            }
            dirty |= self.signed_in();
        }
    }

    /// Handles a key press, returning whether to quit.
    fn key(&mut self, key: KeyEvent) -> bool {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let busy = !matches!(self.status, Status::Idle);
        if self.picker.is_some() {
            self.pick(key);
            return false;
        }
        if self.providers.is_some() {
            self.pick_provider(key);
            return false;
        }
        if self.signin.is_some() {
            if key.code == KeyCode::Esc || (ctrl && key.code == KeyCode::Char('c')) {
                self.signin = None;
                self.say(&view::notice("Sign-in to OpenRouter cancelled"));
            }
            return false;
        }
        match key.code {
            KeyCode::Char('c') if ctrl => {
                if matches!(
                    self.status,
                    Status::Reading(_)
                        | Status::Running(_)
                        | Status::Waiting(_)
                        | Status::Generating { .. }
                ) {
                    self.interrupt();
                } else if !self.input.is_empty() {
                    self.input.take();
                } else {
                    return true;
                }
            }
            KeyCode::Char('d') if ctrl && self.input.is_empty() => return true,
            // With no turn to stop, esc sets a plan with steps left aside.
            KeyCode::Esc if !busy && self.ready && self.waiting() => {
                self.plan = Plan::default();
                self.say(&[vec![span("● The plan is set aside").dark_grey()]]);
                self.worker.send(Request::SetPlanAside);
            }
            KeyCode::Esc => self.interrupt(),
            KeyCode::Enter
                if key
                    .modifiers
                    .intersects(KeyModifiers::SHIFT | KeyModifiers::ALT) =>
            {
                self.input.insert("\n")
            }
            KeyCode::Char('j') if ctrl => self.input.insert("\n"),
            KeyCode::Char('o') if ctrl => {
                self.verbose = !self.verbose;
                if self.verbose
                    && self.segment == Segment::Thought
                    && matches!(self.status, Status::Generating { .. })
                {
                    self.commit(false);
                }
            }
            KeyCode::Enter
                if !busy || (!self.ready && self.input.text().trim().starts_with('/')) =>
            {
                self.submit()
            }
            // While the model works, a message changes its course: the
            // model is stopped to take it in, and goes on from the plan.
            KeyCode::Enter if self.working() && !self.input.text().trim().starts_with('/') => {
                self.change_course()
            }
            KeyCode::Enter => {}
            KeyCode::Char('a') if ctrl => self.input.home(),
            KeyCode::Char('e') if ctrl => self.input.end(),
            KeyCode::Char('u') if ctrl => self.input.kill_to_start(),
            KeyCode::Char('k') if ctrl => self.input.kill_to_end(),
            KeyCode::Char('w') if ctrl => self.input.delete_word(),
            KeyCode::Backspace if key.modifiers.contains(KeyModifiers::ALT) => {
                self.input.delete_word()
            }
            KeyCode::Char(c) if !ctrl => self.input.insert(c.encode_utf8(&mut [0; 4])),
            KeyCode::Backspace => self.input.backspace(),
            KeyCode::Delete => self.input.delete(),
            KeyCode::Left => self.input.left(),
            KeyCode::Right => self.input.right(),
            KeyCode::Home => self.input.home(),
            KeyCode::End => self.input.end(),
            KeyCode::Up => self.recall(-1),
            KeyCode::Down => self.recall(1),
            _ => {}
        }
        false
    }

    /// Handles a key press while the model picker is open: left and right
    /// change the provider, up and down move over the models it runs,
    /// enter picks, and esc closes the picker. Picking the model the
    /// session is on, where it runs, shows it; picking another, or another
    /// provider of it, moves the conversation there.
    fn pick(&mut self, key: KeyEvent) {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let Some(picker) = &mut self.picker else {
            return;
        };
        match key.code {
            KeyCode::Up => picker.step(false),
            KeyCode::Down => picker.step(true),
            KeyCode::Char('p') if ctrl => picker.step(false),
            KeyCode::Char('n') if ctrl => picker.step(true),
            KeyCode::Left => picker.turn(false),
            KeyCode::Right | KeyCode::Tab => picker.turn(true),
            KeyCode::Esc => self.picker = None,
            KeyCode::Char('c') if ctrl => self.picker = None,
            KeyCode::Enter => {
                let picked = picker.picked();
                self.picker = None;
                match picked {
                    Some((provider, variant))
                        if Some(provider) == self.provider && variant == self.variant =>
                    {
                        let lines = self.model_listing();
                        self.say(&lines);
                    }
                    Some((provider, variant)) => self.switch(provider, variant),
                    None => {}
                }
            }
            _ => {}
        }
    }

    /// Handles a key press while the provider picker is open: up and down
    /// move over the providers, enter picks, and esc closes the picker.
    /// Picking the provider the session is on shows it, but for
    /// OpenRouter, which is signed in to again, for a key that has expired
    /// or been revoked; picking another moves the conversation there, after
    /// signing in to it if there is no key to it.
    fn pick_provider(&mut self, key: KeyEvent) {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let Some(providers) = &mut self.providers else {
            return;
        };
        match key.code {
            KeyCode::Up => providers.step(false),
            KeyCode::Down => providers.step(true),
            KeyCode::Char('p') if ctrl => providers.step(false),
            KeyCode::Char('n') if ctrl => providers.step(true),
            KeyCode::Esc => self.providers = None,
            KeyCode::Char('c') if ctrl => self.providers = None,
            KeyCode::Enter => {
                let picked = providers.picked();
                self.providers = None;
                match picked {
                    Some(Kind::OpenRouter)
                        if openrouter::key().is_none()
                            || self.provider == Some(Kind::OpenRouter) =>
                    {
                        self.sign_in()
                    }
                    Some(provider) if Some(provider) == self.provider => {
                        self.say(&view::notice(view::provider_notice(provider, self.variant)));
                    }
                    Some(provider) => self.switch(provider, self.variant_at(provider)),
                    None => {}
                }
            }
            _ => {}
        }
    }

    /// The model the session goes on with at `provider`: the one it is on,
    /// but for the large one, which does not run on this machine or at
    /// Cerebras.
    fn variant_at(&self, provider: Kind) -> Variant {
        match provider {
            Kind::Local | Kind::Cerebras => Variant::Medium,
            Kind::OpenRouter | Kind::Vercel => self.variant,
        }
    }

    /// Moves the conversation to `variant` at `provider`.
    fn switch(&mut self, provider: Kind, variant: Variant) {
        self.reply.clear();
        self.committed = 0;
        self.interrupted = false;
        self.status = Status::Switching(Instant::now());
        self.worker.send(Request::Model(provider, variant));
    }

    /// Starts signing in to OpenRouter: opens the browser, and waits for
    /// the key on a thread of its own.
    fn sign_in(&mut self) {
        let login = match Login::start() {
            Ok(login) => login,
            Err(e) => {
                self.say(&view::notice(format!("Cannot sign in to OpenRouter: {e}")));
                return;
            }
        };
        let url = login.url();
        let opened = browser::open(&url).is_ok();
        let cancelled = Arc::new(AtomicBool::new(false));
        let (sender, key) = mpsc::channel();
        thread::spawn({
            let cancelled = cancelled.clone();
            move || {
                let _ = sender.send(login.wait(&cancelled).map_err(|e| e.to_string()));
            }
        });
        self.signin = Some(Signin {
            url,
            opened,
            since: Instant::now(),
            cancelled,
            key,
        });
    }

    /// Goes on from a sign-in that is over, if it is: keeps the key, and
    /// moves the conversation to OpenRouter, or opens it there again with
    /// the new key. Returns whether it was over.
    fn signed_in(&mut self) -> bool {
        let Some(signin) = &self.signin else {
            return false;
        };
        let key = match signin.key.try_recv() {
            Ok(key) => key,
            Err(TryRecvError::Empty) => return false,
            Err(TryRecvError::Disconnected) => Err("the sign-in stopped unexpectedly".to_string()),
        };
        self.signin = None;
        match key.and_then(|key| match key {
            Some(key) => openrouter::login::save(&key)
                .map(Some)
                .map_err(|e| format!("cannot keep the key: {e}")),
            None => Ok(None),
        }) {
            Ok(Some(path)) => {
                self.say(&view::notice(format!(
                    "Signed in to OpenRouter · key kept in {}",
                    view::tilde(&path)
                )));
                if env::var(openrouter::KEY).is_ok_and(|key| !key.trim().is_empty()) {
                    self.say(&view::notice(format!(
                        "{} is set, and is used ahead of the key kept: unset it to use this one",
                        openrouter::KEY
                    )));
                }
                self.switch(Kind::OpenRouter, self.variant);
            }
            Ok(None) => {}
            Err(e) => {
                self.lines.push(vec![
                    span("● ").red(),
                    span(format!("error: signing in to OpenRouter: {e}")),
                ]);
                self.lines.push(Line::new());
            }
        }
        true
    }

    /// Stops the reply being generated, if any.
    fn interrupt(&mut self) {
        if self.working() {
            self.worker.interrupt();
            self.interrupted = true;
        }
    }

    /// Whether the model is at work on a turn.
    fn working(&self) -> bool {
        self.ready
            && matches!(
                self.status,
                Status::Reading(_)
                    | Status::Running(_)
                    | Status::Waiting(_)
                    | Status::Generating { .. }
            )
    }

    /// Whether a plan with steps left waits for the user.
    fn waiting(&self) -> bool {
        !self.plan.is_empty() && !self.plan.finished() && !self.working()
    }

    /// Takes what the input box holds as a message that changes the course
    /// of the work under way: stops the model, and keeps the message to
    /// send once the turn has ended. The plan is the harness's to keep, so
    /// nothing done is lost by stopping: the model gets the message with
    /// the plan as it stands, and goes on from there.
    fn change_course(&mut self) {
        let message = self.input.text().trim().to_string();
        if message.is_empty() {
            return;
        }
        self.input.take();
        self.recalled = None;
        if self.history.last() != Some(&message) {
            self.history.push(message.clone());
        }
        self.redirect = Some(match self.redirect.take() {
            Some(earlier) => format!("{earlier}\n\n{message}"),
            None => message,
        });
        self.interrupt();
    }

    /// Steps through the messages sent so far, like a shell's history.
    fn recall(&mut self, step: isize) {
        if self.input.text().contains('\n') || self.history.is_empty() {
            return;
        }
        let last = self.history.len() - 1;
        self.recalled = match (self.recalled, step) {
            (None, -1) => Some(last),
            (None, _) => None,
            (Some(i), -1) => Some(i.saturating_sub(1)),
            (Some(i), _) if i < last => Some(i + 1),
            (Some(_), _) => None,
        };
        let text = self
            .recalled
            .map(|i| self.history[i].clone())
            .unwrap_or_default();
        self.input.set(text);
    }

    fn submit(&mut self) {
        let message = self.input.text().trim().to_string();
        if message.is_empty() {
            return;
        }
        self.input.take();
        self.recalled = None;
        if self.history.last() != Some(&message) {
            self.history.push(message.clone());
        }

        self.echo(&message);

        let request = match message.strip_prefix('/').map(Command::parse) {
            None => Request::Message(message),
            Some(Ok(Command::Compact)) => Request::Compact,
            Some(Ok(Command::SetThinkingBudget(tokens))) => Request::ThinkingBudget(tokens),
            Some(Ok(Command::SetAuto(auto))) => Request::Auto(auto),
            Some(Ok(Command::Server(action))) => Request::Server(action),
            Some(Ok(Command::Model)) => {
                let groups = session::providers();
                if groups.iter().all(|group| group.unavailable.is_some()) {
                    self.say(&view::notice(view::no_provider(false)));
                } else {
                    self.picker = Some(Picker::new(groups, self.provider, self.variant));
                }
                return;
            }
            Some(Ok(Command::Provider)) => {
                self.providers = Some(Providers::new(
                    session::provider_choices(self.provider),
                    self.provider,
                ));
                return;
            }
            Some(Ok(command)) => {
                self.command(command);
                return;
            }
            Some(Err(what)) => {
                self.say(&view::notice(what));
                return;
            }
        };

        if !self.ready {
            self.say(&view::notice(match self.provider {
                Some(_) => "The model is still loading · /model picks another".to_string(),
                None => view::no_provider(openrouter::key().is_some()),
            }));
            return;
        }
        self.start(request);
    }

    /// Prints a message from the user into the conversation.
    fn echo(&mut self, message: &str) {
        let (columns, _) = self.screen.size();
        self.lines.push(Line::new());
        for (i, line) in tui::wrap(message, columns.saturating_sub(2))
            .into_iter()
            .enumerate()
        {
            let prompt = if i == 0 {
                span("› ").bold()
            } else {
                span("  ")
            };
            self.lines.push(vec![prompt, span(line).bold()]);
        }
        self.lines.push(Line::new());
    }

    /// Asks the model's thread for `request`, a turn, starting a reply.
    fn start(&mut self, request: Request) {
        let (columns, _) = self.screen.size();
        self.reply.clear();
        self.segment = Segment::Text;
        self.reply_width = columns.saturating_sub(2);
        self.committed = 0;
        self.interrupted = false;
        self.doing = None;
        self.unheeded = matches!(request, Request::Message(_));
        self.status = Status::Reading(Instant::now());
        self.worker.send(request);
    }

    fn reply(&mut self, reply: Reply) -> Result<(), Box<dyn Error>> {
        match reply {
            Reply::Status(status) => self.status_update(status),
            Reply::Ready(loading) => {
                self.loading = loading;
                self.ready = true;
                self.status = Status::Idle;
            }
            Reply::Reopened => self.status = Status::Reading(Instant::now()),
            Reply::Event(event) => self.event(event),
            Reply::Done(stats) => {
                self.stats = stats;
                self.reply.truncate(self.reply.trim_end().len());
                self.commit(true);
                self.doing = None;
                self.unheeded = false;
                self.status = Status::Idle;
                // What the user wrote while the model worked is the next
                // message, which the model was stopped for.
                match self.redirect.take() {
                    Some(message) if self.ready => {
                        self.finish();
                        self.echo(&message);
                        self.start(Request::Message(message));
                    }
                    _ if self.interrupted => {
                        self.lines.push(vec![span("  ⎿ Interrupted").dark_grey()]);
                    }
                    _ => {}
                }
            }
            Reply::Notice(text) => {
                self.finish();
                self.lines.push(vec![span(format!("● {text}")).dark_grey()]);
                self.lines.push(Line::new());
            }
            Reply::ThinkingBudget(tokens) => {
                self.thinking_budget = tokens;
                self.finish();
                self.lines.push(vec![
                    span(format!("● {}", view::budget_set(tokens))).dark_grey(),
                ]);
                self.lines.push(Line::new());
            }
            Reply::Auto(auto) => {
                self.auto = auto;
                self.finish();
                self.lines.push(vec![
                    span(format!("● {}", view::auto_notice(auto))).dark_grey(),
                ]);
                self.lines.push(Line::new());
            }
            Reply::Model { variant, provider } => {
                let notice = match self.provider {
                    Some(_) => view::switched(variant, provider),
                    None => view::provider_notice(provider, variant),
                };
                self.variant = variant;
                self.model = variant.name().to_string();
                self.provider = Some(provider);
                // Picked while the model was loading: the pick loads now.
                if !self.ready {
                    self.status = Status::Switching(Instant::now());
                }
                self.finish();
                self.lines
                    .push(vec![span(format!("● {notice}")).dark_grey()]);
                self.lines.push(Line::new());
            }
            Reply::Error(e) => {
                self.finish();
                self.lines
                    .push(vec![span("● ").red(), span(format!("error: {e}"))]);
                self.lines.push(Line::new());
            }
            Reply::Unopened(e) => {
                self.finish();
                self.lines
                    .push(vec![span("● ").red(), span(format!("error: {e}"))]);
                self.lines.push(Line::new());
                self.provider = None;
                self.device_name = None;
                self.size = None;
                self.status = Status::Idle;
            }
            Reply::Failed(e) => return Err(e.into()),
        }
        Ok(())
    }

    /// Shows what the server is doing while the session waits on it.
    fn status_update(&mut self, status: session::Status) {
        match status {
            session::Status::Downloading { file, done, total } => {
                // Time each file on its own, so the speed is of this one.
                let since = match &self.status {
                    Status::Downloading {
                        since, file: last, ..
                    } if *last == file => *since,
                    _ => Instant::now(),
                };
                self.status = Status::Downloading {
                    since,
                    file,
                    done,
                    total,
                };
            }
            session::Status::Device(name) => self.device_name = Some(name),
            session::Status::Sized(size) => self.size = Some(size),
            session::Status::Weights { done, total } => {
                let since = match self.status {
                    Status::Starting { since, .. } => since,
                    _ => Instant::now(),
                };
                self.status = if done >= total {
                    Status::Opening(Instant::now())
                } else {
                    Status::Starting { since, done, total }
                };
            }
            session::Status::Restarting => self.status = Status::Restarting(Instant::now()),
            session::Status::Starting => {
                self.status = Status::Starting {
                    since: Instant::now(),
                    done: 0,
                    total: 0,
                };
            }
            session::Status::Opening => {
                // A server this session started opens it once loaded.
                if !matches!(self.status, Status::Starting { .. }) {
                    self.status = Status::Opening(Instant::now());
                }
            }
            session::Status::Prompting { read, total } => {
                let since = match self.status {
                    Status::Prompting { since, .. } => since,
                    _ => Instant::now(),
                };
                self.status = Status::Prompting { since, read, total };
            }
            session::Status::Busy => {
                let since = match self.status {
                    Status::Reading(since) | Status::Waiting(since) => since,
                    _ => Instant::now(),
                };
                self.status = Status::Waiting(since);
            }
        }
    }

    /// Shows something that happened in the turn.
    fn event(&mut self, event: Event) {
        // The model has answered the message once a call of plan goes
        // through, even with the steps unchanged, which reports no plan.
        if let Event::Output(output) = &event
            && self.call == "plan"
            && !output.starts_with("error:")
        {
            self.unheeded = false;
        }
        if let Some(notice) = event.notice() {
            self.finish();
            if event == Event::Compacting {
                self.lines
                    .push(vec![span("● ").dark_grey(), span(notice).dark_grey()]);
                self.status = Status::Reading(Instant::now());
            } else {
                self.lines
                    .push(vec![span(format!("  ⎿ {notice}")).dark_grey()]);
                self.lines.push(Line::new());
                if let Event::Compacted { .. } = event {
                    self.status = Status::Reading(Instant::now());
                }
            }
            return;
        }
        match event {
            Event::Thought(text) => self.generated(Segment::Thought, &text),
            Event::Text(text) => self.generated(Segment::Text, &text),
            Event::Note(text) => self.generated(Segment::Note, &text),
            Event::Token => {
                self.status = match self.status {
                    Status::Generating { start, tokens } => Status::Generating {
                        start,
                        tokens: tokens + 1,
                    },
                    _ => Status::Generating {
                        start: Instant::now(),
                        tokens: 1,
                    },
                };
            }
            Event::Plan(plan) => {
                self.finish();
                let (columns, _) = self.screen.size();
                self.lines
                    .extend(view::plan_news(&self.plan, &plan, columns));
                // What the model was at was the step before's.
                if plan.done() != self.plan.done() {
                    self.doing = None;
                }
                self.plan = plan;
            }
            // The plan shows over the input box, and a step in the
            // conversation as it is finished: not the calls that say so,
            // unless one goes wrong.
            Event::Call { name, .. } if name == "plan" || name == "done" => {
                self.call = name;
                self.status = Status::Reading(Instant::now());
            }
            Event::Output(output)
                if (self.call == "plan" || self.call == "done")
                    && !output.starts_with("error:") => {}
            Event::Call { name, detail } => {
                self.finish();
                let detail: Vec<&str> = detail.split_whitespace().collect();
                let detail = detail.join(" ");
                self.lines.push(vec![
                    span("● "),
                    span(&name).bold(),
                    span(format!("({detail})")).dark_grey(),
                ]);
                self.doing = Some(format!("{name}({detail})"));
                self.call = name;
                self.status = Status::Running(Instant::now());
            }
            Event::Output(output) => {
                if self.call == "plan" || self.call == "done" {
                    self.finish();
                    self.lines
                        .push(vec![span("● "), span(self.call.clone()).bold()]);
                }
                let (columns, _) = self.screen.size();
                let output = tui::sanitize(&output);
                // A file read is long and the model's to read: unless
                // asked, show only the note that ends it, which says what
                // was read.
                let lines = if self.call == "read" && !self.verbose {
                    output.lines().last().into_iter().collect()
                } else {
                    output.lines().collect::<Vec<_>>()
                };
                let mut first = true;
                for line in lines {
                    for text in tui::wrap(line, columns.saturating_sub(4)) {
                        let prefix = if first { "  ⎿ " } else { "    " };
                        first = false;
                        self.lines
                            .push(vec![span(format!("{prefix}{text}")).dark_grey()]);
                    }
                }
                self.lines.push(Line::new());
                // The tool is done, and its output is the model's to read.
                self.status = Status::Reading(Instant::now());
            }
            Event::Cut | Event::Compacting | Event::Compacted { .. } => {}
        }
    }

    /// Adds a piece of the reply to the part being generated, or starts
    /// the next part if it is of a different kind.
    fn generated(&mut self, segment: Segment, text: &str) {
        if segment != self.segment {
            self.finish();
            self.segment = segment;
        }
        let text = tui::sanitize(text);
        // The model tends to open with blank lines.
        let text = if self.reply.is_empty() {
            text.trim_start()
        } else {
            &text
        };
        self.reply.push_str(text);
        if !matches!(self.status, Status::Generating { .. }) {
            self.status = Status::Generating {
                start: Instant::now(),
                tokens: 0,
            };
        }
        self.commit(false);
    }

    /// Prints the rest of the part of the reply being generated, with a
    /// blank line after it if there was anything to print, and starts over.
    fn finish(&mut self) {
        self.reply.truncate(self.reply.trim_end().len());
        self.commit(true);
        if self.committed > 0 {
            self.lines.push(Line::new());
        }
        self.reply.clear();
        self.committed = 0;
    }

    /// Prints the lines of the reply that are finished: all of them once the
    /// reply is done, and otherwise all but the last, which may still grow.
    /// Blank lines are held back until something follows them.
    fn commit(&mut self, done: bool) {
        if self.segment == Segment::Thought && !self.verbose {
            return;
        }
        let lines = tui::wrap(&self.reply, self.reply_width);
        let end = if done { lines.len() } else { lines.len() - 1 };
        let end = lines[..end]
            .iter()
            .rposition(|line| !line.is_empty())
            .map_or(0, |i| i + 1);
        if end > self.committed {
            let segment = self.segment;
            let committed = self.committed;
            self.lines.extend(
                lines
                    .into_iter()
                    .enumerate()
                    .take(end)
                    .skip(committed)
                    .map(|line| view::reply_line(segment, line)),
            );
            self.committed = end;
        }
    }

    /// Runs a slash command that the shell answers by itself, printing
    /// what it has to say: a report under a bullet, the way a tool's output
    /// follows its call, or one line.
    fn command(&mut self, command: Command) {
        let (columns, _) = self.screen.size();
        let lines = match command {
            Command::Stats => {
                let details =
                    session::accounting(&self.stats, self.loading, self.started.elapsed());
                view::report(
                    vec![span("● "), span("This session").bold()],
                    details,
                    columns,
                )
            }
            Command::Transcript => view::notice(match &self.transcript {
                Some(path) => format!("Transcript in {}", view::tilde(path)),
                None => "No transcript is kept".to_string(),
            }),
            Command::Help => view::help(columns),
            Command::ThinkingBudget => view::notice(view::budget_notice(self.thinking_budget)),
            Command::Auto => view::notice(view::auto_notice(self.auto)),
            Command::Model
            | Command::Provider
            | Command::Compact
            | Command::SetThinkingBudget(_)
            | Command::SetAuto(_)
            | Command::Server(_) => {
                return;
            }
        };
        self.say(&lines);
    }

    /// The model the session is on: where it runs, and the room it has.
    fn model_listing(&self) -> Vec<Line> {
        let (columns, _) = self.screen.size();
        let Some(provider) = self.provider else {
            return view::notice(view::no_provider(openrouter::key().is_some()));
        };
        match provider.host() {
            Some(host) => view::hosted_listing(host, self.variant, self.context, columns),
            None => view::model_listing(
                &self.model,
                self.device,
                self.device_name.as_deref(),
                self.size.as_ref(),
                columns,
            ),
        }
    }

    /// Prints `lines` above the live region, cut to the terminal's width,
    /// with a blank line after them.
    fn say(&mut self, lines: &[Line]) {
        let (columns, _) = self.screen.size();
        self.lines
            .extend(lines.iter().map(|line| tui::truncate(line, columns)));
        self.lines.push(Line::new());
    }

    fn draw(&mut self) -> Result<(), Box<dyn Error>> {
        let (columns, rows) = self.screen.size();
        let mut live = Vec::new();

        // The line of the reply still being generated. An empty reply has no
        // line yet: after its thought, the model writes blank lines, or a
        // tool call, which is shown only once it is complete, and a bullet
        // alone would stand for either.
        if matches!(self.status, Status::Generating { .. })
            && (self.segment == Segment::Text || self.verbose)
            && !self.reply.is_empty()
        {
            let lines = tui::wrap(&self.reply, self.reply_width);
            let segment = self.segment;
            live.extend(
                lines
                    .into_iter()
                    .enumerate()
                    .skip(self.committed)
                    .map(|line| view::reply_line(segment, line)),
            );
        }
        // A blank line between the conversation and the input box, unless
        // the conversation already ends in one: the message a reply is to,
        // and a tool's output, do.
        let blank = self.lines.last().map_or(self.blank, Line::is_empty);
        if !live.is_empty() || !blank {
            live.push(Line::new());
        }
        // The plan as it stands, while it has steps left or the model is
        // still at work, and what the user wrote that the model is being
        // stopped for.
        let working = self.working();
        if self.plan.is_empty() && working {
            live.extend(view::no_plan(self.doing.as_deref(), columns));
            live.push(Line::new());
        }
        if !self.plan.is_empty() && (working || !self.plan.finished()) {
            live.extend(view::plan(
                &self.plan,
                self.doing.as_deref().filter(|_| working),
                working,
                self.unheeded || self.redirect.is_some(),
                columns,
            ));
            live.push(Line::new());
        }
        if let Some(message) = &self.redirect {
            for (i, text) in tui::wrap(message, columns.saturating_sub(4))
                .into_iter()
                .enumerate()
            {
                let prefix = if i == 0 { "› " } else { "  " };
                live.push(vec![span(format!("{prefix}{text}")).dark_grey()]);
            }
            live.push(vec![
                span("  ⎿ stopping the model to take this in").dark_grey(),
            ]);
            live.push(Line::new());
        }
        if let Some(status) = self.status() {
            live.push(status);
        }

        let border = |left: &str, right: &str| {
            vec![
                span(format!(
                    "{left}{}{right}",
                    "─".repeat(columns.saturating_sub(2))
                ))
                .dark_grey(),
            ]
        };
        // A picker, or the sign-in, in place of the input box while it is
        // open.
        let panel = if let Some(picker) = &self.picker {
            Some((
                picker.rows(),
                "  ↑↓ model · ←→ provider · enter to select · esc to cancel",
            ))
        } else if let Some(providers) = &self.providers {
            Some((
                providers.rows(),
                "  ↑↓ provider · enter to select · esc to cancel",
            ))
        } else {
            self.signin
                .as_ref()
                .map(|signin| (signin.rows(), "  esc to cancel"))
        };
        if let Some(((rows, row), keys)) = panel {
            live.push(vec![span("─".repeat(columns)).dark_grey()]);
            let cursor = (live.len() + row, 2);
            live.extend(rows);
            live.push(Line::new());
            live.push(vec![span(keys).dark_grey()]);
            let lines = std::mem::take(&mut self.lines);
            if let Some(last) = lines.last() {
                self.blank = last.is_empty();
            }
            self.screen.draw(&lines, &live, cursor)?;
            return Ok(());
        }

        // The input box, as tall as its text but no taller than the screen,
        // scrolled to show the cursor.
        let (text, (row, column)) = self.input.layout(columns.saturating_sub(6).max(1));
        let height = rows.saturating_sub(live.len() + 3).max(1);
        let first = (row + 1).saturating_sub(height);
        live.push(border("╭", "╮"));
        let cursor = (live.len() + row - first, 4 + column);
        for (i, line) in text.into_iter().enumerate().skip(first).take(height) {
            let prompt = match (i, &self.status) {
                (0, Status::Idle) => span("› ").bold(),
                (0, _) => span("› ").dark_grey(),
                _ => span("  "),
            };
            let content = if self.input.is_empty() && self.provider.is_none() {
                span("Type /provider to connect a model").dark_grey()
            } else if self.input.is_empty() && working {
                span("Say what to change, and it changes course · esc stops it").dark_grey()
            } else if self.input.is_empty() && self.waiting() {
                span("Say go on, or what to change in the plan").dark_grey()
            } else if self.input.is_empty() {
                span("Ask anything, or describe what to build").dark_grey()
            } else {
                span(line)
            };
            live.push(tui::boxed(vec![prompt, content], columns));
        }
        live.push(border("╰", "╯"));
        live.push(self.footer());

        let lines = std::mem::take(&mut self.lines);
        if let Some(last) = lines.last() {
            self.blank = last.is_empty();
        }
        self.screen.draw(&lines, &live, cursor)?;
        Ok(())
    }

    /// The line over the input box: what the model is doing, with a
    /// spinner, while it's busy. While the model loads, which the footer
    /// tells of, it says what can be done meanwhile instead.
    fn status(&self) -> Option<Line> {
        if !self.ready {
            let doing = match self.status {
                Status::Downloading { .. } => "downloading",
                _ => "loading",
            };
            return match self.provider {
                None => Some(vec![
                    span("● No model yet"),
                    span(format!(
                        " · {}",
                        view::connect_hint(openrouter::key().is_some())
                    ))
                    .dark_grey(),
                ]),
                Some(Kind::Local) => Some(vec![
                    span(format!("● Local model is {doing}")),
                    span(" · /model picks another model or provider").dark_grey(),
                ]),
                Some(Kind::OpenRouter | Kind::Vercel | Kind::Cerebras) => None,
            };
        }
        let (frame, verb, details) = self.doing()?;
        Some(vec![
            span(format!("{frame} {verb}")).bold(),
            span(format!(" {details}")).dark_grey(),
        ])
    }

    /// The line under the input box: the directory the agent works in, at
    /// the left, and while the model loads, how far along that is, at the
    /// right.
    fn footer(&self) -> Line {
        let (columns, _) = self.screen.size();
        let left = format!("  {}", view::tilde(&self.cwd));
        let right = match self.doing() {
            Some((frame, verb, details)) if !self.ready => {
                let details: Vec<&str> = details.split_whitespace().collect();
                format!("{frame} {verb} {}  ", details.join(" "))
            }
            _ => String::new(),
        };
        let width = |text: &str| tui::width(&vec![span(text)]);
        let line = match columns.checked_sub(width(&left) + width(&right)) {
            _ if right.is_empty() => left,
            Some(pad) if pad >= 2 => format!("{left}{}{right}", " ".repeat(pad)),
            _ => format!("  {right}"),
        };
        tui::truncate(&vec![span(line).dark_grey()], columns)
    }

    /// What the model is doing while it's busy: a frame of the spinner,
    /// what it is doing, and how that goes.
    fn doing(&self) -> Option<(&'static str, String, String)> {
        const SPINNER: [&str; 4] = ["◐", "◓", "◑", "◒"];
        let seconds = |since: Instant| format!("{}s", since.elapsed().as_secs());
        let percent = |done: usize, total: usize| match total {
            0 => "  0%".to_string(),
            _ => format!("{:>3}%", done * 100 / total),
        };
        let frame =
            |since: Instant| SPINNER[(since.elapsed().as_millis() / 150) as usize % SPINNER.len()];
        let (verb, since, details) = match self.status {
            Status::Idle => return None,
            Status::Running(since) => {
                return Some((
                    frame(since),
                    format!("Running {}…", self.call),
                    format!("{} · esc to interrupt", seconds(since)),
                ));
            }
            Status::Downloading {
                since, done, total, ..
            } => (
                "Downloading model…",
                since,
                view::download_details(done, total, since.elapsed()),
            ),
            Status::Connecting(since) => ("Connecting to dwim server…", since, seconds(since)),
            Status::Restarting(since) => ("Restarting dwim server…", since, seconds(since)),
            Status::Opening(since) => ("Opening a session…", since, seconds(since)),
            Status::Starting { since, done, total } => (
                "Starting dwim server…",
                since,
                match total {
                    0 => seconds(since),
                    _ => format!("{} · {}", percent(done, total).trim_start(), seconds(since)),
                },
            ),
            Status::Prompting { since, read, total } => (
                "Reading system prompt…",
                since,
                format!("{} · {}", percent(read, total), seconds(since)),
            ),
            Status::Switching(since) => ("Switching model…", since, seconds(since)),
            Status::Reading(since) => (
                "Reading…",
                since,
                format!("{} · esc to interrupt", seconds(since)),
            ),
            Status::Waiting(since) => (
                "Waiting for the model…",
                since,
                format!(
                    "another session has it · {} · esc to interrupt",
                    seconds(since)
                ),
            ),
            Status::Generating { start, tokens } => (
                match self.segment {
                    Segment::Thought => "Thinking…",
                    Segment::Text => "Generating…",
                    Segment::Note => "Compacting…",
                },
                start,
                {
                    // No rate before a second has passed or a token arrived,
                    // and the word `tok/s` is left out with it.
                    let mut details = view::rate(tokens as f64, start.elapsed())
                        .map_or(String::new(), |rate| format!("{rate:>5.1} tok/s · "));
                    details.push_str(match (self.segment, self.verbose) {
                        (Segment::Thought, false) => "ctrl+o to show · ",
                        (Segment::Thought, true) => "ctrl+o to hide · ",
                        (Segment::Text | Segment::Note, _) => "",
                    });
                    details.push_str("esc to interrupt");
                    details
                },
            ),
        };
        Some((frame(since), verb.to_string(), details))
    }
}
