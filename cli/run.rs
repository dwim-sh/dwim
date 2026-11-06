//! The shell: an input box at the bottom of the terminal, with the model
//! answering above it. The model runs on a thread of its own so the screen
//! stays responsive while it downloads, loads, and generates.

use std::{
    env,
    error::Error,
    fmt::Write,
    ops::ControlFlow,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, Sender, TryRecvError},
    },
    thread,
    time::{Duration, Instant},
};

use crossterm::{
    event::{self, Event as Key, KeyCode, KeyEvent, KeyEventKind, KeyModifiers},
    style::Stylize,
};
use dwim::gpu::BACKEND;
use dwim::harness::{self, Event, Log};

use crate::{
    fetch::Progress,
    model::{self, Size},
    opts::{self, Device},
    session::{self, Loading, Settings},
    tui::{self, Input, Line, Screen, span},
};

/// How often the screen is redrawn while the model is busy, to animate the
/// spinner.
const TICK: Duration = Duration::from_millis(80);

/// Longest the shell waits, on quitting, for the model's thread to stop
/// the turn in progress and end the transcript.
const FINISH: Duration = Duration::from_secs(2);

/// Runs the shell until the user quits, showing the model's thinking and
/// what it reads if `verbose` is set, or once ctrl+o is pressed, and
/// keeping a transcript of the session in `log`.
pub fn run(settings: Settings, verbose: bool, log: Log) -> Result<(), Box<dyn Error>> {
    let transcript = log.path().map(Path::to_path_buf);
    let (requests, requests_rx) = mpsc::channel();
    let (replies_tx, replies) = mpsc::channel();
    let stop = Arc::new(AtomicBool::new(false));
    // Disconnected once the model's thread is done.
    let (done, finished) = mpsc::channel::<()>();
    let model = settings.model.clone();
    let device = settings.device;
    let thinking_budget = settings.thinking_budget;
    {
        let stop = stop.clone();
        thread::spawn(move || {
            let _done = done;
            if let Err(e) = work(&settings, log, requests_rx, &replies_tx, &stop) {
                let _ = replies_tx.send(Reply::Failed(e.to_string()));
            }
        });
    }

    let mut app = App {
        screen: Screen::new()?,
        model,
        device,
        device_name: None,
        size: None,
        transcript,
        cwd: env::current_dir().unwrap_or_default(),
        input: Input::default(),
        history: Vec::new(),
        recalled: None,
        status: Status::Loading {
            since: Instant::now(),
            done: 0,
            total: 0,
        },
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
        lines: Vec::new(),
        blank: false,
        requests,
        replies,
        stop,
    };
    let result = app.run();
    // Leave the conversation on screen, but not the input box, with where
    // the transcript is as the last word and a blank line between it and
    // whatever the shell prints next.
    if let Some(path) = &app.transcript {
        app.lines.push(vec![
            span(format!("  Transcript in {}", tilde(path))).dark_grey(),
        ]);
    }
    app.lines.push(Line::new());
    app.screen.draw(&app.lines, &[], (0, 0))?;
    // With no more messages to answer, and the turn in progress stopped,
    // the model's thread ends the transcript; while the model is still
    // loading, there is no transcript to end.
    let loaded = !matches!(
        app.status,
        Status::Downloading { .. } | Status::Loading { .. }
    );
    app.stop.store(true, Ordering::Relaxed);
    drop(app);
    if loaded {
        let _ = finished.recv_timeout(FINISH);
    }
    result
}

/// How the thinking budget reads when asked.
fn budget_notice(tokens: Option<usize>) -> String {
    match tokens {
        Some(tokens) => format!("Thinking budget: {tokens} tokens"),
        None => "No thinking budget".to_string(),
    }
}

/// How setting the thinking budget is acknowledged.
fn budget_set(tokens: Option<usize>) -> String {
    match tokens {
        Some(tokens) => format!("Thinking budget set to {tokens} tokens"),
        None => "Thinking budget lifted".to_string(),
    }
}

/// What the `/auto` command says of auto mode, and how setting it is
/// acknowledged.
fn auto_notice(auto: bool) -> String {
    if auto {
        "Auto mode on: the model goes on by itself until it says it is done".to_string()
    } else {
        "Auto mode off: the model asks when it has something to ask".to_string()
    }
}

/// The slash commands, each with the line `/help` says of it.
const COMMANDS: [(&str, &str); 7] = [
    (
        "/model",
        "the model, the device it runs on, and its context window",
    ),
    ("/stats", "where this session's time has gone"),
    ("/transcript", "where this session's transcript is"),
    ("/compact", "compact the conversation now"),
    (
        "/thinking-budget N",
        "cap the model's thinking at N tokens, 0 for no cap",
    ),
    (
        "/auto on|off",
        "go on by itself until it says it is done, or ask",
    ),
    ("/help", "this list"),
];

/// A slash command, as the user types it after the slash.
enum Command {
    Model,
    Stats,
    Transcript,
    Compact,
    /// Ask what the thinking budget is, or set it.
    ThinkingBudget,
    SetThinkingBudget(Option<usize>),
    /// Ask whether auto mode is on, or set it.
    Auto,
    SetAuto(bool),
    Help,
}

impl Command {
    /// The command `text` names, or what is wrong with it.
    fn parse(text: &str) -> Result<Self, String> {
        let (name, argument) = text
            .trim()
            .split_once(char::is_whitespace)
            .map_or((text.trim(), None), |(name, rest)| {
                (name, Some(rest.trim()))
            });
        match (name, argument) {
            ("model", None) => Ok(Command::Model),
            ("stats", None) => Ok(Command::Stats),
            ("transcript", None) => Ok(Command::Transcript),
            ("compact", None) => Ok(Command::Compact),
            ("help", None) => Ok(Command::Help),
            ("thinking-budget", None) => Ok(Command::ThinkingBudget),
            ("thinking-budget", Some(tokens)) => tokens
                .parse()
                .map(|tokens| Command::SetThinkingBudget(opts::budget(tokens)))
                .map_err(|_| {
                    "/thinking-budget takes a number of tokens, or 0 for no cap".to_string()
                }),
            ("auto", None) => Ok(Command::Auto),
            ("auto", Some("on")) => Ok(Command::SetAuto(true)),
            ("auto", Some("off")) => Ok(Command::SetAuto(false)),
            ("auto", Some(_)) => Err("/auto takes on or off".to_string()),
            _ => Err(format!("Unknown command /{text} · /help lists them")),
        }
    }
}

/// What the shell asks of the model thread.
enum Request {
    /// A message from the user.
    Message(String),
    /// Compact the conversation now, as the `/compact` command asks.
    Compact,
    /// Limit a thought to this many tokens, or lift the limit, as the
    /// `/thinking-budget` command asks.
    ThinkingBudget(Option<usize>),
    /// Turn auto mode on or off, as the `/auto` command asks.
    Auto(bool),
}

/// What the model thread reports back.
enum Reply {
    /// How far along loading the model is.
    Loading(Loading),
    /// The model is ready for a message, and how long loading it took.
    Ready(Duration),
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
    let loaded = session::load(settings, log, |loading| {
        let _ = replies.send(Reply::Loading(loading));
    })?;
    let mut harness = loaded.harness;
    let _ = replies.send(Reply::Ready(loaded.loading));

    for request in requests {
        let on_event = |event: Event| {
            let _ = replies.send(Reply::Event(event));
            if stop.load(Ordering::Relaxed) {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        };
        let result = match request {
            Request::Message(message) => harness.send(&message, on_event),
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

/// What the model is doing.
enum Status {
    /// Downloading one of the model's files, since when.
    Downloading {
        since: Instant,
        progress: Progress,
    },
    /// Loading the weights, since when: how many tensors so far, out of how
    /// many.
    Loading {
        since: Instant,
        done: usize,
        total: usize,
    },
    /// Reading the system prompt, since when: how many of its tokens so
    /// far, out of how many.
    Prompting {
        since: Instant,
        read: usize,
        total: usize,
    },
    Idle,
    /// Reading the prompt, before the first token of the reply.
    Reading(Instant),
    Generating {
        start: Instant,
        tokens: usize,
    },
}

/// What part of the reply is being generated.
#[derive(Clone, Copy, PartialEq)]
enum Segment {
    /// The model's thought, before it replies.
    Thought,
    /// The reply itself.
    Text,
    /// The note the model goes on from after the conversation is compacted.
    Note,
}

struct App {
    screen: Screen,
    model: String,
    /// The kind of device the model runs on, and its name once it is open.
    device: Device,
    device_name: Option<String>,
    /// The room the model has for conversation, once it is sized.
    size: Option<Size>,
    /// Where the transcript is kept, if one is.
    transcript: Option<PathBuf>,
    cwd: PathBuf,
    input: Input,
    /// Messages sent so far, and which one is recalled into the input.
    history: Vec<String>,
    recalled: Option<usize>,
    status: Status,
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
    /// Lines waiting to be printed above the live region, and whether the
    /// last line printed there was blank.
    lines: Vec<Line>,
    blank: bool,
    requests: Sender<Request>,
    replies: Receiver<Reply>,
    stop: Arc<AtomicBool>,
}

impl App {
    fn run(&mut self) -> Result<(), Box<dyn Error>> {
        let mut dirty = true;
        loop {
            if dirty || !matches!(self.status, Status::Idle) {
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
                match self.replies.try_recv() {
                    Ok(reply) => self.reply(reply)?,
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        return Err("the model stopped unexpectedly".into());
                    }
                }
                dirty = true;
            }
        }
    }

    /// Handles a key press, returning whether to quit.
    fn key(&mut self, key: KeyEvent) -> bool {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let busy = !matches!(self.status, Status::Idle);
        match key.code {
            KeyCode::Char('c') if ctrl => {
                if matches!(self.status, Status::Reading(_) | Status::Generating { .. }) {
                    self.interrupt();
                } else if !self.input.is_empty() {
                    self.input.take();
                } else {
                    return true;
                }
            }
            KeyCode::Char('d') if ctrl && self.input.is_empty() => return true,
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
            KeyCode::Enter if !busy => self.submit(),
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

    /// Stops the reply being generated, if any.
    fn interrupt(&mut self) {
        if matches!(self.status, Status::Reading(_) | Status::Generating { .. }) {
            self.stop.store(true, Ordering::Relaxed);
            self.interrupted = true;
        }
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

        let (columns, _) = self.screen.size();
        self.lines.push(Line::new());
        for (i, line) in tui::wrap(&message, columns.saturating_sub(2))
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

        let request = match message.strip_prefix('/').map(Command::parse) {
            None => Request::Message(message),
            Some(Ok(Command::Compact)) => Request::Compact,
            Some(Ok(Command::SetThinkingBudget(tokens))) => Request::ThinkingBudget(tokens),
            Some(Ok(Command::SetAuto(auto))) => Request::Auto(auto),
            Some(Ok(command)) => {
                self.command(command);
                return;
            }
            Some(Err(what)) => {
                self.say(&notice(what));
                return;
            }
        };

        self.reply.clear();
        self.segment = Segment::Text;
        self.reply_width = columns.saturating_sub(2);
        self.committed = 0;
        self.interrupted = false;
        self.stop.store(false, Ordering::Relaxed);
        self.status = Status::Reading(Instant::now());
        let _ = self.requests.send(request);
    }

    fn reply(&mut self, reply: Reply) -> Result<(), Box<dyn Error>> {
        match reply {
            Reply::Loading(loading) => self.loading(loading),
            Reply::Ready(loading) => {
                self.loading = loading;
                self.status = Status::Idle;
            }
            Reply::Event(event) => self.event(event),
            Reply::Done(stats) => {
                self.stats = stats;
                self.reply.truncate(self.reply.trim_end().len());
                self.commit(true);
                if self.interrupted {
                    self.lines.push(vec![span("  ⎿ Interrupted").dark_grey()]);
                }
                self.status = Status::Idle;
            }
            Reply::Notice(text) => {
                self.finish();
                self.lines.push(vec![span(format!("● {text}")).dark_grey()]);
                self.lines.push(Line::new());
            }
            Reply::ThinkingBudget(tokens) => {
                self.thinking_budget = tokens;
                self.finish();
                self.lines
                    .push(vec![span(format!("● {}", budget_set(tokens))).dark_grey()]);
                self.lines.push(Line::new());
            }
            Reply::Auto(auto) => {
                self.auto = auto;
                self.finish();
                self.lines
                    .push(vec![span(format!("● {}", auto_notice(auto))).dark_grey()]);
                self.lines.push(Line::new());
            }
            Reply::Error(e) => {
                self.finish();
                self.lines
                    .push(vec![span("● ").red(), span(format!("error: {e}"))]);
                self.lines.push(Line::new());
            }
            Reply::Failed(e) => return Err(e.into()),
        }
        Ok(())
    }

    /// Shows how far along loading the model is.
    fn loading(&mut self, loading: Loading) {
        match loading {
            Loading::Downloading(progress) => {
                // Time each file on its own, so the speed is of this one.
                let since = match self.status {
                    Status::Downloading {
                        since,
                        progress: last,
                    } if last.file == progress.file => since,
                    _ => Instant::now(),
                };
                self.status = Status::Downloading { since, progress };
            }
            Loading::Device(name) => self.device_name = Some(name),
            Loading::Sized(size) => self.size = Some(size),
            Loading::Weights { done, total } => {
                let since = match self.status {
                    Status::Loading { since, .. } => since,
                    _ => Instant::now(),
                };
                self.status = Status::Loading { since, done, total };
            }
            Loading::Prompting { read, total } => {
                let since = match self.status {
                    Status::Prompting { since, .. } => since,
                    _ => Instant::now(),
                };
                self.status = Status::Prompting { since, read, total };
            }
        }
    }

    /// Shows something that happened in the turn.
    fn event(&mut self, event: Event) {
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
            Event::Call { name, detail } => {
                self.finish();
                let detail: Vec<&str> = detail.split_whitespace().collect();
                self.lines.push(vec![
                    span("● "),
                    span(&name).bold(),
                    span(format!("({})", detail.join(" "))).dark_grey(),
                ]);
                self.call = name;
                self.status = Status::Reading(Instant::now());
            }
            Event::Output(output) => {
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
                    .map(|line| reply_line(segment, line)),
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
            Command::Model => model_listing(
                &self.model,
                self.device,
                self.device_name.as_deref(),
                self.size.as_ref(),
                columns,
            ),
            Command::Stats => {
                let details = self.stats.report(self.loading, self.started.elapsed());
                report(
                    vec![span("● "), span("This session").bold()],
                    details,
                    columns,
                )
            }
            Command::Transcript => notice(match &self.transcript {
                Some(path) => format!("Transcript in {}", tilde(path)),
                None => "No transcript is kept".to_string(),
            }),
            Command::Help => help(columns),
            Command::ThinkingBudget => notice(budget_notice(self.thinking_budget)),
            Command::Auto => notice(auto_notice(self.auto)),
            Command::Compact | Command::SetThinkingBudget(_) | Command::SetAuto(_) => {
                return;
            }
        };
        self.say(&lines);
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
                    .map(|line| reply_line(segment, line)),
            );
        }
        // A blank line between the conversation and the input box, unless
        // the conversation already ends in one: the message a reply is to,
        // and a tool's output, do.
        let blank = self.lines.last().map_or(self.blank, Line::is_empty);
        if !live.is_empty() || !blank {
            live.push(Line::new());
        }
        if let Some(status) = self.status() {
            live.push(status);
        }

        // The input box, as tall as its text but no taller than the screen,
        // scrolled to show the cursor.
        let (text, (row, column)) = self.input.layout(columns.saturating_sub(6).max(1));
        let height = rows.saturating_sub(live.len() + 3).max(1);
        let first = (row + 1).saturating_sub(height);
        let border = |left: &str, right: &str| {
            vec![
                span(format!(
                    "{left}{}{right}",
                    "─".repeat(columns.saturating_sub(2))
                ))
                .dark_grey(),
            ]
        };
        live.push(border("╭", "╮"));
        let cursor = (live.len() + row - first, 4 + column);
        for (i, line) in text.into_iter().enumerate().skip(first).take(height) {
            let prompt = match (i, &self.status) {
                (0, Status::Idle) => span("› ").bold(),
                (0, _) => span("› ").dark_grey(),
                _ => span("  "),
            };
            let content = if self.input.is_empty() {
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

    /// What the model is doing, with a spinner, while it's busy. Numbers that
    /// would otherwise shift the line are padded to a fixed width, so it does
    /// not shift as they change.
    fn status(&self) -> Option<Line> {
        const SPINNER: [&str; 4] = ["◐", "◓", "◑", "◒"];
        let seconds = |since: Instant| format!("{}s", since.elapsed().as_secs());
        let percent = |done: usize, total: usize| match total {
            0 => "  0%".to_string(),
            _ => format!("{:>3}%", done * 100 / total),
        };
        let (verb, since, details) = match self.status {
            Status::Idle => return None,
            Status::Downloading { since, progress } => (
                "Downloading model…",
                since,
                download_details(progress, since.elapsed()),
            ),
            Status::Loading { since, done, total } => (
                match self.device {
                    Device::Cpu => "Loading model…",
                    Device::Gpu => "Loading model to GPU…",
                },
                since,
                format!("{} · {}", percent(done, total), seconds(since)),
            ),
            Status::Prompting { since, read, total } => (
                "Reading system prompt…",
                since,
                format!("{} · {}", percent(read, total), seconds(since)),
            ),
            Status::Reading(since) => (
                "Reading…",
                since,
                format!("{} · esc to interrupt", seconds(since)),
            ),
            Status::Generating { start, tokens } => (
                match self.segment {
                    Segment::Thought => "Thinking…",
                    Segment::Text => "Generating…",
                    Segment::Note => "Compacting…",
                },
                start,
                format!(
                    "{:>5} tok/s · {}esc to interrupt",
                    rate(tokens as f64, start.elapsed())
                        .map_or(String::new(), |rate| format!("{rate:.1}")),
                    match (self.segment, self.verbose) {
                        (Segment::Thought, false) => "ctrl+o to show · ",
                        (Segment::Thought, true) => "ctrl+o to hide · ",
                        (Segment::Text | Segment::Note, _) => "",
                    }
                ),
            ),
        };
        let frame = SPINNER[(since.elapsed().as_millis() / 150) as usize % SPINNER.len()];
        Some(vec![
            span(format!("{frame} {verb}")).bold(),
            span(format!(" {details}")).dark_grey(),
        ])
    }

    /// The line under the input box: the directory the agent works in, at
    /// the left.
    fn footer(&self) -> Line {
        vec![span(format!("  {}", tilde(&self.cwd))).dark_grey()]
    }
}

/// How fast `done` of something went in `elapsed`, per second; nothing
/// before a second has passed, or before anything was done, since it is
/// too early to tell.
fn rate(done: f64, elapsed: Duration) -> Option<f64> {
    (elapsed >= Duration::from_secs(1) && done > 0.0).then(|| done / elapsed.as_secs_f64())
}

/// How a download is going: how much of the file has arrived, out of how
/// much, at what speed, and the time left once the file's size is known.
/// The fields are padded to fixed widths, as the status line's are.
fn download_details(progress: Progress, elapsed: Duration) -> String {
    let Progress { done, total, .. } = progress;
    let mut details = format!("{:>8}", model::bytes(done));
    if let Some(total) = total {
        details.push_str(&format!(
            " / {:>8} ({:>3}%)",
            model::bytes(total),
            done * 100 / total.max(1)
        ));
    }
    let rate = rate(done as f64, elapsed);
    let speed = rate.map_or(String::new(), |rate| {
        format!("{}/s", model::bytes(rate as u64))
    });
    write!(details, " · {speed:>10}").expect("a String takes any write");
    if let Some(total) = total {
        let left = match rate {
            Some(rate) if done < total => {
                let left = ((total - done) as f64 / rate).round() as u64;
                match left {
                    0..60 => format!("{left}s"),
                    60..3600 => format!("{}m {:02}s", left / 60, left % 60),
                    _ => format!("{}h {:02}m", left / 3600, left % 3600 / 60),
                }
            }
            _ => String::new(),
        };
        write!(details, " · {left:>7} left").expect("a String takes any write");
    }
    details
}

/// A path, with the home directory as `~`.
fn tilde(path: &Path) -> String {
    match dirs::home_dir().and_then(|home| path.strip_prefix(home).ok()) {
        Some(rest) if rest.as_os_str().is_empty() => "~".to_string(),
        Some(rest) => format!("~/{}", rest.display()),
        None => path.display().to_string(),
    }
}

/// The model, for the `/model` command: its source and its weights on
/// disk, the device it runs on and how much conversation it has room for,
/// as a table of labels and values.
fn model_listing(
    name: &str,
    device: Device,
    device_name: Option<&str>,
    size: Option<&Size>,
    columns: usize,
) -> Vec<Line> {
    let header = vec![span("● "), span(name.to_string()).bold()];
    let weights = match model::locate(name) {
        Ok(dir) => {
            let (files, bytes) = model::on_disk(&dir);
            match files {
                0 => "not fetched".to_string(),
                n if n == model::FILES.len() => {
                    format!("{} in {}", model::bytes(bytes), tilde(&dir))
                }
                n => format!(
                    "{n}/{} files, {} in {}",
                    model::FILES.len(),
                    model::bytes(bytes),
                    tilde(&dir)
                ),
            }
        }
        Err(e) => e.to_string(),
    };
    let on = match (device, device_name) {
        (Device::Cpu, _) => "CPU".to_string(),
        (Device::Gpu, None) => "GPU".to_string(),
        (Device::Gpu, Some(gpu)) => format!("GPU · {gpu} via {BACKEND}"),
    };
    let mut rows = vec![
        ("source", model::REPO.to_string()),
        ("weights", weights),
        ("device", on),
    ];
    if let Some(size) = size {
        rows.push((
            "context",
            format!(
                "{} tokens in {} of memory",
                size.context,
                model::bytes(size.bytes)
            ),
        ));
    }
    report(header, table(rows), columns)
}

/// Rows of a label and a value as lines, with the values aligned.
fn table(rows: Vec<(&str, String)>) -> Vec<String> {
    let width = rows.iter().map(|(label, _)| label.len()).max().unwrap_or(0) + 2;
    rows.into_iter()
        .map(|(label, value)| format!("{label:<width$}{value}"))
        .collect()
}

/// What the `/help` command says: the commands, and the keys.
fn help(columns: usize) -> Vec<Line> {
    let width = COMMANDS
        .iter()
        .map(|(name, _)| name.len())
        .max()
        .unwrap_or(0)
        + 2;
    let mut lines: Vec<String> = COMMANDS
        .iter()
        .map(|(name, what)| format!("{name:<width$}{what}"))
        .collect();
    lines.push("ctrl+o shows or hides the thought · esc interrupts · ctrl+d quits".to_string());
    report(vec![span("● "), span("Commands").bold()], lines, columns)
}

/// A report: its header, then its details under `⎿`, dimmed and wrapped to
/// the terminal's `columns`, the way a tool's output follows its call.
fn report(header: Line, details: impl IntoIterator<Item = String>, columns: usize) -> Vec<Line> {
    let mut lines = vec![header];
    let mut first = true;
    for detail in details {
        for text in tui::wrap(&detail, columns.saturating_sub(4)) {
            let prefix = if first { "  ⎿ " } else { "    " };
            first = false;
            lines.push(vec![span(format!("{prefix}{text}")).dark_grey()]);
        }
    }
    lines
}

/// One line said in place of a reply.
fn notice(text: impl Into<String>) -> Vec<Line> {
    vec![vec![span(format!("● {}", text.into()))]]
}

/// A line of a part of the reply, the first one marked with a bullet. A
/// thought is dimmed, so the reply stands out from it, and so is the note
/// the model goes on from after a compaction.
fn reply_line(segment: Segment, (i, text): (usize, &str)) -> Line {
    match segment {
        Segment::Thought | Segment::Note => {
            let bullet = if i == 0 { span("✻ ") } else { span("  ") };
            vec![bullet.dark_grey(), span(text).dark_grey().italic()]
        }
        Segment::Text => {
            let bullet = if i == 0 { span("● ") } else { span("  ") };
            vec![bullet, span(text)]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_commands() {
        assert!(matches!(Command::parse("compact"), Ok(Command::Compact)));
        assert!(matches!(Command::parse("help "), Ok(Command::Help)));
        assert!(matches!(
            Command::parse("thinking-budget"),
            Ok(Command::ThinkingBudget)
        ));
        assert!(matches!(
            Command::parse("thinking-budget 100"),
            Ok(Command::SetThinkingBudget(Some(100)))
        ));
        assert!(matches!(
            Command::parse("thinking-budget 0"),
            Ok(Command::SetThinkingBudget(None))
        ));
        assert!(Command::parse("thinking-budget lots").is_err());
        assert!(matches!(
            Command::parse("auto on"),
            Ok(Command::SetAuto(true))
        ));
        assert!(matches!(Command::parse("auto"), Ok(Command::Auto)));
        assert!(Command::parse("auto maybe").is_err());
        assert!(Command::parse("model now").is_err());
        assert!(
            Command::parse("nothing").is_err_and(|e| e.starts_with("Unknown command /nothing"))
        );
        // Every command has its line in the help.
        for name in ["model", "stats", "transcript", "compact", "help"] {
            assert!(Command::parse(name).is_ok(), "{name}");
            assert!(
                COMMANDS
                    .iter()
                    .any(|(command, _)| command == &format!("/{name}"))
            );
        }
    }

    #[test]
    fn tells_the_rate_once_it_can() {
        assert_eq!(rate(10.0, Duration::from_millis(500)), None);
        assert_eq!(rate(0.0, Duration::from_secs(2)), None);
        assert_eq!(rate(10.0, Duration::from_secs(2)), Some(5.0));
    }
}
