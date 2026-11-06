//! Answering one prompt and exiting, without the shell.
//!
//! The model's reply goes to standard output and nothing else does, so that
//! a script can read the reply on its own. Getting ready, what the model is
//! thinking, the commands it runs, and what they printed all go to standard
//! error.

use std::{
    env,
    error::Error,
    io::{self, IsTerminal, Write},
    ops::ControlFlow,
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};

use dwim_gpu::{Cpu, Device as _, Gpu};
use dwim_harness::{self as harness, Harness, Log};
use dwim_models::{Chat, Gguf, Tokenizer};

use crate::{fetch, models, opts::Device};

/// Answers `prompt` with the model, running the tools it calls, and returns
/// once it replies with text alone, keeping a transcript of it in `log`;
/// then reports where the time went, if
/// `stats` asks for it.
#[allow(clippy::too_many_arguments)]
pub fn once(
    name: &str,
    device: Device,
    context: Option<usize>,
    thinking_budget: Option<usize>,
    effort: harness::Effort,
    temperature: Option<f32>,
    shell: Option<PathBuf>,
    prompt: &str,
    stats: bool,
    log: Log,
) -> Result<(), Box<dyn Error>> {
    let started = Instant::now();
    let (model, dir) = fetch::locate(name)?;
    let mut progress = Progress::new();
    fetch::fetch(model, &dir, |file| {
        let total = file.total.unwrap_or(file.done);
        progress.report(
            format!("downloading {}", file.file),
            megabytes(file.done),
            megabytes(total),
        );
    })?;
    let gguf = model.open(&dir)?;
    let (report, loading) = match device {
        Device::Cpu => answer(
            gguf,
            Cpu,
            model,
            "cpu",
            context,
            thinking_budget,
            effort,
            temperature,
            shell,
            prompt,
            log,
        )?,
        Device::Gpu => {
            let gpu = Gpu::new()?;
            let device = gpu.name();
            answer(
                gguf,
                gpu,
                model,
                &device,
                context,
                thinking_budget,
                effort,
                temperature,
                shell,
                prompt,
                log,
            )?
        }
    };
    if stats {
        for line in report.report(loading, started.elapsed()) {
            eprintln!("{line}");
        }
    }
    Ok(())
}

/// Loads `name` onto `device`, which is called `on`, with room for
/// `context` tokens (or as many as fit), says how much memory that takes,
/// and answers `prompt` with it, keeping a transcript in `log`; returns
/// where the time went and how long the loading took.
#[allow(clippy::too_many_arguments)]
fn answer<D: dwim_gpu::Device + 'static>(
    gguf: Arc<Gguf>,
    device: D,
    which: &'static models::Model,
    on: &str,
    context: Option<usize>,
    thinking_budget: Option<usize>,
    effort: harness::Effort,
    temperature: Option<f32>,
    shell: Option<PathBuf>,
    prompt: &str,
    log: Log,
) -> Result<(harness::Stats, Duration), Box<dyn Error>> {
    let name = which.name;
    let mut progress = Progress::new();
    let tokenizer = Tokenizer::from_gguf(&gguf)?;
    let sampler = models::sampler(&gguf, temperature);
    let loading = Instant::now();
    let size = models::size(&gguf, &device, context)?;
    eprintln!("{}", models::describe(&size));
    let model = models::load(gguf, device, size.context, |done, total| {
        let on = if on == "cpu" { "the CPU" } else { on };
        progress.report(format!("loading {name} on {on}"), done, total);
    })?;
    let loading = loading.elapsed();
    let mut chat = Chat::new(model, tokenizer, sampler)?;
    let cwd = env::current_dir()?;
    let system = harness::system_prompt(&cwd, effort);
    models::start(&mut chat, which, &system, |read, total| {
        progress.report("reading the system prompt".to_string(), read, total);
    })?;

    let mut printer = Printer::default();
    if let Some(path) = log.path() {
        eprintln!("transcript in {}", path.display());
    }
    let mut harness = Harness::new(chat, &cwd, {
        let system = system.clone();
        move |chat| models::start(chat, which, &system, |_, _| {}).map(|_| ())
    })?;
    harness.set_thinking_budget(thinking_budget);
    // No one is here to answer the model, so it goes on by itself.
    harness.set_auto(true);
    if let Some(shell) = shell {
        harness.set_shell(shell);
    }
    harness.keep_log(log, name, on, &system);
    harness.send(prompt, |event| {
        printer.print(event);
        ControlFlow::Continue(())
    })?;
    Ok((harness.stats(), loading))
}

/// Bytes as whole megabytes.
pub fn megabytes(bytes: u64) -> usize {
    (bytes / (1 << 20)) as usize
}

/// How far along getting ready is, on standard error: a line rewritten as it
/// goes when standard error is a terminal, and one line for each stage when
/// it is not.
pub struct Progress {
    terminal: bool,
    stage: String,
}

impl Progress {
    pub fn new() -> Self {
        Self {
            terminal: io::stderr().is_terminal(),
            stage: String::new(),
        }
    }

    pub fn report(&mut self, stage: String, done: usize, total: usize) {
        if self.stage != stage {
            self.stage = stage;
            if !self.terminal {
                eprintln!("{}…", self.stage);
            }
        }
        if self.terminal {
            eprint!("\r\x1b[K{}… {done}/{total}", self.stage);
            if done >= total {
                eprintln!();
            }
            let _ = io::stderr().flush();
        }
    }
}

/// Prints a turn as it happens: the reply to standard output, and the
/// thought behind it, the tools it runs, and the note it writes along the
/// way to standard error.
#[derive(Default)]
struct Printer {
    /// Whether anything of the reply has been printed, so that the blank
    /// lines the model opens with are left out.
    replied: bool,
    /// Whether standard error is part way through a line of thought.
    thinking: bool,
}

impl Printer {
    fn print(&mut self, event: harness::Event) {
        match event {
            harness::Event::Thought(text) => {
                let text = if self.thinking {
                    text
                } else {
                    text.trim_start()
                };
                if !text.is_empty() {
                    eprint!("{text}");
                    let _ = io::stderr().flush();
                    self.thinking = true;
                }
            }
            harness::Event::Text(text) => {
                self.end_thought();
                let text = if self.replied {
                    text
                } else {
                    text.trim_start()
                };
                if !text.is_empty() {
                    print!("{text}");
                    let _ = io::stdout().flush();
                    self.replied = true;
                }
            }
            harness::Event::Token => {}
            harness::Event::Call { name, detail } => {
                self.end_thought();
                match name {
                    "bash" => eprintln!("$ {detail}"),
                    name => eprintln!("{name}({detail})"),
                }
            }
            harness::Event::Output(output) => eprintln!("{output}"),
            harness::Event::Cut => {
                self.end_thought();
                eprintln!("[stopped short: the context window is full]");
            }
            harness::Event::Compacting => {
                self.end_thought();
                eprintln!("[compacting the conversation]");
            }
            harness::Event::Note(text) => self.print(harness::Event::Thought(text)),
            harness::Event::Compacted { before, after } => {
                self.end_thought();
                eprintln!("[compacted from {before} to {after} tokens]");
            }
        }
    }

    /// Ends the line of thought being printed, if there is one.
    fn end_thought(&mut self) {
        if self.thinking {
            eprintln!();
            self.thinking = false;
        }
    }
}

impl Drop for Printer {
    /// Ends the reply with a newline, as a command's output should end.
    fn drop(&mut self) {
        self.end_thought();
        if self.replied {
            println!();
        }
    }
}
