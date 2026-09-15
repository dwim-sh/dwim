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
    path::Path,
};

use hack_gpu::{Cpu, Gpu};
use hack_harness::{self as harness, Harness};
use hack_models::{Chat, Tokenizer};

use crate::{convert, fetch, models, opts::Device};

/// Answers `prompt` with the model, running the tools it calls, and returns
/// once it replies with text alone.
pub fn once(name: &str, device: Device, prompt: &str) -> Result<(), Box<dyn Error>> {
    let (model, dir) = fetch::locate(name)?;
    let mut progress = Progress::new();
    fetch::fetch(model, &dir, |file| {
        let total = file.total.unwrap_or(file.done);
        progress.report(format!("downloading {}", file.file), megabytes(file.done), megabytes(total));
    })?;
    if model.sharded {
        convert::ensure_pack(model, &dir, |stage| {
            let (stage, done, total) = match stage {
                convert::Progress::Planning => ("planning the conversion".to_string(), 0, 0),
                convert::Progress::Downloading { shard, total, download } => (
                    format!("downloading shard {shard}/{total}"),
                    download.map_or(0, |file| megabytes(file.done)),
                    download.map_or(0, |file| megabytes(file.total.unwrap_or(file.done))),
                ),
                convert::Progress::Converting { shard, total, done, tensors } => (format!("converting shard {shard}/{total}"), done, tensors),
            };
            progress.report(stage, done, total);
        })?;
    }
    match device {
        Device::Cpu => answer(&dir, Cpu, name, "the CPU", model.tools, prompt),
        Device::Gpu => {
            let gpu = Gpu::new()?;
            // Drivers append their own name in parentheses; the GPU's is enough.
            let device = gpu.name().split(" (").next().unwrap_or(gpu.name()).to_string();
            answer(&dir, gpu, name, &device, model.tools, prompt)
        }
    }
}

/// Loads `name` onto `device` and answers `prompt` with it, declaring the
/// tools in the form the model writes calls in.
fn answer<D: hack_gpu::Device + 'static>(
    dir: &Path,
    device: D,
    name: &str,
    on: &str,
    tools: harness::ToolFormat,
    prompt: &str,
) -> Result<(), Box<dyn Error>> {
    let mut progress = Progress::new();
    let model = models::load(dir, device, |done, total| {
        progress.report(format!("loading {name} on {on}"), done, total);
    })?;
    let tokenizer = Tokenizer::load(&dir.join("tokenizer.json"))?;
    let mut chat = Chat::new(model, tokenizer, models::sampler(dir))?;
    chat.system(&harness::system_prompt(&env::current_dir()?, tools), |read, total| {
        progress.report("reading the system prompt".to_string(), read, total);
    })?;

    let mut printer = Printer::default();
    let mut harness = Harness::new(chat);
    harness.send(prompt, |event| {
        printer.print(event);
        ControlFlow::Continue(())
    })?;
    Ok(())
}

/// Bytes as whole megabytes.
fn megabytes(bytes: u64) -> usize {
    (bytes / (1 << 20)) as usize
}

/// How far along getting ready is, on standard error: a line rewritten as it
/// goes when standard error is a terminal, and one line for each stage when
/// it is not.
struct Progress {
    terminal: bool,
    stage: String,
}

impl Progress {
    fn new() -> Self {
        Self {
            terminal: io::stderr().is_terminal(),
            stage: String::new(),
        }
    }

    fn report(&mut self, stage: String, done: usize, total: usize) {
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
/// thought behind it and the tools it runs to standard error.
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
                let text = if self.thinking { text } else { text.trim_start() };
                if !text.is_empty() {
                    eprint!("{text}");
                    let _ = io::stderr().flush();
                    self.thinking = true;
                }
            }
            harness::Event::Text(text) => {
                self.end_thought();
                let text = if self.replied { text } else { text.trim_start() };
                if !text.is_empty() {
                    print!("{text}");
                    let _ = io::stdout().flush();
                    self.replied = true;
                }
            }
            harness::Event::Call { name, detail } => {
                self.end_thought();
                match name {
                    "bash" => eprintln!("$ {detail}"),
                    name => eprintln!("{name}({detail})"),
                }
            }
            harness::Event::Output(output) => eprintln!("{output}"),
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
