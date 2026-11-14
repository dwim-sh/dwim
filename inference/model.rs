//! The one model, Qwen3.8-27B, loaded: finding its files, opening the
//! device it runs on, how much conversation it has room for there, and how
//! it samples.

use std::{
    fmt, fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::SystemTime,
};

use serde::{Deserialize, Serialize};

use crate::{
    gpu::{self, Cpu, Device as _, Gpu},
    inference::{Download, Format, Gguf, LanguageModel, Result, Sampler, Tokenizer, fetch, qwen35},
};

/// Tokens of conversation the model has room for when the device cannot
/// tell how much memory it has free, and nothing else says.
const DEFAULT_CONTEXT: usize = 16384;

/// Fewest tokens of conversation the model is given room for when it
/// fits it to the memory free: room for the system prompt, which runs to
/// a few thousand tokens with the project's `AGENTS.md`, the note a
/// compaction writes, and a few turns. At 4096, the system prompt alone
/// took four fifths of the window, and the conversation was compacted at
/// its second message, to no avail.
const MIN_CONTEXT: usize = 16384;

/// Memory left free besides what the model takes, when its context is
/// fitted to the memory free: for the device's scratch buffers, the
/// driver, and whatever else starts in the meantime.
const HEADROOM: u64 = 256 << 20;

/// Where to run the model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Device {
    Cpu,
    Gpu,
}

impl Device {
    /// The API the GPU is driven through on this system.
    pub const BACKEND: &str = gpu::BACKEND;
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Device::Cpu => "cpu",
            Device::Gpu => "gpu",
        })
    }
}

/// How many sessions the model has room for at once, the tokens of
/// conversation each has room for, and the device memory they take:
/// `model` bytes for the weights and the activations, which the sessions
/// share, and `conversation` bytes more for each session, for its recurrent
/// states and its tokens of context.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct Size {
    pub sessions: usize,
    pub context: usize,
    pub model: u64,
    pub conversation: u64,
}

/// How much of the model is in a directory: how many of its files, out
/// of how many, and their size in bytes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OnDisk {
    pub files: usize,
    pub of: usize,
    pub bytes: u64,
}

/// How far loading the model has got.
pub enum Progress {
    /// The device is open: its name.
    Device(String),
    /// How much room the model has for conversation, and the memory it
    /// takes with it.
    Sized(Size),
    /// How many of the model's tensors are loaded, out of how many.
    Weights { done: usize, total: usize },
}

/// The model on its device, with the chat format of its vocabulary.
pub struct Model {
    model: Box<dyn LanguageModel>,
    gguf: Arc<Gguf>,
    format: Format,
    device: String,
    size: Size,
}

impl Model {
    /// The model's name.
    pub const NAME: &str = qwen35::NAME;

    /// Sessions the model has room for at once, unless something else
    /// says: on a 32 GiB Mac, the memory beside the weights has room for
    /// two with a context of twenty-odd thousand tokens each, and three
    /// would each have less than the fewest a session is given.
    pub const SESSIONS: usize = 2;

    /// The repository the model's files come from, and the commit of it
    /// they are fetched at.
    pub const REPO: &str = qwen35::REPO;
    pub const REVISION: &str = qwen35::REVISION;

    /// The directory the files of the model called `name` are kept in,
    /// which is the one model's.
    pub fn dir(name: &str) -> Result<PathBuf> {
        if name != Self::NAME {
            return Err(format!("unknown model '{name}' (known: {})", Self::NAME).into());
        }
        qwen35::dir().ok_or_else(|| "cannot determine the home directory".into())
    }

    /// The file in `dir` that holds the model's weights.
    pub fn weights(dir: &Path) -> PathBuf {
        dir.join(qwen35::WEIGHTS)
    }

    /// How much of the model is in `dir`.
    pub fn on_disk(dir: &Path) -> OnDisk {
        let found = qwen35::FILES
            .iter()
            .filter_map(|file| fs::metadata(dir.join(file.name)).ok());
        OnDisk {
            files: found.clone().count(),
            of: qwen35::FILES.len(),
            bytes: found.map(|metadata| metadata.len()).sum(),
        }
    }

    /// Downloads the model's files into `dir` unless they are there,
    /// telling `on_download` how each is coming in.
    pub fn fetch(dir: &Path, on_download: impl FnMut(Download)) -> Result<()> {
        fetch::fetch(dir, on_download)
    }

    /// The size of the model on `device` with the context it gets by
    /// default, if the model is downloaded and the device opens.
    pub fn default_size(device: Device) -> Option<Size> {
        let weights = Self::weights(&Self::dir(Self::NAME).ok()?);
        if !weights.exists() {
            return None;
        }
        let gguf = Gguf::open(&weights).ok()?;
        size(&gguf, &Opened::open(device).ok()?, None, Self::SESSIONS).ok()
    }

    /// Opens `device` and loads the model in `dir` onto it, with room for
    /// `sessions` sessions of `context` tokens of conversation each, or
    /// as many tokens as fit in the memory the device has free, telling
    /// `on_progress` of each stage.
    pub fn load(
        dir: &Path,
        device: Device,
        context: Option<usize>,
        sessions: usize,
        mut on_progress: impl FnMut(Progress),
    ) -> Result<Self> {
        let gguf = Arc::new(Gguf::open(&Self::weights(dir))?);
        let device = Opened::open(device)?;
        let name = device.name();
        on_progress(Progress::Device(name.clone()));
        let format = Format::new(Tokenizer::from_gguf(&gguf)?)?;
        let size = size(&gguf, &device, context, sessions)?;
        on_progress(Progress::Sized(size));
        let model = device.load(gguf.clone(), size.context, |done, total| {
            on_progress(Progress::Weights { done, total })
        })?;
        Ok(Self {
            model,
            gguf,
            format,
            device: name,
            size,
        })
    }

    /// The chat format of the model's vocabulary, with its tokenizer.
    pub fn format(&self) -> &Format {
        &self.format
    }

    /// The name of the device the model runs on.
    pub fn device(&self) -> &str {
        &self.device
    }

    /// The model's size on its device.
    pub fn size(&self) -> Size {
        self.size
    }

    /// What makes a sampler for the model, seeded anew each time: with
    /// the settings its file recommends where it has them, at the
    /// temperature given instead of the file's if one is. The file does
    /// not carry `min_p`; 0.05 follows instructions more reliably than
    /// none.
    pub fn sampler(&self) -> impl Fn(Option<f32>) -> Sampler + use<> {
        let gguf = self.gguf.clone();
        move |temperature| {
            let number = |key: &str, default: f32| {
                gguf.f32(&format!("general.sampling.{key}"))
                    .unwrap_or(default)
            };
            Sampler::new(
                temperature.unwrap_or_else(|| number("temp", 1.0)),
                number("top_k", 20.0) as usize,
                number("top_p", 0.95),
                number("min_p", 0.05),
                seed(),
            )
        }
    }
}

impl LanguageModel for Model {
    fn save(&self, len: usize) -> Option<Vec<u8>> {
        self.model.save(len)
    }

    fn restore(&mut self, state: &[u8]) -> Result<usize> {
        self.model.restore(state)
    }

    fn forward(&mut self, tokens: &[u32], pos: usize) -> Vec<f32> {
        self.model.forward(tokens, pos)
    }

    fn draft(&mut self, token: u32, pos: usize, k: usize) -> Vec<u32> {
        self.model.draft(token, pos, k)
    }

    fn verify(&mut self, tokens: &[u32], pos: usize) -> Vec<f32> {
        self.model.verify(tokens, pos)
    }

    fn rewind(&mut self, keep: usize) {
        self.model.rewind(keep)
    }

    fn mark(&mut self) -> bool {
        self.model.mark()
    }

    fn rewind_to_mark(&mut self) -> Result<()> {
        self.model.rewind_to_mark()
    }

    fn reset(&mut self) {
        self.model.reset()
    }

    fn max_len(&self) -> usize {
        self.model.max_len()
    }

    fn add_sequence(&mut self) -> Result<usize> {
        self.model.add_sequence()
    }

    fn select_sequence(&mut self, id: usize) -> Result<()> {
        self.model.select_sequence(id)
    }
}

/// A device the model runs on, opened.
enum Opened {
    Cpu(Cpu),
    Gpu(Box<Gpu>),
}

impl Opened {
    fn open(device: Device) -> Result<Self> {
        Ok(match device {
            Device::Cpu => Opened::Cpu(Cpu),
            Device::Gpu => Opened::Gpu(Box::new(Gpu::new()?)),
        })
    }

    /// What the device is called.
    fn name(&self) -> String {
        match self {
            Opened::Cpu(device) => device.name(),
            Opened::Gpu(device) => device.name(),
        }
    }

    /// Bytes of memory free on the device, if it can tell.
    fn memory(&self) -> Option<u64> {
        match self {
            Opened::Cpu(device) => device.memory(),
            Opened::Gpu(device) => device.memory(),
        }
    }

    /// Loads the model onto the device, with room for `context` tokens of
    /// conversation, reporting how many of its tensors are loaded, out of
    /// how many, as it goes.
    fn load(
        self,
        gguf: Arc<Gguf>,
        context: usize,
        on_progress: impl FnMut(usize, usize),
    ) -> Result<Box<dyn LanguageModel>> {
        Ok(match self {
            Opened::Cpu(device) => {
                Box::new(qwen35::Model::load(gguf, device, context, on_progress)?)
            }
            Opened::Gpu(device) => {
                Box::new(qwen35::Model::load(gguf, *device, context, on_progress)?)
            }
        })
    }
}

/// The size of the model on `device`, with room for `sessions` sessions
/// of `context` tokens of conversation each, or as many as fit in the
/// memory it has free.
fn size(gguf: &Gguf, device: &Opened, context: Option<usize>, sessions: usize) -> Result<Size> {
    // Past the positions the model was trained on, its attention degrades.
    let config = qwen35::Config::load(gguf)?;
    let trained = config.context_length;
    let memory = qwen35::Memory::of(&config, gguf)?;
    let context = match context {
        Some(context) => context,
        None => fit(&memory, device, trained, sessions),
    };
    if context == 0 || context > trained {
        return Err(format!(
            "a context of {context} tokens is outside the 1 to {trained} the model was trained for"
        )
        .into());
    }
    Ok(Size {
        sessions,
        context,
        model: memory.fixed,
        conversation: memory.per_sequence + memory.per_token * context as u64,
    })
}

/// Tokens of conversation each of `sessions` sessions has room for in the
/// memory the device has free beside the model, in whole thousands, at
/// least [`MIN_CONTEXT`], and no more than the model was trained for.
fn fit(memory: &qwen35::Memory, device: &Opened, trained: usize, sessions: usize) -> usize {
    let Some(free) = device.memory() else {
        return DEFAULT_CONTEXT.min(trained);
    };
    let each = free.saturating_sub(memory.fixed + HEADROOM) / sessions.max(1) as u64;
    let room = each.saturating_sub(memory.per_sequence) / memory.per_token;
    let tokens = (room as usize).min(trained) / 1024 * 1024;
    tokens.max(MIN_CONTEXT).min(trained)
}

/// Seed for sampling, different on every run.
fn seed() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_nanos() as u64)
        .unwrap_or(0)
}
