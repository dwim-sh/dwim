//! The one model, Qwen3.8-27B: finding its files, the device it runs on,
//! how much conversation it has room for there, and how it samples.

use std::{
    error::Error,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::SystemTime,
};

use dwim::gpu::{Cpu, Device as _, Gpu};
use dwim::inference::{Gguf, LanguageModel, Sampler, qwen35};

pub use dwim::server::protocol::Size;

use crate::opts;

/// Tokens of conversation the model has room for when the device cannot
/// tell how much memory it has free, and the command line does not say.
const DEFAULT_CONTEXT: usize = 16384;

/// Fewest tokens of conversation the model is given room for when it
/// fits it to the memory free: room for the system prompt, which runs to
/// a few thousand tokens with the project's `AGENTS.md`, the note a
/// compaction writes, and a few turns. At 4096, the system prompt alone
/// took four fifths of the window, and the conversation was compacted at
/// its second message, to no avail.
const MIN_CONTEXT: usize = 16384;

/// Sessions the model has room for at once, unless the command line says:
/// on a 32 GiB Mac, the memory beside the weights has room for two with a
/// context of twenty-odd thousand tokens each, and three would each have
/// less than [`MIN_CONTEXT`].
pub const SESSIONS: usize = 2;

/// Memory left free besides what the model takes, when its context is
/// fitted to the memory free: for the device's scratch buffers, the
/// driver, and whatever else starts in the meantime.
const HEADROOM: u64 = 256 << 20;

/// The model's name, as the command line gives it.
pub const NAME: &str = qwen35::NAME;

/// The directory the files of the model called `name` are kept in, which
/// is the one model's.
pub fn locate(name: &str) -> Result<PathBuf, Box<dyn Error>> {
    if name != NAME {
        return Err(format!("unknown model '{name}' (known: {NAME})").into());
    }
    qwen35::dir().ok_or_else(|| "cannot determine the home directory".into())
}

/// Opens the model's weights in `dir`.
pub fn open(dir: &Path) -> Result<Arc<Gguf>, Box<dyn Error>> {
    Ok(Arc::new(Gguf::open(&dir.join(qwen35::WEIGHTS))?))
}

/// How much of the model is in `dir`: how many of its files, and their
/// size in bytes.
pub fn on_disk(dir: &Path) -> (usize, u64) {
    qwen35::FILES
        .iter()
        .filter_map(|file| fs::metadata(dir.join(file.name)).ok())
        .fold((0, 0), |(files, bytes), metadata| {
            (files + 1, bytes + metadata.len())
        })
}

/// A device the model runs on, opened.
pub enum Opened {
    Cpu(Cpu),
    Gpu(Box<Gpu>),
}

impl Opened {
    /// Opens the device the command line chose.
    pub fn open(device: opts::Device) -> Result<Self, Box<dyn Error>> {
        Ok(match device {
            opts::Device::Cpu => Opened::Cpu(Cpu),
            opts::Device::Gpu => Opened::Gpu(Box::new(Gpu::new()?)),
        })
    }

    /// What the device is called.
    pub fn name(&self) -> String {
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
    pub fn load(
        self,
        gguf: Arc<Gguf>,
        context: usize,
        on_progress: impl FnMut(usize, usize),
    ) -> Result<Box<dyn LanguageModel>, Box<dyn Error>> {
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
pub fn size(
    gguf: &Gguf,
    device: &Opened,
    context: Option<usize>,
    sessions: usize,
) -> Result<Size, Box<dyn Error>> {
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

/// The size of the model on `device` with the context it gets by
/// default, if the model is downloaded and the device opens.
pub fn default_size(device: opts::Device) -> Option<Size> {
    let dir = locate(NAME).ok()?;
    if !dir.join(qwen35::WEIGHTS).exists() {
        return None;
    }
    let gguf = open(&dir).ok()?;
    size(&gguf, &Opened::open(device).ok()?, None, SESSIONS).ok()
}

/// `size` as words: the memory the model takes, and the tokens of context
/// and the memory they take.
pub fn describe(size: &Size) -> String {
    if size.sessions == 1 {
        return format!(
            "model: {} · context: {} tokens ({})",
            bytes(size.model),
            size.context,
            bytes(size.conversation)
        );
    }
    format!(
        "model: {} · {} sessions · context: {} tokens ({}) each",
        bytes(size.model),
        size.sessions,
        size.context,
        bytes(size.conversation)
    )
}

/// A byte count for humans, in decimal units.
pub fn bytes(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["KB", "MB", "GB", "TB"];
    if bytes < 1000 {
        return format!("{bytes} B");
    }
    let mut value = bytes as f64 / 1000.0;
    let mut unit = 0;
    while value >= 1000.0 && unit < UNITS.len() - 1 {
        value /= 1000.0;
        unit += 1;
    }
    format!("{value:.1} {}", UNITS[unit])
}

/// The sampler for the model, with the settings its file recommends where
/// it has them, at `temperature` if one is given instead of the file's.
/// The file does not carry `min_p`; 0.05 follows instructions more
/// reliably than none.
pub fn sampler(gguf: &Gguf, temperature: Option<f32>) -> Sampler {
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

/// Seed for sampling, different on every run.
fn seed() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_nanos() as u64)
        .unwrap_or(0)
}
