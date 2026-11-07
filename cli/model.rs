//! The one model, Qwen3.8-27B: finding its files, the device it runs on,
//! how much conversation it has room for there, the state saved after the
//! system prompt, and how it samples.

use std::{
    error::Error,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::SystemTime,
};

use dwim::gpu::{Cpu, Device as _, Gpu};
use dwim::models::{Chat, Gguf, LanguageModel, Sampler, qwen35};

use crate::opts;

/// Saved states kept, newest first: a state is a few hundred megabytes,
/// and the system prompt changes with the project's `AGENTS.md`.
const STATES_KEPT: usize = 4;

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

/// Tokens of conversation the model has room for, and the device memory
/// it takes: `model` bytes for the weights, activations, and recurrent
/// states, whatever the context, and `conversation` bytes more for the
/// tokens of context.
#[derive(Clone, Copy, Debug)]
pub struct Size {
    pub context: usize,
    pub model: u64,
    pub conversation: u64,
}

/// The size of the model on `device`, with room for `context` tokens of
/// conversation, or as many as fit in the memory it has free.
pub fn size(gguf: &Gguf, device: &Opened, context: Option<usize>) -> Result<Size, Box<dyn Error>> {
    // Past the positions the model was trained on, its attention degrades.
    let config = qwen35::Config::load(gguf)?;
    let trained = config.context_length;
    let memory = qwen35::Memory::of(&config, gguf)?;
    let context = match context {
        Some(context) => context,
        None => fit(&memory, device, trained),
    };
    if context == 0 || context > trained {
        return Err(format!(
            "a context of {context} tokens is outside the 1 to {trained} the model was trained for"
        )
        .into());
    }
    Ok(Size {
        context,
        model: memory.fixed,
        conversation: memory.per_token * context as u64,
    })
}

/// Tokens of conversation that fit in the memory `device` has free,
/// beside a model of `memory` and some headroom: a whole number of
/// thousands, at least [`MIN_CONTEXT`], and at most `trained`.
fn fit(memory: &qwen35::Memory, device: &Opened, trained: usize) -> usize {
    let Some(free) = device.memory() else {
        return DEFAULT_CONTEXT.min(trained);
    };
    let room = free.saturating_sub(memory.fixed + HEADROOM) / memory.per_token;
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
    size(&gguf, &Opened::open(device).ok()?, None).ok()
}

/// `size` as words: the memory the model takes, and the tokens of context
/// and the memory they take.
pub fn describe(size: &Size) -> String {
    format!(
        "model: {} · context: {} tokens ({})",
        bytes(size.model),
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

/// Opens the conversation with the system prompt: from the state saved by
/// an earlier run with the same prompt, if there is one and it fits, and
/// otherwise by reading the prompt, reporting progress as `Chat::system`
/// does, and saving the state for the next run. Returns whether the state
/// was restored.
pub fn start<M: LanguageModel>(
    chat: &mut Chat<M>,
    prompt: &str,
    on_progress: impl FnMut(usize, usize),
) -> Result<bool, Box<dyn Error>> {
    let path = state_path(prompt);
    if let Some(path) = &path
        && let Ok(state) = fs::read(path)
        && chat.restore(&state).is_ok()
    {
        return Ok(true);
    }
    chat.system(prompt, on_progress)?;
    if let Some(path) = path
        && let Some(state) = chat.save()
        && let Some(dir) = path.parent()
        && fs::create_dir_all(dir).is_ok()
        && fs::write(&path, state).is_ok()
    {
        prune(dir);
    }
    Ok(false)
}

/// Where the state after `prompt` is kept: named by a hash of the model's
/// name and the prompt, in the cache directory.
fn state_path(prompt: &str) -> Option<PathBuf> {
    // FNV-1a, so that the name is the same from one build to the next.
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in NAME.bytes().chain([0]).chain(prompt.bytes()) {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    Some(
        dirs::cache_dir()?
            .join("dwim")
            .join("states")
            .join(format!("{hash:016x}.bin")),
    )
}

/// Removes all but the newest saved states.
fn prune(dir: &Path) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    let mut states: Vec<_> = entries
        .flatten()
        .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "bin"))
        .filter_map(|entry| Some((entry.metadata().ok()?.modified().ok()?, entry.path())))
        .collect();
    states.sort();
    for (_, path) in states.iter().rev().skip(STATES_KEPT) {
        let _ = fs::remove_file(path);
    }
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
