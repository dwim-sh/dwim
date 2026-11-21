//! A model loaded: the device it runs on, opened, how much conversation it
//! has room for there, and how it samples.

use std::{path::Path, sync::Arc, time::SystemTime};

use crate::{
    gpu::{Cpu, Device as _, Gpu},
    inference::{Gguf, LanguageModel, Result, Sampler, arch},
    model::{Device, Format, Model, Size},
};

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
pub struct Loaded {
    model: Box<dyn LanguageModel>,
    gguf: Arc<Gguf>,
    format: Arc<dyn Format>,
    device: String,
    size: Size,
}

impl Loaded {
    /// Sessions the model has room for at once, unless something else
    /// says: on a 32 GiB Mac, the memory beside the weights has room for
    /// two with a context of twenty-odd thousand tokens each, and three
    /// would each have less than the fewest a session is given.
    pub const SESSIONS: usize = 2;

    /// The size of `model` on `device` with the context it gets by
    /// default, if the model is downloaded and the device opens.
    pub fn default_size(model: Model, device: Device) -> Option<Size> {
        let weights = model.weights()?;
        let path = weights.path(&weights.dir()?);
        if !path.exists() {
            return None;
        }
        let gguf = Gguf::open(&path).ok()?;
        size(&gguf, &Opened::open(device).ok()?, None, Self::SESSIONS).ok()
    }

    /// Opens `device` and loads `model`, whose files are in `dir`, onto
    /// it, with room for `sessions` sessions of `context` tokens of
    /// conversation each, or as many tokens as fit in the memory the
    /// device has free, telling `on_progress` of each stage.
    pub fn load(
        model: Model,
        dir: &Path,
        device: Device,
        context: Option<usize>,
        sessions: usize,
        mut on_progress: impl FnMut(Progress),
    ) -> Result<Self> {
        let weights = model
            .weights()
            .ok_or_else(|| format!("{model} does not run on this machine"))?;
        let gguf = Arc::new(Gguf::open(&weights.path(dir))?);
        let device = Opened::open(device)?;
        let name = device.name();
        on_progress(Progress::Device(name.clone()));
        let format = model.format(&gguf)?;
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
    pub fn format(&self) -> &Arc<dyn Format> {
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

impl LanguageModel for Loaded {
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

    fn batch(&self) -> usize {
        self.model.batch()
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
            Device::Gpu(api) => Opened::Gpu(Box::new(Gpu::open(api)?)),
        })
    }

    /// What the device is called: a GPU's name, and the API it is driven
    /// through.
    fn name(&self) -> String {
        match self {
            Opened::Cpu(device) => device.name(),
            Opened::Gpu(device) => format!("{} via {}", device.name(), device.api()),
        }
    }

    /// Most tokens the device takes at once.
    fn batch(&self) -> usize {
        match self {
            Opened::Cpu(device) => device.batch(),
            Opened::Gpu(device) => device.batch(),
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
        match self {
            Opened::Cpu(device) => arch::load(gguf, device, context, on_progress),
            Opened::Gpu(device) => arch::load(gguf, *device, context, on_progress),
        }
    }
}

/// The size of the model on `device`, with room for `sessions` sessions
/// of `context` tokens of conversation each, or as many as fit in the
/// memory it has free.
fn size(gguf: &Gguf, device: &Opened, context: Option<usize>, sessions: usize) -> Result<Size> {
    let arch::Plan { memory, trained } = arch::plan(gguf, device.batch())?;
    let context = match context {
        Some(context) => context,
        None => memory.fit(device.memory(), trained, sessions),
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

/// Seed for sampling, different on every run.
fn seed() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_nanos() as u64)
        .unwrap_or(0)
}
