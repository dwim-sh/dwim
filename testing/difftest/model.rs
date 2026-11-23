//! The model on the GPU against the model on the CPU, the reference, with
//! the real weights.
//!
//! For each of a few prompts, the CPU runs the prompt and then writes a
//! continuation of it, choosing the likeliest token each time; the tokens,
//! and the logits after those that are compared, are saved in
//! `target/difftest/<key>/`, where the key is a hash of what they depend on:
//! the CPU's code, the model's, the prompts, and the model file. The CPU
//! takes seconds a token, so this happens again only when one of those
//! changes. The GPU then runs the same tokens: the prompt in pieces of
//! every size its kernels tell apart, and the continuation a token at a
//! time, as when the model writes; after every piece, its next-token
//! distribution must agree with the CPU's. Then it runs them again with a
//! detour: halfway through the prompt it marks its state, runs other
//! tokens, and rewinds to the mark, as the session does to drop a thought,
//! and must agree with the CPU's straight run just the same.

use std::{
    hash::Hasher,
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};

use dwim::{
    gpu::{Cpu, Gpu},
    inference::{Gguf, LanguageModel, arch},
    model::{Model, Tokenizer},
};

use crate::Result;

/// The prompts, by name: of a few tokens, of a few dozen, of more than a
/// batch of the model, and of more than a chunk of positions attention
/// splits the keys into.
const PROMPTS: &[(&str, &str)] = &[
    ("capital", "The capital of France is"),
    (
        "recipe",
        "To make a loaf of bread, mix flour, water, salt and yeast, then knead the dough until it \
         is smooth and leave it to rise for",
    ),
    (
        "rust",
        "A function that reverses a string takes the string, splits it into the characters it is \
         made of, puts them back in the opposite order, and returns the result. In Rust, a string \
         is a sequence of bytes in UTF-8, so the characters have to be found first: the chars \
         method gives them one at a time, rev turns the sequence around, and collect gathers them \
         into a new String. Written out in full, the function is",
    ),
    (
        "code",
        "use std::collections::HashMap;\n\n\
         /// Counts how many times each word occurs in `text`, ignoring case.\n\
         pub fn word_counts(text: &str) -> HashMap<String, usize> {\n    \
             let mut counts = HashMap::new();\n    \
             for word in text.split_whitespace() {\n        \
                 let word = word\n            \
                     .trim_matches(|c: char| !c.is_alphanumeric())\n            \
                     .to_lowercase();\n        \
                 if !word.is_empty() {\n            \
                     *counts.entry(word).or_insert(0) += 1;\n        \
                 }\n    \
             }\n    \
             counts\n\
         }\n\n\
         /// The `n` most frequent words in `text`, the most frequent first, and\n\
         /// those as frequent in alphabetical order.\n\
         pub fn most_frequent(text: &str, n: usize) -> Vec<(String, usize)> {\n    \
             let mut counts: Vec<_> = word_counts(text).into_iter().collect();\n    \
             counts.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));\n    \
             counts.truncate(n);\n    \
             counts\n\
         }\n\n\
         #[cfg(test)]\n\
         mod tests {\n    \
             use super::*;\n\n    \
             #[test]\n    \
             fn counts_words_regardless_of_case_and_punctuation() {\n        \
                 let counts = word_counts(\"The cat saw the dog. The dog ran!\");\n        \
                 assert_eq!(counts[\"the\"], 3);\n        \
                 assert_eq!(counts[\"dog\"], 2);\n        \
                 assert_eq!(counts[\"cat\"], 1);\n    \
             }\n\n    \
             #[test]\n    \
             fn orders_by_frequency_then_alphabetically() {\n        \
                 let top = most_frequent(\"b a c b a b\", 2);\n        \
                 assert_eq!(top, vec![(\"b\".to_string(), 3), (\"a\".to_string(), 2)]);\n    \
             }\n\n    \
             #[test]\n    \
             fn ",
    ),
];

/// The sizes of the pieces the GPU runs a prompt in, in turn: a token, a
/// few, one short of eight and eight and one past, one short of the tiled
/// product and the smallest it takes and one past, and a whole batch.
const PIECES: &[usize] = &[1, 2, 7, 8, 9, 31, 32, 33, 64];

/// Tokens the CPU writes after each prompt, which the GPU then runs one at
/// a time.
const CONTINUATION: usize = 32;

/// Largest KL divergence of the GPU's next-token distribution from the
/// CPU's. On an RX 5700 XT under RADV, the GPU is at most 1.1e-5 off.
const MAX_KL: f64 = 1e-3;

/// The same for Inkling, whose router picks each token's experts: a sum
/// that comes out the other way in its last bit rounds a key in the
/// caches differently, and enough of those tip which expert is the
/// eighth, after which the two runs are of slightly different networks.
/// On an M5, the GPU is at most 1.9e-2 off, with the same top token at
/// 171 of 172 checks, and a GB10 through CUDA at most 2.1e-2, at 168,
/// the other four where the CPU's first two are within four points of
/// each other; a kernel that is wrong is off by far more.
const MAX_KL_ROUTED: f64 = 5e-2;

/// What the command line asks for: the model, the default unless one is
/// named, the prompts named, or all, and how many times the GPU runs them.
struct Args {
    model: Model,
    prompts: Vec<(&'static str, &'static str)>,
    repeat: usize,
}

impl Args {
    fn parse(mut args: impl Iterator<Item = String>) -> Result<Self> {
        let mut prompts = Vec::new();
        let mut repeat = 1;
        let mut model = Model::default();
        while let Some(arg) = args.next() {
            if arg == "--model" {
                model = args.next().ok_or("--model takes a model")?.parse()?;
            } else if arg == "--repeat" {
                repeat = args
                    .next()
                    .and_then(|n| n.parse().ok())
                    .ok_or("--repeat takes a number")?;
            } else {
                let prompt = PROMPTS
                    .iter()
                    .find(|(name, _)| *name == arg)
                    .ok_or_else(|| format!("no prompt is named '{arg}'"))?;
                prompts.push(*prompt);
            }
        }
        if prompts.is_empty() {
            prompts = PROMPTS.to_vec();
        }
        Ok(Self {
            model,
            prompts,
            repeat,
        })
    }
}

/// Runs the prompts `args` name, or all, on the GPU, and says whether it
/// agreed with the CPU on all.
pub fn run(args: impl Iterator<Item = String>) -> Result<bool> {
    let args = Args::parse(args)?;
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let named = args.model;
    let weights = named
        .weights()
        .ok_or("the model does not run on this machine")?;
    let model = weights.path(&weights.dir().ok_or("no home directory")?);
    if !model.exists() {
        return Err(format!(
            "no model at {}: run `dwim --download` to fetch it",
            model.display()
        )
        .into());
    }
    let gguf = Arc::new(Gguf::open(&model)?);
    let format = named.format(&gguf)?;
    let dir = root
        .join("target/difftest")
        .join(format!("{:016x}", key(root, &model)?));

    // What the CPU made of each prompt, recorded unless it already is.
    let mut cpu = None;
    let mut expected = Vec::new();
    for &(name, prompt) in &args.prompts {
        let path = dir.join(format!("{name}.bin"));
        let run = match Run::read(&path) {
            Some(run) => run,
            None => {
                let cpu = match &mut cpu {
                    Some(cpu) => cpu,
                    None => cpu.insert(arch::load(gguf.clone(), None, Cpu, 4096, |_, _| {})?),
                };
                let start = Instant::now();
                let run = Run::record(cpu, format.tokenizer(), prompt)?;
                std::fs::create_dir_all(&dir)?;
                run.write(&path)?;
                println!(
                    "recorded {name} on the CPU: {} tokens ({:.0} s)",
                    run.tokens.len(),
                    start.elapsed().as_secs_f64()
                );
                run
            }
        };
        expected.push((name, run));
    }
    drop(cpu);

    let max_kl = match named {
        Model::InklingMini | Model::InklingSmall => MAX_KL_ROUTED,
        Model::Qwen27b | Model::QwenMax => MAX_KL,
    };
    let mut gpu = arch::load(gguf, None, Gpu::new()?, 4096, |_, _| {})?;
    let mut agreed = true;
    for round in 0..args.repeat {
        for (name, want) in &expected {
            for detour in [false, true] {
                let got = want.replay(&mut gpu, detour);
                let (mut worst, mut tops) = ((0.0, 0, 0), 0);
                for ((&(end, piece), got), want) in want.checks.iter().zip(&got).zip(&want.logits) {
                    let kl = kl(want, got);
                    if kl > worst.0 {
                        worst = (kl, end, piece);
                    }
                    tops += usize::from(argmax(got) == argmax(want));
                }
                let ok = worst.0 < max_kl;
                let (kl, end, piece) = worst;
                println!(
                    "{} {name}{}{}: KL at most {kl:.2e} (after token {end}, run {}); the same \
                     top token at {tops} of {}",
                    if ok { "ok  " } else { "FAIL" },
                    if detour { " rewound" } else { "" },
                    if args.repeat > 1 {
                        format!(" #{}", round + 1)
                    } else {
                        String::new()
                    },
                    if piece == 1 {
                        "alone".to_string()
                    } else {
                        format!("in {piece}")
                    },
                    want.checks.len(),
                );
                agreed &= ok;
            }
        }
    }
    Ok(agreed)
}

/// What the CPU made of a prompt: the prompt's tokens and its
/// continuation's, and the logits after each piece the GPU runs them in.
struct Run {
    tokens: Vec<u32>,
    /// Tokens of the prompt; the rest are the continuation.
    prompt: usize,
    /// For each piece, the position of its last token and its size.
    checks: Vec<(usize, usize)>,
    logits: Vec<Vec<f32>>,
}

impl Run {
    /// Has the CPU run `prompt` in the pieces the GPU will, and then write
    /// the continuation a token at a time.
    fn record(cpu: &mut dyn LanguageModel, tokenizer: &Tokenizer, prompt: &str) -> Result<Self> {
        let mut run = Self {
            tokens: tokenizer.encode(prompt)?,
            prompt: 0,
            checks: Vec::new(),
            logits: Vec::new(),
        };
        run.prompt = run.tokens.len();
        let (checks, logits) = run.run(cpu, CONTINUATION, false);
        run.checks = checks;
        run.logits = logits;
        Ok(run)
    }

    /// Runs the tokens through the model as [`record`](Self::record) did,
    /// and returns the logits after each piece. With `detour`, the model
    /// is marked halfway through the prompt, runs the continuation's
    /// tokens there, and is rewound to the mark before the prompt goes on.
    fn replay(&self, model: &mut dyn LanguageModel, detour: bool) -> Vec<Vec<f32>> {
        let mut run = Self {
            tokens: self.tokens.clone(),
            prompt: self.prompt,
            checks: Vec::new(),
            logits: Vec::new(),
        };
        run.run(model, 0, detour).1
    }

    /// Runs the prompt through `model` in pieces, with a detour halfway if
    /// asked, then each token after it alone, adding `write` more chosen
    /// from the logits, and returns where each piece ended, with its size,
    /// and the logits after it.
    fn run(
        &mut self,
        model: &mut dyn LanguageModel,
        write: usize,
        detour: bool,
    ) -> (Vec<(usize, usize)>, Vec<Vec<f32>>) {
        model.reset();
        let (mut checks, mut logits) = (Vec::new(), Vec::new());
        let mut pos = 0;
        let mut detour = detour;
        for &size in PIECES.iter().cycle() {
            if pos == self.prompt {
                break;
            }
            let size = size.min(self.prompt - pos);
            logits.push(model.forward(&self.tokens[pos..pos + size], pos));
            pos += size;
            checks.push((pos - 1, size));
            if detour && pos >= self.prompt / 2 {
                assert!(model.mark(), "the model marks its state");
                model.forward(&self.tokens[self.prompt..], pos);
                model
                    .rewind_to_mark()
                    .expect("the model rewinds to the mark");
                detour = false;
            }
        }
        while pos < self.tokens.len() || pos < self.prompt + write {
            if pos == self.tokens.len() {
                self.tokens.push(argmax(logits.last().unwrap()) as u32);
            }
            logits.push(model.forward(&self.tokens[pos..pos + 1], pos));
            checks.push((pos, 1));
            pos += 1;
        }
        (checks, logits)
    }

    /// Reads a run [`write`](Self::write) saved, if there is one.
    fn read(path: &Path) -> Option<Self> {
        let bytes = std::fs::read(path).ok()?;
        let mut words = bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()));
        let mut next = || words.next();
        let n = next()? as usize;
        let prompt = next()? as usize;
        let tokens = (0..n).map(|_| next()).collect::<Option<Vec<_>>>()?;
        let checks = next()? as usize;
        let vocab = next()? as usize;
        let mut run = Self {
            tokens,
            prompt,
            checks: Vec::new(),
            logits: Vec::new(),
        };
        for _ in 0..checks {
            let end = next()? as usize;
            let size = next()? as usize;
            run.checks.push((end, size));
            let logits = (0..vocab)
                .map(|_| next().map(f32::from_bits))
                .collect::<Option<Vec<_>>>()?;
            run.logits.push(logits);
        }
        Some(run)
    }

    /// Saves the run, under another name first so that a run cut short
    /// leaves nothing to read.
    fn write(&self, path: &Path) -> Result<()> {
        let mut words = vec![self.tokens.len() as u32, self.prompt as u32];
        words.extend(&self.tokens);
        words.push(self.checks.len() as u32);
        words.push(self.logits[0].len() as u32);
        for (&(end, size), logits) in self.checks.iter().zip(&self.logits) {
            words.extend([end as u32, size as u32]);
            words.extend(logits.iter().map(|l| l.to_bits()));
        }
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let partial = path.with_extension("partial");
        std::fs::write(&partial, bytes)?;
        std::fs::rename(partial, path)?;
        Ok(())
    }
}

/// A hash of what the CPU's runs depend on: the code of the CPU device and
/// of the model, the prompts and how they run, and the model file, by its
/// size and when it was written.
fn key(root: &Path, model: &Path) -> Result<u64> {
    let mut files: Vec<PathBuf> = ["cpu.rs", "mod.rs", "quant.rs"]
        .iter()
        .map(|f| root.join("gpu").join(f))
        .collect();
    sources(&root.join("inference"), &mut files)?;
    files.sort();
    let mut hash = Fnv(0xcbf29ce484222325);
    for file in files {
        let name = file.strip_prefix(root).unwrap_or(&file);
        hash.write(name.as_os_str().as_encoded_bytes());
        hash.write(&std::fs::read(&file)?);
    }
    hash.write(format!("{PROMPTS:?} {PIECES:?} {CONTINUATION}").as_bytes());
    let meta = std::fs::metadata(model)?;
    hash.write_u64(meta.len());
    hash.write_u64(
        meta.modified()?
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs(),
    );
    Ok(hash.finish())
}

/// Adds the Rust sources in `dir` to `files`, and those in the directories
/// under it.
fn sources(dir: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            sources(&path, files)?;
        } else if path.extension().is_some_and(|e| e == "rs") {
            files.push(path);
        }
    }
    Ok(())
}

/// The FNV-1a hash, which unlike the standard library's is the same from
/// one build to the next.
struct Fnv(u64);

impl Hasher for Fnv {
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ b as u64).wrapping_mul(0x100000001b3);
        }
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

/// The KL divergence of the distribution `q` gives from the one `p` gives.
fn kl(p: &[f32], q: &[f32]) -> f64 {
    let (p, q) = (softmax(p), softmax(q));
    p.iter()
        .zip(&q)
        .filter(|(p, _)| **p > 0.0)
        .map(|(p, q)| p * (p / q).ln())
        .sum()
}

fn softmax(logits: &[f32]) -> Vec<f64> {
    let max = logits.iter().fold(f32::MIN, |m, &v| m.max(v)) as f64;
    let exp: Vec<f64> = logits.iter().map(|&v| (v as f64 - max).exp()).collect();
    let sum: f64 = exp.iter().sum();
    exp.into_iter().map(|e| e / sum).collect()
}

fn argmax(logits: &[f32]) -> usize {
    (0..logits.len())
        .max_by(|&a, &b| logits[a].total_cmp(&logits[b]))
        .unwrap()
}
