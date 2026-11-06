//! Where a model's files are kept: in `dwim`'s own cache directory, where
//! it downloads them, or in Hugging Face's, where another program may have
//! fetched the same model already.

use std::{
    fs,
    path::{Path, PathBuf},
};

/// A model hosted on Hugging Face.
///
/// The revision pins a commit, so the weights can't change underneath us and
/// outputs stay reproducible.
pub struct Source {
    pub name: &'static str,
    /// The model's name as people say it.
    pub title: &'static str,
    pub repo: &'static str,
    pub revision: &'static str,
    /// The files fetched.
    pub files: &'static [&'static str],
    /// The files that hold the model, any one of which will do: GGUF files
    /// of the same weights in different packings, which carry the
    /// tokenizer and the sampling settings too, the fetched one first.
    pub weights: &'static [&'static str],
}

/// Bonsai 2 27B, in PrismML's 1.75-bit packing, which is fetched, or its
/// 2-bit one, which Splash fetches, if that is what is in the cache.
pub const BONSAI: Source = Source {
    name: "bonsai-2-27b",
    title: "Bonsai 2 27B",
    repo: "prism-ml/Ternary-Bonsai-2-27B-gguf",
    revision: "6ed5e12bf84b7a63069882c91dd9e9218647d17b",
    files: &["LICENSE", "NOTICE.txt", "Ternary-Bonsai-2-27B-PTQ1_0.gguf"],
    weights: &[
        "Ternary-Bonsai-2-27B-PTQ1_0.gguf",
        "Ternary-Bonsai-2-27B-PQ2_0.gguf",
    ],
};

/// The DFlash2 draft trained for Bonsai's target, from its makers' release:
/// the configuration and the bf16 weights.
pub const DRAFT: Source = Source {
    name: "qwen3.8-27b-dflash2",
    title: "DFlash2 draft",
    repo: "incoai/Qwen3.8-27B-DFlash2",
    revision: "015e795645c74b1a0eeef3b570031fb62e769bc5",
    files: &["config.json", "model.safetensors"],
    weights: &["model.safetensors"],
};

impl Source {
    /// URL to download one of the model's files from.
    pub fn url(&self, file: &str) -> String {
        format!(
            "https://huggingface.co/{}/resolve/{}/{}",
            self.repo, self.revision, file
        )
    }

    /// Local directory the model's files are stored in, under the platform's
    /// cache directory (`~/.cache` on Linux, `~/Library/Caches` on macOS):
    /// the files can always be fetched again.
    pub fn dir(&self) -> Option<PathBuf> {
        dirs::cache_dir().map(|dir| dir.join("dwim").join("models").join(self.name))
    }

    /// The file holding the model's weights: in `dir`, its own directory,
    /// or failing that in Hugging Face's cache, in the snapshot of the
    /// pinned revision or else the newest that has one.
    pub fn weights(&self, dir: &Path) -> Option<PathBuf> {
        if let Some(path) = self.weights_in(dir) {
            return Some(path);
        }
        let snapshots = hub()?
            .join(format!("models--{}", self.repo.replace('/', "--")))
            .join("snapshots");
        if let Some(path) = self.weights_in(&snapshots.join(self.revision)) {
            return Some(path);
        }
        let mut others: Vec<_> = fs::read_dir(&snapshots)
            .ok()?
            .flatten()
            .filter_map(|entry| Some((entry.metadata().ok()?.modified().ok()?, entry.path())))
            .collect();
        others.sort();
        others
            .iter()
            .rev()
            .find_map(|(_, snapshot)| self.weights_in(snapshot))
    }

    /// The first of the weights files that is in `dir`.
    fn weights_in(&self, dir: &Path) -> Option<PathBuf> {
        self.weights
            .iter()
            .map(|file| dir.join(file))
            .find(|path| fs::metadata(path).is_ok_and(|meta| meta.is_file()))
    }
}

/// Hugging Face's cache of models, as its `huggingface_hub` library keeps
/// it: `HF_HUB_CACHE`, or `hub` under `HF_HOME`, or under `huggingface` in
/// `XDG_CACHE_HOME` or `~/.cache`.
pub fn hub() -> Option<PathBuf> {
    let var = |name: &str| std::env::var_os(name).filter(|v| !v.is_empty());
    if let Some(cache) = var("HF_HUB_CACHE") {
        return Some(PathBuf::from(cache));
    }
    if let Some(home) = var("HF_HOME") {
        return Some(PathBuf::from(home).join("hub"));
    }
    let cache = match var("XDG_CACHE_HOME") {
        Some(cache) => PathBuf::from(cache),
        None => dirs::home_dir()?.join(".cache"),
    };
    Some(cache.join("huggingface").join("hub"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_weights_in_either_cache() {
        let dir = std::env::temp_dir().join(format!("dwim-cache-{}", std::process::id()));
        let own = dir.join("own");
        let hub = dir.join("hub");
        let model = Source {
            name: "test",
            title: "Test",
            repo: "org/model",
            revision: "pinned",
            files: &["a.gguf"],
            weights: &["a.gguf", "b.gguf"],
        };
        let snapshots = hub.join("models--org--model").join("snapshots");
        fs::create_dir_all(&own).unwrap();
        fs::create_dir_all(snapshots.join("other")).unwrap();
        // The hub's cache is read through its environment variable, which
        // other tests in this process do not touch.
        unsafe { std::env::set_var("HF_HUB_CACHE", &hub) };
        assert_eq!(super::hub().unwrap(), hub);
        assert_eq!(model.weights(&own), None);

        // Another program's snapshot of the other packing.
        let other = snapshots.join("other").join("b.gguf");
        fs::write(&other, b"b").unwrap();
        assert_eq!(model.weights(&own), Some(other.clone()));
        // The pinned revision's snapshot comes first, then the model's own
        // directory.
        fs::create_dir_all(snapshots.join("pinned")).unwrap();
        let pinned = snapshots.join("pinned").join("a.gguf");
        fs::write(&pinned, b"a").unwrap();
        assert_eq!(model.weights(&own), Some(pinned));
        fs::write(own.join("b.gguf"), b"b").unwrap();
        assert_eq!(model.weights(&own), Some(own.join("b.gguf")));
        fs::write(own.join("a.gguf"), b"a").unwrap();
        assert_eq!(model.weights(&own), Some(own.join("a.gguf")));
        unsafe { std::env::remove_var("HF_HUB_CACHE") };
        fs::remove_dir_all(&dir).unwrap();
    }
}
