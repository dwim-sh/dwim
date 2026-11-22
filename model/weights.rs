//! A model's weights as a file has them: where the files come from, where
//! they are kept, and how much of them is there.

use std::{
    fs,
    path::{Path, PathBuf},
};

use crate::model::hub;

/// The weights a model runs from on this machine: the files of a
/// repository at a pinned commit, so the weights can't change underneath
/// us and outputs stay reproducible. A model may be had as other files,
/// quantized another way; these are the ones dwim runs.
pub struct Weights {
    /// The repository the files come from, named `org/name`.
    pub repo: &'static str,
    /// The commit of the repository they are fetched at.
    pub revision: &'static str,
    /// The file holding the model: a GGUF file, which carries the
    /// tokenizer and the sampling settings too.
    pub file: &'static str,
    /// The file holding the model's drafter, where that is a file of its
    /// own: a small network that drafts the tokens ahead, for the model
    /// to verify several of at once.
    pub drafter: Option<&'static str>,
    /// The files fetched.
    pub files: &'static [File],
}

/// A file of a model in its repository.
pub struct File {
    pub name: &'static str,
    /// What the Hugging Face cache names its content by: the SHA-256 of a
    /// file in LFS storage, and the git blob SHA-1 of a small one.
    pub digest: &'static str,
}

/// How much of a model is in a directory: how many of its files, out of
/// how many, and their size in bytes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OnDisk {
    pub files: usize,
    pub of: usize,
    pub bytes: u64,
}

impl Weights {
    /// Where the files are kept: the repository's snapshot at the pinned
    /// revision in the Hugging Face cache, shared with every other tool
    /// that uses it. See [`hub`].
    pub fn dir(&self) -> Option<PathBuf> {
        hub::snapshot_dir(self.repo, self.revision)
    }

    /// The file in `dir` that holds the model.
    pub fn path(&self, dir: &Path) -> PathBuf {
        dir.join(self.file)
    }

    /// The file in `dir` that holds the model's drafter, if it has one of
    /// its own.
    pub fn drafter_path(&self, dir: &Path) -> Option<PathBuf> {
        self.drafter.map(|file| dir.join(file))
    }

    /// How much of the model is in `dir`.
    pub fn on_disk(&self, dir: &Path) -> OnDisk {
        let found = self
            .files
            .iter()
            .filter_map(|file| fs::metadata(dir.join(file.name)).ok());
        OnDisk {
            files: found.clone().count(),
            of: self.files.len(),
            bytes: found.map(|metadata| metadata.len()).sum(),
        }
    }
}
