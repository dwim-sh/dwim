//! The states of system prompts read before, kept in the cache directory:
//! a conversation that opens with a prompt the model has read takes up
//! from the state saved after it, in place of reading it again.

use std::{
    fs,
    path::{Path, PathBuf},
};

use crate::{
    inference::{Chat, Conversation, LanguageModel, Result, qwen35},
    sys,
};

/// How many saved states of system prompts are kept.
const STATES_KEPT: usize = 4;

impl<M: LanguageModel> Chat<M> {
    /// Opens the conversation with the system prompt: from the state saved
    /// by an earlier reading of the same prompt, if there is one and it
    /// fits, and otherwise by reading the prompt, reporting progress as
    /// [`Conversation::system`] does, and saving the state for the next
    /// time. Returns whether the state was restored.
    pub fn start(&mut self, prompt: &str, on_progress: impl FnMut(usize, usize)) -> Result<bool> {
        let path = state_path(prompt);
        if let Some(path) = &path
            && let Ok(state) = fs::read(path)
            && self.restore(&state).is_ok()
        {
            return Ok(true);
        }
        self.system(prompt, on_progress)?;
        if let Some(path) = path
            && let Some(state) = self.save()
            && let Some(dir) = path.parent()
            && fs::create_dir_all(dir).is_ok()
            && fs::write(&path, state).is_ok()
        {
            prune(dir);
        }
        Ok(false)
    }
}

/// Where the state after `prompt` is kept: named by a hash of the model's
/// name and the prompt, in the cache directory.
fn state_path(prompt: &str) -> Option<PathBuf> {
    // FNV-1a, so that the name is the same from one build to the next.
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in qwen35::NAME.bytes().chain([0]).chain(prompt.bytes()) {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    Some(
        sys::dirs::cache()?
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
