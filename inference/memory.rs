//! The device memory a model takes, and the conversation that fits in what
//! is left.

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

/// Device memory a model takes, in bytes, as its architecture works it
/// out: `fixed` for its weights and the activations, which the sequences
/// share, and for each sequence, `per_sequence` for the state it keeps
/// whatever its length and `per_token` more for each token of its
/// context.
#[derive(Clone, Copy, Debug)]
pub struct Memory {
    pub fixed: u64,
    pub per_sequence: u64,
    pub per_token: u64,
}

impl Memory {
    /// Tokens of conversation each of `sessions` sessions has room for in
    /// the `free` bytes a device has beside the model, if it can tell, in
    /// whole thousands, at least [`MIN_CONTEXT`], and no more than the
    /// `trained` positions the model was trained for.
    pub fn fit(&self, free: Option<u64>, trained: usize, sessions: usize) -> usize {
        let Some(free) = free else {
            return DEFAULT_CONTEXT.min(trained);
        };
        let each = free.saturating_sub(self.fixed + HEADROOM) / sessions.max(1) as u64;
        let room = each.saturating_sub(self.per_sequence) / self.per_token;
        let tokens = (room as usize).min(trained) / 1024 * 1024;
        tokens.max(MIN_CONTEXT).min(trained)
    }
}
