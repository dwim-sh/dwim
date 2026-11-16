//! Per-user directories, following each platform's conventions.

use std::path::PathBuf;

/// Returns the user's home directory.
pub fn home() -> Option<PathBuf> {
    dirs::home_dir()
}

/// Returns the user configuration directory: `$XDG_CONFIG_HOME` or
/// `~/.config` on Linux, `~/Library/Application Support` on macOS.
pub fn config() -> Option<PathBuf> {
    dirs::config_dir()
}

/// Returns the user cache directory: `$XDG_CACHE_HOME` or `~/.cache` on
/// Linux, `~/Library/Caches` on macOS.
pub fn cache() -> Option<PathBuf> {
    dirs::cache_dir()
}

/// Returns the user state directory, for data such as logs that persists
/// across runs but is not meant to be edited: `$XDG_STATE_HOME` or
/// `~/.local/state` on Linux. macOS has no equivalent, so this falls back
/// to `~/Library/Application Support`.
pub fn state() -> Option<PathBuf> {
    dirs::state_dir().or_else(dirs::data_local_dir)
}

/// Returns `$XDG_RUNTIME_DIR`, for sockets and other files that live as
/// long as the login session. `None` if it is unset, and always on macOS.
pub fn runtime() -> Option<PathBuf> {
    dirs::runtime_dir()
}
