//! The user's directories, where each system keeps them.

use std::path::PathBuf;

/// The user's home directory.
pub fn home() -> Option<PathBuf> {
    dirs::home_dir()
}

/// The directory for files that can be made again (`~/.cache`, or on
/// macOS `~/Library/Caches`).
pub fn cache() -> Option<PathBuf> {
    dirs::cache_dir()
}

/// The directory for what a program keeps from one run to the next that
/// is not the user's to edit, such as logs (`~/.local/state`, or on macOS,
/// where there is none, `~/Library/Application Support`).
pub fn state() -> Option<PathBuf> {
    dirs::state_dir().or_else(dirs::data_local_dir)
}

/// The directory for sockets and the like that last as long as the user is
/// logged in (`$XDG_RUNTIME_DIR`), where there is one: macOS has none.
pub fn runtime() -> Option<PathBuf> {
    dirs::runtime_dir()
}
