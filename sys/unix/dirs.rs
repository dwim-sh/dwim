//! The user's directories, where each system keeps them.

use std::path::PathBuf;

/// The directory for sockets and the like that last as long as the user is
/// logged in (`$XDG_RUNTIME_DIR`), where there is one: macOS has none.
pub fn runtime() -> Option<PathBuf> {
    dirs::runtime_dir()
}
