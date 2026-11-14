//! Files, where what is asked of them is more than `std::fs` has for
//! every system.

use std::{
    fs::{File, OpenOptions},
    io,
    os::unix::fs::OpenOptionsExt,
    path::Path,
};

/// Creates the file at `path` for writing, empty, and readable by the user
/// alone if it was not there before.
pub fn create_private(path: &Path) -> io::Result<File> {
    OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .open(path)
}
