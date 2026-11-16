//! Files and directories, where what is asked of them is more than
//! `std::fs` has for every system.

use std::{
    fs::{DirBuilder, File, OpenOptions},
    io,
    os::unix::fs::{DirBuilderExt, OpenOptionsExt},
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

/// Creates the directory at `path`, and any of its parents that are not
/// there, readable by the user alone.
pub fn create_private_dir(path: &Path) -> io::Result<()> {
    DirBuilder::new().recursive(true).mode(0o700).create(path)
}
