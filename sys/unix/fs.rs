//! Files and directories, where what is asked of them is more than
//! `std::fs` has for every system.

use std::{
    fs::{self, DirBuilder, File, OpenOptions},
    io,
    os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt, PermissionsExt},
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

/// Makes `path` a symbolic link to `target`, which need not be there.
pub fn symlink(target: &Path, path: &Path) -> io::Result<()> {
    std::os::unix::fs::symlink(target, path)
}

/// Makes `path` a program the user can run, leaving the rest of its
/// permissions as they are.
pub fn make_executable(path: &Path) -> io::Result<()> {
    let mode = mode_of(path)?;
    set_mode(path, mode | 0o100)
}

/// Makes `path` readable, or not, to anyone: a directory that is not is one
/// whose names cannot be listed. The rest of its permissions are left as
/// they are.
pub fn set_readable(path: &Path, readable: bool) -> io::Result<()> {
    let mode = mode_of(path)?;
    let mode = if readable {
        mode | 0o444
    } else {
        mode & !0o444
    };
    set_mode(path, mode)
}

/// Sets the permission bits of `path` to `mode`, as this system counts them.
fn set_mode(path: &Path, mode: u32) -> io::Result<()> {
    fs::set_permissions(path, fs::Permissions::from_mode(mode))
}

/// The permission bits of `path`, as this system counts them.
fn mode_of(path: &Path) -> io::Result<u32> {
    fs::metadata(path).map(|metadata| metadata.mode())
}
