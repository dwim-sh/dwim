//! File system operations that set Unix permission bits.

use std::{
    fs::{self, DirBuilder, File, OpenOptions},
    io,
    os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt, PermissionsExt},
    path::Path,
};

/// Opens `path` for writing, truncating it. If the file is created, its
/// mode is `0600`.
pub fn create_private(path: &Path) -> io::Result<File> {
    OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .open(path)
}

/// Creates `path` and any missing parents with mode `0700`. Directories
/// that already exist are left as they are.
pub fn create_private_dir(path: &Path) -> io::Result<()> {
    DirBuilder::new().recursive(true).mode(0o700).create(path)
}

/// Creates a symbolic link at `path` pointing to `target`, which need not
/// exist.
pub fn symlink(target: &Path, path: &Path) -> io::Result<()> {
    std::os::unix::fs::symlink(target, path)
}

/// Sets the owner execute bit on `path` (`chmod u+x`).
pub fn make_executable(path: &Path) -> io::Result<()> {
    let mode = mode_of(path)?;
    set_mode(path, mode | 0o100)
}

/// Sets or clears the read bits on `path` for owner, group, and others
/// (`chmod a+r` or `a-r`). A directory without them cannot be listed.
pub fn set_readable(path: &Path, readable: bool) -> io::Result<()> {
    let mode = mode_of(path)?;
    let mode = if readable {
        mode | 0o444
    } else {
        mode & !0o444
    };
    set_mode(path, mode)
}

/// Sets the mode of `path` to `mode`.
fn set_mode(path: &Path, mode: u32) -> io::Result<()> {
    fs::set_permissions(path, fs::Permissions::from_mode(mode))
}

/// Returns the mode of `path`, following symbolic links.
fn mode_of(path: &Path) -> io::Result<u32> {
    fs::metadata(path).map(|metadata| metadata.mode())
}
