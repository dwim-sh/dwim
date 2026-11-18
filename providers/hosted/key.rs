//! The key to a host: the environment's, or else the one kept from
//! signing in, in a file of the host's in the user's configuration
//! directory, readable by the user alone.

use std::{env, fs, io, io::Write, path::PathBuf};

use crate::{model::Result, providers::hosted::Host, sys};

/// The key to `host`, if there is one: the environment's, or else the one
/// kept from signing in.
pub fn find(host: &Host) -> Option<String> {
    environment(host).or_else(|| kept(host))
}

/// The key to `host` the environment has, if it has one, which is used
/// ahead of the one kept.
pub fn environment(host: &Host) -> Option<String> {
    env::var(host.key).ok().filter(|key| !key.trim().is_empty())
}

/// The key to `host` kept from signing in, if there is one.
pub fn kept(host: &Host) -> Option<String> {
    let key = fs::read_to_string(path(host)?).ok()?;
    Some(key.trim().to_string()).filter(|key| !key.is_empty())
}

/// Keeps `key` to `host` for the runs to come, readable by the user
/// alone, and returns the file it is in. Its directory is the user's
/// alone too, so that no one else can see the key's name or replace the
/// file.
pub fn keep(host: &Host, key: &str) -> Result<PathBuf> {
    let path = path(host).ok_or("there is no directory to keep the key in")?;
    if let Some(dir) = path.parent() {
        sys::fs::create_private_dir(dir)?;
    }
    let mut file = sys::fs::create_private(&path)?;
    writeln!(file, "{key}")?;
    Ok(path)
}

/// Forgets the key to `host` kept from signing in, and returns whether
/// there was one.
pub fn forget(host: &Host) -> Result<bool> {
    let path = path(host).ok_or("there is no directory the key is kept in")?;
    match fs::remove_file(path) {
        Ok(()) => Ok(true),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(e.into()),
    }
}

/// Where the key to `host` is kept: in the user's configuration directory
/// (`~/.config`, or on macOS `~/Library/Application Support`), in a file
/// named for the host as `--provider` names it.
fn path(host: &Host) -> Option<PathBuf> {
    Some(
        sys::dirs::config()?
            .join("dwim")
            .join(format!("{}.key", host.kind)),
    )
}
