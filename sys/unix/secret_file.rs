//! A secret kept in a file of the user's own, in the user's configuration
//! directory: what a system without a keychain has to keep one in. The
//! file keeps other users out, and nothing the user runs.

use std::{
    fs::{self, OpenOptions},
    io::{self, Read, Write},
    os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt},
    path::{Path, PathBuf},
    process,
};

use super::{dirs, fs::create_private_dir, user};

/// Where a secret is kept.
pub enum Place {
    /// The user's keychain, which gives it only to the program that put
    /// it there.
    Keychain,
    /// A file readable by the user alone.
    File(PathBuf),
}

/// Keeps `secret` as `account` of `service`, and returns where.
pub fn store(service: &str, account: &str, secret: &str) -> io::Result<Place> {
    let path = path(service, account)?;
    write(&path, secret)?;
    Ok(Place::File(path))
}

/// The secret kept as `account` of `service`, if there is one.
pub fn load(service: &str, account: &str) -> io::Result<Option<String>> {
    read(&path(service, account)?)
}

/// Whether a secret is kept as `account` of `service`.
pub fn exists(service: &str, account: &str) -> bool {
    path(service, account)
        .is_ok_and(|path| fs::symlink_metadata(path).is_ok_and(|metadata| metadata.is_file()))
}

/// Forgets the secret kept as `account` of `service`, if there is one.
pub fn delete(service: &str, account: &str) -> io::Result<()> {
    match fs::remove_file(path(service, account)?) {
        Err(e) if e.kind() != io::ErrorKind::NotFound => Err(e),
        _ => Ok(()),
    }
}

/// The file `account` of `service` is kept in: `service/account.key` in
/// the user's configuration directory.
fn path(service: &str, account: &str) -> io::Result<PathBuf> {
    let config = dirs::config().ok_or_else(|| io::Error::other("there is no directory for it"))?;
    Ok(config.join(service).join(format!("{account}.key")))
}

/// Writes `secret` to the file at `path`, readable by the user alone, in a
/// directory that is the user's alone if it is made here. The secret is
/// written to a new file that then takes the place of the old one, so the
/// file is never half written, a file others could read is not written
/// into, and a symbolic link left at `path` is replaced, not followed.
fn write(path: &Path, secret: &str) -> io::Result<()> {
    let (Some(dir), Some(name)) = (path.parent(), path.file_name()) else {
        return Err(io::Error::other("there is no directory for it"));
    };
    create_private_dir(dir)?;
    let new = dir.join(format!(".{}.{}", name.to_string_lossy(), process::id()));
    let _ = fs::remove_file(&new);
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&new)?;
    let written = writeln!(file, "{secret}")
        .and_then(|()| file.sync_all())
        .and_then(|()| fs::rename(&new, path));
    if written.is_err() {
        let _ = fs::remove_file(&new);
    }
    written
}

/// The secret in the file at `path`, if the file is there and has one. A
/// file that is not the user's own, or is a symbolic link, is refused; one
/// that others can read is made the user's alone.
fn read(path: &Path) -> io::Result<Option<String>> {
    let opened = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path);
    let mut file = match opened {
        Ok(file) => file,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e),
    };
    let metadata = file.metadata()?;
    if !metadata.is_file() || metadata.uid() != user::id() {
        return Err(io::Error::other(format!(
            "{} is not a file of the user's own",
            path.display()
        )));
    }
    if metadata.mode() & 0o077 != 0 {
        file.set_permissions(fs::Permissions::from_mode(0o600))?;
    }
    let mut secret = String::new();
    file.read_to_string(&mut secret)?;
    let trimmed = secret.trim();
    Ok((!trimmed.is_empty()).then(|| trimmed.to_string()))
}

#[cfg(test)]
mod tests {
    use std::{env, os::unix::fs::symlink};

    use super::*;

    /// A directory for a test's files, empty, named after `name`.
    fn scratch(name: &str) -> PathBuf {
        let dir = env::temp_dir().join(format!("dwim-secret-{name}-{}", process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    fn mode(path: &Path) -> u32 {
        fs::metadata(path).unwrap().mode() & 0o777
    }

    #[test]
    fn a_secret_written_is_read_back_and_is_the_users_alone() {
        let dir = scratch("kept");
        let path = dir.join("service").join("account.key");
        assert_eq!(read(&path).unwrap(), None);
        write(&path, "first").unwrap();
        write(&path, "second").unwrap();
        assert_eq!(read(&path).unwrap().as_deref(), Some("second"));
        assert_eq!(mode(&path), 0o600);
        assert_eq!(mode(path.parent().unwrap()), 0o700);
        // Nothing is left of the file it was written to first.
        assert_eq!(fs::read_dir(path.parent().unwrap()).unwrap().count(), 1);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_file_others_could_read_is_replaced_by_one_they_cannot() {
        let dir = scratch("loose");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("account.key");
        fs::write(&path, "old").unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();
        // Reading it closes it to others, and so does writing it anew.
        assert_eq!(read(&path).unwrap().as_deref(), Some("old"));
        assert_eq!(mode(&path), 0o600);
        fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();
        write(&path, "new").unwrap();
        assert_eq!(mode(&path), 0o600);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_symbolic_link_is_not_followed() {
        let dir = scratch("link");
        fs::create_dir_all(&dir).unwrap();
        let (target, path) = (dir.join("elsewhere"), dir.join("account.key"));
        fs::write(&target, "theirs").unwrap();
        symlink(&target, &path).unwrap();
        assert!(read(&path).is_err());
        write(&path, "ours").unwrap();
        assert_eq!(fs::read_to_string(&target).unwrap(), "theirs");
        assert_eq!(read(&path).unwrap().as_deref(), Some("ours"));
        fs::remove_dir_all(dir).unwrap();
    }
}
