//! Opening URLs in the default web browser.

use std::{
    io,
    process::{Command, Stdio},
    thread,
};

/// Opens `url` in the default web browser with `open`.
///
/// Returns once `open` is spawned; it is reaped on a background thread.
pub fn open(url: &str) -> io::Result<()> {
    let mut child = Command::new("open")
        .arg(url)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()?;
    thread::spawn(move || {
        let _ = child.wait();
    });
    Ok(())
}
