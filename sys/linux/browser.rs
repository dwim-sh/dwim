//! Opening URLs in the default web browser.

use std::{
    io,
    process::{Command, Stdio},
    thread,
};

/// Opens `url` in the default web browser with `xdg-open`.
///
/// Returns once `xdg-open` is spawned; it is reaped on a background thread.
pub fn open(url: &str) -> io::Result<()> {
    let mut child = Command::new("xdg-open")
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
