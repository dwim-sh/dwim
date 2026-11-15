//! The user's web browser.

use std::{
    io,
    process::{Command, Stdio},
    thread,
};

/// Opens `url` in the user's browser.
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
