//! The terminal the program runs in.

use std::{
    fs::File,
    io::{self, Stdout},
    os::fd::AsFd,
};

/// Standard output as a file of its own, which writes straight through:
/// `Stdout` sends what it is given up to its last newline at once and holds
/// the rest, so a frame written to it reaches the terminal in two parts.
pub fn unbuffered(stdout: &Stdout) -> io::Result<File> {
    Ok(File::from(stdout.as_fd().try_clone_to_owned()?))
}
