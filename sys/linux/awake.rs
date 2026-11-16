//! Holding off idle sleep, with `systemd-inhibit`: while it runs, logind
//! does not suspend the machine for being idle, though it suspends all
//! the same when the lid is closed or the user asks it to. It is given
//! `cat` to run, reading a pipe this process holds the other end of, so
//! that the inhibitor goes when the pipe closes: when the guard is
//! dropped, or when this process ends without dropping it. Where there is
//! no `systemd-inhibit`, or no logind for it to ask, nothing is held and
//! the machine sleeps as it would.

use std::process::{Child, Command, Stdio};

/// Keeps the machine from idle sleep until it is dropped.
pub struct Awake {
    /// The `systemd-inhibit` that holds the inhibitor, if one started.
    inhibitor: Option<Child>,
}

/// Holds off idle sleep, for `reason`, until what it returns is dropped.
/// Where the system has no way to, the machine sleeps as it would.
pub fn hold(reason: &str) -> Awake {
    let inhibitor = Command::new("systemd-inhibit")
        .args(["--what=idle", "--who=dwim", "--mode=block"])
        .arg(format!("--why={reason}"))
        .arg("cat")
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .ok();
    Awake { inhibitor }
}

impl Drop for Awake {
    fn drop(&mut self) {
        if let Some(mut inhibitor) = self.inhibitor.take() {
            // Closing the pipe ends `cat`, and `systemd-inhibit` with it.
            drop(inhibitor.stdin.take());
            let _ = inhibitor.wait();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn holds_and_lets_go() {
        drop(hold("dwim: a test"));
        // More than one may be held at a time.
        let (_a, _b) = (hold("dwim: a test"), hold("dwim: another"));
    }
}
