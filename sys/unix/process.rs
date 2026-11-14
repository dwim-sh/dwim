//! Processes other than this one.

use std::io;

/// Whether a process with `pid` is running. Signal 0 checks without
/// sending anything, and is refused for another user's process, which is
/// running all the same.
pub fn alive(pid: u32) -> bool {
    // SAFETY: a signal of 0 sends nothing, and only asks whether the
    // process is there.
    let refused = unsafe { libc::kill(pid as libc::pid_t, 0) } != 0;
    !refused || io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

#[cfg(test)]
mod tests {
    use std::process::{self, Command};

    use super::*;
    use crate::sys::user;

    #[test]
    fn a_process_is_alive_until_it_is_reaped() {
        assert!(alive(process::id()));
        let mut child = Command::new("true").spawn().unwrap();
        let pid = child.id();
        child.wait().unwrap();
        assert!(!alive(pid));
    }

    #[test]
    fn another_users_process_is_alive() {
        // Process 1 is root's, and is always there.
        if user::id() != 0 {
            assert!(alive(1));
        }
    }
}
