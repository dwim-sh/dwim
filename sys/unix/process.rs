//! Processes other than this one.

use std::{io, os::unix::process::CommandExt, process::Command};

/// Has `command` run in a session of its own, which is a process group of
/// its own with the command's ID, and has no controlling terminal.
pub fn own_session(command: &mut Command) {
    // SAFETY: setsid is safe to call between fork and exec, and the
    // closure does nothing else there but read errno.
    unsafe {
        command.pre_exec(|| {
            if libc::setsid() == -1 {
                return Err(io::Error::last_os_error());
            }
            Ok(())
        });
    }
}

/// Kills every process in the process group `pid` leads, at once.
pub fn kill_group(pid: u32) {
    // SAFETY: kill only sends a signal; a negative ID names a group.
    unsafe { libc::kill(-(pid as libc::pid_t), libc::SIGKILL) };
}

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
    use std::{
        io::{BufRead, BufReader, Read},
        process::{self, Stdio},
        time::{Duration, Instant},
    };

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
    fn killing_a_group_kills_what_its_leader_started() {
        // The `sleep` holds the pipe open, so the pipe ends when it does.
        let mut command = Command::new("sh");
        command
            .args(["-c", "sleep 60 & echo started; wait"])
            .stdout(Stdio::piped());
        own_session(&mut command);
        let mut child = command.spawn().unwrap();
        let mut out = BufReader::new(child.stdout.take().unwrap());
        let mut line = String::new();
        out.read_line(&mut line).unwrap();
        let started = Instant::now();
        kill_group(child.id());
        out.read_to_string(&mut line).unwrap();
        child.wait().unwrap();
        assert!(started.elapsed() < Duration::from_secs(30));
    }

    #[test]
    fn another_users_process_is_alive() {
        // Process 1 is root's, and is always there.
        if user::id() != 0 {
            assert!(alive(1));
        }
    }
}
