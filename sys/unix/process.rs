//! Child processes: sessions, process groups, and signals.

use std::{
    io,
    os::unix::process::{CommandExt, ExitStatusExt},
    process::{Command, ExitStatus},
};

/// Makes `command` call `setsid(2)` before exec, so the child leads a new
/// session and process group and has no controlling terminal.
pub fn own_session(command: &mut Command) {
    // SAFETY: setsid is async-signal-safe, and the closure does nothing
    // else between fork and exec but read errno.
    unsafe {
        command.pre_exec(|| {
            if libc::setsid() == -1 {
                return Err(io::Error::last_os_error());
            }
            Ok(())
        });
    }
}

/// Makes `command` run in a new process group led by the child, so signals
/// sent to this process's group, such as Ctrl-C, do not reach it.
pub fn own_group(command: &mut Command) {
    command.process_group(0);
}

/// Sends `SIGKILL` to the process group led by `pid`.
pub fn kill_group(pid: u32) {
    // SAFETY: kill only sends a signal; a negative ID names a group.
    unsafe { libc::kill(-(pid as libc::pid_t), libc::SIGKILL) };
}

/// Returns whether a process with `pid` exists, including a zombie that
/// has not been reaped.
///
/// Probes with `kill(pid, 0)`, treating `EPERM` (another user's process)
/// as alive.
pub fn alive(pid: u32) -> bool {
    // SAFETY: signal 0 sends nothing; it only checks that `pid` exists.
    let refused = unsafe { libc::kill(pid as libc::pid_t, 0) } != 0;
    !refused || io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

/// Returns the signal that terminated the process, if any.
pub fn signal(status: ExitStatus) -> Option<i32> {
    status.signal()
}

/// Returns the name of `signal`, for common termination signals.
pub fn signal_name(signal: i32) -> Option<&'static str> {
    Some(match signal {
        libc::SIGHUP => "SIGHUP",
        libc::SIGINT => "SIGINT",
        libc::SIGQUIT => "SIGQUIT",
        libc::SIGILL => "SIGILL",
        libc::SIGABRT => "SIGABRT",
        libc::SIGFPE => "SIGFPE",
        libc::SIGKILL => "SIGKILL",
        libc::SIGBUS => "SIGBUS",
        libc::SIGSEGV => "SIGSEGV",
        libc::SIGPIPE => "SIGPIPE",
        libc::SIGALRM => "SIGALRM",
        libc::SIGTERM => "SIGTERM",
        _ => return None,
    })
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
    fn a_child_in_its_own_group_leaves_this_ones() {
        let mut command = Command::new("sleep");
        command.arg("60");
        own_group(&mut command);
        let mut child = command.spawn().unwrap();
        // SAFETY: getpgid only reads what the kernel records of a process,
        // and 0 names this one.
        let (group, ours) = unsafe { (libc::getpgid(child.id() as libc::pid_t), libc::getpgid(0)) };
        assert_eq!(
            group,
            child.id() as libc::pid_t,
            "the child leads its group"
        );
        assert_ne!(group, ours, "a signal to this group does not reach it");
        child.kill().unwrap();
        child.wait().unwrap();
    }

    #[test]
    fn another_users_process_is_alive() {
        // Process 1 is root's, and is always there.
        if user::id() != 0 {
            assert!(alive(1));
        }
    }
}
