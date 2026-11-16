//! Secrets kept from one run to the next, in a file of the user's own.
//! Linux has no keychain that tells one of the user's programs from
//! another: the Secret Service gives a secret to anything on the user's
//! session bus, and the kernel's keyring to any process of the user's. So
//! the file keeps other users out, and nothing the user runs.

use std::io;

pub use crate::sys::unix::secret_file::{Place, delete, exists, load, store};

/// Keeps the user's other processes from reading this one's memory: it
/// can no longer be attached to with `ptrace`, nor read through
/// `/proc/<pid>/mem` and `/proc/<pid>/environ`, and it leaves no core
/// dump. A program this one starts is not kept so.
pub fn guard_memory() -> io::Result<()> {
    // SAFETY: PR_SET_DUMPABLE takes one integer and touches no memory.
    match unsafe { libc::prctl(libc::PR_SET_DUMPABLE, 0) } {
        0 => Ok(()),
        _ => Err(io::Error::last_os_error()),
    }
}
