//! The current user.

/// Returns the real user ID of this process.
pub fn id() -> u32 {
    // SAFETY: getuid has no preconditions and cannot fail.
    unsafe { libc::getuid() }
}
