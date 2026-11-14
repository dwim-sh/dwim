//! The user dwim runs as.

/// The ID of the user.
pub fn id() -> u32 {
    // SAFETY: getuid takes nothing and cannot fail.
    unsafe { libc::getuid() }
}
