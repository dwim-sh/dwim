//! What dwim asks of the operating system: a directory for each system,
//! with the same modules in each, and the ones of the system being built
//! for re-exported here. Code elsewhere uses `sys::memory` and has no need
//! to know which system it is on.
//!
//! Every system dwim is built for is a Unix, and what they do the same way
//! is in `unix`, written once and re-exported here whichever is being
//! built for.

#[cfg(target_vendor = "apple")]
mod darwin;
#[cfg(not(target_vendor = "apple"))]
mod linux;
mod unix;

#[cfg(target_vendor = "apple")]
pub use darwin::memory;
#[cfg(not(target_vendor = "apple"))]
pub use linux::memory;
