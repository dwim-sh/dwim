//! What dwim asks of the operating system: a directory for each system,
//! with the same modules in each, and the ones of the system being built
//! for re-exported here. Code elsewhere uses `sys::memory` and has no need
//! to know which system it is on.

#[cfg(target_vendor = "apple")]
mod darwin;
#[cfg(not(target_vendor = "apple"))]
mod linux;

#[cfg(target_vendor = "apple")]
pub use darwin::memory;
#[cfg(not(target_vendor = "apple"))]
pub use linux::memory;
