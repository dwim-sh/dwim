//! Operating system interfaces.
//!
//! `darwin` (macOS) and `linux` export the same modules, and the ones for
//! the build target are re-exported here. Code common to all Unix targets
//! lives in `unix`.

#[cfg(target_vendor = "apple")]
mod darwin;
#[cfg(not(target_vendor = "apple"))]
mod linux;
mod unix;

#[cfg(target_vendor = "apple")]
pub use darwin::{browser, memory};
#[cfg(not(target_vendor = "apple"))]
pub use linux::{browser, memory};

pub use unix::{dirs, fs, mmap, net, process, terminal, user};
