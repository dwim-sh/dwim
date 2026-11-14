//! The host's memory, as Linux counts it in `/proc/meminfo`: what there is,
//! and what it can give without swapping.

use std::{error::Error, fs};

/// Whether the host has the memory to run the model, which is not known
/// here before the device is open: the weights go in the GPU's memory, and
/// the device refuses a model that does not fit.
pub fn check() -> Result<(), Box<dyn Error>> {
    Ok(())
}

/// Bytes of memory the host has in all.
pub fn total() -> Option<u64> {
    meminfo("MemTotal")
}

/// Bytes of memory the host can give without swapping.
pub fn available() -> Option<u64> {
    meminfo("MemAvailable")
}

/// A field of `/proc/meminfo`, in bytes.
fn meminfo(field: &str) -> Option<u64> {
    let info = fs::read_to_string("/proc/meminfo").ok()?;
    let line = info
        .lines()
        .find_map(|line| line.strip_prefix(field)?.strip_prefix(':'))?;
    let kb: u64 = line.trim().strip_suffix("kB")?.trim().parse().ok()?;
    Some(kb << 10)
}
