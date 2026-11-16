//! Physical memory on Linux, read from `/proc/meminfo`.

use std::{error::Error, fs};

/// Always succeeds. The weights are loaded into GPU memory, so whether the
/// model fits is only known once the device is open, which rejects it if
/// it does not.
pub fn check() -> Result<(), Box<dyn Error>> {
    Ok(())
}

/// Returns total physical memory in bytes (`MemTotal`).
pub fn total() -> Option<u64> {
    meminfo("MemTotal")
}

/// Returns memory available without swapping, in bytes (`MemAvailable`).
pub fn available() -> Option<u64> {
    meminfo("MemAvailable")
}

/// Returns `field` of `/proc/meminfo` in bytes.
fn meminfo(field: &str) -> Option<u64> {
    let info = fs::read_to_string("/proc/meminfo").ok()?;
    let line = info
        .lines()
        .find_map(|line| line.strip_prefix(field)?.strip_prefix(':'))?;
    let kb: u64 = line.trim().strip_suffix("kB")?.trim().parse().ok()?;
    Some(kb << 10)
}
