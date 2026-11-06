//! The host's memory, as the operating system counts it: what there is,
//! and what it can give without swapping. A GPU that shares the host's
//! memory can have no more than that, and the CPU runs in it.

/// Bytes of memory the host has in all.
pub fn total() -> Option<u64> {
    #[cfg(target_vendor = "apple")]
    return sysctl(c"hw.memsize");
    #[cfg(not(target_vendor = "apple"))]
    meminfo("MemTotal")
}

/// Bytes of memory the host can give without swapping: Linux's
/// `MemAvailable`, or on a Mac the share of its memory the kernel counts
/// as available, as `memory_pressure` reports it.
pub fn available() -> Option<u64> {
    #[cfg(target_vendor = "apple")]
    {
        let percent: u32 = sysctl(c"kern.memorystatus_level")?;
        Some(total()? / 100 * u64::from(percent.min(100)))
    }
    #[cfg(not(target_vendor = "apple"))]
    meminfo("MemAvailable")
}

/// A field of `/proc/meminfo`, in bytes.
#[cfg(not(target_vendor = "apple"))]
fn meminfo(field: &str) -> Option<u64> {
    let info = std::fs::read_to_string("/proc/meminfo").ok()?;
    let line = info
        .lines()
        .find_map(|line| line.strip_prefix(field)?.strip_prefix(':'))?;
    let kb: u64 = line.trim().strip_suffix("kB")?.trim().parse().ok()?;
    Some(kb << 10)
}

#[cfg(target_vendor = "apple")]
unsafe extern "C" {
    fn sysctlbyname(
        name: *const std::ffi::c_char,
        old: *mut std::ffi::c_void,
        old_len: *mut usize,
        new: *const std::ffi::c_void,
        new_len: usize,
    ) -> std::ffi::c_int;
}

/// An integer a sysctl's value is read as: a type every bit pattern of is
/// a value, so that whatever the kernel writes into it is one.
#[cfg(target_vendor = "apple")]
trait Sysctl: Default {}
#[cfg(target_vendor = "apple")]
impl Sysctl for u32 {}
#[cfg(target_vendor = "apple")]
impl Sysctl for u64 {}

/// The value of the sysctl `name`, an integer of type `T`.
#[cfg(target_vendor = "apple")]
fn sysctl<T: Sysctl>(name: &std::ffi::CStr) -> Option<T> {
    let mut value = T::default();
    let mut len = size_of::<T>();
    // SAFETY: `value` is a plain integer of `len` bytes, and the kernel
    // writes at most `len` bytes into it, setting `len` to what it wrote;
    // `name` is NUL-terminated and outlives the call.
    let status = unsafe {
        sysctlbyname(
            name.as_ptr(),
            (&raw mut value).cast(),
            &mut len,
            std::ptr::null(),
            0,
        )
    };
    (status == 0 && len == size_of::<T>()).then_some(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn knows_the_hosts_memory() {
        let total = total().expect("the host's memory");
        let available = available().expect("the memory available");
        assert!(total >= 1 << 30, "{total}");
        assert!(available <= total, "{available} of {total}");
    }
}
