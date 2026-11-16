//! Physical memory on macOS, queried with `sysctlbyname(3)`.
//!
//! Memory is unified: the GPU allocates from the same pool as the CPU.

use std::{
    error::Error,
    ffi::{CStr, c_char, c_int, c_void},
    ptr,
};

/// Minimum physical memory to run the model. The weights alone take 16 GiB
/// of the unified memory.
const MIN: u64 = 32 << 30;

/// Returns an error if the machine has less than 32 GiB of memory.
///
/// Cheap enough to call before downloading the weights, so an undersized
/// machine fails early.
pub fn check() -> Result<(), Box<dyn Error>> {
    let memory = total().ok_or("cannot determine the memory size")?;
    if memory < MIN {
        return Err(format!(
            "the model needs a Mac with {} GiB of memory, and this one has {} GiB",
            MIN >> 30,
            memory >> 30
        )
        .into());
    }
    Ok(())
}

/// Returns total physical memory in bytes (`hw.memsize`).
pub fn total() -> Option<u64> {
    sysctl(c"hw.memsize")
}

/// Returns memory available without swapping, in bytes.
///
/// Derived from `kern.memorystatus_level`, the percentage of memory the
/// kernel considers available, as reported by `memory_pressure(1)`.
pub fn available() -> Option<u64> {
    let percent: u32 = sysctl(c"kern.memorystatus_level")?;
    Some(total()? / 100 * u64::from(percent.min(100)))
}

/// Reads the integer sysctl `name`. Returns `None` if it does not exist or
/// is not the size of `T`.
fn sysctl<T: Sysctl>(name: &CStr) -> Option<T> {
    let mut value = T::default();
    let mut len = size_of::<T>();
    // SAFETY: `value` is `len` bytes and the kernel writes at most `len`
    // bytes to it; `name` is NUL-terminated and outlives the call.
    let status = unsafe {
        sysctlbyname(
            name.as_ptr(),
            (&raw mut value).cast(),
            &mut len,
            ptr::null(),
            0,
        )
    };
    (status == 0 && len == size_of::<T>()).then_some(value)
}

/// Integer types a sysctl value can be read into.
///
/// Every bit pattern must be a valid value, since the kernel writes raw
/// bytes into it.
trait Sysctl: Default {}
impl Sysctl for u32 {}
impl Sysctl for u64 {}

unsafe extern "C" {
    fn sysctlbyname(
        name: *const c_char,
        old: *mut c_void,
        old_len: *mut usize,
        new: *const c_void,
        new_len: usize,
    ) -> c_int;
}
