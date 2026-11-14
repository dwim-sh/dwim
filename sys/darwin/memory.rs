//! The host's memory, as macOS counts it in its sysctls: what there is,
//! and what it can give without swapping. The GPU shares it with the CPU.

use std::{
    error::Error,
    ffi::{CStr, c_char, c_int, c_void},
    ptr,
};

/// Memory a Mac needs to run the model: its CPU and GPU share it, and the
/// weights alone take 16 GB of it.
const MIN: u64 = 32 << 30;

/// Refuses a Mac with too little memory to run the model, which is worth
/// finding out before any weights are downloaded.
pub fn check() -> Result<(), Box<dyn Error>> {
    let memory = total().ok_or("cannot determine the memory size")?;
    if memory < MIN {
        return Err(format!(
            "dwim requires {} GiB or more DRAM on a Mac, and this one has {} GiB",
            MIN >> 30,
            memory >> 30
        )
        .into());
    }
    Ok(())
}

/// Bytes of memory the host has in all.
pub fn total() -> Option<u64> {
    sysctl(c"hw.memsize")
}

/// Bytes of memory the host can give without swapping: the share of its
/// memory the kernel counts as available, as `memory_pressure` reports it.
pub fn available() -> Option<u64> {
    let percent: u32 = sysctl(c"kern.memorystatus_level")?;
    Some(total()? / 100 * u64::from(percent.min(100)))
}

/// The value of the sysctl `name`, an integer of type `T`.
fn sysctl<T: Sysctl>(name: &CStr) -> Option<T> {
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
            ptr::null(),
            0,
        )
    };
    (status == 0 && len == size_of::<T>()).then_some(value)
}

/// An integer a sysctl's value is read as: a type every bit pattern of is
/// a value, so that whatever the kernel writes into it is one.
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
