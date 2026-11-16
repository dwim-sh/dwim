//! Holding off idle sleep, with an IOKit power assertion: while one is
//! held, the Mac does not go to sleep for want of a key pressed, though
//! its display may, and it sleeps all the same when its lid is closed or
//! the user asks it to. `pmset -g assertions` lists the assertion by the
//! reason it was taken with.

use std::ffi::{CStr, CString, c_char, c_void};

/// Keeps the machine from idle sleep until it is dropped.
pub struct Awake {
    /// The assertion, if the system gave one.
    assertion: Option<u32>,
}

/// Holds off idle sleep, for `reason`, until what it returns is dropped.
/// Where the system gives no assertion, the machine sleeps as it would.
pub fn hold(reason: &str) -> Awake {
    Awake {
        assertion: assert(reason),
    }
}

impl Drop for Awake {
    fn drop(&mut self) {
        if let Some(assertion) = self.assertion {
            // SAFETY: the assertion is one IOKit gave, released once.
            unsafe { IOPMAssertionRelease(assertion) };
        }
    }
}

/// Takes the assertion that prevents idle system sleep, named `reason`.
fn assert(reason: &str) -> Option<u32> {
    let kind = CfString::new(PREVENT_IDLE_SLEEP)?;
    let name = CfString::new(&CString::new(reason).ok()?)?;
    let mut assertion = 0;
    // SAFETY: both strings are live CFStrings, and IOKit writes the
    // assertion's ID into `assertion`, which outlives the call.
    let status = unsafe { IOPMAssertionCreateWithName(kind.0, LEVEL_ON, name.0, &mut assertion) };
    (status == 0).then_some(assertion)
}

/// `kIOPMAssertionTypePreventUserIdleSystemSleep`: the system does not
/// sleep for being idle, and the display may.
const PREVENT_IDLE_SLEEP: &CStr = c"PreventUserIdleSystemSleep";

/// `kIOPMAssertionLevelOn`.
const LEVEL_ON: u32 = 255;

/// `kCFStringEncodingUTF8`.
const UTF8: u32 = 0x0800_0100;

/// A CFString, released when dropped.
struct CfString(*const c_void);

impl CfString {
    /// A CFString of `text`, if one could be made.
    fn new(text: &CStr) -> Option<Self> {
        // SAFETY: `text` is NUL-terminated and is copied before the call
        // returns; a null allocator is the default one.
        let string = unsafe { CFStringCreateWithCString(std::ptr::null(), text.as_ptr(), UTF8) };
        (!string.is_null()).then_some(Self(string))
    }
}

impl Drop for CfString {
    fn drop(&mut self) {
        // SAFETY: the string is one this made, and is not null.
        unsafe { CFRelease(self.0) };
    }
}

#[link(name = "IOKit", kind = "framework")]
unsafe extern "C" {
    fn IOPMAssertionCreateWithName(
        kind: *const c_void,
        level: u32,
        name: *const c_void,
        assertion: *mut u32,
    ) -> i32;
    fn IOPMAssertionRelease(assertion: u32) -> i32;
}

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    fn CFStringCreateWithCString(
        allocator: *const c_void,
        text: *const c_char,
        encoding: u32,
    ) -> *const c_void;
    fn CFRelease(object: *const c_void);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn holds_and_lets_go() {
        drop(hold("dwim: a test"));
        // More than one may be held at a time.
        let (_a, _b) = (hold("dwim: a test"), hold("dwim: another"));
    }
}
