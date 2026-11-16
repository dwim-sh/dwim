//! Read-only memory-mapped files.

use std::{fs::File, io, ops::Deref, os::unix::io::AsRawFd, ptr, slice};

/// A read-only private mapping of the first `len` bytes of a file.
///
/// Pages are faulted in from the page cache on access. The mapping is
/// removed on drop.
pub struct Map {
    ptr: *const u8,
    len: usize,
}

// SAFETY: the mapping is read-only, so it can be sent and shared across
// threads.
unsafe impl Send for Map {}
unsafe impl Sync for Map {}

impl Map {
    /// Maps the first `len` bytes of `file`, which must be open for
    /// reading.
    ///
    /// The file must be at least `len` bytes long: touching a page past its
    /// end raises `SIGBUS`. Fails if `len` is zero.
    pub fn new(file: &File, len: usize) -> io::Result<Self> {
        // SAFETY: a null address lets the kernel choose one, so no existing
        // mapping is replaced; the descriptor is open for the call.
        let map = unsafe {
            libc::mmap(
                ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            )
        };
        if map == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        Ok(Self {
            ptr: map as *const u8,
            len,
        })
    }
}

impl Deref for Map {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        // SAFETY: the mapping is `len` readable bytes and lives until drop.
        unsafe { slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl Drop for Map {
    fn drop(&mut self) {
        // SAFETY: the mapping is owned by `self`, and no borrows of it
        // outlive `self`.
        unsafe {
            libc::munmap(self.ptr as *mut libc::c_void, self.len);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{env, fs, process};

    use super::*;

    #[test]
    fn a_map_reads_back_the_bytes_of_its_file() {
        let path = env::temp_dir().join(format!("dwim-mmap-{}", process::id()));
        let bytes: Vec<u8> = (0..10_000u32).map(|i| (i % 251) as u8).collect();
        fs::write(&path, &bytes).unwrap();
        let map = Map::new(&File::open(&path).unwrap(), bytes.len()).unwrap();
        fs::remove_file(&path).unwrap();
        assert_eq!(&map[..], &bytes[..]);
    }

    #[test]
    fn a_map_of_nothing_is_refused() {
        let path = env::temp_dir().join(format!("dwim-mmap-empty-{}", process::id()));
        fs::write(&path, b"").unwrap();
        let map = Map::new(&File::open(&path).unwrap(), 0);
        fs::remove_file(&path).unwrap();
        assert!(map.is_err());
    }
}
