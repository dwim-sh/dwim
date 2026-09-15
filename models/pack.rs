//! The pack: one file holding a converted model's tensors, some bf16 and
//! some four-bit, mapped into memory so that weights are read straight from
//! the page cache.
//!
//! ```text
//! "HACKPACK" | header length (u64 LE) | JSON header | padding | tensor data
//! ```
//!
//! The header lists every tensor's dtype, shape, offset into the data, and
//! size in bytes. Tensors are 64-byte aligned.

use std::{
    collections::BTreeMap,
    fs::{File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    os::unix::fs::FileExt,
    path::Path,
    ptr, slice,
};

use serde::{Deserialize, Serialize};

use crate::{Result, Tensor, q4};

const MAGIC: &[u8; 8] = b"HACKPACK";
const ALIGN: u64 = 64;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Dtype {
    Bf16,
    Q4,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Entry {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub offset: u64,
    pub size: u64,
}

#[derive(Serialize, Deserialize)]
struct Header {
    version: u32,
    tensors: BTreeMap<String, Entry>,
}

/// Bytes a tensor of the given dtype and shape takes.
pub fn tensor_bytes(dtype: Dtype, shape: &[usize]) -> u64 {
    let (last, rest) = shape.split_last().expect("a tensor has a shape");
    let rows: usize = rest.iter().product();
    match dtype {
        Dtype::Bf16 => (rows * last * 2) as u64,
        Dtype::Q4 => (rows * q4::row_bytes(*last)) as u64,
    }
}

/// Writes a pack: the layout is fixed up front from the tensors' shapes, and
/// their data can then be written in any order, even across runs.
pub struct Writer {
    file: File,
    tensors: BTreeMap<String, Entry>,
    data_start: u64,
}

impl Writer {
    /// Creates the pack with room for `tensors`, or opens it if it already
    /// exists with the same layout, so that writing can resume.
    pub fn create(path: &Path, tensors: &[(String, Dtype, Vec<usize>)]) -> Result<Self> {
        let mut entries = BTreeMap::new();
        let mut offset = 0;
        for (name, dtype, shape) in tensors {
            let size = tensor_bytes(*dtype, shape);
            entries.insert(
                name.clone(),
                Entry {
                    dtype: *dtype,
                    shape: shape.clone(),
                    offset,
                    size,
                },
            );
            offset += size.div_ceil(ALIGN) * ALIGN;
        }
        let header = serde_json::to_vec(&Header {
            version: 1,
            tensors: entries.clone(),
        })?;
        let data_start = (16 + header.len() as u64).div_ceil(ALIGN) * ALIGN;
        let total = data_start + offset;

        if path.exists() {
            let existing = Pack::open(path)?;
            if existing.tensors == entries {
                let file = OpenOptions::new().write(true).open(path)?;
                return Ok(Self {
                    file,
                    tensors: entries,
                    data_start,
                });
            }
        }
        let mut file = File::create(path)?;
        file.write_all(MAGIC)?;
        file.write_all(&(header.len() as u64).to_le_bytes())?;
        file.write_all(&header)?;
        file.set_len(total)?;
        Ok(Self {
            file,
            tensors: entries,
            data_start,
        })
    }

    pub fn entry(&self, name: &str) -> Option<&Entry> {
        self.tensors.get(name)
    }

    /// Writes `data` at `offset` bytes into the named tensor.
    pub fn write(&self, name: &str, offset: u64, data: &[u8]) -> Result<()> {
        let entry = self.tensors.get(name).ok_or_else(|| format!("no tensor '{name}' in the pack"))?;
        if offset + data.len() as u64 > entry.size {
            return Err(format!("writing past the end of tensor '{name}'").into());
        }
        self.file.write_all_at(data, self.data_start + entry.offset + offset)?;
        Ok(())
    }

    pub fn sync(&self) -> Result<()> {
        Ok(self.file.sync_all()?)
    }
}

/// A pack mapped into memory.
pub struct Pack {
    map: *const u8,
    len: usize,
    data_start: usize,
    tensors: BTreeMap<String, Entry>,
}

// The mapping is read-only and shared freely.
unsafe impl Send for Pack {}
unsafe impl Sync for Pack {}

impl Pack {
    pub fn open(path: &Path) -> Result<Self> {
        let mut file = File::open(path)?;
        let mut magic = [0; 8];
        file.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(format!("{} is not a pack", path.display()).into());
        }
        let mut len = [0; 8];
        file.read_exact(&mut len)?;
        let header_len = u64::from_le_bytes(len);
        let mut header = vec![0; header_len as usize];
        file.read_exact(&mut header)?;
        let header: Header = serde_json::from_slice(&header)?;
        if header.version != 1 {
            return Err(format!("pack version {} is not supported", header.version).into());
        }
        let data_start = ((16 + header_len).div_ceil(ALIGN) * ALIGN) as usize;
        let len = file.seek(SeekFrom::End(0))? as usize;

        let map = unsafe {
            libc::mmap(
                ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                std::os::unix::io::AsRawFd::as_raw_fd(&file),
                0,
            )
        };
        if map == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error().into());
        }
        Ok(Self {
            map: map as *const u8,
            len,
            data_start,
            tensors: header.tensors,
        })
    }

    pub fn entry(&self, name: &str) -> Result<&Entry> {
        self.tensors.get(name).ok_or_else(|| format!("missing tensor '{name}'").into())
    }

    /// The raw bytes of a tensor.
    pub fn bytes(&self, name: &str) -> Result<&[u8]> {
        let entry = self.entry(name)?;
        let start = self.data_start + entry.offset as usize;
        if start + entry.size as usize > self.len {
            return Err(format!("tensor '{name}' runs past the end of the pack").into());
        }
        Ok(unsafe { slice::from_raw_parts(self.map.add(start), entry.size as usize) })
    }

    /// Copies out a bf16 tensor.
    pub fn bf16(&self, name: &str) -> Result<Tensor> {
        let entry = self.entry(name)?;
        if entry.dtype != Dtype::Bf16 {
            return Err(format!("tensor '{name}' is not bf16").into());
        }
        let bytes = self.bytes(name)?;
        Ok(Tensor {
            shape: entry.shape.clone(),
            data: bytes.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect(),
        })
    }
}

impl Drop for Pack {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.map as *mut libc::c_void, self.len);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_and_reads_back() {
        let dir = std::env::temp_dir().join(format!("hack-pack-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.hack");
        let layout = vec![
            ("a".to_string(), Dtype::Bf16, vec![3, 4]),
            ("b".to_string(), Dtype::Q4, vec![2, 5, 64]),
        ];
        let writer = Writer::create(&path, &layout).unwrap();
        let a: Vec<u8> = (0..24).collect();
        writer.write("a", 0, &a).unwrap();
        let b_row = vec![7u8; q4::row_bytes(64)];
        writer.write("b", (q4::row_bytes(64) * 5) as u64, &b_row).unwrap();
        drop(writer);

        // Reopening with the same layout keeps the data.
        let writer = Writer::create(&path, &layout).unwrap();
        assert_eq!(writer.entry("b").unwrap().size, (2 * 5 * q4::row_bytes(64)) as u64);
        drop(writer);

        let pack = Pack::open(&path).unwrap();
        let t = pack.bf16("a").unwrap();
        assert_eq!(t.shape, [3, 4]);
        assert_eq!(t.data[0], u16::from_le_bytes([0, 1]));
        let b = pack.bytes("b").unwrap();
        assert_eq!(b.len(), 2 * 5 * q4::row_bytes(64));
        assert!(b[..q4::row_bytes(64) * 5].iter().all(|&v| v == 0));
        assert!(b[q4::row_bytes(64) * 5..][..q4::row_bytes(64)].iter().all(|&v| v == 7));
        assert!(pack.bytes("c").is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
