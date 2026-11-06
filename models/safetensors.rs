//! Safetensors: the file format the draft ships in, mapped into memory so
//! that weights are read straight from the page cache.
//!
//! ```text
//! header length (u64) | header (JSON) | tensor data
//! ```
//!
//! The header maps each tensor's name to its type, its shape (outermost
//! first), and its byte range in the data.

use std::{
    collections::BTreeMap,
    fs::File,
    io::{self, Read},
    path::Path,
    ptr, slice,
};

use crate::{Result, Tensor};

/// Where a tensor is and what it holds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Info {
    /// Outermost first: a matrix is `[rows, cols]`.
    pub shape: Vec<usize>,
    pub dtype: String,
    /// Byte range in the data.
    pub start: usize,
    pub end: usize,
}

/// A safetensors file mapped into memory.
pub struct Safetensors {
    map: *const u8,
    len: usize,
    data_start: usize,
    tensors: BTreeMap<String, Info>,
}

// The mapping is read-only and shared freely.
unsafe impl Send for Safetensors {}
unsafe impl Sync for Safetensors {}

impl Safetensors {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let len = file.metadata()?.len() as usize;
        let mut reader = io::BufReader::new(&file);
        let mut word = [0u8; 8];
        reader.read_exact(&mut word)?;
        let header_len = u64::from_le_bytes(word) as usize;
        if 8 + header_len > len {
            return Err(format!("{} is not a safetensors file", path.display()).into());
        }
        let mut header = vec![0u8; header_len];
        reader.read_exact(&mut header)?;
        let header: serde_json::Value = serde_json::from_slice(&header)
            .map_err(|e| format!("{}: bad header: {e}", path.display()))?;
        let data_start = 8 + header_len;
        let mut tensors = BTreeMap::new();
        let entries = header
            .as_object()
            .ok_or("the safetensors header is not an object")?;
        for (name, entry) in entries {
            if name == "__metadata__" {
                continue;
            }
            let field = |key: &str| {
                entry
                    .get(key)
                    .ok_or_else(|| format!("tensor '{name}' has no {key}"))
            };
            let shape = field("shape")?
                .as_array()
                .ok_or("a shape is not an array")?
                .iter()
                .map(|v| v.as_u64().map(|v| v as usize))
                .collect::<Option<Vec<_>>>()
                .ok_or("a shape is not of integers")?;
            let offsets = field("data_offsets")?
                .as_array()
                .ok_or("offsets are not an array")?;
            let (start, end) = match offsets.as_slice() {
                [a, b] => (
                    a.as_u64().ok_or("a bad offset")? as usize,
                    b.as_u64().ok_or("a bad offset")? as usize,
                ),
                _ => return Err(format!("tensor '{name}' has bad offsets").into()),
            };
            if start > end || data_start + end > len {
                return Err(format!("tensor '{name}' runs past the end of the file").into());
            }
            tensors.insert(
                name.clone(),
                Info {
                    shape,
                    dtype: field("dtype")?
                        .as_str()
                        .ok_or("a type is not a string")?
                        .to_string(),
                    start,
                    end,
                },
            );
        }
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
            return Err(io::Error::last_os_error().into());
        }
        Ok(Self {
            map: map as *const u8,
            len,
            data_start,
            tensors,
        })
    }

    /// Names of every tensor.
    pub fn tensors(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    pub fn info(&self, name: &str) -> Result<&Info> {
        self.tensors
            .get(name)
            .ok_or_else(|| format!("missing tensor '{name}'").into())
    }

    /// The raw bytes of a tensor.
    pub fn bytes(&self, name: &str) -> Result<&[u8]> {
        let info = self.info(name)?;
        Ok(unsafe {
            slice::from_raw_parts(
                self.map.add(self.data_start + info.start),
                info.end - info.start,
            )
        })
    }

    /// The bf16 bits of a tensor of any shape.
    pub fn bf16_bits(&self, name: &str) -> Result<Vec<u16>> {
        let info = self.info(name)?;
        if info.dtype != "BF16" {
            return Err(format!("tensor '{name}' is {}, not BF16", info.dtype).into());
        }
        let count: usize = info.shape.iter().product();
        let bytes = self.bytes(name)?;
        if bytes.len() != count * 2 {
            return Err(format!("tensor '{name}' has the wrong size").into());
        }
        Ok(bytes
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect())
    }

    /// A bf16 tensor as f32 values.
    pub fn f32s(&self, name: &str) -> Result<Vec<f32>> {
        Ok(self
            .bf16_bits(name)?
            .into_iter()
            .map(dwim_gpu::bf16)
            .collect())
    }

    /// Copies out a bf16 matrix.
    pub fn bf16(&self, name: &str) -> Result<Tensor> {
        let info = self.info(name)?;
        if info.shape.len() != 2 {
            return Err(format!("tensor '{name}' is not a matrix").into());
        }
        Ok(Tensor::Bf16 {
            shape: info.shape.clone(),
            data: self.bf16_bits(name)?,
        })
    }
}

impl Drop for Safetensors {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.map as *mut libc::c_void, self.len);
        }
    }
}

/// Writes a safetensors file of bf16 tensors, for tests: each given by
/// name, shape (outermost first), and bits.
#[cfg(any(test, feature = "testing"))]
pub fn write(path: &Path, tensors: &[(String, Vec<usize>, Vec<u16>)]) -> Result<()> {
    use std::io::Write;

    let mut header = serde_json::Map::new();
    let mut offset = 0usize;
    for (name, shape, data) in tensors {
        let bytes = data.len() * 2;
        header.insert(
            name.clone(),
            serde_json::json!({
                "dtype": "BF16",
                "shape": shape,
                "data_offsets": [offset, offset + bytes],
            }),
        );
        offset += bytes;
    }
    let header = serde_json::to_vec(&serde_json::Value::Object(header))?;
    let mut file = File::create(path)?;
    file.write_all(&(header.len() as u64).to_le_bytes())?;
    file.write_all(&header)?;
    for (_, _, data) in tensors {
        for &v in data {
            file.write_all(&v.to_le_bytes())?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_back_what_it_wrote() {
        let dir = std::env::temp_dir().join(format!("dwim-safetensors-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.safetensors");
        let bits = |v: f32| (v.to_bits() >> 16) as u16;
        write(
            &path,
            &[
                (
                    "w".to_string(),
                    vec![2, 3],
                    (0..6).map(|i| bits(i as f32)).collect(),
                ),
                ("norm".to_string(), vec![4], vec![bits(1.5); 4]),
            ],
        )
        .unwrap();
        let st = Safetensors::open(&path).unwrap();
        assert_eq!(st.tensors().collect::<Vec<_>>(), ["norm", "w"]);
        assert_eq!(st.info("w").unwrap().shape, [2, 3]);
        assert_eq!(st.f32s("norm").unwrap(), [1.5; 4]);
        match st.bf16("w").unwrap() {
            Tensor::Bf16 { shape, data } => {
                assert_eq!(shape, [2, 3]);
                assert_eq!(data[5], bits(5.0));
            }
            _ => panic!("not bf16"),
        }
        assert!(st.bf16("norm").is_err());
        assert!(st.info("missing").is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
