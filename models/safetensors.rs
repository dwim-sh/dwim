use std::{
    collections::HashMap,
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
};

use serde::Deserialize;
use serde_json::Value;

use crate::{Result, Tensor};

/// A safetensors file: an 8-byte little-endian header length, a JSON header
/// giving each tensor's dtype, shape, and byte range, then the tensor data.
pub struct Weights {
    file: File,
    data_start: u64,
    entries: HashMap<String, Entry>,
}

#[derive(Deserialize)]
struct Entry {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

impl Weights {
    pub fn open(path: &Path) -> Result<Self> {
        let mut file = File::open(path)?;
        let mut len = [0; 8];
        file.read_exact(&mut len)?;
        let len = u64::from_le_bytes(len);
        let mut header = vec![0; len as usize];
        file.read_exact(&mut header)?;

        let header: HashMap<String, Value> = serde_json::from_slice(&header)?;
        let mut entries = HashMap::new();
        for (name, entry) in header {
            if name != "__metadata__" {
                entries.insert(name, serde_json::from_value(entry)?);
            }
        }
        Ok(Self {
            file,
            data_start: 8 + len,
            entries,
        })
    }

    pub fn contains(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    /// Names of every tensor in the file.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(String::as_str)
    }

    /// Shape of a tensor.
    pub fn shape(&self, name: &str) -> Result<&[usize]> {
        Ok(&self.entries.get(name).ok_or_else(|| format!("missing tensor '{name}'"))?.shape)
    }

    /// Reads a bf16 tensor.
    pub fn read(&mut self, name: &str) -> Result<Tensor> {
        let entry = self
            .entries
            .get(name)
            .ok_or_else(|| format!("missing tensor '{name}'"))?;
        if entry.dtype != "BF16" {
            return Err(format!("tensor '{name}' is {}, expected BF16", entry.dtype).into());
        }
        let [start, end] = entry.data_offsets;
        let mut bytes = vec![0; (end - start) as usize];
        self.file.seek(SeekFrom::Start(self.data_start + start))?;
        self.file.read_exact(&mut bytes)?;
        Ok(Tensor {
            shape: entry.shape.clone(),
            data: bytes
                .chunks_exact(2)
                .map(|b| u16::from_le_bytes([b[0], b[1]]))
                .collect(),
        })
    }
}
