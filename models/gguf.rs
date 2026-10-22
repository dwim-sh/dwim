//! GGUF: the file format the model ships in, mapped into memory so that
//! weights are read straight from the page cache.
//!
//! ```text
//! "GGUF" | version (u32) | tensor count (u64) | key count (u64)
//! keys: name | type (u32) | value
//! tensors: name | dimensions (u32, u64 each) | type (u32) | offset (u64)
//! padding to the alignment | tensor data
//! ```
//!
//! Strings are a u64 length and the bytes. A tensor's dimensions are
//! innermost first, so a matrix of `[rows, cols]` is written `cols, rows`.

use std::{
    collections::BTreeMap,
    fs::File,
    io::{self, Read},
    path::Path,
    ptr, slice,
};

use dwim_gpu::{bf16, quant};

use crate::{Result, Tensor, quants, ternary};

/// A metadata value.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    Array(Vec<Value>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl Value {
    /// The value as an integer, whichever width it was written in.
    pub fn as_i64(&self) -> Option<i64> {
        Some(match *self {
            Value::U8(v) => v as i64,
            Value::I8(v) => v as i64,
            Value::U16(v) => v as i64,
            Value::I16(v) => v as i64,
            Value::U32(v) => v as i64,
            Value::I32(v) => v as i64,
            Value::U64(v) => v as i64,
            Value::I64(v) => v,
            _ => return None,
        })
    }

    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            Value::F32(v) => Some(v as f64),
            Value::F64(v) => Some(v),
            _ => self.as_i64().map(|v| v as f64),
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match *self {
            Value::Bool(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
}

/// The tensor types the models use: llama.cpp's numbering.
pub const F32: u32 = 0;
pub const Q4_0: u32 = 2;
pub const Q4_1: u32 = 3;
pub const Q8_0: u32 = 8;
pub const Q6_K: u32 = 14;
pub const BF16: u32 = 30;
pub const PTQ1_0: u32 = 143;

/// Bytes of a row of `cols` weights of a tensor type.
pub fn row_bytes(dtype: u32, cols: usize) -> Result<usize> {
    let blocks = |per_block: usize, bytes: usize| {
        if cols.is_multiple_of(per_block) {
            Ok(cols / per_block * bytes)
        } else {
            Err(format!("a row of {cols} is not whole blocks of {per_block}").into())
        }
    };
    match dtype {
        F32 => Ok(cols * 4),
        BF16 => Ok(cols * 2),
        PTQ1_0 => blocks(ternary::BLOCK, ternary::BLOCK_BYTES),
        Q4_0 => blocks(quant::BLOCK, quants::Q4_0_BYTES),
        Q4_1 => blocks(quant::BLOCK, quants::Q4_1_BYTES),
        Q8_0 => blocks(quant::BLOCK, quants::Q8_0_BYTES),
        Q6_K => blocks(quants::Q6_K_BLOCK, quants::Q6_K_BYTES),
        other => Err(format!("tensor type {other} is not supported").into()),
    }
}

/// Rounds an f32 to the nearest bf16, ties to even: bf16 is the top half
/// of an f32.
fn to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    let round = 0x7fff + ((bits >> 16) & 1);
    ((bits.wrapping_add(round)) >> 16) as u16
}

/// Reconstructs a row of weights of a tensor type from its bytes.
pub fn dequantize_row(dtype: u32, row: &[u8], out: &mut [f32]) -> Result<()> {
    if row.len() != row_bytes(dtype, out.len())? {
        return Err("a row of the wrong length".into());
    }
    match dtype {
        F32 => {
            for (o, b) in out.iter_mut().zip(row.chunks_exact(4)) {
                *o = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            }
        }
        BF16 => {
            for (o, b) in out.iter_mut().zip(row.chunks_exact(2)) {
                *o = bf16(u16::from_le_bytes([b[0], b[1]]));
            }
        }
        PTQ1_0 => ternary::dequantize_row(row, out),
        Q4_0 => quants::dequantize_q4_0(row, out),
        Q4_1 => quants::dequantize_q4_1(row, out),
        Q8_0 => quants::dequantize_q8_0(row, out),
        Q6_K => quants::dequantize_q6_k(row, out),
        _ => unreachable!("sized above"),
    }
    Ok(())
}

/// Where a tensor is and what it holds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Info {
    /// Innermost first: `dims[0]` is the length of a row.
    pub dims: Vec<usize>,
    pub dtype: u32,
    pub offset: u64,
}

impl Info {
    /// The shape with the rows first, as [`Tensor`] has it.
    pub fn shape(&self) -> Vec<usize> {
        self.dims.iter().rev().copied().collect()
    }

    /// Bytes the tensor takes.
    pub fn size(&self) -> Result<usize> {
        let (cols, rest) = self.dims.split_first().ok_or("a tensor has dimensions")?;
        let rows: usize = rest.iter().product();
        Ok(rows * row_bytes(self.dtype, *cols)?)
    }
}

/// A GGUF file mapped into memory.
pub struct Gguf {
    map: *const u8,
    len: usize,
    data_start: usize,
    meta: BTreeMap<String, Value>,
    tensors: BTreeMap<String, Info>,
}

// The mapping is read-only and shared freely.
unsafe impl Send for Gguf {}
unsafe impl Sync for Gguf {}

struct Reader<R> {
    inner: R,
    read: usize,
}

impl<R: Read> Reader<R> {
    fn bytes(&mut self, n: usize) -> Result<Vec<u8>> {
        let mut buf = vec![0; n];
        self.inner.read_exact(&mut buf)?;
        self.read += n;
        Ok(buf)
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.bytes(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.bytes(8)?.try_into().unwrap()))
    }

    fn string(&mut self) -> Result<String> {
        let len = self.u64()? as usize;
        Ok(String::from_utf8(self.bytes(len)?)?)
    }

    fn value(&mut self, kind: u32) -> Result<Value> {
        Ok(match kind {
            0 => Value::U8(self.bytes(1)?[0]),
            1 => Value::I8(self.bytes(1)?[0] as i8),
            2 => Value::U16(u16::from_le_bytes(self.bytes(2)?.try_into().unwrap())),
            3 => Value::I16(i16::from_le_bytes(self.bytes(2)?.try_into().unwrap())),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(self.u32()? as i32),
            6 => Value::F32(f32::from_bits(self.u32()?)),
            7 => Value::Bool(self.bytes(1)?[0] != 0),
            8 => Value::Str(self.string()?),
            9 => {
                let kind = self.u32()?;
                let len = self.u64()? as usize;
                let mut values = Vec::with_capacity(len.min(1 << 20));
                for _ in 0..len {
                    values.push(self.value(kind)?);
                }
                Value::Array(values)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(self.u64()? as i64),
            12 => Value::F64(f64::from_bits(self.u64()?)),
            other => return Err(format!("unknown metadata type {other}").into()),
        })
    }
}

impl Gguf {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let len = file.metadata()?.len() as usize;
        let mut r = Reader {
            inner: io::BufReader::new(&file),
            read: 0,
        };
        if r.bytes(4)? != b"GGUF" {
            return Err(format!("{} is not a GGUF file", path.display()).into());
        }
        let version = r.u32()?;
        if version != 3 {
            return Err(format!("GGUF version {version} is not supported").into());
        }
        let n_tensors = r.u64()?;
        let n_meta = r.u64()?;
        let mut meta = BTreeMap::new();
        for _ in 0..n_meta {
            let key = r.string()?;
            let kind = r.u32()?;
            meta.insert(key, r.value(kind)?);
        }
        let mut tensors = BTreeMap::new();
        for _ in 0..n_tensors {
            let name = r.string()?;
            let n_dims = r.u32()? as usize;
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(r.u64()? as usize);
            }
            let dtype = r.u32()?;
            let offset = r.u64()?;
            tensors.insert(
                name,
                Info {
                    dims,
                    dtype,
                    offset,
                },
            );
        }
        let alignment = meta
            .get("general.alignment")
            .and_then(Value::as_i64)
            .unwrap_or(32) as usize;
        let data_start = r.read.div_ceil(alignment) * alignment;

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
            meta,
            tensors,
        })
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.meta.get(key)
    }

    /// The metadata's keys.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.meta.keys().map(String::as_str)
    }

    fn need(&self, key: &str) -> Result<&Value> {
        self.meta
            .get(key)
            .ok_or_else(|| format!("missing metadata '{key}'").into())
    }

    pub fn u32(&self, key: &str) -> Result<u32> {
        let v = self
            .need(key)?
            .as_i64()
            .ok_or_else(|| format!("metadata '{key}' is not an integer"))?;
        Ok(u32::try_from(v)?)
    }

    pub fn f32(&self, key: &str) -> Result<f32> {
        Ok(self
            .need(key)?
            .as_f64()
            .ok_or_else(|| format!("metadata '{key}' is not a number"))? as f32)
    }

    pub fn str(&self, key: &str) -> Result<&str> {
        self.need(key)?
            .as_str()
            .ok_or_else(|| format!("metadata '{key}' is not a string").into())
    }

    pub fn array(&self, key: &str) -> Result<&[Value]> {
        match self.need(key)? {
            Value::Array(values) => Ok(values),
            _ => Err(format!("metadata '{key}' is not an array").into()),
        }
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
        let start = self.data_start + info.offset as usize;
        let size = info.size()?;
        if start + size > self.len {
            return Err(format!("tensor '{name}' runs past the end of the file").into());
        }
        Ok(unsafe { slice::from_raw_parts(self.map.add(start), size) })
    }

    /// Copies out an f32 tensor.
    pub fn f32s(&self, name: &str) -> Result<Vec<f32>> {
        if self.info(name)?.dtype != F32 {
            return Err(format!("tensor '{name}' is not f32").into());
        }
        Ok(self
            .bytes(name)?
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect())
    }

    /// Copies out a bf16 matrix.
    pub fn bf16(&self, name: &str) -> Result<Tensor> {
        let info = self.info(name)?;
        if info.dtype != BF16 || info.dims.len() != 2 {
            return Err(format!("tensor '{name}' is not a bf16 matrix").into());
        }
        Ok(Tensor::Bf16 {
            shape: info.shape(),
            data: self
                .bytes(name)?
                .chunks_exact(2)
                .map(|b| u16::from_le_bytes([b[0], b[1]]))
                .collect(),
        })
    }

    /// Copies out a ternary matrix.
    pub fn ternary(&self, name: &str) -> Result<Tensor> {
        let info = self.info(name)?;
        if info.dtype != PTQ1_0 || info.dims.len() != 2 {
            return Err(format!("tensor '{name}' is not a ternary matrix").into());
        }
        Ok(Tensor::Ternary {
            shape: info.shape(),
            data: self.bytes(name)?.to_vec(),
        })
    }

    /// Copies out a matrix in whichever format a device takes it in: bf16,
    /// ternary, `Q4_0` and `Q8_0` as they are, f32 rounded to bf16, and
    /// `Q4_1` and `Q6_K` quantized again as `Q8_0`.
    pub fn matrix(&self, name: &str) -> Result<Tensor> {
        let info = self.info(name)?;
        if info.dims.len() != 2 {
            return Err(format!("tensor '{name}' is not a matrix").into());
        }
        let (rows, cols) = (info.dims[1], info.dims[0]);
        let bytes = self.bytes(name)?;
        Ok(match info.dtype {
            BF16 => self.bf16(name)?,
            F32 => Tensor::Bf16 {
                shape: info.shape(),
                data: self.f32s(name)?.into_iter().map(to_bf16).collect(),
            },
            PTQ1_0 => self.ternary(name)?,
            Q4_0 => quants::q4_0(bytes, rows, cols),
            Q8_0 => quants::q8_0(bytes, rows, cols),
            Q4_1 | Q6_K => {
                let row = row_bytes(info.dtype, cols)?;
                quants::requantize_q8_0(rows, cols, |r, out| {
                    dequantize_row(info.dtype, &bytes[r * row..][..row], out).expect("sized")
                })
            }
            other => {
                return Err(
                    format!("tensor '{name}' is of type {other}, which no device takes").into(),
                );
            }
        })
    }

    /// Reconstructs row `index` of a matrix into `out`, as wide as a row.
    pub fn row(&self, name: &str, index: usize, out: &mut [f32]) -> Result<()> {
        let info = self.info(name)?;
        if info.dims.len() != 2 || info.dims[0] != out.len() || index >= info.dims[1] {
            return Err(format!("no row {index} of {} in tensor '{name}'", out.len()).into());
        }
        let row = row_bytes(info.dtype, out.len())?;
        dequantize_row(info.dtype, &self.bytes(name)?[index * row..][..row], out)
    }
}

impl Drop for Gguf {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.map as *mut libc::c_void, self.len);
        }
    }
}

/// Writes a GGUF file, for tests: the metadata, and tensors given by name,
/// dimensions (innermost first), type, and data.
#[cfg(any(test, feature = "testing"))]
pub fn write(
    path: &Path,
    meta: &[(&str, Value)],
    tensors: &[(String, Vec<usize>, u32, Vec<u8>)],
) -> Result<()> {
    use std::io::Write;

    fn string(out: &mut Vec<u8>, s: &str) {
        out.extend((s.len() as u64).to_le_bytes());
        out.extend(s.as_bytes());
    }
    fn value(out: &mut Vec<u8>, v: &Value) {
        match v {
            Value::U8(v) => out.push(*v),
            Value::I8(v) => out.push(*v as u8),
            Value::U16(v) => out.extend(v.to_le_bytes()),
            Value::I16(v) => out.extend(v.to_le_bytes()),
            Value::U32(v) => out.extend(v.to_le_bytes()),
            Value::I32(v) => out.extend(v.to_le_bytes()),
            Value::F32(v) => out.extend(v.to_le_bytes()),
            Value::Bool(v) => out.push(*v as u8),
            Value::Str(s) => string(out, s),
            Value::Array(values) => {
                out.extend(kind(&values[0]).to_le_bytes());
                out.extend((values.len() as u64).to_le_bytes());
                for v in values {
                    value(out, v);
                }
            }
            Value::U64(v) => out.extend(v.to_le_bytes()),
            Value::I64(v) => out.extend(v.to_le_bytes()),
            Value::F64(v) => out.extend(v.to_le_bytes()),
        }
    }
    fn kind(v: &Value) -> u32 {
        match v {
            Value::U8(_) => 0,
            Value::I8(_) => 1,
            Value::U16(_) => 2,
            Value::I16(_) => 3,
            Value::U32(_) => 4,
            Value::I32(_) => 5,
            Value::F32(_) => 6,
            Value::Bool(_) => 7,
            Value::Str(_) => 8,
            Value::Array(_) => 9,
            Value::U64(_) => 10,
            Value::I64(_) => 11,
            Value::F64(_) => 12,
        }
    }

    let mut out = Vec::new();
    out.extend(b"GGUF");
    out.extend(3u32.to_le_bytes());
    out.extend((tensors.len() as u64).to_le_bytes());
    out.extend((meta.len() as u64).to_le_bytes());
    for (key, v) in meta {
        string(&mut out, key);
        out.extend(kind(v).to_le_bytes());
        value(&mut out, v);
    }
    let mut offset = 0u64;
    for (name, dims, dtype, data) in tensors {
        string(&mut out, name);
        out.extend((dims.len() as u32).to_le_bytes());
        for &d in dims {
            out.extend((d as u64).to_le_bytes());
        }
        out.extend(dtype.to_le_bytes());
        out.extend(offset.to_le_bytes());
        offset += (data.len() as u64).div_ceil(32) * 32;
    }
    out.resize(out.len().div_ceil(32) * 32, 0);
    for (_, _, _, data) in tensors {
        out.extend(data);
        out.resize(out.len().div_ceil(32) * 32, 0);
    }
    File::create(path)?.write_all(&out)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_back_what_it_wrote() {
        let dir = std::env::temp_dir().join(format!("dwim-gguf-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.gguf");
        let meta = [
            ("general.architecture", Value::Str("qwen35".into())),
            ("qwen35.block_count", Value::U32(2)),
            ("list", Value::Array(vec![Value::I32(-1), Value::I32(1)])),
            ("scale", Value::F32(0.5)),
        ];
        let f32s: Vec<u8> = [1.0f32, 2.0, 3.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let bf16: Vec<u8> = (0..8u16).flat_map(|v| (v << 8).to_le_bytes()).collect();
        let tern = vec![7u8; 2 * ternary::row_bytes(128)];
        let tensors = vec![
            ("norm".to_string(), vec![3], F32, f32s),
            ("w".to_string(), vec![4, 2], BF16, bf16),
            ("t".to_string(), vec![128, 2], PTQ1_0, tern.clone()),
        ];
        write(&path, &meta, &tensors).unwrap();

        let g = Gguf::open(&path).unwrap();
        assert_eq!(g.str("general.architecture").unwrap(), "qwen35");
        assert_eq!(g.u32("qwen35.block_count").unwrap(), 2);
        assert_eq!(g.f32("scale").unwrap(), 0.5);
        assert_eq!(g.array("list").unwrap()[0].as_i64(), Some(-1));
        assert!(g.u32("missing").is_err());
        assert_eq!(g.f32s("norm").unwrap(), [1.0, 2.0, 3.0]);
        match g.bf16("w").unwrap() {
            Tensor::Bf16 { shape, data } => {
                assert_eq!(shape, [2, 4]);
                assert_eq!(data[3], 3 << 8);
            }
            _ => panic!("not bf16"),
        }
        assert_eq!(g.info("t").unwrap().shape(), [2, 128]);
        assert_eq!(g.bytes("t").unwrap(), tern);
        assert!(g.bf16("t").is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
