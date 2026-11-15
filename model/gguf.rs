//! The header of a GGUF file, the format the model ships in: what the file
//! says of the model before its weights, which is all a tokenizer needs
//! of it.
//!
//! ```text
//! "GGUF" | version (u32) | tensor count (u64) | key count (u64)
//! keys: name | type (u32) | value
//! ```
//!
//! Strings are a u64 length and the bytes.

use std::{collections::BTreeMap, io::Read};

use crate::model::Result;

/// The metadata of a GGUF file: its values by key.
pub struct Metadata {
    values: BTreeMap<String, Value>,
}

impl Metadata {
    /// Reads the header of a GGUF file from its start: the metadata, and
    /// how many tensors the file describes after it, where `r` is left.
    pub fn read(r: &mut Reader<impl Read>) -> Result<(Self, u64)> {
        if r.bytes(4)? != b"GGUF" {
            return Err("not a GGUF file".into());
        }
        let version = r.u32()?;
        if version != 3 {
            return Err(format!("GGUF version {version} is not supported").into());
        }
        let n_tensors = r.u64()?;
        let n_meta = r.u64()?;
        let mut values = BTreeMap::new();
        for _ in 0..n_meta {
            let key = r.string()?;
            let kind = r.u32()?;
            values.insert(key, r.value(kind)?);
        }
        Ok((Self { values }, n_tensors))
    }

    /// Metadata of `values`, as a file would carry them.
    pub fn new(values: impl IntoIterator<Item = (String, Value)>) -> Self {
        Self {
            values: values.into_iter().collect(),
        }
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.values.get(key)
    }

    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.values.keys().map(String::as_str)
    }

    fn need(&self, key: &str) -> Result<&Value> {
        self.values
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
}

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

/// Reads the values a GGUF file is written in from a stream of its bytes,
/// counting how many it has read.
pub struct Reader<R> {
    inner: R,
    read: usize,
}

impl<R: Read> Reader<R> {
    pub fn new(inner: R) -> Self {
        Self { inner, read: 0 }
    }

    /// Bytes read so far.
    pub fn position(&self) -> usize {
        self.read
    }

    pub fn bytes(&mut self, n: usize) -> Result<Vec<u8>> {
        let mut buf = vec![0; n];
        self.inner.read_exact(&mut buf)?;
        self.read += n;
        Ok(buf)
    }

    pub fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.bytes(4)?.try_into().unwrap()))
    }

    pub fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.bytes(8)?.try_into().unwrap()))
    }

    pub fn string(&mut self) -> Result<String> {
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
