//! Converts a sharded Hugging Face model into a pack: the routed experts
//! quantized to four bits, everything else kept as bf16. Shards are taken one
//! at a time and deleted once converted, so the whole model never has to be
//! on disk at once, and an interrupted conversion picks up where it left off.

use std::{
    collections::BTreeMap,
    error::Error,
    fs,
    io::Read,
    path::{Path, PathBuf},
    thread,
    time::Duration,
};

use hack_models::{
    Weights,
    pack::{Dtype, Writer},
    q4,
};
use rayon::prelude::*;
use serde::Deserialize;

use crate::{
    fetch::{self, Progress as Download},
    models::Model,
};

/// What the conversion is doing.
#[derive(Clone, Copy)]
pub enum Progress {
    /// Reading the index and the shards' headers.
    Planning,
    /// Waiting for, or downloading, a shard: which one, out of how many.
    Downloading { shard: usize, total: usize, download: Option<Download> },
    /// Converting a shard: which one, out of how many, and how many of its
    /// tensors are done, out of how many.
    Converting { shard: usize, total: usize, done: usize, tensors: usize },
}

#[derive(Deserialize)]
struct Index {
    weight_map: BTreeMap<String, String>,
}

#[derive(Deserialize)]
struct Entry {
    dtype: String,
    shape: Vec<usize>,
}

/// Makes sure `dir/model.hack` holds the converted model, returning its
/// path. Shards already in `dir/shards/` are used, and otherwise fetched.
pub fn ensure_pack(model: &'static Model, dir: &Path, mut progress: impl FnMut(Progress)) -> Result<PathBuf, Box<dyn Error>> {
    let pack_path = dir.join("model.hack");
    let done_path = dir.join("model.hack.done");
    if done_path.exists() {
        return Ok(pack_path);
    }
    progress(Progress::Planning);

    let index: Index = serde_json::from_str(&cached(dir, "model.safetensors.index.json", || {
        Ok(ureq::get(&model.url("model.safetensors.index.json")).call()?.into_body().read_to_string()?)
    })?)?;
    let mut shards: Vec<String> = index.weight_map.values().cloned().collect();
    shards.sort();
    shards.dedup();

    // Every tensor's shape, from the shards' headers, without their data.
    let mut tensors: BTreeMap<String, (String, Vec<usize>)> = BTreeMap::new();
    for shard in &shards {
        let header = cached(dir, &format!("{shard}.header.json"), || fetch_header(model, shard))?;
        let entries: BTreeMap<String, serde_json::Value> = serde_json::from_str(&header)?;
        for (name, value) in entries {
            if name == "__metadata__" {
                continue;
            }
            let entry: Entry = serde_json::from_value(value)?;
            if entry.dtype != "BF16" {
                return Err(format!("tensor '{name}' is {}, expected BF16", entry.dtype).into());
            }
            tensors.insert(name, (shard.clone(), entry.shape));
        }
    }

    // The pack's layout: experts gathered per layer and quantized, the rest
    // as is.
    let mut layout: BTreeMap<String, (Dtype, Vec<usize>)> = BTreeMap::new();
    for (name, (_, shape)) in &tensors {
        match expert(name) {
            Some((prefix, index, kind)) => {
                let entry = layout.entry(format!("{prefix}.{kind}")).or_insert((Dtype::Q4, vec![0, shape[0], shape[1]]));
                entry.1[0] = entry.1[0].max(index + 1);
            }
            None => {
                layout.insert(name.clone(), (Dtype::Bf16, shape.clone()));
            }
        }
    }
    let layout: Vec<(String, Dtype, Vec<usize>)> = layout.into_iter().map(|(name, (dtype, shape))| (name, dtype, shape)).collect();
    let writer = Writer::create(&pack_path, &layout)?;

    let progress_path = dir.join("convert.progress");
    let mut converted: Vec<String> = fs::read_to_string(&progress_path)
        .map(|text| text.lines().map(str::to_string).collect())
        .unwrap_or_default();
    let shard_dir = dir.join("shards");
    fs::create_dir_all(&shard_dir)?;
    for (i, shard) in shards.iter().enumerate() {
        if converted.contains(shard) {
            continue;
        }
        let path = shard_dir.join(shard);
        wait_for_shard(model, shard, &path, |download| {
            progress(Progress::Downloading {
                shard: i + 1,
                total: shards.len(),
                download,
            })
        })?;
        let names: Vec<&String> = tensors.iter().filter(|(_, (in_shard, _))| in_shard == shard).map(|(name, _)| name).collect();
        let mut weights = Weights::open(&path)?;
        for (j, name) in names.iter().enumerate() {
            progress(Progress::Converting {
                shard: i + 1,
                total: shards.len(),
                done: j,
                tensors: names.len(),
            });
            let tensor = weights.read(name)?;
            match expert(name) {
                Some((prefix, index, kind)) => {
                    let cols = tensor.shape[1];
                    let rows = tensor.shape[0];
                    let row_bytes = q4::row_bytes(cols);
                    let mut bytes = vec![0u8; rows * row_bytes];
                    bytes
                        .par_chunks_exact_mut(row_bytes)
                        .zip(tensor.data.par_chunks_exact(cols))
                        .for_each(|(out, row)| {
                            let row: Vec<f32> = row.iter().map(|&v| hack_gpu::bf16(v)).collect();
                            q4::quantize_row(&row, out);
                        });
                    writer.write(&format!("{prefix}.{kind}"), (index * rows * row_bytes) as u64, &bytes)?;
                }
                None => {
                    let bytes: Vec<u8> = tensor.data.iter().flat_map(|v| v.to_le_bytes()).collect();
                    writer.write(name, 0, &bytes)?;
                }
            }
        }
        writer.sync()?;
        converted.push(shard.clone());
        fs::write(&progress_path, converted.join("\n") + "\n")?;
        drop(weights);
        fs::remove_file(&path)?;
    }
    fs::write(&done_path, "")?;
    Ok(pack_path)
}

/// Splits an expert tensor's name, such as
/// `model.layers.3.mlp.experts.17.gate_proj.weight`, into the experts'
/// prefix, the expert's index, and the projection.
fn expert(name: &str) -> Option<(&str, usize, &str)> {
    let (prefix, rest) = name.split_once(".experts.")?;
    let (index, kind) = rest.split_once('.')?;
    let kind = kind.strip_suffix(".weight")?;
    Some((&name[..prefix.len() + ".experts".len()], index.parse().ok()?, kind))
}

/// Reads a small file from `dir`, fetching it with `fetch` first if needed.
fn cached(dir: &Path, name: &str, fetch: impl FnOnce() -> Result<String, Box<dyn Error>>) -> Result<String, Box<dyn Error>> {
    let path = dir.join(name);
    if let Ok(text) = fs::read_to_string(&path) {
        return Ok(text);
    }
    let text = fetch()?;
    fs::write(&path, &text)?;
    Ok(text)
}

/// Fetches just the JSON header of a safetensors shard, by byte range.
fn fetch_header(model: &Model, shard: &str) -> Result<String, Box<dyn Error>> {
    let url = model.url(shard);
    let mut len = Vec::new();
    ureq::get(&url).header("Range", "bytes=0-7").call()?.into_body().into_reader().read_to_end(&mut len)?;
    if len.len() != 8 {
        return Err(format!("{shard}: could not read the header length").into());
    }
    let len = u64::from_le_bytes(len.try_into().unwrap());
    let mut header = Vec::new();
    ureq::get(&url)
        .header("Range", &format!("bytes=8-{}", 7 + len))
        .call()?
        .into_body()
        .into_reader()
        .read_to_end(&mut header)?;
    if header.len() as u64 != len {
        return Err(format!("{shard}: got {} of {len} header bytes", header.len()).into());
    }
    Ok(String::from_utf8(header)?)
}

/// Waits until the shard is at `path`: another process may be downloading
/// it (to a `.part` file beside it); otherwise it is downloaded here.
fn wait_for_shard(model: &Model, shard: &str, path: &Path, mut progress: impl FnMut(Option<Download>)) -> Result<(), Box<dyn Error>> {
    loop {
        if path.exists() {
            return Ok(());
        }
        let partial = fs::read_dir(path.parent().unwrap())?
            .flatten()
            .any(|entry| entry.file_name().to_string_lossy().starts_with(shard) && entry.file_name().to_string_lossy().ends_with(".part"));
        if partial {
            progress(None);
            thread::sleep(Duration::from_secs(2));
            continue;
        }
        let file: &'static str = Box::leak(shard.to_string().into_boxed_str());
        fetch::download(model, file, path, &mut |download| progress(Some(download)))?;
        return Ok(());
    }
}
