use std::{
    error::Error,
    fs::{self, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    thread,
    time::Duration,
};

use crate::models::{MODELS, Model};

/// How long a partial download can go without being written to before it is
/// taken for the leftover of an interrupted run, to be resumed, rather than
/// the work of another run still in progress, to be waited for.
const STALE: Duration = Duration::from_secs(60);

/// How far along the download of one of a model's files is.
#[derive(Clone, Copy)]
pub struct Progress {
    pub file: &'static str,
    /// Bytes downloaded so far.
    pub done: u64,
    /// Size of the file, if the server said, and always once the file is
    /// complete: the last report of a file has `total == Some(done)`.
    pub total: Option<u64>,
}

/// Looks up a model by name, and the directory its files are kept in.
pub fn locate(name: &str) -> Result<(&'static Model, PathBuf), Box<dyn Error>> {
    let model = Model::find(name).ok_or_else(|| {
        let known: Vec<_> = MODELS.iter().map(|model| model.name).collect();
        format!("unknown model '{name}' (known: {})", known.join(", "))
    })?;
    let dir = model.dir().ok_or("cannot determine the cache directory")?;
    Ok((model, dir))
}

/// How much of a model is in `dir`: how many of its files, and their size in
/// bytes.
pub fn on_disk(model: &Model, dir: &Path) -> (usize, u64) {
    model
        .files
        .iter()
        .filter_map(|file| fs::metadata(dir.join(file)).ok())
        .fold((0, 0), |(files, bytes), metadata| {
            (files + 1, bytes + metadata.len())
        })
}

/// Downloads a model's files into `dir`, skipping any that are already there,
/// and reporting progress on each one as it comes in.
pub fn fetch(
    model: &'static Model,
    dir: &Path,
    mut progress: impl FnMut(Progress),
) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(dir)?;
    for file in model.files {
        let path = dir.join(file);
        if path.exists() {
            continue;
        }
        download(model, file, &path, &mut progress)?;
    }
    Ok(())
}

/// Downloads one of the model's files to `path`.
///
/// The file is written under a `.part` name and renamed into place once
/// complete, so an interrupted download never looks like a finished one.
/// A partial file another run is still writing is waited for; one nobody is
/// writing is picked up where it left off, which matters for a model of
/// several gigabytes on a slow link.
pub fn download(
    model: &Model,
    file: &'static str,
    path: &Path,
    progress: &mut impl FnMut(Progress),
) -> Result<(), Box<dyn Error>> {
    let partial = path.with_file_name(format!("{file}.part"));
    let mut done = loop {
        match fs::metadata(&partial) {
            Ok(meta) if meta.modified()?.elapsed().is_ok_and(|age| age < STALE) => {
                progress(Progress {
                    file,
                    done: meta.len(),
                    total: None,
                });
                thread::sleep(Duration::from_secs(2));
                if path.exists() {
                    return Ok(());
                }
            }
            Ok(meta) => break meta.len(),
            Err(_) => break 0,
        }
    };
    progress(Progress {
        file,
        done,
        total: None,
    });
    let mut request = ureq::get(&model.url(file));
    if done > 0 {
        request = request.header("Range", &format!("bytes={done}-"));
    }
    let response = request.call()?;
    // A server that ignores the range sends the whole file, from the start.
    if done > 0 && response.status() != 206 {
        done = 0;
    }
    let result = save(response, &partial, done, file, progress);
    if result.is_err() {
        // Left for the next run to resume, unless nothing arrived.
        if fs::metadata(&partial).is_ok_and(|meta| meta.len() == 0) {
            let _ = fs::remove_file(&partial);
        }
    }
    result?;
    fs::rename(&partial, path)?;
    Ok(())
}

/// Writes the body of a response to `partial` from `done` bytes in, which
/// it holds already, reporting progress.
fn save(
    response: ureq::http::Response<ureq::Body>,
    partial: &Path,
    mut done: u64,
    file: &'static str,
    progress: &mut impl FnMut(Progress),
) -> Result<(), Box<dyn Error>> {
    let total = response
        .headers()
        .get("content-length")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok())
        .map(|length| done + length);
    let mut reader = response.into_body().into_reader();
    let mut writer = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(false)
        .open(partial)?;
    writer.set_len(done)?;
    std::io::Seek::seek(&mut writer, std::io::SeekFrom::Start(done))?;
    let mut buf = vec![0; 1 << 20];
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        writer.write_all(&buf[..n])?;
        done += n as u64;
        // The file's last report, once it is complete, is made below.
        if Some(done) != total {
            progress(Progress { file, done, total });
        }
    }
    if let Some(total) = total
        && done != total
    {
        return Err(format!("{file}: got {done} of {total} bytes").into());
    }
    progress(Progress {
        file,
        done,
        total: Some(done),
    });
    Ok(())
}

/// Formats a byte count for humans, in decimal units.
pub fn size(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["KB", "MB", "GB", "TB"];
    if bytes < 1000 {
        return format!("{bytes} B");
    }
    let mut value = bytes as f64 / 1000.0;
    let mut unit = 0;
    while value >= 1000.0 && unit < UNITS.len() - 1 {
        value /= 1000.0;
        unit += 1;
    }
    format!("{value:.1} {}", UNITS[unit])
}
