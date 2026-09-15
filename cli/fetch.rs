use std::{
    error::Error,
    fs::{self, File},
    io::{self, Read, Write},
    path::{Path, PathBuf},
    time::Duration,
};

use crate::models::{MODELS, Model};

/// How long a partial download can go without being written to before it is
/// taken for the leftover of an interrupted run, rather than the work of
/// another run still in progress, and removed.
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

/// Downloads a model's files into `dir`, skipping any that are already there,
/// and reporting progress on each one as it comes in.
pub fn fetch(model: &'static Model, dir: &Path, mut progress: impl FnMut(Progress)) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(dir)?;
    remove_stale(dir)?;
    for file in model.files {
        let path = dir.join(file);
        if path.exists() {
            continue;
        }
        download(model, file, &path, &mut progress)?;
    }
    Ok(())
}

/// Removes partial downloads left behind by interrupted runs.
fn remove_stale(dir: &Path) -> io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let partial = entry.file_name().to_str().is_some_and(|name| name.ends_with(".part"));
        let untouched = entry.metadata()?.modified()?.elapsed().is_ok_and(|age| age > STALE);
        if partial && untouched {
            fs::remove_file(entry.path())?;
        }
    }
    Ok(())
}

/// Downloads one of the model's files to `path`.
///
/// The file is written under a temporary name and renamed into place once
/// complete, so an interrupted download never looks like a finished one. The
/// name is unique to this process, so that runs started at the same time
/// don't write over each other: whichever finishes first puts the file in
/// place, and the others replace it with their identical copies.
fn download(
    model: &Model,
    file: &'static str,
    path: &Path,
    progress: &mut impl FnMut(Progress),
) -> Result<(), Box<dyn Error>> {
    progress(Progress {
        file,
        done: 0,
        total: None,
    });
    let response = ureq::get(&model.url(file)).call()?;
    let partial = path.with_file_name(format!("{file}.{}.part", std::process::id()));
    let result = save(response, &partial, file, progress);
    if result.is_err() {
        let _ = fs::remove_file(&partial);
    }
    result?;
    fs::rename(&partial, path)?;
    Ok(())
}

/// Writes the body of a response to `partial`, reporting progress.
fn save(
    response: ureq::http::Response<ureq::Body>,
    partial: &Path,
    file: &'static str,
    progress: &mut impl FnMut(Progress),
) -> Result<(), Box<dyn Error>> {
    let total = response
        .headers()
        .get("content-length")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok());
    let mut reader = response.into_body().into_reader();
    let mut writer = File::create(partial)?;
    let mut buf = vec![0; 1 << 20];
    let mut done = 0;
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
