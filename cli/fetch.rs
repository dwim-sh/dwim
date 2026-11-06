//! Downloading the model's files into its directory, resuming what an
//! interrupted run left.

use std::{
    error::Error,
    fs::{self, OpenOptions},
    io::{Read, Write},
    path::Path,
    thread,
    time::Duration,
};

use crate::model;

/// How long a partial download can go without being written to before it is
/// taken for the leftover of an interrupted run, to be resumed, rather than
/// the work of another run still in progress, to be waited for.
const STALE: Duration = Duration::from_secs(60);

/// How far along the download of one of a model's files is.
#[derive(Clone, Copy, Debug)]
pub struct Progress {
    pub file: &'static str,
    /// Bytes downloaded so far.
    pub done: u64,
    /// Size of the file, if the server said, and always once the file is
    /// complete: the last report of a file has `total == Some(done)`.
    pub total: Option<u64>,
}

/// Downloads the model's files into `dir`, skipping any that are already
/// there, and reporting progress on each one as it comes in.
pub fn fetch(dir: &Path, mut progress: impl FnMut(Progress)) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(dir)?;
    for file in model::FILES {
        let path = dir.join(file);
        if path.exists() {
            continue;
        }
        download(file, &path, &mut progress)?;
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
fn download(
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
    let mut request = ureq::get(&model::url(file));
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
