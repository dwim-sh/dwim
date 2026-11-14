//! Downloading the model's files into the Hugging Face cache, resuming what
//! an interrupted run left.

use std::{
    fs::{self, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    os::unix::fs::symlink,
    path::Path,
    thread,
    time::Duration,
};

use crate::inference::{
    Result, hub,
    qwen35::{self, File},
};

/// How long a partial download can go without being written to before it is
/// taken for the leftover of an interrupted run, to be resumed, rather than
/// the work of another run still in progress, to be waited for.
const STALE: Duration = Duration::from_secs(60);

/// How far along the download of one of a model's files is.
#[derive(Clone, Copy, Debug)]
pub struct Download {
    pub file: &'static str,
    /// Bytes downloaded so far.
    pub done: u64,
    /// Size of the file, if the server said, and always once the file is
    /// complete: the last report of a file has `total == Some(done)`.
    pub total: Option<u64>,
}

/// Downloads the model's files into `dir`, its snapshot in the cache,
/// skipping any that are already there, and reporting progress on each
/// one as it comes in. A file's content goes in the repository's blobs,
/// named by its digest, and the file in the snapshot is a link to it, as
/// the cache's other users lay them out; a blob one of them downloaded is
/// linked to without downloading it again.
pub fn fetch(dir: &Path, mut progress: impl FnMut(Download)) -> Result<()> {
    let blobs = hub::blobs_dir(qwen35::REPO).ok_or("cannot determine the home directory")?;
    for file in &qwen35::FILES {
        let path = dir.join(file.name);
        if path.exists() {
            continue;
        }
        let blob = blobs.join(file.digest);
        if !blob.exists() {
            if hub::offline() {
                return Err("the model is not downloaded, and HF_HUB_OFFLINE is set: \
                            run `dwim --download` where there is network"
                    .into());
            }
            fs::create_dir_all(&blobs)?;
            download(file, &blob, &mut progress)?;
        }
        fs::create_dir_all(dir)?;
        // A link left pointing at a blob that was removed is replaced.
        if fs::symlink_metadata(&path).is_ok() {
            fs::remove_file(&path)?;
        }
        symlink(hub::blob_link(file.digest), &path)?;
    }
    Ok(())
}

/// Downloads one of the model's files to `blob`.
///
/// The file is written under an `.incomplete` name, as the cache's other
/// users write theirs, and renamed into place once complete, so an
/// interrupted download never looks like a finished one. A partial file
/// another run is still writing is waited for; one nobody is writing is
/// picked up where it left off, which matters for a model of several
/// gigabytes on a slow link.
fn download(file: &File, blob: &Path, progress: &mut impl FnMut(Download)) -> Result<()> {
    let partial = blob.with_file_name(format!("{}.incomplete", file.digest));
    let mut done = loop {
        match fs::metadata(&partial) {
            Ok(meta) if meta.modified()?.elapsed().is_ok_and(|age| age < STALE) => {
                progress(Download {
                    file: file.name,
                    done: meta.len(),
                    total: None,
                });
                thread::sleep(Duration::from_secs(2));
                if blob.exists() {
                    return Ok(());
                }
            }
            Ok(meta) => break meta.len(),
            Err(_) => break 0,
        }
    };
    progress(Download {
        file: file.name,
        done,
        total: None,
    });
    let mut request = ureq::get(&hub::url(qwen35::REPO, qwen35::REVISION, file.name));
    if done > 0 {
        request = request.header("Range", &format!("bytes={done}-"));
    }
    let response = request.call()?;
    // A server that ignores the range sends the whole file, from the start.
    if done > 0 && response.status() != 206 {
        done = 0;
    }
    let result = save(response, &partial, done, file.name, progress);
    if result.is_err() {
        // Left for the next run to resume, unless nothing arrived.
        if fs::metadata(&partial).is_ok_and(|meta| meta.len() == 0) {
            let _ = fs::remove_file(&partial);
        }
    }
    result?;
    fs::rename(&partial, blob)?;
    Ok(())
}

/// Writes the body of a response to `partial` from `done` bytes in, which
/// it holds already, reporting progress.
fn save(
    response: ureq::http::Response<ureq::Body>,
    partial: &Path,
    mut done: u64,
    file: &'static str,
    progress: &mut impl FnMut(Download),
) -> Result<()> {
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
    writer.seek(SeekFrom::Start(done))?;
    let mut buf = vec![0; 1 << 20];
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        writer.write_all(&buf[..n])?;
        done += n as u64;
        if Some(done) != total {
            progress(Download { file, done, total });
        }
    }
    if let Some(total) = total
        && done != total
    {
        return Err(format!("{file}: got {done} of {total} bytes").into());
    }
    progress(Download {
        file,
        done,
        total: Some(done),
    });
    Ok(())
}
