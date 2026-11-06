//! The Hugging Face cache, where the model's files are kept in the layout
//! the `huggingface_hub` library and the tools built on it use, so that a
//! model any of them downloaded is downloaded once, and theirs is found
//! here. A repository's files at a revision are in `snapshots/<revision>/`,
//! each a symbolic link to its content in `blobs/`, named by its digest:
//! the SHA-256 of a file in LFS storage, and the git blob SHA-1 of a small
//! file. A file still downloading is `blobs/<digest>.incomplete`.

use std::{
    env,
    ffi::OsString,
    path::{Path, PathBuf},
};

/// The server the files come from unless `HF_ENDPOINT` names another.
const ENDPOINT: &str = "https://huggingface.co";

/// Where the cache is: `$HF_HUB_CACHE`, else `hub` under `$HF_HOME`, else
/// `~/.cache/huggingface/hub`, on every platform, as the library has it.
pub fn cache_dir() -> Option<PathBuf> {
    if let Some(dir) = var("HF_HUB_CACHE") {
        return Some(dir.into());
    }
    if let Some(home) = var("HF_HOME") {
        return Some(PathBuf::from(home).join("hub"));
    }
    Some(
        dirs::home_dir()?
            .join(".cache")
            .join("huggingface")
            .join("hub"),
    )
}

/// The directory of the model repository `repo`, named `org/name`.
pub fn repo_dir(repo: &str) -> Option<PathBuf> {
    Some(cache_dir()?.join(format!("models--{}", repo.replace('/', "--"))))
}

/// Where the files of `repo` at `revision`, a commit hash, are kept.
pub fn snapshot_dir(repo: &str, revision: &str) -> Option<PathBuf> {
    Some(repo_dir(repo)?.join("snapshots").join(revision))
}

/// Where the contents of the files of `repo` are kept, named by digest.
pub fn blobs_dir(repo: &str) -> Option<PathBuf> {
    Some(repo_dir(repo)?.join("blobs"))
}

/// The link a file in a snapshot is, to the blob called `digest`.
pub fn blob_link(digest: &str) -> PathBuf {
    Path::new("..").join("..").join("blobs").join(digest)
}

/// URL to download `file` of `repo` at `revision` from.
pub fn url(repo: &str, revision: &str, file: &str) -> String {
    let endpoint = var("HF_ENDPOINT").map_or(ENDPOINT.to_string(), |endpoint| {
        endpoint.to_string_lossy().trim_end_matches('/').to_string()
    });
    format!("{endpoint}/{repo}/resolve/{revision}/{file}")
}

/// Whether `HF_HUB_OFFLINE` says never to use the network.
pub fn offline() -> bool {
    var("HF_HUB_OFFLINE").is_some_and(|value| {
        matches!(
            value.to_string_lossy().to_ascii_uppercase().as_str(),
            "1" | "ON" | "YES" | "TRUE"
        )
    })
}

/// The environment variable `name`, if it is set to anything.
fn var(name: &str) -> Option<OsString> {
    env::var_os(name).filter(|value| !value.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_the_repo_dir_as_the_library_does() {
        let dir = repo_dir("prism-ml/Ternary-Bonsai-2-27B-gguf").unwrap();
        assert_eq!(
            dir.file_name().unwrap(),
            "models--prism-ml--Ternary-Bonsai-2-27B-gguf"
        );
    }

    #[test]
    fn links_a_snapshot_file_to_its_blob() {
        assert_eq!(
            blob_link("abc"),
            Path::new("..").join("..").join("blobs").join("abc")
        );
    }
}
