//! What the project's files are at a moment: each file's path, when it
//! was last modified and its size, without its contents. The harness
//! takes one before a tool call that can change files and one after, and
//! the paths that differ are what the call changed, whichever tool it
//! was: a file `bash` rewrites with `sed -i` is changed as one `edit`
//! rewrites is. The files are those git does not ignore, so that a build
//! or a test run that writes only what the project ignores changes
//! nothing; where there is no git, a walk that leaves out the
//! directories that hold what is built or fetched.

use std::{collections::BTreeMap, fs, path::Path, process::Command, time::SystemTime};

/// The files under a directory, by their paths relative to it with `/`
/// between the parts, each with when it was last modified and its size.
pub struct Snapshot {
    files: BTreeMap<String, (Option<SystemTime>, u64)>,
}

impl Snapshot {
    /// The files under `dir` as they now are.
    pub fn of(dir: &Path) -> Self {
        let mut files = BTreeMap::new();
        for path in listed(dir).unwrap_or_else(|| walked(dir)) {
            // A file deleted but not yet committed is still listed.
            if let Ok(file) = fs::symlink_metadata(dir.join(&path))
                && !file.is_dir()
            {
                files.insert(path, (file.modified().ok(), file.len()));
            }
        }
        Self { files }
    }

    /// The paths of the files that are not in `before` as they are here:
    /// those made, changed or deleted since, in order.
    pub fn changed(&self, before: &Snapshot) -> Vec<String> {
        let mut changed: Vec<String> = self
            .files
            .iter()
            .filter(|(path, file)| before.files.get(*path) != Some(file))
            .map(|(path, _)| path.clone())
            .chain(
                before
                    .files
                    .keys()
                    .filter(|path| !self.files.contains_key(*path))
                    .cloned(),
            )
            .collect();
        changed.sort();
        changed
    }
}

/// The files under `dir` that git tracks or does not ignore, if `dir` is
/// in a repository.
fn listed(dir: &Path) -> Option<Vec<String>> {
    let output = Command::new("git")
        .args([
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ])
        .current_dir(dir)
        .output()
        .ok()
        .filter(|output| output.status.success())?;
    Some(
        output
            .stdout
            .split(|&byte| byte == 0)
            .filter(|path| !path.is_empty())
            .map(|path| String::from_utf8_lossy(path).into_owned())
            .collect(),
    )
}

/// Directories the walk does not enter, with the hidden ones: they hold
/// what is built or fetched, not the project.
const BUILT: [&str; 3] = ["__pycache__", "node_modules", "target"];

/// Most files the walk lists: a directory that is not a project, as a
/// home directory is, is not walked whole before every command.
const CAP: usize = 20_000;

/// The files under `dir` a walk finds, up to `CAP`, not following
/// symbolic links: the first of them in name order, so that two walks
/// list the same files.
fn walked(dir: &Path) -> Vec<String> {
    let mut files = Vec::new();
    walk(dir, "", &mut files);
    files
}

/// Adds the files under `dir`, whose path relative to the root is `at`,
/// to `files`, in name order, until there are `CAP`.
fn walk(dir: &Path, at: &str, files: &mut Vec<String>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<(String, fs::DirEntry)> = entries
        .flatten()
        .filter_map(|entry| Some((entry.file_name().into_string().ok()?, entry)))
        .collect();
    entries.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    for (name, entry) in entries {
        if files.len() == CAP {
            return;
        }
        let path = if at.is_empty() {
            name.clone()
        } else {
            format!("{at}/{name}")
        };
        match entry.file_type() {
            Ok(kind) if kind.is_dir() => {
                if !name.starts_with('.') && !BUILT.contains(&name.as_str()) {
                    walk(&entry.path(), &path, files);
                }
            }
            Ok(_) => files.push(path),
            Err(_) => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{fs, path::PathBuf, process::Command};

    use super::Snapshot;

    /// A directory of its own, called `name`.
    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("dwim-snapshot-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Runs `script` with `sh` in `dir`.
    fn sh(dir: &std::path::Path, script: &str) {
        let status = Command::new("sh")
            .args(["-c", script])
            .current_dir(dir)
            .status()
            .unwrap();
        assert!(status.success(), "{script}");
    }

    #[test]
    fn names_the_files_made_changed_and_deleted() {
        let dir = scratch("changed");
        sh(
            &dir,
            "mkdir src && echo one > src/kept && echo two > src/gone && echo three > src/same",
        );
        let before = Snapshot::of(&dir);
        assert!(Snapshot::of(&dir).changed(&before).is_empty());
        // A change that keeps the size, as a rename of a name does.
        sh(
            &dir,
            "sleep 0.05 && sed -i.bak s/one/uno/ src/kept && rm src/kept.bak src/gone && echo new > made",
        );
        assert_eq!(
            Snapshot::of(&dir).changed(&before),
            ["made", "src/gone", "src/kept"]
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn leaves_out_what_git_ignores() {
        let dir = scratch("ignored");
        sh(
            &dir,
            "git init -q && echo 'target/' > .gitignore && mkdir .github && echo ci > .github/ci.yml",
        );
        let before = Snapshot::of(&dir);
        sh(
            &dir,
            "mkdir target && echo built > target/out && echo more >> .github/ci.yml && echo new > untracked",
        );
        assert_eq!(
            Snapshot::of(&dir).changed(&before),
            [".github/ci.yml", "untracked"]
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn leaves_out_what_is_built_where_there_is_no_git() {
        let dir = scratch("built");
        let before = Snapshot::of(&dir);
        sh(
            &dir,
            "mkdir -p node_modules/a pkg/__pycache__ && echo x > node_modules/a/index.js && echo x > pkg/__pycache__/m.pyc && echo x > pkg/m.py",
        );
        assert_eq!(Snapshot::of(&dir).changed(&before), ["pkg/m.py"]);
        fs::remove_dir_all(&dir).unwrap();
    }
}
