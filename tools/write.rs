//! The `write` tool: writes a file whole, making the directories it goes
//! in, and says how long it is now. It is for a new file, or one to be
//! replaced wholesale; a change within a file is an `edit`. Left with
//! `bash` alone, the model wrote files through heredocs and `printf`,
//! where a quote, a backslash or a dollar sign in the content was lost to
//! the shell as often as not.
//!
//! The call format puts a value on lines of its own between its tags, so
//! the content arrives without a final newline whether or not the model
//! meant one; a file that does not end in a newline is rare and a file
//! that should but does not is a nuisance, so the content gets one.

use std::{fs, path::Path};

use serde_json::Value;

use crate::{plural, string};

/// The tool as the system prompt declares it.
pub const SIGNATURE: &str = r#"{"type": "function", "function": {"name": "write", "description": "Write a file whole, creating it and the directories it goes in, or replacing what it held; the file ends with a newline. For a change within a file, use edit.", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "The file to write."}, "content": {"type": "string", "description": "What the file is to hold."}}, "required": ["path", "content"]}}}"#;

/// Writes the file in `arguments`, returning what the model should see.
pub fn run(arguments: &Value) -> String {
    let Some(path) = string(arguments, "path") else {
        return "error: write needs a 'path' string".to_string();
    };
    let Some(content) = string(arguments, "content") else {
        return "error: write needs a 'content' string, what the file is to hold".to_string();
    };
    let file = Path::new(path);
    if file.is_dir() {
        return format!("error: {path} is a directory");
    }
    let existed = file.exists();
    if let Some(parent) = file.parent()
        && !parent.as_os_str().is_empty()
        && let Err(e) = fs::create_dir_all(parent)
    {
        return format!("error: cannot make {}: {e}", parent.display());
    }
    let mut content = content.to_string();
    if !content.is_empty() && !content.ends_with('\n') {
        content.push('\n');
    }
    if let Err(e) = fs::write(file, &content) {
        return format!("error: cannot write {path}: {e}");
    }
    let lines = content.lines().count() as u64;
    format!(
        "[{} {path}: {lines} {}, {} bytes]",
        if existed { "replaced" } else { "wrote" },
        plural(lines, "line"),
        content.len()
    )
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use serde_json::json;

    use super::*;

    /// A directory of its own.
    fn dir() -> PathBuf {
        static DIRS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = DIRS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("dwim-write-{}-{n}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn writes_a_file_and_the_directories_it_goes_in() {
        let dir = dir();
        let path = dir.join("deep/down/a.txt");
        let path = path.to_str().unwrap();
        assert_eq!(
            run(&json!({ "path": path, "content": "one\ntwo" })),
            format!("[wrote {path}: 2 lines, 8 bytes]")
        );
        assert_eq!(fs::read_to_string(path).unwrap(), "one\ntwo\n");
        assert_eq!(
            run(&json!({ "path": path, "content": "three\n" })),
            format!("[replaced {path}: 1 line, 6 bytes]")
        );
        assert_eq!(fs::read_to_string(path).unwrap(), "three\n");
        assert_eq!(
            run(&json!({ "path": path, "content": "" })),
            format!("[replaced {path}: 0 lines, 0 bytes]")
        );
        assert_eq!(fs::read_to_string(path).unwrap(), "");
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn refuses_what_it_cannot_write() {
        assert!(run(&json!({ "content": "x" })).starts_with("error: write needs a 'path'"));
        assert!(run(&json!({ "path": "a.txt" })).starts_with("error: write needs a 'content'"));
        assert_eq!(
            run(&json!({ "path": "/", "content": "x" })),
            "error: / is a directory"
        );
        assert!(
            run(&json!({ "path": "/dev/null/nowhere/a.txt", "content": "x" }))
                .starts_with("error: cannot make /dev/null/nowhere: ")
        );
    }
}
