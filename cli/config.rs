//! What the `dwim` command keeps from one run to the next: the providers
//! that are disabled. It is a JSON file in the user's configuration
//! directory, written by the shell's `/provider` and fit to edit by hand:
//!
//! ```json
//! { "disabled_providers": ["local"] }
//! ```

use std::{
    error::Error,
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
};

use dwim::{providers::Kind, sys};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// The configuration, as the file has it.
#[derive(Default, Deserialize, Serialize)]
pub struct Config {
    /// The providers that are disabled, by the names `--provider` takes.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    disabled_providers: Vec<String>,
    /// What else the file has, which is another version's to read.
    #[serde(flatten)]
    rest: Map<String, Value>,
}

impl Config {
    /// The user's configuration, which is empty where there is no file.
    pub fn load() -> Result<Self, Box<dyn Error>> {
        match path() {
            Some(path) => Self::read(&path),
            None => Ok(Self::default()),
        }
    }

    /// The configuration in the file at `path`, which is empty if there is
    /// no such file.
    pub fn read(path: &Path) -> Result<Self, Box<dyn Error>> {
        let text = match fs::read_to_string(path) {
            Ok(text) => text,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(Self::default()),
            Err(e) => return Err(format!("{}: {e}", path.display()).into()),
        };
        serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()).into())
    }

    /// Whether `provider` is disabled: a model runs there only when the
    /// command line names it.
    pub fn disabled(&self, provider: Kind) -> bool {
        let name = provider.to_string();
        self.disabled_providers.contains(&name)
    }

    /// Disables `provider`, or enables it again.
    pub fn set_disabled(&mut self, provider: Kind, disabled: bool) {
        let name = provider.to_string();
        self.disabled_providers.retain(|other| *other != name);
        if disabled {
            self.disabled_providers.push(name);
        }
    }

    /// Keeps the configuration for the runs to come.
    pub fn save(&self) -> Result<(), Box<dyn Error>> {
        let path = path().ok_or("there is no directory to keep the configuration in")?;
        self.write(&path)
    }

    /// Writes the configuration to the file at `path`, in a directory
    /// made for it if there is none: to a file beside it first, which
    /// then takes its place, so that the file is never half written.
    pub fn write(&self, path: &Path) -> Result<(), Box<dyn Error>> {
        let failed = |e: io::Error| format!("{}: {e}", path.display());
        if let Some(dir) = path.parent() {
            sys::fs::create_private_dir(dir).map_err(failed)?;
        }
        let text = serde_json::to_string_pretty(self)?;
        let partial = path.with_extension("json.partial");
        let mut file = sys::fs::create_private(&partial).map_err(failed)?;
        writeln!(file, "{text}").map_err(failed)?;
        fs::rename(&partial, path).map_err(failed)?;
        Ok(())
    }
}

/// Where the configuration is kept: in the user's configuration directory
/// (`~/.config`, or on macOS `~/Library/Application Support`).
fn path() -> Option<PathBuf> {
    Some(sys::dirs::config()?.join("dwim").join("config.json"))
}
