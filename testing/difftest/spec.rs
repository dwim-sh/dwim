//! What the subjects checked against a Lean spec in `specs/` share: the
//! spec's oracle, which says what the spec makes of each case, and the
//! generator the cases are made with.

use std::{
    io::Write,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread,
};

use crate::Result;

/// What the oracle `exe` of the specs answers to `input`, a case a line,
/// an answer a line.
pub fn ask(exe: &str, input: &str) -> Result<String> {
    let specs = Path::new(env!("CARGO_MANIFEST_DIR")).join("specs");
    let mut child = Command::new(lake()?)
        .args(["exe", exe])
        .current_dir(specs)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    // The oracle answers as it reads, so it is fed from another thread
    // while its answers are read here, or both pipes could fill.
    let mut stdin = child
        .stdin
        .take()
        .ok_or("the oracle's input is not piped")?;
    let input = input.to_string();
    let feeder = thread::spawn(move || stdin.write_all(input.as_bytes()));
    let output = child.wait_with_output()?;
    feeder
        .join()
        .map_err(|_| "the oracle's feeder panicked")??;
    if !output.status.success() {
        return Err(format!("{exe} failed: {}", String::from_utf8_lossy(&output.stderr)).into());
    }
    Ok(String::from_utf8(output.stdout)?)
}

/// Where `lake` is: on the path, or where elan installs it.
fn lake() -> Result<PathBuf> {
    let elan = std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".elan/bin/lake"));
    std::env::var_os("PATH")
        .iter()
        .flat_map(std::env::split_paths)
        .map(|dir| dir.join("lake"))
        .chain(elan)
        .find(|path| path.is_file())
        .ok_or_else(|| "no lake: install a Lean toolchain with elan".into())
}

/// A xorshift generator, so that a case is the same on every run.
pub struct Rng(pub u64);

impl Rng {
    /// A generator for the case made from `seed`.
    pub fn seeded(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1)
    }

    /// A number below `n`, or 0 if `n` is.
    pub fn below(&mut self, n: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 % n.max(1) as u64) as usize
    }

    /// A budget: one of `sizes`, or a token either side of it, two times
    /// in three; any up to past the largest the third time.
    pub fn around(&mut self, sizes: &[usize]) -> usize {
        let largest = sizes.iter().max().copied().unwrap_or(0);
        if self.below(3) == 0 {
            return self.below(largest + 10);
        }
        let size = sizes[self.below(sizes.len())];
        (size + self.below(3)).saturating_sub(1)
    }
}
