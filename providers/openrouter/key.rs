//! The key to OpenRouter: where it comes from, and how it is held.
//!
//! The key is the environment's, or else the one kept from a sign-in, in
//! the user's keychain where the system has one. The environment's is
//! taken out of the environment when dwim starts, so that no command the
//! model runs is handed it. The key kept is read once, when a model is
//! first asked at OpenRouter: whether there is one is answered without
//! reading it, which the keychain may ask the user about.
//!
//! A key in memory is a [`Secret`], which does not print and is wiped
//! when dropped. That is not every copy there is of it: the response it
//! came in and the request it goes out in are not wiped.

use std::{
    env, fmt, io, ptr,
    sync::{
        Mutex, MutexGuard, Once,
        atomic::{Ordering, compiler_fence},
    },
};

use crate::sys::secrets::{self, Place};

/// The environment variable the key to OpenRouter is read from, ahead of
/// the key kept from a sign-in.
pub const KEY: &str = "OPENROUTER_API_KEY";

/// What the key is kept under, in the keychain or as a file's name.
const SERVICE: &str = "dwim";
const ACCOUNT: &str = "openrouter";

/// A secret in memory: text that is not printed, and is wiped when
/// dropped.
#[derive(Clone)]
pub struct Secret(String);

impl Secret {
    /// `text` as a secret. From the first one on, in a build for release,
    /// the user's other processes are kept from reading this one's
    /// memory, where the system can: a debugger no longer attaches to it.
    pub fn new(text: String) -> Self {
        static GUARD: Once = Once::new();
        if !cfg!(debug_assertions) {
            GUARD.call_once(|| {
                let _ = secrets::guard_memory();
            });
        }
        Self(text)
    }

    /// The secret itself, for where it is sent.
    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for Secret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Secret(…)")
    }
}

impl Drop for Secret {
    fn drop(&mut self) {
        // SAFETY: zeros are valid UTF-8, and each byte is the string's
        // own. The writes are volatile so that they are not dropped as
        // writes to memory about to be freed.
        for byte in unsafe { self.0.as_bytes_mut() } {
            unsafe { ptr::write_volatile(byte, 0) };
        }
        compiler_fence(Ordering::SeqCst);
    }
}

/// Takes the key out of the environment, to be held here: what dwim
/// starts from then on, the model's commands among it, does not inherit
/// it.
///
/// # Safety
///
/// The environment is changed, which no other thread may be reading or
/// changing meanwhile: this is called before any is started.
pub unsafe fn take_from_environment() {
    let key = env::var(KEY).ok();
    // SAFETY: no other thread is running, as the caller has it.
    unsafe { env::remove_var(KEY) };
    held().environment = key
        .filter(|key| !key.trim().is_empty())
        .map(|key| Secret::new(key.trim().to_string()));
}

/// Whether the key in use is the environment's.
pub fn from_environment() -> bool {
    environment().is_some()
}

/// Whether there is a key to OpenRouter, which the user is never asked
/// about.
pub fn has_key() -> bool {
    if environment().is_some() {
        return true;
    }
    match &held().kept {
        Some(kept) => kept.is_some(),
        None => secrets::exists(SERVICE, ACCOUNT),
    }
}

/// The key to OpenRouter, if there is one: the environment's, or else the
/// one kept from a sign-in, which is read once.
pub fn key() -> Option<Secret> {
    if let Some(key) = environment() {
        return Some(key);
    }
    if let Some(kept) = &held().kept {
        return kept.clone();
    }
    // Read with nothing held: the keychain may be asking the user, and
    // whether there is a key is asked meanwhile, as the shell draws.
    let kept = secrets::load(SERVICE, ACCOUNT).ok().flatten();
    held().kept.insert(kept.map(Secret::new)).clone()
}

/// Keeps `key` for the runs to come, and returns where.
pub fn keep(key: Secret) -> io::Result<Place> {
    let place = secrets::store(SERVICE, ACCOUNT, key.expose())?;
    held().kept = Some(Some(key));
    Ok(place)
}

/// The environment's key: the one taken out of it, or the one still in
/// it, in a program that took none.
fn environment() -> Option<Secret> {
    if let Some(key) = &held().environment {
        return Some(key.clone());
    }
    env::var(KEY)
        .ok()
        .filter(|key| !key.trim().is_empty())
        .map(|key| Secret::new(key.trim().to_string()))
}

/// The keys held by this process.
fn held() -> MutexGuard<'static, Held> {
    static HELD: Mutex<Held> = Mutex::new(Held {
        environment: None,
        kept: None,
    });
    HELD.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// The keys a process holds.
struct Held {
    /// The key taken out of the environment, if it had one.
    environment: Option<Secret>,
    /// The key kept from a sign-in, once it has been looked for.
    kept: Option<Option<Secret>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_secret_does_not_print() {
        let secret = Secret::new("sk-or-v1-0123".to_string());
        assert!(!format!("{secret:?}").contains("0123"));
        assert_eq!(secret.expose(), "sk-or-v1-0123");
    }
}
