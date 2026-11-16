//! Signing in to OpenRouter for a key, in the user's browser.
//!
//! It goes as OpenRouter's OAuth with PKCE does. A server on this machine
//! waits at a port of its own. The browser is sent to that server, which
//! sends it on to OpenRouter with a challenge: the hash of a secret made up
//! for this sign-in. Once the user agrees there, OpenRouter sends the
//! browser back to the server with a code, and the code and the secret
//! together are exchanged for a key. The address the browser is first sent
//! to is short enough to type, where it does not open by itself.

use std::{
    io::{self, Read, Write},
    net::{TcpListener, TcpStream},
    sync::atomic::{AtomicBool, Ordering},
    thread,
    time::Duration,
};

use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use ring::{
    digest,
    rand::{SecureRandom, SystemRandom},
};
use serde_json::{Value, json};
use ureq::Agent;

use crate::model::Result;

use super::Secret;

/// Where the user agrees to a key being made.
const AUTH: &str = "https://openrouter.ai/auth";

/// Where a code is exchanged for a key.
const KEYS: &str = "https://openrouter.ai/api/v1/auth/keys";

/// How often the server looks for the browser, and whether the sign-in
/// was called off.
const POLL: Duration = Duration::from_millis(50);

/// Longest the server waits for a browser that connected to say what it
/// wants.
const READ: Duration = Duration::from_secs(2);

/// Most bytes of a request the server reads.
const REQUEST: usize = 8192;

/// A sign-in under way: the server the browser comes back to, and the
/// secrets of this sign-in.
pub struct Login {
    listener: TcpListener,
    port: u16,
    /// The secret whose hash OpenRouter is given first, and which the code
    /// is exchanged with.
    verifier: String,
    /// What OpenRouter is to send back with the code, so that a code from
    /// anywhere else is not taken.
    state: String,
}

impl Login {
    /// Starts a sign-in: the server listens, and the browser is yet to be
    /// sent to [`Login::url`].
    pub fn start() -> Result<Self> {
        let listener = TcpListener::bind(("127.0.0.1", 0))?;
        listener.set_nonblocking(true)?;
        let port = listener.local_addr()?.port();
        Ok(Self {
            listener,
            port,
            verifier: secret(32)?,
            state: secret(16)?,
        })
    }

    /// The address to open in the browser, which sends it on to
    /// OpenRouter.
    pub fn url(&self) -> String {
        format!("http://localhost:{}", self.port)
    }

    /// Waits for the user to agree in the browser, and returns the key
    /// OpenRouter made; or nothing, once `cancelled` is set.
    pub fn wait(self, cancelled: &AtomicBool) -> Result<Option<Secret>> {
        match self.code(cancelled)? {
            Some(code) => exchange(&code, &self.verifier).map(Some),
            None => Ok(None),
        }
    }

    /// Serves the browser until it comes back from OpenRouter with a code,
    /// which is returned; or nothing, once `cancelled` is set.
    fn code(&self, cancelled: &AtomicBool) -> Result<Option<String>> {
        while !cancelled.load(Ordering::Relaxed) {
            match self.listener.accept() {
                Ok((stream, _)) => {
                    if let Some(code) = self.answer(stream) {
                        return Ok(Some(code));
                    }
                }
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => thread::sleep(POLL),
                Err(e) => return Err(e.into()),
            }
        }
        Ok(None)
    }

    /// Answers a request of the browser's: sends it on to OpenRouter from
    /// the front page, and takes the code it comes back with, which is
    /// returned if it came with this sign-in's state.
    fn answer(&self, mut stream: TcpStream) -> Option<String> {
        stream.set_nonblocking(false).ok()?;
        stream.set_read_timeout(Some(READ)).ok()?;
        let target = target(&mut stream)?;
        let (path, query) = target.split_once('?').unwrap_or((&target, ""));
        let value = |name: &str| {
            query
                .split('&')
                .filter_map(|pair| pair.split_once('='))
                .find(|(key, _)| *key == name)
                .map(|(_, value)| decode(value))
        };
        match path {
            "/" => {
                respond(&mut stream, "302 Found", Some(&self.auth_url()), "");
                None
            }
            "/callback" => match value("code") {
                Some(code) if value("state").as_deref() == Some(&self.state) => {
                    respond(
                        &mut stream,
                        "200 OK",
                        None,
                        &page("You can close this tab and go back to dwim."),
                    );
                    Some(code)
                }
                _ => {
                    respond(
                        &mut stream,
                        "400 Bad Request",
                        None,
                        &page("This is not the sign-in dwim is waiting for."),
                    );
                    None
                }
            },
            _ => {
                respond(&mut stream, "404 Not Found", None, "");
                None
            }
        }
    }

    /// Where the browser is sent at OpenRouter: to agree to a key, and
    /// come back to the server with a code.
    fn auth_url(&self) -> String {
        format!(
            "{AUTH}?callback_url={}&code_challenge={}&code_challenge_method=S256&state={}&key_label=dwim",
            encode(&format!("{}/callback", self.url())),
            challenge(&self.verifier),
            self.state
        )
    }
}

/// Exchanges `code` for a key, with the secret whose hash OpenRouter was
/// given.
fn exchange(code: &str, verifier: &str) -> Result<Secret> {
    let agent = Agent::config_builder()
        .http_status_as_error(false)
        .timeout_global(Some(Duration::from_secs(30)))
        .build()
        .new_agent();
    let body = json!({
        "code": code,
        "code_verifier": verifier,
        "code_challenge_method": "S256",
    });
    let response = agent
        .post(KEYS)
        .header("Content-Type", "application/json")
        .send(&body.to_string())?;
    let status = response.status();
    let text = response.into_body().read_to_string()?;
    let answer: Value = serde_json::from_str(&text).unwrap_or_default();
    match answer["key"].as_str() {
        Some(key) => Ok(Secret::new(key.trim().to_string())),
        None => {
            let why = answer["error"]["message"].as_str().unwrap_or(text.trim());
            Err(format!("OpenRouter gave no key ({status}): {why}").into())
        }
    }
}

/// The challenge OpenRouter is given for `verifier`: its SHA-256, in
/// base64 as a URL takes it.
fn challenge(verifier: &str) -> String {
    URL_SAFE_NO_PAD.encode(digest::digest(&digest::SHA256, verifier.as_bytes()))
}

/// A secret of `bytes` random bytes, in base64 as a URL takes it.
fn secret(bytes: usize) -> Result<String> {
    let mut secret = vec![0; bytes];
    SystemRandom::new()
        .fill(&mut secret)
        .map_err(|_| "the system gave no random bytes")?;
    Ok(URL_SAFE_NO_PAD.encode(secret))
}

/// What a request on `stream` asks for: the path, with its query.
fn target(stream: &mut TcpStream) -> Option<String> {
    let mut request = Vec::new();
    let mut buffer = [0; 1024];
    while !request.windows(2).any(|pair| pair == b"\r\n") && request.len() < REQUEST {
        match stream.read(&mut buffer) {
            Ok(0) | Err(_) => break,
            Ok(n) => request.extend_from_slice(&buffer[..n]),
        }
    }
    let request = String::from_utf8_lossy(&request);
    let mut words = request.lines().next()?.split_whitespace();
    (words.next()? == "GET").then(|| words.next().map(str::to_string))?
}

/// Answers the request on `stream` with `status`, sending the browser on
/// to `location` if there is one, and with `body` as the page.
fn respond(stream: &mut TcpStream, status: &str, location: Option<&str>, body: &str) {
    let location = location.map_or(String::new(), |url| format!("Location: {url}\r\n"));
    let _ = write!(
        stream,
        "HTTP/1.1 {status}\r\n{location}Content-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let _ = stream.flush();
}

/// A page that says `text`.
fn page(text: &str) -> String {
    format!(
        "<!doctype html><meta charset=\"utf-8\"><title>dwim</title>\
         <body style=\"font: 16px system-ui; margin: 20vh auto; max-width: 30em; text-align: center\">\
         <h1>dwim</h1><p>{text}</p></body>"
    )
}

/// `text` as it goes in a URL's query: everything but letters, digits and
/// `-._~` as its bytes in hex.
fn encode(text: &str) -> String {
    let mut encoded = String::new();
    for byte in text.bytes() {
        if byte.is_ascii_alphanumeric() || b"-._~".contains(&byte) {
            encoded.push(byte as char);
        } else {
            encoded.push_str(&format!("%{byte:02X}"));
        }
    }
    encoded
}

/// `text` out of a URL's query, where a byte may be written in hex.
fn decode(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut decoded = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let hex = bytes
            .get(i + 1..i + 3)
            .and_then(|hex| std::str::from_utf8(hex).ok())
            .and_then(|hex| u8::from_str_radix(hex, 16).ok());
        match (bytes[i], hex) {
            (b'%', Some(byte)) => {
                decoded.push(byte);
                i += 3;
            }
            (b'+', _) => {
                decoded.push(b' ');
                i += 1;
            }
            (byte, _) => {
                decoded.push(byte);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&decoded).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What the sign-in's server answers a request for `target` with.
    fn get(login: &Login, target: &str) -> String {
        let mut stream = TcpStream::connect(("127.0.0.1", login.port)).unwrap();
        write!(stream, "GET {target} HTTP/1.1\r\nHost: localhost\r\n\r\n").unwrap();
        let mut answer = String::new();
        stream.read_to_string(&mut answer).unwrap();
        answer
    }

    #[test]
    fn takes_the_code_the_browser_comes_back_with() {
        let login = Login::start().unwrap();
        let cancelled = AtomicBool::new(false);
        thread::scope(|scope| {
            let code = scope.spawn(|| login.code(&cancelled).unwrap());
            // A code from somewhere else, without this sign-in's state, is
            // not taken, and the server goes on waiting.
            assert!(get(&login, "/callback?code=theirs&state=made-up").starts_with("HTTP/1.1 400"));
            let target = format!("/callback?code=a%2Fb&state={}", login.state);
            assert!(get(&login, &target).starts_with("HTTP/1.1 200"));
            assert_eq!(code.join().unwrap().as_deref(), Some("a/b"));
        });
    }

    #[test]
    fn sends_the_browser_on_to_openrouter_and_back() {
        let login = Login::start().unwrap();
        let cancelled = AtomicBool::new(false);
        thread::scope(|scope| {
            let code = scope.spawn(|| login.code(&cancelled).unwrap());
            let answer = get(&login, "/");
            let back = format!(
                "callback_url=http%3A%2F%2Flocalhost%3A{}%2Fcallback",
                login.port
            );
            assert!(answer.starts_with("HTTP/1.1 302"));
            assert!(answer.contains("Location: https://openrouter.ai/auth?"));
            assert!(answer.contains(&back), "{answer}");
            cancelled.store(true, Ordering::Relaxed);
            assert_eq!(code.join().unwrap(), None);
        });
    }

    #[test]
    fn challenges_with_the_hash_of_the_secret() {
        // The SHA-256 of the secret as Python's hashlib has it.
        assert_eq!(
            challenge("dBjftJeZ4CVP-mBKkjh1eEcaWmGl1ZtfN4GVwpz01E0"),
            "2Bc6cgbURaiBQ4hI_cNSL6WTaBXnjU548CxQ_jku6UI"
        );
    }
}
