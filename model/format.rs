//! The chat format Qwen models are trained on, ChatML, as every session
//! with one speaks it, wherever the model runs: its tags, the turns of a
//! conversation as the chat template writes them, and what reading a reply
//! as it comes takes.

use std::{
    collections::HashMap,
    hash::{DefaultHasher, Hash, Hasher},
    sync::Arc,
};

use serde::{Deserialize, Serialize};

use crate::model::{Result, Tokenizer};

/// What follows a `<tool_call>` tag that opens a call, as the chat template
/// writes one: a tag followed by anything else is text.
pub const CALL_OPENING: &str = "\n<function=";

/// A thought that writes the same [`REPEAT_WINDOW`] tokens for the
/// [`REPEATS`]th time is going round in circles, and is ended as one that
/// runs to its limit is: a small model at a long context sometimes falls
/// into saying the same thing over and over, and would not stop on its own
/// before the context ran out. A window this long is not written three
/// times by a thought that is getting anywhere.
pub const REPEAT_WINDOW: usize = 32;
pub const REPEATS: u32 = 3;

/// The tags of the chat format, which are special tokens of the model's
/// vocabulary.
pub const IM_START: &str = "<|im_start|>";
pub const IM_END: &str = "<|im_end|>";
pub const END_OF_TEXT: &str = "<|endoftext|>";
pub const THINK: &str = "<think>";
pub const THINK_END: &str = "</think>";
pub const TOOL_CALL: &str = "<tool_call>";
pub const TOOL_CALL_END: &str = "</tool_call>";
pub const TOOL_RESPONSE: &str = "<tool_response>";
pub const TOOL_RESPONSE_END: &str = "</tool_response>";

/// The chat format's tags, as a vocabulary for it has them: the ones
/// starting with `<|` are control tokens, the rest user-defined.
#[cfg(test)]
pub const SPECIAL_TOKENS: [&str; 9] = [
    END_OF_TEXT,
    IM_START,
    IM_END,
    TOOL_CALL,
    TOOL_CALL_END,
    TOOL_RESPONSE,
    TOOL_RESPONSE_END,
    THINK,
    THINK_END,
];

/// A turn of a conversation, as a transcript of it keeps them: what the
/// user said; what the model replied, as the text it showed and the tool
/// calls it made as written; and what the tools returned.
///
/// A reply has the thought behind it where the provider that made it gave
/// one to keep, and is empty otherwise. The chat template writes the
/// thoughts of the replies since the user's last message and leaves out
/// the ones before: a model that calls a tool goes on from what it had
/// worked out, and one that is not shown its thought works it out again
/// after every call.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub enum Turn {
    User(String),
    Reply {
        text: String,
        calls: Vec<String>,
        #[serde(default, skip_serializing_if = "String::is_empty")]
        thought: String,
    },
    Results(Vec<String>),
}

/// The template's opening of a reply, for the model to go on from: with
/// the thought open, for it to think, or with an empty thought closed at
/// once, as the template opens a reply made without thinking.
pub fn opening(tokenizer: &Tokenizer, thinking: bool) -> Result<Vec<u32>> {
    let mut tokens = vec![tokenizer.special(IM_START)?];
    tokens.extend(tokenizer.encode("assistant\n")?);
    if thinking {
        tokens.push(tokenizer.special(THINK)?);
        tokens.extend(tokenizer.encode("\n")?);
    } else {
        tokens.extend(closed_thought(tokenizer)?);
        tokens.extend(tokenizer.encode("\n\n")?);
    }
    Ok(tokens)
}

/// An empty thought, opened and closed, as the template writes one before
/// a reply made without thinking, and before a reply written over.
pub fn closed_thought(tokenizer: &Tokenizer) -> Result<Vec<u32>> {
    let mut tokens = vec![tokenizer.special(THINK)?];
    tokens.extend(tokenizer.encode("\n\n")?);
    tokens.push(tokenizer.special(THINK_END)?);
    Ok(tokens)
}

/// The chat format as the model's vocabulary has it: the tokenizer, and
/// the ids of the format's tags, which are special tokens of it. It
/// encodes the turns of a conversation as the chat template writes them.
/// Cloning one shares the tokenizer.
#[derive(Clone)]
pub struct Format {
    pub tokenizer: Arc<Tokenizer>,
    pub im_start: u32,
    pub im_end: u32,
    pub end_of_text: u32,
    pub think: u32,
    pub think_end: u32,
    pub tool_call: u32,
    pub tool_call_end: u32,
    pub tool_response: u32,
    pub tool_response_end: u32,
}

impl Format {
    /// The format over `tokenizer`'s vocabulary, which must have the tags.
    pub fn new(tokenizer: Tokenizer) -> Result<Self> {
        Ok(Self {
            im_start: tokenizer.special(IM_START)?,
            im_end: tokenizer.special(IM_END)?,
            end_of_text: tokenizer.special(END_OF_TEXT)?,
            think: tokenizer.special(THINK)?,
            think_end: tokenizer.special(THINK_END)?,
            tool_call: tokenizer.special(TOOL_CALL)?,
            tool_call_end: tokenizer.special(TOOL_CALL_END)?,
            tool_response: tokenizer.special(TOOL_RESPONSE)?,
            tool_response_end: tokenizer.special(TOOL_RESPONSE_END)?,
            tokenizer: Arc::new(tokenizer),
        })
    }

    /// Number of tokens `turn` takes when fed, as `replay` writes it.
    pub fn measure(&self, turn: &Turn) -> Result<usize> {
        Ok(self.encode(turn)?.len())
    }

    /// Most tokens a plain answer to `message` takes with its framing, if
    /// it is limited to `limit` tokens: what a plain answer needs room
    /// for. Fails only if the message cannot be encoded.
    pub fn answer_tokens(&self, message: &str, limit: usize) -> Result<usize> {
        Ok(self.user(message)?.len()
            + opening(&self.tokenizer, false)?.len()
            + limit
            + self.end_of_reply(false)?.len())
    }

    /// `turn` as tokens, as `replay` writes it.
    pub fn encode(&self, turn: &Turn) -> Result<Vec<u32>> {
        match turn {
            Turn::User(message) => self.user(message),
            Turn::Reply { text, calls, .. } => self.reply(text, calls),
            Turn::Results(outputs) => self.results(outputs),
        }
    }

    /// A message from the user as a turn.
    pub fn user(&self, message: &str) -> Result<Vec<u32>> {
        let content = self.tokenizer.encode(message)?;
        self.turn("user", content)
    }

    /// The results of tool calls as a turn: a `<tool_response>` block each.
    pub fn results(&self, outputs: &[String]) -> Result<Vec<u32>> {
        let mut content = Vec::new();
        for (i, output) in outputs.iter().enumerate() {
            if i > 0 {
                content.extend(self.tokenizer.encode("\n")?);
            }
            content.push(self.tool_response);
            content.extend(self.tokenizer.encode(&format!("\n{output}\n"))?);
            content.push(self.tool_response_end);
        }
        self.turn("user", content)
    }

    /// A reply already made as a turn: an empty thought, its text, and its
    /// tool calls as written, each in a `<tool_call>` block. A reply is
    /// written without its thought wherever it comes: the model on this
    /// machine reads a turn once, and to leave a thought out after the
    /// user's next message would have it read the turn over. The newlines
    /// the model puts between the thought and the text are the template's
    /// to write, so the text's leading newlines are left out.
    pub fn reply(&self, text: &str, calls: &[String]) -> Result<Vec<u32>> {
        let mut content = closed_thought(&self.tokenizer)?;
        let text = text.trim_start_matches('\n');
        content.extend(self.tokenizer.encode(&format!("\n\n{text}"))?);
        for (i, call) in calls.iter().enumerate() {
            if i > 0 {
                content.extend(self.tokenizer.encode("\n")?);
            }
            content.push(self.tool_call);
            content.extend(self.tokenizer.encode(call)?);
            content.push(self.tool_call_end);
        }
        self.turn("assistant", content)
    }

    /// Encodes a turn of the conversation around its content's tokens.
    pub fn turn(&self, role: &str, content: Vec<u32>) -> Result<Vec<u32>> {
        let mut turn = vec![self.im_start];
        turn.extend(self.tokenizer.encode(&format!("{role}\n"))?);
        turn.extend(content);
        turn.push(self.im_end);
        turn.extend(self.tokenizer.encode("\n")?);
        Ok(turn)
    }

    /// What ends a reply the way the chat format expects, ready for the
    /// next message: the thought closed first if it is still open.
    pub fn end_of_reply(&self, thinking: bool) -> Result<Vec<u32>> {
        let mut end = Vec::new();
        if thinking {
            end.extend(self.tokenizer.encode("\n")?);
            end.push(self.think_end);
            end.extend(self.tokenizer.encode("\n\n")?);
        }
        end.push(self.im_end);
        end.extend(self.tokenizer.encode("\n")?);
        Ok(end)
    }
}

/// Whether a thought is going round in circles: its tokens so far, how
/// often each window of [`REPEAT_WINDOW`] of them has been written, and
/// whether one has been written [`REPEATS`] times.
#[derive(Default)]
pub struct Repeats {
    recent: Vec<u32>,
    windows: HashMap<u64, u32>,
    pub repeating: bool,
}

impl Repeats {
    /// Takes the thought's next token.
    pub fn push(&mut self, token: u32) {
        self.recent.push(token);
        if self.recent.len() >= REPEAT_WINDOW {
            let mut hasher = DefaultHasher::new();
            self.recent[self.recent.len() - REPEAT_WINDOW..].hash(&mut hasher);
            let count = self.windows.entry(hasher.finish()).or_insert(0);
            *count += 1;
            self.repeating |= *count >= REPEATS;
        }
    }
}

/// Turns a stream of bytes into text, holding back a UTF-8 character split
/// across tokens until the rest of it arrives.
#[derive(Default)]
pub struct Utf8Stream {
    pending: Vec<u8>,
}

impl Utf8Stream {
    pub fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        let complete = match std::str::from_utf8(&self.pending) {
            Ok(_) => self.pending.len(),
            // An incomplete character at the end: wait for the rest.
            Err(e) if e.error_len().is_none() => e.valid_up_to(),
            Err(_) => self.pending.len(),
        };
        let text = String::from_utf8_lossy(&self.pending[..complete]).into_owned();
        self.pending.drain(..complete);
        text
    }
}
