//! A chat format: a conversation as the tokens a family of models is
//! trained to read it in, as every session with one speaks it, wherever
//! the model runs. A family's format writes the turns as its chat template
//! does, and reads a reply as it comes; the families' own are in their
//! directories, and what holds one holds a [`Format`], whichever it is.

use std::{
    collections::HashMap,
    hash::{DefaultHasher, Hash, Hasher},
    ops::ControlFlow,
};

use serde::{Deserialize, Serialize};

use crate::model::{Chunk, Result, Tokenizer};

/// A thought that writes the same [`REPEAT_WINDOW`] tokens for the
/// [`REPEATS`]th time is going round in circles, and is ended as one that
/// runs to its limit is: a small model at a long context sometimes falls
/// into saying the same thing over and over, and would not stop on its own
/// before the context ran out. A window this long is not written three
/// times by a thought that is getting anywhere.
pub const REPEAT_WINDOW: usize = 32;
pub const REPEATS: u32 = 3;

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

/// The chat format of a family of models, over the vocabulary of one of
/// them. The format's tags are special tokens, which it puts in by id;
/// what the user and the tools say is encoded as text, so that a message
/// which spells out a tag stays text.
pub trait Format: Send + Sync {
    /// The tokenizer of the vocabulary.
    fn tokenizer(&self) -> &Tokenizer;

    /// The system prompt as the turn that opens a conversation. It is the
    /// harness's to write, and may have the format's tags in it.
    fn system(&self, text: &str) -> Result<Vec<u32>>;

    /// A message from the user as a turn.
    fn user(&self, message: &str) -> Result<Vec<u32>>;

    /// The results of tool calls as a turn.
    fn results(&self, outputs: &[String]) -> Result<Vec<u32>>;

    /// A reply already made as a turn: its text, and its tool calls as
    /// written, without the thought behind it.
    fn reply(&self, text: &str, calls: &[String]) -> Result<Vec<u32>>;

    /// The opening of a reply, for the model to go on from: with the
    /// thought open, for it to think, or as the format opens a reply made
    /// without thinking.
    fn opening(&self, thinking: bool) -> Result<Vec<u32>>;

    /// What ends a reply the way the format expects, ready for the next
    /// message: what the reply was in the middle of closed first, if it
    /// is still `open`, as [`Reader::open`] says.
    fn end_of_reply(&self, open: bool) -> Result<Vec<u32>>;

    /// What ends a plain answer, made from the opening of a reply without
    /// thinking.
    fn end_of_answer(&self) -> Result<Vec<u32>> {
        self.end_of_reply(false)
    }

    /// What writes `closing` into a thought as the model's own and ends
    /// the thought there, for the reply to go on from.
    fn close_thought(&self, closing: &str) -> Result<Vec<u32>>;

    /// The tokens a reply that must write its answer out may not pick:
    /// the ones that end it, and what opens a call.
    fn unanswering(&self) -> Vec<u32>;

    /// The tokens a plain answer may not pick: what opens and closes a
    /// thought, and what opens a call.
    fn unplain(&self) -> Vec<u32>;

    /// Whether `token` ends a reply.
    fn ends(&self, token: u32) -> bool;

    /// Reads a reply from the opening the format gives it, with the
    /// thought open.
    fn reader(&self) -> Box<dyn Reader>;

    /// `turn` as tokens, as a conversation read over again has it.
    fn encode(&self, turn: &Turn) -> Result<Vec<u32>> {
        match turn {
            Turn::User(message) => self.user(message),
            Turn::Reply { text, calls, .. } => self.reply(text, calls),
            Turn::Results(outputs) => self.results(outputs),
        }
    }

    /// Number of tokens `turn` takes when fed.
    fn measure(&self, turn: &Turn) -> Result<usize> {
        Ok(self.encode(turn)?.len())
    }

    /// Most tokens a plain answer to `message` takes with its framing, if
    /// it is limited to `limit` tokens: what a plain answer needs room
    /// for. Fails only if the message cannot be encoded.
    fn answer_tokens(&self, message: &str, limit: usize) -> Result<usize> {
        Ok(self.user(message)?.len()
            + self.opening(false)?.len()
            + limit
            + self.end_of_answer()?.len())
    }
}

/// A reply as it is read, a token at a time: whether it is in its thought,
/// and what it has said and called so far.
pub trait Reader {
    /// Whether the reply is in its thought.
    fn thinking(&self) -> bool;

    /// Whether the reply is in the middle of something its end must close
    /// first: its thought, or in a format of messages, the one being
    /// written.
    fn open(&self) -> bool {
        self.thinking()
    }

    /// Tokens of the thought since it opened.
    fn thought(&self) -> usize;

    /// Whether the thought is going round in circles.
    fn repeating(&self) -> bool;

    /// Whether the reply has said anything, or called a tool.
    fn replied(&self) -> bool;

    /// Takes the token picked as the reply's next, showing `on_chunk`
    /// what it adds to the thought or to the text, and says what the
    /// token comes to.
    fn read(&mut self, token: u32, on_chunk: &mut dyn FnMut(Chunk) -> ControlFlow<()>) -> Read;

    /// Shows `on_chunk` what of the thought was held back, if anything
    /// was, as a thought that is about to be closed is no longer waiting
    /// to find out what it is.
    fn flush(&mut self, on_chunk: &mut dyn FnMut(Chunk) -> ControlFlow<()>) -> ControlFlow<()>;

    /// Takes the thought as closed, by what [`Format::close_thought`]
    /// wrote into it.
    fn close_thought(&mut self);

    /// The reply as it was read: its text as shown, and its calls as
    /// written.
    fn finish(self: Box<Self>) -> (String, Vec<String>);
}

/// What a token picked for a reply comes to.
pub enum Read {
    /// The reply goes on with the token, or with the one that takes its
    /// place, and whether whoever is shown the reply ended it there.
    Token(u32, ControlFlow<()>),
    /// Whoever is shown the reply ended it before the token.
    Interrupted,
    /// The token ends the reply.
    End,
    /// The token would end a reply that has said nothing and called
    /// nothing: the model answered only in its thought.
    Unanswered,
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
