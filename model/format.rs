//! The chat format Qwen models are trained on, ChatML, as every session
//! with one speaks it, wherever the model runs: its tags, the turns of a
//! conversation as the chat template writes them, and what reading a reply
//! as it comes takes.

use std::{
    collections::HashMap,
    hash::{DefaultHasher, Hash, Hasher},
    ops::ControlFlow,
    sync::Arc,
};

use serde::{Deserialize, Serialize};

use crate::model::{Chunk, Result, Tokenizer};

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

    /// What writes `closing` into a thought as the model's own and ends
    /// the thought there, for the reply to go on from.
    pub fn close_thought(&self, closing: &str) -> Result<Vec<u32>> {
        let mut tokens = self.tokenizer.encode(closing)?;
        tokens.push(self.think_end);
        tokens.extend(self.tokenizer.encode("\n\n")?);
        Ok(tokens)
    }

    /// The tokens a reply that must write its answer out may not pick:
    /// the ones that end it, and the one that opens a call.
    pub fn unanswering(&self) -> [u32; 3] {
        [self.im_end, self.end_of_text, self.tool_call]
    }

    /// Reads a reply from the opening the format gives it, with the
    /// thought open.
    pub fn reader(&self) -> Reader {
        Reader {
            format: self.clone(),
            text: Utf8Stream::default(),
            thinking: true,
            thought: 0,
            repeats: Repeats::default(),
            held: None,
            call: None,
            replied: false,
            shown: String::new(),
            calls: Vec::new(),
        }
    }
}

/// A reply as it is read, a token at a time: whether it is in its thought,
/// and what it has said and called so far.
///
/// Qwen models call tools from inside the thought as well as after it: a
/// `<tool_call>` tag followed by a `<function=…>` block ends the thought
/// and opens the call, wherever it comes, as the model's makers read a
/// reply. A `<tool_call>` tag followed by anything else, as one the model
/// only mentions, is part of the thought. Left in the thought, a call the
/// model meant gets no result, and the model goes on thinking about what
/// it has asked for: a thought that never ends.
pub struct Reader {
    format: Format,
    text: Utf8Stream,
    thinking: bool,
    /// Tokens of the thought since it opened, and whether it is going
    /// round in circles.
    thought: usize,
    repeats: Repeats,
    /// What has followed a `<tool_call>` tag written in the thought, while
    /// it is not yet known whether the tag opens a call.
    held: Option<String>,
    /// The call being written, as far as it has got.
    call: Option<String>,
    replied: bool,
    /// The reply's text as shown, and the calls it has made, as written.
    shown: String,
    calls: Vec<String>,
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

impl Reader {
    /// Whether the reply is in its thought.
    pub fn thinking(&self) -> bool {
        self.thinking
    }

    /// Tokens of the thought since it opened.
    pub fn thought(&self) -> usize {
        self.thought
    }

    /// Whether the thought is going round in circles.
    pub fn repeating(&self) -> bool {
        self.repeats.repeating
    }

    /// Whether the reply has said anything, or called a tool.
    pub fn replied(&self) -> bool {
        self.replied
    }

    /// Takes the token picked as the reply's next, showing `on_chunk`
    /// what it adds to the thought or to the text, and says what the
    /// token comes to.
    pub fn read(
        &mut self,
        mut token: u32,
        on_chunk: &mut dyn FnMut(Chunk) -> ControlFlow<()>,
    ) -> Read {
        let format = &self.format;
        let special = [
            format.im_end,
            format.end_of_text,
            format.think,
            format.think_end,
            format.tool_call,
            format.tool_call_end,
        ]
        .contains(&token);
        // A <tool_call> written in the thought is held until what
        // follows it says what it is. The opening of a <function=…>
        // block makes it a call, and the thought ends there, as the
        // model's makers read it: the model calls tools from inside
        // the thought, without closing it first. Anything else, as a
        // tag the model only mentions, is part of the thought, shown
        // as written.
        let mut consumed = false;
        if let Some(pending) = &mut self.held {
            let settled = if special {
                self.held.take()
            } else {
                consumed = true;
                pending.push_str(&self.text.push(format.tokenizer.decode(token)));
                if pending.starts_with(CALL_OPENING) {
                    self.thinking = false;
                    self.replied = true;
                    self.call = self.held.take();
                    None
                } else if CALL_OPENING.starts_with(pending.as_str()) {
                    None
                } else {
                    self.held.take()
                }
            };
            if let Some(pending) = settled
                && on_chunk(Chunk::Thought(&format!("<tool_call>{pending}"))).is_break()
            {
                return Read::Interrupted;
            }
        }
        if !consumed && (token == format.im_end || token == format.end_of_text) {
            if self.thinking {
                // The model sometimes ends its reply while still
                // thinking, which leaves nothing to show for it: close
                // the thought instead, so that it goes on to reply.
                token = format.think_end;
            } else if self.replied {
                return Read::End;
            } else {
                return Read::Unanswered;
            }
        }
        if self.thinking {
            self.thought += 1;
            self.repeats.push(token);
        }
        let flow = if consumed {
            ControlFlow::Continue(())
        } else if token == format.think {
            self.thinking = true;
            self.thought = 0;
            self.repeats = Repeats::default();
            ControlFlow::Continue(())
        } else if token == format.think_end {
            self.thinking = false;
            ControlFlow::Continue(())
        } else if token == format.tool_call && self.thinking {
            self.held = Some(String::new());
            ControlFlow::Continue(())
        } else if token == format.tool_call_end && self.thinking {
            let chunk = self.text.push(format.tokenizer.decode(token));
            on_chunk(Chunk::Thought(&chunk))
        } else if token == format.tool_call {
            self.replied = true;
            self.call = Some(String::new());
            ControlFlow::Continue(())
        } else if token == format.tool_call_end {
            self.calls.extend(self.call.take());
            ControlFlow::Continue(())
        } else {
            let chunk = self.text.push(format.tokenizer.decode(token));
            match &mut self.call {
                Some(call) => {
                    call.push_str(&chunk);
                    ControlFlow::Continue(())
                }
                None if chunk.is_empty() => ControlFlow::Continue(()),
                None if self.thinking => on_chunk(Chunk::Thought(&chunk)),
                None => {
                    self.replied |= !chunk.trim().is_empty();
                    self.shown.push_str(&chunk);
                    on_chunk(Chunk::Text(&chunk))
                }
            }
        };
        Read::Token(token, flow)
    }

    /// Shows `on_chunk` what of the thought was held back, if anything
    /// was, as a thought that is about to be closed is no longer waiting
    /// to find out what it is.
    pub fn flush(&mut self, on_chunk: &mut dyn FnMut(Chunk) -> ControlFlow<()>) -> ControlFlow<()> {
        match self.held.take() {
            Some(pending) => on_chunk(Chunk::Thought(&format!("<tool_call>{pending}"))),
            None => ControlFlow::Continue(()),
        }
    }

    /// Takes the thought as closed, by what [`Format::close_thought`]
    /// wrote into it.
    pub fn close_thought(&mut self) {
        self.thinking = false;
    }

    /// The reply as it was read: its text as shown, and its calls as
    /// written.
    pub fn finish(self) -> (String, Vec<String>) {
        (self.shown, self.calls)
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
