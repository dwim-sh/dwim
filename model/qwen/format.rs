//! ChatML, the chat format Qwen models are trained on, with their tags for
//! thoughts and tool calls:
//!
//! ```text
//! <|im_start|>user
//! Hello!<|im_end|>
//! <|im_start|>assistant
//! <think>
//! The user is greeting me.
//! </think>
//!
//! Hi there!<|im_end|>
//! ```
//!
//! The model thinks inside `<think>` tags before it replies: its chat
//! template opens the tag for it. It may also call tools, if the system
//! prompt describes them, by replying with `<tool_call>` blocks. Their
//! results go back to it in a user turn of `<tool_response>` blocks.

use std::{ops::ControlFlow, sync::Arc};

use crate::model::{
    Chunk, Format, Result, Tokenizer,
    format::{Read, Reader, Repeats, Utf8Stream},
};

/// What follows a `<tool_call>` tag that opens a call, as the chat template
/// writes one: a tag followed by anything else is text.
const CALL_OPENING: &str = "\n<function=";

/// The tags of the chat format, which are special tokens of the model's
/// vocabulary.
const IM_START: &str = "<|im_start|>";
const IM_END: &str = "<|im_end|>";
const END_OF_TEXT: &str = "<|endoftext|>";
const THINK: &str = "<think>";
const THINK_END: &str = "</think>";
const TOOL_CALL: &str = "<tool_call>";
const TOOL_CALL_END: &str = "</tool_call>";
const TOOL_RESPONSE: &str = "<tool_response>";
const TOOL_RESPONSE_END: &str = "</tool_response>";

/// The chat format's tags, as a vocabulary for it has them: the ones
/// starting with `<|` are control tokens, the rest user-defined.
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

/// ChatML as a Qwen model's vocabulary has it: the tokenizer, and the ids
/// of the format's tags, which are special tokens of it. It encodes the
/// turns of a conversation as the chat template writes them. Cloning one
/// shares the tokenizer.
#[derive(Clone)]
pub struct ChatMl {
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

impl ChatMl {
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

    /// Encodes a turn of the conversation around its content's tokens.
    fn turn(&self, role: &str, content: Vec<u32>) -> Result<Vec<u32>> {
        let mut turn = vec![self.im_start];
        turn.extend(self.tokenizer.encode(&format!("{role}\n"))?);
        turn.extend(content);
        turn.push(self.im_end);
        turn.extend(self.tokenizer.encode("\n")?);
        Ok(turn)
    }

    /// An empty thought, opened and closed, as the template writes one
    /// before a reply made without thinking, and before a reply written
    /// over.
    fn closed_thought(&self) -> Result<Vec<u32>> {
        let mut tokens = vec![self.think];
        tokens.extend(self.tokenizer.encode("\n\n")?);
        tokens.push(self.think_end);
        Ok(tokens)
    }
}

impl Format for ChatMl {
    fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    fn system(&self, text: &str) -> Result<Vec<u32>> {
        let content = self.tokenizer.encode_with_special(text)?;
        self.turn("system", content)
    }

    fn user(&self, message: &str) -> Result<Vec<u32>> {
        let content = self.tokenizer.encode(message)?;
        self.turn("user", content)
    }

    /// A `<tool_response>` block for each result, in a user turn.
    fn results(&self, outputs: &[String]) -> Result<Vec<u32>> {
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

    /// An empty thought, the reply's text, and its tool calls as written,
    /// each in a `<tool_call>` block. A reply is written without its
    /// thought wherever it comes: the model on this machine reads a turn
    /// once, and to leave a thought out after the user's next message
    /// would have it read the turn over. The newlines the model puts
    /// between the thought and the text are the template's to write, so
    /// the text's leading newlines are left out.
    fn reply(&self, text: &str, calls: &[String]) -> Result<Vec<u32>> {
        let mut content = self.closed_thought()?;
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

    /// The template's opening of a reply: with the thought open, or with
    /// an empty thought closed at once, as the template opens a reply made
    /// without thinking.
    fn opening(&self, thinking: bool) -> Result<Vec<u32>> {
        let mut tokens = vec![self.im_start];
        tokens.extend(self.tokenizer.encode("assistant\n")?);
        if thinking {
            tokens.push(self.think);
            tokens.extend(self.tokenizer.encode("\n")?);
        } else {
            tokens.extend(self.closed_thought()?);
            tokens.extend(self.tokenizer.encode("\n\n")?);
        }
        Ok(tokens)
    }

    fn end_of_reply(&self, open: bool) -> Result<Vec<u32>> {
        let mut end = Vec::new();
        if open {
            end.extend(self.tokenizer.encode("\n")?);
            end.push(self.think_end);
            end.extend(self.tokenizer.encode("\n\n")?);
        }
        end.push(self.im_end);
        end.extend(self.tokenizer.encode("\n")?);
        Ok(end)
    }

    fn close_thought(&self, closing: &str) -> Result<Vec<u32>> {
        let mut tokens = self.tokenizer.encode(closing)?;
        tokens.push(self.think_end);
        tokens.extend(self.tokenizer.encode("\n\n")?);
        Ok(tokens)
    }

    fn unanswering(&self) -> Vec<u32> {
        vec![self.im_end, self.end_of_text, self.tool_call]
    }

    fn unplain(&self) -> Vec<u32> {
        vec![self.think, self.think_end, self.tool_call]
    }

    fn ends(&self, token: u32) -> bool {
        token == self.im_end || token == self.end_of_text
    }

    fn reader(&self) -> Box<dyn Reader> {
        Box::new(Reading {
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
        })
    }
}

/// A reply being read, a token at a time.
///
/// Qwen models call tools from inside the thought as well as after it: a
/// `<tool_call>` tag followed by a `<function=…>` block ends the thought
/// and opens the call, wherever it comes, as the model's makers read a
/// reply. A `<tool_call>` tag followed by anything else, as one the model
/// only mentions, is part of the thought. Left in the thought, a call the
/// model meant gets no result, and the model goes on thinking about what
/// it has asked for: a thought that never ends.
struct Reading {
    format: ChatMl,
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

impl Reader for Reading {
    fn thinking(&self) -> bool {
        self.thinking
    }

    fn thought(&self) -> usize {
        self.thought
    }

    fn repeating(&self) -> bool {
        self.repeats.repeating
    }

    fn replied(&self) -> bool {
        self.replied
    }

    fn read(&mut self, mut token: u32, on_chunk: &mut dyn FnMut(Chunk) -> ControlFlow<()>) -> Read {
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

    fn flush(&mut self, on_chunk: &mut dyn FnMut(Chunk) -> ControlFlow<()>) -> ControlFlow<()> {
        match self.held.take() {
            Some(pending) => on_chunk(Chunk::Thought(&format!("<tool_call>{pending}"))),
            None => ControlFlow::Continue(()),
        }
    }

    fn close_thought(&mut self) {
        self.thinking = false;
    }

    fn finish(self: Box<Self>) -> (String, Vec<String>) {
        (self.shown, self.calls)
    }
}
