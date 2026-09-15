use std::ops::ControlFlow;

use serde::Deserialize;
use serde_json::Value;

use crate::{Device, Result, Sampler, Tokenizer, qwen3::{BATCH, Model, State}};

/// A conversation with an instruction-tuned model, in the ChatML format that
/// Qwen models are trained on:
///
/// ```text
/// <|im_start|>user
/// Hello!<|im_end|>
/// <|im_start|>assistant
/// <think>
/// The user is greeting me.
/// </think>
///
/// Hi there!<|im_end|>
/// ```
///
/// The model thinks inside `<think>` tags before it replies, and the thought
/// is streamed apart from the reply. It may also call tools, if the system
/// prompt describes them, by replying with `<tool_call>` blocks. Their
/// results go back to it in a user turn of `<tool_response>` blocks. `Chat`
/// only speaks the format: what the tools are and running them is up to the
/// caller.
///
/// The whole conversation stays in the key/value cache, so each turn only
/// runs the model over its new tokens.
pub struct Chat<D: Device> {
    model: Model<D>,
    state: State<D>,
    tokenizer: Tokenizer,
    sampler: Sampler,
    /// Number of tokens in the conversation so far.
    len: usize,
    im_start: u32,
    im_end: u32,
    end_of_text: u32,
    think: u32,
    think_end: u32,
    tool_call: u32,
    tool_call_end: u32,
}

/// A piece of the model's reply, as it is generated.
pub enum Chunk<'a> {
    /// Part of the model's thought, before it replies.
    Thought(&'a str),
    /// Part of the reply itself.
    Text(&'a str),
}

/// A call the model made to a tool, as written in a `<tool_call>` block.
#[derive(Debug, Deserialize)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Value,
}

impl ToolCall {
    /// Parses the body of a `<tool_call>` block.
    pub fn parse(text: &str) -> Result<Self> {
        Ok(serde_json::from_str(text.trim())?)
    }
}

impl<D: Device> Chat<D> {
    /// Starts a conversation with room for `max_len` tokens.
    pub fn new(model: Model<D>, tokenizer: Tokenizer, sampler: Sampler, max_len: usize) -> Result<Self> {
        Ok(Self {
            state: State::new(&model, max_len),
            model,
            im_start: tokenizer.special("<|im_start|>")?,
            im_end: tokenizer.special("<|im_end|>")?,
            end_of_text: tokenizer.special("<|endoftext|>")?,
            think: tokenizer.special("<think>")?,
            think_end: tokenizer.special("</think>")?,
            tool_call: tokenizer.special("<tool_call>")?,
            tool_call_end: tokenizer.special("</tool_call>")?,
            tokenizer,
            sampler,
            len: 0,
        })
    }

    /// Opens the conversation with a system prompt, reporting how many of
    /// its tokens the model has read so far, out of how many, as it goes.
    /// Tags in the prompt, such as `<tool_call>`, are encoded as special
    /// tokens.
    pub fn system(&mut self, text: &str, mut on_progress: impl FnMut(usize, usize)) -> Result<()> {
        if self.len != 0 {
            return Err("the conversation has already started".into());
        }
        let content = self.tokenizer.encode_with_special(text)?;
        let turn = self.turn("system", content)?;
        on_progress(0, turn.len());
        for (i, batch) in turn.chunks(BATCH).enumerate() {
            self.feed(batch)?;
            on_progress(i * BATCH + batch.len(), turn.len());
        }
        Ok(())
    }

    /// Number of tokens in the conversation so far.
    pub fn tokens(&self) -> usize {
        self.len
    }

    /// Sends a message from the user, streaming the reply to `on_chunk` as
    /// it is generated, and returns the tool calls the reply made, as the
    /// model wrote them. The reply ends early if `on_chunk` breaks, and then
    /// makes no calls.
    pub fn send(&mut self, message: &str, on_chunk: impl FnMut(Chunk) -> ControlFlow<()>) -> Result<Vec<String>> {
        let content = self.tokenizer.encode(message)?;
        let turn = self.turn("user", content)?;
        self.feed(&turn)?;
        self.generate(on_chunk)
    }

    /// Sends the results of the tool calls the last reply made, in order,
    /// and streams the reply to them like `send`.
    pub fn respond(&mut self, outputs: &[String], on_chunk: impl FnMut(Chunk) -> ControlFlow<()>) -> Result<Vec<String>> {
        let responses: Vec<String> = outputs
            .iter()
            .map(|output| format!("<tool_response>\n{output}\n</tool_response>"))
            .collect();
        let content = self.tokenizer.encode_with_special(&responses.join("\n"))?;
        let turn = self.turn("user", content)?;
        self.feed(&turn)?;
        self.generate(on_chunk)
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

    /// Generates the assistant's reply to the conversation so far.
    fn generate(&mut self, mut on_chunk: impl FnMut(Chunk) -> ControlFlow<()>) -> Result<Vec<String>> {
        let mut prompt = vec![self.im_start];
        prompt.extend(self.tokenizer.encode("assistant\n")?);

        let mut logits = self.feed(&prompt)?;
        let mut text = Utf8Stream::default();
        let mut calls = Vec::new();
        // Whether the model is inside its <think> block.
        let mut thinking = false;
        // The body of the tool call being written, if the model is in one.
        let mut call: Option<String> = None;
        let mut interrupted = false;
        loop {
            let token = self.sampler.sample(&logits);
            if token == self.im_end || token == self.end_of_text {
                break;
            }
            let flow = if token == self.think {
                thinking = true;
                ControlFlow::Continue(())
            } else if token == self.think_end {
                thinking = false;
                ControlFlow::Continue(())
            } else if token == self.tool_call {
                call = Some(String::new());
                ControlFlow::Continue(())
            } else if token == self.tool_call_end {
                calls.extend(call.take());
                ControlFlow::Continue(())
            } else {
                let chunk = text.push(self.tokenizer.decode(token));
                match &mut call {
                    Some(call) => {
                        call.push_str(&chunk);
                        ControlFlow::Continue(())
                    }
                    None if chunk.is_empty() => ControlFlow::Continue(()),
                    None if thinking => on_chunk(Chunk::Thought(&chunk)),
                    None => on_chunk(Chunk::Text(&chunk)),
                }
            };
            // Feed the token even if the reply ends here, so that the model
            // remembers the reply exactly as far as it was shown.
            logits = self.feed(&[token])?;
            if flow.is_break() {
                interrupted = true;
                break;
            }
        }

        // End the reply the way the chat format expects, ready for the next
        // message, even if it was cut short.
        let mut end = vec![self.im_end];
        end.extend(self.tokenizer.encode("\n")?);
        self.feed(&end)?;
        Ok(if interrupted { Vec::new() } else { calls })
    }

    /// Runs tokens through the model, returning the logits after the last one.
    fn feed(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        if self.len + tokens.len() > self.state.max_len() {
            return Err("the conversation no longer fits in the context window".into());
        }
        let logits = self.model.forward(&mut self.state, tokens, self.len);
        self.len += tokens.len();
        Ok(logits)
    }
}

/// Turns a stream of bytes into text, holding back a UTF-8 character split
/// across tokens until the rest of it arrives.
#[derive(Default)]
struct Utf8Stream {
    pending: Vec<u8>,
}

impl Utf8Stream {
    fn push(&mut self, bytes: &[u8]) -> String {
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
