//! A session with the model, as the harness holds one, and what its
//! replies come to.

use std::ops::ControlFlow;

use serde::{Deserialize, Serialize};

use crate::model::{Result, Sampling, Turn};

/// A session with the model, a conversation with it, as the harness holds
/// one, wherever the model runs.
pub trait Session {
    /// Forgets the conversation, so that a system prompt can open another.
    fn clear(&mut self) -> Result<()>;

    /// Feeds `turns` as the conversation so far, without a reply to them.
    fn replay(&mut self, turns: &[Turn]) -> Result<()>;

    /// Number of tokens `turn` takes when fed.
    fn measure(&mut self, turn: &Turn) -> Result<usize>;

    /// Most tokens a plain answer to `message` takes with its framing, if
    /// it is limited to `limit` tokens.
    fn answer_tokens(&mut self, message: &str, limit: usize) -> Result<usize>;

    /// Number of tokens in the conversation so far.
    fn tokens(&self) -> usize;

    /// Number of tokens the conversation has room for in all.
    fn capacity(&self) -> usize;

    /// Tokens the conversation has room for before the reserve.
    fn free(&self) -> usize;

    /// Keeps `tokens` of the context free: a reply stops short of them.
    fn reserve(&mut self, tokens: usize);

    /// Ends a thought that runs to `tokens`, if they are limited, or that
    /// repeats itself, with `closing` written into it as the model's own.
    fn limit_thoughts(&mut self, tokens: Option<usize>, closing: &str);

    /// Feeds the results of the tool calls the last reply made, without a
    /// reply.
    fn feed_results(&mut self, outputs: &[String]) -> Result<()>;

    /// Where the conversation's time has gone so far.
    fn stats(&self) -> Stats;

    /// How the model's tokens are sampled.
    fn sampling(&self) -> Sampling;

    /// Sends a message from the user, streaming the reply to `on_chunk`,
    /// which ends it early by breaking.
    fn send(
        &mut self,
        message: &str,
        on_chunk: impl FnMut(Chunk) -> ControlFlow<()>,
    ) -> Result<Reply>;

    /// Sends the results of the tool calls the last reply made, and
    /// streams the reply to them like `send`.
    fn respond(
        &mut self,
        outputs: &[String],
        on_chunk: impl FnMut(Chunk) -> ControlFlow<()>,
    ) -> Result<Reply>;

    /// Has the model answer `message` plainly, under `heads` in turn, in
    /// at most `limit` tokens, streaming the answer to `on_text`.
    fn answer(
        &mut self,
        message: &str,
        heads: &[&str],
        limit: usize,
        on_text: impl FnMut(&str) -> ControlFlow<()>,
    ) -> Result<Answer>;
}

/// A piece of the model's reply, as it is generated.
pub enum Chunk<'a> {
    /// Part of the model's thought, before it replies.
    Thought(&'a str),
    /// Part of the reply itself.
    Text(&'a str),
    /// A token sampled, as it may be hidden (a tool-call tag or body, a
    /// thought that closes) rather than shown as a `Thought` or `Text`.
    /// It carries no text.
    Token,
}

/// What a reply came to: the tool calls it made, as the model wrote them,
/// whether it was stopped short because the context was full, but for the
/// reserve, and whether its thought was ended at the limit
/// [`Session::limit_thoughts`] sets, rather than by the model.
#[derive(Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct Reply {
    pub calls: Vec<String>,
    pub cut: bool,
    pub thought_ended: bool,
}

/// A plain answer, and whether it was stopped short for room.
#[derive(Debug, Deserialize, PartialEq, Serialize)]
pub struct Answer {
    pub text: String,
    pub cut: bool,
}

/// Where a conversation's time has gone: the tokens the model read as
/// prompts, the ones it generated while thinking, and the ones it
/// generated as replies, with the time spent on each; and how many tokens
/// were drafted, and how many of those the sampler accepted.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct Stats {
    /// Tokens taken from a saved state rather than read.
    pub cached: Tally,
    pub prompt: Tally,
    pub thought: Tally,
    pub answer: Tally,
    pub drafted: usize,
    pub accepted: usize,
}

/// Tokens of one kind, and the seconds the model took over them.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct Tally {
    pub tokens: usize,
    pub seconds: f64,
}

impl Tally {
    /// Tokens a second, or zero before there are any.
    pub fn rate(&self) -> f64 {
        if self.seconds > 0.0 {
            self.tokens as f64 / self.seconds
        } else {
            0.0
        }
    }
}
