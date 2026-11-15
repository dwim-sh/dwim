//! Where a session's model runs: what the harness reaches one through.
//!
//! A provider is asked the way chat APIs are: with the whole conversation
//! each time, which it answers with a reply, keeping nothing the next
//! request depends on. What it has read of a conversation before is its
//! own to reuse. A [`Conversation`] is the session the harness holds on
//! one.

mod conversation;
pub mod dwim;
pub mod openrouter;

use std::ops::ControlFlow;

use crate::model::{Answer, Chat, Chunk, Generation, Reply, Result, Sampling, State, Turn};

pub use conversation::Conversation;

/// A model somewhere, asked with the whole conversation each time.
pub trait Provider {
    /// Number of tokens a conversation has room for in all.
    fn capacity(&self) -> usize;

    /// How the model's tokens are sampled.
    fn sampling(&self) -> Sampling;

    /// Number of tokens `turn` takes in a conversation.
    fn measure(&mut self, turn: &Turn) -> Result<usize>;

    /// Most tokens a plain answer to `message` takes with its framing, if
    /// it is limited to `limit` tokens.
    fn answer_tokens(&mut self, message: &str, limit: usize) -> Result<usize>;

    /// Reads `chat` without replying to it, so that a reply to it, or to
    /// what follows it, starts from there, reporting how many of the
    /// tokens it had yet to read it has read so far, out of how many.
    fn read(&mut self, chat: &Chat, on_progress: impl FnMut(usize, usize)) -> Result<State>;

    /// Replies to `chat`, which ends with a message from the user or the
    /// results of tool calls, streaming the reply to `on_chunk`, which
    /// ends it early by breaking.
    fn reply(
        &mut self,
        chat: &Chat,
        generation: &Generation,
        on_chunk: impl FnMut(Chunk) -> ControlFlow<()>,
    ) -> Result<Completion>;

    /// Answers the message `chat` ends with plainly, under `heads` in
    /// turn, in at most `limit` tokens, streaming the answer to `on_text`.
    fn answer(
        &mut self,
        chat: &Chat,
        heads: &[&str],
        limit: usize,
        generation: &Generation,
        on_text: impl FnMut(&str) -> ControlFlow<()>,
    ) -> Result<(Answer, State)>;
}

/// What a reply came to: the reply, its text as it was shown, which with
/// the reply's calls is the turn it adds to the conversation, and where
/// the conversation is after it.
#[derive(Debug, Default, PartialEq)]
pub struct Completion {
    pub reply: Reply,
    pub text: String,
    pub state: State,
}
