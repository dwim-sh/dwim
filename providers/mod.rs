//! Where a session's model runs: what the harness reaches one through.
//!
//! A provider is asked the way chat APIs are: with the whole conversation
//! each time, which it answers with a reply, keeping nothing the next
//! request depends on. What it has read of a conversation before is its
//! own to reuse. A [`Conversation`] is the session the harness holds on
//! one.
//!
//! There are two kinds: the dwim server on this machine, and a host of the
//! model, which is OpenRouter or Vercel's AI Gateway.

mod conversation;
pub mod dwim;
pub mod hosted;
pub mod openrouter;
pub mod vercel;

use std::{fmt, ops::ControlFlow, str::FromStr};

use crate::model::{Answer, Chat, Chunk, Generation, Reply, Result, Sampling, State, Turn};

use hosted::{Host, Hosted};

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
/// the reply's calls is the turn it adds to the conversation, the thought
/// behind it if the turn is to keep it, and where the conversation is
/// after it.
#[derive(Debug, Default, PartialEq)]
pub struct Completion {
    pub reply: Reply,
    pub text: String,
    pub thought: String,
    pub state: State,
}

/// Which provider a session's model runs at.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Kind {
    /// The dwim server, on this machine.
    Local,
    OpenRouter,
    /// Vercel's AI Gateway.
    Vercel,
}

impl Kind {
    /// Who hosts a model that runs at this provider, if it is hosted.
    pub fn host(self) -> Option<&'static Host> {
        match self {
            Kind::Local => None,
            Kind::OpenRouter => Some(&openrouter::HOST),
            Kind::Vercel => Some(&vercel::HOST),
        }
    }
}

impl fmt::Display for Kind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Kind::Local => "local",
            Kind::OpenRouter => "openrouter",
            Kind::Vercel => "vercel",
        })
    }
}

impl FromStr for Kind {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, String> {
        match value {
            "local" => Ok(Kind::Local),
            "openrouter" => Ok(Kind::OpenRouter),
            "vercel" => Ok(Kind::Vercel),
            _ => Err(format!(
                "unknown provider '{value}' (expected local, openrouter or vercel)"
            )),
        }
    }
}

/// A provider of either kind, as a command that runs on whichever was
/// asked for holds one.
pub enum Any {
    Dwim(dwim::Remote),
    Hosted(Hosted),
}

impl Any {
    /// Which provider this is.
    pub fn kind(&self) -> Kind {
        match self {
            Any::Dwim(_) => Kind::Local,
            Any::Hosted(hosted) => hosted.host().kind,
        }
    }

    /// The session on the dwim server, if that is the provider.
    pub fn dwim(&mut self) -> Option<&mut dwim::Remote> {
        match self {
            Any::Dwim(remote) => Some(remote),
            Any::Hosted(_) => None,
        }
    }
}

impl Provider for Any {
    fn capacity(&self) -> usize {
        match self {
            Any::Dwim(provider) => provider.capacity(),
            Any::Hosted(provider) => provider.capacity(),
        }
    }

    fn sampling(&self) -> Sampling {
        match self {
            Any::Dwim(provider) => provider.sampling(),
            Any::Hosted(provider) => provider.sampling(),
        }
    }

    fn measure(&mut self, turn: &Turn) -> Result<usize> {
        match self {
            Any::Dwim(provider) => provider.measure(turn),
            Any::Hosted(provider) => provider.measure(turn),
        }
    }

    fn answer_tokens(&mut self, message: &str, limit: usize) -> Result<usize> {
        match self {
            Any::Dwim(provider) => provider.answer_tokens(message, limit),
            Any::Hosted(provider) => provider.answer_tokens(message, limit),
        }
    }

    fn read(&mut self, chat: &Chat, on_progress: impl FnMut(usize, usize)) -> Result<State> {
        match self {
            Any::Dwim(provider) => provider.read(chat, on_progress),
            Any::Hosted(provider) => provider.read(chat, on_progress),
        }
    }

    fn reply(
        &mut self,
        chat: &Chat,
        generation: &Generation,
        on_chunk: impl FnMut(Chunk) -> ControlFlow<()>,
    ) -> Result<Completion> {
        match self {
            Any::Dwim(provider) => provider.reply(chat, generation, on_chunk),
            Any::Hosted(provider) => provider.reply(chat, generation, on_chunk),
        }
    }

    fn answer(
        &mut self,
        chat: &Chat,
        heads: &[&str],
        limit: usize,
        generation: &Generation,
        on_text: impl FnMut(&str) -> ControlFlow<()>,
    ) -> Result<(Answer, State)> {
        match self {
            Any::Dwim(provider) => provider.answer(chat, heads, limit, generation, on_text),
            Any::Hosted(provider) => provider.answer(chat, heads, limit, generation, on_text),
        }
    }
}
