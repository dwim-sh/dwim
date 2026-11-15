//! A conversation as a request for a reply carries it: whole, with how the
//! reply is to be made, and what comes back besides the reply.

use serde::{Deserialize, Serialize};

use crate::model::{Stats, Turn};

/// A conversation, whole: the system prompt it opens with, and the turns
/// since.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct Chat {
    pub system: String,
    pub turns: Vec<Turn>,
}

/// How a reply is generated, which goes with each request for one.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct Generation {
    /// Tokens of the context kept free: the reply stops short of them.
    pub reserve: usize,
    /// Most tokens a thought may run to, if they are limited, and what is
    /// written to close a thought ended early.
    pub thought_limit: Option<usize>,
    pub closing: Option<String>,
}

/// Where a conversation is: the tokens in it so far, and where its time
/// has gone.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct State {
    pub tokens: usize,
    pub stats: Stats,
}
