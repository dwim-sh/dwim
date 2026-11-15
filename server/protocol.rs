//! What a client and the server say to each other, over a Unix socket:
//! JSON, one message a line. A request is answered with a response, with
//! what the server has to say in the meantime, such as a reply's pieces or
//! how loading is going, streamed ahead of it. One connection holds one
//! session, and makes one request at a time.
//!
//! A request for a reply carries the whole conversation, as a chat API's
//! does, and nothing a client asks depends on what it asked before. The
//! server keeps what its session has read of the conversation, and reads
//! only what a request adds to it.

use serde::{Deserialize, Serialize};

use crate::model::{Answer, Chat, Chunk, Generation, Reply, Sampling, Size, State, Turn};

/// What a client asks of the server.
#[derive(Debug, Deserialize, Serialize)]
pub enum Request {
    /// Who is calling: the build of dwim the client is. Answered with
    /// [`Response::Welcome`] at once, whatever the server is doing.
    Hello { build: String },
    /// Asks the server to go: to give up its socket at once, so that
    /// another can take it, and to exit as soon as it has no sessions.
    /// Answered with [`Response::Retired`].
    Retire,
    /// Opens the connection's session: a conversation with the model,
    /// sampled at `temperature`, or as the model's file recommends, with
    /// `speculate` tokens drafted after each token picked. Answered with
    /// [`Response::Opened`].
    Open {
        temperature: Option<f32>,
        speculate: usize,
    },
    /// Reads the conversation, without a reply to it: what the session has
    /// not read of it yet, or all of it if it is not the conversation the
    /// session has read, from the state saved after an earlier reading of
    /// its system prompt where there is one, reporting
    /// [`Status::Prompting`] as the prompt is read. Answered with
    /// [`Response::Read`].
    Read { chat: Chat },
    /// How many tokens a turn takes when fed. Answered with
    /// [`Response::Count`] at once, without waiting for the model.
    Measure { turn: Turn },
    /// Most tokens a plain answer to the message takes with its framing,
    /// limited to `limit` tokens. Answered with [`Response::Count`] at
    /// once.
    AnswerTokens { message: String, limit: usize },
    /// Replies to the conversation, which ends with a message from the
    /// user or the results of tool calls, having read it as
    /// [`Request::Read`] does: the reply's pieces are streamed as
    /// [`Response::Chunk`]s, and then it is answered with
    /// [`Response::Reply`].
    Reply { chat: Chat, generation: Generation },
    /// Has the model answer the message the conversation ends with
    /// plainly, under `heads` in turn, in at most `limit` tokens: the
    /// answer's text is streamed as [`Response::Chunk`]s of text, and then
    /// it is answered with [`Response::Answered`].
    Answer {
        chat: Chat,
        heads: Vec<String>,
        limit: usize,
        generation: Generation,
    },
    /// Ends the reply being generated for this connection early, as
    /// breaking from the chunks does in this process. It is not answered:
    /// the reply it ends is, as it would have been. Sent before the
    /// request it ends was taken up, it ends the reply as it starts.
    Interrupt,
}

/// What the server says back.
#[derive(Debug, Deserialize, Serialize)]
pub enum Response {
    /// The answer to [`Request::Hello`].
    Welcome(Welcome),
    /// The answer to [`Request::Retire`]: how many sessions the server
    /// has, which it exits after.
    Retired { sessions: usize },
    /// What the server is doing while a request waits: loading the model
    /// when it is new, or serving another session. Sent ahead of a
    /// request's answer, to every connection while the model loads.
    Status(Status),
    /// The answer to [`Request::Open`].
    Opened(Opened),
    /// A piece of the reply being generated.
    Chunk(Piece),
    /// The answer to [`Request::Read`]: where the conversation is.
    Read(State),
    /// The answer to [`Request::Measure`] and [`Request::AnswerTokens`].
    Count(usize),
    /// The answer to [`Request::Reply`]: the reply, its text as it was
    /// streamed, which with its calls is the turn it adds to the
    /// conversation, and where the conversation is after it.
    Reply {
        reply: Reply,
        text: String,
        state: State,
    },
    /// The answer to [`Request::Answer`].
    Answered { answer: Answer, state: State },
    /// The request failed, or could not be read.
    Error { message: String },
}

/// Who the server is: its build, its process, the model it has or is
/// loading, the device it runs on, the context it was asked for, if it was
/// asked for one, how many sessions it has room for, and how many it has.
///
/// A session reads this from a server of any build, to find out whether to
/// replace it, so a field added to it has a default, for a server that
/// does not send it.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Welcome {
    pub build: String,
    pub pid: u32,
    pub model: String,
    pub device: String,
    pub context: Option<usize>,
    #[serde(default)]
    pub room: usize,
    pub sessions: usize,
}

/// A session as opened: the device the model runs on, its size, the tokens
/// a conversation has room for, and how its tokens are sampled.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Opened {
    pub device: String,
    pub size: Size,
    pub capacity: usize,
    pub sampling: Sampling,
}

/// A piece of a reply, as [`crate::model::Chunk`] has it, owned.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub enum Piece {
    Thought(String),
    Text(String),
    Token,
}

impl Piece {
    /// `chunk`, owned.
    pub fn of(chunk: Chunk) -> Self {
        match chunk {
            Chunk::Thought(text) => Piece::Thought(text.to_string()),
            Chunk::Text(text) => Piece::Text(text.to_string()),
            Chunk::Token => Piece::Token,
        }
    }
}

/// What the server is doing while a request waits.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub enum Status {
    /// Downloading one of the model's files: how many bytes so far, out of
    /// how many if that is known.
    Downloading {
        file: String,
        done: u64,
        total: Option<u64>,
    },
    /// The device the model is being loaded onto is open: its name.
    Device(String),
    /// How much room the model has for conversation, and the memory it
    /// takes with it.
    Sized(Size),
    /// How many of the model's tensors are loaded, out of how many.
    Weights { done: usize, total: usize },
    /// The model is reading a system prompt: how many of its tokens so
    /// far, out of how many.
    Prompting { read: usize, total: usize },
    /// The model is busy with another session's request, and this one
    /// waits its turn.
    Busy,
    /// The server running is not as the session asks for, so the session
    /// asks it to go, to start another. The session tells itself this; a
    /// server never sends it.
    Restarting,
    /// No server was running as the session asks for, so the session
    /// started one and waits for it. The session tells itself this; a
    /// server never sends it.
    Starting,
    /// The session reached a server, and opens a conversation on it. The
    /// session tells itself this; a server never sends it.
    Opening,
}
