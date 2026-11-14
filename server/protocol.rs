//! What a client and the server say to each other, over a Unix socket:
//! JSON, one message a line. A request is answered with a response, with
//! what the server has to say in the meantime, such as a reply's pieces or
//! how loading is going, streamed ahead of it. One connection holds one
//! session, and makes one request at a time.

use serde::{Deserialize, Serialize};

use crate::inference::{Answer, Reply, Stats, Turn};

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
    /// Opens the conversation with the system prompt: from the state
    /// saved after an earlier reading of the same prompt, if there is one,
    /// and otherwise by reading it, reporting [`Status::Prompting`] as it
    /// goes. Answered with [`Response::Started`].
    Start { prompt: String },
    /// Opens the conversation with a system prompt, reading it, with
    /// [`Status::Prompting`] as it goes. Answered with [`Response::Ok`].
    System { text: String },
    /// Forgets the conversation. Answered with [`Response::Ok`].
    Clear,
    /// Feeds turns as the conversation so far. Answered with
    /// [`Response::Ok`].
    Replay { turns: Vec<Turn> },
    /// How many tokens a turn takes when fed. Answered with
    /// [`Response::Count`] at once, without waiting for the model.
    Measure { turn: Turn },
    /// Most tokens a plain answer to the message takes with its framing,
    /// limited to `limit` tokens. Answered with [`Response::Count`] at
    /// once.
    AnswerTokens { message: String, limit: usize },
    /// Feeds the results of the last reply's tool calls, without a reply.
    /// Answered with [`Response::Ok`].
    FeedResults { outputs: Vec<String> },
    /// Sends a message from the user: the reply's pieces are streamed as
    /// [`Response::Chunk`]s, and then it is answered with
    /// [`Response::Reply`].
    Send {
        message: String,
        generation: Generation,
    },
    /// Sends the results of the last reply's tool calls, and streams the
    /// reply to them like `Send`.
    Respond {
        outputs: Vec<String>,
        generation: Generation,
    },
    /// Has the model answer the message plainly, under `heads` in turn, in
    /// at most `limit` tokens: the answer's text is streamed as
    /// [`Response::Chunk`]s of text, and then it is answered with
    /// [`Response::Answered`].
    Answer {
        message: String,
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

/// How a reply is generated: what the conversation is set to as the reply
/// starts, which the client keeps and sends with each request for one.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct Generation {
    /// Tokens of the context kept free: the reply stops short of them.
    pub reserve: usize,
    /// Most tokens a thought may run to, if they are limited, and what is
    /// written to close a thought ended early.
    pub thought_limit: Option<usize>,
    pub closing: Option<String>,
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
    /// The answer to [`Request::Start`]: whether the state was restored
    /// rather than the prompt read.
    Started { restored: bool, state: State },
    /// The answer to a request that has nothing to say but that it is
    /// done.
    Ok(State),
    /// The answer to [`Request::Measure`] and [`Request::AnswerTokens`].
    Count(usize),
    /// The answer to [`Request::Send`] and [`Request::Respond`].
    Reply { reply: Reply, state: State },
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
/// the conversation has room for, how its tokens are sampled, and where the
/// conversation is.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Opened {
    pub device: String,
    pub size: Size,
    pub capacity: usize,
    pub sampling: Sampling,
    pub state: State,
}

/// Where a conversation is: the tokens in it so far, and where its time
/// has gone.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct State {
    pub tokens: usize,
    pub stats: Stats,
}

/// How tokens are sampled: a sampler's settings.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct Sampling {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub seed: u64,
}

/// A piece of a reply, as [`crate::inference::Chunk`] has it, owned.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub enum Piece {
    Thought(String),
    Text(String),
    Token,
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

/// How many sessions the model has room for at once, the tokens of
/// conversation each has room for, and the device memory they take:
/// `model` bytes for the weights and the activations, which the sessions
/// share, and `conversation` bytes more for each session, for its recurrent
/// states and its tokens of context.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct Size {
    pub sessions: usize,
    pub context: usize,
    pub model: u64,
    pub conversation: u64,
}
