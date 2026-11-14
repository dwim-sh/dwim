//! A session with the model as the server has it, held over a connection
//! to the server: the [`Conversation`] the harness runs on.

use std::{
    error::Error,
    io::{self, BufRead, BufReader, Write},
    net::Shutdown,
    ops::ControlFlow,
    os::unix::net::UnixStream,
    path::Path,
};

use crate::inference::{Answer, Chunk, Conversation, Reply, Result, Sampler, Size, Stats, Turn};
use crate::server::{
    build,
    protocol::{Generation, Opened, Piece, Request, Response, State, Status, Welcome},
};

/// A connection to the server, before a session is opened on it.
pub struct Connection {
    reader: BufReader<UnixStream>,
    writer: UnixStream,
}

impl Connection {
    /// Connects to the server listening at `path`.
    pub fn connect(path: &Path) -> io::Result<Self> {
        let writer = UnixStream::connect(path)?;
        let reader = BufReader::new(writer.try_clone()?);
        Ok(Self { reader, writer })
    }

    /// Asks the server who it is.
    pub fn hello(&mut self) -> Result<Welcome> {
        match self.call(&Request::Hello { build: build() }, |_| {
            ControlFlow::Continue(())
        })? {
            Response::Welcome(welcome) => Ok(welcome),
            other => Err(unexpected(&other)),
        }
    }

    /// Asks the server to go, and returns how many sessions it exits
    /// after.
    pub fn retire(&mut self) -> Result<usize> {
        match self.call(&Request::Retire, |_| ControlFlow::Continue(()))? {
            Response::Retired { sessions } => Ok(sessions),
            other => Err(unexpected(&other)),
        }
    }

    /// Opens the session: a conversation with the model, sampled at
    /// `temperature` or as the model's file recommends, drafting
    /// `speculate` tokens after each token picked. Waits for the model to
    /// load if it is loading, reporting how that goes to `on_status`.
    pub fn open(
        mut self,
        temperature: Option<f32>,
        speculate: usize,
        mut on_status: impl FnMut(Status),
    ) -> Result<Client> {
        let request = Request::Open {
            temperature,
            speculate,
        };
        let opened = match self.call(&request, |response| {
            if let Response::Status(status) = response {
                on_status(status.clone());
            }
            ControlFlow::Continue(())
        })? {
            Response::Opened(opened) => opened,
            other => return Err(unexpected(&other)),
        };
        let Opened {
            device,
            size,
            capacity,
            sampling,
            state,
        } = opened;
        Ok(Client {
            connection: self,
            device,
            size,
            capacity,
            tokens: state.tokens,
            stats: state.stats,
            sampler: sampling.sampler(),
            generation: Generation::default(),
            on_status: Box::new(|_| {}),
        })
    }

    /// Writes a request and reads what comes back until its answer, passing
    /// what comes ahead of the answer to `on`: when `on` breaks, the server
    /// is asked to end the reply early, and what else comes ahead of the
    /// answer is dropped. An error the server answers with is returned as
    /// one.
    fn call(
        &mut self,
        request: &Request,
        mut on: impl FnMut(&Response) -> ControlFlow<()>,
    ) -> Result<Response> {
        self.write(request)?;
        let mut interrupted = false;
        loop {
            let response = self.read()?;
            match response {
                Response::Status(_) | Response::Chunk(_) => {
                    if !interrupted && on(&response).is_break() {
                        interrupted = true;
                        self.write(&Request::Interrupt)?;
                    }
                }
                Response::Error { message } => return Err(message.into()),
                response => return Ok(response),
            }
        }
    }

    fn write(&mut self, request: &Request) -> Result<()> {
        let mut line = serde_json::to_vec(request)?;
        line.push(b'\n');
        self.writer.write_all(&line)?;
        Ok(())
    }

    fn read(&mut self) -> Result<Response> {
        let mut line = String::new();
        if self.reader.read_line(&mut line)? == 0 {
            return Err("the dwim server hung up".into());
        }
        Ok(serde_json::from_str(&line)?)
    }
}

fn unexpected(response: &Response) -> Box<dyn Error> {
    format!("unexpected response from the dwim server: {response:?}").into()
}

/// A session with the model, held over a connection to the server. What
/// a request changes about the conversation, the tokens in it and where
/// its time has gone, comes back with each answer; what the client sets,
/// the reserve and the thinking budget, goes with each request for a reply.
pub struct Client {
    connection: Connection,
    device: String,
    size: Size,
    capacity: usize,
    tokens: usize,
    stats: Stats,
    sampler: Sampler,
    generation: Generation,
    /// Told when the server has this session wait for another's request.
    on_status: Box<dyn FnMut(Status)>,
}

impl Client {
    /// Ends the session: the server is told it is over, and whatever is
    /// asked of it after fails.
    pub fn close(&mut self) {
        let _ = self.connection.writer.shutdown(Shutdown::Both);
    }

    /// The name of the device the model runs on.
    pub fn device(&self) -> &str {
        &self.device
    }

    /// The model's size on its device.
    pub fn size(&self) -> Size {
        self.size
    }

    /// Has `on_status` told what the server is doing while a request of
    /// this session's waits: that the model is busy with another session's.
    pub fn on_status(&mut self, on_status: impl FnMut(Status) + 'static) {
        self.on_status = Box::new(on_status);
    }

    /// Opens the conversation with the system prompt, from the state the
    /// server saved after an earlier reading of the same prompt if there
    /// is one, reporting the reading as `Conversation::system` does, and
    /// returns whether the state was restored.
    pub fn start(&mut self, prompt: &str, on_progress: impl FnMut(usize, usize)) -> Result<bool> {
        let request = Request::Start {
            prompt: prompt.to_string(),
        };
        match self.call(&request, prompting(on_progress))? {
            Response::Started { restored, state } => {
                self.update(state);
                Ok(restored)
            }
            other => Err(unexpected(&other)),
        }
    }

    /// Makes a request, as [`Connection::call`] does, telling `on_status`
    /// when the session waits for another's request.
    fn call(
        &mut self,
        request: &Request,
        mut on: impl FnMut(&Response) -> ControlFlow<()>,
    ) -> Result<Response> {
        let on_status = &mut self.on_status;
        self.connection.call(request, |response| {
            if let Response::Status(Status::Busy) = response {
                on_status(Status::Busy);
                return ControlFlow::Continue(());
            }
            on(response)
        })
    }

    /// Makes a request whose answer says only that it is done.
    fn done(
        &mut self,
        request: &Request,
        on: impl FnMut(&Response) -> ControlFlow<()>,
    ) -> Result<()> {
        match self.call(request, on)? {
            Response::Ok(state) => {
                self.update(state);
                Ok(())
            }
            other => Err(unexpected(&other)),
        }
    }

    /// Makes a request answered with a count.
    fn count(&mut self, request: &Request) -> Result<usize> {
        match self.call(request, |_| ControlFlow::Continue(()))? {
            Response::Count(count) => Ok(count),
            other => Err(unexpected(&other)),
        }
    }

    /// Makes a request for a reply, streaming its chunks to `on_chunk`.
    fn reply(
        &mut self,
        request: &Request,
        mut on_chunk: impl FnMut(Chunk) -> ControlFlow<()>,
    ) -> Result<Reply> {
        let response = self.call(request, |response| match response {
            Response::Chunk(piece) => on_chunk(piece.chunk()),
            _ => ControlFlow::Continue(()),
        })?;
        match response {
            Response::Reply { reply, state } => {
                self.update(state);
                Ok(reply)
            }
            other => Err(unexpected(&other)),
        }
    }

    fn update(&mut self, state: State) {
        self.tokens = state.tokens;
        self.stats = state.stats;
    }
}

/// What passes the progress of a prompt being read on to `on_progress`.
fn prompting(
    mut on_progress: impl FnMut(usize, usize),
) -> impl FnMut(&Response) -> ControlFlow<()> {
    move |response| {
        if let Response::Status(Status::Prompting { read, total }) = response {
            on_progress(*read, *total);
        }
        ControlFlow::Continue(())
    }
}

impl Piece {
    fn chunk(&self) -> Chunk<'_> {
        match self {
            Piece::Thought(text) => Chunk::Thought(text),
            Piece::Text(text) => Chunk::Text(text),
            Piece::Token => Chunk::Token,
        }
    }
}

impl Conversation for Client {
    fn system(&mut self, text: &str, on_progress: impl FnMut(usize, usize)) -> Result<()> {
        let request = Request::System {
            text: text.to_string(),
        };
        self.done(&request, prompting(on_progress))
    }

    fn clear(&mut self) -> Result<()> {
        self.done(&Request::Clear, |_| ControlFlow::Continue(()))
    }

    fn replay(&mut self, turns: &[Turn]) -> Result<()> {
        let request = Request::Replay {
            turns: turns.to_vec(),
        };
        self.done(&request, |_| ControlFlow::Continue(()))
    }

    fn measure(&mut self, turn: &Turn) -> Result<usize> {
        self.count(&Request::Measure { turn: turn.clone() })
    }

    fn answer_tokens(&mut self, message: &str, limit: usize) -> Result<usize> {
        self.count(&Request::AnswerTokens {
            message: message.to_string(),
            limit,
        })
    }

    fn tokens(&self) -> usize {
        self.tokens
    }

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn free(&self) -> usize {
        self.capacity
            .saturating_sub(self.tokens + self.generation.reserve)
    }

    fn reserve(&mut self, tokens: usize) {
        self.generation.reserve = tokens;
    }

    fn thought_limit(&self) -> Option<usize> {
        self.generation.thought_limit
    }

    fn limit_thoughts(&mut self, tokens: Option<usize>, closing: &str) {
        self.generation.thought_limit = tokens;
        self.generation.closing = Some(closing.to_string());
    }

    fn feed_results(&mut self, outputs: &[String]) -> Result<()> {
        let request = Request::FeedResults {
            outputs: outputs.to_vec(),
        };
        self.done(&request, |_| ControlFlow::Continue(()))
    }

    fn stats(&self) -> Stats {
        self.stats
    }

    fn sampler(&self) -> &Sampler {
        &self.sampler
    }

    fn send(
        &mut self,
        message: &str,
        on_chunk: impl FnMut(Chunk) -> ControlFlow<()>,
    ) -> Result<Reply> {
        let request = Request::Send {
            message: message.to_string(),
            generation: self.generation.clone(),
        };
        self.reply(&request, on_chunk)
    }

    fn respond(
        &mut self,
        outputs: &[String],
        on_chunk: impl FnMut(Chunk) -> ControlFlow<()>,
    ) -> Result<Reply> {
        let request = Request::Respond {
            outputs: outputs.to_vec(),
            generation: self.generation.clone(),
        };
        self.reply(&request, on_chunk)
    }

    fn answer(
        &mut self,
        message: &str,
        heads: &[&str],
        limit: usize,
        mut on_text: impl FnMut(&str) -> ControlFlow<()>,
    ) -> Result<Answer> {
        let request = Request::Answer {
            message: message.to_string(),
            heads: heads.iter().map(|head| head.to_string()).collect(),
            limit,
            generation: self.generation.clone(),
        };
        let response = self.call(&request, |response| match response {
            Response::Chunk(Piece::Text(text)) => on_text(text),
            _ => ControlFlow::Continue(()),
        })?;
        match response {
            Response::Answered { answer, state } => {
                self.update(state);
                Ok(answer)
            }
            other => Err(unexpected(&other)),
        }
    }
}
