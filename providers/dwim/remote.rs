//! The model as the server has it, reached over a connection to the
//! server: a [`Provider`], asked with the whole conversation each time.

use std::{
    error::Error,
    io::{self, BufRead, BufReader, Write},
    net::Shutdown,
    ops::ControlFlow,
    os::unix::net::UnixStream,
    path::Path,
};

use crate::{
    model::{Answer, Chat, Chunk, Generation, Result, Sampling, Size, State, Turn},
    providers::{Completion, Provider},
    server::{
        build,
        protocol::{Opened, Piece, Request, Response, Status, Welcome},
    },
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
    ) -> Result<Remote> {
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
        } = opened;
        Ok(Remote {
            connection: self,
            device,
            size,
            capacity,
            sampling,
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

/// A session on the server, held over a connection to it. The server
/// keeps what the session has read of a conversation, so a request that
/// goes on from the last costs only what it adds.
pub struct Remote {
    connection: Connection,
    device: String,
    size: Size,
    capacity: usize,
    sampling: Sampling,
    /// Told when the server has this session wait for another's request.
    on_status: Box<dyn FnMut(Status)>,
}

impl Remote {
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

    /// Makes a request answered with a count.
    fn count(&mut self, request: &Request) -> Result<usize> {
        match self.call(request, |_| ControlFlow::Continue(()))? {
            Response::Count(count) => Ok(count),
            other => Err(unexpected(&other)),
        }
    }
}

impl Provider for Remote {
    fn capacity(&self) -> usize {
        self.capacity
    }

    fn sampling(&self) -> Sampling {
        self.sampling
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

    fn read(&mut self, chat: &Chat, mut on_progress: impl FnMut(usize, usize)) -> Result<State> {
        let request = Request::Read { chat: chat.clone() };
        let response = self.call(&request, |response| {
            if let Response::Status(Status::Prompting { read, total }) = response {
                on_progress(*read, *total);
            }
            ControlFlow::Continue(())
        })?;
        match response {
            Response::Read(state) => Ok(state),
            other => Err(unexpected(&other)),
        }
    }

    fn reply(
        &mut self,
        chat: &Chat,
        generation: &Generation,
        mut on_chunk: impl FnMut(Chunk) -> ControlFlow<()>,
    ) -> Result<Completion> {
        let request = Request::Reply {
            chat: chat.clone(),
            generation: generation.clone(),
        };
        let response = self.call(&request, |response| match response {
            Response::Chunk(piece) => on_chunk(piece.chunk()),
            _ => ControlFlow::Continue(()),
        })?;
        match response {
            Response::Reply { reply, text, state } => Ok(Completion { reply, text, state }),
            other => Err(unexpected(&other)),
        }
    }

    fn answer(
        &mut self,
        chat: &Chat,
        heads: &[&str],
        limit: usize,
        generation: &Generation,
        mut on_text: impl FnMut(&str) -> ControlFlow<()>,
    ) -> Result<(Answer, State)> {
        let request = Request::Answer {
            chat: chat.clone(),
            heads: heads.iter().map(|head| head.to_string()).collect(),
            limit,
            generation: generation.clone(),
        };
        let response = self.call(&request, |response| match response {
            Response::Chunk(Piece::Text(text)) => on_text(text),
            _ => ControlFlow::Continue(()),
        })?;
        match response {
            Response::Answered { answer, state } => Ok((answer, state)),
            other => Err(unexpected(&other)),
        }
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
