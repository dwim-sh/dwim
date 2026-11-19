//! The server that has the model, and the sessions on it.
//!
//! The model takes most of the memory a machine has, so one process loads
//! it, and every `dwim` session is a client of that process over a Unix
//! socket: `dwim` starts the server itself when none is running, and the
//! server exits once it has gone a while without a session. The harness,
//! the tools, and the terminal stay in the client; the server has the
//! model, the sampling, and each session's conversation with it, as a
//! [`Local`], and speaks the [`protocol`].
//!
//! A request carries the whole conversation, and the server answers it
//! from what it is asked alone, as a chat API does. What a session has
//! read of a conversation stays in its sequence, so a request that goes on
//! from it costs only what it adds: the server reads the turns that are
//! new, and reads the conversation over, from the state saved after its
//! system prompt, only when it is not the one the session has read.
//!
//! The model's weights are shared, and each session has a sequence of its
//! own on the device: its key/value caches, its linear layers' recurrent
//! states, and what the drafter carries over. The server allocates as many
//! sequences as it has room for sessions when it starts, and a session
//! takes one when it opens and gives it back when it closes; a session
//! that finds none free is refused. Changing from one session's sequence to
//! another's copies nothing.
//!
//! Requests are served one at a time, in the order they arrive, each whole,
//! so sessions take turns with the model at reply granularity. The ones
//! that need no model, such as measuring a turn, are answered at once, by
//! the thread that reads the connection.

pub mod protocol;

use std::{
    cell::RefCell,
    collections::{HashMap, hash_map::Entry},
    env,
    error::Error,
    fs,
    io::{self, BufRead, BufReader, Write},
    ops::ControlFlow,
    path::{Path, PathBuf},
    process,
    rc::Rc,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc::{self, Receiver, RecvTimeoutError, Sender},
    },
    thread,
    time::{Duration, UNIX_EPOCH},
};

use crate::{
    inference::{LanguageModel, Local, Sampler},
    model::{Chat, Chunk, Format, Generation, Session, Size, State, Turn},
    server::protocol::{Opened, Piece, Request, Response, Status, Welcome},
    sys::{
        self,
        net::{Listener, Stream},
    },
};

/// How long the server goes without a session before it exits, by
/// default: long enough that a session started soon after the last ended
/// finds the model loaded.
pub const IDLE: Duration = Duration::from_secs(10 * 60);

/// Where the server listens: in the user's runtime directory, or the
/// temporary directory where there is none, named for the user.
pub fn socket_path() -> PathBuf {
    let dir = sys::dirs::runtime().unwrap_or_else(env::temp_dir);
    dir.join(format!("dwim-{}.sock", sys::user::id()))
}

/// Which build of dwim this is: its version, and the size and the
/// modification time of its executable, so that a client and a server of
/// the same version but different builds tell each other apart.
pub fn build() -> String {
    let version = env!("CARGO_PKG_VERSION");
    match env::current_exe().and_then(fs::metadata) {
        Ok(metadata) => {
            let modified = metadata
                .modified()
                .ok()
                .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
                .map_or(0, |elapsed| elapsed.as_secs());
            format!("{version}-{}-{modified}", metadata.len())
        }
        Err(_) => version.to_string(),
    }
}

/// What the server says of itself before the model is loaded: the model,
/// the device it runs on, as the command line names them, the context it
/// was asked for, if it was, and how many sessions it has room for.
pub struct Hello {
    pub model: String,
    pub device: String,
    pub context: Option<usize>,
    pub room: usize,
}

/// A connection, as the threads share it: where its responses are written,
/// and whether the reply being generated for it is to end early.
#[derive(Clone)]
struct Peer {
    writer: Arc<Mutex<Stream>>,
    interrupted: Arc<AtomicBool>,
}

impl Peer {
    /// Writes a response, a line of JSON.
    fn write(&self, response: &Response) -> io::Result<()> {
        let mut line = serde_json::to_vec(response)?;
        line.push(b'\n');
        self.writer
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .write_all(&line)
    }
}

/// What a connection's reader sends the model's thread.
enum Message {
    Request(Request),
    /// The connection is gone.
    Closed,
}

/// What every thread of the server shares.
struct Inner {
    path: PathBuf,
    hello: Hello,
    build: String,
    peers: Mutex<HashMap<usize, Peer>>,
    /// The chat format, once the model is loaded, for measuring turns.
    format: OnceLock<Format>,
    /// How many sessions are open.
    sessions: AtomicUsize,
    /// Whose request the model is on, if any.
    serving: Mutex<Option<usize>>,
    /// Whether the server was asked to go.
    retiring: AtomicBool,
}

/// A session as the server has it: the conversation on the model, and what
/// of a conversation the model has read into it, if anything, which the
/// next request is read on from.
struct Served {
    session: Local<Shared>,
    read: Option<Chat>,
}

/// The server: listening on its socket, with the model loading or loaded.
pub struct Server {
    inner: Arc<Inner>,
    inbox: Receiver<(usize, Message)>,
    idle: Duration,
    /// Whether the state after a system prompt is saved, and taken up
    /// from by the next conversation that opens with the prompt.
    states: bool,
}

impl Server {
    /// Listens on the socket at `path`, saying `hello` to whoever connects,
    /// and takes connections from then on: the model is loaded after, with
    /// [`Server::status`] telling the connections how it goes, and
    /// [`Server::run`] serving them once it is. Fails if another server
    /// has the socket; a socket left by a server that is gone is replaced.
    pub fn bind(path: &Path, hello: Hello) -> io::Result<Self> {
        let listener = match Listener::bind(path) {
            Ok(listener) => listener,
            Err(e) if e.kind() == io::ErrorKind::AddrInUse => {
                if Stream::connect(path).is_ok() {
                    return Err(io::Error::new(
                        io::ErrorKind::AddrInUse,
                        format!("another dwim server is listening on {}", path.display()),
                    ));
                }
                fs::remove_file(path)?;
                Listener::bind(path)?
            }
            Err(e) => return Err(e),
        };
        let inner = Arc::new(Inner {
            path: path.to_path_buf(),
            hello,
            build: build(),
            peers: Mutex::new(HashMap::new()),
            format: OnceLock::new(),
            sessions: AtomicUsize::new(0),
            serving: Mutex::new(None),
            retiring: AtomicBool::new(false),
        });
        let (inbox_tx, inbox) = mpsc::channel();
        thread::spawn({
            let inner = inner.clone();
            move || accept(&listener, &inner, &inbox_tx)
        });
        Ok(Self {
            inner,
            inbox,
            idle: IDLE,
            states: true,
        })
    }

    /// Sets how long the server goes without a session before it exits,
    /// [`IDLE`] unless set.
    pub fn set_idle(&mut self, idle: Duration) {
        self.idle = idle;
    }

    /// Has the server read every system prompt, keeping no state of one
    /// from one conversation to the next.
    #[cfg(test)]
    fn keep_no_states(&mut self) {
        self.states = false;
    }

    /// Tells every connection how loading the model is going.
    pub fn status(&self, status: &Status) {
        self.broadcast(&Response::Status(status.clone()));
    }

    /// Tells every connection that the model could not be loaded: the
    /// server is about to exit.
    pub fn fail(&self, message: &str) {
        self.broadcast(&Response::Error {
            message: message.to_string(),
        });
    }

    fn broadcast(&self, response: &Response) {
        let peers: Vec<Peer> = self.inner.peers().values().cloned().collect();
        for peer in peers {
            let _ = peer.write(response);
        }
    }

    /// Serves the connections with `model`, loaded on `device` at `size`,
    /// giving it a sequence for each of the sessions the size has room
    /// for, in `format`, each session sampling with what `sampler` makes of the
    /// temperature it asks for, until the server has gone the idle time
    /// without a session, or was asked to go and its sessions have ended.
    pub fn run(
        self,
        mut model: Box<dyn LanguageModel>,
        format: Format,
        device: String,
        size: Size,
        sampler: impl Fn(Option<f32>) -> Sampler,
    ) -> Result<(), Box<dyn Error>> {
        let inner = &self.inner;
        if inner.format.set(format.clone()).is_err() {
            return Err("the server has run already".into());
        }
        let mut free = vec![0];
        for _ in 1..size.sessions {
            free.push(model.add_sequence()?);
        }
        free.reverse();
        let pool = Rc::new(RefCell::new(Pool { model, free }));
        let mut sessions: HashMap<usize, Served> = HashMap::new();
        loop {
            if inner.retiring.load(Ordering::Relaxed) && sessions.is_empty() {
                eprintln!("retired: exiting");
                break;
            }
            let (id, message) = match self.inbox.recv_timeout(self.idle) {
                Ok(message) => message,
                Err(RecvTimeoutError::Timeout) if sessions.is_empty() => {
                    eprintln!("idle for {} s: exiting", self.idle.as_secs());
                    break;
                }
                Err(RecvTimeoutError::Timeout) => continue,
                Err(RecvTimeoutError::Disconnected) => break,
            };
            let request = match message {
                Message::Request(request) => request,
                Message::Closed => {
                    if sessions.remove(&id).is_some() {
                        eprintln!("session {id} closed");
                    }
                    inner.sessions.store(sessions.len(), Ordering::Relaxed);
                    continue;
                }
            };
            let Some(peer) = inner.peers().get(&id).cloned() else {
                continue;
            };
            *inner.serving.lock().unwrap_or_else(|e| e.into_inner()) = Some(id);
            let response = match request {
                Request::Open {
                    temperature,
                    speculate,
                } => match sessions.entry(id) {
                    Entry::Occupied(_) => {
                        Err("a session is already open on this connection".into())
                    }
                    Entry::Vacant(entry) => Shared::take(&pool).and_then(|shared| {
                        Local::new(shared, format.clone(), sampler(temperature)).map(
                            |mut session| {
                                session.set_drafts(speculate);
                                let opened = Opened {
                                    device: device.clone(),
                                    size,
                                    capacity: session.capacity(),
                                    sampling: session.sampling(),
                                };
                                eprintln!(
                                    "session {id} opened on sequence {}",
                                    shared_id(&session)
                                );
                                entry.insert(Served {
                                    session,
                                    read: None,
                                });
                                Response::Opened(opened)
                            },
                        )
                    }),
                },
                request => match sessions.get_mut(&id) {
                    Some(served) => {
                        let response = serve(served, request, &peer, self.states);
                        // A reply that failed part way may leave a batch
                        // being verified: it is committed before another
                        // session has the model. What the model has read
                        // is then not known to be a conversation's, so the
                        // next request is read from its start.
                        served.session.settle();
                        if response.is_err() {
                            served.read = None;
                        }
                        response
                    }
                    None => Err("no session is open on this connection".into()),
                },
            };
            inner.sessions.store(sessions.len(), Ordering::Relaxed);
            *inner.serving.lock().unwrap_or_else(|e| e.into_inner()) = None;
            let response = response.unwrap_or_else(|e: Box<dyn Error>| Response::Error {
                message: e.to_string(),
            });
            // A client that is gone is found out by its reader, which
            // says so.
            let _ = peer.write(&response);
        }
        Ok(())
    }
}

impl Drop for Server {
    /// Gives up the socket, unless it was given up already.
    fn drop(&mut self) {
        if !self.inner.retiring.load(Ordering::Relaxed) {
            let _ = fs::remove_file(&self.inner.path);
        }
    }
}

impl Inner {
    fn peers(&self) -> std::sync::MutexGuard<'_, HashMap<usize, Peer>> {
        self.peers.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Whether the model is on another connection's request.
    fn busy_for(&self, id: usize) -> bool {
        self.serving
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_some_and(|serving| serving != id)
    }
}

/// Takes connections as they come, and reads each on a thread of its own.
fn accept(listener: &Listener, inner: &Arc<Inner>, inbox: &Sender<(usize, Message)>) {
    for (id, stream) in listener.incoming().enumerate() {
        let Ok(stream) = stream else {
            continue;
        };
        let Ok(reading) = stream.try_clone() else {
            continue;
        };
        let peer = Peer {
            writer: Arc::new(Mutex::new(stream)),
            interrupted: Arc::new(AtomicBool::new(false)),
        };
        inner.peers().insert(id, peer.clone());
        thread::spawn({
            let inner = inner.clone();
            let inbox = inbox.clone();
            move || {
                read(id, reading, &peer, &inner, &inbox);
                inner.peers().remove(&id);
                let _ = inbox.send((id, Message::Closed));
            }
        });
    }
}

/// Reads a connection's requests until it closes: answers the ones that
/// need no model, and passes the rest to the model's thread, telling the
/// client when they must wait for another session's.
fn read(id: usize, stream: Stream, peer: &Peer, inner: &Inner, inbox: &Sender<(usize, Message)>) {
    for line in BufReader::new(stream).lines() {
        let Ok(line) = line else {
            break;
        };
        let request: Request = match serde_json::from_str(&line) {
            Ok(request) => request,
            Err(e) => {
                let _ = peer.write(&Response::Error {
                    message: format!("malformed request: {e}"),
                });
                continue;
            }
        };
        let response = match request {
            Request::Hello { .. } => Response::Welcome(Welcome {
                build: inner.build.clone(),
                pid: process::id(),
                model: inner.hello.model.clone(),
                device: inner.hello.device.clone(),
                context: inner.hello.context,
                room: inner.hello.room,
                sessions: inner.sessions.load(Ordering::Relaxed),
            }),
            Request::Retire => {
                inner.retiring.store(true, Ordering::Relaxed);
                let _ = fs::remove_file(&inner.path);
                Response::Retired {
                    sessions: inner.sessions.load(Ordering::Relaxed),
                }
            }
            Request::Measure { turn } => match inner.format.get() {
                Some(format) => count(format.measure(&turn)),
                None => not_loaded(),
            },
            Request::AnswerTokens { message, limit } => match inner.format.get() {
                Some(format) => count(format.answer_tokens(&message, limit)),
                None => not_loaded(),
            },
            Request::Interrupt => {
                peer.interrupted.store(true, Ordering::Relaxed);
                continue;
            }
            request => {
                if matches!(request, Request::Reply { .. } | Request::Answer { .. }) {
                    peer.interrupted.store(false, Ordering::Relaxed);
                }
                if inner.busy_for(id) {
                    let _ = peer.write(&Response::Status(Status::Busy));
                }
                if inbox.send((id, Message::Request(request))).is_err() {
                    break;
                }
                continue;
            }
        };
        if peer.write(&response).is_err() {
            break;
        }
    }
}

fn count(result: crate::inference::Result<usize>) -> Response {
    match result {
        Ok(count) => Response::Count(count),
        Err(e) => Response::Error {
            message: e.to_string(),
        },
    }
}

fn not_loaded() -> Response {
    Response::Error {
        message: "the model is not loaded yet".to_string(),
    }
}

/// Where `session` is, for a response.
fn state<M: LanguageModel>(session: &Local<M>) -> State {
    State {
        tokens: session.tokens(),
        stats: session.stats(),
    }
}

/// Serves a session's request on the model, streaming what there is to
/// `peer` along the way. The state after a system prompt is saved and
/// taken up from if `states` says so.
fn serve(
    served: &mut Served,
    request: Request,
    peer: &Peer,
    states: bool,
) -> Result<Response, Box<dyn Error>> {
    let progress = |read: usize, total: usize| {
        let _ = peer.write(&Response::Status(Status::Prompting { read, total }));
    };
    // A chunk goes to the client; the reply ends early when the client
    // asked it to, or is gone.
    let stream = |piece: Piece| -> ControlFlow<()> {
        if peer.write(&Response::Chunk(piece)).is_err() || peer.interrupted.load(Ordering::Relaxed)
        {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    };
    Ok(match request {
        Request::Read { chat } => {
            served.catch_up(&chat.system, &chat.turns, states, progress)?;
            Response::Read(state(&served.session))
        }
        Request::Reply { chat, generation } => {
            let (last, before) = asked(&chat)?;
            served.catch_up(&chat.system, before, states, progress)?;
            generation.apply(&mut served.session);
            // The reply's text as it is streamed, which is the turn the
            // reply adds, and what the client is told it is.
            let mut text = String::new();
            let on_chunk = |chunk: Chunk| {
                if let Chunk::Text(piece) = chunk {
                    text.push_str(piece);
                }
                stream(Piece::of(chunk))
            };
            let reply = match last {
                Turn::User(message) => served.session.send(message, on_chunk)?,
                Turn::Results(outputs) => served.session.respond(outputs, on_chunk)?,
                Turn::Reply { .. } => unreachable!("asked ends with what is replied to"),
            };
            served.add(last.clone());
            served.add(Turn::Reply {
                text: text.clone(),
                calls: reply.calls.clone(),
                thought: String::new(),
            });
            Response::Reply {
                reply,
                text,
                state: state(&served.session),
            }
        }
        Request::Answer {
            chat,
            heads,
            limit,
            generation,
        } => {
            let (last, before) = asked(&chat)?;
            let Turn::User(message) = last else {
                return Err("a plain answer is to a message from the user".into());
            };
            served.catch_up(&chat.system, before, states, progress)?;
            generation.apply(&mut served.session);
            let heads: Vec<&str> = heads.iter().map(String::as_str).collect();
            let answer = served.session.answer(message, &heads, limit, |text| {
                stream(Piece::Text(text.to_string()))
            })?;
            served.add(last.clone());
            served.add(Turn::Reply {
                text: answer.text.clone(),
                calls: Vec::new(),
                thought: String::new(),
            });
            Response::Answered {
                answer,
                state: state(&served.session),
            }
        }
        Request::Hello { .. }
        | Request::Retire
        | Request::Open { .. }
        | Request::Measure { .. }
        | Request::AnswerTokens { .. }
        | Request::Interrupt => return Err("the request is not a session's".into()),
    })
}

/// What `chat` asks a reply to, a message from the user or the results of
/// tool calls, and the turns before it.
fn asked(chat: &Chat) -> Result<(&Turn, &[Turn]), Box<dyn Error>> {
    match chat.turns.split_last() {
        Some((last, before)) if !matches!(last, Turn::Reply { .. }) => Ok((last, before)),
        _ => Err("the conversation does not end with something to reply to".into()),
    }
}

impl Served {
    /// Has the model read the conversation that opens with `system` and
    /// goes on with `turns`: the turns it has yet to read, if the
    /// conversation goes on from the one it has read, and otherwise all of
    /// it, from the start, telling `on_progress` how the reading of the
    /// system prompt goes.
    fn catch_up(
        &mut self,
        system: &str,
        turns: &[Turn],
        states: bool,
        on_progress: impl FnMut(usize, usize),
    ) -> Result<(), Box<dyn Error>> {
        let goes_on = self
            .read
            .as_ref()
            .is_some_and(|read| read.system == system && turns.starts_with(&read.turns));
        if !goes_on {
            self.read = None;
            self.session.clear()?;
            if states {
                self.session.start(system, on_progress)?;
            } else {
                self.session.system(system, on_progress)?;
            }
            self.read = Some(Chat {
                system: system.to_string(),
                turns: Vec::new(),
            });
        }
        let read = self.read.as_mut().expect("the conversation is open");
        let new = &turns[read.turns.len()..];
        if !new.is_empty() {
            // A reading that fails part way leaves the model with turns
            // that are not known here.
            let read = self.read.take().expect("the conversation is open");
            self.session.replay(new)?;
            let mut read = read;
            read.turns.extend_from_slice(new);
            self.read = Some(read);
        }
        Ok(())
    }

    /// Adds a turn the model has read, or said, to what it has read.
    fn add(&mut self, turn: Turn) {
        if let Some(read) = &mut self.read {
            read.turns.push(turn);
        }
    }
}

impl Generation {
    /// Sets `session` as the client has it for the reply to come.
    fn apply<M: LanguageModel>(&self, session: &mut Local<M>) {
        session.reserve(self.reserve);
        if let Some(closing) = &self.closing {
            session.limit_thoughts(self.thought_limit, closing);
        }
    }
}

/// The model, shared by the sessions, and the numbers of its sequences
/// that no session has.
struct Pool {
    model: Box<dyn LanguageModel>,
    free: Vec<usize>,
}

/// A session's handle on the shared model: a language model that is the
/// model running the session's own sequence, which it selects as it is
/// used, and gives back when it is dropped.
struct Shared {
    id: usize,
    pool: Rc<RefCell<Pool>>,
}

impl Shared {
    /// Takes a sequence no session has, as before any token, or fails if
    /// every one is taken.
    fn take(pool: &Rc<RefCell<Pool>>) -> Result<Self, Box<dyn Error>> {
        let mut p = pool.borrow_mut();
        let Some(id) = p.free.pop() else {
            return Err(
                "the dwim server has no room for another session: quit one, or start the server with more --sessions"
                    .into(),
            );
        };
        p.model.select_sequence(id)?;
        p.model.reset();
        drop(p);
        Ok(Self {
            id,
            pool: pool.clone(),
        })
    }

    /// The model, running this session's sequence.
    fn model(&self) -> std::cell::RefMut<'_, Box<dyn LanguageModel>> {
        let mut pool = self.pool.borrow_mut();
        pool.model
            .select_sequence(self.id)
            .expect("no batch is verified across requests");
        std::cell::RefMut::map(pool, |pool| &mut pool.model)
    }
}

/// The number of the sequence a session runs on.
fn shared_id(session: &Local<Shared>) -> usize {
    session.model().id
}

impl Drop for Shared {
    fn drop(&mut self) {
        self.pool.borrow_mut().free.push(self.id);
    }
}

impl LanguageModel for Shared {
    fn save(&self, len: usize) -> Option<Vec<u8>> {
        self.model().save(len)
    }

    fn restore(&mut self, state: &[u8]) -> crate::inference::Result<usize> {
        self.model().restore(state)
    }

    fn forward(&mut self, tokens: &[u32], pos: usize) -> Vec<f32> {
        self.model().forward(tokens, pos)
    }

    fn draft(&mut self, token: u32, pos: usize, k: usize) -> Vec<u32> {
        self.model().draft(token, pos, k)
    }

    fn verifiable(&self, pos: usize) -> usize {
        self.model().verifiable(pos)
    }

    fn verify(&mut self, tokens: &[u32], pos: usize) -> Vec<f32> {
        self.model().verify(tokens, pos)
    }

    fn rewind(&mut self, keep: usize) {
        self.model().rewind(keep)
    }

    fn mark(&mut self) -> bool {
        self.model().mark()
    }

    fn rewind_to_mark(&mut self) -> crate::inference::Result<()> {
        self.model().rewind_to_mark()
    }

    fn reset(&mut self) {
        self.model().reset()
    }

    fn max_len(&self) -> usize {
        self.pool.borrow().model.max_len()
    }

    /// A session runs the one sequence it took.
    fn add_sequence(&mut self) -> crate::inference::Result<usize> {
        Err("a session runs one sequence".into())
    }

    fn select_sequence(&mut self, id: usize) -> crate::inference::Result<()> {
        if id == 0 {
            Ok(())
        } else {
            Err("a session runs one sequence".into())
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use crate::{
        inference::testing::Scripted,
        model::{Session, Tokenizer, Turn},
        providers::{
            Conversation, Provider,
            dwim::{Connection, Remote},
        },
    };

    use super::*;

    /// A server on a socket of its own, serving a scripted model that says
    /// `scripts` in turn, on a thread that ends when the server exits. If
    /// `hold` is given, the model holds once it has said that many tokens,
    /// until the server has been asked to interrupt a reply.
    fn server(scripts: &[&str], hold: Option<usize>) -> (PathBuf, thread::JoinHandle<()>) {
        static SERVERS: AtomicUsize = AtomicUsize::new(0);
        let n = SERVERS.fetch_add(1, Ordering::Relaxed);
        let path = env::temp_dir().join(format!("dwim-test-{}-{n}.sock", process::id()));
        let scripts: Vec<String> = scripts.iter().map(|s| s.to_string()).collect();
        let hello = Hello {
            model: "tiny".to_string(),
            device: "cpu".to_string(),
            context: None,
            room: 2,
        };
        let mut server = Server::bind(&path, hello).unwrap();
        server.keep_no_states();
        let inner = server.inner.clone();
        let handle = thread::spawn(move || {
            let tokenizer = Tokenizer::tiny();
            let scripts: Vec<&str> = scripts.iter().map(String::as_str).collect();
            let mut model =
                Scripted::new(&tokenizer, &scripts, 1 << 12).answers(&tokenizer, "noted");
            if let Some(after) = hold {
                // It goes on after a while regardless, for the reply to
                // run to its end and the test to fail on it, rather than
                // hang, if the interrupt never comes.
                model = model.holds(after, move || {
                    let deadline = Instant::now() + Duration::from_secs(10);
                    while Instant::now() < deadline
                        && !inner
                            .peers()
                            .values()
                            .any(|peer| peer.interrupted.load(Ordering::Relaxed))
                    {
                        thread::sleep(Duration::from_millis(1));
                    }
                });
            }
            let format = Format::new(tokenizer).unwrap();
            let size = Size {
                sessions: 2,
                context: 1 << 12,
                model: 0,
                conversation: 0,
            };
            server
                .run(
                    Box::new(model),
                    format,
                    "cpu".to_string(),
                    size,
                    |temperature| Sampler::new(temperature.unwrap_or(0.0), 1, 1.0, 0.0, 1),
                )
                .unwrap();
        });
        (path, handle)
    }

    fn open(path: &Path) -> Conversation<Remote> {
        let connection = Connection::connect(path).unwrap();
        Conversation::new(connection.open(None, 0, |_| {}).unwrap())
    }

    /// The text of a reply to `message`.
    fn ask(session: &mut Conversation<Remote>, message: &str) -> (String, crate::model::Reply) {
        let mut text = String::new();
        let reply = session
            .send(message, |chunk| {
                if let Chunk::Text(piece) = chunk {
                    text.push_str(piece);
                }
                ControlFlow::Continue(())
            })
            .unwrap();
        (text, reply)
    }

    #[test]
    fn sessions_take_turns_with_their_own_state() {
        let (path, handle) = server(
            &[
                "</think>\n\nOne.",
                "</think>\n\nTwo.",
                "</think>\n\nThree.",
                "</think>\n\nFour.",
            ],
            None,
        );
        let mut a = open(&path);
        let mut b = open(&path);
        a.start("Be brief.", |_, _| {}).unwrap();
        b.start("Be brief.", |_, _| {}).unwrap();
        assert_eq!(a.tokens(), b.tokens());
        let start = a.tokens();

        let (one, _) = ask(&mut a, "first");
        assert_eq!(one.trim(), "One.");
        let after_one = a.tokens();
        assert!(after_one > start);
        let (two, _) = ask(&mut b, "second");
        assert_eq!(two.trim(), "Two.");
        // The sessions' states are their own: b has one exchange, a has
        // one, and a's goes on from where it was.
        assert_eq!(
            b.tokens(),
            a.tokens() - a.measure(&Turn::User("first".into())).unwrap()
                + b.measure(&Turn::User("second".into())).unwrap()
        );
        let (three, _) = ask(&mut a, "third");
        assert_eq!(three.trim(), "Three.");
        assert!(a.tokens() > after_one);
        let (four, reply) = ask(&mut b, "fourth");
        assert_eq!(four.trim(), "Four.");
        assert!(reply.calls.is_empty());

        // Measuring needs no model, and agrees with a session's measure.
        let turn = Turn::Reply {
            text: "Ok.".to_string(),
            calls: vec!["<function=bash>\n</function>".to_string()],
            thought: String::new(),
        };
        let format = Format::new(Tokenizer::tiny()).unwrap();
        assert_eq!(a.measure(&turn).unwrap(), format.measure(&turn).unwrap());
        assert_eq!(
            a.answer_tokens("note", 100).unwrap(),
            format.answer_tokens("note", 100).unwrap()
        );

        // A plain answer, under heads: the model says its piece under the
        // first and has nothing left for the second.
        let answer = a
            .answer("note", &["A:", "B:"], 20, |_| ControlFlow::Continue(()))
            .unwrap();
        assert_eq!(answer.text, "A:notedB:");

        // Starting over, and reading turns back.
        let tokens = a.tokens();
        a.clear().unwrap();
        assert_eq!(a.tokens(), 0);
        a.start("Be brief.", |_, _| {}).unwrap();
        a.replay(&[Turn::User("hi".to_string())]).unwrap();
        assert!(a.tokens() > start && a.tokens() < tokens);

        // A third session finds no room, until one of the two closes, and
        // then it starts afresh on the sequence given back.
        let refused = Connection::connect(&path)
            .unwrap()
            .open(None, 0, |_| {})
            .err()
            .unwrap();
        assert!(refused.to_string().contains("no room"), "{refused}");
        drop(b);
        // The server hears of the close on a thread of its own, so the
        // open may come first, and is tried again.
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut c = loop {
            match Connection::connect(&path).unwrap().open(None, 0, |_| {}) {
                Ok(session) => break Conversation::new(session),
                Err(_) if Instant::now() < deadline => thread::sleep(Duration::from_millis(10)),
                Err(e) => panic!("{e}"),
            }
        };
        c.start("Be brief.", |_, _| {}).unwrap();
        assert_eq!(c.tokens(), start);
        let b = c;

        // The server goes once asked to, and its sessions are gone.
        let mut connection = Connection::connect(&path).unwrap();
        let welcome = connection.hello().unwrap();
        assert_eq!(welcome.sessions, 2);
        assert_eq!(welcome.build, build());
        assert_eq!(connection.retire().unwrap(), 2);
        assert!(!path.exists());
        drop(connection);
        drop(a);
        drop(b);
        handle.join().unwrap();
    }

    #[test]
    fn a_conversation_sent_whole_again_is_read_once() {
        let (path, handle) = server(&["</think>\n\nOne.", "</think>\n\nTwo."], None);
        let mut session = open(&path);
        session.start("Be brief.", |_, _| {}).unwrap();
        let start = session.stats().prompt.tokens;
        ask(&mut session, "first");
        let first = session.stats().prompt.tokens - start;
        // The second request carries the first exchange again, and the
        // server reads only what it adds: as much as the first did, the
        // messages and the replies being as long.
        let (two, _) = ask(&mut session, "again");
        assert_eq!(two.trim(), "Two.");
        assert_eq!(session.stats().prompt.tokens - start, 2 * first);

        let mut connection = Connection::connect(&path).unwrap();
        connection.retire().unwrap();
        drop(connection);
        drop(session);
        handle.join().unwrap();
    }

    #[test]
    fn a_reply_is_interrupted_and_the_session_goes_on() {
        let thought: String = (1..).map(|i: u32| i.to_string()).take(200).collect();
        // The model holds partway into the thought until the server has
        // heard the interrupt, so that the interrupt lands inside the reply
        // however the threads are scheduled.
        let (path, handle) = server(
            &[
                &format!(
                    "{thought}</think>\n\nSaid.<tool_call>\n<function=bash>\n<parameter=command>\necho\n</parameter>\n</function>\n</tool_call>"
                ),
                "</think>\n\nNext.",
            ],
            Some(20),
        );
        let mut session = open(&path);
        session.start("Be brief.", |_, _| {}).unwrap();
        let mut thoughts = 0;
        let reply = session
            .send("go", |chunk| {
                if let Chunk::Thought(_) = chunk {
                    thoughts += 1;
                }
                if thoughts >= 3 {
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                }
            })
            .unwrap();
        // The reply ended at the first token after the model went on: the
        // rest of the thought, and the call it was about to make, were
        // never said.
        assert!(thoughts < 100, "{thoughts}");
        assert!(reply.calls.is_empty());
        let (next, _) = ask(&mut session, "on");
        assert_eq!(next.trim(), "Next.");

        // A server with no sessions exits once idle.
        let mut connection = Connection::connect(&path).unwrap();
        connection.retire().unwrap();
        drop(connection);
        drop(session);
        handle.join().unwrap();
    }

    #[test]
    fn a_welcome_from_an_older_build_is_read() {
        let older = r#"{"Welcome":{"build":"0.0.10-1-2","pid":7,"model":"qwen3.8-27b","device":"gpu","context":null,"sessions":1}}"#;
        let Response::Welcome(welcome) = serde_json::from_str(older).unwrap() else {
            panic!("not a welcome");
        };
        assert_eq!((welcome.pid, welcome.room, welcome.sessions), (7, 0, 1));
    }

    #[test]
    fn an_idle_server_exits() {
        let path = env::temp_dir().join(format!("dwim-test-idle-{}.sock", process::id()));
        let hello = Hello {
            model: "tiny".to_string(),
            device: "cpu".to_string(),
            context: Some(7),
            room: 1,
        };
        let mut server = Server::bind(&path, hello).unwrap();
        server.set_idle(Duration::from_millis(200));
        // A client that connects while the model loads is told of it, and
        // of who the server is at once.
        let mut connection = Connection::connect(&path).unwrap();
        let welcome = connection.hello().unwrap();
        assert_eq!(welcome.context, Some(7));
        assert_eq!(welcome.pid, process::id());
        server.status(&Status::Weights { done: 1, total: 2 });
        let started = Instant::now();
        let handle = thread::spawn(move || {
            let tokenizer = Tokenizer::tiny();
            let model = Scripted::new(&tokenizer, &[], 1 << 12);
            server
                .run(
                    Box::new(model),
                    Format::new(tokenizer).unwrap(),
                    "cpu".to_string(),
                    Size {
                        sessions: 1,
                        context: 1 << 12,
                        model: 0,
                        conversation: 0,
                    },
                    |_| Sampler::new(0.0, 1, 1.0, 0.0, 1),
                )
                .unwrap();
        });
        let mut seen = Vec::new();
        let session = connection
            .open(None, 0, |status| seen.push(status))
            .unwrap();
        assert_eq!(seen, [Status::Weights { done: 1, total: 2 }]);
        assert_eq!(session.capacity(), 1 << 12);
        drop(session);
        handle.join().unwrap();
        assert!(started.elapsed() >= Duration::from_millis(200));
        assert!(!path.exists());
    }
}
