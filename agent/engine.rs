//! The model loaded in this program, in place of a server of its own: an
//! engine, which a session's settings carry.
//!
//! The model is loaded on a thread of the program, the first time a
//! session is opened on the engine, and runs there. That thread is a
//! server without a socket: a session reaches it over a connection made in
//! the program, and is served as a session of `dwim` is by the server
//! process, with a sequence of its own beside the weights the sessions
//! share. So the sessions of an engine take turns with the model, and
//! what holds of the server holds here: a conversation is read on from
//! what the session has read of it, and from the state saved after its
//! system prompt.
//!
//! The model stays loaded while the engine is held: by the settings it is
//! in, their clones, and the sessions opened on it. It is dropped when
//! the last of them is.

use std::{
    error::Error,
    sync::{Arc, Mutex},
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use crate::{
    agent::{Engine, Settings, Status, load_model},
    model::{Device, Model},
    providers::dwim::Connection,
    server::{Connector, Hello, Server},
};

/// The model loaded in this program, once a session is opened on it.
/// Clones are the same engine, and the sessions opened with them share its
/// weights.
#[derive(Clone, Default)]
pub struct Embedded {
    running: Arc<Mutex<Option<Running>>>,
    /// What serves in place of loading the model, in a test.
    #[cfg(test)]
    stand_in: Option<Arc<dyn Fn(Server) + Send + Sync>>,
}

/// The engine with the model loading or loaded: what it was asked for,
/// how its thread is reached, and the thread.
struct Running {
    asked: Asked,
    connector: Connector,
    thread: JoinHandle<()>,
}

/// What an engine is asked for: the model, the device it runs on, and the
/// room it has, for how many sessions and of how many tokens each.
#[derive(Clone, Copy, PartialEq)]
struct Asked {
    model: Model,
    device: Device,
    context: Option<usize>,
    sessions: usize,
}

/// Longest an engine is given to find that a session's connection has
/// closed, before the session counts as one it has.
const CLOSING: Duration = Duration::from_millis(200);

/// How often the engine's connections are counted again meanwhile.
const POLL: Duration = Duration::from_millis(2);

impl Embedded {
    /// Sessions an engine has room for at once, unless the settings say:
    /// one, which then has all the context that fits in memory.
    pub const SESSIONS: usize = 1;

    /// Connects to the engine, which starts loading the model as
    /// `settings` have it if it is not loaded: the session opened on the
    /// connection waits for that, and is told how it goes. An engine with
    /// another model, or the model on another device or with
    /// other room than asked for, is replaced if it has no sessions; with
    /// sessions, it is an error, since two models would not fit in
    /// memory.
    pub fn connect(&self, settings: &Settings) -> Result<Connection, Box<dyn Error>> {
        let asked = Asked {
            model: settings.model,
            device: settings.device,
            context: settings.context,
            sessions: settings.sessions.unwrap_or(Self::SESSIONS),
        };
        if asked.sessions == 0 {
            return Err("an engine needs room for one session at least".into());
        }
        let mut running = self.running.lock().unwrap_or_else(|e| e.into_inner());
        // An engine whose thread is over failed to load the model.
        if running.as_ref().is_some_and(|run| run.thread.is_finished()) {
            *running = None;
        }
        if let Some(run) = running.take_if(|run| run.asked != asked) {
            // A session that has just ended is gone from the engine once
            // its thread has read that the connection closed.
            let closing = Instant::now() + CLOSING;
            while run.connector.connections() > 0 && Instant::now() < closing {
                thread::sleep(POLL);
            }
            let sessions = run.connector.connections();
            if sessions > 0 {
                let model = run.asked.model;
                *running = Some(run);
                return Err(format!(
                    "{model} is loaded in this program as {sessions} other {} asked for it, and not as this one does: end {} first, or open this one with the same settings",
                    if sessions == 1 { "session" } else { "sessions" },
                    if sessions == 1 { "it" } else { "them" }
                )
                .into());
            }
            // With no one to reach it, the engine ends, and the model
            // goes before the next is loaded.
            let Running {
                thread, connector, ..
            } = run;
            drop(connector);
            let _ = thread.join();
        }
        if let Some(run) = running.as_ref() {
            return Ok(Connection::over(run.connector.connect()?)?);
        }
        let hello = Hello {
            model: asked.model.name().to_string(),
            device: asked.device.to_string(),
            context: asked.context,
            room: asked.sessions,
        };
        let (server, connector) = Server::unbound(hello);
        // The connection is made before the model loads, so that the
        // session on it is told of all of the loading.
        let connection = Connection::over(connector.connect()?)?;
        let thread = thread::Builder::new()
            .name("dwim engine".to_string())
            .spawn({
                // The thread has the settings without the engine it is
                // the thread of: with it, it would hold itself, and the
                // model would stay loaded with no one left to use it.
                let settings = Settings {
                    engine: Engine::Embedded(Embedded::default()),
                    ..settings.clone()
                };
                #[cfg(test)]
                let stand_in = self.stand_in.clone();
                move || {
                    #[cfg(test)]
                    if let Some(stand_in) = stand_in {
                        return stand_in(server);
                    }
                    // A failure is told to the sessions that wait.
                    let _ = serve(server, &settings, asked.sessions, |_| {});
                }
            })?;
        *running = Some(Running {
            asked,
            connector,
            thread,
        });
        Ok(connection)
    }
}

/// Loads the model as `settings` have it, with room for `sessions`
/// sessions, and serves `server`'s connections with it until the server
/// ends: what the engine's thread does, and the server process. The
/// connections are told how the loading goes, and `on_status` is, and
/// they are told if it fails.
pub fn serve(
    server: Server,
    settings: &Settings,
    sessions: usize,
    mut on_status: impl FnMut(&Status),
) -> Result<(), Box<dyn Error>> {
    let model = match load_model(settings, sessions, |status| {
        on_status(&status);
        server.status(&status);
    }) {
        Ok(model) => model,
        Err(e) => {
            server.fail(&e.to_string());
            return Err(e);
        }
    };
    let format = model.format().clone();
    let (device, size) = (model.device().to_string(), model.size());
    let temperature = settings.temperature;
    let sampler = model.sampler();
    server.run(Box::new(model), format, device, size, move |asked| {
        sampler(asked.or(temperature))
    })
}

#[cfg(test)]
mod tests {
    use std::{
        ops::ControlFlow,
        path::Path,
        sync::atomic::{AtomicUsize, Ordering},
    };

    use crate::{
        harness::{Event, Harness},
        inference::{Sampler, testing::Scripted},
        model::{Effort, Size, Tokenizer, qwen::ChatMl},
        providers::Conversation,
    };

    use super::*;

    /// An engine that serves a scripted model that says `scripts` in turn
    /// in place of loading one, with room for two sessions, and how many
    /// times it has been started, and has ended.
    fn engine(scripts: &[&str]) -> (Embedded, Arc<AtomicUsize>, Arc<AtomicUsize>) {
        let scripts: Vec<String> = scripts.iter().map(|s| s.to_string()).collect();
        let started = Arc::new(AtomicUsize::new(0));
        let ended = Arc::new(AtomicUsize::new(0));
        let stand_in = {
            let (started, ended) = (started.clone(), ended.clone());
            move |mut server: Server| {
                started.fetch_add(1, Ordering::Relaxed);
                server.keep_no_states();
                let tokenizer = Tokenizer::tiny();
                let scripts: Vec<&str> = scripts.iter().map(String::as_str).collect();
                let model = Scripted::new(&tokenizer, &scripts, 1 << 12);
                let size = Size {
                    sessions: 2,
                    context: 1 << 12,
                    model: 0,
                    conversation: 0,
                };
                server
                    .run(
                        Box::new(model),
                        Arc::new(ChatMl::new(tokenizer).unwrap()),
                        "cpu".to_string(),
                        size,
                        |temperature| Sampler::new(temperature.unwrap_or(0.0), 1, 1.0, 0.0, 1),
                    )
                    .unwrap();
                ended.fetch_add(1, Ordering::Relaxed);
            }
        };
        let embedded = Embedded {
            running: Arc::default(),
            stand_in: Some(Arc::new(stand_in)),
        };
        (embedded, started, ended)
    }

    /// Waits for `count` to come to `n`, for a while.
    fn reaches(count: &AtomicUsize, n: usize) -> bool {
        let deadline = Instant::now() + Duration::from_secs(10);
        while count.load(Ordering::Relaxed) != n && Instant::now() < deadline {
            thread::sleep(POLL);
        }
        count.load(Ordering::Relaxed) == n
    }

    fn settings(embedded: &Embedded) -> Settings {
        Settings {
            dir: std::env::temp_dir(),
            model: Model::Qwen27b,
            provider: None,
            disabled: Vec::new(),
            engine: Engine::Embedded(embedded.clone()),
            device: Device::Cpu,
            context: None,
            sessions: Some(2),
            thinking_budget: None,
            effort: Effort::Low,
            temperature: None,
            speculate: 0,
            shell: None,
            tools: Vec::new(),
            instructions: None,
        }
    }

    /// A harness on a session opened on the engine `settings` carry.
    fn harness(
        embedded: &Embedded,
        settings: &Settings,
    ) -> Harness<Conversation<crate::providers::dwim::Remote>> {
        let remote = embedded
            .connect(settings)
            .unwrap()
            .open(None, 0, |_| {})
            .unwrap();
        let mut session = Conversation::new(remote);
        session.start("Be brief.", |_, _| {}).unwrap();
        Harness::new(session, Path::new("."), |session| {
            session.start("Be brief.", |_, _| {})
        })
        .unwrap()
    }

    /// The text of the reply to `message`.
    fn ask(
        harness: &mut Harness<Conversation<crate::providers::dwim::Remote>>,
        message: &str,
    ) -> String {
        let mut text = String::new();
        harness
            .send(message, |event| {
                if let Event::Text(piece) = event {
                    text.push_str(&piece);
                }
                ControlFlow::Continue(())
            })
            .unwrap();
        text.trim().to_string()
    }

    #[test]
    fn the_sessions_of_an_engine_share_the_model_loaded_in_the_program() {
        let (embedded, started, _) = engine(&["</think>\n\nOne.", "</think>\n\nTwo."]);
        let settings = settings(&embedded);
        let mut first = harness(&embedded, &settings);
        assert_eq!(ask(&mut first, "first"), "One.");
        // The clone in the settings is the same engine, with the model
        // loaded once.
        let Engine::Embedded(clone) = &settings.engine else {
            unreachable!()
        };
        let mut second = harness(clone, &settings);
        assert_eq!(ask(&mut second, "second"), "Two.");
        assert_eq!(started.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn an_agent_opened_on_one_thread_answers_on_another() {
        let (embedded, _, _) = engine(&["</think>\n\nOne."]);
        let mut harness = harness(&embedded, &settings(&embedded));
        let answer = thread::spawn(move || ask(&mut harness, "first"));
        assert_eq!(answer.join().unwrap(), "One.");
    }

    #[test]
    fn the_model_goes_when_the_engine_and_its_sessions_are_let_go_of() {
        let (embedded, _, ended) = engine(&["</think>\n\nOne."]);
        let settings = settings(&embedded);
        let mut harness = harness(&embedded, &settings);
        drop((embedded, settings));
        // The session still has the model.
        assert_eq!(ask(&mut harness, "first"), "One.");
        assert_eq!(ended.load(Ordering::Relaxed), 0);
        drop(harness);
        assert!(reaches(&ended, 1));
    }

    #[test]
    fn an_engine_asked_for_another_model_is_replaced_once_it_has_no_sessions() {
        let (embedded, started, ended) = engine(&["</think>\n\nOne."]);
        let settings = settings(&embedded);
        let other = Settings {
            context: Some(1 << 10),
            ..settings.clone()
        };
        let mut first = harness(&embedded, &settings);
        let error = embedded.connect(&other).err().unwrap().to_string();
        assert!(error.contains("1 other session"), "{error}");
        // The session it has goes on as it was.
        assert_eq!(ask(&mut first, "first"), "One.");
        drop(first);
        let mut second = harness(&embedded, &other);
        // The model it had went before the next was loaded.
        assert_eq!(ended.load(Ordering::Relaxed), 1);
        assert_eq!(ask(&mut second, "second"), "One.");
        assert_eq!(started.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn a_session_is_told_when_the_model_cannot_be_loaded() {
        let started = Arc::new(AtomicUsize::new(0));
        let embedded = Embedded {
            running: Arc::default(),
            stand_in: Some(Arc::new({
                let started = started.clone();
                move |server: Server| {
                    started.fetch_add(1, Ordering::Relaxed);
                    server.fail("the weights are not there");
                }
            })),
        };
        let settings = settings(&embedded);
        for attempt in 1..=2 {
            let error = embedded
                .connect(&settings)
                .unwrap()
                .open(None, 0, |_| {})
                .err()
                .unwrap()
                .to_string();
            assert_eq!(error, "the weights are not there");
            assert_eq!(started.load(Ordering::Relaxed), attempt);
            // Once the engine's thread is over, the next session has the
            // model loaded afresh.
            while !embedded
                .running
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .thread
                .is_finished()
            {
                thread::sleep(POLL);
            }
        }
    }
}
