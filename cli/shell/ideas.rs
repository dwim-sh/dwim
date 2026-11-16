//! Ideas while the user types: before a message is sent, the model says in
//! a few marked lines what it would do with the draft so far, what the
//! draft leaves open, and what is risky in it, so that the user can settle
//! it by rewording, before anything runs. It runs on a thread and a session of its own,
//! with a short prompt of its own and no thinking, so that it answers in
//! about a second; a draft that changes while it answers ends the answer,
//! and the newer draft is read instead.

use std::{
    env,
    ops::ControlFlow,
    process::Command,
    sync::mpsc::{self, Receiver, Sender, TryRecvError},
    thread,
};

use dwim::harness;
use dwim::model::Session as _;

use crate::session::{self, Session, Settings};

/// Most tokens of an idea.
const LIMIT: usize = 120;

/// Most of the project's file names the prompt lists.
const MAX_FILES: usize = 400;

/// How the model is asked for ideas, ahead of where it is and the files.
const PROMPT: &str = "You watch a developer type a request to dwim, a coding agent working in the project below. Before they send it, tell them what they cannot see from what they wrote: what dwim would do, and what in the request is open. Never restate the request: they know what they asked. Reply with at most three lines of at most fifteen words each, and nothing else, each starting with one of these marks:\n\
→ what dwim would do, as steps beginning with a verb, naming the files the file list makes likely, and ending with what it hands back: a report, or changed files and how they are checked\n\
? something left open that would change the work, and the choice dwim makes if not told (\"? fix them too, or only report? assumes report\")\n\
! a risk, only where dwim would change something: a change that is large, hard to undo, or outside the project\n\
Leave out a mark with nothing to say, never write one with \"none\": a clear, safe request gets a single → line. No preamble, no code, no promises. A draft cut off mid-word is still being typed: read what it is heading for.";

/// What the thread says of a draft: the idea so far, and whether it is
/// whole.
pub struct Idea {
    pub draft: u64,
    pub text: String,
    pub done: bool,
}

/// The thread that has ideas, and the way to and from it.
pub struct Ideas {
    drafts: Sender<(u64, String)>,
    ideas: Receiver<Idea>,
}

impl Ideas {
    /// Starts the thread, which opens its session where `settings` have
    /// the model when it is first given a draft.
    pub fn spawn(settings: Settings) -> Self {
        let (drafts, drafts_rx) = mpsc::channel();
        let (ideas_tx, ideas) = mpsc::channel();
        thread::spawn(move || work(&settings, &drafts_rx, &ideas_tx));
        Self { drafts, ideas }
    }

    /// Gives the thread draft number `draft`, `text`, to read in place of
    /// any it was given before.
    pub fn read(&self, draft: u64, text: &str) {
        let _ = self.drafts.send((draft, text.to_string()));
    }

    /// The next thing the thread has to say, if there is anything.
    pub fn try_recv(&self) -> Option<Idea> {
        self.ideas.try_recv().ok()
    }
}

/// Reads drafts as they come, the latest of them, until the shell hangs
/// up, telling `ideas` the idea for each as it grows.
fn work(settings: &Settings, drafts: &Receiver<(u64, String)>, ideas: &Sender<Idea>) {
    let system = prompt();
    let mut session: Option<Session> = None;
    let mut next: Option<(u64, String)> = None;
    loop {
        let mut draft = match next.take() {
            Some(draft) => draft,
            None => match drafts.recv() {
                Ok(draft) => draft,
                Err(_) => return,
            },
        };
        while let Ok(newer) = drafts.try_recv() {
            draft = newer;
        }
        let (number, text) = draft;
        let failed = |error: String| {
            let _ = ideas.send(Idea {
                draft: number,
                text: format!("(no idea: {error})"),
                done: true,
            });
        };
        if session.is_none() {
            match session::open_with(settings, system.clone(), |_| {}) {
                Ok((opened, _)) => session = Some(opened),
                Err(e) => {
                    failed(e.to_string());
                    continue;
                }
            }
        }
        let Some(open) = session.as_mut() else {
            continue;
        };
        // Each draft is read afresh: the last one is no part of it.
        if let Err(e) = open.clear().and_then(|()| open.start(&system, |_, _| {})) {
            failed(e.to_string());
            session = None;
            continue;
        }
        open.limit_thoughts(Some(0), "");
        let mut idea = String::new();
        let answered = open.answer(&format!("Draft so far:\n{text}"), &[], LIMIT, |piece| {
            match drafts.try_recv() {
                Ok(newer) => {
                    next = Some(newer);
                    return ControlFlow::Break(());
                }
                Err(TryRecvError::Disconnected) => return ControlFlow::Break(()),
                Err(TryRecvError::Empty) => {}
            }
            idea.push_str(piece);
            let _ = ideas.send(Idea {
                draft: number,
                text: idea.trim().to_string(),
                done: false,
            });
            ControlFlow::Continue(())
        });
        match answered {
            Ok(_) if next.is_none() => {
                let _ = ideas.send(Idea {
                    draft: number,
                    text: idea.trim().to_string(),
                    done: true,
                });
            }
            Ok(_) => {}
            Err(e) => {
                failed(e.to_string());
                session = None;
            }
        }
    }
}

/// The prompt ideas are asked under: how to answer, where the agent is,
/// and the files of the project, as git has them, so that an idea can
/// name where the work would land.
fn prompt() -> String {
    let dir = env::current_dir().unwrap_or_default();
    let mut prompt = format!("{PROMPT}\n\n{}", harness::environment(&dir));
    let files = Command::new("git")
        .args(["ls-files"])
        .current_dir(&dir)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).into_owned())
        .unwrap_or_default();
    if !files.is_empty() {
        let files: Vec<&str> = files.lines().take(MAX_FILES).collect();
        prompt.push_str("\n\nFiles in the project:\n");
        prompt.push_str(&files.join("\n"));
    }
    prompt
}
