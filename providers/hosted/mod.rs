//! A model at a host, asked over its chat completions API: OpenRouter,
//! Vercel's AI Gateway, or Cerebras. The first two pass the request on to
//! a provider of the model, and Cerebras runs it itself. What is asked of
//! each in its own way is its [`Host`]'s.
//!
//! The conversation goes as messages, and the provider writes the prompt
//! the model reads from them. The harness declares the tools in the
//! system prompt, as the model's chat template writes them; here they are
//! taken out of it and declared to the provider, whose template writes the
//! same declaration back. A provider that is not told of the tools may
//! drop a call the model writes, taking it for one and having nowhere to
//! put it. A reply's calls go as the calls they were, and their results as
//! messages from the tools.
//!
//! A reply's thought is kept with it, and goes back with the replies made
//! since the user's last message, as the model's chat template writes
//! them: asked again after a tool call without what it had thought, the
//! model works its plan out over again, at length, and calls another tool
//! in place of acting on it.
//!
//! What only a server of dwim's own can do is done without. A closing
//! cannot be written into a thought, and a host is asked how much the
//! model reasons as an effort, not as a number of tokens, so a thought
//! that runs to its limit is ended here: the reply is dropped where it
//! is, and asked for again without a thought, with the thought so far
//! and the closing put to the model as a reminder to act on. A thought
//! it has all the same is held to the limit as the first was, a few
//! times over. A plain answer is asked to write its headings itself, and
//! tokens are counted by what the provider said the last request took,
//! with an estimate of what was added since.

pub mod key;
mod stream;

use std::{
    io::BufReader,
    ops::ControlFlow,
    thread,
    time::{Duration, Instant},
};

use serde_json::{Value, json};
use ureq::Agent;

use crate::{
    model::{
        Answer, Chat, Chunk, Effort, Generation, Model, Reply, Result, Sampling, State, Stats, Turn,
    },
    providers::{Completion, Kind, Provider},
    tools::{ToolCall, declared},
};

use stream::{Outcome, Usage};

/// Who a model is hosted by: where it is asked, and what is asked there
/// in the host's own way.
pub struct Host {
    /// Which provider of a session's model this is.
    pub kind: Kind,
    /// The host as it is named to the user.
    pub name: &'static str,
    /// Where a chat is completed.
    pub url: &'static str,
    /// A model as the host names it.
    pub model: fn(Model) -> &'static str,
    /// Adds to a request for a reply what the host is asked in its own
    /// way: to say what the reply used, how the tokens are sampled beyond
    /// what every host takes, and the `effort` the model reasons at, or
    /// not to reason when there is none. The host writes the effort's
    /// line into the system prompt as the model's chat template does, and
    /// the line for `xhigh` when it is not asked, so it is always asked.
    pub ask: fn(body: &mut Value, sampling: &Sampling, effort: Option<Effort>),
    /// The environment variable the key to the host is read from, ahead
    /// of the key kept from signing in.
    pub key: &'static str,
    /// What to do about a key the host refuses.
    pub refused: &'static str,
}

/// Tokens of conversation a session has room for unless told: half of
/// what the models were trained with. Every request sends the whole
/// conversation, and is charged for it, so a conversation compacted
/// sooner costs less.
pub const CONTEXT: usize = 131_072;

/// Most tokens a reply may run to, which every provider of the models
/// allows.
const MAX_REPLY: usize = 32_768;

/// Tokens a plain answer may think for, beside the tokens of the answer:
/// the provider counts both against the most a reply may run to.
const ANSWER_THOUGHT: usize = 2048;

/// Most times the thought of a reply is ended at its limit, which is
/// more than once only where a provider has the model think when it is
/// asked not to. The thought after the last is let run, for the reply
/// that has to come of it.
const ENDINGS: usize = 3;

/// Bytes of a thought to a token, about, as prose runs: a thought is
/// counted against its limit as it streams, in text.
const THOUGHT_BYTES: usize = 4;

/// How often a request that was refused for now, or did not get through,
/// is made again, and how long it waits first.
const RETRIES: [Duration; 2] = [Duration::from_secs(1), Duration::from_secs(4)];

/// A model at a host.
pub struct Hosted {
    host: &'static Host,
    agent: Agent,
    /// Where a chat is completed: the host's address.
    url: String,
    key: String,
    model: Model,
    /// How much the model reasons before it replies.
    effort: Effort,
    capacity: usize,
    sampling: Sampling,
    stats: Stats,
    /// The conversation the provider last counted, and its tokens.
    counted: Option<(Chat, usize)>,
}

impl Hosted {
    /// `model` at `host`, asked with `key`, reasoning at
    /// `effort`, with room for `context` tokens of conversation, or
    /// [`CONTEXT`], sampled at `temperature`, or as the model's makers
    /// recommend.
    pub fn new(
        host: &'static Host,
        key: String,
        model: Model,
        effort: Effort,
        context: Option<usize>,
        temperature: Option<f32>,
    ) -> Self {
        let agent = Agent::config_builder()
            .http_status_as_error(false)
            .timeout_connect(Some(Duration::from_secs(30)))
            .build()
            .new_agent();
        Self {
            host,
            agent,
            url: host.url.to_string(),
            key,
            model,
            effort,
            capacity: context.unwrap_or(CONTEXT),
            sampling: Sampling {
                temperature: temperature.unwrap_or(model.sampling().temperature),
                ..model.sampling()
            },
            stats: Stats::default(),
            counted: None,
        }
    }

    /// Who hosts the model.
    pub fn host(&self) -> &'static Host {
        self.host
    }

    /// The model as its host names it.
    pub fn model(&self) -> &'static str {
        (self.host.model)(self.model)
    }

    /// Tokens in `chat`: what the provider counted of it, and an estimate
    /// of the turns since.
    fn count(&self, chat: &Chat) -> usize {
        match &self.counted {
            Some((counted, tokens))
                if counted.system == chat.system && chat.turns.starts_with(&counted.turns) =>
            {
                tokens
                    + chat.turns[counted.turns.len()..]
                        .iter()
                        .map(estimate)
                        .sum::<usize>()
            }
            _ => tokens_of(&chat.system) + chat.turns.iter().map(estimate).sum::<usize>(),
        }
    }

    /// Asks for a reply to `messages`, with the tools they declare, of at
    /// most `limit` tokens, reasoned at `effort`, streaming it to
    /// `on_chunk`, and counts what it took. A thought that runs to the
    /// tokens of `ending`, if it has an ending, is ended with the closing
    /// there, as the model's own: the reply is asked for again, without
    /// a thought, to go on from the thought so far and the closing. A
    /// thought of that reply, if it has one all the same, is ended the
    /// same way, [`ENDINGS`] times in all.
    fn complete(
        &mut self,
        messages: (Vec<Value>, Vec<Value>),
        limit: usize,
        effort: Effort,
        ending: Option<(usize, &str)>,
        mut on_chunk: impl FnMut(Chunk) -> ControlFlow<()>,
    ) -> Result<Outcome> {
        let (messages, tools) = messages;
        let mut body = json!({
            "model": self.model(),
            "messages": messages,
            "stream": true,
            "max_tokens": limit.clamp(1, MAX_REPLY),
            "temperature": self.sampling.temperature,
            "top_p": self.sampling.top_p,
        });
        if !tools.is_empty() {
            body["tools"] = Value::Array(tools);
        }
        (self.host.ask)(&mut body, &self.sampling, Some(effort));
        let most = ending.map(|(tokens, _)| tokens * THOUGHT_BYTES);
        let asked = messages
            .last()
            .and_then(|last| last["content"].as_str())
            .unwrap_or("")
            .to_string();
        let mut thought = String::new();
        let mut endings = 0;
        loop {
            // A reply has to come of the thought, so the one after the
            // last ending is let run.
            let most = most.filter(|_| endings < ENDINGS);
            let (outcome, over) = self.ask_within(&body, most, &mut on_chunk)?;
            if endings > 0 && !outcome.thought.trim().is_empty() {
                thought.push_str("\n\n");
            }
            let Some((_, closing)) = ending.filter(|_| over) else {
                if endings == 0 {
                    return Ok(outcome);
                }
                thought.push_str(outcome.thought.trim());
                return Ok(Outcome {
                    thought,
                    thought_ended: true,
                    ..outcome
                });
            };
            endings += 1;
            thought = format!("{thought}{}\n\n{closing}", outcome.thought.trim());
            if on_chunk(Chunk::Thought(&format!("\n\n{closing}\n"))).is_break() {
                return Ok(Outcome {
                    thought,
                    thought_ended: true,
                    ..outcome
                });
            }
            // The model is asked not to think again. A provider that
            // has it think all the same has that thought held to the
            // limit as the last was.
            if let Some(last) = body["messages"]
                .as_array_mut()
                .and_then(|messages| messages.last_mut())
            {
                last["content"] = Value::String(format!("{asked}\n\n{}", ended(&thought)));
            }
            (self.host.ask)(&mut body, &self.sampling, None);
        }
    }

    /// Asks for a reply as [`Hosted::ask`] does, and drops it where its
    /// thought runs to `most` bytes, if they are limited: what there was
    /// of the reply, and whether it was dropped there.
    fn ask_within(
        &mut self,
        body: &Value,
        most: Option<usize>,
        on_chunk: &mut impl FnMut(Chunk) -> ControlFlow<()>,
    ) -> Result<(Outcome, bool)> {
        let mut bytes = 0;
        let mut over = false;
        let outcome = self.ask(body, |chunk| {
            let thinking = matches!(chunk, Chunk::Thought(_));
            if let Chunk::Thought(piece) = &chunk {
                bytes += piece.len();
            }
            on_chunk(chunk)?;
            if thinking && most.is_some_and(|most| bytes >= most) {
                over = true;
                return ControlFlow::Break(());
            }
            ControlFlow::Continue(())
        })?;
        Ok((outcome, over))
    }

    /// Posts a request for a reply, reads the reply as it streams to
    /// `on_chunk`, and counts what the provider says it took: a reply
    /// ended early is not counted, since the provider does not get to say.
    fn ask(
        &mut self,
        body: &Value,
        on_chunk: impl FnMut(Chunk) -> ControlFlow<()>,
    ) -> Result<Outcome> {
        let started = Instant::now();
        let response = self.post(&body.to_string())?;
        let waited = started.elapsed().as_secs_f64();
        let outcome = stream::read(BufReader::new(response), self.host.name, on_chunk)?;
        if let Some(usage) = outcome.usage {
            let generating = started.elapsed().as_secs_f64() - waited;
            let share = usage.reasoning as f64 / usage.completion.max(1) as f64;
            let stats = &mut self.stats;
            stats.cached.tokens += usage.cached;
            stats.prompt.tokens += usage.prompt - usage.cached.min(usage.prompt);
            stats.prompt.seconds += waited;
            stats.thought.tokens += usage.reasoning;
            stats.thought.seconds += generating * share;
            stats.answer.tokens += usage.completion - usage.reasoning.min(usage.completion);
            stats.answer.seconds += generating * (1.0 - share);
            stats.cost += usage.cost;
        }
        Ok(outcome)
    }

    /// Posts a request, again after a wait if it is refused for now or
    /// does not get through, and returns the body of the answer, which is
    /// the reply as it streams.
    fn post(&self, body: &str) -> Result<impl std::io::Read + use<>> {
        let name = self.host.name;
        let mut retries = RETRIES.iter();
        loop {
            let sent = self
                .agent
                .post(&self.url)
                .header("Authorization", &format!("Bearer {}", self.key))
                .header("Content-Type", "application/json")
                .header("X-Title", "dwim")
                .send(body);
            let failure = match sent {
                Ok(response) if response.status().is_success() => {
                    return Ok(response.into_body().into_reader());
                }
                Ok(response) => {
                    let status = response.status();
                    let text = response.into_body().read_to_string().unwrap_or_default();
                    let message = match serde_json::from_str::<Value>(&text) {
                        Ok(answer) if answer.get("error").is_some() => {
                            stream::error_message(name, &answer["error"])
                        }
                        _ => format!("{name}: {status}"),
                    };
                    if status.as_u16() == 401 {
                        return Err(format!(
                            "{message} · the key to {name} was refused: {}",
                            self.host.refused
                        )
                        .into());
                    }
                    if !(status.as_u16() == 429 || status.is_server_error()) {
                        return Err(message.into());
                    }
                    message
                }
                Err(e) => format!("{name}: {e}"),
            };
            match retries.next() {
                Some(wait) => thread::sleep(*wait),
                None => return Err(failure.into()),
            }
        }
    }

    /// Where `chat` is once `reply` follows it: counted by the provider
    /// if the reply's stream said what it `used`, and estimated otherwise,
    /// as when the reply was ended early.
    fn after(&mut self, chat: &Chat, reply: Turn, used: Option<Usage>) -> State {
        let kept = matches!(&reply, Turn::Reply { thought, .. } if !thought.is_empty());
        let tokens = match used {
            // A thought the reply keeps goes back with the turn's
            // requests; one it does not is no part of what goes on.
            Some(usage) if kept => usage.prompt + usage.completion,
            Some(usage) => usage.prompt + usage.completion - usage.reasoning.min(usage.completion),
            None => self.count(chat) + estimate(&reply),
        };
        let mut counted = chat.clone();
        counted.turns.push(reply);
        self.counted = Some((counted, tokens));
        State {
            tokens,
            stats: self.stats,
        }
    }
}

impl Provider for Hosted {
    fn capacity(&self) -> usize {
        self.capacity
    }

    fn sampling(&self) -> Sampling {
        self.sampling
    }

    fn measure(&mut self, turn: &Turn) -> Result<usize> {
        Ok(estimate(turn))
    }

    fn answer_tokens(&mut self, message: &str, limit: usize) -> Result<usize> {
        Ok(estimate(&Turn::User(message.to_string())) + limit + FRAMING)
    }

    /// Nothing is read ahead of a request: where the conversation is said
    /// to be is what the provider last counted, and an estimate of the
    /// rest.
    fn read(&mut self, chat: &Chat, _: impl FnMut(usize, usize)) -> Result<State> {
        Ok(State {
            tokens: self.count(chat),
            stats: self.stats,
        })
    }

    fn reply(
        &mut self,
        chat: &Chat,
        generation: &Generation,
        on_chunk: impl FnMut(Chunk) -> ControlFlow<()>,
    ) -> Result<Completion> {
        let room = self
            .capacity
            .saturating_sub(self.count(chat) + generation.reserve);
        if room == 0 {
            // The reply is stopped short before it starts, as one that
            // ran out of room is, for the conversation to be compacted.
            return Ok(Completion {
                reply: Reply {
                    cut: true,
                    ..Reply::default()
                },
                text: String::new(),
                thought: String::new(),
                state: State {
                    tokens: self.count(chat),
                    stats: self.stats,
                },
            });
        }
        let outcome = self.complete(
            messages(chat),
            room,
            self.effort,
            generation.thought_limit.zip(generation.closing.as_deref()),
            on_chunk,
        )?;
        let reply = Turn::Reply {
            text: outcome.text.clone(),
            calls: outcome.calls.clone(),
            thought: outcome.thought.clone(),
        };
        let state = self.after(chat, reply, outcome.usage);
        Ok(Completion {
            reply: Reply {
                calls: outcome.calls,
                cut: outcome.cut,
                thought_ended: outcome.thought_ended,
            },
            text: outcome.text,
            thought: outcome.thought,
            state,
        })
    }

    fn answer(
        &mut self,
        chat: &Chat,
        heads: &[&str],
        limit: usize,
        _: &Generation,
        mut on_text: impl FnMut(&str) -> ControlFlow<()>,
    ) -> Result<(Answer, State)> {
        // A plain answer calls no tools, so none are declared.
        let (mut messages, _) = messages(chat);
        // The headings cannot be written into the answer for the model, so
        // it is asked to write them.
        if !heads.is_empty()
            && let Some(last) = messages.last_mut()
        {
            let heads: Vec<&str> = heads.iter().map(|head| head.trim()).collect();
            let asked = format!(
                "{}\n\nWrite each of these headings yourself, at the start of a line, in this order, each followed by its section: {}",
                last["content"].as_str().unwrap_or(""),
                heads.join(" ")
            );
            last["content"] = Value::String(asked);
        }
        // A plain answer's thought has no closing to be ended with, so
        // it is asked to be brief, whatever the session reasons at, and
        // not to take the answer's tokens.
        let outcome = self.complete(
            (messages, Vec::new()),
            limit + ANSWER_THOUGHT,
            Effort::Low,
            None,
            |chunk| match chunk {
                Chunk::Text(text) => on_text(text),
                _ => ControlFlow::Continue(()),
            },
        )?;
        // A plain answer stands without its thought.
        let reply = Turn::Reply {
            text: outcome.text.clone(),
            calls: Vec::new(),
            thought: String::new(),
        };
        let state = self.after(chat, reply, outcome.usage);
        let answer = Answer {
            text: outcome.text,
            cut: outcome.cut,
        };
        Ok((answer, state))
    }
}

/// Tokens a turn's framing takes, about: the tags around it, and its role.
const FRAMING: usize = 8;

/// About how many tokens `turn` takes: its text at three bytes a token,
/// which code and prose both come in under, so that room is run out of
/// early rather than late.
fn estimate(turn: &Turn) -> usize {
    FRAMING
        + match turn {
            Turn::User(message) => tokens_of(message),
            Turn::Reply {
                text,
                calls,
                thought,
            } => {
                tokens_of(thought)
                    + tokens_of(text)
                    + calls.iter().map(|call| tokens_of(call) + 2).sum::<usize>()
            }
            Turn::Results(outputs) => outputs.iter().map(|output| tokens_of(output) + 4).sum(),
        }
}

fn tokens_of(text: &str) -> usize {
    text.len().div_ceil(3)
}

/// What a model whose thought was ended at its limit is told in place of
/// thinking again: the `thought`, with its closing, which it goes on from.
fn ended(thought: &str) -> String {
    format!(
        "<system-reminder>Your thinking for this reply reached its limit and was ended. This is what you thought:\n\n{thought}\n\nDo not think it over again. Act on it now: call a tool, or reply.</system-reminder>"
    )
}

/// `chat` as the messages of a request, and the tools it declares: the
/// system prompt without its declaration of the tools, and a message for
/// each turn. A reply's calls go as calls, each with a name of its own for
/// its result to answer to, and a turn of results as a message from the
/// tool for each. A reply made since the user's last message goes with
/// the thought behind it, as the model's chat template writes one.
fn messages(chat: &Chat) -> (Vec<Value>, Vec<Value>) {
    let (system, tools) = declared(&chat.system);
    let asked = chat
        .turns
        .iter()
        .rposition(|turn| matches!(turn, Turn::User(_)));
    let mut messages = vec![json!({ "role": "system", "content": system })];
    // The names of the calls of the last reply, for their results.
    let mut ids: Vec<String> = Vec::new();
    for (n, turn) in chat.turns.iter().enumerate() {
        match turn {
            Turn::User(message) => {
                messages.push(json!({ "role": "user", "content": message }));
            }
            Turn::Reply {
                text,
                calls,
                thought,
            } => {
                ids = (0..calls.len()).map(|i| format!("call_{n}_{i}")).collect();
                let mut message = json!({ "role": "assistant", "content": text });
                if !thought.is_empty() && asked.is_none_or(|asked| n > asked) {
                    message["reasoning"] = Value::String(thought.clone());
                }
                if !calls.is_empty() {
                    let calls: Vec<Value> = calls.iter().zip(&ids).map(call).collect();
                    message["tool_calls"] = Value::Array(calls);
                }
                messages.push(message);
            }
            Turn::Results(outputs) => {
                for (i, output) in outputs.iter().enumerate() {
                    messages.push(match ids.get(i) {
                        Some(id) => {
                            json!({ "role": "tool", "tool_call_id": id, "content": output })
                        }
                        // A result of no call the reply made goes as the
                        // model's chat template would have written it.
                        None => json!({
                            "role": "user",
                            "content": format!("<tool_response>\n{output}\n</tool_response>"),
                        }),
                    });
                }
                ids.clear();
            }
        }
    }
    (messages, tools)
}

/// A call the model made, as a message carries one, answered to as `id`.
/// One that cannot be read as a call goes as a call of no tool, with what
/// was written as its argument, so that its result still has a call to
/// answer to.
fn call((written, id): (&String, &String)) -> Value {
    let (name, arguments) = match ToolCall::parse(written) {
        Ok(call) => (call.name, call.arguments),
        Err(_) => ("invalid".to_string(), json!({ "written": written })),
    };
    json!({
        "id": id,
        "type": "function",
        "function": { "name": name, "arguments": arguments.to_string() },
    })
}

#[cfg(test)]
mod tests {
    use std::{
        io::{BufRead, Read, Write},
        net::TcpListener,
    };

    use super::*;
    use crate::providers::openrouter;

    /// A reply as a provider streams it: an event for each delta, and
    /// then what it used.
    fn events(deltas: &[Value]) -> String {
        let mut body = String::new();
        for delta in deltas {
            let event = json!({ "choices": [{ "delta": delta }] });
            body.push_str(&format!("data: {event}\n\n"));
        }
        let usage =
            json!({ "choices": [], "usage": { "prompt_tokens": 10, "completion_tokens": 2 } });
        body.push_str(&format!("data: {usage}\n\ndata: [DONE]\n\n"));
        body
    }

    /// A model at a provider on this machine that answers the requests it
    /// gets with `replies` in turn, and the thread that hands back what
    /// it was asked.
    fn provider(replies: Vec<String>) -> (Hosted, thread::JoinHandle<Vec<Value>>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/", listener.local_addr().unwrap());
        let serving = thread::spawn(move || {
            let mut asked = Vec::new();
            for reply in replies {
                let (stream, _) = listener.accept().unwrap();
                let mut reader = BufReader::new(stream);
                let mut length = 0;
                loop {
                    let mut line = String::new();
                    reader.read_line(&mut line).unwrap();
                    let line = line.trim().to_ascii_lowercase();
                    if let Some(value) = line.strip_prefix("content-length:") {
                        length = value.trim().parse().unwrap();
                    }
                    if line.is_empty() {
                        break;
                    }
                }
                let mut body = vec![0; length];
                reader.read_exact(&mut body).unwrap();
                asked.push(serde_json::from_slice(&body).unwrap());
                // A client that has heard enough is gone before the end.
                let _ = reader.get_mut().write_all(
                    format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n{reply}")
                        .as_bytes(),
                );
            }
            asked
        });
        let mut provider = Hosted::new(
            &openrouter::HOST,
            "key".to_string(),
            Model::QwenMax,
            Effort::Xhigh,
            None,
            None,
        );
        provider.url = url;
        (provider, serving)
    }

    #[test]
    fn a_reply_is_asked_at_the_effort_of_the_session_and_a_plain_answer_at_low() {
        let (mut provider, serving) = provider(vec![
            events(&[json!({ "content": "Done." })]),
            events(&[json!({ "content": "A note." })]),
        ]);
        let chat = Chat {
            system: "Be brief.".to_string(),
            turns: vec![Turn::User("Fix it.".to_string())],
        };
        let generation = Generation {
            reserve: 0,
            thought_limit: Some(2048),
            closing: Some("I will decide now.".to_string()),
        };
        provider
            .reply(&chat, &generation, |_| ControlFlow::Continue(()))
            .unwrap();
        provider
            .answer(&chat, &[], 100, &generation, |_| ControlFlow::Continue(()))
            .unwrap();
        let asked = serving.join().unwrap();

        assert_eq!(asked[0]["reasoning"], json!({ "effort": "xhigh" }));
        assert_eq!(asked[1]["reasoning"], json!({ "effort": "low" }));
    }

    #[test]
    fn a_thought_that_runs_to_its_limit_is_ended_and_the_reply_asked_again() {
        // The provider was asked for a thought of two tokens, and thinks
        // on: the reply is dropped there, and asked for again, from the
        // thought so far and the closing.
        let thinking: Vec<Value> = (0..50)
            .map(|_| json!({ "reasoning": "Let me reconsider. " }))
            .chain([json!({ "content": "Never said." })])
            .collect();
        let (mut provider, serving) = provider(vec![
            events(&thinking),
            events(&[json!({ "content": "Done." })]),
        ]);
        let chat = Chat {
            system: "Be brief.".to_string(),
            turns: vec![Turn::User("Fix it.".to_string())],
        };
        let generation = Generation {
            reserve: 0,
            thought_limit: Some(2),
            closing: Some("I will decide now.".to_string()),
        };
        let mut thoughts = String::new();
        let completion = provider
            .reply(&chat, &generation, |chunk| {
                if let Chunk::Thought(piece) = chunk {
                    thoughts.push_str(piece);
                }
                ControlFlow::Continue(())
            })
            .unwrap();
        let asked = serving.join().unwrap();

        assert_eq!(completion.text, "Done.");
        assert!(completion.reply.thought_ended);
        assert_eq!(
            completion.thought,
            "Let me reconsider.\n\nI will decide now."
        );
        assert_eq!(thoughts, "Let me reconsider. \n\nI will decide now.\n");
        // The second request has the thought and its closing, after the
        // message the reply is to.
        assert_eq!(asked.len(), 2);
        let last = asked[1]["messages"].as_array().unwrap().last().unwrap();
        let content = last["content"].as_str().unwrap();
        assert_eq!(last["role"], "user");
        assert!(content.starts_with("Fix it."), "{content}");
        assert!(
            content.contains("Let me reconsider.\n\nI will decide now."),
            "{content}"
        );
    }

    #[test]
    fn a_thought_of_the_reply_asked_again_is_held_to_the_limit_too() {
        // The provider has the model think past the limit however it
        // is asked: its thought is ended each time, and the last reply
        // is let run.
        let thinking: Vec<Value> = (0..50)
            .map(|_| json!({ "reasoning": "Once more. " }))
            .chain([json!({ "content": "Done." })])
            .collect();
        let (mut provider, serving) = provider(vec![events(&thinking); 4]);
        let chat = Chat {
            system: "Be brief.".to_string(),
            turns: vec![Turn::User("Fix it.".to_string())],
        };
        let generation = Generation {
            reserve: 0,
            thought_limit: Some(2),
            closing: Some("I will decide now.".to_string()),
        };
        let completion = provider
            .reply(&chat, &generation, |_| ControlFlow::Continue(()))
            .unwrap();
        let asked = serving.join().unwrap();

        assert_eq!(completion.text, "Done.");
        assert!(completion.reply.thought_ended);
        // Three thoughts were ended, and none of them ran on.
        assert_eq!(asked.len(), 4);
        assert_eq!(completion.thought.matches("I will decide now.").count(), 3);
        assert!(
            completion
                .thought
                .starts_with("Once more.\n\nI will decide now.\n\nOnce more.\n\nI will"),
            "{}",
            completion.thought
        );
        // Each reply asked again is asked not to think, and told all
        // that was thought, after the one message it is to.
        assert_eq!(asked[0]["reasoning"], json!({ "effort": "xhigh" }));
        for request in &asked[1..] {
            assert_eq!(request["reasoning"], json!({ "effort": "none" }));
        }
        let last = asked[3]["messages"].as_array().unwrap().last().unwrap();
        let content = last["content"].as_str().unwrap();
        assert_eq!(content.matches("Fix it.").count(), 1, "{content}");
        assert_eq!(content.matches("I will decide now.").count(), 3);
    }

    #[test]
    fn a_reply_goes_with_its_thought_until_the_user_says_more() {
        let reply = |thought: &str| Turn::Reply {
            text: String::new(),
            calls: vec![r#"{"name": "bash", "arguments": {"command": "ls"}}"#.to_string()],
            thought: thought.to_string(),
        };
        let mut chat = Chat {
            system: "Be brief.".to_string(),
            turns: vec![
                Turn::User("What is here?".to_string()),
                reply("I will list the files, then read the first."),
                Turn::Results(vec!["a.rs".to_string()]),
            ],
        };
        // Asked again after the tool's result, the model has what it
        // thought before the call.
        let (messages, _) = messages(&chat);
        assert_eq!(
            messages[2]["reasoning"],
            "I will list the files, then read the first."
        );

        // Once the user has said more, the thoughts before are left out,
        // and the ones since go.
        chat.turns.extend([
            Turn::User("And there?".to_string()),
            reply("The same again, elsewhere."),
            Turn::Results(vec!["b.rs".to_string()]),
        ]);
        let (messages, _) = super::messages(&chat);
        assert_eq!(messages[2]["role"], "assistant");
        assert!(messages[2].get("reasoning").is_none());
        assert_eq!(messages[5]["reasoning"], "The same again, elsewhere.");
    }

    /// A system prompt that declares a tool as the harness does.
    const SYSTEM: &str = "# Tools\n\nYou have access to the following functions:\n\n<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"bash\", \"description\": \"Run.\", \"parameters\": {\"type\": \"object\", \"properties\": {}, \"required\": []}}}\n</tools>\n\nIf you choose to call a function ONLY reply in the following format:\n\n<tool_call>\n</tool_call>\n\n<IMPORTANT>\nReminder\n</IMPORTANT>\n\nBe brief.";

    #[test]
    fn the_tools_are_declared_to_the_provider_and_not_twice() {
        let chat = Chat {
            system: SYSTEM.to_string(),
            turns: Vec::new(),
        };
        let (messages, tools) = messages(&chat);
        assert_eq!(messages[0]["content"], "Be brief.");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["function"]["name"], "bash");
    }

    #[test]
    fn a_reply_goes_with_its_calls_and_each_result_answers_its_call() {
        let chat = Chat {
            system: "Be brief.".to_string(),
            turns: vec![
                Turn::User("What is here?".to_string()),
                Turn::Reply {
                    text: "Looking.".to_string(),
                    calls: vec![
                        "\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>\n"
                            .to_string(),
                        r#"{"name": "read", "arguments": {"path": "a.rs"}}"#.to_string(),
                    ],
                    thought: String::new(),
                },
                Turn::Results(vec!["a.rs".to_string(), "fn main() {}".to_string()]),
            ],
        };
        let (messages, tools) = messages(&chat);
        assert!(tools.is_empty());
        assert_eq!(messages[0]["content"], "Be brief.");
        assert_eq!(messages[2]["content"], "Looking.");
        let calls = messages[2]["tool_calls"].as_array().unwrap();
        let said: Vec<(&str, Value)> = calls
            .iter()
            .map(|call| {
                let function = &call["function"];
                let arguments = function["arguments"].as_str().unwrap();
                (
                    function["name"].as_str().unwrap(),
                    serde_json::from_str(arguments).unwrap(),
                )
            })
            .collect();
        assert_eq!(
            said,
            [
                ("bash", json!({ "command": "ls" })),
                ("read", json!({ "path": "a.rs" })),
            ]
        );
        // Each result is the tool's, and answers the call it is of.
        for (result, call) in messages[3..].iter().zip(calls) {
            assert_eq!(result["role"], "tool");
            assert_eq!(result["tool_call_id"], call["id"]);
        }
        assert_eq!(messages[4]["content"], "fn main() {}");
    }
}
