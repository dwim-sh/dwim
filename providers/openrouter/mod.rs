//! A model at OpenRouter, asked over its chat completions API.
//!
//! The conversation goes as messages, and OpenRouter's provider writes the
//! prompt the model reads from them. The harness declares the tools in the
//! system prompt, as the model's chat template writes them; here they are
//! taken out of it and declared to the provider, whose template writes the
//! same declaration back. A provider that is not told of the tools may
//! drop a call the model writes, taking it for one and having nowhere to
//! put it. A reply's calls go as the calls they were, and their results as
//! messages from the tools.
//!
//! What only a server of dwim's own can do is done without: a thought is
//! limited by asking the provider to, with no closing written into it, a
//! plain answer is asked to write its headings itself, and tokens are
//! counted by what the provider said the last request took, with an
//! estimate of what was added since.

mod stream;

use std::{
    env,
    io::BufReader,
    ops::ControlFlow,
    thread,
    time::{Duration, Instant},
};

use serde_json::{Value, json};
use ureq::Agent;

use crate::{
    model::{
        Answer, Chat, Chunk, Generation, Reply, Result, Sampling, State, Stats, Turn, Variant,
    },
    providers::{Completion, Provider},
    tools::ToolCall,
};

use stream::Outcome;

/// The environment variable the key to OpenRouter is read from.
pub const KEY: &str = "OPENROUTER_API_KEY";

/// Where a chat is completed.
const URL: &str = "https://openrouter.ai/api/v1/chat/completions";

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

/// How often a request that was refused for now, or did not get through,
/// is made again, and how long it waits first.
const RETRIES: [Duration; 2] = [Duration::from_secs(1), Duration::from_secs(4)];

/// A model at OpenRouter.
pub struct OpenRouter {
    agent: Agent,
    key: String,
    variant: Variant,
    capacity: usize,
    sampling: Sampling,
    stats: Stats,
    /// The conversation the provider last counted, and its tokens.
    counted: Option<(Chat, usize)>,
}

impl OpenRouter {
    /// The model `variant` at OpenRouter, with the key in the environment,
    /// with room for `context` tokens of conversation, or [`CONTEXT`],
    /// sampled at `temperature`, or as the model's makers recommend.
    pub fn new(variant: Variant, context: Option<usize>, temperature: Option<f32>) -> Result<Self> {
        let key = key().ok_or_else(|| format!("{KEY} is not set: it is the key to OpenRouter"))?;
        let agent = Agent::config_builder()
            .http_status_as_error(false)
            .timeout_connect(Some(Duration::from_secs(30)))
            .build()
            .new_agent();
        Ok(Self {
            agent,
            key,
            variant,
            capacity: context.unwrap_or(CONTEXT),
            sampling: Sampling {
                temperature: temperature.unwrap_or(0.6),
                top_k: 20,
                top_p: 0.95,
                min_p: 0.0,
                seed: 0,
            },
            stats: Stats::default(),
            counted: None,
        })
    }

    /// The model as OpenRouter names it.
    pub fn model(&self) -> &'static str {
        model(self.variant)
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
    /// most `limit` tokens, thinking
    /// for at most `thought` of them if that is limited, streaming it to
    /// `on_chunk`, and counts what it took.
    fn complete(
        &mut self,
        messages: (Vec<Value>, Vec<Value>),
        limit: usize,
        thought: Option<usize>,
        on_chunk: impl FnMut(Chunk) -> ControlFlow<()>,
    ) -> Result<Outcome> {
        let (messages, tools) = messages;
        let mut body = json!({
            "model": self.model(),
            "messages": messages,
            "stream": true,
            "usage": { "include": true },
            "max_tokens": limit.clamp(1, MAX_REPLY),
            "temperature": self.sampling.temperature,
            "top_p": self.sampling.top_p,
            "top_k": self.sampling.top_k,
        });
        if !tools.is_empty() {
            body["tools"] = Value::Array(tools);
        }
        if let Some(thought) = thought {
            body["reasoning"] = json!({ "max_tokens": thought });
        }
        let started = Instant::now();
        let response = self.post(&body.to_string())?;
        let waited = started.elapsed().as_secs_f64();
        let outcome = stream::read(BufReader::new(response), on_chunk)?;
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
        let mut retries = RETRIES.iter();
        loop {
            let sent = self
                .agent
                .post(URL)
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
                            stream::error_message(&answer["error"])
                        }
                        _ => format!("OpenRouter: {status}"),
                    };
                    if !(status.as_u16() == 429 || status.is_server_error()) {
                        return Err(message.into());
                    }
                    message
                }
                Err(e) => format!("OpenRouter: {e}"),
            };
            match retries.next() {
                Some(wait) => thread::sleep(*wait),
                None => return Err(failure.into()),
            }
        }
    }

    /// Where `chat` is once a reply of `text` and `calls` follows it:
    /// counted by the provider if the reply's stream said, and estimated
    /// otherwise, as when the reply was ended early.
    fn after(&mut self, chat: &Chat, outcome: &Outcome) -> State {
        let reply = Turn::Reply {
            text: outcome.text.clone(),
            calls: outcome.calls.clone(),
        };
        let tokens = match outcome.usage {
            // The thought is not part of the conversation that goes on.
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

impl Provider for OpenRouter {
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
                state: State {
                    tokens: self.count(chat),
                    stats: self.stats,
                },
            });
        }
        let outcome = self.complete(messages(chat), room, generation.thought_limit, on_chunk)?;
        let state = self.after(chat, &outcome);
        Ok(Completion {
            reply: Reply {
                calls: outcome.calls,
                cut: outcome.cut,
                thought_ended: false,
            },
            text: outcome.text,
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
        let outcome = self.complete(
            (messages, Vec::new()),
            limit + ANSWER_THOUGHT,
            Some(ANSWER_THOUGHT),
            |chunk| match chunk {
                Chunk::Text(text) => on_text(text),
                _ => ControlFlow::Continue(()),
            },
        )?;
        let state = self.after(chat, &outcome);
        let answer = Answer {
            text: outcome.text,
            cut: outcome.cut,
        };
        Ok((answer, state))
    }
}

/// `variant` as OpenRouter names it.
pub fn model(variant: Variant) -> &'static str {
    match variant {
        Variant::Medium => "qwen/qwen3.8-27b",
        Variant::Large => "qwen/qwen3.8-2.4t-a95b",
    }
}

/// The key to OpenRouter, if the environment has one.
pub fn key() -> Option<String> {
    env::var(KEY).ok().filter(|key| !key.trim().is_empty())
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
            Turn::Reply { text, calls } => {
                tokens_of(text) + calls.iter().map(|call| tokens_of(call) + 2).sum::<usize>()
            }
            Turn::Results(outputs) => outputs.iter().map(|output| tokens_of(output) + 4).sum(),
        }
}

fn tokens_of(text: &str) -> usize {
    text.len().div_ceil(3)
}

/// `chat` as the messages of a request, and the tools it declares: the
/// system prompt without its declaration of the tools, and a message for
/// each turn. A reply's calls go as calls, each with a name of its own for
/// its result to answer to, and a turn of results as a message from the
/// tool for each.
fn messages(chat: &Chat) -> (Vec<Value>, Vec<Value>) {
    let (system, tools) = declared(&chat.system);
    let mut messages = vec![json!({ "role": "system", "content": system })];
    // The names of the calls of the last reply, for their results.
    let mut ids: Vec<String> = Vec::new();
    for (n, turn) in chat.turns.iter().enumerate() {
        match turn {
            Turn::User(message) => {
                messages.push(json!({ "role": "user", "content": message }));
            }
            Turn::Reply { text, calls } => {
                ids = (0..calls.len()).map(|i| format!("call_{n}_{i}")).collect();
                let mut message = json!({ "role": "assistant", "content": text });
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

/// What opens and closes the harness's declaration of the tools in a
/// system prompt, and the signatures in it, a line each.
const DECLARATION: (&str, &str) = ("# Tools\n", "</IMPORTANT>");
const SIGNATURES: (&str, &str) = ("<tools>\n", "\n</tools>");

/// A system prompt without its declaration of the tools, and the tools it
/// declared, as a request declares them. A prompt that declares none, or
/// not as expected, is left as it is, and the model calls the tools as the
/// prompt tells it to.
fn declared(system: &str) -> (String, Vec<Value>) {
    let parts = || {
        let start = system.find(DECLARATION.0)?;
        let end = start + system[start..].find(DECLARATION.1)? + DECLARATION.1.len();
        let declaration = &system[start..end];
        let from = declaration.find(SIGNATURES.0)? + SIGNATURES.0.len();
        let to = from + declaration[from..].find(SIGNATURES.1)?;
        let tools: Vec<Value> = declaration[from..to]
            .lines()
            .map(serde_json::from_str)
            .collect::<std::result::Result<_, _>>()
            .ok()?;
        let rest = format!("{}{}", &system[..start], system[end..].trim_start());
        Some((rest, tools))
    };
    parts().unwrap_or_else(|| (system.to_string(), Vec::new()))
}

#[cfg(test)]
mod tests {
    use super::*;

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
