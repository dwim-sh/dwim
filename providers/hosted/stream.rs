//! A reply as a host streams it: server-sent events, each a piece of
//! the reply as JSON, read into the thought, the text and the tool calls
//! the harness is given.
//!
//! The model writes its calls as its chat template has it, in
//! `<tool_call>` blocks in the text, since the tools are declared in the
//! system prompt and not to the provider. Some providers read the blocks
//! out of the text regardless and send the calls as fields; both come to
//! the same calls here.

use std::{error::Error, fmt, io::BufRead, ops::ControlFlow};

use serde_json::{Value, json};

use crate::model::Chunk;

/// What a reply came to, as the stream gave it.
#[derive(Debug, Default, PartialEq)]
pub struct Outcome {
    /// The thought behind the reply, as it was streamed.
    pub thought: String,
    /// The reply's text as shown, without its calls.
    pub text: String,
    /// The calls the reply made, each as a `<tool_call>` block's body.
    pub calls: Vec<String>,
    /// Whether the thought was ended at its limit, rather than by the
    /// model: the stream does not say, and whoever ended it does.
    pub thought_ended: bool,
    /// Whether the reply was ended by `on_chunk` breaking.
    pub interrupted: bool,
    /// Whether the reply ran to the most tokens it was allowed.
    pub cut: bool,
    /// What the provider counted and charged, if the stream got that far.
    pub usage: Option<Usage>,
    /// Who ran the model, where the host passes the request on and says.
    pub provider: Option<String>,
    /// What the requests for the reply that were made again failed with.
    pub retried: Vec<String>,
    /// Whether the server sent the thought apart from the text.
    pub apart: bool,
}

/// Where a reply's thought comes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Thinking {
    /// Only in its own field.
    Apart,
    /// In its own field, or in the text between `<think>` and `</think>`.
    Tagged,
    /// In its own field, or in the text up to `</think>`: the chat
    /// template opened it in the prompt.
    Opened,
}

/// Why a reply did not come: what went wrong, and whether the request
/// may be answered if it is made again, as one a provider failed at is
/// and one the host refuses for what it asks is not.
#[derive(Debug)]
pub struct Failure {
    pub message: String,
    pub again: bool,
}

impl Failure {
    /// A failure of the stream itself, which another request may not have.
    fn passing(host: &str, error: impl fmt::Display) -> Self {
        Self {
            message: format!("{host}: {error}"),
            again: true,
        }
    }
}

impl fmt::Display for Failure {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl Error for Failure {}

/// What a request used, as the provider counts it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Usage {
    /// Tokens of the prompt, and how many of them the provider had read
    /// before.
    pub prompt: usize,
    pub cached: usize,
    /// Tokens generated, and how many of them were thought.
    pub completion: usize,
    pub reasoning: usize,
    /// Dollars charged.
    pub cost: f64,
}

/// Reads a reply from the events in `body`, streaming it to `on_chunk`, which ends it
/// early by breaking.
pub fn read(
    body: impl BufRead,
    host: &str,
    thinking: Thinking,
    mut on_chunk: impl FnMut(Chunk) -> ControlFlow<()>,
) -> Result<Outcome, Failure> {
    let mut outcome = Outcome::default();
    let mut tags = Tags::new(thinking);
    let mut text = Splitter::default();
    // Calls sent as fields: the name and the arguments of each, as they
    // come in pieces.
    let mut fields: Vec<(String, String)> = Vec::new();
    for line in body.lines() {
        let line = line.map_err(|e| Failure::passing(host, e))?;
        let Some(data) = line.strip_prefix("data:") else {
            continue;
        };
        let data = data.trim();
        if data == "[DONE]" {
            break;
        }
        let event: Value = serde_json::from_str(data).map_err(|e| Failure::passing(host, e))?;
        if let Some(error) = event.get("error") {
            return Err(Failure {
                message: error_message(host, error),
                again: passes(error),
            });
        }
        if let Some(provider) = event["provider"].as_str() {
            outcome.provider = Some(provider.to_string());
        }
        if let Some(usage) = event.get("usage").filter(|usage| usage.is_object()) {
            outcome.usage = Some(Usage::of(usage));
        }
        let Some(choice) = event["choices"].get(0) else {
            continue;
        };
        if choice["finish_reason"] == "length" {
            outcome.cut = true;
        }
        let delta = &choice["delta"];
        let mut flow = ControlFlow::Continue(());
        let thought = delta["reasoning"]
            .as_str()
            .or_else(|| delta["reasoning_content"].as_str())
            .filter(|thought| !thought.is_empty());
        if let Some(thought) = thought {
            outcome.apart = true;
            let held = tags.apart();
            flow = think(thought, &mut outcome, &mut on_chunk);
            if flow.is_continue() && !held.is_empty() {
                flow = text.push(&held, &mut outcome, &mut on_chunk);
            }
        }
        if flow.is_continue()
            && let Some(content) = delta["content"].as_str()
        {
            let (thought, content) = tags.split(content);
            flow = think(&thought, &mut outcome, &mut on_chunk);
            if flow.is_continue() && !content.is_empty() {
                flow = text.push(&content, &mut outcome, &mut on_chunk);
            }
        }
        for call in delta["tool_calls"].as_array().into_iter().flatten() {
            let index = call["index"].as_u64().unwrap_or(0) as usize;
            if fields.len() <= index {
                fields.resize(index + 1, Default::default());
            }
            let function = &call["function"];
            fields[index]
                .0
                .push_str(function["name"].as_str().unwrap_or(""));
            fields[index]
                .1
                .push_str(function["arguments"].as_str().unwrap_or(""));
        }
        if flow.is_continue() {
            flow = on_chunk(Chunk::Token);
        }
        if flow.is_break() {
            outcome.interrupted = true;
            break;
        }
    }
    if !outcome.interrupted {
        let (thought, content) = tags.finish();
        if tags.unclosed() && !outcome.cut {
            // Never closed: the server parsed out an empty thought, so
            // this was the text all along.
            outcome.apart = true;
            let content = std::mem::take(&mut outcome.thought) + &thought;
            let _ = text.push(&content, &mut outcome, &mut on_chunk);
        } else {
            let _ = think(&thought, &mut outcome, &mut on_chunk);
            let _ = text.push(&content, &mut outcome, &mut on_chunk);
        }
        text.finish(&mut outcome, &mut on_chunk);
        for (name, arguments) in fields {
            // Arguments that are not the JSON they should be go on as the
            // text they are, for the harness to tell the model of.
            let arguments =
                serde_json::from_str::<Value>(&arguments).unwrap_or(Value::String(arguments));
            outcome
                .calls
                .push(json!({ "name": name, "arguments": arguments }).to_string());
        }
    }
    // What a call was about to be is not run once the reply is ended
    // early, since the results would have nothing to follow.
    if outcome.interrupted || outcome.cut {
        outcome.calls.clear();
    }
    // A thought that was not expected in the text: calls inside it are
    // not made.
    if thinking != Thinking::Apart
        && outcome.thought.is_empty()
        && let Some(at) = outcome.text.find(THOUGHT)
    {
        let inside = text.offsets.iter().filter(|&&offset| offset <= at).count();
        outcome.calls.drain(..inside.min(outcome.calls.len()));
        let text = outcome.text.split_off(at + THOUGHT.len());
        outcome.text.truncate(at);
        outcome.thought = std::mem::replace(&mut outcome.text, text.trim_start().to_string());
    }
    // llama.cpp does not count reasoning tokens.
    if let Some(usage) = &mut outcome.usage
        && usage.reasoning == 0
        && !outcome.thought.is_empty()
    {
        usage.reasoning = (outcome.thought.len() / super::THOUGHT_BYTES).min(usage.completion);
    }
    Ok(outcome)
}

fn think(
    thought: &str,
    outcome: &mut Outcome,
    on_chunk: &mut impl FnMut(Chunk) -> ControlFlow<()>,
) -> ControlFlow<()> {
    if thought.is_empty() {
        return ControlFlow::Continue(());
    }
    outcome.thought.push_str(thought);
    on_chunk(Chunk::Thought(thought))
}

/// What an error says, as `host` writes one: its message, with the
/// provider it is of and the provider's own words where the host passes
/// them on.
pub fn error_message(host: &str, error: &Value) -> String {
    let message = error["message"].as_str().unwrap_or("the request failed");
    let metadata = &error["metadata"];
    let mut said = format!("{host}: {message}");
    if let Some(provider) = metadata["provider_name"].as_str() {
        said.push_str(&format!(" ({provider})"));
    }
    if let Some(raw) = metadata["raw"].as_str() {
        said.push_str(&format!(": {raw}"));
    }
    said
}

/// Whether an error in a stream may pass if the request is made again:
/// one with the code of a request refused for now or of a server's
/// failure, or with none, as a provider that failed mid-reply is passed
/// on. A code of the request's own fault, such as one too long, is not.
fn passes(error: &Value) -> bool {
    let code = match &error["code"] {
        Value::String(code) => code.parse().ok(),
        code => code.as_u64(),
    };
    code.is_none_or(|code| code == 408 || code == 429 || code >= 500)
}

impl Usage {
    fn of(usage: &Value) -> Self {
        let count = |value: &Value| value.as_u64().unwrap_or(0) as usize;
        Self {
            prompt: count(&usage["prompt_tokens"]),
            cached: count(&usage["prompt_tokens_details"]["cached_tokens"]),
            completion: count(&usage["completion_tokens"]),
            reasoning: count(&usage["completion_tokens_details"]["reasoning_tokens"]),
            // A host may write the dollars as a number, or as text.
            cost: match &usage["cost"] {
                Value::String(cost) => cost.parse().unwrap_or(0.0),
                cost => cost.as_f64().unwrap_or(0.0),
            },
        }
    }
}

const THINK: &str = "<think>";
const THOUGHT: &str = "</think>";

/// Splits a thought written in the text from the text, as it streams.
struct Tags {
    /// Text that may be the start of a tag.
    held: String,
    at: Part,
    /// Whether the reply starts inside a thought the prompt opened.
    opened: bool,
}

#[derive(PartialEq)]
enum Part {
    Start,
    Thought,
    /// After the thought, skipping blank lines.
    Between,
    Text,
}

impl Tags {
    fn new(thinking: Thinking) -> Self {
        Self {
            held: String::new(),
            at: match thinking {
                Thinking::Apart => Part::Text,
                Thinking::Tagged | Thinking::Opened => Part::Start,
            },
            opened: thinking == Thinking::Opened,
        }
    }

    /// The server sends the thought apart: what was held is text.
    fn apart(&mut self) -> String {
        if self.at != Part::Start {
            return String::new();
        }
        self.at = Part::Text;
        std::mem::take(&mut self.held)
    }

    fn unclosed(&self) -> bool {
        self.opened && self.at == Part::Thought
    }

    /// The next piece of the text, as thought and text.
    fn split(&mut self, piece: &str) -> (String, String) {
        match self.at {
            Part::Text => return (String::new(), piece.to_string()),
            Part::Between => {
                let text = piece.trim_start();
                if !text.is_empty() {
                    self.at = Part::Text;
                }
                return (String::new(), text.to_string());
            }
            Part::Start | Part::Thought => self.held.push_str(piece),
        }
        if self.at == Part::Start {
            let start = self.held.trim_start();
            if let Some(thought) = start.strip_prefix(THINK) {
                self.held = thought.to_string();
                self.at = Part::Thought;
                self.opened = false;
            } else if THINK.starts_with(start) {
                return (String::new(), String::new());
            } else if self.opened {
                self.at = Part::Thought;
            } else {
                self.at = Part::Text;
                return (String::new(), std::mem::take(&mut self.held));
            }
        }
        if let Some(at) = self.held.find(THOUGHT) {
            let thought = self.held[..at].to_string();
            let rest = self.held[at + THOUGHT.len()..].to_string();
            self.held.clear();
            self.at = Part::Between;
            return (thought, self.split(&rest).1);
        }
        let keep = (1..THOUGHT.len())
            .rev()
            .find(|&n| self.held.ends_with(&THOUGHT[..n]))
            .unwrap_or(0);
        let thought = self.held.drain(..self.held.len() - keep).collect();
        (thought, String::new())
    }

    fn finish(&mut self) -> (String, String) {
        let held = std::mem::take(&mut self.held);
        match self.at {
            Part::Thought => (held, String::new()),
            _ => (String::new(), held),
        }
    }
}

/// The tags a call is written between. The model now and then writes the
/// opening in the plural, with the block otherwise as it should be.
const OPENINGS: [&str; 2] = ["<tool_call>", "<tool_calls>"];
const CLOSINGS: [&str; 2] = ["</tool_call>", "</tool_calls>"];

/// Splits a reply's text, as it comes in pieces, into what is shown and
/// the calls written in it.
#[derive(Default)]
struct Splitter {
    /// Text not yet passed on: the start of what may be a tag, or the body
    /// of a call so far.
    held: String,
    in_call: bool,
    /// The length of the text shown when each call was written.
    offsets: Vec<usize>,
}

impl Splitter {
    /// Takes the next piece of the text.
    fn push(
        &mut self,
        piece: &str,
        outcome: &mut Outcome,
        on_chunk: &mut impl FnMut(Chunk) -> ControlFlow<()>,
    ) -> ControlFlow<()> {
        self.held.push_str(piece);
        loop {
            if self.in_call {
                let Some((at, tag)) = find(&self.held, &CLOSINGS) else {
                    return ControlFlow::Continue(());
                };
                outcome.calls.push(self.held[..at].to_string());
                self.offsets.push(outcome.text.len());
                self.held.drain(..at + tag.len());
                self.in_call = false;
            } else if let Some((at, tag)) = find(&self.held, &OPENINGS) {
                let shown: String = self.held.drain(..at).collect();
                self.held.drain(..tag.len());
                self.in_call = true;
                show(&shown, outcome, on_chunk)?;
            } else {
                // All of it is text but an end that may be the start of a
                // tag, which waits for the rest.
                let keep = OPENINGS
                    .iter()
                    .flat_map(|tag| (1..tag.len()).map(|n| &tag[..n]))
                    .filter(|start| self.held.ends_with(start))
                    .map(str::len)
                    .max()
                    .unwrap_or(0);
                let shown: String = self.held.drain(..self.held.len() - keep).collect();
                return show(&shown, outcome, on_chunk);
            }
        }
    }

    /// Ends the text: what was held is shown, and a call left open is
    /// taken as far as it got.
    fn finish(
        &mut self,
        outcome: &mut Outcome,
        on_chunk: &mut impl FnMut(Chunk) -> ControlFlow<()>,
    ) {
        let held = std::mem::take(&mut self.held);
        if self.in_call {
            outcome.calls.push(held);
            self.offsets.push(outcome.text.len());
        } else {
            let _ = show(&held, outcome, on_chunk);
        }
    }
}

/// Passes on text of the reply that is shown.
fn show(
    text: &str,
    outcome: &mut Outcome,
    on_chunk: &mut impl FnMut(Chunk) -> ControlFlow<()>,
) -> ControlFlow<()> {
    if text.is_empty() {
        return ControlFlow::Continue(());
    }
    outcome.text.push_str(text);
    on_chunk(Chunk::Text(text))
}

/// Where the first of `tags` is in `text`, and which it is.
fn find<'a>(text: &str, tags: &[&'a str]) -> Option<(usize, &'a str)> {
    tags.iter()
        .filter_map(|tag| text.find(tag).map(|at| (at, *tag)))
        .min_by_key(|(at, _)| *at)
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    /// The events of a reply, a delta each.
    fn events(deltas: &[Value]) -> Cursor<String> {
        let mut body = String::from(": OPENROUTER PROCESSING\n\n");
        for delta in deltas {
            let event = json!({ "choices": [{ "delta": delta, "finish_reason": null }] });
            body.push_str(&format!("data: {event}\n\n"));
        }
        body.push_str("data: [DONE]\n\n");
        Cursor::new(body)
    }

    fn reply(deltas: &[Value]) -> (Outcome, String, String) {
        reply_as(Thinking::Tagged, deltas)
    }

    /// The reply read from `deltas`, and its thought and text as streamed.
    fn reply_as(thinking: Thinking, deltas: &[Value]) -> (Outcome, String, String) {
        let (mut thought, mut text) = (String::new(), String::new());
        let outcome = read(events(deltas), "Endpoint", thinking, |chunk| {
            match chunk {
                Chunk::Thought(piece) => thought.push_str(piece),
                Chunk::Text(piece) => text.push_str(piece),
                Chunk::Token => {}
            }
            ControlFlow::Continue(())
        })
        .unwrap();
        (outcome, thought, text)
    }

    #[test]
    fn a_call_written_in_the_text_is_a_call_however_it_is_cut_up() {
        let (outcome, thought, text) = reply(&[
            json!({ "reasoning": "I will " }),
            json!({ "reasoning": "look." }),
            json!({ "content": "Looking.<tool" }),
            json!({ "content": "_call>\n<function=bash>\n<parameter=command>\nls" }),
            json!({ "content": "\n</parameter>\n</function>\n</tool_" }),
            json!({ "content": "call>" }),
        ]);
        assert_eq!(thought, "I will look.");
        assert_eq!(text, "Looking.");
        assert_eq!(outcome.text, "Looking.");
        assert_eq!(
            outcome.calls,
            ["\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>\n"]
        );
    }

    #[test]
    fn a_thought_written_in_the_text_is_the_thought_however_it_is_cut_up() {
        let pieces = ["\n<thi", "nk>\nI will ", "look.</th", "ink>\n", "\nDone."];
        let deltas: Vec<Value> = pieces.iter().map(|p| json!({ "content": p })).collect();
        let (outcome, thought, text) = reply(&deltas);

        assert_eq!(outcome.thought.trim(), "I will look.");
        assert_eq!(thought, outcome.thought);
        assert_eq!((outcome.text.as_str(), text.as_str()), ("Done.", "Done."));
    }

    #[test]
    fn a_reply_that_does_not_open_with_a_thought_is_all_text() {
        for content in ["Use <think> tags.", "<thinking> is a word.", "  "] {
            let (outcome, thought, _) = reply(&[json!({ "content": content })]);
            assert_eq!((outcome.text.as_str(), thought.as_str()), (content, ""));
        }
    }

    #[test]
    fn a_thought_the_prompt_opened_is_the_thought_once_the_reply_is_read() {
        let (outcome, _, _) = reply(&[
            json!({ "content": "I will count.\n</th" }),
            json!({ "content": "ink>\n\nIt has 48 lines." }),
        ]);
        assert_eq!(outcome.thought, "I will count.\n");
        assert_eq!(outcome.text, "It has 48 lines.");
    }

    const REMOVE: &str = "<tool_call>\n<function=bash>\n<parameter=command>\nrm x\n</parameter>\n</function>\n</tool_call>";

    #[test]
    fn a_thought_the_prompt_opened_is_streamed_as_the_thought_and_its_calls_are_not_made() {
        let (outcome, thought, text) = reply_as(
            Thinking::Opened,
            &[
                json!({ "content": format!("I could run {REMOVE}, but no.\n</th") }),
                json!({ "content": "ink>\n\nDone." }),
            ],
        );
        assert_eq!(thought, outcome.thought);
        assert!(outcome.thought.starts_with("I could run <tool_call>"));
        assert_eq!((outcome.text.as_str(), text.as_str()), ("Done.", "Done."));
        assert!(outcome.calls.is_empty(), "{:?}", outcome.calls);
    }

    #[test]
    fn a_thought_sent_apart_leaves_the_text_as_text_whatever_the_prompt_opened() {
        let (outcome, thought, text) = reply_as(
            Thinking::Opened,
            &[
                json!({ "reasoning_content": "I will look." }),
                json!({ "content": "Looking." }),
                json!({ "content": REMOVE }),
            ],
        );
        assert_eq!(
            (thought.as_str(), text.as_str()),
            ("I will look.", "Looking.")
        );
        assert_eq!(outcome.calls.len(), 1);
        assert!(outcome.apart);
    }

    #[test]
    fn a_reply_the_prompt_opened_a_thought_for_that_never_closes_it_is_the_text() {
        let (outcome, _, text) = reply_as(
            Thinking::Opened,
            &[
                json!({ "content": "Looking." }),
                json!({ "content": REMOVE }),
            ],
        );
        assert_eq!(
            (outcome.thought.as_str(), outcome.text.as_str()),
            ("", "Looking.")
        );
        assert_eq!(text, "Looking.");
        assert_eq!(outcome.calls.len(), 1);
        assert!(outcome.apart);
    }

    #[test]
    fn a_thought_the_prompt_opened_that_is_cut_off_is_the_thought() {
        let body = events(&[json!({ "content": "Let me think about" })])
            .into_inner()
            .replace("data: [DONE]", "");
        let cut =
            json!({ "choices": [{ "delta": { "content": " it." }, "finish_reason": "length" }] });
        let body = format!("{body}data: {cut}\n\ndata: [DONE]\n\n");
        let outcome = read(Cursor::new(body), "Endpoint", Thinking::Opened, |_| {
            ControlFlow::Continue(())
        })
        .unwrap();
        assert_eq!(
            (outcome.thought.as_str(), outcome.text.as_str()),
            ("Let me think about it.", "")
        );
        assert!(outcome.cut);
    }

    #[test]
    fn a_closing_tag_a_host_sends_in_the_text_is_text() {
        let content = "Qwen ends its thought with </think>, then answers.";
        let (outcome, thought, text) = reply_as(Thinking::Apart, &[json!({ "content": content })]);
        assert_eq!((outcome.text.as_str(), text.as_str()), (content, content));
        assert_eq!((outcome.thought.as_str(), thought.as_str()), ("", ""));
    }

    #[test]
    fn a_call_written_before_a_closing_not_known_to_be_opened_is_not_made() {
        let (outcome, _, _) = reply(&[
            json!({ "content": format!("I could run {REMOVE}.\n</think>\n\n") }),
            json!({ "content": format!("Listing. {}", REMOVE.replace("rm x", "ls")) }),
        ]);
        assert_eq!(outcome.text, "Listing. ");
        assert_eq!(outcome.calls.len(), 1);
        assert!(outcome.calls[0].contains("ls"), "{:?}", outcome.calls);
    }

    #[test]
    fn a_thought_the_server_does_not_count_is_estimated() {
        let mut body = events(&[json!({ "reasoning_content": "Let me look at it first." })])
            .into_inner()
            .replace("data: [DONE]", "");
        let usage =
            json!({ "choices": [], "usage": { "prompt_tokens": 10, "completion_tokens": 20 } });
        body.push_str(&format!("data: {usage}\n\ndata: [DONE]\n\n"));
        let outcome = read(Cursor::new(body), "Endpoint", Thinking::Tagged, |_| {
            ControlFlow::Continue(())
        })
        .unwrap();

        let usage = outcome.usage.unwrap();
        assert!((1..20).contains(&usage.reasoning), "{usage:?}");
    }

    #[test]
    fn text_that_only_looks_like_a_tag_is_shown() {
        let (outcome, _, text) = reply(&[
            json!({ "content": "a < b and <tool" }),
            json!({ "content": "box> is not a tag, nor is <" }),
        ]);
        assert_eq!(text, "a < b and <toolbox> is not a tag, nor is <");
        assert!(outcome.calls.is_empty());
    }

    #[test]
    fn a_call_sent_as_fields_is_a_call_the_harness_can_parse() {
        let (outcome, _, text) = reply(&[
            json!({ "tool_calls": [{ "index": 0, "function": { "name": "bash", "arguments": "{\"comm" } }] }),
            json!({ "tool_calls": [{ "index": 0, "function": { "arguments": "and\": \"ls\"}" } }] }),
        ]);
        assert_eq!(text, "");
        let call = crate::tools::ToolCall::parse(&outcome.calls[0]).unwrap();
        assert_eq!(call.name, "bash");
        assert_eq!(call.arguments["command"], "ls");
    }

    #[test]
    fn a_reply_ended_early_makes_no_calls() {
        let outcome = read(
            events(&[
                json!({ "content": "One." }),
                json!({ "content": "<tool_call>\n<function=bash>\n</function>\n</tool_call>" }),
            ]),
            "OpenRouter",
            Thinking::Apart,
            |chunk| match chunk {
                Chunk::Text(_) => ControlFlow::Break(()),
                _ => ControlFlow::Continue(()),
            },
        )
        .unwrap();
        assert!(outcome.interrupted);
        assert_eq!(outcome.text, "One.");
        assert!(outcome.calls.is_empty());
    }

    #[test]
    fn an_error_in_the_stream_fails_the_reply() {
        let body = Cursor::new(
            "data: {\"error\":{\"message\":\"Provider returned error\",\"metadata\":{\"raw\":\"overloaded\"}}}\n\n",
        );
        let error = read(body, "OpenRouter", Thinking::Apart, |_| {
            ControlFlow::Continue(())
        })
        .unwrap_err();
        assert!(error.to_string().contains("overloaded"), "{error}");
        assert!(error.again);
        // A request refused for what it asks is not worth making again.
        let body = Cursor::new(
            "data: {\"error\":{\"code\":400,\"message\":\"the prompt is too long\"}}\n\n",
        );
        let error = read(body, "OpenRouter", Thinking::Apart, |_| {
            ControlFlow::Continue(())
        })
        .unwrap_err();
        assert!(!error.again, "{error}");
    }
}
