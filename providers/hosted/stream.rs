//! A reply as a host streams it: server-sent events, each a piece of
//! the reply as JSON, read into the thought, the text and the tool calls
//! the harness is given.
//!
//! The model writes its calls as its chat template has it, in
//! `<tool_call>` blocks in the text, since the tools are declared in the
//! system prompt and not to the provider. Some providers read the blocks
//! out of the text regardless and send the calls as fields; both come to
//! the same calls here.

use std::{io::BufRead, ops::ControlFlow};

use serde_json::{Value, json};

use crate::model::{Chunk, Result};

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
}

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

/// Reads a reply from the events in `body`, which `host` sends, streaming
/// it to `on_chunk`, which ends it early by breaking.
pub fn read(
    body: impl BufRead,
    host: &str,
    mut on_chunk: impl FnMut(Chunk) -> ControlFlow<()>,
) -> Result<Outcome> {
    let mut outcome = Outcome::default();
    let mut text = Splitter::default();
    // Calls sent as fields: the name and the arguments of each, as they
    // come in pieces.
    let mut fields: Vec<(String, String)> = Vec::new();
    for line in body.lines() {
        let line = line?;
        let Some(data) = line.strip_prefix("data:") else {
            continue;
        };
        let data = data.trim();
        if data == "[DONE]" {
            break;
        }
        let event: Value = serde_json::from_str(data)?;
        if let Some(error) = event.get("error") {
            return Err(error_message(host, error).into());
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
        if let Some(thought) = delta["reasoning"].as_str().filter(|s| !s.is_empty()) {
            outcome.thought.push_str(thought);
            flow = on_chunk(Chunk::Thought(thought));
        }
        if flow.is_continue()
            && let Some(content) = delta["content"].as_str().filter(|s| !s.is_empty())
        {
            flow = text.push(content, &mut outcome, &mut on_chunk);
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
    Ok(outcome)
}

/// What an error says, as `host` writes one: its message, and the
/// provider's own where the host passes one on.
pub fn error_message(host: &str, error: &Value) -> String {
    let message = error["message"].as_str().unwrap_or("the request failed");
    match error["metadata"]["raw"].as_str() {
        Some(raw) => format!("{host}: {message}: {raw}"),
        None => format!("{host}: {message}"),
    }
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

    /// The reply read from `deltas`, and its thought and its text as they
    /// were streamed.
    fn reply(deltas: &[Value]) -> (Outcome, String, String) {
        let (mut thought, mut text) = (String::new(), String::new());
        let outcome = read(events(deltas), "OpenRouter", |chunk| {
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
        let error = read(body, "OpenRouter", |_| ControlFlow::Continue(())).unwrap_err();
        assert!(error.to_string().contains("overloaded"), "{error}");
    }
}
