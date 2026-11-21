//! The chat format Inkling is trained on: a conversation as messages, each
//! a header saying whose it is, a tag saying what it holds, the content,
//! and an end:
//!
//! ```text
//! <|message_user|><|content_text|>Hello!<|end_message|>
//! <|message_model|><|content_thinking|>The user is greeting me.<|end_message|>
//! <|message_model|><|content_text|>Hi there!<|end_message|>
//! <|content_model_end_sampling|>
//! ```
//!
//! without the line breaks, which are here to read it by. A reply is a
//! message for each thing the model does: a thought, a text, and a call
//! for each tool it calls, which has the tool's name in its header and its
//! arguments as JSON:
//!
//! ```text
//! <|message_model|>read<|content_invoke_tool_json|>{"name":"read","args":{"path":"a.rs"}}<|end_message|>
//! ```
//!
//! The model opens each of its messages itself, and ends its reply with
//! `<|content_model_end_sampling|>`. The tools are declared in a system
//! message of their own, as JSON, ahead of the system prompt; how much the
//! model is to think is a number from 0 to 0.99 in a system message after
//! it; and a tool's result comes back in a message of the tool's.

use std::{ops::ControlFlow, sync::Arc};

use serde_json::{Value, json};

use crate::{
    model::{
        Chunk, Effort, Format, Result, Tokenizer,
        format::{Read, Reader, Repeats, Utf8Stream},
    },
    tools::{ToolCall, declared},
};

/// The tags of the chat format, which are special tokens of the model's
/// vocabulary.
const MESSAGE_USER: &str = "<|message_user|>";
const MESSAGE_MODEL: &str = "<|message_model|>";
const MESSAGE_SYSTEM: &str = "<|message_system|>";
const MESSAGE_TOOL: &str = "<|message_tool|>";
const CONTENT_TEXT: &str = "<|content_text|>";
const CONTENT_THINKING: &str = "<|content_thinking|>";
const CONTENT_XML: &str = "<|content_xml|>";
const CONTENT_INVOKE_TOOL_JSON: &str = "<|content_invoke_tool_json|>";
const END_MESSAGE: &str = "<|end_message|>";
const END_SAMPLING: &str = "<|content_model_end_sampling|>";
const END_OF_TEXT: &str = "<|endoftext|>";

/// The chat format's tags, as a vocabulary for it has them.
#[cfg(test)]
pub const SPECIAL_TOKENS: [&str; 11] = [
    END_OF_TEXT,
    MESSAGE_USER,
    MESSAGE_MODEL,
    MESSAGE_SYSTEM,
    MESSAGE_TOOL,
    CONTENT_TEXT,
    CONTENT_THINKING,
    CONTENT_XML,
    CONTENT_INVOKE_TOOL_JSON,
    END_MESSAGE,
    END_SAMPLING,
];

/// What the header of the system message that declares the tools says.
const TOOL_DECLARE: &str = "tool_declare";

/// How much the model is to think, as the number Inkling's chat template
/// writes for an effort of the same name.
const fn level(effort: Effort) -> &'static str {
    match effort {
        Effort::Xhigh => "0.99",
        Effort::Medium => "0.7",
        Effort::Low => "0.2",
    }
}

/// Inkling's chat format as the model's vocabulary has it: the tokenizer,
/// and the ids of the format's tags, which are special tokens of it. It
/// encodes the turns of a conversation as the chat template writes them.
/// Cloning one shares the tokenizer.
#[derive(Clone)]
pub struct Blocks {
    pub tokenizer: Arc<Tokenizer>,
    pub message_user: u32,
    pub message_model: u32,
    pub message_system: u32,
    pub message_tool: u32,
    pub content_text: u32,
    pub content_thinking: u32,
    pub content_xml: u32,
    pub content_invoke_tool_json: u32,
    pub end_message: u32,
    pub end_sampling: u32,
    pub end_of_text: u32,
}

impl Blocks {
    /// The format over `tokenizer`'s vocabulary, which must have the tags.
    pub fn new(tokenizer: Tokenizer) -> Result<Self> {
        Ok(Self {
            message_user: tokenizer.special(MESSAGE_USER)?,
            message_model: tokenizer.special(MESSAGE_MODEL)?,
            message_system: tokenizer.special(MESSAGE_SYSTEM)?,
            message_tool: tokenizer.special(MESSAGE_TOOL)?,
            content_text: tokenizer.special(CONTENT_TEXT)?,
            content_thinking: tokenizer.special(CONTENT_THINKING)?,
            content_xml: tokenizer.special(CONTENT_XML)?,
            content_invoke_tool_json: tokenizer.special(CONTENT_INVOKE_TOOL_JSON)?,
            end_message: tokenizer.special(END_MESSAGE)?,
            end_sampling: tokenizer.special(END_SAMPLING)?,
            end_of_text: tokenizer.special(END_OF_TEXT)?,
            tokenizer: Arc::new(tokenizer),
        })
    }

    /// A message: whose it is, with `name` in its header if it has one,
    /// what it holds, and the content.
    fn message(&self, role: u32, name: &str, kind: u32, content: &str) -> Result<Vec<u32>> {
        let mut message = vec![role];
        message.extend(self.tokenizer.encode(name)?);
        message.push(kind);
        message.extend(self.tokenizer.encode(content)?);
        message.push(self.end_message);
        Ok(message)
    }

    /// A call the model made, written as dwim keeps one, as the message
    /// the chat template writes for it: the arguments by name in order.
    /// One that cannot be read as a call goes as a call of no tool, with
    /// what was written as its argument, so that its result still has a
    /// call to answer to.
    fn call(&self, written: &str) -> Result<Vec<u32>> {
        let (name, arguments) = match ToolCall::parse(written) {
            Ok(call) => (call.name, call.arguments),
            Err(_) => ("invalid".to_string(), json!({ "written": written })),
        };
        let call = format!(
            "{{\"name\":{},\"args\":{arguments}}}",
            Value::String(name.clone())
        );
        self.message(
            self.message_model,
            &name,
            self.content_invoke_tool_json,
            &call,
        )
    }
}

impl Format for Blocks {
    fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// The tools the prompt declares, in a message of their own, as the
    /// chat template declares them, then the rest of the prompt, then how
    /// much the model is to think. The harness's prompt declares the
    /// tools and opens with the effort's line as Qwen's template has
    /// them: both are taken out of it and written as this format has
    /// them. A prompt that opens with no effort's line is at the effort
    /// that has none.
    fn system(&self, text: &str) -> Result<Vec<u32>> {
        let (text, tools) = declared(text);
        let (effort, text) = [Effort::Xhigh, Effort::Low]
            .into_iter()
            .find_map(|effort| {
                let rest = text.strip_prefix(effort.instruction()?)?;
                Some((effort, rest.trim_start_matches('\n')))
            })
            .unwrap_or((Effort::Medium, &text));
        let mut turn = Vec::new();
        if !tools.is_empty() {
            let specs: Vec<Value> = tools.iter().map(spec).collect();
            turn.extend(self.message(
                self.message_system,
                TOOL_DECLARE,
                self.content_xml,
                &Value::Array(specs).to_string(),
            )?);
        }
        turn.extend(self.message(self.message_system, "", self.content_text, text)?);
        turn.extend(self.message(
            self.message_system,
            "",
            self.content_text,
            &format!("Thinking effort level: {}", level(effort)),
        )?);
        Ok(turn)
    }

    fn user(&self, message: &str) -> Result<Vec<u32>> {
        self.message(self.message_user, "", self.content_text, message)
    }

    /// A message of the tool's for each result.
    fn results(&self, outputs: &[String]) -> Result<Vec<u32>> {
        let mut turn = Vec::new();
        for output in outputs {
            turn.extend(self.message(self.message_tool, "", self.content_text, output)?);
        }
        Ok(turn)
    }

    /// The reply's text, in a message, unless it has none and calls a
    /// tool; a message for each of its tool calls; and the end of the
    /// reply.
    fn reply(&self, text: &str, calls: &[String]) -> Result<Vec<u32>> {
        let mut turn = Vec::new();
        if !text.is_empty() || calls.is_empty() {
            turn.extend(self.message(self.message_model, "", self.content_text, text)?);
        }
        for call in calls {
            turn.extend(self.call(call)?);
        }
        turn.push(self.end_sampling);
        Ok(turn)
    }

    /// The model opens its messages itself, but for the first: a thought,
    /// for it to think, or a text, for a reply made without thinking.
    fn opening(&self, thinking: bool) -> Result<Vec<u32>> {
        Ok(vec![
            self.message_model,
            if thinking {
                self.content_thinking
            } else {
                self.content_text
            },
        ])
    }

    fn end_of_reply(&self, open: bool) -> Result<Vec<u32>> {
        Ok(match open {
            true => vec![self.end_message, self.end_sampling],
            false => vec![self.end_sampling],
        })
    }

    /// A plain answer is a text left open.
    fn end_of_answer(&self) -> Result<Vec<u32>> {
        self.end_of_reply(true)
    }

    fn close_thought(&self, closing: &str) -> Result<Vec<u32>> {
        let mut tokens = self.tokenizer.encode(closing)?;
        tokens.push(self.end_message);
        Ok(tokens)
    }

    fn unanswering(&self) -> Vec<u32> {
        vec![
            self.end_sampling,
            self.end_of_text,
            self.content_invoke_tool_json,
        ]
    }

    /// A plain answer is one text: it may open no other message, nor
    /// anything else in the one it is in.
    fn unplain(&self) -> Vec<u32> {
        vec![
            self.message_user,
            self.message_model,
            self.message_system,
            self.message_tool,
            self.content_text,
            self.content_thinking,
            self.content_xml,
            self.content_invoke_tool_json,
        ]
    }

    /// The end of the reply, and the end of a message, which ends a plain
    /// answer.
    fn ends(&self, token: u32) -> bool {
        token == self.end_message || token == self.end_sampling || token == self.end_of_text
    }

    fn reader(&self) -> Box<dyn Reader> {
        Box::new(Reading {
            format: self.clone(),
            text: Utf8Stream::default(),
            at: At::Thought,
            thought: 0,
            repeats: Repeats::default(),
            replied: false,
            shown: String::new(),
            calls: Vec::new(),
        })
    }
}

/// A tool as the chat template declares one, from its signature as the
/// harness lists it: what it is called, what it does, and its parameters.
fn spec(tool: &Value) -> Value {
    let function = tool.get("function").unwrap_or(tool);
    json!({
        "description": function.get("description").cloned().unwrap_or_else(|| json!("")),
        "name": function.get("name").cloned().unwrap_or(Value::Null),
        "parameters": function.get("parameters").cloned().unwrap_or_else(|| json!({})),
        "type": tool.get("type").cloned().unwrap_or_else(|| json!("function")),
    })
}

/// A call as dwim keeps one, a JSON object of the tool's name and its
/// arguments, from the message the model wrote for it: the name in its
/// header, and a body of the name again and the arguments as `args`. A
/// body that is not that is kept as written, for whoever reads the call to
/// say what is wrong with it.
fn call(name: &str, body: &str) -> String {
    let Ok(Value::Object(mut written)) = serde_json::from_str(body) else {
        return body.to_string();
    };
    let Some(arguments) = written.remove("args") else {
        return body.to_string();
    };
    let name = written
        .remove("name")
        .unwrap_or_else(|| Value::String(name.trim().to_string()));
    json!({ "name": name, "arguments": arguments }).to_string()
}

/// Where a reply being read is: in a message, by what it holds, or
/// between messages.
enum At {
    Thought,
    Text,
    /// In the header of a message, with what it has said so far: a call's
    /// has the tool's name.
    Header(String),
    /// In a call, with the name its header gave and its body so far.
    Call {
        name: String,
        body: String,
    },
    Between,
}

/// A reply being read, a token at a time: the model opens each message
/// with `<|message_model|>`, says what it holds, and ends it, and ends the
/// reply once it has opened no other.
struct Reading {
    format: Blocks,
    text: Utf8Stream,
    at: At,
    /// Tokens of the thought since it opened, and whether it is going
    /// round in circles.
    thought: usize,
    repeats: Repeats,
    replied: bool,
    /// The reply's text as shown, and the calls it has made, as dwim
    /// keeps them.
    shown: String,
    calls: Vec<String>,
}

impl Reader for Reading {
    fn thinking(&self) -> bool {
        matches!(self.at, At::Thought)
    }

    fn open(&self) -> bool {
        !matches!(self.at, At::Between)
    }

    fn thought(&self) -> usize {
        self.thought
    }

    fn repeating(&self) -> bool {
        self.repeats.repeating
    }

    fn replied(&self) -> bool {
        self.replied
    }

    fn read(&mut self, mut token: u32, on_chunk: &mut dyn FnMut(Chunk) -> ControlFlow<()>) -> Read {
        let format = &self.format;
        if token == format.end_sampling || token == format.end_of_text {
            match self.at {
                At::Between if self.replied => return Read::End,
                At::Between => return Read::Unanswered,
                // The model sometimes ends its reply inside a message, a
                // thought most of all, which leaves nothing to show for
                // it: end the message instead, so that it goes on.
                _ => token = format.end_message,
            }
        }
        if self.thinking() {
            self.thought += 1;
            self.repeats.push(token);
        }
        let flow = if token == format.end_message {
            if let At::Call { name, body } = std::mem::replace(&mut self.at, At::Between) {
                self.calls.push(call(&name, &body));
            }
            ControlFlow::Continue(())
        } else if token == format.message_model && matches!(self.at, At::Between) {
            self.at = At::Header(String::new());
            ControlFlow::Continue(())
        } else if token == format.content_thinking && matches!(self.at, At::Header(_)) {
            self.at = At::Thought;
            self.thought = 0;
            self.repeats = Repeats::default();
            ControlFlow::Continue(())
        } else if token == format.content_text && matches!(self.at, At::Header(_)) {
            self.at = At::Text;
            // A text after another is a paragraph of its own.
            if self.shown.is_empty() {
                ControlFlow::Continue(())
            } else {
                self.shown.push_str("\n\n");
                on_chunk(Chunk::Text("\n\n"))
            }
        } else if token == format.content_invoke_tool_json && matches!(self.at, At::Header(_)) {
            let At::Header(name) = std::mem::replace(&mut self.at, At::Between) else {
                unreachable!("matched above");
            };
            self.replied = true;
            self.at = At::Call {
                name,
                body: String::new(),
            };
            ControlFlow::Continue(())
        } else {
            let chunk = self.text.push(format.tokenizer.decode(token));
            match &mut self.at {
                At::Header(name) => {
                    name.push_str(&chunk);
                    ControlFlow::Continue(())
                }
                At::Call { body, .. } => {
                    body.push_str(&chunk);
                    ControlFlow::Continue(())
                }
                _ if chunk.is_empty() => ControlFlow::Continue(()),
                At::Thought => on_chunk(Chunk::Thought(&chunk)),
                // What is said outside any message is text all the same.
                At::Text | At::Between => {
                    self.replied |= !chunk.trim().is_empty();
                    self.shown.push_str(&chunk);
                    on_chunk(Chunk::Text(&chunk))
                }
            }
        };
        Read::Token(token, flow)
    }

    fn flush(&mut self, _on_chunk: &mut dyn FnMut(Chunk) -> ControlFlow<()>) -> ControlFlow<()> {
        ControlFlow::Continue(())
    }

    fn close_thought(&mut self) {
        self.at = At::Between;
    }

    fn finish(self: Box<Self>) -> (String, Vec<String>) {
        (self.shown, self.calls)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn format() -> Blocks {
        Blocks::new(Tokenizer::tiny_inkling()).unwrap()
    }

    /// The tokens as text, tags and all.
    fn text(format: &Blocks, tokens: &[u32]) -> String {
        let bytes: Vec<u8> = tokens
            .iter()
            .flat_map(|&token| format.tokenizer.decode(token).to_vec())
            .collect();
        String::from_utf8(bytes).unwrap()
    }

    /// Reads `written`, the model's reply after the opening, tags and
    /// all, and returns what the reader made of it: the thought and the
    /// text as shown, the calls, the tokens it took, and how it ended.
    fn read(written: &str) -> (String, String, Vec<String>, String, bool) {
        let format = format();
        let mut reader = format.reader();
        let (mut thought, mut shown) = (String::new(), String::new());
        let mut taken = Vec::new();
        let mut ended = false;
        for token in format.tokenizer.encode_with_special(written).unwrap() {
            let mut on_chunk = |chunk: Chunk| {
                match chunk {
                    Chunk::Thought(chunk) => thought.push_str(chunk),
                    Chunk::Text(chunk) => shown.push_str(chunk),
                    _ => {}
                }
                ControlFlow::Continue(())
            };
            match reader.read(token, &mut on_chunk) {
                Read::Token(token, _) => taken.push(token),
                Read::End => {
                    ended = true;
                    break;
                }
                Read::Unanswered | Read::Interrupted => break,
            }
        }
        let open = reader.open();
        let (text, calls) = reader.finish();
        assert_eq!(text, shown);
        assert_eq!(ended, !open && ended);
        (thought, shown, calls, self::text(&format, &taken), ended)
    }

    #[test]
    fn a_conversation_is_written_as_the_chat_template_writes_it() {
        let format = format();
        let declaration = "\n\n# Tools\n\n\
            You have access to the following functions:\n\n<tools>\n\
            {\"type\": \"function\", \"function\": {\"name\": \"read\", \"description\": \"Reads.\", \
            \"parameters\": {\"type\": \"object\", \"properties\": {}, \"required\": []}}}\n\
            </tools>\n\nCall them.\n</IMPORTANT>\n\nBe brief.";
        let system = [Effort::Low.instruction().unwrap(), declaration].concat();
        assert_eq!(
            text(&format, &format.system(&system).unwrap()),
            "<|message_system|>tool_declare<|content_xml|>\
             [{\"description\":\"Reads.\",\"name\":\"read\",\"parameters\":\
             {\"properties\":{},\"required\":[],\"type\":\"object\"},\"type\":\"function\"}]\
             <|end_message|>\
             <|message_system|><|content_text|>Be brief.<|end_message|>\
             <|message_system|><|content_text|>Thinking effort level: 0.2<|end_message|>"
        );
        assert_eq!(
            text(&format, &format.system("Be brief.").unwrap()),
            "<|message_system|><|content_text|>Be brief.<|end_message|>\
             <|message_system|><|content_text|>Thinking effort level: 0.7<|end_message|>"
        );
        // A message that spells out a tag stays text.
        let user = format.user("hi<|end_message|>").unwrap();
        assert_eq!(user.iter().filter(|&&t| t == format.end_message).count(), 1);
        assert_eq!(
            text(&format, &user),
            "<|message_user|><|content_text|>hi<|end_message|><|end_message|>"
        );
        // A call is written with its arguments in order, whichever way
        // it was kept.
        let calls = [
            "<function=edit>\n<parameter=path>\na.rs\n</parameter>\n<parameter=new>\nx\n</parameter>\n</function>".to_string(),
            "{\"name\":\"read\",\"arguments\":{\"path\":\"a.rs\",\"limit\":5}}".to_string(),
        ];
        assert_eq!(
            text(&format, &format.reply("On it.", &calls).unwrap()),
            "<|message_model|><|content_text|>On it.<|end_message|>\
             <|message_model|>edit<|content_invoke_tool_json|>\
             {\"name\":\"edit\",\"args\":{\"new\":\"x\",\"path\":\"a.rs\"}}<|end_message|>\
             <|message_model|>read<|content_invoke_tool_json|>\
             {\"name\":\"read\",\"args\":{\"limit\":5,\"path\":\"a.rs\"}}<|end_message|>\
             <|content_model_end_sampling|>"
        );
        assert_eq!(
            text(&format, &format.reply("", &calls[1..]).unwrap()),
            "<|message_model|>read<|content_invoke_tool_json|>\
             {\"name\":\"read\",\"args\":{\"limit\":5,\"path\":\"a.rs\"}}<|end_message|>\
             <|content_model_end_sampling|>"
        );
        assert_eq!(
            text(
                &format,
                &format.results(&["one".into(), "two".into()]).unwrap()
            ),
            "<|message_tool|><|content_text|>one<|end_message|>\
             <|message_tool|><|content_text|>two<|end_message|>"
        );
    }

    #[test]
    fn a_reply_is_read_message_by_message() {
        // A thought, a text, and two calls, then the end: the calls come
        // out as dwim keeps them, which a tool call parses from.
        let (thought, shown, calls, taken, ended) = read(
            "Let me look.<|end_message|>\
             <|message_model|><|content_text|>Reading it.<|end_message|>\
             <|message_model|>read<|content_invoke_tool_json|>\
             {\"name\":\"read\",\"args\":{\"path\":\"a.rs\",\"limit\":5}}<|end_message|>\
             <|message_model|>bash<|content_invoke_tool_json|>\
             {\"name\":\"bash\",\"args\":{\"command\":\"ls\"}}<|end_message|>\
             <|content_model_end_sampling|>",
        );
        assert_eq!(thought, "Let me look.");
        assert_eq!(shown, "Reading it.");
        assert!(ended);
        assert!(
            taken.ends_with("{\"name\":\"bash\",\"args\":{\"command\":\"ls\"}}<|end_message|>")
        );
        let parsed: Vec<ToolCall> = calls.iter().map(|c| ToolCall::parse(c).unwrap()).collect();
        assert_eq!(parsed[0].name, "read");
        assert_eq!(parsed[0].arguments, json!({ "path": "a.rs", "limit": 5 }));
        assert_eq!(parsed[1].name, "bash");
        assert_eq!(parsed[1].arguments, json!({ "command": "ls" }));
        // What the reader made of the reply is written back as the model
        // wrote it, without the thought.
        let format = format();
        assert_eq!(
            text(&format, &format.reply(&shown, &calls).unwrap()),
            "<|message_model|><|content_text|>Reading it.<|end_message|>\
             <|message_model|>read<|content_invoke_tool_json|>\
             {\"name\":\"read\",\"args\":{\"limit\":5,\"path\":\"a.rs\"}}<|end_message|>\
             <|message_model|>bash<|content_invoke_tool_json|>\
             {\"name\":\"bash\",\"args\":{\"command\":\"ls\"}}<|end_message|>\
             <|content_model_end_sampling|>"
        );
    }

    #[test]
    fn a_reply_that_ends_in_its_thought_goes_on() {
        // The end of the reply inside the thought ends the thought
        // instead; then the reply has said nothing, and may not end.
        let (thought, shown, calls, taken, ended) =
            read("Hmm.<|content_model_end_sampling|><|content_model_end_sampling|>");
        assert_eq!((thought.as_str(), shown.as_str()), ("Hmm.", ""));
        assert!(calls.is_empty() && !ended);
        assert_eq!(taken, "Hmm.<|end_message|>");
        // A second text is a paragraph of its own.
        let (_, shown, ..) = read(
            "<|end_message|><|message_model|><|content_text|>One.<|end_message|>\
             <|message_model|><|content_text|>Two.<|end_message|><|content_model_end_sampling|>",
        );
        assert_eq!(shown, "One.\n\nTwo.");
    }
}
