//! The user's own chat completions server, such as llama.cpp's or vLLM's.

use std::time::Duration;

use serde_json::{Value, json};
use ureq::Agent;

use crate::{
    model::{Effort, Model, Sampling},
    providers::{
        Kind,
        hosted::{Host, Hosted, key},
    },
};

pub const HOST: Host = Host {
    kind: Kind::Endpoint,
    name: "Endpoint",
    url: "",
    model: Model::name,
    ask,
    key: "",
    refused: "/login takes a key the server takes",
};

/// Where the user's server is, and the model's name there if it serves several.
#[derive(Clone, Debug, PartialEq)]
pub struct Address {
    pub url: String,
    pub model: Option<String>,
}

impl Address {
    pub fn base(&self) -> String {
        base_of(&self.url)
    }
}

const ASKING: Duration = Duration::from_secs(5);

/// What the server says it serves.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Served {
    pub base: String,
    pub answered: bool,
    /// The model's name at the server.
    pub name: Option<String>,
    /// Which of dwim's models that is, if any.
    pub model: Option<Model>,
    pub context: Option<usize>,
}

pub fn nowhere() -> String {
    "there is no server to ask: /login takes its address, as http://localhost:8080".to_string()
}

/// Asks the server at `address` what it serves.
pub fn served(address: &Address) -> Served {
    let key = key::kept(&HOST).unwrap_or_default();
    ask_server(address.base(), address.model.clone(), &key)
}

/// `model` at the server `served` describes, without asking it again.
pub fn open(
    served: &Served,
    model: Model,
    effort: Effort,
    context: Option<usize>,
    temperature: Option<f32>,
) -> Hosted {
    let context = match (context, served.context) {
        (Some(asked), Some(room)) => Some(asked.min(room)),
        (asked, room) => asked.or(room),
    };
    let key = key::kept(&HOST).unwrap_or_default();
    let name = served
        .name
        .clone()
        .unwrap_or_else(|| model.name().to_string());
    let url = format!("{}/chat/completions", served.base);
    Hosted::new(&HOST, key, model, effort, context, temperature)
        .at(url, name)
        .tagged()
}

/// `url` without the chat completions path, or with `/v1` added to a bare host.
fn base_of(url: &str) -> String {
    let url = url.trim_end_matches('/');
    let url = url.strip_suffix("/chat/completions").unwrap_or(url);
    let path = url.split_once("://").map_or(url, |(_, rest)| rest);
    if path.contains('/') {
        url.to_string()
    } else {
        format!("{url}/v1")
    }
}

fn ask_server(base: String, named: Option<String>, key: &str) -> Served {
    let agent: Agent = Agent::config_builder()
        .timeout_global(Some(ASKING))
        .build()
        .into();
    let get = |url: String| -> Option<Value> {
        let mut request = agent.get(url);
        if !key.is_empty() {
            request = request.header("Authorization", format!("Bearer {key}"));
        }
        let text = request.call().ok()?.into_body().read_to_string().ok()?;
        serde_json::from_str(&text).ok()
    };
    let Some(listing) = get(format!("{base}/models")) else {
        let name = named;
        let model = name.as_deref().and_then(recognize);
        return Served {
            base,
            answered: false,
            name,
            model,
            context: None,
        };
    };
    let (name, listed) = listed(&listing, named);
    let context = listed
        .and_then(|model| model["max_model_len"].as_u64())
        .or_else(|| {
            let props = get(format!("{}/props", base.strip_suffix("/v1")?))?;
            props["default_generation_settings"]["n_ctx"].as_u64()
        })
        .map(|tokens| tokens as usize);
    Served {
        model: name.as_deref().and_then(recognize),
        base,
        answered: true,
        name,
        context,
    }
}

/// The model named, or else the only one listed, and its entry.
fn listed(listing: &Value, named: Option<String>) -> (Option<String>, Option<&Value>) {
    let models = listing["data"].as_array().map_or(&[][..], Vec::as_slice);
    match named {
        Some(name) => {
            let entry = models.iter().find(|model| model["id"] == name.as_str());
            (Some(name), entry)
        }
        None => match models {
            [model] => (model["id"].as_str().map(str::to_string), Some(model)),
            _ => (None, None),
        },
    }
}

/// Which of dwim's models a server's name for one is, ignoring case and punctuation.
pub fn recognize(name: &str) -> Option<Model> {
    let name = letters(name);
    Model::ALL
        .into_iter()
        .filter(|model| name.contains(&letters(model.name())))
        .max_by_key(|model| model.name().len())
}

fn letters(text: &str) -> String {
    text.chars()
        .filter(char::is_ascii_alphanumeric)
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

fn ask(body: &mut Value, sampling: &Sampling, effort: Option<Effort>) {
    body["stream_options"] = json!({ "include_usage": true });
    body["reasoning_format"] = json!("deepseek");
    // Calls are parsed from the text; vLLM refuses tools without a tool
    // parser unless told not to parse them.
    if body.get("tools").is_some() {
        body["tool_choice"] = json!("none");
    }
    if sampling.top_k > 0 {
        body["top_k"] = json!(sampling.top_k);
    }
    body["chat_template_kwargs"] = match effort {
        Some(effort) => json!({ "reasoning_effort": effort.name() }),
        None => json!({ "enable_thinking": false }),
    };
    match effort.filter(|&effort| effort != Effort::Xhigh) {
        Some(effort) => body["reasoning_effort"] = json!(effort.name()),
        None => {
            if let Some(body) = body.as_object_mut() {
                body.remove("reasoning_effort");
            }
        }
    }
    for message in body["messages"].as_array_mut().into_iter().flatten() {
        if let Some(thought) = message.get("reasoning").cloned() {
            message["reasoning_content"] = thought;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{env, ops::ControlFlow};

    use super::*;
    use crate::{
        harness,
        model::{Chat, Chunk, Generation, Turn},
        providers::Provider,
        tools::ToolCall,
    };

    #[test]
    fn the_server_alone_is_asked_under_v1_and_any_other_address_as_given() {
        for (given, base) in [
            ("http://192.168.1.122:8080", "http://192.168.1.122:8080/v1"),
            ("localhost:11434/", "localhost:11434/v1"),
            ("http://localhost:8000/v1/", "http://localhost:8000/v1"),
            (
                "https://example.com/api/v1/chat/completions",
                "https://example.com/api/v1",
            ),
        ] {
            assert_eq!(base_of(given), base, "{given}");
        }
    }

    #[test]
    fn a_reply_asked_again_without_reasoning_drops_the_effort_asked_before() {
        let sampling = Model::Qwen27b.sampling();
        let mut body = json!({});
        ask(&mut body, &sampling, Some(Effort::Medium));
        assert_eq!(body["reasoning_effort"], "medium");
        ask(&mut body, &sampling, None);
        assert_eq!(body.get("reasoning_effort"), None);
        assert_eq!(
            body["chat_template_kwargs"],
            json!({ "enable_thinking": false })
        );
        // vLLM refuses a `reasoning_effort` the API does not name.
        ask(&mut body, &sampling, Some(Effort::Xhigh));
        assert_eq!(body.get("reasoning_effort"), None);
    }

    #[test]
    fn a_model_is_recognized_by_the_name_a_server_or_a_repository_gives_it() {
        for (name, model) in [
            ("/models/Qwen3.8-27B-UD-Q4_K_M.gguf", Some(Model::Qwen27b)),
            ("Qwen/Qwen3.8-2.4T-A95B", Some(Model::QwenMax)),
            ("thinkingmachines/inkling-2-mini", Some(Model::InklingMini)),
            ("unsloth/Inkling-2-Small-GGUF", Some(Model::InklingSmall)),
            ("llama3.3:70b", None),
        ] {
            assert_eq!(recognize(name), model, "{name}");
        }
    }

    #[test]
    fn the_name_and_room_come_from_the_one_model_listed_or_the_one_named() {
        let vllm = json!({ "data": [{ "id": "Qwen/Qwen3.8-27B", "max_model_len": 32768 }] });
        let (name, entry) = listed(&vllm, None);
        assert_eq!(name.as_deref(), Some("Qwen/Qwen3.8-27B"));
        assert_eq!(entry.unwrap()["max_model_len"], 32768);

        let ollama = json!({ "data": [{ "id": "qwen3.8:27b" }, { "id": "llama3.3:70b" }] });
        assert_eq!(listed(&ollama, None), (None, None));
        let (name, entry) = listed(&ollama, Some("llama3.3:70b".to_string()));
        assert_eq!(name.as_deref(), Some("llama3.3:70b"));
        assert_eq!(entry, Some(&ollama["data"][1]));
    }

    #[test]
    fn a_thought_goes_back_where_the_servers_read_it() {
        let mut body = json!({ "messages": [
            { "role": "user", "content": "Fix it." },
            { "role": "assistant", "content": "", "reasoning": "First, the test." },
        ] });
        ask(&mut body, &Model::Qwen27b.sampling(), Some(Effort::Low));
        assert_eq!(body["messages"][1]["reasoning_content"], "First, the test.");
        assert_eq!(body["messages"][0].get("reasoning_content"), None);
    }

    const TEST_SERVER: &str = "DWIM_TEST_ENDPOINT";

    /// The model at the server in `DWIM_TEST_ENDPOINT`, run as `DWIM_TEST_MODEL` or Qwen3.8-27B.
    fn real_endpoint() -> Option<(Hosted, Model)> {
        let Some(url) = env::var(TEST_SERVER).ok().filter(|url| !url.is_empty()) else {
            eprintln!("skipped: {TEST_SERVER} is not set");
            return None;
        };
        let address = Address { url, model: None };
        let model =
            env::var("DWIM_TEST_MODEL").map_or(Model::Qwen27b, |name| name.parse().unwrap());
        let served = served(&address);
        Some((open(&served, model, Effort::Low, None, None), model))
    }

    fn chat(model: Model, message: &str) -> Chat {
        Chat {
            system: harness::system_prompt(&env::current_dir().unwrap(), model, None),
            turns: vec![Turn::User(message.to_string())],
        }
    }

    #[test]
    #[ignore = "needs a server at DWIM_TEST_ENDPOINT"]
    fn real_endpoint_answers_and_counts_the_tokens() {
        let Some((mut provider, model)) = real_endpoint() else {
            return;
        };
        let chat = chat(model, "What is 17 * 23? Reply with the number alone.");
        let mut streamed = String::new();
        let completion = provider
            .reply(&chat, &Generation::default(), |chunk| {
                if let Chunk::Text(text) = chunk {
                    streamed.push_str(text);
                }
                ControlFlow::Continue(())
            })
            .unwrap();

        assert!(completion.text.contains("391"), "{completion:?}");
        assert_eq!(completion.text.trim(), streamed.trim());
        assert!(completion.reply.calls.is_empty(), "{completion:?}");
        assert!(completion.state.tokens > 1000, "{completion:?}");
    }

    #[test]
    #[ignore = "needs a server at DWIM_TEST_ENDPOINT"]
    fn real_endpoint_calls_a_tool_the_system_prompt_declares() {
        let Some((mut provider, model)) = real_endpoint() else {
            return;
        };
        let chat = chat(
            model,
            "Count the lines of Cargo.toml with `wc -l Cargo.toml`, using the bash tool.",
        );
        let completion = provider
            .reply(&chat, &Generation::default(), |_| ControlFlow::Continue(()))
            .unwrap();

        assert_eq!(completion.reply.calls.len(), 1, "{completion:?}");
        let call = ToolCall::parse(&completion.reply.calls[0]).unwrap();
        assert_eq!(call.name, "bash");
        assert!(call.arguments.to_string().contains("wc -l"), "{call:?}");
    }

    #[test]
    #[ignore = "needs a server at DWIM_TEST_ENDPOINT"]
    fn real_endpoint_reads_the_thought_behind_a_reply_since_the_last_message() {
        let read = |thought: &str| {
            let (mut provider, model) = real_endpoint()?;
            let mut chat = chat(model, "How many lines has Cargo.toml?");
            chat.turns.push(Turn::Reply {
                text: String::new(),
                calls: vec![
                    json!({ "name": "bash", "arguments": { "command": "wc -l Cargo.toml" } })
                        .to_string(),
                ],
                thought: thought.to_string(),
            });
            chat.turns
                .push(Turn::Results(vec!["42 Cargo.toml".to_string()]));
            let completion = provider
                .reply(&chat, &Generation::default(), |_| ControlFlow::Continue(()))
                .unwrap();
            let stats = completion.state.stats;
            Some(stats.prompt.tokens + stats.cached.tokens)
        };
        let thought =
            "The manifest says what the package is, so I count its lines first. ".repeat(20);
        let (Some(with), Some(without)) = (read(&thought), read("")) else {
            return;
        };

        // The thought is about 270 tokens.
        assert!(
            with >= without + 200,
            "{with} tokens with the thought, {without} without"
        );
    }

    #[test]
    #[ignore = "needs a server at DWIM_TEST_ENDPOINT"]
    fn real_endpoint_ends_a_thought_at_its_limit_and_answers_without_one() {
        let Some((mut provider, model)) = real_endpoint() else {
            return;
        };
        let chat = chat(
            model,
            "Think it through step by step: which is larger, 2^100 or 3^60? Then reply \
             with the larger one alone.",
        );
        let generation = Generation {
            reserve: 0,
            thought_limit: Some(32),
            closing: Some("I have to answer now.".to_string()),
        };
        let completion = provider
            .reply(&chat, &generation, |_| ControlFlow::Continue(()))
            .unwrap();

        assert!(completion.reply.thought_ended, "{completion:?}");
        assert!(completion.thought.contains("I have to answer now."));
        // What comes of the thought may be a call as well as an answer.
        let acted = !completion.text.trim().is_empty() || !completion.reply.calls.is_empty();
        assert!(acted, "{completion:?}");
    }
}
