//! OpenRouter as the host of a model: where it is asked, what it is asked
//! in its own way, and the key to it, which is the environment's or one
//! kept from a sign-in.

pub mod login;

use serde_json::{Value, json};

use crate::{
    model::{Model, Result, Sampling},
    providers::{
        Kind,
        hosted::{self, Host, Hosted},
    },
};

/// The environment variable the key to OpenRouter is read from, ahead of
/// the key kept from a sign-in.
pub const KEY: &str = "OPENROUTER_API_KEY";

/// OpenRouter, as the host of a model.
pub const HOST: Host = Host {
    kind: Kind::OpenRouter,
    name: "OpenRouter",
    url: "https://openrouter.ai/api/v1/chat/completions",
    model,
    ask,
    key: KEY,
    refused: "/login signs in again",
};

/// `model` at OpenRouter, with the key in the environment or
/// kept from a sign-in, with room for `context` tokens of conversation,
/// or as many as a model at a host has unless told, sampled at
/// `temperature`, or as the model's makers recommend.
pub fn open(model: Model, context: Option<usize>, temperature: Option<f32>) -> Result<Hosted> {
    let key = key()
        .ok_or_else(|| format!("there is no key to OpenRouter: /login signs in, or set {KEY}"))?;
    Ok(Hosted::new(&HOST, key, model, context, temperature))
}

/// `model` as OpenRouter names it.
pub fn model(model: Model) -> &'static str {
    match model {
        Model::Qwen27b => "qwen/qwen3.8-27b",
        Model::QwenMax => "qwen/qwen3.8-2.4t-a95b",
    }
}

/// The key to OpenRouter, if there is one: the environment's, or else the
/// one kept from a sign-in.
pub fn key() -> Option<String> {
    hosted::key::find(&HOST)
}

/// Adds to a request what OpenRouter is asked in its own way: to say what
/// the reply used and cost, to sample from the likeliest tokens only, and
/// to limit the thought to a number of tokens.
fn ask(body: &mut Value, sampling: &Sampling, thought: Option<usize>, _ended: bool) {
    body["usage"] = json!({ "include": true });
    body["top_k"] = json!(sampling.top_k);
    if let Some(thought) = thought {
        body["reasoning"] = json!({ "max_tokens": thought });
    }
}
