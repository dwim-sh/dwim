//! OpenRouter as the host of a model: where it is asked, what it is asked
//! in its own way, and the key to it, which is the environment's or one
//! kept from a sign-in.

pub mod login;

use std::env;

use serde_json::{Value, json};

use crate::{
    model::{Result, Sampling, Variant},
    providers::{
        Kind,
        hosted::{Host, Hosted},
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
    refused: "/provider signs in again",
};

/// The model `variant` at OpenRouter, with the key in the environment or
/// kept from a sign-in, with room for `context` tokens of conversation,
/// or as many as a model at a host has unless told, sampled at
/// `temperature`, or as the model's makers recommend.
pub fn open(variant: Variant, context: Option<usize>, temperature: Option<f32>) -> Result<Hosted> {
    let key = key().ok_or_else(|| {
        format!("there is no key to OpenRouter: /provider signs in, or set {KEY}")
    })?;
    Ok(Hosted::new(&HOST, key, variant, context, temperature))
}

/// `variant` as OpenRouter names it.
pub fn model(variant: Variant) -> &'static str {
    match variant {
        Variant::Medium => "qwen/qwen3.8-27b",
        Variant::Large => "qwen/qwen3.8-2.4t-a95b",
    }
}

/// The key to OpenRouter, if there is one: the environment's, or else the
/// one kept from a sign-in.
pub fn key() -> Option<String> {
    env::var(KEY)
        .ok()
        .filter(|key| !key.trim().is_empty())
        .or_else(login::saved)
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
