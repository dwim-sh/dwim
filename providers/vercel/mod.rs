//! Vercel's AI Gateway as the host of a model: where it is asked, what it
//! is asked in its own way, and the key to it, which is the environment's.
//!
//! The gateway makes keys in its dashboard only, so there is no signing in
//! to it from here, and no key kept.

use std::env;

use serde_json::{Value, json};

use crate::{
    model::{Result, Sampling, Variant},
    providers::{
        Kind,
        hosted::{Host, Hosted},
    },
};

/// The environment variable the key to the gateway is read from.
pub const KEY: &str = "AI_GATEWAY_API_KEY";

/// Vercel's AI Gateway, as the host of a model.
pub const HOST: Host = Host {
    kind: Kind::Vercel,
    name: "Vercel AI Gateway",
    url: "https://ai-gateway.vercel.sh/v1/chat/completions",
    model,
    ask,
    refused: "make another in the Vercel dashboard, and set AI_GATEWAY_API_KEY to it",
};

/// The model `variant` at the gateway, with the key in the environment,
/// with room for `context` tokens of conversation, or as many as a model
/// at a host has unless told, sampled at `temperature`, or as the model's
/// makers recommend.
pub fn open(variant: Variant, context: Option<usize>, temperature: Option<f32>) -> Result<Hosted> {
    let key = key().ok_or_else(|| {
        format!(
            "there is no key to Vercel AI Gateway: set {KEY} to one made in the Vercel dashboard"
        )
    })?;
    Ok(Hosted::new(&HOST, key, variant, context, temperature))
}

/// `variant` as the gateway names it.
pub fn model(variant: Variant) -> &'static str {
    match variant {
        Variant::Medium => "alibaba/qwen3.8-27b",
        Variant::Large => "alibaba/qwen3.8-2.4t-a95b",
    }
}

/// The key to the gateway, if the environment has one.
pub fn key() -> Option<String> {
    env::var(KEY).ok().filter(|key| !key.trim().is_empty())
}

/// Adds to a request what the gateway is asked in its own way: to say
/// what the reply used, as the last event of the stream. The gateway
/// lists no way to sample from the likeliest tokens only, so that is left
/// to the provider.
///
/// The gateway takes how much these models think as an effort, not as a
/// number of tokens. A thought that is ended here at its limit is left to
/// the model, and one that is not, which is a plain answer's, is asked to
/// be brief.
fn ask(body: &mut Value, _sampling: &Sampling, thought: Option<usize>, ended: bool) {
    body["stream_options"] = json!({ "include_usage": true });
    if thought.is_some() && !ended {
        body["reasoning"] = json!({ "effort": "low" });
    }
}
