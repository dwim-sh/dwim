//! Vercel's AI Gateway as the host of a model: where it is asked, what it
//! is asked in its own way, and the key to it, which is the environment's
//! or one kept from signing in.
//!
//! The gateway makes keys in its dashboard only, so signing in to it from
//! here is giving a key made there.

use serde_json::{Value, json};

use crate::{
    model::{Effort, Model, Result, Sampling},
    providers::{
        Kind,
        hosted::{self, Host, Hosted},
    },
};

/// The environment variable the key to the gateway is read from, ahead
/// of the key kept from signing in.
pub const KEY: &str = "AI_GATEWAY_API_KEY";

/// Vercel's AI Gateway, as the host of a model.
pub const HOST: Host = Host {
    kind: Kind::Vercel,
    name: "Vercel",
    url: "https://ai-gateway.vercel.sh/v1/chat/completions",
    model,
    ask,
    key: KEY,
    refused: "make another in the Vercel dashboard, and give it with /login",
};

/// `model` at the gateway, with the key in the environment
/// or kept from signing in, reasoning at `effort`, with room for
/// `context` tokens of conversation, or as many as a model at a host has
/// unless told, sampled at `temperature`, or as the model's makers
/// recommend. Inkling is run on this machine only.
pub fn open(
    model: Model,
    effort: Effort,
    context: Option<usize>,
    temperature: Option<f32>,
) -> Result<Hosted> {
    if model == Model::InklingMini {
        return Err(format!("{model} is not run at Vercel").into());
    }
    let key = key()
        .ok_or_else(|| format!("there is no key to Vercel: /login takes one, or set {KEY}"))?;
    Ok(Hosted::new(&HOST, key, model, effort, context, temperature))
}

/// `model` as the gateway names it. Inkling is named as it would be, for
/// [`open`] to refuse.
pub fn model(model: Model) -> &'static str {
    match model {
        Model::Qwen27b => "alibaba/qwen3.8-27b",
        Model::QwenMax => "alibaba/qwen3.8-2.4t-a95b",
        Model::InklingMini => "thinkingmachines/inkling-2-mini",
    }
}

/// The key to the gateway, if there is one: the environment's, or else
/// the one kept from signing in.
pub fn key() -> Option<String> {
    hosted::key::find(&HOST)
}

/// Adds to a request what the gateway is asked in its own way: to say
/// what the reply used, as the last event of the stream. The gateway
/// lists no way to sample from the likeliest tokens only, so that is left
/// to the provider. The model is asked to reason at `effort`, or not to
/// when there is none.
fn ask(body: &mut Value, _sampling: &Sampling, effort: Option<Effort>) {
    body["stream_options"] = json!({ "include_usage": true });
    body["reasoning"] = json!({ "effort": effort.map_or("none", Effort::name) });
}
