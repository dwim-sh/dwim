//! Cerebras as the host of a model: where it is asked, what it is asked
//! in its own way, and the key to it, which is the environment's or one
//! kept from signing in.
//!
//! Cerebras runs the medium model only. Its keys are made in its cloud
//! console, so signing in to it from here is giving a key made there.

use serde_json::{Value, json};

use crate::{
    model::{Result, Sampling, Variant},
    providers::{
        Kind,
        hosted::{self, Host, Hosted},
    },
};

/// The environment variable the key to Cerebras is read from, ahead of
/// the key kept from signing in.
pub const KEY: &str = "CEREBRAS_API_KEY";

/// Cerebras, as the host of a model.
pub const HOST: Host = Host {
    kind: Kind::Cerebras,
    name: "Cerebras",
    url: "https://api.cerebras.ai/v1/chat/completions",
    model,
    ask,
    key: KEY,
    refused: "make another in the Cerebras cloud console, and give it with /login",
};

/// The medium model at Cerebras, with the key in the environment or kept
/// from signing in, with
/// room for `context` tokens of conversation, or as many as a model at a
/// host has unless told, sampled at `temperature`, or as the model's
/// makers recommend. The large model does not run there.
pub fn open(variant: Variant, context: Option<usize>, temperature: Option<f32>) -> Result<Hosted> {
    if variant == Variant::Large {
        return Err(format!("{variant} does not run at Cerebras").into());
    }
    let key = key()
        .ok_or_else(|| format!("there is no key to Cerebras: /login takes one, or set {KEY}"))?;
    Ok(Hosted::new(&HOST, key, variant, context, temperature))
}

/// `variant` as Cerebras names it. Only the medium model runs there; the
/// large one is named as it would be, for [`open`] to refuse.
pub fn model(variant: Variant) -> &'static str {
    match variant {
        Variant::Medium => "qwen-3.8-27b",
        Variant::Large => "qwen-3.8-2.4t-a95b",
    }
}

/// The key to Cerebras, if there is one: the environment's, or else the
/// one kept from signing in.
pub fn key() -> Option<String> {
    hosted::key::find(&HOST)
}

/// Adds to a request what Cerebras is asked in its own way. It says what
/// the reply used without being asked, and lists no way to sample from
/// the likeliest tokens only, so that is left to it.
///
/// Cerebras takes how much the model thinks as an effort, not as a
/// number of tokens, and thinks at high effort unless told. A thought
/// that is ended here at its limit is left to the model, and one that is
/// not, which is a plain answer's, is asked to be brief.
fn ask(body: &mut Value, _sampling: &Sampling, thought: Option<usize>, ended: bool) {
    if thought.is_some() && !ended {
        body["reasoning_effort"] = json!("low");
    }
}
