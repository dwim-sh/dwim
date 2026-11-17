//! Cerebras as the host of a model: where it is asked, what it is asked
//! in its own way, and the key to it, which is the environment's.
//!
//! Cerebras runs the medium model only. Its keys are made in its cloud
//! console, so there is no signing in to it from here, and no key kept.

use std::env;

use serde_json::{Value, json};

use crate::{
    model::{Result, Sampling, Variant},
    providers::{
        Kind,
        hosted::{Host, Hosted},
    },
};

/// The environment variable the key to Cerebras is read from.
pub const KEY: &str = "CEREBRAS_API_KEY";

/// Cerebras, as the host of a model.
pub const HOST: Host = Host {
    kind: Kind::Cerebras,
    name: "Cerebras",
    url: "https://api.cerebras.ai/v1/chat/completions",
    model,
    ask,
    refused: "make another in the Cerebras cloud console, and set CEREBRAS_API_KEY to it",
};

/// The medium model at Cerebras, with the key in the environment, with
/// room for `context` tokens of conversation, or as many as a model at a
/// host has unless told, sampled at `temperature`, or as the model's
/// makers recommend. The large model does not run there.
pub fn open(variant: Variant, context: Option<usize>, temperature: Option<f32>) -> Result<Hosted> {
    if variant == Variant::Large {
        return Err(format!("{variant} does not run at Cerebras").into());
    }
    let key = key().ok_or_else(|| {
        format!("there is no key to Cerebras: set {KEY} to one made in the Cerebras cloud console")
    })?;
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

/// The key to Cerebras, if the environment has one.
pub fn key() -> Option<String> {
    env::var(KEY).ok().filter(|key| !key.trim().is_empty())
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
