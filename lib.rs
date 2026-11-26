//! A coding agent harness: the agent, the model, and the engine that runs it.
//!
//! A program that wants the agent in it opens an [`agent::Agent`], which
//! puts the rest together. The modules under it are the parts: the
//! `harness` and its `tools`, the `providers` a session's model runs at,
//! and, for a model that runs on this machine, the `server` that has it,
//! `inference` and the `gpu`.

pub mod agent;
pub mod gpu;
pub mod harness;
pub mod inference;
pub mod model;
pub mod providers;
pub mod server;
pub mod sys;
pub mod tools;
