# Hack

A coding agent harness in Rust.

## Layout

* `cli/`: the `hack` command: arguments, fetching the model, and the terminal UI.
* `models/`: language models: the tokenizer, safetensors weights, the chat, and `qwen3`, the transformer as operations on a device.
* `harness/`: the agent around the model: runs the tools it calls and feeds the results back until it replies with text alone.
* `gpu/`: devices a model runs on: the `Device` trait, `cpu` as the reference, `vulkan` with hand-written WGSL compute kernels compiled to SPIR-V at build time, and `metal` with hand-written Metal Shading Language kernels compiled when the device opens. `Gpu` is Metal on Apple platforms and Vulkan elsewhere.
* `third_party/`: reference implementations studied for design, not built.

## Building

`cargo build` needs nothing beyond Rust: the Vulkan kernels are compiled by `naga` in `gpu/build.rs`, and Metal compiles its own at runtime. Running on the GPU needs Metal on macOS, and a Vulkan 1.1 driver with `VK_KHR_push_descriptor` elsewhere. Weights are fetched into `~/.cache/hack/models/` on first use.

## Verifying

* `cargo test -p hack-gpu` checks every Vulkan and Metal kernel against the CPU reference, and skips a backend whose GPU is not available.
* Exercise `./target/debug/hack` in a terminal (or tmux) and ask the model something before calling a change done.

## Coding Style

* Never use `pub(crate)` in source code. A definition is either `pub` or not.
* Use one `use` per crate with nested paths (`use std::{path::Path, sync::Arc};`, not a `use` per item), grouped as `std`, external crates, then `crate`, separated by blank lines.
* Doc comments say what a thing is or does, in plain sentences.
