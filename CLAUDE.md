# Hack

A coding agent harness in Rust.

## Layout

* `cli/`: the `hack` command: arguments, fetching the model, and the terminal UI.
* `models/`: language models: the tokenizer, safetensors weights, the chat, and the transformers as operations on a device: `qwen3` (dense) and `qwen3_moe` (the same attention with a mixture of experts). `pack` is the converted-model file format, `q4` the four-bit weight format with its AVX2 dot product, and `experts` the routed experts on the CPU.
* `harness/`: the agent around the model: runs the tools it calls and feeds the results back until it replies with text alone. The system prompt declares the `bash` tool in the form the model's chat template uses: JSON for Qwen3, `<function=…>` blocks for Qwen3-Coder; each entry in `cli/models.rs` says which.
* `gpu/`: devices a model runs on: the `Device` trait, `cpu` as the reference, `vulkan` with hand-written WGSL compute kernels compiled to SPIR-V at build time, and `metal` with hand-written Metal Shading Language kernels compiled when the device opens. `Gpu` is Metal on Apple platforms and Vulkan elsewhere.
* `scripts/`: `swebench.py`, which runs the agent over SWE-bench Lite instances and writes the patches it makes as predictions.
* `third_party/`: reference implementations studied for design, not built.

## Models

* `qwen3-0.6b` (default) is one safetensors file, run entirely on the device in bf16.
* `qwen3-coder-30b-a3b` is 30B parameters in 16 shards (61 GB bf16). `cli/convert.rs` streams the shards one at a time into `~/.cache/hack/models/qwen3-coder-30b-a3b/model.hack`, quantizing the routed experts to four bits (about 16 GB) and keeping the dense layers bf16, deleting each shard once converted; it resumes where it left off. At run time the dense layers live on the GPU and the experts stay in the pack's memory mapping, run on the CPU; each layer's activations are read back for the router and the experts it picks, both on the CPU. The experts want to stay in the page cache. The model is not offered on Apple platforms, where the CPU experts have no fast path.

## Building

`cargo build` needs nothing beyond Rust: the Vulkan kernels are compiled by `naga` in `gpu/build.rs`, and Metal compiles its own at runtime. Running on the GPU needs Metal on macOS, and a Vulkan 1.1 driver with `VK_KHR_push_descriptor` elsewhere. Weights are fetched into `~/.cache/hack/models/` on first use.

## Verifying

* `cargo test -p hack-gpu` checks every Vulkan and Metal kernel against the CPU reference, and skips a backend whose GPU is not available.
* `cargo test -p hack-models` checks the four-bit format, the pack, the CPU experts, and runs a tiny random Qwen3-MoE on the CPU and the GPU and compares their logits.
* Exercise `./target/debug/hack` in a terminal (or tmux) and ask the model something before calling a change done.
* `CONTRIBUTING.md` has the longer version of all this, and how to run SWE-bench Lite.

## Coding Style

* Never use `pub(crate)` in source code. A definition is either `pub` or not.
* Use one `use` per crate with nested paths (`use std::{path::Path, sync::Arc};`, not a `use` per item), grouped as `std`, external crates, then `crate`, separated by blank lines.
* Doc comments say what a thing is or does, in plain sentences.
