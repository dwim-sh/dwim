# `dwim`

A coding agent harness in Rust.

## Layout

* `cli/`: the `dwim` command: arguments, fetching the model, and the terminal UI.
* `models/`: the language model: `gguf` reads the file the weights ship in, `tokenizer` builds the byte-level BPE tokenizer from the vocabulary in it, `bonsai` is the transformer as operations on a device, and `chat` is the conversation around it.
* `harness/`: the agent around the model: runs the tools it calls and feeds the results back until it replies with text alone. The system prompt declares the `bash` tool in the form the model's chat template uses: a JSON signature, and calls as `<function=…>` blocks.
* `gpu/`: devices a model runs on: the `Device` trait, `cpu` as the reference, `vulkan` with hand-written WGSL compute kernels compiled to SPIR-V at build time, and `metal` with hand-written Metal Shading Language kernels compiled when the device opens. `ternary` is the 1.75-bit weight format. `Gpu` is Metal on Apple platforms and Vulkan elsewhere. The Vulkan ternary matmul comes in three kernels picked by shape: a lane per row for matrices of few rows, eight rows per workgroup for tall ones, and four tokens at a time for batches; its attention splits positions into chunks of 256 across workgroups and merges them, reading each key/value head once for the query heads that share it; and it submits command buffers every 128 kernels from a ring so the GPU runs while the CPU records.
* `scripts/`: `swebench.py`, which runs the agent over SWE-bench Lite instances and writes the patches it makes as predictions.
* `third_party/`: reference implementations studied for design, not built.

## Model

The one model is `bonsai-2-27b`: PrismML's Ternary Bonsai 2 27B, Qwen3.8-27B with its matrices made ternary (`{-1, 0, 1}` with an f16 scale per 128 weights, 5.95 GB as shipped in the `PTQ1_0` GGUF packing). It is a hybrid: 64 layers of which every fourth is full attention (24 query heads, 4 key/value heads, 256 wide, the first 64 elements rotated by RoPE) and the rest Gated DeltaNet linear attention (16 key heads and 48 value heads of 128, a causal convolution of 4 taps, and a 128×128 recurrent state per value head). The matrices are stored in a rotated basis: activations are rotated by a blockwise (1024) Walsh-Hadamard transform with fixed signs before every ternary matmul, and the embedding table's rows are rotated back after lookup. The file stores the linear-attention value heads in llama.cpp's tiled head order; `bonsai.rs` puts them back into the checkpoint's grouped order when loading. The whole model runs on the device; only the embedding lookup is on the CPU.

The model thinks by default (its template opens `<think>` for it), calls tools as `<tool_call><function=bash>…`, and samples at temperature 1.0, top-k 20, top-p 0.95 as the file recommends. Its own chat template puts the tools at the top of the system prompt, which the harness follows.

## Building

`cargo build` needs nothing beyond Rust: the Vulkan kernels are compiled by `naga` in `gpu/build.rs`, and Metal compiles its own at runtime. Running on the GPU needs Metal on macOS, and a Vulkan 1.1 driver with `VK_KHR_push_descriptor` elsewhere. The weights are fetched into `~/.cache/dwim/models/` on first use (a resumable 6 GB download) and need about 6 GB of device memory plus 64 KB per token of context for the key/value caches.

## Verifying

* `cargo test -p dwim-gpu` checks every Vulkan and Metal kernel against the CPU reference, and skips a backend whose GPU is not available.
* `cargo test -p dwim-models` checks the GGUF reader and runs a tiny random model on the CPU and the GPU and compares their logits. `cargo test -p dwim-models real_model -- --ignored --nocapture` completes a prompt with the real model if it is in the cache: it must say Paris, as PrismML's llama.cpp does, and it reports prompt and decode speeds.
* Exercise `./target/debug/dwim` in a terminal (or tmux) and ask the model something before calling a change done.
* `CONTRIBUTING.md` has the longer version of all this, and how to run SWE-bench Lite.

## Coding Style

* Never use `pub(crate)` in source code. A definition is either `pub` or not.
* Use one `use` per crate with nested paths (`use std::{path::Path, sync::Arc};`, not a `use` per item), grouped as `std`, external crates, then `crate`, separated by blank lines.
* Doc comments say what a thing is or does, in plain sentences.
