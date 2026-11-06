# `dwim`

A coding agent harness in Rust: the agent, the model, and the engine that
runs it, in one program.

## Layout

* `cli/`: the `dwim` command: arguments, fetching the model, the terminal UI,
  and `bench`, which times the model at depths of context.
* `models/`: the language model: `gguf` reads the weights file, `quants`
  unpacks the block formats llama.cpp quantizes to, `tokenizer` builds the
  BPE tokenizer from it, `qwen35` is the transformer as operations on a
  device, `chat` is the conversation around it, and `load` picks the
  transformer by the file's architecture.
* `harness/`: the agent around the model: the system prompt (which includes
  this file), the tool loop, compaction near the end of the context, the
  thinking budget if one is set and the cap on a turn's calls, and the
  session transcript.
* `tools/`: the tools, one file each, and `call`, which parses a call as the
  model writes it.
* `gpu/`: devices a model runs on: the `Device` trait, `cpu` as the reference,
  `vulkan` (WGSL kernels compiled to SPIR-V at build time), and `metal`
  (MSL kernels compiled when the device opens). `ternary` is the 1.75-bit
  weight format; `quant` the 4- and 8-bit block formats.
* `scripts/`: `swebench.py` runs SWE-bench Lite; `release` cuts a release.
* `testing/`: `difftest`, differential testing of the GPU against the CPU
  with the real weights.
* `man/`: `dwim.1`, the manual page.
* `third_party/`: reference implementations studied for design, not built.

## Models

Both models are Qwen3.8-27B, a hybrid of full attention every fourth layer
and Gated DeltaNet linear attention elsewhere, which `models/qwen35.rs` runs
from whichever weights the file has and documents in detail. Both think by
default and call tools as `<tool_call><function=bash>…`.

* `bonsai-2-27b`, the default, is PrismML's Ternary Bonsai 2 27B: the
  matrices made ternary (6 GB), in a rotated basis the model undoes with
  Hadamard rotations of the activations.
* `qwen3.8-27b` is the model as Qwen released it, in bartowski's Q4_0 GGUF
  conversion (16 GB): Q4_0 and Q8_0 matrices run as they are, and Q4_1 and
  Q6_K are quantized again as Q8_0 on loading. It needs a 32 GiB Mac.

## Building

`cargo build` needs nothing beyond Rust. The GPU needs Metal on macOS or a
Vulkan 1.1 driver with `VK_KHR_push_descriptor` elsewhere. The weights (6 GB)
are fetched into the user's cache directory on first use.

## Verifying

* `cargo test --workspace`, `cargo clippy --workspace --all-targets -- -D
  warnings`, and `cargo fmt --check`, as CI runs them. GPU tests skip a
  backend whose GPU is not available.
* `cargo test -p dwim-models real_model -- --ignored --nocapture` and
  `cargo run -p dwim-difftest` check the real models that are in the cache
  (`--model qwen3.8-27b` picks the other for difftest).
* Exercise `./target/debug/dwim` in a terminal and ask the model something
  before calling a change done.
* `CONTRIBUTING.md` has the longer version, and how to run SWE-bench Lite.

## Coding Style

* Never use `pub(crate)`. A definition is either `pub` or not.
* One `use` per crate with nested paths, grouped as `std`, external crates,
  then `crate`, separated by blank lines.
* Doc comments say what a thing is or does, in plain sentences.
