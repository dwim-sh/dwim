# `dwim`

A coding agent harness in Rust: the agent, the model, and the engine that
runs it, in one program.

## Layout

* `cli/`: the `dwim` command: arguments, fetching the model, the terminal UI,
  and `bench`, which times the model at depths of context.
* `models/`: the language model: `gguf` reads the weights file, `tokenizer`
  builds the BPE tokenizer from it, `bonsai` is the transformer as operations
  on a device, `chat` is the conversation around it. `draft` is the DFlash2
  draft that proposes tokens, read by `safetensors`, and `speculative` runs
  it ahead of the model; `cache` finds both models' files.
* `harness/`: the agent around the model: the system prompt (which includes
  this file), the tool loop, compaction near the end of the context, the
  thinking budget if one is set and the cap on a turn's calls, and the
  session transcript.
* `tools/`: the tools, one file each, and `call`, which parses a call as the
  model writes it.
* `gpu/`: devices a model runs on: the `Device` trait, `cpu` as the reference,
  `vulkan` (WGSL kernels compiled to SPIR-V at build time), and `metal`
  (MSL kernels compiled when the device opens). `ternary` is the 1.75-bit
  weight format, `twobit` the 2-bit one a file may come in, and `affine4`
  the 4-bit one the draft's weights are quantized to.
* `scripts/`: `swebench.py` runs SWE-bench Lite; `release` cuts a release.
* `testing/`: `difftest`, differential testing of the GPU against the CPU
  with the real weights.
* `man/`: `dwim.1`, the manual page.
* `third_party/`: reference implementations studied for design, not built.

## Model

The one model is `bonsai-2-27b`, PrismML's Ternary Bonsai 2 27B: Qwen3.8-27B
with ternary matrices, a hybrid of full attention every fourth layer and
Gated DeltaNet linear attention elsewhere. It thinks by default and calls
tools as `<tool_call><function=bash>…`. Its details are documented in
`models/bonsai.rs`.

It decodes with a draft, `incoai/Qwen3.8-27B-DFlash2`, unless `--no-draft`
says otherwise: a five-layer transformer that reads the model's hidden
states and proposes seven tokens at a time, which the model checks in one
pass, keeping those it would have picked itself. The output is the model's
own, token for token at a temperature of zero; a cycle yields several tokens
for little more than one token's pass. Its bf16 weights are quantized to
four bits as they load. Its details are documented in `models/draft.rs` and
`models/speculative.rs`. Its file comes in two packings of the same weights,
`PTQ1_0` at 1.75 bits a weight, which `dwim` fetches, and `PQ2_0` at two,
which other programs fetch; either is read, from `dwim`'s cache or Hugging
Face's (`models/cache.rs`), into the 1.75-bit packing, which the devices
compute on: holding the weights two bits each on the device was measured
slower, as a token generated costs a read of every weight.

## Building

`cargo build` needs nothing beyond Rust. The GPU needs Metal on macOS or a
Vulkan 1.1 driver with `VK_KHR_push_descriptor` elsewhere. The weights (6 GB,
and 4 GB for the draft, which loads as 1.2 GB) are fetched into the user's
cache directory on first use, unless Hugging Face's cache has them.

## Verifying

* `cargo test --workspace`, `cargo clippy --workspace --all-targets -- -D
  warnings`, and `cargo fmt --check`, as CI runs them. GPU tests skip a
  backend whose GPU is not available.
* `cargo test -p dwim-models real_model -- --ignored --nocapture` and
  `cargo run -p dwim-difftest` check the real model if it is in the cache.
* Exercise `./target/debug/dwim` in a terminal and ask the model something
  before calling a change done.
* `CONTRIBUTING.md` has the longer version, and how to run SWE-bench Lite.

## Coding Style

* Never use `pub(crate)`. A definition is either `pub` or not.
* One `use` per crate with nested paths, grouped as `std`, external crates,
  then `crate`, separated by blank lines.
* Doc comments say what a thing is or does, in plain sentences.
