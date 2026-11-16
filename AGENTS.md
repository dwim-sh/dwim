This is `dwim`, a coding agent harness in Rust: the agent, the model
(`qwen3.8-27b`, a 16 GB Q4_0 GGUF), and the engine that runs it, in one
program. One package: `lib.rs` is the library, `cli/` and
`testing/difftest/` are the binaries.

## Building and Verifying

- **Build**: `cargo build`
- **Test**: `cargo test`
- **Lint**: `cargo clippy --all-targets -- -D warnings`
- **Format**: `cargo fmt --check`
- **Real model**, if the weights are in the cache:
  `cargo test real_model -- --ignored --nocapture` and
  `cargo run --bin difftest`
- **Fetch the weights** (16 GB, into the Hugging Face cache):
  `dwim --download`
- Before calling a change done, run `./target/debug/dwim` in a terminal
  and ask the model something.
- GPU tests skip a backend whose GPU is not there. The GPU is Metal on
  macOS, Vulkan 1.1 with `VK_KHR_push_descriptor` elsewhere.

## Layout

- `cli/` - the `dwim` command: options, the shell, `serve`, `bench`
- `harness/` - the agent: system prompt (includes this file), tool loop,
  the plan of the work, compaction, thinking budget, transcript
- `providers/` - where a session's model runs: `Provider` is a model
  asked with the whole conversation each time, as a chat API is, and
  `conversation` the `Session` the harness holds on one. A directory
  each: `dwim` is the dwim server over its Unix socket (`remote`), and
  reaching or starting that server (`launch`); `openrouter` is a model
  at OpenRouter over its chat completions API
- `server/` - the process that has the model, and the `protocol` it is
  asked in: a request carries the whole conversation, and the server
  reads only what it has not read of it
- `inference/` - running the model on this machine: `gguf`, `quants`,
  `qwen35` (the transformer, documented there), `model` (loading and
  sizing), `local`, `sampler`, `states` (saved system prompts), `fetch`
  (downloads into the cache)
- `model/` - what is known of the model wherever it runs: `session` (the
  `Session` the harness holds), `chat` (a conversation as a request
  carries it), `format` (the chat format), `tokenizer`, `gguf` (a file's
  header), `hub` (the Hugging Face cache). It takes
  nothing from the directories above, and `providers/` take nothing from
  `inference/`
- `tools/` - the tools, one file each; `call` parses a call
- `gpu/` - the `Device` trait; `cpu` is the reference, `vulkan` (WGSL
  kernels, compiled by `build.rs`), `metal` (MSL kernels), `quant`, `float`
- `sys/` - what is asked of the operating system: `darwin/` and `linux/`
  with the same modules, `unix/` for what both do the same way, all
  re-exported as `sys::memory` and so on
- `scripts/` - `swebench.py`, `release`
- `man/dwim.1` - the manual page
- `third_party/` - references to read, not built

## Tests

A test checks behavior a user or the model would find broken. **Never**
add a test that only pins how the source is written:

- The wording, order, or width of help text, a status line, or any text
  for a person to read. Run `dwim` and read it.
- A table checked against itself or another table: sorted, no duplicates,
  names every entry.
- A value copied from the source, or a one-line function restated as its
  expected value.

If a test can only fail when someone edits text or a table on purpose,
do not write it.

## Coding Style

- **Never `pub(crate)`.** A definition is `pub` or not.
- One `use` per crate with nested paths, grouped `std`, external crates,
  `crate`, separated by blank lines.
- Doc comments say what a thing is or does, in plain sentences.
- A file reads top to bottom: a function comes before the ones it calls,
  helpers and foreign declarations last.
- Code that differs by system goes in `sys/`, not behind a `cfg` where it
  is used.

## Commits and Pull Requests

- A subject is `area: What changes`, as in `tools: Read only regular
  files`: the area is the directory or module (`sys`, `gpu/vulkan`, `ci`),
  the rest imperative, all within 72 characters.
- It names the things that change, not the intent: `sys: Move libc, mmap
  and dirs callers behind sys/unix`, not `Move what dwim asks of the
  operating system into sys/`.
- The body says why, and what was and was not verified.
- A pull request's title is a subject too, written the same way.

`CONTRIBUTING.md` has the longer version, and how to run SWE-bench Lite.
