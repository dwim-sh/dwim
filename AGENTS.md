This is `dwim`, a coding agent harness in Rust: the agent, the model
(`inkling2-mini`, a 17.6 GB Q4_K_XL GGUF), and the engine that runs it,
in one program. One package: `lib.rs` is the library, `cli/` and
`testing/difftest/` are the binaries.

## Building and Verifying

- **Build**: `cargo build`
- **Test**: `cargo test`
- **Lint**: `cargo clippy --all-targets -- -D warnings`
- **Format**: `cargo fmt --check`
- **Specs**: `make specs` checks the Lean specs: their
  examples and their proofs. It needs `lake` from a Lean toolchain.
- **Code against the specs**: `make difftest-compaction` runs
  `difftest compaction`, which checks compaction's code against its Lean
  spec on generated cases.
- **All of the above**: `make check`
- **Spec site**: `make site` renders the Lean specs as a web site and
  prints where it is.
- **Real model**, if the weights are in the cache:
  `cargo test real_model -- --ignored --nocapture` and
  `cargo run --bin difftest model`
- **Your own server**, if one is running:
  `DWIM_TEST_ENDPOINT=http://host:8080 cargo test real_endpoint -- --ignored`
- **Fetch the weights** (17.6 GB, into the Hugging Face cache):
  `dwim --download`
- Before calling a change done, run `./target/debug/dwim` in a terminal
  and ask the model something.
- GPU tests skip a backend whose GPU is not there. The GPU is Metal on
  macOS, and elsewhere an NVIDIA driver with the CUDA toolkit's NVRTC, or
  Vulkan 1.1 with `VK_KHR_push_descriptor`.

## Layout

- `cli/` - the `dwim` command: options, `config` (what is kept in
  `config.json`), and `cmd/`, what it does, a module each: the shell,
  `once`, `serve`, `bench`, `download`
- `harness/` - the agent: system prompt (includes this file), tool loop,
  the plan of the work, compaction, thinking budget, transcript
- `providers/` - where a session's model runs: `Provider` is a model
  asked with the whole conversation each time, as a chat API is, and
  `conversation` the `Session` the harness holds on one. A directory
  each: `dwim` is the dwim server over its Unix socket (`remote`), and
  reaching or starting that server (`launch`); `hosted` is a model at a
  host over its chat completions API, and `openrouter`, `vercel`
  (Vercel's AI Gateway) and `cerebras` the hosts: where each is asked,
  what it is asked in its own way, and the key to it; `endpoint` the
  user's own server
- `server/` - the process that has the model, and the `protocol` it is
  asked in: a request carries the whole conversation, and the server
  reads only what it has not read of it
- `inference/` - running the model on this machine: `gguf`, `quants`,
  `arch` (which architecture a file is of), `qwen35` (the transformer,
  documented there) and `inkling2` (Inkling 2 Mini's, likewise), `loaded`
  (loading and sizing), `local`, `sampler`, `states` (saved system
  prompts), `fetch` (downloads into the cache)
- `model/` - what is known of the model wherever it runs: `session` (the
  `Session` the harness holds), `chat` (a conversation as a request
  carries it), `format` (the chat format), `tokenizer`, `gguf` (a file's
  header), `hub` (the Hugging Face cache), and a directory for each
  family of models with what is its own: `qwen`, and `inkling` (Inkling
  2 Mini, which `--model inkling2-mini` runs, on Metal, CUDA and
  Vulkan). It takes nothing from the directories above, and `providers/`
  take nothing from `inference/`
- `tools/` - the tools, one file each; `call` parses a call
- `gpu/` - the `Device` trait; `cpu` is the reference, `cuda` (CUDA C
  kernels, compiled by NVRTC when the device opens), `vulkan` (WGSL
  kernels, compiled by `build.rs`), `metal` (MSL kernels), `platform`
  (`Gpu`, whichever of them opens: CUDA before Vulkan), `quant`, `float`
- `sys/` - what is asked of the operating system: `darwin/` and `linux/`
  with the same modules, `unix/` for what both do the same way, all
  re-exported as `sys::memory` and so on
- `evals/` - the benchmarks `dwim` is run over, a directory each with its
  `run.py`: `swebench` (Lite, Verified, Verified Mini); `results/` is
  the report of each run over a whole dataset, and `runs/` what the runs
  leave, which is not kept in git
- `specs/` - Lean specs of the code, a file for each Rust one they
  specify: the intents, what the part is for, in plain sentences; the
  design as Lean definitions, with examples checked when built; and
  proofs that the design meets the intents; `site/` renders them as a web
  site
- `scripts/` - `release`
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

A pull request's description is mandatory and is written for a reader
who has not seen the code or the commits, and does not know how the
part being changed works. It stands on its own, in four parts:

1. **Background**: how the part being changed works today, as far as
   the problem needs it. A few sentences.
2. **Problem**: what is wrong or missing, and why it matters, with the
   case that showed it.
3. **Solution**: what the change does about it, as a whole. When most
   of the files changed are plumbing, say which have the logic.
4. **Results**: numbers for the claims, such as a benchmark, timings,
   sizes, or tokens, measured before and after where there is a before.
   Then what was verified and what was not. Say so when a claim cannot
   be measured.

Write it as it would be explained aloud: short sentences, plain words,
a number or an example for a claim, a table for before and after.

Leave the details of each change to the commits. A description does not
list the commits, retell their messages, or list every flag and fix. When
a pull request changes, rewrite its description to match.

`CONTRIBUTING.md` has the longer version, and how to run SWE-bench.
