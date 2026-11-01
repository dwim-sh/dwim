# Contributing

## Building

`cargo build` needs nothing beyond Rust. The Vulkan kernels are compiled from
WGSL to SPIR-V by `naga` in `gpu/build.rs`, and Metal compiles its own kernels
from source when the device opens.

```
cargo build              # debug, with the model and the devices optimized
cargo build --release    # for anything that runs the model for long
make install             # a release build and the manual page, in ~/.local
```

`make install PREFIX=/usr/local` installs system-wide, which may need `sudo`;
`make uninstall` removes what `make install` put there.

Running the model on a GPU needs Metal on macOS, and elsewhere a Vulkan 1.1
driver with `VK_KHR_push_descriptor`. `--device cpu` runs anywhere, slowly.

The weights are downloaded on first use into `~/Library/Caches/dwim/models`
on macOS, and `~/.cache/dwim/models` elsewhere. Bonsai 2 27B, the one model,
is a 6 GB file; an interrupted download resumes on the next run.

## Running

```
dwim                     # the shell
dwim "what changed in the last commit?"
dwim --device cpu --context 4096 "what files are here?"
```

With a prompt, `dwim` answers it and exits. The reply goes to standard output
and nothing else does, so `dwim "…" 2>/dev/null` gives the reply alone, while
progress, the model's thinking, the commands it runs, and their output go to
standard error.

## Verifying a change

```
cargo test --workspace   # the GPU kernels are checked against the CPU reference
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --check
```

CI runs these with the toolchain `rust-toolchain.toml` pins, which `rustup`
picks up here too.

`cargo test -p dwim-gpu` skips a backend whose GPU is not available, so it
passes on a machine with no GPU without having checked anything. With the
model in the cache, `cargo test -p dwim-models real_model -- --ignored
--nocapture` completes a prompt with it and reports the speed, and
`cargo run -p dwim-difftest` checks the GPU against the CPU with it, on a
few prompts run in pieces of different sizes and on their continuations
run a token at a time. The CPU's side is
recorded into `target/difftest/` on the first run, which takes minutes, and
again only when the CPU's or the model's code changes; after that a run
takes seconds. `--repeat <n>` runs the GPU's side again, for mistakes it
makes only now and then.

Run the agent before calling a change done. The model is easy to break in
ways tests don't catch, such as replies that end with nothing to show, so
ask it something in the shell, and check a one-off prompt too.

The coding style is in `CLAUDE.md`. Commit messages say what changed and why,
in the imperative, with the reasoning in the body. A commit whose change
`dwim` made ends with the line `Generated-by: dwim`. The person who commits
it has read the change and stands behind it, and the message is theirs.

## SWE-bench Lite

[SWE-bench](https://www.swebench.com) gives an agent a real GitHub issue and
a repository at the commit before the fix, and checks the agent's patch by
running the project's tests. Lite is 300 instances from 12 Python projects.

```
scripts/swebench.py --instance pallets__flask-4045     # one instance
scripts/swebench.py --limit 10                         # the first ten
scripts/swebench.py                                    # all 300
```

The script builds `dwim`, and for each instance takes the repository as the
instance's SWE-bench image has it, with the project and what it needs
installed, gives `dwim` the issue with its shell commands running in a
container of that image (`dwim --shell`), takes what it changed as the
patch, and scores the patch with the official harness, which runs the
project's tests in the same image. It tells you what to install if Docker
or the harness is missing; the harness needs Python 3.10, and on Apple
Silicon a Docker that runs x86_64 images, such as Colima started with
`--vz-rosetta`. `--no-score` writes `predictions.jsonl` without scoring,
`--score-only` scores what an earlier run wrote, and `--help` lists the rest.

**The agent's commands run in the container, but the model runs here, and
its read and edit tools work on the checkout as you, on an untrusted
issue.** Run it in a virtual machine, not on a machine you care about.

## Releasing

The version is in the workspace's `Cargo.toml`, which every crate inherits.

```
scripts/release 0.2.0
git push --atomic origin main v0.2.0
```

`scripts/release` sets the version in `Cargo.toml`, `Cargo.lock` and the
manual page, commits it as `dwim v0.2.0`, and tags the commit `v0.2.0`, but
pushes nothing. Pushing the tag runs `.github/workflows/release.yml`, which
checks that the tag matches `Cargo.toml`, builds `dwim` for Apple silicon
macOS and x86-64 and ARM64 Linux, and publishes them as a GitHub release.
`https://dwim.sh/install.sh` installs the latest release from there.
