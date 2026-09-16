# Contributing

## Building

`cargo build` needs nothing beyond Rust. The Vulkan kernels are compiled from
WGSL to SPIR-V by `naga` in `gpu/build.rs`, and Metal compiles its own kernels
from source when the device opens.

```
cargo build              # debug, with the model and the devices optimized
cargo build --release    # for anything that runs the model for long
```

Running the model on a GPU needs Metal on macOS, and elsewhere a Vulkan 1.1
driver with `VK_KHR_push_descriptor`. `--device cpu` runs anywhere, slowly.

The weights are downloaded on first use into `~/Library/Caches/hack/models`
on macOS, and `~/.cache/hack/models` elsewhere. Qwen3-0.6B, the default, is
about 1.5 GB.

## Running

```
hack                     # the shell
hack "what changed in the last commit?"
hack --model qwen3-0.6b --device cpu "what files are here?"
```

With a prompt, hack answers it and exits. The reply goes to standard output
and nothing else does, so `hack "…" 2>/dev/null` gives the reply alone, while
progress, the model's thinking, the commands it runs, and their output go to
standard error.

## Verifying a change

```
cargo test --workspace   # the GPU kernels are checked against the CPU reference
cargo clippy --workspace --all-targets
```

`cargo test -p hack-gpu` skips a backend whose GPU is not available, so it
passes on a machine with no GPU without having checked anything.

Run the agent before calling a change done. A model of this size is easy to
break in ways tests don't catch, such as replies that end with nothing to
show, so ask it something in the shell, and check a one-off prompt too.

The coding style is in `CLAUDE.md`. Commit messages say what changed and why,
in the imperative, with the reasoning in the body.

## SWE-bench Lite

[SWE-bench](https://www.swebench.com) gives an agent a real GitHub issue and
a repository at the commit before the fix, and checks the agent's patch by
running the project's tests. Lite is 300 instances from 12 Python projects.

```
scripts/swebench.py --instance pallets__flask-4045     # one instance
scripts/swebench.py --limit 10                         # the first ten
scripts/swebench.py                                    # all 300
```

The script builds hack, checks each repository out at its base commit, gives
hack the issue, takes what it changed as the patch, and scores the patches
with the official harness, which runs each project's tests in Docker. It
tells you what to install if Docker or the harness is missing. `--no-score`
writes `predictions.jsonl` without scoring, `--score-only` scores what an
earlier run wrote, and `--help` lists the rest.

**The agent runs shell commands from an untrusted issue, in a checkout of an
untrusted repository, as you.** Run it in a container or a virtual machine,
not on a machine you care about.
