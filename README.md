<div align="center">
  <img src=".github/assets/hero.png" width="200" alt="dwim logo: a small robot sitting cross-legged, eyes closed, at peace">

# dwim

**A coding agent that runs on your own GPU.**

The agent, the model, and the engine that runs it, in one Rust program:
no API key, no server, and nothing leaves your machine.

[![Model: Bonsai 2 27B](https://img.shields.io/badge/Model-Bonsai_2_27B-3ec98a)](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE.md)

</div>

> [!WARNING]
> **dwim is experimental software.** Expect bugs, rough edges, and changes that
> break things from one commit to the next. The model runs shell commands on
> your machine without asking first, so use it on code you have committed or
> backed up.

## Installation

To install the latest release:

```console
curl -fsSL https://dwim.sh/install.sh | sh
```

To install the latest development version:

```console
cargo install --git https://github.com/dwim-sh/dwim
```

Either way, the model (about 6 GB) is downloaded on first use.

## Features

- **Fully local coding agent.** Runs commands, reads their output, and keeps
  going.
- **Bonsai 2 27B model.** Qwen3.8-27B compressed to 6 GB with ternary
  weights.
- **Fast local inference.** Metal on macOS, Vulkan on Linux.
- **Bring your own sandbox.** `dwim` expects to run inside a sandbox, such as
  a container or a virtual machine, and does not confine the model itself.
  Commands run without confirmation and without a time limit, until they
  finish.

## Getting started

Run `dwim` in a project directory. It opens a shell: type what you want, and
the model reads files, runs commands, and answers.

<div align="center">
  <img src=".github/assets/session.png" width="760" alt="A dwim session: after typing Explain this codebase to me in one sentence, the model reads README.md and replies with one sentence">
</div>

To ask one thing and exit, give the prompt on the command line:

```console
dwim "Explain this codebase to me in one sentence."
```

The manual page has the options, the shell's commands and keys, and the files
it reads:

```console
man dwim
```

## License

[MIT](LICENSE.md).
