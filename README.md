<div align="center">
  <img src=".github/assets/hero.png" width="200" alt="dwim logo: a small robot sitting cross-legged, eyes closed, at peace">

# dwim

*Do what I mean.*

**A coding agent that you control.**

The agent, the model, and the engine that runs it, in one Rust program.
Local by default: no API key, no cloud, and nothing leaves your machine
unless you send a session to a bigger model.

[![Model: Inkling-2-Mini](https://img.shields.io/badge/Model-Inkling--2--Mini-3ec98a)](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF)
[![Model: Inkling-2-Small](https://img.shields.io/badge/Model-Inkling--2--Small-3ec98a)](https://openrouter.ai/thinkingmachines/inkling-2-small)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE.md)

</div>

> [!NOTE]
> dwim runs shell commands on your machine as part of its work, so please
> run it in a secure environment such as a sandbox, and keep the code you
> ask it to work on committed or backed up.

## Installation

To install the latest release:

```console
curl -fsSL https://dwim.sh/install.sh | sh
```

To install the latest development version:

```console
cargo install --git https://github.com/dwim-sh/dwim
```

Or, from a checkout, with the manual page:

```console
make install                     # in ~/.local
make install PREFIX=/usr/local   # system-wide, which may need sudo
```

Either way, the model (17.6 GB) is downloaded on first use into your
Hugging Face cache, where `hf download` and other tools would put it, and
used from there if one of them already did. `dwim --download` fetches it
ahead of time.

## Features

- **Local by default.** Runs commands, reads their output, and keeps
  going, all on your machine.
- **A bigger model when you need it.** `/provider` signs in to OpenRouter
  in your browser, and `/model` moves the session to Inkling-2-Small there,
  and back. The conversation goes with it. With a key in
  `AI_GATEWAY_API_KEY`, Vercel's AI Gateway runs the models too.
- **Inkling-2-Mini.** 26B parameters, 3.4B active, at four bits, 17.6 GB,
  on a machine with 32 GiB of memory or more.
- **Fast local inference.** Metal on macOS, CUDA or Vulkan on Linux.
- **Built-in local inference server.** The model is loaded once, in a
  `dwim` server that every `dwim` you run connects to. The first one starts
  it in the background, and it exits when idle. Sessions share the weights,
  and each has its own context on the GPU, so you can run several at once.
- **Bring your own sandbox.** `dwim` expects to run inside a sandbox, such as
  a container or a virtual machine, and does not confine the model itself.
  Commands run without confirmation and without a time limit, until they
  finish.

## Getting started

Run `dwim` in a project directory. It opens a shell: type what you want, and
the model reads files, runs commands, and answers.

<div align="center">
  <img src="https://dwim.sh/demo.gif" width="720" alt="A dwim session: the user types Hello!, the model thinks, and replies that it is in the dwim project and asks what to work on">
</div>

Run `dwim` in another terminal and it connects to the same server, ready in
a moment instead of loading the model again. Two sessions fit at once by
default, each with its own context; `--sessions` sets how many, and
`/server` in the shell shows, stops, or restarts the server.

`/model` picks the model and where it runs; `--model large` starts a
session on the large one.

To ask one thing and exit, give the prompt on the command line:

```console
dwim "Explain this codebase to me in one sentence."
```

The manual page has the options, the shell's commands and keys, and the files
it reads:

```console
man dwim
```

## Performance

How fast the local model, Inkling-2-Mini at four bits, reads a prompt
(prefill) and writes its reply (decode), measured on the prompt "Summarize
this repository in one paragraph":

| Machine               |    Prefill |   Decode |
|-----------------------|-----------:|---------:|
| NVIDIA DGX Spark      | 2090 tok/s | 38 tok/s |
| MacBook Pro, Apple M5 |  240 tok/s | 20 tok/s |

`dwim --bench` measures both on your machine, at several depths of
context.

## Using dwim as a sub-agent

dwim also makes a good sub-agent: your main agent plans the change, and
dwim writes the code. The [dwim skills](https://github.com/dwim-sh/skills)
teach an agent such as Claude Code or Codex to delegate to it:

```console
npx skills add dwim-sh/skills
```

Then, in the agent, ask for the change "with dwim":

1. `/dwim plan` reads the code, decides the design, and writes one
   self-contained task file per commit under `.dwim/plans/<name>/`, with
   a `PLAN.md` that holds the goal, the design, the checks, and the steps
   as a checklist. It runs nothing: you read the plan and edit what you
   want.
2. `/dwim do` runs the plan step by step: it hands each task file to
   dwim, watches the run, reviews the change as it would a pull request,
   fixes and verifies it, commits it with `Generated-by: dwim`, ticks the
   step off, and goes on to the next until a step fails review or none
   remain. `/dwim do next` or `/dwim do 3` runs just one step,
   and `/dwim status` shows where the plan stands.

The main agent spends its tokens on the decisions and the review, and
dwim, on your own GPU, on the code.

## License

[MIT](LICENSE.md).
