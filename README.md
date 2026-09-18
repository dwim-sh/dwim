```
HACK(1)                          User Commands                          HACK(1)

NAME
       hack - a coding agent

SYNOPSIS
       hack [OPTIONS] [PROMPT]

DESCRIPTION
       hack is a coding agent that runs in your terminal. It reads and
       edits files, runs commands, and works through a task with you in
       an interactive shell.

       With no arguments, hack starts an interactive session in the
       current directory. If PROMPT is given, hack answers it without
       the shell and exits: the reply goes to standard output and
       nothing else does, while progress, the model's thinking, the
       commands it runs, and their output go to standard error.

OPTIONS
       --model <model>
              Model to run (default: bonsai-2-27b).

       --device <device>
              Device to run the model on: gpu (default), or cpu.

       --context <tokens>
              Maximum conversation length in tokens (default: 32768).
              Cannot exceed the model's supported context length.
              Larger values reserve more memory on the selected device
              at startup.

       --help Display usage information.

INTERACTIVE SHELL
       The shell reads one message at a time. Anything you type is sent
       to the agent. Lines starting with a slash are commands:

       /model List the models hack knows, where their weights are kept
              and how much of each has been fetched, marking the one
              running.

       Ctrl-C or Esc interrupts the agent while it works. Ctrl-C or
       Ctrl-D on an empty line exits hack, and Ctrl-C on a line with
       text clears it.

FILES
       AGENTS.md
              If present in the working directory, its contents are
              given to the agent as project instructions.

EXIT STATUS
       0      Success.

       1      An error occurred.

SEE ALSO
       claude(1), codex(1), opencode(1)

hack                              2026-09-18                            HACK(1)
```
