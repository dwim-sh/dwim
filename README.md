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
       current directory. If PROMPT is given, it is sent as the first
       message of the session.

OPTIONS
       --model <model>
              Model to run (default: qwen3-0.6b).

       --device <device>
              Device to run the model on: gpu (default), or cpu.

       --help Display usage information.

INTERACTIVE SHELL
       The shell reads one message at a time. Anything you type is sent
       to the agent. Lines starting with a slash are commands:

       /help  List the available commands.

       /clear Start a new conversation, discarding the current context.

       /quit  Exit hack. Ctrl-D and Ctrl-C twice do the same.

FILES
       AGENTS.md
              If present in the working directory, its contents are
              given to the agent as project instructions.

EXIT STATUS
       0      Success.

       1      An error occurred.

SEE ALSO
       claude(1), codex(1), opencode(1)

hack                              2026-09-15                            HACK(1)
```
