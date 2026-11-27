# SWE-bench results

The report of each run over a whole dataset, as `evals/swebench/run.py`
wrote it. Each report says what was run, how, and what became of every
instance; this page says which is which.

| Report | Agent | Model | Served by | Reasoning effort | Resolved |
|---|---|---|---|---|---|
| [dwim-swebench-mini.md](dwim-swebench-mini.md) | dwim, commit a8aa882b1a49 | Qwen3.8-27B (`qwen3.8-27b`) | Vercel AI Gateway | low | 32 of 50 |
| [dwim-inkling2-mini-swebench-mini.md](dwim-inkling2-mini-swebench-mini.md) | dwim, commit 8169aae0c3eb | Inkling 2 Mini (`inkling2-mini`) | OpenRouter: PrimeIntellect | medium | 33 of 50 |

Both are of SWE-bench Verified Mini, 50 issues from Django and Sphinx,
under the same method, which each report gives in full:

- One go of the agent at each issue, and one run: the models are not
  seeded, so a count moves by a few from one run to the next, and a
  difference of two or three issues between reports is not one to read
  much into.
- The agent has 600 seconds an issue. Its shell commands run in the
  issue's SWE-bench image with no network, and its patch is scored by the
  official SWE-bench harness (swebench 5.0.2).
- `dwim` thinks at most 2,048 tokens a thought in both of its runs.

The two reports are of different commits as well as different models, so
they do not compare the models alone. The Inkling run is of `main` three
days after the Qwen run, at medium effort where the Qwen run was at low,
and with the changes to the harness made in between: Inkling 2 Mini has a
system prompt of its own, and is told the rules again in the user's
message and where it breaks one. It also puts the issue to the model in
other words: it is told to leave the tests that are already there as they
are and to change its fix until they pass, where the Qwen run was told to
update the tests its change breaks. Each report gives the words it used.

Inkling 2 Mini resolved 30, 31 and 32 of 50 at low effort on three
earlier commits, `53a9b2b59e88`, `efd9019a1088` and `67e964d8cc3e`, and
31 of 50 at medium effort on `ee74aa0edb2a`.
