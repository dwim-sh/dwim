# dwim-swebench-mini

| | |
|---|---|
| Dataset | verified-mini, 50 instances |
| Agent | dwim 0.3.0, commit a8aa882b1a49 |
| Model | qwen3.8-27b, asked for as `medium`, served as Vercel · alibaba/qwen3.8-27b with a context of 131072 tokens |
| Reasoning effort | low |
| Thinking budget | 2048 tokens a thought |
| Sampling | temperature 0.6, top-p 0.95 |
| Arguments | --provider vercel |
| Time limit | 600 seconds an instance |
| Started | 2026-11-21T10:31:01+00:00 |

**32 of 50 resolved.**

| Outcome | Instances |
|---|---|
| resolved | 32 |
| unresolved | 17 |
| empty patch | 1 |

The agent timed out on 1, which are counted above by what it had changed by then.

The agent failed on 1, which are counted above by what it had changed by then.

## Thinking

| | |
|---|---|
| Thinking tokens an instance, median | 7477 |
| Thinking tokens an instance, most | 182784 |
| Thinking tokens an instance as the host counted them, median | 6270 |
| Thinking tokens, all instances | 1043454 |
| Replies with a thought | 2542 |
| Thoughts ended at the budget | 284, in 29 instances |
| Longest thought | 2114 tokens |

A thought's tokens are estimated from its text at four characters a token,
which is how `dwim` counts a thought against the budget as it streams from
a host: the host does not say what a reply that was ended early took. A
thought that runs to the budget is ended, and the reply asked for again
with no thought, up to three times in a reply. The host's own count is of the
replies that ran to their end only, so it leaves out the thoughts that
were ended.

## Method

Each instance is one go of the agent at one issue, and the run is one go at
each instance: the model's sampling is not seeded, so the count moves by a
few from one run to the next.

- **The task.** The agent is given the issue's text, asked to make the
  smallest change that fixes it in the checkout, to update the tests the
  change breaks, and to run them. It is not given the tests the fix is
  scored by.
- **The checkout.** The repository is copied out of the instance's
  SWE-bench evaluation image, at the commit before the fix, with the
  project installed. The agent's shell commands run in a container of that
  image that has no network; the model is asked from this machine, and the
  agent's tools that read and edit files work on the checkout here.
- **The time.** The agent has the time limit above for an instance. One
  that runs out of it is stopped, and what it had changed by then is its
  patch.
- **The patch.** Whatever differs from the checkout as the image had it,
  files added as well as changed, committed or not.
- **The score.** The official SWE-bench harness (swebench 5.0.2) applies
  the patch in a fresh container of the same image and runs the tests of
  the instance: resolved is every test the fix was to make pass passing,
  and none that passed before failing. An instance with no patch is not
  resolved.
- **The settings.** The agent runs with its defaults and the arguments
  above. For `dwim`, the table has what each session's transcript says it
  was started with: where the model was served, the reasoning effort it
  was asked for, the thinking budget, and the sampling.

## Instances

| Instance | Agent | Seconds | Tool calls | Thinking tokens | Thoughts ended | Outcome |
|---|---|---|---|---|---|---|
| django__django-11790 | finished | 99 | 46 | 11236 | 2 | resolved |
| django__django-11815 | finished | 51 | 21 | 5106 | 1 | resolved |
| django__django-11848 | finished | 55 | 34 | 11348 | 2 | unresolved |
| django__django-11880 | finished | 41 | 37 | 4816 | 0 | resolved |
| django__django-11885 | timed out | 600 | 156 | 182784 | 71 | unresolved |
| django__django-11951 | finished | 24 | 22 | 3537 | 0 | resolved |
| django__django-11964 | finished | 162 | 78 | 29589 | 7 | unresolved |
| django__django-11999 | finished | 162 | 56 | 23094 | 6 | resolved |
| django__django-12039 | finished | 43 | 34 | 5876 | 0 | resolved |
| django__django-12050 | finished | 54 | 34 | 6335 | 0 | resolved |
| django__django-12143 | finished | 21 | 17 | 666 | 0 | resolved |
| django__django-12155 | finished | 22 | 18 | 2025 | 0 | resolved |
| django__django-12193 | finished | 89 | 58 | 8620 | 0 | resolved |
| django__django-12209 | finished | 451 | 136 | 137866 | 50 | resolved |
| django__django-12262 | finished | 30 | 23 | 3001 | 1 | resolved |
| django__django-12273 | finished | 112 | 53 | 21822 | 7 | unresolved |
| django__django-12276 | finished | 29 | 28 | 2410 | 0 | resolved |
| django__django-12304 | finished | 75 | 49 | 9299 | 2 | resolved |
| django__django-12308 | finished | 54 | 33 | 4038 | 0 | unresolved |
| django__django-12325 | finished | 73 | 40 | 6642 | 1 | unresolved |
| django__django-12406 | finished | 135 | 52 | 13067 | 2 | unresolved |
| django__django-12708 | finished | 46 | 40 | 3663 | 0 | resolved |
| django__django-12713 | finished | 29 | 31 | 1922 | 0 | resolved |
| django__django-12774 | finished | 36 | 29 | 6580 | 0 | resolved |
| django__django-9296 | finished | 22 | 21 | 1291 | 0 | resolved |
| sphinx-doc__sphinx-10323 | finished | 96 | 56 | 17230 | 4 | resolved |
| sphinx-doc__sphinx-10435 | finished | 38 | 22 | 4412 | 1 | resolved |
| sphinx-doc__sphinx-10466 | finished | 201 | 80 | 28642 | 4 | resolved |
| sphinx-doc__sphinx-10673 | finished | 230 | 125 | 33768 | 6 | unresolved |
| sphinx-doc__sphinx-11510 | finished | 473 | 89 | 51513 | 14 | unresolved |
| sphinx-doc__sphinx-7590 | finished | 165 | 65 | 26690 | 10 | unresolved |
| sphinx-doc__sphinx-7748 | finished | 144 | 61 | 19637 | 5 | unresolved |
| sphinx-doc__sphinx-7757 | finished | 51 | 28 | 6250 | 0 | resolved |
| sphinx-doc__sphinx-7985 | finished | 202 | 90 | 54055 | 18 | unresolved |
| sphinx-doc__sphinx-8035 | finished | 116 | 45 | 13942 | 2 | resolved |
| sphinx-doc__sphinx-8056 | finished | 80 | 52 | 10553 | 2 | resolved |
| sphinx-doc__sphinx-8265 | finished | 29 | 21 | 3889 | 1 | unresolved |
| sphinx-doc__sphinx-8269 | finished | 46 | 34 | 10127 | 2 | unresolved |
| sphinx-doc__sphinx-8475 | finished | 48 | 32 | 5397 | 0 | resolved |
| sphinx-doc__sphinx-8548 | finished | 74 | 43 | 9680 | 2 | unresolved |
| sphinx-doc__sphinx-8551 | finished | 198 | 88 | 30082 | 5 | resolved |
| sphinx-doc__sphinx-8638 | finished | 74 | 55 | 6075 | 1 | unresolved |
| sphinx-doc__sphinx-8721 | finished | 40 | 28 | 2337 | 0 | resolved |
| sphinx-doc__sphinx-9229 | failed | 560 | 165 | 156441 | 50 | empty patch |
| sphinx-doc__sphinx-9230 | finished | 39 | 35 | 2440 | 0 | resolved |
| sphinx-doc__sphinx-9281 | finished | 37 | 29 | 2325 | 0 | resolved |
| sphinx-doc__sphinx-9320 | finished | 45 | 39 | 7813 | 0 | resolved |
| sphinx-doc__sphinx-9367 | finished | 27 | 16 | 7141 | 0 | resolved |
| sphinx-doc__sphinx-9461 | finished | 180 | 91 | 25367 | 5 | unresolved |
| sphinx-doc__sphinx-9698 | finished | 24 | 26 | 1015 | 0 | resolved |
