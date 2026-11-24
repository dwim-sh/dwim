# dwim-inkling2-mini-swebench-mini

| | |
|---|---|
| Dataset | verified-mini, 50 instances |
| Agent | dwim 0.4.1, commit 8169aae0c3eb |
| Model | inkling2-mini, served as OpenRouter · thinkingmachines/inkling-2-mini with a context of 131072 tokens |
| Reasoning effort | medium |
| Thinking budget | 2048 tokens a thought |
| Sampling | temperature 1.0, top-p 1.0 |
| Providers | PrimeIntellect (2674 replies), Io Net (1 replies) |
| Arguments | --provider openrouter |
| Time limit | 600 seconds an instance |
| Started | 2026-11-24T20:15:25+00:00 |

**33 of 50 resolved.**

| Outcome | Instances |
|---|---|
| resolved | 33 |
| unresolved | 15 |
| empty patch | 2 |

The agent timed out on 2, which are counted above by what it had changed by then.

## Thinking

| | |
|---|---|
| Thinking tokens an instance, median | 4727 |
| Thinking tokens an instance, most | 49577 |
| Thinking tokens an instance as the host counted them, median | 3952 |
| Thinking tokens, all instances | 457692 |
| Replies with a thought | 1662 |
| Thoughts ended at the budget | 39, in 20 instances |
| Longest thought | 2105 tokens |

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
  smallest change that fixes it in the checkout, to run the project's
  tests to check it, to leave the tests that are already there as they
  are, and to change its fix until they pass, but for one that checks
  the wrong behavior the issue describes. It is not given the tests the
  fix is scored by.
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
| django__django-11790 | finished | 92 | 47 | 7414 | 1 | unresolved |
| django__django-11815 | finished | 72 | 37 | 5405 | 1 | resolved |
| django__django-11848 | finished | 21 | 12 | 2986 | 1 | resolved |
| django__django-11880 | finished | 41 | 31 | 1603 | 0 | resolved |
| django__django-11885 | finished | 393 | 156 | 35415 | 3 | unresolved |
| django__django-11951 | finished | 21 | 14 | 470 | 0 | resolved |
| django__django-11964 | finished | 143 | 61 | 11633 | 1 | unresolved |
| django__django-11999 | finished | 47 | 28 | 3746 | 0 | resolved |
| django__django-12039 | finished | 42 | 28 | 1599 | 0 | resolved |
| django__django-12050 | finished | 34 | 23 | 838 | 0 | resolved |
| django__django-12143 | finished | 51 | 29 | 1814 | 0 | resolved |
| django__django-12155 | finished | 36 | 26 | 1393 | 0 | resolved |
| django__django-12193 | finished | 52 | 37 | 2470 | 0 | resolved |
| django__django-12209 | finished | 246 | 105 | 25450 | 1 | resolved |
| django__django-12262 | finished | 87 | 34 | 7648 | 2 | resolved |
| django__django-12273 | finished | 168 | 74 | 23856 | 3 | unresolved |
| django__django-12276 | finished | 60 | 44 | 2324 | 0 | resolved |
| django__django-12304 | finished | 28 | 16 | 1301 | 0 | resolved |
| django__django-12308 | finished | 69 | 37 | 6437 | 1 | unresolved |
| django__django-12325 | finished | 76 | 32 | 6381 | 1 | unresolved |
| django__django-12406 | finished | 78 | 51 | 3210 | 0 | unresolved |
| django__django-12708 | finished | 87 | 55 | 6740 | 0 | resolved |
| django__django-12713 | finished | 49 | 25 | 1405 | 0 | resolved |
| django__django-12774 | finished | 22 | 17 | 636 | 0 | resolved |
| django__django-9296 | finished | 34 | 22 | 1030 | 0 | resolved |
| sphinx-doc__sphinx-10323 | finished | 43 | 20 | 1223 | 0 | resolved |
| sphinx-doc__sphinx-10435 | finished | 85 | 41 | 8612 | 2 | resolved |
| sphinx-doc__sphinx-10466 | finished | 66 | 39 | 3539 | 0 | resolved |
| sphinx-doc__sphinx-10673 | finished | 191 | 75 | 13010 | 2 | resolved |
| sphinx-doc__sphinx-11510 | finished | 386 | 105 | 30836 | 7 | unresolved |
| sphinx-doc__sphinx-7590 | finished | 63 | 34 | 7845 | 0 | unresolved |
| sphinx-doc__sphinx-7748 | finished | 187 | 105 | 15352 | 1 | unresolved |
| sphinx-doc__sphinx-7757 | finished | 64 | 40 | 3075 | 0 | resolved |
| sphinx-doc__sphinx-7985 | finished | 96 | 53 | 12191 | 0 | unresolved |
| sphinx-doc__sphinx-8035 | finished | 143 | 68 | 10565 | 0 | resolved |
| sphinx-doc__sphinx-8056 | finished | 190 | 100 | 19051 | 1 | unresolved |
| sphinx-doc__sphinx-8265 | finished | 153 | 61 | 5808 | 0 | unresolved |
| sphinx-doc__sphinx-8269 | finished | 70 | 34 | 2370 | 0 | resolved |
| sphinx-doc__sphinx-8475 | finished | 52 | 32 | 1580 | 0 | resolved |
| sphinx-doc__sphinx-8548 | finished | 234 | 69 | 14842 | 2 | unresolved |
| sphinx-doc__sphinx-8551 | finished | 239 | 61 | 12778 | 3 | resolved |
| sphinx-doc__sphinx-8638 | timed out | 600 | 167 | 39305 | 1 | empty patch |
| sphinx-doc__sphinx-8721 | finished | 149 | 54 | 5445 | 0 | resolved |
| sphinx-doc__sphinx-9229 | timed out | 600 | 168 | 49577 | 2 | empty patch |
| sphinx-doc__sphinx-9230 | finished | 129 | 37 | 2802 | 0 | resolved |
| sphinx-doc__sphinx-9281 | finished | 184 | 59 | 4050 | 0 | resolved |
| sphinx-doc__sphinx-9320 | finished | 17 | 15 | 519 | 0 | resolved |
| sphinx-doc__sphinx-9367 | finished | 19 | 14 | 518 | 0 | resolved |
| sphinx-doc__sphinx-9461 | finished | 259 | 105 | 28260 | 3 | unresolved |
| sphinx-doc__sphinx-9698 | finished | 28 | 18 | 1335 | 0 | resolved |
