# dwim-inkling2-mini-swebench-mini

| | |
|---|---|
| Dataset | verified-mini, 50 instances |
| Agent | dwim 0.3.0, commit 53a9b2b59e88 |
| Model | inkling2-mini, served as OpenRouter · thinkingmachines/inkling-2-mini with a context of 131072 tokens |
| Reasoning effort | low |
| Thinking budget | 2048 tokens a thought |
| Sampling | temperature 1.0, top-p 1.0 |
| Arguments | --provider openrouter |
| Time limit | 600 seconds an instance |
| Started | 2026-11-22T07:58:41+00:00 |

**30 of 50 resolved.**

| Outcome | Instances |
|---|---|
| resolved | 30 |
| unresolved | 19 |
| empty patch | 1 |

The agent timed out on 3, which are counted above by what it had changed by then.

The agent failed on 3, which are counted above by what it had changed by then.

## Thinking

| | |
|---|---|
| Thinking tokens an instance, median | 1945 |
| Thinking tokens an instance, most | 48961 |
| Thinking tokens an instance as the host counted them, median | 2099 |
| Thinking tokens, all instances | 221995 |
| Replies with a thought | 1157 |
| Thoughts ended at the budget | 7, in 4 instances |
| Longest thought | 2102 tokens |

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
| django__django-11790 | finished | 114 | 23 | 1639 | 0 | unresolved |
| django__django-11815 | finished | 140 | 22 | 773 | 0 | resolved |
| django__django-11848 | failed | 72 | 11 | 929 | 0 | resolved |
| django__django-11880 | failed | 13 | 0 | 0 | 0 | empty patch |
| django__django-11885 | failed | 115 | 12 | 3151 | 0 | unresolved |
| django__django-11951 | finished | 65 | 16 | 866 | 0 | resolved |
| django__django-11964 | finished | 395 | 56 | 6710 | 0 | unresolved |
| django__django-11999 | finished | 89 | 22 | 1404 | 0 | resolved |
| django__django-12039 | finished | 297 | 35 | 1570 | 0 | resolved |
| django__django-12050 | finished | 53 | 25 | 766 | 0 | resolved |
| django__django-12143 | finished | 94 | 25 | 543 | 0 | resolved |
| django__django-12155 | finished | 53 | 9 | 450 | 0 | resolved |
| django__django-12193 | finished | 83 | 33 | 1939 | 0 | unresolved |
| django__django-12209 | finished | 176 | 48 | 5193 | 0 | resolved |
| django__django-12262 | finished | 62 | 21 | 1117 | 0 | resolved |
| django__django-12273 | finished | 358 | 71 | 17991 | 2 | unresolved |
| django__django-12276 | finished | 207 | 30 | 896 | 0 | resolved |
| django__django-12304 | finished | 131 | 31 | 2565 | 0 | resolved |
| django__django-12308 | finished | 39 | 21 | 1217 | 0 | unresolved |
| django__django-12325 | finished | 116 | 33 | 2217 | 0 | unresolved |
| django__django-12406 | finished | 143 | 56 | 2778 | 0 | unresolved |
| django__django-12708 | finished | 239 | 77 | 16874 | 3 | resolved |
| django__django-12713 | finished | 64 | 23 | 757 | 0 | resolved |
| django__django-12774 | finished | 76 | 18 | 899 | 0 | resolved |
| django__django-9296 | finished | 16 | 8 | 201 | 0 | resolved |
| sphinx-doc__sphinx-10323 | finished | 22 | 11 | 439 | 0 | resolved |
| sphinx-doc__sphinx-10435 | finished | 80 | 36 | 2267 | 0 | unresolved |
| sphinx-doc__sphinx-10466 | finished | 37 | 10 | 702 | 0 | resolved |
| sphinx-doc__sphinx-10673 | finished | 176 | 62 | 5935 | 0 | resolved |
| sphinx-doc__sphinx-11510 | finished | 148 | 29 | 4001 | 0 | unresolved |
| sphinx-doc__sphinx-7590 | finished | 72 | 27 | 2225 | 0 | unresolved |
| sphinx-doc__sphinx-7748 | finished | 152 | 51 | 5339 | 1 | unresolved |
| sphinx-doc__sphinx-7757 | finished | 45 | 25 | 2556 | 0 | resolved |
| sphinx-doc__sphinx-7985 | timed out | 600 | 55 | 8434 | 0 | unresolved |
| sphinx-doc__sphinx-8035 | finished | 100 | 37 | 1952 | 0 | resolved |
| sphinx-doc__sphinx-8056 | finished | 194 | 48 | 7170 | 0 | unresolved |
| sphinx-doc__sphinx-8265 | finished | 133 | 42 | 3066 | 0 | resolved |
| sphinx-doc__sphinx-8269 | finished | 53 | 17 | 2147 | 0 | unresolved |
| sphinx-doc__sphinx-8475 | finished | 40 | 12 | 1125 | 0 | resolved |
| sphinx-doc__sphinx-8548 | finished | 180 | 72 | 5830 | 0 | unresolved |
| sphinx-doc__sphinx-8551 | finished | 71 | 30 | 1238 | 0 | resolved |
| sphinx-doc__sphinx-8638 | timed out | 600 | 144 | 24419 | 0 | unresolved |
| sphinx-doc__sphinx-8721 | finished | 69 | 35 | 3753 | 0 | resolved |
| sphinx-doc__sphinx-9229 | timed out | 600 | 168 | 48961 | 1 | unresolved |
| sphinx-doc__sphinx-9230 | finished | 189 | 85 | 7103 | 0 | resolved |
| sphinx-doc__sphinx-9281 | finished | 44 | 32 | 1014 | 0 | resolved |
| sphinx-doc__sphinx-9320 | finished | 23 | 14 | 1172 | 0 | resolved |
| sphinx-doc__sphinx-9367 | finished | 18 | 11 | 227 | 0 | resolved |
| sphinx-doc__sphinx-9461 | finished | 182 | 89 | 7015 | 0 | unresolved |
| sphinx-doc__sphinx-9698 | finished | 30 | 17 | 460 | 0 | resolved |
