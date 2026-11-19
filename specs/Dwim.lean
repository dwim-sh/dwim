import Dwim.Model.Format
import Dwim.Harness
import Dwim.Harness.Compaction
import Dwim.Harness.Turn

/-!
# dwim, specified

dwim is a coding agent: a language model, the harness that runs it, and
the tools it calls, in one program. These pages say, in Lean, what parts
of dwim must do, and prove that what they say holds for every input, not
only the ones tried.

Each page specifies one part of the code, and mirrors the Rust file it
specifies: it defines the terms, states the requirements the part must
meet, defines the part as Lean functions with examples checked when the
page is built, and proves theorems that the requirements hold. A page is
written from what the part should do, not from what the code does: where
the two differ, the code is wrong.

- [Chat format](Dwim/Model/Format/): the turns a conversation is made of,
  and the grammar they follow, in which every tool call is answered.
- [Harness](Dwim/Harness/): the transcript the harness keeps.
- [Compaction](Dwim/Harness/Compaction/): what is kept of a conversation
  when it no longer fits in the context window.
- [Turns](Dwim/Harness/Turn/): how a turn ends, and that however it ends,
  every call is answered.

**How the code is checked against them.** A proof covers the Lean
definitions; the Rust is checked against them by differential testing,
as in "How we built Cedar with automated reasoning and differential
testing" (Amazon Science, 2023). For each part, `difftest` runs the code
on thousands of generated cases and asks the spec, through an oracle
built from these definitions, what it makes of each: `difftest
compaction` compares what a compaction keeps, and `difftest harness`
checks that every turn the real harness plays leaves every call
answered. A disagreement is a bug in the code.

`make check` checks the proofs and the examples, makes these pages, and
runs the difftests.
-/
