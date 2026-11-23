import Dwim.Model.Format
import Dwim.Harness
import Dwim.Harness.Compaction
import Dwim.Harness.Turn

/-!
# The dwim specification

dwim is a coding agent: a language model, the harness that runs it, and
the tools it calls, in one program. This document specifies, in Lean,
parts of dwim's semantics and proves them formally.

To check the proofs and examples, and generate this document:

```
make spec
```

To run the differential random tests:

```
make check
```

## Formal models and proofs

Each section specifies one part of the code, and mirrors the Rust file it
specifies: it defines the terms, states the requirements the part must
meet, defines the part as Lean functions with examples checked when the
document is built, and proves theorems that the requirements hold. A
section is written from what the part should do, not from what the code
does: where the two differ, the code is wrong.

- [Chat format](Dwim/Model/Format/): the turns a conversation is made of,
  and the grammar they follow, in which every tool call is answered.
- [Harness](Dwim/Harness/): the transcript the harness keeps.
- [Compaction](Dwim/Harness/Compaction/): what is kept of a conversation
  when it no longer fits in the context window
  ([animation](compaction.html)).
- [Turns](Dwim/Harness/Turn/): how a turn ends, and that however it ends,
  every call is answered.

## Differential random testing

A proof covers the Lean definitions; the Rust is checked against them
by differential random testing, as Cedar's is (Hicks, 2023). For
each part, `difftest` runs the code on thousands of generated cases and
asks the spec, through an oracle built from these definitions, what it
makes of each: `difftest compaction` compares what a compaction keeps,
and `difftest harness` checks that every turn the real harness plays
leaves every call answered. A disagreement is a bug in the code.

## References

- Mike Hicks. 2023. [How we built Cedar with automated
  reasoning and differential testing](https://www.amazon.science/blog/how-we-built-cedar-with-automated-reasoning-and-differential-testing).
  _Amazon Science_.
-/
