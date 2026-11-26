import Dwim.Harness.Compaction

/-!
# The dwim specification

Each file here specifies a part of dwim, and mirrors the Rust file it
specifies: `Dwim/Harness/Compaction.lean` specifies
`harness/compaction.rs`. A spec opens with its intents, what the part is
for, in plain sentences; defines the part as Lean functions, with
examples checked when the spec is built; and proves theorems that make
each intent precise. A spec is written from what the part should do, not
from what the code does: where the two differ, the code is wrong.

To check every definition, example and proof: `make check`.
-/
