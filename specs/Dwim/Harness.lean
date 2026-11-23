import Dwim.Model.Format

/-!
# Harness

A transcript of the turns since the system prompt, each with the tokens
it takes.

The one assumption about tokens is that a conversation takes the sum of
what its turns take, which `Entry.tokens` records for each.

See `harness/mod.rs`.
-/

namespace Dwim.Harness

open Dwim.Model

/-- A turn as the transcript keeps it: the tokens it takes when fed, and
whether it is the harness's own prompt, such as the head of a compaction,
rather than a message of the user's. -/
structure Entry where
  turn : Turn
  tokens : Nat
  prompt : Bool := false
  deriving DecidableEq, Repr

namespace Entry

/-- Whether the entry is a reply. -/
def isReply (e : Entry) : Bool :=
  match e.turn with
  | .reply _ => true
  | _ => false

/-- Whether the entry is a message the user wrote. -/
def isOwn (e : Entry) : Bool := e.turn == .user && !e.prompt

end Entry

/-- The tokens `es` take. -/
def size (es : List Entry) : Nat := (es.map Entry.tokens).sum

end Dwim.Harness
