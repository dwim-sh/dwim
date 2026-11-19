/-!
# Chat format

The turns a conversation is made of, and the order they may come in.

See `model/format.rs`.
-/

namespace Dwim.Model

/-- A turn of the conversation: a message, a reply and whether it made
calls, or the results of the calls a reply made. -/
inductive Turn where
  | user
  | reply (calls : Bool)
  | results
  deriving DecidableEq, Repr

/-- Turn `b` may come right after turn `a`: tool results come after a
reply that made calls, and a reply that made calls is followed by their
results. -/
def Follows (a b : Turn) : Prop :=
  b = .results ↔ a = .reply true

/-- Every two turns in a row in `ts` follow each other. -/
def Paired : List Turn → Prop
  | a :: b :: rest => Follows a b ∧ Paired (b :: rest)
  | _ => True

/-- `ts` follows the chat grammar: it does not open with tool results, and
calls and their results come in pairs. A reply whose calls are not yet
answered can come only last: its results are still to be fed. -/
def WellFormed (ts : List Turn) : Prop :=
  ts.head? ≠ some .results ∧ Paired ts

end Dwim.Model
