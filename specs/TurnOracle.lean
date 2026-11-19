import Dwim.Harness.Turn

/-!
# Turn oracle

Answers, for each conversation it reads, whether it is as a conversation
between turns must be (`turn_ends_answered`): well-formed, and not ending
with a call. `difftest harness` asks it of what the harness leaves after
each turn.

Each line read is a conversation, its turns by kind:

    user call res answer

and each line written is `answered` or `unanswered`.
-/

open Dwim.Model Dwim.Harness.Turn

/-- The turn of the kind `word`. -/
def turn? : String → Option Turn
  | "user" => some .user
  | "call" => some (.reply true)
  | "answer" => some (.reply false)
  | "res" => some .results
  | _ => none

def main : IO Unit := do
  let stdin ← IO.getStdin
  let stdout ← IO.getStdout
  repeat
    let line ← stdin.getLine
    if line.isEmpty then break
    let words := (line.trimAscii.toString.splitOn " ").filter (· ≠ "")
    match words.mapM turn? with
    | some turns => stdout.putStrLn (if answered turns then "answered" else "unanswered")
    | none => throw <| IO.userError s!"not a conversation: {line.trimAscii}"
