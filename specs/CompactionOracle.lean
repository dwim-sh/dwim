import Dwim.Harness.Compaction

/-!
# Compaction oracle

Answers, for each transcript it reads, what the spec selects of it: where
the tail starts, and which user messages the head carries.
`difftest compaction` asks the same of the code and compares the two.

Each line read is the capacity of the window, a transcript in the
notation of the spec, and after a `|` the messages the user has sent:

    720 msg 10 call 50 res 20 answer 30 msg 10 call 40 res 20 answer 30 | msg 10 msg 10

and each line written is what `select` keeps: the index where the tail
starts, and the indices in the messages sent of those the head carries:

    tail=5 messages=0,1
-/

open Dwim.Harness Dwim.Harness.Compaction

/-- The turn written as `kind tokens`. -/
def turn? (kind : String) (tokens : Nat) : Option Entry :=
  match kind with
  | "msg" => some (msg tokens)
  | "prompt" => some (prompt tokens)
  | "call" => some (call tokens)
  | "res" => some (res tokens)
  | "answer" => some (answer tokens)
  | _ => none

/-- The transcript written as `kind tokens` pairs. -/
def transcript? : List String → Option (List Entry)
  | [] => some []
  | kind :: tokens :: rest => do
    let entry ← turn? kind (← tokens.toNat?)
    return entry :: (← transcript? rest)
  | [_] => none

/-- The words of `s`, split at spaces. -/
def words (s : String) : List String := (s.splitOn " ").filter (· ≠ "")

/-- What the spec keeps of the transcript on `line`. The messages carried
are a suffix of the candidates (`messages_suffix`), which are the first
of the messages sent, so their indices end where the candidates do. -/
def select? (line : String) : Option String := do
  let [conversation, sent] := line.splitOn "|" | none
  let capacity :: rest := words conversation | none
  let t ← transcript? rest
  let said ← transcript? (words sent)
  let (start, carried) := select (← capacity.toNat?) said t
  let last := (candidates said (t.drop start)).length
  let indices := List.range' (last - carried.length) carried.length
  return s!"tail={start} messages={",".intercalate (indices.map toString)}"

def main : IO Unit := do
  let stdin ← IO.getStdin
  let stdout ← IO.getStdout
  repeat
    let line ← stdin.getLine
    if line.isEmpty then break
    let line := line.trimAscii.toString
    match select? line with
    | some selected => stdout.putStrLn selected
    | none => throw <| IO.userError s!"not a transcript: {line}"
