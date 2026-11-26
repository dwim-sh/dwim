import Dwim.Harness.Compaction

/-!
# Compaction oracle

Answers for the spec of compaction, a line for each line it reads, so
that `difftest compaction` can ask the code the same and compare. A line
is a case of compaction or a window's budgets.

A case is the capacity of the window, where the session's transcript is,
the turns of the conversation, and after a `|` the messages the user has
sent. A turn is its kind and its tokens, and a message of the user's its
name too:

    case 720 /t.jsonl user:m0:10 call:50 results:20 reply:30 prompt:5 call:40 | user:m0:10

The answer is what `compact` makes of it: where the tail starts, the
names of the messages the briefing carries, and where the briefing says
the transcript is:

    tail=3 carried=m0 transcript=/t.jsonl

A window's budgets are asked for with its capacity, and answered with
the high-water mark and the budgets of the note, the tail and the
messages:

    budgets 16384
    high_water=13107 note=512 tail=2048 messages=1024
-/

open Dwim.Harness.Compaction

/-- `tokens` is the tokenizer of the oracle: a text ends in its tokens,
after a colon. -/
def tokens : Tokenizer where
  count text := ((text.splitOn ":").getLast?.bind String.toNat?).getD 0
  frame := 0

/-- `turn?` is the turn written as `kind:tokens`, or `user:name:tokens`.
-/
def turn? (written : String) : Option Turn :=
  match written.splitOn ":" with
  | ["user", name, n] => n.toNat?.map fun _ => .user s!"{name}:{n}"
  | ["prompt", n] => n.toNat?.map fun _ => .prompt n
  | ["reply", n] => n.toNat?.map fun _ => .reply n false
  | ["call", n] => n.toNat?.map fun _ => .reply n true
  | ["results", n] => n.toNat?.map fun _ => .results n
  | _ => none

/-- `words` is the words of `line`, split at spaces. -/
def words (line : String) : List String := (line.splitOn " ").filter (· ≠ "")

/-- `name` is the name of a message of the user's, written `name:tokens`.
-/
def name (text : String) : String := (text.splitOn ":").headD ""

/-- `answer?` is the spec's answer to `line`. -/
def answer? (line : String) : Option String :=
  match words line with
  | ["budgets", C] => do
    let C ← C.toNat?
    return s!"high_water={highWater C} note={noteBudget C} tail={tailBudget C} messages={messageBudget C}"
  | "case" :: C :: path :: rest => do
    let C ← C.toNat?
    let (written, sent) := rest.span (· ≠ "|")
    let conversation ← written.mapM turn?
    let said ← (sent.drop 1).mapM fun t => (turn? t).bind Turn.said?
    let h : Harness :=
      { capacity := C, tokenizer := tokens, plan := "", transcript := path, write := fun _ => "" }
    let .briefing b :: kept := (compact h { said, conversation }).conversation | none
    let start := conversation.length - kept.length
    return s!"tail={start} carried={",".intercalate (b.messages.map name)} transcript={b.transcript}"
  | _ => none

#guard answer? "budgets 16384" = some "high_water=13107 note=512 tail=2048 messages=1024"
#guard answer? "case 720 /t.jsonl user:m0:10 call:50 results:20 reply:30 prompt:5 call:40 | user:m0:10"
  = some "tail=3 carried=m0 transcript=/t.jsonl"
#guard answer? "case 160 /t.jsonl user:m0:10 call:50 results:20 reply:30 call:40 | user:m0:10"
  = some "tail=4 carried=m0 transcript=/t.jsonl"

def main : IO Unit := do
  let stdin ← IO.getStdin
  let stdout ← IO.getStdout
  repeat
    let line ← stdin.getLine
    if line.isEmpty then break
    match answer? line.trimAscii.toString with
    | some answer => stdout.putStrLn answer
    | none => throw <| IO.userError s!"not a case: {line}"
