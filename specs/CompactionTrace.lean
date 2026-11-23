import Lean.Data.Json
import Dwim.Harness.Compaction

/-!
# Compaction trace

Plays a session through the spec, a turn at a time, with `add`, and
writes every step as a script for `compaction.html` to animate:

    lake exe compaction-trace > compaction-trace.js

The page only draws what this writes. Which turns a compaction keeps,
where the tail starts and which messages the head carries are the
spec's `select`; the conversation after each step is checked against
the spec's `add`, so the page shows the spec and nothing else.

Turns have no identity in the spec, only a kind and a size, so the
trace follows each one by its place: the turns a compaction keeps are
those from where the tail starts, and the messages the head carries a
suffix of the candidates, which are the first of `said`
(`messages_suffix`).
-/

open Lean Dwim.Harness Dwim.Harness.Compaction

/-- The window, as when dwim runs the model itself. -/
def capacity : Nat := 16384

/-- What the head takes besides the messages it carries: where the agent
is, the note of `C/32` tokens, and the plan. -/
def overhead : Nat := 1100

/-- The head: what it takes besides, and the messages it carries. -/
def headOf (ms : List Entry) : Entry := prompt (overhead + size ms)

/-- The session: each turn, with what it says. -/
def session : List (Entry × String) := [
  (msg 42, "The tests in harness/ fail since the rebase. Find out why."),
  (call 120, "read harness/mod.rs"),
  (res 4200, "harness/mod.rs, 1,300 lines"),
  (call 90, "cargo test"),
  (res 2600, "test output: compacts_when_the_context_fills failed"),
  (call 80, "grep -n said harness/"),
  (res 900, "12 matches"),
  (answer 380, "said takes a message before make_room compacts for it, so the head carries it twice."),
  (msg 24, "Fix it, and add a test that would have caught it."),
  (call 640, "edit harness/mod.rs: push to said in record"),
  (res 60, "edited"),
  (call 90, "cargo test"),
  (res 3100, "test output: 250 passed"),
  (answer 210, "Fixed: said takes a message when the transcript records it."),
  (msg 1500, "CI still fails. Here is the log: [1,500 tokens pasted]"),
  (call 110, "read .github/workflows/ci.yml"),
  (res 1300, ".github/workflows/ci.yml"),
  (call 90, "cargo clippy --all-targets"),
  (res 2400, "clippy: 3 warnings"),
  (call 520, "edit harness/compaction.rs"),
  (res 60, "edited"),
  (call 90, "cargo clippy --all-targets"),
  (res 200, "clippy: no warnings"),
  (answer 260, "CI failed on clippy warnings; fixed."),
  (msg 30, "Now make the note's Findings section longer."),
  (call 130, "read harness/compaction.rs"),
  (res 5200, "harness/compaction.rs, 700 lines"),
  (call 700, "edit harness/compaction.rs: NOTE_HEADS"),
  (res 60, "edited"),
  (call 90, "cargo test"),
  (res 3000, "test output: 250 passed"),
  (answer 240, "Findings now takes half of the note."),
  (msg 18, "Commit it."),
  (call 100, "git commit"),
  (res 120, "[main 2bef2f8] harness: More space for findings"),
  (answer 60, "Committed."),
  (msg 12, "Thanks!"),
  (answer 20, "You're welcome."),
  (msg 26, "One more: why is bench slower on Vulkan than on Metal?"),
  (call 110, "dwim bench"),
  (res 2800, "bench: 41 tokens/s"),
  (call 140, "read gpu/vulkan/mod.rs"),
  (res 4600, "gpu/vulkan/mod.rs, 1,500 lines"),
  (answer 420, "Each matmul waits for the one before: the command buffer is submitted per kernel."),
  (msg 16, "Batch them."),
  (call 760, "edit gpu/vulkan/mod.rs: one command buffer a step"),
  (res 60, "edited"),
  (call 110, "dwim bench"),
  (res 2800, "bench: 58 tokens/s"),
  (answer 180, "58 tokens/s, from 41.")
]

/-- The name of a turn's kind on the page. -/
def kind (e : Entry) : String :=
  match e.turn, e.prompt with
  | .user, true => "head"
  | .user, false => "msg"
  | .reply true, _ => "call"
  | .reply false, _ => "answer"
  | .results, _ => "res"

def turnJson (id : Nat) (e : Entry) (text : String) : Json :=
  Json.mkObj [("id", id), ("kind", kind e), ("tokens", e.tokens), ("text", text)]

/-- What the trace has followed so far: the session as the spec has it,
and the turns of its conversation and of `said` by their ids. -/
structure State where
  session : Session := {}
  conversation : List Nat := []
  said : List Nat := []
  turns : Array Json := #[]
  steps : Array Json := #[]

def nats (ns : List Nat) : Json := Json.arr (ns.map toJson).toArray

/-- Adds turn `id`, `e`, as the spec's `add` does, and records the step. -/
def step (st : State) (id : Nat) (e : Entry) (text : String) : Except String State := do
  let s := st.session
  let t := s.conversation
  let mut ids := st.conversation
  let mut turns := st.turns.push (turnJson id e text)
  let mut compaction := Json.null
  if full capacity t e then
    let (start, carried) := select capacity s.said t
    let cands := candidates s.said (t.drop start)
    let first := cands.length - carried.length
    let carriedIds := (st.said.drop first).take carried.length
    let headId := 1000 + st.steps.size
    let h := headOf carried
    turns := turns.push (turnJson headId h s!"note and plan, {carried.length} of your messages")
    compaction := Json.mkObj [
      ("before", nats ids), ("size", size t), ("start", start),
      ("candidates", nats (st.said.take cands.length)), ("carried", nats carriedIds),
      ("head", headId)]
    ids := headId :: ids.drop start
  let after := add capacity headOf s e
  ids := ids ++ [id]
  if ids.length ≠ after.conversation.length then
    throw s!"step {id}: the trace has {ids.length} turns, the spec {after.conversation.length}"
  let said := if e.isOwn then st.said ++ [id] else st.said
  return { session := after, conversation := ids, said, turns,
           steps := st.steps.push (Json.mkObj [
             ("add", id), ("compaction", compaction), ("conversation", nats ids),
             ("said", nats said), ("size", size after.conversation)]) }

def trace : Except String Json := do
  let mut st : State := {}
  for (e, text) in session, id in List.range session.length do
    st ← step st id e text
  return Json.mkObj [
    ("capacity", capacity), ("highWater", toJson (capacity * 4 / 5)),
    ("tailBudget", toJson (capacity / 8)), ("messageBudget", toJson (capacity / 16)),
    ("turns", Json.arr st.turns), ("steps", Json.arr st.steps)]

def main : IO UInt32 := do
  match trace with
  | .ok json =>
    IO.println s!"// Written by `lake exe compaction-trace` from the spec in Dwim/Harness/Compaction.lean.\nconst TRACE = {json.compress};"
    return 0
  | .error e =>
    IO.eprintln s!"compaction-trace: {e}"
    return 1
