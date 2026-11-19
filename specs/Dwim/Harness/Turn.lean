import Dwim.Harness.Compaction

/-!
# Turns

A *turn* begins with a message and ends when the harness waits for the
next one. In between, the model replies; a reply that calls tools is
followed by the tools' outputs, the *results*, and the model replies to
those in turn, until it replies without calling a tool. The harness may
compact the conversation between turns and before results, to make room.

A turn can end in several ways: the model replies without calling a
tool; the reply is cut short for lack of room; the user interrupts while
the tools run, or while a compaction's note is written; the turn reaches
the most calls it may make. Whichever way it ends, the conversation it
leaves must follow the chat grammar, with every call answered by its
results: the next message follows a whole exchange, and a model, or a
hosted API, never sees a call without an answer.

**Requirements.**

1. *Every call is answered.* A reply that calls tools is followed by
   their results, whatever happens while they run: the outputs, or word
   that a call was not run because the turn was interrupted.
2. *Results always fit.* Results can be larger than the whole window: a
   tool may print more than the model can read. Then the largest output
   is replaced by a *note* saying how large it was and to ask for less,
   then the next largest, one at a time, until the results fit. The turn
   goes on: the model reads the notes and can ask again for less, such as
   a range of lines.
3. *A turn ends answered.* When a turn ends, by any of the ways above,
   the conversation is well-formed and does not end with a call.

The assumption about the window is that a note in place of every output
fits: notes are short, and the cap on calls bounds how many there are.

![A turn: a message, replies with calls and their results, and a reply with no call, after which the harness waits; the results of a reply are recorded whatever happens.](turn.svg)

See `harness/mod.rs`.
-/

namespace Dwim.Harness.Turn

open Dwim.Model Dwim.Harness Dwim.Harness.Compaction

/-! ## Results that do not fit

Each output is given by the tokens it takes. When they do not fit in the
room there is, the largest output is replaced by a note of `note`
tokens, and so on, until they fit, or no output is larger than a note.
-/

/-- The largest of `outputs`, or 0 if there are none. -/
def largest (outputs : List Nat) : Nat := outputs.foldr max 0

/-- Where the largest of `outputs` is: the first of them, if several are
as large. -/
def largestAt : List Nat → Nat
  | [] => 0
  | x :: xs => if largest xs ≤ x then 0 else largestAt xs + 1

#guard largestAt [30, 900, 20, 900] = 1

theorem le_largest {outputs : List Nat} {x : Nat} (h : x ∈ outputs) : x ≤ largest outputs := by
  induction outputs with
  | nil => cases h
  | cons y ys ih =>
    simp only [largest, List.foldr_cons] at ih ⊢
    rcases List.mem_cons.mp h with rfl | h
    · exact Nat.le_max_left _ _
    · exact Nat.le_trans (ih h) (Nat.le_max_right _ _)

theorem largestAt_lt {outputs : List Nat} (h : outputs ≠ []) : largestAt outputs < outputs.length := by
  induction outputs with
  | nil => contradiction
  | cons x xs ih =>
    simp only [largestAt]
    split
    · simp
    · rename_i hx
      have hxs : xs ≠ [] := by
        rintro rfl
        simp [largest] at hx
      simpa using ih hxs

theorem getElem_largestAt {outputs : List Nat} (h : largestAt outputs < outputs.length) :
    outputs[largestAt outputs] = largest outputs := by
  induction outputs with
  | nil => simp at h
  | cons x xs ih =>
    simp only [largestAt] at h ⊢
    split
    · rename_i hx
      simp [largest, List.foldr_cons] at hx ⊢
      omega
    · rename_i hx
      have h' : largestAt xs < xs.length := by simp only [hx, ite_false, List.length_cons] at h; omega
      simp only [List.getElem_cons_succ, ih h', largest, List.foldr_cons] at hx ⊢
      omega

theorem sum_set {outputs : List Nat} {i : Nat} (h : i < outputs.length) (x : Nat) :
    (outputs.set i x).sum + outputs[i] = outputs.sum + x := by
  induction outputs generalizing i with
  | nil => simp at h
  | cons y ys ih =>
    cases i with
    | zero => simp; omega
    | succ i =>
      have := ih (i := i) (by simpa using h)
      simp
      omega

theorem sum_le_of_le {outputs : List Nat} {n : Nat} (h : ∀ x ∈ outputs, x ≤ n) :
    outputs.sum ≤ outputs.length * n := by
  induction outputs with
  | nil => simp
  | cons x xs ih =>
    have hx := h x (List.mem_cons_self ..)
    have := ih fun y hy => h y (List.mem_cons_of_mem _ hy)
    simp [Nat.succ_mul]
    omega

/-- The outputs as recorded in `room` tokens: as they are, if they fit;
otherwise with the largest replaced by a note of `note` tokens, one at a
time, until they fit or none is larger than a note. -/
def fit (note room : Nat) (outputs : List Nat) : List Nat :=
  if _h : room < outputs.sum ∧ note < largest outputs then
    fit note room (outputs.set (largestAt outputs) note)
  else outputs
termination_by outputs.sum
decreasing_by
  have hne : outputs ≠ [] := by
    rintro rfl
    simp [largest] at _h
  have hlt := largestAt_lt hne
  have := sum_set hlt note
  rw [getElem_largestAt hlt] at this
  omega

-- Results that fit are recorded as they are.
#guard fit 40 1000 [300, 200] = [300, 200]
-- Two outputs of 900 tokens in 1000: the first is replaced by a note.
#guard fit 40 1000 [900, 900] = [40, 900]
-- Three large outputs in 500: the largest first, until they fit.
#guard fit 40 500 [300, 900, 400] = [300, 40, 40]

/-- Every call keeps its own result. -/
theorem fit_length (note room : Nat) (outputs : List Nat) :
    (fit note room outputs).length = outputs.length := by
  fun_induction fit note room outputs with
  | case1 outputs h ih => rw [ih, List.length_set]
  | case2 => rfl

/-- The results fit, when a note in place of every output does. -/
theorem fit_fits (note room : Nat) (outputs : List Nat) (h : outputs.length * note ≤ room) :
    (fit note room outputs).sum ≤ room := by
  fun_induction fit note room outputs with
  | case1 outputs _ ih => exact ih (by rw [List.length_set]; exact h)
  | case2 outputs hn =>
    by_cases hr : room < outputs.sum
    · have hl : largest outputs ≤ note := by
        apply Nat.le_of_not_lt
        intro hl
        exact hn ⟨hr, hl⟩
      have := sum_le_of_le fun x hx => Nat.le_trans (le_largest hx) hl
      omega
    · omega

/-- Results that fit are recorded as they are. -/
theorem fit_of_fits (note room : Nat) (outputs : List Nat) (h : outputs.sum ≤ room) :
    fit note room outputs = outputs := by
  rw [fit]
  simp only [dite_eq_right_iff]
  intro hr
  omega

/-! ## A turn

A turn is a sequence of steps. Where the turn is decides which steps can
come next: with no turn under way, a message; while the harness waits
for the model, a reply; while the tools of a reply that called them run,
their results, recorded in the room there is. Between turns, and before
results, the conversation may be compacted.

A reply cut short for lack of room keeps no calls, so it is a reply that
calls none. A turn interrupted while the tools run, or while a
compaction's note is written before their results, or that reaches the
cap on calls, still records the results, and then ends; the steps that
follow, such as the account the cap asks for, are messages and replies
like any other.
-/

/-- Where a turn is. -/
inductive Phase where
  /-- No turn under way: the harness waits for a message. -/
  | idle
  /-- The harness waits for the model's reply. -/
  | replying
  /-- The tools of a reply that called them are running. -/
  | running
  deriving DecidableEq, Repr

/-- What the harness holds: where the turn is, the conversation, and every
message the user has sent. -/
structure State where
  phase : Phase
  conversation : List Entry
  said : List Entry

/-- Before the first message. -/
def State.init : State := { phase := .idle, conversation := [], said := [] }

/-- The steps of a turn, in a window of `capacity` tokens, with notes of
`note` tokens in place of outputs that do not fit, and heads built by
`head`. -/
inductive Step (capacity note : Nat) (head : List Entry → Entry) : State → State → Prop where
  /-- With no turn under way, a message arrives: the user's, or a prompt
  of the harness's, such as the one that has the model go on. -/
  | message (s : State) (tokens : Nat) (prompt : Bool) (h : s.phase = .idle) :
      Step capacity note head s
        { phase := .replying
          conversation := s.conversation ++ [{ turn := .user, tokens, prompt }]
          said := if prompt then s.said else s.said ++ [{ turn := .user, tokens }] }
  /-- The model replies, calling tools or not. A turn whose reply calls
  none ends. -/
  | reply (s : State) (calls : Bool) (tokens : Nat) (h : s.phase = .replying) :
      Step capacity note head s
        { s with
          phase := if calls then .running else .idle
          conversation := s.conversation ++ [{ turn := .reply calls, tokens }] }
  /-- The tools ran, and their results are recorded in `room` tokens; the
  model is asked to reply to them. -/
  | results (s : State) (outputs : List Nat) (room : Nat) (h : s.phase = .running) :
      Step capacity note head s
        { s with
          phase := .replying
          conversation := s.conversation ++ [{ turn := .results, tokens := (fit note room outputs).sum }] }
  /-- The turn ends at its calls, interrupted or at the cap: their results
  are recorded all the same. -/
  | close (s : State) (outputs : List Nat) (room : Nat) (h : s.phase = .running) :
      Step capacity note head s
        { s with
          phase := .idle
          conversation := s.conversation ++ [{ turn := .results, tokens := (fit note room outputs).sum }] }
  /-- The conversation is compacted, between turns or before results. -/
  | compact (s : State) (h : s.phase ≠ .replying) :
      Step capacity note head s
        { s with conversation := Compaction.compact capacity head s.said s.conversation }

/-- The states a harness can reach from the first message on. -/
inductive Reachable (capacity note : Nat) (head : List Entry → Entry) : State → Prop where
  | init : Reachable capacity note head .init
  | step {s s' : State} :
      Reachable capacity note head s → Step capacity note head s s' →
        Reachable capacity note head s'

/-- The kind of the last turn of the conversation. -/
def State.last (s : State) : Option Turn := (s.conversation.map Entry.turn).getLast?

/-- The conversation is well-formed, and ends as where the turn is says:
between turns, not with a call; waiting for the model, with a message or
results; with the tools running, with the reply that called them. -/
def Settled (s : State) : Prop :=
  WellFormed (s.conversation.map Entry.turn) ∧
    match s.phase with
    | .idle => s.last ≠ some (.reply true)
    | .replying => s.last = some .user ∨ s.last = some .results
    | .running => s.last = some (.reply true)

/-!
The proof that every step keeps a state settled rests on two facts: the
grammar holds of a conversation with one more turn when that turn may
follow the last; and a compaction keeps the last turn, or leaves the
head last, which is not a call.
-/

theorem paired_append {l : List Turn} {x : Turn} (h : Paired l)
    (hx : ∀ a, l.getLast? = some a → Follows a x) : Paired (l ++ [x]) := by
  induction l with
  | nil => trivial
  | cons a t ih =>
    cases t with
    | nil => exact ⟨hx a rfl, trivial⟩
    | cons b rest =>
      refine ⟨h.1, ih h.2 fun c hc => hx c ?_⟩
      simpa [List.getLast?_cons] using hc

theorem wellFormed_append {l : List Turn} {x : Turn} (h : WellFormed l)
    (hx : ∀ a, l.getLast? = some a → Follows a x) (hne : l = [] → x ≠ .results) :
    WellFormed (l ++ [x]) := by
  refine ⟨?_, paired_append h.2 hx⟩
  cases l with
  | nil => simpa using hne rfl
  | cons a t => simpa using h.1

theorem replies_of_last {t : List Entry} (h : (t.map Entry.turn).getLast? = some (.reply true)) :
    t.length - 1 ∈ replies t := by
  rw [List.getLast?_map, List.getLast?_eq_getElem?] at h
  obtain ⟨e, he, het⟩ := Option.map_eq_some_iff.mp h
  have hlen : t.length - 1 < t.length := (List.getElem?_eq_some_iff.mp he).1
  rw [mem_replies]
  refine ⟨hlen, ?_⟩
  rw [he]
  simp [Entry.isReply, het]

theorem last_compact (capacity : Nat) (head : List Entry → Entry) (said t : List Entry)
    (hk : tail t (capacity / 8) < t.length) :
    ((Compaction.compact capacity head said t).map Entry.turn).getLast? =
      (t.map Entry.turn).getLast? := by
  simp only [Compaction.compact, select, List.map_cons, List.getLast?_cons, List.getLast?_map,
    List.getLast?_drop]
  have hn : ¬ t.length ≤ tail t (capacity / 8) := by omega
  simp only [hn, ↓reduceIte]
  cases h : t.getLast? with
  | none =>
    rw [List.getLast?_eq_none_iff] at h
    subst h
    simp at hk
  | some e => rfl

theorem last_compact_head (capacity : Nat) (head : List Entry → Entry) (said t : List Entry)
    (hk : t.length ≤ tail t (capacity / 8)) :
    ((Compaction.compact capacity head said t).map Entry.turn).getLast? =
      some (head (select capacity said t).2).turn := by
  simp only [Compaction.compact, select, List.map_cons, List.getLast?_cons, List.getLast?_map,
    List.getLast?_drop]
  simp only [hk, ↓reduceIte]
  rfl

/-- Every step keeps a state settled. -/
theorem step_settled {capacity note : Nat} {head : List Entry → Entry}
    (hhead : ∀ ms, (head ms).turn = .user) {s s' : State}
    (hs : Settled s) (hstep : Step capacity note head s s') : Settled s' := by
  cases hstep with
  | message tokens prompt h =>
    simp only [Settled, h] at hs
    refine ⟨?_, ?_⟩
    · rw [List.map_append]
      refine wellFormed_append hs.1 (fun a ha => ?_) (fun _ => by simp)
      simp only [Follows]
      constructor
      · intro hx; cases hx
      · intro ha'; subst ha'; exact absurd ha hs.2
    · simp [State.last]
  | reply calls tokens h =>
    simp only [Settled, h] at hs
    refine ⟨?_, ?_⟩
    · simp only [List.map_append]
      refine wellFormed_append hs.1 (fun a ha => ?_) (fun _ => by simp)
      simp only [Follows]
      constructor
      · intro hx; cases hx
      · intro ha'
        subst ha'
        rcases hs.2 with h2 | h2 <;> simp [State.last, ha] at h2
    · cases calls <;> simp [State.last]
  | results outputs room h =>
    simp only [Settled, h] at hs
    refine ⟨?_, ?_⟩
    · simp only [List.map_append]
      refine wellFormed_append hs.1 (fun a ha => ?_) (fun hnil => ?_)
      · simp only [State.last] at hs
        rw [hs.2] at ha
        cases ha
        simp [Follows]
      · simp [State.last, hnil] at hs
    · simp [State.last]
  | close outputs room h =>
    simp only [Settled, h] at hs
    refine ⟨?_, ?_⟩
    · simp only [List.map_append]
      refine wellFormed_append hs.1 (fun a ha => ?_) (fun hnil => ?_)
      · simp only [State.last] at hs
        rw [hs.2] at ha
        cases ha
        simp [Follows]
      · simp [State.last, hnil] at hs
    · simp [State.last]
  | compact h =>
    refine ⟨compact_wellFormed capacity head s.said s.conversation hhead hs.1, ?_⟩
    have hs2 := hs.2
    cases hp : s.phase with
    | replying => exact absurd hp h
    | running =>
      rw [hp] at hs2
      simp only [State.last] at hs2 ⊢
      have hmem := replies_of_last hs2
      have hr : replies s.conversation ≠ [] := List.ne_nil_of_mem hmem
      have hlast := le_of_getLast? (replies_sorted _) (getLast?_replies hr) hmem
      have hlt : last s.conversation < s.conversation.length :=
        (mem_replies.mp (List.mem_of_getLast? (getLast?_replies hr))).1
      have hk := tail_le_last s.conversation (capacity / 8)
      rw [last_compact capacity head s.said s.conversation (by omega)]
      exact hs2
    | idle =>
      rw [hp] at hs2
      simp only [State.last] at hs2 ⊢
      by_cases hk : tail s.conversation (capacity / 8) < s.conversation.length
      · rw [last_compact capacity head s.said s.conversation hk]
        exact hs2
      · rw [last_compact_head capacity head s.said s.conversation (by omega), hhead]
        simp

/-!
The harness can check this of any conversation between turns: whether it
is well-formed and does not end with a call.
-/

/-- Whether a conversation between turns is as `turn_ends_answered` says:
well-formed, and not ending with a call. -/
def answered (ts : List Turn) : Bool :=
  decide (WellFormed ts) && ts.getLast? != some (.reply true)

#guard answered [.user, .reply true, .results, .reply false]
-- Interrupted while the tools ran: the results are recorded all the same.
#guard answered [.user, .reply true, .results]
-- What the harness left when results did not fit: a call with no answer.
#guard !answered [.user, .reply true]
-- And the next message after it.
#guard !answered [.user, .reply true, .user]

/-! ## Theorems -/

/-- Every state a harness can reach is settled. -/
theorem reachable_settled {capacity note : Nat} {head : List Entry → Entry}
    (hhead : ∀ ms, (head ms).turn = .user) {s : State}
    (h : Reachable capacity note head s) : Settled s := by
  induction h with
  | init => exact ⟨⟨by simp [State.init], trivial⟩, by simp [State.init, State.last]⟩
  | step _ hstep ih => exact step_settled hhead ih hstep

/-- However a turn ends, the conversation it leaves is well-formed, and
does not end with a call: every call is answered (requirements 1
and 3). -/
theorem turn_ends_answered {capacity note : Nat} {head : List Entry → Entry}
    (hhead : ∀ ms, (head ms).turn = .user) {s : State}
    (h : Reachable capacity note head s) (hidle : s.phase = .idle) :
    WellFormed (s.conversation.map Entry.turn) ∧ s.last ≠ some (.reply true) := by
  have hs := reachable_settled hhead h
  simp only [Settled, hidle] at hs
  exact hs

end Dwim.Harness.Turn
