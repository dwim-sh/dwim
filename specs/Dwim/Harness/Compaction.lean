import Dwim.Harness

/-!
# Compaction

A language model conditions each reply on its whole context: the system
prompt followed by the conversation so far. The context is bounded by
the model's *context window*, a fixed capacity of `C` tokens; when dwim
runs the model itself, `C` is at least 16,384. Every turn appended to the
conversation consumes tokens and none is ever removed implicitly, so a
long task eventually exhausts the window.

*Compaction* keeps the context bounded. Before a turn is appended, the
harness checks whether the conversation would exceed the *high-water
mark* of `4/5 · C` tokens. If it would, the harness

1. asks the model to summarize the work so far in a *note* of at most
   `C/32` tokens, but no fewer than 256 and no more than 2,048;
2. discards the conversation, keeping the system prompt; and
3. starts a new conversation made of a *head* followed by a *tail*.

**Terms.**

- A *turn* is one element of the conversation: a user message, a reply,
  or tool results.
- A *user message* is a turn written by the user. The harness also writes
  turns in the user's role, such as a head; these are *prompts*, not user
  messages.
- A *reply* is a turn written by the model. It may *call* tools.
- *Tool results* are the outputs of the calls made by the reply
  immediately before them.
- An *exchange* is a reply together with the turns that follow it, up to
  the next reply.
- The *head* is the first turn of the compacted conversation: a prompt
  holding the note, the plan of the work if there is one, and a selection
  of the user's earlier messages.
- The *tail* is the part of the old conversation kept verbatim: its last
  few exchanges.

![Compaction replaces the turns before the tail with the head, which holds the note, the plan and the user messages, and keeps the tail unchanged.](compaction.svg)

**Requirements.** The compacted conversation must preserve what the
model needs to continue, and must be small:

1. *The state of the work.* The note records the task, the findings, what
   has been done and what remains, in the model's own words.
2. *The user's intent, verbatim.* A summary is lossy. The head therefore
   carries, word for word, the most recent of all the messages the user
   has sent that are not in the tail, within `C/16` tokens, however many
   compactions came before. The user's latest message always survives,
   even when it alone exceeds the budget.
3. *The recent history, verbatim.* The tail keeps the last exchanges
   within `C/8` tokens. It always contains the last exchange, even when
   that alone exceeds the budget: the last reply may have called tools
   whose results are appended right after the compaction, and they must
   follow the reply that called them.
4. *Well-formedness.* Every tool result follows the reply whose calls it
   answers, before and after compaction.
5. *Bounded size.* The compacted conversation takes at most the head's
   note and plan plus `C/16 + C/8` tokens, with the user's latest message
   in place of `C/16` and the last exchange in place of `C/8` when they
   are larger. The bound does not depend on earlier compactions, so
   repeated compaction cannot make the conversation grow.

**This page** defines the selection of the tail, the selection of the
user's messages, and compaction itself as Lean functions. Each definition
is illustrated by examples, and each `#guard` line is checked when the
page is built: if a definition stops producing the stated result, the
build fails. The page ends with theorems that prove the requirements for
every conversation. The content of the note is outside the model: only
the number of tokens it takes is represented.

The code is checked against this page by differential testing:
`difftest compaction` generates 10,000 transcripts that the chat grammar
allows, with budgets at and around the sizes of their suffixes, and
requires the Rust code to select the same tail and the same user
messages as the definitions here, given the same messages the user has
sent. `make check` runs it, together with
the proofs.

See `harness/compaction.rs`.
-/

namespace Dwim.Harness.Compaction

open Dwim.Model

/-! ## Notation

Compaction depends on two properties of a turn only: its kind and the
number of tokens it takes. The examples therefore write a conversation as
a list of turns, each given by its kind and its size:

| Notation     | Turn                                         |
|--------------|----------------------------------------------|
| `msg 10`     | a user message of 10 tokens                  |
| `prompt 100` | a prompt of 100 tokens, such as a head       |
| `call 50`    | a reply of 50 tokens that calls tools        |
| `res 20`     | tool results of 20 tokens                    |
| `answer 30`  | a reply of 30 tokens that calls no tools     |
-/

def msg (tokens : Nat) : Entry := { turn := .user, tokens }
def prompt (tokens : Nat) : Entry := { turn := .user, tokens, prompt := true }
def call (tokens : Nat) : Entry := { turn := .reply true, tokens }
def res (tokens : Nat) : Entry := { turn := .results, tokens }
def answer (tokens : Nat) : Entry := { turn := .reply false, tokens }

/-- The conversation used in most examples: two user messages, each
answered by a reply that calls a tool, the tool results, and a reply
that calls none. The numbers are the turns' indices.

```
0       1        2       3          4       5        6       7
msg 10  call 50  res 20  answer 30  msg 10  call 40  res 20  answer 30
```
-/
def conversation : List Entry :=
  [msg 10, call 50, res 20, answer 30, msg 10, call 40, res 20, answer 30]

/-! ## Selecting the tail

The tail is a suffix of the conversation, so selecting it means choosing
the index at which it starts. Not every index will do: a tail starting
between a reply and its tool results would begin with results that answer
no call, violating requirement 4. The tail therefore starts at a reply,
that is, at the start of an exchange.
-/

/-- The indices at which an exchange starts, that is, of the replies, in
increasing order. -/
def replies (t : List Entry) : List Nat :=
  (List.range t.length).filter fun i => t[i]?.any Entry.isReply

#guard replies conversation = [1, 3, 5, 7]

/-- The index of the last reply, or the length of `t` if it has none. -/
def last (t : List Entry) : Nat := (replies t).getLast?.getD t.length

#guard last conversation = 7
#guard last [msg 10] = 1

/-!
The tail is the longest suffix that starts at a reply and fits in the
budget. The suffixes starting at each reply of the example take:

```
        0       1        2       3          4       5        6       7
        msg 10  call 50  res 20  answer 30  msg 10  call 40  res 20  answer 30
from 7                                                               |- 30 --|
from 5                                              |---------- 90 ----------|
from 3                           |------------------- 130 -------------------|
from 1          |--------------------------- 200 ----------------------------|
```

With a budget of 90 the tail starts at 5. With a budget of 129 it still
starts at 5: index 4 holds a user message, not a reply, and the next
candidate, 3, needs 130. The user message at 4 is not lost: it falls in
the discarded part, from which the head carries user messages.

If no suffix fits, the tail is the last exchange regardless of its size
(requirement 3). The budget can therefore be exceeded, but only by the
last exchange; theorem `tail_fits` states this precisely.
-/

/-- The index at which the tail starts: the first reply from which the
rest of `t` fits in `budget`, or the last reply if there is none. -/
def tail (t : List Entry) (budget : Nat) : Nat :=
  ((replies t).find? fun r => size (t.drop r) ≤ budget).getD (last t)

#guard tail conversation 90 = 5
#guard tail conversation 129 = 5
#guard tail conversation 130 = 3
-- Everything fits: the tail starts at the first reply. The user message
-- before it is carried by the head.
#guard tail conversation 1000 = 1
-- Nothing fits: the tail is the last exchange.
#guard tail conversation 0 = 7
-- No reply: the tail is empty.
#guard tail [msg 10] 100 = 1

/-! ## Selecting the user's messages

The harness keeps every message the user has sent, in order; this page
calls that list `said`. The head carries user messages from `said`, not
from the conversation being discarded: a message carried by the head of
an earlier compaction is no longer in the conversation, but it is still
in `said`, so it can be carried again. However many times the
conversation is compacted, the user's words survive (requirement 2).

The user messages in the tail are the last ones in `said`, since the
tail is the end of the conversation. They need no carrying, so the
*candidates* are the user messages in `said` but for those last ones.
No user message is then both carried and kept.

Of the candidates, the head carries the most recent that fit in the
budget, in their original order: the longest suffix that fits. It stops
at the first message that does not fit, even if an older, shorter one
would, so the head shows an unbroken run of the most recent messages,
never one with gaps.

One message is carried whatever it takes: the user's latest, when the
tail holds no user message. It states what the model is working on, and
leaving it out because it is long, a pasted log say, would leave the
model working on a request it can no longer read. The budget can
therefore be exceeded, but only by that one message.

Prompts are never carried. Among them are the heads of earlier
compactions; if heads were carried, each head would contain its
predecessor, and the head would grow with every compaction, violating
requirement 5.
-/

/-- The user messages in `es`, without the prompts. -/
def own (es : List Entry) : List Entry := es.filter Entry.isOwn

/-- The longest suffix of `ms` that fits in `budget`. -/
def fitting (ms : List Entry) (budget : Nat) : List Entry :=
  let k := (List.range (ms.length + 1)).find? fun k => size (ms.drop k) ≤ budget
  ms.drop (k.getD ms.length)

#guard fitting [msg 30, msg 20, msg 40] 60 = [msg 20, msg 40]
-- msg 5 would fit, but msg 50 after it does not, so the suffix ends there.
#guard fitting [msg 5, msg 50, msg 10] 20 = [msg 10]

/-- The user messages the head may carry: those in `said`, but for the
last ones, which are in `tail`. -/
def candidates (said tail : List Entry) : List Entry :=
  (own said).take ((own said).length - (own tail).length)

#guard candidates [msg 30, msg 20, msg 40] [answer 5, msg 40, answer 5] = [msg 30, msg 20]

/-- The user messages the head carries: the longest suffix of the
candidates that fits in `budget`, but at least the user's latest message
when `tail` holds no user message. -/
def messages (said tail : List Entry) (budget : Nat) : List Entry :=
  let ms := fitting (candidates said tail) budget
  if ms.isEmpty && (own tail).isEmpty then (candidates said tail).getLast?.toList else ms

-- With an empty tail, every message is a candidate.
#guard messages [msg 30, msg 20, msg 40] [] 100 = [msg 30, msg 20, msg 40]
#guard messages [msg 30, msg 20, msg 40] [] 60 = [msg 20, msg 40]
#guard messages [msg 30, msg 20, msg 40] [] 59 = [msg 40]
-- Nothing fits, and the tail holds no user message: the latest is carried
-- whatever it takes.
#guard messages [msg 30, msg 20, msg 40] [] 39 = [msg 40]
-- The latest is in the tail, so it is not carried, and nothing else fits.
#guard messages [msg 30, msg 20, msg 40] [msg 40, answer 5] 10 = []
#guard messages [msg 30, msg 20, msg 40] [msg 40, answer 5] 50 = [msg 30, msg 20]
-- Prompts are never carried.
#guard messages [msg 30, prompt 50, msg 20] [] 100 = [msg 30, msg 20]

/-! ## Compacting a conversation

Compaction selects the tail within `C/8` tokens, selects the user
messages from `said` within `C/16` tokens, and returns the head built
from those messages followed by the tail.

For a window of `C = 720` tokens, the tail may take 90 tokens and the
user messages 45:

```
before  msg 10  call 50  res 20  answer 30  msg 10  call 40  res 20  answer 30
        |-------------- into head ---------------|  |--------- tail ---------|
after   head [msg 10, msg 10]                       call 40  res 20  answer 30
```

Both user messages fit in 45 tokens, so the head carries both. How the
head is built is a parameter of the definition; the examples take a
head of 100 tokens for the note and the plan, plus the messages.
-/

/-- What compacting `t` keeps in a window of `capacity` tokens, when the
user has said `said`: where the tail starts, and the user messages the
head carries. -/
def select (capacity : Nat) (said t : List Entry) : Nat × List Entry :=
  let start := tail t (capacity / 8)
  (start, messages said (t.drop start) (capacity / 16))

/-- The conversation that compacting `t` leaves: the head that `head`
builds from the selected user messages, followed by the tail. -/
def compact (capacity : Nat) (head : List Entry → Entry) (said t : List Entry) :
    List Entry :=
  head (select capacity said t).2 :: t.drop (select capacity said t).1

/-- The head in the examples: a prompt of 100 tokens for the note and the
plan, plus the user messages it carries. -/
def head (ms : List Entry) : Entry := prompt (100 + size ms)

#guard compact 720 head [msg 10, msg 10] conversation = [prompt 120, call 40, res 20, answer 30]

/-!
The conversation then continues and is compacted again. Here one more
user message is answered after a tool call:
-/

def compacted : List Entry :=
  compact 720 head [msg 10, msg 10] conversation ++ [msg 10, call 40, res 20, answer 30]

/-!
The second head carries all three user messages: the two carried by the
first head come from `said`, not from the conversation, which holds them
only inside that head. A user message survives any number of
compactions while it fits.
-/

#guard compact 720 head [msg 10, msg 10, msg 10] compacted = [prompt 130, call 40, res 20, answer 30]

/-! ## Theorems

The examples cover particular conversations; the theorems below hold
for all of them. Lean checks every step of every proof.

The proofs rest on four lemmas: the reply indices are strictly
increasing; an index is a reply index exactly when a reply is there; in
a strictly increasing list, the first element satisfying a predicate is
no greater than any other element satisfying it; and the last element
is no smaller than any.
-/

theorem replies_sorted (t : List Entry) : (replies t).Pairwise (· < ·) :=
  List.pairwise_lt_range.filter _

theorem mem_replies {t : List Entry} {r : Nat} :
    r ∈ replies t ↔ r < t.length ∧ (t[r]?.any Entry.isReply) = true := by
  simp [replies, List.mem_filter, List.mem_range]

theorem find?_le {l : List Nat} (hl : l.Pairwise (· < ·)) {p : Nat → Bool} {k j : Nat}
    (hk : l.find? p = some k) (hj : j ∈ l) (hp : p j = true) : k ≤ j := by
  obtain ⟨-, as, bs, rfl, has⟩ := List.find?_eq_some_iff_append.mp hk
  rw [List.mem_append, List.mem_cons] at hj
  rcases hj with hj | rfl | hj
  · have := has j hj
    simp [hp] at this
  · exact Nat.le_refl _
  · exact Nat.le_of_lt (List.rel_of_pairwise_cons (List.pairwise_append.mp hl).2.1 hj)

theorem le_of_getLast? {l : List Nat} (hl : l.Pairwise (· < ·)) {z x : Nat}
    (hz : l.getLast? = some z) (hx : x ∈ l) : x ≤ z := by
  obtain ⟨ys, rfl⟩ := List.getLast?_eq_some_iff.mp hz
  rw [List.mem_append, List.mem_singleton] at hx
  rcases hx with hx | rfl
  · exact Nat.le_of_lt ((List.pairwise_append.mp hl).2.2 x hx z (List.mem_singleton_self z))
  · exact Nat.le_refl _

theorem getLast?_replies {t : List Entry} (h : replies t ≠ []) :
    (replies t).getLast? = some (last t) := by
  cases hz : (replies t).getLast? with
  | none => simp [List.getLast?_eq_none_iff] at hz; contradiction
  | some z => simp [last, hz]

theorem tail_mem {t : List Entry} {budget : Nat} (h : replies t ≠ []) :
    tail t budget ∈ replies t := by
  unfold tail
  cases hf : (replies t).find? (fun r => decide (size (t.drop r) ≤ budget)) with
  | none => exact List.mem_of_getLast? (getLast?_replies h)
  | some r => exact List.mem_of_find?_eq_some hf

theorem tail_of_nil {t : List Entry} {budget : Nat} (h : replies t = []) :
    tail t budget = t.length := by
  simp [tail, last, h]

/-!
**The tail.** It starts at a reply (requirement 4), contains the last
exchange (requirement 3), fits in its budget unless the last exchange
alone does not, and is the longest suffix that does.
-/

/-- The tail starts at a reply, if `t` has one. -/
theorem tail_at_reply (t : List Entry) (budget : Nat) (h : replies t ≠ []) :
    (t.drop (tail t budget)).head?.any Entry.isReply := by
  rw [List.head?_drop]
  exact (mem_replies.mp (tail_mem h)).2

/-- The tail contains the last exchange. -/
theorem tail_le_last (t : List Entry) (budget : Nat) : tail t budget ≤ last t := by
  by_cases h : replies t = []
  · simp [tail, last, h]
  · exact le_of_getLast? (replies_sorted t) (getLast?_replies h) (tail_mem h)

/-- The tail fits in the budget, or it is exactly the last exchange. -/
theorem tail_fits (t : List Entry) (budget : Nat) :
    size (t.drop (tail t budget)) ≤ budget ∨ tail t budget = last t := by
  unfold tail
  cases hf : (replies t).find? (fun r => decide (size (t.drop r) ≤ budget)) with
  | none => exact Or.inr rfl
  | some s => exact Or.inl (by simpa using List.find?_some hf)

/-- No suffix starting at an earlier reply fits in the budget. -/
theorem tail_longest (t : List Entry) (budget : Nat) :
    ∀ r ∈ replies t, r < tail t budget → budget < size (t.drop r) := by
  intro r hr hlt
  apply Nat.lt_of_not_le
  intro hp
  unfold tail at hlt
  cases hf : (replies t).find? (fun r => decide (size (t.drop r) ≤ budget)) with
  | none => exact List.find?_eq_none.mp hf r hr (by simpa using hp)
  | some s =>
    rw [hf] at hlt
    have := find?_le (replies_sorted t) hf hr (by simpa using hp)
    simp at hlt
    omega

/-!
**The user messages.** The selection contains user messages only, so
heads never nest (requirement 5); it is a suffix of the candidates,
hence the most recent in their original order and none of those in the
tail (requirement 2); it fits in its budget unless it is the user's
latest message alone; no longer suffix fits; and the user's latest
message is always carried when the tail holds none (requirement 2).

These rest on three facts about `fitting`: it is a suffix, it fits, and
no longer suffix does.
-/

theorem fitting_drop (ms : List Entry) (budget : Nat) :
    ∃ k, fitting ms budget = ms.drop k :=
  ⟨_, rfl⟩

theorem fitting_fit (ms : List Entry) (budget : Nat) : size (fitting ms budget) ≤ budget := by
  simp only [fitting]
  cases hf : (List.range (ms.length + 1)).find?
      (fun k => decide (size (ms.drop k) ≤ budget)) with
  | none => simp [size]
  | some k => simpa using List.find?_some hf

theorem fitting_longest (ms m : List Entry) (budget : Nat) (hm : m <:+ ms)
    (hfit : size m ≤ budget) : m.length ≤ (fitting ms budget).length := by
  simp only [fitting]
  have hle := hm.length_le
  have hj : m = ms.drop (ms.length - m.length) := List.suffix_iff_eq_drop.mp hm
  have hp : decide (size (ms.drop (ms.length - m.length)) ≤ budget) = true := by
    rw [← hj]; simpa using hfit
  have hmem : ms.length - m.length ∈ List.range (ms.length + 1) := by
    simp [List.mem_range]; omega
  cases hf : (List.range (ms.length + 1)).find?
      (fun k => decide (size (ms.drop k) ≤ budget)) with
  | none => exact absurd hp (List.find?_eq_none.mp hf _ hmem)
  | some k =>
    have := find?_le List.pairwise_lt_range hf hmem hp
    simp [List.length_drop]
    omega

theorem candidates_of_nil {said tail : List Entry} (h : own tail = []) :
    candidates said tail = own said := by
  simp [candidates, h]

theorem getLast?_suffix (l : List Entry) : l.getLast?.toList <:+ l := by
  cases h : l.getLast? with
  | none => exact List.nil_suffix
  | some z =>
    obtain ⟨ys, rfl⟩ := List.getLast?_eq_some_iff.mp h
    exact List.suffix_append ys [z]

/-- The selection contains user messages only, no prompts. -/
theorem messages_own (said tail : List Entry) (budget : Nat) :
    ∀ e ∈ messages said tail budget, e.isOwn := by
  intro e he
  have hc : e ∈ candidates said tail := by
    simp only [messages] at he
    split at he
    · exact (getLast?_suffix _).subset he
    · obtain ⟨k, hk⟩ := fitting_drop (candidates said tail) budget
      rw [hk] at he
      exact List.mem_of_mem_drop he
  exact (List.mem_filter.mp (List.mem_of_mem_take hc)).2

/-- The selection is a suffix of the candidates: the most recent of them,
in order. -/
theorem messages_suffix (said tail : List Entry) (budget : Nat) :
    messages said tail budget <:+ candidates said tail := by
  simp only [messages]
  split
  · exact getLast?_suffix _
  · obtain ⟨k, hk⟩ := fitting_drop (candidates said tail) budget
    rw [hk]
    exact List.drop_suffix _ _

/-- The selection fits in the budget, or it is the user's latest message
alone, and the tail holds no user message. -/
theorem messages_fit (said tail : List Entry) (budget : Nat) :
    size (messages said tail budget) ≤ budget ∨
      (own tail = [] ∧ messages said tail budget = (own said).getLast?.toList) := by
  simp only [messages]
  split
  · rename_i h
    simp only [Bool.and_eq_true, List.isEmpty_iff] at h
    exact Or.inr ⟨h.2, by rw [candidates_of_nil h.2]⟩
  · exact Or.inl (fitting_fit _ _)

/-- No longer suffix of the candidates fits in the budget. -/
theorem messages_longest (said tail m : List Entry) (budget : Nat)
    (hm : m <:+ candidates said tail) (hfit : size m ≤ budget) :
    m.length ≤ (messages said tail budget).length := by
  have := fitting_longest _ m budget hm hfit
  simp only [messages]
  split
  · rename_i h
    simp only [Bool.and_eq_true, List.isEmpty_iff] at h
    rw [h.1, List.length_nil] at this
    omega
  · exact this

/-- When the tail holds no user message, the user's latest message is
carried. -/
theorem messages_latest (said tail : List Entry) (budget : Nat) (h : own tail = []) :
    (messages said tail budget).getLast? = (own said).getLast? := by
  simp only [messages, candidates_of_nil h]
  split
  · cases (own said).getLast? <;> rfl
  · rename_i hne
    simp only [h, List.isEmpty_nil, Bool.and_true, List.isEmpty_iff] at hne
    obtain ⟨k, hk⟩ := fitting_drop (own said) budget
    rw [hk] at hne ⊢
    rw [List.getLast?_drop]
    have : ¬ (own said).length ≤ k := by
      intro hle
      exact hne (List.drop_eq_nil_of_le hle)
    simp [this]

/-!
**The compaction.** Compaction preserves well-formedness
(requirement 4), and the size of its result is bounded independently of
the conversation's history (requirement 5).

The first proof uses that well-formedness is preserved by taking a
suffix; the second, that the size of a list is the size of its first
turn plus the size of the rest.
-/

theorem paired_of_cons {a : Turn} {l : List Turn} (h : Paired (a :: l)) : Paired l := by
  cases l with
  | nil => trivial
  | cons b rest => exact h.2

theorem paired_drop {l : List Turn} (n : Nat) (h : Paired l) : Paired (l.drop n) := by
  induction n generalizing l with
  | zero => simpa using h
  | succ n ih =>
    cases l with
    | nil => simp [Paired]
    | cons a l => exact ih (paired_of_cons h)

theorem size_cons (e : Entry) (es : List Entry) : size (e :: es) = e.tokens + size es := by
  simp [size]

/-- Compacting a well-formed conversation yields a well-formed one. -/
theorem compact_wellFormed (capacity : Nat) (head : List Entry → Entry) (said t : List Entry)
    (hhead : ∀ ms, (head ms).turn = .user) (ht : WellFormed (t.map Entry.turn)) :
    WellFormed ((compact capacity head said t).map Entry.turn) := by
  have hp : Paired ((t.drop (tail t (capacity / 8))).map Entry.turn) := by
    rw [List.map_drop]
    exact paired_drop _ ht.2
  simp only [compact, select, List.map_cons, hhead]
  refine ⟨by simp, ?_⟩
  cases hd : t.drop (tail t (capacity / 8)) with
  | nil => trivial
  | cons e rest =>
    rw [hd] at hp
    have hr : replies t ≠ [] := by
      intro h
      rw [tail_of_nil h, List.drop_length] at hd
      cases hd
    have hreply := tail_at_reply t (capacity / 8) hr
    rw [hd] at hreply
    refine ⟨?_, hp⟩
    simp only [Follows]
    constructor
    · intro he
      simp [Entry.isReply, he] at hreply
    · intro h
      cases h

/-- The compacted conversation takes at most the head's `overhead` for the
note and the plan; `C/16` tokens of user messages, or the user's latest
message in place of `C/16` when it is larger; and `C/8` tokens of tail,
or the last exchange in place of `C/8` when it is larger. -/
theorem compact_size (capacity overhead : Nat) (head : List Entry → Entry) (said t : List Entry)
    (hhead : ∀ ms, (head ms).tokens ≤ overhead + size ms) :
    size (compact capacity head said t)
      ≤ overhead + max (capacity / 16) (size (own said).getLast?.toList)
        + max (capacity / 8) (size (t.drop (last t))) := by
  simp only [compact, select, size_cons]
  have h1 := hhead (messages said (t.drop (tail t (capacity / 8))) (capacity / 16))
  have h2 : size (messages said (t.drop (tail t (capacity / 8))) (capacity / 16))
      ≤ max (capacity / 16) (size (own said).getLast?.toList) := by
    rcases messages_fit said (t.drop (tail t (capacity / 8))) (capacity / 16) with h | ⟨-, h⟩
    · omega
    · rw [h]; omega
  rcases tail_fits t (capacity / 8) with h3 | h3
  · omega
  · have : size (t.drop (tail t (capacity / 8))) = size (t.drop (last t)) := by rw [h3]
    omega

end Dwim.Harness.Compaction
