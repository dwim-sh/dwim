/-!
# Compaction

Each token a language model outputs depends on its whole context: the
system prompt and the conversation so far. The context holds at most `C`
tokens, the model's *context window*. A conversation only grows, so a
long task fills the window. *Compaction* is what dwim does then: it
replaces the conversation with a shorter one, and the work goes on from
there.

## Intents

1. **Compaction frees up space in the context window so the model can
   continue working.** The conversation grows with every turn, and the
   window does not. Compaction makes the conversation short again, short
   enough to go on, however many times it has been compacted before.
2. **Compaction removes only information that can be retrieved and notes
   where to find it.** Forgetting is safe only when the model can get
   back what it forgot. Everything compaction removes stays in the
   session's transcript, and the compacted conversation says where the
   transcript is.
3. **Compaction never changes the meaning of the information it keeps.**
   A model working from a distorted record is worse off than one that
   forgot. What compaction keeps, the user's words and the last
   exchanges, stays exactly as it was. The note is the model's own
   summary, and is kept apart as such.

## The design

When the next turn would take the conversation past four fifths of the
window, the *high-water mark*, the harness compacts it first:

1. The model writes a *note* about the conversation: the task, what it
   found, what it has done, and what is next. The note takes at most
   `C/32` tokens, and no fewer than 256 or more than 2,048.
2. The harness keeps the *tail*: the last exchanges of the conversation,
   within `C/8` tokens, exactly as they were.
3. It puts a *briefing* before the tail: one turn that holds the note,
   the plan of the work, the user's earlier messages word for word,
   within `C/16` tokens, and where the session's transcript is.
4. The *transcript* holds every turn of the session, and every briefing,
   in the order they came. Nothing is ever removed from it, so whatever
   leaves the conversation can be read there.

See `harness/compaction.rs`.
-/

namespace Dwim.Harness.Compaction

/-! ## Turns

A conversation is a list of turns. A turn is a message from the user, a
prompt from the harness, a reply from the model, the results of the
tools a reply called, or a briefing. Compaction removes the turns before
the tail and puts a briefing in their place, so a compacted conversation
is a briefing followed by the tail.
-/

/-- A `Briefing` is the turn compaction puts in place of the turns it
removes. Its `note` is the model's summary of the conversation, written
just before compaction. Its `plan` is the plan of the work, as it
stands. Its `messages` are the user's most recent messages, word for
word, as many as fit in the message budget, so the user's instructions
are never left to the summary. Its `transcript` is where the session's
transcript is: the file that holds every turn of the session, so the
model can read whatever compaction removed. -/
structure Briefing where
  note : String
  plan : String
  messages : List String
  transcript : String
  deriving DecidableEq, Repr

/-- A `Turn` is a turn of the conversation: a message the `user` wrote;
a `prompt` the harness writes in the user's place, such as asking the
model to go on; a `reply` of the model's, and whether it calls tools;
the `results` of the tools the reply before called; or a `briefing`,
which compaction puts in place of the turns it removes. -/
inductive Turn where
  | user (text : String)
  | prompt (text : String)
  | reply (text : String) (calls : Bool)
  | results (text : String)
  | briefing (briefing : Briefing)
  deriving DecidableEq, Repr

namespace Turn

/-- `isReply` is whether the turn is a reply. -/
def isReply : Turn → Bool
  | reply .. => true
  | _ => false

/-- `isBriefing` is whether the turn is a briefing. -/
def isBriefing : Turn → Bool
  | briefing _ => true
  | _ => false

/-- `said?` is the words of the turn, if the user wrote it. -/
def said? : Turn → Option String
  | user text => some text
  | _ => none

end Turn

/-- `users` is the messages the user wrote among `turns`, in order. -/
def users (turns : List Turn) : List String := turns.filterMap Turn.said?

/-! ## Tokens

What compaction does depends on how many tokens each turn takes. A
tokenizer counts the tokens of a text; a briefing takes those of what it
holds, and a fixed `frame` for the words around them. The theorems hold
for every tokenizer.
-/

/-- A `Tokenizer` is how many tokens a text takes, its `count`, and how
many the fixed words of a briefing take, its `frame`. -/
structure Tokenizer where
  count : String → Nat
  frame : Nat

/-- `cost` is the tokens a list of texts takes. -/
def cost (k : Tokenizer) (texts : List String) : Nat := (texts.map k.count).sum

/-- `Turn.tokens` is the tokens a turn takes. -/
def Turn.tokens (k : Tokenizer) : Turn → Nat
  | user text | prompt text | reply text _ | results text => k.count text
  | briefing b => k.frame + k.count b.note + k.count b.plan + k.count b.transcript + cost k b.messages

/-- `size` is the tokens a list of turns takes. -/
def size (k : Tokenizer) (turns : List Turn) : Nat := (turns.map (Turn.tokens k)).sum

/-- `words` is the tokenizer of the examples: a token is a word, and a
briefing's fixed words take 10. -/
def words : Tokenizer where
  count text := ((text.splitOn " ").filter (· ≠ "")).length
  frame := 10

#guard words.count "Fix the failing test" = 4

/-! ## Budgets

Each part of a compacted conversation has its share of the window. The
examples are for a window of 16,384 tokens, the smallest dwim runs the
model in.
-/

/-- `highWater` is the high-water mark: four fifths of the window. -/
def highWater (C : Nat) : Nat := C * 4 / 5

/-- `noteBudget` is the most tokens the note may take: a thirty-second
of the window, but no fewer than 256 and no more than 2,048. -/
def noteBudget (C : Nat) : Nat := min 2048 (max 256 (C / 32))

/-- `tailBudget` is the most tokens the tail may take: an eighth of the
window. -/
def tailBudget (C : Nat) : Nat := C / 8

/-- `messageBudget` is the most tokens the user's messages in the
briefing may take: a sixteenth of the window. -/
def messageBudget (C : Nat) : Nat := C / 16

#guard highWater 16384 = 13107
#guard noteBudget 16384 = 512
#guard tailBudget 16384 = 2048
#guard messageBudget 16384 = 1024
-- The note's bounds, in a small window and a large one.
#guard noteBudget 4096 = 256
#guard noteBudget 262144 = 2048

/-! ## An example

The examples follow one session. The user asks for a fix, and the model
reads, runs and edits until it is done; then the user asks for more.
-/

def fix := Turn.user "Fix the failing test in parser.rs"
def run := Turn.reply "I will run the tests first" true
def failed := Turn.results "test parse_empty failed: expected None but got Some"
def read := Turn.reply "The parser accepts empty input, so I will read parse" true
def source := Turn.results "fn parse(input: &str) -> Option<Ast> { let tokens = lex(input); Some(build(tokens)) }"
def edit := Turn.reply "I will return None when there are no tokens" true
def edited := Turn.results "parser.rs: 1 line changed"
def done := Turn.reply "Fixed: parse returns None on empty input, and the test passes" false
def more := Turn.user "Thanks. Now handle input that is only whitespace too"
def rerun := Turn.reply "I will add a test for whitespace and run it" true

/-- `conversation` is the session so far: two requests, and the work on
them, up to a call whose results are yet to come. -/
def conversation : List Turn := [fix, run, failed, read, source, edit, edited, done, more, rerun]

#guard conversation.map (Turn.tokens words) = [6, 6, 8, 10, 12, 9, 4, 11, 9, 10]
#guard size words conversation = 85

/-! ## The harness

Compaction takes what the harness knows: the window, the tokenizer, the
plan, where the transcript is, and the model, which writes the note. The
model may write any note; the harness stops it at the note's budget,
which the theorems take as given.
-/

/-- The `Harness` is what compaction takes from the harness: the
`capacity` of the context window, `C`; the `tokenizer`; the `plan` of
the work, as it stands; where the session's `transcript` is; and the
note the model would `write` about a conversation. -/
structure Harness where
  capacity : Nat
  tokenizer : Tokenizer
  plan : String
  transcript : String
  write : List Turn → String

/-- `harness` is the harness of the examples: a window of 160 words, so
a tail of 20 and messages of 10. -/
def harness : Harness where
  capacity := 160
  tokenizer := words
  plan := ""
  transcript := "~/.dwim/transcripts/session.jsonl"
  write _ := "Task: fix parse on empty input. Done: fixed. Next: whitespace"

/-! ## Selecting the tail

The tail is the end of the conversation, so selecting it means choosing
the turn it starts at. It must start at a reply: a tail that started
between a reply and its results would begin with results that answer no
call. The tail is the longest end of the conversation that starts at a
reply and fits in its budget. If none fits, it is the last exchange, the
last reply and what follows it, whatever that takes: the last reply may
have called tools whose results come right after the compaction, and
they must follow the reply that called them.
-/

/-- `replies` is where the replies are in `turns`, in order. -/
def replies (turns : List Turn) : List Nat :=
  (List.range turns.length).filter fun i => turns[i]?.any Turn.isReply

/-- `last` is where the last reply is, or the length of `turns` if there
is none. -/
def last (turns : List Turn) : Nat := (replies turns).getLast?.getD turns.length

/-- `tail` is where the tail starts: at the first reply from which
the rest of `turns` fits in `budget`, or at the last reply if there
is none. -/
def tail (k : Tokenizer) (turns : List Turn) (budget : Nat) : Nat :=
  ((replies turns).find? fun r => size k (turns.drop r) ≤ budget).getD (last turns)

#guard replies conversation = [1, 3, 5, 7, 9]
-- From the last reply, `rerun`, the rest takes 10 words; from `done`, 30.
#guard tail words conversation 20 = 9
#guard tail words conversation 30 = 7
-- Nothing fits: the tail is the last exchange all the same.
#guard tail words conversation 5 = 9

/-! ## Selecting the user's messages

The harness keeps every message the user has sent, in order; this page
calls the list `said`. The briefing carries messages from `said`, not
from the conversation being compacted: a message carried by an earlier
briefing is no longer in the conversation, but it is still in `said`.

The user's messages in the tail are the last ones in `said`, and they
need no carrying. The others are the *candidates*. The briefing carries
the most recent candidates that fit in the budget, oldest first. It
stops at the first that does not fit, even if an older, shorter one
would, so the messages it carries have no gaps. When the tail holds no
message of the user's, the briefing carries the latest whatever it
takes: it says what the model is working on.
-/

/-- `candidates` is the messages the briefing may carry: those in `said`
but for the last ones, which are in `kept`. -/
def candidates (said : List String) (kept : List Turn) : List String :=
  said.take (said.length - (users kept).length)

/-- `fitting` is the longest end of `texts` that fits in `budget`. -/
def fitting (k : Tokenizer) (texts : List String) (budget : Nat) : List String :=
  let i := (List.range (texts.length + 1)).find? fun i => cost k (texts.drop i) ≤ budget
  texts.drop (i.getD texts.length)

/-- `carried` is the messages the briefing carries: the most recent
candidates that fit in `budget`, and at least the latest when `kept`
holds no message of the user's. -/
def carried (k : Tokenizer) (said : List String) (kept : List Turn) (budget : Nat) :
    List String :=
  let texts := fitting k (candidates said kept) budget
  if texts.isEmpty && (users kept).isEmpty then (candidates said kept).getLast?.toList
  else texts

/-- `said` is what the user said in the example session. -/
def said : List String := ["Fix the failing test in parser.rs", "Thanks. Now handle input that is only whitespace too"]

-- The tail from `rerun` holds no message of the user's: both are
-- candidates, and in 10 words only the latest fits.
#guard carried words said (conversation.drop 9) 10 = ["Thanks. Now handle input that is only whitespace too"]
#guard carried words said (conversation.drop 9) 15 = said
-- The tail from `more` holds the latest, so it is not carried too.
#guard carried words said (conversation.drop 8) 10 = ["Fix the failing test in parser.rs"]
-- Nothing fits, and the tail holds no message of the user's: the latest
-- is carried whatever it takes.
#guard carried words said (conversation.drop 9) 3 = ["Thanks. Now handle input that is only whitespace too"]

/-! ## Compacting

A session is what the harness holds: the transcript, what the user has
said, and the conversation. Compacting it replaces the conversation with
a briefing and the tail, and adds the briefing to the transcript.
-/

/-- A `Session` is what the harness holds: the `transcript`, every turn
of the session and every briefing, in order; what the user has `said`,
every message, in order; and the `conversation`, which the model is
given after the system prompt. -/
structure Session where
  transcript : List Turn := []
  said : List String := []
  conversation : List Turn := []
  deriving Repr

/-- `compact` is the session after its conversation is compacted. -/
def compact (h : Harness) (s : Session) : Session :=
  let kept := s.conversation.drop (tail h.tokenizer s.conversation (tailBudget h.capacity))
  let briefing := Turn.briefing {
    note := h.write s.conversation
    plan := h.plan
    messages := carried h.tokenizer s.said kept (messageBudget h.capacity)
    transcript := h.transcript }
  { s with conversation := briefing :: kept, transcript := s.transcript ++ [briefing] }

/-- `full` is whether adding `t` would take the conversation past the
high-water mark. An empty conversation is never compacted. -/
def full (h : Harness) (s : Session) (t : Turn) : Bool :=
  !s.conversation.isEmpty &&
    highWater h.capacity < size h.tokenizer s.conversation + t.tokens h.tokenizer

/-- `add` is the session after `t` is added to it, compacted first if
`t` would take it past the high-water mark. -/
def add (h : Harness) (s : Session) (t : Turn) : Session :=
  let s := if full h s t then compact h s else s
  { transcript := s.transcript ++ [t], said := s.said ++ t.said?.toList, conversation := s.conversation ++ [t] }

/-- `play` is the session after the turns `ts`, added one at a time. -/
def play (h : Harness) (ts : List Turn) : Session := ts.foldl (add h) {}

/-!
The example session takes 85 words, under the high-water mark of 128.
The next turn is the results of the tests `rerun` ran, 46 words, which
would take it to 131, so the conversation is compacted first. The tail
is the last exchange: `rerun` alone, 10 words of the 20 it may take, and
the results must follow it. It holds no message of the user's, so the
briefing carries the latest, which fits in 10 words; the one before it
would not. Then the results go in, after the call they answer.
-/

def report := Turn.results "running 6 tests: parse_empty ok, parse_whitespace ok, parse_tabs FAILED, parse_newlines FAILED, parse_mixed ok, parse_unicode ok. failures: parse_tabs expected None but got Some for a tab; parse_newlines expected None but got Some for a newline. test result: FAILED. 4 passed, 2 failed, 0 ignored, in 0.02 seconds"

#guard report.tokens words = 46

def session := play harness (conversation ++ [report])

def compacted : Turn := .briefing {
  note := "Task: fix parse on empty input. Done: fixed. Next: whitespace"
  plan := ""
  messages := ["Thanks. Now handle input that is only whitespace too"]
  transcript := "~/.dwim/transcripts/session.jsonl" }

#guard session.conversation = [compacted, rerun, report]

-- The transcript has every turn, the briefing among them, in the order they came.
#guard session.transcript = conversation ++ [compacted, report]

/-! # The intents, proved

## Lemmas
-/

theorem getLast?_suffix {α : Type} (l : List α) : l.getLast?.toList <:+ l := by
  cases h : l.getLast? with
  | none => exact List.nil_suffix
  | some z =>
    obtain ⟨ys, rfl⟩ := List.getLast?_eq_some_iff.mp h
    exact List.suffix_append ys [z]

theorem size_cons (k : Tokenizer) (t : Turn) (ts : List Turn) :
    size k (t :: ts) = t.tokens k + size k ts := by
  simp [size]

/-- `tail_fits` says the tail fits in its budget, or it is the last
exchange. -/
theorem tail_fits (k : Tokenizer) (turns : List Turn) (budget : Nat) :
    size k (turns.drop (tail k turns budget)) ≤ budget ∨ tail k turns budget = last turns := by
  unfold tail
  cases hf : (replies turns).find? (fun r => decide (size k (turns.drop r) ≤ budget)) with
  | none => exact Or.inr rfl
  | some r => exact Or.inl (by simpa using List.find?_some hf)

/-- `fitting_fits` says the longest end of a list that fits in a budget
does fit in it. -/
theorem fitting_fits (k : Tokenizer) (texts : List String) (budget : Nat) :
    cost k (fitting k texts budget) ≤ budget := by
  simp only [fitting]
  cases hf : (List.range (texts.length + 1)).find?
      (fun i => decide (cost k (texts.drop i) ≤ budget)) with
  | none => simp [cost]
  | some i => simpa using List.find?_some hf

theorem candidates_of_nil {said : List String} {kept : List Turn} (h : users kept = []) :
    candidates said kept = said := by
  simp [candidates, h]

/-- `carried_fits` says the messages the briefing carries fit in their
budget, or they are the user's latest message alone, carried because the
tail holds none. -/
theorem carried_fits (k : Tokenizer) (said : List String) (kept : List Turn) (budget : Nat) :
    cost k (carried k said kept budget) ≤ budget ∨
      (users kept = [] ∧ carried k said kept budget = said.getLast?.toList) := by
  simp only [carried]
  split
  · rename_i h
    simp only [Bool.and_eq_true, List.isEmpty_iff] at h
    exact Or.inr ⟨h.2, by rw [candidates_of_nil h.2]⟩
  · exact Or.inl (fitting_fits _ _ _)

/-- `carried_suffix` says the messages the briefing carries are the last
of the candidates. -/
theorem carried_suffix (k : Tokenizer) (said : List String) (kept : List Turn) (budget : Nat) :
    carried k said kept budget <:+ candidates said kept := by
  simp only [carried]
  split
  · exact getLast?_suffix _
  · exact List.drop_suffix _ _

theorem users_suffix {a b : List Turn} (h : a <:+ b) : users a <:+ users b := by
  obtain ⟨p, rfl⟩ := h
  exact ⟨users p, by simp [users, List.filterMap_append]⟩

theorem users_append (ts : List Turn) (t : Turn) :
    users (ts ++ [t]) = users ts ++ t.said?.toList := by
  cases t <;> simp [users, List.filterMap_append, Turn.said?]

@[simp] theorem Turn.isBriefing_briefing (b : Briefing) : (Turn.briefing b).isBriefing = true := rfl

theorem compact_in_transcript (h : Harness) (s : Session) (hs : ∀ t ∈ s.conversation, t ∈ s.transcript) :
    ∀ t ∈ (compact h s).conversation, t ∈ (compact h s).transcript := by
  intro t ht
  simp only [compact, List.mem_cons] at ht ⊢
  rcases ht with rfl | ht
  · exact List.mem_append_right _ (List.mem_singleton_self _)
  · exact List.mem_append_left _ (hs t (List.mem_of_mem_drop ht))

theorem add_in_transcript (h : Harness) (s : Session) (t : Turn)
    (hs : ∀ u ∈ s.conversation, u ∈ s.transcript) :
    ∀ u ∈ (add h s t).conversation, u ∈ (add h s t).transcript := by
  intro u hu
  simp only [add] at hu ⊢
  split at hu <;> rename_i hf <;> simp only [hf, ite_true] <;>
    rw [List.mem_append] at hu ⊢ <;> rcases hu with hu | hu
  · exact Or.inl (compact_in_transcript h s hs u hu)
  · exact Or.inr hu
  · exact Or.inl (hs u hu)
  · exact Or.inr hu

theorem foldl_in_transcript (h : Harness) (ts : List Turn) (s : Session)
    (hs : ∀ u ∈ s.conversation, u ∈ s.transcript) :
    ∀ u ∈ (ts.foldl (add h) s).conversation, u ∈ (ts.foldl (add h) s).transcript := by
  induction ts generalizing s with
  | nil => exact hs
  | cons t ts ih => exact ih _ (add_in_transcript h s t hs)

theorem add_transcript (h : Harness) (s : Session) (t : Turn) (ht : t.isBriefing = false) :
    (add h s t).transcript.filter (!·.isBriefing) = s.transcript.filter (!·.isBriefing) ++ [t] := by
  simp only [add]
  split <;> simp [compact, List.filter_append, ht]

theorem foldl_transcript (h : Harness) (ts : List Turn) (s : Session)
    (hts : ∀ t ∈ ts, t.isBriefing = false) :
    (ts.foldl (add h) s).transcript.filter (!·.isBriefing) = s.transcript.filter (!·.isBriefing) ++ ts := by
  induction ts generalizing s with
  | nil => simp
  | cons t ts ih =>
    rw [List.foldl_cons, ih _ (fun u hu => hts u (List.mem_cons_of_mem _ hu)),
      add_transcript h s t (hts t (List.mem_cons_self ..))]
    simp

theorem add_users (h : Harness) (s : Session) (t : Turn) (hs : users s.conversation <:+ s.said) :
    users (add h s t).conversation <:+ (add h s t).said := by
  have hc : users (compact h s).conversation <:+ (compact h s).said := by
    simp only [compact, users, List.filterMap_cons, Turn.said?]
    exact (users_suffix (List.drop_suffix _ _)).trans hs
  have step : ∀ s' : Session, users s'.conversation <:+ s'.said →
      users (s'.conversation ++ [t]) <:+ s'.said ++ t.said?.toList := by
    intro s' ⟨p, hp⟩
    exact ⟨p, by rw [users_append, ← List.append_assoc, hp]⟩
  simp only [add]
  split
  · exact step _ hc
  · exact step _ hs

theorem foldl_users (h : Harness) (ts : List Turn) (s : Session)
    (hs : users s.conversation <:+ s.said) :
    users (ts.foldl (add h) s).conversation <:+ (ts.foldl (add h) s).said := by
  induction ts generalizing s with
  | nil => exact hs
  | cons t ts ih => exact ih _ (add_users h s t hs)

/-! ## Intent 1: Compaction frees up space in the context window so the model can continue working

Precisely: however long the conversation was, the compacted one takes at
most the briefing's fixed parts, the note's budget, `C/16` tokens of the
user's messages, and `C/8` tokens of tail. The bound does not depend on
the conversation before, so compacting again and again cannot make the
conversation grow.

The design allows two exceptions, each for the model to be able to go
on: the briefing carries the user's latest message whatever it takes,
and the tail keeps the last exchange whatever it takes. When they are
within their budgets, in a window of at least 8,192 tokens, the
compacted conversation leaves at least half the window before the next
compaction, less the briefing's fixed parts: the frame, the plan and the
transcript's path.
-/

/-- `compact_size` says **the compacted conversation is bounded,
whatever came before.** When the note is within its budget, it takes at
most the briefing's fixed parts, the note's budget, `C/16` tokens of the
user's messages or their latest message if it is larger, and `C/8`
tokens of tail or the last exchange if it is larger. -/
theorem compact_size (h : Harness) (s : Session)
    (hnote : h.tokenizer.count (h.write s.conversation) ≤ noteBudget h.capacity) :
    size h.tokenizer (compact h s).conversation
      ≤ h.tokenizer.frame + h.tokenizer.count h.plan + h.tokenizer.count h.transcript
        + noteBudget h.capacity
        + max (messageBudget h.capacity) (cost h.tokenizer s.said.getLast?.toList)
        + max (tailBudget h.capacity)
            (size h.tokenizer (s.conversation.drop (last s.conversation))) := by
  have hm := carried_fits h.tokenizer s.said
    (s.conversation.drop (tail h.tokenizer s.conversation (tailBudget h.capacity)))
    (messageBudget h.capacity)
  have ht := tail_fits h.tokenizer s.conversation (tailBudget h.capacity)
  simp only [compact, size_cons, Turn.tokens]
  generalize tail h.tokenizer s.conversation (tailBudget h.capacity) = start at hm ht ⊢
  rcases ht with ht | rfl <;> rcases hm with hm | ⟨-, hm⟩ <;> (try rw [hm]) <;> omega

/-- `compact_room` says **compaction leaves at least half the window to
work in.** In a window of at least 8,192 tokens, when the note, the
user's latest message and the last exchange are within their budgets,
the compacted conversation leaves at least half the window before the
high-water mark, less the briefing's fixed parts. -/
theorem compact_room (h : Harness) (s : Session) (hC : 8192 ≤ h.capacity)
    (hnote : h.tokenizer.count (h.write s.conversation) ≤ noteBudget h.capacity)
    (hlatest : cost h.tokenizer s.said.getLast?.toList ≤ messageBudget h.capacity)
    (hlast : size h.tokenizer (s.conversation.drop (last s.conversation)) ≤ tailBudget h.capacity) :
    size h.tokenizer (compact h s).conversation + h.capacity / 2
      ≤ highWater h.capacity
        + (h.tokenizer.frame + h.tokenizer.count h.plan + h.tokenizer.count h.transcript) := by
  have := compact_size h s hnote
  have hn : noteBudget h.capacity ≤ h.capacity / 32 := by
    unfold noteBudget; omega
  unfold highWater
  unfold messageBudget tailBudget at *
  omega

/-!
For dwim's smallest window, of 16,384 tokens, the bound is the
briefing's fixed parts and 3,584 tokens: 512 of note, 1,024 of the
user's messages and 2,048 of tail. The high-water mark is at 13,107, so
9,523 tokens are left for the work to go on, less the fixed parts: at
least half the window, as `compact_room` says. In the example session,
the 85 words of the conversation became 40.
-/

#guard noteBudget 16384 + messageBudget 16384 + tailBudget 16384 = 3584
#guard highWater 16384 - 3584 = 9523
#guard size words conversation = 85
#guard size words [compacted, rerun] = 40

/-! ## Intent 2: Compaction removes only information that can be retrieved and notes where to find it

Precisely: the transcript holds everything that was ever in the
conversation, so whatever compaction removes is in the transcript; and
the briefing names where the transcript is. Read the transcript and
leave out the briefings, and you have every turn of the session, in the
order it came.
-/

/-- `compact_transcript` says **compacting removes nothing from the
transcript.** It adds the briefing to it, the same briefing the
conversation now starts with. -/
theorem compact_transcript (h : Harness) (s : Session) :
    ∃ b, (compact h s).transcript = s.transcript ++ [.briefing b] ∧ (compact h s).conversation.head? = some (.briefing b) :=
  ⟨_, rfl, rfl⟩

/-- `compact_retrievable` says **what compaction removes is in the
transcript.** When everything in the conversation is in the transcript,
so is everything compaction removes, and everything it leaves. -/
theorem compact_retrievable (h : Harness) (s : Session)
    (hs : ∀ t ∈ s.conversation, t ∈ s.transcript) :
    (∀ t ∈ s.conversation, t ∈ (compact h s).transcript) ∧
      (∀ t ∈ (compact h s).conversation, t ∈ (compact h s).transcript) :=
  ⟨fun t ht => List.mem_append_left _ (hs t ht), compact_in_transcript h s hs⟩

/-- `compact_names_transcript` says **the briefing names where the
transcript is.** -/
theorem compact_names_transcript (h : Harness) (s : Session) :
    ∃ b kept, (compact h s).conversation = .briefing b :: kept ∧ b.transcript = h.transcript :=
  ⟨_, _, rfl, rfl⟩

/-- `play_in_transcript` says **everything in the conversation is in the
transcript, throughout a session.** So whatever a compaction removes can
be read there. -/
theorem play_in_transcript (h : Harness) (ts : List Turn) :
    ∀ u ∈ (play h ts).conversation, u ∈ (play h ts).transcript :=
  foldl_in_transcript h ts {} (by simp)

/-- `play_transcript` says **the transcript, with the briefings left
out, is every turn of the session, in the order it came.** The turns of
a session are the user's, the harness's prompts, the model's replies and
the tools' results; only compaction writes briefings. -/
theorem play_transcript (h : Harness) (ts : List Turn) (hts : ∀ t ∈ ts, t.isBriefing = false) :
    (play h ts).transcript.filter (!·.isBriefing) = ts := by
  simpa [play] using foldl_transcript h ts {} hts

/-!
In the example session, the user's first message is no longer in the
conversation, but it is in the transcript, and the briefing says where
the transcript is:
-/

#guard !session.conversation.contains fix && session.transcript.contains fix
#guard session.conversation.head? = some compacted
#guard (match compacted with | .briefing b => b.transcript | _ => "") = "~/.dwim/transcripts/session.jsonl"
#guard session.transcript.filter (!·.isBriefing) = conversation ++ [report]

/-! ## Intent 3: Compaction never changes the meaning of the information it keeps

Precisely: the compacted conversation is a briefing followed by the end
of the old conversation, turn for turn as it was; and the user's
messages in it, those the briefing carries and then those in the tail,
are messages the user sent, word for word, in the order the user sent
them.

What Lean cannot check is the note. It is the model's summary of what
was removed, and whether it keeps the meaning is the model's doing. The
design does two things about it: the briefing labels the note as the
model's own, apart from the user's messages and the plan; and, by intent
2, everything the note summarizes can be read in the transcript as it
was.
-/

/-- `compact_keeps` says **what compaction keeps stays as it was.** The
compacted conversation is a briefing followed by the end of the old
conversation, turn for turn. -/
theorem compact_keeps (h : Harness) (s : Session) :
    ∃ b n, (compact h s).conversation = .briefing b :: s.conversation.drop n :=
  ⟨_, _, rfl⟩

/-- `compact_in_order` says **the user's messages are kept word for
word, in the order sent.** When the user's messages in the conversation
are the last the user sent, the user's messages in the compacted
conversation, those the briefing carries and then those in the tail, are
messages the user sent, word for word, in the order sent. -/
theorem compact_in_order (h : Harness) (s : Session) (hs : users s.conversation <:+ s.said) :
    ∃ b n, (compact h s).conversation = .briefing b :: s.conversation.drop n ∧
      List.Sublist (b.messages ++ users (s.conversation.drop n)) s.said := by
  refine ⟨_, _, rfl, ?_⟩
  dsimp only
  have hk := (users_suffix (List.drop_suffix
    (tail h.tokenizer s.conversation (tailBudget h.capacity)) s.conversation)).trans hs
  generalize s.conversation.drop (tail h.tokenizer s.conversation (tailBudget h.capacity)) = kept
    at hk ⊢
  have hsplit : s.said.take (s.said.length - (users kept).length) ++ users kept = s.said := by
    obtain ⟨p, hp⟩ := hk
    rw [← hp]
    simp
  have := (carried_suffix h.tokenizer s.said kept (messageBudget h.capacity)).sublist.append
    (List.Sublist.refl (users kept))
  rwa [candidates, hsplit] at this

/-- `play_users` says **the user's messages in the conversation are the
last the user sent, throughout a session.** So `compact_in_order` holds
of every compaction. -/
theorem play_users (h : Harness) (ts : List Turn) :
    users (play h ts).conversation <:+ (play h ts).said :=
  foldl_users h ts {} (by simp [users])

/-!
In the example session, the tail is `rerun` as it was, and the briefing
carries the user's latest message in the user's own words, not the
note's:
-/

#guard session.conversation.tail = [rerun, report]
#guard (match compacted with | .briefing b => b.messages | _ => []) = [said.getLast!]
#guard said.getLast! = "Thanks. Now handle input that is only whitespace too"

end Dwim.Harness.Compaction
