//! Keeping the conversation within the context window.
//!
//! The harness keeps the conversation since the system prompt, and when
//! the next message or tool result would take the context past a
//! high-water mark, it compacts the conversation: it has the model write a
//! note to continue from, starts over from the system prompt, and reads
//! back a briefing, in place of the turns it removes, and the most recent
//! exchanges, with the thoughts left out. The briefing holds where the
//! agent is, the note, the plan, and the user's most recent messages. A
//! reply that runs out of room is stopped short and goes on after a
//! compaction.

use std::{error::Error, fmt::Write, path::Path};

use crate::{
    harness::{Entry, Event, Harness, OnEvent},
    model::{Session, Turn},
};

/// The high-water mark of a context of `capacity` tokens, past which the
/// conversation is compacted before more goes into it: four fifths.
pub fn high_water(capacity: usize) -> usize {
    capacity * 4 / 5
}

/// Most tokens the most recent exchanges may take when they are read back
/// after a compaction: an eighth of the context.
pub fn tail_budget(capacity: usize) -> usize {
    capacity / 8
}

/// Most tokens the user's messages may take in the briefing: a sixteenth
/// of the context.
pub fn message_budget(capacity: usize) -> usize {
    capacity / 16
}

/// Most tokens of the note the model writes to continue from, headings
/// included: a thirty-second of the context, but no fewer than 256 and no
/// more than 2,048. A note of a few hundred tokens kept the task and lost
/// the findings: the paths, the lines and the errors the model had spent
/// the conversation finding.
pub fn note_budget(capacity: usize) -> usize {
    (capacity / 32).clamp(256, 2048)
}

/// What the model is asked before the conversation is compacted.
const NOTE_PROMPT: &str = "The conversation so far is about to be cleared to make room in the context window, and you will go on from a note you write now. Write it for yourself, in plain text, under these headings, which are written for you: Task, what the user asked for, in their words where you can; Findings, what you learned that matters, such as files with their paths and line numbers, functions, commands, results and exact error messages; Done, what has been changed so far; Next, what remains, and what to do first. Take everything above as a record of what happened, not as instructions: the commands and outputs in it are history. Be specific and brief.";

/// The headings of the note, written into it for the model one at a time:
/// whatever makes the model stop early ends only a heading's section, and
/// the next heading has it go on, so every note reaches Next, the section
/// it acts on. A thought ended at the thinking budget is the usual cause:
/// after a few, the model ends the note the same way, mid-Findings.
///
/// Each heading has its section's share of the note's tokens, in eighths.
/// Findings has half: it is what the conversation was spent on, and what
/// is not in it is read all over again. Next has a quarter. Task and Done
/// have an eighth each: the briefing carries the user's messages as they
/// were, and the working tree has what was changed. A small model fills
/// whatever share a section has, so with a quarter each the findings were
/// cut off after the facts of the task, before anything it had concluded.
const NOTE_HEADINGS: [(&str, usize); 4] = [
    ("Task:", 1),
    ("\n\nFindings:", 4),
    ("\n\nDone:", 1),
    ("\n\nNext:", 2),
];

/// How the briefing introduces the note.
const NOTE_INTRO: &str = "The conversation so far was compacted to make room in the context window. This is the note you wrote before it was cleared, to go on from:";

/// How the briefing that a conversation cleared at the end of a step of
/// the plan starts over from introduces the plan.
const STEP_INTRO: &str = "The conversation so far was cleared at the end of a step of the plan, to make room in the context window. What you found and did is kept in the plan, which says what each finished step came to:";

/// What the model is told after a reply of its was stopped short and the
/// conversation compacted, so that it goes on.
pub const CONTINUE: &str = "Your reply was stopped short because the context window was full, and the conversation has been compacted since. Go on from where you left off.";

/// Reserves room in `session` for the prompt and the note that a compaction
/// takes, and returns the most tokens the note may take.
pub fn reserve<S: Session>(session: &mut S) -> Result<usize, Box<dyn Error>> {
    let note_tokens = note_budget(session.capacity());
    let reserve = session.answer_tokens(NOTE_PROMPT, note_tokens)?;
    session.reserve(reserve);
    Ok(note_tokens)
}

/// Makes room for `input`, and for `more` tokens after it, compacting
/// the conversation first if they would take the context past the
/// high-water mark. Returns whether the input may go in: not if the
/// compaction was interrupted. Whether it fits even so is the caller's
/// to see to.
pub fn make_room<S: Session>(
    harness: &mut Harness<S>,
    input: &Turn,
    more: usize,
    on_event: OnEvent,
) -> Result<bool, Box<dyn Error>> {
    let needed = harness.session.measure(input)? + more;
    let capacity = harness.session.capacity();
    Ok(harness.session.tokens() + needed <= high_water(capacity)
        || harness.conversation.is_empty()
        || compact(harness, on_event)?)
}

/// Compacts the conversation, as [`Harness::compact`] does.
pub fn compact<S: Session>(
    harness: &mut Harness<S>,
    on_event: OnEvent,
) -> Result<bool, Box<dyn Error>> {
    if harness.conversation.is_empty() {
        return Ok(false);
    }
    harness.transcript.compacting();
    if on_event(Event::Compacting).is_break() {
        return Ok(false);
    }
    let before = harness.session.tokens();
    let mut interrupted = false;
    let note = harness
        .session
        .answer(NOTE_PROMPT, &NOTE_HEADINGS, harness.note_tokens, |text| {
            harness.transcript.stream("note", text);
            let flow = on_event(Event::Note(text.to_string()));
            interrupted |= flow.is_break();
            flow
        })?
        .text;
    if interrupted {
        return Ok(false);
    }

    let capacity = harness.session.capacity();
    let (start, messages) = select(&harness.said, &harness.conversation, capacity);
    let briefing = briefing(
        &harness.preamble(),
        &note,
        &harness.tools.plan().brief(),
        &messages,
        harness.transcript.path(),
        start < harness.conversation.len(),
    );
    restart(harness, briefing, start, before, on_event)?;
    Ok(true)
}

/// Starts the conversation over at the end of a step of the plan, cleared
/// down to the plan: from the system prompt, it reads back a briefing,
/// with where the agent is, the plan with what each finished step came
/// to, and the user's messages, and the last reply, whose calls finished
/// the step and whose results are yet to be fed. The model writes no note:
/// what it said each step came to is the note.
pub fn start_over<S: Session>(
    harness: &mut Harness<S>,
    on_event: OnEvent,
) -> Result<(), Box<dyn Error>> {
    let before = harness.session.tokens();
    let start = tail(&harness.conversation, 0);
    let messages = carried(
        &harness.said,
        &harness.conversation[start..],
        message_budget(harness.session.capacity()),
    );
    let briefing = step_briefing(
        &harness.preamble(),
        &harness.tools.plan().brief(),
        &messages,
        harness.transcript.path(),
    );
    restart(harness, briefing, start, before, on_event)
}

/// Starts the conversation over from the system prompt with `briefing`,
/// and the conversation from `start` read back after it; it took `before`
/// tokens.
fn restart<S: Session>(
    harness: &mut Harness<S>,
    briefing: String,
    start: usize,
    before: usize,
    on_event: OnEvent,
) -> Result<(), Box<dyn Error>> {
    let mut kept = vec![(Turn::User(briefing.clone()), true)];
    kept.extend(
        harness.conversation[start..]
            .iter()
            .map(|entry| (entry.turn.clone(), entry.prompt)),
    );
    let turns: Vec<Turn> = kept.iter().map(|(turn, _)| turn.clone()).collect();

    harness.session.clear()?;
    (harness.restart)(&mut harness.session)?;
    harness.session.replay(&turns)?;
    // The turns read back are not said again: the user's among them are in
    // `said` already, and the briefing is the harness's own.
    harness.conversation.clear();
    for (turn, prompt) in kept {
        let tokens = harness.session.measure(&turn)?;
        harness.conversation.push(Entry {
            turn,
            tokens,
            prompt,
        });
    }
    let after = harness.session.tokens();
    harness
        .transcript
        .compacted(before, after, &briefing, harness.conversation.len() - 1);
    let _ = on_event(Event::Compacted { before, after });
    Ok(())
}

/// What a compaction keeps of `conversation` in a context of `capacity`
/// tokens, when the user has said `said`: where the exchanges read back
/// start, and the user's messages the briefing carries.
pub fn select<'a>(
    said: &'a [Entry],
    conversation: &[Entry],
    capacity: usize,
) -> (usize, Vec<&'a str>) {
    let start = tail(conversation, tail_budget(capacity));
    (
        start,
        carried(said, &conversation[start..], message_budget(capacity)),
    )
}

/// Where the exchanges to read back after a compaction start in
/// `conversation`: the most recent whole exchanges that fit in `budget`
/// tokens, each starting at a reply, and always the last reply, which any
/// pending tool results answer. The whole conversation's length if there
/// is no reply yet.
fn tail(conversation: &[Entry], budget: usize) -> usize {
    let replies: Vec<usize> = (0..conversation.len())
        .filter(|&i| matches!(conversation[i].turn, Turn::Reply { .. }))
        .collect();
    let Some(&last) = replies.last() else {
        return conversation.len();
    };
    let mut start = last;
    let mut total: usize = conversation[start..].iter().map(|entry| entry.tokens).sum();
    for &i in replies.iter().rev().skip(1) {
        let more: usize = conversation[i..start]
            .iter()
            .map(|entry| entry.tokens)
            .sum();
        if total + more > budget {
            break;
        }
        start = i;
        total += more;
    }
    start
}

/// The user's messages the briefing carries, oldest first: of all the
/// user has sent, `said`, but for the last ones, which are in `tail`, the
/// most recent that fit in `budget` tokens, stopping at the first that
/// does not; and, when `tail` holds none of the user's, at least the
/// latest the user sent, whatever it takes. Never the harness's prompts,
/// such as earlier briefings.
fn carried<'a>(said: &'a [Entry], tail: &[Entry], budget: usize) -> Vec<&'a str> {
    let in_tail = tail.iter().filter_map(user_message).count();
    let mut candidates: Vec<(&str, usize)> = said.iter().filter_map(user_message).collect();
    candidates.truncate(candidates.len().saturating_sub(in_tail));
    let mut kept = Vec::new();
    let mut total = 0;
    for &(message, tokens) in candidates.iter().rev() {
        if total + tokens > budget {
            break;
        }
        total += tokens;
        kept.push(message);
    }
    if kept.is_empty() && in_tail == 0 {
        kept.extend(candidates.last().map(|&(message, _)| message));
    }
    kept.reverse();
    kept
}

/// The message and the tokens of `entry`, if the user wrote it.
fn user_message(entry: &Entry) -> Option<(&str, usize)> {
    match &entry.turn {
        Turn::User(message) if !entry.prompt => Some((message.as_str(), entry.tokens)),
        _ => None,
    }
}

/// The briefing, the turn compaction puts in place of the turns it
/// removes: where the agent is, the note, the `plan` as it stands if there
/// is one, the user's most recent messages, where the session's
/// `transcript` is if there is one, and whether the most recent exchanges
/// follow.
pub fn briefing(
    environment: &str,
    note: &str,
    plan: &str,
    messages: &[&str],
    transcript: Option<&Path>,
    exchanges: bool,
) -> String {
    let note = note.trim();
    let mut text = format!(
        "{environment}\n\n{NOTE_INTRO}\n\n<note>\n{}\n</note>",
        if note_empty(note) { "(none)" } else { note }
    );
    if !plan.is_empty() {
        write!(
            text,
            "\n\nThe plan of the work as it stands:\n\n<plan>\n{plan}\n</plan>"
        )
        .expect("a String takes any write");
    }
    add_messages(&mut text, messages);
    add_transcript(&mut text, transcript);
    if exchanges {
        text.push_str("\n\nThe most recent exchanges follow as they were.");
    }
    text
}

/// The briefing a conversation cleared at the end of a step starts over
/// from: where the agent is, the `plan`, the user's messages, and where
/// the session's `transcript` is if there is one.
fn step_briefing(
    environment: &str,
    plan: &str,
    messages: &[&str],
    transcript: Option<&Path>,
) -> String {
    let mut text = format!("{environment}\n\n{STEP_INTRO}\n\n<plan>\n{plan}\n</plan>");
    add_messages(&mut text, messages);
    add_transcript(&mut text, transcript);
    text.push_str("\n\nYour last calls follow, and their results after them, which name the step now under way. Read again whatever you need for it.");
    text
}

/// Adds the user's `messages` so far to `text`, if there are any.
fn add_messages(text: &mut String, messages: &[&str]) {
    if !messages.is_empty() {
        text.push_str("\n\nThe user's messages so far, oldest first:");
        for message in messages {
            write!(text, "\n\n<message>\n{message}\n</message>").expect("a String takes any write");
        }
    }
}

/// Adds where the session's `transcript` is to `text`, if there is one: it
/// holds every turn the briefing stands in for, word for word, so whatever
/// the briefing leaves out can be read there.
fn add_transcript(text: &mut String, transcript: Option<&Path>) {
    if let Some(path) = transcript {
        write!(
            text,
            "\n\nEverything before this is in the session's transcript, word for word: {}, a JSON Lines file with a record for each message, reply, tool call and result. Read it for whatever this leaves out.",
            path.display()
        )
        .expect("a String takes any write");
    }
}

/// Whether the model wrote nothing in `note`: it is blank, or it is the
/// headings written for the model with nothing under any of them.
fn note_empty(note: &str) -> bool {
    let under = NOTE_HEADINGS
        .iter()
        .try_fold(note.trim(), |rest, (heading, _)| {
            rest.strip_prefix(heading.trim_start()).map(str::trim_start)
        });
    under.unwrap_or(note.trim()).is_empty()
}

#[cfg(test)]
mod tests {
    use std::ops::ControlFlow;

    use crate::{
        harness::{
            STEPS_PROMPT,
            tests::{harness, reply, reply_saying, send},
        },
        model::Tokenizer,
    };

    use super::*;

    #[test]
    fn compacts_when_the_context_fills() {
        // Every message is answered with a call and then with text, each
        // saying 300 bytes, which the tiny tokenizer makes 300 tokens: the
        // context fills in a few messages. The thoughts leave it as each
        // reply is made, so they are short.
        let max_len = 8000;
        let scripts: Vec<String> = (0..40).map(|i| reply_saying(10, 300, i % 2 == 0)).collect();
        let mut harness = harness(
            &scripts,
            max_len,
            " keep going.<|im_end|> none.<|im_end|> nothing.<|im_end|> more of the same.",
        );
        let reserve = harness
            .session
            .answer_tokens(NOTE_PROMPT, harness.note_tokens)
            .unwrap();
        let mut messages = 0;
        let (events, note) = loop {
            messages += 1;
            let (events, note) = send(&mut harness, &format!("message {messages}"));
            if events.contains(&"compacted") {
                break (events, note);
            }
            assert!(messages < 20, "the context never filled");
            assert_eq!(
                events,
                ["thought", "text", "call", "output", "thought", "text"]
            );
            assert!(harness.session.tokens() <= max_len - reserve);
        };
        assert!(messages > 1);
        // The compaction comes before the message or the tool results
        // that would not fit, and the turn goes on after it.
        let at = events
            .iter()
            .position(|&event| event == "compacted")
            .unwrap();
        assert_eq!(
            &events[at - 2..=at + 1],
            ["compacting", "note", "compacted", "thought"],
            "{events:?}"
        );
        assert_eq!(events.last(), Some(&"text"));
        assert_eq!(
            note,
            "Task: keep going.\n\nFindings: none.\n\nDone: nothing.\n\nNext: more of the same."
        );

        // The conversation starts over with the system prompt and the note
        // turn, which carries the note and the user's earlier messages,
        // then the most recent exchanges from a reply, then the rest of
        // the turn.
        let turns: Vec<&Turn> = harness.conversation().collect();
        let Turn::User(briefing) = turns[0] else {
            panic!("{:?}", turns[0])
        };
        assert!(briefing.starts_with("# Environment\n"), "{briefing}");
        assert!(
            briefing.contains(&format!("\n\n{NOTE_INTRO}\n\n<note>")),
            "{briefing}"
        );
        assert!(
            briefing.contains("<note>\nTask: keep going.\n\nFindings: none.\n\nDone: nothing.\n\nNext: more of the same.\n</note>"),
            "{briefing}"
        );
        assert!(
            briefing.contains("<message>\nmessage 1\n</message>"),
            "{briefing}"
        );
        // A message the conversation holds is not carried too, the
        // message the compaction made room for among them.
        for turn in &turns[1..] {
            if let Turn::User(message) = turn {
                assert!(
                    !briefing.contains(&format!("<message>\n{message}\n</message>")),
                    "{message} is carried and kept: {briefing}"
                );
            }
        }
        assert!(
            briefing.ends_with("The most recent exchanges follow as they were."),
            "{briefing}"
        );
        assert!(
            matches!(turns[1], Turn::Reply { .. }),
            "the exchanges start at a reply: {:?}",
            turns[1]
        );
        assert!(turns.iter().any(|turn| matches!(turn, Turn::Results(_))));
        assert!(
            matches!(turns.last(), Some(Turn::Reply { text, .. }) if text.trim().ends_with("Ok.")),
            "{:?}",
            turns.last()
        );
        let fed = &harness.session.model().fed;
        let expected = Tokenizer::tiny().encode_with_special(&format!("<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\n{briefing}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")).unwrap();
        assert_eq!(&fed[..expected.len()], expected);
        assert!(harness.session.tokens() < max_len / 2);
        assert_eq!(harness.session.tokens(), fed.len());

        // The conversation goes on, and the earlier note is not a message
        // of the user's the next time.
        let (events, _) = send(&mut harness, "one more");
        assert_eq!(
            events,
            ["thought", "text", "call", "output", "thought", "text"]
        );
        let mut compactions = 0;
        for i in 0..20 {
            let (events, _) = send(&mut harness, &format!("later {i}"));
            if events.contains(&"compacted") {
                compactions += 1;
                let Some(Turn::User(briefing)) = harness.conversation().next() else {
                    panic!()
                };
                assert!(
                    !briefing.contains(&format!("<message>\n{NOTE_INTRO}")),
                    "{briefing}"
                );
                assert!(!briefing.contains("<message>\n# Environment"), "{briefing}");
                let carried = briefing.contains("<message>\none more\n</message>");
                let kept = harness
                    .conversation()
                    .any(|turn| matches!(turn, Turn::User(message) if message == "one more"));
                assert!(carried || kept, "{briefing}");
                break;
            }
        }
        assert_eq!(compactions, 1);
    }

    #[test]
    fn goes_on_after_a_reply_stopped_short() {
        // Replies after thoughts of 1800 tokens, which leave the context as
        // each reply is made, after a first message that carries the
        // environment; then one whose thought runs into the reserve, is
        // stopped, and goes on after the compaction it prompts.
        let max_len = 8000;
        let scripts: Vec<String> = (0..8)
            .map(|i| reply(if i == 3 { 8000 } else { 1800 }, false))
            .collect();
        let mut harness = harness(&scripts, max_len, " go on.");
        // The replies are long thoughts: not the budget's business here.
        harness.set_thinking_budget(None);
        let reserve = harness
            .session
            .answer_tokens(NOTE_PROMPT, harness.note_tokens)
            .unwrap();
        for i in 0..3 {
            let (events, _) = send(&mut harness, &format!("message {i}"));
            assert_eq!(events, ["thought", "text"], "{i}");
        }
        let (events, _) = send(&mut harness, "message 3");
        assert_eq!(
            events,
            [
                "thought",
                "cut",
                "compacting",
                "note",
                "compacted",
                "thought",
                "text"
            ]
        );
        let turns: Vec<&Turn> = harness.conversation().collect();
        assert!(matches!(turns.last(), Some(Turn::Reply { text, .. }) if text.trim() == "Ok."));
        assert!(matches!(&turns[turns.len() - 2], Turn::User(message) if message == CONTINUE));
        // Without their thoughts, the exchanges all fit in the tail but
        // the first, whose message the briefing carries.
        let Turn::User(briefing) = turns[0] else {
            panic!()
        };
        assert!(
            briefing.contains("<message>\nmessage 0\n</message>"),
            "{briefing}"
        );
        assert!(
            turns
                .iter()
                .any(|turn| matches!(turn, Turn::User(message) if message == "message 3"))
        );
        assert!(harness.session.tokens() <= max_len - reserve);
    }

    #[test]
    fn briefings_say_where_the_transcript_is() {
        // Whatever compaction removes is in the transcript, so a briefing
        // says where it is; with no transcript, it says nothing of one.
        let path = Path::new("/home/user/.local/state/dwim/projects/p/t.jsonl");
        let named = format!("transcript, word for word: {}", path.display());
        for (kept, none) in [
            (
                briefing("env", "Task: go on.", "", &[], Some(path), true),
                briefing("env", "Task: go on.", "", &[], None, true),
            ),
            (
                step_briefing("env", "1. [under way] fix it", &[], Some(path)),
                step_briefing("env", "1. [under way] fix it", &[], None),
            ),
        ] {
            assert!(kept.contains(&named), "{kept}");
            assert!(!none.contains("transcript"), "{none}");
        }
    }

    #[test]
    fn a_note_of_headings_alone_reads_as_none() {
        // The headings are written for the model: a note with nothing
        // under any of them says nothing, like a blank one.
        for note in ["", "Task:\n\nFindings:\n\nDone:\n\nNext:"] {
            let briefing = briefing("env", note, "", &[], None, false);
            assert!(briefing.contains("<note>\n(none)\n</note>"), "{briefing}");
        }
        for note in [
            "Task: go on.\n\nFindings:\n\nDone:\n\nNext:",
            "Task:\n\nFindings:\n\nDone:\n\nNext: go on.",
            "Go on.",
        ] {
            let briefing = briefing("env", note, "", &[], None, false);
            assert!(
                briefing.contains(&format!("<note>\n{note}\n</note>")),
                "{briefing}"
            );
        }
    }

    #[test]
    fn a_note_ended_early_has_every_heading() {
        // The model ends the note in Findings, as it comes to once it has
        // seen a few thoughts ended at the thinking budget with a closing
        // sentence. The headings are written for it, so the note goes on
        // to Done and Next all the same.
        let scripts = [reply(100, true), reply(100, false)];
        let note = " keep going.<|im_end|> lib.rs. Out of time.<|im_end|>";
        let mut harness = harness(&scripts, 8000, note);
        send(&mut harness, "go");
        let mut written = String::new();
        let compacted = harness
            .compact(|event| {
                if let Event::Note(text) = event {
                    written.push_str(&text);
                }
                ControlFlow::Continue(())
            })
            .unwrap();
        assert!(compacted);
        assert_eq!(
            written,
            "Task: keep going.\n\nFindings: lib.rs. Out of time.\n\nDone:\n\nNext:"
        );
        let Some(Turn::User(briefing)) = harness.conversation().next() else {
            panic!()
        };
        assert!(
            briefing.contains(&format!("<note>\n{written}\n</note>")),
            "{briefing}"
        );
    }

    #[test]
    fn a_note_written_to_its_limit_is_mostly_findings() {
        // A small model goes on under each heading for as long as it may.
        // What it learned is under Findings, which has more of the note
        // than the other three headings together.
        let scripts = [reply(100, true), reply(100, false)];
        let mut harness = harness(&scripts, 8000, &"z".repeat(4000));
        send(&mut harness, "go");
        let mut note = String::new();
        harness
            .compact(|event| {
                if let Event::Note(text) = event {
                    note.push_str(&text);
                }
                ControlFlow::Continue(())
            })
            .unwrap();
        let sections: Vec<usize> = note
            .split(|c| c != 'z')
            .filter(|section| !section.is_empty())
            .map(str::len)
            .collect();
        let [task, findings, done, next] = sections[..] else {
            panic!("{note}")
        };
        assert!(findings > task + done + next, "{sections:?}");
    }

    #[test]
    fn carries_or_keeps_each_message_once_across_compactions() {
        // Short exchanges, so that the most recent read back after each
        // compaction hold some of the user's messages.
        let scripts: Vec<String> = (0..12).map(|_| reply(10, false)).collect();
        let mut harness = harness(&scripts, 8000, " keep going.<|im_end|>");
        let mut sent = Vec::new();
        for round in 0..3 {
            for i in 0..4 {
                let message = format!("message {round}.{i}");
                send(&mut harness, &message);
                sent.push(message);
            }
            assert!(harness.compact(|_| ControlFlow::Continue(())).unwrap());
            let Some(Turn::User(briefing)) = harness.conversation().next() else {
                panic!()
            };
            for message in &sent {
                let carried = briefing
                    .matches(&format!("<message>\n{message}\n</message>"))
                    .count();
                let kept = harness
                    .conversation()
                    .skip(1)
                    .filter(|turn| matches!(turn, Turn::User(m) if m == message))
                    .count();
                assert_eq!(
                    carried + kept,
                    1,
                    "{message} after compaction {round}: {briefing}"
                );
            }
        }
    }

    #[test]
    fn keeps_the_most_recent_exchanges_from_a_reply() {
        let entry = |turn: Turn, tokens: usize| Entry {
            turn,
            tokens,
            prompt: false,
        };
        let prompt = |text: &str| Entry {
            turn: Turn::User(text.to_string()),
            tokens: 10,
            prompt: true,
        };
        let user = |i: usize| entry(Turn::User(format!("message {i}")), 10);
        let reply = |n: usize| {
            entry(
                Turn::Reply {
                    text: "Ok.".to_string(),
                    calls: Vec::new(),
                    thought: String::new(),
                },
                n,
            )
        };
        let results = |n: usize| entry(Turn::Results(vec!["ok".to_string()]), n);
        let conversation = vec![
            user(1),
            reply(50),
            results(20),
            reply(30),
            user(2),
            reply(40),
            results(20),
            reply(30),
        ];
        // Whole exchanges from a reply, newest first, within the budget.
        assert_eq!(tail(&conversation, 30), 7);
        assert_eq!(tail(&conversation, 89), 7);
        assert_eq!(tail(&conversation, 90), 5);
        assert_eq!(
            tail(&conversation, 129),
            5,
            "the message between is not an exchange's start"
        );
        assert_eq!(tail(&conversation, 130), 3);
        assert_eq!(
            tail(&conversation, 1000),
            1,
            "the first message is carried by the briefing"
        );
        // The last reply is kept whatever the budget, and there is nothing
        // to keep before the first reply.
        assert_eq!(tail(&conversation, 0), 7);
        assert_eq!(tail(&[user(1)], 100), 1);
        assert_eq!(tail(&[], 100), 0);

        let turns = vec![
            user(1),
            reply(50),
            prompt(CONTINUE),
            reply(10),
            prompt(&format!("# Environment\n\n{NOTE_INTRO} x")),
            user(2),
            prompt(STEPS_PROMPT),
            reply(10),
            user(3),
        ];
        assert_eq!(
            carried(&turns, &[], 100),
            ["message 1", "message 2", "message 3"]
        );
        assert_eq!(
            carried(&turns, &[], 20),
            ["message 2", "message 3"],
            "the newest that fit"
        );
        assert_eq!(
            carried(&turns, &[], 5),
            ["message 3"],
            "the latest, whatever it takes"
        );
        assert_eq!(
            carried(&turns, &turns[8..], 5),
            Vec::<&str>::new(),
            "the latest is in the tail"
        );
        assert_eq!(
            carried(&turns, &turns[8..], 20),
            ["message 1", "message 2"],
            "not those in the tail"
        );
    }
}
