//! A model and a tokenizer for tests, here and in the crates around this
//! one: the tokenizer over a tiny vocabulary, and a model that says what
//! it is told to.

use std::collections::{HashMap, VecDeque};

use crate::models::{LanguageModel, Result, Tokenizer, chat};

/// A model that says what it is told to: it records every token fed to it,
/// and each time the chat template opens a reply for it, it says the next
/// of its scripts, then ends the reply. Once its scripts run out, it ends
/// every reply at once. Its state is the tokens fed, which it saves and
/// restores.
pub struct Scripted {
    pub fed: Vec<u32>,
    scripts: VecDeque<Vec<u32>>,
    /// What it says when asked for a plain answer, if not the next script.
    answer: Option<Vec<u32>>,
    saying: VecDeque<u32>,
    /// Whether the last batch ended with the template's opening of a
    /// reply, and with the thought closed, so the next sampled token
    /// starts one.
    pending: Option<bool>,
    /// Where the mark is, in tokens fed, if one has been made.
    mark: Option<usize>,
    /// The sequence it is running, and the others' tokens fed, replies
    /// under way, and marks, by number.
    sequence: usize,
    parked: HashMap<usize, Parked>,
    vocab: usize,
    /// How the chat template opens a reply, with the thought open or with
    /// an empty one when the reply is to be made without thinking.
    opening: Vec<u32>,
    unthinking: Vec<u32>,
    end: u32,
    max_len: usize,
}

impl Scripted {
    /// A model over `tokenizer`'s vocabulary that says `scripts` in turn,
    /// each given as text with the chat format's tags in it, and has room
    /// for `max_len` tokens.
    pub fn new(tokenizer: &Tokenizer, scripts: &[&str], max_len: usize) -> Self {
        Self {
            fed: Vec::new(),
            scripts: scripts
                .iter()
                .map(|script| tokenizer.encode_with_special(script).unwrap())
                .collect(),
            answer: None,
            saying: VecDeque::new(),
            pending: None,
            mark: None,
            sequence: 0,
            parked: HashMap::new(),
            vocab: tokenizer.vocab_size(),
            opening: chat::opening(tokenizer, true).unwrap(),
            unthinking: chat::opening(tokenizer, false).unwrap(),
            end: tokenizer.special(chat::IM_END).unwrap(),
            max_len,
        }
    }

    /// Has the model say `text` whenever it is asked for a plain answer,
    /// rather than its next script. The text may have the chat format's
    /// tags in it: an `<|im_end|>` in the middle ends the answer there, and
    /// what follows is said if the answer is pulled on with a head.
    pub fn answers(mut self, tokenizer: &Tokenizer, text: &str) -> Self {
        self.answer = Some(tokenizer.encode_with_special(text).unwrap());
        self
    }

    /// What the next reply says: the answer, if there is one and the reply
    /// is to be made without thinking, and otherwise the next script,
    /// which it takes.
    fn script(&mut self, unthinking: bool) -> Vec<u32> {
        match (&self.answer, unthinking) {
            (Some(answer), true) => answer.clone(),
            _ => self.scripts.pop_front().unwrap_or_default(),
        }
    }

    /// The first token of what the next reply says, leaving it in place.
    fn peek(&self, unthinking: bool) -> Option<u32> {
        match (&self.answer, unthinking) {
            (Some(answer), true) => answer.first().copied(),
            _ => self.scripts.front()?.first().copied(),
        }
    }
}

/// A sequence the model is not running, as it was left.
type Parked = (Vec<u32>, VecDeque<u32>, Option<bool>, Option<usize>);

impl LanguageModel for Scripted {
    fn forward(&mut self, tokens: &[u32], pos: usize) -> Vec<f32> {
        assert_eq!(
            pos,
            self.fed.len(),
            "the conversation must only append tokens"
        );
        assert!(
            pos + tokens.len() <= self.max_len,
            "tokens past the end of the context"
        );
        self.fed.extend(tokens);
        // A reply's prompt ends with the template's opening of one, and
        // the reply is then sampled a token at a time. The script is taken
        // up only once the sampling starts: a prompt read in batches may
        // end a batch at an opening it goes on past, and the chat may
        // write tokens into a reply it has just opened, such as a head,
        // for the script to be said from. Either way, a batch that does
        // not end at an opening leaves the reply as it was: opened and
        // yet to be said, or under way.
        if tokens.len() > 1 {
            if self.fed.ends_with(&self.unthinking) {
                self.pending = Some(true);
                self.saying.clear();
            } else if self.fed.ends_with(&self.opening) {
                self.pending = Some(false);
                self.saying.clear();
            }
        } else {
            if let Some(unthinking) = self.pending.take() {
                self.saying = self.script(unthinking).into();
            }
            // The token the logits said is said once it is fed. A token
            // written in instead, such as a head of one token, leaves it
            // to be said after.
            if self.saying.front() == Some(&tokens[0]) {
                self.saying.pop_front();
            }
        }
        let next = match self.pending {
            Some(unthinking) => self.peek(unthinking).unwrap_or(self.end),
            None => {
                let next = self.saying.front().copied().unwrap_or(self.end);
                // The chat ends a reply, or a section of an answer, at
                // `<|im_end|>` without feeding it: it is said now.
                if next == self.end {
                    self.saying.pop_front();
                }
                next
            }
        };
        let mut logits = vec![0.0; self.vocab];
        logits[next as usize] = 1.0;
        logits
    }

    /// The tokens fed so far, as bytes.
    fn save(&self, len: usize) -> Option<Vec<u8>> {
        Some(
            self.fed[..len]
                .iter()
                .flat_map(|token| token.to_le_bytes())
                .collect(),
        )
    }

    fn restore(&mut self, state: &[u8]) -> Result<usize> {
        self.reset();
        self.fed = state
            .chunks_exact(4)
            .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        Ok(self.fed.len())
    }

    fn mark(&mut self) -> bool {
        self.mark = Some(self.fed.len());
        true
    }

    /// Forgets the tokens fed since the mark.
    fn rewind_to_mark(&mut self) -> Result<()> {
        let mark = self.mark.ok_or("no mark to rewind to")?;
        self.fed.truncate(mark);
        self.saying.clear();
        self.pending = None;
        Ok(())
    }

    fn reset(&mut self) {
        self.fed.clear();
        self.saying.clear();
        self.pending = None;
        self.mark = None;
    }

    fn max_len(&self) -> usize {
        self.max_len
    }

    fn add_sequence(&mut self) -> Result<usize> {
        let id = self.parked.len() + 1;
        self.parked.insert(id, Default::default());
        Ok(id)
    }

    /// Parks the sequence it is running, and takes up sequence `id`: its
    /// scripts are shared, said in turn whichever sequence asks.
    fn select_sequence(&mut self, id: usize) -> Result<()> {
        if id == self.sequence {
            return Ok(());
        }
        let (fed, saying, pending, mark) = self
            .parked
            .remove(&id)
            .ok_or_else(|| format!("no sequence {id}"))?;
        let old = (
            std::mem::replace(&mut self.fed, fed),
            std::mem::replace(&mut self.saying, saying),
            std::mem::replace(&mut self.pending, pending),
            std::mem::replace(&mut self.mark, mark),
        );
        self.parked.insert(self.sequence, old);
        self.sequence = id;
        Ok(())
    }
}
