//! Drafts copied from the context. A reply often repeats what the context
//! already holds: an edit retypes lines of a file the model has read, a
//! command names a path a tool printed, an answer quotes the question.
//! Where the last tokens of the context were written before, the tokens
//! that followed them then are a draft of what follows now, which costs
//! nothing to make: the model verifies it as it does its drafter's.

use std::collections::HashMap;

/// Fewest tokens at the end of the context that must have been written
/// before, in the same order, for what followed them to be copied. Fewer
/// match by chance too often, in the middle of something else; more wait
/// longer than need be, since a copy the reply does not keep costs less
/// than the drafter's tokens it is verified in place of.
const MATCH: usize = 4;

/// Most tokens back an earlier place is compared with the end of the
/// context. Of the places that match, the one that matches furthest back
/// is copied from, and the latest of those that match as far.
const DEPTH: usize = 32;

/// The tokens of a context from some position on, and where each is.
#[derive(Default)]
pub struct Copies {
    /// The position of the first token held: a context taken up from a
    /// saved state has tokens before it that were never seen.
    start: usize,
    tokens: Vec<u32>,
    /// Where each token is in `tokens`, in order.
    places: HashMap<u32, Vec<u32>>,
}

impl Copies {
    /// A context whose tokens before position `start` are not known.
    pub fn starting(start: usize) -> Self {
        Self {
            start,
            ..Self::default()
        }
    }

    /// Adds `tokens` to the end of the context.
    pub fn extend(&mut self, tokens: &[u32]) {
        for &token in tokens {
            self.places
                .entry(token)
                .or_default()
                .push(self.tokens.len() as u32);
            self.tokens.push(token);
        }
    }

    /// Forgets the tokens from position `len` on.
    pub fn truncate(&mut self, len: usize) {
        let len = len.saturating_sub(self.start);
        while self.tokens.len() > len {
            let token = self.tokens.pop().expect("longer than len");
            self.places
                .get_mut(&token)
                .expect("every token has a place")
                .pop();
        }
    }

    /// Up to `k` tokens to draft after the context: the ones that followed
    /// its last tokens where they were written before, and none if they
    /// were not.
    pub fn following(&self, k: usize) -> &[u32] {
        let tokens = &self.tokens;
        let Some(last) = tokens.len().checked_sub(1) else {
            return &[];
        };
        // How many tokens back from `place` are the ones back from the
        // end, up to `DEPTH`.
        let matched = |place: usize| {
            (1..DEPTH.min(place + 1))
                .find(|n| tokens[place - n] != tokens[last - n])
                .unwrap_or(DEPTH.min(place + 1))
        };
        let mut best = None;
        let mut depth = MATCH - 1;
        // The last token's places are in order, the end itself the last of
        // them.
        let places = &self.places[&tokens[last]];
        for &place in places[..places.len() - 1].iter().rev() {
            let n = matched(place as usize);
            if n > depth {
                (best, depth) = (Some(place as usize), n);
                if n == DEPTH {
                    break;
                }
            }
        }
        match best {
            Some(place) => &tokens[place + 1..tokens.len().min(place + 1 + k)],
            None => &[],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn copies(tokens: &[u32]) -> Copies {
        let mut copies = Copies::default();
        copies.extend(tokens);
        copies
    }

    #[test]
    fn copies_what_followed_the_last_tokens_before() {
        let line: Vec<u32> = (100..110).collect();
        let mut context = line.clone();
        context.extend([1, 2, 3, 4, 5, 6]);
        context.extend(&line);
        let copies = copies(&context);
        assert_eq!(copies.following(4), [1, 2, 3, 4]);
        assert_eq!(copies.following(8), [1, 2, 3, 4, 5, 6, 100, 101]);
    }

    #[test]
    fn copies_nothing_after_a_few_tokens_written_before() {
        let mut context: Vec<u32> = (100..110).collect();
        context.extend([1, 2, 3]);
        // Three tokens of the line: one short.
        context.extend(107..110);
        assert!(copies(&context).following(4).is_empty());
        assert!(copies(&[]).following(4).is_empty());
        assert!(copies(&[7]).following(4).is_empty());
    }

    #[test]
    fn copies_from_the_place_that_matches_furthest_back() {
        // Two places end with the same eight tokens: the earlier one has
        // the two before them as well, as the end of the context does.
        let tail: Vec<u32> = (100..108).collect();
        let mut context = vec![50, 51];
        context.extend(&tail);
        context.push(1);
        context.extend([60, 61]);
        context.extend(&tail);
        context.push(2);
        context.extend([50, 51]);
        context.extend(&tail);
        assert_eq!(copies(&context).following(1), [1]);
    }

    #[test]
    fn copies_from_the_latest_of_places_that_match_as_far() {
        let line: Vec<u32> = (100..140).collect();
        let mut context = line.clone();
        context.push(1);
        context.extend(&line);
        context.push(2);
        context.extend(&line);
        assert_eq!(copies(&context).following(1), [2]);
    }

    #[test]
    fn forgets_the_tokens_truncated() {
        let line: Vec<u32> = (100..110).collect();
        let mut copies = Copies::starting(1000);
        copies.extend(&line);
        copies.extend(&[1, 2]);
        copies.extend(&line);
        assert_eq!(copies.following(2), [1, 2]);
        // Back to the middle of the second line, and on with another
        // token: the line is not what the context ends with any more.
        copies.truncate(1000 + 17);
        copies.extend(&[9]);
        assert!(copies.following(2).is_empty());
        copies.truncate(1000 + 12);
        copies.extend(&line);
        assert_eq!(copies.following(2), [1, 2]);
    }
}
