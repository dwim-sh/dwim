//! Language models: Qwen3.8-27B, its tokenizer and weights, and a chat
//! around it.

mod chat;
pub mod gguf;
pub mod hub;
pub mod quants;
pub mod qwen35;
mod sampler;
#[cfg(test)]
pub mod testing;
mod tokenizer;

pub use chat::{Answer, Chat, Chunk, DRAFTS, Format, Reply, Stats, Tally, Turn};
pub use gguf::Gguf;
pub use sampler::Sampler;
pub use tokenizer::Tokenizer;

use std::ops::ControlFlow;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// A model that predicts the next token, keeping the sequence so far in
/// its own state.
pub trait LanguageModel {
    /// The model's state after the first `len` positions, as bytes for
    /// [`restore`](Self::restore) to take up from, if the model can give
    /// it.
    fn save(&self, _len: usize) -> Option<Vec<u8>> {
        None
    }

    /// Puts the model in a state [`save`](Self::save) gave, and returns
    /// how many positions it holds.
    fn restore(&mut self, _state: &[u8]) -> Result<usize> {
        Err("this model cannot restore a state".into())
    }

    /// Runs `tokens`, the first at position `pos`, through the model, and
    /// returns the logits for the token that follows the last of them.
    fn forward(&mut self, tokens: &[u32], pos: usize) -> Vec<f32>;

    /// Drafts up to `k` tokens to follow `token`, which was picked at
    /// position `pos` from the logits the latest run gave: the most likely
    /// token after each by the drafter's reckoning, if the model has a
    /// drafter and nothing is being verified, and none otherwise. The
    /// token and the drafts are then run as one batch by
    /// [`verify`](Self::verify).
    fn draft(&mut self, _token: u32, _pos: usize, _k: usize) -> Vec<u32> {
        Vec::new()
    }

    /// Runs `tokens`, the first at position `pos`, as a batch to be
    /// verified: a token picked and the tokens drafted after it. Returns
    /// the logits after each, a row per token, and commits none of them
    /// until [`rewind`](Self::rewind) says how many to keep. Only for a
    /// model that drafted them.
    fn verify(&mut self, _tokens: &[u32], _pos: usize) -> Vec<f32> {
        unreachable!("the model does not draft")
    }

    /// Commits the first `keep` tokens of the batch being verified, one at
    /// least, and drops the rest.
    fn rewind(&mut self, _keep: usize) {
        unreachable!("the model does not draft")
    }

    /// Marks the sequence so far, keeping what
    /// [`rewind_to_mark`](Self::rewind_to_mark) needs to take it up again
    /// from here, if the model can: returns whether it did. A mark is kept
    /// until the next, or until the model is reset or restored.
    fn mark(&mut self) -> bool {
        false
    }

    /// Puts the model back at the mark, forgetting the positions after
    /// it: the next tokens start where the mark was made.
    fn rewind_to_mark(&mut self) -> Result<()> {
        Err("this model cannot rewind to a mark".into())
    }

    /// Forgets the sequence so far, so that the next tokens start at
    /// position 0.
    fn reset(&mut self);

    /// Longest sequence the state has room for.
    fn max_len(&self) -> usize;

    /// Allocates another sequence, with state of its own and room for as
    /// many tokens as the first, beside the model's others, which share its
    /// weights, and returns its number. A model starts with one sequence,
    /// number 0, which it runs until another is selected.
    fn add_sequence(&mut self) -> Result<usize> {
        Err("this model runs one sequence".into())
    }

    /// Has the model run sequence `id` from here on: every other method
    /// is of the sequence selected. Fails while a batch is being verified.
    fn select_sequence(&mut self, id: usize) -> Result<()> {
        if id == 0 {
            Ok(())
        } else {
            Err(format!("no sequence {id}").into())
        }
    }
}

impl<M: LanguageModel + ?Sized> LanguageModel for Box<M> {
    fn save(&self, len: usize) -> Option<Vec<u8>> {
        (**self).save(len)
    }

    fn restore(&mut self, state: &[u8]) -> Result<usize> {
        (**self).restore(state)
    }

    fn forward(&mut self, tokens: &[u32], pos: usize) -> Vec<f32> {
        (**self).forward(tokens, pos)
    }

    fn draft(&mut self, token: u32, pos: usize, k: usize) -> Vec<u32> {
        (**self).draft(token, pos, k)
    }

    fn verify(&mut self, tokens: &[u32], pos: usize) -> Vec<f32> {
        (**self).verify(tokens, pos)
    }

    fn rewind(&mut self, keep: usize) {
        (**self).rewind(keep)
    }

    fn mark(&mut self) -> bool {
        (**self).mark()
    }

    fn rewind_to_mark(&mut self) -> Result<()> {
        (**self).rewind_to_mark()
    }

    fn reset(&mut self) {
        (**self).reset()
    }

    fn max_len(&self) -> usize {
        (**self).max_len()
    }

    fn add_sequence(&mut self) -> Result<usize> {
        (**self).add_sequence()
    }

    fn select_sequence(&mut self, id: usize) -> Result<()> {
        (**self).select_sequence(id)
    }
}

/// The cosines and sines of the angles rotary position embeddings rotate
/// by, laid out as [`Device::rope`] expects, for positions up to `max_len`
/// and `rot_dim` rotated elements per head. Each pair of elements rotates at
/// its own frequency.
pub fn rope_table(max_len: usize, rot_dim: usize, theta: f32) -> Vec<f32> {
    let half = rot_dim / 2;
    let mut table = Vec::with_capacity(max_len * rot_dim);
    for pos in 0..max_len {
        for i in 0..half {
            let freq = 1.0 / theta.powf((2 * i) as f32 / rot_dim as f32);
            let (sin, cos) = (pos as f32 * freq).sin_cos();
            table.extend([cos, sin]);
        }
    }
    table
}

/// A conversation with the model, as the harness holds one: a [`Chat`] on
/// a model in this process, or one kept with the server that has the
/// model, and reached over its socket. The methods are [`Chat`]'s.
pub trait Conversation {
    /// Opens the conversation with a system prompt, reporting how many of
    /// its tokens the model has read so far, out of how many, as it goes.
    fn system(&mut self, text: &str, on_progress: impl FnMut(usize, usize)) -> Result<()>;

    /// Forgets the conversation, so that a system prompt can open another.
    fn clear(&mut self) -> Result<()>;

    /// Feeds `turns` as the conversation so far, without a reply to them.
    fn replay(&mut self, turns: &[Turn]) -> Result<()>;

    /// Number of tokens `turn` takes when fed.
    fn measure(&mut self, turn: &Turn) -> Result<usize>;

    /// Most tokens a plain answer to `message` takes with its framing, if
    /// it is limited to `limit` tokens.
    fn answer_tokens(&mut self, message: &str, limit: usize) -> Result<usize>;

    /// Number of tokens in the conversation so far.
    fn tokens(&self) -> usize;

    /// Number of tokens the conversation has room for in all.
    fn capacity(&self) -> usize;

    /// Tokens the conversation has room for before the reserve.
    fn free(&self) -> usize;

    /// Keeps `tokens` of the context free: a reply stops short of them.
    fn reserve(&mut self, tokens: usize);

    /// Most tokens a thought may run to, if they are limited.
    fn thought_limit(&self) -> Option<usize>;

    /// Ends a thought that runs to `tokens`, if they are limited, or that
    /// repeats itself, with `closing` written into it as the model's own.
    fn limit_thoughts(&mut self, tokens: Option<usize>, closing: &str);

    /// Feeds the results of the tool calls the last reply made, without a
    /// reply.
    fn feed_results(&mut self, outputs: &[String]) -> Result<()>;

    /// Where the conversation's time has gone so far.
    fn stats(&self) -> Stats;

    /// The sampler the model's tokens are picked with.
    fn sampler(&self) -> &Sampler;

    /// Sends a message from the user, streaming the reply to `on_chunk`,
    /// which ends it early by breaking.
    fn send(
        &mut self,
        message: &str,
        on_chunk: impl FnMut(Chunk) -> ControlFlow<()>,
    ) -> Result<Reply>;

    /// Sends the results of the tool calls the last reply made, and
    /// streams the reply to them like `send`.
    fn respond(
        &mut self,
        outputs: &[String],
        on_chunk: impl FnMut(Chunk) -> ControlFlow<()>,
    ) -> Result<Reply>;

    /// Has the model answer `message` plainly, under `heads` in turn, in
    /// at most `limit` tokens, streaming the answer to `on_text`.
    fn answer(
        &mut self,
        message: &str,
        heads: &[&str],
        limit: usize,
        on_text: impl FnMut(&str) -> ControlFlow<()>,
    ) -> Result<Answer>;
}

impl<M: LanguageModel> Conversation for Chat<M> {
    fn system(&mut self, text: &str, on_progress: impl FnMut(usize, usize)) -> Result<()> {
        Chat::system(self, text, on_progress)
    }

    fn clear(&mut self) -> Result<()> {
        Chat::clear(self);
        Ok(())
    }

    fn replay(&mut self, turns: &[Turn]) -> Result<()> {
        Chat::replay(self, turns)
    }

    fn measure(&mut self, turn: &Turn) -> Result<usize> {
        Chat::measure(self, turn)
    }

    fn answer_tokens(&mut self, message: &str, limit: usize) -> Result<usize> {
        Chat::answer_tokens(self, message, limit)
    }

    fn tokens(&self) -> usize {
        Chat::tokens(self)
    }

    fn capacity(&self) -> usize {
        Chat::capacity(self)
    }

    fn free(&self) -> usize {
        Chat::free(self)
    }

    fn reserve(&mut self, tokens: usize) {
        Chat::reserve(self, tokens)
    }

    fn thought_limit(&self) -> Option<usize> {
        Chat::thought_limit(self)
    }

    fn limit_thoughts(&mut self, tokens: Option<usize>, closing: &str) {
        Chat::limit_thoughts(self, tokens, closing)
    }

    fn feed_results(&mut self, outputs: &[String]) -> Result<()> {
        Chat::feed_results(self, outputs)
    }

    fn stats(&self) -> Stats {
        Chat::stats(self)
    }

    fn sampler(&self) -> &Sampler {
        Chat::sampler(self)
    }

    fn send(
        &mut self,
        message: &str,
        on_chunk: impl FnMut(Chunk) -> ControlFlow<()>,
    ) -> Result<Reply> {
        Chat::send(self, message, on_chunk)
    }

    fn respond(
        &mut self,
        outputs: &[String],
        on_chunk: impl FnMut(Chunk) -> ControlFlow<()>,
    ) -> Result<Reply> {
        Chat::respond(self, outputs, on_chunk)
    }

    fn answer(
        &mut self,
        message: &str,
        heads: &[&str],
        limit: usize,
        on_text: impl FnMut(&str) -> ControlFlow<()>,
    ) -> Result<Answer> {
        Chat::answer(self, message, heads, limit, on_text)
    }
}
