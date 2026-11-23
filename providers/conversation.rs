use std::ops::ControlFlow;

use crate::{
    model::{
        Answer, Chat, Chunk, Generation, Reply, Result, Sampling, Session, State, Stats, Turn,
    },
    providers::Provider,
};

/// A session on a provider: the conversation so far, kept here and sent
/// whole with each request, and where the provider said it was after the
/// last one.
pub struct Conversation<P> {
    provider: P,
    chat: Chat,
    generation: Generation,
    state: State,
}

impl<P: Provider> Conversation<P> {
    /// A conversation with the model `provider` reaches, yet to be opened
    /// with a system prompt.
    pub fn new(provider: P) -> Self {
        Self {
            provider,
            chat: Chat::default(),
            generation: Generation::default(),
            state: State::default(),
        }
    }

    /// The provider the conversation is on.
    pub fn provider(&mut self) -> &mut P {
        &mut self.provider
    }

    /// Opens the conversation with the system prompt, which the provider
    /// reads, reporting how many of its tokens so far, out of how many.
    pub fn start(&mut self, system: &str, on_progress: impl FnMut(usize, usize)) -> Result<()> {
        if !self.chat.system.is_empty() || !self.chat.turns.is_empty() {
            return Err("the conversation has already started".into());
        }
        self.chat.system = system.to_string();
        self.read(on_progress)
    }

    /// Has the provider read the conversation as it is now.
    fn read(&mut self, on_progress: impl FnMut(usize, usize)) -> Result<()> {
        self.state = self.provider.read(&self.chat, on_progress)?;
        Ok(())
    }

    /// Asks for the reply to the conversation as it is now, and adds it.
    fn complete(&mut self, on_chunk: impl FnMut(Chunk) -> ControlFlow<()>) -> Result<Reply> {
        let completion = self
            .provider
            .reply(&self.chat, &self.generation, on_chunk)?;
        self.state = completion.state;
        self.chat.turns.push(Turn::Reply {
            text: completion.text,
            calls: completion.reply.calls.clone(),
            thought: completion.thought,
        });
        Ok(completion.reply)
    }
}

impl<P: Provider> Session for Conversation<P> {
    fn clear(&mut self) -> Result<()> {
        self.chat = Chat::default();
        self.state.tokens = 0;
        Ok(())
    }

    fn replay(&mut self, turns: &[Turn]) -> Result<()> {
        self.chat.turns.extend_from_slice(turns);
        self.read(|_, _| {})
    }

    fn measure(&mut self, turn: &Turn) -> Result<usize> {
        self.provider.measure(turn)
    }

    fn answer_tokens(&mut self, message: &str, limit: usize) -> Result<usize> {
        self.provider.answer_tokens(message, limit)
    }

    fn tokens(&self) -> usize {
        self.state.tokens
    }

    fn capacity(&self) -> usize {
        self.provider.capacity()
    }

    fn free(&self) -> usize {
        self.capacity()
            .saturating_sub(self.state.tokens + self.generation.reserve)
    }

    fn reserve(&mut self, tokens: usize) {
        self.generation.reserve = tokens;
    }

    fn limit_thoughts(&mut self, tokens: Option<usize>, closing: &str) {
        self.generation.thought_limit = tokens;
        self.generation.closing = Some(closing.to_string());
    }

    fn feed_results(&mut self, outputs: &[String]) -> Result<()> {
        self.chat.turns.push(Turn::Results(outputs.to_vec()));
        self.read(|_, _| {})
    }

    fn stats(&self) -> Stats {
        self.state.stats
    }

    fn sampling(&self) -> Sampling {
        self.provider.sampling()
    }

    fn send(
        &mut self,
        message: &str,
        on_chunk: impl FnMut(Chunk) -> ControlFlow<()>,
    ) -> Result<Reply> {
        self.chat.turns.push(Turn::User(message.to_string()));
        self.complete(on_chunk)
    }

    fn respond(
        &mut self,
        outputs: &[String],
        on_chunk: impl FnMut(Chunk) -> ControlFlow<()>,
    ) -> Result<Reply> {
        self.chat.turns.push(Turn::Results(outputs.to_vec()));
        self.complete(on_chunk)
    }

    fn answer(
        &mut self,
        message: &str,
        heads: &[(&str, usize)],
        limit: usize,
        on_text: impl FnMut(&str) -> ControlFlow<()>,
    ) -> Result<Answer> {
        self.chat.turns.push(Turn::User(message.to_string()));
        let (answer, state) =
            self.provider
                .answer(&self.chat, heads, limit, &self.generation, on_text)?;
        self.state = state;
        self.chat.turns.push(Turn::Reply {
            text: answer.text.clone(),
            calls: Vec::new(),
            thought: String::new(),
        });
        Ok(answer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::Completion;

    /// A provider that keeps the conversations it is asked with, and
    /// replies to each with a call to a tool.
    #[derive(Default)]
    struct Asked {
        chats: Vec<Chat>,
    }

    impl Provider for Asked {
        fn capacity(&self) -> usize {
            1000
        }

        fn sampling(&self) -> Sampling {
            Sampling {
                temperature: 0.0,
                top_k: 1,
                top_p: 1.0,
                min_p: 0.0,
                seed: 1,
            }
        }

        fn measure(&mut self, _: &Turn) -> Result<usize> {
            Ok(1)
        }

        fn answer_tokens(&mut self, _: &str, limit: usize) -> Result<usize> {
            Ok(limit)
        }

        fn read(&mut self, chat: &Chat, _: impl FnMut(usize, usize)) -> Result<State> {
            self.chats.push(chat.clone());
            Ok(State {
                tokens: 10 * (1 + chat.turns.len()),
                ..State::default()
            })
        }

        fn reply(
            &mut self,
            chat: &Chat,
            _: &Generation,
            mut on_chunk: impl FnMut(Chunk) -> ControlFlow<()>,
        ) -> Result<Completion> {
            self.chats.push(chat.clone());
            let _ = on_chunk(Chunk::Text("Looking."));
            Ok(Completion {
                reply: Reply {
                    calls: vec!["ls".to_string()],
                    ..Reply::default()
                },
                text: "Looking.".to_string(),
                thought: "They ask what is here.".to_string(),
                state: State {
                    tokens: 10 * (2 + chat.turns.len()),
                    ..State::default()
                },
            })
        }

        fn answer(
            &mut self,
            chat: &Chat,
            _: &[(&str, usize)],
            _: usize,
            _: &Generation,
            _: impl FnMut(&str) -> ControlFlow<()>,
        ) -> Result<(Answer, State)> {
            self.chats.push(chat.clone());
            let answer = Answer {
                text: "noted".to_string(),
                cut: false,
            };
            Ok((answer, State::default()))
        }
    }

    #[test]
    fn asks_with_the_whole_conversation_each_time() {
        let mut conversation = Conversation::new(Asked::default());
        conversation.start("Be brief.", |_, _| {}).unwrap();
        let reply = conversation
            .send("What is here?", |_| ControlFlow::Continue(()))
            .unwrap();
        assert_eq!(reply.calls, ["ls"]);
        conversation
            .respond(&["a.rs".to_string()], |_| ControlFlow::Continue(()))
            .unwrap();

        let last = conversation.provider().chats.last().unwrap().clone();
        assert_eq!(last.system, "Be brief.");
        assert_eq!(
            last.turns,
            [
                Turn::User("What is here?".to_string()),
                // The reply goes back with the thought behind it.
                Turn::Reply {
                    text: "Looking.".to_string(),
                    calls: vec!["ls".to_string()],
                    thought: "They ask what is here.".to_string(),
                },
                Turn::Results(vec!["a.rs".to_string()]),
            ]
        );
        // Where the provider said the conversation was, after its reply.
        assert_eq!(conversation.tokens(), 50);
        assert_eq!(conversation.free(), 950);
    }

    #[test]
    fn opens_over_with_what_is_replayed_once_cleared() {
        let mut conversation = Conversation::new(Asked::default());
        conversation.start("Be brief.", |_, _| {}).unwrap();
        conversation
            .send("What is here?", |_| ControlFlow::Continue(()))
            .unwrap();
        conversation.clear().unwrap();
        assert_eq!(conversation.tokens(), 0);
        conversation.start("Be brief.", |_, _| {}).unwrap();
        let note = [Turn::User("The note.".to_string())];
        conversation.replay(&note).unwrap();

        let last = conversation.provider().chats.last().unwrap().clone();
        assert_eq!(last.system, "Be brief.");
        assert_eq!(last.turns, note);
    }
}
