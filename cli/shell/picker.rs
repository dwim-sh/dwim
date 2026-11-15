//! The model picker the `/model` command opens: the models by size, and
//! under them the provider on a line of its own, changed with the left
//! and right arrows. A model the provider does not run is shown, but not to
//! be picked, and neither is a provider that cannot be used.

use crossterm::style::Stylize;
use dwim::model::Variant;
use dwim::providers::Kind;

use crate::tui::{Line, span};

/// The providers and the models to pick from, and which the cursor is on.
pub struct Picker {
    groups: Vec<Group>,
    /// The model the session is on, and where it runs.
    current: (Kind, Variant),
    /// Which of the providers is picked, and the model the cursor is on.
    provider: usize,
    model: Variant,
}

/// A provider and the models it runs, or why it cannot be used.
pub struct Group {
    pub provider: Kind,
    pub models: Vec<Variant>,
    pub unavailable: Option<String>,
}

impl Picker {
    /// A picker of the models `groups` run, on the provider and the model
    /// the session is on, `current`.
    pub fn new(groups: Vec<Group>, current: (Kind, Variant)) -> Self {
        let usable = |group: &Group| group.unavailable.is_none();
        let provider = groups
            .iter()
            .position(|group| group.provider == current.0 && usable(group))
            .or_else(|| groups.iter().position(usable))
            .unwrap_or(0);
        let mut picker = Self {
            groups,
            current,
            provider,
            model: current.1,
        };
        picker.settle();
        picker
    }

    /// Changes the provider to the next one that can be used, to the right
    /// or to the left, if there is one that way. The cursor stays on its
    /// model if the provider runs it.
    pub fn turn(&mut self, right: bool) {
        let usable = |&i: &usize| self.groups[i].unavailable.is_none();
        let next = if right {
            (self.provider + 1..self.groups.len()).find(usable)
        } else {
            (0..self.provider).rfind(usable)
        };
        self.provider = next.unwrap_or(self.provider);
        self.settle();
    }

    /// Moves the cursor down, or up, to the next model the provider runs,
    /// if there is one that way.
    pub fn step(&mut self, down: bool) {
        let models = self.models();
        let runs = |&&model: &&Variant| self.groups[self.provider].models.contains(&model);
        let at = models.iter().position(|&model| model == self.model);
        let next = match (at, down) {
            (Some(at), true) => models[at + 1..].iter().find(runs),
            (Some(at), false) => models[..at].iter().rfind(runs),
            (None, _) => None,
        };
        self.model = next.copied().unwrap_or(self.model);
    }

    /// The model the cursor is on and the provider picked for it, if the
    /// provider can be used and runs it.
    pub fn picked(&self) -> Option<(Kind, Variant)> {
        let group = &self.groups[self.provider];
        (group.unavailable.is_none() && group.models.contains(&self.model))
            .then_some((group.provider, self.model))
    }

    /// The picker as rows, and which of them the cursor is on: the models
    /// in columns, the one the session is on ticked and one the provider
    /// does not run dimmed; then the providers on a line, the one picked
    /// filled, and one that cannot be used struck out, with why.
    pub fn rows(&self) -> (Vec<Line>, usize) {
        let group = &self.groups[self.provider];
        let models = self.models();
        let sizes = models.iter().map(|m| m.size().len()).max().unwrap_or(0);
        let mut rows = vec![vec![span("  Select model").bold()], Line::new()];
        let mut cursor = 0;
        for &model in &models {
            let size = format!("{:<sizes$}  ", model.size());
            let tick = if (group.provider, model) == self.current {
                "  ✓"
            } else {
                ""
            };
            rows.push(if !group.models.contains(&model) {
                vec![span(format!("    {size}{model}")).dark_grey()]
            } else if model == self.model {
                cursor = rows.len();
                vec![span(format!("  › {size}{model}{tick}")).bold()]
            } else {
                vec![
                    span(format!("    {size}")),
                    span(model.to_string()).dark_grey(),
                    span(tick),
                ]
            });
        }

        let mut providers = vec![span("  Provider  ")];
        for (i, group) in self.groups.iter().enumerate() {
            let name = group.provider.to_string();
            match &group.unavailable {
                Some(why) => {
                    providers.push(span(format!(" {name} ")).dark_grey().crossed_out());
                    providers.push(span(format!(" {why}")).dark_grey());
                }
                None if i == self.provider => {
                    providers.push(span(format!(" {name} ")).bold().reverse())
                }
                None => providers.push(span(format!(" {name} "))),
            }
            providers.push(span(" "));
        }
        rows.push(Line::new());
        rows.push(providers);
        (rows, cursor)
    }

    /// Leaves the cursor on its model if the provider runs it, and puts it
    /// on the first model the provider runs otherwise.
    fn settle(&mut self) {
        let models = &self.groups[self.provider].models;
        if !models.contains(&self.model) {
            self.model = models.first().copied().unwrap_or(self.model);
        }
    }

    /// The models, in the order they are shown.
    fn models(&self) -> Vec<Variant> {
        let mut models = Vec::new();
        for &model in self.groups.iter().flat_map(|group| &group.models) {
            if !models.contains(&model) {
                models.push(model);
            }
        }
        models
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn picker(openrouter: Option<String>) -> Picker {
        Picker::new(
            vec![
                Group {
                    provider: Kind::Local,
                    models: vec![Variant::Medium],
                    unavailable: None,
                },
                Group {
                    provider: Kind::OpenRouter,
                    models: vec![Variant::Medium, Variant::Large],
                    unavailable: openrouter,
                },
            ],
            (Kind::Local, Variant::Medium),
        )
    }

    #[test]
    fn picks_a_model_at_the_provider_turned_to() {
        let mut picker = picker(None);
        assert_eq!(picker.picked(), Some((Kind::Local, Variant::Medium)));
        picker.turn(true);
        assert_eq!(picker.picked(), Some((Kind::OpenRouter, Variant::Medium)));
        picker.step(true);
        assert_eq!(picker.picked(), Some((Kind::OpenRouter, Variant::Large)));
    }

    #[test]
    fn does_not_pick_a_model_the_provider_does_not_run() {
        let mut picker = picker(None);
        picker.step(true);
        assert_eq!(picker.picked(), Some((Kind::Local, Variant::Medium)));
        picker.turn(true);
        picker.step(true);
        picker.turn(false);
        assert_eq!(picker.picked(), Some((Kind::Local, Variant::Medium)));
    }

    #[test]
    fn does_not_turn_to_a_provider_that_cannot_be_used() {
        let mut picker = picker(Some("no key".to_string()));
        picker.turn(true);
        picker.step(true);
        assert_eq!(picker.picked(), Some((Kind::Local, Variant::Medium)));
    }
}
