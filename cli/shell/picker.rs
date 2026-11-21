//! The model picker the `/model` command opens: the models by size, and
//! under them the provider on a line of its own, changed with the left
//! and right arrows. A model the provider does not run is shown, but not
//! to be picked. Only the providers a model can run at are shown.
//!
//! And the provider picker the `/provider` command opens: the providers,
//! each with a box that is ticked unless it is disabled, and whether a
//! model can run there.
//!
//! And the sign-in picker the `/login` command opens: the hosts, each with
//! whether there is a key to it.

use crossterm::style::Stylize;
use dwim::model::Model;
use dwim::providers::{Kind, hosted::Host};
use unicode_width::UnicodeWidthStr;

use crate::tui::{Line, span};

/// The providers and the models to pick from, and which the cursor is on.
pub struct Picker {
    /// The providers a model can run at, of which there is one at least.
    groups: Vec<Group>,
    /// Whether there are more providers, that a model cannot run at as
    /// things are.
    more: bool,
    /// The model the session is on, and where it runs, if anywhere.
    current: Option<(Kind, Model)>,
    /// Which of the providers is picked, and the model the cursor is on.
    provider: usize,
    model: Model,
}

/// A provider and the models it runs.
pub struct Group {
    pub provider: Kind,
    pub name: &'static str,
    pub models: Vec<Model>,
}

impl Picker {
    /// A picker of the models `groups` run, on the provider the session's
    /// model runs at, if it runs anywhere, and the model, `model`.
    /// `more` says there are providers besides, to point at `/provider`
    /// for.
    pub fn new(groups: Vec<Group>, more: bool, provider: Option<Kind>, model: Model) -> Self {
        let at = groups
            .iter()
            .position(|group| Some(group.provider) == provider)
            .unwrap_or(0);
        let mut picker = Self {
            groups,
            more,
            current: provider.map(|provider| (provider, model)),
            provider: at,
            model,
        };
        picker.settle();
        picker
    }

    /// Changes the provider to the next one, to the right or to the left,
    /// if there is one that way. The cursor stays on its model if the
    /// provider runs it.
    pub fn turn(&mut self, right: bool) {
        if right {
            self.provider = (self.provider + 1).min(self.groups.len() - 1);
        } else {
            self.provider = self.provider.saturating_sub(1);
        }
        self.settle();
    }

    /// Moves the cursor down, or up, to the next model the provider runs,
    /// if there is one that way.
    pub fn step(&mut self, down: bool) {
        let models = self.models();
        let runs = |&&model: &&Model| self.groups[self.provider].models.contains(&model);
        let at = models.iter().position(|&model| model == self.model);
        let next = match (at, down) {
            (Some(at), true) => models[at + 1..].iter().find(runs),
            (Some(at), false) => models[..at].iter().rfind(runs),
            (None, _) => None,
        };
        self.model = next.copied().unwrap_or(self.model);
    }

    /// The model the cursor is on and the provider picked for it, if the
    /// provider runs it.
    pub fn picked(&self) -> Option<(Kind, Model)> {
        let group = &self.groups[self.provider];
        group
            .models
            .contains(&self.model)
            .then_some((group.provider, self.model))
    }

    /// The picker as rows, and which of them the cursor is on: the models
    /// in columns, the one the session is on ticked and one the provider
    /// does not run dimmed; then the providers on a line, the one picked
    /// filled; and where to enable more, if there are more.
    pub fn rows(&self) -> (Vec<Line>, usize) {
        let group = &self.groups[self.provider];
        let models = self.models();
        let alias = |model: Model| model.alias().unwrap_or("");
        let sizes = models.iter().map(|&m| alias(m).len()).max().unwrap_or(0);
        let mut rows = vec![vec![span("  Select model").bold()], Line::new()];
        let mut cursor = 0;
        for &model in &models {
            let size = format!("{:<sizes$}  ", alias(model));
            let tick = if Some((group.provider, model)) == self.current {
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
            let name = format!(" {} ", group.name);
            providers.push(if i == self.provider {
                span(name).bold().reverse()
            } else {
                span(name)
            });
            providers.push(span(" "));
        }
        rows.push(Line::new());
        rows.push(providers);
        if self.more {
            rows.push(Line::new());
            rows.push(vec![
                span("  To enable more providers, use /provider").dark_grey(),
            ]);
        }
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
    fn models(&self) -> Vec<Model> {
        let mut models = Vec::new();
        for &model in self.groups.iter().flat_map(|group| &group.models) {
            if !models.contains(&model) {
                models.push(model);
            }
        }
        models
    }
}

/// The providers, and which the cursor is on.
pub struct Providers {
    choices: Vec<Choice>,
    /// Where the session's model runs, if anywhere.
    current: Option<Kind>,
    at: usize,
}

/// A provider: whether it is disabled, which keeps a model from running
/// there unless the command line names it, and what else keeps one from
/// running there, if anything does.
pub struct Choice {
    pub provider: Kind,
    pub name: &'static str,
    pub problem: Option<String>,
    pub disabled: bool,
}

impl Providers {
    /// A picker of `choices`, on the provider the session's model runs
    /// at, `current`, or on the first of them.
    pub fn new(choices: Vec<Choice>, current: Option<Kind>) -> Self {
        let at = choices
            .iter()
            .position(|choice| Some(choice.provider) == current)
            .unwrap_or(0);
        Self {
            choices,
            current,
            at,
        }
    }

    /// Moves the cursor down, or up, to the next provider, if there is
    /// one that way.
    pub fn step(&mut self, down: bool) {
        if down {
            self.at = (self.at + 1).min(self.choices.len() - 1);
        } else {
            self.at = self.at.saturating_sub(1);
        }
    }

    /// The provider the cursor is on, as it is shown.
    pub fn choice(&self) -> &Choice {
        &self.choices[self.at]
    }

    /// Shows the provider the cursor is on as disabled, or as enabled.
    pub fn set_disabled(&mut self, disabled: bool) {
        self.choices[self.at].disabled = disabled;
    }

    /// The picker as rows, and which of them the cursor is on: the
    /// providers, each with a box that is ticked unless it is disabled,
    /// and whether a model can run there: off if it is disabled, ready,
    /// or what it wants. The one the session is on is said to be in use.
    pub fn rows(&self) -> (Vec<Line>, usize) {
        let names = self.choices.iter().map(|c| c.name.width());
        let names = names.max().unwrap_or(0);
        let mut rows = vec![vec![span("  Providers").bold()], Line::new()];
        let mut cursor = 0;
        for (i, choice) in self.choices.iter().enumerate() {
            let name = format!(
                "{} {}{}  ",
                if choice.disabled { "[ ]" } else { "[✓]" },
                choice.name,
                " ".repeat(names - choice.name.width())
            );
            let status = match &choice.problem {
                _ if choice.disabled => span("off").dark_grey(),
                Some(problem) => span(problem).dark_yellow(),
                None => span("ready").dark_green(),
            };
            let used = if Some(choice.provider) == self.current {
                " · in use"
            } else {
                ""
            };
            rows.push(if i == self.at {
                cursor = rows.len();
                vec![span(format!("  › {name}")).bold(), status, span(used)]
            } else {
                vec![span(format!("    {name}")), status, span(used).dark_grey()]
            });
        }
        (rows, cursor)
    }
}

/// The hosts to sign in to, and which the cursor is on.
pub struct Logins {
    accounts: Vec<Account>,
    at: usize,
}

/// A host to sign in to: what there is of a key to it, and whether one is
/// kept from signing in, which signing out forgets.
pub struct Account {
    pub host: &'static Host,
    pub note: String,
    pub kept: bool,
}

impl Logins {
    /// A picker of `accounts`, on the first of them.
    pub fn new(accounts: Vec<Account>) -> Self {
        Self { accounts, at: 0 }
    }

    /// Moves the cursor down, or up, to the next host, if there is one
    /// that way.
    pub fn step(&mut self, down: bool) {
        if down {
            self.at = (self.at + 1).min(self.accounts.len() - 1);
        } else {
            self.at = self.at.saturating_sub(1);
        }
    }

    /// The host the cursor is on.
    pub fn account(&self) -> &Account {
        &self.accounts[self.at]
    }

    /// The picker as rows, and which of them the cursor is on: the hosts,
    /// each with what there is of a key to it.
    pub fn rows(&self) -> (Vec<Line>, usize) {
        let names = self.accounts.iter().map(|a| a.host.name.len());
        let names = names.max().unwrap_or(0);
        let mut rows = vec![vec![span("  Sign in").bold()], Line::new()];
        let mut cursor = 0;
        for (i, account) in self.accounts.iter().enumerate() {
            let name = format!("{:<names$}  ", account.host.name);
            rows.push(if i == self.at {
                cursor = rows.len();
                vec![span(format!("  › {name}")).bold(), span(&account.note)]
            } else {
                vec![span(format!("    {name}")), span(&account.note).dark_grey()]
            });
        }
        (rows, cursor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn picker() -> Picker {
        Picker::new(
            vec![
                Group {
                    provider: Kind::Local,
                    name: "Local",
                    models: vec![Model::Qwen27b],
                },
                Group {
                    provider: Kind::OpenRouter,
                    name: "OpenRouter",
                    models: vec![Model::Qwen27b, Model::QwenMax],
                },
            ],
            false,
            Some(Kind::Local),
            Model::Qwen27b,
        )
    }

    #[test]
    fn picks_a_model_at_the_provider_turned_to() {
        let mut picker = picker();
        assert_eq!(picker.picked(), Some((Kind::Local, Model::Qwen27b)));
        picker.turn(true);
        assert_eq!(picker.picked(), Some((Kind::OpenRouter, Model::Qwen27b)));
        picker.step(true);
        assert_eq!(picker.picked(), Some((Kind::OpenRouter, Model::QwenMax)));
    }

    #[test]
    fn does_not_pick_a_model_the_provider_does_not_run() {
        let mut picker = picker();
        picker.step(true);
        assert_eq!(picker.picked(), Some((Kind::Local, Model::Qwen27b)));
        picker.turn(true);
        picker.step(true);
        picker.turn(false);
        assert_eq!(picker.picked(), Some((Kind::Local, Model::Qwen27b)));
    }
}
