//! The providers as the shell's pickers list them: for `/model`, the
//! models each runs; for `/provider`, whether each is disabled; and for
//! `/login`, the hosts and the keys to them.

use dwim::{
    model::Model,
    providers::{
        Kind, cerebras, endpoint,
        hosted::{self, Host},
        openrouter, vercel,
    },
    sys::memory,
};

use crate::{
    cmd::shell::picker::{Account, Choice, Group},
    config::Config,
    session::{self, ready, runs},
};

/// The providers a model can run at, each with the models it runs, for
/// the model picker, and whether there are more that it cannot: one that
/// is disabled in `config`, this machine without the memory for a model,
/// a host without a key to it. They are this machine first, then the
/// hosts by name. Where the session's model runs, `current`, is among
/// them whatever is said of it, since the model does run there. The
/// user's own server lists only what it `served`, if known.
pub fn providers(
    config: &Config,
    current: Option<Kind>,
    served: Option<Model>,
) -> (Vec<Group>, bool) {
    let usable = |&kind: &Kind| {
        Some(kind) == current
            || (!config.disabled(kind) && Model::SHOWN.iter().any(|&model| ready(kind, model)))
    };
    let groups: Vec<Group> = SHOWN
        .into_iter()
        .filter(usable)
        .map(|kind| Group {
            provider: kind,
            name: label(kind),
            models: match (kind, served) {
                (Kind::Endpoint, Some(model)) => vec![model],
                _ => Model::SHOWN
                    .into_iter()
                    .filter(|&model| runs(kind, model))
                    .collect(),
            },
        })
        .collect();
    let more = groups.len() < SHOWN.len();
    (groups, more)
}

/// The providers, for the provider picker: this machine first, then the
/// hosts by name, each with whether it is disabled in `config`, and what
/// keeps a model from running there if anything does: this machine wants
/// the memory for a model, and a host a key to it.
pub fn provider_choices(config: &Config) -> Vec<Choice> {
    SHOWN
        .into_iter()
        .map(|kind| Choice {
            provider: kind,
            name: label(kind),
            problem: match (kind, kind.host()) {
                (Kind::Endpoint, _) => session::server_address()
                    .is_none()
                    .then(|| "no server · /login takes its address".to_string()),
                (_, None) => memory::check().err().map(|e| e.to_string()),
                (_, Some(host)) => hosted::key::find(host)
                    .is_none()
                    .then(|| "no key · /login signs in".to_string()),
            },
            disabled: config.disabled(kind),
        })
        .collect()
}

/// The providers in the order the pickers show them: this machine, then
/// the hosts by name, then the user's own server.
const SHOWN: [Kind; 5] = [
    Kind::Local,
    Kind::Cerebras,
    Kind::OpenRouter,
    Kind::Vercel,
    Kind::Endpoint,
];

/// What `kind` is called in a picker: this machine with a house by it.
fn label(kind: Kind) -> &'static str {
    kind.host().map_or("Local 🏠", |host| host.name)
}

/// The hosts, each with what there is of a key to it, for the sign-in
/// picker: the environment's, which is used ahead of one kept and is not
/// the shell's to change, or one kept from signing in, or none; then the
/// user's own server.
pub fn accounts() -> Vec<Account> {
    let hosts: [&'static Host; 3] = [&cerebras::HOST, &openrouter::HOST, &vercel::HOST];
    let mut accounts: Vec<Account> = hosts
        .into_iter()
        .map(|host| {
            let kept = hosted::key::kept(host).is_some();
            let browser = host.kind == Kind::OpenRouter;
            let note = if hosted::key::environment(host).is_some() {
                format!("key in {}", host.key)
            } else if kept && browser {
                "signed in · enter signs in again".to_string()
            } else if kept {
                "signed in · enter takes another key".to_string()
            } else if browser {
                "sign in with your browser".to_string()
            } else {
                "paste a key".to_string()
            };
            Account { host, note, kept }
        })
        .collect();
    accounts.push(server_account());
    accounts
}

fn server_account() -> Account {
    let host = &endpoint::HOST;
    let configured = Config::load().ok().and_then(|config| config.endpoint());
    let kept = configured.is_some() || hosted::key::kept(host).is_some();
    let note = match configured {
        Some(address) => format!("{} · enter changes it", address.url),
        None => "give the server's address".to_string(),
    };
    Account { host, note, kept }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cmd::shell::picker::Picker;

    #[test]
    fn the_users_own_server_offers_the_model_it_serves_alone() {
        let current = Some(Kind::Endpoint);
        let (groups, more) = providers(&Config::default(), current, Some(Model::Qwen27b));
        let endpoint = groups.iter().find(|group| group.provider == Kind::Endpoint);
        assert_eq!(endpoint.unwrap().models, [Model::Qwen27b]);

        let picker = Picker::new(groups, more, current, Model::Qwen27b);
        assert_eq!(picker.picked(), Some((Kind::Endpoint, Model::Qwen27b)));
    }
}
