//! The slash commands: what the user can ask of the shell itself.

use crate::opts;

/// A slash command, as the user types it after the slash.
pub enum Command {
    Model,
    Stats,
    Transcript,
    Compact,
    /// Ask what the thinking budget is, or set it.
    ThinkingBudget,
    SetThinkingBudget(Option<usize>),
    /// Ask whether auto mode is on, or set it.
    Auto,
    SetAuto(bool),
    Server(ServerAction),
    Help,
}

impl Command {
    /// The command `text` names, or what is wrong with it.
    pub fn parse(text: &str) -> Result<Self, String> {
        let (name, argument) = text
            .trim()
            .split_once(char::is_whitespace)
            .map_or((text.trim(), None), |(name, rest)| {
                (name, Some(rest.trim()))
            });
        match (name, argument) {
            ("model", None) => Ok(Command::Model),
            ("stats", None) => Ok(Command::Stats),
            ("transcript", None) => Ok(Command::Transcript),
            ("compact", None) => Ok(Command::Compact),
            ("help", None) => Ok(Command::Help),
            ("thinking-budget", None) => Ok(Command::ThinkingBudget),
            ("thinking-budget", Some(tokens)) => tokens
                .parse()
                .map(|tokens| Command::SetThinkingBudget(opts::budget(tokens)))
                .map_err(|_| {
                    "/thinking-budget takes a number of tokens, or 0 for no cap".to_string()
                }),
            ("auto", None) => Ok(Command::Auto),
            ("auto", Some("on")) => Ok(Command::SetAuto(true)),
            ("auto", Some("off")) => Ok(Command::SetAuto(false)),
            ("auto", Some(_)) => Err("/auto takes on or off".to_string()),
            ("server", None) => Ok(Command::Server(ServerAction::Show)),
            ("server", Some("stop")) => Ok(Command::Server(ServerAction::Stop)),
            ("server", Some("start")) => Ok(Command::Server(ServerAction::Start)),
            ("server", Some("restart")) => Ok(Command::Server(ServerAction::Restart)),
            ("server", Some(_)) => Err("/server takes stop, start or restart".to_string()),
            _ => Err(format!("Unknown command /{text} · /help lists them")),
        }
    }
}

/// What the `/server` command asks of the server.
#[derive(Clone, Copy)]
pub enum ServerAction {
    /// Show the server: who it is and how many sessions it has.
    Show,
    Stop,
    Start,
    Restart,
}

/// The slash commands, each with the line `/help` says of it.
pub const COMMANDS: [(&str, &str); 8] = [
    (
        "/model",
        "the model, the device it runs on, and its context window",
    ),
    ("/stats", "where this session's time has gone"),
    ("/transcript", "where this session's transcript is"),
    ("/compact", "compact the conversation now"),
    (
        "/thinking-budget N",
        "cap the model's thinking at N tokens, 0 for no cap",
    ),
    (
        "/auto on|off",
        "go on by itself until it says it is done, or ask",
    ),
    (
        "/server stop|start|restart",
        "the dwim server the model runs in; stop, start or restart it",
    ),
    ("/help", "this list"),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_commands() {
        assert!(matches!(Command::parse("compact"), Ok(Command::Compact)));
        assert!(matches!(Command::parse("help "), Ok(Command::Help)));
        assert!(matches!(
            Command::parse("thinking-budget"),
            Ok(Command::ThinkingBudget)
        ));
        assert!(matches!(
            Command::parse("thinking-budget 100"),
            Ok(Command::SetThinkingBudget(Some(100)))
        ));
        assert!(matches!(
            Command::parse("thinking-budget 0"),
            Ok(Command::SetThinkingBudget(None))
        ));
        assert!(Command::parse("thinking-budget lots").is_err());
        assert!(matches!(
            Command::parse("auto on"),
            Ok(Command::SetAuto(true))
        ));
        assert!(matches!(Command::parse("auto"), Ok(Command::Auto)));
        assert!(Command::parse("auto maybe").is_err());
        assert!(Command::parse("model now").is_err());
        assert!(
            Command::parse("nothing").is_err_and(|e| e.starts_with("Unknown command /nothing"))
        );
    }
}
