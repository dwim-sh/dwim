//! Answering one prompt and exiting, without the shell.
//!
//! The model's reply goes to standard output and nothing else does, so that
//! a script can read the reply on its own. A prompt starting with `/` and
//! the name of a skill invokes it, as `/skill arguments` in the shell does:
//! the skill's instructions, loaded the way the skill's tool would load
//! them, followed by the arguments after the first whitespace, are the
//! message sent. Any other prompt, such as one starting with a path, is sent
//! as it is. Getting ready, what the model is thinking, the commands it
//! runs, and what they printed all go to standard error.

use std::{
    env,
    error::Error,
    io::{self, Write},
    ops::ControlFlow,
    path::Path,
    time::Instant,
};

use dwim::harness::{Event, Log};
use dwim::skills::{self, Skill};

use crate::session::{self, Settings, Stderr};

/// The message `prompt` sends: a skill's instructions with the arguments
/// after them, when it is `/` and one of `skills`' names, or the prompt as
/// it is.
fn message(prompt: &str, skills: &[Skill]) -> Result<String, String> {
    let Some(rest) = prompt.strip_prefix('/') else {
        return Ok(prompt.to_string());
    };
    let (name, arguments) = rest.split_once(char::is_whitespace).unwrap_or((rest, ""));
    if !skills.iter().any(|skill| skill.name == name) {
        return Ok(prompt.to_string());
    }
    skills::invoke(skills, name, arguments.trim())
}

/// Answers `prompt` with the model, running the tools it calls, and returns
/// once it replies with text alone, keeping a transcript of it in `log`;
/// then reports where the time went, if `stats` asks for it.
pub fn once(
    settings: &Settings,
    prompt: &str,
    stats: bool,
    log: Log,
) -> Result<(), Box<dyn Error>> {
    let started = Instant::now();
    let transcript = log.path().map(Path::to_path_buf);
    let mut stderr = Stderr::new();
    let skills = skills::find(&env::current_dir()?).skills;
    // Before the model loads, so that a skill that cannot be read says so
    // at once.
    let message = message(prompt, &skills)?;
    let mut loaded = session::load(settings, skills, log, |loading| stderr.report(&loading))?;
    if let Some(path) = transcript {
        eprintln!("transcript in {}", path.display());
    }
    // No one is here to answer the model, so it goes on by itself.
    loaded.harness.set_auto(true);
    let mut printer = Printer::default();
    loaded.harness.send(&message, |event| {
        printer.print(event);
        ControlFlow::Continue(())
    })?;
    drop(printer);
    if stats {
        for line in loaded
            .harness
            .stats()
            .report(loaded.loading, started.elapsed())
        {
            eprintln!("{line}");
        }
    }
    Ok(())
}

/// Prints a turn as it happens: the reply to standard output, and the
/// thought behind it, the tools it runs, and the note it writes along the
/// way to standard error.
#[derive(Default)]
struct Printer {
    /// Whether anything of the reply has been printed, so that the blank
    /// lines the model opens with are left out.
    replied: bool,
    /// Whether standard error is part way through a line of thought.
    thinking: bool,
}

impl Printer {
    fn print(&mut self, event: Event) {
        if let Some(notice) = event.notice() {
            self.end_thought();
            eprintln!("[{notice}]");
            return;
        }
        match event {
            Event::Thought(text) | Event::Note(text) => {
                let text = if self.thinking {
                    text.as_str()
                } else {
                    text.trim_start()
                };
                if !text.is_empty() {
                    eprint!("{text}");
                    let _ = io::stderr().flush();
                    self.thinking = true;
                }
            }
            Event::Text(text) => {
                self.end_thought();
                let text = if self.replied {
                    text.as_str()
                } else {
                    text.trim_start()
                };
                if !text.is_empty() {
                    print!("{text}");
                    let _ = io::stdout().flush();
                    self.replied = true;
                }
            }
            Event::Call { name, detail } => {
                self.end_thought();
                eprintln!("{name}({detail})");
            }
            Event::Output(output) => eprintln!("{output}"),
            Event::Token | Event::Cut | Event::Compacting | Event::Compacted { .. } => {}
        }
    }

    /// Ends the line of thought being printed, if there is one.
    fn end_thought(&mut self) {
        if self.thinking {
            eprintln!();
            self.thinking = false;
        }
    }
}

impl Drop for Printer {
    /// Ends the reply with a newline, as a command's output should end.
    fn drop(&mut self) {
        self.end_thought();
        if self.replied {
            println!();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use dwim::skills::Skill;

    use super::message;

    #[test]
    fn sends_a_skill_or_the_prompt_as_it_is() {
        let dir = std::env::temp_dir().join(format!("dwim-once-{}", std::process::id()));
        fs::remove_dir_all(&dir).ok();
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("SKILL.md"),
            "---\nname: greet\ndescription: Greets.\n---\nSay hello.\n",
        )
        .unwrap();
        let skills = [Skill {
            name: "greet".into(),
            description: "Greets.".into(),
            dir: dir.clone(),
        }];
        let sent = message("/greet Pekka", &skills).unwrap();
        assert!(sent.starts_with("<skill name=\"greet\""), "{sent}");
        assert!(sent.ends_with("\n\nPekka"), "{sent}");
        for prompt in [
            "fix the build",
            "/etc/nginx/nginx.conf has a syntax error, fix it",
            "/tmp is full",
            "/gret Pekka",
        ] {
            assert_eq!(message(prompt, &skills).unwrap(), prompt);
        }
        fs::remove_dir_all(&dir).unwrap();
    }
}
