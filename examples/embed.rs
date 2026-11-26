//! The agent in a program of one's own: answers the prompt on the command
//! line, working in the directory it is run in, and prints the reply.
//!
//! ```text
//! cargo run --release --example embed -- "what does this project do?"
//! ```
//!
//! The model runs where `dwim` would run it, but for one thing: on this
//! machine, it is loaded in this program, with no server and no `dwim`
//! beside it. That takes the memory the model takes, so a `dwim` server
//! with the model loaded has to be stopped first.

use std::{
    env,
    error::Error,
    io::{self, Write},
    ops::ControlFlow,
};

use dwim::{
    agent::{Agent, Settings, Status},
    harness::{Event, Log},
    model::Model,
};

fn main() -> Result<(), Box<dyn Error>> {
    let prompt = env::args().skip(1).collect::<Vec<_>>().join(" ");
    if prompt.trim().is_empty() {
        return Err("usage: embed <prompt>".into());
    }
    let settings = Settings::new(&env::current_dir()?, Model::default());
    let mut agent = Agent::open(settings, Log::none(), |status| match status {
        Status::Device(name) => eprintln!("running on {name}"),
        Status::Weights { done, total } if done == total => eprintln!("loaded the model"),
        _ => {}
    })?;
    // No one is here to answer the model, so it goes on by itself.
    agent.harness().set_auto(true);
    agent.send(&prompt, |event| {
        match event {
            Event::Text(text) => {
                print!("{text}");
                let _ = io::stdout().flush();
            }
            Event::Call { name, detail } => eprintln!("{name}({detail})"),
            _ => {}
        }
        ControlFlow::Continue(())
    })?;
    println!();
    Ok(())
}
