//! The `done` tool: the model says the step of the plan under way is
//! finished, and what it came to, and is told the step that is under way
//! next. What it says the step came to is what is kept of the step when
//! the conversation is cleared to make room, so it is asked for what the
//! steps to come need, not for a report.

use serde_json::Value;

use crate::tools::{Parameter, Signature, Tool, required, string};

const NAME: &str = "done";

pub const TOOL: Tool = Tool {
    name: NAME,
    signature,
    changes: false,
    describe,
    run: |tools, arguments| {
        let result = required(arguments, NAME, "result")?;
        match tools.plan.finish(result) {
            Some(at) => Ok(format!("Step {} is done. {}", at + 1, tools.plan.next())),
            None if tools.plan.is_empty() => {
                Err("there is no plan: call plan with the steps first".to_string())
            }
            None => Err("every step of the plan is already done".to_string()),
        }
    },
};

fn signature() -> Signature {
    Signature {
        description: "Say that the step of the plan under way is finished. Call it once the step's work is made and checked, before starting the next step, which the result names.".to_string(),
        parameters: vec![Parameter::string(
            "result",
            "What the step came to, in a line or two, with what the steps to come need from it: the files, lines and names found or changed, as `retries go in send(), stream.rs:120`. It is what you keep of the step if the conversation is cleared to make room.",
        )],
    }
}

/// How a call reads on screen: what the step came to.
fn describe(arguments: &Value) -> Option<String> {
    string(arguments, "result").map(str::to_string)
}
