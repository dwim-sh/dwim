//! The plan of the work, and the `plan` tool that lays it out.
//!
//! The model lays out the steps it will take, and the choices it made
//! where the request left them open, and the tools keep the [`Plan`] from
//! then on: which steps are done, which one is under way, and what each
//! came to. The model does not say where the plan stands, since a small
//! model lets that go stale; it says when a step is finished, with the
//! `done` tool, and is told the next one. What a step changed, and the
//! last command run after its last change, are kept with it as the tools
//! see them, so that the user reads what a finished step rests on and not
//! only that the model called it finished.
//!
//! The user's interface shows the plan as it stands while the model works.
//! A choice the request leaves open is made and said in the plan, for the
//! user to overrule, not asked.

use serde_json::Value;

use crate::tools::{Parameter, Signature, Tool};

const NAME: &str = "plan";

pub const TOOL: Tool = Tool {
    name: NAME,
    signature,
    changes: false,
    verb: None,
    describe,
    run: |tools, arguments| run(&mut tools.plan, arguments),
};

fn signature() -> Signature {
    Signature {
        description: "Lay out the plan of the work, which the user watches while you work: the steps you will take, in order. The first is under way at once; call done as each is finished, and you are told the next. Call plan again to change the steps that are not done yet, which the steps given replace, or to confirm them after a message from the user; the ones done are kept, and so is where a step given again was.".to_string(),
        parameters: vec![
            Parameter::string(
                "steps",
                "The steps to take, one a line, each in ten words or fewer saying what you will do and where, as `add backoff to send() in stream.rs`: the user reads them at a glance, and the detail belongs in your work, not in the step. End with the step that checks the work, such as running the tests.",
            ),
            Parameter::string(
                "assumes",
                "The choices you made that the request did not state, one a line and a few words each: a number, a default, a name, which cases are covered, as `3 tries, doubling from 0.1 s` or `retry on 429 and 5xx only`. The user reads them, and says if they want otherwise. `none` if the request left nothing open.",
            ),
        ],
    }
}

/// The plan of the work: the choices made where the request left them
/// open, and the steps, of which the first that is not done is under way.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Plan {
    assumes: Vec<String>,
    steps: Vec<Step>,
}

/// A step of the plan, and what is known of it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Step {
    /// What the step is, as the model put it.
    pub text: String,
    pub done: bool,
    /// What the step came to, as the model said when it finished it.
    pub result: String,
    /// The files the step changed, in the order it first changed them.
    pub changed: Vec<String>,
    /// The last command run since the step's last change, if any was.
    pub check: Option<Check>,
}

/// A command run after a change, and whether it succeeded.
#[derive(Clone, Debug, PartialEq)]
pub struct Check {
    pub command: String,
    pub passed: bool,
}

impl Plan {
    /// Whether no plan has been laid out.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// The steps, in order.
    pub fn steps(&self) -> &[Step] {
        &self.steps
    }

    /// The choices made where the request left them open.
    pub fn assumes(&self) -> &[String] {
        &self.assumes
    }

    /// Which step is under way: the first that is not done.
    pub fn under_way(&self) -> Option<usize> {
        self.steps.iter().position(|step| !step.done)
    }

    /// How many steps are done.
    pub fn done(&self) -> usize {
        self.steps.iter().filter(|step| step.done).count()
    }

    /// Whether there is a plan and every step of it is done.
    pub fn finished(&self) -> bool {
        !self.is_empty() && self.under_way().is_none()
    }

    /// Forgets the plan.
    pub fn clear(&mut self) {
        *self = Self::default();
    }

    /// Lays out `steps` as the ones yet to do, in place of those that were,
    /// keeping the ones done, and `assumes` as the choices made if any are
    /// given. A step that was already there, not done, keeps what is known
    /// of it.
    pub fn lay(&mut self, steps: Vec<String>, assumes: Option<Vec<String>>) {
        let mut before: Vec<Step> = Vec::new();
        let mut kept = Vec::new();
        for step in self.steps.drain(..) {
            if step.done {
                kept.push(step);
            } else {
                before.push(step);
            }
        }
        for text in steps {
            if kept.iter().any(|step| step.text == text) {
                continue;
            }
            let step = match before.iter().position(|step| step.text == text) {
                Some(i) => before.remove(i),
                None => Step {
                    text,
                    ..Step::default()
                },
            };
            kept.push(step);
        }
        self.steps = kept;
        if let Some(assumes) = assumes {
            self.assumes = assumes;
        }
    }

    /// Finishes the step under way, which came to `result`, and returns
    /// which it was: none if no step is under way.
    pub fn finish(&mut self, result: &str) -> Option<usize> {
        let at = self.under_way()?;
        self.steps[at].done = true;
        self.steps[at].result = result.trim().to_string();
        Some(at)
    }

    /// Notes that the step under way changed the file at `path`: whatever
    /// was run before the change no longer says anything of it.
    pub fn changed(&mut self, path: &str) {
        if let Some(step) = self.current() {
            if !step.changed.iter().any(|changed| changed == path) {
                step.changed.push(path.to_string());
            }
            step.check = None;
        }
    }

    /// Notes that `command` was run, and whether it `passed`: for the step
    /// under way, if it has changed anything for the command to check, and
    /// for each step done that changed something and has had nothing run
    /// since, as when the last step of a plan runs the tests for them all.
    pub fn ran(&mut self, command: &str, passed: bool) {
        let under_way = self.under_way();
        let check = Check {
            command: command.lines().next().unwrap_or_default().to_string(),
            passed,
        };
        for (i, step) in self.steps.iter_mut().enumerate() {
            if !step.changed.is_empty() && (under_way == Some(i) || step.check.is_none()) {
                step.check = Some(check.clone());
            }
        }
    }

    /// The plan as the model is told it: the choices made, then the steps,
    /// numbered, each with where it is and what it came to.
    pub fn brief(&self) -> String {
        let mut lines = Vec::new();
        for assume in &self.assumes {
            lines.push(format!("Assuming: {assume}"));
        }
        let under_way = self.under_way();
        for (i, step) in self.steps.iter().enumerate() {
            let state = match (step.done, under_way == Some(i)) {
                (true, _) => "done",
                (false, true) => "under way",
                (false, false) => "to come",
            };
            let result = if step.result.is_empty() {
                String::new()
            } else {
                format!(": {}", step.result)
            };
            lines.push(format!("{}. [{state}] {}{result}", i + 1, step.text));
        }
        lines.join("\n")
    }

    /// What the model is told of the step now under way, or that none is.
    pub fn next(&self) -> String {
        match self.under_way() {
            Some(at) => format!(
                "Step {} of {} is under way: {}. When it is finished, call done.",
                at + 1,
                self.steps.len(),
                self.steps[at].text
            ),
            None => {
                "Every step of the plan is done: reply to the user with what you did.".to_string()
            }
        }
    }

    /// The step under way.
    fn current(&mut self) -> Option<&mut Step> {
        let at = self.under_way()?;
        self.steps.get_mut(at)
    }
}

/// How a call reads on screen: how many steps it lays out.
fn describe(arguments: &Value) -> Option<String> {
    let steps = list(arguments, "steps")?.len();
    Some(format!(
        "{steps} {}",
        if steps == 1 { "step" } else { "steps" }
    ))
}

/// Lays out the plan in `arguments`, returning what the model should see.
fn run(plan: &mut Plan, arguments: &Value) -> Result<String, String> {
    let steps = list(arguments, "steps").unwrap_or_default();
    if steps.is_empty() {
        return Err("plan needs its 'steps': give them one a line".to_string());
    }
    let assumes = list(arguments, "assumes").map(|assumes| {
        assumes
            .into_iter()
            .filter(|assume| !assume.trim_end_matches('.').eq_ignore_ascii_case("none"))
            .collect()
    });
    plan.lay(steps, assumes);
    Ok(format!("[the user sees the plan]\n{}", plan.next()))
}

/// The list the argument `key` gives, if there is one: the lines of a
/// string, or the strings of an array, which a model writes as often for a
/// list, as the argument itself or as the text of one.
fn list(arguments: &Value, key: &str) -> Option<Vec<String>> {
    let items = |items: &[Value]| {
        let text: Vec<&str> = items.iter().filter_map(Value::as_str).collect();
        lines(&text.join("\n"))
    };
    match arguments.get(key)? {
        Value::Array(array) => Some(items(array)),
        Value::String(text) => Some(match serde_json::from_str(text.trim()) {
            Ok(Value::Array(array)) => items(&array),
            _ => lines(text),
        }),
        _ => None,
    }
}

/// The lines of `text` that say something, each without the bullet, the
/// number or the checkbox a model tends to open a list's line with. A line
/// whose box is checked is left out: it is a step the model says is done,
/// and the plan keeps those itself.
fn lines(text: &str) -> Vec<String> {
    text.lines()
        .filter_map(|line| {
            let line = line.trim().trim_start_matches(['-', '*', '•']).trim_start();
            let digits = line.chars().take_while(char::is_ascii_digit).count();
            let line = match line[digits..].strip_prefix(['.', ')']) {
                Some(rest) if digits > 0 => rest.trim_start(),
                _ => line,
            };
            let line = match line.get(..3).map(str::to_ascii_lowercase).as_deref() {
                Some("[x]") => return None,
                Some("[ ]" | "[>]" | "[~]") => line[3..].trim_start(),
                _ => line,
            };
            (!line.is_empty()).then(|| line.to_string())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn texts(plan: &Plan) -> Vec<(&str, bool)> {
        plan.steps()
            .iter()
            .map(|step| (step.text.as_str(), step.done))
            .collect()
    }

    #[test]
    fn lays_out_the_steps_however_the_model_lists_them() {
        let mut plan = Plan::default();
        let said = run(
            &mut plan,
            &json!({
                "steps": "- [ ] read stream.rs\n2. add backoff\n\n* [x] already did this\n3) run cargo test\n- [X] ",
                "assumes": "- retry on 429 and 5xx only\n",
            }),
        )
        .unwrap();
        assert_eq!(
            texts(&plan),
            [
                ("read stream.rs", false),
                ("add backoff", false),
                ("run cargo test", false)
            ]
        );
        assert_eq!(plan.assumes(), ["retry on 429 and 5xx only"]);
        let mut open = Plan::default();
        run(&mut open, &json!({ "steps": "look", "assumes": "None." })).unwrap();
        assert!(open.assumes().is_empty());
        assert_eq!(plan.under_way(), Some(0));
        assert!(
            said.contains("Step 1 of 3 is under way: read stream.rs"),
            "{said}"
        );
        // A list written as an array, or as the text of one, is its items.
        for steps in [json!(["look", "- change"]), json!("[\"look\", \"change\"]")] {
            let mut plan = Plan::default();
            run(&mut plan, &json!({ "steps": steps })).unwrap();
            assert_eq!(texts(&plan), [("look", false), ("change", false)]);
        }
        assert!(run(&mut plan, &json!({ "steps": "\n  \n" })).is_err());
        assert!(run(&mut plan, &json!({})).is_err());
    }

    #[test]
    fn a_plan_laid_again_keeps_what_is_done_and_what_is_known() {
        let mut plan = Plan::default();
        plan.lay(
            vec!["look".to_string(), "change".to_string(), "test".to_string()],
            Some(vec!["three tries".to_string()]),
        );
        assert_eq!(plan.finish("it is in send()"), Some(0));
        plan.changed("a.rs");
        // The steps to come are replaced, the one done is not, though the
        // model lists it again; the one under way keeps its change, and
        // the choices stand when none are given.
        plan.lay(
            vec![
                "look".to_string(),
                "change".to_string(),
                "document".to_string(),
            ],
            None,
        );
        assert_eq!(
            texts(&plan),
            [("look", true), ("change", false), ("document", false)]
        );
        assert_eq!(plan.steps()[1].changed, ["a.rs"]);
        assert_eq!(plan.assumes(), ["three tries"]);
        assert_eq!(
            plan.brief(),
            "Assuming: three tries\n1. [done] look: it is in send()\n2. [under way] change\n3. [to come] document"
        );
    }

    #[test]
    fn keeps_what_a_step_changed_and_what_was_run_after() {
        let mut plan = Plan::default();
        plan.lay(vec!["change".to_string(), "test".to_string()], None);
        // A command run before anything changed checks nothing.
        plan.ran("ls", true);
        assert_eq!(plan.steps()[0].check, None);
        plan.changed("a.rs");
        plan.ran("cargo test\n# all of them", false);
        plan.changed("a.rs");
        assert_eq!(plan.steps()[0].changed, ["a.rs"]);
        assert_eq!(plan.steps()[0].check, None, "a change since the run");
        plan.ran("cargo test", true);
        plan.finish("done");
        assert_eq!(
            plan.steps()[0].check,
            Some(Check {
                command: "cargo test".to_string(),
                passed: true
            })
        );
        // What comes after is the next step's; a command run then says
        // nothing of a step done that had one run, and vouches for one
        // that had none.
        plan.changed("b.rs");
        assert_eq!(plan.steps()[0].changed, ["a.rs"]);
        assert_eq!(plan.steps()[1].changed, ["b.rs"]);
        plan.ran("false", false);
        assert!(plan.steps()[0].check.as_ref().unwrap().passed);
        assert!(!plan.steps()[1].check.as_ref().unwrap().passed);
        let mut later = Plan::default();
        later.lay(vec!["change".to_string(), "test".to_string()], None);
        later.changed("a.rs");
        later.finish("");
        later.ran("cargo test", true);
        assert!(later.steps()[0].check.as_ref().unwrap().passed);
        assert_eq!(later.steps()[1].check, None);
        plan.finish("");
        assert!(plan.finished());
        assert_eq!(plan.finish("again"), None);
    }
}
