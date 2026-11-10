//! The `skill` tool: the instructions of a skill, by name. The skills are
//! listed in the system prompt by name and description, and this loads
//! one's instructions only when a task calls for them, so the prompt does
//! not carry what the model will not use.

use serde_json::Value;

use crate::skills::Skill;
use crate::tools::{Parameter, Signature, Tool, required, string};

const NAME: &str = "skill";

pub const TOOL: Tool = Tool {
    name: NAME,
    signature,
    changes: false,
    describe,
    run: |tools, arguments| run(&tools.skills, arguments),
};

fn signature() -> Signature {
    Signature {
        description: "Load a skill's instructions by name, from the skills listed in the system prompt, when the task matches the skill's description or the user names it. Follow the instructions it returns; a path in them is relative to the skill's directory, which the result names.".to_string(),
        parameters: vec![Parameter::string("name", "The skill's name, as listed.")],
    }
}

/// How a call reads on screen: the skill's name, as given.
fn describe(arguments: &Value) -> Option<String> {
    string(arguments, "name").map(str::to_string)
}

/// The skill named in `arguments` from `skills`, and its instructions.
fn run(skills: &[Skill], arguments: &Value) -> Result<String, String> {
    let name = required(arguments, NAME, "name")?;
    let skill = skills
        .iter()
        .find(|skill| skill.name == name)
        .ok_or_else(|| {
            format!("no skill called '{name}'; the skills are listed in the system prompt")
        })?;
    crate::skills::load(skill)
}

#[cfg(test)]
mod tests {
    use std::fs;

    use crate::tools::{ToolCall, Tools, describe};

    #[test]
    fn loads_a_skill() {
        let dir = std::env::temp_dir().join(format!("dwim-skill-{}", std::process::id()));
        fs::remove_dir_all(&dir).ok();
        let skill_dir = dir.join(".dwim/skills/demo");
        fs::create_dir_all(skill_dir.join("scripts")).unwrap();
        fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: demo\ndescription: A demo.\n---\nDo the demo.\n",
        )
        .unwrap();
        fs::write(skill_dir.join("scripts/run.sh"), "").unwrap();
        let mut tools = Tools::new(&dir);
        tools.set_skills(crate::skills::find(&dir).skills);
        let call = ToolCall::parse(r#"{"name": "skill", "arguments": {"name": "demo"}}"#).unwrap();
        let output = tools.run(&call);
        assert!(
            output.text.starts_with("<skill name=\"demo\" directory="),
            "{}",
            output.text
        );
        assert!(
            output.text.contains("Do the demo.\n</skill>"),
            "{}",
            output.text
        );
        assert!(
            output
                .text
                .ends_with("Files there besides SKILL.md: scripts/"),
            "{}",
            output.text
        );
        assert!(!output.changed);
        let nope = ToolCall::parse(r#"{"name": "skill", "arguments": {"name": "nope"}}"#).unwrap();
        let refused = tools.run(&nope);
        assert!(
            refused.text.starts_with("error: no skill called 'nope'"),
            "{}",
            refused.text
        );
        assert_eq!(describe(&call), "demo");
        fs::remove_dir_all(&dir).unwrap();
    }
}
