//! A tool call, as the model writes it.

use std::error::Error;

use serde::Deserialize;
use serde_json::Value;

/// A call the model made to a tool, as written in a `<tool_call>` block.
#[derive(Debug, Deserialize)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Value,
}

impl ToolCall {
    /// Parses the body of a `<tool_call>` block: either a JSON object with
    /// the name and the arguments, as Qwen3 writes it, or a `<function=name>`
    /// block of `<parameter=name>` values, as Qwen3-Coder writes it, whose
    /// values are all strings.
    pub fn parse(text: &str) -> Result<Self, Box<dyn Error>> {
        let text = text.trim();
        if text.starts_with('{') {
            return Ok(serde_json::from_str(text)?);
        }
        let rest = text
            .strip_prefix("<function=")
            .ok_or("a tool call is a JSON object or a <function=...> block")?;
        let (name, mut rest) = rest.split_once('>').ok_or("unterminated function name")?;
        let mut arguments = serde_json::Map::new();
        loop {
            rest = rest.trim_start();
            if let Some(after) = rest.strip_prefix("</function>") {
                if !after.trim().is_empty() {
                    return Err("text after </function>".into());
                }
                break;
            }
            let param = rest
                .strip_prefix("<parameter=")
                .ok_or("expected <parameter=...> or </function>")?;
            let (key, rest_of_param) =
                param.split_once('>').ok_or("unterminated parameter name")?;
            let (value, after) = rest_of_param
                .split_once("</parameter>")
                .ok_or("unterminated parameter")?;
            // The value sits on its own lines between the tags.
            let value = value.strip_prefix('\n').unwrap_or(value);
            let value = value.strip_suffix('\n').unwrap_or(value);
            arguments.insert(key.to_string(), Value::String(value.to_string()));
            rest = after;
        }
        Ok(Self {
            name: name.to_string(),
            arguments: Value::Object(arguments),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_calls_of_both_kinds() {
        let json = r#"{"name": "bash", "arguments": {"command": "ls"}}"#;
        let call = ToolCall::parse(&format!("\n{json}\n")).unwrap();
        assert_eq!(
            (call.name.as_str(), call.arguments["command"].as_str()),
            ("bash", Some("ls"))
        );

        let coder = "<function=bash>\n<parameter=command>\ndate\n</parameter>\n</function>";
        let call = ToolCall::parse(&format!("\n{coder}\n")).unwrap();
        assert_eq!(
            (call.name.as_str(), call.arguments["command"].as_str()),
            ("bash", Some("date"))
        );
    }
}
