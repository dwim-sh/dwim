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
    /// values are all strings. A value runs to the first `</parameter>` after
    /// which the rest of the block is well formed, so a value may itself hold
    /// a tool call, as a command that writes a test of one does. An error
    /// quotes the line the parser stopped on.
    pub fn parse(text: &str) -> Result<Self, Box<dyn Error>> {
        let text = text.trim();
        if text.starts_with('{') {
            return Ok(serde_json::from_str(text)?);
        }
        let rest = text.strip_prefix("<function=").ok_or_else(|| {
            stopped(
                "a tool call is a JSON object or a <function=...> block",
                text,
            )
        })?;
        let (name, rest) = rest
            .split_once('>')
            .ok_or_else(|| stopped("unterminated function name", rest))?;
        let parameters = parameters(rest)?;
        let arguments = parameters
            .into_iter()
            .map(|(key, value)| (key.to_string(), Value::String(value.to_string())))
            .collect();
        Ok(Self {
            name: name.to_string(),
            arguments: Value::Object(arguments),
        })
    }
}

const CLOSE: &str = "</parameter>";

/// A tag of a tool call.
#[derive(Clone, Copy, PartialEq)]
enum Tag {
    Function,
    FunctionEnd,
    Parameter,
    ParameterEnd,
}

/// How a `<function=...>` block parses from a place in it on: to its end,
/// or as a parameter and the place after the value's `</parameter>`, by
/// its number among the places.
#[derive(Clone, Copy)]
enum Step<'a> {
    End,
    Value {
        key: &'a str,
        value: &'a str,
        next: usize,
    },
}

/// Why a `<function=...>` block does not parse from a place in it on.
#[derive(Clone, Copy)]
enum Fault<'a> {
    /// What was wrong, and where in the block the parser stopped.
    At(&'static str, usize),
    /// A parameter, by name, that no `</parameter>` ends.
    Unterminated(&'a str),
}

impl Fault<'_> {
    /// The error the fault makes in the block `text`.
    fn describe(self, text: &str) -> String {
        match self {
            Fault::At(what, at) => stopped(what, &text[at..]),
            Fault::Unterminated(key) => format!("unterminated parameter {key}"),
        }
    }
}

/// Parses the `<parameter=name>` values of a `<function=...>` block, `text`.
/// A value ends at the first `</parameter>` after which the rest of the block
/// parses without repeating a parameter, preferring ends before which the
/// value's tags nest. The places the parse may go on from are parsed from the
/// last to the first, so each is parsed once and without recursion.
fn parameters(text: &str) -> std::result::Result<Vec<(&str, &str)>, String> {
    let tags: Vec<(usize, Tag)> = text
        .match_indices('<')
        .filter_map(|(at, _)| {
            let rest = &text[at..];
            let tag = if rest.starts_with("<function=") {
                Tag::Function
            } else if rest.starts_with("</function>") {
                Tag::FunctionEnd
            } else if rest.starts_with("<parameter=") {
                Tag::Parameter
            } else if rest.starts_with(CLOSE) {
                Tag::ParameterEnd
            } else {
                return None;
            };
            Some((at, tag))
        })
        .collect();
    let places: Vec<usize> = std::iter::once(0)
        .chain(
            tags.iter()
                .filter(|(_, tag)| *tag == Tag::ParameterEnd)
                .map(|(at, _)| at + CLOSE.len()),
        )
        .collect();
    let mut steps = vec![Ok(Step::End); places.len()];
    for i in (0..places.len()).rev() {
        steps[i] = step(text, &tags, &places, &steps, places[i]);
    }
    let mut values = Vec::new();
    let mut i = 0;
    loop {
        match steps[i].map_err(|fault| fault.describe(text))? {
            Step::End => return Ok(values),
            Step::Value { key, value, next } => {
                values.push((key, value));
                i = next;
            }
        }
    }
}

/// How the block `text`, whose `tags` and `places` `parameters` found, parses
/// from `from` on, given how it parses from each of the places after.
fn step<'a>(
    text: &'a str,
    tags: &[(usize, Tag)],
    places: &[usize],
    steps: &[std::result::Result<Step<'a>, Fault<'a>>],
    from: usize,
) -> std::result::Result<Step<'a>, Fault<'a>> {
    // Where a suffix of the block starts in it.
    let at = |rest: &str| text.len() - rest.len();
    let rest = text[from..].trim_start();
    if let Some(after) = rest.strip_prefix("</function>") {
        return if after.trim().is_empty() {
            Ok(Step::End)
        } else {
            Err(Fault::At("text after </function>", at(after)))
        };
    }
    let param = rest.strip_prefix("<parameter=").ok_or(Fault::At(
        "expected <parameter=...> or </function>",
        at(rest),
    ))?;
    let (key, body) = param
        .split_once('>')
        .ok_or(Fault::At("unterminated parameter name", at(param)))?;
    let start = at(body);
    // The fault of the first end tried after which the rest could go on
    // from a value, being empty or a tag that may follow one: that end is
    // the likeliest to be the value's, so its fault says best what is
    // wrong with the call.
    let mut likeliest = None;
    // The value ending at `at`, if the block parses after it.
    let mut end = |at: usize| {
        let next = places
            .binary_search(&(at + CLOSE.len()))
            .expect("every </parameter> is a place");
        // Whether the rest gives the value's parameter again.
        let mut later = Some(next);
        while let Some(i) = later {
            later = match steps[i] {
                Ok(Step::Value { key: k, next, .. }) if k != key => Some(next),
                Ok(Step::Value { .. }) => return None,
                _ => None,
            };
        }
        match steps[next] {
            Ok(_) => {
                let value = &text[start..at];
                // The value sits on its own lines between the tags.
                let value = value.strip_prefix('\n').unwrap_or(value);
                let value = value.strip_suffix('\n').unwrap_or(value);
                Some(Step::Value { key, value, next })
            }
            Err(fault) => {
                let rest = text[places[next]..].trim_start();
                if likeliest.is_none()
                    && (rest.is_empty()
                        || rest.starts_with("<parameter=")
                        || rest.starts_with("</function>"))
                {
                    likeliest = Some(fault);
                }
                None
            }
        }
    };
    // The ends before which the value's tags nest, in one walk over them,
    // up to the first closing tag that does not match, after which none
    // does; then the others, from the first.
    let first = tags.partition_point(|&(at, _)| at < start);
    let mut opened = Vec::new();
    let mut open = Vec::new();
    let mut walked = tags.len();
    for (i, &(at, tag)) in tags.iter().enumerate().skip(first) {
        let matched = match tag {
            Tag::Function | Tag::Parameter => {
                opened.push(tag);
                true
            }
            Tag::FunctionEnd => opened.pop() == Some(Tag::Function),
            Tag::ParameterEnd => {
                if opened.is_empty() {
                    if let Some(step) = end(at) {
                        return Ok(step);
                    }
                } else {
                    open.push(at);
                }
                opened.pop() == Some(Tag::Parameter)
            }
        };
        if !matched {
            walked = i + 1;
            break;
        }
    }
    let later = tags[walked..]
        .iter()
        .filter(|(_, tag)| *tag == Tag::ParameterEnd)
        .map(|(at, _)| *at);
    for at in open.into_iter().chain(later) {
        if let Some(step) = end(at) {
            return Ok(step);
        }
    }
    // Every end was tried; the error is from the likeliest of them, or
    // else from the last, which is the last place.
    if let Some(fault) = likeliest {
        return Err(fault);
    }
    match steps.last() {
        Some(&Err(fault)) if places[places.len() - 1] > start => Err(fault),
        _ => Err(Fault::Unterminated(key)),
    }
}

/// Most characters of a line an error in parsing a tool call quotes.
const QUOTE: usize = 80;

/// An error in parsing a tool call, with the line of `rest` it stopped on,
/// cut to its first `QUOTE` characters.
fn stopped(what: &str, rest: &str) -> String {
    let line = rest.trim_start().lines().next().unwrap_or("");
    match line.trim() {
        "" => format!("{what}, at the end of the call"),
        line => match line.char_indices().nth(QUOTE) {
            Some((cut, _)) => format!("{what}, at {:?}...", &line[..cut]),
            None => format!("{what}, at {line:?}"),
        },
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

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

    fn parsed(text: &str) -> (String, Value) {
        let call = ToolCall::parse(text).unwrap();
        (call.name, call.arguments)
    }

    fn rejected(text: &str) -> String {
        ToolCall::parse(text).unwrap_err().to_string()
    }

    #[test]
    fn parses_a_call_whose_value_holds_a_call() {
        // A command that writes a test of a tool call, as the model does.
        let inner = "<function=bash>\n<parameter=command>\ndate\n</parameter>\n</function>";
        let command = format!(
            "cat > test.rs <<'EOF'\nlet call = \"{}\";\nEOF",
            inner.replace('\n', "\\n")
        );
        let text =
            format!("<function=bash>\n<parameter=command>\n{command}\n</parameter>\n</function>");
        assert_eq!(
            parsed(&text),
            ("bash".into(), json!({ "command": command }))
        );
        // The same with the tags on their own lines, as a heredoc holds them.
        let command = format!("cat > call.txt <<'EOF'\n{inner}\nEOF");
        let text =
            format!("<function=bash>\n<parameter=command>\n{command}\n</parameter>\n</function>");
        assert_eq!(
            parsed(&text),
            ("bash".into(), json!({ "command": command }))
        );
        // And with a call of two parameters inside, whose first
        // `</parameter>` is followed by a `<parameter=` of its own.
        let inner = "<function=read>\n<parameter=path>\na.txt\n</parameter>\n<parameter=start>\n3\n</parameter>\n</function>";
        let command = format!("cat > call.txt <<'EOF'\n{inner}\nEOF");
        let text =
            format!("<function=bash>\n<parameter=command>\n{command}\n</parameter>\n</function>");
        assert_eq!(
            parsed(&text),
            ("bash".into(), json!({ "command": command }))
        );
    }

    #[test]
    fn keeps_apart_values_whose_lone_tags_would_pair_up() {
        // A `<function=` in one value and a `</function>` in the next, as
        // in an edit of the parser's own source, are two values, not one
        // that holds a call.
        let text = "<function=read>\n<parameter=path>\nx<function=\n</parameter>\n<parameter=start>\n</function>\n</parameter>\n</function>";
        assert_eq!(
            parsed(text),
            (
                "read".into(),
                json!({ "path": "x<function=", "start": "</function>" })
            )
        );
    }

    #[test]
    fn keeps_a_value_whole_rather_than_give_its_parameter_twice() {
        // A heredoc that holds a `</parameter>` and a `<parameter=command>`
        // is part of the command, not a second command in its place.
        let command = "cat <<EOF\n</parameter>\n<parameter=command>\nfoo\nEOF";
        let text =
            format!("<function=bash>\n<parameter=command>\n{command}\n</parameter>\n</function>");
        assert_eq!(
            parsed(&text),
            ("bash".into(), json!({ "command": command }))
        );
        // So a call that does give a parameter twice is read as one value
        // that holds the other, which no tool takes for what was meant.
        assert_eq!(
            parsed(
                "<function=bash>\n<parameter=command>\nls\n</parameter>\n<parameter=command>\npwd\n</parameter>\n</function>"
            ),
            (
                "bash".into(),
                json!({ "command": "ls\n</parameter>\n<parameter=command>\npwd" })
            )
        );
    }

    #[test]
    fn parses_a_long_value_of_many_tags() {
        // A command that writes a file of thousands of tool calls, each of
        // whose `</parameter>` the value does not end at: the tags are
        // looked at once for each place the parse may go on from, rather
        // than once for each of their ends.
        let calls =
            "<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>\n".repeat(5_000);
        let command = format!("cat > calls.txt <<'EOF'\n{calls}EOF");
        let text =
            format!("<function=bash>\n<parameter=command>\n{command}\n</parameter>\n</function>");
        assert_eq!(
            parsed(&text),
            ("bash".into(), json!({ "command": command }))
        );
    }

    #[test]
    fn parses_a_value_of_many_parameters_on_a_small_stack() {
        // A command that writes thousands of parameters, each of which the
        // parse may go on from, on a thread with the stack the harness's
        // worker has: it must parse rather than overflow the stack.
        let pairs: String = (0..20_000)
            .map(|i| format!("</parameter>\n<parameter=p{i}>\n"))
            .collect();
        let text = format!(
            "<function=bash>\n<parameter=command>\ncat <<'EOF'\n{pairs}EOF\n</parameter>\n</function>"
        );
        let parsed = std::thread::Builder::new()
            .stack_size(2 << 20)
            .spawn(move || ToolCall::parse(&text).is_ok())
            .unwrap()
            .join()
            .unwrap();
        assert!(parsed);
    }

    #[test]
    fn parses_several_parameters() {
        let text = "<function=read>\n<parameter=path>\na.txt\n</parameter>\n<parameter=start>\n3\n</parameter>\n</function>";
        assert_eq!(
            parsed(text),
            ("read".into(), json!({ "path": "a.txt", "start": "3" }))
        );
        assert_eq!(
            parsed("<function=bash>\n</function>"),
            ("bash".into(), json!({}))
        );
    }

    #[test]
    fn says_where_a_malformed_call_stopped() {
        assert_eq!(
            rejected("<bash>ls</bash>"),
            "a tool call is a JSON object or a <function=...> block, at \"<bash>ls</bash>\""
        );
        assert_eq!(
            rejected("<function=bash\nls"),
            "unterminated function name, at \"bash\""
        );
        assert_eq!(
            rejected("<function=bash>\nls\n</function>"),
            "expected <parameter=...> or </function>, at \"ls\""
        );
        assert_eq!(
            rejected("<function=bash>\n<parameter=command>\nls"),
            "unterminated parameter command"
        );
        assert_eq!(
            rejected("<function=bash>\n<parameter=command>\nls\n</parameter>"),
            "expected <parameter=...> or </function>, at the end of the call"
        );
        assert_eq!(
            rejected("<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>\nDone."),
            "text after </function>, at \"Done.\""
        );
        // A long line is quoted only as far as its start.
        let long = format!("<bash>{}", "é".repeat(10_000));
        assert_eq!(
            rejected(&long),
            format!(
                "a tool call is a JSON object or a <function=...> block, at \"<bash>{}\"...",
                "é".repeat(QUOTE - 6)
            )
        );
        // Spaces or a carriage return before the text are passed over too.
        for gap in ["  \n", "\r\n", " \t"] {
            assert_eq!(
                rejected(&format!(
                    "<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>{gap}Done."
                )),
                "text after </function>, at \"Done.\""
            );
        }
        // A value's `</parameter>` that does not end the block is passed over,
        // and the error is where the parser stopped after the last one.
        assert_eq!(
            rejected(
                "<function=bash>\n<parameter=command>\necho '</parameter>'\n</parameter>\n</function>\nDone."
            ),
            "text after </function>, at \"Done.\""
        );
        // Text after the call that holds a `</parameter>` of its own is
        // quoted from its start, where the call's own end is followed by
        // what could go on from it, rather than after the stray tag.
        assert_eq!(
            rejected(
                "<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>\nI ran </parameter> above."
            ),
            "text after </function>, at \"I ran </parameter> above.\""
        );
        // A call whose value holds a call, but that is missing its own
        // `</function>`, ends early, not at the text after the inner call.
        assert_eq!(
            rejected(
                "<function=bash>\n<parameter=command>\ncat <<'EOF'\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>\nEOF\n</parameter>"
            ),
            "expected <parameter=...> or </function>, at the end of the call"
        );
    }
}
