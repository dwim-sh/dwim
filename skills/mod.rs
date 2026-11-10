//! Skills: instructions for a kind of task, one per directory, each a
//! `SKILL.md` file. The file holds YAML-like frontmatter between a first
//! line of `---` and the next line of `---`, with `name` and
//! `description` lines, then the instructions. Skills are looked for in
//! the project's `.dwim/skills`, `.agents/skills` and `.claude/skills`
//! directories, then the user's: `$XDG_CONFIG_HOME/dwim/skills` (or
//! `~/.config/dwim/skills` when the variable is not set),
//! `~/.agents/skills` and `~/.claude/skills`, in that order.

use std::{
    collections::HashMap,
    env, fs,
    path::{Path, PathBuf},
};

/// The file a skill is.
pub const FILE: &str = "SKILL.md";

/// Most files of a skill's directory the loaded skill lists.
pub const MAX_FILES: usize = 20;

/// A skill: its name, what it is for, and the directory its SKILL.md is in.
#[derive(Clone, Debug, PartialEq)]
pub struct Skill {
    pub name: String,
    pub description: String,
    pub dir: PathBuf,
}

/// The skills found, and the SKILL.md files skipped, each with why: one
/// that does not parse, or one shadowed by a skill of its name found first.
#[derive(Debug, Default)]
pub struct Found {
    pub skills: Vec<Skill>,
    pub skipped: Vec<(PathBuf, String)>,
}

/// The frontmatter and the body of a SKILL.md, or `None` when the file has
/// no frontmatter or its frontmatter is never closed.
fn split(text: &str) -> Option<(&str, &str)> {
    let mut lines = text.split_inclusive('\n');
    let first = lines.next()?;
    if first.trim() != "---" {
        return None;
    }
    let start = first.len();
    let mut end = start;
    for line in lines {
        if line.trim() == "---" {
            return Some((&text[start..end], &text[end + line.len()..]));
        }
        end += line.len();
    }
    None
}

/// The value without the quotes around it, if it starts and ends with the
/// same one, `"` or `'`, and with what is escaped inside them undone: `''`
/// within single quotes, and `\"` and `\\` within double quotes.
fn unquote(value: &str) -> String {
    let Some(first) = value.chars().next() else {
        return value.to_string();
    };
    if !(first == '"' || first == '\'') || value.len() < 2 || !value.ends_with(first) {
        return value.to_string();
    }
    let inner = &value[1..value.len() - 1];
    if first == '\'' {
        return inner.replace("''", "'");
    }
    let mut unescaped = String::with_capacity(inner.len());
    let mut chars = inner.chars();
    while let Some(c) = chars.next() {
        match (c, chars.clone().next()) {
            ('\\', Some(next @ ('"' | '\\'))) => {
                unescaped.push(next);
                chars.next();
            }
            _ => unescaped.push(c),
        }
    }
    unescaped
}

/// A run of whitespace in a value, collapsed to one space.
fn collapse(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// A skill, from the text of the SKILL.md in `dir`, or what is wrong with
/// it: no frontmatter, no description, or a name that is not lowercase
/// letters, digits and dashes, or is longer than 64 characters. The name
/// is the directory's when the frontmatter gives none, or gives it empty.
pub fn parse(text: &str, dir: &Path) -> Result<Skill, String> {
    let Some((frontmatter, _)) = split(text) else {
        return Err("no frontmatter".to_string());
    };
    // Each key's value as written, with the lines continuing it, and
    // whether it is a block (`>` or `|`), whose text is taken as it is.
    let mut values: HashMap<String, (String, bool)> = HashMap::new();
    let mut last: Option<String> = None;
    for line in frontmatter.lines() {
        if line.starts_with([' ', '\t']) {
            if let Some(key) = &last {
                let continued = collapse(line);
                if !continued.is_empty() {
                    let (value, _) = values.get_mut(key).unwrap();
                    *value = if value.is_empty() {
                        continued
                    } else {
                        format!("{value} {continued}")
                    };
                }
            }
            continue;
        }
        let Some(colon) = line.find(':') else {
            continue;
        };
        let key = line[..colon].trim();
        if key.is_empty() {
            continue;
        }
        let raw = collapse(&line[colon + 1..]);
        let value = if matches!(raw.as_str(), ">" | ">-" | "|" | "|-") {
            (String::new(), true)
        } else {
            (raw, false)
        };
        last = Some(key.to_string());
        values.insert(key.to_string(), value);
    }
    let values: HashMap<String, String> = values
        .into_iter()
        .map(|(key, (value, block))| (key, if block { value } else { unquote(&value) }))
        .collect();
    let Some(description) = values.get("description").filter(|d| !d.is_empty()) else {
        return Err("no description".to_string());
    };
    let name = values
        .get("name")
        .filter(|name| !name.is_empty())
        .cloned()
        .unwrap_or_else(|| dir.file_name().unwrap_or_default().to_string_lossy().into());
    if name.is_empty() {
        return Err("name \"\" is not lowercase letters, digits and dashes".to_string());
    }
    if name.len() > 64 {
        return Err(format!("name \"{name}\" is longer than 64 characters"));
    }
    if !name
        .chars()
        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
    {
        return Err(format!(
            "name \"{name}\" is not lowercase letters, digits and dashes"
        ));
    }
    Ok(Skill {
        name,
        description: description.clone(),
        dir: dir.to_path_buf(),
    })
}

/// The directories skills are looked for in, in order: the project's
/// `.dwim/skills`, `.agents/skills` and `.claude/skills` under `dir`, then
/// the user's `dwim/skills` under `$XDG_CONFIG_HOME` or `~/.config`, and
/// `.agents/skills` and `.claude/skills` under the home directory.
pub fn roots(dir: &Path) -> Vec<PathBuf> {
    let mut roots = vec![
        dir.join(".dwim/skills"),
        dir.join(".agents/skills"),
        dir.join(".claude/skills"),
    ];
    if let Some(home) = dirs::home_dir() {
        let config = env::var("XDG_CONFIG_HOME")
            .ok()
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| home.join(".config"));
        for root in [
            config.join("dwim/skills"),
            home.join(".agents/skills"),
            home.join(".claude/skills"),
        ] {
            if !roots.contains(&root) {
                roots.push(root);
            }
        }
    }
    roots
}

/// The skills in `roots`, each a directory of skill directories, sorted
/// by name, with the SKILL.md files skipped and why. Hidden directories
/// are left out, and of two skills of a name the one in the earlier root
/// wins, the other skipped as shadowed by it; a root that cannot be read
/// is empty.
pub fn scan(roots: &[PathBuf]) -> Found {
    let mut found = Found::default();
    for root in roots {
        let Ok(entries) = fs::read_dir(root) else {
            continue;
        };
        let mut entries = entries
            .filter_map(|e| e.ok())
            .filter(|e| {
                let name = e.file_name().to_string_lossy().into_owned();
                !name.starts_with('.')
            })
            .collect::<Vec<_>>();
        entries.sort_by_key(|e| e.file_name());
        for entry in entries {
            let path = entry.path().join(FILE);
            if !path.is_file() {
                continue;
            }
            let text = match fs::read_to_string(&path) {
                Ok(text) => text,
                Err(error) => {
                    found.skipped.push((path, format!("cannot read: {error}")));
                    continue;
                }
            };
            match parse(&text, &entry.path()) {
                Ok(skill) => match found.skills.iter().find(|s| s.name == skill.name) {
                    Some(first) => found.skipped.push((
                        path,
                        format!("shadowed by {}", first.dir.join(FILE).display()),
                    )),
                    None => found.skills.push(skill),
                },
                Err(error) => found.skipped.push((path, error)),
            }
        }
    }
    found.skills.sort_by(|a, b| a.name.cmp(&b.name));
    found
}

/// The skills of the project in `dir` and the user's: those in [`roots`].
pub fn find(dir: &Path) -> Found {
    scan(&roots(dir))
}

/// The message invoking the skill `name` with `arguments` sends: the skill's
/// instructions as [`load`] gives them, followed by a blank line and the
/// arguments when they are not empty.
pub fn invoke(skills: &[Skill], name: &str, arguments: &str) -> Result<String, String> {
    let skill = skills
        .iter()
        .find(|skill| skill.name == name)
        .ok_or_else(|| format!("no skill called '{name}'"))?;
    let mut message = load(skill)?;
    if !arguments.is_empty() {
        message.push_str(&format!("\n\n{arguments}"));
    }
    Ok(message)
}

/// The skill's instructions as the model gets them: its `SKILL.md`'s body
/// after the frontmatter, whole when there is no frontmatter, named with
/// the directory it is in, with the files of that directory beside it.
pub fn load(skill: &Skill) -> Result<String, String> {
    let path = skill.dir.join(FILE);
    let text =
        fs::read_to_string(&path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let body = split(&text)
        .map(|(_, body)| body)
        .unwrap_or(&text)
        .trim()
        .to_string();
    let mut files = Vec::new();
    if let Ok(entries) = fs::read_dir(&skill.dir) {
        let mut entries = entries
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|name| name != FILE && !name.starts_with('.'))
            .collect::<Vec<_>>();
        entries.sort();
        for name in entries {
            files.push(if skill.dir.join(&name).is_dir() {
                format!("{name}/")
            } else {
                name
            });
        }
    }
    let files = if files.len() > MAX_FILES {
        let mut shown = files[..MAX_FILES].join(", ");
        shown.push_str(&format!(" and {} more", files.len() - MAX_FILES));
        shown
    } else {
        files.join(", ")
    };
    let tail = if files.is_empty() {
        String::from("No other files are there.")
    } else {
        format!(
            "Paths in these instructions are relative to {}. Files there besides {FILE}: {files}",
            skill.dir.display()
        )
    };
    Ok(format!(
        "<skill name=\"{}\" directory=\"{}\">\n{body}\n</skill>\n\n{tail}",
        skill.name,
        skill.dir.display()
    ))
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::{Path, PathBuf},
    };

    use super::{FILE, Skill, invoke, load, parse, roots, scan};

    #[test]
    fn parses_the_frontmatter() {
        let dir = Path::new("x/demo");
        let skill = parse(
            "---\nname: demo\ndescription: Does a thing.\n---\nDo the thing.\n",
            dir,
        )
        .unwrap();
        assert_eq!(
            skill,
            Skill {
                name: "demo".into(),
                description: "Does a thing.".into(),
                dir: dir.to_path_buf(),
            }
        );

        let dir = Path::new("x/other-skill");
        let skill = parse("---\ndescription: Does another.\n---\n", dir).unwrap();
        assert_eq!(skill.name, "other-skill");

        let skill = parse(
            "---\nname: quoted\ndescription: \"Quoted.\"\n---\n",
            Path::new("x/quoted"),
        )
        .unwrap();
        assert_eq!(skill.description, "Quoted.");

        let skill = parse(
            "---\nname: block\ndescription: >\n  First line.\n  Second line.\n---\n",
            Path::new("x/block"),
        )
        .unwrap();
        assert_eq!(skill.description, "First line. Second line.");

        let skill = parse(
            "---\nname: continued\ndescription: A\n  continued\n---\n",
            Path::new("x/continued"),
        )
        .unwrap();
        assert_eq!(skill.description, "A continued");

        // A quoted value continued on the next line loses its quotes, and
        // what is escaped in it is undone.
        let skill = parse(
            "---\nname: wrapped\ndescription: \"Use when \\\"asked\\\"\n  to deploy.\"\n---\n",
            Path::new("x/wrapped"),
        )
        .unwrap();
        assert_eq!(skill.description, "Use when \"asked\" to deploy.");
        let skill = parse(
            "---\nname: single\ndescription: 'It''s here.'\n---\n",
            Path::new("x/single"),
        )
        .unwrap();
        assert_eq!(skill.description, "It's here.");
        // A block's text is kept as it is, quotes and all.
        let skill = parse(
            "---\nname: literal\ndescription: |\n  \"Quoted.\"\n---\n",
            Path::new("x/literal"),
        )
        .unwrap();
        assert_eq!(skill.description, "\"Quoted.\"");

        // An empty name is the directory's.
        for empty in ["name:", "name: \"\""] {
            let skill = parse(
                &format!("---\n{empty}\ndescription: Has one.\n---\n"),
                Path::new("x/unnamed"),
            )
            .unwrap();
            assert_eq!(skill.name, "unnamed");
        }

        let skill = parse(
            "---\nname: keyed\ndescription: Has one.\nlicense: MIT\n---\n",
            Path::new("x/keyed"),
        )
        .unwrap();
        assert_eq!(skill.name, "keyed");
        assert_eq!(skill.description, "Has one.");
    }

    #[test]
    fn refuses_bad_frontmatter() {
        let dir = Path::new("x/bad");
        assert_eq!(
            parse("name: x\ndescription: Has one.\n", dir).unwrap_err(),
            "no frontmatter"
        );
        assert_eq!(
            parse("---\nname: x\ndescription: Has one.\n", dir).unwrap_err(),
            "no frontmatter"
        );
        assert_eq!(
            parse("---\nname: x\n---\n", dir).unwrap_err(),
            "no description"
        );
        assert_eq!(
            parse("---\nname: Has Space\ndescription: Has one.\n---\n", dir).unwrap_err(),
            "name \"Has Space\" is not lowercase letters, digits and dashes"
        );
        assert_eq!(
            parse(
                &format!(
                    "---\nname: {}\ndescription: Has one.\n---\n",
                    "a".repeat(65)
                ),
                dir
            )
            .unwrap_err(),
            "name \"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\" is longer than 64 characters"
        );
    }

    #[test]
    fn scans_the_roots_in_order() {
        let dir = std::env::temp_dir().join(format!("dwim-skills-{}", std::process::id()));
        fs::remove_dir_all(&dir).ok();
        for sub in ["a/one", "a/two", "b/one", "b/.hidden", "b/bad"] {
            fs::create_dir_all(dir.join(sub)).unwrap();
        }
        fs::write(
            dir.join("a/one/SKILL.md"),
            "---\nname: one\ndescription: From a.\n---\n",
        )
        .unwrap();
        fs::write(
            dir.join("a/two/SKILL.md"),
            "---\nname: two\ndescription: From a too.\n---\n",
        )
        .unwrap();
        fs::write(
            dir.join("b/one/SKILL.md"),
            "---\nname: one\ndescription: From b.\n---\n",
        )
        .unwrap();
        fs::write(
            dir.join("b/.hidden/SKILL.md"),
            "---\nname: hidden\ndescription: Hidden.\n---\n",
        )
        .unwrap();
        fs::write(dir.join("b/bad/SKILL.md"), "---\nname: bad\n---\n").unwrap();

        let found = scan(&[dir.join("a"), dir.join("b")]);
        assert_eq!(
            found.skills,
            vec![
                Skill {
                    name: "one".into(),
                    description: "From a.".into(),
                    dir: dir.join("a/one"),
                },
                Skill {
                    name: "two".into(),
                    description: "From a too.".into(),
                    dir: dir.join("a/two"),
                },
            ]
        );
        assert_eq!(
            found.skipped,
            vec![
                (dir.join("b/bad/SKILL.md"), "no description".to_string()),
                (
                    dir.join("b/one/SKILL.md"),
                    format!("shadowed by {}", dir.join("a/one/SKILL.md").display())
                ),
            ]
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn loads_the_instructions() {
        let dir = std::env::temp_dir().join(format!("dwim-skill-load-{}", std::process::id()));
        fs::remove_dir_all(&dir).ok();
        let one = dir.join("one");
        let two = dir.join("two");
        fs::create_dir_all(one.join("scripts")).unwrap();
        fs::create_dir_all(&two).unwrap();
        fs::write(
            one.join("SKILL.md"),
            "---\nname: one\ndescription: One.\n---\nDo one.\n",
        )
        .unwrap();
        fs::write(one.join("scripts/run.sh"), "").unwrap();
        fs::write(one.join("a.txt"), "").unwrap();
        fs::write(one.join(".hidden"), "").unwrap();
        fs::write(
            two.join("SKILL.md"),
            "No frontmatter here.\nJust the body.\n",
        )
        .unwrap();
        let one = Skill {
            name: "one".into(),
            description: "One.".into(),
            dir: one,
        };
        let text = load(&one).unwrap();
        assert!(text.starts_with("<skill name=\"one\" directory="), "{text}");
        assert!(text.contains("Do one.\n</skill>"), "{text}");
        assert!(
            text.ends_with(
                &format!(
                    "Paths in these instructions are relative to {}. Files there besides {FILE}: a.txt, scripts/",
                    one.dir.display()
                )
            ),
            "{text}"
        );
        let two = Skill {
            name: "two".into(),
            description: "Two.".into(),
            dir: two,
        };
        let text = load(&two).unwrap();
        assert!(
            text.contains("No frontmatter here.\nJust the body.\n</skill>"),
            "{text}"
        );
        assert!(text.ends_with("No other files are there."), "{text}");
        let missing = Skill {
            name: "gone".into(),
            description: "Gone.".into(),
            dir: dir.join("gone"),
        };
        let err = load(&missing).unwrap_err();
        assert!(err.starts_with("cannot read "), "{err}");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn invokes_a_skill_with_arguments() {
        let dir = std::env::temp_dir().join(format!("dwim-skill-invoke-{}", std::process::id()));
        fs::remove_dir_all(&dir).ok();
        let one = dir.join("one");
        fs::create_dir_all(&one).unwrap();
        fs::write(
            one.join("SKILL.md"),
            "---\nname: one\ndescription: One.\n---\nDo one.\n",
        )
        .unwrap();
        let one = Skill {
            name: "one".into(),
            description: "One.".into(),
            dir: one,
        };
        let message = invoke(std::slice::from_ref(&one), "one", "x").unwrap();
        assert!(
            message.starts_with("<skill name=\"one\" directory="),
            "{message}"
        );
        assert!(message.ends_with("\n\nx"), "{message}");
        let message = invoke(std::slice::from_ref(&one), "one", "").unwrap();
        assert!(!message.ends_with("\n\n"), "{message}");
        assert_eq!(
            invoke(&[one], "two", "x").unwrap_err(),
            "no skill called 'two'"
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn roots_start_in_the_project() {
        let roots = roots(Path::new("/p"));
        assert!(roots.starts_with(&[
            PathBuf::from("/p/.dwim/skills"),
            PathBuf::from("/p/.agents/skills"),
            PathBuf::from("/p/.claude/skills"),
        ]));
    }
}
