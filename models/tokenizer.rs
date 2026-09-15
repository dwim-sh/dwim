use std::{collections::HashMap, fs, path::Path};

use fancy_regex::Regex;
use serde_json::Value;

use crate::Result;

/// A byte-level BPE tokenizer, loaded from a Hugging Face `tokenizer.json`.
///
/// Text is first split into words by a regular expression. Each word starts
/// out as one token per UTF-8 byte, and adjacent tokens are then merged
/// pairwise, lowest-ranked merge first, until no merge applies.
pub struct Tokenizer {
    /// Splits text into words.
    pattern: Regex,
    /// Bytes of every token, indexed by id.
    tokens: Vec<Vec<u8>>,
    /// Token id of every single byte.
    bytes: [u32; 256],
    /// For each pair of adjacent tokens that can merge: the merge's rank and
    /// the merged token.
    merges: HashMap<(u32, u32), (usize, u32)>,
    /// Ids of special tokens such as `<|im_start|>`, by content.
    special: HashMap<String, u32>,
}

impl Tokenizer {
    pub fn load(path: &Path) -> Result<Self> {
        let json: Value = serde_json::from_str(&fs::read_to_string(path)?)?;

        // Byte-level vocabularies write each byte as a printable character, so
        // decode every token back into the bytes it stands for.
        let to_byte: HashMap<char, u8> = byte_chars()
            .iter()
            .enumerate()
            .map(|(byte, &c)| (c, byte as u8))
            .collect();
        let vocab = json["model"]["vocab"].as_object().ok_or("missing vocabulary")?;
        let mut ids = HashMap::new();
        let mut tokens = Vec::new();
        for (text, id) in vocab {
            let id = id.as_u64().ok_or("invalid token id")? as usize;
            let bytes = text
                .chars()
                .map(|c| to_byte.get(&c).copied().ok_or("invalid byte-level token"))
                .collect::<std::result::Result<Vec<u8>, _>>()?;
            if tokens.len() <= id {
                tokens.resize(id + 1, Vec::new());
            }
            tokens[id] = bytes;
            ids.insert(text.as_str(), id as u32);
        }

        let mut bytes = [0; 256];
        for (byte, c) in byte_chars().iter().enumerate() {
            bytes[byte] = *ids.get(c.to_string().as_str()).ok_or("missing byte token")?;
        }

        let mut merges = HashMap::new();
        let list = json["model"]["merges"].as_array().ok_or("missing merges")?;
        for (rank, merge) in list.iter().enumerate() {
            // Merges are either ["a", "b"] pairs or "a b" strings.
            let (a, b) = match merge {
                Value::Array(pair) => (pair[0].as_str(), pair[1].as_str()),
                Value::String(pair) => match pair.split_once(' ') {
                    Some((a, b)) => (Some(a), Some(b)),
                    None => (None, None),
                },
                _ => (None, None),
            };
            let (a, b) = a.zip(b).ok_or("invalid merge")?;
            let merged = format!("{a}{b}");
            let (Some(&a), Some(&b), Some(&merged)) =
                (ids.get(a), ids.get(b), ids.get(merged.as_str()))
            else {
                return Err("merge refers to an unknown token".into());
            };
            merges.insert((a, b), (rank, merged));
        }

        let mut special = HashMap::new();
        for token in json["added_tokens"].as_array().ok_or("missing added tokens")? {
            let id = token["id"].as_u64().ok_or("invalid token id")? as usize;
            let content = token["content"].as_str().ok_or("invalid token")?;
            if tokens.len() <= id {
                tokens.resize(id + 1, Vec::new());
            }
            tokens[id] = content.as_bytes().to_vec();
            special.insert(content.to_string(), id as u32);
        }

        Ok(Self {
            pattern: Regex::new(split_pattern(&json["pre_tokenizer"]).ok_or("missing split pattern")?)?,
            tokens,
            bytes,
            merges,
            special,
        })
    }

    /// Encodes text into tokens. Special tokens in the text are treated as
    /// plain text; insert them by id with [`special`](Self::special), or
    /// encode text that is meant to contain them with
    /// [`encode_with_special`](Self::encode_with_special).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let mut out = Vec::new();
        self.encode_into(text, &mut out)?;
        Ok(out)
    }

    /// Encodes text like [`encode`](Self::encode), but with special tokens
    /// written in it, such as `<tool_call>`, encoded by id. For text of the
    /// chat template, not text from the user, who could otherwise end a
    /// turn.
    pub fn encode_with_special(&self, text: &str) -> Result<Vec<u32>> {
        let mut out = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            // The earliest special token in the text, and the longest one
            // if several start there.
            let next = self
                .special
                .iter()
                .filter_map(|(content, &id)| rest.find(content.as_str()).map(|at| (at, content.len(), id)))
                .min_by_key(|&(at, len, _)| (at, std::cmp::Reverse(len)));
            let Some((at, len, id)) = next else {
                self.encode_into(rest, &mut out)?;
                break;
            };
            self.encode_into(&rest[..at], &mut out)?;
            out.push(id);
            rest = &rest[at + len..];
        }
        Ok(out)
    }

    fn encode_into(&self, text: &str, out: &mut Vec<u32>) -> Result<()> {
        for word in self.pattern.find_iter(text) {
            self.merge(word?.as_str().as_bytes(), out);
        }
        Ok(())
    }

    /// Bytes a token stands for. A token can end partway through a UTF-8
    /// character.
    pub fn decode(&self, token: u32) -> &[u8] {
        &self.tokens[token as usize]
    }

    /// Id of a special token.
    pub fn special(&self, content: &str) -> Result<u32> {
        Ok(*self
            .special
            .get(content)
            .ok_or_else(|| format!("missing special token '{content}'"))?)
    }

    fn merge(&self, word: &[u8], out: &mut Vec<u32>) {
        let mut parts: Vec<u32> = word.iter().map(|&b| self.bytes[b as usize]).collect();
        while let Some((_, i, merged)) = parts
            .windows(2)
            .enumerate()
            .filter_map(|(i, pair)| {
                let &(rank, merged) = self.merges.get(&(pair[0], pair[1]))?;
                Some((rank, i, merged))
            })
            .min()
        {
            parts[i] = merged;
            parts.remove(i + 1);
        }
        out.extend(parts);
    }
}

/// GPT-2's mapping from bytes to printable characters, which byte-level
/// vocabularies are written in: printable bytes stand for themselves, and the
/// rest are shifted past 255 in order.
fn byte_chars() -> [char; 256] {
    let mut chars = ['\0'; 256];
    let mut next = 256;
    for (byte, c) in chars.iter_mut().enumerate() {
        let printable = matches!(byte, 33..=126 | 161..=172 | 174..=255);
        *c = if printable {
            char::from(byte as u8)
        } else {
            next += 1;
            char::from_u32(next - 1).unwrap()
        };
    }
    chars
}

/// Finds the regular expression a pre-tokenizer splits words with.
fn split_pattern(pre_tokenizer: &Value) -> Option<&str> {
    if pre_tokenizer["type"] == "Split" {
        return pre_tokenizer["pattern"]["Regex"].as_str();
    }
    pre_tokenizer["pretokenizers"]
        .as_array()?
        .iter()
        .find_map(split_pattern)
}
