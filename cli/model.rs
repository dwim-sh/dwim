//! The model's size as the command line shows it, in words.

use dwim::inference::Size;

/// `size` as words: the memory the model takes, and the tokens of context
/// and the memory they take.
pub fn describe(size: &Size) -> String {
    if size.sessions == 1 {
        return format!(
            "model: {} · context: {} tokens ({})",
            bytes(size.model),
            size.context,
            bytes(size.conversation)
        );
    }
    format!(
        "model: {} · {} sessions · context: {} tokens ({}) each",
        bytes(size.model),
        size.sessions,
        size.context,
        bytes(size.conversation)
    )
}

/// A byte count for humans, in decimal units.
pub fn bytes(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["KB", "MB", "GB", "TB"];
    if bytes < 1000 {
        return format!("{bytes} B");
    }
    let mut value = bytes as f64 / 1000.0;
    let mut unit = 0;
    while value >= 1000.0 && unit < UNITS.len() - 1 {
        value /= 1000.0;
        unit += 1;
    }
    format!("{value:.1} {}", UNITS[unit])
}
