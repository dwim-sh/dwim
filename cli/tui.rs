//! The terminal, driven like an inline chat UI, after Titania's: finished
//! output scrolls up into the terminal's own scrollback like any other
//! program's, while a live region at the bottom is redrawn in place.

use std::{
    fs::File,
    io::{self, Write},
    mem,
    ops::Range,
};

use crossterm::{
    cursor::{MoveDown, MoveToColumn, MoveUp},
    event::{
        DisableBracketedPaste, EnableBracketedPaste, KeyboardEnhancementFlags,
        PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
    },
    execute, queue,
    style::{Print, PrintStyledContent, StyledContent, Stylize},
    terminal::{self, BeginSynchronizedUpdate, Clear, ClearType, EndSynchronizedUpdate},
};
use dwim::sys;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

/// A line of styled text.
pub type Line = Vec<StyledContent<String>>;

/// Display width of a line.
pub fn width(line: &Line) -> usize {
    line.iter().map(|span| span.content().width()).sum()
}

/// Cuts a line down to at most `columns` wide.
pub fn truncate(line: &Line, columns: usize) -> Line {
    let mut left = columns;
    let mut truncated = Line::new();
    for span in line {
        let mut text = String::new();
        for c in span.content().chars() {
            let char_width = c.width().unwrap_or(0);
            if char_width > left {
                left = 0;
                break;
            }
            text.push(c);
            left -= char_width;
        }
        truncated.push(StyledContent::new(*span.style(), text));
        if left == 0 {
            break;
        }
    }
    truncated
}

/// `text` cut down to at most `columns` wide, with an ellipsis for what
/// is cut.
pub fn fit(text: &str, columns: usize) -> String {
    if text.width() <= columns {
        return text.to_string();
    }
    let mut left = columns.saturating_sub(1);
    let mut fitted = String::new();
    for c in text.chars() {
        let width = c.width().unwrap_or(0);
        if width > left {
            break;
        }
        fitted.push(c);
        left -= width;
    }
    fitted.truncate(fitted.trim_end().len());
    fitted.push('…');
    fitted
}

/// A row of a box `columns` wide: `content` between its sides, padded or
/// cut to fit.
pub fn boxed(content: Line, columns: usize) -> Line {
    let inner = columns.saturating_sub(4);
    let content = truncate(&content, inner);
    let pad = inner - width(&content);
    let mut line = vec![span("│ ").dark_grey()];
    line.extend(content);
    line.push(span(" ".repeat(pad)));
    line.push(span(" │").dark_grey());
    line
}

/// A styled piece of text.
pub fn span(text: impl Into<String>) -> StyledContent<String> {
    StyledContent::new(Default::default(), text.into())
}

/// The terminal: finished lines are printed into the scrollback, and a live
/// region below them (the line being generated, a status line, and the input
/// box) is redrawn in place.
///
/// A draw never blanks the live region first: it writes over only the rows
/// that changed, and sends the whole frame at once, so the terminal never
/// shows a frame half drawn, whether or not it supports synchronized updates.
pub struct Screen {
    /// Standard output, unbuffered, so that a frame is written whole.
    out: File,
    /// The live region as last drawn, a row each.
    drawn: Vec<Line>,
    /// Row and column the cursor was left at within the live region.
    cursor: (usize, usize),
    /// Terminal width at the last draw, or 0 before the first.
    columns: usize,
    /// Whether keyboard enhancement was pushed, and must be popped on exit.
    enhanced: bool,
}

impl Screen {
    pub fn new() -> io::Result<Self> {
        terminal::enable_raw_mode()?;
        let mut out = sys::terminal::unbuffered(&io::stdout())?;
        execute!(out, EnableBracketedPaste)?;
        // Lets Shift+Enter be told apart from Enter, where supported.
        let enhanced = terminal::supports_keyboard_enhancement().unwrap_or(false);
        if enhanced {
            execute!(
                out,
                PushKeyboardEnhancementFlags(KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES)
            )?;
        }
        Ok(Self {
            out,
            drawn: Vec::new(),
            cursor: (0, 0),
            columns: 0,
            enhanced,
        })
    }

    /// Terminal size, in columns and rows.
    pub fn size(&self) -> (usize, usize) {
        let (columns, rows) = terminal::size().unwrap_or((80, 24));
        (columns.max(1) as usize, rows.max(1) as usize)
    }

    /// Prints `lines` into the scrollback, then redraws the live region below
    /// them, leaving the cursor at `cursor` (row, column) within it.
    pub fn draw(
        &mut self,
        lines: &[Line],
        live: &[Line],
        cursor: (usize, usize),
    ) -> io::Result<()> {
        let (columns, rows) = self.size();
        // A line of the live region that wrapped would push it out of place
        // for the next draw, so they are cut to the terminal's width. A
        // region taller than the screen could not be moved back to its top,
        // so it loses rows from the top, where the cursor is not.
        let mut live: Vec<Line> = live.iter().map(|line| truncate(line, columns)).collect();
        if live.is_empty() {
            live.push(Line::new());
        }
        let cut = live.len().saturating_sub(rows);
        live.drain(..cut);
        let cursor = (
            cursor.0.saturating_sub(cut).min(live.len() - 1),
            cursor.1.min(columns - 1),
        );

        // If the terminal was resized, it may have rewrapped what was drawn
        // onto more rows, so the region is cleared and drawn whole, as it is
        // the first time.
        let resized = columns != self.columns;
        let row_count = |width: usize| width.div_ceil(columns).max(1);
        let up = self.drawn[..self.cursor.0]
            .iter()
            .map(|line| row_count(width(line)))
            .sum::<usize>()
            + self.cursor.1 / columns;

        let mut frame = Vec::new();
        queue!(frame, BeginSynchronizedUpdate, MoveToColumn(0))?;
        if up > 0 {
            queue!(frame, MoveUp(up as u16))?;
        }
        if resized {
            queue!(frame, Clear(ClearType::FromCursorDown))?;
        }
        let before: &[Line] = if resized { &[] } else { &self.drawn };
        // The row of the screen being written, counted from the region's top.
        let mut row = 0;
        for (i, line) in lines.iter().chain(&live).enumerate() {
            if i > 0 {
                queue!(frame, Print("\r\n"))?;
            }
            if before.get(row) == Some(line) {
                row += 1;
                continue;
            }
            let line_width = width(line);
            for span in line {
                queue!(frame, PrintStyledContent(span.clone()))?;
            }
            // A line as wide as the screen leaves the cursor on its last
            // column, where clearing would erase that column.
            if !line_width.is_multiple_of(columns) || line_width == 0 {
                queue!(frame, Clear(ClearType::UntilNewLine))?;
            }
            row += row_count(line_width);
        }
        // Rows of a region taller than this one are left below it.
        if before.len() > row {
            queue!(
                frame,
                MoveDown(1),
                MoveToColumn(0),
                Clear(ClearType::FromCursorDown),
                MoveUp(1)
            )?;
        }
        let below = live.len() - 1 - cursor.0;
        if below > 0 {
            queue!(frame, MoveUp(below as u16))?;
        }
        queue!(frame, MoveToColumn(cursor.1 as u16), EndSynchronizedUpdate)?;
        self.out.write_all(&frame)?;
        self.out.flush()?;

        self.drawn = live;
        self.cursor = cursor;
        self.columns = columns;
        Ok(())
    }
}

impl Drop for Screen {
    fn drop(&mut self) {
        if self.enhanced {
            let _ = execute!(self.out, PopKeyboardEnhancementFlags);
        }
        let _ = execute!(self.out, DisableBracketedPaste);
        let _ = terminal::disable_raw_mode();
    }
}

/// Wraps text into lines at most `width` columns wide, breaking between words
/// where possible.
///
/// Appending text only ever changes the last line, so the lines before it can
/// be printed while the text is still streaming in.
pub fn wrap(text: &str, width: usize) -> Vec<&str> {
    wrap_ranges(text, width)
        .into_iter()
        .map(|range| &text[range])
        .collect()
}

/// Wraps text like `wrap`, returning where in the text each line is.
fn wrap_ranges(text: &str, width: usize) -> Vec<Range<usize>> {
    let width = width.max(1);
    let mut lines = Vec::new();
    let mut pos = 0;
    for paragraph in text.split('\n') {
        let mut start = pos;
        let mut line_width = 0;
        for word in words(paragraph) {
            let word_width = word.width();
            if line_width + word_width <= width {
                line_width += word_width;
            } else if word.starts_with(' ') {
                lines.push(start..pos);
                start = pos + word.len();
                line_width = 0;
            } else {
                if line_width > 0 {
                    lines.push(start..pos);
                    start = pos;
                    line_width = 0;
                }
                for (i, c) in word.char_indices() {
                    let char_width = c.width().unwrap_or(0);
                    if line_width + char_width > width && line_width > 0 {
                        lines.push(start..pos + i);
                        start = pos + i;
                        line_width = 0;
                    }
                    line_width += char_width;
                }
            }
            pos += word.len();
        }
        lines.push(start..pos);
        pos += 1;
    }
    for line in &mut lines {
        line.end = line.start + text[line.clone()].trim_end_matches(' ').len();
    }
    lines
}

/// Splits text into alternating runs of spaces and of everything else.
fn words(text: &str) -> impl Iterator<Item = &str> {
    let mut rest = text;
    std::iter::from_fn(move || {
        let space = rest.starts_with(' ');
        let end = rest
            .find(|c: char| (c == ' ') != space)
            .unwrap_or(rest.len());
        let (word, tail) = rest.split_at(end);
        rest = tail;
        (!word.is_empty()).then_some(word)
    })
}

/// Makes text safe to print: tabs become spaces, line endings become `\n`,
/// and other control characters, which would move the cursor, are dropped.
pub fn sanitize(text: &str) -> String {
    text.replace("\r\n", "\n")
        .chars()
        .filter_map(|c| match c {
            '\t' => Some(' '),
            '\r' => Some('\n'),
            c if c.is_control() && c != '\n' => None,
            c => Some(c),
        })
        .collect()
}

/// A line editor's text and cursor.
#[derive(Default)]
pub struct Input {
    text: String,
    /// Byte offset of the cursor.
    cursor: usize,
}

impl Input {
    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    /// Replaces the text, with the cursor at the end.
    pub fn set(&mut self, text: String) {
        self.cursor = text.len();
        self.text = text;
    }

    /// Takes the text out, leaving the editor empty.
    pub fn take(&mut self) -> String {
        self.cursor = 0;
        mem::take(&mut self.text)
    }

    pub fn insert(&mut self, text: &str) {
        self.text.insert_str(self.cursor, text);
        self.cursor += text.len();
    }

    pub fn backspace(&mut self) {
        let start = self.prev(self.cursor);
        self.text.drain(start..self.cursor);
        self.cursor = start;
    }

    pub fn delete(&mut self) {
        let end = self.next(self.cursor);
        self.text.drain(self.cursor..end);
    }

    pub fn left(&mut self) {
        self.cursor = self.prev(self.cursor);
    }

    pub fn right(&mut self) {
        self.cursor = self.next(self.cursor);
    }

    /// Moves to the start of the current line.
    pub fn home(&mut self) {
        self.cursor = self.line_start();
    }

    /// Moves to the end of the current line.
    pub fn end(&mut self) {
        self.cursor = self.line_end();
    }

    /// Deletes from the start of the current line to the cursor.
    pub fn kill_to_start(&mut self) {
        let start = self.line_start();
        self.text.drain(start..self.cursor);
        self.cursor = start;
    }

    /// Deletes from the cursor to the end of the current line.
    pub fn kill_to_end(&mut self) {
        let end = self.line_end();
        self.text.drain(self.cursor..end);
    }

    /// Deletes the word before the cursor.
    pub fn delete_word(&mut self) {
        let before = &self.text[..self.cursor];
        let trimmed = before.trim_end_matches(' ');
        let start = trimmed.rfind([' ', '\n']).map_or(0, |i| i + 1);
        self.text.drain(start..self.cursor);
        self.cursor = start;
    }

    /// Lays the text out in lines at most `width` columns wide, returning
    /// them along with the cursor's row and column.
    pub fn layout(&self, width: usize) -> (Vec<&str>, (usize, usize)) {
        let lines = wrap_ranges(&self.text, width);
        // Spaces at the end of a line are dropped from it, but the cursor
        // still moves over them as they are typed, as far as the line is
        // wide.
        let row = lines
            .iter()
            .rposition(|line| line.start <= self.cursor)
            .unwrap_or(0);
        let line = &lines[row];
        let column = self.text[line.start..self.cursor].width().min(width);
        let lines = lines.into_iter().map(|line| &self.text[line]).collect();
        (lines, (row, column))
    }

    fn line_start(&self) -> usize {
        self.text[..self.cursor].rfind('\n').map_or(0, |i| i + 1)
    }

    fn line_end(&self) -> usize {
        self.text[self.cursor..]
            .find('\n')
            .map_or(self.text.len(), |i| self.cursor + i)
    }

    fn prev(&self, i: usize) -> usize {
        self.text[..i]
            .char_indices()
            .next_back()
            .map_or(0, |(i, _)| i)
    }

    fn next(&self, i: usize) -> usize {
        self.text[i..]
            .chars()
            .next()
            .map_or(i, |c| i + c.len_utf8())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrap_breaks_between_words() {
        assert_eq!(wrap("the quick brown fox", 10), ["the quick", "brown fox"]);
        assert_eq!(wrap("one\n\ntwo", 10), ["one", "", "two"]);
        assert_eq!(wrap("abcdefghij", 4), ["abcd", "efgh", "ij"]);
    }

    #[test]
    fn wrap_only_changes_last_line_when_appending() {
        let text = "`dwim` is a coding agent harness: a loop around a model that runs the tools \
                    it calls and feeds the results back. Supercalifragilisticexpialidocious   \
                    words    and\n\nparagraphs.";
        for width in 1..30 {
            let full = wrap(text, width);
            for end in (0..=text.len()).filter(|&end| text.is_char_boundary(end)) {
                let prefix = wrap(&text[..end], width);
                let stable = prefix.len() - 1;
                assert_eq!(
                    prefix[..stable],
                    full[..stable],
                    "width {width}, text {:?}",
                    &text[..end]
                );
            }
        }
    }

    #[test]
    fn layout_places_cursor() {
        let mut input = Input::default();
        input.set("abcdef".to_string());
        assert_eq!(input.layout(3), (vec!["abc", "def"], (1, 3)));
        input.left();
        assert_eq!(input.layout(3).1, (1, 2));
        input.set("hello world".to_string());
        assert_eq!(input.layout(8), (vec!["hello", "world"], (1, 5)));
        input.set("hello ".to_string());
        assert_eq!(input.layout(5), (vec!["hello", ""], (1, 0)));
        assert_eq!(input.layout(10), (vec!["hello"], (0, 6)));
        input.set("ab\ncd".to_string());
        input.home();
        assert_eq!(input.layout(10).1, (1, 0));
    }
}
