use crossterm::style::Color;

use crate::ui::Stylize;

pub(crate) fn report_title(text: &str) -> String {
    format!("{}", text.bold().with(Color::Yellow))
}

pub(crate) fn report_section(text: &str) -> String {
    format!("{}", text.bold().with(Color::Cyan))
}

pub(crate) fn report_label(text: &str) -> String {
    format!("{}", text.with(Color::DarkGrey))
}

pub(crate) fn short_sha(value: &str) -> &str {
    value.get(..12).unwrap_or(value)
}

pub(crate) fn truncate_for_summary(text: &str, max_chars: usize) -> String {
    let trimmed = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut chars = trimmed.chars();
    let truncated = chars.by_ref().take(max_chars).collect::<String>();
    if chars.next().is_some() {
        format!("{truncated}…")
    } else {
        truncated
    }
}

pub(crate) fn list_or_none(values: &[String]) -> String {
    if values.is_empty() {
        "none".to_string()
    } else {
        values.join(",")
    }
}

pub(crate) fn sequence_or_none(values: &[String]) -> String {
    if values.is_empty() {
        "none".to_string()
    } else {
        values.join(" -> ")
    }
}

pub(crate) fn shell_quote(value: &str) -> String {
    if value.is_empty() {
        return "''".to_string();
    }
    if value
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.' | '/' | ':' | '+'))
    {
        return value.to_string();
    }
    format!("'{}'", value.replace('\'', "'\\''"))
}
