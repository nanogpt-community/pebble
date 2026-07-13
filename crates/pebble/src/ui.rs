#![allow(dead_code)]
//! Shared terminal UI primitives for the Pebble REPL.
//!
//! This module concentrates the visual vocabulary of the interactive shell:
//! the welcome banner, status panels, colored badges for tool calls, turn
//! separators, prompt styling, and other reusable widgets. Keeping these in
//! one place means the REPL presents a consistent, cohesive look instead of
//! a grab-bag of ad-hoc `println!`s.

use crossterm::style::{Attribute, Color, ContentStyle};
use crossterm::terminal::size as terminal_size;
use std::ffi::OsString;
use std::fmt::{Display, Formatter, Write as _};
use std::io::IsTerminal;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

/// Width of framed panels (banner, status cards, tool-call boxes).
pub const PANEL_WIDTH: usize = 72;
static COLOR_OUTPUT_ENABLED: AtomicBool = AtomicBool::new(true);

pub trait Stylize: Sized {
    fn into_styled_text(self) -> StyledText;

    fn with(self, color: Color) -> StyledText {
        let mut styled = self.into_styled_text();
        styled.style.foreground_color = Some(color);
        styled
    }

    fn bold(self) -> StyledText {
        let mut styled = self.into_styled_text();
        styled.style.attributes.set(Attribute::Bold);
        styled
    }

    fn italic(self) -> StyledText {
        let mut styled = self.into_styled_text();
        styled.style.attributes.set(Attribute::Italic);
        styled
    }

    fn underlined(self) -> StyledText {
        let mut styled = self.into_styled_text();
        styled.style.attributes.set(Attribute::Underlined);
        styled
    }
}

pub struct StyledText {
    text: String,
    style: ContentStyle,
}

impl Display for StyledText {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        if COLOR_OUTPUT_ENABLED.load(Ordering::Relaxed) {
            Display::fmt(&self.style.apply(self.text.as_str()), formatter)
        } else {
            formatter.write_str(&self.text)
        }
    }
}

impl Stylize for StyledText {
    fn into_styled_text(self) -> StyledText {
        self
    }
}

impl Stylize for String {
    fn into_styled_text(self) -> StyledText {
        StyledText {
            text: self,
            style: ContentStyle::default(),
        }
    }
}

impl Stylize for &str {
    fn into_styled_text(self) -> StyledText {
        self.to_string().into_styled_text()
    }
}

impl Stylize for &String {
    fn into_styled_text(self) -> StyledText {
        self.clone().into_styled_text()
    }
}

/// Applies the standard terminal color conventions before any UI is rendered.
pub fn configure_color_output() {
    let enabled = should_enable_color_output(std::io::stdout().is_terminal(), |name| {
        std::env::var_os(name)
    });
    COLOR_OUTPUT_ENABLED.store(enabled, Ordering::Relaxed);
    crossterm::style::force_color_output(enabled);
}

#[must_use]
pub fn color_output_is_enabled() -> bool {
    COLOR_OUTPUT_ENABLED.load(Ordering::Relaxed)
}

fn should_enable_color_output(is_terminal: bool, value: impl Fn(&str) -> Option<OsString>) -> bool {
    if value("NO_COLOR").is_some()
        || value("CLICOLOR").is_some_and(|value| value == "0")
        || value("TERM").is_some_and(|value| value.eq_ignore_ascii_case("dumb"))
    {
        return false;
    }
    if value("CLICOLOR_FORCE").is_some_and(|value| !value.is_empty() && value != "0") {
        return true;
    }
    is_terminal
}

/// Unicode/ANSI palette used by the shell. Centralising the palette makes it
/// easy to tweak the whole look and feel in one place.
pub mod palette {
    use crossterm::style::Color;

    pub const ACCENT: Color = Color::Cyan;
    pub const ACCENT_DIM: Color = Color::DarkCyan;
    pub const BRAND: Color = Color::Magenta;
    pub const BRAND_DIM: Color = Color::DarkMagenta;
    pub const MUTED: Color = Color::DarkGrey;
    pub const OK: Color = Color::Green;
    pub const WARN: Color = Color::Yellow;
    pub const ERR: Color = Color::Red;
    pub const INFO: Color = Color::Blue;

    /// Color used to highlight the name of a tool call.
    pub const TOOL_NAME: Color = Color::Cyan;
    /// Color for arguments rendered inline next to a tool call.
    pub const TOOL_ARG: Color = Color::DarkGrey;
    /// Color for permission-mode badges.
    pub const PERMISSION: Color = Color::Yellow;
}

/// Compute the printable width of a string, ignoring ANSI escape sequences.
/// Falls back to a simple char count for anything that isn't an escape.
#[must_use]
pub fn visible_width(text: &str) -> usize {
    let mut width = 0usize;
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\u{1b}' {
            // Eat a CSI sequence (`ESC [ ... letter`).
            if chars.peek() == Some(&'[') {
                chars.next();
                for next in chars.by_ref() {
                    if next.is_ascii_alphabetic() {
                        break;
                    }
                }
                continue;
            }
        }
        width += 1;
    }
    width
}

/// A single row to place inside a bordered panel.
#[derive(Debug, Clone)]
pub enum PanelRow {
    /// Blank spacer row inside the frame.
    Blank,
    /// Horizontal rule row inside the frame.
    Divider,
    /// A label/value pair rendered in two columns.
    Field { label: String, value: String },
    /// A free-form already-styled text line.
    Line(String),
    /// A subdued section heading.
    Section(String),
}

/// Build a rounded, colour-accented box around the provided rows. The title
/// bar is printed in the accent color; inside rows may carry their own ANSI
/// styling without breaking alignment thanks to [`visible_width`].
#[must_use]
pub fn panel(title: &str, rows: &[PanelRow]) -> String {
    panel_with_width(title, rows, PANEL_WIDTH)
}

/// Build a rounded panel with a caller-selected width. Widths smaller than a
/// practical minimum are clamped so borders and labels remain readable.
#[must_use]
pub fn panel_with_width(title: &str, rows: &[PanelRow], width: usize) -> String {
    let width = width.max(44);
    let inner = width.saturating_sub(4); // borders + one pad on each side

    let mut out = String::new();

    // Top border with inline title: ╭── Title ──────────────╮
    let title_segment = format!(" {} ", title.bold().with(palette::ACCENT));
    let title_visible = visible_width(&title_segment);
    let remaining = inner.saturating_sub(title_visible).saturating_add(2);
    let top = format!(
        "{}{}{}{}",
        "╭─".with(palette::ACCENT_DIM),
        title_segment,
        "─".repeat(remaining).with(palette::ACCENT_DIM),
        "╮".with(palette::ACCENT_DIM),
    );
    out.push_str(&top);
    out.push('\n');

    for row in rows {
        let body = match row {
            PanelRow::Blank => String::new(),
            PanelRow::Divider => "┈".repeat(inner).with(palette::MUTED).to_string(),
            PanelRow::Section(text) => {
                format!("{}", text.clone().bold().with(palette::BRAND))
            }
            PanelRow::Field { label, value } => {
                let label_width = 14usize;
                let label_styled = format!("{label:<label_width$}");
                format!(
                    "{} {}",
                    label_styled.with(palette::MUTED),
                    value.clone().with(Color::White),
                )
            }
            PanelRow::Line(text) => text.clone(),
        };
        out.push_str(&frame_line(&body, inner));
        out.push('\n');
    }

    // Bottom border
    let bottom = format!(
        "{}{}{}",
        "╰".with(palette::ACCENT_DIM),
        "─"
            .repeat(width.saturating_sub(2))
            .with(palette::ACCENT_DIM),
        "╯".with(palette::ACCENT_DIM),
    );
    out.push_str(&bottom);
    out
}

fn frame_line(content: &str, inner_width: usize) -> String {
    let content_visible = visible_width(content);
    let pad = inner_width.saturating_sub(content_visible);
    format!(
        "{} {}{} {}",
        "│".with(palette::ACCENT_DIM),
        content,
        " ".repeat(pad),
        "│".with(palette::ACCENT_DIM),
    )
}

/// Render the welcome banner shown at REPL startup.
///
/// Startup should orient the user, then get out of the way. A large framed
/// settings card made every launch feel like opening a control panel, so this
/// deliberately stays to four short lines.
#[must_use]
pub fn welcome_banner(info: &BannerInfo<'_>) -> String {
    let service = info.provider.map_or_else(
        || info.service.to_string(),
        |provider| format!("{} via {provider}", info.service),
    );
    let mut output = format!(
        "{}  {}\n  {} {} {}\n  {} {} {}",
        "◆ pebble".bold().with(palette::BRAND),
        format!("v{}", info.version).with(palette::MUTED),
        short_model_name(info.model).bold().with(palette::ACCENT),
        "on".with(palette::MUTED),
        service,
        info.collaboration_mode.with(palette::BRAND),
        "·".with(palette::MUTED),
        friendly_permission(info.permission_mode).with(palette::PERMISSION),
    );
    if let Some(cwd) = info.cwd {
        let _ = write!(output, "\n  {}", cwd.with(palette::MUTED));
    }
    if let Some(login_command) = info.auth_hint {
        let _ = write!(
            output,
            "\n  {} {} {}",
            "Not connected".bold().with(palette::WARN),
            "· start with".with(palette::MUTED),
            login_command.with(palette::ACCENT),
        );
    }
    let _ = write!(
        output,
        "\n  {}",
        format!(
            "{} for commands · Tab switches mode",
            "/help".with(palette::ACCENT)
        )
        .with(palette::MUTED)
    );
    output
}

fn help_hint() -> String {
    format!(
        "{} {}",
        "type".to_string().with(palette::MUTED),
        "/help".bold().with(palette::ACCENT),
    )
}

/// Public entry point for caller-provided banner data.
#[derive(Debug, Clone, Copy)]
pub struct BannerInfo<'a> {
    pub version: &'a str,
    pub service: &'a str,
    pub model: &'a str,
    pub provider: Option<&'a str>,
    pub auth_hint: Option<&'a str>,
    pub collaboration_mode: &'a str,
    pub permission_mode: &'a str,
    pub cwd: Option<&'a str>,
}

/// Emit a subdued horizontal separator used between turns. The separator is
/// visually quiet so it doesn't steal attention from assistant output.
#[must_use]
pub fn turn_separator() -> String {
    let line = "─".repeat(PANEL_WIDTH);
    format!("\n{}\n", line.with(palette::MUTED))
}

/// Render the primary line-editor prompt. We use a chevron glyph in the
/// brand color so users can find the input cursor at a glance, even on
/// output-heavy sessions.
#[must_use]
pub fn prompt_string(collaboration_mode: &str) -> String {
    #[cfg(windows)]
    {
        format!("{collaboration_mode}> ")
    }
    #[cfg(not(windows))]
    {
        let mode_color = if collaboration_mode == "plan" {
            palette::ACCENT
        } else {
            palette::BRAND
        };
        format!(
            "{} {} ",
            collaboration_mode.with(mode_color),
            "❯".bold().with(mode_color)
        )
    }
}

fn friendly_permission(permission_mode: &str) -> &str {
    match permission_mode {
        "read-only" => "read only",
        "workspace-write" => "workspace",
        "danger-full-access" => "full access",
        other => other,
    }
}

fn short_model_name(model: &str) -> &str {
    model.rsplit('/').next().unwrap_or(model)
}

#[allow(clippy::cast_precision_loss)]
fn context_percent(info: ContextWindowInfo) -> f64 {
    if info.max_tokens == 0 {
        0.0
    } else {
        (info.used_tokens as f64 / info.max_tokens as f64) * 100.0
    }
}

#[allow(clippy::cast_precision_loss)]
fn format_compact_tokens(tokens: u64) -> String {
    if tokens >= 1_000_000 {
        format!("{:.1}M", tokens as f64 / 1_000_000.0)
    } else if tokens >= 1_000 {
        format!("{:.1}k", tokens as f64 / 1_000.0)
    } else {
        tokens.to_string()
    }
}

#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn format_context_window_usage(info: ContextWindowInfo) -> String {
    let percent = context_percent(info);
    format!(
        "{}/{} {:.1}%",
        format_compact_tokens(info.used_tokens),
        format_compact_tokens(info.max_tokens),
        percent
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextWindowInfo {
    pub used_tokens: u64,
    pub max_tokens: u64,
}

/// Render the header that precedes a tool call in the transcript.
///
/// Example:
///   ⏺ Bash › git status
///
/// The leading glyph makes tool activity skimmable even in a dense log.
#[must_use]
pub fn tool_call_header(name: &str, summary: &str) -> String {
    let icon = tool_icon(name);
    let name_styled = tool_label(name).bold().with(palette::TOOL_NAME).to_string();
    if summary.is_empty() {
        format!("{} {}", icon.with(palette::ACCENT), name_styled)
    } else {
        let summary_styled = summary.to_owned().with(palette::TOOL_ARG);
        format!(
            "{} {}  {}",
            icon.with(palette::ACCENT),
            name_styled,
            summary_styled
        )
    }
}

fn tool_label(tool_name: &str) -> &str {
    match tool_name {
        "bash" | "Bash" => "Shell",
        "read_file" | "Read" => "Read",
        "write_file" | "Write" => "Write",
        "edit_file" | "Edit" | "MultiEdit" => "Edit",
        "apply_patch" => "Patch",
        "glob_search" | "Glob" => "Find files",
        "grep_search" | "Grep" => "Search",
        "web_search" | "WebSearch" => "Web search",
        "WebFetch" | "web_scrape" | "WebScrape" => "Web fetch",
        "ls" | "Ls" => "List",
        other => other,
    }
}

/// Pick a single-width glyph per tool family so transcript columns stay
/// aligned in terminals that disagree about emoji width.
fn tool_icon(tool_name: &str) -> &'static str {
    match tool_name {
        "bash" | "Bash" => "$",
        "read_file" | "Read" => "›",
        "write_file" | "Write" | "edit_file" | "Edit" | "MultiEdit" | "apply_patch" => "+",
        "glob_search" | "Glob" | "grep_search" | "Grep" => "?",
        "web_search" | "WebSearch" | "WebFetch" | "web_scrape" | "WebScrape" => "@",
        "ls" | "Ls" => "·",
        _ => "•",
    }
}

/// Emit a short dim note. Used for auxiliary notices (auto-compaction, budget
/// warnings, tool result truncations) where we want to inform without
/// distracting from the conversation flow.
#[must_use]
pub fn dim_note(text: &str) -> String {
    format!("{} {}", "·".with(palette::MUTED), text.with(palette::MUTED))
}

#[must_use]
pub fn activity_note(text: &str) -> String {
    format!(
        "{} {}",
        "◇".with(palette::BRAND_DIM),
        text.with(palette::MUTED)
    )
}

/// Emit a warning-styled note (yellow, with a warning glyph).
#[must_use]
pub fn warning_note(text: &str) -> String {
    format!("{} {}", "⚠".with(palette::WARN), text.with(palette::WARN))
}

/// Emit a success note (green, with a check glyph).
#[must_use]
pub fn success_note(text: &str) -> String {
    format!("{} {}", "✔".with(palette::OK), text.with(palette::OK))
}

/// Emit an error note (red, with a cross glyph).
#[must_use]
pub fn error_note(text: &str) -> String {
    format!("{} {}", "✘".with(palette::ERR), text.with(palette::ERR))
}

/// Styling for the "thinking" stream that surfaces model-internal reasoning
/// in a clearly-distinct, dim colour so it never competes visually with the
/// primary response.
#[must_use]
pub fn thinking_chunk(text: &str) -> String {
    format!("{}", text.italic().with(palette::MUTED))
}

/// Styled leader printed once before a stream of thinking output.
#[must_use]
pub fn thinking_lead() -> String {
    format!(
        "{} {}\n",
        "✦".with(palette::BRAND_DIM),
        "thinking".italic().with(palette::MUTED),
    )
}

/// Render a compact, indented tool-result block that visually hangs off the
/// tool-call header above it. The layout intentionally does NOT use markdown
/// because the tool-call header already announced the call; what we want
/// underneath is a terse, dim confirmation, not a second heading.
///
/// `summary_lines` is an ordered list of `(label, value)` pairs. Labels are
/// rendered muted, values are rendered normally. Pass an empty `label` to
/// render a single freeform line.
///
/// Example output:
/// ```text
///   ⎿ wrote 1 line to /tmp/test.txt
/// ```
#[must_use]
pub fn tool_result_block(summary_lines: &[(&str, &str)]) -> String {
    if summary_lines.is_empty() {
        return String::new();
    }
    let mut out = String::new();
    for (idx, (label, value)) in summary_lines.iter().enumerate() {
        let glyph = if idx == 0 { "⎿" } else { " " };
        let label_part = if label.is_empty() {
            String::new()
        } else {
            format!("{} ", format!("{label}:").with(palette::MUTED))
        };
        let _ = std::fmt::Write::write_fmt(
            &mut out,
            format_args!(
                "  {} {label_part}{}\n",
                glyph.to_string().with(palette::MUTED),
                value.to_string().with(palette::MUTED),
            ),
        );
    }
    out
}

/// Render a multi-line tool result whose body is an already-captured chunk of
/// text (bash stdout, grep hits, file range). The body is indented and
/// dimmed, with a `⎿` tree-drawing prefix on the first line. Long bodies
/// should be truncated by the caller before being passed in.
#[must_use]
pub fn tool_result_body(heading: &str, body: &str) -> String {
    let mut out = String::new();
    let _ = std::fmt::Write::write_fmt(
        &mut out,
        format_args!(
            "  {} {}\n",
            "⎿".with(palette::MUTED),
            heading.to_string().with(palette::MUTED),
        ),
    );
    for line in body.lines() {
        let _ = std::fmt::Write::write_fmt(
            &mut out,
            format_args!("    {}\n", line.to_string().with(palette::MUTED)),
        );
    }
    out
}

/// Prefix emitted once per assistant response to visually anchor the model's
/// voice. Rendered as a bold brand-colored bullet on its own line so the
/// reply is easy to locate in a tool-heavy transcript.
#[must_use]
pub fn assistant_lead() -> String {
    format!(
        "{} {}\n",
        "◆".with(palette::BRAND),
        "Pebble".bold().with(palette::BRAND)
    )
}

/// Render a compact banner for resumed sessions. It avoids repeating the full
/// welcome card while still orienting the user before the prompt appears.
#[must_use]
pub fn resume_banner(info: &ResumeBannerInfo<'_>) -> String {
    format!(
        "{}  {}\n  {}\n  {} {} {}\n  {}",
        "◆ pebble".bold().with(palette::BRAND),
        "resumed".with(palette::OK),
        short_model_name(info.model).bold().with(palette::ACCENT),
        info.collaboration_mode.with(palette::BRAND),
        "·".with(palette::MUTED),
        friendly_permission(info.permission_mode).with(palette::PERMISSION),
        format!("session {}", info.session_id).with(palette::MUTED),
    )
}

#[derive(Debug, Clone, Copy)]
pub struct ResumeBannerInfo<'a> {
    pub session_id: &'a str,
    pub model: &'a str,
    pub collaboration_mode: &'a str,
    pub permission_mode: &'a str,
}

/// A narrower panel for command feedback and prompts. These cards are meant
/// to be glanced at, not studied.
#[must_use]
pub fn compact_panel(title: &str, rows: &[PanelRow]) -> String {
    let mut output = format!("{}", title.bold().with(palette::ACCENT));
    for row in rows {
        match row {
            PanelRow::Blank => output.push('\n'),
            PanelRow::Divider => {
                let _ = write!(output, "\n  {}", "─".repeat(40).with(palette::MUTED));
            }
            PanelRow::Field { label, value } => {
                let _ = write!(
                    output,
                    "\n  {:<14} {}",
                    label.clone().with(palette::MUTED),
                    value
                );
            }
            PanelRow::Line(line) => {
                let _ = write!(output, "\n  {line}");
            }
            PanelRow::Section(section) => {
                let _ = write!(output, "\n{}", section.as_str().bold().with(palette::BRAND));
            }
        }
    }
    output
}

/// Render a concise success card for a setting change.
#[must_use]
pub fn setting_changed(title: &str, fields: &[(&str, &str)]) -> String {
    let mut output = format!("{} {}", "✓".with(palette::OK), title.bold());
    for (label, value) in fields {
        let _ = write!(
            output,
            "\n  {}  {}",
            label.to_string().with(palette::MUTED),
            value
        );
    }
    output
}

/// Summary printed at the end of an assistant turn.
#[must_use]
pub fn turn_summary(info: &TurnSummaryInfo) -> String {
    let mut pieces = vec![format!("Done in {}", format_duration(info.elapsed))];
    if info.iterations > 1 {
        pieces.push(format!("{} passes", info.iterations));
    }
    if info.tool_calls > 0 {
        pieces.push(format!("{} tools", info.tool_calls));
    }
    if info.changed_files > 0 {
        let noun = if info.changed_files == 1 {
            "file"
        } else {
            "files"
        };
        pieces.push(format!("{} {noun} changed", info.changed_files));
    }
    let total_tokens = info
        .usage
        .input_tokens
        .saturating_add(info.usage.output_tokens);
    if total_tokens > 0 {
        pieces.push(format!(
            "{} tokens",
            format_compact_tokens(u64::from(total_tokens))
        ));
    }
    if let Some(context_window) = info.context_window {
        pieces.push(format!("{:.0}% context", context_percent(context_window)));
    }
    let separator = format!(" {} ", "·".with(palette::MUTED));
    let joined = pieces.join(&separator);
    let terminal_width = terminal_size()
        .ok()
        .map_or(usize::MAX, |(columns, _)| usize::from(columns));
    if visible_width(&joined).saturating_add(2) <= terminal_width || pieces.len() == 1 {
        format!("{} {joined}", "✓".with(palette::OK))
    } else {
        format!(
            "{} {}\n  {}",
            "✓".with(palette::OK),
            pieces[0],
            pieces[1..].join(&separator)
        )
    }
}

fn format_duration(duration: Duration) -> String {
    if duration.as_secs() >= 60 {
        format!(
            "{}m {:02}s",
            duration.as_secs() / 60,
            duration.as_secs() % 60
        )
    } else if duration.as_secs() >= 10 {
        format!("{}s", duration.as_secs())
    } else {
        format!("{:.1}s", duration.as_secs_f64())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TurnSummaryInfo {
    pub elapsed: Duration,
    pub iterations: usize,
    pub tool_calls: usize,
    pub changed_files: usize,
    pub usage: runtime::TokenUsage,
    pub context_window: Option<ContextWindowInfo>,
}

/// Render the current permission request as a clear approval card.
#[must_use]
pub fn permission_prompt_header(tool: &str, current: &str, required: &str) -> String {
    format!(
        "{} {}\n  {} wants to run {}\n  {} → {}",
        "!".with(palette::WARN),
        "Permission needed".bold(),
        "Pebble".with(palette::MUTED),
        tool_label(tool).with(palette::TOOL_NAME),
        friendly_permission(current).with(palette::MUTED),
        friendly_permission(required).with(palette::PERMISSION),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn visible_width_strips_ansi() {
        let raw = format!("{}", "hello".with(palette::ACCENT));
        assert_eq!(visible_width(&raw), 5);
    }

    #[test]
    fn panel_contains_title_and_rows() {
        let rendered = panel(
            "Demo",
            &[
                PanelRow::Field {
                    label: "key".into(),
                    value: "value".into(),
                },
                PanelRow::Blank,
                PanelRow::Line("freeform".into()),
            ],
        );
        assert!(rendered.contains("Demo"));
        assert!(rendered.contains("key"));
        assert!(rendered.contains("value"));
        assert!(rendered.contains("freeform"));
        assert!(rendered.contains("╭"));
        assert!(rendered.contains("╯"));
    }

    #[test]
    fn welcome_banner_mentions_core_fields() {
        let banner = welcome_banner(&BannerInfo {
            version: "0.2.0",
            service: "NanoGPT",
            model: "zai-org/glm-5.1",
            provider: Some("fireworks"),
            auth_hint: Some("/login nanogpt"),
            collaboration_mode: "build",
            permission_mode: "workspace-write",
            cwd: Some("/tmp/project"),
        });
        assert!(banner.contains("pebble"));
        assert!(banner.contains("v0.2.0"));
        assert!(banner.contains("glm-5.1"));
        assert!(banner.contains("fireworks"));
        assert!(banner.contains("build"));
        assert!(banner.contains("workspace"));
        assert!(banner.contains("/tmp/project"));
        assert!(banner.contains("Not connected"));
        assert!(banner.contains("/login nanogpt"));
    }

    #[test]
    fn tool_call_header_includes_icon_and_name() {
        let header = tool_call_header("Bash", "git status");
        assert!(header.contains("Shell"));
        assert!(header.contains("git status"));
    }

    #[test]
    fn format_compact_tokens_scales() {
        assert_eq!(format_compact_tokens(500), "500");
        assert_eq!(format_compact_tokens(1_500), "1.5k");
        assert_eq!(format_compact_tokens(2_400_000), "2.4M");
    }

    #[test]
    fn plain_terminal_conventions_disable_color() {
        let value = |name: &str| match name {
            "TERM" => Some(OsString::from("xterm-256color")),
            _ => None,
        };
        assert!(!should_enable_color_output(false, value));

        for (name, value) in [("NO_COLOR", "1"), ("CLICOLOR", "0"), ("TERM", "dumb")] {
            assert!(!should_enable_color_output(true, |candidate| {
                (candidate == name).then(|| OsString::from(value))
            }));
        }
    }

    #[test]
    fn clicolor_force_enables_color_for_redirected_output() {
        assert!(should_enable_color_output(false, |name| {
            (name == "CLICOLOR_FORCE").then(|| OsString::from("1"))
        }));
    }
}
