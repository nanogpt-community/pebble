use std::borrow::Cow;
use std::fs;
use std::io::{self, IsTerminal, Read, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use rustyline::completion::{Completer, Pair};
use rustyline::error::ReadlineError;
use rustyline::highlight::{CmdKind, Highlighter};
use rustyline::hint::Hinter;
use rustyline::history::DefaultHistory;
use rustyline::validate::Validator;
use rustyline::{
    Cmd, CompletionType, ConditionalEventHandler, Config, Context, Editor, Event, EventContext,
    EventHandler, Helper, KeyCode, KeyEvent, Modifiers,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReadOutcome {
    Submit(String),
    Cancel,
    Exit,
    ToggleMode,
}

struct SlashCommandHelper {
    completions: Vec<String>,
}

impl SlashCommandHelper {
    fn new(completions: Vec<String>) -> Self {
        Self { completions }
    }

    fn set_completions(&mut self, completions: Vec<String>) {
        self.completions = completions;
    }
}

impl Completer for SlashCommandHelper {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &Context<'_>,
    ) -> rustyline::Result<(usize, Vec<Self::Candidate>)> {
        let Some((start, prefix)) = slash_command_prefix(line, pos) else {
            return Ok((0, Vec::new()));
        };

        let matches = self
            .completions
            .iter()
            .filter(|candidate| candidate.starts_with(prefix))
            .map(|candidate| Pair {
                display: candidate.clone(),
                replacement: candidate.clone(),
            })
            .collect();

        Ok((start, matches))
    }
}

impl Hinter for SlashCommandHelper {
    type Hint = String;
}

impl Highlighter for SlashCommandHelper {
    fn highlight<'l>(&self, line: &'l str, _pos: usize) -> Cow<'l, str> {
        Cow::Borrowed(line)
    }

    fn highlight_char(&self, _line: &str, _pos: usize, _kind: CmdKind) -> bool {
        false
    }
}

impl Validator for SlashCommandHelper {}
impl Helper for SlashCommandHelper {}

struct PasteSafeSubmitHandler;

impl ConditionalEventHandler for PasteSafeSubmitHandler {
    fn handle(
        &self,
        _evt: &Event,
        _n: rustyline::RepeatCount,
        _positive: bool,
        ctx: &EventContext,
    ) -> Option<Cmd> {
        if ctx.line().is_empty() {
            None
        } else {
            Some(Cmd::AcceptLine)
        }
    }
}

struct EmptyLineTabHandler {
    toggled: Arc<AtomicBool>,
}

impl ConditionalEventHandler for EmptyLineTabHandler {
    fn handle(
        &self,
        _evt: &Event,
        _n: rustyline::RepeatCount,
        _positive: bool,
        ctx: &EventContext,
    ) -> Option<Cmd> {
        if ctx.line().is_empty() {
            self.toggled.store(true, Ordering::SeqCst);
            Some(Cmd::Interrupt)
        } else {
            None
        }
    }
}

fn paste_safe_mode_enabled() -> bool {
    env_flag_enabled("PEBBLE_PASTE_SAFE")
}

pub struct LineEditor {
    prompt: String,
    completions: Vec<String>,
    editor: Editor<SlashCommandHelper, DefaultHistory>,
    pending_mode_toggle: Arc<AtomicBool>,
    history_path: Option<PathBuf>,
}

impl LineEditor {
    #[must_use]
    pub fn new(prompt: impl Into<String>, completions: Vec<String>) -> Self {
        let pending_mode_toggle = Arc::new(AtomicBool::new(false));
        let editor = Self::build_editor(completions.clone(), pending_mode_toggle.clone());

        Self {
            prompt: prompt.into(),
            completions,
            editor,
            pending_mode_toggle,
            history_path: None,
        }
    }

    #[must_use]
    pub fn with_history_path(mut self, path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        if path.is_file() {
            let _ = self.editor.load_history(&path);
        }
        self.history_path = Some(path);
        self
    }

    /// Update the visible input prompt glyph. Callers typically do this when
    /// switching modes (e.g. toggling thinking) so the prompt stays in sync
    /// with ambient state.
    #[allow(dead_code)]
    pub fn set_prompt(&mut self, prompt: impl Into<String>) {
        self.prompt = prompt.into();
    }

    pub fn push_history(&mut self, entry: impl Into<String>) {
        let entry = entry.into();
        if entry.trim().is_empty() {
            return;
        }

        if is_sensitive_history_entry(&entry) {
            return;
        }

        let _ = self.editor.add_history_entry(entry);
    }

    pub fn set_completions(&mut self, completions: Vec<String>) {
        self.completions = completions.clone();
        if let Some(helper) = self.editor.helper_mut() {
            helper.set_completions(completions);
        }
    }

    pub fn read_line(&mut self) -> io::Result<ReadOutcome> {
        loop {
            if !io::stdin().is_terminal() || !io::stdout().is_terminal() {
                return self.read_line_fallback();
            }
            self.pending_mode_toggle.store(false, Ordering::SeqCst);

            match self.editor.readline(&self.prompt) {
                Ok(line) => {
                    if self.handle_submission(&line)? {
                        continue;
                    }
                    return Ok(ReadOutcome::Submit(line));
                }
                Err(ReadlineError::Interrupted) => {
                    if self.pending_mode_toggle.swap(false, Ordering::SeqCst) {
                        self.finish_interrupted_read()?;
                        return Ok(ReadOutcome::ToggleMode);
                    }
                    self.finish_interrupted_read()?;
                    return Ok(ReadOutcome::Cancel);
                }
                Err(ReadlineError::Eof) => {
                    self.finish_interrupted_read()?;
                    return Ok(ReadOutcome::Exit);
                }
                Err(error) => return Err(io::Error::other(error)),
            }
        }
    }

    fn build_editor(
        completions: Vec<String>,
        pending_mode_toggle: Arc<AtomicBool>,
    ) -> Editor<SlashCommandHelper, DefaultHistory> {
        let paste_safe_mode = paste_safe_mode_enabled();
        let config = Config::builder()
            .completion_type(CompletionType::List)
            .build();
        let mut editor = Editor::<SlashCommandHelper, DefaultHistory>::with_config(config)
            .expect("rustyline editor should initialize");
        editor.set_helper(Some(SlashCommandHelper::new(completions)));
        editor.bind_sequence(KeyEvent(KeyCode::Enter, Modifiers::SHIFT), Cmd::Newline);
        editor.bind_sequence(KeyEvent(KeyCode::Char('J'), Modifiers::CTRL), Cmd::Newline);
        if paste_safe_mode {
            editor.bind_sequence(KeyEvent(KeyCode::Enter, Modifiers::NONE), Cmd::Newline);
            editor.bind_sequence(
                KeyEvent(KeyCode::Char('D'), Modifiers::CTRL),
                EventHandler::Conditional(Box::new(PasteSafeSubmitHandler)),
            );
        }
        editor.bind_sequence(KeyEvent(KeyCode::Up, Modifiers::NONE), Cmd::PreviousHistory);
        editor.bind_sequence(KeyEvent(KeyCode::Down, Modifiers::NONE), Cmd::NextHistory);
        editor.bind_sequence(
            KeyEvent(KeyCode::Tab, Modifiers::NONE),
            EventHandler::Conditional(Box::new(EmptyLineTabHandler {
                toggled: pending_mode_toggle,
            })),
        );
        editor
    }

    fn finish_interrupted_read(&mut self) -> io::Result<()> {
        let mut stdout = io::stdout();
        writeln!(stdout)
    }

    fn read_line_fallback(&mut self) -> io::Result<ReadOutcome> {
        loop {
            let mut stdout = io::stdout();
            write!(stdout, "{}", self.prompt)?;
            stdout.flush()?;

            let Some(buffer) = read_full_fallback_input(&mut io::stdin())? else {
                return Ok(ReadOutcome::Exit);
            };

            if self.handle_submission(&buffer)? {
                continue;
            }

            return Ok(ReadOutcome::Submit(buffer));
        }
    }

    fn handle_submission(&mut self, _line: &str) -> io::Result<bool> {
        Ok(false)
    }
}

impl Drop for LineEditor {
    fn drop(&mut self) {
        let Some(path) = self.history_path.as_deref() else {
            return;
        };
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let _ = self.editor.save_history(path);
    }
}

fn is_sensitive_history_entry(entry: &str) -> bool {
    let trimmed = entry.trim();
    let mut parts = trimmed.split_whitespace();
    let Some(command) = parts.next() else {
        return false;
    };
    if !matches!(command, "/login" | "/auth") {
        return false;
    }
    let remaining = parts.collect::<Vec<_>>();
    remaining.len() > 1 || remaining.iter().any(|part| part.contains("api-key"))
}

fn env_flag_enabled(name: &str) -> bool {
    std::env::var(name).ok().is_some_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

fn read_full_fallback_input(reader: &mut impl Read) -> io::Result<Option<String>> {
    let mut buffer = String::new();
    reader.read_to_string(&mut buffer)?;
    if buffer.is_empty() {
        return Ok(None);
    }

    while matches!(buffer.chars().last(), Some('\n' | '\r')) {
        buffer.pop();
    }

    Ok(Some(buffer))
}

fn slash_command_prefix(line: &str, pos: usize) -> Option<(usize, &str)> {
    if pos != line.len() {
        return None;
    }

    let prefix = &line[..pos];
    if !prefix.starts_with('/') {
        return None;
    }

    Some((0, prefix))
}

#[cfg(test)]
mod tests {
    use super::{
        is_sensitive_history_entry, paste_safe_mode_enabled, read_full_fallback_input,
        slash_command_prefix, LineEditor, SlashCommandHelper,
    };
    use rustyline::completion::Completer;
    use rustyline::history::{DefaultHistory, History};
    use rustyline::Context;
    use std::io::Cursor;

    #[test]
    fn extracts_only_terminal_slash_command_prefixes() {
        assert_eq!(slash_command_prefix("/he", 3), Some((0, "/he")));
        assert_eq!(slash_command_prefix("/help me", 5), None);
        assert_eq!(slash_command_prefix("hello", 5), None);
        assert_eq!(slash_command_prefix("/help", 2), None);
    }

    #[test]
    fn completes_matching_slash_commands() {
        let helper = SlashCommandHelper::new(vec![
            "/help".to_string(),
            "/hello".to_string(),
            "/status".to_string(),
        ]);
        let history = DefaultHistory::new();
        let ctx = Context::new(&history);
        let (start, matches) = helper
            .complete("/he", 3, &ctx)
            .expect("completion should work");

        assert_eq!(start, 0);
        assert_eq!(
            matches
                .into_iter()
                .map(|candidate| candidate.replacement)
                .collect::<Vec<_>>(),
            vec!["/help".to_string(), "/hello".to_string()]
        );
    }

    #[test]
    fn completes_contextual_slash_commands_with_arguments() {
        let helper = SlashCommandHelper::new(vec![
            "/help auth".to_string(),
            "/help sessions".to_string(),
            "/permissions workspace-write".to_string(),
        ]);
        let history = DefaultHistory::new();
        let ctx = Context::new(&history);
        let (start, matches) = helper
            .complete("/help a", 7, &ctx)
            .expect("completion should work");

        assert_eq!(start, 0);
        assert_eq!(
            matches
                .into_iter()
                .map(|candidate| candidate.replacement)
                .collect::<Vec<_>>(),
            vec!["/help auth".to_string()]
        );
    }

    #[test]
    fn ignores_non_slash_command_completion_requests() {
        let helper = SlashCommandHelper::new(vec!["/help".to_string()]);
        let history = DefaultHistory::new();
        let ctx = Context::new(&history);
        let (_, matches) = helper
            .complete("hello", 5, &ctx)
            .expect("completion should work");

        assert!(matches.is_empty());
    }

    #[test]
    fn push_history_ignores_blank_entries() {
        let mut editor = LineEditor::new("› ", vec!["/help".to_string()]);
        editor.push_history("   ");
        editor.push_history("/help");

        assert_eq!(editor.editor.history().len(), 1);
    }

    #[test]
    fn history_skips_inline_credentials() {
        assert!(is_sensitive_history_entry("/login nanogpt secret-key"));
        assert!(is_sensitive_history_entry(
            "/auth opencode-go --api-key secret-key"
        ));
        assert!(!is_sensitive_history_entry("/login nanogpt"));
        assert!(!is_sensitive_history_entry("summarize the project"));
    }

    #[test]
    fn handle_submission_does_not_intercept_plain_input() {
        let mut editor = LineEditor::new("> ", vec!["/help".to_string()]);
        editor.push_history("/help");

        let handled = editor
            .handle_submission("hello")
            .expect("submission handling should succeed");

        assert!(!handled);
        assert_eq!(editor.editor.history().len(), 1);
    }

    #[test]
    fn fallback_input_reads_full_multiline_payload() {
        let mut input = Cursor::new("first line\nsecond line\nthird line\n");

        let result = read_full_fallback_input(&mut input).expect("fallback read should succeed");

        assert_eq!(
            result,
            Some("first line\nsecond line\nthird line".to_string())
        );
    }

    #[test]
    fn fallback_input_trims_final_crlf_without_touching_internal_newlines() {
        let mut input = Cursor::new("alpha\r\nbeta\r\n");

        let result = read_full_fallback_input(&mut input).expect("fallback read should succeed");

        assert_eq!(result, Some("alpha\r\nbeta".to_string()));
    }

    #[test]
    fn fallback_input_returns_none_for_empty_stdin() {
        let mut input = Cursor::new("");

        let result = read_full_fallback_input(&mut input).expect("fallback read should succeed");

        assert_eq!(result, None);
    }

    #[test]
    fn paste_safe_mode_reads_env_flag() {
        std::env::set_var("PEBBLE_PASTE_SAFE", "1");
        assert!(paste_safe_mode_enabled());

        std::env::remove_var("PEBBLE_PASTE_SAFE");
        assert!(!paste_safe_mode_enabled());
    }
}
