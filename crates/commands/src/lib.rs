use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;

use platform::{pebble_config_home, write_atomic};
use runtime::{compact_session, CompactionConfig, Session};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandManifestEntry {
    pub name: String,
    pub source: CommandSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandSource {
    Builtin,
    InternalOnly,
    FeatureGated,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CommandRegistry {
    entries: Vec<CommandManifestEntry>,
}

impl CommandRegistry {
    #[must_use]
    pub fn new(entries: Vec<CommandManifestEntry>) -> Self {
        Self { entries }
    }

    #[must_use]
    pub fn entries(&self) -> &[CommandManifestEntry] {
        &self.entries
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlashCommandCategory {
    Core,
    Session,
    Workspace,
    Repl,
}

impl SlashCommandCategory {
    const fn label(self) -> &'static str {
        match self {
            Self::Core => "Core",
            Self::Session => "Session",
            Self::Workspace => "Workspace",
            Self::Repl => "REPL",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlashCommandSpec {
    pub name: &'static str,
    pub aliases: &'static [&'static str],
    pub summary: &'static str,
    pub argument_hint: Option<&'static str>,
    pub category: SlashCommandCategory,
    pub detail: &'static str,
    pub resume_supported: bool,
}

const SLASH_COMMAND_SPECS: &[SlashCommandSpec] = &[
    SlashCommandSpec {
        name: "help",
        aliases: &[],
        summary: "Show available slash commands",
        argument_hint: None,
        category: SlashCommandCategory::Core,
        detail: "Use `/help <topic>` for focused guidance on auth, sessions, extensions, or web tools.",
        resume_supported: true,
    },
    SlashCommandSpec {
        name: "status",
        aliases: &[],
        summary: "Show current session status",
        argument_hint: None,
        category: SlashCommandCategory::Core,
        detail: "Shows model, service, permissions, runtime toggles, workspace context, MCP counts, and web tool readiness.",
        resume_supported: true,
    },
    SlashCommandSpec {
        name: "compact",
        aliases: &[],
        summary: "Compact local session history",
        argument_hint: None,
        category: SlashCommandCategory::Session,
        detail: "Condenses older messages into a resumable summary while keeping the active session usable.",
        resume_supported: true,
    },
    SlashCommandSpec {
        name: "archives",
        aliases: &[],
        summary: "List or inspect archived tool outputs",
        argument_hint: Some(
            "[list|show <message-id-or-tool-call-id>|page <message-id-or-tool-call-id>|save <message-id-or-tool-call-id> [file]]",
        ),
        category: SlashCommandCategory::Session,
        detail: "Shows tool outputs that were archived during compaction and can print, page, or restore a specific archived result by message ID or tool call ID.",
        resume_supported: true,
    },
    SlashCommandSpec {
        name: "undo",
        aliases: &[],
        summary: "Undo the previous turn and restore changed files",
        argument_hint: None,
        category: SlashCommandCategory::Session,
        detail: "Restores the workspace and conversation to the state before the most recent prompt.",
        resume_supported: false,
    },
    SlashCommandSpec {
        name: "redo",
        aliases: &[],
        summary: "Redo the most recently undone turn",
        argument_hint: None,
        category: SlashCommandCategory::Session,
        detail: "Reapplies the last undone conversation turn and its tracked file changes.",
        resume_supported: false,
    },
    SlashCommandSpec {
        name: "timeline",
        aliases: &[],
        summary: "Show message timeline for the active session",
        argument_hint: None,
        category: SlashCommandCategory::Session,
        detail: "Lists message ids, roles, and short previews for selecting fork points.",
        resume_supported: true,
    },
    SlashCommandSpec {
        name: "fork",
        aliases: &[],
        summary: "Fork the active session",
        argument_hint: Some("[message-id-or-index]"),
        category: SlashCommandCategory::Session,
        detail: "Creates a new managed session from the current history, optionally truncated after a message.",
        resume_supported: false,
    },
    SlashCommandSpec {
        name: "rename",
        aliases: &[],
        summary: "Rename the active session",
        argument_hint: Some("<title>"),
        category: SlashCommandCategory::Session,
        detail: "Stores a display title in session metadata for session lists and filtering.",
        resume_supported: false,
    },
    SlashCommandSpec {
        name: "reasoning",
        aliases: &["thinking"],
        summary: "Show or set reasoning effort",
        argument_hint: Some("[default|minimal|low|medium|high|xhigh]"),
        category: SlashCommandCategory::Core,
        detail: "Without an argument, reports the current reasoning effort. `/thinking` remains as a compatibility alias.",
        resume_supported: false,
    },
    SlashCommandSpec {
        name: "fast",
        aliases: &[],
        summary: "Show or toggle ChatGPT fast mode",
        argument_hint: Some("[on|off]"),
        category: SlashCommandCategory::Core,
        detail: "Without an argument, reports whether fast mode is enabled for the current session.",
        resume_supported: false,
    },
    SlashCommandSpec {
        name: "mode",
        aliases: &[],
        summary: "Show or switch between build and plan mode",
        argument_hint: Some("[build|plan]"),
        category: SlashCommandCategory::Core,
        detail: "Without an argument, reports the current collaboration mode. Press Tab on an empty prompt to toggle modes quickly.",
        resume_supported: false,
    },
    SlashCommandSpec {
        name: "model",
        aliases: &[],
        summary: "Show or switch the active model",
        argument_hint: Some("[model]"),
        category: SlashCommandCategory::Core,
        detail: "With no argument, opens the model picker. Model IDs are service-qualified, e.g. `opencode-go/kimi-k2.6`.",
        resume_supported: false,
    },
    SlashCommandSpec {
        name: "provider",
        aliases: &["providers"],
        summary: "Choose a model provider",
        argument_hint: Some("[name]"),
        category: SlashCommandCategory::Core,
        detail: "With no argument, opens the unified picker on the current provider. Pass `nanogpt`, `neuralwatt`, `lilac`, `grok`, `synthetic`, `openai-codex`, or `opencode-go` to focus it.",
        resume_supported: false,
    },
    SlashCommandSpec {
        name: "route",
        aliases: &["routing"],
        summary: "Choose NanoGPT's upstream route",
        argument_hint: Some("[route-id|default]"),
        category: SlashCommandCategory::Core,
        detail: "Only applies to NanoGPT models. With no argument, opens NanoGPT's route picker; `default` restores platform routing.",
        resume_supported: false,
    },
    SlashCommandSpec {
        name: "logout",
        aliases: &[],
        summary: "Remove saved credentials for a service",
        argument_hint: Some("[service]"),
        category: SlashCommandCategory::Core,
        detail: "With no argument, opens the auth service picker. Removes saved credentials from `~/.pebble/credentials.json`, but shell environment variables still take precedence if set.",
        resume_supported: false,
    },
    SlashCommandSpec {
        name: "mcp",
        aliases: &[],
        summary: "Inspect configured MCP servers or list exposed MCP tools",
        argument_hint: Some("[status|tools|reload|add <name>|enable <name>|disable <name>]"),
        category: SlashCommandCategory::Workspace,
        detail: "Use `/mcp add <name>` to scaffold a server entry in `.pebble/settings.json`, then `/mcp enable <name>` or `/mcp disable <name>` to manage local overrides in `.pebble/settings.local.json`.",
        resume_supported: false,
    },
    SlashCommandSpec {
        name: "permissions",
        aliases: &["bypass"],
        summary: "Show or switch the active permission mode",
        argument_hint: Some("[read-only|workspace-write|danger-full-access]"),
        category: SlashCommandCategory::Core,
        detail: "`/bypass` is a shortcut for `/permissions danger-full-access`.",
        resume_supported: false,
    },
    SlashCommandSpec {
        name: "clear",
        aliases: &[],
        summary: "Start a fresh local session",
        argument_hint: Some("[--confirm]"),
        category: SlashCommandCategory::Session,
        detail: "Creates a new managed session while preserving the current model and runtime toggles.",
        resume_supported: true,
    },
    SlashCommandSpec {
        name: "resume",
        aliases: &[],
        summary: "Load a saved session into the REPL",
        argument_hint: Some("[session-id-or-path]"),
        category: SlashCommandCategory::Session,
        detail: "Use `/resume last` to reopen the most recently modified saved session.",
        resume_supported: false,
    },
    SlashCommandSpec {
        name: "config",
        aliases: &[],
        summary: "Inspect Pebble config files or merged sections",
        argument_hint: Some("[check|env|hooks|model|plugins]"),
        category: SlashCommandCategory::Workspace,
        detail: "Shows merged runtime config, validates settings, and lists which config files were loaded.",
        resume_supported: true,
    },
    SlashCommandSpec {
        name: "memory",
        aliases: &[],
        summary: "Inspect loaded Pebble instruction memory files",
        argument_hint: None,
        category: SlashCommandCategory::Workspace,
        detail: "Lists project and user instruction files discovered by the runtime.",
        resume_supported: true,
    },
    SlashCommandSpec {
        name: "init",
        aliases: &[],
        summary: "Create a starter PEBBLE.md for this repo",
        argument_hint: None,
        category: SlashCommandCategory::Workspace,
        detail: "Bootstraps common project-local Pebble files for a repository.",
        resume_supported: true,
    },
    SlashCommandSpec {
        name: "diff",
        aliases: &[],
        summary: "Show git diff for current workspace changes",
        argument_hint: None,
        category: SlashCommandCategory::Workspace,
        detail: "Runs a local git diff against the current worktree.",
        resume_supported: true,
    },
    SlashCommandSpec {
        name: "patch",
        aliases: &[],
        summary: "Check or apply a patch",
        argument_hint: Some("[--check|--apply] [patch-file-or-inline-diff]"),
        category: SlashCommandCategory::Workspace,
        detail: "Validates or applies a unified diff/OpenAI-style patch block. Defaults to --check unless --apply is provided.",
        resume_supported: true,
    },
    SlashCommandSpec {
        name: "version",
        aliases: &[],
        summary: "Show CLI version and build information",
        argument_hint: None,
        category: SlashCommandCategory::Core,
        detail: "Prints the Pebble build version and related metadata.",
        resume_supported: true,
    },
    SlashCommandSpec {
        name: "branch",
        aliases: &[],
        summary: "List, create, or switch git branches",
        argument_hint: Some("[list|create <name>|switch <name>]"),
        category: SlashCommandCategory::Workspace,
        detail: "Simple git branch helpers for listing, creating, and switching local branches.",
        resume_supported: false,
    },
    SlashCommandSpec {
        name: "worktree",
        aliases: &[],
        summary: "List, add, remove, or prune git worktrees",
        argument_hint: Some("[list|add <path> [branch]|remove <path>|prune]"),
        category: SlashCommandCategory::Workspace,
        detail: "Convenience wrapper over `git worktree` commands.",
        resume_supported: false,
    },
    SlashCommandSpec {
        name: "export",
        aliases: &[],
        summary: "Export the current conversation to a file",
        argument_hint: Some("[file]"),
        category: SlashCommandCategory::Session,
        detail: "Writes the current session transcript to a text file in the working directory unless an explicit path is provided.",
        resume_supported: true,
    },
    SlashCommandSpec {
        name: "session",
        aliases: &[],
        summary: "List or switch managed local sessions",
        argument_hint: Some("[list|switch <session-id>|timeline|fork [message-id]|rename <title>]"),
        category: SlashCommandCategory::Session,
        detail: "Use `switch` to move between saved sessions, `timeline` to inspect message ids, `fork` to branch, or `rename` to store a title.",
        resume_supported: false,
    },
    SlashCommandSpec {
        name: "sessions",
        aliases: &[],
        summary: "List recent managed local sessions",
        argument_hint: None,
        category: SlashCommandCategory::Session,
        detail: "Shows recent managed sessions with model and last-prompt metadata.",
        resume_supported: false,
    },
    SlashCommandSpec {
        name: "plugins",
        aliases: &["plugin", "marketplace"],
        summary: "Manage Pebble plugins",
        argument_hint: Some(
            "[list|help|install <path>|enable <id>|disable <id>|uninstall <id>|update <id>]",
        ),
        category: SlashCommandCategory::Workspace,
        detail: "Use `/plugins help` for local plugin layout, install, and enable/disable guidance.",
        resume_supported: false,
    },
    SlashCommandSpec {
        name: "agents",
        aliases: &[],
        summary: "List configured Pebble agents",
        argument_hint: Some("[list|help]"),
        category: SlashCommandCategory::Workspace,
        detail: "Lists configured agent definitions discovered from project and user config roots.",
        resume_supported: true,
    },
    SlashCommandSpec {
        name: "skills",
        aliases: &[],
        summary: "List available Pebble skills",
        argument_hint: Some("[list|help|init <name>]"),
        category: SlashCommandCategory::Workspace,
        detail: "Use `/skills init <name>` to scaffold `.pebble/skills/<name>/SKILL.md` in the current project.",
        resume_supported: true,
    },
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HelpTopic {
    pub name: &'static str,
    pub summary: &'static str,
    pub detail: &'static str,
}

const HELP_TOPICS: &[HelpTopic] = &[
    HelpTopic {
        name: "help",
        summary: "Understand command help and topic help",
        detail: "Run `/help` for the full command guide or `/help <topic>` for focused help. Supported topics: help, auth, sessions, extensions, web.",
    },
    HelpTopic {
        name: "auth",
        summary: "Authenticate model services and Exa",
        detail: "Run `/login` or `/auth` with no argument to open a service picker, or `/logout [service]` to remove saved credentials. Model services are `nanogpt`, `neuralwatt`, `lilac`, `grok`, `synthetic`, `openai-codex`, and `opencode-go`; `exa` powers web tools. API keys and OpenAI device-code tokens live in `~/.pebble/credentials.json`. Grok OAuth stays in the official Grok CLI, and shell environment variables still take precedence.",
    },
    HelpTopic {
        name: "sessions",
        summary: "Manage, resume, and restore sessions",
        detail: "Use `/sessions` to list recent sessions, `/resume` to open the picker, `/resume last` for the most recent saved session, and `/session switch <id>` to jump directly.",
    },
    HelpTopic {
        name: "extensions",
        summary: "Add skills, MCP servers, and plugins",
        detail: "Use `/skills init <name>` to scaffold a project skill, `/mcp add <name>` to add a starter MCP server entry to `.pebble/settings.json`, `/mcp enable <name>` or `/mcp disable <name>` for local MCP overrides, and `/plugins help` for local plugin install and enable guidance.",
    },
    HelpTopic {
        name: "web",
        summary: "Provider-agnostic web search and scrape",
        detail: "Pebble always uses Exa for `WebSearch` and `WebScrape`. Configure `EXA_API_KEY` or store Exa credentials through `/login exa`.",
    },
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SlashCommand {
    Help {
        topic: Option<String>,
    },
    Status,
    Compact,
    Archives {
        action: Option<String>,
        target: Option<String>,
    },
    Reasoning {
        effort: Option<String>,
    },
    Fast {
        enabled: Option<bool>,
    },
    Mode {
        mode: Option<String>,
    },
    Model {
        model: Option<String>,
    },
    Provider {
        provider: Option<String>,
    },
    Route {
        route: Option<String>,
    },
    Logout {
        service: Option<String>,
    },
    Mcp {
        action: Option<String>,
    },
    Permissions {
        mode: Option<String>,
    },
    Clear {
        confirm: bool,
    },
    Resume {
        session_path: Option<String>,
    },
    Config {
        section: Option<String>,
    },
    Memory,
    Init,
    Diff,
    Patch {
        args: Option<String>,
    },
    Version,
    Branch {
        action: Option<String>,
        target: Option<String>,
    },
    Worktree {
        action: Option<String>,
        path: Option<String>,
        branch: Option<String>,
    },
    Export {
        path: Option<String>,
    },
    Undo,
    Redo,
    Timeline,
    Fork {
        target: Option<String>,
    },
    Rename {
        title: Option<String>,
    },
    Session {
        action: Option<String>,
        target: Option<String>,
    },
    Sessions,
    Plugins {
        action: Option<String>,
        target: Option<String>,
    },
    Agents {
        args: Option<String>,
    },
    Skills {
        args: Option<String>,
    },
    Unknown(String),
}

impl SlashCommand {
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn parse(input: &str) -> Option<Self> {
        let trimmed = input.trim();
        if !trimmed.starts_with('/') {
            return None;
        }

        let mut parts = trimmed.trim_start_matches('/').split_whitespace();
        let command = parts.next().unwrap_or_default();
        Some(match command {
            "help" => Self::Help {
                topic: parts.next().map(ToOwned::to_owned),
            },
            "status" => Self::Status,
            "compact" => Self::Compact,
            "archives" => Self::Archives {
                action: parts.next().map(ToOwned::to_owned),
                target: {
                    let remainder = parts.collect::<Vec<_>>().join(" ");
                    (!remainder.is_empty()).then_some(remainder)
                },
            },
            "undo" => Self::Undo,
            "redo" => Self::Redo,
            "timeline" => Self::Timeline,
            "fork" => Self::Fork {
                target: parts.next().map(ToOwned::to_owned),
            },
            "rename" => Self::Rename {
                title: {
                    let remainder = parts.collect::<Vec<_>>().join(" ");
                    (!remainder.is_empty()).then_some(remainder)
                },
            },
            "reasoning" | "thinking" => Self::Reasoning {
                effort: parts.next().map(ToOwned::to_owned),
            },
            "fast" => Self::Fast {
                enabled: match parts.next() {
                    Some("on") => Some(true),
                    Some("off") => Some(false),
                    Some(_) | None => None,
                },
            },
            "mode" => Self::Mode {
                mode: parts.next().map(ToOwned::to_owned),
            },
            "model" => Self::Model {
                model: parts.next().map(ToOwned::to_owned),
            },
            "provider" | "providers" => Self::Provider {
                provider: parts.next().map(ToOwned::to_owned),
            },
            "route" | "routing" => Self::Route {
                route: parts.next().map(ToOwned::to_owned),
            },
            "logout" => Self::Logout {
                service: parts.next().map(ToOwned::to_owned),
            },
            "mcp" => Self::Mcp {
                action: parts.next().map(ToOwned::to_owned),
            },
            "bypass" => Self::Permissions {
                mode: Some("danger-full-access".to_string()),
            },
            "permissions" => Self::Permissions {
                mode: parts.next().map(ToOwned::to_owned),
            },
            "clear" => Self::Clear {
                confirm: parts.next() == Some("--confirm"),
            },
            "resume" => Self::Resume {
                session_path: parts.next().map(ToOwned::to_owned),
            },
            "config" => Self::Config {
                section: parts.next().map(ToOwned::to_owned),
            },
            "memory" => Self::Memory,
            "init" => Self::Init,
            "diff" => Self::Diff,
            "patch" => Self::Patch {
                args: remainder_after_command(trimmed, command),
            },
            "version" => Self::Version,
            "branch" => Self::Branch {
                action: parts.next().map(ToOwned::to_owned),
                target: parts.next().map(ToOwned::to_owned),
            },
            "worktree" => Self::Worktree {
                action: parts.next().map(ToOwned::to_owned),
                path: parts.next().map(ToOwned::to_owned),
                branch: parts.next().map(ToOwned::to_owned),
            },
            "export" => Self::Export {
                path: parts.next().map(ToOwned::to_owned),
            },
            "session" => Self::Session {
                action: parts.next().map(ToOwned::to_owned),
                target: {
                    let remainder = parts.collect::<Vec<_>>().join(" ");
                    (!remainder.is_empty()).then_some(remainder)
                },
            },
            "sessions" => Self::Sessions,
            "plugin" | "plugins" | "marketplace" => Self::Plugins {
                action: parts.next().map(ToOwned::to_owned),
                target: {
                    let remainder = parts.collect::<Vec<_>>().join(" ");
                    (!remainder.is_empty()).then_some(remainder)
                },
            },
            "agents" => Self::Agents {
                args: remainder_after_command(trimmed, command),
            },
            "skills" => Self::Skills {
                args: remainder_after_command(trimmed, command),
            },
            other => Self::Unknown(other.to_string()),
        })
    }
}

fn remainder_after_command(input: &str, command: &str) -> Option<String> {
    input
        .trim()
        .strip_prefix(&format!("/{command}"))
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

#[must_use]
pub fn slash_command_specs() -> &'static [SlashCommandSpec] {
    SLASH_COMMAND_SPECS
}

#[must_use]
pub fn resume_supported_slash_commands() -> Vec<&'static SlashCommandSpec> {
    slash_command_specs()
        .iter()
        .filter(|spec| spec.resume_supported)
        .collect()
}

#[must_use]
pub fn render_slash_command_help() -> String {
    render_slash_command_help_topic(None)
}

#[must_use]
pub fn render_slash_command_help_topic(topic: Option<&str>) -> String {
    let topic = topic
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_ascii_lowercase);
    let Some(topic) = topic else {
        return render_full_slash_command_help();
    };

    if let Some(help_topic) = HELP_TOPICS.iter().find(|candidate| candidate.name == topic) {
        return format!(
            "Help: {}\n  {}\n\n{}",
            help_topic.name, help_topic.summary, help_topic.detail
        );
    }

    if let Some(spec) = slash_command_specs()
        .iter()
        .find(|spec| spec.name == topic || spec.aliases.iter().any(|alias| *alias == topic))
    {
        let aliases = if spec.aliases.is_empty() {
            "none".to_string()
        } else {
            spec.aliases.join(", ")
        };
        let usage = slash_command_usage(spec);
        let resume = if spec.resume_supported { "yes" } else { "no" };
        return format!(
            "Help: /{}\n  Category         {}\n  Usage            {}\n  Aliases          {}\n  Resume           {}\n\n{}",
            spec.name,
            spec.category.label(),
            usage,
            aliases,
            resume,
            spec.detail
        );
    }

    format!(
        "Unknown help topic `{topic}`.\n\n{}",
        render_help_topics_overview()
    )
}

#[must_use]
pub fn render_help_topics_overview() -> String {
    let mut lines = vec!["Help topics".to_string()];
    for topic in HELP_TOPICS {
        lines.push(format!("  {:<18} {}", topic.name, topic.summary));
    }
    lines.join("\n")
}

#[must_use]
pub fn command_names_and_aliases() -> Vec<String> {
    slash_command_specs()
        .iter()
        .flat_map(|spec| {
            std::iter::once(format!("/{}", spec.name)).chain(
                spec.aliases
                    .iter()
                    .map(|alias| format!("/{alias}"))
                    .collect::<Vec<_>>(),
            )
        })
        .collect()
}

fn render_full_slash_command_help() -> String {
    let mut lines = vec![
        "Slash commands".to_string(),
        "  [resume] means the command also works with --resume SESSION.json".to_string(),
    ];
    for category in [
        SlashCommandCategory::Core,
        SlashCommandCategory::Session,
        SlashCommandCategory::Workspace,
        SlashCommandCategory::Repl,
    ] {
        let mut category_lines = slash_command_specs()
            .iter()
            .filter(|spec| spec.category == category)
            .map(|spec| {
                let aliases = if spec.aliases.is_empty() {
                    String::new()
                } else {
                    format!(" (aliases: {})", spec.aliases.join(", "))
                };
                let resume = if spec.resume_supported {
                    " [resume]"
                } else {
                    ""
                };
                format!(
                    "  {:<32} {}{}{}",
                    slash_command_usage(spec),
                    spec.summary,
                    aliases,
                    resume
                )
            })
            .collect::<Vec<_>>();
        if category_lines.is_empty() {
            continue;
        }
        lines.push(String::new());
        lines.push(category.label().to_string());
        lines.append(&mut category_lines);
    }
    lines.push(String::new());
    lines.push(render_help_topics_overview());
    lines.join("\n")
}

fn slash_command_usage(spec: &SlashCommandSpec) -> String {
    match spec.argument_hint {
        Some(argument_hint) => format!("/{} {}", spec.name, argument_hint),
        None => format!("/{}", spec.name),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlashCommandResult {
    pub message: String,
    pub session: Session,
}

#[must_use]
pub fn handle_slash_command(
    input: &str,
    session: &Session,
    compaction: CompactionConfig,
) -> Option<SlashCommandResult> {
    match SlashCommand::parse(input)? {
        SlashCommand::Compact => {
            let result = compact_session(session, compaction);
            let message = if result.removed_message_count == 0 {
                "Compaction skipped: session is below the compaction threshold.".to_string()
            } else {
                format!(
                    "Compacted {} messages into a resumable system summary.",
                    result.removed_message_count
                )
            };
            Some(SlashCommandResult {
                message,
                session: result.compacted_session,
            })
        }
        SlashCommand::Help { topic } => Some(SlashCommandResult {
            message: render_slash_command_help_topic(topic.as_deref()),
            session: session.clone(),
        }),
        SlashCommand::Archives { .. }
        | SlashCommand::Status
        | SlashCommand::Undo
        | SlashCommand::Redo
        | SlashCommand::Timeline
        | SlashCommand::Fork { .. }
        | SlashCommand::Rename { .. }
        | SlashCommand::Reasoning { .. }
        | SlashCommand::Fast { .. }
        | SlashCommand::Mode { .. }
        | SlashCommand::Model { .. }
        | SlashCommand::Provider { .. }
        | SlashCommand::Route { .. }
        | SlashCommand::Logout { .. }
        | SlashCommand::Mcp { .. }
        | SlashCommand::Permissions { .. }
        | SlashCommand::Clear { .. }
        | SlashCommand::Resume { .. }
        | SlashCommand::Config { .. }
        | SlashCommand::Memory
        | SlashCommand::Init
        | SlashCommand::Diff
        | SlashCommand::Patch { .. }
        | SlashCommand::Version
        | SlashCommand::Branch { .. }
        | SlashCommand::Worktree { .. }
        | SlashCommand::Export { .. }
        | SlashCommand::Session { .. }
        | SlashCommand::Sessions
        | SlashCommand::Plugins { .. }
        | SlashCommand::Agents { .. }
        | SlashCommand::Skills { .. }
        | SlashCommand::Unknown(_) => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum DefinitionSource {
    Project,
    UserConfigHome,
}

impl DefinitionSource {
    fn label(self) -> &'static str {
        match self {
            Self::Project => "Project (.pebble)",
            Self::UserConfigHome => "User (PEBBLE_CONFIG_HOME)",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AgentSummary {
    name: String,
    description: Option<String>,
    model: Option<String>,
    reasoning_effort: Option<String>,
    source: DefinitionSource,
    shadowed_by: Option<DefinitionSource>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SkillSummary {
    name: String,
    description: Option<String>,
    source: DefinitionSource,
    shadowed_by: Option<DefinitionSource>,
    origin: SkillOrigin,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SkillOrigin {
    SkillsDir,
    LegacyCommandsDir,
}

impl SkillOrigin {
    fn detail_label(self) -> Option<&'static str> {
        match self {
            Self::SkillsDir => None,
            Self::LegacyCommandsDir => Some("legacy /commands"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SkillRoot {
    source: DefinitionSource,
    path: PathBuf,
    origin: SkillOrigin,
}

pub fn handle_branch_slash_command(
    action: Option<&str>,
    target: Option<&str>,
    cwd: &Path,
) -> io::Result<String> {
    match normalize_optional_args(action) {
        None | Some("list") => {
            let branches = git_stdout(cwd, &["branch", "--list", "--verbose"])?;
            let trimmed = branches.trim();
            Ok(if trimmed.is_empty() {
                "Branch\n  Result           no branches found".to_string()
            } else {
                format!("Branch\n  Result           listed\n\n{trimmed}")
            })
        }
        Some("create") => {
            let Some(target) = target.filter(|value| !value.trim().is_empty()) else {
                return Ok("Usage: /branch create <name>".to_string());
            };
            git_status_ok(cwd, &["switch", "-c", target])?;
            Ok(format!(
                "Branch\n  Result           created and switched\n  Branch           {target}"
            ))
        }
        Some("switch") => {
            let Some(target) = target.filter(|value| !value.trim().is_empty()) else {
                return Ok("Usage: /branch switch <name>".to_string());
            };
            git_status_ok(cwd, &["switch", target])?;
            Ok(format!(
                "Branch\n  Result           switched\n  Branch           {target}"
            ))
        }
        Some(other) => Ok(format!(
            "Unknown /branch action '{other}'. Use /branch list, /branch create <name>, or /branch switch <name>."
        )),
    }
}

pub fn handle_worktree_slash_command(
    action: Option<&str>,
    path: Option<&str>,
    branch: Option<&str>,
    cwd: &Path,
) -> io::Result<String> {
    match normalize_optional_args(action) {
        None | Some("list") => {
            let worktrees = git_stdout(cwd, &["worktree", "list"])?;
            let trimmed = worktrees.trim();
            Ok(if trimmed.is_empty() {
                "Worktree\n  Result           no worktrees found".to_string()
            } else {
                format!("Worktree\n  Result           listed\n\n{trimmed}")
            })
        }
        Some("add") => {
            let Some(path) = path.filter(|value| !value.trim().is_empty()) else {
                return Ok("Usage: /worktree add <path> [branch]".to_string());
            };
            if let Some(branch) = branch.filter(|value| !value.trim().is_empty()) {
                if branch_exists(cwd, branch) {
                    git_status_ok(cwd, &["worktree", "add", path, branch])?;
                } else {
                    git_status_ok(cwd, &["worktree", "add", path, "-b", branch])?;
                }
                Ok(format!(
                    "Worktree\n  Result           added\n  Path             {path}\n  Branch           {branch}"
                ))
            } else {
                git_status_ok(cwd, &["worktree", "add", path])?;
                Ok(format!(
                    "Worktree\n  Result           added\n  Path             {path}"
                ))
            }
        }
        Some("remove") => {
            let Some(path) = path.filter(|value| !value.trim().is_empty()) else {
                return Ok("Usage: /worktree remove <path>".to_string());
            };
            git_status_ok(cwd, &["worktree", "remove", path])?;
            Ok(format!(
                "Worktree\n  Result           removed\n  Path             {path}"
            ))
        }
        Some("prune") => {
            git_status_ok(cwd, &["worktree", "prune"])?;
            Ok("Worktree\n  Result           pruned".to_string())
        }
        Some(other) => Ok(format!(
            "Unknown /worktree action '{other}'. Use /worktree list, /worktree add <path> [branch], /worktree remove <path>, or /worktree prune."
        )),
    }
}

pub fn handle_agents_slash_command(args: Option<&str>, cwd: &Path) -> io::Result<String> {
    match normalize_optional_args(args) {
        None | Some("list") => {
            let roots = discover_definition_roots(cwd, "agents");
            let agents = load_agents_from_roots(&roots)?;
            Ok(render_agents_report(&agents))
        }
        Some("-h" | "--help" | "help") => Ok(render_agents_usage(None)),
        Some(args) => Ok(render_agents_usage(Some(args))),
    }
}

pub fn handle_skills_slash_command(args: Option<&str>, cwd: &Path) -> io::Result<String> {
    match normalize_optional_args(args) {
        None | Some("list") => {
            let roots = discover_skill_roots(cwd);
            let skills = load_skills_from_roots(&roots)?;
            Ok(render_skills_report(&skills))
        }
        Some("-h" | "--help" | "help") => Ok(render_skills_usage(None)),
        Some(args) if args.starts_with("init ") => scaffold_skill(args, cwd),
        Some(args) => Ok(render_skills_usage(Some(args))),
    }
}

fn scaffold_skill(args: &str, cwd: &Path) -> io::Result<String> {
    let name = args.trim_start_matches("init").trim();
    if name.is_empty() {
        return Ok("Skills\n  error: missing skill name\n  usage: /skills init <name>".to_string());
    }
    if name.contains('/') || name.contains('\\') {
        return Ok(format!(
            "Skills\n  error: invalid skill name `{name}`\n  hint: skill names become directory names under .pebble/skills/"
        ));
    }

    let skill_dir = cwd.join(".pebble").join("skills").join(name);
    fs::create_dir_all(&skill_dir)?;
    let skill_file = skill_dir.join("SKILL.md");
    if skill_file.exists() {
        return Ok(format!(
            "Skills\n  result:  exists\n  name:    {name}\n  file:    {}",
            skill_file.display()
        ));
    }

    let body = format!(
        "# {name}\n\nDescribe when this skill should be used and what it should do.\n\n## Workflow\n\n1. Capture the goal.\n2. Read only the files you need.\n3. Perform the task.\n4. Report the result.\n"
    );
    write_atomic(&skill_file, body)?;
    Ok(format!(
        "Skills\n  result:  initialized\n  name:    {name}\n  file:    {}",
        skill_file.display()
    ))
}

fn normalize_optional_args(args: Option<&str>) -> Option<&str> {
    args.map(str::trim).filter(|value| !value.is_empty())
}

fn git_stdout(cwd: &Path, args: &[&str]) -> io::Result<String> {
    run_command_stdout("git", args, cwd)
}

fn git_status_ok(cwd: &Path, args: &[&str]) -> io::Result<()> {
    run_command_success("git", args, cwd)
}

fn run_command_stdout(program: &str, args: &[&str], cwd: &Path) -> io::Result<String> {
    let output = Command::new(program).args(args).current_dir(cwd).output()?;
    if !output.status.success() {
        return Err(io::Error::other(command_failure(program, args, &output)));
    }
    String::from_utf8(output.stdout)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
}

fn run_command_success(program: &str, args: &[&str], cwd: &Path) -> io::Result<()> {
    let output = Command::new(program).args(args).current_dir(cwd).output()?;
    if !output.status.success() {
        return Err(io::Error::other(command_failure(program, args, &output)));
    }
    Ok(())
}

fn command_failure(program: &str, args: &[&str], output: &std::process::Output) -> String {
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let detail = if stderr.is_empty() { stdout } else { stderr };
    if detail.is_empty() {
        format!("{program} {} failed", args.join(" "))
    } else {
        format!("{program} {} failed: {detail}", args.join(" "))
    }
}

fn branch_exists(cwd: &Path, branch: &str) -> bool {
    Command::new("git")
        .args([
            "show-ref",
            "--verify",
            "--quiet",
            &format!("refs/heads/{branch}"),
        ])
        .current_dir(cwd)
        .output()
        .is_ok_and(|output| output.status.success())
}

fn discover_definition_roots(cwd: &Path, leaf: &str) -> Vec<(DefinitionSource, PathBuf)> {
    let mut roots = Vec::new();

    for ancestor in cwd.ancestors() {
        push_unique_root(
            &mut roots,
            DefinitionSource::Project,
            ancestor.join(".pebble").join(leaf),
        );
    }

    if let Some(config_home) = pebble_config_home() {
        push_unique_root(
            &mut roots,
            DefinitionSource::UserConfigHome,
            config_home.join(leaf),
        );
    }

    roots
}

fn discover_skill_roots(cwd: &Path) -> Vec<SkillRoot> {
    let mut roots = Vec::new();

    for ancestor in cwd.ancestors() {
        push_unique_skill_root(
            &mut roots,
            DefinitionSource::Project,
            ancestor.join(".pebble").join("skills"),
            SkillOrigin::SkillsDir,
        );
        push_unique_skill_root(
            &mut roots,
            DefinitionSource::Project,
            ancestor.join(".pebble").join("commands"),
            SkillOrigin::LegacyCommandsDir,
        );
    }

    if let Some(config_home) = pebble_config_home() {
        push_unique_skill_root(
            &mut roots,
            DefinitionSource::UserConfigHome,
            config_home.join("skills"),
            SkillOrigin::SkillsDir,
        );
        push_unique_skill_root(
            &mut roots,
            DefinitionSource::UserConfigHome,
            config_home.join("commands"),
            SkillOrigin::LegacyCommandsDir,
        );
    }

    roots
}

fn push_unique_root(
    roots: &mut Vec<(DefinitionSource, PathBuf)>,
    source: DefinitionSource,
    path: PathBuf,
) {
    if path.is_dir() && !roots.iter().any(|(_, existing)| existing == &path) {
        roots.push((source, path));
    }
}

fn push_unique_skill_root(
    roots: &mut Vec<SkillRoot>,
    source: DefinitionSource,
    path: PathBuf,
    origin: SkillOrigin,
) {
    if path.is_dir() && !roots.iter().any(|existing| existing.path == path) {
        roots.push(SkillRoot {
            source,
            path,
            origin,
        });
    }
}

fn load_agents_from_roots(roots: &[(DefinitionSource, PathBuf)]) -> io::Result<Vec<AgentSummary>> {
    let mut agents = Vec::new();
    let mut active_sources = BTreeMap::<String, DefinitionSource>::new();

    for (source, root) in roots {
        let mut root_agents = Vec::new();
        for entry in fs::read_dir(root)? {
            let entry = entry?;
            if entry.path().extension().is_none_or(|ext| ext != "toml") {
                continue;
            }
            let contents = fs::read_to_string(entry.path())?;
            let fallback_name = entry.path().file_stem().map_or_else(
                || entry.file_name().to_string_lossy().to_string(),
                |stem| stem.to_string_lossy().to_string(),
            );
            root_agents.push(AgentSummary {
                name: parse_toml_string(&contents, "name").unwrap_or(fallback_name),
                description: parse_toml_string(&contents, "description"),
                model: parse_toml_string(&contents, "model"),
                reasoning_effort: parse_toml_string(&contents, "model_reasoning_effort"),
                source: *source,
                shadowed_by: None,
            });
        }
        root_agents.sort_by(|left, right| left.name.cmp(&right.name));

        for mut agent in root_agents {
            let key = agent.name.to_ascii_lowercase();
            if let Some(existing) = active_sources.get(&key) {
                agent.shadowed_by = Some(*existing);
            } else {
                active_sources.insert(key, agent.source);
            }
            agents.push(agent);
        }
    }

    Ok(agents)
}

fn load_skills_from_roots(roots: &[SkillRoot]) -> io::Result<Vec<SkillSummary>> {
    let mut skills = Vec::new();
    let mut active_sources = BTreeMap::<String, DefinitionSource>::new();

    for root in roots {
        let mut root_skills = Vec::new();
        for entry in fs::read_dir(&root.path)? {
            let entry = entry?;
            match root.origin {
                SkillOrigin::SkillsDir => {
                    if !entry.path().is_dir() {
                        continue;
                    }
                    let skill_path = entry.path().join("SKILL.md");
                    if !skill_path.is_file() {
                        continue;
                    }
                    let contents = fs::read_to_string(&skill_path)?;
                    let name = entry.file_name().to_string_lossy().to_string();
                    root_skills.push(SkillSummary {
                        name,
                        description: parse_skill_description(&contents),
                        source: root.source,
                        shadowed_by: None,
                        origin: root.origin,
                    });
                }
                SkillOrigin::LegacyCommandsDir => {
                    if entry.path().extension().is_none_or(|ext| ext != "md") {
                        continue;
                    }
                    let contents = fs::read_to_string(entry.path())?;
                    let name = entry.path().file_stem().map_or_else(
                        || entry.file_name().to_string_lossy().to_string(),
                        |stem| stem.to_string_lossy().to_string(),
                    );
                    root_skills.push(SkillSummary {
                        name,
                        description: parse_skill_description(&contents),
                        source: root.source,
                        shadowed_by: None,
                        origin: root.origin,
                    });
                }
            }
        }
        root_skills.sort_by(|left, right| left.name.cmp(&right.name));

        for mut skill in root_skills {
            let key = skill.name.to_ascii_lowercase();
            if let Some(existing) = active_sources.get(&key) {
                skill.shadowed_by = Some(*existing);
            } else {
                active_sources.insert(key, skill.source);
            }
            skills.push(skill);
        }
    }

    Ok(skills)
}

fn parse_toml_string(contents: &str, key: &str) -> Option<String> {
    let key = format!("{key} = ");
    contents
        .lines()
        .map(str::trim)
        .find(|line| line.starts_with(&key))
        .and_then(|line| line.split_once('='))
        .map(|(_, value)| value.trim().trim_matches('"').to_string())
        .filter(|value| !value.is_empty())
}

fn parse_skill_description(contents: &str) -> Option<String> {
    contents
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty() && !line.starts_with('#'))
        .map(ToOwned::to_owned)
}

fn render_agents_report(agents: &[AgentSummary]) -> String {
    let mut lines = vec!["Agents".to_string()];
    if agents.is_empty() {
        lines.push("  No agents found.".to_string());
        return lines.join("\n");
    }

    for agent in agents {
        let mut line = format!("  {}", agent.name);
        if let Some(model) = &agent.model {
            let _ = write!(line, "  model={model}");
        }
        let _ = write!(line, "  source={}", agent.source.label());
        if let Some(shadowed_by) = agent.shadowed_by {
            let _ = write!(line, "  shadowed-by={}", shadowed_by.label());
        }
        lines.push(line);
        if let Some(description) = &agent.description {
            lines.push(format!("    {description}"));
        }
        if let Some(reasoning_effort) = &agent.reasoning_effort {
            lines.push(format!("    reasoning_effort={reasoning_effort}"));
        }
    }

    lines.join("\n")
}

fn render_skills_report(skills: &[SkillSummary]) -> String {
    let mut lines = vec!["Skills".to_string()];
    if skills.is_empty() {
        lines.push("  No skills found.".to_string());
        return lines.join("\n");
    }

    for skill in skills {
        let mut line = format!("  {}  source={}", skill.name, skill.source.label());
        if let Some(detail) = skill.origin.detail_label() {
            let _ = write!(line, "  type={detail}");
        }
        if let Some(shadowed_by) = skill.shadowed_by {
            let _ = write!(line, "  shadowed-by={}", shadowed_by.label());
        }
        lines.push(line);
        if let Some(description) = &skill.description {
            lines.push(format!("    {description}"));
        }
    }

    lines.join("\n")
}

fn render_agents_usage(args: Option<&str>) -> String {
    let mut lines = vec![
        "Agents".to_string(),
        "  Usage            /agents [list|help]".to_string(),
    ];
    if let Some(args) = args {
        lines.push(format!("  Unsupported      {}", args.trim()));
    }
    lines.join("\n")
}

fn render_skills_usage(args: Option<&str>) -> String {
    let mut lines = vec![
        "Skills".to_string(),
        "  Usage            /skills [list|help|init <name>]".to_string(),
        "  Init             Create .pebble/skills/<name>/SKILL.md in the current project"
            .to_string(),
    ];
    if let Some(args) = args {
        lines.push(format!("  Unsupported      {}", args.trim()));
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::{
        handle_slash_command, render_slash_command_help, resume_supported_slash_commands,
        slash_command_specs, SlashCommand,
    };
    use runtime::{CompactionConfig, ContentBlock, ConversationMessage, MessageRole, Session};

    #[test]
    #[allow(clippy::too_many_lines)]
    fn parses_supported_slash_commands() {
        assert_eq!(
            SlashCommand::parse("/help"),
            Some(SlashCommand::Help { topic: None })
        );
        assert_eq!(SlashCommand::parse(" /status "), Some(SlashCommand::Status));
        assert_eq!(
            SlashCommand::parse("/model claude-opus"),
            Some(SlashCommand::Model {
                model: Some("claude-opus".to_string()),
            })
        );
        assert_eq!(
            SlashCommand::parse("/model"),
            Some(SlashCommand::Model { model: None })
        );
        assert_eq!(
            SlashCommand::parse("/provider lilac"),
            Some(SlashCommand::Provider {
                provider: Some("lilac".to_string()),
            })
        );
        assert_eq!(
            SlashCommand::parse("/route default"),
            Some(SlashCommand::Route {
                route: Some("default".to_string()),
            })
        );
        assert_eq!(
            SlashCommand::parse("/logout openai-codex"),
            Some(SlashCommand::Logout {
                service: Some("openai-codex".to_string()),
            })
        );
        assert_eq!(
            SlashCommand::parse("/permissions read-only"),
            Some(SlashCommand::Permissions {
                mode: Some("read-only".to_string()),
            })
        );
        assert_eq!(
            SlashCommand::parse("/bypass"),
            Some(SlashCommand::Permissions {
                mode: Some("danger-full-access".to_string()),
            })
        );
        assert_eq!(
            SlashCommand::parse("/clear"),
            Some(SlashCommand::Clear { confirm: false })
        );
        assert_eq!(
            SlashCommand::parse("/clear --confirm"),
            Some(SlashCommand::Clear { confirm: true })
        );
        assert_eq!(
            SlashCommand::parse("/reasoning high"),
            Some(SlashCommand::Reasoning {
                effort: Some("high".to_string()),
            })
        );
        assert_eq!(
            SlashCommand::parse("/thinking"),
            Some(SlashCommand::Reasoning { effort: None })
        );
        assert_eq!(
            SlashCommand::parse("/fast on"),
            Some(SlashCommand::Fast {
                enabled: Some(true)
            })
        );
        assert_eq!(
            SlashCommand::parse("/mode plan"),
            Some(SlashCommand::Mode {
                mode: Some("plan".to_string()),
            })
        );
        assert_eq!(
            SlashCommand::parse("/resume session.json"),
            Some(SlashCommand::Resume {
                session_path: Some("session.json".to_string()),
            })
        );
        assert_eq!(
            SlashCommand::parse("/archives"),
            Some(SlashCommand::Archives {
                action: None,
                target: None,
            })
        );
        assert_eq!(
            SlashCommand::parse("/archives show tool-1"),
            Some(SlashCommand::Archives {
                action: Some("show".to_string()),
                target: Some("tool-1".to_string()),
            })
        );
        assert_eq!(SlashCommand::parse("/undo"), Some(SlashCommand::Undo));
        assert_eq!(SlashCommand::parse("/redo"), Some(SlashCommand::Redo));
        assert_eq!(
            SlashCommand::parse("/timeline"),
            Some(SlashCommand::Timeline)
        );
        assert_eq!(
            SlashCommand::parse("/fork msg-1"),
            Some(SlashCommand::Fork {
                target: Some("msg-1".to_string())
            })
        );
        assert_eq!(
            SlashCommand::parse("/rename Working title"),
            Some(SlashCommand::Rename {
                title: Some("Working title".to_string())
            })
        );
        assert_eq!(
            SlashCommand::parse("/config"),
            Some(SlashCommand::Config { section: None })
        );
        assert_eq!(
            SlashCommand::parse("/config env"),
            Some(SlashCommand::Config {
                section: Some("env".to_string())
            })
        );
        assert_eq!(SlashCommand::parse("/memory"), Some(SlashCommand::Memory));
        assert_eq!(SlashCommand::parse("/init"), Some(SlashCommand::Init));
        assert_eq!(SlashCommand::parse("/diff"), Some(SlashCommand::Diff));
        assert_eq!(
            SlashCommand::parse("/patch --apply changes.diff"),
            Some(SlashCommand::Patch {
                args: Some("--apply changes.diff".to_string())
            })
        );
        assert_eq!(SlashCommand::parse("/version"), Some(SlashCommand::Version));
        assert_eq!(
            SlashCommand::parse("/branch create feat/demo"),
            Some(SlashCommand::Branch {
                action: Some("create".to_string()),
                target: Some("feat/demo".to_string()),
            })
        );
        assert_eq!(
            SlashCommand::parse("/worktree add ../demo feature/demo"),
            Some(SlashCommand::Worktree {
                action: Some("add".to_string()),
                path: Some("../demo".to_string()),
                branch: Some("feature/demo".to_string()),
            })
        );
        assert_eq!(
            SlashCommand::parse("/export notes.txt"),
            Some(SlashCommand::Export {
                path: Some("notes.txt".to_string())
            })
        );
        assert_eq!(
            SlashCommand::parse("/session switch abc123"),
            Some(SlashCommand::Session {
                action: Some("switch".to_string()),
                target: Some("abc123".to_string())
            })
        );
        assert_eq!(
            SlashCommand::parse("/plugins enable demo@external"),
            Some(SlashCommand::Plugins {
                action: Some("enable".to_string()),
                target: Some("demo@external".to_string())
            })
        );
        assert_eq!(
            SlashCommand::parse("/marketplace list"),
            Some(SlashCommand::Plugins {
                action: Some("list".to_string()),
                target: None
            })
        );
        assert_eq!(
            SlashCommand::parse("/agents"),
            Some(SlashCommand::Agents { args: None })
        );
        assert_eq!(
            SlashCommand::parse("/skills help"),
            Some(SlashCommand::Skills {
                args: Some("help".to_string())
            })
        );
    }

    #[test]
    fn renders_help_from_shared_specs() {
        let help = render_slash_command_help();
        assert!(help.contains("works with --resume SESSION.json"));
        assert!(help.contains("/help"));
        assert!(help.contains("/status"));
        assert!(help.contains("/compact"));
        assert!(help.contains(
            "/archives [list|show <message-id-or-tool-call-id>|page <message-id-or-tool-call-id>|save <message-id-or-tool-call-id> [file]]"
        ));
        assert!(help.contains("/undo"));
        assert!(help.contains("/redo"));
        assert!(help.contains("/timeline"));
        assert!(help.contains("/fork [message-id-or-index]"));
        assert!(help.contains("/rename <title>"));
        assert!(help.contains("/reasoning [default|minimal|low|medium|high|xhigh]"));
        assert!(help.contains("aliases: thinking"));
        assert!(help.contains("/fast [on|off]"));
        assert!(help.contains("/mode [build|plan]"));
        assert!(help.contains("/model [model]"));
        assert!(help.contains("/provider [name]"));
        assert!(help.contains("/route [route-id|default]"));
        assert!(help.contains("/logout [service]"));
        assert!(help.contains("/mcp [status|tools|reload|add <name>|enable <name>|disable <name>]"));
        assert!(help.contains("/permissions [read-only|workspace-write|danger-full-access]"));
        assert!(help.contains("aliases: bypass"));
        assert!(help.contains("/clear [--confirm]"));
        assert!(help.contains("/resume [session-id-or-path]"));
        assert!(help.contains("/config [check|env|hooks|model|plugins]"));
        assert!(help.contains("/memory"));
        assert!(help.contains("/init"));
        assert!(help.contains("/diff"));
        assert!(help.contains("/patch [--check|--apply] [patch-file-or-inline-diff]"));
        assert!(help.contains("/version"));
        assert!(help.contains("/branch [list|create <name>|switch <name>]"));
        assert!(help.contains("/worktree [list|add <path> [branch]|remove <path>|prune]"));
        assert!(help.contains("/export [file]"));
        assert!(help.contains(
            "/session [list|switch <session-id>|timeline|fork [message-id]|rename <title>]"
        ));
        assert!(help.contains("/sessions"));
        assert!(help.contains(
            "/plugins [list|help|install <path>|enable <id>|disable <id>|uninstall <id>|update <id>]"
        ));
        assert!(help.contains("/agents [list|help]"));
        assert!(help.contains("/skills [list|help|init <name>]"));
        assert!(help.contains("Help topics"));
        assert_eq!(slash_command_specs().len(), 34);
        assert_eq!(resume_supported_slash_commands().len(), 15);
    }

    #[test]
    fn compacts_sessions_via_slash_command() {
        let root = std::env::temp_dir().join(format!(
            "commands-compact-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time")
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).expect("temp dir");
        let previous = std::env::current_dir().expect("cwd");
        std::env::set_current_dir(&root).expect("set cwd");

        let session = Session {
            version: 1,
            messages: vec![
                ConversationMessage::user_text("a ".repeat(200)),
                ConversationMessage::assistant(vec![ContentBlock::Text {
                    text: "b ".repeat(200),
                }]),
                ConversationMessage::tool_result("1", "bash", "ok ".repeat(200), false),
                ConversationMessage::assistant(vec![ContentBlock::Text {
                    text: "recent".to_string(),
                }]),
            ],
            metadata: None,
        };

        let result = handle_slash_command(
            "/compact",
            &session,
            CompactionConfig {
                preserve_recent_messages: 2,
                max_estimated_tokens: 1,
                preserve_recent_tokens: None,
                ..CompactionConfig::default()
            },
        )
        .expect("slash command should be handled");

        std::env::set_current_dir(previous).expect("restore cwd");
        let _ = std::fs::remove_dir_all(root);

        assert!(result.message.contains("Compacted 4 messages"));
        assert_eq!(result.session.messages[0].role, MessageRole::System);
    }

    #[test]
    fn help_command_is_non_mutating() {
        let session = Session::new();
        let result = handle_slash_command("/help", &session, CompactionConfig::default())
            .expect("help command should be handled");
        assert_eq!(result.session, session);
        assert!(result.message.contains("Slash commands"));
    }

    #[test]
    fn ignores_unknown_or_runtime_bound_slash_commands() {
        let session = Session::new();
        assert!(handle_slash_command("/unknown", &session, CompactionConfig::default()).is_none());
        assert!(handle_slash_command("/status", &session, CompactionConfig::default()).is_none());
        assert!(handle_slash_command("/archives", &session, CompactionConfig::default()).is_none());
        assert!(
            handle_slash_command("/model claude", &session, CompactionConfig::default()).is_none()
        );
        assert!(handle_slash_command(
            "/permissions read-only",
            &session,
            CompactionConfig::default()
        )
        .is_none());
        assert!(handle_slash_command("/clear", &session, CompactionConfig::default()).is_none());
        assert!(
            handle_slash_command("/clear --confirm", &session, CompactionConfig::default())
                .is_none()
        );
        assert!(
            handle_slash_command("/reasoning high", &session, CompactionConfig::default())
                .is_none()
        );
        assert!(handle_slash_command(
            "/resume session.json",
            &session,
            CompactionConfig::default()
        )
        .is_none());
        assert!(handle_slash_command("/config", &session, CompactionConfig::default()).is_none());
        assert!(
            handle_slash_command("/config env", &session, CompactionConfig::default()).is_none()
        );
        assert!(handle_slash_command("/diff", &session, CompactionConfig::default()).is_none());
        assert!(
            handle_slash_command("/patch changes.diff", &session, CompactionConfig::default())
                .is_none()
        );
        assert!(handle_slash_command("/version", &session, CompactionConfig::default()).is_none());
        assert!(
            handle_slash_command("/branch list", &session, CompactionConfig::default()).is_none()
        );
        assert!(
            handle_slash_command("/worktree list", &session, CompactionConfig::default()).is_none()
        );
        assert!(
            handle_slash_command("/export note.txt", &session, CompactionConfig::default())
                .is_none()
        );
        assert!(
            handle_slash_command("/session list", &session, CompactionConfig::default()).is_none()
        );
        assert!(
            handle_slash_command("/plugins list", &session, CompactionConfig::default()).is_none()
        );
        assert!(handle_slash_command("/agents", &session, CompactionConfig::default()).is_none());
        assert!(handle_slash_command("/skills", &session, CompactionConfig::default()).is_none());
    }
}
