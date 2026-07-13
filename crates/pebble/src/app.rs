use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::env;
use std::fmt::Write as _;
use std::fs;
use std::io::{self, IsTerminal, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::process::Stdio;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use api::{
    resolve_api_key as resolve_nanogpt_api_key, resolve_api_key_for, resolve_base_url_for,
    save_openai_codex_credentials, ApiError, ApiService, InputMessage, MessageRequest,
    MessageResponse, NanoGptClient, OpenAiCodexCredentials, OutputContentBlock, ReasoningEffort,
    OPENAI_CODEX_CLIENT_ID, OPENAI_CODEX_ISSUER,
};
use crossterm::event::{
    self, DisableBracketedPaste, EnableBracketedPaste, Event, KeyCode, KeyEvent, KeyEventKind,
    KeyModifiers,
};
use crossterm::execute;
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};
use platform::{pebble_config_home as resolve_pebble_config_home, write_atomic};
use plugins::{PluginError, PluginManager, PluginSummary};
use reqwest::blocking::Client as BlockingClient;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use sha2::{Digest, Sha256};

use crate::eval::{
    eval_compare_json_report, eval_history_json_report, eval_replay_json_report, load_eval_report,
    load_eval_suite, rebuild_eval_history_index, render_eval_compare_report,
    render_eval_history_report, render_eval_replay_report, run_eval_capture,
    write_eval_history_index, EvalCaptureOptions, EvalHistoryEntry, EvalHistoryFilter,
    EvalReplayOptions, EVAL_HISTORY_INDEX_FILE,
};
use crate::eval_runner::run_eval_suite;
use crate::init::{initialize_repo, initialize_repo_with_pebble_md, render_init_pebble_md};
use crate::input;
use crate::interrupt::InterruptGuard;
use crate::mcp::{
    add_mcp_server_interactive, handle_mcp_action, load_mcp_catalog, print_mcp_status,
    print_mcp_tools, set_mcp_server_enabled, McpCatalog, McpCommand,
};
use crate::models::{
    context_length_for_model, current_service_or_default, default_model_or,
    infer_service_for_model, load_model_state, max_output_tokens_for_model_or, open_model_picker,
    open_model_picker_for_service, open_provider_picker, persist_current_model,
    persist_provider_for_model, persist_proxy_tool_calls, provider_for_model,
    proxy_tool_calls_enabled, refresh_model_catalog, save_model_state, service_from_selector,
    validate_provider_for_model, verify_model_service_credentials,
};
use crate::provider_auth::{
    parse_auth_command, parse_login_tokens, parse_logout_command, parse_logout_tokens,
    prompt_for_auth_service_selection, remove_saved_credentials, run_grok_auth_command,
    save_credentials, AuthService, CredentialRemovalOutcome,
};
#[cfg(test)]
use crate::provider_auth::{LoginCommand, LogoutCommand};
use crate::proxy::{build_proxy_system_prompt, parse_proxy_value, ProxyCommand, RuntimeToolSpec};
use crate::report::{report_label, report_section, report_title, truncate_for_summary};
use crate::runtime_client::{
    effective_reasoning_effort, permission_policy, CliToolExecutor, PebbleRuntimeClient,
};
use crate::session_store::{
    append_undo_snapshot, build_turn_snapshot, create_managed_session_handle,
    current_timestamp_rfc3339ish, derive_session_metadata, file_changes_from_turn_messages,
    list_managed_sessions, pop_redo_snapshot, pop_undo_snapshot, push_redo_snapshot,
    push_undo_snapshot, render_session_list, render_session_timeline, resolve_session_reference,
    resolve_timeline_target, restore_snapshot_files, session_runtime_state, SessionHandle,
    SnapshotDirection, WorktreeSnapshot,
};
use crate::trace_view::{
    load_turn_trace, render_replay_report, render_trace_report, replay_json_report,
    trace_json_report,
};
use crate::ui::{self, Stylize};
use commands::{
    command_names_and_aliases, handle_agents_slash_command, handle_branch_slash_command,
    handle_skills_slash_command, handle_worktree_slash_command, render_help_topics_overview,
    render_slash_command_help, render_slash_command_help_topic, SlashCommand,
};
use compat_harness::{extract_manifest, UpstreamPaths};
use runtime::{
    apply_patch, auto_compaction_threshold_from_env, get_compact_continuation_message,
    get_tool_result_context_output, load_system_prompt_with_model_family, resolve_sandbox_status,
    CancellationToken, CompactionConfig, ConfigLoader, ConfigSource, ContentBlock,
    ConversationMessage, ConversationRuntime, MessageRole, PermissionMode,
    PermissionPromptDecision, PermissionPrompter, PermissionRequest, RuntimeError,
    RuntimeJsonValue, RuntimeRetentionConfig, Session, SessionTurnSnapshot, TokenUsage,
    TurnSummary, UsageTracker,
};
use tools::{build_plugin_manager, current_tool_registry, GlobalToolRegistry};

pub(crate) const DEFAULT_MODEL: &str = "zai-org/glm-5.1";
const DEFAULT_MAX_TOKENS: u32 = 4096;
const INIT_PEBBLE_MD_MAX_TOKENS: u32 = 2_048;
const BUILD_DATE: &str = match option_env!("PEBBLE_BUILD_DATE") {
    Some(date) => date,
    None => "unknown",
};
const SECRET_PROMPT_STALE_ENTER_WINDOW: Duration = Duration::from_millis(150);
const OPENAI_CODEX_DEVICE_AUTH_TIMEOUT: Duration = Duration::from_secs(15 * 60);
const OPENAI_CODEX_DEVICE_POLL_SAFETY_MARGIN: Duration = Duration::from_secs(3);
const MAX_INIT_CONTEXT_CHARS: usize = 1_200;
const MAX_INIT_CONTEXT_FILES: usize = 6;
const MAX_INIT_TOP_LEVEL_ENTRIES: usize = 40;
const AUTO_COMPACTION_CONTEXT_UTILIZATION_PERCENT: u64 = 85;
const AUTO_COMPACTION_CONTEXT_SAFETY_MARGIN_TOKENS: u64 = 8_192;
pub(crate) const MAX_TURN_SNAPSHOT_STACK_ENTRIES: usize = 20;
const VERSION: &str = env!("CARGO_PKG_VERSION");
const BUILD_TARGET: Option<&str> = option_env!("PEBBLE_BUILD_TARGET");
const SELF_UPDATE_REPOSITORY: &str = "nanogpt-community/pebble";
const SELF_UPDATE_LATEST_RELEASE_URL: &str =
    "https://api.github.com/repos/nanogpt-community/pebble/releases/latest";
const SELF_UPDATE_USER_AGENT: &str = "pebble-self-update";
const OLD_SESSION_COMPACTION_AGE_SECS: u64 = 60 * 60 * 24;
const SECS_PER_DAY: u64 = 60 * 60 * 24;
const CHECKSUM_ASSET_CANDIDATES: &[&str] = &[
    "SHA256SUMS",
    "SHA256SUMS.txt",
    "sha256sums",
    "sha256sums.txt",
    "checksums.txt",
    "checksums.sha256",
];

fn current_date() -> String {
    time::OffsetDateTime::now_local()
        .unwrap_or_else(|_| time::OffsetDateTime::now_utc())
        .date()
        .to_string()
}

pub(crate) type AllowedToolSet = BTreeSet<String>;

#[allow(clippy::too_many_lines)]
pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    match parse_args(&args)? {
        CliAction::DumpManifests => dump_manifests(),
        CliAction::BootstrapPlan => print_bootstrap_plan(),
        CliAction::PrintSystemPrompt { cwd, date } => print_system_prompt(cwd, date),
        CliAction::Model { model } => handle_model_action(model)?,
        CliAction::Provider { provider } => handle_provider_action(provider)?,
        CliAction::Route { route } => handle_route_action(route)?,
        CliAction::Proxy { mode } => handle_proxy_action(mode)?,
        CliAction::Mcp { action } => handle_mcp_action(action)?,
        CliAction::Config {
            section,
            output_format,
        } => run_config_command(section.as_deref(), output_format)?,
        CliAction::Plugins { action, target } => {
            println!(
                "{}",
                handle_plugins_command(action.as_deref(), target.as_deref())?
            );
        }
        CliAction::Branch { action, target } => println!(
            "{}",
            handle_branch_slash_command(
                action.as_deref(),
                target.as_deref(),
                &env::current_dir()?
            )?
        ),
        CliAction::Worktree {
            action,
            path,
            branch,
        } => println!(
            "{}",
            handle_worktree_slash_command(
                action.as_deref(),
                path.as_deref(),
                branch.as_deref(),
                &env::current_dir()?,
            )?
        ),
        CliAction::Agents { args } => println!(
            "{}",
            handle_agents_slash_command(args.as_deref(), &env::current_dir()?)?
        ),
        CliAction::Skills { args } => println!(
            "{}",
            handle_skills_slash_command(args.as_deref(), &env::current_dir()?)?
        ),
        CliAction::Init => run_init()?,
        CliAction::Doctor { command } => run_doctor(command)?,
        CliAction::Ci { command } => run_ci(command)?,
        CliAction::Release { command } => run_release(command)?,
        CliAction::SelfUpdate => run_self_update()?,
        CliAction::Eval {
            suite_path,
            model,
            allowed_tools,
            permission_mode,
            collaboration_mode,
            reasoning_effort,
            fast_mode,
            check_only,
            fail_on_failures,
        } => run_eval_suite(
            &suite_path,
            model,
            allowed_tools.as_ref(),
            permission_mode,
            collaboration_mode,
            reasoning_effort,
            fast_mode,
            check_only,
            fail_on_failures,
        )?,
        CliAction::EvalCompare {
            old_path,
            new_path,
            output_format,
        } => run_eval_compare(&old_path, &new_path, output_format)?,
        CliAction::EvalHistory {
            filter,
            output_format,
        } => run_eval_history(&filter, output_format)?,
        CliAction::EvalCapture { options } => run_eval_capture(&options)?,
        CliAction::EvalReplay {
            options,
            output_format,
        } => run_eval_replay(&options, output_format)?,
        CliAction::Trace {
            trace_path,
            output_format,
        } => run_trace_viewer(&trace_path, output_format)?,
        CliAction::Replay {
            trace_path,
            output_format,
        } => run_trace_replay(&trace_path, output_format)?,
        CliAction::Gc { dry_run } => run_gc(dry_run)?,
        CliAction::ResumeSession {
            session_path,
            commands,
        } => resume_session_cli(session_path.as_deref(), &commands)?,
        CliAction::Prompt {
            prompt,
            model,
            allowed_tools,
            permission_mode,
            collaboration_mode,
            reasoning_effort,
            fast_mode,
            output_format,
        } => LiveCli::new(
            model,
            true,
            allowed_tools,
            permission_mode,
            collaboration_mode,
            reasoning_effort,
            fast_mode,
            matches!(output_format, CliOutputFormat::Text),
        )?
        .run_turn_with_output(&prompt, output_format)?,
        CliAction::Repl {
            model,
            allowed_tools,
            permission_mode,
            collaboration_mode,
            reasoning_effort,
            fast_mode,
        } => run_repl(
            model,
            allowed_tools,
            permission_mode,
            collaboration_mode,
            reasoning_effort,
            fast_mode,
        )?,
        CliAction::Login { service, api_key } => login(service, api_key)?,
        CliAction::Logout { service } => logout(service)?,
        CliAction::Help => print_help(),
        CliAction::Version => print_version(),
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
enum CliAction {
    DumpManifests,
    BootstrapPlan,
    PrintSystemPrompt {
        cwd: PathBuf,
        date: String,
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
    Proxy {
        mode: ProxyCommand,
    },
    Mcp {
        action: McpCommand,
    },
    Config {
        section: Option<String>,
        output_format: CliOutputFormat,
    },
    Plugins {
        action: Option<String>,
        target: Option<String>,
    },
    Branch {
        action: Option<String>,
        target: Option<String>,
    },
    Worktree {
        action: Option<String>,
        path: Option<String>,
        branch: Option<String>,
    },
    Agents {
        args: Option<String>,
    },
    Skills {
        args: Option<String>,
    },
    Init,
    Doctor {
        command: DoctorCommand,
    },
    Ci {
        command: CiCommand,
    },
    Release {
        command: ReleaseCommand,
    },
    SelfUpdate,
    Eval {
        suite_path: PathBuf,
        model: String,
        allowed_tools: Option<AllowedToolSet>,
        permission_mode: PermissionMode,
        collaboration_mode: CollaborationMode,
        reasoning_effort: Option<ReasoningEffort>,
        fast_mode: FastMode,
        check_only: bool,
        fail_on_failures: bool,
    },
    EvalCompare {
        old_path: PathBuf,
        new_path: PathBuf,
        output_format: CliOutputFormat,
    },
    EvalHistory {
        filter: EvalHistoryFilter,
        output_format: CliOutputFormat,
    },
    EvalCapture {
        options: EvalCaptureOptions,
    },
    EvalReplay {
        options: EvalReplayOptions,
        output_format: CliOutputFormat,
    },
    Trace {
        trace_path: PathBuf,
        output_format: CliOutputFormat,
    },
    Replay {
        trace_path: PathBuf,
        output_format: CliOutputFormat,
    },
    Gc {
        dry_run: bool,
    },
    ResumeSession {
        session_path: Option<PathBuf>,
        commands: Vec<String>,
    },
    Prompt {
        prompt: String,
        model: String,
        allowed_tools: Option<AllowedToolSet>,
        permission_mode: PermissionMode,
        collaboration_mode: CollaborationMode,
        reasoning_effort: Option<ReasoningEffort>,
        fast_mode: FastMode,
        output_format: CliOutputFormat,
    },
    Login {
        service: Option<AuthService>,
        api_key: Option<String>,
    },
    Logout {
        service: Option<AuthService>,
    },
    Repl {
        model: String,
        allowed_tools: Option<AllowedToolSet>,
        permission_mode: PermissionMode,
        collaboration_mode: CollaborationMode,
        reasoning_effort: Option<ReasoningEffort>,
        fast_mode: FastMode,
    },
    Help,
    Version,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CliOutputFormat {
    Text,
    Json,
}

impl CliOutputFormat {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "text" => Ok(Self::Text),
            "json" => Ok(Self::Json),
            other => Err(format!(
                "unsupported value for --output-format: {other} (expected text or json)"
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DoctorCommand {
    Check,
    Bundle,
    Providers { json: bool },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CiCommand {
    Check {
        output_format: CliOutputFormat,
        save_report: bool,
    },
    History {
        output_format: CliOutputFormat,
        limit: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReleaseCommand {
    Check {
        output_format: CliOutputFormat,
        save_report: bool,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CollaborationMode {
    Build,
    Plan,
}

impl CollaborationMode {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Build => "build",
            Self::Plan => "plan",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FastMode {
    Off,
    On,
}

impl FastMode {
    pub(crate) const fn enabled(self) -> bool {
        matches!(self, Self::On)
    }

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::On => "on",
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct OpenAiCodexTokenResponse {
    access_token: String,
    refresh_token: String,
    #[serde(default)]
    id_token: Option<String>,
    #[serde(default)]
    expires_in: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OpenAiCodexDeviceCodeResponse {
    device_auth_id: String,
    user_code: String,
    interval: String,
}

#[derive(Debug, Clone, Deserialize)]
struct OpenAiCodexDeviceTokenResponse {
    authorization_code: String,
    code_verifier: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct StartupRuntimeDefaults {
    permission_mode: PermissionMode,
    collaboration_mode: CollaborationMode,
    reasoning_effort: Option<ReasoningEffort>,
    fast_mode: FastMode,
}

fn parse_args(args: &[String]) -> Result<CliAction, String> {
    let startup_defaults = startup_runtime_defaults();
    let mut model = resolve_model_alias(&default_model_or(DEFAULT_MODEL)).to_string();
    let mut model_was_set = false;
    let mut permission_mode = startup_defaults.permission_mode;
    let mut collaboration_mode = startup_defaults.collaboration_mode;
    let mut reasoning_effort = startup_defaults.reasoning_effort;
    let mut fast_mode = startup_defaults.fast_mode;
    let mut output_format = CliOutputFormat::Text;
    let mut allowed_tool_values = Vec::new();
    let mut rest = Vec::new();
    let mut index = 0;

    while index < args.len() {
        match args[index].as_str() {
            "--model" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| "missing value for --model".to_string())?;
                model = resolve_model_alias(value).to_string();
                model_was_set = true;
                index += 2;
            }
            flag if flag.starts_with("--model=") => {
                model = resolve_model_alias(&flag[8..]).to_string();
                model_was_set = true;
                index += 1;
            }
            "--permission-mode" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| "missing value for --permission-mode".to_string())?;
                permission_mode = parse_permission_mode_arg(value)?;
                index += 2;
            }
            flag if flag.starts_with("--permission-mode=") => {
                permission_mode = parse_permission_mode_arg(&flag[18..])?;
                index += 1;
            }
            "--mode" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| "missing value for --mode".to_string())?;
                collaboration_mode = parse_collaboration_mode_arg(value)?;
                index += 2;
            }
            flag if flag.starts_with("--mode=") => {
                collaboration_mode = parse_collaboration_mode_arg(&flag[7..])?;
                index += 1;
            }
            "--reasoning" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| "missing value for --reasoning".to_string())?;
                reasoning_effort = parse_reasoning_effort_arg(value)?;
                index += 2;
            }
            flag if flag.starts_with("--reasoning=") => {
                reasoning_effort = parse_reasoning_effort_arg(&flag[12..])?;
                index += 1;
            }
            "--thinking" => {
                reasoning_effort = Some(ReasoningEffort::Medium);
                index += 1;
            }
            "--fast" => {
                fast_mode = FastMode::On;
                index += 1;
            }
            "--output-format" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| "missing value for --output-format".to_string())?;
                output_format = CliOutputFormat::parse(value)?;
                index += 2;
            }
            flag if flag.starts_with("--output-format=") => {
                output_format = CliOutputFormat::parse(&flag[16..])?;
                index += 1;
            }
            "--json" => {
                output_format = CliOutputFormat::Json;
                index += 1;
            }
            "-p" => {
                let prompt = args[index + 1..].join(" ");
                if prompt.trim().is_empty() {
                    return Err("-p requires a prompt string".to_string());
                }
                return Ok(CliAction::Prompt {
                    prompt,
                    model: resolve_model_alias(&model).to_string(),
                    allowed_tools: normalize_allowed_tools(&allowed_tool_values)?,
                    permission_mode,
                    collaboration_mode,
                    reasoning_effort,
                    fast_mode,
                    output_format,
                });
            }
            "--print" => {
                output_format = CliOutputFormat::Text;
                index += 1;
            }
            "--allowedTools" | "--allowed-tools" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| "missing value for --allowedTools".to_string())?;
                allowed_tool_values.push(value.clone());
                index += 2;
            }
            flag if flag.starts_with("--allowedTools=") => {
                allowed_tool_values.push(flag[15..].to_string());
                index += 1;
            }
            flag if flag.starts_with("--allowed-tools=") => {
                allowed_tool_values.push(flag[16..].to_string());
                index += 1;
            }
            flag @ ("--help" | "-h" | "--version" | "-V" | "--resume") => {
                rest.push(flag.to_string());
                index += 1;
            }
            flag if flag.starts_with('-') && rest.is_empty() => {
                return Err(format!("unknown option: {flag}"));
            }
            other => {
                rest.push(other.to_string());
                index += 1;
            }
        }
    }

    let allowed_tools = normalize_allowed_tools(&allowed_tool_values)?;

    if rest.is_empty() {
        return Ok(CliAction::Repl {
            model,
            allowed_tools,
            permission_mode,
            collaboration_mode,
            reasoning_effort,
            fast_mode,
        });
    }
    if matches!(rest.first().map(String::as_str), Some("--help" | "-h")) {
        return Ok(CliAction::Help);
    }
    if matches!(rest.first().map(String::as_str), Some("--version" | "-V")) {
        return Ok(CliAction::Version);
    }
    if rest.first().map(String::as_str) == Some("--resume") {
        return parse_resume_args(&rest[1..]);
    }

    match rest[0].as_str() {
        "dump-manifests" => Ok(CliAction::DumpManifests),
        "bootstrap-plan" => Ok(CliAction::BootstrapPlan),
        "system-prompt" => parse_system_prompt_args(&rest[1..]),
        "login" | "auth" => parse_login_args(&rest[1..]),
        "logout" => parse_logout_args(&rest[1..]),
        "model" | "models" => parse_model_args(&rest[1..]),
        "provider" | "providers" => parse_provider_args(&rest[1..]),
        "route" | "routing" => parse_route_args(&rest[1..]),
        "proxy" => parse_proxy_args(&rest[1..]),
        "mcp" => parse_mcp_args(&rest[1..]),
        "config" => parse_config_args(&rest[1..], output_format),
        "resume" => parse_resume_args(&rest[1..]),
        "plugins" | "plugin" | "marketplace" => Ok(CliAction::Plugins {
            action: rest.get(1).cloned(),
            target: {
                let remainder = rest.iter().skip(2).cloned().collect::<Vec<_>>().join(" ");
                (!remainder.is_empty()).then_some(remainder)
            },
        }),
        "branch" => Ok(CliAction::Branch {
            action: rest.get(1).cloned(),
            target: rest.get(2).cloned(),
        }),
        "worktree" => Ok(CliAction::Worktree {
            action: rest.get(1).cloned(),
            path: rest.get(2).cloned(),
            branch: rest.get(3).cloned(),
        }),
        "agents" => Ok(CliAction::Agents {
            args: join_optional_args(&rest[1..]),
        }),
        "skills" => Ok(CliAction::Skills {
            args: join_optional_args(&rest[1..]),
        }),
        "init" => Ok(CliAction::Init),
        "doctor" => parse_doctor_args(&rest[1..], output_format),
        "ci" => parse_ci_args(&rest[1..], output_format),
        "release" => parse_release_args(&rest[1..], output_format),
        "self-update" => Ok(CliAction::SelfUpdate),
        "trace" => parse_trace_args(&rest[1..], output_format),
        "replay" => parse_replay_args(&rest[1..], output_format),
        "gc" => parse_gc_args(&rest[1..]),
        "eval" => parse_eval_args(
            &rest[1..],
            model,
            allowed_tools,
            permission_mode,
            collaboration_mode,
            reasoning_effort,
            fast_mode,
            model_was_set,
            output_format,
        ),
        "prompt" => {
            let prompt = rest[1..].join(" ");
            if prompt.trim().is_empty() {
                return Err("prompt subcommand requires a prompt string".to_string());
            }
            Ok(CliAction::Prompt {
                prompt,
                model,
                allowed_tools,
                permission_mode,
                collaboration_mode,
                reasoning_effort,
                fast_mode,
                output_format,
            })
        }
        other if other.starts_with('/') => parse_direct_slash_cli_action(&rest),
        other if !other.starts_with('/') => Ok(CliAction::Prompt {
            prompt: rest.join(" "),
            model,
            allowed_tools,
            permission_mode,
            collaboration_mode,
            reasoning_effort,
            fast_mode,
            output_format,
        }),
        other => Err(format!("unknown subcommand: {other}")),
    }
}

fn parse_gc_args(args: &[String]) -> Result<CliAction, String> {
    let mut dry_run = false;
    for arg in args {
        match arg.as_str() {
            "--dry-run" | "-n" => dry_run = true,
            other => return Err(format!("gc: unsupported argument `{other}`")),
        }
    }
    Ok(CliAction::Gc { dry_run })
}

fn parse_doctor_args(args: &[String], output_format: CliOutputFormat) -> Result<CliAction, String> {
    match args {
        [] => Ok(CliAction::Doctor {
            command: DoctorCommand::Check,
        }),
        [value] if value == "bundle" => Ok(CliAction::Doctor {
            command: DoctorCommand::Bundle,
        }),
        [value] if value == "providers" => Ok(CliAction::Doctor {
            command: DoctorCommand::Providers {
                json: output_format == CliOutputFormat::Json,
            },
        }),
        [value, flag] if value == "providers" && flag == "--json" => Ok(CliAction::Doctor {
            command: DoctorCommand::Providers { json: true },
        }),
        [value] => Err(format!("doctor: unsupported argument `{value}`")),
        _ => Err("usage: pebble doctor [bundle | providers [--json]]".to_string()),
    }
}

fn parse_ci_args(args: &[String], default_format: CliOutputFormat) -> Result<CliAction, String> {
    let mut output_format = default_format;
    let mut save_report = false;
    let mut command = None;
    let mut limit = 10;
    let mut limit_was_set = false;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--json" => {
                output_format = CliOutputFormat::Json;
                index += 1;
            }
            "--output-format" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| "ci --output-format requires a value".to_string())?;
                output_format = CliOutputFormat::parse(value)?;
                index += 2;
            }
            value if value.starts_with("--output-format=") => {
                output_format = CliOutputFormat::parse(&value[16..])?;
                index += 1;
            }
            "--save-report" => {
                save_report = true;
                index += 1;
            }
            "--limit" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| "ci --limit requires a value".to_string())?;
                limit = parse_ci_history_limit(value)?;
                limit_was_set = true;
                index += 2;
            }
            value if value.starts_with("--limit=") => {
                limit = parse_ci_history_limit(&value[8..])?;
                limit_was_set = true;
                index += 1;
            }
            "check" if command.is_none() => {
                command = Some("check");
                index += 1;
            }
            "history" if command.is_none() => {
                command = Some("history");
                index += 1;
            }
            value if value.starts_with('-') => {
                return Err(format!("ci: unsupported argument `{value}`"));
            }
            value => return Err(format!("ci: unsupported argument `{value}`")),
        }
    }
    let command = match command {
        Some("history") => {
            if save_report {
                return Err("ci history does not support --save-report".to_string());
            }
            CiCommand::History {
                output_format,
                limit,
            }
        }
        Some("check") | None => {
            if limit_was_set {
                return Err("ci check does not support --limit".to_string());
            }
            CiCommand::Check {
                output_format,
                save_report,
            }
        }
        Some(other) => return Err(format!("ci: unsupported command `{other}`")),
    };
    Ok(CliAction::Ci { command })
}

fn parse_ci_history_limit(value: &str) -> Result<usize, String> {
    let limit = value
        .parse::<usize>()
        .map_err(|_| format!("ci history --limit must be a positive integer: {value}"))?;
    if limit == 0 {
        return Err("ci history --limit must be greater than 0".to_string());
    }
    Ok(limit)
}

fn parse_release_args(
    args: &[String],
    default_format: CliOutputFormat,
) -> Result<CliAction, String> {
    let mut output_format = default_format;
    let mut save_report = false;
    let mut command = None;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--json" => {
                output_format = CliOutputFormat::Json;
                index += 1;
            }
            "--output-format" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| "release --output-format requires a value".to_string())?;
                output_format = CliOutputFormat::parse(value)?;
                index += 2;
            }
            value if value.starts_with("--output-format=") => {
                output_format = CliOutputFormat::parse(&value[16..])?;
                index += 1;
            }
            "--save-report" => {
                save_report = true;
                index += 1;
            }
            "check" if command.is_none() => {
                command = Some("check");
                index += 1;
            }
            value if value.starts_with('-') => {
                return Err(format!("release: unsupported argument `{value}`"));
            }
            value => return Err(format!("release: unsupported argument `{value}`")),
        }
    }
    match command {
        Some("check") | None => Ok(CliAction::Release {
            command: ReleaseCommand::Check {
                output_format,
                save_report,
            },
        }),
        Some(other) => Err(format!("release: unsupported command `{other}`")),
    }
}

fn parse_config_args(
    args: &[String],
    default_format: CliOutputFormat,
) -> Result<CliAction, String> {
    let mut output_format = default_format;
    let mut section = None;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--json" => {
                output_format = CliOutputFormat::Json;
                index += 1;
            }
            "--output-format" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| "config --output-format requires a value".to_string())?;
                output_format = CliOutputFormat::parse(value)?;
                index += 2;
            }
            value if value.starts_with("--output-format=") => {
                output_format = CliOutputFormat::parse(&value[16..])?;
                index += 1;
            }
            value if value.starts_with('-') => {
                return Err(format!("config: unsupported argument `{value}`"));
            }
            value if section.is_none() => {
                section = Some(value.to_string());
                index += 1;
            }
            _ => return Err("config accepts at most one section or `check`".to_string()),
        }
    }
    Ok(CliAction::Config {
        section,
        output_format,
    })
}

fn parse_trace_args(args: &[String], default_format: CliOutputFormat) -> Result<CliAction, String> {
    let (trace_path, output_format) =
        parse_single_path_debug_args(args, default_format, "trace", "trace JSON path")?;
    Ok(CliAction::Trace {
        trace_path,
        output_format,
    })
}

fn parse_replay_args(
    args: &[String],
    default_format: CliOutputFormat,
) -> Result<CliAction, String> {
    let (trace_path, output_format) =
        parse_single_path_debug_args(args, default_format, "replay", "trace JSON path")?;
    Ok(CliAction::Replay {
        trace_path,
        output_format,
    })
}

fn parse_single_path_debug_args(
    args: &[String],
    default_format: CliOutputFormat,
    command: &str,
    path_label: &str,
) -> Result<(PathBuf, CliOutputFormat), String> {
    let mut path = None;
    let mut output_format = default_format;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--json" => {
                output_format = CliOutputFormat::Json;
                index += 1;
            }
            "--output-format" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| format!("{command} --output-format requires a value"))?;
                output_format = CliOutputFormat::parse(value)?;
                index += 2;
            }
            value if value.starts_with("--output-format=") => {
                output_format = CliOutputFormat::parse(&value[16..])?;
                index += 1;
            }
            value if value.starts_with('-') => {
                return Err(format!("{command}: unsupported argument `{value}`"));
            }
            value if path.is_none() => {
                path = Some(PathBuf::from(value));
                index += 1;
            }
            _ => return Err(format!("{command} accepts exactly one {path_label}")),
        }
    }
    let path = path.ok_or_else(|| format!("{command} subcommand requires a {path_label}"))?;
    Ok((path, output_format))
}

fn resolve_model_alias(model: &str) -> &str {
    match model.trim().to_ascii_lowercase().as_str() {
        "default" | "glm" | "glm5.1" | "glm-5.1" | "glm_5_1" | "zai-org/glm-5.1" => {
            "zai-org/glm-5.1"
        }
        "glm5" | "glm-5" | "glm_5" | "zai-org/glm-5" => "zai-org/glm-5",
        _ => model,
    }
}

fn parse_direct_slash_cli_action(rest: &[String]) -> Result<CliAction, String> {
    let raw = rest.join(" ");
    match SlashCommand::parse(&raw) {
        Some(SlashCommand::Help { .. }) => Ok(CliAction::Help),
        Some(SlashCommand::Agents { args }) => Ok(CliAction::Agents { args }),
        Some(SlashCommand::Skills { args }) => Ok(CliAction::Skills { args }),
        Some(command) => Err(format!(
            "unsupported direct slash command outside the REPL: {command_name}",
            command_name = match command {
                SlashCommand::Unknown(name) => format!("/{name}"),
                _ => rest[0].clone(),
            }
        )),
        None => Err(format!("unknown subcommand: {}", rest[0])),
    }
}

fn join_optional_args(args: &[String]) -> Option<String> {
    let joined = args.join(" ");
    let trimmed = joined.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

fn normalize_allowed_tools(values: &[String]) -> Result<Option<AllowedToolSet>, String> {
    current_tool_registry()?.normalize_allowed_tools(values)
}

fn parse_system_prompt_args(args: &[String]) -> Result<CliAction, String> {
    let mut cwd = env::current_dir().map_err(|error| error.to_string())?;
    let mut date = current_date();
    let mut index = 0;

    while index < args.len() {
        match args[index].as_str() {
            "--cwd" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| "missing value for --cwd".to_string())?;
                cwd = PathBuf::from(value);
                index += 2;
            }
            "--date" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| "missing value for --date".to_string())?;
                date.clone_from(value);
                index += 2;
            }
            other => return Err(format!("unknown system-prompt option: {other}")),
        }
    }

    Ok(CliAction::PrintSystemPrompt { cwd, date })
}

fn parse_model_args(args: &[String]) -> Result<CliAction, String> {
    if args.len() > 1 {
        return Err("model accepts at most one optional model id".to_string());
    }
    Ok(CliAction::Model {
        model: args.first().cloned(),
    })
}

fn parse_provider_args(args: &[String]) -> Result<CliAction, String> {
    if args.len() > 1 {
        return Err("provider accepts at most one optional provider name".to_string());
    }
    Ok(CliAction::Provider {
        provider: args.first().cloned(),
    })
}

fn parse_route_args(args: &[String]) -> Result<CliAction, String> {
    if args.len() > 1 {
        return Err("route accepts at most one optional NanoGPT route id".to_string());
    }
    Ok(CliAction::Route {
        route: args.first().cloned(),
    })
}

fn parse_login_args(args: &[String]) -> Result<CliAction, String> {
    let tokens = args.iter().map(String::as_str).collect::<Vec<_>>();
    let parsed = parse_login_tokens(&tokens)?;
    Ok(CliAction::Login {
        service: parsed.service,
        api_key: parsed.api_key,
    })
}

fn parse_logout_args(args: &[String]) -> Result<CliAction, String> {
    let tokens = args.iter().map(String::as_str).collect::<Vec<_>>();
    let parsed = parse_logout_tokens(&tokens)?;
    Ok(CliAction::Logout {
        service: parsed.service,
    })
}

fn parse_proxy_args(args: &[String]) -> Result<CliAction, String> {
    if args.len() > 1 {
        return Err("proxy accepts at most one optional argument".to_string());
    }
    Ok(CliAction::Proxy {
        mode: parse_proxy_value(args.first().map(String::as_str))?,
    })
}

#[allow(clippy::too_many_arguments)]
fn parse_eval_args(
    args: &[String],
    model: String,
    allowed_tools: Option<AllowedToolSet>,
    permission_mode: PermissionMode,
    collaboration_mode: CollaborationMode,
    reasoning_effort: Option<ReasoningEffort>,
    fast_mode: FastMode,
    model_was_set: bool,
    default_output_format: CliOutputFormat,
) -> Result<CliAction, String> {
    if args.first().map(String::as_str) == Some("history") {
        let (mut filter, output_format) =
            parse_eval_history_args(&args[1..], default_output_format)?;
        if filter.model.is_none() && model_was_set {
            filter.model = Some(model);
        }
        return Ok(CliAction::EvalHistory {
            filter,
            output_format,
        });
    }
    if args.first().map(String::as_str) == Some("capture") {
        return parse_eval_capture_args(&args[1..]);
    }
    if args.first().map(String::as_str) == Some("replay") {
        return parse_eval_replay_args(&args[1..], default_output_format);
    }
    if args.first().map(String::as_str) == Some("compare") {
        return parse_eval_compare_args(&args[1..], default_output_format);
    }

    let mut check_only = false;
    let mut fail_on_failures = false;
    let mut suite_path = None;

    for arg in args {
        match arg.as_str() {
            "--check" | "--dry-run" => check_only = true,
            "--fail-on-failures" => fail_on_failures = true,
            other if other.starts_with('-') => {
                return Err(format!("unknown eval option: {other}"));
            }
            path if suite_path.is_none() => suite_path = Some(PathBuf::from(path)),
            _ => return Err("eval accepts exactly one suite path".to_string()),
        }
    }

    Ok(CliAction::Eval {
        suite_path: suite_path
            .ok_or_else(|| "eval subcommand requires a suite path".to_string())?,
        model,
        allowed_tools,
        permission_mode,
        collaboration_mode,
        reasoning_effort,
        fast_mode,
        check_only,
        fail_on_failures,
    })
}

fn parse_eval_history_args(
    args: &[String],
    default_format: CliOutputFormat,
) -> Result<(EvalHistoryFilter, CliOutputFormat), String> {
    let mut filter = EvalHistoryFilter::default();
    let mut output_format = default_format;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--json" => {
                output_format = CliOutputFormat::Json;
                index += 1;
            }
            "--output-format" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| "eval history --output-format requires a value".to_string())?;
                output_format = CliOutputFormat::parse(value)?;
                index += 2;
            }
            value if value.starts_with("--output-format=") => {
                output_format = CliOutputFormat::parse(&value[16..])?;
                index += 1;
            }
            "--suite" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| "eval history --suite requires a value".to_string())?;
                filter.suite = Some(value.clone());
                index += 2;
            }
            value if value.starts_with("--suite=") => {
                filter.suite = Some(value[8..].to_string());
                index += 1;
            }
            "--model" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| "eval history --model requires a value".to_string())?;
                filter.model = Some(resolve_model_alias(value).to_string());
                index += 2;
            }
            value if value.starts_with("--model=") => {
                filter.model = Some(resolve_model_alias(&value[8..]).to_string());
                index += 1;
            }
            "--limit" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| "eval history --limit requires a value".to_string())?;
                filter.limit = parse_history_limit(value)?;
                index += 2;
            }
            value if value.starts_with("--limit=") => {
                filter.limit = parse_history_limit(&value[8..])?;
                index += 1;
            }
            other => return Err(format!("eval history: unsupported argument `{other}`")),
        }
    }
    Ok((filter, output_format))
}

fn parse_history_limit(value: &str) -> Result<usize, String> {
    let limit = value
        .parse::<usize>()
        .map_err(|_| format!("eval history --limit must be a positive integer: {value}"))?;
    if limit == 0 {
        return Err("eval history --limit must be greater than 0".to_string());
    }
    Ok(limit)
}

fn parse_eval_compare_args(
    args: &[String],
    default_format: CliOutputFormat,
) -> Result<CliAction, String> {
    let mut old_path = None;
    let mut new_path = None;
    let mut output_format = default_format;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--json" => {
                output_format = CliOutputFormat::Json;
                index += 1;
            }
            "--output-format" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| "eval compare --output-format requires a value".to_string())?;
                output_format = CliOutputFormat::parse(value)?;
                index += 2;
            }
            value if value.starts_with("--output-format=") => {
                output_format = CliOutputFormat::parse(&value[16..])?;
                index += 1;
            }
            value if value.starts_with('-') => {
                return Err(format!("eval compare: unsupported argument `{value}`"));
            }
            value if old_path.is_none() => {
                old_path = Some(PathBuf::from(value));
                index += 1;
            }
            value if new_path.is_none() => {
                new_path = Some(PathBuf::from(value));
                index += 1;
            }
            _ => return Err("eval compare requires old and new report paths".to_string()),
        }
    }
    Ok(CliAction::EvalCompare {
        old_path: old_path.ok_or_else(|| "eval compare requires old report path".to_string())?,
        new_path: new_path.ok_or_else(|| "eval compare requires new report path".to_string())?,
        output_format,
    })
}

fn parse_eval_replay_args(
    args: &[String],
    default_format: CliOutputFormat,
) -> Result<CliAction, String> {
    let mut report_path = None;
    let mut case_id = None;
    let mut output_format = default_format;
    let mut index = 0;

    while index < args.len() {
        match args[index].as_str() {
            "--json" => {
                output_format = CliOutputFormat::Json;
                index += 1;
            }
            "--output-format" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| "eval replay --output-format requires a value".to_string())?;
                output_format = CliOutputFormat::parse(value)?;
                index += 2;
            }
            value if value.starts_with("--output-format=") => {
                output_format = CliOutputFormat::parse(&value[16..])?;
                index += 1;
            }
            "--case" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| "eval replay --case requires a value".to_string())?;
                case_id = Some(value.clone());
                index += 2;
            }
            value if value.starts_with("--case=") => {
                case_id = Some(value[7..].to_string());
                index += 1;
            }
            value if value.starts_with('-') => {
                return Err(format!("eval replay: unsupported argument `{value}`"));
            }
            value if report_path.is_none() => {
                report_path = Some(PathBuf::from(value));
                index += 1;
            }
            _ => return Err("eval replay accepts exactly one report path".to_string()),
        }
    }

    Ok(CliAction::EvalReplay {
        options: EvalReplayOptions {
            report_path: report_path
                .ok_or_else(|| "eval replay requires an eval report path".to_string())?,
            case_id,
        },
        output_format,
    })
}

fn parse_eval_capture_args(args: &[String]) -> Result<CliAction, String> {
    let mut trace_path = None;
    let mut suite_path = None;
    let mut name = None;
    let mut force = false;
    let mut index = 0;

    while index < args.len() {
        match args[index].as_str() {
            "--suite" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| "eval capture --suite requires a value".to_string())?;
                suite_path = Some(PathBuf::from(value));
                index += 2;
            }
            value if value.starts_with("--suite=") => {
                suite_path = Some(PathBuf::from(&value[8..]));
                index += 1;
            }
            "--name" | "--id" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| "eval capture --name requires a value".to_string())?;
                name = Some(value.clone());
                index += 2;
            }
            value if value.starts_with("--name=") => {
                name = Some(value[7..].to_string());
                index += 1;
            }
            value if value.starts_with("--id=") => {
                name = Some(value[5..].to_string());
                index += 1;
            }
            "--force" => {
                force = true;
                index += 1;
            }
            value if value.starts_with('-') => {
                return Err(format!("eval capture: unsupported argument `{value}`"));
            }
            value if trace_path.is_none() => {
                trace_path = Some(PathBuf::from(value));
                index += 1;
            }
            _ => return Err("eval capture accepts exactly one trace path".to_string()),
        }
    }

    Ok(CliAction::EvalCapture {
        options: EvalCaptureOptions {
            trace_path: trace_path
                .ok_or_else(|| "eval capture requires a trace JSON path".to_string())?,
            suite_path: suite_path
                .ok_or_else(|| "eval capture requires --suite <path>".to_string())?,
            name,
            force,
        },
    })
}

fn parse_mcp_args(args: &[String]) -> Result<CliAction, String> {
    let action = match args.first().map(String::as_str) {
        None | Some("status") => McpCommand::Status,
        Some("tools") => McpCommand::Tools,
        Some("reload") => McpCommand::Reload,
        Some("add") => {
            let name = args
                .get(1)
                .ok_or_else(|| "mcp add requires a server name".to_string())?;
            if args.len() > 2 {
                return Err("mcp add accepts exactly one server name".to_string());
            }
            McpCommand::Add { name: name.clone() }
        }
        Some("enable") => {
            let name = args
                .get(1)
                .ok_or_else(|| "mcp enable requires a server name".to_string())?;
            if args.len() > 2 {
                return Err("mcp enable accepts exactly one server name".to_string());
            }
            McpCommand::Enable { name: name.clone() }
        }
        Some("disable") => {
            let name = args
                .get(1)
                .ok_or_else(|| "mcp disable requires a server name".to_string())?;
            if args.len() > 2 {
                return Err("mcp disable accepts exactly one server name".to_string());
            }
            McpCommand::Disable { name: name.clone() }
        }
        Some(other) => {
            return Err(format!(
                "mcp accepts status, tools, reload, add <name>, enable <name>, or disable <name> (got {other})"
            ));
        }
    };
    if !matches!(
        action,
        McpCommand::Add { .. } | McpCommand::Enable { .. } | McpCommand::Disable { .. }
    ) && args.len() > 1
    {
        return Err(
            "mcp accepts at most one optional argument unless using add <name>, enable <name>, or disable <name>".to_string(),
        );
    }
    Ok(CliAction::Mcp { action })
}

fn parse_resume_args(args: &[String]) -> Result<CliAction, String> {
    let (session_path, commands) = match args.first() {
        None => (None, Vec::new()),
        Some(first) if first.trim_start().starts_with('/') => {
            return Err("resume without a session id/path opens the session picker and does not accept trailing commands".to_string());
        }
        Some(first) => (Some(PathBuf::from(first)), args[1..].to_vec()),
    };
    if commands
        .iter()
        .any(|command| !command.trim_start().starts_with('/'))
    {
        return Err("--resume trailing arguments must be slash commands".to_string());
    }
    Ok(CliAction::ResumeSession {
        session_path,
        commands,
    })
}

fn dump_manifests() {
    let workspace_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let paths = UpstreamPaths::from_workspace_dir(&workspace_dir);
    match extract_manifest(&paths) {
        Ok(manifest) => {
            println!("commands: {}", manifest.commands.entries().len());
            println!("tools: {}", manifest.tools.entries().len());
            println!("bootstrap phases: {}", manifest.bootstrap.phases().len());
        }
        Err(error) => {
            eprintln!("failed to extract manifests: {error}");
            std::process::exit(1);
        }
    }
}

fn print_bootstrap_plan() {
    for phase in runtime::BootstrapPlan::pebble_default().phases() {
        println!("- {phase:?}");
    }
}

fn print_system_prompt(cwd: PathBuf, date: String) {
    let model = default_model_or(DEFAULT_MODEL);
    let service = infer_service_for_model(&model);
    match load_system_prompt_with_model_family(
        cwd,
        date,
        env::consts::OS,
        "unknown",
        prompt_model_family(service, &model),
    ) {
        Ok(sections) => println!("{}", sections.join("\n\n")),
        Err(error) => {
            eprintln!("failed to build system prompt: {error}");
            std::process::exit(1);
        }
    }
}

fn resume_session_cli(
    session_path: Option<&Path>,
    commands: &[String],
) -> Result<(), Box<dyn std::error::Error>> {
    let handle = match session_path {
        Some(session_path) => resolve_session_reference(&session_path.display().to_string())?,
        None => match prompt_for_session_selection(None)? {
            Some(handle) => handle,
            None => return Ok(()),
        },
    };
    let session = match Session::load_from_path(&handle.path) {
        Ok(session) => session,
        Err(error) => {
            eprintln!("failed to restore session: {error}");
            std::process::exit(1);
        }
    };

    if commands.is_empty() {
        return run_repl_from_session(handle, session);
    }

    let mut session = session;
    for raw_command in commands {
        let Some(command) = SlashCommand::parse(raw_command) else {
            eprintln!("unsupported resumed command: {raw_command}");
            std::process::exit(2);
        };
        match run_resume_command(&handle.path, &session, &command) {
            Ok(ResumeCommandOutcome {
                session: next_session,
                message,
            }) => {
                session = next_session.clone();
                if let Err(error) = next_session.save_to_path(&handle.path) {
                    eprintln!("failed to persist resumed session: {error}");
                    std::process::exit(1);
                }
                if let Some(message) = message {
                    println!("{message}");
                }
            }
            Err(error) => {
                eprintln!("{error}");
                std::process::exit(2);
            }
        }
    }
    Ok(())
}

fn prompt_for_session_selection(
    active_session_id: Option<&str>,
) -> Result<Option<SessionHandle>, Box<dyn std::error::Error>> {
    let sessions = list_managed_sessions()?;
    if sessions.is_empty() {
        println!("No managed sessions saved yet.");
        return Ok(None);
    }

    print!("Filter sessions (optional, press Enter for all): ");
    io::stdout().flush()?;
    let mut filter = String::new();
    io::stdin().read_line(&mut filter)?;
    let filter = filter.trim().to_ascii_lowercase();
    let sessions = if filter.is_empty() {
        sessions
    } else {
        sessions
            .into_iter()
            .filter(|session| {
                fuzzy_session_match(&session.id, &filter)
                    || session
                        .model
                        .as_deref()
                        .is_some_and(|model| fuzzy_session_match(model, &filter))
                    || session
                        .title
                        .as_deref()
                        .is_some_and(|title| fuzzy_session_match(title, &filter))
                    || session
                        .last_prompt
                        .as_deref()
                        .is_some_and(|prompt| fuzzy_session_match(prompt, &filter))
            })
            .collect::<Vec<_>>()
    };
    if sessions.is_empty() {
        println!("No sessions matched that filter.");
        return Ok(None);
    }

    println!("Recent sessions");
    for (index, session) in sessions.iter().enumerate() {
        let marker = if active_session_id == Some(session.id.as_str()) {
            "current"
        } else if index == 0 {
            "last"
        } else {
            "saved"
        };
        let model = session.model.as_deref().unwrap_or("unknown");
        let last_prompt = session.last_prompt.as_deref().map_or_else(
            || "-".to_string(),
            |prompt| truncate_for_summary(prompt, 48),
        );
        let title = session
            .title
            .as_deref()
            .map_or_else(|| "-".to_string(), |title| truncate_for_summary(title, 28));
        println!(
            "  {idx:>2}. {id:<22} {marker:<7} model={model:<24} msgs={msgs:<4} title={title:<28} last={last}",
            idx = index + 1,
            id = session.id,
            marker = marker,
            model = model,
            msgs = session.message_count,
            title = title,
            last = last_prompt,
        );
    }
    println!();
    print!(
        "Select a session to resume [1-{}] or press Enter to cancel: ",
        sessions.len()
    );
    io::stdout().flush()?;
    let mut buffer = String::new();
    io::stdin().read_line(&mut buffer)?;
    let selection = buffer.trim();
    if selection.is_empty() {
        return Ok(None);
    }
    let index = selection
        .parse::<usize>()
        .map_err(|_| format!("invalid selection: {selection}"))?;
    let Some(session) = sessions.get(index.saturating_sub(1)) else {
        return Err(format!("selection out of range: {selection}").into());
    };
    Ok(Some(SessionHandle {
        id: session.id.clone(),
        path: session.path.clone(),
    }))
}

fn fuzzy_session_match(haystack: &str, query: &str) -> bool {
    let haystack = haystack.to_ascii_lowercase();
    if haystack.contains(query) {
        return true;
    }

    let mut query_chars = query.chars();
    let mut current = match query_chars.next() {
        Some(ch) => ch,
        None => return true,
    };

    for hay in haystack.chars() {
        if hay == current {
            match query_chars.next() {
                Some(next) => current = next,
                None => return true,
            }
        }
    }
    false
}

fn login(
    service: Option<AuthService>,
    api_key: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let service = match service {
        Some(service) => service,
        None => match prompt_for_auth_service_selection()? {
            Some(service) => service,
            None => return Ok(()),
        },
    };
    if service == AuthService::Grok {
        if api_key.is_some() {
            return Err(
                "Grok login uses the official CLI OAuth flow and does not accept API keys".into(),
            );
        }
        run_grok_auth_command("login")?;
        println!("Grok OAuth login complete. Subscription access remains managed by the official Grok CLI.");
        return Ok(());
    }
    if service == AuthService::OpenAiCodex {
        if api_key.is_some() {
            return Err(
                "OpenAI Codex login uses device-code authentication and does not accept API keys"
                    .into(),
            );
        }
        let credentials_path = login_openai_codex()?;
        println!(
            "Saved {} credentials to {}",
            service.display_name(),
            credentials_path.display()
        );
        if let Some(note) = login_model_guidance(service) {
            println!("{note}");
        }
        return Ok(());
    }
    let api_key = resolve_auth_api_key(service, api_key)?;
    let verification = service.runtime_service().and_then(|runtime_service| {
        matches!(
            runtime_service,
            ApiService::NanoGpt | ApiService::Neuralwatt | ApiService::Lilac
        )
        .then(|| verify_model_service_credentials(runtime_service, &api_key))
    });
    if let Some(Err(ApiError::Api {
        status, message, ..
    })) = &verification
    {
        if matches!(
            *status,
            reqwest::StatusCode::UNAUTHORIZED | reqwest::StatusCode::FORBIDDEN
        ) {
            return Err(format!(
                "{} rejected this API key{}; credentials were not saved",
                service.display_name(),
                message
                    .as_deref()
                    .map(|message| format!(": {message}"))
                    .unwrap_or_default()
            )
            .into());
        }
    }
    let credentials_path = save_credentials(service, &api_key)?;
    println!(
        "Saved {} credentials to {}",
        service.display_name(),
        credentials_path.display()
    );
    match verification {
        Some(Ok(model_count)) => {
            println!(
                "Verified {} with {} available models.",
                service.display_name(),
                model_count
            );
            if let Some(runtime_service) = service.runtime_service() {
                let _ = refresh_model_catalog(runtime_service);
            }
        }
        Some(Err(error)) => println!(
            "Saved, but {} could not be verified: {error}",
            service.display_name()
        ),
        None => {}
    }
    if let Some(note) = login_model_guidance(service) {
        println!("{note}");
    }
    Ok(())
}

fn login_model_guidance(service: AuthService) -> Option<String> {
    let target_service = service.runtime_service()?;
    let active_model = default_model_or(DEFAULT_MODEL);
    let active_service = infer_service_for_model(&active_model);
    if active_service == target_service {
        return None;
    }

    let switch_hint = match service {
        AuthService::Synthetic => {
            "Run `/model` and choose a Synthetic model id (usually prefixed with `hf:`)."
        }
        AuthService::OpenAiCodex => {
            "Run `/model` and choose an OpenAI Codex model id prefixed with `openai-codex/`."
        }
        AuthService::OpencodeGo => {
            "Run `/model` and choose an OpenCode Go model id prefixed with `opencode-go/`."
        }
        AuthService::NanoGpt => {
            "Run `/model zai-org/glm-5.1` or another NanoGPT-backed model if you want to use this key immediately."
        }
        AuthService::Neuralwatt => {
            "Run `/model` and choose a model under the Neuralwatt provider."
        }
        AuthService::Lilac => "Run `/model` and choose a model under the Lilac provider.",
        AuthService::Grok => "Run `/model` and choose Grok 4.5 under the Grok provider.",
        AuthService::Exa => return None,
    };

    Some(format!(
        "Note: your current model is `{active_model}` on {}. Logging into {} saves credentials but does not switch the active model. {switch_hint}",
        active_service.display_name(),
        service.display_name(),
    ))
}

fn logout(service: Option<AuthService>) -> Result<(), Box<dyn std::error::Error>> {
    let service = match service {
        Some(service) => service,
        None => match prompt_for_auth_service_selection()? {
            Some(service) => service,
            None => return Ok(()),
        },
    };

    if service == AuthService::Grok {
        run_grok_auth_command("logout")?;
        println!("Grok OAuth session removed by the official Grok CLI.");
        return Ok(());
    }

    let outcome = remove_saved_credentials(service)?;
    match outcome {
        CredentialRemovalOutcome::Removed { path } => {
            println!(
                "Removed saved {} credentials from {}",
                service.display_name(),
                path.display()
            );
        }
        CredentialRemovalOutcome::Missing { path } => {
            println!(
                "No saved {} credentials found in {}",
                service.display_name(),
                path.display()
            );
        }
    }
    Ok(())
}

fn login_openai_codex() -> Result<PathBuf, Box<dyn std::error::Error>> {
    login_openai_codex_device_code()
}

fn login_openai_codex_device_code() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let client = BlockingClient::builder()
        .timeout(Duration::from_secs(20))
        .build()?;
    let response = client
        .post(format!(
            "{OPENAI_CODEX_ISSUER}/api/accounts/deviceauth/usercode"
        ))
        .header("content-type", "application/json")
        .json(&serde_json::json!({ "client_id": OPENAI_CODEX_CLIENT_ID }))
        .send()?;
    let response = response.error_for_status()?;
    let payload: OpenAiCodexDeviceCodeResponse = response.json()?;
    let interval_secs = payload
        .interval
        .trim()
        .parse::<u64>()
        .ok()
        .filter(|value| *value > 0)
        .unwrap_or(5);

    println!(
        "Open this URL in your browser and sign in with ChatGPT:\n\n{issuer}/codex/device\n\nEnter this one-time code:\n\n{code}\n",
        issuer = OPENAI_CODEX_ISSUER,
        code = payload.user_code
    );

    let deadline = std::time::Instant::now() + OPENAI_CODEX_DEVICE_AUTH_TIMEOUT;
    loop {
        let response = client
            .post(format!(
                "{OPENAI_CODEX_ISSUER}/api/accounts/deviceauth/token"
            ))
            .header("content-type", "application/json")
            .json(&serde_json::json!({
                "device_auth_id": payload.device_auth_id,
                "user_code": payload.user_code,
            }))
            .send()?;

        if response.status().is_success() {
            let payload: OpenAiCodexDeviceTokenResponse = response.json()?;
            let tokens = exchange_openai_codex_authorization_code(
                &payload.authorization_code,
                &format!("{OPENAI_CODEX_ISSUER}/deviceauth/callback"),
                &payload.code_verifier,
            )?;
            return persist_openai_codex_tokens(tokens);
        }

        let status = response.status();
        if !matches!(status.as_u16(), 403 | 404) {
            let body = response.text().unwrap_or_default();
            return Err(
                format!("device code authorization failed with status {status}: {body}").into(),
            );
        }

        if std::time::Instant::now() >= deadline {
            return Err("device code authorization timed out after 15 minutes".into());
        }

        std::thread::sleep(
            Duration::from_secs(interval_secs)
                .saturating_add(OPENAI_CODEX_DEVICE_POLL_SAFETY_MARGIN),
        );
    }
}

fn exchange_openai_codex_authorization_code(
    code: &str,
    redirect_uri: &str,
    verifier: &str,
) -> Result<OpenAiCodexTokenResponse, Box<dyn std::error::Error>> {
    let client = BlockingClient::builder()
        .timeout(Duration::from_secs(20))
        .build()?;
    let response = client
        .post(format!("{OPENAI_CODEX_ISSUER}/oauth/token"))
        .header("content-type", "application/x-www-form-urlencoded")
        .form(&[
            ("grant_type", "authorization_code"),
            ("code", code),
            ("redirect_uri", redirect_uri),
            ("client_id", OPENAI_CODEX_CLIENT_ID),
            ("code_verifier", verifier),
        ])
        .send()?;
    let response = response.error_for_status()?;
    Ok(response.json()?)
}

fn persist_openai_codex_tokens(
    tokens: OpenAiCodexTokenResponse,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let expires_at = tokens
        .expires_in
        .map(|seconds| current_epoch_millis().saturating_add(seconds.saturating_mul(1_000)));
    let account_id = tokens
        .id_token
        .as_deref()
        .and_then(extract_openai_codex_account_id)
        .or_else(|| extract_openai_codex_account_id(&tokens.access_token));
    let credentials = OpenAiCodexCredentials {
        access_token: tokens.access_token,
        refresh_token: tokens.refresh_token,
        expires_at,
        account_id,
    };
    Ok(save_openai_codex_credentials(&credentials)?)
}

fn extract_openai_codex_account_id(token: &str) -> Option<String> {
    let claims = decode_jwt_claims(token)?;
    claims
        .get("chatgpt_account_id")
        .and_then(JsonValue::as_str)
        .map(ToOwned::to_owned)
        .or_else(|| {
            claims
                .get("https://api.openai.com/auth")
                .and_then(JsonValue::as_object)
                .and_then(|value| value.get("chatgpt_account_id"))
                .and_then(JsonValue::as_str)
                .map(ToOwned::to_owned)
        })
        .or_else(|| {
            claims
                .get("organizations")
                .and_then(JsonValue::as_array)
                .and_then(|values| values.first())
                .and_then(JsonValue::as_object)
                .and_then(|value| value.get("id"))
                .and_then(JsonValue::as_str)
                .map(ToOwned::to_owned)
        })
}

fn decode_jwt_claims(token: &str) -> Option<JsonValue> {
    let payload = token.split('.').nth(1)?;
    let decoded = decode_base64_url(payload)?;
    serde_json::from_slice(&decoded).ok()
}

fn decode_base64_url(value: &str) -> Option<Vec<u8>> {
    let mut output = Vec::new();
    let mut buffer = 0_u32;
    let mut bits = 0_u8;

    for byte in value.bytes() {
        let sextet = match byte {
            b'A'..=b'Z' => byte - b'A',
            b'a'..=b'z' => byte - b'a' + 26,
            b'0'..=b'9' => byte - b'0' + 52,
            b'-' => 62,
            b'_' => 63,
            b'=' => break,
            _ => return None,
        };
        buffer = (buffer << 6) | u32::from(sextet);
        bits += 6;
        while bits >= 8 {
            bits -= 8;
            output.push(((buffer >> bits) & 0xFF) as u8);
        }
    }

    Some(output)
}

fn handle_model_action(model: Option<String>) -> Result<(), Box<dyn std::error::Error>> {
    match model {
        Some(model) => {
            let model = resolve_model_alias(&model).to_string();
            persist_current_model(model.clone())?;
            println!("{}", ui::setting_changed("model", &[("selected", &model)]));
        }
        None => match open_model_picker()?.selected_model {
            Some(model) => println!("{}", ui::setting_changed("model", &[("selected", &model)])),
            None => println!(
                "{}",
                ui::setting_changed("model", &[("result", "selection cancelled")])
            ),
        },
    }
    Ok(())
}

fn handle_provider_action(provider: Option<String>) -> Result<(), Box<dyn std::error::Error>> {
    let service = provider
        .as_deref()
        .map(|provider| {
            service_from_selector(provider).ok_or_else(|| unknown_provider_message(provider))
        })
        .transpose()?;
    match open_model_picker_for_service(service)?.selected_model {
        Some(model) => println!(
            "{}",
            ui::setting_changed(
                "model provider",
                &[
                    ("provider", infer_service_for_model(&model).display_name()),
                    ("model", &model),
                ],
            )
        ),
        None => println!(
            "{}",
            ui::setting_changed("model provider", &[("result", "selection cancelled")])
        ),
    }
    Ok(())
}

fn unknown_provider_message(provider: &str) -> String {
    format!(
        "unknown model provider `{provider}`; choose nanogpt, neuralwatt, lilac, grok, synthetic, openai-codex, or opencode-go"
    )
}

fn handle_route_action(provider: Option<String>) -> Result<(), Box<dyn std::error::Error>> {
    let model = default_model_or(DEFAULT_MODEL);
    if infer_service_for_model(&model) != ApiService::NanoGpt {
        return Err(format!(
            "routing overrides are only available for NanoGPT models; current model {} is on {}. Use `pebble provider` to switch model providers",
            model,
            infer_service_for_model(&model).display_name()
        )
        .into());
    }
    match provider {
        Some(provider) if is_clear_provider_arg(&provider) => {
            persist_provider_for_model(&model, None)?;
            println!(
                "{}",
                ui::setting_changed(
                    "NanoGPT route",
                    &[("model", &model), ("route", "platform default")],
                )
            );
        }
        Some(provider) => {
            validate_provider_for_model(&model, &provider)?;
            persist_provider_for_model(&model, Some(provider.clone()))?;
            println!(
                "{}",
                ui::setting_changed(
                    "NanoGPT route",
                    &[
                        ("model", &model),
                        ("route", &provider),
                        ("routing", "paygo routing enabled"),
                    ],
                )
            );
        }
        None => match open_provider_picker(&model)?.selected_provider {
            Some(provider) => {
                println!(
                    "{}",
                    ui::setting_changed(
                        "NanoGPT route",
                        &[
                            ("model", &model),
                            ("route", &provider),
                            ("routing", "paygo routing enabled"),
                        ],
                    )
                );
            }
            None => println!(
                "{}",
                ui::setting_changed(
                    "NanoGPT route",
                    &[("model", &model), ("route", "platform default")],
                )
            ),
        },
    }
    Ok(())
}

fn provider_label_for_service_model(service: ApiService, model: &str) -> Option<String> {
    (service == ApiService::NanoGpt)
        .then(|| provider_for_model(model))
        .flatten()
}

fn startup_auth_hint(service: ApiService) -> Option<String> {
    model_auth_status(service)
        .starts_with("missing")
        .then(|| format!("/login {}", service.as_str().replace('_', "-")))
}

fn handle_proxy_action(mode: ProxyCommand) -> Result<(), Box<dyn std::error::Error>> {
    let current = proxy_tool_calls_enabled();
    let next = match mode {
        ProxyCommand::Toggle => !current,
        ProxyCommand::Enable => true,
        ProxyCommand::Disable => false,
        ProxyCommand::Status => current,
    };
    if !matches!(mode, ProxyCommand::Status) {
        persist_proxy_tool_calls(next)?;
    }
    println!(
        "{}",
        ui::setting_changed(
            "proxy tool calls",
            &[("state", if next { "enabled" } else { "disabled" })],
        )
    );
    if next {
        println!(
            "{}",
            ui::dim_note("native tool schemas disabled; XML <tool_call> blocks enabled")
        );
    }
    Ok(())
}

fn resolve_auth_api_key(
    service: AuthService,
    api_key: Option<String>,
) -> Result<String, Box<dyn std::error::Error>> {
    if matches!(service, AuthService::OpenAiCodex | AuthService::Grok) {
        return Err(format!("{} login does not accept API keys", service.display_name()).into());
    }
    match api_key {
        Some(api_key) if !api_key.trim().is_empty() => Ok(api_key),
        Some(_) => Err(format!("{} API key cannot be empty", service.display_name()).into()),
        None => {
            let api_key = read_secret(&format!("{} API key: ", service.display_name()))?;
            if api_key.trim().is_empty() {
                return Err(format!("{} API key cannot be empty", service.display_name()).into());
            }
            Ok(api_key)
        }
    }
}

fn pebble_config_home() -> Result<PathBuf, Box<dyn std::error::Error>> {
    resolve_pebble_config_home()
        .ok_or_else(|| "could not resolve PEBBLE_CONFIG_HOME, HOME, or USERPROFILE".into())
}

fn parse_model_command(input: &str) -> Option<Option<String>> {
    let mut parts = input.split_whitespace();
    let command = parts.next()?;
    if command != "/model" && command != "/models" {
        return None;
    }

    let remainder = parts.collect::<Vec<_>>().join(" ");
    if remainder.is_empty() {
        Some(None)
    } else {
        Some(Some(remainder))
    }
}

fn parse_provider_command(input: &str) -> Option<Option<String>> {
    let mut parts = input.split_whitespace();
    let command = parts.next()?;
    if command != "/provider" && command != "/providers" {
        return None;
    }

    let remainder = parts.collect::<Vec<_>>().join(" ");
    if remainder.is_empty() {
        Some(None)
    } else {
        Some(Some(remainder))
    }
}

fn parse_route_command(input: &str) -> Option<Option<String>> {
    let mut parts = input.split_whitespace();
    let command = parts.next()?;
    if command != "/route" && command != "/routing" {
        return None;
    }

    let remainder = parts.collect::<Vec<_>>().join(" ");
    if remainder.is_empty() {
        Some(None)
    } else {
        Some(Some(remainder))
    }
}

fn parse_proxy_command(input: &str) -> Option<Result<ProxyCommand, String>> {
    let mut parts = input.split_whitespace();
    let command = parts.next()?;
    if command != "/proxy" {
        return None;
    }

    let remainder = parts.collect::<Vec<_>>().join(" ");
    Some(parse_proxy_value(
        (!remainder.is_empty()).then_some(remainder.as_str()),
    ))
}

fn parse_reasoning_command(input: &str) -> Option<Result<Option<Option<ReasoningEffort>>, String>> {
    let mut parts = input.split_whitespace();
    let command = parts.next()?;
    if command != "/reasoning" && command != "/thinking" {
        return None;
    }

    let remainder = parts.collect::<Vec<_>>().join(" ");
    let trimmed = remainder.trim();
    Some(if trimmed.is_empty() {
        Ok(None)
    } else {
        parse_reasoning_effort_arg(trimmed).map(Some)
    })
}

fn parse_mode_command(input: &str) -> Option<Result<Option<CollaborationMode>, String>> {
    let mut parts = input.split_whitespace();
    let command = parts.next()?;
    if command != "/mode" {
        return None;
    }

    let remainder = parts.collect::<Vec<_>>().join(" ");
    let trimmed = remainder.trim();
    Some(if trimmed.is_empty() {
        Ok(None)
    } else {
        parse_collaboration_mode_arg(trimmed).map(Some)
    })
}

fn parse_fast_command(input: &str) -> Option<Result<Option<FastMode>, String>> {
    let mut parts = input.split_whitespace();
    let command = parts.next()?;
    if command != "/fast" {
        return None;
    }

    let remainder = parts.collect::<Vec<_>>().join(" ");
    let trimmed = remainder.trim();
    Some(match trimmed {
        "" => Ok(None),
        "on" => Ok(Some(FastMode::On)),
        "off" => Ok(Some(FastMode::Off)),
        other => Err(format!(
            "/fast accepts one optional argument: on or off (got {other})"
        )),
    })
}

fn parse_mcp_command(input: &str) -> Option<Result<McpCommand, String>> {
    let mut parts = input.split_whitespace();
    let command = parts.next()?;
    if command != "/mcp" {
        return None;
    }

    let args = parts.collect::<Vec<_>>();
    Some(match args.as_slice() {
        [] | ["status"] => Ok(McpCommand::Status),
        ["tools"] => Ok(McpCommand::Tools),
        ["reload"] => Ok(McpCommand::Reload),
        ["add", name] => Ok(McpCommand::Add {
            name: (*name).to_string(),
        }),
        ["enable", name] => Ok(McpCommand::Enable {
            name: (*name).to_string(),
        }),
        ["disable", name] => Ok(McpCommand::Disable {
            name: (*name).to_string(),
        }),
        [other, ..] => Err(format!(
            "/mcp accepts status, tools, reload, add <name>, enable <name>, or disable <name> (got {other})"
        )),
    })
}

fn parse_permissions_command(input: &str) -> Option<Result<Option<PermissionMode>, String>> {
    let mut parts = input.split_whitespace();
    let command = parts.next()?;
    if command == "/bypass" {
        let remainder = parts.collect::<Vec<_>>().join(" ");
        if !remainder.trim().is_empty() {
            return Some(Err("/bypass does not accept arguments".to_string()));
        }
        return Some(Ok(Some(PermissionMode::DangerFullAccess)));
    }
    if command != "/permissions" {
        return None;
    }

    let remainder = parts.collect::<Vec<_>>().join(" ");
    if remainder.trim().is_empty() {
        return Some(Ok(None));
    }
    Some(parse_permission_mode_arg(remainder.trim()).map(Some))
}

fn normalize_permission_mode(mode: &str) -> Option<&'static str> {
    match mode.trim().to_ascii_lowercase().as_str() {
        "read-only" | "readonly" | "read_only" => Some("read-only"),
        "workspace-write" | "workspacewrite" | "workspace_write" => Some("workspace-write"),
        "danger-full-access" | "dangerfullaccess" | "danger_full_access" => {
            Some("danger-full-access")
        }
        _ => None,
    }
}

fn permission_mode_from_label(mode: &str) -> PermissionMode {
    match mode {
        "read-only" => PermissionMode::ReadOnly,
        "workspace-write" => PermissionMode::WorkspaceWrite,
        "danger-full-access" => PermissionMode::DangerFullAccess,
        other => panic!("unsupported permission mode label: {other}"),
    }
}

pub(crate) fn parse_permission_mode_arg(value: &str) -> Result<PermissionMode, String> {
    normalize_permission_mode(value)
        .ok_or_else(|| {
            format!(
                "unsupported permission mode '{value}'. Use read-only, workspace-write, or danger-full-access."
            )
        })
        .map(permission_mode_from_label)
}

pub(crate) fn parse_collaboration_mode_arg(value: &str) -> Result<CollaborationMode, String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "build" => Ok(CollaborationMode::Build),
        "plan" => Ok(CollaborationMode::Plan),
        other => Err(format!("unsupported mode '{other}'. Use build or plan.")),
    }
}

pub(crate) fn parse_reasoning_effort_arg(value: &str) -> Result<Option<ReasoningEffort>, String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "" | "default" | "auto" | "off" => Ok(None),
        "minimal" => Ok(Some(ReasoningEffort::Minimal)),
        "low" => Ok(Some(ReasoningEffort::Low)),
        "medium" | "on" => Ok(Some(ReasoningEffort::Medium)),
        "high" => Ok(Some(ReasoningEffort::High)),
        "xhigh" | "x-high" => Ok(Some(ReasoningEffort::XHigh)),
        other => Err(format!(
            "unsupported reasoning effort '{other}'. Use default, minimal, low, medium, high, or xhigh."
        )),
    }
}

fn startup_runtime_defaults() -> StartupRuntimeDefaults {
    let state = load_model_state().ok();
    let permission_mode = permission_mode_from_env()
        .or_else(|| {
            state
                .as_ref()
                .and_then(|state| state.permission_mode.as_deref())
                .and_then(|value| parse_permission_mode_arg(value).ok())
        })
        .unwrap_or(PermissionMode::WorkspaceWrite);
    let collaboration_mode = state
        .as_ref()
        .and_then(|state| state.collaboration_mode.as_deref())
        .and_then(|value| parse_collaboration_mode_arg(value).ok())
        .unwrap_or(CollaborationMode::Build);
    let reasoning_effort = state
        .as_ref()
        .and_then(|state| state.reasoning_effort.as_deref())
        .and_then(|value| parse_reasoning_effort_arg(value).ok())
        .flatten();
    let fast_mode = state.as_ref().map_or(FastMode::Off, |state| {
        if state.fast_mode {
            FastMode::On
        } else {
            FastMode::Off
        }
    });

    StartupRuntimeDefaults {
        permission_mode,
        collaboration_mode,
        reasoning_effort,
        fast_mode,
    }
}

fn default_permission_mode() -> PermissionMode {
    permission_mode_from_env()
        .or_else(|| {
            load_model_state()
                .ok()
                .and_then(|state| state.permission_mode)
                .and_then(|value| parse_permission_mode_arg(&value).ok())
        })
        .unwrap_or(PermissionMode::WorkspaceWrite)
}

fn permission_mode_from_env() -> Option<PermissionMode> {
    env::var("PEBBLE_PERMISSION_MODE")
        .ok()
        .as_deref()
        .and_then(normalize_permission_mode)
        .map(permission_mode_from_label)
}

fn is_clear_provider_arg(value: &str) -> bool {
    matches!(value, "default" | "none" | "clear")
}

fn read_secret(prompt: &str) -> io::Result<String> {
    let mut stdout = io::stdout();
    write!(stdout, "{prompt}")?;
    stdout.flush()?;

    if !io::stdin().is_terminal() || !io::stdout().is_terminal() {
        let mut buffer = String::new();
        io::stdin().read_line(&mut buffer)?;
        while matches!(buffer.chars().last(), Some('\n' | '\r')) {
            buffer.pop();
        }
        return Ok(buffer);
    }

    enable_raw_mode()?;
    if let Err(error) = execute!(stdout, EnableBracketedPaste) {
        let _ = disable_raw_mode();
        return Err(error);
    }
    let result = read_secret_raw(&mut stdout);
    let disable_paste_result = execute!(stdout, DisableBracketedPaste);
    let disable_raw_result = disable_raw_mode();
    writeln!(stdout)?;
    disable_paste_result?;
    disable_raw_result?;
    result
}

fn read_secret_raw(out: &mut impl Write) -> io::Result<String> {
    let mut secret = String::new();
    let opened_at = Instant::now();
    loop {
        match event::read()? {
            Event::Paste(data) => {
                secret.push_str(trim_trailing_line_endings(&data));
            }
            Event::Key(KeyEvent { kind, .. }) if kind == KeyEventKind::Release => {}
            Event::Key(KeyEvent {
                code: KeyCode::Enter,
                ..
            }) if should_ignore_stale_secret_submit(&secret, opened_at.elapsed()) => {}
            Event::Key(KeyEvent {
                code: KeyCode::Enter,
                ..
            }) => return Ok(secret),
            Event::Key(KeyEvent {
                code: KeyCode::Backspace,
                ..
            }) => {
                secret.pop();
            }
            Event::Key(KeyEvent {
                code: KeyCode::Char('c'),
                modifiers,
                ..
            }) if modifiers.contains(KeyModifiers::CONTROL) => {
                return Err(io::Error::new(
                    io::ErrorKind::Interrupted,
                    "login cancelled",
                ));
            }
            Event::Key(KeyEvent {
                code: KeyCode::Char(ch),
                modifiers,
                ..
            }) if modifiers.is_empty() || modifiers == KeyModifiers::SHIFT => {
                secret.push(ch);
            }
            _ => {}
        }
        out.flush()?;
    }
}

fn should_ignore_stale_secret_submit(secret: &str, elapsed: Duration) -> bool {
    secret.is_empty() && elapsed <= SECRET_PROMPT_STALE_ENTER_WINDOW
}

fn trim_trailing_line_endings(value: &str) -> &str {
    value.trim_end_matches(['\r', '\n'])
}

fn base_runtime_tool_specs(tool_registry: &GlobalToolRegistry) -> Vec<RuntimeToolSpec> {
    tool_registry
        .entries()
        .iter()
        .map(|entry| RuntimeToolSpec {
            name: entry.definition.name.clone(),
            description: tuned_tool_description(
                &entry.definition.name,
                &entry
                    .definition
                    .description
                    .clone()
                    .unwrap_or_else(|| entry.definition.name.clone()),
            ),
            input_schema: entry.definition.input_schema.clone(),
            required_permission: entry.required_permission,
        })
        .collect()
}

fn tuned_tool_description(name: &str, base: &str) -> String {
    match name {
        "WebSearch" => format!(
            "{base} Prefer this for current information, release notes, changelogs, news, and finding relevant pages before reading them."
        ),
        "WebScrape" => format!(
            "{base} Prefer this when you already know the docs/article URLs and need readable page content or markdown to inspect."
        ),
        "WebFetch" => format!(
            "{base} Prefer this for a single known URL when you only need a quick fetch/summary; use WebScrape for richer doc/article reading."
        ),
        _ => base.to_string(),
    }
}

fn available_runtime_tool_specs(
    tool_registry: &GlobalToolRegistry,
    mcp_catalog: &McpCatalog,
) -> Vec<RuntimeToolSpec> {
    let mut specs = base_runtime_tool_specs(tool_registry);
    specs.extend(mcp_catalog.tool_specs());
    specs
}

fn filter_runtime_tool_specs(
    specs: Vec<RuntimeToolSpec>,
    allowed_tools: Option<&AllowedToolSet>,
) -> Vec<RuntimeToolSpec> {
    specs
        .into_iter()
        .filter(|spec| allowed_tools.is_none_or(|allowed| allowed.contains(&spec.name)))
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct StatusContext {
    cwd: PathBuf,
    session_path: Option<PathBuf>,
    loaded_config_files: usize,
    discovered_config_files: usize,
    instruction_file_count: usize,
    memory_file_count: usize,
    project_root: Option<PathBuf>,
    git_branch: Option<String>,
    sandbox_summary: String,
    web_tools_summary: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct StatusUsage {
    message_count: usize,
    turns: u32,
    undo_count: usize,
    redo_count: usize,
    latest: TokenUsage,
    cumulative: TokenUsage,
    estimated_tokens: usize,
    context_window: Option<ui::ContextWindowInfo>,
}

fn session_undo_redo_counts(session: &Session) -> (usize, usize) {
    session.metadata.as_ref().map_or((0, 0), |metadata| {
        (
            metadata.undo_stack.as_ref().map_or(0, Vec::len),
            metadata.redo_stack.as_ref().map_or(0, Vec::len),
        )
    })
}

#[derive(Debug, Clone)]
struct ResumeCommandOutcome {
    session: Session,
    message: Option<String>,
}

fn status_context(
    session_path: Option<&Path>,
) -> Result<StatusContext, Box<dyn std::error::Error>> {
    let cwd = env::current_dir()?;
    let loader = ConfigLoader::default_for(&cwd);
    let discovered_config_files = loader.discover().len();
    let runtime_config = loader.load()?;
    let sandbox_status = resolve_sandbox_status(runtime_config.sandbox(), &cwd);
    let project_context = runtime::ProjectContext::discover_with_git(&cwd, current_date())?;
    let (project_root, git_branch) =
        parse_git_status_metadata(project_context.git_status.as_deref());
    Ok(StatusContext {
        cwd,
        session_path: session_path.map(Path::to_path_buf),
        loaded_config_files: runtime_config.loaded_entries().len(),
        discovered_config_files,
        instruction_file_count: project_context.instruction_files.len(),
        memory_file_count: project_context.memory_files.len(),
        project_root,
        git_branch,
        sandbox_summary: format_sandbox_status(&sandbox_status),
        web_tools_summary: format_web_tools_status(),
    })
}

fn parse_git_status_metadata(status: Option<&str>) -> (Option<PathBuf>, Option<String>) {
    let Some(status) = status else {
        return (None, None);
    };
    let mut branch = None;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("## ") {
            branch = Some(rest.split("...").next().unwrap_or(rest).trim().to_string());
            break;
        }
    }
    let project_root = Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .current_dir(env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|stdout| PathBuf::from(stdout.trim()));
    (project_root, branch)
}

fn config_source_label(source: ConfigSource) -> &'static str {
    match source {
        ConfigSource::User => "user",
        ConfigSource::Project => "project",
        ConfigSource::Local => "local",
    }
}

fn format_status_context_window(context_window: Option<ui::ContextWindowInfo>) -> String {
    context_window.map_or_else(|| "unknown".to_string(), ui::format_context_window_usage)
}

fn format_status_report(
    service: ApiService,
    model: &str,
    usage: StatusUsage,
    permission_mode: &str,
    provider: Option<&str>,
    proxy_tool_calls: bool,
    collaboration_mode: CollaborationMode,
    reasoning_effort: Option<ReasoningEffort>,
    fast_mode: FastMode,
    mcp_catalog: &McpCatalog,
    context: &StatusContext,
) -> String {
    let backend = provider.map_or_else(
        || service.display_name().to_string(),
        |provider| format!("{} via {provider}", service.display_name()),
    );
    let permission = match permission_mode {
        "read-only" => "read only",
        "workspace-write" => "workspace",
        "danger-full-access" => "full access",
        other => other,
    };
    let mut flags = vec![
        collaboration_mode.as_str().to_string(),
        permission.to_string(),
        format!("think {}", reasoning_effort_label(reasoning_effort)),
    ];
    if fast_mode.enabled() {
        flags.push("fast".to_string());
    }
    if proxy_tool_calls {
        flags.push("proxy tools".to_string());
    }

    let project = context
        .project_root
        .as_deref()
        .unwrap_or(&context.cwd)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("workspace");
    let branch = context.git_branch.as_deref().unwrap_or("no git branch");
    let sandbox = summary_field(&context.sandbox_summary, "Status").unwrap_or("unknown");
    let web_service = summary_field(&context.web_tools_summary, "service").unwrap_or("disabled");
    let web_search = summary_field(&context.web_tools_summary, "web_search")
        .is_some_and(|value| value == "available");

    let mut output = String::new();
    let _ = writeln!(output, "{}", report_title("Pebble status"));
    let _ = writeln!(
        output,
        "  {} on {}\n  {}",
        short_model_name(model),
        backend,
        flags.join(" · ")
    );
    let turn_count = format!(
        "{} {}",
        usage.turns,
        if usage.turns == 1 { "turn" } else { "turns" }
    );
    let _ = writeln!(
        output,
        "  {} · {} · {}",
        turn_count,
        counted(usage.message_count, "message", "messages"),
        format_status_context_window(usage.context_window)
    );

    let _ = writeln!(output, "\n{}", report_section("Workspace"));
    let _ = writeln!(output, "  {project} on {branch}");
    let _ = writeln!(output, "  {}", context.cwd.display());
    let _ = writeln!(
        output,
        "  config {}/{} · {} · {}",
        context.loaded_config_files,
        context.discovered_config_files,
        counted(
            context.instruction_file_count,
            "instruction",
            "instructions"
        ),
        counted(context.memory_file_count, "memory", "memories"),
    );

    let _ = writeln!(output, "\n{}", report_section("Session"));
    let _ = writeln!(
        output,
        "  {} input · {} output tokens · {} latest",
        usage.cumulative.input_tokens,
        usage.cumulative.output_tokens,
        usage.latest.total_tokens(),
    );
    let _ = writeln!(
        output,
        "  undo {} · redo {} · estimated {} tokens",
        usage.undo_count, usage.redo_count, usage.estimated_tokens
    );
    let _ = writeln!(
        output,
        "  session {}",
        context.session_path.as_ref().map_or_else(
            || "live".to_string(),
            |path| path
                .file_stem()
                .and_then(|name| name.to_str())
                .unwrap_or("saved")
                .to_string()
        )
    );

    let _ = writeln!(output, "\n{}", report_section("Runtime"));
    let _ = writeln!(output, "  model auth {}", model_auth_status(service));
    let _ = writeln!(output, "  sandbox {sandbox}");
    let _ = writeln!(
        output,
        "  MCP {} · {}",
        counted(mcp_catalog.servers.len(), "server", "servers"),
        counted(mcp_catalog.tools.len(), "tool", "tools")
    );
    let _ = write!(
        output,
        "  web {web_service}{}",
        if web_search {
            " · ready"
        } else {
            " · unavailable"
        }
    );
    output
}

fn model_auth_status(service: ApiService) -> String {
    if service == ApiService::Grok {
        let executable = env::var("PEBBLE_GROK_CLI").unwrap_or_else(|_| "grok".to_string());
        return if command_is_available(&executable) {
            "official Grok CLI available".to_string()
        } else {
            "missing · run /login grok".to_string()
        };
    }
    if resolve_api_key_for(service).is_ok() {
        "connected".to_string()
    } else {
        format!(
            "missing · run /login {}",
            service.as_str().replace('_', "-")
        )
    }
}

fn command_is_available(command: &str) -> bool {
    let path = Path::new(command);
    if path.components().count() > 1 {
        return path.is_file();
    }
    env::var_os("PATH").is_some_and(|paths| {
        env::split_paths(&paths).any(|directory| directory.join(command).is_file())
    })
}

fn summary_field<'a>(summary: &'a str, key: &str) -> Option<&'a str> {
    summary.lines().find_map(|line| {
        let line = line.trim();
        line.strip_prefix(key)
            .map(|value| value.trim_start_matches([' ', '=']).trim())
            .filter(|value| !value.is_empty())
    })
}

fn counted(count: usize, singular: &str, plural: &str) -> String {
    format!("{count} {}", if count == 1 { singular } else { plural })
}

fn run_trace_viewer(
    trace_path: &Path,
    output_format: CliOutputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    let trace = load_turn_trace(trace_path)?;
    match output_format {
        CliOutputFormat::Text => println!("{}", render_trace_report(trace_path, &trace)),
        CliOutputFormat::Json => print_json(&trace_json_report(trace_path, &trace))?,
    }
    Ok(())
}

fn run_trace_replay(
    trace_path: &Path,
    output_format: CliOutputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    let trace = load_turn_trace(trace_path)?;
    match output_format {
        CliOutputFormat::Text => println!("{}", render_replay_report(trace_path, &trace)),
        CliOutputFormat::Json => print_json(&replay_json_report(trace_path, &trace))?,
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct GcReport {
    dry_run: bool,
    scanned: usize,
    deleted: usize,
    reclaimed_bytes: u64,
    entries: Vec<GcDeletedEntry>,
    errors: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct GcDeletedEntry {
    kind: &'static str,
    path: PathBuf,
    bytes: u64,
    reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ArtifactCandidate {
    path: PathBuf,
    modified: SystemTime,
    bytes: u64,
}

fn run_gc(dry_run: bool) -> Result<(), Box<dyn std::error::Error>> {
    let cwd = env::current_dir()?;
    let config = retention_config_for(&cwd);
    let report = collect_generated_artifacts(&cwd, config, dry_run);
    println!("{}", render_gc_report(&cwd, &config, &report));
    Ok(())
}

pub(crate) fn retention_config_for(cwd: &Path) -> RuntimeRetentionConfig {
    ConfigLoader::default_for(cwd)
        .load()
        .map(|config| config.retention())
        .unwrap_or_default()
}

fn collect_generated_artifacts(
    cwd: &Path,
    config: RuntimeRetentionConfig,
    dry_run: bool,
) -> GcReport {
    let mut report = GcReport {
        dry_run,
        scanned: 0,
        deleted: 0,
        reclaimed_bytes: 0,
        entries: Vec::new(),
        errors: Vec::new(),
    };
    collect_artifact_dir(
        &mut report,
        "trace",
        &cwd.join(".pebble").join("runs"),
        config.trace_days,
        config.max_trace_files,
    );
    collect_artifact_dir(
        &mut report,
        "eval",
        &cwd.join(".pebble").join("evals"),
        config.eval_days,
        config.max_eval_reports,
    );
    collect_artifact_dir(
        &mut report,
        "ci",
        &cwd.join(".pebble").join("ci"),
        config.ci_days,
        config.max_ci_reports,
    );

    if dry_run {
        report.reclaimed_bytes = report
            .entries
            .iter()
            .map(|entry| entry.bytes)
            .fold(0_u64, u64::saturating_add);
    } else {
        for entry in &report.entries {
            match fs::remove_file(&entry.path) {
                Ok(()) => {
                    report.deleted += 1;
                    report.reclaimed_bytes = report.reclaimed_bytes.saturating_add(entry.bytes);
                }
                Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                Err(error) => report
                    .errors
                    .push(format!("{}: {error}", entry.path.display())),
            }
        }
    }
    report
}

pub(crate) fn prune_generated_artifacts(cwd: &Path) {
    let _gc_report = collect_generated_artifacts(cwd, retention_config_for(cwd), false);
}

fn collect_artifact_dir(
    report: &mut GcReport,
    kind: &'static str,
    dir: &Path,
    max_age_days: Option<usize>,
    max_files: Option<usize>,
) {
    let mut candidates = match artifact_candidates(dir) {
        Ok(candidates) => candidates,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return,
        Err(error) => {
            report.errors.push(format!("{}: {error}", dir.display()));
            return;
        }
    };
    report.scanned += candidates.len();
    let mut removals = BTreeMap::<PathBuf, GcDeletedEntry>::new();
    let now = SystemTime::now();

    if let Some(days) = max_age_days {
        let max_age = Duration::from_secs(
            u64::try_from(days)
                .unwrap_or(u64::MAX / SECS_PER_DAY)
                .saturating_mul(SECS_PER_DAY),
        );
        for candidate in &candidates {
            if now.duration_since(candidate.modified).unwrap_or_default() > max_age {
                removals.insert(
                    candidate.path.clone(),
                    GcDeletedEntry {
                        kind,
                        path: candidate.path.clone(),
                        bytes: candidate.bytes,
                        reason: format!("age>{days}d"),
                    },
                );
            }
        }
    }

    if let Some(max_files) = max_files {
        candidates.sort_by(|left, right| {
            left.modified
                .cmp(&right.modified)
                .then_with(|| left.path.cmp(&right.path))
        });
        let overflow = candidates.len().saturating_sub(max_files);
        for candidate in candidates.iter().take(overflow) {
            removals
                .entry(candidate.path.clone())
                .and_modify(|entry| {
                    if !entry.reason.contains("count") {
                        let _ = write!(entry.reason, ", count>{max_files}");
                    }
                })
                .or_insert_with(|| GcDeletedEntry {
                    kind,
                    path: candidate.path.clone(),
                    bytes: candidate.bytes,
                    reason: format!("count>{max_files}"),
                });
        }
    }

    report.entries.extend(removals.into_values());
}

fn artifact_candidates(dir: &Path) -> io::Result<Vec<ArtifactCandidate>> {
    fs::read_dir(dir)?
        .map(|entry| {
            let entry = entry?;
            let path = entry.path();
            let metadata = entry.metadata()?;
            Ok((path, metadata))
        })
        .filter_map(|result| match result {
            Ok((path, metadata))
                if metadata.is_file()
                    && path.extension().is_some_and(|ext| ext == "json")
                    && path
                        .file_name()
                        .is_none_or(|name| name != EVAL_HISTORY_INDEX_FILE) =>
            {
                Some(Ok(ArtifactCandidate {
                    path,
                    modified: metadata.modified().unwrap_or(UNIX_EPOCH),
                    bytes: metadata.len(),
                }))
            }
            Ok(_) => None,
            Err(error) => Some(Err(error)),
        })
        .collect()
}

fn render_gc_report(cwd: &Path, config: &RuntimeRetentionConfig, report: &GcReport) -> String {
    let mut output = String::new();
    let _ = writeln!(
        output,
        "{}",
        report_title(if report.dry_run {
            "Pebble GC Dry Run"
        } else {
            "Pebble GC"
        })
    );
    let _ = writeln!(output, "  {} {}", report_label("root"), cwd.display());
    let _ = writeln!(
        output,
        "  {} traces: days={} max_files={} | evals: days={} max_reports={} | ci: days={} max_reports={}",
        report_label("policy"),
        optional_retention_label(config.trace_days),
        optional_retention_label(config.max_trace_files),
        optional_retention_label(config.eval_days),
        optional_retention_label(config.max_eval_reports),
        optional_retention_label(config.ci_days),
        optional_retention_label(config.max_ci_reports),
    );
    let _ = writeln!(
        output,
        "  {} scanned={} matched={} deleted={} {}={}",
        report_label("summary"),
        report.scanned,
        report.entries.len(),
        report.deleted,
        if report.dry_run {
            "reclaimable"
        } else {
            "reclaimed"
        },
        format_bytes(report.reclaimed_bytes),
    );

    if !report.entries.is_empty() {
        let _ = writeln!(output);
        let _ = writeln!(output, "  {}", report_section("Artifacts"));
        for entry in report.entries.iter().take(20) {
            let _ = writeln!(
                output,
                "    {} {} {} {}",
                report_label(entry.kind),
                entry.reason,
                format_bytes(entry.bytes),
                entry.path.display()
            );
        }
        if report.entries.len() > 20 {
            let _ = writeln!(output, "    ... {} more", report.entries.len() - 20);
        }
    }

    if !report.errors.is_empty() {
        let _ = writeln!(output);
        let _ = writeln!(output, "  {}", report_section("Errors"));
        for error in &report.errors {
            let _ = writeln!(output, "    - {error}");
        }
    }

    output.trim_end().to_string()
}

fn optional_retention_label(value: Option<usize>) -> String {
    value.map_or_else(|| "unlimited".to_string(), |value| value.to_string())
}

fn format_bytes(bytes: u64) -> String {
    const KIB: u64 = 1024;
    const MIB: u64 = KIB * 1024;
    if bytes >= MIB {
        format_scaled_bytes(bytes, MIB, "MiB")
    } else if bytes >= KIB {
        format_scaled_bytes(bytes, KIB, "KiB")
    } else {
        format!("{bytes}B")
    }
}

fn format_scaled_bytes(bytes: u64, unit: u64, suffix: &str) -> String {
    let tenths = bytes.saturating_mul(10).saturating_add(unit / 2) / unit;
    format!("{}.{:01}{suffix}", tenths / 10, tenths % 10)
}

fn print_json<T: Serialize>(value: &T) -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", serde_json::to_string_pretty(value)?);
    Ok(())
}

fn run_eval_replay(
    options: &EvalReplayOptions,
    output_format: CliOutputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    let report = load_eval_report(&options.report_path)?;
    if let Some(case_id) = &options.case_id {
        let found = report
            .cases
            .iter()
            .any(|case| case.case.id == *case_id || case.result.id == *case_id);
        if !found {
            return Err(format!(
                "eval report `{}` does not contain case `{case_id}`",
                options.report_path.display()
            )
            .into());
        }
    }
    match output_format {
        CliOutputFormat::Text => println!("{}", render_eval_replay_report(options, &report)),
        CliOutputFormat::Json => print_json(&eval_replay_json_report(options, &report))?,
    }
    Ok(())
}

fn run_eval_history(
    filter: &EvalHistoryFilter,
    output_format: CliOutputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    let cwd = env::current_dir()?;
    let index = rebuild_eval_history_index(&cwd)?;
    write_eval_history_index(&cwd, &index)?;
    match output_format {
        CliOutputFormat::Text => println!("{}", render_eval_history_report(filter, &index)),
        CliOutputFormat::Json => print_json(&eval_history_json_report(filter, &index))?,
    }
    Ok(())
}

fn run_eval_compare(
    old_path: &Path,
    new_path: &Path,
    output_format: CliOutputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    let old_report = load_eval_report(old_path)?;
    let new_report = load_eval_report(new_path)?;
    match output_format {
        CliOutputFormat::Text => println!(
            "{}",
            render_eval_compare_report(old_path, &old_report, new_path, &new_report)
        ),
        CliOutputFormat::Json => print_json(&eval_compare_json_report(
            old_path,
            &old_report,
            new_path,
            &new_report,
        ))?,
    }
    Ok(())
}

fn format_web_tools_status() -> String {
    let api_key_configured = resolve_exa_api_key().is_ok();
    let base_url = resolve_exa_base_url();
    let (web_search_available, web_scrape_available) = current_tool_registry()
        .map(|registry| {
            let mut has_search = false;
            let mut has_scrape = false;
            for entry in registry.entries() {
                if entry.definition.name == "WebSearch" {
                    has_search = true;
                }
                if entry.definition.name == "WebScrape" {
                    has_scrape = true;
                }
            }
            (has_search, has_scrape)
        })
        .unwrap_or((false, false));

    format!(
        "service=Exa\nbase_url={base_url}\napi_key={}\nweb_search={}\nweb_scrape={}",
        if api_key_configured {
            "configured"
        } else {
            "missing"
        },
        if web_search_available {
            "available"
        } else {
            "missing"
        },
        if web_scrape_available {
            "available"
        } else {
            "missing"
        },
    )
}

fn resolve_exa_base_url() -> String {
    env::var("EXA_BASE_URL").unwrap_or_else(|_| "https://api.exa.ai".to_string())
}

fn resolve_exa_api_key() -> Result<String, Box<dyn std::error::Error>> {
    match env::var("EXA_API_KEY") {
        Ok(value) if !value.trim().is_empty() => Ok(value),
        Ok(_) => Err("EXA_API_KEY is empty".into()),
        Err(env::VarError::NotPresent) => {
            let path = pebble_config_home()?.join("credentials.json");
            let contents = fs::read_to_string(path)?;
            let parsed = serde_json::from_str::<serde_json::Value>(&contents)?;
            parsed
                .get("exa_api_key")
                .and_then(serde_json::Value::as_str)
                .filter(|value| !value.trim().is_empty())
                .map(ToOwned::to_owned)
                .ok_or_else(|| "missing exa_api_key".into())
        }
        Err(error) => Err(Box::new(error)),
    }
}

fn format_resume_report(session_path: &str, message_count: usize, turns: u32) -> String {
    ui::setting_changed(
        "session resumed",
        &[
            ("file", session_path),
            ("messages", &message_count.to_string()),
            ("turns", &turns.to_string()),
        ],
    )
}

fn format_export_report(path: &str, messages: usize) -> String {
    ui::setting_changed(
        "export",
        &[
            ("result", "wrote transcript"),
            ("file", path),
            ("messages", &messages.to_string()),
        ],
    )
}

fn format_compact_report(removed: usize) -> String {
    ui::setting_changed(
        "compacted",
        &[
            ("result", "session context reduced"),
            ("messages", &removed.to_string()),
        ],
    )
}

fn format_clear_report(
    model: &str,
    collaboration_mode: CollaborationMode,
    permission_mode: PermissionMode,
    session_id: &str,
) -> String {
    ui::setting_changed(
        "session cleared",
        &[
            ("mode", "fresh session"),
            ("model", model),
            ("session mode", collaboration_mode.as_str()),
            ("permissions", permission_mode.as_str()),
            ("session", session_id),
        ],
    )
}

fn format_sandbox_status(status: &runtime::SandboxStatus) -> String {
    let mode = status.filesystem_mode.as_str();
    let active = if status.active { "active" } else { "inactive" };
    let network = if status.network_active {
        "isolated"
    } else if status.requested.network_isolation {
        "requested-unavailable"
    } else {
        "shared"
    };
    let namespace = if status.namespace_active {
        "restricted"
    } else if status.requested.namespace_restrictions {
        "requested-unavailable"
    } else {
        "shared"
    };
    let mounts = if status.allowed_mounts.is_empty() {
        "<none>".to_string()
    } else {
        status.allowed_mounts.join(", ")
    };
    let mut line = format!(
        "Enabled          {}\n  Status           {}\n  Namespace        {}\n  Network          {}\n  Filesystem       {}\n  Allowed mounts   {}\n  Container        {}",
        if status.enabled { "yes" } else { "no" },
        active,
        namespace,
        network,
        mode,
        mounts,
        if status.in_container { "yes" } else { "no" },
    );
    if let Some(reason) = &status.fallback_reason {
        let _ = write!(line, "\n  Fallback         {reason}");
    }
    line
}

fn format_auto_compaction_notice(event: runtime::AutoCompactionEvent) -> String {
    match (event.removed_message_count, event.pruned_tool_result_count) {
        (removed, 0) => format!("[auto-compacted: removed {removed} messages]"),
        (0, pruned) => format!("[auto-compacted: pruned {pruned} stale tool results]"),
        (removed, pruned) => {
            format!(
                "[auto-compacted: removed {removed} messages, pruned {pruned} stale tool results]"
            )
        }
    }
}

fn current_epoch_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis().try_into().unwrap_or(u64::MAX))
        .unwrap_or_default()
}

fn current_epoch_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default()
}

fn persist_runtime_defaults(
    permission_mode: PermissionMode,
    collaboration_mode: CollaborationMode,
    reasoning_effort: Option<ReasoningEffort>,
    fast_mode: FastMode,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut state = load_model_state()?;
    state.permission_mode = Some(permission_mode.as_str().to_string());
    state.collaboration_mode = Some(collaboration_mode.as_str().to_string());
    state.reasoning_effort =
        reasoning_effort.map(|effort| reasoning_effort_label(Some(effort)).to_string());
    state.fast_mode = fast_mode.enabled();
    save_model_state(&state)
}

fn session_age_secs(modified_epoch_secs: u64) -> u64 {
    current_epoch_secs().saturating_sub(modified_epoch_secs)
}

fn auto_compact_inactive_sessions(
    active_session_id: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    for summary in list_managed_sessions()? {
        if summary.id == active_session_id
            || session_age_secs(summary.modified_epoch_secs) < OLD_SESSION_COMPACTION_AGE_SECS
        {
            continue;
        }
        let path = summary.path.clone();
        let Ok(session) = Session::load_from_path(&path) else {
            continue;
        };
        if !runtime::should_compact(&session, CompactionConfig::default()) {
            continue;
        }
        let mut compacted =
            runtime::compact_session(&session, CompactionConfig::default()).compacted_session;
        let model = compacted.metadata.as_ref().map_or_else(
            || DEFAULT_MODEL.to_string(),
            |metadata| metadata.model.clone(),
        );
        let state = session_runtime_state(
            &compacted,
            &model,
            None,
            default_permission_mode(),
            CollaborationMode::Build,
            None,
            FastMode::Off,
            false,
        );
        compacted.metadata = Some(derive_session_metadata(
            &compacted,
            &model,
            state.allowed_tools.as_ref(),
            state.permission_mode,
            state.collaboration_mode,
            state.reasoning_effort,
            state.fast_mode,
            state.proxy_tool_calls,
        ));
        compacted.save_to_path(&path)?;
    }
    Ok(())
}

fn run_config_command(
    section: Option<&str>,
    output_format: CliOutputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    if section == Some("check") {
        let cwd = env::current_dir()?;
        let loader = ConfigLoader::default_for(&cwd);
        let report = loader.check();
        match output_format {
            CliOutputFormat::Text => println!("{}", render_config_check_report(&cwd, &report)),
            CliOutputFormat::Json => print_json(&config_check_json_report(&cwd, &report))?,
        }
        if report.is_ok() {
            return Ok(());
        }
        return Err("config check failed".into());
    }

    match output_format {
        CliOutputFormat::Text => println!("{}", render_config_report(section)?),
        CliOutputFormat::Json => {
            let cwd = env::current_dir()?;
            let loader = ConfigLoader::default_for(&cwd);
            let runtime_config = loader.load()?;
            print_json(&serde_json::json!({
                "kind": "config",
                "cwd": cwd,
                "section": section,
                "loaded_files": runtime_config.loaded_entries().iter().map(|entry| {
                    serde_json::json!({
                        "source": config_source_label(entry.source),
                        "path": entry.path,
                    })
                }).collect::<Vec<_>>(),
                "merged": runtime_config.as_json().render(),
            }))?;
        }
    }
    Ok(())
}

fn render_config_check_report(cwd: &Path, report: &runtime::ConfigCheckReport) -> String {
    let mut lines = vec![
        report_title("Config Check"),
        format!("  {} {}", report_label("cwd"), cwd.display()),
        format!(
            "  {} {}",
            report_label("result"),
            if report.is_ok() { "ok" } else { "failed" }
        ),
        format!(
            "  {} loaded={} discovered={}",
            report_label("files"),
            report.loaded_entries.len(),
            report.discovered_entries.len()
        ),
        String::new(),
        format!("  {}", report_section("Files")),
    ];

    for entry in &report.discovered_entries {
        let source = config_source_label(entry.source);
        let status = if report
            .loaded_entries
            .iter()
            .any(|loaded_entry| loaded_entry.path == entry.path)
        {
            "loaded"
        } else {
            "missing"
        };
        lines.push(format!(
            "    {source:<7} {status:<7} {}",
            entry.path.display()
        ));
    }

    lines.push(String::new());
    lines.push(format!("  {}", report_section("Issues")));
    if report.issues.is_empty() {
        lines.push("    none".to_string());
    } else {
        for issue in &report.issues {
            let path = issue.path.as_ref().map_or_else(
                || "merged settings".to_string(),
                |path| path.display().to_string(),
            );
            let field = issue.field_path.as_deref().unwrap_or("<file>");
            lines.push(format!("    - {path}: {field}: {}", issue.message));
        }
    }

    lines.join("\n")
}

fn config_check_json_report(cwd: &Path, report: &runtime::ConfigCheckReport) -> JsonValue {
    serde_json::json!({
        "kind": "config_check",
        "cwd": cwd,
        "ok": report.is_ok(),
        "discovered_files": report.discovered_entries.iter().map(|entry| {
            serde_json::json!({
                "source": config_source_label(entry.source),
                "path": entry.path,
                "loaded": report.loaded_entries.iter().any(|loaded_entry| loaded_entry.path == entry.path),
            })
        }).collect::<Vec<_>>(),
        "issues": report.issues.iter().map(|issue| {
            serde_json::json!({
                "path": issue.path,
                "field_path": issue.field_path,
                "message": issue.message,
            })
        }).collect::<Vec<_>>(),
    })
}

fn render_config_report(section: Option<&str>) -> Result<String, Box<dyn std::error::Error>> {
    if section == Some("check") {
        let cwd = env::current_dir()?;
        let loader = ConfigLoader::default_for(&cwd);
        return Ok(render_config_check_report(&cwd, &loader.check()));
    }

    let cwd = env::current_dir()?;
    let loader = ConfigLoader::default_for(&cwd);
    let discovered = loader.discover();
    let runtime_config = loader.load()?;

    let mut lines = vec![
        format!(
            "Config\n  Working directory {}\n  Loaded files      {}\n  Merged keys       {}",
            cwd.display(),
            runtime_config.loaded_entries().len(),
            runtime_config.merged().len()
        ),
        "Discovered files".to_string(),
    ];
    for entry in discovered {
        let source = config_source_label(entry.source);
        let status = if runtime_config
            .loaded_entries()
            .iter()
            .any(|loaded_entry| loaded_entry.path == entry.path)
        {
            "loaded"
        } else {
            "missing"
        };
        lines.push(format!(
            "  {source:<7} {status:<7} {}",
            entry.path.display()
        ));
    }

    if let Some(section) = section {
        lines.push(format!("Merged section: {section}"));
        let value = match section {
            "env" => runtime_config.get("env"),
            "hooks" => runtime_config.get("hooks"),
            "model" => runtime_config.get("model"),
            "plugins" => runtime_config.get("plugins"),
            other => {
                lines.push(format!(
                    "  Unsupported config section '{other}'. Use env, hooks, model, or plugins."
                ));
                return Ok(lines.join("\n"));
            }
        };
        lines.push(format!(
            "  {}",
            match value {
                Some(value) => value.render(),
                None => "<unset>".to_string(),
            }
        ));
        return Ok(lines.join("\n"));
    }

    lines.push("Merged JSON".to_string());
    lines.push(format!("  {}", runtime_config.as_json().render()));
    Ok(lines.join("\n"))
}

fn current_plugin_manager() -> Result<PluginManager, Box<dyn std::error::Error>> {
    let cwd = env::current_dir()?;
    let loader = ConfigLoader::default_for(&cwd);
    let runtime_config = loader.load()?;
    Ok(build_plugin_manager(&cwd, &loader, &runtime_config))
}

fn render_plugins_report(plugins: &[PluginSummary]) -> String {
    let mut lines = vec![
        report_title("Plugins"),
        format!("  {} {}", report_label("count:"), plugins.len()),
    ];
    if plugins.is_empty() {
        lines.push(format!(
            "  {} no plugins discovered",
            report_label("state:")
        ));
        return lines.join("\n");
    }

    for plugin in plugins {
        lines.push(String::new());
        lines.push(format!(
            "  {}",
            format!("{}", plugin.metadata.name.as_str().bold())
        ));
        lines.push(format!(
            "    {} {}  {} {}  {} {}  {} {}",
            report_label("id"),
            plugin.metadata.id,
            report_label("kind"),
            plugin.metadata.kind,
            report_label("version"),
            plugin.metadata.version,
            report_label("state"),
            if plugin.enabled {
                "enabled"
            } else {
                "disabled"
            }
        ));
        lines.push(format!("    {}", plugin.metadata.description));
        if let Some(root) = &plugin.metadata.root {
            lines.push(format!("    {} {}", report_label("root"), root.display()));
        }
    }

    lines.join("\n")
}

fn resolve_plugin_summary(
    manager: &PluginManager,
    target: &str,
) -> Result<PluginSummary, PluginError> {
    let plugins = manager.list_installed_plugins()?;
    plugins
        .into_iter()
        .find(|plugin| plugin.metadata.id == target || plugin.metadata.name == target)
        .ok_or_else(|| PluginError::NotFound(format!("plugin `{target}` was not found")))
}

fn render_plugin_action_result(
    action: &str,
    plugin_id: &str,
    name: &str,
    version_line: &str,
    state: &str,
) -> String {
    format!(
        "{}\n  {} {}\n  {} {}\n  {} {}\n  {} {}\n  {} {}",
        report_title("Plugins"),
        report_label("action:"),
        action,
        report_label("plugin:"),
        name,
        report_label("id:"),
        plugin_id,
        report_label("version:"),
        version_line,
        report_label("state:"),
        state
    )
}

fn handle_plugins_command(
    action: Option<&str>,
    target: Option<&str>,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut manager = current_plugin_manager()?;
    match action {
        None | Some("list") => Ok(render_plugins_report(&manager.list_installed_plugins()?)),
        Some("help") => Ok([
            "Plugins".to_string(),
            "  Usage            /plugins [list|help|install <path>|enable <id>|disable <id>|uninstall <id>|update <id>]".to_string(),
            "  Install          Point at a local plugin root that contains `.codex-plugin/plugin.json`.".to_string(),
            "  Example          /plugins install ./plugins/my-plugin".to_string(),
            "  Enable           /plugins enable <id>".to_string(),
            "  Disable          /plugins disable <id>".to_string(),
            "  Layout           Local plugins typically store skills, optional MCP manifests, and plugin metadata under the plugin root.".to_string(),
        ]
        .join("\n")),
        Some("install") => {
            let Some(target) = target else {
                return Ok("Plugins\n  error: missing install target\n  usage: /plugins install <path>".to_string());
            };
            let install = manager.install(target)?;
            let plugin = resolve_plugin_summary(&manager, &install.plugin_id).ok();
            Ok(render_plugin_action_result(
                "installed",
                &install.plugin_id,
                plugin
                    .as_ref()
                    .map_or(install.plugin_id.as_str(), |plugin| plugin.metadata.name.as_str()),
                &install.version,
                if plugin.as_ref().is_some_and(|plugin| plugin.enabled) {
                    "enabled"
                } else {
                    "disabled"
                },
            ))
        }
        Some("enable") => {
            let Some(target) = target else {
                return Ok("Plugins\n  error: missing enable target\n  usage: /plugins enable <id>".to_string());
            };
            let plugin = resolve_plugin_summary(&manager, target)?;
            manager.enable(&plugin.metadata.id)?;
            Ok(render_plugin_action_result(
                "enabled",
                &plugin.metadata.id,
                &plugin.metadata.name,
                &plugin.metadata.version,
                "enabled",
            ))
        }
        Some("disable") => {
            let Some(target) = target else {
                return Ok("Plugins\n  error: missing disable target\n  usage: /plugins disable <id>".to_string());
            };
            let plugin = resolve_plugin_summary(&manager, target)?;
            manager.disable(&plugin.metadata.id)?;
            Ok(render_plugin_action_result(
                "disabled",
                &plugin.metadata.id,
                &plugin.metadata.name,
                &plugin.metadata.version,
                "disabled",
            ))
        }
        Some("uninstall") => {
            let Some(target) = target else {
                return Ok("Plugins\n  error: missing uninstall target\n  usage: /plugins uninstall <id>".to_string());
            };
            let plugin = resolve_plugin_summary(&manager, target)?;
            manager.uninstall(&plugin.metadata.id)?;
            Ok(format!(
                "Plugins\n  action:  uninstalled\n  plugin:  {}\n  id:      {}",
                plugin.metadata.name, plugin.metadata.id
            ))
        }
        Some("update") => {
            let Some(target) = target else {
                return Ok("Plugins\n  error: missing update target\n  usage: /plugins update <id>".to_string());
            };
            let plugin = resolve_plugin_summary(&manager, target)?;
            let update = manager.update(&plugin.metadata.id)?;
            Ok(render_plugin_action_result(
                "updated",
                &update.plugin_id,
                &plugin.metadata.name,
                &format!("{} -> {}", update.old_version, update.new_version),
                if resolve_plugin_summary(&manager, &update.plugin_id)
                    .ok()
                    .is_some_and(|summary| summary.enabled)
                {
                    "enabled"
                } else {
                    "disabled"
                },
            ))
        }
        Some(other) => Ok(format!(
            "Plugins\n  error: unsupported action `{other}`\n  usage: /plugins [list|install|enable|disable|uninstall|update]"
        )),
    }
}

fn plugins_command_is_mutating(action: Option<&str>) -> bool {
    matches!(
        action.unwrap_or("list"),
        "install" | "enable" | "disable" | "uninstall" | "update"
    )
}

fn render_memory_report() -> Result<String, Box<dyn std::error::Error>> {
    let cwd = env::current_dir()?;
    let project_context = runtime::ProjectContext::discover(&cwd, current_date())?;
    let mut lines = vec![format!(
        "Memory\n  Working directory {}\n  Instruction files {}\n  Memory files      {}",
        cwd.display(),
        project_context.instruction_files.len(),
        project_context.memory_files.len(),
    )];

    lines.push("Instruction files".to_string());
    if project_context.instruction_files.is_empty() {
        lines.push("  No instruction markdown files discovered.".to_string());
    } else {
        for (index, file) in project_context.instruction_files.iter().enumerate() {
            let preview = file.content.lines().next().unwrap_or("").trim();
            lines.push(format!("  {}. {}", index + 1, file.path.display()));
            lines.push(format!(
                "     lines={} preview={}",
                file.content.lines().count(),
                if preview.is_empty() {
                    "<empty>"
                } else {
                    preview
                }
            ));
        }
    }

    lines.push("Memory files".to_string());
    if project_context.memory_files.is_empty() {
        lines.push("  No `.pebble/memory` files discovered.".to_string());
    } else {
        for (index, file) in project_context.memory_files.iter().enumerate() {
            let preview = file.content.lines().next().unwrap_or("").trim();
            lines.push(format!("  {}. {}", index + 1, file.path.display()));
            lines.push(format!(
                "     lines={} preview={}",
                file.content.lines().count(),
                if preview.is_empty() {
                    "<empty>"
                } else {
                    preview
                }
            ));
        }
    }

    Ok(lines.join("\n"))
}

fn render_diff_report() -> Result<String, Box<dyn std::error::Error>> {
    let cwd = env::current_dir()?;
    let staged = run_git_capture(&cwd, &["diff", "--cached"])?;
    let unstaged = run_git_capture(&cwd, &["diff"])?;

    let mut sections = Vec::new();
    if !staged.trim().is_empty() {
        sections.push(format!("Staged changes:\n{}", staged.trim_end()));
    }
    if !unstaged.trim().is_empty() {
        sections.push(format!("Unstaged changes:\n{}", unstaged.trim_end()));
    }

    if sections.is_empty() {
        return Ok(ui::setting_changed(
            "diff",
            &[
                ("result", "clean working tree"),
                ("detail", "no current changes"),
            ],
        ));
    }

    Ok(format!("Diff\n\n{}", sections.join("\n\n")))
}

fn render_patch_report(args: Option<&str>) -> Result<String, Box<dyn std::error::Error>> {
    let (dry_run, patch, source) = load_patch_command(args)?;
    let output = apply_patch(&patch, dry_run)?;
    Ok(format_patch_report(&output, &source))
}

fn load_patch_command(
    args: Option<&str>,
) -> Result<(bool, String, String), Box<dyn std::error::Error>> {
    let mut dry_run = true;
    let mut source = args.unwrap_or_default().trim();
    if source == "--apply" {
        return Err("usage: /patch --apply <patch-file-or-inline-diff>".into());
    } else if let Some(rest) = source.strip_prefix("--apply ") {
        dry_run = false;
        source = rest.trim_start();
    } else if source == "--check" {
        return Err("usage: /patch --check <patch-file-or-inline-diff>".into());
    } else if let Some(rest) = source.strip_prefix("--check ") {
        dry_run = true;
        source = rest.trim_start();
    }

    if source.is_empty() {
        return Err("usage: /patch [--check|--apply] <patch-file-or-inline-diff>".into());
    }

    if looks_like_inline_patch(source) {
        return Ok((
            dry_run,
            source.to_string(),
            inline_patch_source_label(source),
        ));
    }

    let path = split_command_arguments(source)
        .into_iter()
        .next()
        .ok_or("usage: /patch [--check|--apply] <patch-file-or-inline-diff>")?;
    let content = fs::read_to_string(&path)?;
    Ok((dry_run, content, path))
}

fn looks_like_inline_patch(value: &str) -> bool {
    let trimmed = value.trim_start();
    trimmed.starts_with("*** Begin Patch")
        || trimmed.starts_with("diff --git ")
        || trimmed.starts_with("--- ")
}

fn inline_patch_source_label(value: &str) -> String {
    let line_count = value.lines().count().max(1);
    format!("inline patch ({line_count} lines)")
}

fn format_patch_report(output: &runtime::ApplyPatchOutput, source: &str) -> String {
    let mode = if output.dry_run { "check" } else { "apply" };
    let result = if output.dry_run {
        "validated"
    } else {
        "applied"
    };
    let mut lines = vec![
        "Patch".to_string(),
        format!("  Mode             {mode}"),
        format!("  Result           {result}"),
        format!("  Source           {source}"),
        format!("  Summary          {}", output.summary),
    ];
    if !output.changed_files.is_empty() {
        lines.push("  Files".to_string());
        for file in &output.changed_files {
            lines.push(format!("    {:<8} {}", file.action, file.file_path));
        }
    }
    lines.join("\n")
}

fn run_git_capture(cwd: &Path, args: &[&str]) -> Result<String, Box<dyn std::error::Error>> {
    let output = Command::new("git").args(args).current_dir(cwd).output()?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(format!("git {} failed: {stderr}", args.join(" ")).into());
    }
    Ok(String::from_utf8(output.stdout)?)
}

fn render_version_report() -> String {
    let target = BUILD_TARGET.unwrap_or("unknown");
    format!(
        "Version\n  Version          {VERSION}\n  Target           {target}\n  Build date       {BUILD_DATE}"
    )
}

fn render_export_text(session: &Session, session_path: Option<&Path>) -> String {
    let mut lines = vec!["# Conversation Export".to_string(), String::new()];
    if let Some(summary) = render_export_archived_tool_summary(session, session_path) {
        lines.push(summary);
        lines.push(String::new());
    }
    for (index, message) in session.messages.iter().enumerate() {
        let role = match message.role {
            MessageRole::System => "system",
            MessageRole::User => "user",
            MessageRole::Assistant => "assistant",
            MessageRole::Tool => "tool",
        };
        lines.push(format!("## {}. {role}", index + 1));
        for block in &message.blocks {
            match block {
                ContentBlock::Text { text } => lines.push(text.clone()),
                ContentBlock::Thinking { text, signature } => {
                    lines.push(format!(
                        "[thinking hidden chars={} signature={}]",
                        text.chars().count(),
                        if signature.is_some() {
                            "present"
                        } else {
                            "absent"
                        }
                    ));
                }
                ContentBlock::ToolUse { id, name, input } => {
                    lines.push(format!("[tool_use id={id} name={name}] {input}"));
                }
                ContentBlock::ToolResult {
                    tool_use_id,
                    tool_name,
                    output,
                    is_error,
                    compacted,
                    archived_output_path,
                } => {
                    let output = get_tool_result_context_output(output, *compacted);
                    lines.push(format!(
                        "[tool_result id={tool_use_id} name={tool_name} error={is_error}] {output}"
                    ));
                    if let Some(archived_output_path) = archived_output_path {
                        lines.push(format!("[tool_result_archive path={archived_output_path}]"));
                    }
                }
                ContentBlock::CompactionSummary {
                    summary,
                    recent_messages_preserved,
                    ..
                } => lines.push(get_compact_continuation_message(
                    summary,
                    true,
                    *recent_messages_preserved,
                )),
            }
        }
        lines.push(String::new());
    }
    lines.join("\n")
}

fn render_export_archived_tool_summary(
    session: &Session,
    session_path: Option<&Path>,
) -> Option<String> {
    let summary = summarize_archived_tool_results(session, session_path);
    if summary.entries.is_empty() {
        return None;
    }

    let mut lines = vec![
        "## Archived Tool Outputs".to_string(),
        format!("- Count: {}", summary.entries.len()),
        format!("- Available sidecars: {}", summary.available_count),
        format!("- Missing sidecars: {}", summary.missing_count),
    ];
    if summary.unrecoverable_count > 0 {
        lines.push(format!(
            "- Unrecoverable entries: {}",
            summary.unrecoverable_count
        ));
    }
    lines.push("- Suggested commands:".to_string());
    lines.push("  /archives list".to_string());
    for entry in summary.entries.iter().take(3) {
        lines.push(format!("  /archives show {}", entry.entry.tool_use_id));
        lines.push(format!(
            "  /archives save {} [file]",
            entry.entry.tool_use_id
        ));
    }
    lines.push("- Archived entries:".to_string());
    for entry in summary.entries.iter().take(8) {
        lines.push(format!(
            "  - tool_call={} message={} tool={} status={} file={}",
            entry.entry.tool_use_id,
            entry.entry.message_id,
            entry.entry.tool_name,
            entry.status.label(),
            entry
                .entry
                .archived_output_path
                .as_deref()
                .unwrap_or("(none)")
        ));
    }
    if summary.entries.len() > 8 {
        lines.push(format!("  - ... and {} more", summary.entries.len() - 8));
    }

    Some(lines.join("\n"))
}

pub(crate) fn assistant_text_from_messages(messages: &[ConversationMessage]) -> String {
    messages
        .iter()
        .flat_map(|message| message.blocks.iter())
        .filter_map(|block| match block {
            ContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

fn default_export_filename(session: &Session) -> String {
    let stem = session
        .messages
        .iter()
        .find_map(|message| match message.role {
            MessageRole::User => message.blocks.iter().find_map(|block| match block {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            }),
            _ => None,
        })
        .map_or("conversation", |text| {
            text.lines().next().unwrap_or("conversation")
        })
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect::<String>()
        .split('-')
        .filter(|part| !part.is_empty())
        .take(8)
        .collect::<Vec<_>>()
        .join("-");
    let fallback = if stem.is_empty() {
        "conversation"
    } else {
        &stem
    };
    format!("{fallback}.txt")
}

fn resolve_export_path(
    requested_path: Option<&str>,
    session: &Session,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let cwd = env::current_dir()?;
    let file_name =
        requested_path.map_or_else(|| default_export_filename(session), ToOwned::to_owned);
    let final_name = if Path::new(&file_name)
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("txt"))
    {
        file_name
    } else {
        format!("{file_name}.txt")
    };
    Ok(cwd.join(final_name))
}

fn render_repl_help() -> String {
    format!(
        "{}\n\n  {:<18} {}\n  {:<18} {}\n  {:<18} {}\n  {:<18} {}\n  {:<18} {}\n  {:<18} {}\n\n{}\n  {}\n  {}\n  {}",
        report_title("Pebble help"),
        "/model",
        "switch model",
        "/mode build|plan",
        "switch mode",
        "/permissions",
        "tool access",
        "/status",
        "session details",
        "/diff",
        "workspace changes",
        "/undo /redo",
        "restore last turn",
        report_section("More help"),
        "/help commands  full reference",
        "/help sessions  session workflows",
        "/help auth      model login",
    )
}

fn repl_completion_candidates(cli: &LiveCli) -> Vec<String> {
    let mut candidates = BTreeSet::new();

    for candidate in command_names_and_aliases() {
        candidates.insert(candidate);
    }

    for topic in ["commands", "help", "auth", "sessions", "extensions", "web"] {
        candidates.insert(format!("/help {topic}"));
    }

    for service in [
        "nanogpt",
        "neuralwatt",
        "lilac",
        "grok",
        "synthetic",
        "openai-codex",
        "opencode-go",
        "exa",
    ] {
        candidates.insert(format!("/login {service}"));
        candidates.insert(format!("/auth {service}"));
        candidates.insert(format!("/logout {service}"));
    }

    for mode in ["read-only", "workspace-write", "danger-full-access"] {
        candidates.insert(format!("/permissions {mode}"));
    }
    candidates.insert("/bypass".to_string());

    for value in ["on", "off"] {
        candidates.insert(format!("/thinking {value}"));
        candidates.insert(format!("/proxy {value}"));
    }
    candidates.insert("/proxy status".to_string());

    candidates.insert("/mcp status".to_string());
    candidates.insert("/mcp tools".to_string());
    candidates.insert("/mcp reload".to_string());
    candidates.insert("/mcp add".to_string());
    candidates.insert("/mcp enable".to_string());
    candidates.insert("/mcp disable".to_string());

    candidates.insert("/branch list".to_string());
    candidates.insert("/branch create".to_string());
    candidates.insert("/branch switch".to_string());

    candidates.insert("/worktree list".to_string());
    candidates.insert("/worktree add".to_string());
    candidates.insert("/worktree remove".to_string());
    candidates.insert("/worktree prune".to_string());

    candidates.insert("/plugins help".to_string());
    candidates.insert("/plugins list".to_string());
    candidates.insert("/plugins install".to_string());
    candidates.insert("/plugins enable".to_string());
    candidates.insert("/plugins disable".to_string());
    candidates.insert("/plugins uninstall".to_string());
    candidates.insert("/plugins update".to_string());

    candidates.insert("/skills list".to_string());
    candidates.insert("/skills help".to_string());
    candidates.insert("/skills init".to_string());
    candidates.insert("/agents list".to_string());
    candidates.insert("/agents help".to_string());
    candidates.insert("/session list".to_string());
    candidates.insert("/session switch".to_string());
    candidates.insert("/session timeline".to_string());
    candidates.insert("/session fork".to_string());
    candidates.insert("/session rename".to_string());
    candidates.insert("/timeline".to_string());
    candidates.insert("/fork".to_string());
    candidates.insert("/rename".to_string());
    candidates.insert("/undo".to_string());
    candidates.insert("/redo".to_string());
    candidates.insert("/resume last".to_string());

    if let Ok(commands) =
        load_custom_slash_commands(&env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
    {
        for name in commands.keys() {
            candidates.insert(format!("/{name}"));
        }
    }

    for model in model_completion_candidates(&cli.model) {
        candidates.insert(format!("/model {model}"));
    }

    for provider in [
        "nanogpt",
        "neuralwatt",
        "lilac",
        "grok",
        "synthetic",
        "openai-codex",
        "opencode-go",
    ] {
        candidates.insert(format!("/provider {provider}"));
    }

    if cli.service == ApiService::NanoGpt {
        candidates.insert("/route default".to_string());
        if let Some(provider) = provider_for_model(&cli.model) {
            candidates.insert(format!("/route {provider}"));
        }
    }

    if let Ok(sessions) = list_managed_sessions() {
        for session in sessions {
            candidates.insert(format!("/resume {}", session.id));
            candidates.insert(format!("/resume {}", session.path.display()));
            candidates.insert(format!("/session switch {}", session.id));
        }
    }

    if let Ok(entries) = fs::read_dir(env::current_dir().unwrap_or_else(|_| PathBuf::from("."))) {
        for entry in entries.flatten().take(64) {
            let path = entry.path();
            let display = path.display().to_string();
            candidates.insert(format!("/export {display}"));
            candidates.insert(format!("/plugins install {display}"));
            if path.is_dir() {
                candidates.insert(format!("/worktree add {display}"));
                candidates.insert(format!("/worktree remove {display}"));
            }
        }
    }

    candidates.into_iter().collect()
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CustomSlashCommand {
    template: String,
    description: Option<String>,
}

fn parse_custom_slash_invocation(input: &str) -> Option<(&str, &str)> {
    let trimmed = input.trim();
    let body = trimmed.strip_prefix('/')?;
    let (name, arguments) = body
        .split_once(char::is_whitespace)
        .map_or((body, ""), |(name, rest)| (name, rest.trim()));
    (!name.is_empty()).then_some((name, arguments))
}

fn load_custom_slash_commands(cwd: &Path) -> io::Result<BTreeMap<String, CustomSlashCommand>> {
    let mut commands = BTreeMap::new();

    if let Ok(config) = ConfigLoader::default_for(cwd).load() {
        if let Some(command_map) = config.get("command").and_then(RuntimeJsonValue::as_object) {
            for (name, value) in command_map {
                if let Some(command) = custom_command_from_config(value) {
                    commands.insert(name.clone(), command);
                }
            }
        }
    }

    for root in custom_command_roots(cwd) {
        if !root.is_dir() {
            continue;
        }
        for entry in fs::read_dir(root)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_none_or(|ext| ext != "md") {
                continue;
            }
            let Some(name) = path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .map(ToOwned::to_owned)
            else {
                continue;
            };
            let template = fs::read_to_string(&path)?;
            commands.insert(
                name,
                CustomSlashCommand {
                    description: first_non_heading_line(&template),
                    template,
                },
            );
        }
    }

    Ok(commands)
}

fn custom_command_from_config(value: &RuntimeJsonValue) -> Option<CustomSlashCommand> {
    if let Some(template) = value.as_str() {
        return Some(CustomSlashCommand {
            template: template.to_string(),
            description: None,
        });
    }
    let object = value.as_object()?;
    let template = object.get("template").and_then(RuntimeJsonValue::as_str)?;
    Some(CustomSlashCommand {
        template: template.to_string(),
        description: object
            .get("description")
            .and_then(RuntimeJsonValue::as_str)
            .map(ToOwned::to_owned),
    })
}

fn custom_command_roots(cwd: &Path) -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Some(config_home) = resolve_pebble_config_home() {
        roots.push(config_home.join("commands"));
    }
    for ancestor in cwd.ancestors().collect::<Vec<_>>().into_iter().rev() {
        roots.push(ancestor.join(".pebble").join("commands"));
    }
    roots
}

fn render_custom_command_template(template: &str, arguments: &str) -> String {
    let args = split_command_arguments(arguments);
    let mut rendered = template.replace("$ARGUMENTS", arguments);
    let mut last_numbered = 0_usize;
    for token in numbered_placeholders(template) {
        last_numbered = last_numbered.max(token);
    }
    for token in numbered_placeholders(template) {
        let replacement = if token == last_numbered {
            args.get(token.saturating_sub(1)..)
                .map(|values| values.join(" "))
                .unwrap_or_default()
        } else {
            args.get(token.saturating_sub(1))
                .cloned()
                .unwrap_or_default()
        };
        rendered = rendered.replace(&format!("${token}"), &replacement);
    }
    if !template.contains("$ARGUMENTS") && last_numbered == 0 && !arguments.trim().is_empty() {
        rendered.push_str("\n\n");
        rendered.push_str(arguments);
    }
    rendered.trim().to_string()
}

fn numbered_placeholders(template: &str) -> BTreeSet<usize> {
    let mut values = BTreeSet::new();
    let chars = template.chars().collect::<Vec<_>>();
    let mut index = 0;
    while index + 1 < chars.len() {
        if chars[index] != '$' || !chars[index + 1].is_ascii_digit() {
            index += 1;
            continue;
        }
        let start = index + 1;
        let mut end = start;
        while end < chars.len() && chars[end].is_ascii_digit() {
            end += 1;
        }
        if let Ok(value) = chars[start..end]
            .iter()
            .collect::<String>()
            .parse::<usize>()
        {
            values.insert(value);
        }
        index = end;
    }
    values
}

fn split_command_arguments(arguments: &str) -> Vec<String> {
    let mut args = Vec::new();
    let mut current = String::new();
    let mut quote = None;
    let mut escaped = false;
    for ch in arguments.chars() {
        if escaped {
            current.push(ch);
            escaped = false;
            continue;
        }
        if ch == '\\' {
            escaped = true;
            continue;
        }
        if quote == Some(ch) {
            quote = None;
            continue;
        }
        if quote.is_none() && (ch == '"' || ch == '\'') {
            quote = Some(ch);
            continue;
        }
        if quote.is_none() && ch.is_whitespace() {
            if !current.is_empty() {
                args.push(std::mem::take(&mut current));
            }
            continue;
        }
        current.push(ch);
    }
    if !current.is_empty() {
        args.push(current);
    }
    args
}

fn first_non_heading_line(markdown: &str) -> Option<String> {
    markdown
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty() && !line.starts_with('#'))
        .map(ToOwned::to_owned)
}

fn model_completion_candidates(current_model: &str) -> Vec<String> {
    let mut candidates = BTreeSet::new();
    for alias in [
        "default",
        "glm",
        "glm5",
        "glm-5",
        "glm5.1",
        "glm-5.1",
        "zai-org/glm-5",
        "zai-org/glm-5.1",
    ] {
        candidates.insert(alias.to_string());
    }

    candidates.insert(DEFAULT_MODEL.to_string());
    candidates.insert(current_model.to_string());

    if let Ok(state) = load_model_state() {
        if let Some(model) = state.current_model {
            candidates.insert(model);
        }
        for favorite in state.favorite_models {
            candidates.insert(favorite);
        }
    }

    candidates.into_iter().collect()
}

fn run_repl(
    model: String,
    allowed_tools: Option<AllowedToolSet>,
    permission_mode: PermissionMode,
    collaboration_mode: CollaborationMode,
    reasoning_effort: Option<ReasoningEffort>,
    fast_mode: FastMode,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut cli = LiveCli::new(
        model,
        true,
        allowed_tools,
        permission_mode,
        collaboration_mode,
        reasoning_effort,
        fast_mode,
        true,
    )?;
    run_repl_loop(&mut cli)
}

fn run_repl_from_session(
    handle: SessionHandle,
    session: Session,
) -> Result<(), Box<dyn std::error::Error>> {
    let model = session.metadata.as_ref().map_or_else(
        || default_model_or(DEFAULT_MODEL),
        |metadata| metadata.model.clone(),
    );
    let mut cli = LiveCli::from_session(
        handle,
        session,
        model,
        None,
        default_permission_mode(),
        CollaborationMode::Build,
        None,
        FastMode::Off,
        true,
    )?;
    run_repl_loop(&mut cli)
}

fn run_repl_loop(cli: &mut LiveCli) -> Result<(), Box<dyn std::error::Error>> {
    let mut editor = input::LineEditor::new(
        ui::prompt_string(cli.collaboration_mode.as_str()),
        repl_completion_candidates(cli),
    );
    if let Some(config_home) = resolve_pebble_config_home() {
        editor = editor.with_history_path(config_home.join("repl-history"));
    }

    // Welcome banner: a single bordered panel that tells the user who they
    // are talking to, how the agent is configured, and which keystrokes to
    // remember. Printed once per REPL session.
    let cwd_display = env::current_dir()
        .ok()
        .map(|path| path.display().to_string());
    let provider_label = provider_label_for_service_model(cli.service, &cli.model);
    let auth_hint = startup_auth_hint(cli.service);
    let banner = if cli.runtime.session().messages.is_empty() {
        ui::welcome_banner(&ui::BannerInfo {
            version: VERSION,
            service: cli.service.display_name(),
            model: &cli.model,
            provider: provider_label.as_deref(),
            auth_hint: auth_hint.as_deref(),
            collaboration_mode: cli.collaboration_mode.as_str(),
            permission_mode: cli.permission_mode.as_str(),
            cwd: cwd_display.as_deref(),
        })
    } else {
        ui::resume_banner(&ui::ResumeBannerInfo {
            session_id: &cli.session.id,
            model: &cli.model,
            collaboration_mode: cli.collaboration_mode.as_str(),
            permission_mode: cli.permission_mode.as_str(),
        })
    };
    println!("{banner}");
    println!();

    loop {
        editor.set_completions(repl_completion_candidates(cli));
        editor.set_prompt(ui::prompt_string(cli.collaboration_mode.as_str()));
        let input = match editor.read_line()? {
            input::ReadOutcome::Submit(input) => input,
            input::ReadOutcome::Cancel => continue,
            input::ReadOutcome::Exit => break,
            input::ReadOutcome::ToggleMode => {
                cli.toggle_mode()?;
                continue;
            }
        };
        let trimmed = input.trim();
        if trimmed.is_empty() {
            continue;
        }
        editor.push_history(trimmed.to_string());
        let command_result = (|| -> Result<bool, Box<dyn std::error::Error>> {
            if let Some(login_command) = parse_auth_command(trimmed) {
                login(login_command.service, login_command.api_key)?;
                return Ok(false);
            }
            if let Some(logout_command) = parse_logout_command(trimmed) {
                logout(logout_command.service)?;
                return Ok(false);
            }
            if let Some(model) = parse_model_command(trimmed) {
                match model {
                    Some(model) => cli.set_model(model)?,
                    None => {
                        if let Some(model) = open_model_picker()?.selected_model {
                            cli.set_model(model)?;
                        }
                    }
                }
                return Ok(false);
            }
            if let Some(provider) = parse_provider_command(trimmed) {
                let service = provider
                    .as_deref()
                    .map(|provider| {
                        service_from_selector(provider)
                            .ok_or_else(|| unknown_provider_message(provider))
                    })
                    .transpose()?;
                if let Some(model) = open_model_picker_for_service(service)?.selected_model {
                    cli.set_model(model)?;
                }
                return Ok(false);
            }
            if let Some(route) = parse_route_command(trimmed) {
                match route {
                    Some(route) if is_clear_provider_arg(&route) => cli.set_route(None)?,
                    Some(route) => cli.set_route(Some(route))?,
                    None => match open_provider_picker(&cli.model)?.selected_provider {
                        Some(route) => cli.set_route(Some(route))?,
                        None => cli.set_route(None)?,
                    },
                }
                return Ok(false);
            }
            if let Some(mode) = parse_proxy_command(trimmed) {
                handle_proxy_runtime_command(cli, mode?)?;
                return Ok(false);
            }
            if let Some(reasoning_effort) = parse_reasoning_command(trimmed) {
                cli.set_reasoning(reasoning_effort?)?;
                return Ok(false);
            }
            if let Some(mode) = parse_mode_command(trimmed) {
                cli.set_mode(mode?)?;
                return Ok(false);
            }
            if let Some(fast_mode) = parse_fast_command(trimmed) {
                cli.set_fast_mode(fast_mode?)?;
                return Ok(false);
            }
            if let Some(command) = parse_mcp_command(trimmed) {
                handle_mcp_runtime_command(cli, command?)?;
                return Ok(false);
            }
            if let Some(mode) = parse_permissions_command(trimmed) {
                cli.set_permissions(mode?)?;
                return Ok(false);
            }
            match trimmed {
                "/exit" | "/quit" => return Ok(true),
                _ if trimmed.starts_with('/') => {
                    let Some(command) = SlashCommand::parse(trimmed) else {
                        return Ok(false);
                    };
                    match command {
                        SlashCommand::Help { topic } => println!(
                            "{}",
                            match topic.as_deref() {
                                None => render_repl_help(),
                                Some("commands" | "all") => render_slash_command_help(),
                                Some(topic) => render_slash_command_help_topic(Some(topic)),
                            }
                        ),
                        SlashCommand::Status => cli.print_status(),
                        SlashCommand::Compact => cli.compact()?,
                        SlashCommand::Archives { action, target } => println!(
                            "{}",
                            render_archived_tool_results_report(
                                cli.runtime.session(),
                                Some(&cli.session.path),
                                action.as_deref(),
                                target.as_deref(),
                            )?
                        ),
                        SlashCommand::Undo => cli.undo_turn()?,
                        SlashCommand::Redo => cli.redo_turn()?,
                        SlashCommand::Timeline => println!("{}", cli.render_timeline()),
                        SlashCommand::Fork { target } => cli.fork_session(target.as_deref())?,
                        SlashCommand::Rename { title } => cli.rename_session(title.as_deref())?,
                        SlashCommand::Reasoning { effort } => {
                            cli.set_reasoning(match effort.as_deref() {
                                Some(value) => Some(parse_reasoning_effort_arg(value)?),
                                None => None,
                            })?;
                        }
                        SlashCommand::Fast { enabled } => {
                            cli.set_fast_mode(enabled.map(|enabled| {
                                if enabled {
                                    FastMode::On
                                } else {
                                    FastMode::Off
                                }
                            }))?;
                        }
                        SlashCommand::Mode { mode } => cli.set_mode(
                            mode.as_deref()
                                .map(parse_collaboration_mode_arg)
                                .transpose()?,
                        )?,
                        SlashCommand::Permissions { mode } => cli.set_permissions(
                            mode.as_deref().map(parse_permission_mode_arg).transpose()?,
                        )?,
                        SlashCommand::Clear { confirm } => cli.clear_session(confirm)?,
                        SlashCommand::Resume { session_path } => {
                            cli.resume_session(session_path)?;
                        }
                        SlashCommand::Config { section } => {
                            println!("{}", render_config_report(section.as_deref())?);
                        }
                        SlashCommand::Memory => println!("{}", render_memory_report()?),
                        SlashCommand::Init => run_init_with_model(cli.service, &cli.model)?,
                        SlashCommand::Diff => println!("{}", render_diff_report()?),
                        SlashCommand::Patch { args } => cli.run_patch_command(args.as_deref())?,
                        SlashCommand::Version => print_version(),
                        SlashCommand::Branch { action, target } => println!(
                            "{}",
                            handle_branch_slash_command(
                                action.as_deref(),
                                target.as_deref(),
                                &env::current_dir()?,
                            )?
                        ),
                        SlashCommand::Worktree {
                            action,
                            path,
                            branch,
                        } => println!(
                            "{}",
                            handle_worktree_slash_command(
                                action.as_deref(),
                                path.as_deref(),
                                branch.as_deref(),
                                &env::current_dir()?,
                            )?
                        ),
                        SlashCommand::Export { path } => cli.export_session(path.as_deref())?,
                        SlashCommand::Session { action, target } => {
                            cli.handle_session_command(action.as_deref(), target.as_deref())?;
                        }
                        SlashCommand::Sessions => {
                            println!("{}", render_session_list(&cli.session.id)?);
                        }
                        SlashCommand::Plugins { action, target } => {
                            cli.handle_plugins_command(action.as_deref(), target.as_deref())?;
                        }
                        SlashCommand::Agents { args } => println!(
                            "{}",
                            handle_agents_slash_command(args.as_deref(), &env::current_dir()?)?
                        ),
                        SlashCommand::Skills { args } => println!(
                            "{}",
                            handle_skills_slash_command(args.as_deref(), &env::current_dir()?)?
                        ),
                        SlashCommand::Unknown(name) => {
                            if !cli.run_custom_slash_command(trimmed)? {
                                eprintln!(
                                    "{}",
                                    ui::error_note(&format!(
                                        "Unknown command /{name}. Type /help for the essentials."
                                    ))
                                );
                            }
                        }
                        SlashCommand::Model { .. }
                        | SlashCommand::Provider { .. }
                        | SlashCommand::Route { .. }
                        | SlashCommand::Logout { .. }
                        | SlashCommand::Mcp { .. } => {
                            unreachable!("handled before shared slash command dispatch")
                        }
                    }
                }
                _ => {
                    if let Err(error) = cli.run_turn(trimmed) {
                        if error
                            .downcast_ref::<RuntimeError>()
                            .is_some_and(RuntimeError::is_cancelled)
                        {
                            println!();
                            println!("{}", ui::dim_note("Cancelled."));
                        } else {
                            eprintln!("{}", ui::error_note(&format!("Request failed: {error}")));
                        }
                    }
                }
            }
            Ok(false)
        })();
        match command_result {
            Ok(true) => break,
            Ok(false) => {}
            Err(error) => eprintln!("{}", ui::error_note(&error.to_string())),
        }
    }

    println!(
        "{}",
        ui::dim_note(&format!(
            "Saved {}.\n  Use /resume to return.",
            cli.session.id
        ))
    );
    Ok(())
}

pub(crate) struct LiveCli {
    service: ApiService,
    model: String,
    allowed_tools: Option<AllowedToolSet>,
    permission_mode: PermissionMode,
    collaboration_mode: CollaborationMode,
    reasoning_effort: Option<ReasoningEffort>,
    fast_mode: FastMode,
    system_prompt: Vec<String>,
    proxy_tool_calls: bool,
    context_window_tokens: Option<u64>,
    mcp_catalog: McpCatalog,
    runtime: ConversationRuntime<PebbleRuntimeClient, CliToolExecutor>,
    session: SessionHandle,
    render_model_output: bool,
}

impl LiveCli {
    pub(crate) fn new(
        model: String,
        enable_tools: bool,
        allowed_tools: Option<AllowedToolSet>,
        permission_mode: PermissionMode,
        collaboration_mode: CollaborationMode,
        reasoning_effort: Option<ReasoningEffort>,
        fast_mode: FastMode,
        render_model_output: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let service = infer_service_for_model(&model);
        let context_window_tokens = context_length_for_model(&model);
        let system_prompt = build_system_prompt(service, &model, collaboration_mode)?;
        let proxy_tool_calls = proxy_tool_calls_enabled();
        let mcp_catalog = load_mcp_catalog(&env::current_dir()?)?;
        let session = create_managed_session_handle()?;
        auto_compact_inactive_sessions(&session.id)?;
        let runtime = build_runtime(
            Session::new(),
            service,
            model.clone(),
            system_prompt.clone(),
            enable_tools,
            proxy_tool_calls,
            mcp_catalog.clone(),
            allowed_tools.clone(),
            permission_mode,
            collaboration_mode,
            reasoning_effort,
            fast_mode,
            render_model_output,
        )?;
        let cli = Self {
            service,
            model,
            allowed_tools,
            permission_mode,
            collaboration_mode,
            reasoning_effort,
            fast_mode,
            system_prompt,
            proxy_tool_calls,
            context_window_tokens,
            mcp_catalog,
            runtime,
            session,
            render_model_output,
        };
        cli.persist_session()?;
        Ok(cli)
    }

    fn from_session(
        session_handle: SessionHandle,
        session: Session,
        model: String,
        allowed_tools: Option<AllowedToolSet>,
        permission_mode: PermissionMode,
        collaboration_mode: CollaborationMode,
        reasoning_effort: Option<ReasoningEffort>,
        fast_mode: FastMode,
        render_model_output: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let restored = session_runtime_state(
            &session,
            &model,
            allowed_tools.as_ref(),
            permission_mode,
            collaboration_mode,
            reasoning_effort,
            fast_mode,
            proxy_tool_calls_enabled(),
        );
        let system_prompt = build_system_prompt(
            restored.service,
            &restored.model,
            restored.collaboration_mode,
        )?;
        let context_window_tokens = context_length_for_model(&restored.model);
        let mcp_catalog = load_mcp_catalog(&env::current_dir()?)?;
        let runtime = build_runtime(
            session,
            restored.service,
            restored.model.clone(),
            system_prompt.clone(),
            true,
            restored.proxy_tool_calls,
            mcp_catalog.clone(),
            restored.allowed_tools.clone(),
            restored.permission_mode,
            restored.collaboration_mode,
            restored.reasoning_effort,
            restored.fast_mode,
            render_model_output,
        )?;
        Ok(Self {
            service: restored.service,
            model: restored.model,
            allowed_tools: restored.allowed_tools,
            permission_mode: restored.permission_mode,
            collaboration_mode: restored.collaboration_mode,
            reasoning_effort: restored.reasoning_effort,
            fast_mode: restored.fast_mode,
            system_prompt,
            proxy_tool_calls: restored.proxy_tool_calls,
            context_window_tokens,
            mcp_catalog,
            runtime,
            session: session_handle,
            render_model_output,
        })
    }

    fn run_turn_with_output(
        &mut self,
        input: &str,
        output_format: CliOutputFormat,
    ) -> Result<(), Box<dyn std::error::Error>> {
        match output_format {
            CliOutputFormat::Text => self.run_turn(input),
            CliOutputFormat::Json => self.run_prompt_json(input),
        }
    }

    fn run_runtime_turn(
        &mut self,
        input: &str,
        prompter: &mut dyn PermissionPrompter,
    ) -> Result<TurnSummary, RuntimeError> {
        let cancellation = CancellationToken::new();
        let _interrupt = InterruptGuard::install(&cancellation)?;
        self.runtime
            .run_turn_cancellable(input, Some(prompter), &cancellation)
    }

    fn run_turn(&mut self, input: &str) -> Result<(), Box<dyn std::error::Error>> {
        let cwd = env::current_dir()?;
        let before_message_count = self.runtime.session().messages.len();
        let before_session = self.runtime.session().clone();
        let before_files = WorktreeSnapshot::capture(&cwd);
        println!();
        println!(
            "{}",
            ui::activity_note(if self.collaboration_mode == CollaborationMode::Plan {
                "Planning..."
            } else {
                "Working..."
            })
        );
        let started_at = Instant::now();
        let mut permission_prompter = CliPermissionPrompter::new(self.permission_mode);
        let result = self.run_runtime_turn(input, &mut permission_prompter);
        match result {
            Ok(summary) => {
                let mut session = self.runtime.session().clone();
                let snapshot =
                    build_turn_snapshot(&cwd, &before_files, before_message_count, &session);
                let changed_files = snapshot.as_ref().map_or(0, |snapshot| snapshot.files.len());
                if let Some(snapshot) = snapshot {
                    append_undo_snapshot(&mut session, snapshot);
                    self.runtime.replace_session(session);
                }
                self.persist_session()?;
                let trace_path = persist_turn_trace(&cwd, &self.session.id, &summary.trace)?;
                if let Some(event) = summary.auto_compaction {
                    println!("{}", ui::dim_note(&format_auto_compaction_notice(event)));
                }
                println!(
                    "{}",
                    ui::turn_summary(&ui::TurnSummaryInfo {
                        elapsed: started_at.elapsed(),
                        iterations: summary.iterations,
                        tool_calls: collect_tool_uses(&summary).len(),
                        changed_files,
                        usage: summary.usage,
                        context_window: self.context_window_usage(
                            u64::try_from(self.runtime.estimated_tokens()).unwrap_or(u64::MAX),
                        ),
                    })
                );
                if env_trace_notice_enabled() {
                    println!(
                        "{}",
                        ui::dim_note(&format!("trace: {}", trace_path.display()))
                    );
                }
                println!();
                Ok(())
            }
            Err(error) => {
                if self.runtime.session().messages.len() <= before_message_count + 1 {
                    self.runtime.replace_session(before_session);
                    self.persist_session()?;
                    return Err(Box::new(error));
                }

                let mut session = self.runtime.session().clone();
                if let Some(snapshot) =
                    build_turn_snapshot(&cwd, &before_files, before_message_count, &session)
                {
                    append_undo_snapshot(&mut session, snapshot);
                    self.runtime.replace_session(session);
                }
                self.persist_session()?;
                if error.is_cancelled() {
                    return Err(Box::new(error));
                }
                Err(
                    format!("{error}\n  Partial turn saved. Use /undo to revert its file changes.")
                        .into(),
                )
            }
        }
    }

    fn run_prompt_json(&mut self, input: &str) -> Result<(), Box<dyn std::error::Error>> {
        let cwd = env::current_dir()?;
        let before_message_count = self.runtime.session().messages.len();
        let before_session = self.runtime.session().clone();
        let before_files = WorktreeSnapshot::capture(&cwd);
        let mut permission_prompter = CliPermissionPrompter::new(self.permission_mode);
        let summary = match self.run_runtime_turn(input, &mut permission_prompter) {
            Ok(summary) => summary,
            Err(error) => {
                if self.runtime.session().messages.len() <= before_message_count + 1 {
                    self.runtime.replace_session(before_session);
                    self.persist_session()?;
                } else {
                    let mut session = self.runtime.session().clone();
                    if let Some(snapshot) =
                        build_turn_snapshot(&cwd, &before_files, before_message_count, &session)
                    {
                        append_undo_snapshot(&mut session, snapshot);
                        self.runtime.replace_session(session);
                    }
                    self.persist_session()?;
                }
                return Err(Box::new(error));
            }
        };
        let mut session = self.runtime.session().clone();
        if let Some(snapshot) =
            build_turn_snapshot(&cwd, &before_files, before_message_count, &session)
        {
            append_undo_snapshot(&mut session, snapshot);
            self.runtime.replace_session(session);
        }
        self.persist_session()?;
        let trace_path = persist_turn_trace(&cwd, &self.session.id, &summary.trace)?;
        let context_window = self.context_window_usage(
            u64::try_from(self.runtime.estimated_tokens()).unwrap_or(u64::MAX),
        );
        println!(
            "{}",
            serde_json::json!({
                "message": assistant_text_from_messages(&summary.assistant_messages),
                "model": self.model,
                "iterations": summary.iterations,
                "auto_compaction": summary.auto_compaction.map(|event| serde_json::json!({
                    "removed_messages": event.removed_message_count,
                    "pruned_tool_results": event.pruned_tool_result_count,
                    "notice": format_auto_compaction_notice(event),
                })),
                "tool_uses": collect_tool_uses(&summary),
                "tool_results": collect_tool_results(&summary),
                "usage": {
                    "input_tokens": summary.usage.input_tokens,
                    "output_tokens": summary.usage.output_tokens,
                    "cache_creation_input_tokens": summary.usage.cache_creation_input_tokens,
                    "cache_read_input_tokens": summary.usage.cache_read_input_tokens,
                },
                "context_window": context_window.map(|info| serde_json::json!({
                    "used_tokens": info.used_tokens,
                    "max_tokens": info.max_tokens,
                    "percent": if info.max_tokens == 0 {
                        0.0
                    } else {
                        (info.used_tokens as f64 / info.max_tokens as f64) * 100.0
                    },
                    "display": ui::format_context_window_usage(info),
                })),
                "trace": summary.trace,
                "trace_file": trace_path,
            })
        );
        Ok(())
    }

    fn run_patch_command(&mut self, args: Option<&str>) -> Result<(), Box<dyn std::error::Error>> {
        let (dry_run, patch, source) = load_patch_command(args)?;
        let cwd = env::current_dir()?;
        let before_message_count = self.runtime.session().messages.len();
        let before_files = (!dry_run).then(|| WorktreeSnapshot::capture(&cwd));
        let output = apply_patch(&patch, dry_run)?;
        if let Some(before_files) = before_files {
            let mut session = self.runtime.session().clone();
            let output_json = serde_json::to_string(&output)?;
            let synthetic_messages = vec![ConversationMessage::tool_result(
                "patch-command",
                "apply_patch",
                output_json,
                false,
            )];
            let mut snapshot = SessionTurnSnapshot {
                timestamp: current_timestamp_rfc3339ish(),
                message_count_before: before_message_count.try_into().unwrap_or(u32::MAX),
                prompt: Some(format!("/patch {source}")),
                messages: Vec::new(),
                files: file_changes_from_turn_messages(&cwd, &synthetic_messages)
                    .into_values()
                    .collect(),
            };
            if snapshot.files.is_empty() {
                if let Some(git_snapshot) =
                    build_turn_snapshot(&cwd, &before_files, before_message_count, &session)
                {
                    snapshot = git_snapshot;
                }
            }
            if !snapshot.files.is_empty() {
                append_undo_snapshot(&mut session, snapshot);
                self.runtime.replace_session(session);
                self.persist_session()?;
            }
        }
        println!("{}", format_patch_report(&output, &source));
        Ok(())
    }

    fn print_status(&self) {
        let cumulative = self.runtime.usage().cumulative_usage();
        let latest = self.runtime.usage().current_turn_usage();
        let context = status_context(Some(&self.session.path)).expect("status context should load");
        let (undo_count, redo_count) = session_undo_redo_counts(self.runtime.session());
        let estimated_tokens = self.runtime.estimated_tokens();
        println!(
            "{}",
            format_status_report(
                self.service,
                &self.model,
                StatusUsage {
                    message_count: self.runtime.session().messages.len(),
                    turns: self.runtime.usage().turns(),
                    undo_count,
                    redo_count,
                    latest,
                    cumulative,
                    estimated_tokens,
                    context_window: self
                        .context_window_usage(u64::try_from(estimated_tokens).unwrap_or(u64::MAX),),
                },
                self.permission_mode.as_str(),
                provider_label_for_service_model(self.service, &self.model).as_deref(),
                self.proxy_tool_calls,
                self.collaboration_mode,
                self.effective_reasoning_effort(),
                self.fast_mode,
                &self.mcp_catalog,
                &context,
            )
        );
    }

    fn context_window_usage(&self, used_tokens: u64) -> Option<ui::ContextWindowInfo> {
        self.context_window_tokens
            .filter(|max_tokens| *max_tokens > 0)
            .map(|max_tokens| ui::ContextWindowInfo {
                used_tokens,
                max_tokens,
            })
    }

    fn compact(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let result = self.runtime.compact(CompactionConfig::default());
        let removed = result.removed_message_count;
        self.rebuild_runtime(result.compacted_session)?;
        self.persist_session()?;
        println!("{}", format_compact_report(removed));
        Ok(())
    }

    fn set_model(&mut self, model: String) -> Result<(), Box<dyn std::error::Error>> {
        let model = resolve_model_alias(&model).to_string();
        persist_current_model(model.clone())?;
        self.service = infer_service_for_model(&model);
        self.model = model.clone();
        self.rebuild_runtime(self.runtime.session().clone())?;
        self.persist_session()?;
        let service = self.service.display_name().to_string();
        let mut fields = vec![("service", service.as_str()), ("model", model.as_str())];
        if let Some(provider) = provider_label_for_service_model(self.service, &self.model) {
            fields.push(("route", provider.as_str()));
            println!("{}", ui::setting_changed("model", &fields));
        } else {
            println!("{}", ui::setting_changed("model", &fields));
        }
        Ok(())
    }

    fn set_route(&mut self, provider: Option<String>) -> Result<(), Box<dyn std::error::Error>> {
        if self.service != ApiService::NanoGpt {
            return Err(format!(
                "routing overrides are only supported for NanoGPT models; current model {} is on {}. Use `/provider` to switch model providers",
                self.model,
                self.service.display_name()
            )
            .into());
        }
        let session = self.runtime.session().clone();
        if let Some(provider) = provider.as_deref() {
            validate_provider_for_model(&self.model, provider)?;
        }
        let provider_label = provider
            .clone()
            .unwrap_or_else(|| "<platform default>".to_string());
        persist_provider_for_model(&self.model, provider)?;
        self.rebuild_runtime(session)?;
        self.persist_session()?;
        let routing = if provider_label == "<platform default>" {
            "platform default"
        } else {
            "paygo routing enabled"
        };
        println!(
            "{}",
            ui::setting_changed(
                "NanoGPT route",
                &[
                    ("model", self.model.as_str()),
                    ("route", provider_label.as_str()),
                    ("routing", routing),
                ],
            )
        );
        Ok(())
    }

    fn set_proxy_tool_calls(&mut self, enabled: bool) -> Result<(), Box<dyn std::error::Error>> {
        let session = self.runtime.session().clone();
        persist_proxy_tool_calls(enabled)?;
        self.proxy_tool_calls = enabled;
        self.rebuild_runtime(session)?;
        self.persist_session()?;
        println!(
            "{}",
            ui::setting_changed(
                "proxy tool calls",
                &[("state", if enabled { "enabled" } else { "disabled" })],
            )
        );
        if enabled {
            println!(
                "{}",
                ui::dim_note("native tool schemas disabled; XML <tool_call> blocks enabled")
            );
        }
        Ok(())
    }

    fn reload_mcp(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let session = self.runtime.session().clone();
        self.mcp_catalog = load_mcp_catalog(&env::current_dir()?)?;
        self.rebuild_runtime(session)?;
        self.persist_session()?;
        println!("{}", ui::success_note("reloaded MCP config"));
        self.print_mcp_status();
        Ok(())
    }

    fn print_mcp_status(&self) {
        print_mcp_status(&self.mcp_catalog);
    }

    fn print_mcp_tools(&self) {
        print_mcp_tools(&self.mcp_catalog);
    }

    fn set_permissions(
        &mut self,
        mode: Option<PermissionMode>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let Some(mode) = mode else {
            println!(
                "{}",
                ui::setting_changed("permissions", &[("mode", self.permission_mode.as_str())])
            );
            return Ok(());
        };
        if mode == self.permission_mode {
            self.persist_runtime_defaults()?;
            println!(
                "{}",
                ui::setting_changed("permissions", &[("mode", self.permission_mode.as_str())])
            );
            return Ok(());
        }

        self.permission_mode = mode;
        self.rebuild_runtime(self.runtime.session().clone())?;
        self.persist_runtime_defaults()?;
        self.persist_session()?;
        println!(
            "{}",
            ui::setting_changed("permissions", &[("mode", self.permission_mode.as_str())])
        );
        Ok(())
    }

    fn toggle_mode(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.set_mode(Some(match self.collaboration_mode {
            CollaborationMode::Build => CollaborationMode::Plan,
            CollaborationMode::Plan => CollaborationMode::Build,
        }))
    }

    fn set_mode(
        &mut self,
        mode: Option<CollaborationMode>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let Some(mode) = mode else {
            println!("{}", format_mode_report(self.collaboration_mode));
            return Ok(());
        };
        if mode == self.collaboration_mode {
            self.persist_runtime_defaults()?;
            println!("{}", format_mode_report(self.collaboration_mode));
            return Ok(());
        }

        self.collaboration_mode = mode;
        self.rebuild_runtime(self.runtime.session().clone())?;
        self.persist_runtime_defaults()?;
        self.persist_session()?;
        println!("{}", format_mode_switch_report(self.collaboration_mode));
        Ok(())
    }

    fn set_reasoning(
        &mut self,
        reasoning_effort: Option<Option<ReasoningEffort>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let Some(reasoning_effort) = reasoning_effort else {
            println!(
                "{}",
                format_reasoning_report(self.reasoning_effort, self.effective_reasoning_effort())
            );
            return Ok(());
        };
        if reasoning_effort == self.reasoning_effort {
            self.persist_runtime_defaults()?;
            println!(
                "{}",
                format_reasoning_report(self.reasoning_effort, self.effective_reasoning_effort())
            );
            return Ok(());
        }

        self.reasoning_effort = reasoning_effort;
        self.rebuild_runtime(self.runtime.session().clone())?;
        self.persist_runtime_defaults()?;
        self.persist_session()?;
        println!(
            "{}",
            format_reasoning_switch_report(
                self.reasoning_effort,
                self.effective_reasoning_effort()
            )
        );
        Ok(())
    }

    fn set_fast_mode(
        &mut self,
        fast_mode: Option<FastMode>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let Some(fast_mode) = fast_mode else {
            println!("{}", format_fast_mode_report(self.fast_mode));
            return Ok(());
        };
        if fast_mode == self.fast_mode {
            self.persist_runtime_defaults()?;
            println!("{}", format_fast_mode_report(self.fast_mode));
            return Ok(());
        }

        self.fast_mode = fast_mode;
        self.rebuild_runtime(self.runtime.session().clone())?;
        self.persist_runtime_defaults()?;
        self.persist_session()?;
        println!("{}", format_fast_mode_switch_report(self.fast_mode));
        Ok(())
    }

    fn effective_reasoning_effort(&self) -> Option<ReasoningEffort> {
        effective_reasoning_effort(self.collaboration_mode, self.reasoning_effort)
    }

    fn persist_runtime_defaults(&self) -> Result<(), Box<dyn std::error::Error>> {
        persist_runtime_defaults(
            self.permission_mode,
            self.collaboration_mode,
            self.reasoning_effort,
            self.fast_mode,
        )
    }

    fn rebuild_runtime(&mut self, session: Session) -> Result<(), Box<dyn std::error::Error>> {
        self.system_prompt =
            build_system_prompt(self.service, &self.model, self.collaboration_mode)?;
        self.runtime = build_runtime(
            session,
            self.service,
            self.model.clone(),
            self.system_prompt.clone(),
            true,
            self.proxy_tool_calls,
            self.mcp_catalog.clone(),
            self.allowed_tools.clone(),
            self.permission_mode,
            self.collaboration_mode,
            self.reasoning_effort,
            self.fast_mode,
            self.render_model_output,
        )?;
        Ok(())
    }

    fn handle_plugins_command(
        &mut self,
        action: Option<&str>,
        target: Option<&str>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        println!("{}", handle_plugins_command(action, target)?);
        if plugins_command_is_mutating(action) {
            self.rebuild_runtime(self.runtime.session().clone())?;
            self.persist_session()?;
        }
        Ok(())
    }

    fn clear_session(&mut self, confirm: bool) -> Result<(), Box<dyn std::error::Error>> {
        if !confirm {
            println!(
                "{}",
                ui::setting_changed(
                    "clear session",
                    &[
                        ("result", "confirmation required"),
                        ("run", "/clear --confirm"),
                    ],
                )
            );
            return Ok(());
        }

        self.session = create_managed_session_handle()?;
        self.rebuild_runtime(Session::new())?;
        self.persist_session()?;
        println!(
            "{}",
            format_clear_report(
                &self.model,
                self.collaboration_mode,
                self.permission_mode,
                &self.session.id,
            )
        );
        Ok(())
    }

    fn resume_session(
        &mut self,
        session_path: Option<String>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let session_ref = match session_path {
            Some(session_ref) => session_ref,
            None => match prompt_for_session_selection(Some(&self.session.id))? {
                Some(handle) => {
                    return self.resume_handle(handle);
                }
                None => return Ok(()),
            },
        };
        let handle = resolve_session_reference(&session_ref)?;
        self.resume_handle(handle)
    }

    fn resume_handle(&mut self, handle: SessionHandle) -> Result<(), Box<dyn std::error::Error>> {
        let session = Session::load_from_path(&handle.path)?;
        let message_count = session.messages.len();
        self.restore_session_runtime(handle.clone(), session)?;
        self.session = handle;
        self.persist_session()?;
        println!(
            "{}",
            format_resume_report(
                &self.session.path.display().to_string(),
                message_count,
                self.runtime.usage().turns(),
            )
        );
        Ok(())
    }

    fn export_session(
        &self,
        requested_path: Option<&str>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let export_path = resolve_export_path(requested_path, self.runtime.session())?;
        write_atomic(
            &export_path,
            render_export_text(self.runtime.session(), Some(&self.session.path)),
        )?;
        println!(
            "{}",
            format_export_report(
                &export_path.display().to_string(),
                self.runtime.session().messages.len()
            )
        );
        Ok(())
    }

    fn undo_turn(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let mut session = self.runtime.session().clone();
        let Some(snapshot) = pop_undo_snapshot(&mut session) else {
            println!(
                "{}",
                ui::setting_changed("undo", &[("result", "nothing to undo")])
            );
            return Ok(());
        };
        restore_snapshot_files(&env::current_dir()?, &snapshot, SnapshotDirection::Before)?;
        session
            .messages
            .truncate(snapshot.message_count_before as usize);
        push_redo_snapshot(&mut session, snapshot.clone());
        self.runtime.replace_session(session);
        self.persist_session()?;
        println!(
            "{}",
            ui::setting_changed(
                "undo",
                &[
                    ("result", "restored previous turn"),
                    ("messages removed", &snapshot.messages.len().to_string()),
                    ("files restored", &snapshot.files.len().to_string()),
                ],
            )
        );
        Ok(())
    }

    fn redo_turn(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let mut session = self.runtime.session().clone();
        let Some(snapshot) = pop_redo_snapshot(&mut session) else {
            println!(
                "{}",
                ui::setting_changed("redo", &[("result", "nothing to redo")])
            );
            return Ok(());
        };
        restore_snapshot_files(&env::current_dir()?, &snapshot, SnapshotDirection::After)?;
        session
            .messages
            .truncate(snapshot.message_count_before as usize);
        session.messages.extend(snapshot.messages.clone());
        push_undo_snapshot(&mut session, snapshot.clone(), false);
        self.runtime.replace_session(session);
        self.persist_session()?;
        println!(
            "{}",
            ui::setting_changed(
                "redo",
                &[
                    ("result", "replayed turn"),
                    ("messages restored", &snapshot.messages.len().to_string()),
                    ("files restored", &snapshot.files.len().to_string()),
                ],
            )
        );
        Ok(())
    }

    fn render_timeline(&self) -> String {
        render_session_timeline(self.runtime.session())
    }

    fn fork_session(&mut self, target: Option<&str>) -> Result<(), Box<dyn std::error::Error>> {
        let mut forked = self.runtime.session().clone();
        if let Some(target) = target.filter(|value| !value.trim().is_empty()) {
            let index = resolve_timeline_target(&forked, target)
                .ok_or_else(|| format!("unknown timeline target: {target}"))?;
            forked.messages.truncate(index + 1);
        }
        let source_id = self.session.id.clone();
        let fork_title = forked.metadata.as_ref().and_then(|metadata| {
            metadata
                .title
                .as_ref()
                .map(|title| format!("Fork of {title}"))
        });
        if let Some(metadata) = &mut forked.metadata {
            metadata.undo_stack = None;
            metadata.redo_stack = None;
        }
        let handle = create_managed_session_handle()?;
        forked.metadata = Some(derive_session_metadata(
            &forked,
            &self.model,
            self.allowed_tools.as_ref(),
            self.permission_mode,
            self.collaboration_mode,
            self.reasoning_effort,
            self.fast_mode,
            self.proxy_tool_calls,
        ));
        if let Some(metadata) = &mut forked.metadata {
            metadata.title = Some(fork_title.unwrap_or_else(|| format!("Fork of {source_id}")));
            metadata.undo_stack = None;
            metadata.redo_stack = None;
        }
        forked.save_to_path(&handle.path)?;
        self.restore_session_runtime(handle.clone(), forked)?;
        self.persist_session()?;
        println!(
            "{}",
            ui::setting_changed(
                "session forked",
                &[
                    ("source", source_id.as_str()),
                    ("active", handle.id.as_str()),
                    ("file", &handle.path.display().to_string()),
                ],
            )
        );
        Ok(())
    }

    fn rename_session(&mut self, title: Option<&str>) -> Result<(), Box<dyn std::error::Error>> {
        let Some(title) = title.map(str::trim).filter(|value| !value.is_empty()) else {
            println!("Usage: /rename <title>");
            return Ok(());
        };
        let mut session = self.runtime.session().clone();
        if session.metadata.is_none() {
            session.metadata = Some(derive_session_metadata(
                &session,
                &self.model,
                self.allowed_tools.as_ref(),
                self.permission_mode,
                self.collaboration_mode,
                self.reasoning_effort,
                self.fast_mode,
                self.proxy_tool_calls,
            ));
        }
        if let Some(metadata) = &mut session.metadata {
            metadata.title = Some(title.to_string());
        }
        self.runtime.replace_session(session);
        self.persist_session()?;
        println!(
            "{}",
            ui::setting_changed(
                "session renamed",
                &[("active", self.session.id.as_str()), ("title", title)],
            )
        );
        Ok(())
    }

    fn run_custom_slash_command(
        &mut self,
        input: &str,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        let Some((name, arguments)) = parse_custom_slash_invocation(input) else {
            return Ok(false);
        };
        let commands = load_custom_slash_commands(&env::current_dir()?)?;
        let Some(command) = commands.get(name) else {
            return Ok(false);
        };
        let prompt = render_custom_command_template(&command.template, arguments);
        if prompt.trim().is_empty() {
            println!("Command /{name} expanded to an empty prompt.");
            return Ok(true);
        }
        self.run_turn(&prompt)?;
        Ok(true)
    }

    fn handle_session_command(
        &mut self,
        action: Option<&str>,
        target: Option<&str>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        match action {
            None | Some("list") => println!("{}", render_session_list(&self.session.id)?),
            Some("switch") => {
                let Some(target) = target else {
                    println!("Usage: /session switch <session-id>");
                    return Ok(());
                };
                let handle = resolve_session_reference(target)?;
                let session = Session::load_from_path(&handle.path)?;
                let message_count = session.messages.len();
                self.restore_session_runtime(handle.clone(), session)?;
                self.session = handle;
                self.persist_session()?;
                println!(
                    "{}",
                    ui::setting_changed(
                        "session switched",
                        &[
                            ("active", self.session.id.as_str()),
                            ("file", &self.session.path.display().to_string()),
                            ("messages", &message_count.to_string()),
                        ],
                    )
                );
            }
            Some("timeline") => println!("{}", self.render_timeline()),
            Some("fork") => self.fork_session(target)?,
            Some("rename") => self.rename_session(target)?,
            Some(other) => println!(
                "Unknown /session action '{other}'. Use /session list, /session switch <session-id>, /session timeline, /session fork [message-id], or /session rename <title>."
            ),
        }
        Ok(())
    }

    fn restore_session_runtime(
        &mut self,
        handle: SessionHandle,
        session: Session,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let state = session_runtime_state(
            &session,
            &self.model,
            self.allowed_tools.as_ref(),
            self.permission_mode,
            self.collaboration_mode,
            self.reasoning_effort,
            self.fast_mode,
            self.proxy_tool_calls,
        );
        self.model = state.model;
        self.service = state.service;
        self.allowed_tools = state.allowed_tools;
        self.permission_mode = state.permission_mode;
        self.collaboration_mode = state.collaboration_mode;
        self.reasoning_effort = state.reasoning_effort;
        self.fast_mode = state.fast_mode;
        self.proxy_tool_calls = state.proxy_tool_calls;
        self.mcp_catalog = load_mcp_catalog(&env::current_dir()?)?;
        self.rebuild_runtime(session)?;
        self.session = handle;
        Ok(())
    }

    pub(crate) fn persist_session(&self) -> Result<(), Box<dyn std::error::Error>> {
        let mut session = self.runtime.session().clone();
        session.metadata = Some(derive_session_metadata(
            &session,
            &self.model,
            self.allowed_tools.as_ref(),
            self.permission_mode,
            self.collaboration_mode,
            self.reasoning_effort,
            self.fast_mode,
            self.proxy_tool_calls,
        ));
        session.save_to_path(&self.session.path)?;
        auto_compact_inactive_sessions(&self.session.id)?;
        Ok(())
    }

    pub(crate) fn message_count(&self) -> usize {
        self.runtime.session().messages.len()
    }

    pub(crate) fn run_eval_turn(
        &mut self,
        prompt: String,
        prompter: &mut CliPermissionPrompter,
    ) -> Result<TurnSummary, RuntimeError> {
        self.run_runtime_turn(&prompt, prompter)
    }

    pub(crate) fn current_session(&self) -> Session {
        self.runtime.session().clone()
    }

    pub(crate) fn replace_session_for_eval(&mut self, session: Session) {
        self.runtime.replace_session(session);
    }

    pub(crate) fn session_id(&self) -> &str {
        &self.session.id
    }

    pub(crate) fn session_path(&self) -> &Path {
        &self.session.path
    }
}

fn handle_proxy_runtime_command(
    cli: &mut LiveCli,
    mode: ProxyCommand,
) -> Result<(), Box<dyn std::error::Error>> {
    match mode {
        ProxyCommand::Status => {
            println!(
                "{}",
                ui::setting_changed(
                    "proxy tool calls",
                    &[(
                        "state",
                        if cli.proxy_tool_calls {
                            "enabled"
                        } else {
                            "disabled"
                        },
                    )],
                )
            );
            Ok(())
        }
        ProxyCommand::Toggle => cli.set_proxy_tool_calls(!cli.proxy_tool_calls),
        ProxyCommand::Enable => cli.set_proxy_tool_calls(true),
        ProxyCommand::Disable => cli.set_proxy_tool_calls(false),
    }
}

fn handle_mcp_runtime_command(
    cli: &mut LiveCli,
    command: McpCommand,
) -> Result<(), Box<dyn std::error::Error>> {
    match command {
        McpCommand::Status => {
            cli.print_mcp_status();
            Ok(())
        }
        McpCommand::Tools => {
            cli.print_mcp_tools();
            Ok(())
        }
        McpCommand::Reload => cli.reload_mcp(),
        McpCommand::Add { name } => {
            println!(
                "{}",
                add_mcp_server_interactive(&env::current_dir()?, &name)?
            );
            cli.reload_mcp()
        }
        McpCommand::Enable { name } => {
            println!(
                "{}",
                set_mcp_server_enabled(&env::current_dir()?, &name, true)?
            );
            cli.reload_mcp()
        }
        McpCommand::Disable { name } => {
            println!(
                "{}",
                set_mcp_server_enabled(&env::current_dir()?, &name, false)?
            );
            cli.reload_mcp()
        }
    }
}

pub(crate) struct CliPermissionPrompter {
    current_mode: PermissionMode,
}

impl CliPermissionPrompter {
    pub(crate) fn new(current_mode: PermissionMode) -> Self {
        Self { current_mode }
    }
}

impl PermissionPrompter for CliPermissionPrompter {
    fn decide(&mut self, request: &PermissionRequest) -> PermissionPromptDecision {
        println!();
        println!(
            "{}",
            ui::permission_prompt_header(
                &request.tool_name,
                self.current_mode.as_str(),
                request.required_mode.as_str(),
            )
        );
        if let Some(preview) = render_permission_diff_preview(request) {
            println!("{preview}");
        } else {
            println!(
                "  {}  {}",
                "Input".with(ui::palette::MUTED),
                truncate_for_summary(&request.input, 1_000)
            );
        }
        print!(
            "\n{} ",
            "Allow once? [y/N]".to_string().with(ui::palette::WARN)
        );
        let _ = io::stdout().flush();

        let mut response = String::new();
        match io::stdin().read_line(&mut response) {
            Ok(_) => {
                let normalized = response.trim().to_ascii_lowercase();
                if matches!(normalized.as_str(), "y" | "yes") {
                    PermissionPromptDecision::Allow
                } else {
                    PermissionPromptDecision::Deny {
                        reason: format!(
                            "tool '{}' denied by user approval prompt",
                            request.tool_name
                        ),
                    }
                }
            }
            Err(error) => PermissionPromptDecision::Deny {
                reason: format!("permission approval failed: {error}"),
            },
        }
    }
}

fn render_permission_diff_preview(request: &PermissionRequest) -> Option<String> {
    match request.tool_name.as_str() {
        "write_file" => render_write_permission_preview(&request.input),
        "edit_file" => render_edit_permission_preview(&request.input),
        "apply_patch" => render_apply_patch_permission_preview(&request.input),
        _ => None,
    }
}

fn render_write_permission_preview(input: &str) -> Option<String> {
    let value = serde_json::from_str::<JsonValue>(input).ok()?;
    let path = value
        .get("path")
        .or_else(|| value.get("file_path"))
        .and_then(JsonValue::as_str)?;
    let after = value.get("content").and_then(JsonValue::as_str)?;
    let before = fs::read_to_string(path).ok();
    Some(render_permission_file_diff(
        path,
        before.is_some(),
        before.as_deref().unwrap_or(""),
        true,
        after,
    ))
}

fn render_edit_permission_preview(input: &str) -> Option<String> {
    let value = serde_json::from_str::<JsonValue>(input).ok()?;
    let path = value
        .get("path")
        .or_else(|| value.get("file_path"))
        .and_then(JsonValue::as_str)?;
    let old_string = value.get("old_string").and_then(JsonValue::as_str)?;
    let new_string = value.get("new_string").and_then(JsonValue::as_str)?;
    let replace_all = value
        .get("replace_all")
        .and_then(JsonValue::as_bool)
        .unwrap_or(false);
    let before = fs::read_to_string(path).ok()?;
    let after = if replace_all {
        before.replace(old_string, new_string)
    } else {
        before.replacen(old_string, new_string, 1)
    };
    Some(render_permission_file_diff(
        path, true, &before, true, &after,
    ))
}

fn render_apply_patch_permission_preview(input: &str) -> Option<String> {
    let value = serde_json::from_str::<JsonValue>(input).ok()?;
    let patch = value.get("patch").and_then(JsonValue::as_str)?;
    let checked = match apply_patch(patch, true) {
        Ok(output) => output,
        Err(error) => {
            return Some(format!(
                "  Patch check      failed before approval: {error}\n  Input            {}",
                truncate_for_summary(input, 1_000)
            ));
        }
    };

    let mut sections = vec![format!("  Patch check      {}", checked.summary)];
    for file in checked.changed_files.iter().take(6) {
        sections.push(render_permission_file_diff(
            &file.file_path,
            file.before_exists,
            file.before_content.as_deref().unwrap_or(""),
            file.after_exists,
            file.after_content.as_deref().unwrap_or(""),
        ));
    }
    if checked.changed_files.len() > 6 {
        sections.push(format!(
            "  ...              {} more files",
            checked.changed_files.len() - 6
        ));
    }
    Some(sections.join("\n"))
}

fn render_permission_file_diff(
    path: &str,
    before_exists: bool,
    before: &str,
    after_exists: bool,
    after: &str,
) -> String {
    let before_lines = split_preview_lines(before);
    let after_lines = split_preview_lines(after);
    let removed = before_lines.len();
    let added = after_lines.len();
    let mut lines = vec![
        format!("  Diff             {path}"),
        format!("    --- {}", if before_exists { path } else { "/dev/null" }),
        format!("    +++ {}", if after_exists { path } else { "/dev/null" }),
    ];

    if before_exists && after_exists {
        let prefix = common_prefix_len(&before_lines, &after_lines);
        let suffix = common_suffix_len(&before_lines[prefix..], &after_lines[prefix..]);
        let before_start = prefix.saturating_sub(3);
        let before_end = before_lines
            .len()
            .saturating_sub(suffix)
            .saturating_add(3)
            .min(before_lines.len());
        let after_start = prefix.saturating_sub(3);
        let after_end = after_lines
            .len()
            .saturating_sub(suffix)
            .saturating_add(3)
            .min(after_lines.len());
        lines.push(format!(
            "    @@ -{},{} +{},{} @@",
            before_start + 1,
            before_end.saturating_sub(before_start),
            after_start + 1,
            after_end.saturating_sub(after_start)
        ));
        for line in &before_lines[before_start..prefix] {
            lines.push(format!("     {}", truncate_for_summary(line, 180)));
        }
        for line in &before_lines[prefix..before_lines.len().saturating_sub(suffix)] {
            lines.push(format!("    -{}", truncate_for_summary(line, 180)));
        }
        for line in &after_lines[prefix..after_lines.len().saturating_sub(suffix)] {
            lines.push(format!("    +{}", truncate_for_summary(line, 180)));
        }
        let suffix_start = after_lines.len().saturating_sub(suffix);
        for line in &after_lines[suffix_start..after_end] {
            lines.push(format!("     {}", truncate_for_summary(line, 180)));
        }
    } else if after_exists {
        lines.push(format!("    @@ created file: {added} lines @@"));
        for line in after_lines.iter().take(40) {
            lines.push(format!("    +{}", truncate_for_summary(line, 180)));
        }
    } else {
        lines.push(format!("    @@ deleted file: {removed} lines @@"));
        for line in before_lines.iter().take(40) {
            lines.push(format!("    -{}", truncate_for_summary(line, 180)));
        }
    }

    truncate_diff_lines(lines, 80, 8_000).join("\n")
}

fn split_preview_lines(content: &str) -> Vec<String> {
    if content.is_empty() {
        Vec::new()
    } else {
        content.lines().map(ToOwned::to_owned).collect()
    }
}

fn common_prefix_len(left: &[String], right: &[String]) -> usize {
    left.iter()
        .zip(right)
        .take_while(|(left, right)| left == right)
        .count()
}

fn common_suffix_len(left: &[String], right: &[String]) -> usize {
    left.iter()
        .rev()
        .zip(right.iter().rev())
        .take_while(|(left, right)| left == right)
        .count()
}

fn truncate_diff_lines(mut lines: Vec<String>, max_lines: usize, max_chars: usize) -> Vec<String> {
    let mut chars = 0usize;
    let mut truncated = false;
    let mut kept = Vec::new();
    for line in lines.drain(..) {
        if kept.len() == max_lines || chars.saturating_add(line.len()) > max_chars {
            truncated = true;
            break;
        }
        chars += line.len();
        kept.push(line);
    }
    if truncated {
        kept.push("    ... diff preview truncated".to_string());
    }
    kept
}

fn build_system_prompt(
    service: ApiService,
    model: &str,
    collaboration_mode: CollaborationMode,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let mut prompt = load_system_prompt_with_model_family(
        env::current_dir()?,
        current_date(),
        env::consts::OS,
        "unknown",
        prompt_model_family(service, model),
    )?;
    prompt.push(
        [
            "# Web Research Guidance",
            " - Use WebSearch when you need current or web-sourced information, such as docs discovery, release notes, changelogs, product details, or anything that may have changed recently.",
            " - Use WebScrape when you already know the URL and need to read the page contents closely, especially documentation pages, blog posts, articles, and reference material.",
            " - Use WebFetch only for a quick single-page fetch/summary when you do not need richer scraping output.",
            " - For documentation work, the preferred sequence is usually: WebSearch to find the right page, then WebScrape to read it carefully.",
        ]
        .join("\n"),
    );
    prompt.push(mode_instructions(collaboration_mode));
    Ok(prompt)
}

fn mode_instructions(collaboration_mode: CollaborationMode) -> String {
    match collaboration_mode {
        CollaborationMode::Build => [
            "# Collaboration Mode: Build",
            "",
            "You are in Build mode. Execute the user's request directly and make the requested changes when appropriate.",
            "Prefer taking action over only proposing plans, and make reasonable assumptions when the repo can answer them.",
        ]
        .join("\n"),
        CollaborationMode::Plan => [
            "# Collaboration Mode: Plan",
            "",
            "You are in Plan mode until the system prompt changes again.",
            "Plan mode is not changed by user intent or imperative phrasing; if the user asks you to execute, plan that execution instead of doing it.",
            "",
            "Rules:",
            " - You may explore the repo, inspect files, and run non-mutating commands that improve the plan.",
            " - You must not edit files, apply patches, run mutating formatters/codegen, or otherwise perform implementation work.",
            " - Ask focused questions only when the answer cannot be discovered from the environment and materially changes the plan.",
            " - Final answers in this mode should be implementation-ready plans, not code changes.",
        ]
        .join("\n"),
    }
}

fn prompt_model_family(service: ApiService, model: &str) -> String {
    match service {
        ApiService::NanoGpt => "NanoGPT Messages API".to_string(),
        ApiService::Synthetic => format!("Synthetic ({model})"),
        ApiService::OpenAiCodex => format!("OpenAI Codex ({model})"),
        ApiService::OpencodeGo => format!("OpenCode Go ({model})"),
        ApiService::Neuralwatt => format!("Neuralwatt ({model})"),
        ApiService::Lilac => format!("Lilac ({model})"),
        ApiService::Grok => format!("Grok subscription ({model})"),
    }
}

fn build_runtime_feature_config(
) -> Result<runtime::RuntimeFeatureConfig, Box<dyn std::error::Error>> {
    let cwd = env::current_dir()?;
    let loader = ConfigLoader::default_for(&cwd);
    let runtime_config = loader.load()?;
    let mut feature_config = runtime_config.feature_config().clone();
    let plugin_manager = build_plugin_manager(&cwd, &loader, &runtime_config);
    let plugin_hooks = plugin_manager.aggregated_hooks()?;
    let plugin_pre_hooks = plugin_hooks.pre_tool_use.clone();
    let plugin_post_hooks = plugin_hooks.post_tool_use.clone();
    let merged_hooks = runtime::RuntimeHookConfig::new(
        feature_config
            .hooks()
            .pre_tool_use()
            .iter()
            .cloned()
            .chain(plugin_pre_hooks)
            .collect(),
        feature_config
            .hooks()
            .post_tool_use()
            .iter()
            .cloned()
            .chain(plugin_post_hooks)
            .collect(),
        feature_config.hooks().post_tool_use_failure().to_vec(),
    );
    feature_config = feature_config.with_hooks(merged_hooks);
    Ok(feature_config)
}

fn build_runtime(
    session: Session,
    service: ApiService,
    model: String,
    system_prompt: Vec<String>,
    enable_tools: bool,
    proxy_tool_calls: bool,
    mcp_catalog: McpCatalog,
    allowed_tools: Option<AllowedToolSet>,
    permission_mode: PermissionMode,
    collaboration_mode: CollaborationMode,
    reasoning_effort: Option<ReasoningEffort>,
    fast_mode: FastMode,
    render_model_output: bool,
) -> Result<ConversationRuntime<PebbleRuntimeClient, CliToolExecutor>, Box<dyn std::error::Error>> {
    let tool_registry = current_tool_registry()
        .map_err(|error| io::Error::other(format!("failed to load tool registry: {error}")))?;
    let tool_specs = if enable_tools {
        filter_runtime_tool_specs(
            available_runtime_tool_specs(&tool_registry, &mcp_catalog),
            allowed_tools.as_ref(),
        )
    } else {
        Vec::new()
    };
    let mut runtime_prompt = system_prompt;
    if enable_tools && proxy_tool_calls {
        runtime_prompt.push(build_proxy_system_prompt(&tool_specs));
    }
    let permission_policy = permission_policy(
        permission_mode,
        &available_runtime_tool_specs(&tool_registry, &mcp_catalog),
    );
    let max_output_tokens = max_output_tokens_for_model_or(&model, DEFAULT_MAX_TOKENS);
    let feature_config = build_runtime_feature_config()?;
    let compaction_config = feature_config.compaction();
    let auto_compaction_threshold = configured_auto_compaction_threshold().unwrap_or_else(|| {
        derive_auto_compaction_threshold(&model, max_output_tokens, compaction_config.reserved)
            .unwrap_or_else(auto_compaction_threshold_from_env)
    });
    Ok(ConversationRuntime::new_with_features(
        session,
        PebbleRuntimeClient::new(
            service,
            model.clone(),
            max_output_tokens,
            provider_for_model(&model),
            enable_tools,
            proxy_tool_calls,
            tool_specs.clone(),
            collaboration_mode,
            reasoning_effort,
            fast_mode,
            render_model_output,
        )?,
        CliToolExecutor::new(
            service,
            tool_registry,
            mcp_catalog,
            tool_specs,
            allowed_tools,
            render_model_output,
        ),
        permission_policy,
        runtime_prompt,
        &feature_config,
    )
    .with_auto_compaction_input_tokens_threshold(auto_compaction_threshold))
}

fn configured_auto_compaction_threshold() -> Option<u32> {
    std::env::var("PEBBLE_AUTO_COMPACT_INPUT_TOKENS")
        .ok()
        .and_then(|value| value.trim().parse::<u32>().ok())
        .filter(|threshold| *threshold > 0)
}

fn derive_auto_compaction_threshold(
    model: &str,
    max_output_tokens: u32,
    reserved_override: Option<u32>,
) -> Option<u32> {
    let context_length = context_length_for_model(model)?;
    let reserved_output_tokens = reserved_override.map_or_else(
        || {
            u64::from(max_output_tokens)
                .saturating_add(AUTO_COMPACTION_CONTEXT_SAFETY_MARGIN_TOKENS)
        },
        u64::from,
    );
    let available_input_tokens = context_length.saturating_sub(reserved_output_tokens);
    if available_input_tokens == 0 {
        return None;
    }

    let threshold =
        available_input_tokens.saturating_mul(AUTO_COMPACTION_CONTEXT_UTILIZATION_PERCENT) / 100;
    u32::try_from(threshold.max(1).min(u64::from(u32::MAX))).ok()
}

pub(crate) fn reasoning_effort_label(reasoning_effort: Option<ReasoningEffort>) -> &'static str {
    match reasoning_effort {
        Some(ReasoningEffort::Minimal) => "minimal",
        Some(ReasoningEffort::Low) => "low",
        Some(ReasoningEffort::Medium) => "medium",
        Some(ReasoningEffort::High) => "high",
        Some(ReasoningEffort::XHigh) => "xhigh",
        None => "default",
    }
}

fn format_mode_report(collaboration_mode: CollaborationMode) -> String {
    ui::setting_changed(
        "mode",
        &[
            ("active", collaboration_mode.as_str()),
            ("toggle", "Tab on empty prompt or /mode build|plan"),
        ],
    )
}

fn format_mode_switch_report(collaboration_mode: CollaborationMode) -> String {
    ui::setting_changed(
        "mode updated",
        &[
            ("result", collaboration_mode.as_str()),
            ("applies", "subsequent requests"),
        ],
    )
}

fn format_reasoning_report(
    configured: Option<ReasoningEffort>,
    effective: Option<ReasoningEffort>,
) -> String {
    ui::setting_changed(
        "reasoning",
        &[
            ("configured", reasoning_effort_label(configured)),
            ("effective", reasoning_effort_label(effective)),
            ("set", "/reasoning default|minimal|low|medium|high|xhigh"),
        ],
    )
}

fn format_reasoning_switch_report(
    configured: Option<ReasoningEffort>,
    effective: Option<ReasoningEffort>,
) -> String {
    ui::setting_changed(
        "reasoning updated",
        &[
            ("configured", reasoning_effort_label(configured)),
            ("effective", reasoning_effort_label(effective)),
            ("applies", "subsequent requests"),
        ],
    )
}

fn format_fast_mode_report(fast_mode: FastMode) -> String {
    ui::setting_changed(
        "fast mode",
        &[
            ("active", fast_mode.as_str()),
            ("toggle", "/fast on or /fast off"),
        ],
    )
}

fn format_fast_mode_switch_report(fast_mode: FastMode) -> String {
    ui::setting_changed(
        "fast mode updated",
        &[
            ("result", fast_mode.as_str()),
            ("applies", "subsequent requests"),
        ],
    )
}

/// Derive a compact model label for the prompt status strip. Model IDs can
/// be very long (e.g. `anthropic/claude-opus-4-20250514`); we trim the
/// provider-style prefix so the strip stays readable on narrow terminals.
fn short_model_name(model: &str) -> &str {
    model.rsplit('/').next().unwrap_or(model)
}

fn collect_tool_uses(summary: &runtime::TurnSummary) -> Vec<JsonValue> {
    summary
        .assistant_messages
        .iter()
        .flat_map(|message| message.blocks.iter())
        .filter_map(|block| match block {
            ContentBlock::ToolUse { id, name, input } => Some(serde_json::json!({
                "id": id,
                "name": name,
                "input": input,
            })),
            _ => None,
        })
        .collect()
}

fn collect_tool_results(summary: &runtime::TurnSummary) -> Vec<JsonValue> {
    summary
        .tool_results
        .iter()
        .flat_map(|message| message.blocks.iter())
        .filter_map(|block| match block {
            ContentBlock::ToolResult {
                tool_use_id,
                tool_name,
                output,
                is_error,
                compacted,
                archived_output_path,
            } => Some(serde_json::json!({
                "tool_use_id": tool_use_id,
                "tool_name": tool_name,
                "output": get_tool_result_context_output(output, *compacted),
                "is_error": is_error,
                "compacted": compacted,
                "archived_output_path": archived_output_path,
            })),
            _ => None,
        })
        .collect()
}

pub(crate) fn persist_turn_trace(
    cwd: &Path,
    session_id: &str,
    trace: &runtime::TurnTrace,
) -> io::Result<PathBuf> {
    let runs_dir = cwd.join(".pebble").join("runs");
    fs::create_dir_all(&runs_dir)?;
    let safe_session_id = sanitize_trace_filename(session_id);
    let path = runs_dir.join(format!(
        "{}-{}.json",
        safe_session_id, trace.started_at_unix_ms
    ));
    let safe_trace = trace.redacted();
    let bytes = serde_json::to_vec_pretty(&safe_trace)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    write_atomic(&path, bytes)?;
    let _gc_report = collect_generated_artifacts(cwd, retention_config_for(cwd), false);
    Ok(path)
}

fn sanitize_trace_filename(value: &str) -> String {
    let sanitized = value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_') {
                ch
            } else {
                '-'
            }
        })
        .collect::<String>();
    if sanitized.is_empty() {
        "session".to_string()
    } else {
        sanitized
    }
}

fn env_trace_notice_enabled() -> bool {
    std::env::var("PEBBLE_TRACE_NOTICE")
        .ok()
        .is_some_and(|value| matches!(value.as_str(), "1" | "true" | "yes"))
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ArchivedToolResultEntry {
    message_id: String,
    tool_use_id: String,
    tool_name: String,
    archived_output_path: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ArchivedToolResultStatus {
    Available,
    Missing,
    NotRecoverable,
}

impl ArchivedToolResultStatus {
    const fn label(self) -> &'static str {
        match self {
            Self::Available => "available",
            Self::Missing => "missing",
            Self::NotRecoverable => "not recoverable",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ArchivedToolResultRow {
    entry: ArchivedToolResultEntry,
    status: ArchivedToolResultStatus,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ArchivedToolResultSummary {
    entries: Vec<ArchivedToolResultRow>,
    available_count: usize,
    missing_count: usize,
    unrecoverable_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct LoadedArchivedToolResult {
    entry: ArchivedToolResultEntry,
    resolved_path: PathBuf,
    output: String,
}

fn collect_archived_tool_results(session: &Session) -> Vec<ArchivedToolResultEntry> {
    session
        .messages
        .iter()
        .flat_map(|message| {
            message.blocks.iter().filter_map(move |block| match block {
                ContentBlock::ToolResult {
                    tool_use_id,
                    tool_name,
                    compacted,
                    archived_output_path,
                    ..
                } if *compacted || archived_output_path.is_some() => {
                    Some(ArchivedToolResultEntry {
                        message_id: message.id.clone(),
                        tool_use_id: tool_use_id.clone(),
                        tool_name: tool_name.clone(),
                        archived_output_path: archived_output_path.clone(),
                    })
                }
                _ => None,
            })
        })
        .collect()
}

fn summarize_archived_tool_results(
    session: &Session,
    session_path: Option<&Path>,
) -> ArchivedToolResultSummary {
    let entries = collect_archived_tool_results(session)
        .into_iter()
        .map(|entry| {
            let status = archived_tool_result_status(&entry, session_path);
            ArchivedToolResultRow { entry, status }
        })
        .collect::<Vec<_>>();
    let available_count = entries
        .iter()
        .filter(|entry| entry.status == ArchivedToolResultStatus::Available)
        .count();
    let missing_count = entries
        .iter()
        .filter(|entry| entry.status == ArchivedToolResultStatus::Missing)
        .count();
    let unrecoverable_count = entries
        .iter()
        .filter(|entry| entry.status == ArchivedToolResultStatus::NotRecoverable)
        .count();

    ArchivedToolResultSummary {
        entries,
        available_count,
        missing_count,
        unrecoverable_count,
    }
}

fn archived_tool_result_status(
    entry: &ArchivedToolResultEntry,
    session_path: Option<&Path>,
) -> ArchivedToolResultStatus {
    let Some(path) = entry.archived_output_path.as_deref() else {
        return ArchivedToolResultStatus::NotRecoverable;
    };
    if resolve_archived_output_path(path, session_path).is_some_and(|candidate| candidate.exists())
    {
        ArchivedToolResultStatus::Available
    } else {
        ArchivedToolResultStatus::Missing
    }
}

fn render_archived_tool_results_report(
    session: &Session,
    session_path: Option<&Path>,
    action: Option<&str>,
    target: Option<&str>,
) -> Result<String, Box<dyn std::error::Error>> {
    match action.unwrap_or("list") {
        "list" => Ok(render_archived_tool_results_list(session, session_path)),
        "show" => render_archived_tool_result_show(session, session_path, target),
        "page" => page_archived_tool_result(session, session_path, target),
        "save" => save_archived_tool_result(session, session_path, target),
        other => Ok(archive_report(&[
            ("result", format!("unsupported action `{other}`")),
            (
                "usage",
                "/archives [list|show <id>|page <id>|save <id> [file]]".to_string(),
            ),
        ])),
    }
}

fn archive_report(fields: &[(&str, String)]) -> String {
    let rows = fields
        .iter()
        .map(|(label, value)| ui::PanelRow::Field {
            label: (*label).to_string(),
            value: value.clone(),
        })
        .collect::<Vec<_>>();
    ui::compact_panel("archives", &rows)
}

fn archive_report_with_body(fields: &[(&str, String)], body: &str) -> String {
    let mut rows = fields
        .iter()
        .map(|(label, value)| ui::PanelRow::Field {
            label: (*label).to_string(),
            value: value.clone(),
        })
        .collect::<Vec<_>>();
    rows.push(ui::PanelRow::Blank);
    rows.push(ui::PanelRow::Line(body.to_string()));
    ui::compact_panel("archives", &rows)
}

fn render_archived_tool_results_list(session: &Session, session_path: Option<&Path>) -> String {
    let summary = summarize_archived_tool_results(session, session_path);
    if summary.entries.is_empty() {
        return archive_report(&[("result", "no archived tool outputs".to_string())]);
    }

    let mut lines = vec![
        "Archives".to_string(),
        format!("  Count            {}", summary.entries.len()),
        format!("  Available        {}", summary.available_count),
        format!("  Missing          {}", summary.missing_count),
    ];
    if summary.unrecoverable_count > 0 {
        lines.push(format!(
            "  Unrecoverable    {}",
            summary.unrecoverable_count
        ));
    }

    for (index, entry) in summary.entries.iter().enumerate() {
        lines.push(String::new());
        lines.push(format!(
            "{}. Message         {}",
            index + 1,
            entry.entry.message_id
        ));
        lines.push(format!("   Tool call       {}", entry.entry.tool_use_id));
        lines.push(format!("   Tool            {}", entry.entry.tool_name));
        lines.push(format!(
            "   File            {}",
            entry
                .entry
                .archived_output_path
                .as_deref()
                .unwrap_or("(none)")
        ));
        lines.push(format!("   Status          {}", entry.status.label()));
    }

    lines.join("\n")
}

fn render_archived_tool_result_show(
    session: &Session,
    session_path: Option<&Path>,
    target: Option<&str>,
) -> Result<String, Box<dyn std::error::Error>> {
    let Some(target) = parse_archived_tool_result_target(target) else {
        return Ok(archive_report(&[
            ("result", "missing target".to_string()),
            (
                "usage",
                "/archives show <message-id-or-tool-call-id>".to_string(),
            ),
        ]));
    };

    match load_archived_tool_result(session, session_path, target) {
        Ok(loaded) => Ok(archive_report_with_body(
            &[
                ("result", "showing archived tool output".to_string()),
                ("message", loaded.entry.message_id),
                ("tool call", loaded.entry.tool_use_id),
                ("tool", loaded.entry.tool_name),
                ("file", loaded.resolved_path.display().to_string()),
            ],
            &loaded.output,
        )),
        Err(report) => Ok(report),
    }
}

fn page_archived_tool_result(
    session: &Session,
    session_path: Option<&Path>,
    target: Option<&str>,
) -> Result<String, Box<dyn std::error::Error>> {
    let Some(target) = parse_archived_tool_result_target(target) else {
        return Ok(archive_report(&[
            ("result", "missing target".to_string()),
            (
                "usage",
                "/archives page <message-id-or-tool-call-id>".to_string(),
            ),
        ]));
    };

    let loaded = match load_archived_tool_result(session, session_path, target) {
        Ok(loaded) => loaded,
        Err(report) => return Ok(report),
    };

    if !io::stdin().is_terminal() || !io::stdout().is_terminal() {
        return Ok(archive_report_with_body(
            &[
                (
                    "result",
                    "pager unavailable outside an interactive terminal".to_string(),
                ),
                ("message", loaded.entry.message_id),
                ("tool call", loaded.entry.tool_use_id),
                ("tool", loaded.entry.tool_name),
                ("file", loaded.resolved_path.display().to_string()),
            ],
            &loaded.output,
        ));
    }

    let pager_command = env::var("PAGER")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| "less -FRX".to_string());
    let mut parts = pager_command.split_whitespace();
    let Some(program) = parts.next() else {
        return Ok(archive_report_with_body(
            &[
                ("result", "pager unavailable".to_string()),
                ("message", loaded.entry.message_id),
                ("tool call", loaded.entry.tool_use_id),
                ("tool", loaded.entry.tool_name),
                ("file", loaded.resolved_path.display().to_string()),
            ],
            &loaded.output,
        ));
    };
    let args = parts.collect::<Vec<_>>();

    let mut child = match Command::new(program)
        .args(args)
        .stdin(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(_) => {
            return Ok(archive_report_with_body(
                &[
                    ("result", "pager unavailable".to_string()),
                    ("message", loaded.entry.message_id),
                    ("tool call", loaded.entry.tool_use_id),
                    ("tool", loaded.entry.tool_name),
                    ("file", loaded.resolved_path.display().to_string()),
                ],
                &loaded.output,
            ));
        }
    };
    if let Some(stdin) = child.stdin.as_mut() {
        let header = format!(
            "Archived tool output\nMessage: {}\nTool call: {}\nTool: {}\nFile: {}\n\n",
            loaded.entry.message_id,
            loaded.entry.tool_use_id,
            loaded.entry.tool_name,
            loaded.resolved_path.display(),
        );
        stdin.write_all(header.as_bytes())?;
        stdin.write_all(loaded.output.as_bytes())?;
    }
    let status = child.wait()?;
    if !status.success() {
        return Ok(archive_report(&[
            ("result", "pager exited unsuccessfully".to_string()),
            ("message", loaded.entry.message_id),
            ("tool call", loaded.entry.tool_use_id),
            ("tool", loaded.entry.tool_name),
            ("file", loaded.resolved_path.display().to_string()),
        ]));
    }

    Ok(archive_report(&[
        ("result", "viewed in pager".to_string()),
        ("message", loaded.entry.message_id),
        ("tool call", loaded.entry.tool_use_id),
        ("tool", loaded.entry.tool_name),
        ("file", loaded.resolved_path.display().to_string()),
    ]))
}

fn save_archived_tool_result(
    session: &Session,
    session_path: Option<&Path>,
    target: Option<&str>,
) -> Result<String, Box<dyn std::error::Error>> {
    let Some((target, path)) = parse_archived_tool_result_save_args(target) else {
        return Ok(archive_report(&[
            ("result", "missing target".to_string()),
            (
                "usage",
                "/archives save <message-id-or-tool-call-id> [file]".to_string(),
            ),
        ]));
    };

    let loaded = match load_archived_tool_result(session, session_path, target) {
        Ok(loaded) => loaded,
        Err(report) => return Ok(report),
    };
    let destination = match path {
        Some(path) => resolve_archive_save_path(path)?,
        None => env::current_dir()?.join(default_archived_tool_result_filename(
            &loaded.entry.tool_use_id,
            &loaded.entry.tool_name,
        )),
    };
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)?;
    }
    write_atomic(&destination, &loaded.output)?;

    Ok(archive_report(&[
        ("result", "wrote archived tool output".to_string()),
        ("message", loaded.entry.message_id),
        ("tool call", loaded.entry.tool_use_id),
        ("tool", loaded.entry.tool_name),
        ("source", loaded.resolved_path.display().to_string()),
        ("saved to", destination.display().to_string()),
    ]))
}

fn resolve_archived_output_path(
    archived_output_path: &str,
    session_path: Option<&Path>,
) -> Option<PathBuf> {
    let relative = PathBuf::from(archived_output_path);
    if relative.is_absolute() {
        return Some(relative);
    }

    let cwd_candidate = env::current_dir().ok().map(|cwd| cwd.join(&relative));
    let session_candidate = session_project_root(session_path).map(|root| root.join(&relative));

    cwd_candidate
        .as_ref()
        .filter(|candidate| candidate.exists())
        .cloned()
        .or_else(|| {
            session_candidate
                .as_ref()
                .filter(|candidate| candidate.exists())
                .cloned()
        })
        .or(cwd_candidate)
        .or(session_candidate)
}

fn session_project_root(session_path: Option<&Path>) -> Option<PathBuf> {
    let session_path = session_path?;
    let sessions_dir = session_path.parent()?;
    let pebble_dir = sessions_dir.parent()?;
    (pebble_dir.file_name()? == ".pebble").then(|| pebble_dir.parent().map(Path::to_path_buf))?
}

fn parse_archived_tool_result_target(target: Option<&str>) -> Option<&str> {
    target.map(str::trim).filter(|value| !value.is_empty())
}

fn parse_archived_tool_result_save_args(target: Option<&str>) -> Option<(&str, Option<&str>)> {
    let target = parse_archived_tool_result_target(target)?;
    match target.split_once(char::is_whitespace) {
        Some((identifier, path)) => {
            let path = path.trim();
            Some((identifier.trim(), (!path.is_empty()).then_some(path)))
        }
        None => Some((target, None)),
    }
}

fn load_archived_tool_result(
    session: &Session,
    session_path: Option<&Path>,
    target: &str,
) -> Result<LoadedArchivedToolResult, String> {
    let entries = collect_archived_tool_results(session);
    let Some(entry) = entries
        .into_iter()
        .find(|entry| entry.message_id == target || entry.tool_use_id == target)
    else {
        return Err(archive_report(&[(
            "result",
            format!("no archived tool output matched `{target}`"),
        )]));
    };

    let Some(archived_output_path) = entry.archived_output_path.as_deref() else {
        return Err(archive_report(&[
            ("result", "archived output unavailable".to_string()),
            ("message", entry.message_id),
            ("tool call", entry.tool_use_id),
            ("tool", entry.tool_name),
        ]));
    };

    let resolved_path = resolve_archived_output_path(archived_output_path, session_path)
        .ok_or_else(|| {
            archive_report(&[("result", "unable to resolve archive path".to_string())])
        })?;
    let output = match fs::read_to_string(&resolved_path) {
        Ok(output) => output,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            return Err(archive_report(&[
                ("result", "archive file missing".to_string()),
                ("message", entry.message_id),
                ("tool call", entry.tool_use_id),
                ("tool", entry.tool_name),
                ("file", resolved_path.display().to_string()),
            ]));
        }
        Err(error) => return Err(error.to_string()),
    };

    Ok(LoadedArchivedToolResult {
        entry,
        resolved_path,
        output,
    })
}

fn default_archived_tool_result_filename(tool_use_id: &str, tool_name: &str) -> String {
    format!(
        "archived-{}-{}.txt",
        sanitize_filename_component(tool_use_id),
        sanitize_filename_component(tool_name)
    )
}

fn resolve_archive_save_path(path: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let path = PathBuf::from(path.trim());
    if path.is_absolute() {
        Ok(path)
    } else {
        Ok(env::current_dir()?.join(path))
    }
}

fn sanitize_filename_component(value: &str) -> String {
    let sanitized = value
        .chars()
        .map(|char| {
            if char.is_ascii_alphanumeric() || matches!(char, '-' | '_') {
                char
            } else {
                '-'
            }
        })
        .collect::<String>();
    if sanitized.is_empty() {
        "item".to_string()
    } else {
        sanitized
    }
}

fn run_resume_command(
    session_path: &Path,
    session: &Session,
    command: &SlashCommand,
) -> Result<ResumeCommandOutcome, Box<dyn std::error::Error>> {
    match command {
        SlashCommand::Help { topic } => Ok(ResumeCommandOutcome {
            session: session.clone(),
            message: Some(topic.as_deref().map_or_else(render_repl_help, |topic| {
                render_slash_command_help_topic(Some(topic))
            })),
        }),
        SlashCommand::Compact => {
            let result = compact_resumed_session(session)
                .unwrap_or_else(|_| compact_resumed_session_local(session));
            let removed = result.removed_message_count;
            let kept = result.compacted_session.messages.len();
            let skipped = removed == 0;
            Ok(ResumeCommandOutcome {
                session: result.compacted_session,
                message: Some(if skipped {
                    ui::setting_changed(
                        "compact",
                        &[
                            ("result", "skipped"),
                            ("reason", "session below compaction threshold"),
                            ("messages kept", &kept.to_string()),
                        ],
                    )
                } else {
                    ui::setting_changed(
                        "compact",
                        &[
                            ("result", "compacted"),
                            ("messages removed", &removed.to_string()),
                            ("messages kept", &kept.to_string()),
                        ],
                    )
                }),
            })
        }
        SlashCommand::Archives { action, target } => Ok(ResumeCommandOutcome {
            session: session.clone(),
            message: Some(render_archived_tool_results_report(
                session,
                Some(session_path),
                action.as_deref(),
                target.as_deref(),
            )?),
        }),
        SlashCommand::Clear { confirm } => {
            if !confirm {
                return Ok(ResumeCommandOutcome {
                    session: session.clone(),
                    message: Some(ui::setting_changed(
                        "clear session",
                        &[
                            ("result", "confirmation required"),
                            ("run", "/clear --confirm"),
                        ],
                    )),
                });
            }
            let cleared = Session::new();
            Ok(ResumeCommandOutcome {
                session: cleared,
                message: Some(ui::setting_changed(
                    "session cleared",
                    &[("file", &session_path.display().to_string())],
                )),
            })
        }
        SlashCommand::Status => {
            let tracker = UsageTracker::from_session(session);
            let usage = tracker.cumulative_usage();
            let state = session_runtime_state(
                session,
                DEFAULT_MODEL,
                None,
                default_permission_mode(),
                CollaborationMode::Build,
                None,
                FastMode::Off,
                false,
            );
            let (undo_count, redo_count) = session_undo_redo_counts(session);
            Ok(ResumeCommandOutcome {
                session: session.clone(),
                message: Some(format_status_report(
                    state.service,
                    &state.model,
                    StatusUsage {
                        message_count: session.messages.len(),
                        turns: tracker.turns(),
                        undo_count,
                        redo_count,
                        latest: tracker.current_turn_usage(),
                        cumulative: usage,
                        estimated_tokens: 0,
                        context_window: None,
                    },
                    state.permission_mode.as_str(),
                    provider_label_for_service_model(state.service, &state.model).as_deref(),
                    state.proxy_tool_calls,
                    state.collaboration_mode,
                    effective_reasoning_effort(state.collaboration_mode, state.reasoning_effort),
                    state.fast_mode,
                    &McpCatalog::default(),
                    &status_context(Some(session_path))?,
                )),
            })
        }
        SlashCommand::Config { section } => Ok(ResumeCommandOutcome {
            session: session.clone(),
            message: Some(render_config_report(section.as_deref())?),
        }),
        SlashCommand::Memory => Ok(ResumeCommandOutcome {
            session: session.clone(),
            message: Some(render_memory_report()?),
        }),
        SlashCommand::Init => {
            let state = session_runtime_state(
                session,
                DEFAULT_MODEL,
                None,
                default_permission_mode(),
                CollaborationMode::Build,
                None,
                FastMode::Off,
                false,
            );
            let (report, warning) =
                initialize_repo_for_model(&env::current_dir()?, state.service, &state.model)?;
            let mut message = String::new();
            if let Some(warning) = warning {
                writeln!(&mut message, "{warning}")?;
                writeln!(&mut message)?;
            }
            message.push_str(&report.render());
            Ok(ResumeCommandOutcome {
                session: session.clone(),
                message: Some(message),
            })
        }
        SlashCommand::Diff => Ok(ResumeCommandOutcome {
            session: session.clone(),
            message: Some(render_diff_report()?),
        }),
        SlashCommand::Patch { args } => Ok(ResumeCommandOutcome {
            session: session.clone(),
            message: Some(render_patch_report(args.as_deref())?),
        }),
        SlashCommand::Version => Ok(ResumeCommandOutcome {
            session: session.clone(),
            message: Some(render_version_report()),
        }),
        SlashCommand::Timeline => Ok(ResumeCommandOutcome {
            session: session.clone(),
            message: Some(render_session_timeline(session)),
        }),
        SlashCommand::Export { path } => {
            let export_path = resolve_export_path(path.as_deref(), session)?;
            write_atomic(
                &export_path,
                render_export_text(session, Some(session_path)),
            )?;
            Ok(ResumeCommandOutcome {
                session: session.clone(),
                message: Some(format_export_report(
                    &export_path.display().to_string(),
                    session.messages.len(),
                )),
            })
        }
        SlashCommand::Resume { .. }
        | SlashCommand::Undo
        | SlashCommand::Redo
        | SlashCommand::Fork { .. }
        | SlashCommand::Rename { .. }
        | SlashCommand::Model { .. }
        | SlashCommand::Provider { .. }
        | SlashCommand::Route { .. }
        | SlashCommand::Logout { .. }
        | SlashCommand::Mcp { .. }
        | SlashCommand::Permissions { .. }
        | SlashCommand::Reasoning { .. }
        | SlashCommand::Fast { .. }
        | SlashCommand::Mode { .. }
        | SlashCommand::Session { .. }
        | SlashCommand::Sessions
        | SlashCommand::Branch { .. }
        | SlashCommand::Worktree { .. }
        | SlashCommand::Plugins { .. }
        | SlashCommand::Agents { .. }
        | SlashCommand::Skills { .. }
        | SlashCommand::Unknown(_) => Err("unsupported resumed slash command".into()),
    }
}

fn compact_resumed_session(
    session: &Session,
) -> Result<runtime::CompactionResult, Box<dyn std::error::Error>> {
    let fallback_model = session.metadata.as_ref().map_or_else(
        || default_model_or(DEFAULT_MODEL),
        |metadata| metadata.model.clone(),
    );
    let restored = session_runtime_state(
        session,
        &fallback_model,
        None,
        default_permission_mode(),
        CollaborationMode::Build,
        None,
        FastMode::Off,
        proxy_tool_calls_enabled(),
    );
    let system_prompt = build_system_prompt(
        restored.service,
        &restored.model,
        restored.collaboration_mode,
    )?;
    let mcp_catalog = load_mcp_catalog(&env::current_dir()?)?;
    let mut runtime = build_runtime(
        session.clone(),
        restored.service,
        restored.model,
        system_prompt,
        true,
        restored.proxy_tool_calls,
        mcp_catalog,
        restored.allowed_tools,
        restored.permission_mode,
        restored.collaboration_mode,
        restored.reasoning_effort,
        restored.fast_mode,
        false,
    )?;
    Ok(runtime.compact(force_compaction_config()))
}

fn compact_resumed_session_local(session: &Session) -> runtime::CompactionResult {
    runtime::compact_session(session, force_compaction_config())
}

fn force_compaction_config() -> CompactionConfig {
    CompactionConfig {
        max_estimated_tokens: 0,
        ..CompactionConfig::default()
    }
}

fn print_help() {
    println!("Pebble");
    println!();
    println!("Usage");
    println!(
        "  pebble [--model MODEL] [--permission-mode MODE] [--mode MODE] [--reasoning LEVEL] [--fast] [--allowedTools TOOL[,TOOL...]]"
    );
    println!("                                               Start interactive REPL");
    println!("  pebble login [SERVICE] [--api-key KEY]    Connect a model or research provider");
    println!(
        "                                               Services: nanogpt, neuralwatt, lilac, grok, synthetic, openai-codex, opencode-go, exa"
    );
    println!("  pebble logout [SERVICE]                   Remove saved credentials for a service");
    println!("  pebble model [MODEL_ID]                   Choose or persist a default model");
    println!("  pebble provider [NAME]                   Choose a model provider, then a model");
    println!("  pebble route [ROUTE_ID|default]          Override NanoGPT's upstream route");
    println!("  pebble proxy [on|off|status]              Toggle XML tool-call proxy mode");
    println!(
        "  pebble mcp [status|tools|reload|add <name>|enable <name>|disable <name>] Inspect configured MCP servers and tools"
    );
    println!("  pebble plugins [list|help|install|enable|disable|uninstall|update] [TARGET]");
    println!("  pebble branch [list|create|switch] [ARG]  Inspect or change git branches");
    println!("  pebble worktree [list|add|remove|prune]   Inspect or manage git worktrees");
    println!("  pebble agents [list|help]                 List configured Pebble agents");
    println!("  pebble skills [list|help|init <name>]     List or scaffold Pebble skills");
    println!("  pebble init                               Create starter Pebble project files");
    println!("  pebble doctor                             Run local environment diagnostics");
    println!("  pebble doctor bundle                      Write a redacted diagnostics bundle");
    println!("  pebble doctor providers [--json]          Probe provider auth and model catalogs");
    println!("  pebble ci check [--json] [--save-report] Run local CI harness safety checks");
    println!("  pebble ci history [--json] [--limit N]   Show saved CI check reports");
    println!("  pebble release check [--json] [--save-report]");
    println!("                                               Summarize ship readiness from saved harness reports");
    println!("  pebble trace TRACE.json                   Render a saved .pebble/runs trace");
    println!("  pebble replay TRACE.json                  Replay a saved trace timeline");
    println!("  pebble gc [--dry-run]                    Prune generated trace and eval artifacts");
    println!(
        "  pebble eval [--check] [--fail-on-failures] SUITE.json Run or validate an eval suite"
    );
    println!("  pebble eval capture TRACE.json --suite SUITE.json [--name NAME] [--force]");
    println!(
        "                                               Append a trace-derived regression case"
    );
    println!("  pebble eval compare OLD.json NEW.json     Compare two eval run reports");
    println!("  pebble eval replay REPORT.json [--case ID]");
    println!("                                               Explain failed eval cases with saved traces");
    println!("  pebble eval history [--suite S] [--model M] [--limit N]");
    println!("                                               Show recent eval pass-rate trends");
    println!("  Add --json to trace/replay/eval debug commands for machine-readable output");
    println!("  pebble config check [--json]              Validate Pebble settings files");
    println!("  pebble self-update                        Update from GitHub releases");
    println!("  pebble resume [SESSION_ID_OR_PATH]");
    println!("                                               Resume a saved session, or pick one and enter the REPL");
    println!("  pebble --resume [SESSION_ID_OR_PATH] [/status] [/compact] [...]");
    println!("                                               Resume a saved session and optionally run slash commands");
    println!(
        "  pebble prompt [--model MODEL] [--permission-mode MODE] [--mode MODE] [--reasoning LEVEL] [--fast] [--output-format text|json] TEXT"
    );
    println!(
        "                                               Send one prompt and stream the response"
    );
    println!(
        "  pebble [--model MODEL] [--permission-mode MODE] [--mode MODE] [--reasoning LEVEL] [--fast] [--output-format text|json] TEXT"
    );
    println!(
        "                                               Shorthand non-interactive prompt mode"
    );
    println!("  pebble dump-manifests");
    println!("  pebble bootstrap-plan");
    println!("  pebble system-prompt [--cwd PATH] [--date YYYY-MM-DD]");
    println!("  pebble --version");
    println!(
        "  --permission-mode MODE                     read-only, workspace-write, or danger-full-access"
    );
    println!("  --mode MODE                               build or plan");
    println!(
        "  --reasoning LEVEL                         default, minimal, low, medium, high, or xhigh"
    );
    println!(
        "  --fast                                     Enable ChatGPT fast mode for OpenAI Codex"
    );
    println!(
        "  --thinking                                 Compatibility alias for --reasoning medium"
    );
    println!(
        "  --output-format FORMAT                     Non-interactive output format: text or json"
    );
    println!();
    println!("{}", render_repl_help());
    println!();
    println!("{}", render_help_topics_overview());
}

fn print_version() {
    println!("{}", render_version_report());
}

fn run_init() -> Result<(), Box<dyn std::error::Error>> {
    let model = default_model_or(DEFAULT_MODEL);
    run_init_with_model(infer_service_for_model(&model), &model)
}

fn run_init_with_model(service: ApiService, model: &str) -> Result<(), Box<dyn std::error::Error>> {
    let cwd = env::current_dir()?;
    let (report, warning) = initialize_repo_for_model(&cwd, service, model)?;
    if let Some(warning) = warning {
        eprintln!("{warning}");
    }
    println!("{}", report.render());
    Ok(())
}

fn initialize_repo_for_model(
    cwd: &Path,
    service: ApiService,
    model: &str,
) -> Result<(crate::init::InitReport, Option<String>), Box<dyn std::error::Error>> {
    if cwd.join("PEBBLE.md").exists() {
        return Ok((initialize_repo(cwd)?, None));
    }

    match generate_pebble_md(cwd, service, model) {
        Ok(content) => Ok((initialize_repo_with_pebble_md(cwd, &content)?, None)),
        Err(error) => Ok((
            initialize_repo(cwd)?,
            Some(format_init_generation_warning(service, model, &*error)),
        )),
    }
}

fn generate_pebble_md(
    cwd: &Path,
    service: ApiService,
    model: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let request = MessageRequest {
        model: model.to_string(),
        max_tokens: INIT_PEBBLE_MD_MAX_TOKENS,
        messages: vec![InputMessage::user_text(build_init_generation_prompt(cwd)?)],
        system: Some(init_generation_system_prompt().to_string()),
        tools: None,
        tool_choice: None,
        thinking: None,
        reasoning_effort: None,
        fast_mode: false,
        stream: false,
    };
    let mut client = NanoGptClient::from_service_env(service)?;
    if service == ApiService::NanoGpt {
        client = client.with_provider(provider_for_model(model));
    }

    let runtime = tokio::runtime::Runtime::new()?;
    let response = runtime.block_on(async { client.send_message(&request).await })?;
    extract_generated_pebble_md(response)
}

fn init_generation_system_prompt() -> &'static str {
    "You write repo-specific PEBBLE.md files for a coding assistant. Return only markdown for the file with no code fences or prefatory text. Stay concrete, concise, and factual. Do not invent commands, directories, workflows, or architecture details. If a detail is uncertain from the provided context, say to verify it or omit it."
}

fn build_init_generation_prompt(cwd: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let project_context = runtime::ProjectContext::discover_with_git(cwd, current_date())?;
    let mut prompt = String::new();
    writeln!(
        &mut prompt,
        "Create a project-specific `PEBBLE.md` for this repository."
    )?;
    writeln!(
        &mut prompt,
        "Use the supplied repository context to replace the generic starter template with concrete guidance."
    )?;
    writeln!(&mut prompt)?;
    writeln!(&mut prompt, "Required output shape:")?;
    writeln!(&mut prompt, "- `# PEBBLE.md`")?;
    writeln!(&mut prompt, "- `## Project Overview`")?;
    writeln!(&mut prompt, "- `## Repository Shape`")?;
    writeln!(&mut prompt, "- `## Commands`")?;
    writeln!(&mut prompt, "- `## Working Agreement`")?;
    writeln!(&mut prompt)?;
    writeln!(&mut prompt, "Instructions:")?;
    writeln!(
        &mut prompt,
        "- Keep it concise, actionable, and specific to this repo."
    )?;
    writeln!(
        &mut prompt,
        "- Prefer bullet lists for commands and operational guidance."
    )?;
    writeln!(
        &mut prompt,
        "- Mention verification commands only when supported by the provided files."
    )?;
    writeln!(
        &mut prompt,
        "- If the context is incomplete, say what to verify instead of guessing."
    )?;
    writeln!(&mut prompt)?;
    writeln!(&mut prompt, "Repository context")?;
    writeln!(&mut prompt, "Working directory: {}", cwd.display())?;
    writeln!(
        &mut prompt,
        "Top-level entries:\n{}",
        render_init_top_level_entries(cwd)?
    )?;

    if let Some(git_status) = project_context.git_status.as_deref() {
        let trimmed = git_status.trim();
        if !trimmed.is_empty() {
            writeln!(&mut prompt)?;
            writeln!(&mut prompt, "Git status:")?;
            writeln!(&mut prompt, "```text")?;
            writeln!(&mut prompt, "{trimmed}")?;
            writeln!(&mut prompt, "```")?;
        }
    }

    let context_files = collect_init_context_files(cwd, &project_context);
    if !context_files.is_empty() {
        writeln!(&mut prompt)?;
        writeln!(&mut prompt, "Key file excerpts:")?;
        for path in context_files {
            if let Some(snippet) = read_init_context_file(&path) {
                writeln!(&mut prompt, "### {}", display_init_context_path(cwd, &path))?;
                writeln!(&mut prompt, "```text")?;
                writeln!(&mut prompt, "{snippet}")?;
                writeln!(&mut prompt, "```")?;
            }
        }
    }

    writeln!(&mut prompt)?;
    writeln!(&mut prompt, "Starter template to improve:")?;
    writeln!(&mut prompt, "```markdown")?;
    writeln!(&mut prompt, "{}", render_init_pebble_md(cwd))?;
    writeln!(&mut prompt, "```")?;

    Ok(prompt)
}

fn render_init_top_level_entries(cwd: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let mut entries = fs::read_dir(cwd)?
        .flatten()
        .map(|entry| {
            let path = entry.path();
            let mut name = entry.file_name().to_string_lossy().into_owned();
            if path.is_dir() {
                name.push('/');
            }
            name
        })
        .collect::<Vec<_>>();
    entries.sort();

    let remaining = entries.len().saturating_sub(MAX_INIT_TOP_LEVEL_ENTRIES);
    entries.truncate(MAX_INIT_TOP_LEVEL_ENTRIES);
    if remaining > 0 {
        entries.push(format!("... and {remaining} more"));
    }

    Ok(entries
        .into_iter()
        .map(|entry| format!("- {entry}"))
        .collect::<Vec<_>>()
        .join("\n"))
}

fn collect_init_context_files(
    cwd: &Path,
    project_context: &runtime::ProjectContext,
) -> Vec<PathBuf> {
    let mut seen = HashSet::new();
    let mut paths = Vec::new();
    for path in [
        cwd.join("README.md"),
        cwd.join("Cargo.toml"),
        cwd.join("package.json"),
        cwd.join("pyproject.toml"),
        cwd.join("go.mod"),
        cwd.join("Makefile"),
        cwd.join("justfile"),
        cwd.join("Justfile"),
    ] {
        if path.is_file() && seen.insert(path.clone()) {
            paths.push(path);
        }
    }
    for file in &project_context.instruction_files {
        if let Some(name) = file.path.file_name().and_then(|name| name.to_str()) {
            if name.eq_ignore_ascii_case("PEBBLE.md") {
                continue;
            }
        }
        if file.path.is_file() && seen.insert(file.path.clone()) {
            paths.push(file.path.clone());
        }
    }
    paths.truncate(MAX_INIT_CONTEXT_FILES);
    paths
}

fn read_init_context_file(path: &Path) -> Option<String> {
    let contents = fs::read_to_string(path).ok()?;
    let normalized = contents.replace("\r\n", "\n");
    let trimmed = normalized.trim();
    if trimmed.is_empty() {
        return None;
    }
    Some(truncate_chars(trimmed, MAX_INIT_CONTEXT_CHARS))
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    let mut truncated = text.chars().take(max_chars).collect::<String>();
    if text.chars().count() > max_chars {
        truncated.push_str("\n... [truncated]");
    }
    truncated
}

fn display_init_context_path(cwd: &Path, path: &Path) -> String {
    path.strip_prefix(cwd).unwrap_or(path).display().to_string()
}

fn extract_generated_pebble_md(
    response: MessageResponse,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut text = String::new();
    for block in response.content {
        if let OutputContentBlock::Text { text: block_text } = block {
            text.push_str(&block_text);
        }
    }

    normalize_generated_pebble_md(&text)
        .ok_or_else(|| "model returned no markdown content for PEBBLE.md".into())
}

fn normalize_generated_pebble_md(raw: &str) -> Option<String> {
    let normalized = raw.replace("\r\n", "\n");
    let trimmed = normalized.trim();
    if trimmed.is_empty() {
        return None;
    }

    let unfenced = strip_markdown_code_fence(trimmed).unwrap_or(trimmed).trim();
    if unfenced.is_empty() {
        return None;
    }

    let mut content = unfenced.find("# PEBBLE.md").map_or_else(
        || unfenced.to_string(),
        |index| unfenced[index..].to_string(),
    );
    if !content.starts_with("# PEBBLE.md") {
        content = format!("# PEBBLE.md\n\n{content}");
    }
    if !content.ends_with('\n') {
        content.push('\n');
    }
    Some(content)
}

fn strip_markdown_code_fence(text: &str) -> Option<&str> {
    let rest = text.strip_prefix("```")?;
    let newline = rest.find('\n')?;
    let body = &rest[newline + 1..];
    body.strip_suffix("\n```")
        .or_else(|| body.strip_suffix("```"))
}

fn format_init_generation_warning(
    service: ApiService,
    model: &str,
    error: &dyn std::fmt::Display,
) -> String {
    format!(
        "warning: failed to generate a repo-specific PEBBLE.md with {}/{}; used the starter template instead: {error}",
        service.display_name(),
        model
    )
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
struct GitHubRelease {
    tag_name: String,
    #[serde(default)]
    body: String,
    #[serde(default)]
    assets: Vec<GitHubReleaseAsset>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
struct GitHubReleaseAsset {
    name: String,
    browser_download_url: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SelectedReleaseAssets {
    binary: GitHubReleaseAsset,
    checksum: GitHubReleaseAsset,
}

fn run_self_update() -> Result<(), Box<dyn std::error::Error>> {
    let Some(release) = fetch_latest_release()? else {
        println!(
            "{}",
            render_update_report(
                "No published release available",
                Some(VERSION),
                None,
                Some("GitHub latest release endpoint returned no published release for nanogpt-community/pebble."),
                None,
            )
        );
        return Ok(());
    };

    let latest_version = normalize_version_tag(&release.tag_name);
    if !is_newer_version(VERSION, &latest_version) {
        println!(
            "{}",
            render_update_report(
                "Already up to date",
                Some(VERSION),
                Some(&latest_version),
                Some("Current binary already matches the latest published release."),
                Some(&release.body),
            )
        );
        return Ok(());
    }

    let selected = match select_release_assets(&release) {
        Ok(selected) => selected,
        Err(message) => {
            println!(
                "{}",
                render_update_report(
                    "Release found, but no installable asset matched this platform",
                    Some(VERSION),
                    Some(&latest_version),
                    Some(&message),
                    Some(&release.body),
                )
            );
            return Ok(());
        }
    };

    #[cfg(windows)]
    {
        println!(
            "{}",
            render_update_report(
                "Manual install required on Windows",
                Some(VERSION),
                Some(&latest_version),
                Some(&format!(
                    "Download {} from the latest GitHub release and replace pebble.exe manually. In-place self-update is not supported on Windows yet.",
                    selected.binary.name
                )),
                Some(&release.body),
            )
        );
        return Ok(());
    }

    #[cfg(not(windows))]
    {
        let client = build_self_update_client()?;
        let binary_bytes = download_bytes(&client, &selected.binary.browser_download_url)?;
        let checksum_manifest = download_text(&client, &selected.checksum.browser_download_url)?;
        let expected_checksum = parse_checksum_for_asset(&checksum_manifest, &selected.binary.name)
            .ok_or_else(|| {
                format!(
                    "checksum manifest did not contain an entry for {}",
                    selected.binary.name
                )
            })?;
        let actual_checksum = sha256_hex(&binary_bytes);
        if actual_checksum != expected_checksum {
            return Err(format!(
                "downloaded asset checksum mismatch for {} (expected {}, got {})",
                selected.binary.name, expected_checksum, actual_checksum
            )
            .into());
        }

        replace_current_executable(&binary_bytes)?;

        println!(
            "{}",
            render_update_report(
                "Update installed",
                Some(VERSION),
                Some(&latest_version),
                Some(&format!(
                    "Installed {} from GitHub release assets for {}.",
                    selected.binary.name,
                    current_target()
                )),
                Some(&release.body),
            )
        );
        Ok(())
    }
}

fn fetch_latest_release() -> Result<Option<GitHubRelease>, Box<dyn std::error::Error>> {
    let client = build_self_update_client()?;
    let response = client
        .get(SELF_UPDATE_LATEST_RELEASE_URL)
        .header(reqwest::header::ACCEPT, "application/vnd.github+json")
        .send()?;

    if response.status() == reqwest::StatusCode::NOT_FOUND {
        return Ok(None);
    }

    let response = response.error_for_status()?;
    Ok(Some(response.json()?))
}

fn build_self_update_client() -> Result<BlockingClient, reqwest::Error> {
    BlockingClient::builder()
        .user_agent(SELF_UPDATE_USER_AGENT)
        .build()
}

fn download_bytes(
    client: &BlockingClient,
    url: &str,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let response = client.get(url).send()?.error_for_status()?;
    Ok(response.bytes()?.to_vec())
}

fn download_text(client: &BlockingClient, url: &str) -> Result<String, Box<dyn std::error::Error>> {
    let response = client.get(url).send()?.error_for_status()?;
    Ok(response.text()?)
}

fn normalize_version_tag(version: &str) -> String {
    version.trim().trim_start_matches('v').to_string()
}

fn is_newer_version(current: &str, latest: &str) -> bool {
    compare_versions(latest, current).is_gt()
}

fn compare_versions(left: &str, right: &str) -> std::cmp::Ordering {
    let left = normalize_version_tag(left);
    let right = normalize_version_tag(right);
    let left_parts = version_components(&left);
    let right_parts = version_components(&right);
    let max_len = left_parts.len().max(right_parts.len());
    for index in 0..max_len {
        let left_part = *left_parts.get(index).unwrap_or(&0);
        let right_part = *right_parts.get(index).unwrap_or(&0);
        match left_part.cmp(&right_part) {
            std::cmp::Ordering::Equal => {}
            ordering => return ordering,
        }
    }
    std::cmp::Ordering::Equal
}

fn version_components(version: &str) -> Vec<u64> {
    version
        .split(['.', '-'])
        .map(|part| {
            part.chars()
                .take_while(char::is_ascii_digit)
                .collect::<String>()
        })
        .filter(|part| !part.is_empty())
        .filter_map(|part| part.parse::<u64>().ok())
        .collect()
}

fn current_target() -> String {
    BUILD_TARGET.map_or_else(default_target_triple, str::to_string)
}

fn default_target_triple() -> String {
    let os = match env::consts::OS {
        "linux" => "unknown-linux-gnu",
        "macos" => "apple-darwin",
        "windows" => "pc-windows-msvc",
        other => other,
    };
    format!("{}-{os}", env::consts::ARCH)
}

fn target_name_candidates() -> Vec<String> {
    let mut candidates = Vec::new();
    if let Some(target) = BUILD_TARGET {
        candidates.push(target.to_string());
    }
    candidates.push(default_target_triple());
    candidates.push(format!("{}-{}", env::consts::ARCH, env::consts::OS));
    candidates.sort();
    candidates.dedup();
    candidates
}

fn release_asset_candidates() -> Vec<String> {
    let mut candidates = target_name_candidates()
        .into_iter()
        .flat_map(|target| {
            let mut names = vec![format!("pebble-{target}")];
            if env::consts::OS == "windows" {
                names.push(format!("pebble-{target}.exe"));
            }
            names
        })
        .collect::<Vec<_>>();
    if env::consts::OS == "windows" {
        candidates.push("pebble.exe".to_string());
    }
    candidates.push("pebble".to_string());
    candidates.sort();
    candidates.dedup();
    candidates
}

fn select_release_assets(release: &GitHubRelease) -> Result<SelectedReleaseAssets, String> {
    let binary = release_asset_candidates()
        .into_iter()
        .find_map(|candidate| {
            release
                .assets
                .iter()
                .find(|asset| asset.name == candidate)
                .cloned()
        })
        .ok_or_else(|| {
            format!(
                "no binary asset matched target {} (expected one of: {})",
                current_target(),
                release_asset_candidates().join(", ")
            )
        })?;

    let checksum = CHECKSUM_ASSET_CANDIDATES
        .iter()
        .find_map(|candidate| {
            release
                .assets
                .iter()
                .find(|asset| asset.name == *candidate)
                .cloned()
        })
        .ok_or_else(|| {
            format!(
                "release did not include a checksum manifest (expected one of: {})",
                CHECKSUM_ASSET_CANDIDATES.join(", ")
            )
        })?;

    Ok(SelectedReleaseAssets { binary, checksum })
}

fn parse_checksum_for_asset(manifest: &str, asset_name: &str) -> Option<String> {
    manifest.lines().find_map(|line| {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return None;
        }
        if let Some((left, right)) = trimmed.split_once(" = ") {
            return left
                .strip_prefix("SHA256 (")
                .and_then(|value| value.strip_suffix(')'))
                .filter(|file| *file == asset_name)
                .map(|_| right.to_ascii_lowercase());
        }
        let mut parts = trimmed.split_whitespace();
        let checksum = parts.next()?;
        let file = parts
            .next_back()
            .or_else(|| parts.next())?
            .trim_start_matches('*');
        (file == asset_name).then(|| checksum.to_ascii_lowercase())
    })
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn replace_current_executable(binary_bytes: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
    let current = env::current_exe()?;
    replace_executable_at(&current, binary_bytes)
}

fn replace_executable_at(
    current: &Path,
    binary_bytes: &[u8],
) -> Result<(), Box<dyn std::error::Error>> {
    let temp_path = current.with_extension("download");
    let backup_path = current.with_extension("bak");

    if backup_path.exists() {
        fs::remove_file(&backup_path)?;
    }
    fs::write(&temp_path, binary_bytes)?;
    copy_executable_permissions(current, &temp_path)?;

    fs::rename(current, &backup_path)?;
    if let Err(error) = fs::rename(&temp_path, current) {
        let _ = fs::rename(&backup_path, current);
        let _ = fs::remove_file(&temp_path);
        return Err(format!("failed to replace current executable: {error}").into());
    }

    if let Err(error) = fs::remove_file(&backup_path) {
        eprintln!(
            "warning: failed to remove self-update backup {}: {error}",
            backup_path.display()
        );
    }
    Ok(())
}

#[cfg(unix)]
fn copy_executable_permissions(
    source: &Path,
    destination: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::os::unix::fs::PermissionsExt;

    let mode = fs::metadata(source)?.permissions().mode();
    fs::set_permissions(destination, fs::Permissions::from_mode(mode))?;
    Ok(())
}

#[cfg(not(unix))]
fn copy_executable_permissions(
    _source: &Path,
    _destination: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    Ok(())
}

fn render_update_report(
    result: &str,
    current_version: Option<&str>,
    latest_version: Option<&str>,
    detail: Option<&str>,
    changelog: Option<&str>,
) -> String {
    let mut report = String::from("Self-update\n");
    let _ = writeln!(report, "  Repository       {SELF_UPDATE_REPOSITORY}");
    let _ = writeln!(report, "  Result           {result}");
    if let Some(current_version) = current_version {
        let _ = writeln!(report, "  Current version  {current_version}");
    }
    if let Some(latest_version) = latest_version {
        let _ = writeln!(report, "  Latest version   {latest_version}");
    }
    if let Some(detail) = detail {
        let _ = writeln!(report, "  Detail           {detail}");
    }
    let trimmed = changelog.map(str::trim).filter(|value| !value.is_empty());
    if let Some(changelog) = trimmed {
        report.push_str("\nChangelog\n");
        report.push_str(changelog);
    }
    report.trim_end().to_string()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DiagnosticLevel {
    Ok,
    Warn,
    Fail,
}

impl DiagnosticLevel {
    const fn label(self) -> &'static str {
        match self {
            Self::Ok => "OK",
            Self::Warn => "WARN",
            Self::Fail => "FAIL",
        }
    }

    const fn is_failure(self) -> bool {
        matches!(self, Self::Fail)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DiagnosticCheck {
    name: &'static str,
    level: DiagnosticLevel,
    summary: String,
    details: Vec<String>,
}

impl DiagnosticCheck {
    fn new(name: &'static str, level: DiagnosticLevel, summary: impl Into<String>) -> Self {
        Self {
            name,
            level,
            summary: summary.into(),
            details: Vec::new(),
        }
    }

    fn with_details(mut self, details: Vec<String>) -> Self {
        self.details = details;
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ConfigFileCheck {
    path: PathBuf,
    exists: bool,
    valid: bool,
    note: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DoctorReport {
    checks: Vec<DiagnosticCheck>,
}

impl DoctorReport {
    fn has_failures(&self) -> bool {
        self.checks.iter().any(|check| check.level.is_failure())
    }

    fn render(&self) -> String {
        let mut lines = vec!["Doctor diagnostics".to_string()];
        let ok_count = self
            .checks
            .iter()
            .filter(|check| check.level == DiagnosticLevel::Ok)
            .count();
        let warn_count = self
            .checks
            .iter()
            .filter(|check| check.level == DiagnosticLevel::Warn)
            .count();
        let fail_count = self
            .checks
            .iter()
            .filter(|check| check.level == DiagnosticLevel::Fail)
            .count();
        lines.push(format!(
            "Summary\n  OK               {ok_count}\n  Warnings         {warn_count}\n  Failures         {fail_count}"
        ));
        lines.extend(self.checks.iter().map(render_diagnostic_check));
        lines.join("\n\n")
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DiagnosticsBundle {
    path: PathBuf,
    files: Vec<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CiCheckReport {
    kind: String,
    ok: bool,
    cwd: PathBuf,
    steps: Vec<CiCheckStepReport>,
    diagnostics_bundle: Option<PathBuf>,
    report_path: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CiCheckStepReport {
    name: String,
    ok: bool,
    duration_ms: u128,
    error: Option<String>,
    #[serde(default)]
    artifact: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
struct CiHistoryReport {
    kind: &'static str,
    cwd: PathBuf,
    limit: usize,
    reports: Vec<CiCheckReport>,
}

#[derive(Debug, Clone, Serialize)]
struct ReleaseCheckReport {
    kind: &'static str,
    ok: bool,
    cwd: PathBuf,
    version: String,
    git: ReleaseGitStatus,
    latest_ci: Option<ReleaseCiSummary>,
    latest_eval: Option<ReleaseEvalSummary>,
    config: ReleaseConfigSummary,
    diagnostics_bundle: Option<PathBuf>,
    golden_trace_regressions: Option<ReleaseStepSummary>,
    diagnostics_redaction: Option<ReleaseStepSummary>,
    report_path: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize)]
struct ReleaseGitStatus {
    branch: Option<String>,
    commit: Option<String>,
    dirty: Option<bool>,
    error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct ReleaseCiSummary {
    ok: bool,
    report_path: Option<PathBuf>,
    duration_ms: u128,
    failed_steps: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct ReleaseEvalSummary {
    ok: bool,
    run_id: String,
    suite: String,
    model: String,
    passed: usize,
    failed: usize,
    duration_ms: u128,
    report_file: PathBuf,
}

#[derive(Debug, Clone, Serialize)]
struct ReleaseConfigSummary {
    ok: bool,
    loaded_files: usize,
    discovered_files: usize,
    issues: usize,
}

#[derive(Debug, Clone, Serialize)]
struct ReleaseStepSummary {
    ok: bool,
    duration_ms: u128,
    error: Option<String>,
    artifact: Option<PathBuf>,
}

#[derive(Debug, Default)]
struct CiStepOutput {
    diagnostics_bundle: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CiStepFailure {
    message: String,
    artifact: Option<PathBuf>,
}

impl CiStepFailure {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            artifact: None,
        }
    }

    fn with_artifact(message: impl Into<String>, artifact: PathBuf) -> Self {
        Self {
            message: message.into(),
            artifact: Some(artifact),
        }
    }
}

fn render_diagnostic_check(check: &DiagnosticCheck) -> String {
    let mut section = vec![format!(
        "{}\n  Status           {}\n  Summary          {}",
        check.name,
        check.level.label(),
        check.summary
    )];
    if !check.details.is_empty() {
        section.push("  Details".to_string());
        section.extend(check.details.iter().map(|detail| format!("    - {detail}")));
    }
    section.join("\n")
}

fn run_doctor(command: DoctorCommand) -> Result<(), Box<dyn std::error::Error>> {
    let cwd = env::current_dir()?;
    match command {
        DoctorCommand::Check => run_doctor_check(&cwd),
        DoctorCommand::Bundle => run_doctor_bundle(&cwd),
        DoctorCommand::Providers { json } => crate::provider_diagnostics::run(json),
    }
}

fn run_ci(command: CiCommand) -> Result<(), Box<dyn std::error::Error>> {
    match command {
        CiCommand::Check {
            output_format,
            save_report,
        } => run_ci_check(output_format, save_report),
        CiCommand::History {
            output_format,
            limit,
        } => run_ci_history(output_format, limit),
    }
}

fn run_release(command: ReleaseCommand) -> Result<(), Box<dyn std::error::Error>> {
    match command {
        ReleaseCommand::Check {
            output_format,
            save_report,
        } => run_release_check(output_format, save_report),
    }
}

fn run_release_check(
    output_format: CliOutputFormat,
    save_report: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let cwd = env::current_dir()?;
    let mut report = build_release_check_report(&cwd)?;
    if save_report {
        write_release_check_report(&cwd, current_epoch_millis(), &mut report)?;
    }

    match output_format {
        CliOutputFormat::Text => println!("{}", render_release_check_report(&report)),
        CliOutputFormat::Json => print_json(&report)?,
    }

    if report.ok {
        Ok(())
    } else {
        Err("release check failed".into())
    }
}

fn build_release_check_report(
    cwd: &Path,
) -> Result<ReleaseCheckReport, Box<dyn std::error::Error>> {
    let latest_ci = load_ci_history_report(cwd, 1).reports.into_iter().next();
    let latest_eval = rebuild_eval_history_index(cwd)?
        .runs
        .into_iter()
        .next_back();
    let config_report = ConfigLoader::default_for(cwd).check();
    let golden_trace_regressions = latest_ci
        .as_ref()
        .and_then(|report| release_step_summary(report, "golden trace regressions"));
    let diagnostics_redaction = latest_ci
        .as_ref()
        .and_then(|report| release_step_summary(report, "diagnostics bundle"));

    let latest_ci_summary = latest_ci.as_ref().map(release_ci_summary);
    let latest_eval_summary = latest_eval.as_ref().map(release_eval_summary);
    let diagnostics_bundle = latest_ci
        .as_ref()
        .and_then(|report| report.diagnostics_bundle.clone());
    let config = ReleaseConfigSummary {
        ok: config_report.is_ok(),
        loaded_files: config_report.loaded_entries.len(),
        discovered_files: config_report.discovered_entries.len(),
        issues: config_report.issues.len(),
    };
    let ok = latest_ci_summary.as_ref().is_some_and(|summary| summary.ok)
        && latest_eval_summary
            .as_ref()
            .is_some_and(|summary| summary.ok)
        && config.ok
        && golden_trace_regressions
            .as_ref()
            .is_some_and(|step| step.ok)
        && diagnostics_redaction.as_ref().is_some_and(|step| step.ok);

    Ok(ReleaseCheckReport {
        kind: "release_check",
        ok,
        cwd: cwd.to_path_buf(),
        version: VERSION.to_string(),
        git: release_git_status(cwd),
        latest_ci: latest_ci_summary,
        latest_eval: latest_eval_summary,
        config,
        diagnostics_bundle,
        golden_trace_regressions,
        diagnostics_redaction,
        report_path: None,
    })
}

fn release_ci_summary(report: &CiCheckReport) -> ReleaseCiSummary {
    ReleaseCiSummary {
        ok: report.ok,
        report_path: report.report_path.clone(),
        duration_ms: report.steps.iter().map(|step| step.duration_ms).sum(),
        failed_steps: report
            .steps
            .iter()
            .filter(|step| !step.ok)
            .map(|step| step.name.clone())
            .collect(),
    }
}

fn release_eval_summary(entry: &EvalHistoryEntry) -> ReleaseEvalSummary {
    ReleaseEvalSummary {
        ok: entry.failed == 0,
        run_id: entry.run_id.clone(),
        suite: entry.suite.clone(),
        model: entry.model.clone(),
        passed: entry.passed,
        failed: entry.failed,
        duration_ms: entry.duration_ms,
        report_file: entry.report_file.clone(),
    }
}

fn release_step_summary(report: &CiCheckReport, name: &str) -> Option<ReleaseStepSummary> {
    report
        .steps
        .iter()
        .find(|step| step.name == name)
        .map(|step| ReleaseStepSummary {
            ok: step.ok,
            duration_ms: step.duration_ms,
            error: step.error.clone(),
            artifact: step.artifact.clone(),
        })
}

fn release_git_status(cwd: &Path) -> ReleaseGitStatus {
    let branch = git_output(cwd, &["rev-parse", "--abbrev-ref", "HEAD"]);
    let commit = git_output(cwd, &["rev-parse", "--short", "HEAD"]);
    let dirty = git_output(cwd, &["status", "--porcelain"]).map(|output| !output.is_empty());
    let error = branch
        .as_ref()
        .err()
        .or_else(|| commit.as_ref().err())
        .or_else(|| dirty.as_ref().err())
        .cloned();
    ReleaseGitStatus {
        branch: branch.ok(),
        commit: commit.ok(),
        dirty: dirty.ok(),
        error,
    }
}

fn git_output(cwd: &Path, args: &[&str]) -> Result<String, String> {
    let output = Command::new("git")
        .args(args)
        .current_dir(cwd)
        .output()
        .map_err(|error| error.to_string())?;
    if !output.status.success() {
        return Err(format!(
            "git {} failed with {}{}",
            args.join(" "),
            output.status,
            stderr_summary(&output.stderr)
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn write_release_check_report(
    cwd: &Path,
    run_id: u64,
    report: &mut ReleaseCheckReport,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let dir = cwd.join(".pebble").join("release");
    fs::create_dir_all(&dir)?;
    let path = dir.join(format!("release-check-{run_id}.json"));
    report.report_path = Some(path.clone());
    write_atomic(&path, serde_json::to_vec_pretty(report)?)?;
    Ok(path)
}

fn render_release_check_report(report: &ReleaseCheckReport) -> String {
    let mut output = String::new();
    let _ = writeln!(output, "{}", report_title("Release Check"));
    let _ = writeln!(output, "  {} {}", report_label("cwd"), report.cwd.display());
    let _ = writeln!(output, "  {} {}", report_label("version"), report.version);
    let _ = writeln!(
        output,
        "  {} {}",
        report_label("result"),
        if report.ok { "ok" } else { "failed" }
    );
    if let Some(path) = &report.report_path {
        let _ = writeln!(output, "  {} {}", report_label("report"), path.display());
    }
    let _ = writeln!(
        output,
        "\n  {} branch={} commit={} dirty={}",
        report_section("Git"),
        report.git.branch.as_deref().unwrap_or("<unknown>"),
        report.git.commit.as_deref().unwrap_or("<unknown>"),
        report
            .git
            .dirty
            .map_or_else(|| "unknown".to_string(), |dirty| dirty.to_string())
    );
    if let Some(error) = &report.git.error {
        let _ = writeln!(output, "    {}", truncate_for_summary(error, 180));
    }
    if let Some(ci) = &report.latest_ci {
        let _ = writeln!(
            output,
            "\n  {} status={} duration_ms={} report={}",
            report_section("Latest CI"),
            if ci.ok { "ok" } else { "failed" },
            ci.duration_ms,
            ci.report_path.as_ref().map_or_else(
                || "<unsaved>".to_string(),
                |path| path.display().to_string()
            )
        );
        if !ci.failed_steps.is_empty() {
            let _ = writeln!(output, "    failed_steps {}", ci.failed_steps.join(", "));
        }
    } else {
        let _ = writeln!(output, "\n  {} missing", report_section("Latest CI"));
    }
    if let Some(eval) = &report.latest_eval {
        let _ = writeln!(
            output,
            "\n  {} status={} suite={} model={} passed={} failed={} report={}",
            report_section("Latest Eval"),
            if eval.ok { "ok" } else { "failed" },
            eval.suite,
            eval.model,
            eval.passed,
            eval.failed,
            eval.report_file.display()
        );
    } else {
        let _ = writeln!(output, "\n  {} missing", report_section("Latest Eval"));
    }
    let _ = writeln!(
        output,
        "\n  {} status={} loaded={} discovered={} issues={}",
        report_section("Config"),
        if report.config.ok { "ok" } else { "failed" },
        report.config.loaded_files,
        report.config.discovered_files,
        report.config.issues
    );
    render_release_step(
        &mut output,
        "Golden Trace Regressions",
        &report.golden_trace_regressions,
    );
    render_release_step(
        &mut output,
        "Diagnostics Redaction",
        &report.diagnostics_redaction,
    );
    if let Some(bundle) = &report.diagnostics_bundle {
        let _ = writeln!(
            output,
            "\n  {} {}",
            report_label("diagnostics_bundle"),
            bundle.display()
        );
    }
    output.trim_end().to_string()
}

fn render_release_step(output: &mut String, label: &str, step: &Option<ReleaseStepSummary>) {
    if let Some(step) = step {
        let _ = writeln!(
            output,
            "\n  {} status={} duration_ms={}",
            report_section(label),
            if step.ok { "ok" } else { "failed" },
            step.duration_ms
        );
        if let Some(error) = &step.error {
            let _ = writeln!(output, "    {}", truncate_for_summary(error, 180));
        }
        if let Some(artifact) = &step.artifact {
            let _ = writeln!(output, "    artifact {}", artifact.display());
        }
    } else {
        let _ = writeln!(output, "\n  {} missing", report_section(label));
    }
}

fn run_ci_check(
    output_format: CliOutputFormat,
    save_report: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let cwd = env::current_dir()?;
    let render_text = matches!(output_format, CliOutputFormat::Text);
    let run_id = current_epoch_millis();
    let artifact_dir = (!render_text).then(|| {
        cwd.join(".pebble")
            .join("ci")
            .join(format!("ci-check-{run_id}"))
    });
    if render_text {
        println!("{}", report_title("CI Check"));
    }

    let mut report = CiCheckReport {
        kind: "ci_check".to_string(),
        ok: true,
        cwd: cwd.clone(),
        steps: Vec::new(),
        diagnostics_bundle: None,
        report_path: None,
    };

    run_ci_report_step(&mut report, "golden trace regressions", render_text, || {
        run_ci_cargo_command(
            &cwd,
            &["test", "-p", "pebble", "golden"],
            render_text,
            artifact_dir.as_deref(),
            "golden-trace-regressions",
        )
    });
    if report.ok {
        run_ci_report_step(&mut report, "config schema", render_text, || {
            run_ci_config_check(&cwd, render_text)
        });
    }
    if report.ok {
        run_ci_report_step(&mut report, "eval suites", render_text, || {
            run_ci_eval_check(&PathBuf::from("evals/smoke.json"), render_text)
        });
    }
    if report.ok {
        run_ci_report_step(&mut report, "diagnostics bundle", render_text, || {
            run_ci_diagnostics_bundle(&cwd, render_text)
        });
    }

    if save_report {
        let path = write_ci_check_report(&cwd, run_id, &mut report)?;
        if render_text {
            println!("  {} {}", report_label("report"), path.display());
        }
    }

    match output_format {
        CliOutputFormat::Text => {
            println!(
                "  {} {}",
                report_label("result"),
                if report.ok { "ok" } else { "failed" }
            );
        }
        CliOutputFormat::Json => print_json(&report)?,
    }

    if report.ok {
        Ok(())
    } else {
        Err("ci check failed".into())
    }
}

fn write_ci_check_report(
    cwd: &Path,
    run_id: u64,
    report: &mut CiCheckReport,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let dir = cwd.join(".pebble").join("ci");
    fs::create_dir_all(&dir)?;
    let path = dir.join(format!("ci-check-{run_id}.json"));
    report.report_path = Some(path.clone());
    write_atomic(&path, serde_json::to_vec_pretty(report)?)?;
    Ok(path)
}

fn run_ci_history(
    output_format: CliOutputFormat,
    limit: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let cwd = env::current_dir()?;
    let report = load_ci_history_report(&cwd, limit);
    match output_format {
        CliOutputFormat::Text => println!("{}", render_ci_history_report(&report)),
        CliOutputFormat::Json => print_json(&report)?,
    }
    Ok(())
}

fn load_ci_history_report(cwd: &Path, limit: usize) -> CiHistoryReport {
    let dir = cwd.join(".pebble").join("ci");
    let reports = newest_json_artifacts(&dir, limit.saturating_mul(2))
        .into_iter()
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("ci-check-"))
        })
        .filter_map(|path| load_ci_check_report(&path).ok())
        .take(limit)
        .collect::<Vec<_>>();
    CiHistoryReport {
        kind: "ci_history",
        cwd: cwd.to_path_buf(),
        limit,
        reports,
    }
}

fn load_ci_check_report(path: &Path) -> Result<CiCheckReport, Box<dyn std::error::Error>> {
    let mut report = serde_json::from_str::<CiCheckReport>(&fs::read_to_string(path)?)?;
    if report.report_path.is_none() {
        report.report_path = Some(path.to_path_buf());
    }
    Ok(report)
}

fn render_ci_history_report(report: &CiHistoryReport) -> String {
    let mut output = String::new();
    let _ = writeln!(output, "{}", report_title("CI History"));
    let _ = writeln!(output, "  {} {}", report_label("cwd"), report.cwd.display());
    let _ = writeln!(
        output,
        "  {} {}",
        report_label("reports"),
        report.reports.len()
    );
    if report.reports.is_empty() {
        let _ = writeln!(output, "  {} no saved CI reports", report_label("note"));
        return output.trim_end().to_string();
    }
    for saved in &report.reports {
        let total_ms = saved
            .steps
            .iter()
            .map(|step| step.duration_ms)
            .sum::<u128>();
        let path = saved.report_path.as_ref().map_or_else(
            || "<unknown>".to_string(),
            |path| path.display().to_string(),
        );
        let _ = writeln!(
            output,
            "\n  {} {}",
            report_section(if saved.ok { "ok" } else { "failed" }),
            path
        );
        let _ = writeln!(output, "    duration_ms {total_ms}");
        if let Some(bundle) = &saved.diagnostics_bundle {
            let _ = writeln!(output, "    bundle {}", bundle.display());
        }
        for step in &saved.steps {
            let _ = writeln!(
                output,
                "    - {} {}ms {}",
                if step.ok { "ok" } else { "failed" },
                step.duration_ms,
                step.name
            );
            if let Some(error) = &step.error {
                let _ = writeln!(output, "      {}", truncate_for_summary(error, 180));
            }
            if let Some(artifact) = &step.artifact {
                let _ = writeln!(output, "      artifact {}", artifact.display());
            }
        }
    }
    output.trim_end().to_string()
}

fn run_ci_report_step(
    report: &mut CiCheckReport,
    name: &'static str,
    render_text: bool,
    run: impl FnOnce() -> Result<CiStepOutput, CiStepFailure>,
) {
    if render_text {
        println!("  {} {name}", report_label("step"));
    }
    let started = Instant::now();
    let result = run();
    let duration_ms = started.elapsed().as_millis();
    match result {
        Ok(output) => {
            if let Some(path) = output.diagnostics_bundle {
                report.diagnostics_bundle = Some(path);
            }
            if render_text {
                println!("  {} {name}", report_label("ok"));
            }
            report.steps.push(CiCheckStepReport {
                name: name.to_string(),
                ok: true,
                duration_ms,
                error: None,
                artifact: None,
            });
        }
        Err(error) => {
            let message = error.message;
            if render_text {
                println!("  {} {name}", report_label("failed"));
                println!("    {message}");
            }
            report.ok = false;
            report.steps.push(CiCheckStepReport {
                name: name.to_string(),
                ok: false,
                duration_ms,
                error: Some(message),
                artifact: error.artifact,
            });
        }
    }
}

fn run_ci_cargo_command(
    cwd: &Path,
    args: &[&str],
    render_output: bool,
    artifact_dir: Option<&Path>,
    artifact_name: &str,
) -> Result<CiStepOutput, CiStepFailure> {
    if render_output {
        let status = Command::new("cargo")
            .args(args)
            .current_dir(cwd)
            .status()
            .map_err(|error| CiStepFailure::new(error.to_string()))?;
        if status.success() {
            return Ok(CiStepOutput::default());
        }
        return Err(CiStepFailure::new(format!(
            "cargo {} failed with {status}",
            args.join(" ")
        )));
    }

    let output = Command::new("cargo")
        .args(args)
        .current_dir(cwd)
        .output()
        .map_err(|error| CiStepFailure::new(error.to_string()))?;
    if output.status.success() {
        return Ok(CiStepOutput::default());
    }
    let message = format!("cargo {} failed with {}", args.join(" "), output.status);
    if let Some(dir) = artifact_dir {
        match write_ci_step_artifact(dir, artifact_name, &message, &output.stdout, &output.stderr) {
            Ok(path) => return Err(CiStepFailure::with_artifact(message, path)),
            Err(error) => {
                return Err(CiStepFailure::new(format!(
                    "{message}; failed to write artifact: {error}{}{}",
                    stderr_summary(&output.stderr),
                    stdout_summary(&output.stdout)
                )));
            }
        }
    }
    Err(CiStepFailure::new(format!(
        "{message}{}{}",
        stderr_summary(&output.stderr),
        stdout_summary(&output.stdout)
    )))
}

fn run_ci_config_check(cwd: &Path, render_output: bool) -> Result<CiStepOutput, CiStepFailure> {
    let loader = ConfigLoader::default_for(cwd);
    let report = loader.check();
    if report.is_ok() {
        return Ok(CiStepOutput::default());
    }
    if render_output {
        println!("{}", render_config_check_report(cwd, &report));
    }
    Err(CiStepFailure::new("config check failed"))
}

fn run_ci_eval_check(
    suite_path: &Path,
    render_output: bool,
) -> Result<CiStepOutput, CiStepFailure> {
    let suite =
        load_eval_suite(suite_path).map_err(|error| CiStepFailure::new(error.to_string()))?;
    if render_output {
        crate::eval::print_eval_suite_check(&suite, suite_path);
    }
    Ok(CiStepOutput::default())
}

fn run_ci_diagnostics_bundle(
    cwd: &Path,
    render_output: bool,
) -> Result<CiStepOutput, CiStepFailure> {
    let bundle =
        write_diagnostics_bundle(cwd).map_err(|error| CiStepFailure::new(error.to_string()))?;
    let readme = fs::read_to_string(bundle.path.join("README.txt"))
        .map_err(|error| CiStepFailure::new(error.to_string()))?;
    validate_diagnostics_bundle_readme_contract(&readme).map_err(CiStepFailure::new)?;
    if render_output {
        println!("    bundle {}", bundle.path.display());
    }
    Ok(CiStepOutput {
        diagnostics_bundle: Some(bundle.path),
    })
}

fn write_ci_step_artifact(
    dir: &Path,
    name: &str,
    message: &str,
    stdout: &[u8],
    stderr: &[u8],
) -> io::Result<PathBuf> {
    fs::create_dir_all(dir)?;
    let path = dir.join(format!("{name}.log"));
    let mut log = String::new();
    let _ = writeln!(log, "{message}");
    append_ci_log_section(&mut log, "stderr", stderr);
    append_ci_log_section(&mut log, "stdout", stdout);
    write_atomic(&path, log)?;
    Ok(path)
}

fn append_ci_log_section(log: &mut String, label: &str, bytes: &[u8]) {
    let text = String::from_utf8_lossy(bytes);
    let _ = writeln!(log);
    let _ = writeln!(log, "## {label}");
    if text.trim().is_empty() {
        let _ = writeln!(log, "<empty>");
    } else {
        let _ = write!(log, "{}", text.trim_end());
        let _ = writeln!(log);
    }
}

fn stderr_summary(stderr: &[u8]) -> String {
    stream_summary("stderr", stderr)
}

fn stdout_summary(stdout: &[u8]) -> String {
    stream_summary("stdout", stdout)
}

fn stream_summary(label: &str, bytes: &[u8]) -> String {
    let text = String::from_utf8_lossy(bytes);
    let trimmed = text.trim();
    if trimmed.is_empty() {
        String::new()
    } else {
        format!("\n{label}:\n{}", truncate_for_summary(trimmed, 4_000))
    }
}

fn validate_diagnostics_bundle_readme_contract(readme: &str) -> Result<(), String> {
    for term in [
        "API keys",
        "full prompts",
        "tool inputs",
        "live API key validation",
        "network connectivity probes",
    ] {
        if !readme.contains(term) {
            return Err(format!(
                "diagnostics bundle README missing redaction contract term `{term}`"
            ));
        }
    }
    Ok(())
}

fn run_doctor_check(cwd: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let report = build_doctor_report(cwd, true);
    println!("{}", report.render());
    if report.has_failures() {
        return Err("doctor found failing checks".into());
    }
    Ok(())
}

fn build_doctor_report(cwd: &Path, include_live_checks: bool) -> DoctorReport {
    let config_loader = ConfigLoader::default_for(cwd);
    let config = config_loader.load();
    let mut checks = if include_live_checks {
        vec![
            check_api_key_validity(),
            check_web_tools_health(),
            check_config_files(&config_loader, config.as_ref()),
            check_git_availability(cwd),
            check_mcp_server_health(cwd),
            check_network_connectivity(),
            check_system_info(cwd, config.as_ref().ok()),
        ]
    } else {
        vec![
            DiagnosticCheck::new(
                "Live checks",
                DiagnosticLevel::Warn,
                "live API key and network checks are skipped in diagnostics bundles",
            ),
            check_web_tools_health(),
            check_config_files(&config_loader, config.as_ref()),
            check_git_availability(cwd),
            check_mcp_server_health(cwd),
            check_system_info(cwd, config.as_ref().ok()),
        ]
    };
    checks.sort_by_key(|check| check.name);
    DoctorReport { checks }
}

fn run_doctor_bundle(cwd: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let bundle = write_diagnostics_bundle(cwd)?;
    println!("{}", render_diagnostics_bundle_report(&bundle));
    Ok(())
}

fn write_diagnostics_bundle(cwd: &Path) -> Result<DiagnosticsBundle, Box<dyn std::error::Error>> {
    let root = cwd
        .join(".pebble")
        .join("diagnostics")
        .join(format!("bundle-{}", current_epoch_millis()));
    fs::create_dir_all(&root)?;

    let mut files = Vec::new();
    let doctor = build_doctor_report(cwd, false);
    write_bundle_file(&root, &mut files, "README.txt", diagnostics_bundle_readme())?;
    write_bundle_file(&root, &mut files, "doctor.txt", doctor.render())?;
    write_bundle_json_file(
        &root,
        &mut files,
        "doctor.json",
        &doctor_json_report(&doctor),
    )?;

    let config_loader = ConfigLoader::default_for(cwd);
    let config_check = config_loader.check();
    write_bundle_file(
        &root,
        &mut files,
        "config-check.txt",
        render_config_check_report(cwd, &config_check),
    )?;
    write_bundle_json_file(
        &root,
        &mut files,
        "config-check.json",
        &config_check_json_report(cwd, &config_check),
    )?;

    write_bundle_json_file(&root, &mut files, "system.json", &system_bundle_json(cwd))?;
    write_bundle_json_file(&root, &mut files, "sessions.json", &session_bundle_json())?;
    write_bundle_json_file(&root, &mut files, "traces.json", &trace_bundle_json(cwd))?;
    write_bundle_json_file(&root, &mut files, "evals.json", &eval_bundle_json(cwd))?;

    let mcp = mcp_bundle_reports(cwd);
    write_bundle_file(&root, &mut files, "mcp-status.txt", mcp.0)?;
    write_bundle_json_file(&root, &mut files, "mcp-status.json", &mcp.1)?;

    Ok(DiagnosticsBundle { path: root, files })
}

fn write_bundle_file(
    root: &Path,
    files: &mut Vec<PathBuf>,
    name: &str,
    contents: impl AsRef<[u8]>,
) -> Result<(), Box<dyn std::error::Error>> {
    let path = root.join(name);
    write_atomic(&path, contents)?;
    files.push(path);
    Ok(())
}

fn write_bundle_json_file(
    root: &Path,
    files: &mut Vec<PathBuf>,
    name: &str,
    value: &JsonValue,
) -> Result<(), Box<dyn std::error::Error>> {
    write_bundle_file(root, files, name, serde_json::to_vec_pretty(value)?)
}

fn render_diagnostics_bundle_report(bundle: &DiagnosticsBundle) -> String {
    let mut output = String::new();
    let _ = writeln!(output, "{}", report_title("Diagnostics Bundle"));
    let _ = writeln!(
        output,
        "  {} {}",
        report_label("path"),
        bundle.path.display()
    );
    let _ = writeln!(output, "  {} {}", report_label("files"), bundle.files.len());
    for file in &bundle.files {
        let _ = writeln!(output, "    {}", file.display());
    }
    output.trim_end().to_string()
}

fn diagnostics_bundle_readme() -> &'static str {
    "Pebble diagnostics bundle\n\nIncluded:\n- offline doctor checks\n- config validation results\n- loaded config file paths, not raw config contents\n- local system metadata\n- managed session metadata without message bodies\n- recent trace summaries without prompt/output previews\n- recent eval report summaries without final answers\n- MCP server discovery status without tool invocation\n\nExcluded:\n- API keys, OAuth tokens, credentials, and environment values\n- full prompts, assistant responses, tool inputs, and tool outputs\n- live API key validation and network connectivity probes\n"
}

fn doctor_json_report(report: &DoctorReport) -> JsonValue {
    serde_json::json!({
        "kind": "doctor",
        "ok": !report.has_failures(),
        "checks": report.checks.iter().map(|check| serde_json::json!({
            "name": check.name,
            "level": check.level.label(),
            "summary": check.summary,
            "details": check.details,
        })).collect::<Vec<_>>(),
    })
}

fn system_bundle_json(cwd: &Path) -> JsonValue {
    let git_branch = Command::new("git")
        .args(["branch", "--show-current"])
        .current_dir(cwd)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .filter(|branch| !branch.is_empty());
    let git_status = Command::new("git")
        .args(["status", "--short"])
        .current_dir(cwd)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| {
            let text = String::from_utf8_lossy(&output.stdout);
            serde_json::json!({
                "changed_lines": text.lines().count(),
                "clean": text.trim().is_empty(),
            })
        });
    serde_json::json!({
        "kind": "system",
        "cwd": cwd,
        "version": VERSION,
        "os": env::consts::OS,
        "arch": env::consts::ARCH,
        "shell": env::var("SHELL").ok().and_then(|value| {
            Path::new(&value)
                .file_name()
                .and_then(|name| name.to_str())
                .map(str::to_string)
        }),
        "git": {
            "branch": git_branch,
            "status": git_status,
        },
    })
}

fn session_bundle_json() -> JsonValue {
    match list_managed_sessions() {
        Ok(sessions) => serde_json::json!({
            "kind": "sessions",
            "count": sessions.len(),
            "sessions": sessions.into_iter().take(10).map(|session| serde_json::json!({
                "id": session.id,
                "path": session.path,
                "modified_epoch_secs": session.modified_epoch_secs,
                "message_count": session.message_count,
                "title": session.title,
                "model": session.model,
                "started_at": session.started_at,
                "last_prompt_present": session.last_prompt.is_some(),
            })).collect::<Vec<_>>(),
        }),
        Err(error) => serde_json::json!({
            "kind": "sessions",
            "error": error.to_string(),
        }),
    }
}

fn trace_bundle_json(cwd: &Path) -> JsonValue {
    let dir = cwd.join(".pebble").join("runs");
    let traces = newest_json_artifacts(&dir, 10)
        .into_iter()
        .map(|path| match load_turn_trace(&path) {
            Ok(trace) => serde_json::json!({
                "path": path,
                "schema_version": trace.schema_version,
                "started_at_unix_ms": trace.started_at_unix_ms,
                "duration_ms": trace.duration_ms,
                "initial_message_count": trace.initial_message_count,
                "final_message_count": trace.final_message_count,
                "user_input": {
                    "chars": trace.user_input.chars,
                    "sha256": crate::report::short_sha(&trace.user_input.sha256),
                    "truncated": trace.user_input.truncated,
                    "redacted": trace.user_input.redacted,
                },
                "api_calls": trace.api_calls.len(),
                "tool_calls": trace.tool_calls.len(),
                "permissions": trace.permissions.len(),
                "compactions": trace.compactions.len(),
                "errors": trace.errors.len(),
            }),
            Err(error) => serde_json::json!({
                "path": path,
                "error": error.to_string(),
            }),
        })
        .collect::<Vec<_>>();
    serde_json::json!({
        "kind": "traces",
        "directory": dir,
        "count": traces.len(),
        "traces": traces,
    })
}

fn eval_bundle_json(cwd: &Path) -> JsonValue {
    let dir = cwd.join(".pebble").join("evals");
    let reports = newest_json_artifacts(&dir, 10)
        .into_iter()
        .filter(|path| {
            path.file_name()
                .is_none_or(|name| name != EVAL_HISTORY_INDEX_FILE)
        })
        .map(|path| match load_eval_report(&path) {
            Ok(report) => serde_json::json!({
                "path": path,
                "schema_version": report.schema_version,
                "run_id": report.run_id,
                "suite": report.suite,
                "model": report.model,
                "started_at_unix_ms": report.started_at_unix_ms,
                "duration_ms": report.duration_ms,
                "passed": report.passed,
                "failed": report.failed,
                "cases": report.cases.len(),
                "errored_cases": report.cases.iter().filter(|case| case.error.is_some()).count(),
            }),
            Err(error) => serde_json::json!({
                "path": path,
                "error": error.to_string(),
            }),
        })
        .collect::<Vec<_>>();
    serde_json::json!({
        "kind": "evals",
        "directory": dir,
        "count": reports.len(),
        "reports": reports,
    })
}

fn mcp_bundle_reports(cwd: &Path) -> (String, JsonValue) {
    match load_mcp_catalog(cwd) {
        Ok(catalog) => {
            let mut text = String::from("MCP Status\n");
            if catalog.servers.is_empty() {
                text.push_str("  no MCP servers configured\n");
            }
            for server in &catalog.servers {
                let _ = writeln!(
                    text,
                    "  {} enabled={} loaded={} transport={:?} tools={} note={}",
                    server.server_name,
                    server.enabled,
                    server.loaded,
                    server.transport,
                    server.tool_count,
                    server.note
                );
            }
            let json = serde_json::json!({
                "kind": "mcp_status",
                "servers": catalog.servers.iter().map(|server| serde_json::json!({
                    "name": server.server_name,
                    "scope": config_source_label(server.scope),
                    "enabled": server.enabled,
                    "transport": format!("{:?}", server.transport),
                    "loaded": server.loaded,
                    "tool_count": server.tool_count,
                    "note": server.note,
                })).collect::<Vec<_>>(),
                "tools": catalog.tools.iter().map(|tool| serde_json::json!({
                    "exposed_name": tool.exposed_name,
                    "server_name": tool.server_name,
                    "upstream_name": tool.upstream_name,
                    "description": tool.description,
                })).collect::<Vec<_>>(),
            });
            (text.trim_end().to_string(), json)
        }
        Err(error) => (
            format!("MCP Status\n  error: {error}"),
            serde_json::json!({
                "kind": "mcp_status",
                "error": error.to_string(),
            }),
        ),
    }
}

fn newest_json_artifacts(dir: &Path, limit: usize) -> Vec<PathBuf> {
    let Ok(mut candidates) = artifact_candidates(dir) else {
        return Vec::new();
    };
    candidates.sort_by(|left, right| {
        right
            .modified
            .cmp(&left.modified)
            .then_with(|| right.path.cmp(&left.path))
    });
    candidates
        .into_iter()
        .take(limit)
        .map(|candidate| candidate.path)
        .collect()
}

fn check_api_key_validity() -> DiagnosticCheck {
    let api_key = match resolve_nanogpt_api_key() {
        Ok(value) => value,
        Err(ApiError::MissingApiKey) => {
            return DiagnosticCheck::new(
                "API key validity",
                DiagnosticLevel::Warn,
                "no NanoGPT API key is configured",
            );
        }
        Err(error) => {
            return DiagnosticCheck::new(
                "API key validity",
                DiagnosticLevel::Fail,
                format!("failed to resolve NanoGPT API key: {error}"),
            );
        }
    };

    let request = MessageRequest {
        model: default_model_or(DEFAULT_MODEL),
        max_tokens: 1,
        messages: vec![InputMessage::user_text("Reply with OK.")],
        system: None,
        tools: None,
        tool_choice: None,
        thinking: None,
        reasoning_effort: None,
        fast_mode: false,
        stream: false,
    };

    let runtime = match tokio::runtime::Runtime::new() {
        Ok(runtime) => runtime,
        Err(error) => {
            return DiagnosticCheck::new(
                "API key validity",
                DiagnosticLevel::Fail,
                format!("failed to create async runtime: {error}"),
            );
        }
    };

    match runtime.block_on(NanoGptClient::new(api_key).send_message(&request)) {
        Ok(response) => DiagnosticCheck::new(
            "API key validity",
            DiagnosticLevel::Ok,
            "NanoGPT API accepted the configured API key",
        )
        .with_details(vec![format!(
            "request_id={} input_tokens={} output_tokens={}",
            response.request_id.unwrap_or_else(|| "<none>".to_string()),
            response.usage.input_tokens,
            response.usage.output_tokens
        )]),
        Err(ApiError::Api { status, .. }) if status == 401 || status == 403 => {
            DiagnosticCheck::new(
                "API key validity",
                DiagnosticLevel::Fail,
                format!("NanoGPT API rejected the API key with HTTP {status}"),
            )
        }
        Err(error) => DiagnosticCheck::new(
            "API key validity",
            DiagnosticLevel::Warn,
            format!("unable to conclusively validate the API key: {error}"),
        ),
    }
}

fn validate_config_file(path: &Path) -> ConfigFileCheck {
    match fs::read_to_string(path) {
        Ok(contents) => {
            if contents.trim().is_empty() {
                return ConfigFileCheck {
                    path: path.to_path_buf(),
                    exists: true,
                    valid: true,
                    note: "exists but is empty".to_string(),
                };
            }
            match serde_json::from_str::<serde_json::Value>(&contents) {
                Ok(serde_json::Value::Object(_)) => ConfigFileCheck {
                    path: path.to_path_buf(),
                    exists: true,
                    valid: true,
                    note: "valid JSON object".to_string(),
                },
                Ok(_) => ConfigFileCheck {
                    path: path.to_path_buf(),
                    exists: true,
                    valid: false,
                    note: "top-level JSON value is not an object".to_string(),
                },
                Err(error) => ConfigFileCheck {
                    path: path.to_path_buf(),
                    exists: true,
                    valid: false,
                    note: format!("invalid JSON: {error}"),
                },
            }
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => ConfigFileCheck {
            path: path.to_path_buf(),
            exists: false,
            valid: true,
            note: "not present".to_string(),
        },
        Err(error) => ConfigFileCheck {
            path: path.to_path_buf(),
            exists: true,
            valid: false,
            note: format!("unreadable: {error}"),
        },
    }
}

fn check_config_files(
    config_loader: &ConfigLoader,
    config: Result<&runtime::RuntimeConfig, &runtime::ConfigError>,
) -> DiagnosticCheck {
    let file_checks = config_loader
        .discover()
        .into_iter()
        .map(|entry| validate_config_file(&entry.path))
        .collect::<Vec<_>>();
    let existing_count = file_checks.iter().filter(|check| check.exists).count();
    let invalid_count = file_checks
        .iter()
        .filter(|check| check.exists && !check.valid)
        .count();
    let mut details = file_checks
        .iter()
        .map(|check| format!("{} => {}", check.path.display(), check.note))
        .collect::<Vec<_>>();
    match config {
        Ok(runtime_config) => details.push(format!(
            "merged load succeeded with {} loaded file(s)",
            runtime_config.loaded_entries().len()
        )),
        Err(error) => details.push(format!("merged load failed: {error}")),
    }
    DiagnosticCheck::new(
        "Config files",
        if invalid_count > 0 || config.is_err() {
            DiagnosticLevel::Fail
        } else if existing_count == 0 {
            DiagnosticLevel::Warn
        } else {
            DiagnosticLevel::Ok
        },
        format!(
            "discovered {} candidate file(s), {} existing, {} invalid",
            file_checks.len(),
            existing_count,
            invalid_count
        ),
    )
    .with_details(details)
}

fn check_git_availability(cwd: &Path) -> DiagnosticCheck {
    match Command::new("git").arg("--version").output() {
        Ok(version_output) if version_output.status.success() => {
            let version = String::from_utf8_lossy(&version_output.stdout)
                .trim()
                .to_string();
            match Command::new("git")
                .args(["rev-parse", "--show-toplevel"])
                .current_dir(cwd)
                .output()
            {
                Ok(root_output) if root_output.status.success() => DiagnosticCheck::new(
                    "Git availability",
                    DiagnosticLevel::Ok,
                    "git is installed and the current directory is inside a repository",
                )
                .with_details(vec![
                    version,
                    format!(
                        "repo_root={}",
                        String::from_utf8_lossy(&root_output.stdout).trim()
                    ),
                ]),
                Ok(_) => DiagnosticCheck::new(
                    "Git availability",
                    DiagnosticLevel::Warn,
                    "git is installed but the current directory is not a repository",
                )
                .with_details(vec![version]),
                Err(error) => DiagnosticCheck::new(
                    "Git availability",
                    DiagnosticLevel::Warn,
                    format!("git is installed but repo detection failed: {error}"),
                )
                .with_details(vec![version]),
            }
        }
        Ok(output) => DiagnosticCheck::new(
            "Git availability",
            DiagnosticLevel::Fail,
            format!("git --version exited with status {}", output.status),
        ),
        Err(error) => DiagnosticCheck::new(
            "Git availability",
            DiagnosticLevel::Fail,
            format!("failed to execute git: {error}"),
        ),
    }
}

fn check_mcp_server_health(cwd: &Path) -> DiagnosticCheck {
    match load_mcp_catalog(cwd) {
        Ok(catalog) if catalog.servers.is_empty() => DiagnosticCheck::new(
            "MCP server health",
            DiagnosticLevel::Warn,
            "no MCP servers are configured",
        ),
        Ok(catalog) => {
            let level = if catalog.servers.iter().any(|server| !server.loaded) {
                DiagnosticLevel::Warn
            } else {
                DiagnosticLevel::Ok
            };
            DiagnosticCheck::new(
                "MCP server health",
                level,
                format!("checked {} configured MCP server(s)", catalog.servers.len()),
            )
            .with_details(
                catalog
                    .servers
                    .iter()
                    .map(|server| {
                        format!(
                            "{} [{:?}] {} tool(s): {}",
                            server.server_name, server.transport, server.tool_count, server.note
                        )
                    })
                    .collect(),
            )
        }
        Err(error) => DiagnosticCheck::new(
            "MCP server health",
            DiagnosticLevel::Fail,
            format!("failed to inspect MCP servers: {error}"),
        ),
    }
}

fn check_network_connectivity() -> DiagnosticCheck {
    let address = match ("nano-gpt.com", 443).to_socket_addrs() {
        Ok(mut addrs) => match addrs.next() {
            Some(addr) => addr,
            None => {
                return DiagnosticCheck::new(
                    "Network connectivity",
                    DiagnosticLevel::Fail,
                    "DNS resolution returned no addresses for nano-gpt.com",
                );
            }
        },
        Err(error) => {
            return DiagnosticCheck::new(
                "Network connectivity",
                DiagnosticLevel::Fail,
                format!("failed to resolve nano-gpt.com: {error}"),
            );
        }
    };
    match TcpStream::connect_timeout(&address, Duration::from_secs(5)) {
        Ok(stream) => {
            let _ = stream.shutdown(std::net::Shutdown::Both);
            DiagnosticCheck::new(
                "Network connectivity",
                DiagnosticLevel::Ok,
                format!("connected to {address}"),
            )
        }
        Err(error) => DiagnosticCheck::new(
            "Network connectivity",
            DiagnosticLevel::Fail,
            format!("failed to connect to {address}: {error}"),
        ),
    }
}

fn check_web_tools_health() -> DiagnosticCheck {
    let service = current_service_or_default();
    let base_url = resolve_base_url_for(service);
    let api_key_configured = resolve_api_key_for(service).is_ok();
    match current_tool_registry() {
        Ok(registry) => {
            let mut has_search = false;
            let mut has_scrape = false;
            for entry in registry.entries() {
                if entry.definition.name == "WebSearch" {
                    has_search = true;
                }
                if entry.definition.name == "WebScrape" {
                    has_scrape = true;
                }
            }
            let level = if has_search && has_scrape && api_key_configured {
                DiagnosticLevel::Ok
            } else if has_search || has_scrape {
                DiagnosticLevel::Warn
            } else {
                DiagnosticLevel::Fail
            };
            DiagnosticCheck::new(
                "Web tools",
                level,
                if has_search && has_scrape {
                    "web tools are registered"
                } else {
                    "one or more web tools are unavailable"
                },
            )
            .with_details(vec![
                format!("service={}", service.display_name()),
                format!("base_url={base_url}"),
                format!(
                    "api_key={}",
                    if api_key_configured {
                        "configured"
                    } else {
                        "missing"
                    }
                ),
                format!(
                    "WebSearch={}",
                    if has_search { "available" } else { "missing" }
                ),
                format!(
                    "WebScrape={}",
                    if has_scrape { "available" } else { "missing" }
                ),
            ])
        }
        Err(error) => DiagnosticCheck::new(
            "Web tools",
            DiagnosticLevel::Fail,
            format!("failed to load tool registry: {error}"),
        )
        .with_details(vec![format!("base_url={base_url}")]),
    }
}

fn check_system_info(cwd: &Path, config: Option<&runtime::RuntimeConfig>) -> DiagnosticCheck {
    let mut details = vec![
        format!("os={} arch={}", env::consts::OS, env::consts::ARCH),
        format!("cwd={}", cwd.display()),
        format!("cli_version={VERSION}"),
    ];
    if let Some(config) = config {
        let sandbox_status = resolve_sandbox_status(config.sandbox(), cwd);
        details.push(format!(
            "resolved_model={} loaded_config_files={}",
            config.model().unwrap_or(DEFAULT_MODEL),
            config.loaded_entries().len()
        ));
        details.push(format!(
            "sandbox enabled={} active={} namespace_active={} network_active={} filesystem_mode={}",
            sandbox_status.enabled,
            sandbox_status.active,
            sandbox_status.namespace_active,
            sandbox_status.network_active,
            sandbox_status.filesystem_mode.as_str()
        ));
        if !sandbox_status.allowed_mounts.is_empty() {
            details.push(format!(
                "sandbox_allowed_mounts={}",
                sandbox_status.allowed_mounts.join(", ")
            ));
        }
        if let Some(reason) = sandbox_status.fallback_reason {
            details.push(format!("sandbox_fallback={reason}"));
        }
    }
    DiagnosticCheck::new(
        "System info",
        DiagnosticLevel::Ok,
        "captured local runtime and build metadata",
    )
    .with_details(details)
}

#[cfg(test)]
mod tests;
