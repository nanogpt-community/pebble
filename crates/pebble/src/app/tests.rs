use super::{
    append_undo_snapshot, available_runtime_tool_specs, build_system_prompt, build_turn_snapshot,
    collect_generated_artifacts, filter_runtime_tool_specs, format_status_report,
    format_web_tools_status, load_custom_slash_commands, load_turn_trace, login_model_guidance,
    normalize_generated_pebble_md, parse_args, parse_auth_command, parse_checksum_for_asset,
    parse_logout_command, parse_mcp_command, parse_model_command, parse_permissions_command,
    parse_provider_command, parse_proxy_command, parse_route_command, persist_runtime_defaults,
    remove_saved_credentials, render_archived_tool_results_report, render_config_check_report,
    render_custom_command_template, render_export_text, render_gc_report,
    render_permission_diff_preview, render_replay_report, render_session_timeline,
    render_trace_report, render_update_report, resolve_model_alias,
    should_ignore_stale_secret_submit, strip_markdown_code_fence, trim_trailing_line_endings,
    tuned_tool_description, write_diagnostics_bundle, AuthService, CiCommand, CliAction,
    CliOutputFormat, CollaborationMode, CredentialRemovalOutcome, DoctorCommand, FastMode,
    GitHubRelease, GitHubReleaseAsset, LoginCommand, LogoutCommand, McpCatalog, McpCommand,
    ReleaseCommand, RuntimeToolSpec, StatusContext, StatusUsage, WorktreeSnapshot,
    DEFAULT_MAX_TOKENS, DEFAULT_MODEL,
};
use crate::eval::{
    eval_replay_json_report, load_eval_report, load_eval_suite, rebuild_eval_history_index,
    render_eval_capture_report, render_eval_compare_report, render_eval_history_report,
    render_eval_replay_report, write_captured_eval_case, EvalCaptureOptions, EvalHistoryFilter,
    EvalReplayOptions, EvalRunCaseReport, EvalRunReport, EVAL_HISTORY_INDEX_FILE,
    EVAL_REPORT_SCHEMA_VERSION, LEGACY_EVAL_REPORT_SCHEMA_VERSION,
};
use crate::proxy::ProxyCommand;
use crate::report::shell_quote;
use crate::runtime_client::{
    append_proxy_text_events, convert_messages, extract_first_json_object, parse_tool_input_value,
    prompt_to_content_blocks, proxy_response_to_events, push_output_block,
    render_streamed_tool_call_start, response_to_events, should_retry_proxy_tool_prompt,
    PebbleRuntimeClient,
};
use crate::tool_render::render_tool_result_block;
use crate::trace_view::eval_case_from_trace;
use api::{
    ApiService, InputContentBlock, MessageResponse, OutputContentBlock, ReasoningEffort, Usage,
};
use compat_harness::{EvalCase, EvalCaseResult, EvalFailureKind};
use runtime::{
    ApiCallTrace, AssistantEvent, ConfigLoader, ContentBlock, ConversationMessage, PermissionMode,
    PermissionRequest, PermissionTrace, RuntimeRetentionConfig, Session, TokenUsage, ToolCallTrace,
    TracePayloadSummary, TurnTrace,
};
use serde_json::json;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tools::current_tool_registry;

fn env_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .expect("env test lock should not be poisoned")
}

fn with_isolated_config_home<T>(run: impl FnOnce() -> T) -> T {
    let _guard = env_lock();
    let root = std::env::temp_dir().join(format!(
        "pebble-main-test-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should be after epoch")
            .as_nanos()
    ));
    std::fs::create_dir_all(&root).expect("config dir should exist");
    std::env::set_var("PEBBLE_CONFIG_HOME", &root);
    let output = run();
    std::env::remove_var("PEBBLE_CONFIG_HOME");
    std::fs::remove_dir_all(root).expect("temp config dir should be removed");
    output
}

fn tool_specs() -> Vec<RuntimeToolSpec> {
    available_runtime_tool_specs(
        &current_tool_registry().expect("tool registry should load"),
        &McpCatalog::default(),
    )
}

#[test]
fn defaults_to_repl_when_no_args() {
    with_isolated_config_home(|| {
        assert_eq!(
            parse_args(&[]).expect("args should parse"),
            CliAction::Repl {
                model: DEFAULT_MODEL.to_string(),
                allowed_tools: None,
                permission_mode: PermissionMode::WorkspaceWrite,
                collaboration_mode: CollaborationMode::Build,
                reasoning_effort: None,
                fast_mode: FastMode::Off,
            }
        );
    });
}

#[test]
fn rejects_unknown_options_instead_of_sending_them_to_the_model() {
    let error = parse_args(&["--modle".to_string(), "glm-5.1".to_string()])
        .expect_err("unknown option should fail before starting a prompt");

    assert_eq!(error, "unknown option: --modle");
}

#[test]
fn parses_eval_check_and_fail_gate_options() {
    with_isolated_config_home(|| {
        let args = vec![
            "eval".to_string(),
            "--check".to_string(),
            "--fail-on-failures".to_string(),
            "evals/smoke.json".to_string(),
        ];
        assert_eq!(
            parse_args(&args).expect("args should parse"),
            CliAction::Eval {
                suite_path: PathBuf::from("evals/smoke.json"),
                model: DEFAULT_MODEL.to_string(),
                allowed_tools: None,
                permission_mode: PermissionMode::WorkspaceWrite,
                collaboration_mode: CollaborationMode::Build,
                reasoning_effort: None,
                fast_mode: FastMode::Off,
                check_only: true,
                fail_on_failures: true,
            }
        );
    });
}

#[test]
fn parses_eval_compare_subcommand() {
    with_isolated_config_home(|| {
        let args = vec![
            "eval".to_string(),
            "compare".to_string(),
            ".pebble/evals/old.json".to_string(),
            ".pebble/evals/new.json".to_string(),
        ];
        assert_eq!(
            parse_args(&args).expect("args should parse"),
            CliAction::EvalCompare {
                old_path: PathBuf::from(".pebble/evals/old.json"),
                new_path: PathBuf::from(".pebble/evals/new.json"),
                output_format: CliOutputFormat::Text,
            }
        );
    });
}

#[test]
fn parses_eval_capture_subcommand() {
    with_isolated_config_home(|| {
        let args = vec![
            "eval".to_string(),
            "capture".to_string(),
            ".pebble/runs/run.json".to_string(),
            "--suite".to_string(),
            "evals/regressions.json".to_string(),
            "--name".to_string(),
            "Handles denied write".to_string(),
            "--force".to_string(),
        ];
        assert_eq!(
            parse_args(&args).expect("args should parse"),
            CliAction::EvalCapture {
                options: EvalCaptureOptions {
                    trace_path: PathBuf::from(".pebble/runs/run.json"),
                    suite_path: PathBuf::from("evals/regressions.json"),
                    name: Some("Handles denied write".to_string()),
                    force: true,
                },
            }
        );
    });
}

#[test]
fn parses_eval_replay_subcommand() {
    with_isolated_config_home(|| {
        let args = vec![
            "eval".to_string(),
            "replay".to_string(),
            ".pebble/evals/eval-1.json".to_string(),
            "--case=handles-denied-write".to_string(),
        ];
        assert_eq!(
            parse_args(&args).expect("args should parse"),
            CliAction::EvalReplay {
                options: EvalReplayOptions {
                    report_path: PathBuf::from(".pebble/evals/eval-1.json"),
                    case_id: Some("handles-denied-write".to_string()),
                },
                output_format: CliOutputFormat::Text,
            }
        );
    });
}

#[test]
fn parses_eval_history_subcommand_with_filters() {
    with_isolated_config_home(|| {
        let args = vec![
            "eval".to_string(),
            "history".to_string(),
            "--suite".to_string(),
            "smoke".to_string(),
            "--model=openai/gpt-5.2".to_string(),
            "--limit".to_string(),
            "5".to_string(),
        ];
        assert_eq!(
            parse_args(&args).expect("args should parse"),
            CliAction::EvalHistory {
                filter: EvalHistoryFilter {
                    suite: Some("smoke".to_string()),
                    model: Some("openai/gpt-5.2".to_string()),
                    limit: 5,
                },
                output_format: CliOutputFormat::Text,
            }
        );
    });
}

#[test]
fn parses_debug_commands_with_json_output_flags() {
    with_isolated_config_home(|| {
        assert_eq!(
            parse_args(&[
                "trace".to_string(),
                ".pebble/runs/run.json".to_string(),
                "--json".to_string(),
            ])
            .expect("trace args should parse"),
            CliAction::Trace {
                trace_path: PathBuf::from(".pebble/runs/run.json"),
                output_format: CliOutputFormat::Json,
            }
        );
        assert_eq!(
            parse_args(&[
                "eval".to_string(),
                "history".to_string(),
                "--json".to_string(),
            ])
            .expect("history args should parse"),
            CliAction::EvalHistory {
                filter: EvalHistoryFilter::default(),
                output_format: CliOutputFormat::Json,
            }
        );
        assert_eq!(
            parse_args(&[
                "eval".to_string(),
                "compare".to_string(),
                "old.json".to_string(),
                "new.json".to_string(),
                "--json".to_string(),
            ])
            .expect("compare args should parse"),
            CliAction::EvalCompare {
                old_path: PathBuf::from("old.json"),
                new_path: PathBuf::from("new.json"),
                output_format: CliOutputFormat::Json,
            }
        );
    });
}

#[test]
fn renders_eval_compare_report_with_regressions_and_fixes() {
    let old_report = EvalRunReport {
        schema_version: EVAL_REPORT_SCHEMA_VERSION,
        run_id: "eval-old".to_string(),
        suite: "smoke".to_string(),
        model: DEFAULT_MODEL.to_string(),
        started_at_unix_ms: 1,
        duration_ms: 100,
        passed: 1,
        failed: 1,
        cases: vec![
            eval_case_report("regressed", true, 1, 1, Vec::new(), None, 0),
            eval_case_report(
                "fixed",
                false,
                2,
                2,
                vec!["missing answer".to_string()],
                None,
                1,
            ),
        ],
    };
    let new_report = EvalRunReport {
        schema_version: EVAL_REPORT_SCHEMA_VERSION,
        run_id: "eval-new".to_string(),
        suite: "smoke".to_string(),
        model: DEFAULT_MODEL.to_string(),
        started_at_unix_ms: 2,
        duration_ms: 150,
        passed: 1,
        failed: 1,
        cases: vec![
            eval_case_report(
                "regressed",
                false,
                3,
                2,
                vec!["forbidden tool `bash` was used".to_string()],
                None,
                2,
            ),
            eval_case_report("fixed", true, 1, 1, Vec::new(), None, 0),
        ],
    };

    let report = render_eval_compare_report(
        Path::new("old.json"),
        &old_report,
        Path::new("new.json"),
        &new_report,
    );

    assert!(report.contains("Eval Compare"));
    assert!(report.contains("schema_version"));
    assert!(report.contains("pass_rate"));
    assert!(report.contains("+0.0pp"));
    assert!(report.contains("duration_ms"));
    assert!(report.contains("+50"));
    assert!(report.contains("Failure Categories"));
    assert!(report.contains("missing_answer_substring"));
    assert!(report.contains("regressions"));
    assert!(report.contains("regressed: forbidden tool `bash` was used"));
    assert!(report.contains("fixes"));
    assert!(report.contains("fixed"));
}

#[test]
fn renders_eval_replay_report_with_trace_timeline() {
    let root = std::env::temp_dir().join(format!(
        "pebble-eval-replay-test-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should work")
            .as_nanos()
    ));
    let evals = root.join("evals");
    std::fs::create_dir_all(&evals).expect("evals dir should exist");
    let report_path = evals.join("eval-1.json");
    let trace_path = evals.join("trace.json");
    let mut trace = TurnTrace::start("avoid bash", 0);
    trace.api_calls.push(ApiCallTrace {
        iteration: 1,
        request_message_count: 1,
        request_estimated_tokens: 20,
        duration_ms: 6,
        result_event_count: Some(1),
        usage: None,
        error: None,
    });
    trace.tool_calls.push(ToolCallTrace {
        iteration: 1,
        tool_use_id: "tool-1".to_string(),
        tool_name: "bash".to_string(),
        input: TracePayloadSummary::from_text(r#"{"cmd":"pwd"}"#),
        effective_input: None,
        output: TracePayloadSummary::from_text("/tmp/project"),
        duration_ms: 3,
        permission_outcome: "allow".to_string(),
        is_error: false,
    });
    std::fs::write(
        &trace_path,
        serde_json::to_vec_pretty(&trace).expect("trace should serialize"),
    )
    .expect("trace should write");
    let eval = EvalRunReport {
        schema_version: EVAL_REPORT_SCHEMA_VERSION,
        run_id: "eval-1".to_string(),
        suite: "regressions".to_string(),
        model: DEFAULT_MODEL.to_string(),
        started_at_unix_ms: 1,
        duration_ms: 10,
        passed: 0,
        failed: 1,
        cases: vec![EvalRunCaseReport {
            case: EvalCase {
                id: "avoid-bash".to_string(),
                prompt: "avoid bash".to_string(),
                forbidden_tools: vec!["bash".to_string()],
                ..EvalCase::default()
            },
            result: EvalCaseResult {
                id: "avoid-bash".to_string(),
                passed: false,
                failures: vec!["forbidden tool `bash` was used".to_string()],
                failure_categories: vec![EvalFailureKind::ForbiddenToolUsed],
                iterations: 1,
                tool_calls: 1,
                api_calls: 1,
                duration_ms: Some(9),
            },
            final_answer: "I ran pwd.".to_string(),
            trace_file: Some(PathBuf::from("trace.json")),
            session_file: Some(PathBuf::from("session.json")),
            error: None,
            changed_files: 2,
        }],
    };
    let output = render_eval_replay_report(
        &EvalReplayOptions {
            report_path: report_path.clone(),
            case_id: None,
        },
        &eval,
    );
    let json_output = eval_replay_json_report(
        &EvalReplayOptions {
            report_path: report_path.clone(),
            case_id: None,
        },
        &eval,
    );

    assert!(output.contains("Eval Replay"));
    assert!(output.contains("avoid-bash"));
    assert!(output.contains("forbidden_tool_used"));
    assert!(output.contains("forbidden tool `bash` was used"));
    assert!(output.contains("changed_files"));
    assert!(output.contains("I ran pwd."));
    assert!(output.contains("Trace Replay"));
    assert!(output.contains("Pebble Replay"));
    assert!(output.contains("bash"));
    assert!(output.contains("/tmp/project"));
    assert_eq!(json_output["kind"], "eval_replay");
    assert_eq!(json_output["cases"][0]["id"], "avoid-bash");
    assert_eq!(
        json_output["cases"][0]["failure_categories"][0],
        "forbidden_tool_used"
    );
    assert_eq!(json_output["cases"][0]["trace"]["loaded"], true);
    assert_eq!(
        json_output["cases"][0]["trace"]["timeline"][2]["kind"],
        "tool"
    );
    std::fs::remove_dir_all(root).expect("temp root should be removed");
}

#[test]
fn load_eval_report_defaults_missing_schema_version_to_legacy() {
    let root = std::env::temp_dir().join(format!(
        "pebble-eval-report-schema-test-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should work")
            .as_nanos()
    ));
    std::fs::create_dir_all(&root).expect("temp root should exist");
    let path = root.join("eval.json");
    std::fs::write(
        &path,
        r#"{
          "run_id": "eval-legacy",
          "suite": "smoke",
          "model": "model",
          "started_at_unix_ms": 1,
          "duration_ms": 2,
          "passed": 0,
          "failed": 0,
          "cases": []
        }"#,
    )
    .expect("eval report should write");

    let report = load_eval_report(&path).expect("legacy eval report should load");

    assert_eq!(report.schema_version, LEGACY_EVAL_REPORT_SCHEMA_VERSION);
    std::fs::remove_dir_all(root).expect("temp root should be removed");
}

#[test]
fn rebuilds_and_renders_eval_history_with_deltas() {
    let root = std::env::temp_dir().join(format!(
        "pebble-eval-history-test-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should work")
            .as_nanos()
    ));
    let evals = root.join(".pebble").join("evals");
    std::fs::create_dir_all(&evals).expect("evals dir should exist");
    std::fs::write(evals.join(EVAL_HISTORY_INDEX_FILE), "{}")
        .expect("index placeholder should write");
    write_eval_report_fixture(
        &evals.join("eval-1.json"),
        EvalRunReport {
            schema_version: EVAL_REPORT_SCHEMA_VERSION,
            run_id: "eval-1".to_string(),
            suite: "smoke".to_string(),
            model: DEFAULT_MODEL.to_string(),
            started_at_unix_ms: 1,
            duration_ms: 10,
            passed: 0,
            failed: 1,
            cases: Vec::new(),
        },
    );
    write_eval_report_fixture(
        &evals.join("eval-2.json"),
        EvalRunReport {
            schema_version: EVAL_REPORT_SCHEMA_VERSION,
            run_id: "eval-2".to_string(),
            suite: "smoke".to_string(),
            model: DEFAULT_MODEL.to_string(),
            started_at_unix_ms: 2,
            duration_ms: 12,
            passed: 1,
            failed: 0,
            cases: Vec::new(),
        },
    );

    let index = rebuild_eval_history_index(&root).expect("history should rebuild");
    let report = render_eval_history_report(
        &EvalHistoryFilter {
            suite: Some("smoke".to_string()),
            model: Some(DEFAULT_MODEL.to_string()),
            limit: 20,
        },
        &index,
    );

    assert_eq!(index.runs.len(), 2);
    assert!(report.contains("Eval History"));
    assert!(report.contains("eval-2"));
    assert!(report.contains("delta=+100.0pp"));
    assert!(report.contains(".pebble/evals/eval-2.json"));
    std::fs::remove_dir_all(root).expect("temp root should be removed");
}

#[test]
fn captures_trace_into_eval_suite_with_sequence_and_permissions() {
    let root = std::env::temp_dir().join(format!(
        "pebble-eval-capture-test-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should work")
            .as_nanos()
    ));
    std::fs::create_dir_all(&root).expect("temp root should exist");
    let suite_path = root.join("evals").join("regressions.json");
    let trace_path = root.join(".pebble").join("runs").join("trace.json");
    std::fs::create_dir_all(trace_path.parent().expect("trace parent should exist"))
        .expect("trace parent should be created");
    let mut trace = TurnTrace::start("Please inspect Cargo.toml", 1);
    trace.api_calls.push(ApiCallTrace {
        iteration: 1,
        request_message_count: 2,
        request_estimated_tokens: 100,
        duration_ms: 25,
        result_event_count: Some(1),
        usage: None,
        error: None,
    });
    trace.permissions.push(PermissionTrace {
        iteration: 1,
        tool_use_id: "tool-1".to_string(),
        tool_name: "read".to_string(),
        outcome: "allow".to_string(),
        reason: None,
    });
    trace.tool_calls.push(ToolCallTrace {
        iteration: 1,
        tool_use_id: "tool-1".to_string(),
        tool_name: "read".to_string(),
        input: TracePayloadSummary::from_text(r#"{"file":"Cargo.toml"}"#),
        effective_input: None,
        output: TracePayloadSummary::from_text("workspace manifest"),
        duration_ms: 5,
        permission_outcome: "allow".to_string(),
        is_error: false,
    });
    std::fs::write(
        &trace_path,
        serde_json::to_vec_pretty(&trace).expect("trace should serialize"),
    )
    .expect("trace should write");

    let loaded = load_turn_trace(&trace_path).expect("trace should load");
    let case = eval_case_from_trace(
        &loaded,
        "handles-inspection",
        Path::new(".pebble/runs/trace.json"),
    );
    let outcome = write_captured_eval_case(&suite_path, case, false).expect("case should capture");
    let report = render_eval_capture_report(
        &EvalCaptureOptions {
            trace_path: trace_path.clone(),
            suite_path: suite_path.clone(),
            name: Some("handles inspection".to_string()),
            force: false,
        },
        &loaded,
        &outcome,
    );
    let suite = load_eval_suite(&suite_path).expect("captured suite should load");

    assert_eq!(suite.cases.len(), 1);
    assert_eq!(suite.cases[0].id, "handles-inspection");
    assert_eq!(suite.cases[0].required_tools, vec!["read"]);
    assert_eq!(suite.cases[0].required_tool_sequence, vec!["read"]);
    assert_eq!(suite.cases[0].required_permission_outcomes.len(), 1);
    assert_eq!(suite.cases[0].max_tool_calls, Some(1));
    assert_eq!(suite.cases[0].max_api_calls, Some(1));
    assert!(suite.cases[0].require_successful_tool);
    assert!(report.contains("Eval Capture"));
    assert!(report.contains("tool_sequence"));
    assert!(report.contains("read=allow"));
    std::fs::remove_dir_all(root).expect("temp root should be removed");
}

fn eval_case_report(
    id: &str,
    passed: bool,
    tool_calls: usize,
    api_calls: usize,
    failures: Vec<String>,
    error: Option<String>,
    changed_files: usize,
) -> EvalRunCaseReport {
    let failure_categories = if failures.is_empty() {
        Vec::new()
    } else {
        vec![EvalFailureKind::MissingAnswerSubstring]
    };
    EvalRunCaseReport {
        case: EvalCase {
            id: id.to_string(),
            prompt: format!("case {id}"),
            ..EvalCase::default()
        },
        result: EvalCaseResult {
            id: id.to_string(),
            passed,
            failures,
            failure_categories,
            iterations: api_calls,
            tool_calls,
            api_calls,
            duration_ms: Some(10),
        },
        final_answer: String::new(),
        trace_file: None,
        session_file: None,
        error,
        changed_files,
    }
}

fn write_eval_report_fixture(path: &Path, report: EvalRunReport) {
    std::fs::write(
        path,
        serde_json::to_vec_pretty(&report).expect("eval report should serialize"),
    )
    .expect("eval report should write");
}

#[test]
fn uses_persisted_runtime_defaults_for_new_repl() {
    with_isolated_config_home(|| {
        persist_runtime_defaults(
            PermissionMode::DangerFullAccess,
            CollaborationMode::Plan,
            Some(ReasoningEffort::High),
            FastMode::On,
        )
        .expect("runtime defaults should persist");

        assert_eq!(
            parse_args(&[]).expect("args should parse"),
            CliAction::Repl {
                model: DEFAULT_MODEL.to_string(),
                allowed_tools: None,
                permission_mode: PermissionMode::DangerFullAccess,
                collaboration_mode: CollaborationMode::Plan,
                reasoning_effort: Some(ReasoningEffort::High),
                fast_mode: FastMode::On,
            }
        );
    });
}

#[test]
fn permission_env_overrides_persisted_runtime_default() {
    with_isolated_config_home(|| {
        persist_runtime_defaults(
            PermissionMode::DangerFullAccess,
            CollaborationMode::Build,
            None,
            FastMode::Off,
        )
        .expect("runtime defaults should persist");
        let original = std::env::var("PEBBLE_PERMISSION_MODE").ok();
        std::env::set_var("PEBBLE_PERMISSION_MODE", "read-only");

        let parsed = parse_args(&[]).expect("args should parse");

        match original {
            Some(value) => std::env::set_var("PEBBLE_PERMISSION_MODE", value),
            None => std::env::remove_var("PEBBLE_PERMISSION_MODE"),
        }
        assert_eq!(
            parsed,
            CliAction::Repl {
                model: DEFAULT_MODEL.to_string(),
                allowed_tools: None,
                permission_mode: PermissionMode::ReadOnly,
                collaboration_mode: CollaborationMode::Build,
                reasoning_effort: None,
                fast_mode: FastMode::Off,
            }
        );
    });
}

#[test]
fn stale_empty_secret_submit_is_ignored_briefly() {
    assert!(should_ignore_stale_secret_submit(
        "",
        Duration::from_millis(25)
    ));
    assert!(!should_ignore_stale_secret_submit(
        "sk-live",
        Duration::from_millis(25)
    ));
    assert!(!should_ignore_stale_secret_submit(
        "",
        Duration::from_millis(300)
    ));
}

#[test]
fn trims_trailing_line_endings_from_pasted_secret() {
    assert_eq!(trim_trailing_line_endings("abc123\r\n"), "abc123");
    assert_eq!(trim_trailing_line_endings("abc123\n"), "abc123");
    assert_eq!(trim_trailing_line_endings("abc123"), "abc123");
}

#[test]
fn runtime_client_constructor_defers_api_key_lookup() {
    with_isolated_config_home(|| {
        let original = std::env::var("NANOGPT_API_KEY").ok();
        std::env::remove_var("NANOGPT_API_KEY");

        let client = PebbleRuntimeClient::new(
            ApiService::NanoGpt,
            DEFAULT_MODEL.to_string(),
            DEFAULT_MAX_TOKENS,
            None,
            true,
            false,
            Vec::new(),
            CollaborationMode::Build,
            None,
            FastMode::Off,
            false,
        );

        match original {
            Some(value) => std::env::set_var("NANOGPT_API_KEY", value),
            None => std::env::remove_var("NANOGPT_API_KEY"),
        }

        assert!(
            client.is_ok(),
            "runtime client should initialize without credentials"
        );
    });
}

#[test]
fn synthetic_login_guidance_explains_active_model_mismatch() {
    with_isolated_config_home(|| {
        let note = login_model_guidance(AuthService::Synthetic)
            .expect("synthetic login should show model guidance by default");
        assert!(note.contains("current model is `zai-org/glm-5.1`"));
        assert!(note.contains("Logging into Synthetic saves credentials"));
        assert!(note.contains("prefixed with `hf:`"));
    });
}

#[test]
fn tuned_web_tool_descriptions_push_search_and_scrape_workflow() {
    let search = tuned_tool_description("WebSearch", "Search the web.");
    let scrape = tuned_tool_description("WebScrape", "Scrape pages.");
    let fetch = tuned_tool_description("WebFetch", "Fetch a URL.");

    assert!(search.contains("current information"));
    assert!(scrape.contains("readable page content"));
    assert!(fetch.contains("WebScrape"));
}

#[test]
fn system_prompt_includes_web_research_guidance() {
    let prompt = build_system_prompt(ApiService::NanoGpt, DEFAULT_MODEL, CollaborationMode::Build)
        .expect("system prompt should build")
        .join("\n\n");
    assert!(prompt.contains("# Web Research Guidance"));
    assert!(prompt.contains("WebSearch"));
    assert!(prompt.contains("WebScrape"));
}

#[test]
fn web_tools_status_mentions_auth_and_tool_availability() {
    let summary = format_web_tools_status();
    assert!(summary.contains("api_key="));
    assert!(summary.contains("web_search="));
    assert!(summary.contains("web_scrape="));
}

#[test]
fn status_report_includes_runtime_health() {
    let report = format_status_report(
        ApiService::NanoGpt,
        DEFAULT_MODEL,
        StatusUsage {
            message_count: 0,
            turns: 0,
            undo_count: 0,
            redo_count: 0,
            latest: TokenUsage {
                input_tokens: 0,
                output_tokens: 0,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
            },
            cumulative: TokenUsage {
                input_tokens: 0,
                output_tokens: 0,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
            },
            estimated_tokens: 0,
            context_window: None,
        },
        "workspace-write",
        Some("<platform default>"),
        false,
        CollaborationMode::Build,
        None,
        FastMode::Off,
        &McpCatalog::default(),
        &StatusContext {
            cwd: PathBuf::from("."),
            session_path: None,
            loaded_config_files: 0,
            discovered_config_files: 0,
            instruction_file_count: 0,
            memory_file_count: 0,
            project_root: None,
            git_branch: None,
            sandbox_summary: "sandbox".to_string(),
            web_tools_summary: "web".to_string(),
        },
    );
    assert!(report.contains("Runtime"));
    assert!(report.contains("model auth"));
    assert!(report.contains("web disabled"));
}

#[test]
fn parses_prompt_subcommand() {
    with_isolated_config_home(|| {
        let args = vec![
            "prompt".to_string(),
            "hello".to_string(),
            "world".to_string(),
        ];
        assert_eq!(
            parse_args(&args).expect("args should parse"),
            CliAction::Prompt {
                prompt: "hello world".to_string(),
                model: DEFAULT_MODEL.to_string(),
                allowed_tools: None,
                permission_mode: PermissionMode::WorkspaceWrite,
                collaboration_mode: CollaborationMode::Build,
                reasoning_effort: None,
                fast_mode: FastMode::Off,
                output_format: CliOutputFormat::Text,
            }
        );
    });
}

#[test]
fn parses_eval_subcommand_with_runtime_flags() {
    with_isolated_config_home(|| {
        let args = vec![
            "--model".to_string(),
            "openai/gpt-5.2".to_string(),
            "--permission-mode".to_string(),
            "read-only".to_string(),
            "eval".to_string(),
            "suite.json".to_string(),
        ];
        assert_eq!(
            parse_args(&args).expect("args should parse"),
            CliAction::Eval {
                suite_path: PathBuf::from("suite.json"),
                model: "openai/gpt-5.2".to_string(),
                allowed_tools: None,
                permission_mode: PermissionMode::ReadOnly,
                collaboration_mode: CollaborationMode::Build,
                reasoning_effort: None,
                fast_mode: FastMode::Off,
                check_only: false,
                fail_on_failures: false,
            }
        );
    });
}

#[test]
fn parses_trace_subcommand() {
    with_isolated_config_home(|| {
        let args = vec!["trace".to_string(), ".pebble/runs/run.json".to_string()];
        assert_eq!(
            parse_args(&args).expect("args should parse"),
            CliAction::Trace {
                trace_path: PathBuf::from(".pebble/runs/run.json"),
                output_format: CliOutputFormat::Text,
            }
        );
    });
}

#[test]
fn parses_replay_subcommand() {
    with_isolated_config_home(|| {
        let args = vec!["replay".to_string(), ".pebble/runs/run.json".to_string()];
        assert_eq!(
            parse_args(&args).expect("args should parse"),
            CliAction::Replay {
                trace_path: PathBuf::from(".pebble/runs/run.json"),
                output_format: CliOutputFormat::Text,
            }
        );
    });
}

#[test]
fn parses_gc_subcommand() {
    with_isolated_config_home(|| {
        let args = vec!["gc".to_string(), "--dry-run".to_string()];
        assert_eq!(
            parse_args(&args).expect("args should parse"),
            CliAction::Gc { dry_run: true }
        );
    });
}

#[test]
fn parses_config_check_subcommand() {
    with_isolated_config_home(|| {
        let args = vec![
            "config".to_string(),
            "check".to_string(),
            "--json".to_string(),
        ];
        assert_eq!(
            parse_args(&args).expect("args should parse"),
            CliAction::Config {
                section: Some("check".to_string()),
                output_format: CliOutputFormat::Json,
            }
        );
    });
}

#[test]
fn renders_config_check_report_with_field_paths() {
    let root = std::env::temp_dir().join(format!(
        "pebble-config-check-test-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should work")
            .as_nanos()
    ));
    let cwd = root.join("project");
    let home = root.join("home").join(".pebble");
    std::fs::create_dir_all(cwd.join(".pebble")).expect("project config dir should exist");
    std::fs::create_dir_all(&home).expect("home config dir should exist");
    std::fs::write(
        cwd.join(".pebble").join("settings.json"),
        r#"{"retention":{"traceDays":"soon"}}"#,
    )
    .expect("settings should write");

    let loader = ConfigLoader::new(&cwd, &home);
    let report = loader.check();
    let rendered = render_config_check_report(&cwd, &report);

    assert!(rendered.contains("Config Check"));
    assert!(rendered.contains("result"));
    assert!(rendered.contains("failed"));
    assert!(rendered.contains("retention.traceDays"));
    assert!(rendered.contains("field traceDays must be an integer"));

    std::fs::remove_dir_all(root).expect("temp config dir should be removed");
}

#[test]
fn renders_trace_report_with_api_tool_and_permission_summary() {
    let mut trace = TurnTrace::start("inspect repository", 2);
    trace.duration_ms = Some(42);
    trace.final_message_count = Some(6);
    trace.api_calls.push(ApiCallTrace {
        iteration: 1,
        request_message_count: 3,
        request_estimated_tokens: 1234,
        duration_ms: 20,
        result_event_count: Some(4),
        usage: Some(TokenUsage {
            input_tokens: 100,
            output_tokens: 25,
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: 0,
        }),
        error: None,
    });
    trace.permissions.push(PermissionTrace {
        iteration: 1,
        tool_use_id: "toolu_1".to_string(),
        tool_name: "bash".to_string(),
        outcome: "allow".to_string(),
        reason: Some("workspace write".to_string()),
    });
    trace.tool_calls.push(ToolCallTrace {
        iteration: 1,
        tool_use_id: "toolu_1".to_string(),
        tool_name: "bash".to_string(),
        input: TracePayloadSummary::from_text(r#"{"cmd":"rg TODO"}"#),
        effective_input: None,
        output: TracePayloadSummary::from_text("src/main.rs:1:TODO"),
        duration_ms: 7,
        permission_outcome: "allow".to_string(),
        is_error: false,
    });

    let report = render_trace_report(Path::new(".pebble/runs/run.json"), &trace);

    assert!(report.contains("Pebble Trace"));
    assert!(report.contains("schema_version"));
    assert!(report.contains("api=1 tool=1 permission=1 compaction=0 errors=0"));
    assert!(report.contains("peak_estimated_tokens=1234"));
    assert!(report.contains("API Calls"));
    assert!(report.contains("usage=125"));
    assert!(report.contains("Suggested Eval"));
    assert!(report.contains(
        "pebble eval capture .pebble/runs/run.json --suite evals/regressions.json --name bash"
    ));
    assert!(report.contains("tools"));
    assert!(report.contains("bash=allow"));
    assert!(report.contains("iterations=1 tool_calls=1 api_calls=1"));
    assert!(report.contains("Tool Calls"));
    assert!(report.contains("bash"));
    assert!(report.contains("Permissions"));
    assert!(report.contains("workspace write"));
}

#[test]
fn suggested_eval_shell_quotes_paths_and_names() {
    assert_eq!(shell_quote("plain/path.json"), "plain/path.json");
    assert_eq!(
        shell_quote("runs/trace with space.json"),
        "'runs/trace with space.json'"
    );
    assert_eq!(shell_quote("Bob's trace.json"), "'Bob'\\''s trace.json'");
}

#[test]
fn renders_replay_report_as_ordered_timeline() {
    let mut trace = TurnTrace::start("inspect repository", 2);
    trace.duration_ms = Some(42);
    trace.api_calls.push(ApiCallTrace {
        iteration: 1,
        request_message_count: 3,
        request_estimated_tokens: 1234,
        duration_ms: 20,
        result_event_count: Some(4),
        usage: None,
        error: None,
    });
    trace.permissions.push(PermissionTrace {
        iteration: 1,
        tool_use_id: "toolu_1".to_string(),
        tool_name: "bash".to_string(),
        outcome: "allow".to_string(),
        reason: None,
    });
    trace.tool_calls.push(ToolCallTrace {
        iteration: 1,
        tool_use_id: "toolu_1".to_string(),
        tool_name: "bash".to_string(),
        input: TracePayloadSummary::from_text(r#"{"cmd":"rg TODO"}"#),
        effective_input: None,
        output: TracePayloadSummary::from_text("src/main.rs:1:TODO"),
        duration_ms: 7,
        permission_outcome: "allow".to_string(),
        is_error: false,
    });

    let report = render_replay_report(Path::new(".pebble/runs/run.json"), &trace);

    assert!(report.contains("Pebble Replay"));
    assert!(report.contains("schema_version"));
    assert!(report.contains("Timeline"));
    assert!(report.contains("1. user"));
    assert!(report.contains("api"));
    assert!(report.contains("permission"));
    assert!(report.contains("tool"));
    assert!(report.contains("tool_result"));
    assert!(report.contains("src/main.rs:1:TODO"));
}

#[test]
fn load_turn_trace_reads_saved_json() {
    let root = std::env::temp_dir().join(format!(
        "pebble-trace-test-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should work")
            .as_nanos()
    ));
    std::fs::create_dir_all(&root).expect("temp root should exist");
    let path = root.join("trace.json");
    let trace = TurnTrace::start("hello", 0);
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(&trace).expect("trace should serialize"),
    )
    .expect("trace should write");

    let loaded = load_turn_trace(&path).expect("trace should load");

    assert_eq!(loaded.user_input.preview, "hello");
    assert_eq!(loaded.schema_version, runtime::TURN_TRACE_SCHEMA_VERSION);
    std::fs::remove_dir_all(root).expect("temp root should be removed");
}

#[test]
fn load_turn_trace_redacts_legacy_secret_preview() {
    let root = std::env::temp_dir().join(format!(
        "pebble-trace-redaction-test-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should work")
            .as_nanos()
    ));
    std::fs::create_dir_all(&root).expect("temp root should exist");
    let path = root.join("trace.json");
    std::fs::write(
        &path,
        r#"{
          "started_at_unix_ms": 1,
          "duration_ms": null,
          "initial_message_count": 0,
          "final_message_count": null,
          "user_input": {
            "chars": 48,
            "sha256": "legacy",
            "preview": "use api_key=sk-live-abcdefghijklmnopqrstuvwxyz",
            "truncated": false
          },
          "api_calls": [
            {
              "iteration": 1,
              "request_message_count": 1,
              "request_estimated_tokens": 10,
              "duration_ms": 1,
              "result_event_count": null,
              "usage": null,
              "error": "Authorization: Bearer sk-live-abcdefghijklmnopqrstuvwxyz"
            }
          ],
          "permissions": [],
          "tool_calls": [],
          "compactions": [],
          "errors": []
        }"#,
    )
    .expect("trace should write");

    let loaded = load_turn_trace(&path).expect("trace should load");

    assert!(loaded.user_input.redacted);
    assert_eq!(
        loaded.schema_version,
        runtime::LEGACY_TURN_TRACE_SCHEMA_VERSION
    );
    assert!(loaded.user_input.preview.contains("[REDACTED]"));
    assert!(!loaded
        .user_input
        .preview
        .contains("abcdefghijklmnopqrstuvwxyz"));
    assert_eq!(
        loaded.api_calls[0].error.as_deref(),
        Some("Authorization: Bearer [REDACTED]")
    );
    std::fs::remove_dir_all(root).expect("temp root should be removed");
}

#[test]
fn gc_prunes_generated_artifacts_by_count() {
    let root = std::env::temp_dir().join(format!(
        "pebble-gc-test-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should work")
            .as_nanos()
    ));
    let runs_dir = root.join(".pebble").join("runs");
    let evals_dir = root.join(".pebble").join("evals");
    let ci_dir = root.join(".pebble").join("ci");
    std::fs::create_dir_all(&runs_dir).expect("runs dir should exist");
    std::fs::create_dir_all(&evals_dir).expect("evals dir should exist");
    std::fs::create_dir_all(&ci_dir).expect("ci dir should exist");
    for index in 0..3 {
        std::fs::write(runs_dir.join(format!("trace-{index}.json")), "{}")
            .expect("trace should write");
        std::fs::write(evals_dir.join(format!("eval-{index}.json")), "{}")
            .expect("eval should write");
        std::fs::write(ci_dir.join(format!("ci-check-{index}.json")), "{}")
            .expect("ci report should write");
    }
    std::fs::write(evals_dir.join(EVAL_HISTORY_INDEX_FILE), "{}")
        .expect("history index should write");

    let config = RuntimeRetentionConfig {
        trace_days: None,
        max_trace_files: Some(1),
        eval_days: None,
        max_eval_reports: Some(2),
        ci_days: None,
        max_ci_reports: Some(1),
    };
    let dry_run = collect_generated_artifacts(&root, config, true);
    assert_eq!(dry_run.scanned, 9);
    assert_eq!(dry_run.entries.len(), 5);
    assert_eq!(dry_run.deleted, 0);
    assert_eq!(dry_run.reclaimed_bytes, 10);
    assert_eq!(std::fs::read_dir(&runs_dir).expect("runs").count(), 3);

    let report = collect_generated_artifacts(&root, config, false);
    assert_eq!(report.deleted, 5);
    assert_eq!(std::fs::read_dir(&runs_dir).expect("runs").count(), 1);
    assert_eq!(std::fs::read_dir(&evals_dir).expect("evals").count(), 3);
    assert_eq!(std::fs::read_dir(&ci_dir).expect("ci").count(), 1);
    assert!(evals_dir.join(EVAL_HISTORY_INDEX_FILE).exists());
    let rendered = render_gc_report(&root, &config, &report);
    assert!(rendered.contains("Pebble GC"));
    assert!(rendered.contains("ci: days=unlimited max_reports=1"));

    std::fs::remove_dir_all(root).expect("temp root should be removed");
}

#[test]
fn load_eval_suite_accepts_document_shape() {
    let root = std::env::temp_dir().join(format!(
        "pebble-eval-suite-test-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should work")
            .as_nanos()
    ));
    std::fs::create_dir_all(&root).expect("temp root should exist");
    let path = root.join("suite.json");
    std::fs::write(
        &path,
        r#"{
          "name": "smoke",
          "cases": [
            {
              "id": "hello",
              "prompt": "say hello",
              "required_answer_substrings": ["hello"],
              "max_iterations": 1
            }
          ]
        }"#,
    )
    .expect("suite should write");

    let suite = load_eval_suite(&path).expect("suite should load");

    assert_eq!(suite.name, "smoke");
    assert_eq!(suite.cases.len(), 1);
    assert_eq!(suite.cases[0].id, "hello");
    std::fs::remove_dir_all(root).expect("temp root should be removed");
}

#[test]
fn parses_bare_prompt_with_json_output_flag() {
    with_isolated_config_home(|| {
        let args = vec![
            "--output-format=json".to_string(),
            "summarize".to_string(),
            "this".to_string(),
            "repo".to_string(),
        ];
        assert_eq!(
            parse_args(&args).expect("args should parse"),
            CliAction::Prompt {
                prompt: "summarize this repo".to_string(),
                model: DEFAULT_MODEL.to_string(),
                allowed_tools: None,
                permission_mode: PermissionMode::WorkspaceWrite,
                collaboration_mode: CollaborationMode::Build,
                reasoning_effort: None,
                fast_mode: FastMode::Off,
                output_format: CliOutputFormat::Json,
            }
        );
    });
}

#[test]
fn parses_dash_p_prompt_shorthand() {
    with_isolated_config_home(|| {
        let args = vec![
            "-p".to_string(),
            "summarize".to_string(),
            "this".to_string(),
        ];
        assert_eq!(
            parse_args(&args).expect("args should parse"),
            CliAction::Prompt {
                prompt: "summarize this".to_string(),
                model: DEFAULT_MODEL.to_string(),
                allowed_tools: None,
                permission_mode: PermissionMode::WorkspaceWrite,
                collaboration_mode: CollaborationMode::Build,
                reasoning_effort: None,
                fast_mode: FastMode::Off,
                output_format: CliOutputFormat::Text,
            }
        );
    });
}

#[test]
fn parses_print_flag_as_text_output() {
    with_isolated_config_home(|| {
        let args = vec![
            "--output-format=json".to_string(),
            "--print".to_string(),
            "summarize".to_string(),
            "this".to_string(),
        ];
        assert_eq!(
            parse_args(&args).expect("args should parse"),
            CliAction::Prompt {
                prompt: "summarize this".to_string(),
                model: DEFAULT_MODEL.to_string(),
                allowed_tools: None,
                permission_mode: PermissionMode::WorkspaceWrite,
                collaboration_mode: CollaborationMode::Build,
                reasoning_effort: None,
                fast_mode: FastMode::Off,
                output_format: CliOutputFormat::Text,
            }
        );
    });
}

#[test]
fn parses_allowed_tools_flags_with_aliases_and_lists() {
    with_isolated_config_home(|| {
        let args = vec![
            "--allowedTools".to_string(),
            "read,glob".to_string(),
            "--allowed-tools=write_file".to_string(),
        ];
        assert_eq!(
            parse_args(&args).expect("args should parse"),
            CliAction::Repl {
                model: DEFAULT_MODEL.to_string(),
                allowed_tools: Some(
                    ["glob_search", "read_file", "write_file"]
                        .into_iter()
                        .map(str::to_string)
                        .collect()
                ),
                permission_mode: PermissionMode::WorkspaceWrite,
                collaboration_mode: CollaborationMode::Build,
                reasoning_effort: None,
                fast_mode: FastMode::Off,
            }
        );
    });
}

#[test]
fn parses_permission_mode_flag() {
    with_isolated_config_home(|| {
        let args = vec!["--permission-mode=read-only".to_string()];
        assert_eq!(
            parse_args(&args).expect("args should parse"),
            CliAction::Repl {
                model: DEFAULT_MODEL.to_string(),
                allowed_tools: None,
                permission_mode: PermissionMode::ReadOnly,
                collaboration_mode: CollaborationMode::Build,
                reasoning_effort: None,
                fast_mode: FastMode::Off,
            }
        );
    });
}

#[test]
fn rejects_unknown_allowed_tools() {
    with_isolated_config_home(|| {
        let error = parse_args(&["--allowedTools".to_string(), "teleport".to_string()])
            .expect_err("tool should be rejected");
        assert!(error.contains("unsupported tool in --allowedTools: teleport"));
    });
}

#[test]
fn parses_login_subcommand() {
    with_isolated_config_home(|| {
        let args = vec!["login".to_string(), "--api-key=nano-key".to_string()];
        assert_eq!(
            parse_args(&args).expect("args should parse"),
            CliAction::Login {
                service: None,
                api_key: Some("nano-key".to_string()),
            }
        );
    });
}

#[test]
fn parses_logout_subcommand() {
    with_isolated_config_home(|| {
        let args = vec!["logout".to_string(), "openai-codex".to_string()];
        assert_eq!(
            parse_args(&args).expect("args should parse"),
            CliAction::Logout {
                service: Some(AuthService::OpenAiCodex),
            }
        );
    });
}

#[test]
fn parses_model_subcommand() {
    with_isolated_config_home(|| {
        let args = vec!["model".to_string(), "openai/gpt-5.2".to_string()];
        assert_eq!(
            parse_args(&args).expect("args should parse"),
            CliAction::Model {
                model: Some("openai/gpt-5.2".to_string()),
            }
        );
    });
}

#[test]
fn parses_provider_subcommand() {
    with_isolated_config_home(|| {
        let args = vec!["provider".to_string(), "lilac".to_string()];
        assert_eq!(
            parse_args(&args).expect("args should parse"),
            CliAction::Provider {
                provider: Some("lilac".to_string()),
            }
        );

        let args = vec!["route".to_string(), "openrouter".to_string()];
        assert_eq!(
            parse_args(&args).expect("route args should parse"),
            CliAction::Route {
                route: Some("openrouter".to_string()),
            }
        );
    });
}

#[test]
fn parses_proxy_subcommand() {
    with_isolated_config_home(|| {
        let args = vec!["proxy".to_string(), "on".to_string()];
        assert_eq!(
            parse_args(&args).expect("args should parse"),
            CliAction::Proxy {
                mode: ProxyCommand::Enable,
            }
        );
    });
}

#[test]
fn parses_mcp_subcommand() {
    with_isolated_config_home(|| {
        let args = vec!["mcp".to_string(), "tools".to_string()];
        assert_eq!(
            parse_args(&args).expect("args should parse"),
            CliAction::Mcp {
                action: McpCommand::Tools,
            }
        );
        let args = vec![
            "mcp".to_string(),
            "disable".to_string(),
            "context7".to_string(),
        ];
        assert_eq!(
            parse_args(&args).expect("args should parse"),
            CliAction::Mcp {
                action: McpCommand::Disable {
                    name: "context7".to_string(),
                },
            }
        );
    });
}

#[test]
fn parses_system_prompt_options() {
    with_isolated_config_home(|| {
        let args = vec![
            "system-prompt".to_string(),
            "--cwd".to_string(),
            "/tmp/project".to_string(),
            "--date".to_string(),
            "2026-04-01".to_string(),
        ];
        assert_eq!(
            parse_args(&args).expect("args should parse"),
            CliAction::PrintSystemPrompt {
                cwd: PathBuf::from("/tmp/project"),
                date: "2026-04-01".to_string(),
            }
        );
    });
}

#[test]
fn parses_resume_flag_with_slash_command() {
    with_isolated_config_home(|| {
        let args = vec![
            "--resume".to_string(),
            "session.json".to_string(),
            "/compact".to_string(),
        ];
        assert_eq!(
            parse_args(&args).expect("args should parse"),
            CliAction::ResumeSession {
                session_path: Some(PathBuf::from("session.json")),
                commands: vec!["/compact".to_string()],
            }
        );
    });
}

#[test]
fn parses_resume_flag_with_multiple_slash_commands() {
    with_isolated_config_home(|| {
        let args = vec![
            "--resume".to_string(),
            "session.json".to_string(),
            "/status".to_string(),
            "/export".to_string(),
        ];
        assert_eq!(
            parse_args(&args).expect("args should parse"),
            CliAction::ResumeSession {
                session_path: Some(PathBuf::from("session.json")),
                commands: vec!["/status".to_string(), "/export".to_string()],
            }
        );
    });
}

#[test]
fn parses_resume_without_path_as_picker_action() {
    with_isolated_config_home(|| {
        let args = vec!["resume".to_string()];
        assert_eq!(
            parse_args(&args).expect("args should parse"),
            CliAction::ResumeSession {
                session_path: None,
                commands: Vec::new(),
            }
        );
    });
}

#[test]
fn parses_version_flag() {
    with_isolated_config_home(|| {
        let args = vec!["--version".to_string()];
        assert_eq!(
            parse_args(&args).expect("args should parse"),
            CliAction::Version
        );
    });
}

#[test]
fn parses_self_update_subcommand() {
    with_isolated_config_home(|| {
        assert_eq!(
            parse_args(&["self-update".to_string()]).expect("self-update should parse"),
            CliAction::SelfUpdate
        );
    });
}

#[test]
fn parses_doctor_bundle_subcommand() {
    with_isolated_config_home(|| {
        assert_eq!(
            parse_args(&["doctor".to_string(), "bundle".to_string()])
                .expect("doctor bundle should parse"),
            CliAction::Doctor {
                command: DoctorCommand::Bundle,
            }
        );
    });
}

#[test]
fn parses_provider_diagnostics_json_subcommand() {
    with_isolated_config_home(|| {
        assert_eq!(
            parse_args(&[
                "doctor".to_string(),
                "providers".to_string(),
                "--json".to_string(),
            ])
            .expect("provider diagnostics should parse"),
            CliAction::Doctor {
                command: DoctorCommand::Providers { json: true },
            }
        );
    });
}

#[test]
fn parses_ci_check_subcommand() {
    with_isolated_config_home(|| {
        assert_eq!(
            parse_args(&["ci".to_string(), "check".to_string()]).expect("ci check should parse"),
            CliAction::Ci {
                command: CiCommand::Check {
                    output_format: CliOutputFormat::Text,
                    save_report: false,
                },
            }
        );
    });
}

#[test]
fn parses_ci_as_check_by_default() {
    with_isolated_config_home(|| {
        assert_eq!(
            parse_args(&["ci".to_string()]).expect("ci should parse"),
            CliAction::Ci {
                command: CiCommand::Check {
                    output_format: CliOutputFormat::Text,
                    save_report: false,
                },
            }
        );
    });
}

#[test]
fn parses_ci_check_json_output() {
    with_isolated_config_home(|| {
        assert_eq!(
            parse_args(&["ci".to_string(), "check".to_string(), "--json".to_string()])
                .expect("ci check --json should parse"),
            CliAction::Ci {
                command: CiCommand::Check {
                    output_format: CliOutputFormat::Json,
                    save_report: false,
                },
            }
        );
    });
}

#[test]
fn parses_ci_check_save_report() {
    with_isolated_config_home(|| {
        assert_eq!(
            parse_args(&[
                "ci".to_string(),
                "check".to_string(),
                "--json".to_string(),
                "--save-report".to_string(),
            ])
            .expect("ci check --save-report should parse"),
            CliAction::Ci {
                command: CiCommand::Check {
                    output_format: CliOutputFormat::Json,
                    save_report: true,
                },
            }
        );
    });
}

#[test]
fn parses_release_check_json_save_report() {
    with_isolated_config_home(|| {
        assert_eq!(
            parse_args(&[
                "release".to_string(),
                "check".to_string(),
                "--json".to_string(),
                "--save-report".to_string(),
            ])
            .expect("release check should parse"),
            CliAction::Release {
                command: ReleaseCommand::Check {
                    output_format: CliOutputFormat::Json,
                    save_report: true,
                },
            }
        );
    });
}

#[test]
fn ci_step_artifact_writes_stdout_and_stderr() {
    let root = std::env::temp_dir().join(format!(
        "pebble-ci-artifact-test-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should work")
            .as_nanos()
    ));
    let artifact = super::write_ci_step_artifact(
        &root,
        "golden-trace-regressions",
        "cargo test failed",
        b"stdout line\n",
        b"stderr line\n",
    )
    .expect("artifact should write");

    let log = std::fs::read_to_string(&artifact).expect("artifact should be readable");
    assert!(log.contains("cargo test failed"));
    assert!(log.contains("## stderr"));
    assert!(log.contains("stderr line"));
    assert!(log.contains("## stdout"));
    assert!(log.contains("stdout line"));

    std::fs::remove_dir_all(root).expect("temp artifact dir should be removed");
}

#[test]
fn parses_ci_history_with_limit_and_json_output() {
    with_isolated_config_home(|| {
        assert_eq!(
            parse_args(&[
                "ci".to_string(),
                "history".to_string(),
                "--limit".to_string(),
                "20".to_string(),
                "--json".to_string(),
            ])
            .expect("ci history should parse"),
            CliAction::Ci {
                command: CiCommand::History {
                    output_format: CliOutputFormat::Json,
                    limit: 20,
                },
            }
        );
    });
}

#[test]
fn diagnostics_bundle_writes_redacted_summary_files() {
    with_isolated_config_home(|| {
        let root = std::env::temp_dir().join(format!(
            "pebble-diagnostics-bundle-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("time should work")
                .as_nanos()
        ));
        std::fs::create_dir_all(root.join(".pebble")).expect("pebble dir should exist");
        std::fs::write(
            root.join(".pebble").join("settings.json"),
            r#"{"retention":{"traceDays":14}}"#,
        )
        .expect("settings should write");

        let bundle = write_diagnostics_bundle(&root).expect("bundle should write");

        assert!(bundle
            .path
            .starts_with(root.join(".pebble").join("diagnostics")));
        for name in [
            "README.txt",
            "doctor.txt",
            "doctor.json",
            "config-check.txt",
            "config-check.json",
            "system.json",
            "sessions.json",
            "traces.json",
            "evals.json",
            "mcp-status.txt",
            "mcp-status.json",
        ] {
            assert!(bundle.path.join(name).exists(), "{name} should exist");
        }
        let readme = std::fs::read_to_string(bundle.path.join("README.txt"))
            .expect("README should be readable");
        assert!(readme.contains("Excluded"));
        assert!(readme.contains("full prompts"));
        assert!(readme.contains("API keys"));
        super::validate_diagnostics_bundle_readme_contract(&readme)
            .expect("bundle README should include the redaction contract");

        std::fs::remove_dir_all(root).expect("temp bundle dir should be removed");
    });
}

#[test]
fn parses_checksum_manifest_for_named_asset() {
    let manifest = "abc123 *pebble-aarch64-apple-darwin\ndef456 other-file\n";
    assert_eq!(
        parse_checksum_for_asset(manifest, "pebble-aarch64-apple-darwin"),
        Some("abc123".to_string())
    );
}

#[test]
fn select_release_assets_requires_checksum_file() {
    let asset_name = super::release_asset_candidates()
        .into_iter()
        .next()
        .expect("at least one asset candidate");
    let release = GitHubRelease {
        tag_name: "v0.2.0".to_string(),
        body: String::new(),
        assets: vec![GitHubReleaseAsset {
            name: asset_name,
            browser_download_url: "https://example.invalid/pebble".to_string(),
        }],
    };

    let error = super::select_release_assets(&release).expect_err("missing checksum should error");
    assert!(error.contains("checksum manifest"));
}

#[test]
fn update_report_includes_changelog_when_present() {
    let report = render_update_report(
        "Already up to date",
        Some("0.1.3"),
        Some("0.1.3"),
        Some("No action taken."),
        Some("- Added self-update"),
    );
    assert!(report.contains("Self-update"));
    assert!(report.contains("Changelog"));
    assert!(report.contains("- Added self-update"));
    assert!(report.contains("nanogpt-community/pebble"));
}

#[test]
fn filtered_tool_specs_respect_allowlist() {
    let allowed = ["read_file", "grep_search"]
        .into_iter()
        .map(str::to_string)
        .collect();
    let filtered = filter_runtime_tool_specs(
        available_runtime_tool_specs(
            &current_tool_registry().expect("tool registry should load"),
            &McpCatalog::default(),
        ),
        Some(&allowed),
    );
    let names = filtered
        .into_iter()
        .map(|spec| spec.name)
        .collect::<Vec<_>>();
    assert_eq!(names, vec!["read_file", "grep_search"]);
}

#[test]
fn parses_auth_slash_command() {
    assert_eq!(
        parse_auth_command("/login"),
        Some(LoginCommand {
            service: None,
            api_key: None,
        })
    );
    assert_eq!(
        parse_auth_command("/auth nano-key"),
        Some(LoginCommand {
            service: None,
            api_key: Some("nano-key".to_string()),
        })
    );
    assert_eq!(
        parse_auth_command("/login synthetic"),
        Some(LoginCommand {
            service: Some(AuthService::Synthetic),
            api_key: None,
        })
    );
    assert_eq!(
        parse_auth_command("/login neuralwatt nw-key"),
        Some(LoginCommand {
            service: Some(AuthService::Neuralwatt),
            api_key: Some("nw-key".to_string()),
        })
    );
    assert_eq!(
        parse_auth_command("/login lilac lilac-key"),
        Some(LoginCommand {
            service: Some(AuthService::Lilac),
            api_key: Some("lilac-key".to_string()),
        })
    );
    assert_eq!(
        parse_auth_command("/login grok"),
        Some(LoginCommand {
            service: Some(AuthService::Grok),
            api_key: None,
        })
    );
    assert_eq!(
        parse_auth_command("/login openai-codex"),
        Some(LoginCommand {
            service: Some(AuthService::OpenAiCodex),
            api_key: None,
        })
    );
    assert_eq!(
        parse_auth_command("/login opencode-go"),
        Some(LoginCommand {
            service: Some(AuthService::OpencodeGo),
            api_key: None,
        })
    );
    assert_eq!(
        parse_auth_command("/login exa"),
        Some(LoginCommand {
            service: Some(AuthService::Exa),
            api_key: None,
        })
    );
    assert_eq!(parse_auth_command("/status"), None);
}

#[test]
fn parses_logout_slash_command() {
    assert_eq!(
        parse_logout_command("/logout"),
        Some(LogoutCommand { service: None })
    );
    assert_eq!(
        parse_logout_command("/logout openai-codex"),
        Some(LogoutCommand {
            service: Some(AuthService::OpenAiCodex),
        })
    );
    assert_eq!(parse_logout_command("/status"), None);
}

#[test]
fn removes_saved_credentials_for_selected_service() {
    with_isolated_config_home(|| {
        let config_home =
            std::env::var("PEBBLE_CONFIG_HOME").expect("isolated config home should be set");
        let credentials_path = PathBuf::from(config_home).join("credentials.json");
        std::fs::write(
            &credentials_path,
            serde_json::json!({
                "openai_codex_auth": {
                    "access_token": "token",
                    "refresh_token": "refresh"
                },
                "nanogpt_api_key": "nano-key"
            })
            .to_string(),
        )
        .expect("credentials should be written");

        let outcome = remove_saved_credentials(AuthService::OpenAiCodex)
            .expect("logout should remove saved credentials");
        assert_eq!(
            outcome,
            CredentialRemovalOutcome::Removed {
                path: credentials_path.clone(),
            }
        );

        let parsed: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&credentials_path)
                .expect("credentials should remain readable"),
        )
        .expect("credentials should remain valid json");
        assert!(parsed.get("openai_codex_auth").is_none());
        assert_eq!(
            parsed
                .get("nanogpt_api_key")
                .and_then(serde_json::Value::as_str),
            Some("nano-key")
        );
    });
}

#[test]
fn parses_bypass_as_danger_full_access() {
    assert!(matches!(
        parse_permissions_command("/bypass"),
        Some(Ok(Some(PermissionMode::DangerFullAccess)))
    ));
    assert!(matches!(
        parse_permissions_command("/bypass now"),
        Some(Err(message)) if message == "/bypass does not accept arguments"
    ));
}

#[test]
fn parses_model_slash_command() {
    assert_eq!(parse_model_command("/model"), Some(None));
    assert_eq!(
        parse_model_command("/models openai/gpt-5.2"),
        Some(Some("openai/gpt-5.2".to_string()))
    );
    assert_eq!(parse_model_command("/status"), None);
}

#[test]
fn resolves_known_pebble_model_aliases() {
    assert_eq!(resolve_model_alias("default"), "zai-org/glm-5.1");
    assert_eq!(resolve_model_alias("glm"), "zai-org/glm-5.1");
    assert_eq!(resolve_model_alias("glm5"), "zai-org/glm-5");
    assert_eq!(resolve_model_alias("glm-5.1"), "zai-org/glm-5.1");
    assert_eq!(resolve_model_alias("openai/gpt-5.2"), "openai/gpt-5.2");
}

#[test]
fn parses_provider_slash_command() {
    assert_eq!(parse_provider_command("/provider"), Some(None));
    assert_eq!(
        parse_provider_command("/providers lilac"),
        Some(Some("lilac".to_string()))
    );
    assert_eq!(parse_provider_command("/status"), None);

    assert_eq!(parse_route_command("/route"), Some(None));
    assert_eq!(
        parse_route_command("/routing openrouter"),
        Some(Some("openrouter".to_string()))
    );
}

#[test]
fn parses_proxy_slash_command() {
    assert_eq!(
        parse_proxy_command("/proxy").expect("proxy command should parse"),
        Ok(ProxyCommand::Toggle)
    );
    assert_eq!(
        parse_proxy_command("/proxy status").expect("proxy status should parse"),
        Ok(ProxyCommand::Status)
    );
    assert!(parse_proxy_command("/status").is_none());
}

#[test]
fn parses_mcp_slash_command() {
    assert_eq!(
        parse_mcp_command("/mcp tools").expect("mcp command should parse"),
        Ok(McpCommand::Tools)
    );
    assert_eq!(
        parse_mcp_command("/mcp enable context7").expect("mcp enable should parse"),
        Ok(McpCommand::Enable {
            name: "context7".to_string(),
        })
    );
    assert_eq!(
        parse_mcp_command("/mcp").expect("mcp default should parse"),
        Ok(McpCommand::Status)
    );
    assert!(parse_mcp_command("/status").is_none());
}

#[test]
fn recovers_json_object_from_noisy_proxy_input() {
    let recovered = extract_first_json_object(
        "Here you go {\"path\":\"README.md\",\"offset\":0} and then some commentary",
    )
    .expect("json object should be recovered");
    assert_eq!(
        recovered,
        serde_json::json!({"path":"README.md","offset":0})
    );
}

#[test]
fn parses_inline_proxy_arg_fragment_for_tool_execution() {
    let value = parse_tool_input_value(
        "read_file",
        "<arg name=\"path\">README.md</arg><arg name=\"offset\" type=\"integer\">0</arg>",
        &tool_specs(),
    )
    .expect("proxy arg fragment should parse");
    assert_eq!(value, serde_json::json!({"path":"README.md","offset":0}));
}

#[test]
fn converts_tool_roundtrip_messages() {
    let messages = vec![
        ConversationMessage::user_text("hello"),
        ConversationMessage::assistant(vec![ContentBlock::ToolUse {
            id: "tool-1".to_string(),
            name: "bash".to_string(),
            input: "{\"command\":\"pwd\"}".to_string(),
        }]),
        ConversationMessage::tool_result("tool-1", "bash", "ok", false),
    ];

    let converted =
        convert_messages(&messages, ApiService::NanoGpt).expect("messages should convert");
    assert_eq!(converted.len(), 3);
    assert_eq!(converted[1].role, "assistant");
    assert_eq!(converted[2].role, "user");
}

#[test]
fn archives_report_lists_archived_tool_results() {
    let temp = std::env::temp_dir().join(format!(
        "pebble-archives-list-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time after epoch")
            .as_nanos()
    ));
    let session_path = temp.join(".pebble/sessions/test.json");
    let archive_path = temp.join(".pebble/tool-results/tool-1-bash.txt");
    std::fs::create_dir_all(session_path.parent().expect("session dir")).expect("session dir");
    std::fs::create_dir_all(archive_path.parent().expect("archive dir")).expect("archive dir");
    std::fs::write(&archive_path, "archived output").expect("archive file");

    let mut message = ConversationMessage::compacted_tool_result("tool-1", "bash", "", false);
    let message_id = message.id.clone();
    if let ContentBlock::ToolResult {
        archived_output_path,
        ..
    } = &mut message.blocks[0]
    {
        *archived_output_path = Some(".pebble/tool-results/tool-1-bash.txt".to_string());
    }
    let session = Session {
        version: 1,
        messages: vec![message],
        metadata: None,
    };

    let report = render_archived_tool_results_report(&session, Some(&session_path), None, None)
        .expect("archives report should render");
    std::fs::remove_dir_all(&temp).expect("cleanup temp dir");

    assert!(report.contains("Archives"));
    assert!(report.contains(&message_id));
    assert!(report.contains("tool-1"));
    assert!(report.contains("available"));
}

#[test]
fn archives_report_shows_archived_tool_result_contents() {
    let temp = std::env::temp_dir().join(format!(
        "pebble-archives-show-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time after epoch")
            .as_nanos()
    ));
    let session_path = temp.join(".pebble/sessions/test.json");
    let archive_path = temp.join(".pebble/tool-results/tool-1-bash.txt");
    std::fs::create_dir_all(session_path.parent().expect("session dir")).expect("session dir");
    std::fs::create_dir_all(archive_path.parent().expect("archive dir")).expect("archive dir");
    std::fs::write(&archive_path, "archived output").expect("archive file");

    let mut message = ConversationMessage::compacted_tool_result("tool-1", "bash", "", false);
    if let ContentBlock::ToolResult {
        archived_output_path,
        ..
    } = &mut message.blocks[0]
    {
        *archived_output_path = Some(".pebble/tool-results/tool-1-bash.txt".to_string());
    }
    let session = Session {
        version: 1,
        messages: vec![message],
        metadata: None,
    };

    let report = render_archived_tool_results_report(
        &session,
        Some(&session_path),
        Some("show"),
        Some("tool-1"),
    )
    .expect("archives report should render");
    std::fs::remove_dir_all(&temp).expect("cleanup temp dir");

    assert!(report.contains("showing archived tool output"));
    assert!(report.contains("archived output"));
    assert!(report.contains("tool-1"));
}

#[test]
fn archives_report_saves_archived_tool_result_to_requested_path() {
    let temp = std::env::temp_dir().join(format!(
        "pebble-archives-save-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time after epoch")
            .as_nanos()
    ));
    let session_path = temp.join(".pebble/sessions/test.json");
    let archive_path = temp.join(".pebble/tool-results/tool-1-bash.txt");
    let save_path = temp.join("restored/tool-output.txt");
    std::fs::create_dir_all(session_path.parent().expect("session dir")).expect("session dir");
    std::fs::create_dir_all(archive_path.parent().expect("archive dir")).expect("archive dir");
    std::fs::write(&archive_path, "archived output").expect("archive file");

    let mut message = ConversationMessage::compacted_tool_result("tool-1", "bash", "", false);
    if let ContentBlock::ToolResult {
        archived_output_path,
        ..
    } = &mut message.blocks[0]
    {
        *archived_output_path = Some(".pebble/tool-results/tool-1-bash.txt".to_string());
    }
    let session = Session {
        version: 1,
        messages: vec![message],
        metadata: None,
    };

    let report = render_archived_tool_results_report(
        &session,
        Some(&session_path),
        Some("save"),
        Some(&format!("tool-1 {}", save_path.display())),
    )
    .expect("archives report should render");
    let restored = std::fs::read_to_string(&save_path).expect("restored file should exist");
    std::fs::remove_dir_all(&temp).expect("cleanup temp dir");

    assert!(report.contains("wrote archived tool output"));
    assert!(report.contains(&save_path.display().to_string()));
    assert_eq!(restored, "archived output");
}

#[test]
fn export_includes_archived_tool_summary_and_commands() {
    let temp = std::env::temp_dir().join(format!(
        "pebble-export-archives-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time after epoch")
            .as_nanos()
    ));
    let session_path = temp.join(".pebble/sessions/test.json");
    let archive_path = temp.join(".pebble/tool-results/tool-1-bash.txt");
    std::fs::create_dir_all(session_path.parent().expect("session dir")).expect("session dir");
    std::fs::create_dir_all(archive_path.parent().expect("archive dir")).expect("archive dir");
    std::fs::write(&archive_path, "archived output").expect("archive file");

    let mut message = ConversationMessage::compacted_tool_result("tool-1", "bash", "", false);
    let message_id = message.id.clone();
    if let ContentBlock::ToolResult {
        archived_output_path,
        ..
    } = &mut message.blocks[0]
    {
        *archived_output_path = Some(".pebble/tool-results/tool-1-bash.txt".to_string());
    }
    let session = Session {
        version: 1,
        messages: vec![ConversationMessage::user_text("hello"), message],
        metadata: None,
    };

    let export = render_export_text(&session, Some(&session_path));
    std::fs::remove_dir_all(&temp).expect("cleanup temp dir");

    assert!(export.contains("## Archived Tool Outputs"));
    assert!(export.contains("- Count: 1"));
    assert!(export.contains("- Available sidecars: 1"));
    assert!(export.contains("/archives list"));
    assert!(export.contains("/archives show tool-1"));
    assert!(export.contains("/archives save tool-1 [file]"));
    assert!(export.contains(&format!("message={message_id}")));
}

#[test]
fn convert_messages_drops_dangling_tool_use_blocks() {
    let messages = vec![
        ConversationMessage::user_text("hello"),
        ConversationMessage::assistant(vec![ContentBlock::ToolUse {
            id: "tool-1".to_string(),
            name: "bash".to_string(),
            input: "{\"command\":\"pwd\"}".to_string(),
        }]),
        ConversationMessage::assistant(vec![ContentBlock::Text {
            text: "I can still answer normally.".to_string(),
        }]),
    ];

    let converted =
        convert_messages(&messages, ApiService::NanoGpt).expect("messages should convert");
    assert_eq!(converted.len(), 2);
    assert_eq!(converted[0].role, "user");
    assert_eq!(converted[1].role, "assistant");
    assert!(matches!(
        &converted[1].content[0],
        InputContentBlock::Text { text } if text == "I can still answer normally."
    ));
}

#[test]
fn convert_messages_drops_orphan_tool_results() {
    let messages = vec![
        ConversationMessage::user_text("hello"),
        ConversationMessage::tool_result("tool-missing", "bash", "ok", false),
    ];

    let converted =
        convert_messages(&messages, ApiService::NanoGpt).expect("messages should convert");
    assert_eq!(converted.len(), 1);
    assert_eq!(converted[0].role, "user");
    assert!(matches!(
        &converted[0].content[0],
        InputContentBlock::Text { text } if text == "hello"
    ));
}

#[test]
fn convert_messages_preserves_reasoning_for_opencode_go_assistant_messages() {
    let messages = vec![
        ConversationMessage::assistant(vec![
            ContentBlock::Thinking {
                text: "reasoning trail".to_string(),
                signature: None,
            },
            ContentBlock::ToolUse {
                id: "tool-1".to_string(),
                name: "bash".to_string(),
                input: "{\"command\":\"pwd\"}".to_string(),
            },
        ]),
        ConversationMessage::tool_result("tool-1", "bash", "ok", false),
    ];

    let converted =
        convert_messages(&messages, ApiService::OpencodeGo).expect("messages should convert");
    assert_eq!(converted.len(), 2);
    assert_eq!(
        converted[0].reasoning_content.as_deref(),
        Some("reasoning trail")
    );
    assert_eq!(converted[0].reasoning.as_deref(), Some("reasoning trail"));
}

#[test]
fn prompt_to_content_blocks_keeps_text_only_prompt() {
    let blocks =
        prompt_to_content_blocks("hello world", Path::new(".")).expect("text prompt should parse");
    assert_eq!(
        blocks,
        vec![InputContentBlock::Text {
            text: "hello world".to_string()
        }]
    );
}

#[test]
fn prompt_to_content_blocks_embeds_at_image_refs() {
    let temp = temp_fixture_dir("at-image-ref");
    let image_path = temp.join("sample.png");
    std::fs::write(&image_path, [1_u8, 2, 3]).expect("fixture write");
    let prompt = format!("describe @{} please", image_path.display());

    let blocks = prompt_to_content_blocks(&prompt, Path::new(".")).expect("image ref should parse");

    assert!(matches!(
        &blocks[0],
        InputContentBlock::Text { text } if text == "describe "
    ));
    assert!(matches!(
        &blocks[1],
        InputContentBlock::Image { source }
            if source.kind == "base64"
                && source.media_type == "image/png"
                && source.data == "AQID"
    ));
    assert!(matches!(
        &blocks[2],
        InputContentBlock::Text { text } if text == " please"
    ));
}

#[test]
fn prompt_to_content_blocks_embeds_markdown_image_refs() {
    let temp = temp_fixture_dir("markdown-image-ref");
    let image_path = temp.join("sample.webp");
    std::fs::write(&image_path, [255_u8]).expect("fixture write");
    let prompt = format!("see ![asset]({}) now", image_path.display());

    let blocks =
        prompt_to_content_blocks(&prompt, Path::new(".")).expect("markdown image ref should parse");

    assert!(matches!(
        &blocks[1],
        InputContentBlock::Image { source }
            if source.media_type == "image/webp" && source.data == "/w=="
    ));
}

#[test]
fn prompt_to_content_blocks_rejects_unsupported_formats() {
    let temp = temp_fixture_dir("unsupported-image-ref");
    let image_path = temp.join("sample.bmp");
    std::fs::write(&image_path, [1_u8]).expect("fixture write");
    let prompt = format!("describe @{}", image_path.display());

    let error = prompt_to_content_blocks(&prompt, Path::new("."))
        .expect_err("unsupported image ref should fail");

    assert!(error.contains("unsupported image format"));
}

#[test]
fn prompt_to_content_blocks_embeds_text_file_refs() {
    let temp = temp_fixture_dir("file-ref");
    let file_path = temp.join("README.md");
    std::fs::write(&file_path, "hello\nworld\n").expect("fixture write");
    let prompt = format!("read @{}", file_path.display());

    let blocks = prompt_to_content_blocks(&prompt, Path::new(".")).expect("file ref should parse");

    assert!(matches!(
        &blocks[1],
        InputContentBlock::Text { text }
            if text.contains("File reference:")
                && text.contains("hello\nworld")
    ));
}

#[test]
fn prompt_to_content_blocks_embeds_directory_refs() {
    let temp = temp_fixture_dir("directory-ref");
    std::fs::create_dir_all(temp.join("src")).expect("src dir");
    std::fs::write(temp.join("src/lib.rs"), "pub fn demo() {}\n").expect("fixture write");
    let prompt = format!("inspect @{}", temp.join("src").display());

    let blocks =
        prompt_to_content_blocks(&prompt, Path::new(".")).expect("directory ref should parse");

    assert!(matches!(
        &blocks[1],
        InputContentBlock::Text { text }
            if text.contains("Directory reference:")
                && text.contains("lib.rs")
    ));
}

#[test]
fn custom_command_templates_expand_arguments() {
    assert_eq!(
        render_custom_command_template("Review $1 then $2", r#"api "web app""#),
        "Review api then web app"
    );
    assert_eq!(
        render_custom_command_template("Review: $ARGUMENTS", "all changes"),
        "Review: all changes"
    );
    assert_eq!(
        render_custom_command_template("Review this", "extra context"),
        "Review this\n\nextra context"
    );
}

#[test]
fn renders_session_timeline_with_message_previews() {
    let mut session = Session::new();
    session
        .messages
        .push(ConversationMessage::user_text("hello timeline"));
    let timeline = render_session_timeline(&session);
    assert!(timeline.contains("Timeline"));
    assert!(timeline.contains("user"));
    assert!(timeline.contains("hello timeline"));
}

#[test]
fn loads_custom_commands_from_config_and_command_dir() {
    let _guard = env_lock();
    let temp = temp_fixture_dir("custom-commands");
    std::fs::create_dir_all(temp.join(".pebble/commands")).expect("commands dir");
    std::fs::write(
        temp.join(".pebble/settings.json"),
        r#"{"command":{"review":{"template":"Review $ARGUMENTS","description":"review work"}}}"#,
    )
    .expect("settings write");
    std::fs::write(temp.join(".pebble/commands/fix.md"), "Fix $1").expect("command write");

    let commands = load_custom_slash_commands(&temp).expect("commands load");

    assert_eq!(commands["review"].template, "Review $ARGUMENTS");
    assert_eq!(commands["fix"].template, "Fix $1");
}

#[test]
fn turn_snapshots_record_messages_and_git_file_changes() {
    let _guard = env_lock();
    let temp = temp_fixture_dir("turn-snapshot");
    std::process::Command::new("git")
        .args(["init"])
        .current_dir(&temp)
        .output()
        .expect("git init");
    std::fs::write(temp.join("tracked.txt"), "old\n").expect("tracked write");
    std::process::Command::new("git")
        .args(["add", "tracked.txt"])
        .current_dir(&temp)
        .output()
        .expect("git add");
    std::process::Command::new("git")
        .args([
            "-c",
            "user.email=a@example.test",
            "-c",
            "user.name=Test",
            "commit",
            "-m",
            "init",
        ])
        .current_dir(&temp)
        .output()
        .expect("git commit");

    let before = WorktreeSnapshot::capture(&temp);
    std::fs::write(temp.join("tracked.txt"), "new\n").expect("tracked edit");
    std::fs::write(temp.join("new.txt"), "created\n").expect("new write");
    let mut session = Session::new();
    session
        .messages
        .push(ConversationMessage::user_text("change files"));

    let snapshot = build_turn_snapshot(&temp, &before, 0, &session).expect("snapshot should exist");

    assert_eq!(snapshot.messages.len(), 1);
    assert!(snapshot
        .files
        .iter()
        .any(|file| file.path == "tracked.txt" && file.before == "old\n" && file.after == "new\n"));
    assert!(snapshot
        .files
        .iter()
        .any(|file| file.path == "new.txt" && !file.before_exists && file.after_exists));
}

#[test]
fn turn_snapshots_record_file_tool_outputs_without_git() {
    let _guard = env_lock();
    let temp = temp_fixture_dir("turn-snapshot-nongit");
    let file = temp.join("plain.txt");
    std::fs::write(&file, "old\n").expect("fixture write");
    let before = WorktreeSnapshot::capture(&temp);
    std::fs::write(&file, "new\n").expect("fixture edit");

    let mut session = Session::new();
    session
        .messages
        .push(ConversationMessage::user_text("edit"));
    session.messages.push(ConversationMessage::tool_result(
        "tool-1",
        "write_file",
        serde_json::json!({
            "type": "update",
            "filePath": file.display().to_string(),
            "content": "new\n",
            "originalFile": "old\n",
            "structuredPatch": [],
            "gitDiff": null
        })
        .to_string(),
        false,
    ));

    let snapshot = build_turn_snapshot(&temp, &before, 0, &session).expect("snapshot should exist");
    let change = snapshot
        .files
        .iter()
        .find(|file| file.path == "plain.txt")
        .expect("plain file change should be tracked");
    assert_eq!(change.before, "old\n");
    assert_eq!(change.after, "new\n");
    assert!(change.before_exists);
    assert!(change.after_exists);
}

#[test]
fn turn_snapshot_stacks_are_bounded() {
    let mut session = Session::new();
    for index in 0..(super::MAX_TURN_SNAPSHOT_STACK_ENTRIES + 5) {
        append_undo_snapshot(
            &mut session,
            runtime::SessionTurnSnapshot {
                timestamp: format!("snapshot-{index}"),
                message_count_before: index as u32,
                prompt: None,
                messages: Vec::new(),
                files: Vec::new(),
            },
        );
    }

    let undo_stack = session
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.undo_stack.as_ref())
        .expect("undo stack should be initialized");
    assert_eq!(undo_stack.len(), super::MAX_TURN_SNAPSHOT_STACK_ENTRIES);
    assert_eq!(undo_stack.first().unwrap().timestamp, "snapshot-5");
    assert_eq!(undo_stack.last().unwrap().timestamp, "snapshot-24");
}

#[test]
fn permission_preview_shows_file_diff_for_write() {
    let _guard = env_lock();
    let temp = temp_fixture_dir("permission-diff");
    let file = temp.join("demo.txt");
    std::fs::write(&file, "alpha\nbeta\n").expect("fixture write");
    let request = PermissionRequest {
        tool_name: "write_file".to_string(),
        input: serde_json::json!({
            "path": file.display().to_string(),
            "content": "alpha\nomega\n"
        })
        .to_string(),
        current_mode: PermissionMode::ReadOnly,
        required_mode: PermissionMode::WorkspaceWrite,
        reason: None,
    };

    let preview = render_permission_diff_preview(&request).expect("preview should render");
    assert!(preview.contains("Diff"));
    assert!(preview.contains("-beta"));
    assert!(preview.contains("+omega"));
}

#[test]
fn convert_messages_expands_user_text_image_refs() {
    let temp = temp_fixture_dir("convert-message-image-ref");
    let image_path = temp.join("sample.gif");
    std::fs::write(&image_path, [71_u8, 73, 70]).expect("fixture write");
    let messages = vec![ConversationMessage::user_text(format!(
        "inspect @{}",
        image_path.display()
    ))];

    let converted =
        convert_messages(&messages, ApiService::NanoGpt).expect("messages should convert");

    assert_eq!(converted.len(), 1);
    assert!(matches!(
        &converted[0].content[1],
        InputContentBlock::Image { source }
            if source.media_type == "image/gif" && source.data == "R0lG"
    ));
}

fn temp_fixture_dir(label: &str) -> PathBuf {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock should advance")
        .as_nanos();
    let path = std::env::temp_dir().join(format!("pebble-{label}-{unique}"));
    std::fs::create_dir_all(&path).expect("temp dir should exist");
    path
}

#[test]
fn proxy_message_responses_preserve_native_tool_calls() {
    let response = MessageResponse {
        id: "msg_123".to_string(),
        kind: "message".to_string(),
        role: "assistant".to_string(),
        content: vec![
            OutputContentBlock::Text {
                text: "I will inspect this.\n\n".to_string(),
            },
            OutputContentBlock::ToolUse {
                id: "toolu_1".to_string(),
                name: "read_file".to_string(),
                input: json!({"path":"README.md"}),
            },
        ],
        model: "zai-org/glm-5.1".to_string(),
        stop_reason: Some("end_turn".to_string()),
        stop_sequence: None,
        usage: Usage {
            input_tokens: 10,
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: 0,
            output_tokens: 5,
        },
        request_id: None,
    };

    let events = proxy_response_to_events(response, &mut Vec::new(), &tool_specs())
        .expect("proxy response should convert");
    assert!(matches!(
        &events[0],
        AssistantEvent::TextDelta(text) if text == "I will inspect this.\n\n"
    ));
    assert!(matches!(
        &events[1],
        AssistantEvent::ToolUse { id, name, input }
            if id == "toolu_1"
                && name == "read_file"
                && input == "{\"path\":\"README.md\"}"
    ));
}

#[test]
fn retries_proxy_when_reply_only_narrates_tool_intent() {
    let events = vec![
        AssistantEvent::TextDelta(
            "Let me explore the project structure to understand what this is about.".to_string(),
        ),
        AssistantEvent::MessageStop,
    ];

    assert!(should_retry_proxy_tool_prompt(&events));
}

#[test]
fn does_not_retry_proxy_when_tool_call_is_already_present() {
    let events = vec![
        AssistantEvent::TextDelta("Let me inspect that.".to_string()),
        AssistantEvent::ToolUse {
            id: "toolu_1".to_string(),
            name: "read_file".to_string(),
            input: "{\"path\":\"README.md\"}".to_string(),
        },
        AssistantEvent::MessageStop,
    ];

    assert!(!should_retry_proxy_tool_prompt(&events));
}

#[test]
fn read_file_tui_preview_is_compact() {
    let content = (1..=80)
        .map(|line| format!("line {line}"))
        .collect::<Vec<_>>()
        .join("\n");
    let output = serde_json::json!({
        "type": "text",
        "file": {
            "filePath": "/tmp/demo.rs",
            "content": content,
            "numLines": 80,
            "startLine": 1,
            "totalLines": 80
        }
    })
    .to_string();

    let block = render_tool_result_block("read_file", &output);
    let plain = strip_ansi_for_test(&block);

    // The compact block mentions the path and a range, but never spills
    // the full file contents into the transcript.
    assert!(plain.contains("/tmp/demo.rs"));
    assert!(plain.contains("range"));
    assert!(plain.contains("lines 1"));
    assert!(!plain.contains("line 1\n"));
    assert!(!plain.contains("line 80"));
}

#[test]
fn proxy_write_file_xml_is_not_rendered_to_tui() {
    let text = "Now let me create the file: <tool_call name=\"write_file\"><arg name=\"path\">test.md</arg><arg name=\"content\">hello world</arg></tool_call>";
    let mut rendered = Vec::new();
    let mut events = Vec::new();

    append_proxy_text_events(text, &mut rendered, &mut events, &tool_specs())
        .expect("proxy text should parse");

    let rendered_text = String::from_utf8(rendered).expect("rendered bytes should be utf8");
    assert!(rendered_text.trim().is_empty());
    assert!(events.iter().any(|event| matches!(
        event,
        AssistantEvent::ToolUse { name, .. } if name == "write_file"
    )));
}

#[test]
fn bash_tui_preview_truncates_large_stdout() {
    let stdout = (1..=100)
        .map(|line| format!("output {line}"))
        .collect::<Vec<_>>()
        .join("\n");
    let output = serde_json::json!({
        "stdout": stdout,
        "stderr": ""
    })
    .to_string();

    let block = render_tool_result_block("bash", &output);
    let plain = strip_ansi_for_test(&block);

    // The compact block labels the stream and keeps at most a small
    // preview; the tail of the 100-line output must not leak in.
    assert!(plain.contains("stdout"));
    assert!(plain.contains("output 1"));
    assert!(!plain.contains("output 100"));
}

#[test]
fn streaming_tool_use_defers_empty_object_until_json_deltas_arrive() {
    let mut rendered = Vec::new();
    let mut events = Vec::new();
    let mut pending_tool = None;

    push_output_block(
        OutputContentBlock::ToolUse {
            id: "toolu_1".to_string(),
            name: "read_file".to_string(),
            input: json!({}),
        },
        &mut rendered,
        &mut events,
        &mut pending_tool,
        true,
    )
    .expect("tool block should be accepted");

    assert!(rendered.is_empty());
    assert!(events.is_empty());
    assert_eq!(
        pending_tool,
        Some((
            "toolu_1".to_string(),
            "read_file".to_string(),
            String::new()
        ))
    );
}

#[test]
fn streamed_tool_call_start_renders_after_accumulation() {
    let mut rendered = Vec::new();

    render_streamed_tool_call_start(&mut rendered, "read_file", r#"{"path":"README.md"}"#)
        .expect("rendered tool call should succeed");

    let text = String::from_utf8(rendered).expect("rendered bytes should be utf8");
    // The renderer now emits an ANSI-styled, glyph-prefixed header
    // (see `ui::tool_call_header`). Strip ANSI for deterministic
    // substring checks and assert on the stable human-visible parts.
    let plain = strip_ansi_for_test(&text);
    assert!(plain.contains("Read"));
    assert!(plain.contains("README.md"));
    assert!(!plain.contains("{}"));
}

fn strip_ansi_for_test(input: &str) -> String {
    let mut out = String::new();
    let mut chars = input.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\u{1b}' {
            if chars.peek() == Some(&'[') {
                chars.next();
                for next in chars.by_ref() {
                    if next.is_ascii_alphabetic() {
                        break;
                    }
                }
            }
        } else {
            out.push(ch);
        }
    }
    out
}

#[test]
fn non_stream_response_preserves_empty_object_tool_input() {
    let response = MessageResponse {
        id: "msg_123".to_string(),
        kind: "message".to_string(),
        role: "assistant".to_string(),
        content: vec![OutputContentBlock::ToolUse {
            id: "toolu_1".to_string(),
            name: "read_file".to_string(),
            input: json!({}),
        }],
        model: "zai-org/glm-5.1".to_string(),
        stop_reason: Some("tool_use".to_string()),
        stop_sequence: None,
        usage: Usage {
            input_tokens: 1,
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: 0,
            output_tokens: 1,
        },
        request_id: None,
    };

    let events =
        response_to_events(response, &mut Vec::new()).expect("response conversion should succeed");

    assert!(matches!(
        &events[0],
        AssistantEvent::ToolUse { name, input, .. }
            if name == "read_file" && input == "{}"
    ));
}

#[test]
fn response_to_events_ignores_redacted_thinking_blocks() {
    let events = response_to_events(
        MessageResponse {
            id: "msg_2".to_string(),
            kind: "message".to_string(),
            role: "assistant".to_string(),
            content: vec![
                OutputContentBlock::RedactedThinking {
                    data: json!({"reason":"hidden"}),
                },
                OutputContentBlock::Text {
                    text: "Final answer".to_string(),
                },
            ],
            model: "zai-org/glm-5.1".to_string(),
            stop_reason: Some("end_turn".to_string()),
            stop_sequence: None,
            usage: Usage {
                input_tokens: 1,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
                output_tokens: 1,
            },
            request_id: None,
        },
        &mut Vec::new(),
    )
    .expect("response conversion should succeed");

    assert!(matches!(
        &events[0],
        AssistantEvent::TextDelta(text) if text == "Final answer"
    ));
    assert!(!events.iter().any(|event| matches!(
        event,
        AssistantEvent::ThinkingDelta(_) | AssistantEvent::ThinkingSignature(_)
    )));
}

#[test]
fn strip_markdown_code_fence_removes_wrapping_block() {
    let stripped = strip_markdown_code_fence("```markdown\n# PEBBLE.md\n\nRules\n```")
        .expect("code fence should be removed");

    assert_eq!(stripped, "# PEBBLE.md\n\nRules");
}

#[test]
fn normalize_generated_pebble_md_adds_heading_when_missing() {
    let normalized =
        normalize_generated_pebble_md("Repository guidance").expect("markdown should normalize");

    assert_eq!(normalized, "# PEBBLE.md\n\nRepository guidance\n");
}

#[test]
fn normalize_generated_pebble_md_prefers_embedded_pebble_heading() {
    let normalized = normalize_generated_pebble_md(
        "Here is the file you requested:\n\n# PEBBLE.md\n\nProject rules",
    )
    .expect("markdown should normalize");

    assert_eq!(normalized, "# PEBBLE.md\n\nProject rules\n");
}
