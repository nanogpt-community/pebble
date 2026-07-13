use std::env;
use std::path::Path;
use std::time::Instant;

use api::ReasoningEffort;
use compat_harness::{evaluate_trace, EvalCase, EvalCaseResult, EvalFailureKind};
use crossterm::style::Color;
use platform::write_atomic;
use runtime::PermissionMode;

use crate::app::{
    assistant_text_from_messages, persist_turn_trace, prune_generated_artifacts, AllowedToolSet,
    CliPermissionPrompter, CollaborationMode, FastMode, LiveCli,
};
use crate::eval::{
    evals_dir, load_eval_suite, print_eval_suite_check, rebuild_eval_history_index,
    write_eval_history_index, EvalRunCaseReport, EvalRunReport, EVAL_REPORT_SCHEMA_VERSION,
};
use crate::session_store::{append_undo_snapshot, build_turn_snapshot, WorktreeSnapshot};
use crate::ui::Stylize;

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_eval_suite(
    suite_path: &Path,
    model: String,
    allowed_tools: Option<&AllowedToolSet>,
    permission_mode: PermissionMode,
    collaboration_mode: CollaborationMode,
    reasoning_effort: Option<ReasoningEffort>,
    fast_mode: FastMode,
    check_only: bool,
    fail_on_failures: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let suite = load_eval_suite(suite_path)?;
    if check_only {
        print_eval_suite_check(&suite, suite_path);
        return Ok(());
    }

    let cwd = env::current_dir()?;
    let started_at_unix_ms = unix_timestamp_ms();
    let started = Instant::now();
    let run_id = format!("eval-{started_at_unix_ms}");
    let mut reports = Vec::new();

    println!("Eval");
    println!("  suite   {}", suite.name);
    println!("  model   {model}");
    println!("  cases   {}", suite.cases.len());
    println!();

    for (index, case) in suite.cases.iter().enumerate() {
        println!(
            "[{}/{}] {}",
            index + 1,
            suite.cases.len(),
            case.id.as_str().bold()
        );
        reports.push(run_eval_case(
            &cwd,
            case,
            &model,
            allowed_tools,
            permission_mode,
            collaboration_mode,
            reasoning_effort,
            fast_mode,
        ));
    }

    let passed = reports
        .iter()
        .filter(|report| report.result.passed && report.error.is_none())
        .count();
    let failed = reports.len().saturating_sub(passed);
    let report = EvalRunReport {
        schema_version: EVAL_REPORT_SCHEMA_VERSION,
        run_id: run_id.clone(),
        suite: suite.name,
        model,
        started_at_unix_ms,
        duration_ms: started.elapsed().as_millis(),
        passed,
        failed,
        cases: reports,
    };
    let report_path = evals_dir()?.join(format!("{run_id}.json"));
    write_atomic(&report_path, serde_json::to_vec_pretty(&report)?)?;
    prune_generated_artifacts(&cwd);
    if let Ok(index) = rebuild_eval_history_index(&cwd) {
        let _index_path = write_eval_history_index(&cwd, &index);
    }

    println!();
    println!(
        "Eval complete: {} passed, {} failed",
        passed.to_string().with(if failed == 0 {
            Color::Green
        } else {
            Color::Yellow
        }),
        failed.to_string().with(if failed == 0 {
            Color::DarkGrey
        } else {
            Color::Red
        })
    );
    println!("Report: {}", report_path.display());
    if fail_on_failures && failed > 0 {
        return Err(format!("eval suite failed: {passed} passed, {failed} failed").into());
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_eval_case(
    cwd: &Path,
    case: &EvalCase,
    model: &str,
    allowed_tools: Option<&AllowedToolSet>,
    permission_mode: PermissionMode,
    collaboration_mode: CollaborationMode,
    reasoning_effort: Option<ReasoningEffort>,
    fast_mode: FastMode,
) -> EvalRunCaseReport {
    let mut cli = match LiveCli::new(
        model.to_string(),
        true,
        allowed_tools.cloned(),
        permission_mode,
        collaboration_mode,
        reasoning_effort,
        fast_mode,
        false,
    ) {
        Ok(cli) => cli,
        Err(error) => {
            return failed_eval_case(
                case,
                format!("setup failed: {error}"),
                EvalFailureKind::SetupError,
            );
        }
    };

    let before_message_count = cli.message_count();
    let before_files = WorktreeSnapshot::capture(cwd);
    let mut permission_prompter = CliPermissionPrompter::new(permission_mode);
    let summary = match cli.run_eval_turn(case.prompt.clone(), &mut permission_prompter) {
        Ok(summary) => summary,
        Err(error) => {
            return failed_eval_case(case, error.to_string(), EvalFailureKind::RuntimeError);
        }
    };
    let final_answer = assistant_text_from_messages(&summary.assistant_messages);
    let trace_file = persist_turn_trace(cwd, cli.session_id(), &summary.trace).ok();

    let mut session = cli.current_session();
    let snapshot = build_turn_snapshot(cwd, &before_files, before_message_count, &session);
    let changed_files = snapshot.as_ref().map_or(0, |snapshot| snapshot.files.len());
    if let Some(snapshot) = snapshot {
        append_undo_snapshot(&mut session, snapshot);
        cli.replace_session_for_eval(session);
    }
    let session_file = cli
        .persist_session()
        .ok()
        .map(|()| cli.session_path().to_path_buf());
    let result = evaluate_trace(case, &summary.trace, &final_answer);
    if result.passed {
        println!("  {}", "pass".with(Color::Green));
    } else {
        println!("  {}", "fail".with(Color::Red));
        for failure in &result.failures {
            println!("    - {failure}");
        }
    }

    EvalRunCaseReport {
        case: case.clone(),
        result,
        final_answer,
        trace_file,
        session_file,
        error: None,
        changed_files,
    }
}

fn failed_eval_case(
    case: &EvalCase,
    error: String,
    failure_kind: EvalFailureKind,
) -> EvalRunCaseReport {
    println!("  {} {error}", "error".with(Color::Red));
    EvalRunCaseReport {
        case: case.clone(),
        result: EvalCaseResult {
            id: case.id.clone(),
            passed: false,
            failures: vec![error.clone()],
            failure_categories: vec![failure_kind],
            iterations: 0,
            tool_calls: 0,
            api_calls: 0,
            duration_ms: None,
        },
        final_answer: String::new(),
        trace_file: None,
        session_file: None,
        error: Some(error),
        changed_files: 0,
    }
}

fn unix_timestamp_ms() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}
