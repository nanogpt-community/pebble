use std::env;
use std::io;
use std::process::{Command, Stdio};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::process::Command as TokioCommand;
use tokio::runtime::Builder;
use tokio::time::timeout;

use crate::cancellation::{active_cancellation, CancellationToken};
use crate::sandbox::{
    build_linux_sandbox_command, resolve_sandbox_status_for_request, FilesystemIsolationMode,
    SandboxConfig, SandboxStatus,
};
use crate::ConfigLoader;

const DEFAULT_BASH_TIMEOUT_MS: u64 = 10 * 60 * 1_000;
const BASH_TIMEOUT_ENV_VAR: &str = "PEBBLE_BASH_TIMEOUT_MS";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BashCommandInput {
    pub command: String,
    pub timeout: Option<u64>,
    pub description: Option<String>,
    #[serde(rename = "run_in_background")]
    pub run_in_background: Option<bool>,
    #[serde(rename = "dangerouslyDisableSandbox")]
    pub dangerously_disable_sandbox: Option<bool>,
    #[serde(rename = "namespaceRestrictions")]
    pub namespace_restrictions: Option<bool>,
    #[serde(rename = "isolateNetwork")]
    pub isolate_network: Option<bool>,
    #[serde(rename = "filesystemMode")]
    pub filesystem_mode: Option<FilesystemIsolationMode>,
    #[serde(rename = "allowedMounts")]
    pub allowed_mounts: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BashCommandOutput {
    pub stdout: String,
    pub stderr: String,
    #[serde(rename = "rawOutputPath")]
    pub raw_output_path: Option<String>,
    pub interrupted: bool,
    #[serde(rename = "isImage")]
    pub is_image: Option<bool>,
    #[serde(rename = "backgroundTaskId")]
    pub background_task_id: Option<String>,
    #[serde(rename = "backgroundedByUser")]
    pub backgrounded_by_user: Option<bool>,
    #[serde(rename = "assistantAutoBackgrounded")]
    pub assistant_auto_backgrounded: Option<bool>,
    #[serde(rename = "dangerouslyDisableSandbox")]
    pub dangerously_disable_sandbox: Option<bool>,
    #[serde(rename = "returnCodeInterpretation")]
    pub return_code_interpretation: Option<String>,
    #[serde(rename = "noOutputExpected")]
    pub no_output_expected: Option<bool>,
    #[serde(rename = "structuredContent")]
    pub structured_content: Option<Vec<serde_json::Value>>,
    #[serde(rename = "persistedOutputPath")]
    pub persisted_output_path: Option<String>,
    #[serde(rename = "persistedOutputSize")]
    pub persisted_output_size: Option<u64>,
    #[serde(rename = "sandboxStatus")]
    pub sandbox_status: Option<SandboxStatus>,
}

pub fn execute_bash(input: BashCommandInput) -> io::Result<BashCommandOutput> {
    let cwd = env::current_dir().or_else(|error| {
        if error.kind() == io::ErrorKind::NotFound {
            Ok(env::temp_dir())
        } else {
            Err(error)
        }
    })?;
    let cwd = if cwd.is_dir() { cwd } else { env::temp_dir() };
    let sandbox_status = sandbox_status_for_input(&input, &cwd);

    if input.run_in_background.unwrap_or(false) {
        let mut child = prepare_command(&input.command, &cwd, &sandbox_status, false)?;
        let child = child
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()?;

        return Ok(BashCommandOutput {
            stdout: String::new(),
            stderr: String::new(),
            raw_output_path: None,
            interrupted: false,
            is_image: None,
            background_task_id: Some(child.id().to_string()),
            backgrounded_by_user: Some(false),
            assistant_auto_backgrounded: Some(false),
            dangerously_disable_sandbox: input.dangerously_disable_sandbox,
            return_code_interpretation: None,
            no_output_expected: Some(true),
            structured_content: None,
            persisted_output_path: None,
            persisted_output_size: None,
            sandbox_status: Some(sandbox_status),
        });
    }

    let runtime = Builder::new_current_thread().enable_all().build()?;
    runtime.block_on(execute_bash_async(input, sandbox_status, cwd))
}

async fn execute_bash_async(
    input: BashCommandInput,
    sandbox_status: SandboxStatus,
    cwd: std::path::PathBuf,
) -> io::Result<BashCommandOutput> {
    let mut command = prepare_tokio_command(&input.command, &cwd, &sandbox_status, true)?;
    command.kill_on_drop(true);
    let timeout_ms = input.timeout.unwrap_or_else(default_bash_timeout_ms);

    let active_cancellation = active_cancellation();
    let output_result = tokio::select! {
        result = timeout(Duration::from_millis(timeout_ms), command.output()) => Some(result),
        () = wait_for_cancellation(active_cancellation.as_ref()) => None,
    };
    let output_result = match output_result {
        Some(Ok(result)) => (result?, false),
        Some(Err(_)) => {
            return Ok(BashCommandOutput {
                stdout: String::new(),
                stderr: format!("Command exceeded timeout of {timeout_ms} ms"),
                raw_output_path: None,
                interrupted: true,
                is_image: None,
                background_task_id: None,
                backgrounded_by_user: None,
                assistant_auto_backgrounded: None,
                dangerously_disable_sandbox: input.dangerously_disable_sandbox,
                return_code_interpretation: Some(String::from("timeout")),
                no_output_expected: Some(true),
                structured_content: None,
                persisted_output_path: None,
                persisted_output_size: None,
                sandbox_status: Some(sandbox_status),
            });
        }
        None => {
            return Ok(BashCommandOutput {
                stdout: String::new(),
                stderr: "Command cancelled".to_string(),
                raw_output_path: None,
                interrupted: true,
                is_image: None,
                background_task_id: None,
                backgrounded_by_user: None,
                assistant_auto_backgrounded: None,
                dangerously_disable_sandbox: input.dangerously_disable_sandbox,
                return_code_interpretation: Some(String::from("cancelled")),
                no_output_expected: Some(true),
                structured_content: None,
                persisted_output_path: None,
                persisted_output_size: None,
                sandbox_status: Some(sandbox_status),
            });
        }
    };

    let (output, interrupted) = output_result;
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    let no_output_expected = Some(stdout.trim().is_empty() && stderr.trim().is_empty());
    let return_code_interpretation = output.status.code().and_then(|code| {
        if code == 0 {
            None
        } else {
            Some(format!("exit_code:{code}"))
        }
    });

    Ok(BashCommandOutput {
        stdout,
        stderr,
        raw_output_path: None,
        interrupted,
        is_image: None,
        background_task_id: None,
        backgrounded_by_user: None,
        assistant_auto_backgrounded: None,
        dangerously_disable_sandbox: input.dangerously_disable_sandbox,
        return_code_interpretation,
        no_output_expected,
        structured_content: None,
        persisted_output_path: None,
        persisted_output_size: None,
        sandbox_status: Some(sandbox_status),
    })
}

async fn wait_for_cancellation(cancellation: Option<&CancellationToken>) {
    let Some(cancellation) = cancellation else {
        std::future::pending::<()>().await;
        return;
    };
    while !cancellation.is_cancelled() {
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

fn default_bash_timeout_ms() -> u64 {
    parse_bash_timeout_ms(std::env::var(BASH_TIMEOUT_ENV_VAR).ok().as_deref())
}

fn parse_bash_timeout_ms(value: Option<&str>) -> u64 {
    value
        .and_then(|raw| raw.trim().parse::<u64>().ok())
        .filter(|timeout_ms| *timeout_ms > 0)
        .unwrap_or(DEFAULT_BASH_TIMEOUT_MS)
}

fn sandbox_status_for_input(input: &BashCommandInput, cwd: &std::path::Path) -> SandboxStatus {
    let config = ConfigLoader::default_for(cwd).load().map_or_else(
        |_| SandboxConfig::default(),
        |runtime_config| runtime_config.sandbox().clone(),
    );
    let request = config.resolve_request(
        input.dangerously_disable_sandbox.map(|disabled| !disabled),
        input.namespace_restrictions,
        input.isolate_network,
        input.filesystem_mode,
        input.allowed_mounts.clone(),
    );
    resolve_sandbox_status_for_request(&request, cwd)
}

#[cfg_attr(not(windows), allow(clippy::unnecessary_wraps))]
fn prepare_command(
    command: &str,
    cwd: &std::path::Path,
    sandbox_status: &SandboxStatus,
    create_dirs: bool,
) -> io::Result<Command> {
    if create_dirs {
        prepare_sandbox_dirs(cwd);
    }

    if let Some(launcher) = build_linux_sandbox_command(command, cwd, sandbox_status) {
        let mut prepared = Command::new(launcher.program);
        prepared.args(launcher.args);
        prepared.current_dir(cwd);
        prepared.env("PWD", cwd);
        prepared.envs(launcher.env);
        return Ok(prepared);
    }

    let shell_program = default_shell_program();
    #[cfg(windows)]
    let shell_program = shell_program?;
    let mut prepared = Command::new(shell_program);
    prepared.arg("-lc").arg(command).current_dir(cwd);
    prepared.env("PWD", cwd);
    if sandbox_status.filesystem_active {
        prepared.env("HOME", cwd.join(".sandbox-home"));
        prepared.env("TMPDIR", cwd.join(".sandbox-tmp"));
    }
    Ok(prepared)
}

#[cfg_attr(not(windows), allow(clippy::unnecessary_wraps))]
fn prepare_tokio_command(
    command: &str,
    cwd: &std::path::Path,
    sandbox_status: &SandboxStatus,
    create_dirs: bool,
) -> io::Result<TokioCommand> {
    if create_dirs {
        prepare_sandbox_dirs(cwd);
    }

    if let Some(launcher) = build_linux_sandbox_command(command, cwd, sandbox_status) {
        let mut prepared = TokioCommand::new(launcher.program);
        prepared.args(launcher.args);
        prepared.current_dir(cwd);
        prepared.env("PWD", cwd);
        prepared.envs(launcher.env);
        return Ok(prepared);
    }

    let shell_program = default_shell_program();
    #[cfg(windows)]
    let shell_program = shell_program?;
    let mut prepared = TokioCommand::new(shell_program);
    prepared.arg("-lc").arg(command).current_dir(cwd);
    prepared.env("PWD", cwd);
    if sandbox_status.filesystem_active {
        prepared.env("HOME", cwd.join(".sandbox-home"));
        prepared.env("TMPDIR", cwd.join(".sandbox-tmp"));
    }
    Ok(prepared)
}

fn prepare_sandbox_dirs(cwd: &std::path::Path) {
    let _ = std::fs::create_dir_all(cwd.join(".sandbox-home"));
    let _ = std::fs::create_dir_all(cwd.join(".sandbox-tmp"));
}

#[cfg(windows)]
fn default_shell_program() -> io::Result<&'static str> {
    if command_exists("sh") {
        return Ok("sh");
    }
    if command_exists("bash") {
        return Ok("bash");
    }
    Err(io::Error::new(
        io::ErrorKind::NotFound,
        "bash tool requires a POSIX shell on Windows; install Git Bash/MSYS2 or use the PowerShell tool instead",
    ))
}

#[cfg(not(windows))]
fn default_shell_program() -> &'static str {
    if std::path::Path::new("/bin/sh").exists() {
        "/bin/sh"
    } else {
        "sh"
    }
}

#[cfg(windows)]
fn command_exists(command: &str) -> bool {
    let Some(paths) = env::var_os("PATH") else {
        return false;
    };
    env::split_paths(&paths).any(|path| {
        shell_command_candidates(command)
            .iter()
            .any(|candidate| path.join(candidate).exists())
    })
}

#[cfg(windows)]
fn shell_command_candidates(command: &str) -> Vec<String> {
    let mut candidates = vec![command.to_string()];
    candidates.push(format!("{command}.exe"));
    candidates.push(format!("{command}.cmd"));
    candidates.push(format!("{command}.bat"));
    candidates.push(format!("{command}.com"));
    candidates
}

#[cfg(test)]
mod tests {
    use super::{execute_bash, parse_bash_timeout_ms, BashCommandInput, DEFAULT_BASH_TIMEOUT_MS};
    use crate::sandbox::FilesystemIsolationMode;

    #[test]
    fn executes_simple_command() {
        let output = execute_bash(BashCommandInput {
            command: String::from("printf 'hello'"),
            timeout: Some(1_000),
            description: None,
            run_in_background: Some(false),
            dangerously_disable_sandbox: Some(false),
            namespace_restrictions: Some(false),
            isolate_network: Some(false),
            filesystem_mode: Some(FilesystemIsolationMode::WorkspaceOnly),
            allowed_mounts: None,
        })
        .expect("bash command should execute");

        assert_eq!(output.stdout, "hello");
        assert!(!output.interrupted);
        assert!(output.sandbox_status.is_some());
    }

    #[test]
    fn disables_sandbox_when_requested() {
        let output = execute_bash(BashCommandInput {
            command: String::from("printf 'hello'"),
            timeout: Some(1_000),
            description: None,
            run_in_background: Some(false),
            dangerously_disable_sandbox: Some(true),
            namespace_restrictions: Some(false),
            isolate_network: Some(false),
            filesystem_mode: Some(FilesystemIsolationMode::WorkspaceOnly),
            allowed_mounts: None,
        })
        .expect("bash command should execute");

        assert!(!output.sandbox_status.expect("sandbox status").enabled);
    }

    #[test]
    fn bash_timeout_defaults_and_rejects_invalid_values() {
        assert_eq!(parse_bash_timeout_ms(None), DEFAULT_BASH_TIMEOUT_MS);
        assert_eq!(parse_bash_timeout_ms(Some("2500")), 2_500);
        assert_eq!(parse_bash_timeout_ms(Some("0")), DEFAULT_BASH_TIMEOUT_MS);
        assert_eq!(
            parse_bash_timeout_ms(Some("not-a-number")),
            DEFAULT_BASH_TIMEOUT_MS
        );
    }

    #[test]
    fn times_out_foreground_commands() {
        let output = execute_bash(BashCommandInput {
            command: String::from("sleep 1"),
            timeout: Some(20),
            description: None,
            run_in_background: Some(false),
            dangerously_disable_sandbox: Some(true),
            namespace_restrictions: Some(false),
            isolate_network: Some(false),
            filesystem_mode: Some(FilesystemIsolationMode::WorkspaceOnly),
            allowed_mounts: None,
        })
        .expect("timed command should return a result");

        assert!(output.interrupted);
        assert!(output.stderr.contains("exceeded timeout of 20 ms"));
        assert_eq!(
            output.return_code_interpretation.as_deref(),
            Some("timeout")
        );
    }
}
