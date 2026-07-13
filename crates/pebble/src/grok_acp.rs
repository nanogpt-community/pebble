use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use runtime::RuntimeError;
use serde_json::Value;

const CONTROL_RESPONSE_TIMEOUT: Duration = Duration::from_secs(30);
const PROMPT_RESPONSE_TIMEOUT: Duration = Duration::from_secs(600);
const RESPONSE_POLL_INTERVAL: Duration = Duration::from_millis(50);

pub(crate) fn run(
    model: &str,
    prompt: &str,
    system: &str,
    cancelled: &AtomicBool,
    mut on_text: impl FnMut(&str) -> Result<(), RuntimeError>,
) -> Result<String, RuntimeError> {
    let executable = std::env::var("PEBBLE_GROK_CLI").unwrap_or_else(|_| "grok".to_string());
    run_with_executable(&executable, model, prompt, system, cancelled, &mut on_text)
}

fn run_with_executable(
    executable: &str,
    model: &str,
    prompt: &str,
    system: &str,
    cancelled: &AtomicBool,
    on_text: &mut dyn FnMut(&str) -> Result<(), RuntimeError>,
) -> Result<String, RuntimeError> {
    let model = model.strip_prefix("grok/").unwrap_or(model);
    let cwd = std::env::current_dir().map_err(|error| RuntimeError::new(error.to_string()))?;
    let mut child = Command::new(executable)
        .args(["--no-auto-update", "--model", model, "agent", "stdio"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|error| {
            RuntimeError::new(format!(
                "could not launch the official Grok CLI; install it from https://x.ai/cli and run `/login grok`: {error}"
            ))
        })?;
    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| RuntimeError::new("official Grok CLI did not open ACP stdin"))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| RuntimeError::new("official Grok CLI did not open ACP stdout"))?;
    let (sender, receiver) = mpsc::channel();
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            if sender.send(line).is_err() {
                break;
            }
        }
    });

    let result = (|| {
        write_request(
            &mut stdin,
            1,
            "initialize",
            &serde_json::json!({
                "protocolVersion": 1,
                "clientCapabilities": {
                    "fs": { "readTextFile": false, "writeTextFile": false },
                    "terminal": false
                }
            }),
        )?;
        let initialized = read_response(&receiver, 1, cancelled, on_text)?;
        let auth_methods = initialized
            .get("authMethods")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let method_id = if std::env::var("XAI_API_KEY").is_ok()
            && auth_methods
                .iter()
                .any(|method| method.get("id").and_then(Value::as_str) == Some("xai.api_key"))
        {
            "xai.api_key"
        } else if auth_methods
            .iter()
            .any(|method| method.get("id").and_then(Value::as_str) == Some("cached_token"))
        {
            "cached_token"
        } else {
            return Err(RuntimeError::new(
                "Grok is not signed in; run `/login grok` first",
            ));
        };
        write_request(
            &mut stdin,
            2,
            "authenticate",
            &serde_json::json!({ "methodId": method_id, "_meta": { "headless": true } }),
        )?;
        read_response(&receiver, 2, cancelled, on_text)?;
        write_request(
            &mut stdin,
            3,
            "session/new",
            &serde_json::json!({ "cwd": cwd, "mcpServers": [] }),
        )?;
        let session = read_response(&receiver, 3, cancelled, on_text)?;
        let session_id = session
            .get("sessionId")
            .and_then(Value::as_str)
            .ok_or_else(|| RuntimeError::new("Grok ACP session/new returned no sessionId"))?;
        let full_prompt = format!("# System instructions\n{system}\n\n# Conversation\n{prompt}");
        write_request(
            &mut stdin,
            4,
            "session/prompt",
            &serde_json::json!({
                "sessionId": session_id,
                "prompt": [{ "type": "text", "text": full_prompt }]
            }),
        )?;
        let mut text = String::new();
        read_prompt_response(&receiver, cancelled, on_text, &mut text)?;
        collect_tail(&receiver, cancelled, on_text, &mut text)?;
        Ok(text.trim().to_string())
    })();
    let _ = child.kill();
    let _ = child.wait();
    result
}

fn write_request(
    stdin: &mut impl Write,
    id: u64,
    method: &str,
    params: &Value,
) -> Result<(), RuntimeError> {
    writeln!(
        stdin,
        "{}",
        serde_json::json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params })
    )
    .and_then(|()| stdin.flush())
    .map_err(|error| RuntimeError::new(format!("failed writing Grok ACP request: {error}")))
}

fn read_response(
    receiver: &mpsc::Receiver<String>,
    expected_id: u64,
    cancelled: &AtomicBool,
    on_text: &mut dyn FnMut(&str) -> Result<(), RuntimeError>,
) -> Result<Value, RuntimeError> {
    read_until(
        receiver,
        expected_id,
        CONTROL_RESPONSE_TIMEOUT,
        cancelled,
        |line| deliver_text_chunk(line, on_text).map(|_| ()),
    )
}

fn read_prompt_response(
    receiver: &mpsc::Receiver<String>,
    cancelled: &AtomicBool,
    on_text: &mut dyn FnMut(&str) -> Result<(), RuntimeError>,
    output: &mut String,
) -> Result<Value, RuntimeError> {
    read_until(receiver, 4, PROMPT_RESPONSE_TIMEOUT, cancelled, |line| {
        if let Some(text) = deliver_text_chunk(line, on_text)? {
            output.push_str(&text);
        }
        Ok(())
    })
}

fn read_until(
    receiver: &mpsc::Receiver<String>,
    expected_id: u64,
    timeout: Duration,
    cancelled: &AtomicBool,
    mut inspect: impl FnMut(&str) -> Result<(), RuntimeError>,
) -> Result<Value, RuntimeError> {
    let deadline = Instant::now() + timeout;
    loop {
        if cancelled.load(Ordering::SeqCst) {
            return Err(RuntimeError::cancelled());
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(RuntimeError::new("Grok ACP response timed out"));
        }
        let line = match receiver.recv_timeout(remaining.min(RESPONSE_POLL_INTERVAL)) {
            Ok(line) => line,
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                return Err(RuntimeError::new(
                    "official Grok CLI closed its ACP output before replying",
                ));
            }
        };
        inspect(&line)?;
        let Ok(message) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        if message.get("id").and_then(Value::as_u64) != Some(expected_id) {
            continue;
        }
        if let Some(error) = message.get("error") {
            return Err(RuntimeError::new(format!("Grok ACP error: {error}")));
        }
        return Ok(message.get("result").cloned().unwrap_or(Value::Null));
    }
}

fn collect_tail(
    receiver: &mpsc::Receiver<String>,
    cancelled: &AtomicBool,
    on_text: &mut dyn FnMut(&str) -> Result<(), RuntimeError>,
    output: &mut String,
) -> Result<(), RuntimeError> {
    for _ in 0..2 {
        if cancelled.load(Ordering::SeqCst) {
            return Err(RuntimeError::cancelled());
        }
        match receiver.recv_timeout(Duration::from_millis(150)) {
            Ok(line) => {
                if let Some(text) = deliver_text_chunk(&line, on_text)? {
                    output.push_str(&text);
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout | mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
    Ok(())
}

fn deliver_text_chunk(
    line: &str,
    on_text: &mut dyn FnMut(&str) -> Result<(), RuntimeError>,
) -> Result<Option<String>, RuntimeError> {
    let Ok(message) = serde_json::from_str::<Value>(line) else {
        return Ok(None);
    };
    if message.get("method").and_then(Value::as_str) != Some("session/update") {
        return Ok(None);
    }
    let update = &message["params"]["update"];
    if update.get("sessionUpdate").and_then(Value::as_str) != Some("agent_message_chunk") {
        return Ok(None);
    }
    let Some(text) = update["content"]["text"]
        .as_str()
        .filter(|text| !text.is_empty())
    else {
        return Ok(None);
    };
    on_text(text)?;
    Ok(Some(text.to_string()))
}

#[cfg(test)]
mod tests {
    use super::run_with_executable;
    use std::fs;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[cfg(unix)]
    fn stub(contents: &str) -> std::path::PathBuf {
        use std::os::unix::fs::PermissionsExt;

        let path = std::env::temp_dir().join(format!(
            "pebble-grok-acp-stub-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        fs::write(&path, contents).expect("stub should be written");
        fs::set_permissions(&path, fs::Permissions::from_mode(0o700))
            .expect("stub should be executable");
        path
    }

    #[cfg(unix)]
    #[test]
    fn streams_acp_chunks_without_putting_prompts_in_process_arguments() {
        let path = stub(
            "#!/bin/sh\nwhile IFS= read -r line; do\ncase \"$line\" in\n*'\"method\":\"initialize\"'*) echo '{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"authMethods\":[{\"id\":\"cached_token\"}]}}' ;;\n*'\"method\":\"authenticate\"'*) echo '{\"jsonrpc\":\"2.0\",\"id\":2,\"result\":{}}' ;;\n*'\"method\":\"session/new\"'*) echo '{\"jsonrpc\":\"2.0\",\"id\":3,\"result\":{\"sessionId\":\"s1\"}}' ;;\n*'\"method\":\"session/prompt\"'*) echo '{\"jsonrpc\":\"2.0\",\"method\":\"session/update\",\"params\":{\"update\":{\"sessionUpdate\":\"agent_message_chunk\",\"content\":{\"text\":\"hello \"}}}}'; echo '{\"jsonrpc\":\"2.0\",\"method\":\"session/update\",\"params\":{\"update\":{\"sessionUpdate\":\"agent_message_chunk\",\"content\":{\"text\":\"from acp\"}}}}'; echo '{\"jsonrpc\":\"2.0\",\"id\":4,\"result\":{\"stopReason\":\"end_turn\"}}' ;;\nesac\ndone\n",
        );
        let cancelled = AtomicBool::new(false);
        let mut chunks = Vec::new();
        let output = run_with_executable(
            path.to_str().expect("utf-8 path"),
            "grok/grok-4.5",
            "secret prompt",
            "secret system",
            &cancelled,
            &mut |text| {
                chunks.push(text.to_string());
                Ok(())
            },
        )
        .expect("stub should run");
        let _ = fs::remove_file(path);

        assert_eq!(chunks, ["hello ", "from acp"]);
        assert_eq!(output, "hello from acp");
    }

    #[cfg(unix)]
    #[test]
    fn cancels_a_running_acp_prompt_after_a_streamed_chunk() {
        let path = stub(
            "#!/bin/sh\nwhile IFS= read -r line; do\ncase \"$line\" in\n*'\"method\":\"initialize\"'*) echo '{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"authMethods\":[{\"id\":\"cached_token\"}]}}' ;;\n*'\"method\":\"authenticate\"'*) echo '{\"jsonrpc\":\"2.0\",\"id\":2,\"result\":{}}' ;;\n*'\"method\":\"session/new\"'*) echo '{\"jsonrpc\":\"2.0\",\"id\":3,\"result\":{\"sessionId\":\"s1\"}}' ;;\n*'\"method\":\"session/prompt\"'*) echo '{\"jsonrpc\":\"2.0\",\"method\":\"session/update\",\"params\":{\"update\":{\"sessionUpdate\":\"agent_message_chunk\",\"content\":{\"text\":\"started\"}}}}' ;;\nesac\ndone\n",
        );
        let cancelled = AtomicBool::new(false);
        let error = run_with_executable(
            path.to_str().expect("utf-8 path"),
            "grok/grok-4.5",
            "prompt",
            "system",
            &cancelled,
            &mut |_| {
                cancelled.store(true, Ordering::SeqCst);
                Ok(())
            },
        )
        .expect_err("cancelled prompt should fail");
        let _ = fs::remove_file(path);

        assert!(error.is_cancelled());
    }
}
