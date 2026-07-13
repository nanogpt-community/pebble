use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::Path;

use platform::write_atomic;
use reqwest::header::{HeaderName, HeaderValue};
use serde_json::Value as JsonValue;

use crate::proxy::RuntimeToolSpec;
use crate::report::{report_label, report_title};
use crate::ui::{self, Stylize};
use runtime::{
    mcp_tool_name, spawn_mcp_stdio_process, CancellationToken, ConfigLoader, ConfigSource,
    JsonRpcId, JsonRpcRequest, JsonRpcResponse, McpClientAuth, McpClientBootstrap,
    McpClientTransport, McpInitializeClientInfo, McpInitializeParams, McpListToolsParams,
    McpListToolsResult, McpToolCallParams, McpToolCallResult, McpTransport, PermissionMode,
    ScopedMcpServerConfig,
};

const MCP_DISCOVERY_TIMEOUT_SECS: u64 = 30;
const VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum McpCommand {
    Status,
    Tools,
    Reload,
    Add { name: String },
    Enable { name: String },
    Disable { name: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct McpServerStatus {
    pub(crate) server_name: String,
    pub(crate) scope: ConfigSource,
    pub(crate) enabled: bool,
    pub(crate) transport: McpTransport,
    pub(crate) loaded: bool,
    pub(crate) tool_count: usize,
    pub(crate) note: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct McpToolBinding {
    pub(crate) exposed_name: String,
    pub(crate) server_name: String,
    pub(crate) upstream_name: String,
    pub(crate) description: String,
    pub(crate) input_schema: JsonValue,
    pub(crate) config: ScopedMcpServerConfig,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct McpCatalog {
    pub(crate) servers: Vec<McpServerStatus>,
    pub(crate) tools: Vec<McpToolBinding>,
}

pub(crate) fn handle_mcp_action(action: McpCommand) -> Result<(), Box<dyn std::error::Error>> {
    let cwd = env::current_dir()?;
    match action {
        McpCommand::Status | McpCommand::Reload => {
            let catalog = load_mcp_catalog(&cwd)?;
            if matches!(action, McpCommand::Reload) {
                println!(
                    "{}",
                    ui::success_note(&format!("reloaded MCP config from {}", cwd.display()))
                );
            }
            print_mcp_status(&catalog);
        }
        McpCommand::Tools => {
            let catalog = load_mcp_catalog(&cwd)?;
            print_mcp_tools(&catalog);
        }
        McpCommand::Add { name } => println!("{}", add_mcp_server_interactive(&cwd, &name)?),
        McpCommand::Enable { name } => println!("{}", set_mcp_server_enabled(&cwd, &name, true)?),
        McpCommand::Disable { name } => {
            println!("{}", set_mcp_server_enabled(&cwd, &name, false)?);
        }
    }
    Ok(())
}

pub(crate) fn add_mcp_server_interactive(
    cwd: &Path,
    name: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    if name.trim().is_empty() {
        return Err("mcp server name cannot be empty".into());
    }
    let normalized_name = name.trim();
    println!("Add MCP server: {normalized_name}");
    print!("Transport [stdio/http] (default: stdio): ");
    io::stdout().flush()?;
    let mut transport = String::new();
    io::stdin().read_line(&mut transport)?;
    let transport = match transport.trim().to_ascii_lowercase().as_str() {
        "" | "stdio" => "stdio",
        "http" => "http",
        other => return Err(format!("unsupported transport: {other}").into()),
    };

    let server_config = if transport == "stdio" {
        let command = prompt_text("Command: ")?;
        if command.trim().is_empty() {
            return Err("stdio MCP command cannot be empty".into());
        }
        let args = prompt_text("Args (space-separated, optional): ")?;
        let args = args
            .split_whitespace()
            .map(|value| JsonValue::String(value.to_string()))
            .collect::<Vec<_>>();
        serde_json::json!({
            "type": "stdio",
            "command": command.trim(),
            "args": args,
        })
    } else {
        let url = prompt_text("URL: ")?;
        if url.trim().is_empty() {
            return Err("http MCP url cannot be empty".into());
        }
        serde_json::json!({
            "type": "http",
            "url": url.trim(),
        })
    };

    let settings_dir = cwd.join(".pebble");
    fs::create_dir_all(&settings_dir)?;
    let settings_path = settings_dir.join("settings.json");
    let mut root = match fs::read_to_string(&settings_path) {
        Ok(contents) => serde_json::from_str::<serde_json::Value>(&contents)
            .unwrap_or_else(|_| serde_json::json!({})),
        Err(error) if error.kind() == io::ErrorKind::NotFound => serde_json::json!({}),
        Err(error) => return Err(Box::new(error)),
    };
    if !root.is_object() {
        root = serde_json::json!({});
    }
    let Some(root_object) = root.as_object_mut() else {
        return Err("settings root must be an object".into());
    };
    let mcp_servers = root_object
        .entry("mcpServers")
        .or_insert_with(|| serde_json::json!({}));
    if !mcp_servers.is_object() {
        *mcp_servers = serde_json::json!({});
    }
    let Some(servers_object) = mcp_servers.as_object_mut() else {
        return Err("mcpServers must be an object".into());
    };
    servers_object.insert(normalized_name.to_string(), server_config);
    write_atomic(&settings_path, serde_json::to_string_pretty(&root)?)?;

    Ok(format!(
        "MCP\n  result:  added\n  server:  {normalized_name}\n  file:    {}\n  next:    run /mcp reload",
        settings_path.display()
    ))
}

pub(crate) fn set_mcp_server_enabled(
    cwd: &Path,
    name: &str,
    enabled: bool,
) -> Result<String, Box<dyn std::error::Error>> {
    let normalized_name = name.trim();
    if normalized_name.is_empty() {
        return Err("mcp server name cannot be empty".into());
    }

    let config = ConfigLoader::default_for(cwd).load()?;
    let scoped = config
        .mcp()
        .get(normalized_name)
        .ok_or_else(|| format!("unknown MCP server: {normalized_name}"))?;

    let settings_dir = cwd.join(".pebble");
    fs::create_dir_all(&settings_dir)?;
    let settings_path = settings_dir.join("settings.local.json");
    let mut root = match fs::read_to_string(&settings_path) {
        Ok(contents) => serde_json::from_str::<serde_json::Value>(&contents)
            .unwrap_or_else(|_| serde_json::json!({})),
        Err(error) if error.kind() == io::ErrorKind::NotFound => serde_json::json!({}),
        Err(error) => return Err(Box::new(error)),
    };
    if !root.is_object() {
        root = serde_json::json!({});
    }
    let Some(root_object) = root.as_object_mut() else {
        return Err("settings root must be an object".into());
    };
    let mcp_servers = root_object
        .entry("mcpServers")
        .or_insert_with(|| serde_json::json!({}));
    if !mcp_servers.is_object() {
        *mcp_servers = serde_json::json!({});
    }
    let Some(servers_object) = mcp_servers.as_object_mut() else {
        return Err("mcpServers must be an object".into());
    };
    servers_object.insert(
        normalized_name.to_string(),
        mcp_server_config_to_json(&scoped.config, enabled),
    );
    write_atomic(&settings_path, serde_json::to_string_pretty(&root)?)?;

    Ok(format!(
        "MCP\n  result:  {}\n  server:  {normalized_name}\n  file:    {}\n  next:    run /mcp reload",
        if enabled { "enabled" } else { "disabled" },
        settings_path.display()
    ))
}

fn mcp_server_config_to_json(
    config: &runtime::McpServerConfig,
    enabled: bool,
) -> serde_json::Value {
    match config {
        runtime::McpServerConfig::Stdio(config) => serde_json::json!({
            "type": "stdio",
            "command": config.command,
            "args": config.args,
            "env": config.env,
            "stderr": match config.stderr {
                runtime::McpStdioStderrMode::Inherit => "inherit",
                runtime::McpStdioStderrMode::Null => "null",
            },
            "enabled": enabled,
        }),
        runtime::McpServerConfig::Sse(config) => serde_json::json!({
            "type": "sse",
            "url": config.url,
            "headers": config.headers,
            "headersHelper": config.headers_helper,
            "oauth": mcp_oauth_to_json(config.oauth.as_ref()),
            "enabled": enabled,
        }),
        runtime::McpServerConfig::Http(config) => serde_json::json!({
            "type": "http",
            "url": config.url,
            "headers": config.headers,
            "headersHelper": config.headers_helper,
            "oauth": mcp_oauth_to_json(config.oauth.as_ref()),
            "enabled": enabled,
        }),
        runtime::McpServerConfig::Ws(config) => serde_json::json!({
            "type": "ws",
            "url": config.url,
            "headers": config.headers,
            "headersHelper": config.headers_helper,
            "enabled": enabled,
        }),
        runtime::McpServerConfig::Sdk(config) => serde_json::json!({
            "type": "sdk",
            "name": config.name,
            "enabled": enabled,
        }),
        runtime::McpServerConfig::ClaudeAiProxy(config) => serde_json::json!({
            "type": "claudeai-proxy",
            "url": config.url,
            "id": config.id,
            "enabled": enabled,
        }),
    }
}

fn mcp_oauth_to_json(oauth: Option<&runtime::McpOAuthConfig>) -> serde_json::Value {
    oauth.map_or(serde_json::Value::Null, |oauth| {
        serde_json::json!({
            "clientId": oauth.client_id,
            "callbackPort": oauth.callback_port,
            "authServerMetadataUrl": oauth.auth_server_metadata_url,
            "xaa": oauth.xaa,
        })
    })
}

pub(crate) fn load_mcp_catalog(cwd: &Path) -> Result<McpCatalog, Box<dyn std::error::Error>> {
    let config = ConfigLoader::default_for(cwd).load()?;
    let mut catalog = McpCatalog::default();
    let servers = configured_mcp_servers(&config)?;

    for (server_name, scoped) in servers {
        let mut status = McpServerStatus {
            server_name: server_name.clone(),
            scope: scoped.scope,
            enabled: scoped.enabled,
            transport: scoped.transport(),
            loaded: false,
            tool_count: 0,
            note: String::new(),
        };

        if !scoped.is_enabled() {
            status.note = "disabled in config".to_string();
            catalog.servers.push(status);
            continue;
        }

        match scoped.transport() {
            McpTransport::Stdio => match load_stdio_mcp_tools(&server_name, &scoped) {
                Ok(tools) => {
                    status.loaded = true;
                    status.tool_count = tools.len();
                    status.note = "stdio tools loaded".to_string();
                    catalog.tools.extend(tools);
                }
                Err(error) => {
                    status.note = format!("load failed: {error}");
                }
            },
            McpTransport::Http => match load_http_mcp_tools(&server_name, &scoped) {
                Ok(tools) => {
                    status.loaded = true;
                    status.tool_count = tools.len();
                    status.note = "http tools loaded".to_string();
                    catalog.tools.extend(tools);
                }
                Err(error) => {
                    status.note = format!("load failed: {error}");
                }
            },
            other => {
                status.note =
                    format!("{other:?} transport is configured but not executable in Pebble yet");
            }
        }

        catalog.servers.push(status);
    }

    catalog
        .servers
        .sort_by(|left, right| left.server_name.cmp(&right.server_name));
    catalog
        .tools
        .sort_by(|left, right| left.exposed_name.cmp(&right.exposed_name));

    Ok(catalog)
}

fn configured_mcp_servers(
    config: &runtime::RuntimeConfig,
) -> Result<Vec<(String, ScopedMcpServerConfig)>, Box<dyn std::error::Error>> {
    let mut servers = config
        .mcp()
        .servers()
        .iter()
        .map(|(name, scoped)| (name.clone(), scoped.clone()))
        .collect::<Vec<_>>();

    servers.sort_by(|left, right| left.0.cmp(&right.0));
    Ok(servers)
}

fn load_stdio_mcp_tools(
    server_name: &str,
    scoped: &ScopedMcpServerConfig,
) -> Result<Vec<McpToolBinding>, Box<dyn std::error::Error>> {
    let bootstrap = McpClientBootstrap::from_scoped_config(server_name, scoped);
    let runtime = tokio::runtime::Runtime::new()?;
    runtime.block_on(async move {
        let mut process = spawn_mcp_stdio_process(&bootstrap)?;
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(MCP_DISCOVERY_TIMEOUT_SECS),
            async {
                let initialize = process
                    .initialize(
                        JsonRpcId::Number(1),
                        McpInitializeParams {
                            protocol_version: "2025-03-26".to_string(),
                            capabilities: serde_json::json!({"roots": {}}),
                            client_info: McpInitializeClientInfo {
                                name: "pebble".to_string(),
                                version: VERSION.to_string(),
                            },
                        },
                    )
                    .await?;
                if let Some(error) = initialize.error {
                    return Err::<Vec<McpToolBinding>, Box<dyn std::error::Error>>(
                        format!("initialize failed: {}", error.message).into(),
                    );
                }
                process.send_initialized_notification().await?;

                let mut next_cursor = None;
                let mut bindings = Vec::new();
                let mut next_id = 2u64;
                loop {
                    let response = process
                        .list_tools(
                            JsonRpcId::Number(next_id),
                            Some(McpListToolsParams {
                                cursor: next_cursor.clone(),
                            }),
                        )
                        .await?;
                    next_id += 1;
                    if let Some(error) = response.error {
                        return Err(format!("tools/list failed: {}", error.message).into());
                    }
                    let Some(result) = response.result else {
                        break;
                    };
                    for tool in result.tools {
                        bindings.push(McpToolBinding {
                            exposed_name: mcp_tool_name(server_name, &tool.name),
                            server_name: server_name.to_string(),
                            upstream_name: tool.name,
                            description: tool.description.unwrap_or_else(|| "MCP tool".to_string()),
                            input_schema: tool
                                .input_schema
                                .unwrap_or_else(|| serde_json::json!({"type":"object"})),
                            config: scoped.clone(),
                        });
                    }
                    if result.next_cursor.is_none() {
                        break;
                    }
                    next_cursor = result.next_cursor;
                }

                Ok(bindings)
            },
        )
        .await;
        let _ = process.terminate().await;
        let _ = process.wait().await;

        match result {
            Ok(result) => result,
            Err(_) => Err::<Vec<McpToolBinding>, Box<dyn std::error::Error>>(
                format!("timed out after {MCP_DISCOVERY_TIMEOUT_SECS}s during stdio MCP discovery")
                    .into(),
            ),
        }
    })
}

fn load_http_mcp_tools(
    server_name: &str,
    scoped: &ScopedMcpServerConfig,
) -> Result<Vec<McpToolBinding>, Box<dyn std::error::Error>> {
    let McpClientTransport::Http(transport) =
        McpClientBootstrap::from_scoped_config(server_name, scoped).transport
    else {
        return Err("server is not an HTTP MCP transport".into());
    };

    if transport.headers_helper.is_some() {
        return Err("headers_helper for remote MCP servers is not wired yet".into());
    }
    if transport.auth != McpClientAuth::None {
        return Err("OAuth-backed remote MCP servers are not wired yet".into());
    }

    let runtime = tokio::runtime::Runtime::new()?;
    let server_name = server_name.to_string();
    let scoped = scoped.clone();
    runtime.block_on(async move {
        let initialize = http_jsonrpc_request::<JsonValue>(
            &transport.url,
            &transport.headers,
            JsonRpcId::Number(1),
            "initialize",
            Some(serde_json::json!({
                "protocolVersion": "2025-03-26",
                "capabilities": {"roots": {}},
                "clientInfo": {"name": "pebble", "version": VERSION}
            })),
        )
        .await?;
        if let Some(error) = initialize.error {
            return Err::<Vec<McpToolBinding>, Box<dyn std::error::Error>>(
                format!("initialize failed: {}", error.message).into(),
            );
        }
        http_jsonrpc_notification(
            &transport.url,
            &transport.headers,
            "notifications/initialized",
            Some(serde_json::json!({})),
        )
        .await?;

        let mut next_cursor = None;
        let mut bindings = Vec::new();
        let mut next_id = 2_u64;
        loop {
            let response = http_jsonrpc_request::<McpListToolsResult>(
                &transport.url,
                &transport.headers,
                JsonRpcId::Number(next_id),
                "tools/list",
                Some(serde_json::to_value(McpListToolsParams {
                    cursor: next_cursor.clone(),
                })?),
            )
            .await?;
            next_id += 1;
            if let Some(error) = response.error {
                return Err(format!("tools/list failed: {}", error.message).into());
            }
            let Some(result) = response.result else {
                break;
            };
            for tool in result.tools {
                bindings.push(McpToolBinding {
                    exposed_name: mcp_tool_name(&server_name, &tool.name),
                    server_name: server_name.clone(),
                    upstream_name: tool.name,
                    description: tool.description.unwrap_or_else(|| "MCP tool".to_string()),
                    input_schema: tool
                        .input_schema
                        .unwrap_or_else(|| serde_json::json!({"type":"object"})),
                    config: scoped.clone(),
                });
            }
            if result.next_cursor.is_none() {
                break;
            }
            next_cursor = result.next_cursor;
        }

        Ok(bindings)
    })
}

pub(crate) fn call_mcp_tool(
    binding: &McpToolBinding,
    input: &JsonValue,
    cancellation: &CancellationToken,
) -> Result<String, Box<dyn std::error::Error>> {
    match binding.config.transport() {
        McpTransport::Stdio => call_stdio_mcp_tool(binding, input, cancellation),
        McpTransport::Http => call_http_mcp_tool(binding, input, cancellation),
        other => Err(format!("MCP transport {other:?} is not executable in Pebble yet").into()),
    }
}

fn call_stdio_mcp_tool(
    binding: &McpToolBinding,
    input: &JsonValue,
    cancellation: &CancellationToken,
) -> Result<String, Box<dyn std::error::Error>> {
    let bootstrap = McpClientBootstrap::from_scoped_config(&binding.server_name, &binding.config);
    let runtime = tokio::runtime::Runtime::new()?;
    runtime.block_on(async move {
        let mut process = spawn_mcp_stdio_process(&bootstrap)?;
        let initialize = tokio::select! {
            result = process.initialize(
                JsonRpcId::Number(1),
                McpInitializeParams {
                    protocol_version: "2025-03-26".to_string(),
                    capabilities: serde_json::json!({"roots": {}}),
                    client_info: McpInitializeClientInfo {
                        name: "pebble".to_string(),
                        version: VERSION.to_string(),
                    },
                },
            ) => Some(result),
            () = wait_for_cancellation(cancellation) => None,
        };
        let Some(initialize) = initialize else {
            let _ = process.terminate().await;
            let _ = process.wait().await;
            return Err("request cancelled".into());
        };
        let initialize = initialize?;
        if let Some(error) = initialize.error {
            let _ = process.terminate().await;
            let _ = process.wait().await;
            return Err::<String, Box<dyn std::error::Error>>(
                format!("initialize failed: {}", error.message).into(),
            );
        }
        process.send_initialized_notification().await?;

        let response = tokio::select! {
            result = process.call_tool(
                JsonRpcId::Number(2),
                McpToolCallParams {
                    name: binding.upstream_name.clone(),
                    arguments: Some(input.clone()),
                    meta: None,
                },
            ) => Some(result),
            () = wait_for_cancellation(cancellation) => None,
        };
        let Some(response) = response else {
            let _ = process.terminate().await;
            let _ = process.wait().await;
            return Err("request cancelled".into());
        };
        let response = response?;
        let _ = process.terminate().await;
        let _ = process.wait().await;

        format_mcp_call_result(&binding.server_name, &binding.upstream_name, response)
    })
}

fn call_http_mcp_tool(
    binding: &McpToolBinding,
    input: &JsonValue,
    cancellation: &CancellationToken,
) -> Result<String, Box<dyn std::error::Error>> {
    let McpClientTransport::Http(transport) =
        McpClientBootstrap::from_scoped_config(&binding.server_name, &binding.config).transport
    else {
        return Err("server is not an HTTP MCP transport".into());
    };
    if transport.headers_helper.is_some() {
        return Err("headers_helper for remote MCP servers is not wired yet".into());
    }
    if transport.auth != McpClientAuth::None {
        return Err("OAuth-backed remote MCP servers are not wired yet".into());
    }

    let runtime = tokio::runtime::Runtime::new()?;
    let server_name = binding.server_name.clone();
    let upstream_name = binding.upstream_name.clone();
    let input = input.clone();
    runtime.block_on(async move {
        let initialize = await_mcp_or_cancel(
            http_jsonrpc_request::<JsonValue>(
                &transport.url,
                &transport.headers,
                JsonRpcId::Number(1),
                "initialize",
                Some(serde_json::json!({
                    "protocolVersion": "2025-03-26",
                    "capabilities": {"roots": {}},
                    "clientInfo": {"name": "pebble", "version": VERSION}
                })),
            ),
            cancellation,
        )
        .await?;
        if let Some(error) = initialize.error {
            return Err::<String, Box<dyn std::error::Error>>(
                format!("initialize failed: {}", error.message).into(),
            );
        }
        await_mcp_or_cancel(
            http_jsonrpc_notification(
                &transport.url,
                &transport.headers,
                "notifications/initialized",
                Some(serde_json::json!({})),
            ),
            cancellation,
        )
        .await?;

        let response = await_mcp_or_cancel(
            http_jsonrpc_request::<McpToolCallResult>(
                &transport.url,
                &transport.headers,
                JsonRpcId::Number(2),
                "tools/call",
                Some(serde_json::to_value(McpToolCallParams {
                    name: upstream_name.clone(),
                    arguments: Some(input),
                    meta: None,
                })?),
            ),
            cancellation,
        )
        .await?;
        format_mcp_call_result(&server_name, &upstream_name, response)
    })
}

async fn await_mcp_or_cancel<T>(
    future: impl std::future::Future<Output = Result<T, Box<dyn std::error::Error>>>,
    cancellation: &CancellationToken,
) -> Result<T, Box<dyn std::error::Error>> {
    tokio::select! {
        result = future => result,
        () = wait_for_cancellation(cancellation) => Err("request cancelled".into()),
    }
}

async fn wait_for_cancellation(cancellation: &CancellationToken) {
    while !cancellation.is_cancelled() {
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
}

async fn http_jsonrpc_request<TResult: serde::de::DeserializeOwned>(
    url: &str,
    headers: &std::collections::BTreeMap<String, String>,
    id: JsonRpcId,
    method: &str,
    params: Option<JsonValue>,
) -> Result<JsonRpcResponse<TResult>, Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();
    let mut request = client
        .post(url)
        .header("content-type", "application/json")
        .header("accept", "application/json, text/event-stream");
    for (key, value) in headers {
        request = request.header(
            HeaderName::from_bytes(key.as_bytes())?,
            HeaderValue::from_str(value)?,
        );
    }
    let response = request
        .json(&JsonRpcRequest::new(id, method.to_string(), params))
        .send()
        .await?;
    Ok(response.json::<JsonRpcResponse<TResult>>().await?)
}

async fn http_jsonrpc_notification(
    url: &str,
    headers: &std::collections::BTreeMap<String, String>,
    method: &str,
    params: Option<JsonValue>,
) -> Result<(), Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();
    let mut request = client
        .post(url)
        .header("content-type", "application/json")
        .header("accept", "application/json, text/event-stream");
    for (key, value) in headers {
        request = request.header(
            HeaderName::from_bytes(key.as_bytes())?,
            HeaderValue::from_str(value)?,
        );
    }
    request
        .json(&serde_json::json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params.unwrap_or_else(|| serde_json::json!({}))
        }))
        .send()
        .await?;
    Ok(())
}

fn format_mcp_call_result(
    server_name: &str,
    upstream_name: &str,
    response: JsonRpcResponse<McpToolCallResult>,
) -> Result<String, Box<dyn std::error::Error>> {
    if let Some(error) = response.error {
        return Err(format!("tools/call failed: {}", error.message).into());
    }
    let Some(result) = response.result else {
        return Err("tools/call returned no result".into());
    };
    Ok(serde_json::to_string_pretty(&serde_json::json!({
        "server": server_name,
        "tool": upstream_name,
        "content": result.content,
        "structuredContent": result.structured_content,
        "isError": result.is_error.unwrap_or(false),
    }))?)
}

pub(crate) fn print_mcp_status(catalog: &McpCatalog) {
    if catalog.servers.is_empty() {
        println!("{}", report_title("MCP"));
        println!("  {} {}", report_label("servers:"), 0);
        println!("  {} {}", report_label("tools:"), 0);
        println!(
            "  {} add `mcpServers` to `.pebble/settings.json` to expose MCP tools",
            report_label("hint:")
        );
        return;
    }

    println!("{}", report_title("MCP"));
    println!("  {} {}", report_label("servers:"), catalog.servers.len());
    println!("  {} {}", report_label("tools:"), catalog.tools.len());
    println!();
    for server in &catalog.servers {
        println!("  {}", format!("{}", server.server_name.as_str().bold()));
        println!(
            "    {} {}  {} {:?}  {} {}  {} {}",
            report_label("scope"),
            config_source_label(server.scope),
            report_label("transport"),
            server.transport,
            report_label("tools"),
            server.tool_count,
            report_label("status"),
            if !server.enabled {
                "disabled"
            } else if server.loaded {
                "ready"
            } else {
                "unavailable"
            }
        );
        println!("    {} {}", report_label("note"), server.note);
    }
}

pub(crate) fn print_mcp_tools(catalog: &McpCatalog) {
    if catalog.tools.is_empty() {
        print_mcp_status(catalog);
        return;
    }

    println!("{}", report_title("MCP Tools"));
    println!("  {} {}", report_label("count:"), catalog.tools.len());
    println!();
    for tool in &catalog.tools {
        println!("  {}", format!("{}", tool.exposed_name.as_str().bold()));
        println!(
            "    {} {}\n    {} {}\n    {} {}",
            report_label("upstream"),
            tool.upstream_name,
            report_label("server"),
            tool.server_name,
            report_label("description"),
            tool.description
        );
    }
}

fn config_source_label(source: ConfigSource) -> &'static str {
    match source {
        ConfigSource::User => "user",
        ConfigSource::Project => "project",
        ConfigSource::Local => "local",
    }
}

impl McpCatalog {
    pub(crate) fn tool_specs(&self) -> Vec<RuntimeToolSpec> {
        self.tools
            .iter()
            .map(|tool| RuntimeToolSpec {
                name: tool.exposed_name.clone(),
                description: format!(
                    "MCP {}::{} - {}",
                    tool.server_name, tool.upstream_name, tool.description
                ),
                input_schema: tool.input_schema.clone(),
                required_permission: PermissionMode::DangerFullAccess,
            })
            .collect()
    }

    pub(crate) fn find_tool(&self, name: &str) -> Option<&McpToolBinding> {
        self.tools.iter().find(|tool| tool.exposed_name == name)
    }
}

fn prompt_text(prompt: &str) -> Result<String, Box<dyn std::error::Error>> {
    print!("{prompt}");
    io::stdout().flush()?;
    let mut value = String::new();
    io::stdin().read_line(&mut value)?;
    Ok(value.trim().to_string())
}
