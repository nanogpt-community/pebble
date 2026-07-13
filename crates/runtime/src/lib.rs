mod bash;
mod bootstrap;
mod cancellation;
mod compact;
mod config;
mod conversation;
mod file_ops;
mod hooks;
mod json;
mod mcp;
mod mcp_client;
mod mcp_stdio;
mod oauth;
mod permissions;
mod prompt;
mod remote;
mod sandbox;
mod session;
mod trace;
mod usage;

pub use bash::{execute_bash, BashCommandInput, BashCommandOutput};
pub use bootstrap::{BootstrapPhase, BootstrapPlan};
pub use cancellation::{
    active_cancellation, set_active_cancellation, ActiveCancellationGuard, CancellationToken,
};
pub use compact::{
    build_compaction_prompt, compact_session, compact_session_with_summary,
    estimate_session_tokens, format_compact_summary, get_compact_continuation_message,
    get_tool_result_context_output, prepare_compaction, should_compact, CompactionConfig,
    CompactionResult, PreparedCompaction,
};
pub use config::{
    default_config_home, ConfigCheckIssue, ConfigCheckReport, ConfigEntry, ConfigError,
    ConfigLoader, ConfigSource, McpClaudeAiProxyServerConfig, McpConfigCollection, McpOAuthConfig,
    McpRemoteServerConfig, McpSdkServerConfig, McpServerConfig, McpStdioServerConfig,
    McpStdioStderrMode, McpTransport, McpWebSocketServerConfig, OAuthConfig,
    ResolvedPermissionMode, RuntimeCompactionConfig, RuntimeConfig, RuntimeFeatureConfig,
    RuntimeHookConfig, RuntimePluginConfig, RuntimeRetentionConfig, ScopedMcpServerConfig,
    PEBBLE_SETTINGS_SCHEMA_NAME,
};
pub use conversation::{
    auto_compaction_threshold_from_env, ApiClient, ApiRequest, AssistantEvent, AutoCompactionEvent,
    ConversationRuntime, RuntimeError, StaticToolExecutor, ToolError, ToolExecutor, TurnSummary,
};
pub use file_ops::{
    apply_patch, edit_file, glob_search, grep_search, read_file, write_file, ApplyPatchFileChange,
    ApplyPatchOutput, EditFileOutput, GlobSearchOutput, GrepSearchInput, GrepSearchOutput,
    ReadFileOutput, StructuredPatchHunk, TextFilePayload, WriteFileOutput,
};
pub use hooks::{
    HookAbortSignal, HookEvent, HookPermissionDecision, HookProgressEvent, HookProgressReporter,
    HookRunResult, HookRunner,
};
pub use json::{JsonError, JsonValue as RuntimeJsonValue};
pub use mcp::{
    mcp_server_signature, mcp_tool_name, mcp_tool_prefix, normalize_name_for_mcp,
    scoped_mcp_config_hash, unwrap_ccr_proxy_url,
};
pub use mcp_client::{
    McpClaudeAiProxyTransport, McpClientAuth, McpClientBootstrap, McpClientTransport,
    McpRemoteTransport, McpSdkTransport, McpStdioTransport,
};
pub use mcp_stdio::{
    spawn_mcp_stdio_process, JsonRpcError, JsonRpcId, JsonRpcRequest, JsonRpcResponse,
    ManagedMcpTool, McpInitializeClientInfo, McpInitializeParams, McpInitializeResult,
    McpInitializeServerInfo, McpListResourcesParams, McpListResourcesResult, McpListToolsParams,
    McpListToolsResult, McpReadResourceParams, McpReadResourceResult, McpResource,
    McpResourceContents, McpServerManager, McpServerManagerError, McpStdioProcess, McpTool,
    McpToolCallContent, McpToolCallParams, McpToolCallResult, UnsupportedMcpServer,
};
pub use oauth::{
    code_challenge_s256, generate_pkce_pair, generate_state, loopback_redirect_uri,
    OAuthAuthorizationRequest, OAuthRefreshRequest, OAuthTokenExchangeRequest, OAuthTokenSet,
    PkceChallengeMethod, PkceCodePair,
};
pub use permissions::{
    PermissionContext, PermissionMode, PermissionOutcome, PermissionOverride, PermissionPolicy,
    PermissionPromptDecision, PermissionPrompter, PermissionRequest,
};
pub use prompt::{
    load_system_prompt, load_system_prompt_with_model_family, prepend_bullets, ContextFile,
    ProjectContext, ProjectType, PromptBuildError, RecommendedCheck, RepositoryContext,
    SystemPromptBuilder, FRONTIER_MODEL_NAME, SYSTEM_PROMPT_DYNAMIC_BOUNDARY,
};
pub use remote::{
    inherited_upstream_proxy_env, no_proxy_list, read_token, upstream_proxy_ws_url,
    RemoteSessionContext, UpstreamProxyBootstrap, UpstreamProxyState, DEFAULT_REMOTE_BASE_URL,
    DEFAULT_SESSION_TOKEN_PATH, DEFAULT_SYSTEM_CA_BUNDLE, NO_PROXY_HOSTS, UPSTREAM_PROXY_ENV_KEYS,
};
pub use sandbox::{
    build_linux_sandbox_command, detect_container_environment, detect_container_environment_from,
    resolve_sandbox_status, resolve_sandbox_status_for_request, ContainerEnvironment,
    FilesystemIsolationMode, LinuxSandboxCommand, SandboxConfig, SandboxDetectionInputs,
    SandboxRequest, SandboxStatus,
};
pub use session::{
    ContentBlock, ConversationMessage, EditHistoryEntry, EditHistoryFile, MessageRole, Session,
    SessionError, SessionMetadata, SessionTurnSnapshot,
};
pub use trace::{
    ApiCallTrace, CompactionTrace, PermissionTrace, ToolCallTrace, TracePayloadSummary, TurnTrace,
    LEGACY_TURN_TRACE_SCHEMA_VERSION, TURN_TRACE_SCHEMA_VERSION,
};
pub use usage::{
    format_usd, pricing_for_model, ModelPricing, TokenUsage, UsageCostEstimate, UsageTracker,
};

#[cfg(test)]
pub(crate) fn test_env_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    LOCK.get_or_init(|| std::sync::Mutex::new(()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}
