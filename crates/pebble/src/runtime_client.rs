use std::collections::HashSet;
use std::env;
use std::fmt::Write as _;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

use api::{
    ApiError, ApiService, ContentBlockDelta, ImageSource, InputContentBlock, InputMessage,
    MessageRequest, MessageResponse, NanoGptClient, OutputContentBlock, ReasoningEffort,
    StreamEvent as ApiStreamEvent, ThinkingConfig, ToolChoice, ToolDefinition,
    ToolResultContentBlock,
};
use crossterm::terminal::size as terminal_size;
use serde_json::Value as JsonValue;

use crate::app::{AllowedToolSet, CollaborationMode, FastMode};
use crate::grok_acp;
use crate::mcp::{call_mcp_tool, McpCatalog};
use crate::proxy::{
    convert_messages_for_proxy, parse_proxy_response, ProxyMessage, ProxySegment, RuntimeToolSpec,
};
use crate::render::{MarkdownStreamState, TerminalRenderer};
use crate::report::truncate_for_summary;
use crate::session_store::summarize_tool_payload;
use crate::tool_render::render_tool_result_block;
use crate::ui;
use runtime::{
    get_compact_continuation_message, get_tool_result_context_output, set_active_cancellation,
    ApiClient, ApiRequest, AssistantEvent, CancellationToken, ContentBlock, ConversationMessage,
    MessageRole, PermissionMode, PermissionPolicy, RuntimeError, TokenUsage, ToolError,
    ToolExecutor,
};
use tools::{set_active_backend_service, GlobalToolRegistry};

const DEFAULT_THINKING_BUDGET_TOKENS: u32 = 2_048;
const FILE_REF_MAX_BYTES: u64 = 200_000;
const DIRECTORY_REF_MAX_ENTRIES: usize = 200;
const IMAGE_REF_PREFIX: &str = "@";

async fn await_api_or_cancel<T>(
    future: impl std::future::Future<Output = Result<T, ApiError>>,
    cancellation: &CancellationToken,
) -> Result<T, RuntimeError> {
    tokio::select! {
        result = future => result.map_err(|error| RuntimeError::new(error.to_string())),
        () = wait_for_cancellation(cancellation) => Err(RuntimeError::cancelled()),
    }
}

async fn wait_for_cancellation(cancellation: &CancellationToken) {
    while !cancellation.is_cancelled() {
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }
}

pub(crate) struct PebbleRuntimeClient {
    runtime: tokio::runtime::Runtime,
    service: ApiService,
    model: String,
    provider: Option<String>,
    max_output_tokens: u32,
    enable_tools: bool,
    proxy_tool_calls: bool,
    tool_specs: Vec<RuntimeToolSpec>,
    collaboration_mode: CollaborationMode,
    reasoning_effort: Option<ReasoningEffort>,
    fast_mode: FastMode,
    render_output: bool,
}

impl PebbleRuntimeClient {
    pub(crate) fn new(
        service: ApiService,
        model: String,
        max_output_tokens: u32,
        provider: Option<String>,
        enable_tools: bool,
        proxy_tool_calls: bool,
        tool_specs: Vec<RuntimeToolSpec>,
        collaboration_mode: CollaborationMode,
        reasoning_effort: Option<ReasoningEffort>,
        fast_mode: FastMode,
        render_output: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            runtime: tokio::runtime::Runtime::new()?,
            service,
            model,
            provider,
            max_output_tokens,
            enable_tools,
            proxy_tool_calls,
            tool_specs,
            collaboration_mode,
            reasoning_effort,
            fast_mode,
            render_output,
        })
    }
}

impl ApiClient for PebbleRuntimeClient {
    fn stream(&mut self, request: ApiRequest) -> Result<Vec<AssistantEvent>, RuntimeError> {
        self.stream_cancellable(request, &CancellationToken::new())
    }

    fn stream_cancellable(
        &mut self,
        request: ApiRequest,
        cancellation: &CancellationToken,
    ) -> Result<Vec<AssistantEvent>, RuntimeError> {
        if self.service == ApiService::Grok {
            return self.stream_via_grok_cli(request, cancellation);
        }
        if self.proxy_tool_calls {
            return self.stream_via_proxy(request, cancellation);
        }

        let effective_reasoning_effort =
            effective_reasoning_effort(self.collaboration_mode, self.reasoning_effort);
        let message_request = MessageRequest {
            model: self.model.clone(),
            max_tokens: self.max_output_tokens,
            messages: convert_messages(&request.messages, self.service)?,
            system: (!request.system_prompt.is_empty()).then(|| request.system_prompt.join("\n\n")),
            tools: self.enable_tools.then(|| {
                self.tool_specs
                    .iter()
                    .map(|spec| ToolDefinition {
                        name: spec.name.clone(),
                        description: Some(spec.description.clone()),
                        input_schema: spec.input_schema.clone(),
                    })
                    .collect()
            }),
            tool_choice: self.enable_tools.then_some(ToolChoice::Auto),
            thinking: (self.service != ApiService::OpenAiCodex
                && effective_reasoning_effort.is_some())
            .then_some(ThinkingConfig::enabled(DEFAULT_THINKING_BUDGET_TOKENS)),
            reasoning_effort: effective_reasoning_effort,
            fast_mode: self.fast_mode.enabled(),
            stream: true,
        };

        let client = self.service_client()?;
        self.runtime.block_on(async {
            let mut stream = await_api_or_cancel(
                client.stream_message(&message_request),
                cancellation,
            )
            .await?;
            let mut output: Box<dyn Write> = if self.render_output {
                Box::new(io::stdout())
            } else {
                Box::new(io::sink())
            };
            let renderer = TerminalRenderer::new();
            let mut markdown_stream = MarkdownStreamState::default();
            let mut events = Vec::new();
            let mut pending_tool: Option<(String, String, String)> = None;
            let mut saw_stop = false;
            let mut stream_fallback_requested = false;
            // Print the "● pebble" assistant lead exactly once per streamed
            // response — right before the first text delta — so the model's
            // reply is visually anchored even after a wall of tool output.
            let mut assistant_lead_emitted = false;
            let render_output_enabled = self.render_output;

            loop {
                let next_event = tokio::select! {
                    result = stream.next_event() => result,
                    () = wait_for_cancellation(cancellation) => return Err(RuntimeError::cancelled()),
                };
                let event = match next_event {
                    Ok(Some(event)) => event,
                    Ok(None) => break,
                    Err(ApiError::StreamApi {
                        error_type: Some(error_type),
                        message,
                        ..
                    }) if error_type == "invalid_response_error" => {
                        eprintln!(
                            "[pebble] streaming failed with invalid_response_error{}; retrying non-streaming",
                            message
                                .as_deref()
                                .map(|message| format!(": {message}"))
                                .unwrap_or_default()
                        );
                        stream_fallback_requested = true;
                        break;
                    }
                    Err(error) => return Err(RuntimeError::new(error.to_string())),
                };

                match event {
                    ApiStreamEvent::MessageStart(start) => {
                        for block in start.message.content {
                            push_output_block(
                                block,
                                output.as_mut(),
                                &mut events,
                                &mut pending_tool,
                                true,
                            )?;
                        }
                    }
                    ApiStreamEvent::ContentBlockStart(start) => {
                        push_output_block(
                            start.content_block,
                            output.as_mut(),
                            &mut events,
                            &mut pending_tool,
                            true,
                        )?;
                    }
                    ApiStreamEvent::ContentBlockDelta(delta) => match delta.delta {
                        ContentBlockDelta::TextDelta { text } => {
                            if !text.is_empty() {
                                if render_output_enabled && !assistant_lead_emitted {
                                    write!(output, "{}", ui::assistant_lead())
                                        .and_then(|()| output.flush())
                                        .map_err(|error| RuntimeError::new(error.to_string()))?;
                                    assistant_lead_emitted = true;
                                }
                                if let Some(rendered) = markdown_stream.push(&renderer, &text) {
                                    write!(output, "{rendered}")
                                        .and_then(|()| output.flush())
                                        .map_err(|error| RuntimeError::new(error.to_string()))?;
                                }
                                events.push(AssistantEvent::TextDelta(text));
                            }
                        }
                        ContentBlockDelta::ThinkingDelta { thinking } => {
                            if !thinking.is_empty() {
                                // Keep model reasoning in the trace and
                                // conversation state without dumping it into
                                // the transcript. `/reasoning` controls model
                                // effort, not visibility.
                                events.push(AssistantEvent::ThinkingDelta(thinking));
                            }
                        }
                        ContentBlockDelta::SignatureDelta { signature } => {
                            if !signature.is_empty() {
                                events.push(AssistantEvent::ThinkingSignature(signature));
                            }
                        }
                        ContentBlockDelta::InputJsonDelta { partial_json } => {
                            if let Some((_, _, input)) = &mut pending_tool {
                                input.push_str(&partial_json);
                            }
                        }
                    },
                    ApiStreamEvent::ContentBlockStop(_) => {
                        if let Some(rendered) = markdown_stream.flush(&renderer) {
                            write!(output, "{rendered}")
                                .and_then(|()| output.flush())
                                .map_err(|error| RuntimeError::new(error.to_string()))?;
                        }
                        if let Some((id, name, input)) = pending_tool.take() {
                            render_streamed_tool_call_start(output.as_mut(), &name, &input)?;
                            events.push(AssistantEvent::ToolUse { id, name, input });
                        }
                    }
                    ApiStreamEvent::MessageDelta(delta) => {
                        events.push(AssistantEvent::Usage(TokenUsage {
                            input_tokens: delta.usage.input_tokens,
                            output_tokens: delta.usage.output_tokens,
                            cache_creation_input_tokens: 0,
                            cache_read_input_tokens: 0,
                        }));
                    }
                    ApiStreamEvent::MessageStop(_) => {
                        saw_stop = true;
                        if let Some(rendered) = markdown_stream.flush(&renderer) {
                            write!(output, "{rendered}")
                                .and_then(|()| output.flush())
                                .map_err(|error| RuntimeError::new(error.to_string()))?;
                        }
                        events.push(AssistantEvent::MessageStop);
                    }
                }
            }

            if !stream_fallback_requested
                && !saw_stop
                && events.iter().any(|event| {
                    matches!(event, AssistantEvent::TextDelta(text) if !text.is_empty())
                        || matches!(event, AssistantEvent::ThinkingDelta(text) if !text.is_empty())
                        || matches!(event, AssistantEvent::ThinkingSignature(_))
                        || matches!(event, AssistantEvent::ToolUse { .. })
                })
            {
                events.push(AssistantEvent::MessageStop);
            }

            if events
                .iter()
                .any(|event| matches!(event, AssistantEvent::MessageStop))
            {
                return Ok(events);
            }

            let response = await_api_or_cancel(
                client.send_message(&MessageRequest {
                    stream: false,
                    ..message_request.clone()
                }),
                cancellation,
            )
            .await?;
            response_to_events(response, output.as_mut())
        })
    }
}

impl PebbleRuntimeClient {
    fn service_client(&self) -> Result<NanoGptClient, RuntimeError> {
        let mut client = NanoGptClient::from_service_env(self.service)
            .map_err(|error| RuntimeError::new(error.to_string()))?;
        if self.service == ApiService::NanoGpt {
            client = client.with_provider(self.provider.clone());
        }
        Ok(client)
    }

    fn stream_via_grok_cli(
        &mut self,
        request: ApiRequest,
        cancellation: &CancellationToken,
    ) -> Result<Vec<AssistantEvent>, RuntimeError> {
        let messages = convert_messages_for_proxy(&request.messages).map_err(RuntimeError::new)?;
        let prompt = render_grok_cli_prompt(&messages);
        let mut system = request.system_prompt.join("\n\n");
        if !system.is_empty() {
            system.push_str("\n\n");
        }
        system.push_str(&crate::proxy::build_proxy_system_prompt(&self.tool_specs));

        let cancelled = cancellation.atomic_flag();
        let mut output: Box<dyn Write> = if self.render_output {
            Box::new(io::stdout())
        } else {
            Box::new(io::sink())
        };
        let first_pass = run_grok_proxy_pass(
            &self.model,
            &prompt,
            &system,
            cancelled.as_ref(),
            &self.tool_specs,
            output.as_mut(),
            true,
        )?;

        let mut events = if should_retry_proxy_tool_prompt(&first_pass.events) {
            let retry_prompt = format!("{prompt}\n\n[user]\n{}", proxy_retry_reminder());
            run_grok_proxy_pass(
                &self.model,
                &retry_prompt,
                &system,
                cancelled.as_ref(),
                &self.tool_specs,
                output.as_mut(),
                false,
            )?
            .events
        } else {
            output
                .write_all(&first_pass.deferred_render)
                .and_then(|()| output.flush())
                .map_err(|error| RuntimeError::new(error.to_string()))?;
            first_pass.events
        };
        events.push(AssistantEvent::Usage(TokenUsage::default()));
        events.push(AssistantEvent::MessageStop);
        Ok(events)
    }

    fn stream_via_proxy(
        &mut self,
        request: ApiRequest,
        cancellation: &CancellationToken,
    ) -> Result<Vec<AssistantEvent>, RuntimeError> {
        let effective_reasoning_effort =
            effective_reasoning_effort(self.collaboration_mode, self.reasoning_effort);
        let mut messages = convert_proxy_messages_to_input_messages(
            convert_messages_for_proxy(&request.messages).map_err(RuntimeError::new)?,
        );
        let base_message_request = MessageRequest {
            model: self.model.clone(),
            max_tokens: self.max_output_tokens,
            messages: messages.clone(),
            system: (!request.system_prompt.is_empty()).then(|| request.system_prompt.join("\n\n")),
            tools: None,
            tool_choice: None,
            thinking: (self.service != ApiService::OpenAiCodex
                && effective_reasoning_effort.is_some())
            .then_some(ThinkingConfig::enabled(DEFAULT_THINKING_BUDGET_TOKENS)),
            reasoning_effort: effective_reasoning_effort,
            fast_mode: self.fast_mode.enabled(),
            stream: false,
        };

        let client = self.service_client()?;
        self.runtime.block_on(async {
            let response =
                await_api_or_cancel(client.send_message(&base_message_request), cancellation)
                    .await?;
            let mut first_render = Vec::new();
            let first_events =
                proxy_response_to_events(response, &mut first_render, &self.tool_specs)?;

            if should_retry_proxy_tool_prompt(&first_events) {
                messages.push(InputMessage::user_text(proxy_retry_reminder()));
                let retry_request = MessageRequest {
                    messages,
                    ..base_message_request
                };
                let retry_response =
                    await_api_or_cancel(client.send_message(&retry_request), cancellation).await?;
                let mut output: Box<dyn Write> = if self.render_output {
                    Box::new(io::stdout())
                } else {
                    Box::new(io::sink())
                };
                return proxy_response_to_events(retry_response, output.as_mut(), &self.tool_specs);
            }

            let mut output: Box<dyn Write> = if self.render_output {
                Box::new(io::stdout())
            } else {
                Box::new(io::sink())
            };
            output
                .write_all(&first_render)
                .and_then(|()| output.flush())
                .map_err(|error| RuntimeError::new(error.to_string()))?;
            Ok(first_events)
        })
    }
}

fn render_grok_cli_prompt(messages: &[ProxyMessage]) -> String {
    messages
        .iter()
        .map(|message| format!("[{}]\n{}", message.role, message.content))
        .collect::<Vec<_>>()
        .join("\n\n")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GrokChunkMode {
    Undecided,
    Text,
    Buffered,
}

#[derive(Debug, Default)]
struct GrokChunkGate {
    pending: String,
    mode: Option<GrokChunkMode>,
    emitted_text: bool,
}

impl GrokChunkGate {
    fn push(&mut self, chunk: &str) -> Option<String> {
        self.pending.push_str(chunk);
        let mode = self.mode.unwrap_or(GrokChunkMode::Undecided);
        match mode {
            GrokChunkMode::Buffered => None,
            GrokChunkMode::Text => self.emit_safe_prefix(),
            GrokChunkMode::Undecided => {
                let trimmed = self.pending.trim_start();
                if trimmed.is_empty() {
                    return None;
                }
                if trimmed.starts_with('<') {
                    self.mode = Some(GrokChunkMode::Buffered);
                    return None;
                }
                let normalized = trimmed.to_ascii_lowercase();
                const NARRATION_PREFIXES: [&str; 6] = [
                    "let me ",
                    "i'll ",
                    "i will ",
                    "first, i'll",
                    "first i ",
                    "let's ",
                ];
                let may_be_narration = NARRATION_PREFIXES.iter().any(|prefix| {
                    prefix.starts_with(&normalized) || normalized.starts_with(prefix)
                });
                if may_be_narration {
                    if self.pending.contains('\n') {
                        self.mode = Some(GrokChunkMode::Buffered);
                    }
                    return None;
                }
                self.mode = Some(GrokChunkMode::Text);
                self.emit_safe_prefix()
            }
        }
    }

    fn emit_safe_prefix(&mut self) -> Option<String> {
        if let Some(marker) = self.pending.find('<') {
            self.mode = Some(GrokChunkMode::Buffered);
            let text = self.pending[..marker].to_string();
            self.pending.clear();
            if !text.is_empty() {
                self.emitted_text = true;
                return Some(text);
            }
            return None;
        }
        const HELD_CHARS: usize = 16;
        let char_count = self.pending.chars().count();
        if char_count <= HELD_CHARS {
            return None;
        }
        let split = self
            .pending
            .char_indices()
            .nth(char_count - HELD_CHARS)
            .map_or(0, |(index, _)| index);
        let tail = self.pending.split_off(split);
        let text = std::mem::replace(&mut self.pending, tail);
        if text.is_empty() {
            None
        } else {
            self.emitted_text = true;
            Some(text)
        }
    }

    fn finish_text(&mut self) -> Option<String> {
        if self.mode == Some(GrokChunkMode::Text)
            && !self.pending.contains('<')
            && !self.pending.is_empty()
        {
            self.emitted_text = true;
            return Some(std::mem::take(&mut self.pending));
        }
        None
    }

    fn streamed_text(&self) -> bool {
        self.emitted_text
    }
}

fn run_grok_proxy_pass(
    model: &str,
    prompt: &str,
    system: &str,
    cancelled: &AtomicBool,
    tool_specs: &[RuntimeToolSpec],
    output: &mut (impl Write + ?Sized),
    defer_buffered_output: bool,
) -> Result<GrokPass, RuntimeError> {
    let mut gate = GrokChunkGate::default();
    let mut events = Vec::new();
    let mut assistant_lead_emitted = false;
    let full_text = grok_acp::run(model, prompt, system, cancelled, |chunk| {
        if let Some(text) = gate.push(chunk) {
            render_grok_text_chunk(output, &text, &mut assistant_lead_emitted)?;
            events.push(AssistantEvent::TextDelta(text));
        }
        Ok(())
    })?;
    if let Some(text) = gate.finish_text() {
        render_grok_text_chunk(output, &text, &mut assistant_lead_emitted)?;
        events.push(AssistantEvent::TextDelta(text));
    }
    if gate.streamed_text() {
        for segment in parse_proxy_response(&full_text, tool_specs).map_err(RuntimeError::new)? {
            if let ProxySegment::ToolUse { id, name, input } = segment {
                events.push(AssistantEvent::ToolUse { id, name, input });
            }
        }
    } else {
        let mut deferred_render = Vec::new();
        if defer_buffered_output {
            append_proxy_text_events(&full_text, &mut deferred_render, &mut events, tool_specs)?;
        } else {
            append_proxy_text_events(&full_text, output, &mut events, tool_specs)?;
        }
        return Ok(GrokPass {
            events,
            deferred_render,
        });
    }
    Ok(GrokPass {
        events,
        deferred_render: Vec::new(),
    })
}

struct GrokPass {
    events: Vec<AssistantEvent>,
    deferred_render: Vec<u8>,
}

fn render_grok_text_chunk(
    output: &mut (impl Write + ?Sized),
    text: &str,
    assistant_lead_emitted: &mut bool,
) -> Result<(), RuntimeError> {
    if !*assistant_lead_emitted {
        write!(output, "{}", ui::assistant_lead())
            .and_then(|()| output.flush())
            .map_err(|error| RuntimeError::new(error.to_string()))?;
        *assistant_lead_emitted = true;
    }
    write!(output, "{text}")
        .and_then(|()| output.flush())
        .map_err(|error| RuntimeError::new(error.to_string()))
}

#[cfg(test)]
mod grok_stream_tests {
    use super::{render_grok_cli_prompt, GrokChunkGate, ProxyMessage};

    #[test]
    fn renders_role_delimited_grok_prompt() {
        let prompt = render_grok_cli_prompt(&[
            ProxyMessage {
                role: "user".to_string(),
                content: "inspect this".to_string(),
            },
            ProxyMessage {
                role: "assistant".to_string(),
                content: "working".to_string(),
            },
        ]);
        assert_eq!(prompt, "[user]\ninspect this\n\n[assistant]\nworking");
    }

    #[test]
    fn streams_plain_text_but_never_proxy_markup() {
        let mut text = GrokChunkGate::default();
        let mut visible = String::new();
        for chunk in ["This is a long enough ", "answer to stream safely."] {
            if let Some(chunk) = text.push(chunk) {
                visible.push_str(&chunk);
            }
        }
        visible.push_str(&text.finish_text().unwrap_or_default());
        assert_eq!(visible, "This is a long enough answer to stream safely.");

        let mut tool = GrokChunkGate::default();
        assert_eq!(tool.push("<tool_"), None);
        assert_eq!(tool.push("call name=\"read_file\">"), None);
        assert_eq!(tool.finish_text(), None);

        let mut mixed = GrokChunkGate::default();
        assert!(mixed
            .push("A visible answer that is already long enough to emit ")
            .is_some());
        assert!(mixed.push("<tool_call name=\"read_file\">").is_some());
        assert!(mixed.streamed_text());
    }
}

fn proxy_retry_reminder() -> &'static str {
    "Your previous reply only narrated intent. Do not narrate. If tool use is needed, emit the next XML <tool_call> block immediately with no prefatory text. If no tool is needed, answer directly."
}

pub(crate) fn should_retry_proxy_tool_prompt(events: &[AssistantEvent]) -> bool {
    if events
        .iter()
        .any(|event| matches!(event, AssistantEvent::ToolUse { .. }))
    {
        return false;
    }

    let text = events
        .iter()
        .filter_map(|event| match event {
            AssistantEvent::TextDelta(text) => Some(text.as_str()),
            _ => None,
        })
        .collect::<String>();
    let trimmed = text.trim();
    if trimmed.is_empty() || trimmed.chars().count() > 280 {
        return false;
    }

    let normalized = trimmed.to_ascii_lowercase();
    let intent_prefix = [
        "let me ",
        "i'll ",
        "i will ",
        "first, i'll",
        "first i ",
        "let's ",
    ];
    let tool_intent = [
        "explore",
        "inspect",
        "look at",
        "check",
        "review",
        "understand",
        "start by",
        "begin by",
    ];

    intent_prefix
        .iter()
        .any(|prefix| normalized.starts_with(prefix))
        && tool_intent.iter().any(|phrase| normalized.contains(phrase))
}

pub(crate) fn push_output_block(
    block: OutputContentBlock,
    out: &mut (impl Write + ?Sized),
    events: &mut Vec<AssistantEvent>,
    pending_tool: &mut Option<(String, String, String)>,
    streaming_tool_input: bool,
) -> Result<(), RuntimeError> {
    match block {
        OutputContentBlock::Text { text } => {
            if !text.is_empty() {
                let rendered = TerminalRenderer::new().markdown_to_ansi(&text);
                write!(out, "{rendered}")
                    .and_then(|()| out.flush())
                    .map_err(|error| RuntimeError::new(error.to_string()))?;
                events.push(AssistantEvent::TextDelta(text));
            }
        }
        OutputContentBlock::Thinking {
            thinking,
            signature,
        } => {
            if !thinking.is_empty() {
                events.push(AssistantEvent::ThinkingDelta(thinking));
            }
            if let Some(signature) = signature.filter(|signature| !signature.is_empty()) {
                events.push(AssistantEvent::ThinkingSignature(signature));
            }
        }
        OutputContentBlock::RedactedThinking { .. } => {}
        OutputContentBlock::ToolUse { id, name, input } => {
            let initial_input = if streaming_tool_input
                && input.is_object()
                && input.as_object().is_some_and(serde_json::Map::is_empty)
            {
                String::new()
            } else {
                input.to_string()
            };
            *pending_tool = Some((id, name, initial_input));
        }
    }
    Ok(())
}

pub(crate) fn render_streamed_tool_call_start(
    out: &mut (impl Write + ?Sized),
    name: &str,
    input: &str,
) -> Result<(), RuntimeError> {
    writeln!(out, "{}", format_tool_call_start(name, input))
        .and_then(|()| out.flush())
        .map_err(|error| RuntimeError::new(error.to_string()))
}

fn format_tool_call_start(name: &str, input: &str) -> String {
    let parsed =
        serde_json::from_str::<JsonValue>(input).unwrap_or(JsonValue::String(input.to_string()));
    let detail = match name {
        "bash" | "Bash" => parsed
            .get("command")
            .and_then(JsonValue::as_str)
            .map(|command| truncate_for_summary(command, 120))
            .unwrap_or_default(),
        "read_file" | "Read" => parsed
            .get("file_path")
            .or_else(|| parsed.get("path"))
            .and_then(JsonValue::as_str)
            .unwrap_or("?")
            .to_string(),
        "write_file" | "Write" => {
            let path = parsed
                .get("file_path")
                .or_else(|| parsed.get("path"))
                .and_then(JsonValue::as_str)
                .unwrap_or("?");
            let lines = parsed
                .get("content")
                .and_then(JsonValue::as_str)
                .map_or(0, |content| content.lines().count());
            format!("{path} ({lines} lines)")
        }
        "edit_file" | "Edit" => parsed
            .get("file_path")
            .or_else(|| parsed.get("path"))
            .and_then(JsonValue::as_str)
            .unwrap_or("?")
            .to_string(),
        "apply_patch" => {
            let dry_run = parsed
                .get("dry_run")
                .and_then(JsonValue::as_bool)
                .unwrap_or(false);
            let patch_lines = parsed
                .get("patch")
                .and_then(JsonValue::as_str)
                .map_or(0, |patch| patch.lines().count());
            let mode = if dry_run { "check" } else { "apply" };
            format!("{mode} ({patch_lines} lines)")
        }
        "glob_search" | "Glob" | "grep_search" | "Grep" => parsed
            .get("pattern")
            .and_then(JsonValue::as_str)
            .unwrap_or("?")
            .to_string(),
        "web_search" | "WebSearch" => parsed
            .get("query")
            .and_then(JsonValue::as_str)
            .unwrap_or("?")
            .to_string(),
        _ => summarize_tool_payload(input),
    };

    let detail_width = terminal_size()
        .ok()
        .map_or(80, |(columns, _)| usize::from(columns).saturating_sub(18))
        .max(12);
    ui::tool_call_header(name, &truncate_for_summary(&detail, detail_width))
}

pub(crate) fn response_to_events(
    response: MessageResponse,
    out: &mut (impl Write + ?Sized),
) -> Result<Vec<AssistantEvent>, RuntimeError> {
    let mut events = Vec::new();
    let mut pending_tool = None;
    let mut assistant_lead_emitted = false;

    for block in response.content {
        if matches!(&block, OutputContentBlock::Text { text } if !text.is_empty())
            && !assistant_lead_emitted
        {
            write!(out, "{}", ui::assistant_lead())
                .and_then(|()| out.flush())
                .map_err(|error| RuntimeError::new(error.to_string()))?;
            assistant_lead_emitted = true;
        }
        push_output_block(block, out, &mut events, &mut pending_tool, false)?;
        if let Some((id, name, input)) = pending_tool.take() {
            events.push(AssistantEvent::ToolUse { id, name, input });
        }
    }

    events.push(AssistantEvent::Usage(TokenUsage {
        input_tokens: response.usage.input_tokens,
        output_tokens: response.usage.output_tokens,
        cache_creation_input_tokens: response.usage.cache_creation_input_tokens,
        cache_read_input_tokens: response.usage.cache_read_input_tokens,
    }));
    events.push(AssistantEvent::MessageStop);
    Ok(events)
}

pub(crate) fn proxy_response_to_events(
    response: MessageResponse,
    out: &mut (impl Write + ?Sized),
    tool_specs: &[RuntimeToolSpec],
) -> Result<Vec<AssistantEvent>, RuntimeError> {
    let mut events = Vec::new();
    for block in response.content {
        match block {
            OutputContentBlock::Text { text } => {
                append_proxy_text_events(&text, out, &mut events, tool_specs)?;
            }
            OutputContentBlock::Thinking {
                thinking,
                signature,
            } => {
                if !thinking.is_empty() {
                    events.push(AssistantEvent::ThinkingDelta(thinking));
                }
                if let Some(signature) = signature.filter(|signature| !signature.is_empty()) {
                    events.push(AssistantEvent::ThinkingSignature(signature));
                }
            }
            OutputContentBlock::RedactedThinking { .. } => {}
            OutputContentBlock::ToolUse { id, name, input } => {
                events.push(AssistantEvent::ToolUse {
                    id,
                    name,
                    input: input.to_string(),
                });
            }
        }
    }

    events.push(AssistantEvent::Usage(TokenUsage {
        input_tokens: response.usage.input_tokens,
        output_tokens: response.usage.output_tokens,
        cache_creation_input_tokens: response.usage.cache_creation_input_tokens,
        cache_read_input_tokens: response.usage.cache_read_input_tokens,
    }));
    events.push(AssistantEvent::MessageStop);
    Ok(events)
}

pub(crate) fn append_proxy_text_events(
    text: &str,
    out: &mut (impl Write + ?Sized),
    events: &mut Vec<AssistantEvent>,
    tool_specs: &[RuntimeToolSpec],
) -> Result<(), RuntimeError> {
    let segments = parse_proxy_response(text, tool_specs).map_err(RuntimeError::new)?;
    let has_tool_use = segments
        .iter()
        .any(|segment| matches!(segment, ProxySegment::ToolUse { .. }));
    let mut assistant_lead_emitted = false;
    for segment in segments {
        match segment {
            ProxySegment::Text(text) => {
                if !text.is_empty() && should_render_proxy_text_segment(&text, has_tool_use) {
                    if !assistant_lead_emitted {
                        write!(out, "{}", ui::assistant_lead())
                            .and_then(|()| out.flush())
                            .map_err(|error| RuntimeError::new(error.to_string()))?;
                        assistant_lead_emitted = true;
                    }
                    write!(out, "{text}")
                        .and_then(|()| out.flush())
                        .map_err(|error| RuntimeError::new(error.to_string()))?;
                    events.push(AssistantEvent::TextDelta(text));
                }
            }
            ProxySegment::ToolUse { id, name, input } => {
                events.push(AssistantEvent::ToolUse { id, name, input });
            }
        }
    }
    Ok(())
}

fn should_render_proxy_text_segment(text: &str, has_tool_use: bool) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return false;
    }

    if contains_proxy_markup(trimmed) {
        return false;
    }

    if has_tool_use {
        let normalized = trimmed.to_ascii_lowercase();
        let boilerplate = [
            "now let me",
            "let me",
            "i'll",
            "i will",
            "first, i'll",
            "first i will",
            "first i'll",
            "creating",
            "writing",
            "saving",
            "updating",
            "reading",
            "editing",
        ];
        if boilerplate
            .iter()
            .any(|prefix| normalized.starts_with(prefix))
        {
            return false;
        }
    }

    true
}

fn contains_proxy_markup(text: &str) -> bool {
    let normalized = text.to_ascii_lowercase();
    let markers = [
        "<tool_call",
        "</tool_call",
        "<tool_result",
        "</tool_result",
        "<arg",
        "</arg",
        "</parameter",
        "<read_file",
        "<write_file",
        "<edit_file",
        "<apply_patch",
        "<bash",
        "<glob_search",
        "<grep_search",
        "<webfetch",
        "<websearch",
        "<todowrite",
        "<skill",
        "<agent",
        "<toolsearch",
        "<notebookedit",
        "<sleep",
        "<powershell",
    ];
    markers.iter().any(|marker| normalized.contains(marker))
}

pub(crate) struct CliToolExecutor {
    service: ApiService,
    tool_registry: GlobalToolRegistry,
    mcp_catalog: McpCatalog,
    tool_specs: Vec<RuntimeToolSpec>,
    allowed_tools: Option<AllowedToolSet>,
    emit_output: bool,
}

impl CliToolExecutor {
    pub(crate) fn new(
        service: ApiService,
        tool_registry: GlobalToolRegistry,
        mcp_catalog: McpCatalog,
        tool_specs: Vec<RuntimeToolSpec>,
        allowed_tools: Option<AllowedToolSet>,
        emit_output: bool,
    ) -> Self {
        Self {
            service,
            tool_registry,
            mcp_catalog,
            tool_specs,
            allowed_tools,
            emit_output,
        }
    }
}

impl ToolExecutor for CliToolExecutor {
    fn execute(&mut self, tool_name: &str, input: &str) -> Result<String, ToolError> {
        self.execute_cancellable(tool_name, input, &CancellationToken::new())
    }

    fn execute_cancellable(
        &mut self,
        tool_name: &str,
        input: &str,
        cancellation: &CancellationToken,
    ) -> Result<String, ToolError> {
        if cancellation.is_cancelled() {
            return Err(ToolError::cancelled());
        }
        if self
            .allowed_tools
            .as_ref()
            .is_some_and(|allowed| !allowed.contains(tool_name))
        {
            return Err(ToolError::new(format!(
                "tool `{tool_name}` is not enabled by the current --allowedTools setting"
            )));
        }
        let value = parse_tool_input_value(tool_name, input, &self.tool_specs)?;
        let _service_guard = set_active_backend_service(self.service);
        let _cancellation_guard = set_active_cancellation(cancellation.clone());
        let output = if let Some(tool) = self.mcp_catalog.find_tool(tool_name) {
            call_mcp_tool(tool, &value, cancellation)
                .map_err(|error| ToolError::new(error.to_string()))
        } else {
            execute_registry_tool_cancellable(
                self.tool_registry.clone(),
                tool_name.to_string(),
                value,
                cancellation.clone(),
            )
        };
        if cancellation.is_cancelled() {
            return Err(ToolError::cancelled());
        }
        let output = output?;
        if self.emit_output {
            let block = render_tool_result_block(tool_name, &output);
            if !block.is_empty() {
                let mut stdout = io::stdout();
                write!(stdout, "{block}")
                    .and_then(|()| stdout.flush())
                    .map_err(|error| ToolError::new(error.to_string()))?;
            }
        }
        Ok(output)
    }
}

fn execute_registry_tool_cancellable(
    registry: GlobalToolRegistry,
    tool_name: String,
    input: JsonValue,
    cancellation: CancellationToken,
) -> Result<String, ToolError> {
    let (sender, receiver) = std::sync::mpsc::sync_channel(1);
    let tool_cancellation = cancellation.clone();
    std::thread::spawn(move || {
        let _cancellation_guard = set_active_cancellation(tool_cancellation);
        let result = registry.execute(&tool_name, &input).map_err(ToolError::new);
        let _ = sender.send(result);
    });

    loop {
        if cancellation.is_cancelled() {
            return Err(ToolError::cancelled());
        }
        match receiver.recv_timeout(std::time::Duration::from_millis(20)) {
            Ok(result) => return result,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                return Err(ToolError::new("tool worker stopped without a result"));
            }
        }
    }
}

pub(crate) fn parse_tool_input_value(
    tool_name: &str,
    input: &str,
    tool_specs: &[RuntimeToolSpec],
) -> Result<serde_json::Value, ToolError> {
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(input) {
        return Ok(value);
    }

    if let Some(value) = extract_first_json_object(input) {
        return Ok(value);
    }

    if input.contains("<tool_call") {
        let tool_use = parse_proxy_response(input, tool_specs)
            .map_err(ToolError::new)?
            .into_iter()
            .find_map(|segment| match segment {
                ProxySegment::ToolUse { name, input, .. } if name == tool_name => Some(input),
                _ => None,
            })
            .ok_or_else(|| ToolError::new("proxy tool call did not contain a matching tool"))?;
        return serde_json::from_str(&tool_use).map_err(|error| {
            ToolError::new(format!("invalid recovered proxy tool JSON: {error}"))
        });
    }

    if input.contains("<arg") {
        let wrapped = format!("<tool_call name=\"{tool_name}\">{input}</tool_call>");
        let tool_use = parse_proxy_response(&wrapped, tool_specs)
            .map_err(ToolError::new)?
            .into_iter()
            .find_map(|segment| match segment {
                ProxySegment::ToolUse { input, .. } => Some(input),
                _ => None,
            })
            .ok_or_else(|| ToolError::new("proxy arg fragment did not produce tool input"))?;
        return serde_json::from_str(&tool_use)
            .map_err(|error| ToolError::new(format!("invalid recovered proxy arg JSON: {error}")));
    }

    Err(ToolError::new(format!(
        "invalid tool input JSON: could not parse {input:?}"
    )))
}

pub(crate) fn extract_first_json_object(input: &str) -> Option<serde_json::Value> {
    let start = input.find('{')?;
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;

    for (offset, ch) in input[start..].char_indices() {
        if in_string {
            if escaped {
                escaped = false;
                continue;
            }
            match ch {
                '\\' => escaped = true,
                '"' => in_string = false,
                _ => {}
            }
            continue;
        }

        match ch {
            '"' => in_string = true,
            '{' => depth += 1,
            '}' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    let end = start + offset + ch.len_utf8();
                    return serde_json::from_str(&input[start..end]).ok();
                }
            }
            _ => {}
        }
    }

    None
}

pub(crate) fn permission_policy(
    mode: PermissionMode,
    tool_specs: &[RuntimeToolSpec],
) -> PermissionPolicy {
    tool_specs
        .iter()
        .fold(PermissionPolicy::new(mode), |policy, spec| {
            policy.with_tool_requirement(spec.name.clone(), spec.required_permission)
        })
}

pub(crate) fn convert_messages(
    messages: &[ConversationMessage],
    service: ApiService,
) -> Result<Vec<InputMessage>, RuntimeError> {
    let cwd = env::current_dir().map_err(|error| {
        RuntimeError::new(format!("failed to resolve current directory: {error}"))
    })?;
    sanitize_messages_for_api(messages)
        .iter()
        .filter_map(|message| {
            let role = match message.role {
                MessageRole::System | MessageRole::User | MessageRole::Tool => "user",
                MessageRole::Assistant => "assistant",
            };
            let content = message
                .blocks
                .iter()
                .try_fold(Vec::new(), |mut acc, block| {
                    match block {
                        ContentBlock::Text { text } => {
                            if message.role == MessageRole::User {
                                acc.extend(
                                    prompt_to_content_blocks(text, &cwd)
                                        .map_err(RuntimeError::new)?,
                                );
                            } else {
                                acc.push(InputContentBlock::Text { text: text.clone() });
                            }
                        }
                        ContentBlock::Thinking { .. } => {}
                        ContentBlock::ToolUse { id, name, input } => {
                            acc.push(InputContentBlock::ToolUse {
                                id: id.clone(),
                                name: name.clone(),
                                input: serde_json::from_str(input)
                                    .unwrap_or_else(|_| serde_json::json!({ "raw": input })),
                            });
                        }
                        ContentBlock::ToolResult {
                            tool_use_id,
                            output,
                            is_error,
                            compacted,
                            ..
                        } => acc.push(InputContentBlock::ToolResult {
                            tool_use_id: tool_use_id.clone(),
                            content: vec![ToolResultContentBlock::Text {
                                text: get_tool_result_context_output(output, *compacted)
                                    .into_owned(),
                            }],
                            is_error: *is_error,
                        }),
                        ContentBlock::CompactionSummary {
                            summary,
                            recent_messages_preserved,
                            ..
                        } => acc.push(InputContentBlock::Text {
                            text: get_compact_continuation_message(
                                summary,
                                true,
                                *recent_messages_preserved,
                            ),
                        }),
                    }
                    Ok::<_, RuntimeError>(acc)
                });
            let reasoning_content =
                if service == ApiService::OpencodeGo && message.role == MessageRole::Assistant {
                    let reasoning = message
                        .blocks
                        .iter()
                        .filter_map(|block| match block {
                            ContentBlock::Thinking { text, .. } if !text.is_empty() => {
                                Some(text.as_str())
                            }
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("\n\n");
                    (!reasoning.is_empty()).then_some(reasoning)
                } else {
                    None
                };
            match content {
                Ok(content) if !content.is_empty() => Some(Ok(InputMessage {
                    role: role.to_string(),
                    content,
                    reasoning_content: reasoning_content.clone(),
                    reasoning: reasoning_content,
                })),
                Ok(_) => None,
                Err(error) => Some(Err(error)),
            }
        })
        .collect()
}

fn sanitize_messages_for_api(messages: &[ConversationMessage]) -> Vec<ConversationMessage> {
    let tool_use_ids = messages
        .iter()
        .flat_map(|message| message.blocks.iter())
        .filter_map(|block| match block {
            ContentBlock::ToolUse { id, .. } => Some(id.clone()),
            _ => None,
        })
        .collect::<HashSet<_>>();
    let tool_result_ids = messages
        .iter()
        .flat_map(|message| message.blocks.iter())
        .filter_map(|block| match block {
            ContentBlock::ToolResult { tool_use_id, .. } => Some(tool_use_id.clone()),
            _ => None,
        })
        .collect::<HashSet<_>>();
    let balanced_tool_ids = tool_use_ids
        .intersection(&tool_result_ids)
        .cloned()
        .collect::<HashSet<_>>();

    messages
        .iter()
        .filter_map(|message| {
            let blocks = message
                .blocks
                .iter()
                .filter(|block| match block {
                    ContentBlock::ToolUse { id, .. } => balanced_tool_ids.contains(id),
                    ContentBlock::ToolResult { tool_use_id, .. } => {
                        balanced_tool_ids.contains(tool_use_id)
                    }
                    _ => true,
                })
                .cloned()
                .collect::<Vec<_>>();
            if blocks.is_empty() {
                return None;
            }
            Some(ConversationMessage {
                id: message.id.clone(),
                role: message.role,
                blocks,
                usage: message.usage,
            })
        })
        .collect()
}

pub(crate) fn prompt_to_content_blocks(
    input: &str,
    cwd: &Path,
) -> Result<Vec<InputContentBlock>, String> {
    let mut blocks = Vec::new();
    let mut text_buffer = String::new();
    let mut chars = input.char_indices().peekable();

    while let Some((index, ch)) = chars.next() {
        if ch == '!' && input[index..].starts_with("![") {
            if let Some((_, path_start, path_end)) = parse_markdown_image_ref(input, index) {
                flush_text_block(&mut blocks, &mut text_buffer);
                let path = &input[path_start..path_end];
                blocks.push(load_image_block(path, cwd)?);
                while let Some((next_index, _)) = chars.peek() {
                    if *next_index < path_end + 1 {
                        let _ = chars.next();
                    } else {
                        break;
                    }
                }
                continue;
            }
        }

        if ch == '@' && is_ref_boundary(input[..index].chars().next_back()) {
            let path_end = find_path_end(input, index + 1);
            if path_end > index + 1 {
                let candidate = &input[index + 1..path_end];
                if looks_like_image_ref(candidate, cwd) {
                    flush_text_block(&mut blocks, &mut text_buffer);
                    blocks.push(load_image_block(candidate, cwd)?);
                    while let Some((next_index, _)) = chars.peek() {
                        if *next_index < path_end {
                            let _ = chars.next();
                        } else {
                            break;
                        }
                    }
                    continue;
                }
                if looks_like_unsupported_image_ref(candidate) {
                    flush_text_block(&mut blocks, &mut text_buffer);
                    blocks.push(load_image_block(candidate, cwd)?);
                    while let Some((next_index, _)) = chars.peek() {
                        if *next_index < path_end {
                            let _ = chars.next();
                        } else {
                            break;
                        }
                    }
                    continue;
                }
                if looks_like_file_ref(candidate, cwd) {
                    flush_text_block(&mut blocks, &mut text_buffer);
                    blocks.push(load_file_reference_block(candidate, cwd)?);
                    while let Some((next_index, _)) = chars.peek() {
                        if *next_index < path_end {
                            let _ = chars.next();
                        } else {
                            break;
                        }
                    }
                    continue;
                }
            }
        }

        text_buffer.push(ch);
    }

    flush_text_block(&mut blocks, &mut text_buffer);
    if blocks.is_empty() {
        blocks.push(InputContentBlock::Text {
            text: input.to_string(),
        });
    }
    Ok(blocks)
}

fn parse_markdown_image_ref(input: &str, start: usize) -> Option<(usize, usize, usize)> {
    let after_bang = input.get(start + 2..)?;
    let alt_end_offset = after_bang.find("](")?;
    let path_start = start + 2 + alt_end_offset + 2;
    let remainder = input.get(path_start..)?;
    let path_end_offset = remainder.find(')')?;
    let path_end = path_start + path_end_offset;
    Some((start + 2 + alt_end_offset, path_start, path_end))
}

fn is_ref_boundary(ch: Option<char>) -> bool {
    ch.is_none_or(char::is_whitespace)
}

fn find_path_end(input: &str, start: usize) -> usize {
    input[start..]
        .char_indices()
        .find_map(|(offset, ch)| ch.is_whitespace().then_some(start + offset))
        .unwrap_or(input.len())
}

fn looks_like_image_ref(candidate: &str, cwd: &Path) -> bool {
    let resolved = resolve_prompt_path(candidate, cwd);
    media_type_for_path(Path::new(candidate)).is_some() || media_type_for_path(&resolved).is_some()
}

fn looks_like_file_ref(candidate: &str, cwd: &Path) -> bool {
    let resolved = resolve_prompt_path(candidate, cwd);
    (resolved.is_file() || resolved.is_dir()) && media_type_for_path(Path::new(candidate)).is_none()
}

fn looks_like_unsupported_image_ref(candidate: &str) -> bool {
    Path::new(candidate)
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            matches!(
                extension.to_ascii_lowercase().as_str(),
                "avif" | "bmp" | "heic" | "heif" | "svg" | "tif" | "tiff"
            )
        })
}

fn flush_text_block(blocks: &mut Vec<InputContentBlock>, text_buffer: &mut String) {
    if text_buffer.is_empty() {
        return;
    }
    blocks.push(InputContentBlock::Text {
        text: std::mem::take(text_buffer),
    });
}

fn load_image_block(path_ref: &str, cwd: &Path) -> Result<InputContentBlock, String> {
    let resolved = resolve_prompt_path(path_ref, cwd);
    let media_type = media_type_for_path(&resolved).ok_or_else(|| {
        format!(
            "unsupported image format for reference {IMAGE_REF_PREFIX}{path_ref}; supported: png, jpg, jpeg, gif, webp"
        )
    })?;
    let bytes = fs::read(&resolved).map_err(|error| {
        format!(
            "failed to read image reference {}: {error}",
            resolved.display()
        )
    })?;
    Ok(InputContentBlock::Image {
        source: ImageSource {
            kind: "base64".to_string(),
            media_type: media_type.to_string(),
            data: encode_base64(&bytes),
        },
    })
}

fn load_file_reference_block(path_ref: &str, cwd: &Path) -> Result<InputContentBlock, String> {
    let resolved = resolve_prompt_path(path_ref, cwd);
    if resolved.is_dir() {
        return Ok(InputContentBlock::Text {
            text: render_directory_reference(path_ref, &resolved)?,
        });
    }

    let metadata = fs::metadata(&resolved).map_err(|error| {
        format!(
            "failed to stat file reference {}: {error}",
            resolved.display()
        )
    })?;
    if metadata.len() > FILE_REF_MAX_BYTES {
        return Err(format!(
            "file reference {} is too large ({} bytes, limit {} bytes)",
            resolved.display(),
            metadata.len(),
            FILE_REF_MAX_BYTES
        ));
    }
    let content = fs::read_to_string(&resolved).map_err(|error| {
        format!(
            "failed to read file reference {}: {error}",
            resolved.display()
        )
    })?;
    Ok(InputContentBlock::Text {
        text: format!(
            "File reference: {path_ref}\n```text\n{content}\n```",
            content = content.trim_end()
        ),
    })
}

fn render_directory_reference(path_ref: &str, resolved: &Path) -> Result<String, String> {
    let mut entries = Vec::new();
    collect_directory_reference_entries(resolved, resolved, &mut entries)?;
    entries.sort();
    let truncated = entries.len() > DIRECTORY_REF_MAX_ENTRIES;
    entries.truncate(DIRECTORY_REF_MAX_ENTRIES);
    let mut text = format!("Directory reference: {path_ref}\n");
    if entries.is_empty() {
        text.push_str("(empty directory)");
    } else {
        for entry in &entries {
            let _ = writeln!(&mut text, "- {entry}");
        }
        if truncated {
            let _ = writeln!(
                &mut text,
                "- ... truncated after {DIRECTORY_REF_MAX_ENTRIES} entries"
            );
        }
    }
    Ok(text)
}

fn collect_directory_reference_entries(
    root: &Path,
    dir: &Path,
    entries: &mut Vec<String>,
) -> Result<(), String> {
    if entries.len() > DIRECTORY_REF_MAX_ENTRIES {
        return Ok(());
    }
    let read_dir = fs::read_dir(dir).map_err(|error| {
        format!(
            "failed to read directory reference {}: {error}",
            dir.display()
        )
    })?;
    for entry in read_dir {
        let entry =
            entry.map_err(|error| format!("failed to read directory reference entry: {error}"))?;
        let path = entry.path();
        let name = entry.file_name();
        if name.to_string_lossy().starts_with('.') {
            continue;
        }
        let relative = path
            .strip_prefix(root)
            .unwrap_or(&path)
            .display()
            .to_string();
        if path.is_dir() {
            entries.push(format!("{relative}/"));
            collect_directory_reference_entries(root, &path, entries)?;
        } else {
            entries.push(relative);
        }
        if entries.len() > DIRECTORY_REF_MAX_ENTRIES {
            break;
        }
    }
    Ok(())
}

fn resolve_prompt_path(path_ref: &str, cwd: &Path) -> PathBuf {
    let path = Path::new(path_ref);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        cwd.join(path)
    }
}

fn media_type_for_path(path: &Path) -> Option<&'static str> {
    let extension = path.extension()?.to_str()?.to_ascii_lowercase();
    match extension.as_str() {
        "png" => Some("image/png"),
        "jpg" | "jpeg" => Some("image/jpeg"),
        "gif" => Some("image/gif"),
        "webp" => Some("image/webp"),
        _ => None,
    }
}

fn encode_base64(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut output = String::new();
    let mut index = 0;
    while index + 3 <= bytes.len() {
        let block = (u32::from(bytes[index]) << 16)
            | (u32::from(bytes[index + 1]) << 8)
            | u32::from(bytes[index + 2]);
        output.push(TABLE[((block >> 18) & 0x3F) as usize] as char);
        output.push(TABLE[((block >> 12) & 0x3F) as usize] as char);
        output.push(TABLE[((block >> 6) & 0x3F) as usize] as char);
        output.push(TABLE[(block & 0x3F) as usize] as char);
        index += 3;
    }

    match bytes.len().saturating_sub(index) {
        1 => {
            let block = u32::from(bytes[index]) << 16;
            output.push(TABLE[((block >> 18) & 0x3F) as usize] as char);
            output.push(TABLE[((block >> 12) & 0x3F) as usize] as char);
            output.push('=');
            output.push('=');
        }
        2 => {
            let block = (u32::from(bytes[index]) << 16) | (u32::from(bytes[index + 1]) << 8);
            output.push(TABLE[((block >> 18) & 0x3F) as usize] as char);
            output.push(TABLE[((block >> 12) & 0x3F) as usize] as char);
            output.push(TABLE[((block >> 6) & 0x3F) as usize] as char);
            output.push('=');
        }
        _ => {}
    }

    output
}

fn convert_proxy_messages_to_input_messages(messages: Vec<ProxyMessage>) -> Vec<InputMessage> {
    messages
        .into_iter()
        .map(|message| InputMessage {
            role: message.role,
            content: vec![InputContentBlock::Text {
                text: message.content,
            }],
            reasoning_content: None,
            reasoning: None,
        })
        .collect()
}

pub(crate) fn effective_reasoning_effort(
    collaboration_mode: CollaborationMode,
    reasoning_effort: Option<ReasoningEffort>,
) -> Option<ReasoningEffort> {
    reasoning_effort.or(match collaboration_mode {
        CollaborationMode::Build => None,
        CollaborationMode::Plan => Some(ReasoningEffort::Medium),
    })
}
