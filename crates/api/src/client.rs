use std::collections::{BTreeMap, VecDeque};
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use platform::{pebble_config_home, write_atomic};
use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;

use crate::error::ApiError;
use crate::sse::SseParser;
use crate::types::{
    ChatCompletionContent, ChatCompletionFunction, ChatCompletionMessage, ChatCompletionRequest,
    ChatCompletionResponse, ChatCompletionThinkingConfig, ChatCompletionTool, MessageRequest,
    MessageResponse, ModelsResponse, OutputContentBlock, ProviderSelectionResponse,
    ReasoningEffort, StreamEvent, Usage,
};

const DEFAULT_BASE_URL: &str = "https://nano-gpt.com/api";
const DEFAULT_SYNTHETIC_MESSAGES_BASE_URL: &str = "https://api.synthetic.new/anthropic/v1";
const DEFAULT_OPENCODE_GO_BASE_URL: &str = "https://opencode.ai/zen/go";
const DEFAULT_OPENAI_CODEX_BASE_URL: &str = "https://chatgpt.com/backend-api/codex";
const DEFAULT_NEURALWATT_BASE_URL: &str = "https://api.neuralwatt.com";
const DEFAULT_LILAC_BASE_URL: &str = "https://api.getlilac.com";
const REQUEST_ID_HEADER: &str = "request-id";
const ALT_REQUEST_ID_HEADER: &str = "x-request-id";
const DEFAULT_INITIAL_BACKOFF: Duration = Duration::from_millis(200);
const DEFAULT_MAX_BACKOFF: Duration = Duration::from_secs(2);
const DEFAULT_MAX_RETRIES: u32 = 2;
const DEFAULT_API_TIMEOUT_SECS: u64 = 10 * 60;
const DEFAULT_CONNECT_TIMEOUT_SECS: u64 = 20;
const API_TIMEOUT_ENV_VAR: &str = "PEBBLE_API_TIMEOUT_SECS";
const OPENAI_CODEX_REFRESH_EARLY_MS: u64 = 30_000;
const OPENAI_CODEX_CREDENTIALS_KEY: &str = "openai_codex_auth";
pub const OPENAI_CODEX_ISSUER: &str = "https://auth.openai.com";
pub const OPENAI_CODEX_CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
pub const OPENAI_CODEX_ORIGINATOR: &str = "codex_cli_rs";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenAiCodexCredentials {
    pub access_token: String,
    pub refresh_token: String,
    #[serde(default)]
    pub expires_at: Option<u64>,
    #[serde(default)]
    pub account_id: Option<String>,
}

fn nanogpt_client_debug_enabled() -> bool {
    std::env::var("NANOGPT_CLIENT_DEBUG")
        .ok()
        .is_some_and(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on" | "debug"
            )
        })
}

#[derive(Debug, Clone)]
pub struct NanoGptClient {
    http: reqwest::Client,
    api_key: String,
    openai_codex_auth: Option<Arc<Mutex<OpenAiCodexCredentials>>>,
    base_url: String,
    service: ApiService,
    provider: Option<String>,
    force_paygo: bool,
    max_retries: u32,
    initial_backoff: Duration,
    max_backoff: Duration,
}

impl NanoGptClient {
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            http: build_http_client(),
            api_key: api_key.into(),
            openai_codex_auth: None,
            base_url: DEFAULT_BASE_URL.to_string(),
            service: ApiService::NanoGpt,
            provider: None,
            force_paygo: false,
            max_retries: DEFAULT_MAX_RETRIES,
            initial_backoff: DEFAULT_INITIAL_BACKOFF,
            max_backoff: DEFAULT_MAX_BACKOFF,
        }
    }

    pub fn from_env() -> Result<Self, ApiError> {
        Self::from_service_env(ApiService::NanoGpt)
    }

    pub fn from_service_env(service: ApiService) -> Result<Self, ApiError> {
        let client = match service {
            ApiService::OpenAiCodex => Self::new("")
                .with_service(service)
                .with_openai_codex_auth(resolve_openai_codex_credentials()?),
            _ => Self::new(resolve_api_key_for(service)?).with_service(service),
        };
        Ok(client.with_base_url(resolve_base_url_for(service)))
    }

    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    #[must_use]
    pub fn with_service(mut self, service: ApiService) -> Self {
        self.service = service;
        self
    }

    #[must_use]
    pub fn with_openai_codex_auth(mut self, auth: OpenAiCodexCredentials) -> Self {
        self.openai_codex_auth = Some(Arc::new(Mutex::new(auth)));
        self
    }

    #[must_use]
    pub fn with_provider(mut self, provider: Option<String>) -> Self {
        self.provider = provider.filter(|value| !value.is_empty());
        self.force_paygo = self.provider.is_some();
        self
    }

    #[must_use]
    pub fn with_retry_policy(
        mut self,
        max_retries: u32,
        initial_backoff: Duration,
        max_backoff: Duration,
    ) -> Self {
        self.max_retries = max_retries;
        self.initial_backoff = initial_backoff;
        self.max_backoff = max_backoff;
        self
    }

    /// Rebuild the HTTP client with a shorter request deadline for interactive
    /// probes such as credential verification.
    #[must_use]
    pub fn with_request_timeout(mut self, timeout: Duration) -> Self {
        if let Ok(http) = reqwest::Client::builder()
            .connect_timeout(timeout.min(Duration::from_secs(DEFAULT_CONNECT_TIMEOUT_SECS)))
            .timeout(timeout)
            .build()
        {
            self.http = http;
        }
        self
    }

    pub async fn send_message(
        &self,
        request: &MessageRequest,
    ) -> Result<MessageResponse, ApiError> {
        if self.service == ApiService::OpenAiCodex {
            return self.send_openai_codex_response(request).await;
        }
        if matches!(self.service, ApiService::Neuralwatt | ApiService::Lilac)
            || (self.service == ApiService::OpencodeGo
                && !opencode_go_uses_messages_api(&request.model))
        {
            return self.send_opencode_go_chat_completion(request).await;
        }
        let request = MessageRequest {
            stream: false,
            ..self.normalize_message_request(request)
        };
        let response = self.send_with_retry(&request).await?;
        let request_id = request_id_from_headers(response.headers());
        let mut response = response
            .json::<MessageResponse>()
            .await
            .map_err(ApiError::from)?;
        if response.request_id.is_none() {
            response.request_id = request_id;
        }
        Ok(response)
    }

    pub async fn send_chat_completion(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, ApiError> {
        let request_url = format!(
            "{}{}",
            self.base_url.trim_end_matches('/'),
            self.chat_completions_path()
        );
        if nanogpt_client_debug_enabled() {
            let resolved_base_url = self.base_url.trim_end_matches('/');
            eprintln!("[nanogpt-client] resolved_base_url={resolved_base_url}");
            eprintln!("[nanogpt-client] request_url={request_url}");
        }
        let request_builder = self
            .http
            .post(&request_url)
            .header("content-type", "application/json");
        let request_builder = self.apply_auth_headers(request_builder, true);
        let response = request_builder
            .json(request)
            .send()
            .await
            .map_err(ApiError::from)?;
        let response = expect_success(response).await?;
        response
            .json::<ChatCompletionResponse>()
            .await
            .map_err(ApiError::from)
    }

    pub async fn stream_message(
        &self,
        request: &MessageRequest,
    ) -> Result<MessageStream, ApiError> {
        if matches!(self.service, ApiService::Neuralwatt | ApiService::Lilac) {
            let chat_request = message_request_to_chat_completion_request(
                &request.clone().with_streaming(),
                self,
            )?;
            let response =
                expect_success(self.send_chat_completion_raw(&chat_request).await?).await?;
            return Ok(MessageStream::from_openai_response(response));
        }
        if matches!(
            self.service,
            ApiService::OpenAiCodex
                | ApiService::OpencodeGo
                | ApiService::Neuralwatt
                | ApiService::Lilac
        ) {
            let response = self.send_message(request).await?;
            return Ok(MessageStream::from_message_response(response));
        }
        let response = self
            .send_with_retry(&self.normalize_message_request(&request.clone().with_streaming()))
            .await?;
        Ok(MessageStream::from_http_response(response))
    }

    pub async fn fetch_models(&self, detailed: bool) -> Result<ModelsResponse, ApiError> {
        let response = self
            .send_get_request(
                "/v1/models",
                &[("detailed", if detailed { "true" } else { "false" })],
            )
            .await?;
        response
            .json::<ModelsResponse>()
            .await
            .map_err(ApiError::from)
    }

    pub async fn fetch_providers(
        &self,
        canonical_id: &str,
    ) -> Result<ProviderSelectionResponse, ApiError> {
        let request_url = providers_url(&self.base_url, canonical_id)?;
        if nanogpt_client_debug_enabled() {
            let resolved_base_url = self.base_url.trim_end_matches('/');
            eprintln!("[nanogpt-client] resolved_base_url={resolved_base_url}");
            eprintln!("[nanogpt-client] request_url={request_url}");
        }

        let request_builder = self.http.get(request_url);
        let request_builder = self.apply_auth_headers(request_builder, false);

        let response = request_builder.send().await.map_err(ApiError::from)?;
        let response = expect_success(response).await?;
        response
            .json::<ProviderSelectionResponse>()
            .await
            .map_err(ApiError::from)
    }

    async fn send_with_retry(
        &self,
        request: &MessageRequest,
    ) -> Result<reqwest::Response, ApiError> {
        let mut attempts = 0;
        let mut last_error: Option<ApiError>;

        loop {
            attempts += 1;
            match self.send_raw_request(request).await {
                Ok(response) => match expect_success(response).await {
                    Ok(response) => return Ok(response),
                    Err(error) if error.is_retryable() && attempts <= self.max_retries + 1 => {
                        last_error = Some(error);
                    }
                    Err(error) => return Err(error),
                },
                Err(error) if error.is_retryable() && attempts <= self.max_retries + 1 => {
                    last_error = Some(error);
                }
                Err(error) => return Err(error),
            }

            if attempts > self.max_retries {
                break;
            }

            tokio::time::sleep(self.backoff_for_attempt(attempts)?).await;
        }

        Err(ApiError::RetriesExhausted {
            attempts,
            last_error: Box::new(last_error.expect("retry loop must capture an error")),
        })
    }

    async fn send_raw_request(
        &self,
        request: &MessageRequest,
    ) -> Result<reqwest::Response, ApiError> {
        let request_url = format!(
            "{}{}",
            self.base_url.trim_end_matches('/'),
            self.messages_path()
        );
        if nanogpt_client_debug_enabled() {
            let resolved_base_url = self.base_url.trim_end_matches('/');
            eprintln!("[nanogpt-client] resolved_base_url={resolved_base_url}");
            eprintln!("[nanogpt-client] request_url={request_url}");
        }
        let request_builder = self
            .http
            .post(&request_url)
            .header("content-type", "application/json");
        let request_builder = self.apply_auth_headers(request_builder, true);

        request_builder
            .json(request)
            .send()
            .await
            .map_err(ApiError::from)
    }

    async fn send_chat_completion_raw(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<reqwest::Response, ApiError> {
        self.ensure_openai_codex_access_token(false).await?;
        let request_url = format!(
            "{}{}",
            self.base_url.trim_end_matches('/'),
            self.chat_completions_path()
        );
        if nanogpt_client_debug_enabled() {
            let resolved_base_url = self.base_url.trim_end_matches('/');
            eprintln!("[nanogpt-client] resolved_base_url={resolved_base_url}");
            eprintln!("[nanogpt-client] request_url={request_url}");
        }
        let request_builder = self
            .http
            .post(&request_url)
            .header("content-type", "application/json");
        let request_builder = self.apply_auth_headers(request_builder, true);

        request_builder
            .json(request)
            .send()
            .await
            .map_err(ApiError::from)
    }

    async fn send_get_request(
        &self,
        path: &str,
        query: &[(&str, &str)],
    ) -> Result<reqwest::Response, ApiError> {
        self.ensure_openai_codex_access_token(false).await?;
        let request_url = format!("{}{}", self.base_url.trim_end_matches('/'), path);
        if nanogpt_client_debug_enabled() {
            let resolved_base_url = self.base_url.trim_end_matches('/');
            eprintln!("[nanogpt-client] resolved_base_url={resolved_base_url}");
            eprintln!("[nanogpt-client] request_url={request_url}");
        }

        let request_builder = self.http.get(&request_url).query(query);
        let request_builder = self.apply_auth_headers(request_builder, false);

        let response = request_builder.send().await.map_err(ApiError::from)?;
        expect_success(response).await
    }

    fn current_openai_codex_auth(&self) -> Result<Option<OpenAiCodexCredentials>, ApiError> {
        let Some(auth) = &self.openai_codex_auth else {
            return Ok(None);
        };
        auth.lock()
            .map(|guard| Some(guard.clone()))
            .map_err(|_| std::io::Error::other("openai codex auth mutex poisoned").into())
    }

    async fn ensure_openai_codex_access_token(&self, force_refresh: bool) -> Result<(), ApiError> {
        if self.service != ApiService::OpenAiCodex {
            return Ok(());
        }

        let Some(current) = self.current_openai_codex_auth()? else {
            return Err(ApiError::MissingOpenAiCodexAuth);
        };

        let now = current_epoch_millis();
        let should_refresh = force_refresh
            || current.access_token.trim().is_empty()
            || current
                .expires_at
                .is_some_and(|expires_at| expires_at <= now + OPENAI_CODEX_REFRESH_EARLY_MS);

        if !should_refresh {
            return Ok(());
        }

        if current.refresh_token.trim().is_empty() {
            return if current.access_token.trim().is_empty() {
                Err(ApiError::MissingOpenAiCodexAuth)
            } else {
                Ok(())
            };
        }

        let response = self
            .http
            .post(format!("{OPENAI_CODEX_ISSUER}/oauth/token"))
            .header("content-type", "application/x-www-form-urlencoded")
            .body(
                reqwest::Url::parse_with_params(
                    "https://auth.openai.com/oauth/token",
                    &[
                        ("grant_type", "refresh_token"),
                        ("refresh_token", current.refresh_token.as_str()),
                        ("client_id", OPENAI_CODEX_CLIENT_ID),
                    ],
                )
                .map_err(std::io::Error::other)?
                .query()
                .unwrap_or_default()
                .to_string(),
            )
            .send()
            .await
            .map_err(ApiError::from)?;
        let response = expect_success(response).await?;
        let refresh = response
            .json::<OpenAiCodexRefreshResponse>()
            .await
            .map_err(ApiError::from)?;
        let access_token = refresh
            .access_token
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| {
                ApiError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "refresh token response did not include an access token",
                ))
            })?;

        let updated = OpenAiCodexCredentials {
            access_token,
            refresh_token: refresh
                .refresh_token
                .filter(|value| !value.trim().is_empty())
                .unwrap_or(current.refresh_token),
            expires_at: refresh
                .expires_in
                .map(|seconds| now.saturating_add(seconds.saturating_mul(1_000)))
                .or(current.expires_at),
            account_id: current.account_id,
        };

        if let Some(auth) = &self.openai_codex_auth {
            let mut guard = auth
                .lock()
                .map_err(|_| std::io::Error::other("openai codex auth mutex poisoned"))?;
            *guard = updated.clone();
        }
        save_openai_codex_credentials(&updated)?;
        Ok(())
    }

    fn apply_auth_headers(
        &self,
        request_builder: reqwest::RequestBuilder,
        include_provider: bool,
    ) -> reqwest::RequestBuilder {
        let debug = nanogpt_client_debug_enabled();
        let request_builder = match self.service {
            ApiService::OpenAiCodex => {
                if debug {
                    eprintln!(
                        "[nanogpt-client] headers authorization=Bearer [REDACTED] originator={OPENAI_CODEX_ORIGINATOR}"
                    );
                }
                if let Ok(Some(auth)) = self.current_openai_codex_auth() {
                    let request_builder = request_builder
                        .bearer_auth(&auth.access_token)
                        .header("originator", OPENAI_CODEX_ORIGINATOR);
                    if let Some(account_id) =
                        auth.account_id.as_deref().filter(|value| !value.is_empty())
                    {
                        request_builder.header("ChatGPT-Account-Id", account_id)
                    } else {
                        request_builder
                    }
                } else {
                    request_builder.header("originator", OPENAI_CODEX_ORIGINATOR)
                }
            }
            ApiService::NanoGpt
            | ApiService::Synthetic
            | ApiService::OpencodeGo
            | ApiService::Neuralwatt
            | ApiService::Lilac
                if self.api_key.is_empty() =>
            {
                if debug {
                    eprintln!("[nanogpt-client] headers authorization=<absent> x-api-key=<absent>");
                }
                request_builder
            }
            ApiService::NanoGpt | ApiService::OpencodeGo => {
                if debug {
                    eprintln!(
                        "[nanogpt-client] headers x-api-key=[REDACTED] authorization=Bearer [REDACTED]"
                    );
                }
                request_builder
                    .bearer_auth(&self.api_key)
                    .header("x-api-key", &self.api_key)
            }
            ApiService::Synthetic | ApiService::Neuralwatt | ApiService::Lilac => {
                if debug {
                    eprintln!("[nanogpt-client] headers authorization=Bearer [REDACTED]");
                }
                request_builder.bearer_auth(&self.api_key)
            }
            ApiService::Grok => request_builder,
        };

        if include_provider {
            if let Some(provider) = &self.provider {
                if debug {
                    eprintln!("[nanogpt-client] x-provider={provider}");
                }
                let request_builder = request_builder.header("x-provider", provider);
                if self.force_paygo {
                    if debug {
                        eprintln!("[nanogpt-client] x-billing-mode=paygo");
                    }
                    return request_builder.header("x-billing-mode", "paygo");
                }
                return request_builder;
            }
        }
        request_builder
    }

    fn backoff_for_attempt(&self, attempt: u32) -> Result<Duration, ApiError> {
        let Some(multiplier) = 1_u32.checked_shl(attempt.saturating_sub(1)) else {
            return Err(ApiError::BackoffOverflow {
                attempt,
                base_delay: self.initial_backoff,
            });
        };
        Ok(self
            .initial_backoff
            .checked_mul(multiplier)
            .map_or(self.max_backoff, |delay| delay.min(self.max_backoff)))
    }

    fn messages_path(&self) -> &'static str {
        match self.service {
            ApiService::NanoGpt | ApiService::OpencodeGo => "/v1/messages",
            ApiService::Synthetic | ApiService::Grok => "/messages",
            ApiService::OpenAiCodex => "/responses",
            ApiService::Neuralwatt | ApiService::Lilac => "/v1/chat/completions",
        }
    }

    fn chat_completions_path(&self) -> &'static str {
        match self.service {
            ApiService::NanoGpt
            | ApiService::OpencodeGo
            | ApiService::Neuralwatt
            | ApiService::Lilac => "/v1/chat/completions",
            ApiService::Synthetic | ApiService::Grok => "/chat/completions",
            ApiService::OpenAiCodex => "/responses",
        }
    }

    fn normalize_message_request(&self, request: &MessageRequest) -> MessageRequest {
        let mut normalized = request.clone();
        normalized.model = self.normalize_model_id(&normalized.model);
        normalized
    }

    fn normalize_model_id(&self, model: &str) -> String {
        match self.service {
            ApiService::OpenAiCodex => normalize_openai_codex_model_id(model).to_string(),
            ApiService::OpencodeGo => normalize_opencode_go_model_id(model).to_string(),
            ApiService::Neuralwatt => model
                .strip_prefix("neuralwatt/")
                .unwrap_or(model)
                .to_string(),
            ApiService::Lilac => model.strip_prefix("lilac/").unwrap_or(model).to_string(),
            ApiService::Grok => model.strip_prefix("grok/").unwrap_or(model).to_string(),
            ApiService::NanoGpt | ApiService::Synthetic => model.to_string(),
        }
    }

    async fn send_opencode_go_chat_completion(
        &self,
        request: &MessageRequest,
    ) -> Result<MessageResponse, ApiError> {
        let mut chat_request = message_request_to_chat_completion_request(request, self)?;
        chat_request.stream = false;
        chat_request.stream_options = None;
        let response = self.send_chat_completion_with_retry(&chat_request).await?;
        let request_id = request_id_from_headers(response.headers());
        let response = response
            .json::<ChatCompletionResponse>()
            .await
            .map_err(ApiError::from)?;
        Ok(chat_completion_to_message_response(
            response,
            request.model.clone(),
            request_id,
        ))
    }

    async fn send_openai_codex_response(
        &self,
        request: &MessageRequest,
    ) -> Result<MessageResponse, ApiError> {
        let response_request = message_request_to_openai_codex_request(request, self)?;
        let response = self
            .send_openai_codex_response_with_retry(&response_request)
            .await?;
        let request_id = request_id_from_headers(response.headers());
        collect_openai_codex_message_response(response, request.model.clone(), request_id).await
    }

    async fn send_chat_completion_with_retry(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<reqwest::Response, ApiError> {
        let mut attempts = 0;
        let mut last_error: Option<ApiError>;

        loop {
            attempts += 1;
            match self.send_chat_completion_raw(request).await {
                Ok(response) => match expect_success(response).await {
                    Ok(response) => return Ok(response),
                    Err(ApiError::Api { status, .. })
                        if self.service == ApiService::OpenAiCodex
                            && status == reqwest::StatusCode::UNAUTHORIZED
                            && attempts == 1 =>
                    {
                        self.ensure_openai_codex_access_token(true).await?;
                        continue;
                    }
                    Err(error) if error.is_retryable() && attempts <= self.max_retries + 1 => {
                        last_error = Some(error);
                    }
                    Err(error) => return Err(error),
                },
                Err(error) if error.is_retryable() && attempts <= self.max_retries + 1 => {
                    last_error = Some(error);
                }
                Err(error) => return Err(error),
            }

            if attempts > self.max_retries {
                break;
            }

            tokio::time::sleep(self.backoff_for_attempt(attempts)?).await;
        }

        Err(ApiError::RetriesExhausted {
            attempts,
            last_error: Box::new(last_error.expect("retry loop must capture an error")),
        })
    }

    async fn send_openai_codex_response_raw(
        &self,
        request: &OpenAiCodexResponsesRequest,
    ) -> Result<reqwest::Response, ApiError> {
        self.ensure_openai_codex_access_token(false).await?;
        let request_url = format!(
            "{}{}",
            self.base_url.trim_end_matches('/'),
            self.chat_completions_path()
        );
        if nanogpt_client_debug_enabled() {
            let resolved_base_url = self.base_url.trim_end_matches('/');
            eprintln!("[nanogpt-client] resolved_base_url={resolved_base_url}");
            eprintln!("[nanogpt-client] request_url={request_url}");
        }
        let request_builder = self
            .http
            .post(&request_url)
            .header("content-type", "application/json");
        let request_builder = self.apply_auth_headers(request_builder, false);

        request_builder
            .json(request)
            .send()
            .await
            .map_err(ApiError::from)
    }

    async fn send_openai_codex_response_with_retry(
        &self,
        request: &OpenAiCodexResponsesRequest,
    ) -> Result<reqwest::Response, ApiError> {
        let mut attempts = 0;
        let mut last_error: Option<ApiError>;

        loop {
            attempts += 1;
            match self.send_openai_codex_response_raw(request).await {
                Ok(response) => match expect_success(response).await {
                    Ok(response) => return Ok(response),
                    Err(ApiError::Api { status, .. })
                        if status == reqwest::StatusCode::UNAUTHORIZED && attempts == 1 =>
                    {
                        self.ensure_openai_codex_access_token(true).await?;
                        continue;
                    }
                    Err(error) if error.is_retryable() && attempts <= self.max_retries + 1 => {
                        last_error = Some(error);
                    }
                    Err(error) => return Err(error),
                },
                Err(error) if error.is_retryable() && attempts <= self.max_retries + 1 => {
                    last_error = Some(error);
                }
                Err(error) => return Err(error),
            }

            if attempts > self.max_retries {
                break;
            }

            tokio::time::sleep(self.backoff_for_attempt(attempts)?).await;
        }

        Err(ApiError::RetriesExhausted {
            attempts,
            last_error: Box::new(last_error.expect("retry loop must capture an error")),
        })
    }
}

fn build_http_client() -> reqwest::Client {
    reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(DEFAULT_CONNECT_TIMEOUT_SECS))
        .timeout(Duration::from_secs(api_timeout_secs()))
        .build()
        .unwrap_or_else(|_| reqwest::Client::new())
}

fn api_timeout_secs() -> u64 {
    parse_api_timeout_secs(std::env::var(API_TIMEOUT_ENV_VAR).ok().as_deref())
}

fn parse_api_timeout_secs(value: Option<&str>) -> u64 {
    value
        .and_then(|raw| raw.trim().parse::<u64>().ok())
        .filter(|timeout_secs| *timeout_secs > 0)
        .unwrap_or(DEFAULT_API_TIMEOUT_SECS)
}

fn read_api_key() -> Result<String, ApiError> {
    resolve_api_key_for(ApiService::NanoGpt)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApiService {
    NanoGpt,
    Synthetic,
    OpenAiCodex,
    OpencodeGo,
    Neuralwatt,
    Lilac,
    Grok,
}

impl ApiService {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NanoGpt => "nanogpt",
            Self::Synthetic => "synthetic",
            Self::OpenAiCodex => "openai_codex",
            Self::OpencodeGo => "opencode_go",
            Self::Neuralwatt => "neuralwatt",
            Self::Lilac => "lilac",
            Self::Grok => "grok",
        }
    }

    #[must_use]
    pub const fn display_name(self) -> &'static str {
        match self {
            Self::NanoGpt => "NanoGPT",
            Self::Synthetic => "Synthetic",
            Self::OpenAiCodex => "OpenAI Codex",
            Self::OpencodeGo => "OpenCode Go",
            Self::Neuralwatt => "Neuralwatt",
            Self::Lilac => "Lilac",
            Self::Grok => "Grok",
        }
    }
}

pub fn resolve_api_key_for(service: ApiService) -> Result<String, ApiError> {
    if matches!(service, ApiService::OpenAiCodex | ApiService::Grok) {
        if service == ApiService::Grok {
            return Err(ApiError::MissingApiKey);
        }
        return resolve_openai_codex_credentials().map(|credentials| credentials.access_token);
    }
    match std::env::var(service_api_key_env(service)) {
        Ok(api_key) if !api_key.is_empty() => Ok(api_key),
        Ok(_) => Err(ApiError::MissingApiKey),
        Err(std::env::VarError::NotPresent) => {
            read_api_key_from_credentials_file(service).ok_or(ApiError::MissingApiKey)
        }
        Err(error) => Err(ApiError::from(error)),
    }
}

pub fn resolve_api_key() -> Result<String, ApiError> {
    read_api_key()
}

#[must_use]
pub fn resolve_base_url_for(service: ApiService) -> String {
    match service {
        ApiService::NanoGpt => {
            std::env::var("NANOGPT_BASE_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.to_string())
        }
        ApiService::Synthetic => std::env::var("SYNTHETIC_BASE_URL")
            .unwrap_or_else(|_| DEFAULT_SYNTHETIC_MESSAGES_BASE_URL.to_string()),
        ApiService::OpenAiCodex => std::env::var("OPENAI_CODEX_BASE_URL")
            .unwrap_or_else(|_| DEFAULT_OPENAI_CODEX_BASE_URL.to_string()),
        ApiService::OpencodeGo => std::env::var("OPENCODE_GO_BASE_URL")
            .unwrap_or_else(|_| DEFAULT_OPENCODE_GO_BASE_URL.to_string()),
        ApiService::Neuralwatt => std::env::var("NEURALWATT_BASE_URL")
            .unwrap_or_else(|_| DEFAULT_NEURALWATT_BASE_URL.to_string()),
        ApiService::Lilac => {
            std::env::var("LILAC_BASE_URL").unwrap_or_else(|_| DEFAULT_LILAC_BASE_URL.to_string())
        }
        ApiService::Grok => "grok-cli://local".to_string(),
    }
}

#[must_use]
pub fn resolve_root_url_for(service: ApiService) -> String {
    match service {
        ApiService::NanoGpt => {
            let base = resolve_base_url_for(service);
            let trimmed = base.trim_end_matches('/');
            trimmed.strip_suffix("/api").unwrap_or(trimmed).to_string()
        }
        ApiService::Synthetic => {
            if let Ok(root) = std::env::var("SYNTHETIC_ROOT_URL") {
                return root;
            }
            let base = resolve_base_url_for(service);
            let trimmed = base.trim_end_matches('/');
            trimmed
                .strip_suffix("/anthropic/v1")
                .unwrap_or(trimmed)
                .to_string()
        }
        ApiService::OpenAiCodex
        | ApiService::OpencodeGo
        | ApiService::Neuralwatt
        | ApiService::Lilac
        | ApiService::Grok => resolve_base_url_for(service),
    }
}

pub fn resolve_openai_codex_credentials() -> Result<OpenAiCodexCredentials, ApiError> {
    match std::env::var("OPENAI_CODEX_ACCESS_TOKEN") {
        Ok(access_token) if !access_token.trim().is_empty() => {
            let refresh_token = std::env::var("OPENAI_CODEX_REFRESH_TOKEN").unwrap_or_default();
            let account_id = std::env::var("OPENAI_CODEX_ACCOUNT_ID")
                .ok()
                .filter(|value| !value.trim().is_empty());
            let expires_at = match std::env::var("OPENAI_CODEX_EXPIRES_AT") {
                Ok(value) if !value.trim().is_empty() => {
                    Some(value.parse::<u64>().map_err(|error| {
                        ApiError::Io(std::io::Error::new(
                            std::io::ErrorKind::InvalidInput,
                            format!("invalid OPENAI_CODEX_EXPIRES_AT: {error}"),
                        ))
                    })?)
                }
                _ => None,
            };
            Ok(OpenAiCodexCredentials {
                access_token,
                refresh_token,
                expires_at,
                account_id,
            })
        }
        Ok(_) => Err(ApiError::MissingOpenAiCodexAuth),
        Err(std::env::VarError::NotPresent) => {
            load_openai_codex_credentials_from_credentials_file()
                .ok_or(ApiError::MissingOpenAiCodexAuth)
        }
        Err(error) => Err(ApiError::from(error)),
    }
}

pub fn save_openai_codex_credentials(
    credentials: &OpenAiCodexCredentials,
) -> Result<PathBuf, ApiError> {
    let path = credentials_path().ok_or_else(|| {
        ApiError::Io(std::io::Error::other(
            "could not resolve pebble config home",
        ))
    })?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let mut parsed = match fs::read_to_string(&path) {
        Ok(contents) => serde_json::from_str::<serde_json::Value>(&contents)
            .unwrap_or_else(|_| serde_json::json!({})),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => serde_json::json!({}),
        Err(error) => return Err(ApiError::Io(error)),
    };
    if !parsed.is_object() {
        parsed = serde_json::json!({});
    }
    parsed[OPENAI_CODEX_CREDENTIALS_KEY] = serde_json::to_value(credentials)?;
    write_atomic(&path, serde_json::to_string_pretty(&parsed)?)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600))?;
    }
    Ok(path)
}

fn read_api_key_from_credentials_file(service: ApiService) -> Option<String> {
    let path = credentials_path()?;
    let contents = fs::read_to_string(path).ok()?;
    let parsed = serde_json::from_str::<serde_json::Value>(&contents).ok()?;
    let service_key = match service {
        ApiService::NanoGpt => "nanogpt_api_key",
        ApiService::Synthetic => "synthetic_api_key",
        ApiService::OpenAiCodex | ApiService::Grok => return None,
        ApiService::OpencodeGo => "opencode_go_api_key",
        ApiService::Neuralwatt => "neuralwatt_api_key",
        ApiService::Lilac => "lilac_api_key",
    };
    parsed
        .get(service_key)
        .and_then(serde_json::Value::as_str)
        .or_else(|| {
            (service == ApiService::NanoGpt)
                .then(|| parsed.get("apiKey").and_then(serde_json::Value::as_str))
                .flatten()
        })
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn service_api_key_env(service: ApiService) -> &'static str {
    match service {
        ApiService::NanoGpt => "NANOGPT_API_KEY",
        ApiService::Synthetic => "SYNTHETIC_API_KEY",
        ApiService::OpenAiCodex => "OPENAI_CODEX_ACCESS_TOKEN",
        ApiService::OpencodeGo => "OPENCODE_GO_API_KEY",
        ApiService::Neuralwatt => "NEURALWATT_API_KEY",
        ApiService::Lilac => "LILAC_API_KEY",
        ApiService::Grok => "GROK_ACCESS_TOKEN",
    }
}

fn credentials_path() -> Option<PathBuf> {
    Some(pebble_config_home()?.join("credentials.json"))
}

fn load_openai_codex_credentials_from_credentials_file() -> Option<OpenAiCodexCredentials> {
    let path = credentials_path()?;
    let contents = fs::read_to_string(path).ok()?;
    let parsed = serde_json::from_str::<serde_json::Value>(&contents).ok()?;
    serde_json::from_value(parsed.get(OPENAI_CODEX_CREDENTIALS_KEY)?.clone()).ok()
}

fn request_id_from_headers(headers: &reqwest::header::HeaderMap) -> Option<String> {
    headers
        .get(REQUEST_ID_HEADER)
        .or_else(|| headers.get(ALT_REQUEST_ID_HEADER))
        .and_then(|value| value.to_str().ok())
        .map(ToOwned::to_owned)
}

fn current_epoch_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}

fn providers_url(base_url: &str, canonical_id: &str) -> Result<reqwest::Url, ApiError> {
    let mut url =
        reqwest::Url::parse(&format!("{}/", base_url.trim_end_matches('/'))).map_err(|error| {
            ApiError::Io(std::io::Error::new(std::io::ErrorKind::InvalidInput, error))
        })?;
    let mut segments = url.path_segments_mut().map_err(|()| {
        ApiError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "invalid base url",
        ))
    })?;
    segments.pop_if_empty();
    segments.push("models");
    segments.push(canonical_id);
    segments.push("providers");
    drop(segments);
    Ok(url)
}

#[derive(Debug)]
pub struct MessageStream {
    request_id: Option<String>,
    state: MessageStreamState,
    pending: VecDeque<StreamEvent>,
}

#[derive(Debug)]
enum MessageStreamState {
    Http {
        response: reqwest::Response,
        parser: SseParser,
        done: bool,
    },
    OpenAiHttp {
        response: reqwest::Response,
        parser: OpenAiSseParser,
        done: bool,
    },
    Buffered,
}

impl MessageStream {
    fn from_http_response(response: reqwest::Response) -> Self {
        Self {
            request_id: request_id_from_headers(response.headers()),
            state: MessageStreamState::Http {
                response,
                parser: SseParser::new(),
                done: false,
            },
            pending: VecDeque::new(),
        }
    }

    fn from_message_response(response: MessageResponse) -> Self {
        Self {
            request_id: response.request_id.clone(),
            state: MessageStreamState::Buffered,
            pending: VecDeque::from(message_response_to_stream_events(response)),
        }
    }

    fn from_openai_response(response: reqwest::Response) -> Self {
        Self {
            request_id: request_id_from_headers(response.headers()),
            state: MessageStreamState::OpenAiHttp {
                response,
                parser: OpenAiSseParser::default(),
                done: false,
            },
            pending: VecDeque::new(),
        }
    }

    #[must_use]
    pub fn request_id(&self) -> Option<&str> {
        self.request_id.as_deref()
    }

    pub async fn next_event(&mut self) -> Result<Option<StreamEvent>, ApiError> {
        loop {
            if let Some(event) = self.pending.pop_front() {
                return Ok(Some(event));
            }

            match &mut self.state {
                MessageStreamState::Buffered => return Ok(None),
                MessageStreamState::Http {
                    response,
                    parser,
                    done,
                } => {
                    if *done {
                        let remaining = parser.finish()?;
                        self.pending.extend(remaining);
                        if let Some(event) = self.pending.pop_front() {
                            return Ok(Some(event));
                        }
                        return Ok(None);
                    }

                    match response.chunk().await? {
                        Some(chunk) => {
                            self.pending.extend(parser.push(&chunk)?);
                        }
                        None => {
                            *done = true;
                        }
                    }
                }
                MessageStreamState::OpenAiHttp {
                    response,
                    parser,
                    done,
                } => {
                    if *done {
                        self.pending.extend(parser.finish()?);
                        if let Some(event) = self.pending.pop_front() {
                            return Ok(Some(event));
                        }
                        return Ok(None);
                    }
                    match response.chunk().await? {
                        Some(chunk) => self.pending.extend(parser.push(&chunk)?),
                        None => *done = true,
                    }
                }
            }
        }
    }
}

#[derive(Debug, Default)]
struct OpenAiSseParser {
    buffer: Vec<u8>,
    text_started: bool,
    thinking_started: bool,
    tools: BTreeMap<u32, OpenAiStreamTool>,
    usage: Usage,
    stopped: bool,
}

#[derive(Debug, Default)]
struct OpenAiStreamTool {
    id: String,
    name: String,
    pending_arguments: String,
}

impl OpenAiSseParser {
    fn push(&mut self, chunk: &[u8]) -> Result<Vec<StreamEvent>, ApiError> {
        self.buffer.extend_from_slice(chunk);
        let mut events = Vec::new();
        while let Some(frame) = take_sse_frame(&mut self.buffer) {
            events.extend(self.parse_frame(&frame)?);
        }
        Ok(events)
    }

    fn finish(&mut self) -> Result<Vec<StreamEvent>, ApiError> {
        let mut events = Vec::new();
        if !self.buffer.is_empty() {
            let trailing = String::from_utf8_lossy(&std::mem::take(&mut self.buffer)).into_owned();
            events.extend(self.parse_frame(&trailing)?);
        }
        if !self.stopped {
            events.extend(self.stop_events());
        }
        Ok(events)
    }

    fn parse_frame(&mut self, frame: &str) -> Result<Vec<StreamEvent>, ApiError> {
        let payload = frame
            .lines()
            .filter_map(|line| line.strip_prefix("data:").map(str::trim_start))
            .collect::<Vec<_>>()
            .join("\n");
        if payload.is_empty() {
            return Ok(Vec::new());
        }
        if payload == "[DONE]" {
            return Ok(self.stop_events());
        }
        let value: Value = serde_json::from_str(&payload)?;
        reject_openai_stream_error(&value, &payload)?;
        record_openai_usage(&value, &mut self.usage);
        let mut events = Vec::new();
        let Some(delta) = value
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| choices.first())
            .and_then(|choice| choice.get("delta"))
        else {
            return Ok(events);
        };
        if let Some(thinking) = delta
            .get("reasoning_content")
            .or_else(|| delta.get("reasoning"))
            .and_then(Value::as_str)
            .filter(|text| !text.is_empty())
        {
            if !self.thinking_started {
                self.thinking_started = true;
                events.push(StreamEvent::ContentBlockStart(
                    crate::types::ContentBlockStartEvent {
                        index: 1,
                        content_block: OutputContentBlock::Thinking {
                            thinking: String::new(),
                            signature: None,
                        },
                    },
                ));
            }
            events.push(StreamEvent::ContentBlockDelta(
                crate::types::ContentBlockDeltaEvent {
                    index: 1,
                    delta: crate::types::ContentBlockDelta::ThinkingDelta {
                        thinking: thinking.to_string(),
                    },
                },
            ));
        }
        if let Some(text) = delta
            .get("content")
            .and_then(Value::as_str)
            .filter(|text| !text.is_empty())
        {
            if !self.text_started {
                self.text_started = true;
                events.push(StreamEvent::ContentBlockStart(
                    crate::types::ContentBlockStartEvent {
                        index: 0,
                        content_block: OutputContentBlock::Text {
                            text: String::new(),
                        },
                    },
                ));
            }
            events.push(StreamEvent::ContentBlockDelta(
                crate::types::ContentBlockDeltaEvent {
                    index: 0,
                    delta: crate::types::ContentBlockDelta::TextDelta {
                        text: text.to_string(),
                    },
                },
            ));
        }
        if let Some(calls) = delta.get("tool_calls").and_then(Value::as_array) {
            for call in calls {
                let ordinal = call
                    .get("index")
                    .and_then(Value::as_u64)
                    .unwrap_or(0)
                    .try_into()
                    .unwrap_or(u32::MAX);
                let tool = self.tools.entry(ordinal).or_default();
                if let Some(id) = call.get("id").and_then(Value::as_str) {
                    tool.id.push_str(id);
                }
                if let Some(function) = call.get("function") {
                    if let Some(name) = function.get("name").and_then(Value::as_str) {
                        tool.name.push_str(name);
                    }
                    if let Some(arguments) = function.get("arguments").and_then(Value::as_str) {
                        tool.pending_arguments.push_str(arguments);
                    }
                }
            }
        }
        Ok(events)
    }

    fn stop_events(&mut self) -> Vec<StreamEvent> {
        if self.stopped {
            return Vec::new();
        }
        self.stopped = true;
        let mut events = Vec::new();
        for (ordinal, tool) in &self.tools {
            if !tool.id.is_empty() && !tool.name.is_empty() {
                events.push(StreamEvent::ContentBlockStart(
                    crate::types::ContentBlockStartEvent {
                        index: ordinal + 10,
                        content_block: OutputContentBlock::ToolUse {
                            id: tool.id.clone(),
                            name: tool.name.clone(),
                            input: Value::Object(serde_json::Map::new()),
                        },
                    },
                ));
                if !tool.pending_arguments.is_empty() {
                    events.push(StreamEvent::ContentBlockDelta(
                        crate::types::ContentBlockDeltaEvent {
                            index: ordinal + 10,
                            delta: crate::types::ContentBlockDelta::InputJsonDelta {
                                partial_json: tool.pending_arguments.clone(),
                            },
                        },
                    ));
                }
                events.push(StreamEvent::ContentBlockStop(
                    crate::types::ContentBlockStopEvent {
                        index: ordinal + 10,
                    },
                ));
            }
        }
        if self.thinking_started {
            events.push(StreamEvent::ContentBlockStop(
                crate::types::ContentBlockStopEvent { index: 1 },
            ));
        }
        if self.text_started {
            events.push(StreamEvent::ContentBlockStop(
                crate::types::ContentBlockStopEvent { index: 0 },
            ));
        }
        events.push(StreamEvent::MessageDelta(crate::types::MessageDeltaEvent {
            delta: crate::types::MessageDelta {
                stop_reason: Some("end_turn".to_string()),
                stop_sequence: None,
            },
            usage: self.usage.clone(),
        }));
        events.push(StreamEvent::MessageStop(crate::types::MessageStopEvent {}));
        events
    }
}

fn reject_openai_stream_error(value: &Value, payload: &str) -> Result<(), ApiError> {
    let Some(error) = value.get("error") else {
        return Ok(());
    };
    Err(ApiError::StreamApi {
        error_type: error
            .get("type")
            .and_then(Value::as_str)
            .map(str::to_string),
        message: error
            .get("message")
            .and_then(Value::as_str)
            .map(str::to_string),
        body: payload.to_string(),
    })
}

fn record_openai_usage(value: &Value, usage: &mut Usage) {
    let Some(frame_usage) = value.get("usage") else {
        return;
    };
    usage.input_tokens = frame_usage
        .get("prompt_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0)
        .try_into()
        .unwrap_or(u32::MAX);
    usage.output_tokens = frame_usage
        .get("completion_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0)
        .try_into()
        .unwrap_or(u32::MAX);
}

fn take_sse_frame(buffer: &mut Vec<u8>) -> Option<String> {
    let separator = buffer
        .windows(2)
        .position(|window| window == b"\n\n")
        .map(|position| (position, 2))
        .or_else(|| {
            buffer
                .windows(4)
                .position(|window| window == b"\r\n\r\n")
                .map(|position| (position, 4))
        })?;
    let (position, length) = separator;
    let frame = buffer.drain(..position + length).collect::<Vec<_>>();
    Some(String::from_utf8_lossy(&frame[..position]).into_owned())
}

fn normalize_opencode_go_model_id(model: &str) -> &str {
    model.strip_prefix("opencode-go/").unwrap_or(model)
}

fn normalize_openai_codex_model_id(model: &str) -> &str {
    model.strip_prefix("openai-codex/").unwrap_or(model)
}

fn opencode_go_uses_messages_api(model: &str) -> bool {
    matches!(
        normalize_opencode_go_model_id(model),
        "minimax-m2.5" | "minimax-m2.7"
    )
}

fn opencode_go_prefers_thinking_disabled(model: &str) -> bool {
    matches!(
        normalize_opencode_go_model_id(model),
        "kimi-k2.5" | "kimi-k2.6"
    )
}

fn invalid_request_error(message: impl Into<String>) -> ApiError {
    ApiError::Io(std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        message.into(),
    ))
}

#[derive(Debug, Clone, Serialize, PartialEq)]
struct OpenAiCodexResponsesRequest {
    model: String,
    instructions: String,
    input: Vec<OpenAiCodexInputItem>,
    tools: Vec<OpenAiCodexTool>,
    tool_choice: String,
    parallel_tool_calls: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<OpenAiCodexReasoning>,
    store: bool,
    stream: bool,
    include: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    service_tier: Option<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
struct OpenAiCodexReasoning {
    effort: ReasoningEffort,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
enum OpenAiCodexInputItem {
    Message {
        role: String,
        content: Vec<OpenAiCodexContentItem>,
    },
    FunctionCall {
        name: String,
        arguments: String,
        call_id: String,
    },
    FunctionCallOutput {
        call_id: String,
        output: String,
    },
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
enum OpenAiCodexContentItem {
    InputText { text: String },
    InputImage { image_url: String },
    OutputText { text: String },
}

#[derive(Debug, Clone, Serialize, PartialEq)]
struct OpenAiCodexTool {
    #[serde(rename = "type")]
    kind: String,
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    parameters: Value,
}

#[derive(Debug)]
struct OpenAiCodexResponseAccumulator {
    response_id: Option<String>,
    usage: Usage,
    content: Vec<OutputContentBlock>,
}

impl Default for OpenAiCodexResponseAccumulator {
    fn default() -> Self {
        Self {
            response_id: None,
            usage: Usage {
                input_tokens: 0,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
                output_tokens: 0,
            },
            content: Vec::new(),
        }
    }
}

#[derive(Debug, Deserialize)]
struct OpenAiCodexStreamEventEnvelope {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    item: Option<Value>,
    #[serde(default)]
    response: Option<Value>,
}

fn message_request_to_openai_codex_request(
    request: &MessageRequest,
    client: &NanoGptClient,
) -> Result<OpenAiCodexResponsesRequest, ApiError> {
    let mut input = Vec::new();
    for message in &request.messages {
        input.extend(input_message_to_openai_codex_items(message)?);
    }

    let tools = request
        .tools
        .as_ref()
        .map(|tools| {
            tools
                .iter()
                .map(|tool| OpenAiCodexTool {
                    kind: "function".to_string(),
                    name: tool.name.clone(),
                    description: tool.description.clone(),
                    parameters: normalize_openai_codex_json_schema(&tool.input_schema),
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let reasoning = request
        .reasoning_effort
        .map(|effort| OpenAiCodexReasoning { effort });
    let include = if reasoning.is_some() {
        vec!["reasoning.encrypted_content".to_string()]
    } else {
        Vec::new()
    };

    Ok(OpenAiCodexResponsesRequest {
        model: client.normalize_model_id(&request.model),
        instructions: openai_codex_instructions(request),
        input,
        parallel_tool_calls: tools.len() > 1,
        tools,
        tool_choice: openai_codex_tool_choice(request.tool_choice.as_ref()),
        reasoning,
        store: false,
        stream: true,
        include,
        service_tier: request.fast_mode.then_some("priority".to_string()),
    })
}

fn openai_codex_instructions(request: &MessageRequest) -> String {
    request
        .system
        .as_ref()
        .map(|system| system.trim())
        .filter(|system| !system.is_empty())
        .map_or_else(
            || "You are Pebble, a coding assistant.".to_string(),
            ToOwned::to_owned,
        )
}

fn openai_codex_tool_choice(choice: Option<&crate::types::ToolChoice>) -> String {
    match choice {
        Some(crate::types::ToolChoice::Any | crate::types::ToolChoice::Tool { .. }) => {
            "required".to_string()
        }
        Some(crate::types::ToolChoice::Auto) | None => "auto".to_string(),
    }
}

fn normalize_openai_codex_json_schema(schema: &Value) -> Value {
    match schema {
        Value::Object(map) => {
            let mut normalized = serde_json::Map::new();
            for (key, value) in map {
                let normalized_value = match key.as_str() {
                    "properties" => Value::Object(
                        value
                            .as_object()
                            .map(|properties| {
                                properties
                                    .iter()
                                    .map(|(name, value)| {
                                        (name.clone(), normalize_openai_codex_json_schema(value))
                                    })
                                    .collect()
                            })
                            .unwrap_or_default(),
                    ),
                    "items" | "additionalProperties" | "not" => {
                        normalize_openai_codex_json_schema(value)
                    }
                    "allOf" | "anyOf" | "oneOf" => Value::Array(
                        value
                            .as_array()
                            .map(|items| {
                                items
                                    .iter()
                                    .map(normalize_openai_codex_json_schema)
                                    .collect()
                            })
                            .unwrap_or_default(),
                    ),
                    _ => value.clone(),
                };
                normalized.insert(key.clone(), normalized_value);
            }

            if normalized.get("type").and_then(Value::as_str) == Some("object")
                && !normalized.contains_key("properties")
            {
                normalized.insert(
                    "properties".to_string(),
                    Value::Object(serde_json::Map::new()),
                );
            }

            Value::Object(normalized)
        }
        Value::Array(items) => Value::Array(
            items
                .iter()
                .map(normalize_openai_codex_json_schema)
                .collect(),
        ),
        _ => schema.clone(),
    }
}

fn input_message_to_openai_codex_items(
    message: &crate::types::InputMessage,
) -> Result<Vec<OpenAiCodexInputItem>, ApiError> {
    match message.role.as_str() {
        "assistant" => assistant_input_to_openai_codex_items(message),
        "user" => user_input_to_openai_codex_items(message),
        other => Err(invalid_request_error(format!(
            "unsupported role for OpenAI Codex translation: {other}"
        ))),
    }
}

fn assistant_input_to_openai_codex_items(
    message: &crate::types::InputMessage,
) -> Result<Vec<OpenAiCodexInputItem>, ApiError> {
    let mut items = Vec::new();
    let mut pending_content = Vec::new();

    for block in &message.content {
        match block {
            crate::types::InputContentBlock::Text { text } => {
                if !text.is_empty() {
                    pending_content.push(OpenAiCodexContentItem::OutputText { text: text.clone() });
                }
            }
            crate::types::InputContentBlock::Image { .. } => {
                return Err(invalid_request_error(
                    "assistant image blocks are not supported for OpenAI Codex",
                ));
            }
            crate::types::InputContentBlock::ToolUse { id, name, input } => {
                if !pending_content.is_empty() {
                    items.push(OpenAiCodexInputItem::Message {
                        role: "assistant".to_string(),
                        content: std::mem::take(&mut pending_content),
                    });
                }
                items.push(OpenAiCodexInputItem::FunctionCall {
                    name: name.clone(),
                    arguments: input.to_string(),
                    call_id: id.clone(),
                });
            }
            crate::types::InputContentBlock::ToolResult { .. } => {
                return Err(invalid_request_error(
                    "assistant tool_result blocks cannot be translated to OpenAI Codex",
                ));
            }
        }
    }

    if !pending_content.is_empty() {
        items.push(OpenAiCodexInputItem::Message {
            role: "assistant".to_string(),
            content: pending_content,
        });
    }

    Ok(items)
}

fn user_input_to_openai_codex_items(
    message: &crate::types::InputMessage,
) -> Result<Vec<OpenAiCodexInputItem>, ApiError> {
    let mut items = Vec::new();
    let mut pending_content = Vec::new();

    for block in &message.content {
        match block {
            crate::types::InputContentBlock::Text { text } => {
                if !text.is_empty() {
                    pending_content.push(OpenAiCodexContentItem::InputText { text: text.clone() });
                }
            }
            crate::types::InputContentBlock::Image { source } => {
                pending_content.push(OpenAiCodexContentItem::InputImage {
                    image_url: format!("data:{};base64,{}", source.media_type, source.data),
                });
            }
            crate::types::InputContentBlock::ToolResult {
                tool_use_id,
                content,
                ..
            } => {
                if !pending_content.is_empty() {
                    items.push(OpenAiCodexInputItem::Message {
                        role: "user".to_string(),
                        content: std::mem::take(&mut pending_content),
                    });
                }
                items.push(OpenAiCodexInputItem::FunctionCallOutput {
                    call_id: tool_use_id.clone(),
                    output: tool_result_content_to_string(content)?,
                });
            }
            crate::types::InputContentBlock::ToolUse { .. } => {
                return Err(invalid_request_error(
                    "user tool_use blocks cannot be translated to OpenAI Codex",
                ));
            }
        }
    }

    if !pending_content.is_empty() {
        items.push(OpenAiCodexInputItem::Message {
            role: "user".to_string(),
            content: pending_content,
        });
    }

    Ok(items)
}

async fn collect_openai_codex_message_response(
    mut response: reqwest::Response,
    requested_model: String,
    request_id: Option<String>,
) -> Result<MessageResponse, ApiError> {
    let mut accumulator = OpenAiCodexResponseAccumulator::default();
    let mut buffer = Vec::new();

    while let Some(chunk) = response.chunk().await.map_err(ApiError::from)? {
        buffer.extend_from_slice(&chunk);
        while let Some(frame) = next_sse_frame(&mut buffer) {
            process_openai_codex_sse_frame(&frame, &mut accumulator)?;
        }
    }

    if !buffer.is_empty() {
        let trailing = String::from_utf8_lossy(&buffer).into_owned();
        process_openai_codex_sse_frame(&trailing, &mut accumulator)?;
    }

    Ok(MessageResponse {
        id: accumulator
            .response_id
            .or_else(|| request_id.clone())
            .unwrap_or_else(|| "openai-codex-response".to_string()),
        kind: "message".to_string(),
        role: "assistant".to_string(),
        content: accumulator.content,
        model: requested_model,
        stop_reason: None,
        stop_sequence: None,
        usage: accumulator.usage,
        request_id,
    })
}

fn next_sse_frame(buffer: &mut Vec<u8>) -> Option<String> {
    let separator = buffer
        .windows(2)
        .position(|window| window == b"\n\n")
        .map(|position| (position, 2))
        .or_else(|| {
            buffer
                .windows(4)
                .position(|window| window == b"\r\n\r\n")
                .map(|position| (position, 4))
        })?;

    let (position, separator_len) = separator;
    let frame = buffer.drain(..position + separator_len).collect::<Vec<_>>();
    let frame_len = frame.len().saturating_sub(separator_len);
    Some(String::from_utf8_lossy(&frame[..frame_len]).into_owned())
}

fn process_openai_codex_sse_frame(
    frame: &str,
    accumulator: &mut OpenAiCodexResponseAccumulator,
) -> Result<(), ApiError> {
    let trimmed = frame.trim();
    if trimmed.is_empty() {
        return Ok(());
    }

    let mut data_lines = Vec::new();
    for line in trimmed.lines() {
        if line.starts_with(':') {
            continue;
        }
        if let Some(data) = line.strip_prefix("data:") {
            data_lines.push(data.trim_start());
        }
    }

    if data_lines.is_empty() {
        return Ok(());
    }

    let payload = data_lines.join("\n");
    if payload == "[DONE]" {
        return Ok(());
    }

    let event = serde_json::from_str::<OpenAiCodexStreamEventEnvelope>(&payload)?;
    match event.kind.as_str() {
        "response.output_item.done" => {
            if let Some(item) = event.item {
                accumulator
                    .content
                    .extend(openai_codex_output_item_to_content_blocks(&item));
            }
        }
        "response.completed" => {
            if let Some(response) = event.response {
                accumulator.response_id = json_string_field(&response, "id")
                    .map(ToOwned::to_owned)
                    .or_else(|| accumulator.response_id.take());
                accumulator.usage = openai_codex_usage_from_response(&response);
            }
        }
        "response.failed" => {
            return Err(openai_codex_stream_error(&payload, event.response.as_ref()))
        }
        "response.incomplete" => {
            return Err(ApiError::StreamApi {
                error_type: Some("response_incomplete".to_string()),
                message: json_string_field(
                    event.response.as_ref().unwrap_or(&Value::Null),
                    "status",
                )
                .map(ToOwned::to_owned)
                .or_else(|| Some("OpenAI Codex response completed incompletely".to_string())),
                body: payload,
            });
        }
        _ => {}
    }

    Ok(())
}

fn openai_codex_output_item_to_content_blocks(item: &Value) -> Vec<OutputContentBlock> {
    let Some(kind) = json_string_field(item, "type") else {
        return Vec::new();
    };

    match kind {
        "message" => {
            if json_string_field(item, "role") != Some("assistant") {
                return Vec::new();
            }
            item.get("content")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(|content| match json_string_field(content, "type") {
                    Some("output_text" | "text") => {
                        json_string_field(content, "text").map(|text| OutputContentBlock::Text {
                            text: text.to_string(),
                        })
                    }
                    _ => None,
                })
                .collect()
        }
        "reasoning" => {
            let reasoning = item
                .get("content")
                .and_then(Value::as_array)
                .map(|entries| {
                    entries
                        .iter()
                        .filter_map(|entry| json_string_field(entry, "text"))
                        .collect::<Vec<_>>()
                        .join("\n")
                })
                .filter(|text| !text.is_empty())
                .or_else(|| {
                    item.get("summary")
                        .and_then(Value::as_array)
                        .map(|entries| {
                            entries
                                .iter()
                                .filter_map(|entry| json_string_field(entry, "text"))
                                .collect::<Vec<_>>()
                                .join("\n")
                        })
                })
                .filter(|text| !text.is_empty());

            reasoning
                .map(|thinking| {
                    vec![OutputContentBlock::Thinking {
                        thinking,
                        signature: None,
                    }]
                })
                .unwrap_or_default()
        }
        "function_call" => json_string_field(item, "call_id")
            .zip(json_string_field(item, "name"))
            .map(|(call_id, name)| {
                vec![OutputContentBlock::ToolUse {
                    id: call_id.to_string(),
                    name: name.to_string(),
                    input: parse_json_string_or_string(
                        json_string_field(item, "arguments").unwrap_or_default(),
                    ),
                }]
            })
            .unwrap_or_default(),
        "custom_tool_call" => json_string_field(item, "call_id")
            .zip(json_string_field(item, "name"))
            .map(|(call_id, name)| {
                vec![OutputContentBlock::ToolUse {
                    id: call_id.to_string(),
                    name: name.to_string(),
                    input: parse_json_string_or_string(
                        json_string_field(item, "input").unwrap_or_default(),
                    ),
                }]
            })
            .unwrap_or_default(),
        _ => Vec::new(),
    }
}

fn openai_codex_usage_from_response(response: &Value) -> Usage {
    let usage = response.get("usage").unwrap_or(&Value::Null);
    let input_tokens = json_u64_field(usage, "input_tokens")
        .and_then(|value| u32::try_from(value).ok())
        .unwrap_or_default();
    let output_tokens = json_u64_field(usage, "output_tokens")
        .and_then(|value| u32::try_from(value).ok())
        .unwrap_or_default();
    let cache_read_input_tokens = usage
        .get("input_tokens_details")
        .and_then(|details| json_u64_field(details, "cached_tokens"))
        .and_then(|value| u32::try_from(value).ok())
        .unwrap_or_default();

    Usage {
        input_tokens,
        cache_creation_input_tokens: 0,
        cache_read_input_tokens,
        output_tokens,
    }
}

fn openai_codex_stream_error(payload: &str, response: Option<&Value>) -> ApiError {
    let error = response
        .and_then(|response| response.get("error"))
        .unwrap_or(&Value::Null);
    ApiError::StreamApi {
        error_type: json_string_field(error, "code")
            .or_else(|| json_string_field(error, "type"))
            .map(ToOwned::to_owned),
        message: json_string_field(error, "message").map(ToOwned::to_owned),
        body: payload.to_string(),
    }
}

fn parse_json_string_or_string(value: &str) -> Value {
    serde_json::from_str::<Value>(value).unwrap_or_else(|_| Value::String(value.to_string()))
}

fn json_string_field<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(Value::as_str)
}

fn json_u64_field(value: &Value, key: &str) -> Option<u64> {
    value.get(key).and_then(Value::as_u64)
}

fn message_request_to_chat_completion_request(
    request: &MessageRequest,
    client: &NanoGptClient,
) -> Result<ChatCompletionRequest, ApiError> {
    let mut messages = Vec::new();
    if let Some(system) = request.system.as_ref().filter(|system| !system.is_empty()) {
        messages.push(ChatCompletionMessage {
            role: "system".to_string(),
            content: Some(ChatCompletionContent::Text(system.clone())),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
            reasoning: None,
        });
    }

    for message in &request.messages {
        messages.extend(input_message_to_chat_completion_messages(message)?);
    }

    Ok(ChatCompletionRequest {
        model: client.normalize_model_id(&request.model),
        messages,
        max_tokens: Some(request.max_tokens),
        tools: request.tools.as_ref().map(|tools| {
            tools
                .iter()
                .map(|tool| ChatCompletionTool {
                    kind: "function".to_string(),
                    function: ChatCompletionFunction {
                        name: tool.name.clone(),
                        description: tool.description.clone(),
                        parameters: Some(tool.input_schema.clone()),
                    },
                })
                .collect()
        }),
        tool_choice: request.tool_choice.as_ref().map(map_tool_choice),
        billing_mode: None,
        thinking: (client.service == ApiService::OpencodeGo
            && opencode_go_prefers_thinking_disabled(&request.model))
        .then(ChatCompletionThinkingConfig::disabled),
        reasoning_effort: request.reasoning_effort,
        stream_options: request
            .stream
            .then_some(crate::types::ChatCompletionStreamOptions {
                include_usage: true,
            }),
        stream: request.stream,
    })
}

fn input_message_to_chat_completion_messages(
    message: &crate::types::InputMessage,
) -> Result<Vec<ChatCompletionMessage>, ApiError> {
    match message.role.as_str() {
        "assistant" => assistant_input_to_chat_completion_messages(message),
        "user" => user_input_to_chat_completion_messages(message),
        other => Err(invalid_request_error(format!(
            "unsupported role for chat/completions translation: {other}"
        ))),
    }
}

fn assistant_input_to_chat_completion_messages(
    message: &crate::types::InputMessage,
) -> Result<Vec<ChatCompletionMessage>, ApiError> {
    let mut text_parts = Vec::new();
    let mut tool_calls = Vec::new();

    for block in &message.content {
        match block {
            crate::types::InputContentBlock::Text { text } => {
                if !text.is_empty() {
                    text_parts.push(text.clone());
                }
            }
            crate::types::InputContentBlock::ToolUse { id, name, input } => {
                tool_calls.push(crate::types::ChatCompletionToolCall {
                    id: id.clone(),
                    kind: "function".to_string(),
                    function: crate::types::ChatCompletionFunctionCall {
                        name: name.clone(),
                        arguments: input.to_string(),
                    },
                });
            }
            crate::types::InputContentBlock::Image { .. } => {
                return Err(invalid_request_error(
                    "image inputs are not supported for OpenCode Go chat/completions models",
                ));
            }
            crate::types::InputContentBlock::ToolResult { .. } => {
                return Err(invalid_request_error(
                    "assistant tool_result blocks cannot be translated to chat/completions",
                ));
            }
        }
    }

    Ok(vec![ChatCompletionMessage {
        role: "assistant".to_string(),
        content: (!text_parts.is_empty())
            .then(|| ChatCompletionContent::Text(text_parts.join("\n\n"))),
        tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
        tool_call_id: None,
        reasoning_content: message.reasoning_content.clone(),
        reasoning: message
            .reasoning
            .clone()
            .or(message.reasoning_content.clone()),
    }])
}

fn user_input_to_chat_completion_messages(
    message: &crate::types::InputMessage,
) -> Result<Vec<ChatCompletionMessage>, ApiError> {
    let mut messages = Vec::new();
    let mut pending_parts = Vec::new();
    let mut has_image = false;

    for block in &message.content {
        match block {
            crate::types::InputContentBlock::Text { text } => {
                if !text.is_empty() {
                    pending_parts.push(crate::types::ChatCompletionContentPart {
                        kind: "text".to_string(),
                        text: Some(text.clone()),
                        image_url: None,
                    });
                }
            }
            crate::types::InputContentBlock::ToolResult {
                tool_use_id,
                content,
                ..
            } => {
                if !pending_parts.is_empty() {
                    messages.push(ChatCompletionMessage {
                        role: "user".to_string(),
                        content: Some(chat_content_from_parts(
                            std::mem::take(&mut pending_parts),
                            has_image,
                        )),
                        tool_calls: None,
                        tool_call_id: None,
                        reasoning_content: None,
                        reasoning: None,
                    });
                    has_image = false;
                }
                messages.push(ChatCompletionMessage {
                    role: "tool".to_string(),
                    content: Some(ChatCompletionContent::Text(tool_result_content_to_string(
                        content,
                    )?)),
                    tool_calls: None,
                    tool_call_id: Some(tool_use_id.clone()),
                    reasoning_content: None,
                    reasoning: None,
                });
            }
            crate::types::InputContentBlock::Image { source } => {
                has_image = true;
                pending_parts.push(crate::types::ChatCompletionContentPart {
                    kind: "image_url".to_string(),
                    text: None,
                    image_url: Some(crate::types::ChatCompletionImageUrl {
                        url: format!("data:{};base64,{}", source.media_type, source.data),
                    }),
                });
            }
            crate::types::InputContentBlock::ToolUse { .. } => {
                return Err(invalid_request_error(
                    "user tool_use blocks cannot be translated to chat/completions",
                ));
            }
        }
    }

    if !pending_parts.is_empty() {
        messages.push(ChatCompletionMessage {
            role: "user".to_string(),
            content: Some(chat_content_from_parts(pending_parts, has_image)),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
            reasoning: None,
        });
    }

    Ok(messages)
}

fn chat_content_from_parts(
    parts: Vec<crate::types::ChatCompletionContentPart>,
    has_image: bool,
) -> ChatCompletionContent {
    if has_image {
        ChatCompletionContent::Parts(parts)
    } else {
        ChatCompletionContent::Text(
            parts
                .into_iter()
                .filter_map(|part| part.text)
                .collect::<Vec<_>>()
                .join("\n\n"),
        )
    }
}

fn tool_result_content_to_string(
    content: &[crate::types::ToolResultContentBlock],
) -> Result<String, ApiError> {
    let mut parts = Vec::new();
    for block in content {
        match block {
            crate::types::ToolResultContentBlock::Text { text } => parts.push(text.clone()),
            crate::types::ToolResultContentBlock::Json { value } => parts.push(value.to_string()),
        }
    }
    if parts.is_empty() {
        return Err(invalid_request_error(
            "tool result content cannot be empty for chat/completions translation",
        ));
    }
    Ok(parts.join("\n"))
}

fn map_tool_choice(choice: &crate::types::ToolChoice) -> crate::types::ChatCompletionToolChoice {
    match choice {
        crate::types::ToolChoice::Auto => {
            crate::types::ChatCompletionToolChoice::Mode("auto".to_string())
        }
        crate::types::ToolChoice::Any => {
            crate::types::ChatCompletionToolChoice::Mode("required".to_string())
        }
        crate::types::ToolChoice::Tool { name } => {
            crate::types::ChatCompletionToolChoice::Function {
                kind: "function".to_string(),
                function: crate::types::ChatCompletionNamedFunction { name: name.clone() },
            }
        }
    }
}

fn chat_completion_to_message_response(
    response: ChatCompletionResponse,
    requested_model: String,
    request_id: Option<String>,
) -> MessageResponse {
    let choice =
        response
            .choices
            .into_iter()
            .next()
            .unwrap_or(crate::types::ChatCompletionChoice {
                index: 0,
                message: crate::types::ChatCompletionAssistantMessage {
                    role: "assistant".to_string(),
                    content: None,
                    tool_calls: None,
                    reasoning_content: None,
                    reasoning: None,
                },
                finish_reason: None,
            });

    let mut content = Vec::new();
    if let Some(reasoning) = choice
        .message
        .reasoning
        .clone()
        .or(choice.message.reasoning_content.clone())
        .filter(|text| !text.is_empty())
    {
        content.push(OutputContentBlock::Thinking {
            thinking: reasoning,
            signature: None,
        });
    }
    if let Some(text) = choice
        .message
        .content
        .as_ref()
        .and_then(chat_completion_content_to_text)
        .filter(|text| !text.is_empty())
    {
        content.push(OutputContentBlock::Text { text });
    }
    if let Some(tool_calls) = choice.message.tool_calls {
        for tool_call in tool_calls {
            let input = serde_json::from_str(&tool_call.function.arguments)
                .unwrap_or(serde_json::Value::String(tool_call.function.arguments));
            content.push(OutputContentBlock::ToolUse {
                id: tool_call.id,
                name: tool_call.function.name,
                input,
            });
        }
    }

    let usage = response.usage.unwrap_or(crate::types::ChatCompletionUsage {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0,
    });

    MessageResponse {
        id: response.id,
        kind: "message".to_string(),
        role: choice.message.role,
        content,
        model: requested_model,
        stop_reason: choice.finish_reason,
        stop_sequence: None,
        usage: crate::types::Usage {
            input_tokens: usage.prompt_tokens,
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: 0,
            output_tokens: usage.completion_tokens,
        },
        request_id,
    }
}

fn chat_completion_content_to_text(content: &ChatCompletionContent) -> Option<String> {
    match content {
        ChatCompletionContent::Text(text) => Some(text.clone()),
        ChatCompletionContent::Parts(parts) => {
            let text = parts
                .iter()
                .filter(|part| part.kind == "text" || part.kind == "output_text")
                .filter_map(|part| part.text.as_deref())
                .collect::<Vec<_>>()
                .join("\n");
            (!text.is_empty()).then_some(text)
        }
    }
}

fn message_response_to_stream_events(response: MessageResponse) -> Vec<StreamEvent> {
    let usage = response.usage.clone();
    let delta = crate::types::MessageDelta {
        stop_reason: response.stop_reason.clone(),
        stop_sequence: response.stop_sequence.clone(),
    };
    let content_blocks = response.content.clone();
    let message = MessageResponse {
        content: Vec::new(),
        ..response
    };
    let mut events = vec![StreamEvent::MessageStart(crate::types::MessageStartEvent {
        message,
    })];

    for (index, block) in content_blocks.into_iter().enumerate() {
        let index = u32::try_from(index).unwrap_or(u32::MAX);
        events.push(StreamEvent::ContentBlockStart(
            crate::types::ContentBlockStartEvent {
                index,
                content_block: block,
            },
        ));
        events.push(StreamEvent::ContentBlockStop(
            crate::types::ContentBlockStopEvent { index },
        ));
    }

    events.push(StreamEvent::MessageDelta(crate::types::MessageDeltaEvent {
        delta,
        usage,
    }));
    events.push(StreamEvent::MessageStop(crate::types::MessageStopEvent {}));
    events
}

async fn expect_success(response: reqwest::Response) -> Result<reqwest::Response, ApiError> {
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }

    let body = response.text().await.unwrap_or_else(|_| String::new());
    let parsed_error = serde_json::from_str::<NanoGptErrorEnvelope>(&body).ok();
    let retryable = is_retryable_status(status);

    Err(ApiError::Api {
        status,
        error_type: parsed_error
            .as_ref()
            .map(|error| error.error.error_type.clone()),
        message: parsed_error
            .as_ref()
            .map(|error| error.error.message.clone()),
        body,
        retryable,
    })
}

const fn is_retryable_status(status: reqwest::StatusCode) -> bool {
    matches!(status.as_u16(), 408 | 409 | 429 | 500 | 502 | 503 | 504)
}

#[derive(Debug, Deserialize)]
struct NanoGptErrorEnvelope {
    error: NanoGptErrorBody,
}

#[derive(Debug, Deserialize)]
struct NanoGptErrorBody {
    #[serde(rename = "type")]
    error_type: String,
    message: String,
}

#[derive(Debug, Deserialize)]
struct OpenAiCodexRefreshResponse {
    #[serde(default)]
    access_token: Option<String>,
    #[serde(default)]
    refresh_token: Option<String>,
    #[serde(default)]
    expires_in: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::{
        parse_api_timeout_secs, ALT_REQUEST_ID_HEADER, DEFAULT_API_TIMEOUT_SECS, REQUEST_ID_HEADER,
    };
    use std::sync::{Mutex, OnceLock};
    use std::time::Duration;

    use crate::types::{ContentBlockDelta, MessageRequest};

    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
            .lock()
            .expect("env test lock should not be poisoned")
    }

    fn temp_config_home() -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "pebble-api-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time should be after epoch")
                .as_nanos()
        ))
    }

    #[test]
    fn api_timeout_defaults_and_rejects_invalid_values() {
        assert_eq!(parse_api_timeout_secs(None), DEFAULT_API_TIMEOUT_SECS);
        assert_eq!(parse_api_timeout_secs(Some("45")), 45);
        assert_eq!(parse_api_timeout_secs(Some("0")), DEFAULT_API_TIMEOUT_SECS);
        assert_eq!(
            parse_api_timeout_secs(Some("not-a-number")),
            DEFAULT_API_TIMEOUT_SECS
        );
    }

    #[test]
    fn read_api_key_requires_presence() {
        let _guard = env_lock();
        let root = temp_config_home();
        std::fs::create_dir_all(&root).expect("config dir should exist");
        std::env::remove_var("NANOGPT_API_KEY");
        std::env::set_var("PEBBLE_CONFIG_HOME", &root);
        let error = super::read_api_key().expect_err("missing key should error");
        assert!(matches!(error, crate::error::ApiError::MissingApiKey));
        std::env::remove_var("PEBBLE_CONFIG_HOME");
        std::fs::remove_dir_all(root).expect("temp config dir should be removed");
    }

    #[test]
    fn read_api_key_requires_non_empty_value() {
        let _guard = env_lock();
        let root = temp_config_home();
        std::fs::create_dir_all(&root).expect("config dir should exist");
        std::env::set_var("NANOGPT_API_KEY", "");
        std::env::set_var("PEBBLE_CONFIG_HOME", &root);
        let error = super::read_api_key().expect_err("empty key should error");
        assert!(matches!(error, crate::error::ApiError::MissingApiKey));
        std::env::remove_var("NANOGPT_API_KEY");
        std::env::remove_var("PEBBLE_CONFIG_HOME");
        std::fs::remove_dir_all(root).expect("temp config dir should be removed");
    }

    #[test]
    fn read_api_key_uses_nanogpt_env() {
        let _guard = env_lock();
        let root = temp_config_home();
        std::fs::create_dir_all(&root).expect("config dir should exist");
        std::env::set_var("NANOGPT_API_KEY", "nano-key");
        std::env::set_var("PEBBLE_CONFIG_HOME", &root);
        assert_eq!(
            super::read_api_key().expect("api key should load"),
            "nano-key"
        );
        std::env::remove_var("NANOGPT_API_KEY");
        std::env::remove_var("PEBBLE_CONFIG_HOME");
        std::fs::remove_dir_all(root).expect("temp config dir should be removed");
    }

    #[test]
    fn read_base_url_defaults_to_nanogpt_messages_api_root() {
        let _guard = env_lock();
        std::env::remove_var("NANOGPT_BASE_URL");
        assert_eq!(
            super::resolve_base_url_for(super::ApiService::NanoGpt),
            "https://nano-gpt.com/api"
        );
    }

    #[test]
    fn read_api_key_uses_pebble_credentials_file() {
        let _guard = env_lock();
        let root = temp_config_home();
        std::fs::create_dir_all(&root).expect("config dir should exist");
        std::fs::write(
            root.join("credentials.json"),
            r#"{"nanogpt_api_key":"from-credentials"}"#,
        )
        .expect("credentials should write");

        std::env::remove_var("NANOGPT_API_KEY");
        std::env::set_var("PEBBLE_CONFIG_HOME", &root);
        assert_eq!(
            super::read_api_key().expect("api key should load"),
            "from-credentials"
        );

        std::env::remove_var("PEBBLE_CONFIG_HOME");
        std::fs::remove_dir_all(root).expect("temp config dir should be removed");
    }

    #[test]
    fn resolve_openai_codex_credentials_reads_credentials_file() {
        let _guard = env_lock();
        let root = temp_config_home();
        std::fs::create_dir_all(&root).expect("config dir should exist");
        std::fs::write(
            root.join("credentials.json"),
            r#"{"openai_codex_auth":{"access_token":"access-123","refresh_token":"refresh-123","expires_at":42,"account_id":"org_123"}}"#,
        )
        .expect("credentials should write");

        std::env::remove_var("OPENAI_CODEX_ACCESS_TOKEN");
        std::env::set_var("PEBBLE_CONFIG_HOME", &root);
        let credentials =
            super::resolve_openai_codex_credentials().expect("oauth credentials should load");
        assert_eq!(credentials.access_token, "access-123");
        assert_eq!(credentials.refresh_token, "refresh-123");
        assert_eq!(credentials.expires_at, Some(42));
        assert_eq!(credentials.account_id.as_deref(), Some("org_123"));

        std::env::remove_var("PEBBLE_CONFIG_HOME");
        std::fs::remove_dir_all(root).expect("temp config dir should be removed");
    }

    #[test]
    fn message_request_stream_helper_sets_stream_true() {
        let request = MessageRequest {
            model: "openai/gpt-5.2".to_string(),
            max_tokens: 64,
            messages: vec![],
            system: None,
            tools: None,
            tool_choice: None,
            thinking: None,
            reasoning_effort: None,
            fast_mode: false,
            stream: false,
        };

        assert!(request.with_streaming().stream);
    }

    #[test]
    fn backoff_doubles_until_maximum() {
        let client = super::NanoGptClient::new("test-key").with_retry_policy(
            3,
            Duration::from_millis(10),
            Duration::from_millis(25),
        );
        assert_eq!(
            client.backoff_for_attempt(1).expect("attempt 1"),
            Duration::from_millis(10)
        );
        assert_eq!(
            client.backoff_for_attempt(2).expect("attempt 2"),
            Duration::from_millis(20)
        );
        assert_eq!(
            client.backoff_for_attempt(3).expect("attempt 3"),
            Duration::from_millis(25)
        );
    }

    #[test]
    fn retryable_statuses_are_detected() {
        assert!(super::is_retryable_status(
            reqwest::StatusCode::TOO_MANY_REQUESTS
        ));
        assert!(super::is_retryable_status(
            reqwest::StatusCode::INTERNAL_SERVER_ERROR
        ));
        assert!(!super::is_retryable_status(
            reqwest::StatusCode::UNAUTHORIZED
        ));
    }

    #[test]
    fn tool_delta_variant_round_trips() {
        let delta = ContentBlockDelta::InputJsonDelta {
            partial_json: "{\"city\":\"Paris\"}".to_string(),
        };
        let encoded = serde_json::to_string(&delta).expect("delta should serialize");
        let decoded: ContentBlockDelta =
            serde_json::from_str(&encoded).expect("delta should deserialize");
        assert_eq!(decoded, delta);
    }

    #[test]
    fn request_id_uses_primary_or_fallback_header() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(REQUEST_ID_HEADER, "req_primary".parse().expect("header"));
        assert_eq!(
            super::request_id_from_headers(&headers).as_deref(),
            Some("req_primary")
        );

        headers.clear();
        headers.insert(
            ALT_REQUEST_ID_HEADER,
            "req_fallback".parse().expect("header"),
        );
        assert_eq!(
            super::request_id_from_headers(&headers).as_deref(),
            Some("req_fallback")
        );
    }

    #[test]
    fn openai_codex_request_uses_responses_shape() {
        let client =
            super::NanoGptClient::new("ignored").with_service(super::ApiService::OpenAiCodex);
        let request = MessageRequest {
            model: "openai-codex/gpt-5.4".to_string(),
            max_tokens: 256,
            messages: vec![
                crate::types::InputMessage::user_text("What model are you?"),
                crate::types::InputMessage {
                    role: "assistant".to_string(),
                    content: vec![crate::types::InputContentBlock::ToolUse {
                        id: "call_123".to_string(),
                        name: "read_file".to_string(),
                        input: serde_json::json!({"path":"README.md"}),
                    }],
                    reasoning_content: None,
                    reasoning: None,
                },
                crate::types::InputMessage::user_tool_result("call_123", "ok", false),
            ],
            system: None,
            tools: Some(vec![crate::types::ToolDefinition {
                name: "read_file".to_string(),
                description: Some("Read a file".to_string()),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"]
                }),
            }]),
            tool_choice: Some(crate::types::ToolChoice::Auto),
            thinking: None,
            reasoning_effort: Some(crate::types::ReasoningEffort::High),
            fast_mode: true,
            stream: false,
        };

        let translated = super::message_request_to_openai_codex_request(&request, &client)
            .expect("request should translate");

        assert_eq!(translated.model, "gpt-5.4");
        assert_eq!(
            translated.instructions,
            "You are Pebble, a coding assistant."
        );
        assert!(translated.stream);
        assert!(!translated.store);
        assert_eq!(
            translated.reasoning,
            Some(super::OpenAiCodexReasoning {
                effort: crate::types::ReasoningEffort::High,
            })
        );
        assert_eq!(translated.service_tier.as_deref(), Some("priority"));
        assert_eq!(
            translated.input,
            vec![
                super::OpenAiCodexInputItem::Message {
                    role: "user".to_string(),
                    content: vec![super::OpenAiCodexContentItem::InputText {
                        text: "What model are you?".to_string(),
                    }],
                },
                super::OpenAiCodexInputItem::FunctionCall {
                    name: "read_file".to_string(),
                    arguments: r#"{"path":"README.md"}"#.to_string(),
                    call_id: "call_123".to_string(),
                },
                super::OpenAiCodexInputItem::FunctionCallOutput {
                    call_id: "call_123".to_string(),
                    output: "ok".to_string(),
                },
            ]
        );
        assert_eq!(translated.tools.len(), 1);
        assert_eq!(translated.tool_choice, "auto");
        assert_eq!(
            translated.tools[0].parameters,
            serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"}
                },
                "required": ["path"]
            })
        );
    }

    #[test]
    fn openai_codex_stream_events_assemble_message_response() {
        let mut accumulator = super::OpenAiCodexResponseAccumulator::default();
        super::process_openai_codex_sse_frame(
            concat!(
                "event: response.output_item.done\n",
                "data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"reasoning\",\"summary\":[{\"type\":\"summary_text\",\"text\":\"Inspecting request\"}]}}\n\n"
            ),
            &mut accumulator,
        )
        .expect("reasoning frame should parse");
        super::process_openai_codex_sse_frame(
            concat!(
                "event: response.output_item.done\n",
                "data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"function_call\",\"call_id\":\"call_9\",\"name\":\"read_file\",\"arguments\":\"{\\\"path\\\":\\\"README.md\\\"}\"}}\n\n"
            ),
            &mut accumulator,
        )
        .expect("tool frame should parse");
        super::process_openai_codex_sse_frame(
            concat!(
                "event: response.output_item.done\n",
                "data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"I am Pebble.\"}]}}\n\n"
            ),
            &mut accumulator,
        )
        .expect("message frame should parse");
        super::process_openai_codex_sse_frame(
            concat!(
                "event: response.completed\n",
                "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_123\",\"usage\":{\"input_tokens\":11,\"input_tokens_details\":{\"cached_tokens\":2},\"output_tokens\":7}}}\n\n"
            ),
            &mut accumulator,
        )
        .expect("completed frame should parse");

        assert_eq!(accumulator.response_id.as_deref(), Some("resp_123"));
        assert_eq!(
            accumulator.content,
            vec![
                crate::types::OutputContentBlock::Thinking {
                    thinking: "Inspecting request".to_string(),
                    signature: None,
                },
                crate::types::OutputContentBlock::ToolUse {
                    id: "call_9".to_string(),
                    name: "read_file".to_string(),
                    input: serde_json::json!({"path":"README.md"}),
                },
                crate::types::OutputContentBlock::Text {
                    text: "I am Pebble.".to_string(),
                },
            ]
        );
        assert_eq!(accumulator.usage.input_tokens, 11);
        assert_eq!(accumulator.usage.cache_read_input_tokens, 2);
        assert_eq!(accumulator.usage.output_tokens, 7);
    }

    #[test]
    fn openai_codex_normalizes_object_schema_without_properties() {
        let normalized = super::normalize_openai_codex_json_schema(&serde_json::json!({
            "type": "object",
            "additionalProperties": {
                "type": "object"
            }
        }));

        assert_eq!(
            normalized,
            serde_json::json!({
                "type": "object",
                "properties": {},
                "additionalProperties": {
                    "type": "object",
                    "properties": {}
                }
            })
        );
    }
}
