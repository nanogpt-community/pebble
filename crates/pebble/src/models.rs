use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{self, IsTerminal, Write};
use std::path::PathBuf;
use std::time::Duration;

use api::{
    resolve_api_key_for, resolve_base_url_for, resolve_root_url_for, ApiService, ModelCapabilities,
    ModelInfo, ModelPricing, ModelProvider, NanoGptClient, ProviderPrice,
};
use crossterm::cursor::MoveTo;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, size, Clear, ClearType};
use platform::{pebble_config_home as resolve_pebble_config_home, write_atomic};
use serde::{Deserialize, Serialize};

use crate::model_catalog::{self, CatalogModel};

pub(crate) use crate::model_catalog::{
    refresh_service as refresh_model_catalog,
    verify_credentials as verify_model_service_credentials,
};

const DEFAULT_SYNTHETIC_MODELS_URL: &str = "https://api.synthetic.new/openai/v1/models";
const DEFAULT_OPENAI_CONTEXT_LENGTH: u64 = 256_000;
const DEFAULT_VISIBLE_ROWS: usize = 14;
pub(crate) const AVAILABLE_SERVICES: [ApiService; 7] = [
    ApiService::NanoGpt,
    ApiService::Neuralwatt,
    ApiService::Lilac,
    ApiService::Grok,
    ApiService::Synthetic,
    ApiService::OpenAiCodex,
    ApiService::OpencodeGo,
];

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelState {
    #[serde(default)]
    pub current_model: Option<String>,
    #[serde(default)]
    pub current_service: Option<ApiService>,
    #[serde(default)]
    pub favorite_models: Vec<String>,
    #[serde(default)]
    pub provider_overrides: BTreeMap<String, String>,
    #[serde(default)]
    pub proxy_tool_calls: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub permission_mode: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub collaboration_mode: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[serde(default)]
    pub fast_mode: bool,
    #[serde(default)]
    pub max_output_tokens_by_model: BTreeMap<String, u32>,
    #[serde(default)]
    pub context_length_by_model: BTreeMap<String, u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelSelection {
    pub selected_model: Option<String>,
    pub selected_service: Option<ApiService>,
    pub favorites_changed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderSelection {
    pub selected_provider: Option<String>,
}

pub fn load_model_state() -> Result<ModelState, Box<dyn std::error::Error>> {
    let path = state_path()?;
    match fs::read_to_string(path) {
        Ok(contents) => Ok(serde_json::from_str(&contents)?),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(ModelState::default()),
        Err(error) => Err(Box::new(error)),
    }
}

pub fn save_model_state(state: &ModelState) -> Result<(), Box<dyn std::error::Error>> {
    let config_home = pebble_config_home()?;
    fs::create_dir_all(&config_home)?;
    let path = state_path()?;
    write_atomic(&path, serde_json::to_string_pretty(state)?)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600))?;
    }
    Ok(())
}

pub fn default_model_or(fallback: &str) -> String {
    load_model_state()
        .ok()
        .and_then(|state| state.current_model)
        .unwrap_or_else(|| fallback.to_string())
}

pub fn current_service_or_default() -> ApiService {
    load_model_state()
        .ok()
        .and_then(|state| {
            state
                .current_service
                .or_else(|| state.current_model.as_deref().map(infer_service_for_model))
        })
        .unwrap_or(ApiService::NanoGpt)
}

pub fn infer_service_for_model(model: &str) -> ApiService {
    let trimmed = model.trim();
    if trimmed.starts_with("hf:") {
        ApiService::Synthetic
    } else if trimmed.starts_with("openai-codex/") {
        ApiService::OpenAiCodex
    } else if trimmed.starts_with("opencode-go/") {
        ApiService::OpencodeGo
    } else if trimmed.starts_with("neuralwatt/") {
        ApiService::Neuralwatt
    } else if trimmed.starts_with("lilac/") {
        ApiService::Lilac
    } else if trimmed.starts_with("grok/") {
        ApiService::Grok
    } else {
        ApiService::NanoGpt
    }
}

pub fn persist_current_model(model: String) -> Result<(), Box<dyn std::error::Error>> {
    let mut state = load_model_state()?;
    state.current_service = Some(infer_service_for_model(&model));
    state.current_model = Some(model);
    save_model_state(&state)
}

pub fn provider_for_model(model: &str) -> Option<String> {
    load_model_state()
        .ok()
        .and_then(|state| state.provider_overrides.get(model).cloned())
}

pub fn persist_provider_for_model(
    model: &str,
    provider: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut state = load_model_state()?;
    match provider {
        Some(provider) if !provider.is_empty() => {
            state.provider_overrides.insert(model.to_string(), provider);
        }
        _ => {
            state.provider_overrides.remove(model);
        }
    }
    save_model_state(&state)
}

pub fn proxy_tool_calls_enabled() -> bool {
    load_model_state()
        .map(|state| state.proxy_tool_calls)
        .unwrap_or(false)
}

pub fn max_output_tokens_for_model_or(model: &str, fallback: u32) -> u32 {
    if let Ok(state) = load_model_state() {
        if let Some(value) = state.max_output_tokens_by_model.get(model).copied() {
            return value.max(1);
        }
    }

    let models = match fetch_service_models(infer_service_for_model(model)) {
        Ok(models) => models,
        Err(_) => return fallback,
    };

    let mut state = load_model_state().unwrap_or_default();
    update_model_metadata_cache(
        &mut state,
        &models
            .iter()
            .map(|model| model.info.clone())
            .collect::<Vec<_>>(),
    );
    let resolved = state
        .max_output_tokens_by_model
        .get(model)
        .copied()
        .filter(|value| *value > 0)
        .unwrap_or(fallback);
    let _ = save_model_state(&state);
    resolved
}

pub fn context_length_for_model(model: &str) -> Option<u64> {
    if let Ok(state) = load_model_state() {
        if let Some(value) = state.context_length_by_model.get(model).copied() {
            return Some(value.max(1));
        }
    }

    let models = match fetch_service_models(infer_service_for_model(model)) {
        Ok(models) => models,
        Err(_) => return default_context_length_for_model(model),
    };
    let mut state = load_model_state().unwrap_or_default();
    update_model_metadata_cache(
        &mut state,
        &models
            .iter()
            .map(|model| model.info.clone())
            .collect::<Vec<_>>(),
    );
    let resolved = state
        .context_length_by_model
        .get(model)
        .copied()
        .filter(|value| *value > 0)
        .or_else(|| default_context_length_for_model(model));
    let _ = save_model_state(&state);
    resolved
}

fn default_context_length_for_model(model: &str) -> Option<u64> {
    let trimmed = model.trim();
    if infer_service_for_model(trimmed) == ApiService::OpenAiCodex || trimmed.starts_with("openai/")
    {
        Some(DEFAULT_OPENAI_CONTEXT_LENGTH)
    } else {
        None
    }
}

pub fn persist_proxy_tool_calls(enabled: bool) -> Result<(), Box<dyn std::error::Error>> {
    let mut state = load_model_state()?;
    state.proxy_tool_calls = enabled;
    save_model_state(&state)
}

pub fn open_model_picker() -> Result<ModelSelection, Box<dyn std::error::Error>> {
    open_model_picker_for_service(None)
}

pub fn open_model_picker_for_service(
    service: Option<ApiService>,
) -> Result<ModelSelection, Box<dyn std::error::Error>> {
    let mut state = load_model_state()?;
    let models = fetch_all_sorted_models(&state)?;
    update_model_metadata_cache(
        &mut state,
        &models
            .iter()
            .map(|model| model.info.clone())
            .collect::<Vec<_>>(),
    );
    if models.is_empty() {
        return Err("no models were returned by Pebble providers".into());
    }

    let selection = interactive_model_picker(&models, &mut state, service)?;
    if let Some(model) = &selection.selected_model {
        state.current_model = Some(model.clone());
        state.current_service = selection.selected_service;
    }
    save_model_state(&state)?;
    Ok(selection)
}

pub fn service_from_selector(value: &str) -> Option<ApiService> {
    match value.trim().to_ascii_lowercase().as_str() {
        "nanogpt" | "nano-gpt" | "nano" => Some(ApiService::NanoGpt),
        "neuralwatt" | "neural-watt" => Some(ApiService::Neuralwatt),
        "lilac" | "getlilac" => Some(ApiService::Lilac),
        "grok" | "xai" | "x-ai" => Some(ApiService::Grok),
        "synthetic" | "synthetic.new" => Some(ApiService::Synthetic),
        "openai-codex" | "openai_codex" | "chatgpt" => Some(ApiService::OpenAiCodex),
        "opencode-go" | "opencode_go" | "opencodego" => Some(ApiService::OpencodeGo),
        _ => None,
    }
}

pub fn open_provider_picker(model: &str) -> Result<ProviderSelection, Box<dyn std::error::Error>> {
    if infer_service_for_model(model) != ApiService::NanoGpt {
        return Err(format!(
            "routing overrides are only supported for NanoGPT models; current model is on {}",
            infer_service_for_model(model).display_name()
        )
        .into());
    }
    let response = fetch_provider_selection(model)?;
    if !response.supports_provider_selection {
        return Err(format!("provider selection is not supported for {model}").into());
    }

    let mut state = load_model_state()?;
    let selected_provider = interactive_provider_picker(
        model,
        response.default_price.as_ref(),
        &response.providers,
        state.provider_overrides.get(model).cloned(),
    )?;
    match &selected_provider.selected_provider {
        Some(provider) => {
            state
                .provider_overrides
                .insert(model.to_string(), provider.clone());
        }
        None => {
            state.provider_overrides.remove(model);
        }
    }
    save_model_state(&state)?;
    Ok(selected_provider)
}

pub fn validate_provider_for_model(
    model: &str,
    provider: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if infer_service_for_model(model) != ApiService::NanoGpt {
        return Err(format!(
            "routing overrides are only supported for NanoGPT models; current model is on {}",
            infer_service_for_model(model).display_name()
        )
        .into());
    }
    let response = fetch_provider_selection(model)?;
    if !response.supports_provider_selection {
        return Err(format!("provider selection is not supported for {model}").into());
    }

    let provider_entry = response
        .providers
        .iter()
        .find(|entry| entry.provider == provider)
        .ok_or_else(|| format!("unknown provider for {model}: {provider}"))?;
    if provider_entry.available == Some(false) {
        return Err(format!("provider is currently unavailable for {model}: {provider}").into());
    }
    Ok(())
}

fn build_catalog_client() -> NanoGptClient {
    let api_key = resolve_api_key_for(ApiService::NanoGpt).unwrap_or_default();
    NanoGptClient::new(api_key)
        .with_service(ApiService::NanoGpt)
        .with_base_url(resolve_base_url_for(ApiService::NanoGpt))
}

fn fetch_provider_selection(
    model: &str,
) -> Result<api::ProviderSelectionResponse, Box<dyn std::error::Error>> {
    let client = build_catalog_client();
    let runtime = tokio::runtime::Runtime::new()?;
    Ok(runtime.block_on(client.fetch_providers(model))?)
}

fn fetch_all_sorted_models(
    state: &ModelState,
) -> Result<Vec<CatalogModel>, Box<dyn std::error::Error>> {
    let mut models = model_catalog::load_or_refresh_models()?;
    models.sort_by(|left, right| compare_models(left, right, state));
    Ok(models)
}

pub(crate) fn fetch_service_models(
    service: ApiService,
) -> Result<Vec<CatalogModel>, Box<dyn std::error::Error>> {
    match service {
        ApiService::NanoGpt => fetch_service_models_via_api(service),
        ApiService::Synthetic => fetch_synthetic_models(),
        ApiService::OpenAiCodex => {
            fetch_service_models_via_api(service).or_else(|_| Ok(openai_codex_models()))
        }
        ApiService::OpencodeGo => {
            fetch_service_models_via_api(service).or_else(|_| Ok(opencode_go_models()))
        }
        ApiService::Neuralwatt => fetch_service_models_via_api(service),
        ApiService::Lilac => fetch_service_models_via_api(service).or_else(|_| Ok(lilac_models())),
        ApiService::Grok => fetch_grok_models(),
    }
}

fn fetch_service_models_via_api(
    service: ApiService,
) -> Result<Vec<CatalogModel>, Box<dyn std::error::Error>> {
    let client = match service {
        ApiService::NanoGpt => build_catalog_client(),
        ApiService::Synthetic => {
            return Err("synthetic model catalog uses a dedicated endpoint".into());
        }
        ApiService::OpenAiCodex
        | ApiService::OpencodeGo
        | ApiService::Neuralwatt
        | ApiService::Lilac => {
            NanoGptClient::from_service_env(service)?.with_base_url(resolve_base_url_for(service))
        }
        ApiService::Grok => return fetch_grok_models(),
    };
    let client = client
        .with_request_timeout(Duration::from_secs(15))
        .with_retry_policy(0, Duration::ZERO, Duration::ZERO);
    let runtime = tokio::runtime::Runtime::new()?;
    let response = runtime.block_on(client.fetch_models(true))?;
    let models = response
        .data
        .into_iter()
        .map(|info| CatalogModel {
            service,
            info: canonicalize_model_info(service, info),
        })
        .collect::<Vec<_>>();

    if models.is_empty() {
        Err(format!("no models returned for {}", service.display_name()).into())
    } else {
        Ok(models)
    }
}

fn canonicalize_model_info(service: ApiService, mut info: ModelInfo) -> ModelInfo {
    info.id = canonical_model_id(service, &info.id);
    info
}

fn canonical_model_id(service: ApiService, model_id: &str) -> String {
    let trimmed = model_id.trim();
    match service {
        ApiService::OpenAiCodex if !trimmed.starts_with("openai-codex/") => {
            format!("openai-codex/{trimmed}")
        }
        ApiService::OpencodeGo if !trimmed.starts_with("opencode-go/") => {
            format!("opencode-go/{trimmed}")
        }
        ApiService::Neuralwatt if !trimmed.starts_with("neuralwatt/") => {
            format!("neuralwatt/{trimmed}")
        }
        ApiService::Lilac if !trimmed.starts_with("lilac/") => format!("lilac/{trimmed}"),
        ApiService::Grok if !trimmed.starts_with("grok/") => format!("grok/{trimmed}"),
        ApiService::NanoGpt
        | ApiService::Synthetic
        | ApiService::OpenAiCodex
        | ApiService::OpencodeGo
        | ApiService::Neuralwatt
        | ApiService::Lilac
        | ApiService::Grok => trimmed.to_string(),
    }
}

fn lilac_models() -> Vec<CatalogModel> {
    [
        ("moonshotai/kimi-k2.6", "Kimi K2.6", 262_144),
        ("zai-org/glm-5.2", "GLM 5.2", 524_288),
        ("google/gemma-4-31b-it", "Gemma 4 31B", 262_100),
        ("minimaxai/minimax-m3", "MiniMax M3", 1_048_576),
    ]
    .into_iter()
    .map(|(id, name, context_length)| {
        openai_compatible_catalog_model(
            ApiService::Lilac,
            id,
            name,
            context_length,
            "Lilac model. Sign in to refresh availability and pricing from the live catalog.",
        )
    })
    .collect()
}

fn fetch_grok_models() -> Result<Vec<CatalogModel>, Box<dyn std::error::Error>> {
    let models = discover_grok_model_ids()?;
    if models.is_empty() {
        return Err("official Grok CLI returned no subscription models; run `grok login`".into());
    }
    Ok(models
        .into_iter()
        .map(|id| {
            let name = id
                .split(['-', '_'])
                .map(|part| {
                    let mut chars = part.chars();
                    chars.next().map_or_else(String::new, |first| {
                        first.to_uppercase().collect::<String>() + chars.as_str()
                    })
                })
                .collect::<Vec<_>>()
                .join(" ");
            openai_compatible_catalog_model(
            ApiService::Grok,
            &id,
            &name,
            1_000_000,
            "Grok subscription model accessed through the official Grok CLI and its OAuth session.",
        )
        })
        .collect())
}

fn discover_grok_model_ids() -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let executable = std::env::var("PEBBLE_GROK_CLI").unwrap_or_else(|_| "grok".to_string());
    let output = std::process::Command::new(executable)
        .args(["--no-auto-update", "models"])
        .output()?;
    if !output.status.success() {
        return Err(format!(
            "official Grok CLI model discovery failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )
        .into());
    }
    Ok(parse_grok_model_ids(&String::from_utf8_lossy(
        &output.stdout,
    )))
}

fn parse_grok_model_ids(output: &str) -> Vec<String> {
    output
        .split_whitespace()
        .map(|token| {
            token.trim_matches(|ch: char| {
                !ch.is_ascii_alphanumeric() && !matches!(ch, '-' | '_' | '.')
            })
        })
        .filter(|token| token.starts_with("grok-") || *token == "grok-build")
        .map(str::to_string)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn openai_compatible_catalog_model(
    service: ApiService,
    id: &str,
    name: &str,
    context_length: u64,
    description: &str,
) -> CatalogModel {
    CatalogModel {
        service,
        info: ModelInfo {
            id: canonical_model_id(service, id),
            object: "model".to_string(),
            created: 0,
            owned_by: service.as_str().to_string(),
            name: Some(name.to_string()),
            description: Some(description.to_string()),
            context_length: Some(context_length),
            max_output_tokens: None,
            pricing: None,
            capabilities: Some(ModelCapabilities {
                vision: None,
                reasoning: Some(true),
                tool_calling: Some(true),
                parallel_tool_calls: None,
                structured_output: Some(true),
                pdf_upload: None,
            }),
            category: Some(service.display_name().to_string()),
            cost_estimate: None,
            tags: None,
            supports_provider_selection: Some(false),
        },
    }
}

fn fetch_synthetic_models() -> Result<Vec<CatalogModel>, Box<dyn std::error::Error>> {
    #[derive(Debug, Deserialize)]
    struct SyntheticModelsResponse {
        data: Vec<SyntheticModel>,
    }

    #[derive(Debug, Deserialize)]
    struct SyntheticModel {
        id: String,
        name: Option<String>,
        provider: Option<String>,
        created: Option<i64>,
        context_length: Option<u64>,
        max_output_length: Option<u64>,
        supported_features: Option<Vec<String>>,
    }

    let root = resolve_root_url_for(ApiService::Synthetic);
    let models_url = if root.trim_end_matches('/') == "https://api.synthetic.new" {
        DEFAULT_SYNTHETIC_MODELS_URL.to_string()
    } else {
        format!("{}/openai/v1/models", root.trim_end_matches('/'))
    };

    let runtime = tokio::runtime::Runtime::new()?;
    let payload = runtime.block_on(async {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(20))
            .build()?;
        let mut request = client.get(&models_url);
        if let Ok(api_key) = resolve_api_key_for(ApiService::Synthetic) {
            request = request.bearer_auth(api_key);
        }
        let response = request.send().await?;
        let response = response.error_for_status()?;
        response.json::<SyntheticModelsResponse>().await
    })?;

    Ok(payload
        .data
        .into_iter()
        .map(|model| CatalogModel {
            service: ApiService::Synthetic,
            info: ModelInfo {
                id: model.id,
                object: "model".to_string(),
                created: model.created.unwrap_or_default(),
                owned_by: model.provider.unwrap_or_else(|| "synthetic".to_string()),
                name: model.name,
                description: None,
                context_length: model.context_length,
                max_output_tokens: model.max_output_length,
                pricing: None,
                capabilities: synthetic_capabilities(model.supported_features.as_deref()),
                category: Some("Synthetic".to_string()),
                cost_estimate: None,
                tags: None,
                supports_provider_selection: Some(false),
            },
        })
        .collect())
}

fn synthetic_capabilities(features: Option<&[String]>) -> Option<ModelCapabilities> {
    let features = features?;
    let contains = |needle: &str| features.iter().any(|item| item == needle);
    Some(ModelCapabilities {
        vision: Some(contains("vision")),
        reasoning: Some(contains("reasoning")),
        tool_calling: Some(contains("tools")),
        parallel_tool_calls: None,
        structured_output: Some(contains("structured_outputs") || contains("json_mode")),
        pdf_upload: None,
    })
}

fn opencode_go_models() -> Vec<CatalogModel> {
    [
        ("glm-5", "GLM 5"),
        ("glm-5.1", "GLM 5.1"),
        ("kimi-k2.5", "Kimi K2.5"),
        ("kimi-k2.6", "Kimi K2.6"),
        ("mimo-v2-pro", "MiMo V2 Pro"),
        ("mimo-v2-omni", "MiMo V2 Omni"),
        ("minimax-m2.5", "MiniMax M2.5"),
        ("minimax-m2.7", "MiniMax M2.7"),
        ("qwen3.5-plus", "Qwen3.5 Plus"),
        ("qwen3.6-plus", "Qwen3.6 Plus"),
    ]
    .into_iter()
    .map(|(id, name)| CatalogModel {
        service: ApiService::OpencodeGo,
        info: ModelInfo {
            id: format!("opencode-go/{id}"),
            object: "model".to_string(),
            created: 0,
            owned_by: "opencode-go".to_string(),
            name: Some(name.to_string()),
            description: Some(
                "Curated OpenCode Go model. The available list is sourced from the OpenCode Go docs."
                    .to_string(),
            ),
            context_length: None,
            max_output_tokens: None,
            pricing: None,
            capabilities: None,
            category: Some("OpenCode Go".to_string()),
            cost_estimate: None,
            tags: None,
            supports_provider_selection: Some(false),
        },
    })
    .collect()
}

fn openai_codex_models() -> Vec<CatalogModel> {
    [
        ("gpt-5.1-codex", "GPT-5.1 Codex"),
        ("gpt-5.1-codex-mini", "GPT-5.1 Codex Mini"),
        ("gpt-5.1-codex-max", "GPT-5.1 Codex Max"),
        ("gpt-5.2", "GPT-5.2"),
        ("gpt-5.2-codex", "GPT-5.2 Codex"),
        ("gpt-5.3-codex", "GPT-5.3 Codex"),
        ("gpt-5.4", "GPT-5.4"),
        ("gpt-5.4-mini", "GPT-5.4 Mini"),
        ("gpt-5.5", "GPT-5.5"),
    ]
    .into_iter()
    .map(|(id, name)| CatalogModel {
        service: ApiService::OpenAiCodex,
        info: ModelInfo {
            id: format!("openai-codex/{id}"),
            object: "model".to_string(),
            created: 0,
            owned_by: "openai-codex".to_string(),
            name: Some(name.to_string()),
            description: Some(
                "ChatGPT plan-backed Codex model accessed through OpenAI OAuth/device-code authentication."
                    .to_string(),
            ),
            context_length: Some(DEFAULT_OPENAI_CONTEXT_LENGTH),
            max_output_tokens: None,
            pricing: Some(ModelPricing {
                prompt: Some(0.0),
                completion: Some(0.0),
                currency: Some("USD".to_string()),
                unit: Some("included_with_plan".to_string()),
            }),
            capabilities: Some(ModelCapabilities {
                vision: Some(false),
                reasoning: Some(true),
                tool_calling: Some(true),
                parallel_tool_calls: Some(true),
                structured_output: Some(true),
                pdf_upload: None,
            }),
            category: Some("OpenAI Codex".to_string()),
            cost_estimate: None,
            tags: Some(vec!["chatgpt-plan".to_string(), "oauth".to_string()]),
            supports_provider_selection: Some(false),
        },
    })
    .collect()
}

fn update_model_metadata_cache(state: &mut ModelState, models: &[ModelInfo]) {
    for model in models {
        if let Some(max_output_tokens) = model
            .max_output_tokens
            .map(|value| value.min(u64::from(u32::MAX)) as u32)
            .filter(|value| *value > 0)
        {
            state
                .max_output_tokens_by_model
                .insert(model.id.clone(), max_output_tokens);
        }
        if let Some(context_length) = model.context_length.filter(|value| *value > 0) {
            state
                .context_length_by_model
                .insert(model.id.clone(), context_length);
        }
    }
}

fn compare_models(left: &CatalogModel, right: &CatalogModel, state: &ModelState) -> Ordering {
    let left_favorite = state
        .favorite_models
        .iter()
        .any(|item| item == &left.info.id);
    let right_favorite = state
        .favorite_models
        .iter()
        .any(|item| item == &right.info.id);
    let left_current = state.current_model.as_deref() == Some(left.info.id.as_str());
    let right_current = state.current_model.as_deref() == Some(right.info.id.as_str());

    right_current
        .cmp(&left_current)
        .then_with(|| right_favorite.cmp(&left_favorite))
        .then_with(|| left.service.as_str().cmp(right.service.as_str()))
        .then_with(|| {
            display_name(&left.info)
                .to_ascii_lowercase()
                .cmp(&display_name(&right.info).to_ascii_lowercase())
        })
        .then_with(|| left.info.id.cmp(&right.info.id))
}

fn interactive_model_picker(
    models: &[CatalogModel],
    state: &mut ModelState,
    initial_service: Option<ApiService>,
) -> Result<ModelSelection, Box<dyn std::error::Error>> {
    if !io::stdin().is_terminal() || !io::stdout().is_terminal() {
        return select_model_fallback(models, state, initial_service);
    }

    let current_index = state
        .current_model
        .as_deref()
        .and_then(|model| models.iter().position(|entry| entry.info.id == model))
        .unwrap_or(0);
    let mut current_service = initial_service
        .or(state.current_service)
        .or_else(|| state.current_model.as_deref().map(infer_service_for_model))
        .unwrap_or(ApiService::NanoGpt);
    let mut query = String::new();
    let mut search_mode = false;
    let mut filtered_indices = filtered_model_indices(models, current_service, &query);
    let mut cursor = filtered_indices
        .iter()
        .position(|index| *index == current_index)
        .unwrap_or(0);
    let mut favorites_changed = false;
    enable_raw_mode()?;
    let mut stdout = io::stdout();

    loop {
        draw_model_picker(
            &mut stdout,
            models,
            state,
            &filtered_indices,
            cursor,
            current_service,
            &query,
            search_mode,
        )?;
        match event::read()? {
            Event::Key(
                KeyEvent {
                    code: KeyCode::Tab, ..
                }
                | KeyEvent {
                    code: KeyCode::Right,
                    ..
                },
            ) => {
                current_service = cycle_service(current_service, true);
                filtered_indices = filtered_model_indices(models, current_service, &query);
                cursor = 0;
            }
            Event::Key(KeyEvent {
                code: KeyCode::Left,
                ..
            }) => {
                current_service = cycle_service(current_service, false);
                filtered_indices = filtered_model_indices(models, current_service, &query);
                cursor = 0;
            }
            Event::Key(KeyEvent {
                code: KeyCode::Up, ..
            }) => {
                cursor = cursor.saturating_sub(1);
            }
            Event::Key(KeyEvent {
                code: KeyCode::Down,
                ..
            }) => {
                cursor = (cursor + 1).min(filtered_indices.len().saturating_sub(1));
            }
            Event::Key(KeyEvent {
                code: KeyCode::PageUp,
                ..
            }) => {
                cursor = cursor.saturating_sub(DEFAULT_VISIBLE_ROWS);
            }
            Event::Key(KeyEvent {
                code: KeyCode::PageDown,
                ..
            }) => {
                cursor =
                    (cursor + DEFAULT_VISIBLE_ROWS).min(filtered_indices.len().saturating_sub(1));
            }
            Event::Key(KeyEvent {
                code: KeyCode::Home,
                ..
            }) => cursor = 0,
            Event::Key(KeyEvent {
                code: KeyCode::End, ..
            }) => {
                cursor = filtered_indices.len().saturating_sub(1);
            }
            Event::Key(KeyEvent {
                code: KeyCode::Char('f'),
                modifiers: KeyModifiers::CONTROL,
                ..
            }) => {
                if let Some(model_index) = filtered_indices.get(cursor) {
                    toggle_favorite(&models[*model_index].info.id, state);
                    favorites_changed = true;
                }
            }
            Event::Key(KeyEvent {
                code: KeyCode::Enter,
                ..
            }) => {
                let Some(model_index) = filtered_indices.get(cursor) else {
                    continue;
                };
                disable_raw_mode()?;
                write!(stdout, "\r\n")?;
                return Ok(ModelSelection {
                    selected_model: Some(models[*model_index].info.id.clone()),
                    selected_service: Some(models[*model_index].service),
                    favorites_changed,
                });
            }
            Event::Key(KeyEvent {
                code: KeyCode::Backspace,
                ..
            }) => {
                if !query.is_empty() {
                    query.pop();
                    filtered_indices = filtered_model_indices(models, current_service, &query);
                    cursor = updated_cursor(&filtered_indices, cursor, current_index);
                    search_mode = true;
                }
            }
            Event::Key(KeyEvent {
                code: KeyCode::Char('u'),
                modifiers,
                ..
            }) if modifiers.contains(KeyModifiers::CONTROL) => {
                query.clear();
                filtered_indices = filtered_model_indices(models, current_service, &query);
                cursor = updated_cursor(&filtered_indices, cursor, current_index);
                search_mode = false;
            }
            Event::Key(KeyEvent {
                code: KeyCode::Char(ch),
                modifiers,
                ..
            }) if (modifiers.is_empty() || modifiers == KeyModifiers::SHIFT)
                && (ch != 'q' || search_mode || !query.is_empty()) =>
            {
                query.push(ch);
                search_mode = true;
                filtered_indices = filtered_model_indices(models, current_service, &query);
                cursor = updated_cursor(&filtered_indices, cursor, current_index);
            }
            Event::Key(
                KeyEvent {
                    code: KeyCode::Esc, ..
                }
                | KeyEvent {
                    code: KeyCode::Char('q'),
                    modifiers: KeyModifiers::NONE,
                    ..
                },
            ) => {
                if search_mode || !query.is_empty() {
                    query.clear();
                    filtered_indices = filtered_model_indices(models, current_service, &query);
                    cursor = updated_cursor(&filtered_indices, cursor, current_index);
                    search_mode = false;
                    continue;
                }
                disable_raw_mode()?;
                write!(stdout, "\r\n")?;
                return Ok(ModelSelection {
                    selected_model: None,
                    selected_service: None,
                    favorites_changed,
                });
            }
            Event::Resize(_, _) => {}
            _ => {}
        }
    }
}

fn select_model_fallback(
    models: &[CatalogModel],
    state: &mut ModelState,
    service: Option<ApiService>,
) -> Result<ModelSelection, Box<dyn std::error::Error>> {
    let visible_models = models
        .iter()
        .filter(|model| service.is_none_or(|service| model.service == service))
        .collect::<Vec<_>>();
    println!("Models");
    for (index, model) in visible_models.iter().take(25).enumerate() {
        let current = if state.current_model.as_deref() == Some(model.info.id.as_str()) {
            ">"
        } else {
            " "
        };
        let favorite = if state
            .favorite_models
            .iter()
            .any(|entry| entry == &model.info.id)
        {
            "*"
        } else {
            " "
        };
        println!(
            "{:>2}. {}{} [{}] {} ({})",
            index + 1,
            current,
            favorite,
            model.service.display_name(),
            display_name(&model.info),
            model.info.id
        );
    }
    print!("Choose a model by number or exact id: ");
    io::stdout().flush()?;

    let mut buffer = String::new();
    io::stdin().read_line(&mut buffer)?;
    let input = buffer.trim();
    if input.is_empty() {
        return Ok(ModelSelection {
            selected_model: None,
            selected_service: None,
            favorites_changed: false,
        });
    }

    let selected_model = if let Ok(index) = input.parse::<usize>() {
        visible_models
            .get(index.saturating_sub(1))
            .map(|model| model.info.id.clone())
            .ok_or_else(|| format!("model number {index} is out of range"))?
    } else {
        visible_models
            .iter()
            .find(|model| model.info.id == input)
            .map(|model| model.info.id.clone())
            .ok_or_else(|| format!("unknown model id: {input}"))?
    };

    Ok(ModelSelection {
        selected_model: Some(selected_model.clone()),
        selected_service: Some(infer_service_for_model(&selected_model)),
        favorites_changed: false,
    })
}

fn interactive_provider_picker(
    model: &str,
    _default_price: Option<&ProviderPrice>,
    providers: &[ModelProvider],
    current_provider: Option<String>,
) -> Result<ProviderSelection, Box<dyn std::error::Error>> {
    let mut entries = Vec::with_capacity(providers.len() + 1);
    entries.push((None, "Platform default".to_string()));
    for provider in providers {
        let availability = if provider.available == Some(false) {
            "unavailable"
        } else {
            "available"
        };
        entries.push((
            Some(provider.provider.clone()),
            format!("{} [{}]", provider.provider, availability),
        ));
    }

    if !io::stdin().is_terminal() || !io::stdout().is_terminal() {
        println!("NanoGPT routes for {model}");
        for (index, (provider, label)) in entries.iter().enumerate() {
            let current = if current_provider.as_deref() == provider.as_deref() {
                ">"
            } else {
                " "
            };
            println!("{:>2}. {} {}", index + 1, current, label);
        }
        print!("Choose a route number, exact id, or press Enter for platform default: ");
        io::stdout().flush()?;

        let mut buffer = String::new();
        io::stdin().read_line(&mut buffer)?;
        let input = buffer.trim();
        if input.is_empty() {
            return Ok(ProviderSelection {
                selected_provider: None,
            });
        }

        let selected_provider = if let Ok(index) = input.parse::<usize>() {
            entries
                .get(index.saturating_sub(1))
                .map(|entry| entry.0.clone())
                .ok_or_else(|| format!("provider number {index} is out of range"))?
        } else {
            providers
                .iter()
                .find(|provider| provider.provider == input)
                .map(|provider| Some(provider.provider.clone()))
                .ok_or_else(|| format!("unknown provider id: {input}"))?
        };

        return Ok(ProviderSelection { selected_provider });
    }

    let current_index = current_provider
        .as_deref()
        .and_then(|provider| {
            entries
                .iter()
                .position(|entry| entry.0.as_deref() == Some(provider))
        })
        .unwrap_or(0);
    let mut query = String::new();
    let mut search_mode = false;
    let mut filtered_indices = filtered_provider_indices(&entries, &query);
    let mut cursor = filtered_indices
        .iter()
        .position(|index| *index == current_index)
        .unwrap_or(0);
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    loop {
        execute!(stdout, MoveTo(0, 0), Clear(ClearType::All))?;
        write_raw_line(&mut stdout, &format!("NanoGPT routes for {model}"))?;
        write_raw_line(
            &mut stdout,
            "Up/Down move, Enter select, / search, q cancel",
        )?;
        let search_label = if query.is_empty() {
            if search_mode {
                "Search: ".to_string()
            } else {
                "Search: / to filter".to_string()
            }
        } else if search_mode {
            format!("Search: {query}_")
        } else {
            format!("Search: {query}")
        };
        write_raw_line(&mut stdout, &search_label)?;
        write_raw_line(
            &mut stdout,
            &format!(
                "Current route: {}",
                current_provider.as_deref().unwrap_or("<platform default>")
            ),
        )?;
        write_raw_line(&mut stdout, "")?;

        for (visible_index, entry_index) in filtered_indices.iter().enumerate() {
            let (_, label) = &entries[*entry_index];
            let selected_marker = if visible_index == cursor { ">" } else { " " };
            let current_marker =
                if current_provider.as_deref() == entries[*entry_index].0.as_deref() {
                    "o"
                } else {
                    " "
                };
            write_raw_line(
                &mut stdout,
                &format!("{selected_marker}{current_marker} {label}"),
            )?;
        }
        if filtered_indices.is_empty() {
            write_raw_line(&mut stdout, "No providers match the current search.")?;
        }
        stdout.flush()?;

        match event::read()? {
            Event::Key(KeyEvent {
                code: KeyCode::Char('/'),
                modifiers: KeyModifiers::NONE,
                ..
            }) => {
                search_mode = true;
            }
            Event::Key(KeyEvent {
                code: KeyCode::Up, ..
            }) if !search_mode => cursor = cursor.saturating_sub(1),
            Event::Key(KeyEvent {
                code: KeyCode::Down,
                ..
            }) if !search_mode => {
                cursor = (cursor + 1).min(filtered_indices.len().saturating_sub(1));
            }
            Event::Key(KeyEvent {
                code: KeyCode::Enter,
                ..
            }) => {
                if search_mode {
                    search_mode = false;
                    continue;
                }
                let Some(entry_index) = filtered_indices.get(cursor) else {
                    continue;
                };
                disable_raw_mode()?;
                write!(stdout, "\r\n")?;
                return Ok(ProviderSelection {
                    selected_provider: entries[*entry_index].0.clone(),
                });
            }
            Event::Key(KeyEvent {
                code: KeyCode::Backspace,
                ..
            }) => {
                if !query.is_empty() {
                    query.pop();
                    filtered_indices = filtered_provider_indices(&entries, &query);
                    cursor = updated_cursor(&filtered_indices, cursor, current_index);
                    search_mode = true;
                }
            }
            Event::Key(KeyEvent {
                code: KeyCode::Char('u'),
                modifiers,
                ..
            }) if modifiers.contains(KeyModifiers::CONTROL) => {
                query.clear();
                filtered_indices = filtered_provider_indices(&entries, &query);
                cursor = updated_cursor(&filtered_indices, cursor, current_index);
                search_mode = false;
            }
            Event::Key(KeyEvent {
                code: KeyCode::Char(ch),
                modifiers,
                ..
            }) if search_mode && (modifiers.is_empty() || modifiers == KeyModifiers::SHIFT) => {
                query.push(ch);
                filtered_indices = filtered_provider_indices(&entries, &query);
                cursor = updated_cursor(&filtered_indices, cursor, current_index);
            }
            Event::Key(
                KeyEvent {
                    code: KeyCode::Esc, ..
                }
                | KeyEvent {
                    code: KeyCode::Char('q'),
                    modifiers: KeyModifiers::NONE,
                    ..
                },
            ) => {
                if search_mode || !query.is_empty() {
                    query.clear();
                    filtered_indices = filtered_provider_indices(&entries, &query);
                    cursor = updated_cursor(&filtered_indices, cursor, current_index);
                    search_mode = false;
                    continue;
                }
                disable_raw_mode()?;
                write!(stdout, "\r\n")?;
                return Ok(ProviderSelection {
                    selected_provider: current_provider,
                });
            }
            _ => {}
        }
    }
}

fn draw_model_picker(
    stdout: &mut impl Write,
    models: &[CatalogModel],
    state: &ModelState,
    filtered_indices: &[usize],
    cursor: usize,
    current_service: ApiService,
    query: &str,
    search_mode: bool,
) -> io::Result<()> {
    execute!(stdout, MoveTo(0, 0), Clear(ClearType::All))?;
    let (width, height) = terminal_dimensions();
    let content_width = width.saturating_sub(1).clamp(24, 88);
    let visible_rows = model_visible_rows(height);
    write_raw_line(stdout, "Choose a model")?;
    write_raw_line(
        stdout,
        &fit_line(
            "Type to search every provider  ·  ↑/↓ move  ·  ←/→ provider  ·  Enter use  ·  Esc cancel",
            content_width,
        ),
    )?;
    write_raw_line(
        stdout,
        &format!(
            "Provider: {}",
            service_tab_label(current_service, current_service)
        ),
    )?;
    let provider_rail = AVAILABLE_SERVICES
        .iter()
        .copied()
        .map(|service| service_tab_label(service, current_service))
        .collect::<Vec<_>>()
        .join("  ");
    for line in wrap_text(&provider_rail, content_width, 2) {
        write_raw_line(stdout, &line)?;
    }
    for line in wrap_text(&service_status_line(current_service), content_width, 2) {
        write_raw_line(stdout, &line)?;
    }
    write_raw_line(
        stdout,
        &fit_line(
            &format!(
                "Using: {}",
                state.current_model.as_deref().unwrap_or("not selected")
            ),
            content_width,
        ),
    )?;
    let search_label = if query.is_empty() {
        if search_mode {
            "Search: ".to_string()
        } else {
            "Search: start typing".to_string()
        }
    } else if search_mode {
        format!("Search: {query}_")
    } else {
        format!("Search: {query}")
    };
    write_raw_line(stdout, &fit_line(&search_label, content_width))?;
    write_raw_line(stdout, "")?;

    let start = cursor.saturating_sub(visible_rows / 2);
    let end = (start + visible_rows).min(filtered_indices.len());
    let start = end.saturating_sub(visible_rows);

    for (visible_index, model_index) in filtered_indices
        .iter()
        .enumerate()
        .skip(start)
        .take(end - start)
    {
        let model = &models[*model_index];
        let selected_marker = if visible_index == cursor { ">" } else { " " };
        let favorite_marker = if state
            .favorite_models
            .iter()
            .any(|entry| entry == &model.info.id)
        {
            "*"
        } else {
            " "
        };
        let current_marker = if state.current_model.as_deref() == Some(model.info.id.as_str()) {
            "o"
        } else {
            " "
        };
        write_raw_line(
            stdout,
            &format!(
                "{}{}{} {}",
                selected_marker,
                favorite_marker,
                current_marker,
                fit_line(&model_list_label(model), content_width.saturating_sub(4))
            ),
        )?;
    }

    write_raw_line(stdout, "")?;
    if let Some(model_index) = filtered_indices.get(cursor) {
        let selected = &models[*model_index];
        write_raw_line(
            stdout,
            &fit_line(&selected_summary(selected), content_width),
        )?;
        if let Some(description) = &selected.info.description {
            for line in wrap_text(description, content_width, 2) {
                write_raw_line(stdout, &line)?;
            }
        }
        for line in wrap_text(&detail_line(selected), content_width, 2) {
            write_raw_line(stdout, &line)?;
        }
    } else {
        write_raw_line(stdout, "No models match the current search.")?;
        write_raw_line(
            stdout,
            "Start typing to search every provider. Esc clears the search.",
        )?;
    }
    stdout.flush()
}

fn cycle_service(current: ApiService, forward: bool) -> ApiService {
    let current_index = AVAILABLE_SERVICES
        .iter()
        .position(|service| *service == current)
        .unwrap_or(0);
    let offset = if forward {
        1
    } else {
        AVAILABLE_SERVICES.len().saturating_sub(1)
    };
    AVAILABLE_SERVICES[(current_index + offset) % AVAILABLE_SERVICES.len()]
}

fn service_tab_label(service: ApiService, current: ApiService) -> String {
    let marker = if service_is_ready(service) {
        "●"
    } else {
        "○"
    };
    if service == current {
        format!("[{marker} {}]", service.display_name())
    } else {
        format!("{marker} {}", service.display_name())
    }
}

fn service_is_ready(service: ApiService) -> bool {
    match service {
        ApiService::Grok => command_available(
            &std::env::var("PEBBLE_GROK_CLI").unwrap_or_else(|_| "grok".to_string()),
        ),
        _ => resolve_api_key_for(service).is_ok(),
    }
}

fn service_status_line(service: ApiService) -> String {
    let catalog = model_catalog::health_label(service);
    match service {
        ApiService::Grok if service_is_ready(service) => {
            format!("● Official Grok CLI found · OAuth subscription session · {catalog}")
        }
        ApiService::Grok => {
            "○ Official Grok CLI required · install it from x.ai, then run /login grok".to_string()
        }
        _ if service_is_ready(service) => {
            format!("● {} is connected · {catalog}", service.display_name())
        }
        _ => format!(
            "○ {} needs an API key · /login {}",
            service.display_name(),
            service.as_str().replace('_', "-")
        ),
    }
}

fn command_available(command: &str) -> bool {
    let path = std::path::Path::new(command);
    if path.components().count() > 1 {
        return path.is_file();
    }
    std::env::var_os("PATH").is_some_and(|path| {
        std::env::split_paths(&path).any(|directory| directory.join(command).is_file())
    })
}

fn terminal_dimensions() -> (usize, usize) {
    size()
        .map(|(width, height)| (usize::from(width), usize::from(height)))
        .unwrap_or((120, 32))
}

fn model_visible_rows(height: usize) -> usize {
    height.saturating_sub(10).clamp(6, DEFAULT_VISIBLE_ROWS)
}

fn model_list_label(model: &CatalogModel) -> String {
    let mut segments = vec![display_name(&model.info).to_string()];
    if display_name(&model.info) != model.info.id {
        segments.push(short_model_id(&model.info.id));
    }
    segments.push(model.info.owned_by.clone());
    if let Some(category) = &model.info.category {
        segments.push(category.clone());
    }
    segments.join(" | ")
}

fn selected_summary(model: &CatalogModel) -> String {
    format!(
        "Selected: {} | {} | {}",
        model.service.display_name(),
        display_name(&model.info),
        model.info.id
    )
}

fn detail_line(model: &CatalogModel) -> String {
    let mut parts = Vec::new();
    if let Some(category) = &model.info.category {
        parts.push(format!("category={category}"));
    }
    if let Some(context_length) = model.info.context_length {
        parts.push(format!("ctx={context_length}"));
    }
    if let Some(max_output_tokens) = model.info.max_output_tokens {
        parts.push(format!("max_out={max_output_tokens}"));
    }
    let capabilities = capability_summary(model.info.capabilities.as_ref());
    if !capabilities.is_empty() {
        parts.push(format!("caps={capabilities}"));
    }
    parts.join(" | ")
}

fn capability_summary(capabilities: Option<&ModelCapabilities>) -> String {
    let Some(capabilities) = capabilities else {
        return String::new();
    };
    let mut names = Vec::new();
    if capabilities.vision == Some(true) {
        names.push("vision");
    }
    if capabilities.reasoning == Some(true) {
        names.push("reasoning");
    }
    if capabilities.tool_calling == Some(true) {
        names.push("tools");
    }
    if capabilities.parallel_tool_calls == Some(true) {
        names.push("parallel-tools");
    }
    if capabilities.structured_output == Some(true) {
        names.push("structured-output");
    }
    if capabilities.pdf_upload == Some(true) {
        names.push("pdf");
    }
    names.join(", ")
}

fn display_name(model: &ModelInfo) -> &str {
    model.name.as_deref().unwrap_or(&model.id)
}

fn short_model_id(model_id: &str) -> String {
    const KEEP: usize = 28;
    if model_id.chars().count() <= KEEP {
        return model_id.to_string();
    }
    truncate(model_id, KEEP)
}

fn write_raw_line(stdout: &mut impl Write, line: &str) -> io::Result<()> {
    write!(stdout, "{line}\r\n")
}

fn filtered_model_indices(models: &[CatalogModel], service: ApiService, query: &str) -> Vec<usize> {
    if query.trim().is_empty() {
        return models
            .iter()
            .enumerate()
            .filter_map(|(index, model)| (model.service == service).then_some(index))
            .collect();
    }

    let query = query.to_ascii_lowercase();
    models
        .iter()
        .enumerate()
        .filter_map(|(index, model)| model_matches_query(&model.info, &query).then_some(index))
        .collect()
}

fn model_matches_query(model: &ModelInfo, query: &str) -> bool {
    [
        Some(display_name(model)),
        Some(model.id.as_str()),
        Some(model.owned_by.as_str()),
        model.category.as_deref(),
        model.description.as_deref(),
    ]
    .into_iter()
    .flatten()
    .any(|value| fuzzy_match(&value.to_ascii_lowercase(), query))
}

fn filtered_provider_indices(entries: &[(Option<String>, String)], query: &str) -> Vec<usize> {
    if query.trim().is_empty() {
        return (0..entries.len()).collect();
    }

    let query = query.to_ascii_lowercase();
    entries
        .iter()
        .enumerate()
        .filter_map(|(index, (provider, label))| {
            let matches_provider = provider
                .as_deref()
                .is_some_and(|value| fuzzy_match(&value.to_ascii_lowercase(), &query));
            let matches_label = fuzzy_match(&label.to_ascii_lowercase(), &query);
            (matches_provider || matches_label).then_some(index)
        })
        .collect()
}

fn fuzzy_match(haystack: &str, query: &str) -> bool {
    if query.is_empty() {
        return true;
    }
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

fn updated_cursor(
    filtered_indices: &[usize],
    previous_cursor: usize,
    fallback_index: usize,
) -> usize {
    if filtered_indices.is_empty() {
        return 0;
    }
    if previous_cursor < filtered_indices.len() {
        return previous_cursor;
    }

    filtered_indices
        .iter()
        .position(|model_index| *model_index == fallback_index)
        .unwrap_or(0)
}

fn truncate(input: &str, max_chars: usize) -> String {
    if max_chars == 0 {
        return String::new();
    }
    let mut output = String::new();
    let mut count = 0;
    for ch in input.chars() {
        if count >= max_chars {
            if max_chars <= 3 {
                return ".".repeat(max_chars);
            }
            output = output.chars().take(max_chars - 3).collect();
            output.push_str("...");
            return output;
        }
        output.push(ch);
        count += 1;
    }
    output
}

fn fit_line(input: &str, width: usize) -> String {
    truncate(input, width)
}

fn wrap_text(input: &str, width: usize, max_lines: usize) -> Vec<String> {
    if max_lines == 0 {
        return Vec::new();
    }
    let width = width.max(8);
    let mut lines = Vec::new();
    let mut current = String::new();

    for word in input.split_whitespace() {
        let candidate_len = if current.is_empty() {
            word.chars().count()
        } else {
            current.chars().count() + 1 + word.chars().count()
        };
        if candidate_len <= width {
            if !current.is_empty() {
                current.push(' ');
            }
            current.push_str(word);
            continue;
        }

        if !current.is_empty() {
            lines.push(current);
            if lines.len() == max_lines {
                let last = lines.pop().unwrap_or_default();
                lines.push(truncate(&last, width));
                return lines;
            }
            current = String::new();
        }

        if word.chars().count() > width {
            lines.push(truncate(word, width));
            if lines.len() == max_lines {
                return lines;
            }
        } else {
            current.push_str(word);
        }
    }

    if !current.is_empty() && lines.len() < max_lines {
        lines.push(current);
    }

    lines
}

fn toggle_favorite(model_id: &str, state: &mut ModelState) {
    if let Some(index) = state
        .favorite_models
        .iter()
        .position(|entry| entry == model_id)
    {
        state.favorite_models.remove(index);
    } else {
        state.favorite_models.push(model_id.to_string());
        state.favorite_models.sort();
        state.favorite_models.dedup();
    }
}

fn state_path() -> Result<PathBuf, Box<dyn std::error::Error>> {
    Ok(pebble_config_home()?.join("state.json"))
}

fn pebble_config_home() -> Result<PathBuf, Box<dyn std::error::Error>> {
    resolve_pebble_config_home()
        .ok_or_else(|| "could not resolve PEBBLE_CONFIG_HOME, HOME, or USERPROFILE".into())
}

#[cfg(test)]
mod tests {
    use api::{ApiService, ModelInfo};

    use super::{
        default_context_length_for_model, filtered_model_indices, infer_service_for_model,
        openai_codex_models, parse_grok_model_ids, service_from_selector, toggle_favorite,
        update_model_metadata_cache, CatalogModel, ModelState, DEFAULT_OPENAI_CONTEXT_LENGTH,
    };

    #[test]
    fn toggles_favorite_membership() {
        let mut state = ModelState::default();
        toggle_favorite("openai/gpt-5.2", &mut state);
        assert_eq!(state.favorite_models, vec!["openai/gpt-5.2"]);
        toggle_favorite("openai/gpt-5.2", &mut state);
        assert!(state.favorite_models.is_empty());
    }

    #[test]
    fn searches_models_across_providers_by_name_id_or_category() {
        let models = vec![
            catalog_model(
                ApiService::Lilac,
                "zai-org/glm-5.1",
                Some("GLM 5.1"),
                Some("More"),
            ),
            catalog_model(
                ApiService::NanoGpt,
                "openai/gpt-5.4",
                Some("GPT 5.4"),
                Some("Flagship"),
            ),
        ];

        assert_eq!(
            filtered_model_indices(&models, ApiService::NanoGpt, "glm"),
            vec![0]
        );
        assert_eq!(
            filtered_model_indices(&models, ApiService::NanoGpt, "gpt-5.4"),
            vec![1]
        );
        assert_eq!(
            filtered_model_indices(&models, ApiService::NanoGpt, "flag"),
            vec![1]
        );
        assert_eq!(
            filtered_model_indices(&models, ApiService::NanoGpt, "gm51"),
            vec![0]
        );
    }

    #[test]
    fn infers_opencode_go_service_from_model_prefix() {
        assert_eq!(
            infer_service_for_model("openai-codex/gpt-5.4"),
            ApiService::OpenAiCodex
        );
        assert_eq!(
            infer_service_for_model("opencode-go/glm-5.1"),
            ApiService::OpencodeGo
        );
        assert_eq!(
            infer_service_for_model("neuralwatt/zai-org/glm-5.2"),
            ApiService::Neuralwatt
        );
        assert_eq!(
            infer_service_for_model("lilac/moonshotai/kimi-k2.6"),
            ApiService::Lilac
        );
        assert_eq!(infer_service_for_model("grok/grok-4.5"), ApiService::Grok);
    }

    #[test]
    fn resolves_human_provider_names_and_aliases() {
        assert_eq!(service_from_selector("Nano-GPT"), Some(ApiService::NanoGpt));
        assert_eq!(
            service_from_selector("NeuralWatt"),
            Some(ApiService::Neuralwatt)
        );
        assert_eq!(service_from_selector("getlilac"), Some(ApiService::Lilac));
        assert_eq!(service_from_selector("xai"), Some(ApiService::Grok));
        assert_eq!(
            service_from_selector("chatgpt"),
            Some(ApiService::OpenAiCodex)
        );
        assert_eq!(service_from_selector("nope"), None);
    }

    #[test]
    fn truncates_unicode_without_using_character_counts_as_byte_offsets() {
        assert_eq!(super::truncate("○ Neuralwatt ● Lilac", 12), "○ Neuralw...");
        assert_eq!(super::truncate("●●●●", 3), "...");
    }

    #[test]
    fn parses_and_deduplicates_official_grok_model_output() {
        assert_eq!(
            parse_grok_model_ids("* grok-4.5  Grok 4.5\ngrok-build (default)\ngrok-4.5"),
            vec!["grok-4.5", "grok-build"]
        );
    }

    #[test]
    fn caches_max_output_tokens_from_catalog_models() {
        let mut state = ModelState::default();
        let mut models = vec![model("zai-org/glm-5.1", Some("GLM 5.1"), Some("More"))];
        models[0].max_output_tokens = Some(131_072);

        update_model_metadata_cache(&mut state, &models);

        assert_eq!(
            state.max_output_tokens_by_model.get("zai-org/glm-5.1"),
            Some(&131_072)
        );
    }

    #[test]
    fn caches_context_length_from_catalog_models() {
        let mut state = ModelState::default();
        let mut models = vec![model(
            "openai-codex/gpt-5.4",
            Some("GPT-5.4"),
            Some("Flagship"),
        )];
        models[0].context_length = Some(393_216);

        update_model_metadata_cache(&mut state, &models);

        assert_eq!(
            state.context_length_by_model.get("openai-codex/gpt-5.4"),
            Some(&393_216)
        );
    }

    #[test]
    fn defaults_openai_context_length_to_256k() {
        assert_eq!(
            default_context_length_for_model("openai-codex/gpt-5.4"),
            Some(DEFAULT_OPENAI_CONTEXT_LENGTH)
        );
        assert_eq!(
            default_context_length_for_model("openai/gpt-5.2"),
            Some(DEFAULT_OPENAI_CONTEXT_LENGTH)
        );
        assert_eq!(default_context_length_for_model("anthropic/claude"), None);
    }

    #[test]
    fn openai_codex_fallback_catalog_has_default_context_length() {
        let models = openai_codex_models();

        assert!(models
            .iter()
            .any(|model| model.info.id == "openai-codex/gpt-5.4"));
        assert!(models
            .iter()
            .all(|model| { model.info.context_length == Some(DEFAULT_OPENAI_CONTEXT_LENGTH) }));
    }

    fn model(id: &str, name: Option<&str>, category: Option<&str>) -> ModelInfo {
        ModelInfo {
            id: id.to_string(),
            object: "model".to_string(),
            created: 0,
            owned_by: "provider".to_string(),
            name: name.map(ToOwned::to_owned),
            description: None,
            context_length: None,
            max_output_tokens: None,
            pricing: None,
            capabilities: None,
            category: category.map(ToOwned::to_owned),
            cost_estimate: None,
            tags: None,
            supports_provider_selection: None,
        }
    }

    fn catalog_model(
        service: ApiService,
        id: &str,
        name: Option<&str>,
        category: Option<&str>,
    ) -> CatalogModel {
        CatalogModel {
            service,
            info: model(id, name, category),
        }
    }
}
