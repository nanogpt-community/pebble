use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::PathBuf;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use api::{resolve_base_url_for, ApiError, ApiService, ModelInfo, NanoGptClient};
use platform::{pebble_config_home, write_atomic};
use serde::{Deserialize, Serialize};

use crate::models::{fetch_service_models, AVAILABLE_SERVICES};

const CACHE_TTL_SECS: u64 = 5 * 60;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct CatalogModel {
    pub(crate) service: ApiService,
    pub(crate) info: ModelInfo,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct CatalogCache {
    #[serde(default)]
    services: BTreeMap<String, CachedCatalog>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct CachedCatalog {
    updated_at: u64,
    #[serde(default)]
    models: Vec<CatalogModel>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    last_error: Option<String>,
}

pub(crate) fn load_or_refresh_models() -> Result<Vec<CatalogModel>, Box<dyn std::error::Error>> {
    let cache = load_cache().unwrap_or_default();
    let now = epoch_secs();
    let mut models = cache
        .services
        .values()
        .flat_map(|entry| entry.models.clone())
        .collect::<Vec<_>>();
    let refresh_needed = AVAILABLE_SERVICES.iter().any(|service| {
        cache
            .services
            .get(service.as_str())
            .is_none_or(|entry| now.saturating_sub(entry.updated_at) >= CACHE_TTL_SECS)
    });
    if models.is_empty() {
        models = refresh_all()?
            .services
            .into_values()
            .flat_map(|entry| entry.models)
            .collect();
    } else if refresh_needed {
        std::thread::spawn(|| {
            let _ = refresh_all();
        });
    }
    Ok(models)
}

pub(crate) fn verify_credentials(service: ApiService, api_key: &str) -> Result<usize, ApiError> {
    let client = NanoGptClient::new(api_key)
        .with_service(service)
        .with_base_url(resolve_base_url_for(service))
        .with_request_timeout(Duration::from_secs(10))
        .with_retry_policy(0, Duration::ZERO, Duration::ZERO);
    let runtime = tokio::runtime::Runtime::new().map_err(ApiError::Io)?;
    runtime
        .block_on(client.fetch_models(true))
        .map(|response| response.data.len())
}

pub(crate) fn refresh_service(service: ApiService) -> Result<usize, Box<dyn std::error::Error>> {
    let models = fetch_service_models(service)?;
    let count = models.len();
    let mut cache = load_cache().unwrap_or_default();
    cache.services.insert(
        service.as_str().to_string(),
        CachedCatalog {
            updated_at: epoch_secs(),
            models,
            last_error: None,
        },
    );
    save_cache(&cache)?;
    Ok(count)
}

pub(crate) fn health_label(service: ApiService) -> String {
    let cache = load_cache().unwrap_or_default();
    let Some(entry) = cache.services.get(service.as_str()) else {
        return "catalog not loaded".to_string();
    };
    if let Some(error) = &entry.last_error {
        let state = if entry.models.is_empty() {
            "catalog unavailable"
        } else {
            "cached catalog"
        };
        return format!("{state} · refresh failed: {}", truncate(error, 48));
    }
    let age = epoch_secs().saturating_sub(entry.updated_at);
    if age >= CACHE_TTL_SECS {
        "cached catalog · refreshing".to_string()
    } else {
        format!("catalog refreshed {age}s ago")
    }
}

fn refresh_all() -> Result<CatalogCache, Box<dyn std::error::Error>> {
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::scope(|scope| {
        for service in AVAILABLE_SERVICES {
            let sender = sender.clone();
            scope.spawn(move || {
                let result = fetch_service_models(service).map_err(|error| error.to_string());
                let _ = sender.send((service, result));
            });
        }
    });
    drop(sender);
    let results = receiver.into_iter().collect::<Vec<_>>();
    let mut cache = load_cache().unwrap_or_default();
    let now = epoch_secs();
    for (service, result) in results {
        let entry = cache
            .services
            .entry(service.as_str().to_string())
            .or_default();
        match result {
            Ok(models) => {
                entry.updated_at = now;
                entry.models = models;
                entry.last_error = None;
            }
            Err(error) => {
                entry.updated_at = 0;
                entry.last_error = Some(error);
            }
        }
    }
    save_cache(&cache)?;
    Ok(cache)
}

fn cache_path() -> Result<PathBuf, Box<dyn std::error::Error>> {
    Ok(config_home()?.join("model-catalogs.json"))
}

fn load_cache() -> Result<CatalogCache, Box<dyn std::error::Error>> {
    match fs::read_to_string(cache_path()?) {
        Ok(contents) => Ok(serde_json::from_str(&contents)?),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(CatalogCache::default()),
        Err(error) => Err(error.into()),
    }
}

fn save_cache(cache: &CatalogCache) -> Result<(), Box<dyn std::error::Error>> {
    let path = cache_path()?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    write_atomic(&path, serde_json::to_string_pretty(cache)?)?;
    Ok(())
}

fn epoch_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn config_home() -> Result<PathBuf, Box<dyn std::error::Error>> {
    pebble_config_home()
        .ok_or_else(|| "could not resolve PEBBLE_CONFIG_HOME, HOME, or USERPROFILE".into())
}

fn truncate(input: &str, max_chars: usize) -> String {
    let mut chars = input.chars();
    let output = chars.by_ref().take(max_chars).collect::<String>();
    if chars.next().is_some() {
        format!(
            "{}...",
            output
                .chars()
                .take(max_chars.saturating_sub(3))
                .collect::<String>()
        )
    } else {
        output
    }
}
