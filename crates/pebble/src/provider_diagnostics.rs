use std::time::Instant;

use api::{resolve_api_key_for, ApiService};
use serde::Serialize;

use crate::model_catalog::verify_credentials;
use crate::models::{fetch_service_models, AVAILABLE_SERVICES};

#[derive(Debug, Serialize)]
struct ProviderReport {
    provider: &'static str,
    auth: &'static str,
    status: &'static str,
    model_count: Option<usize>,
    latency_ms: Option<u128>,
    transport: TransportContract,
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct TransportContract {
    streaming: &'static str,
    tool_calls: &'static str,
    vision: &'static str,
}

pub(crate) fn run(json: bool) -> Result<(), Box<dyn std::error::Error>> {
    let reports = AVAILABLE_SERVICES
        .iter()
        .copied()
        .map(check_provider)
        .collect::<Vec<_>>();

    if json {
        println!("{}", serde_json::to_string_pretty(&reports)?);
    } else {
        println!("Provider diagnostics");
        println!("  Live catalog/auth probes only; no prompts are sent and no usage is billed.\n");
        for report in &reports {
            let models = report
                .model_count
                .map_or_else(|| "-".to_string(), |count| count.to_string());
            let latency = report
                .latency_ms
                .map_or_else(|| "-".to_string(), |value| format!("{value} ms"));
            println!(
                "  {:<15} {:<15} {:<8} models {:<6} {}",
                report.provider, report.auth, report.status, models, latency
            );
            println!(
                "    stream {} · tools {} · vision {}",
                report.transport.streaming, report.transport.tool_calls, report.transport.vision
            );
            if let Some(error) = &report.error {
                println!("    {error}");
            }
        }
    }

    if reports.iter().any(|report| report.status == "error") {
        return Err("one or more configured providers failed diagnostics".into());
    }
    Ok(())
}

fn check_provider(service: ApiService) -> ProviderReport {
    let transport = TransportContract {
        streaming: "implemented",
        tool_calls: "implemented",
        vision: if service == ApiService::Grok {
            "not supported"
        } else {
            "implemented"
        },
    };
    let started = Instant::now();
    let result = match service {
        ApiService::Grok => fetch_service_models(service).map(|models| models.len()),
        ApiService::Synthetic => match resolve_api_key_for(service) {
            Ok(_) => fetch_service_models(service).map(|models| models.len()),
            Err(_) => {
                return ProviderReport {
                    provider: service.display_name(),
                    auth: auth_method(service),
                    status: "not configured",
                    model_count: None,
                    latency_ms: None,
                    transport,
                    error: Some(sign_in_hint(service)),
                };
            }
        },
        _ => match resolve_api_key_for(service) {
            Ok(api_key) => {
                verify_credentials(service, &api_key).map_err(Box::<dyn std::error::Error>::from)
            }
            Err(_error) => {
                return ProviderReport {
                    provider: service.display_name(),
                    auth: auth_method(service),
                    status: "not configured",
                    model_count: None,
                    latency_ms: None,
                    transport,
                    error: Some(sign_in_hint(service)),
                };
            }
        },
    };

    match result {
        Ok(model_count) => ProviderReport {
            provider: service.display_name(),
            auth: auth_method(service),
            status: "ready",
            model_count: Some(model_count),
            latency_ms: Some(started.elapsed().as_millis()),
            transport,
            error: None,
        },
        Err(error) => {
            let message = error.to_string();
            let not_configured = missing_credentials(service, &message);
            ProviderReport {
                provider: service.display_name(),
                auth: auth_method(service),
                status: if not_configured {
                    "not configured"
                } else {
                    "error"
                },
                model_count: None,
                latency_ms: Some(started.elapsed().as_millis()),
                transport,
                error: Some(if not_configured {
                    sign_in_hint(service)
                } else {
                    truncate(&message, 180)
                }),
            }
        }
    }
}

const fn auth_method(service: ApiService) -> &'static str {
    if matches!(service, ApiService::Grok) {
        "OAuth/sub"
    } else if matches!(service, ApiService::OpenAiCodex) {
        "device auth"
    } else {
        "API key"
    }
}

fn sign_in_hint(service: ApiService) -> String {
    format!("run `pebble login {}`", service.as_str().replace('_', "-"))
}

fn missing_credentials(service: ApiService, message: &str) -> bool {
    (service == ApiService::Grok
        && (message.contains("No such file")
            || message.contains("not found")
            || message.contains("run `grok login`")
            || message.contains("run `pebble login`")))
        || message.contains("API key")
        || message.contains("api key")
        || message.contains("credentials")
        || message.contains("signed in")
}

fn truncate(value: &str, max_chars: usize) -> String {
    let mut chars = value.chars();
    let prefix = chars.by_ref().take(max_chars).collect::<String>();
    if chars.next().is_some() {
        format!(
            "{}...",
            prefix
                .chars()
                .take(max_chars.saturating_sub(3))
                .collect::<String>()
        )
    } else {
        prefix
    }
}

#[cfg(test)]
mod tests {
    use super::{missing_credentials, truncate};
    use api::ApiService;

    #[test]
    fn classifies_auth_failures_without_exposing_credentials() {
        assert!(missing_credentials(ApiService::Grok, "run `grok login`"));
        assert!(missing_credentials(ApiService::Lilac, "missing API key"));
        assert!(!missing_credentials(ApiService::Lilac, "HTTP 503"));
    }

    #[test]
    fn truncates_provider_errors() {
        assert_eq!(truncate("short", 8), "short");
        assert_eq!(truncate("abcdefghijk", 8), "abcde...");
    }
}
