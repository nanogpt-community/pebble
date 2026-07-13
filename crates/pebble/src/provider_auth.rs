use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::Command;

use api::ApiService;
use platform::{pebble_config_home, write_atomic};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AuthService {
    NanoGpt,
    Neuralwatt,
    Lilac,
    Grok,
    Synthetic,
    OpenAiCodex,
    OpencodeGo,
    Exa,
}

impl AuthService {
    pub(crate) const fn display_name(self) -> &'static str {
        match self {
            Self::NanoGpt => "NanoGPT",
            Self::Neuralwatt => "Neuralwatt",
            Self::Lilac => "Lilac",
            Self::Grok => "Grok",
            Self::Synthetic => "Synthetic",
            Self::OpenAiCodex => "OpenAI Codex",
            Self::OpencodeGo => "OpenCode Go",
            Self::Exa => "Exa",
        }
    }

    pub(crate) const fn slug(self) -> &'static str {
        match self {
            Self::NanoGpt => "nanogpt",
            Self::Neuralwatt => "neuralwatt",
            Self::Lilac => "lilac",
            Self::Grok => "grok",
            Self::Synthetic => "synthetic",
            Self::OpenAiCodex => "openai-codex",
            Self::OpencodeGo => "opencode-go",
            Self::Exa => "exa",
        }
    }

    const fn credential_key(self) -> &'static str {
        match self {
            Self::NanoGpt => "nanogpt_api_key",
            Self::Neuralwatt => "neuralwatt_api_key",
            Self::Lilac => "lilac_api_key",
            Self::Grok => "grok_oauth",
            Self::Synthetic => "synthetic_api_key",
            Self::OpenAiCodex => "openai_codex_auth",
            Self::OpencodeGo => "opencode_go_api_key",
            Self::Exa => "exa_api_key",
        }
    }

    pub(crate) const fn all() -> &'static [Self] {
        &[
            Self::NanoGpt,
            Self::Neuralwatt,
            Self::Lilac,
            Self::Grok,
            Self::Synthetic,
            Self::OpenAiCodex,
            Self::OpencodeGo,
            Self::Exa,
        ]
    }

    pub(crate) const fn runtime_service(self) -> Option<ApiService> {
        match self {
            Self::NanoGpt => Some(ApiService::NanoGpt),
            Self::Neuralwatt => Some(ApiService::Neuralwatt),
            Self::Lilac => Some(ApiService::Lilac),
            Self::Grok => Some(ApiService::Grok),
            Self::Synthetic => Some(ApiService::Synthetic),
            Self::OpenAiCodex => Some(ApiService::OpenAiCodex),
            Self::OpencodeGo => Some(ApiService::OpencodeGo),
            Self::Exa => None,
        }
    }

    pub(crate) const fn auth_method(self) -> &'static str {
        match self {
            Self::OpenAiCodex => "device code",
            Self::Grok => "OAuth subscription",
            _ => "API key",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LoginCommand {
    pub(crate) service: Option<AuthService>,
    pub(crate) api_key: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LogoutCommand {
    pub(crate) service: Option<AuthService>,
}

pub(crate) fn parse_login_tokens(tokens: &[&str]) -> Result<LoginCommand, String> {
    let mut service = None;
    let mut api_key = None;
    let mut index = 0;
    while index < tokens.len() {
        match tokens[index] {
            "--service" => {
                let value = tokens
                    .get(index + 1)
                    .ok_or_else(|| "missing value for --service".to_string())?;
                service = Some(parse_login_service(value)?);
                index += 2;
            }
            flag if flag.starts_with("--service=") => {
                service = Some(parse_login_service(&flag[10..])?);
                index += 1;
            }
            "--api-key" => {
                let value = tokens
                    .get(index + 1)
                    .ok_or_else(|| "missing value for --api-key".to_string())?;
                api_key = Some((*value).to_string());
                index += 2;
            }
            flag if flag.starts_with("--api-key=") => {
                api_key = Some(flag[10..].to_string());
                index += 1;
            }
            value if is_service_alias(value) && api_key.is_none() => {
                service = Some(parse_login_service(value)?);
                index += 1;
            }
            value if api_key.is_none() => {
                api_key = Some(value.to_string());
                index += 1;
            }
            other => return Err(format!("unexpected login argument: {other}")),
        }
    }
    Ok(LoginCommand { service, api_key })
}

pub(crate) fn parse_logout_tokens(tokens: &[&str]) -> Result<LogoutCommand, String> {
    let mut service = None;
    let mut index = 0;
    while index < tokens.len() {
        match tokens[index] {
            "--service" => {
                let value = tokens
                    .get(index + 1)
                    .ok_or_else(|| "missing value for --service".to_string())?;
                service = Some(parse_login_service(value)?);
                index += 2;
            }
            flag if flag.starts_with("--service=") => {
                service = Some(parse_login_service(&flag[10..])?);
                index += 1;
            }
            value if is_service_alias(value) => {
                service = Some(parse_login_service(value)?);
                index += 1;
            }
            other => return Err(format!("unexpected logout argument: {other}")),
        }
    }
    Ok(LogoutCommand { service })
}

pub(crate) fn parse_auth_command(input: &str) -> Option<LoginCommand> {
    let mut parts = input.split_whitespace();
    let command = parts.next()?;
    if command != "/login" && command != "/auth" {
        return None;
    }
    let tokens = parts.collect::<Vec<_>>();
    parse_login_tokens(&tokens).ok()
}

pub(crate) fn parse_logout_command(input: &str) -> Option<LogoutCommand> {
    let mut parts = input.split_whitespace();
    if parts.next()? != "/logout" {
        return None;
    }
    let tokens = parts.collect::<Vec<_>>();
    parse_logout_tokens(&tokens).ok()
}

fn parse_login_service(value: &str) -> Result<AuthService, String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "nanogpt" | "nano-gpt" | "nano" => Ok(AuthService::NanoGpt),
        "neuralwatt" | "neural-watt" => Ok(AuthService::Neuralwatt),
        "lilac" | "getlilac" => Ok(AuthService::Lilac),
        "grok" | "xai" | "x-ai" => Ok(AuthService::Grok),
        "synthetic" | "synthetic.new" => Ok(AuthService::Synthetic),
        "openai-codex" | "openai_codex" | "chatgpt" => Ok(AuthService::OpenAiCodex),
        "opencode-go" | "opencodego" => Ok(AuthService::OpencodeGo),
        "exa" => Ok(AuthService::Exa),
        other => Err(format!(
            "unsupported login service `{other}`; expected nanogpt, neuralwatt, lilac, grok, synthetic, openai-codex, opencode-go, or exa"
        )),
    }
}

fn is_service_alias(value: &str) -> bool {
    parse_login_service(value).is_ok()
}

pub(crate) fn prompt_for_auth_service_selection(
) -> Result<Option<AuthService>, Box<dyn std::error::Error>> {
    println!("Connect a provider");
    for (index, service) in AuthService::all().iter().enumerate() {
        println!(
            "  {:>2}. {:<14} {:<18} {}",
            index + 1,
            service.display_name(),
            service.auth_method(),
            service.slug()
        );
    }
    println!();
    print!(
        "Choose 1-{} or press Enter to cancel: ",
        AuthService::all().len()
    );
    io::stdout().flush()?;
    let mut buffer = String::new();
    io::stdin().read_line(&mut buffer)?;
    let selection = buffer.trim();
    if selection.is_empty() {
        return Ok(None);
    }
    let index = selection
        .parse::<usize>()
        .map_err(|_| format!("invalid selection: {selection}"))?;
    let Some(service) = AuthService::all().get(index.saturating_sub(1)) else {
        return Err(format!("selection out of range: {selection}").into());
    };
    Ok(Some(*service))
}

pub(crate) fn save_credentials(
    service: AuthService,
    api_key: &str,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let config_home = config_home()?;
    fs::create_dir_all(&config_home)?;
    let credentials_path = config_home.join("credentials.json");
    let mut parsed = match fs::read_to_string(&credentials_path) {
        Ok(contents) => serde_json::from_str::<serde_json::Value>(&contents)
            .unwrap_or_else(|_| serde_json::json!({})),
        Err(error) if error.kind() == io::ErrorKind::NotFound => serde_json::json!({}),
        Err(error) => return Err(Box::new(error)),
    };
    if !parsed.is_object() {
        parsed = serde_json::json!({});
    }
    parsed[service.credential_key()] = serde_json::Value::String(api_key.to_string());
    write_atomic(&credentials_path, serde_json::to_string_pretty(&parsed)?)?;
    secure_credentials_file(&credentials_path)?;
    Ok(credentials_path)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CredentialRemovalOutcome {
    Removed { path: PathBuf },
    Missing { path: PathBuf },
}

pub(crate) fn remove_saved_credentials(
    service: AuthService,
) -> Result<CredentialRemovalOutcome, Box<dyn std::error::Error>> {
    let credentials_path = config_home()?.join("credentials.json");
    let contents = match fs::read_to_string(&credentials_path) {
        Ok(contents) => contents,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            return Ok(CredentialRemovalOutcome::Missing {
                path: credentials_path,
            });
        }
        Err(error) => return Err(Box::new(error)),
    };
    let mut parsed = serde_json::from_str::<serde_json::Value>(&contents)
        .unwrap_or_else(|_| serde_json::json!({}));
    if !parsed.is_object() {
        parsed = serde_json::json!({});
    }
    let removed = parsed
        .as_object_mut()
        .is_some_and(|object| object.remove(service.credential_key()).is_some());
    if !removed {
        return Ok(CredentialRemovalOutcome::Missing {
            path: credentials_path,
        });
    }
    write_atomic(&credentials_path, serde_json::to_string_pretty(&parsed)?)?;
    secure_credentials_file(&credentials_path)?;
    Ok(CredentialRemovalOutcome::Removed {
        path: credentials_path,
    })
}

pub(crate) fn run_grok_auth_command(action: &str) -> Result<(), Box<dyn std::error::Error>> {
    let executable = std::env::var("PEBBLE_GROK_CLI").unwrap_or_else(|_| "grok".to_string());
    let status = Command::new(executable)
        .arg(action)
        .status()
        .map_err(|error| {
            io::Error::new(
                error.kind(),
                format!("could not launch the official Grok CLI; install it from https://x.ai/cli first: {error}"),
            )
        })?;
    if !status.success() {
        return Err(format!("`grok {action}` exited with {status}").into());
    }
    Ok(())
}

fn config_home() -> Result<PathBuf, Box<dyn std::error::Error>> {
    pebble_config_home()
        .ok_or_else(|| "could not resolve PEBBLE_CONFIG_HOME, HOME, or USERPROFILE".into())
}

fn secure_credentials_file(path: &std::path::Path) -> io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o600))?;
    }
    Ok(())
}
