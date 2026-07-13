# Pebble

Pebble is a Rust agentic coding harness with a fast, transcript-first terminal interface.

It supports:

- Nano-GPT
- Neuralwatt
- Lilac
- Grok subscriptions through the official Grok CLI
- Synthetic
- OpenAI Codex / ChatGPT plans
- OpenCode Go

Pebble is designed around an interactive REPL, local tools, managed sessions, MCP servers, and a user-controlled permission model. Web retrieval is provider-agnostic and always runs through Exa.

```text
◆ pebble  v0.4.7
  gpt-5.5 on OpenAI Codex
  build · workspace
  ~/dev/my-project
  /help for commands · Tab switches mode

build ❯ find the flaky test and fix the root cause
◇ Working...
$ Shell  cargo test
  └ ok: 0
◆ Pebble
The retry assertion was sharing state across test cases. I isolated the fixture
and added a regression test.
✓ Done in 8.4s · 3 tools · 2 files changed · 1.8k tokens · 4% context
```


Check the [Changelog](CHANGELOG.md) for update/patch notes


## First-run setup

### 1. Authenticate a model service

Pebble can prompt you for credentials interactively:

```bash
pebble login
```

You can also target a specific service directly:

```bash
pebble login synthetic
pebble login neuralwatt
pebble login lilac
pebble login grok
pebble login openai-codex
pebble login opencode-go
pebble login nanogpt
```

For API-key services, you can also pass the key inline:

```bash
pebble login opencode-go --api-key "$OPENCODE_GO_API_KEY"
pebble login neuralwatt --api-key "$NEURALWATT_API_KEY"
pebble login lilac --api-key "$LILAC_API_KEY"
```

`openai-codex` uses ChatGPT device-code auth instead of an API key. `grok`
launches `grok login` and uses the official Grok CLI's refreshable OAuth
session, so Pebble never copies or stores xAI subscription tokens. Install the
[official Grok CLI](https://docs.x.ai/build/overview) before connecting it.
Pebble talks to that CLI over its ACP stdin/stdout protocol; prompts are never
placed in process arguments. Grok replies render as ACP chunks arrive, and
`Ctrl+C` cancels the entire active turn: HTTP streams, Grok ACP sessions, MCP
calls, hooks, and foreground shell/REPL/PowerShell tools all receive the same
cancellation signal and child processes are terminated. API-key logins are
checked before invalid credentials are saved when the provider exposes a model
catalog.

Pebble automatically disables ANSI color when output is redirected, when
`NO_COLOR` is set, with `CLICOLOR=0`, or under `TERM=dumb`. Set
`CLICOLOR_FORCE=1` to keep color in redirected output.

Inside the REPL, the equivalent commands are:

```text
/login
/auth
/login openai-codex
/login grok
/login opencode-go
/auth synthetic
/logout openai-codex
```

If you run `/login` or `/auth` without a service, Pebble opens a picker with:

- `nanogpt`
- `neuralwatt`
- `lilac`
- `grok`
- `synthetic`
- `openai-codex`
- `opencode-go`
- `exa`

### 2. Save your Exa key

Pebble uses Exa for all web search and scrape functionality.

Save it with:

```bash
pebble login exa
```

Or from inside the REPL:

```text
/login exa
/auth exa
```

You can also provide it inline:

```bash
pebble login exa --api-key "$EXA_API_KEY"
```

Or export it in your shell:

```bash
export EXA_API_KEY=...
```

## Daily usage

### Choose a provider and model

Run `/model` to open the unified picker. Left and right switch providers,
typing searches every provider at once, up and down move, and Enter selects.
The provider rail shows which services are connected. Signed-out providers
remain visible with the exact `/login` command they need, and a failed provider
catalog no longer prevents other providers from loading.

Provider catalogs are cached locally for instant startup and refreshed in the
background after five minutes. The provider rail distinguishes fresh, cached,
refreshing, and failed catalogs. Grok's model list comes from the installed
official CLI, so it follows the user's subscription and CLI version.

Neuralwatt and Lilac use their OpenAI-compatible chat-completions APIs. Model
IDs are namespaced in Pebble, such as `neuralwatt/zai-org/glm-5.2` and
`lilac/moonshotai/kimi-k2.6`, so the same upstream model can be selected from
different providers without ambiguity. Both providers stream text, reasoning,
tool calls, and usage natively; image attachments are sent as OpenAI-compatible
image inputs when the selected model supports vision.

### Interactive REPL

Launch the REPL:

```bash
pebble
```

Useful first commands:

```text
/help
/status
/model
/login
/logout
/sessions
```

Basic prompt flow:

```text
> summarize this project
> inspect Cargo.toml and explain the workspace layout
> find the session restore logic
```

### One-shot prompt mode

For a single command without entering the REPL:

```bash
pebble prompt "Summarize this repository"
```

Or:

```bash
pebble "Inspect the current Rust workspace and explain the top-level crates"
```

### Restrict tool access

```bash
pebble --allowedTools read,glob "Summarize Cargo.toml"
```

### Safety limits

Pebble stops runaway turns after 32 model passes. Foreground shell, REPL, and
PowerShell commands default to a 10-minute timeout, and model API requests also
time out after 10 minutes. Override these limits when a project needs more time:

```bash
export PEBBLE_MAX_TURN_ITERATIONS=48
export PEBBLE_BASH_TIMEOUT_MS=900000
export PEBBLE_TOOL_TIMEOUT_MS=900000
export PEBBLE_API_TIMEOUT_SECS=900
```

An explicit tool-call timeout still takes precedence over the default.

### Eval suites and traces

Validate a suite without calling a model:

```bash
pebble eval --check evals/smoke.json
```

Run a suite and fail the process if any case fails:

```bash
pebble eval --fail-on-failures evals/smoke.json
```

Show recent eval trends:

```bash
pebble eval history
pebble eval history --suite smoke --model zai-org/glm-5.1
```

Eval history is rebuilt from `.pebble/evals/*.json` and persisted to
`.pebble/evals/index.json` after each run.

Replay failed eval cases from a saved report:

```bash
pebble eval replay .pebble/evals/<report>.json
pebble eval replay .pebble/evals/<report>.json --case handles-denied-write
```

Eval replay loads each case's saved trace and shows assertion failures, failure
categories, final answer preview, artifacts, and the trace timeline.

Trace, replay, eval history, eval compare, and eval replay support `--json` for
machine-readable diagnostics:

```bash
pebble trace .pebble/runs/<trace>.json --json
pebble eval replay .pebble/evals/<report>.json --json
```

Promote a saved trace into a regression eval:

```bash
pebble eval capture .pebble/runs/<trace>.json --suite evals/regressions.json --name "handles denied write"
```

Captured cases use the trace input preview as the prompt and generate
assertions for required tools, tool order, permission outcomes, API/tool call
limits, and successful tool usage when present. Existing case IDs are protected
unless `--force` is passed.

Inspect a saved turn trace:

```bash
pebble trace .pebble/runs/<trace>.json
```

Replay a saved trace timeline without calling the model or tools:

```bash
pebble replay .pebble/runs/<trace>.json
```

Golden trace regressions protect the trace/replay renderers, JSON reports,
context-window percentage display, compact tool previews, and MCP tool spec
projection:

```bash
cargo test -p pebble golden
PEBBLE_UPDATE_GOLDENS=1 cargo test -p pebble golden
```

Use `PEBBLE_UPDATE_GOLDENS=1` only when the output change is intentional.

Trace previews are redacted before persistence and again when loading older
trace files. Common API keys, bearer tokens, passwords, private keys, and
credential-bearing URLs are replaced with `[REDACTED]` markers.

Trace and eval report JSON files include a `schema_version` field. Files
written before schema versioning are treated as version 1 when loaded; newly
written files use the current schema version.

Prune generated trace and eval artifacts:

```bash
pebble gc --dry-run
pebble gc
```

Retention defaults keep trace JSON files for 30 days or the newest 1000 files,
eval reports for 90 days or the newest 200 reports, and CI check reports for 30
days or the newest 100 reports. Override them in `.pebble/settings.json`:

```json
{
  "retention": {
    "traceDays": 14,
    "maxTraceFiles": 500,
    "evalDays": 60,
    "maxEvalReports": 100,
    "ciDays": 14,
    "maxCiReports": 50
  }
}
```

Validate settings without starting a REPL:

```bash
pebble config check
pebble config check --json
```

Config checks report malformed JSON, non-object settings files, bad field
types, unsupported option values, and the settings file/field path responsible
when Pebble can infer it.

Collect a redacted local support snapshot:

```bash
pebble doctor bundle
```

Probe every configured model provider without sending a model prompt or
incurring inference usage:

```bash
pebble doctor providers
pebble doctor providers --json
```

The provider report checks authentication and the live model catalog, records
latency and model count, and states Pebble's streaming, tool-call, and vision
transport support without printing credentials.

The diagnostics bundle is written under `.pebble/diagnostics/` and includes
offline doctor checks, config validation, local system metadata, session
metadata, recent trace/eval summaries, and MCP discovery status. It excludes
API keys, credentials, raw config contents, full prompts, assistant responses,
tool inputs, tool outputs, and live API/network probes.

## Core REPL commands

Common commands:

- `/help`
- `/help auth`
- `/help sessions`
- `/help extensions`
- `/help web`
- `/status`
- `/model`
- `/login`
- `/logout`
- `/provider`
- `/route`
- `/permissions`
- `/bypass`
- `/proxy`
- `/mcp`
- `/skills`
- `/plugins`
- `/sessions`
- `/resume`
- `/resume last`
- `/session switch <id>`

Notes:

- `/provider [name]` opens model selection, optionally focused on one provider.
- `/route` controls NanoGPT's upstream routing override and only applies to NanoGPT-backed models.
- `Shift+Enter` and `Ctrl+J` insert a newline in the input editor.

## Authentication and config

Pebble stores user config under:

```text
~/.pebble/
```

Credentials are stored in:

```text
~/.pebble/credentials.json
```

Possible stored keys:

- `nanogpt_api_key`
- `synthetic_api_key`
- `openai_codex_auth`
- `opencode_go_api_key`
- `exa_api_key`

Environment variables still take precedence over saved credentials.

Useful environment variables:

- `NANOGPT_API_KEY`
- `SYNTHETIC_API_KEY`
- `OPENAI_CODEX_ACCESS_TOKEN`
- `OPENAI_CODEX_REFRESH_TOKEN`
- `OPENAI_CODEX_ACCOUNT_ID`
- `OPENAI_CODEX_EXPIRES_AT`
- `OPENCODE_GO_API_KEY`
- `EXA_API_KEY`
- `NANOGPT_BASE_URL`
- `SYNTHETIC_BASE_URL`
- `OPENAI_CODEX_BASE_URL`
- `OPENCODE_GO_BASE_URL`
- `EXA_BASE_URL`
- `PEBBLE_CONFIG_HOME`

`EXA_BASE_URL` defaults to `https://api.exa.ai`.
`OPENAI_CODEX_BASE_URL` defaults to `https://chatgpt.com/backend-api/codex`.

## Sessions and restore

Pebble keeps managed sessions under:

```text
.pebble/sessions/
```

Useful flows:

- `/sessions` lists recent sessions
- `/resume` opens the picker
- `/resume last` restores the most recently modified session
- `/session switch <session-id>` switches inside the REPL
- `pebble resume [SESSION_ID_OR_PATH]` resumes from the CLI

Session restore includes more than just transcript history. Pebble persists and restores:

- active model
- permission mode
- thinking toggle
- proxy tool-call toggle
- allowed tool set

That makes restored sessions behave much closer to the original live session.

## Permissions

Pebble supports:

- `read-only`
- `workspace-write`
- `danger-full-access`

Examples:

```text
/permissions
/permissions workspace-write
/bypass
```

`/bypass` is a shortcut for `danger-full-access` in the current session.

## Web search and scrape

Pebble keeps the tool names `WebSearch` and `WebScrape`, but both use Exa.

### WebSearch

- uses Exa `POST /search`
- defaults to Exa search type `auto`
- promotes to `deep` for deeper or more structured requests
- maps allowed and blocked domains into Exa domain filters

### WebScrape

- uses Exa `POST /contents`
- supports one or more URLs
- validates URLs before sending requests
- returns normalized previews in the TUI

### Check readiness

Run:

```text
/status
```

Pebble reports Exa readiness separately from the active model backend.

## Extensions

Pebble has three main extension surfaces:

- skills
- MCP servers
- plugins

### Skills

Create a project-local skill:

```text
/skills init my-skill
```

This creates:

```text
.pebble/skills/my-skill/SKILL.md
```

Useful commands:

```text
/skills
/skills help
```

### MCP servers

Create a starter MCP server entry:

```text
/mcp add my-server
```

This updates:

```text
.pebble/settings.json
```

Inspect what is configured:

```text
/mcp
/mcp tools
/mcp reload
```

Enable or disable a configured server locally:

```text
/mcp disable context7
/mcp enable context7
```

These local toggles are written to:

```text
.pebble/settings.local.json
```

That lets you keep a shared project MCP config while turning specific servers on or off per machine.

### Plugins

Useful commands:

```text
/plugins
/plugins help
/plugins install ./plugins/my-plugin
/plugins enable my-plugin-id
```

Pebble expects plugins to expose:

```text
.pebble-plugin/plugin.json
```

## Proxy mode

Pebble can run in XML proxy tool-call mode:

```text
/proxy status
/proxy on
/proxy off
```

When proxy mode is enabled, tool use is expected through XML `<tool_call>` blocks rather than native tool schemas.

## Troubleshooting

### A model won’t answer

- run `/status`
- confirm you saved credentials with `pebble login`
- or export the matching `*_API_KEY`
- verify the active model with `/model`

### Web tools are unavailable

- run `pebble login exa`
- or export `EXA_API_KEY`
- check `/status` for Exa readiness

### MCP server loads but shows no tools

- run `/mcp`
- run `/mcp tools`
- run `/mcp reload`
- check `.pebble/settings.json`
- check `.pebble/settings.local.json`
- if the server is marked `disabled`, run `/mcp enable <name>`

### Session restore feels wrong

- inspect `/status`
- use `/resume last` or `/session switch <id>`
- verify the session was saved after changing model, permissions, proxy, or thinking state

### Plugin setup is unclear

- run `/plugins help`
- confirm the plugin root contains `.pebble-plugin/plugin.json`

## Install, build, and development

### Build a release binary

```bash
cargo build --release -p pebble
```

Binary output:

```bash
./target/release/pebble
```

On Windows, the binary output is `target\\release\\pebble.exe`. Pebble resolves config and
credentials from `PEBBLE_CONFIG_HOME` first, then `%USERPROFILE%\\.pebble` when `HOME` is not set.

### Run from source

```bash
cargo run -p pebble --
```

### Run tests

```bash
cargo test --workspace -- --test-threads=1
cargo build -p pebble
python3 scripts/pty_smoke.py
```

The PTY suite drives the compiled binary through onboarding, input interrupts,
login and model pickers, narrow-terminal resizing, Unicode search, model and
foreground-tool cancellation, plain redirected output, and clean exit. It runs
on Linux and macOS with isolated temporary configuration and no provider
credentials; CI also runs a Windows redirected-console smoke check.

CI also runs the agent-harness safety checks that protect user-visible
diagnostics and regression tooling:

```bash
cargo run -p pebble -- ci check
cargo run -p pebble -- ci check --json
cargo run -p pebble -- ci check --json --save-report
cargo run -p pebble -- ci history
cargo run -p pebble -- ci history --json --limit 20
cargo run -p pebble -- release check
cargo run -p pebble -- release check --json --save-report
```

That shared entrypoint runs golden trace regressions, config schema validation,
eval suite validation, and a diagnostics bundle redaction-contract check.
`--json` emits step status, durations, and the diagnostics bundle path for
tooling. `--save-report` writes the final JSON report under `.pebble/ci/` and
includes the report path in the output. `ci history` summarizes saved reports
without rerunning the checks. Captured CI failures include a per-step artifact
path with stdout/stderr logs when available.

`release check` is the ship-readiness rollup. It summarizes the current git
branch/commit/dirty state, Pebble version, latest saved CI report, latest eval
history entry, config validation status, golden trace regression status,
diagnostics redaction status, and paths to saved reports/bundles/artifacts.
Use `--save-report` to write the JSON rollup under `.pebble/release/`.

### Project config files

Common project-local files:

- `PEBBLE.md`
- `.pebble/settings.json`
- `.pebble/settings.local.json`
- `.pebble/skills/`
- `.pebble/sessions/`

### Release/update behavior

Pebble’s self-update flow targets the GitHub releases for:

```text
nanogpt-community/pebble
```
