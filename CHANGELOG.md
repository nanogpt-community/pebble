# CHANGELOG (STARTING FROM v0.4.0)

## v0.5.0

- Add Neuralwatt and Lilac as API-key providers using their OpenAI-compatible catalogs and chat-completions APIs.
- Add Grok subscription access through the official Grok CLI and its OAuth session, without storing or copying xAI subscription tokens in Pebble.
- Speak xAI's ACP JSON-RPC protocol over the Grok CLI's stdin/stdout transport, keeping prompts out of process arguments and discovering available Grok models from the installed CLI.
- Render Grok ACP text as it arrives, let `Ctrl+C` cancel a running Grok turn and kill its CLI child, and keep XML tool blocks out of the transcript while streaming.
- Stream Neuralwatt and Lilac responses natively over OpenAI-compatible SSE, including reasoning, tool calls, usage, and image inputs.
- Verify API keys during login, cache provider catalogs for instant picker startup, refresh stale catalogs in the background, and surface catalog health in the provider rail.
- Rebuild model selection around a connected-provider rail, global type-to-search, clearer model details, parallel fault-tolerant catalog loading, and setup guidance for signed-out providers.
- Make `/provider [name]` select an actual model provider and move NanoGPT-specific upstream overrides to the clearer `/route` command.
- Show active model authentication readiness in `/status`, including the exact login command when credentials are missing.
- Show missing model authentication on the first-run banner with the exact login command, without the internal `<platform default>` routing label.
- Add a real PTY smoke suite for startup, input cancellation, login cancellation, narrow-terminal resizing, Unicode model search, picker cancellation, Grok streaming interrupts, and clean exit; run it in CI.
- Unify `Ctrl+C` cancellation across HTTP model streams, Grok ACP, MCP calls, hooks, and foreground shell, REPL, PowerShell, and sleep tools, with typed cancellation errors and child-process cleanup.
- Respect redirected output, `NO_COLOR`, `CLICOLOR`, `CLICOLOR_FORCE`, and `TERM=dumb`; exercise the compiled terminal on Linux and macOS plus redirected-console behavior on Windows CI.
- Add `pebble doctor providers [--json]` for credential-safe live auth/catalog probes, model counts, latency, and transport capability reporting without sending inference prompts.
- Split Grok ACP transport, provider authentication and credential storage, and model-catalog caching into focused modules outside the REPL and picker implementations.
- Fix the model picker's advertised `q` shortcut so it cancels when the search field is empty instead of inserting `q` into the query.
- Fix a Unicode byte-boundary panic that could crash the model picker while truncating provider status markers on narrower terminals.
- Redesign the interactive REPL around a quiet transcript, mode-aware prompt, compact startup identity, readable tool activity, concise help and status views, and consistent command feedback.
- Keep invalid commands and failed model turns inside the REPL instead of ending the process.
- Hide raw model reasoning from the transcript while retaining it in session state and traces.
- Persist REPL history across launches without saving inline login credentials.
- Make `Ctrl+C` cancel the current input and reserve `Ctrl+D`, `/exit`, and `/quit` for leaving the REPL.
- Stop runaway agent turns after 32 model passes by default, with an environment override for longer tasks.
- Add default timeouts for model API requests and foreground shell, REPL, and PowerShell tools so a missing timeout cannot hang Pebble forever.
- Roll back failed requests that never produced an assistant response, while preserving partial tool turns and their undo snapshots.
- Stop injecting legacy `summary-*.md` conversation dumps as global project memory, and give durable memory files a separate 5,000-character prompt budget.
- Reject unknown CLI options instead of accidentally sending them to the model as a paid prompt, and exit cleanly when piped output closes early.
- Build agent context with the real local date instead of the old hardcoded March 2026 value.
- Treat an explicit search path of `.` as a normal workspace-wide search, keeping `.gitignore`, hidden-directory, build-output, and Pebble-state exclusions enabled.

## v0.4.7

- Add structured turn tracing with redacted API call, tool call, permission, compaction, usage, and context-window metadata, plus trace/replay commands with text and JSON output.
- Add an eval harness with suite validation, run reports, history indexing, report comparison, failed-case replay, and trace capture into regression suites.
- Add golden trace regression fixtures for trace rendering, replay output, MCP tool specs, tool previews, and context-window percentage output.
- Add `pebble ci check` and `pebble ci history` to run local harness safety checks for golden regressions, config validation, eval suite validation, and diagnostics bundle redaction; CI failures can now include per-step stdout/stderr artifacts.
- Add `pebble release check` as a ship-readiness rollup over git status, Pebble version, latest saved CI/eval reports, config status, golden trace status, diagnostics redaction status, and saved report/artifact paths.
- Add redacted diagnostics bundles under `.pebble/diagnostics/` with doctor, config, system, session, trace, eval, and MCP status summaries.
- Add `pebble config check` and `/config check` to validate settings files, map schema/shape errors back to source files, and support JSON output.
- Add retention settings and garbage collection for generated trace, eval, and CI artifacts.
- Split the Pebble CLI implementation into focused modules for MCP handling, runtime API streaming, session storage, tool rendering, trace viewing, eval execution, and report formatting.
- Expand README and GitHub Actions coverage for the new diagnostics, eval, CI, release, trace, replay, and retention workflows.

## v0.4.4

- Fix CI instability in tools tests by resolving non-Windows bash commands through `/bin/sh` when available, avoiding sensitivity to tests that temporarily mutate `PATH`.
- Recover from poisoned test environment locks in the local skill-loading test so later tests can continue after prior panics.
- Address clippy's needless pass-by-value warning in runtime file operation tests.

## v0.4.3

- Clarify timeout units in built-in `bash`, `REPL`, and `PowerShell` tool schemas and descriptions so model-facing docs explicitly state milliseconds.
- Enforce `REPL.timeout_ms` for Python, JavaScript/Node, and shell snippets, returning structured output with `timedOut: true` and `Command exceeded timeout of {timeout_ms} ms` on timeout.
- Harden `grep_search.output_mode` handling by validating supported modes (`files_with_matches`, `content`, and `count`) and returning a clear invalid-input error for unknown values.
- Update `grep_search` content-mode pagination so `head_limit` and `offset` apply to returned content lines, with `filenames` and `numFiles` derived from the displayed lines.
- Add targeted tests for REPL timeout enforcement and grep output-mode/pagination semantics.

## v0.4.2

- Add repo-aware defaults for native search tools: broad `grep_search` and `glob_search` now skip hidden/project-state directories, respect `.gitignore`, and avoid `.git`, `target`, `.pebble/sessions`, `.pebble/tool-results`, `.pebble/agents`, `.sandbox-home`, and `.sandbox-tmp` unless explicitly targeted.
- Add workspace-bound path enforcement for `read_file`, `write_file`, `edit_file`, `glob_search`, `grep_search`, and `apply_patch` targets, including lexical and symlink escape checks with clear `path escapes workspace: ...` errors.
- Add focused runtime coverage for workspace path safety, symlink escapes, missing file creation, patch target safety, and repo-aware search behavior.
- Replace direct runtime `walkdir` traversal with `ignore::WalkBuilder` and remove the direct `walkdir` runtime dependency.

## v0.4.1

- Add safer atomic writes for JSON, session, config, plugin, tool, and file-edit persistence to reduce the risk of truncated files.
- Add CI coverage for formatting, clippy, and serial workspace tests.
- Fix runtime prompt tests so they isolate temp roots from ambient project instruction and memory files.

## v0.4.0

- Completely redesigned compaction

Before - haha delete messages go brr 
Now - Full and complete compactions (I totally didn't spend hours recreating opencode's compaction system in rust)

- Fixed openai models not having a context length
- Add fallback for context length to 200K if can't find or is unknown for whatever reason
- Removed Vim Mode
- some black magic to make things better in the backend
- add snapshots (/undo and /redo)
- add session fork/rename/timeline
- add custom command templates
- add @file references
- add changelog.md so github has changelogs
- persist things like permission mode and reasoning effort across sessions and startups
- add patch/diff editing for improved multi-file editing
- many more improvements and fixes
