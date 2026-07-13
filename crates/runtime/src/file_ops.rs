use std::cmp::Reverse;
use std::fs;
use std::io;
use std::path::{Component, Path, PathBuf};
use std::time::Instant;

use glob::Pattern;
use ignore::WalkBuilder;
use platform::write_atomic;
use regex::RegexBuilder;
use serde::{Deserialize, Serialize};

const SEARCH_RESULT_LIMIT: usize = 100;
const DEFAULT_SEARCH_SKIP_DIRS: &[&str] = &[
    ".git",
    "target",
    ".pebble/sessions",
    ".pebble/tool-results",
    ".pebble/agents",
    ".sandbox-home",
    ".sandbox-tmp",
];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TextFilePayload {
    #[serde(rename = "filePath")]
    pub file_path: String,
    pub content: String,
    #[serde(rename = "numLines")]
    pub num_lines: usize,
    #[serde(rename = "startLine")]
    pub start_line: usize,
    #[serde(rename = "totalLines")]
    pub total_lines: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReadFileOutput {
    #[serde(rename = "type")]
    pub kind: String,
    pub file: TextFilePayload,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StructuredPatchHunk {
    #[serde(rename = "oldStart")]
    pub old_start: usize,
    #[serde(rename = "oldLines")]
    pub old_lines: usize,
    #[serde(rename = "newStart")]
    pub new_start: usize,
    #[serde(rename = "newLines")]
    pub new_lines: usize,
    pub lines: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WriteFileOutput {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(rename = "filePath")]
    pub file_path: String,
    pub content: String,
    #[serde(rename = "structuredPatch")]
    pub structured_patch: Vec<StructuredPatchHunk>,
    #[serde(rename = "originalFile")]
    pub original_file: Option<String>,
    #[serde(rename = "gitDiff")]
    pub git_diff: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EditFileOutput {
    #[serde(rename = "filePath")]
    pub file_path: String,
    #[serde(rename = "oldString")]
    pub old_string: String,
    #[serde(rename = "newString")]
    pub new_string: String,
    #[serde(rename = "originalFile")]
    pub original_file: String,
    #[serde(rename = "structuredPatch")]
    pub structured_patch: Vec<StructuredPatchHunk>,
    #[serde(rename = "userModified")]
    pub user_modified: bool,
    #[serde(rename = "replaceAll")]
    pub replace_all: bool,
    #[serde(rename = "gitDiff")]
    pub git_diff: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ApplyPatchOutput {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(rename = "dryRun")]
    pub dry_run: bool,
    #[serde(rename = "changedFiles")]
    pub changed_files: Vec<ApplyPatchFileChange>,
    pub summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ApplyPatchFileChange {
    #[serde(rename = "filePath")]
    pub file_path: String,
    pub action: String,
    #[serde(rename = "beforeExists")]
    pub before_exists: bool,
    #[serde(rename = "afterExists")]
    pub after_exists: bool,
    #[serde(rename = "structuredPatch")]
    pub structured_patch: Vec<StructuredPatchHunk>,
    #[serde(rename = "beforeContent")]
    pub before_content: Option<String>,
    #[serde(rename = "afterContent")]
    pub after_content: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GlobSearchOutput {
    #[serde(rename = "durationMs")]
    pub duration_ms: u128,
    #[serde(rename = "numFiles")]
    pub num_files: usize,
    pub filenames: Vec<String>,
    pub truncated: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GrepSearchInput {
    pub pattern: String,
    pub path: Option<String>,
    pub glob: Option<String>,
    #[serde(rename = "output_mode")]
    pub output_mode: Option<String>,
    #[serde(rename = "-B")]
    pub before: Option<usize>,
    #[serde(rename = "-A")]
    pub after: Option<usize>,
    #[serde(rename = "-C")]
    pub context_short: Option<usize>,
    pub context: Option<usize>,
    #[serde(rename = "-n")]
    pub line_numbers: Option<bool>,
    #[serde(rename = "-i")]
    pub case_insensitive: Option<bool>,
    #[serde(rename = "type")]
    pub file_type: Option<String>,
    pub head_limit: Option<usize>,
    pub offset: Option<usize>,
    pub multiline: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GrepSearchOutput {
    pub mode: Option<String>,
    #[serde(rename = "numFiles")]
    pub num_files: usize,
    pub filenames: Vec<String>,
    pub content: Option<String>,
    #[serde(rename = "numLines")]
    pub num_lines: Option<usize>,
    #[serde(rename = "numMatches")]
    pub num_matches: Option<usize>,
    #[serde(rename = "appliedLimit")]
    pub applied_limit: Option<usize>,
    #[serde(rename = "appliedOffset")]
    pub applied_offset: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GrepOutputMode {
    FilesWithMatches,
    Content,
    Count,
}

impl GrepOutputMode {
    fn parse(value: Option<&str>) -> io::Result<Self> {
        match value.unwrap_or("files_with_matches") {
            "files_with_matches" => Ok(Self::FilesWithMatches),
            "content" => Ok(Self::Content),
            "count" => Ok(Self::Count),
            other => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "unsupported grep_search output_mode `{other}` (expected content, files_with_matches, or count)"
                ),
            )),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::FilesWithMatches => "files_with_matches",
            Self::Content => "content",
            Self::Count => "count",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct GrepContentLine {
    file_path: String,
    rendered: String,
}

pub fn read_file(
    path: &str,
    offset: Option<usize>,
    limit: Option<usize>,
) -> io::Result<ReadFileOutput> {
    let absolute_path = normalize_existing_workspace_path(path)?;
    let content = fs::read_to_string(&absolute_path)?;
    let lines: Vec<&str> = content.lines().collect();
    let start_index = offset.unwrap_or(0).min(lines.len());
    let end_index = limit.map_or(lines.len(), |limit| {
        start_index.saturating_add(limit).min(lines.len())
    });
    let selected = lines[start_index..end_index].join("\n");

    Ok(ReadFileOutput {
        kind: String::from("text"),
        file: TextFilePayload {
            file_path: absolute_path.to_string_lossy().into_owned(),
            content: selected,
            num_lines: end_index.saturating_sub(start_index),
            start_line: start_index.saturating_add(1),
            total_lines: lines.len(),
        },
    })
}

pub fn write_file(path: &str, content: &str) -> io::Result<WriteFileOutput> {
    let absolute_path = normalize_missing_workspace_path(path)?;
    let original_file = fs::read_to_string(&absolute_path).ok();
    if let Some(parent) = absolute_path.parent() {
        fs::create_dir_all(parent)?;
    }
    write_atomic(&absolute_path, content)?;

    Ok(WriteFileOutput {
        kind: if original_file.is_some() {
            String::from("update")
        } else {
            String::from("create")
        },
        file_path: absolute_path.to_string_lossy().into_owned(),
        content: content.to_owned(),
        structured_patch: make_patch(original_file.as_deref().unwrap_or(""), content),
        original_file,
        git_diff: None,
    })
}

pub fn edit_file(
    path: &str,
    old_string: &str,
    new_string: &str,
    replace_all: bool,
) -> io::Result<EditFileOutput> {
    let absolute_path = normalize_existing_workspace_path(path)?;
    let original_file = fs::read_to_string(&absolute_path)?;
    if old_string == new_string {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "old_string and new_string must differ",
        ));
    }
    if !original_file.contains(old_string) {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "old_string not found in file",
        ));
    }

    let updated = if replace_all {
        original_file.replace(old_string, new_string)
    } else {
        original_file.replacen(old_string, new_string, 1)
    };
    write_atomic(&absolute_path, &updated)?;

    Ok(EditFileOutput {
        file_path: absolute_path.to_string_lossy().into_owned(),
        old_string: old_string.to_owned(),
        new_string: new_string.to_owned(),
        original_file: original_file.clone(),
        structured_patch: make_patch(&original_file, &updated),
        user_modified: false,
        replace_all,
        git_diff: None,
    })
}

pub fn apply_patch(patch: &str, dry_run: bool) -> io::Result<ApplyPatchOutput> {
    let cwd = workspace_root()?;
    let plans = parse_patch_plans(patch, &cwd)?;
    if plans.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "patch did not contain any file changes",
        ));
    }

    let mut pending_changes = Vec::new();
    for plan in plans {
        let before = read_optional_text_file(&plan.path)?;
        let after = match &plan.operation {
            PatchOperation::Unified { hunks } => {
                apply_unified_hunks(&plan, before.as_deref(), hunks)?
            }
            PatchOperation::Add { content } => {
                if before.is_some() {
                    return Err(invalid_patch(format!(
                        "cannot add `{}` because it already exists",
                        plan.display_path
                    )));
                }
                Some(content.clone())
            }
            PatchOperation::Delete => {
                if before.is_none() {
                    return Err(invalid_patch(format!(
                        "cannot delete `{}` because it does not exist",
                        plan.display_path
                    )));
                }
                None
            }
            PatchOperation::OpenAiUpdate { hunks } => {
                let Some(before_content) = before.as_deref() else {
                    return Err(invalid_patch(format!(
                        "cannot update `{}` because it does not exist",
                        plan.display_path
                    )));
                };
                Some(apply_openai_update_hunks(
                    &plan.display_path,
                    before_content,
                    hunks,
                )?)
            }
        };

        if before == after {
            continue;
        }

        pending_changes.push(PendingPatchChange {
            path: plan.path,
            display_path: plan.display_path,
            before,
            after,
        });
    }

    if pending_changes.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "patch produced no changes",
        ));
    }

    if !dry_run {
        write_pending_patch_changes(&pending_changes)?;
    }

    let changed_files = pending_changes
        .into_iter()
        .map(|change| {
            let before_exists = change.before.is_some();
            let after_exists = change.after.is_some();
            let action = match (before_exists, after_exists) {
                (false, true) => "create",
                (true, false) => "delete",
                (true, true) => "update",
                (false, false) => "noop",
            }
            .to_string();
            ApplyPatchFileChange {
                file_path: change.path.to_string_lossy().into_owned(),
                action,
                before_exists,
                after_exists,
                structured_patch: make_patch(
                    change.before.as_deref().unwrap_or(""),
                    change.after.as_deref().unwrap_or(""),
                ),
                before_content: change.before,
                after_content: change.after,
            }
        })
        .collect::<Vec<_>>();

    let summary = format_patch_summary(dry_run, &changed_files);
    Ok(ApplyPatchOutput {
        kind: String::from("patch"),
        dry_run,
        changed_files,
        summary,
    })
}

pub fn glob_search(pattern: &str, path: Option<&str>) -> io::Result<GlobSearchOutput> {
    let started = Instant::now();
    let workspace = workspace_root()?;
    let base_dir = path
        .map(normalize_existing_workspace_path)
        .transpose()?
        .unwrap_or_else(|| workspace.clone());
    let explicit_path = path.is_some() && base_dir != workspace;
    let pattern_path = if Path::new(pattern).is_absolute() {
        PathBuf::from(pattern)
    } else {
        base_dir.join(pattern)
    };
    ensure_workspace_path(&normalize_path_lexically(&pattern_path), &workspace)?;
    let search_pattern = pattern_path.to_string_lossy().into_owned();
    let visible_files = if explicit_path {
        None
    } else {
        Some(collect_search_files(&base_dir, false)?)
    };

    let mut matches = Vec::new();
    let entries = glob::glob(&search_pattern)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error.to_string()))?;
    for entry in entries.flatten() {
        if !entry.is_file() {
            continue;
        }
        let entry = normalize_existing_workspace_path_in_workspace(&entry, &workspace)?;
        if let Some(visible_files) = &visible_files {
            if visible_files.binary_search(&entry).is_err() {
                continue;
            }
        }
        matches.push(entry);
    }
    matches.sort();
    matches.dedup();

    matches.sort_by_key(|path| {
        fs::metadata(path)
            .and_then(|metadata| metadata.modified())
            .ok()
            .map(Reverse)
    });

    let truncated = matches.len() > SEARCH_RESULT_LIMIT;
    let filenames = matches
        .into_iter()
        .take(SEARCH_RESULT_LIMIT)
        .map(|path| path.to_string_lossy().into_owned())
        .collect::<Vec<_>>();

    Ok(GlobSearchOutput {
        duration_ms: started.elapsed().as_millis(),
        num_files: filenames.len(),
        filenames,
        truncated,
    })
}

pub fn grep_search(input: &GrepSearchInput) -> io::Result<GrepSearchOutput> {
    let workspace = workspace_root()?;
    let base_path = input
        .path
        .as_deref()
        .map(normalize_existing_workspace_path)
        .transpose()?
        .unwrap_or_else(|| workspace.clone());
    let explicit_path = input.path.is_some() && base_path != workspace;

    let regex = RegexBuilder::new(&input.pattern)
        .case_insensitive(input.case_insensitive.unwrap_or(false))
        .dot_matches_new_line(input.multiline.unwrap_or(false))
        .build()
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error.to_string()))?;

    let glob_filter = input
        .glob
        .as_deref()
        .map(Pattern::new)
        .transpose()
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error.to_string()))?;
    let file_type = input.file_type.as_deref();
    let output_mode = GrepOutputMode::parse(input.output_mode.as_deref())?;
    let context = input.context.or(input.context_short).unwrap_or(0);

    let mut filenames = Vec::new();
    let mut content_lines = Vec::new();
    let mut total_matches = 0usize;

    for file_path in collect_search_files(&base_path, explicit_path)? {
        if !matches_optional_filters(&file_path, glob_filter.as_ref(), file_type) {
            continue;
        }

        let Ok(file_contents) = fs::read_to_string(&file_path) else {
            continue;
        };

        if output_mode == GrepOutputMode::Count {
            let count = regex.find_iter(&file_contents).count();
            if count > 0 {
                filenames.push(file_path.to_string_lossy().into_owned());
                total_matches += count;
            }
            continue;
        }

        let lines: Vec<&str> = file_contents.lines().collect();
        let mut matched_lines = Vec::new();
        for (index, line) in lines.iter().enumerate() {
            if regex.is_match(line) {
                total_matches += 1;
                matched_lines.push(index);
            }
        }

        if matched_lines.is_empty() {
            continue;
        }

        let file_path = file_path.to_string_lossy().into_owned();
        filenames.push(file_path.clone());
        if output_mode == GrepOutputMode::Content {
            for index in matched_lines {
                let start = index.saturating_sub(input.before.unwrap_or(context));
                let end = (index + input.after.unwrap_or(context) + 1).min(lines.len());
                for (current, line) in lines.iter().enumerate().take(end).skip(start) {
                    let prefix = if input.line_numbers.unwrap_or(true) {
                        format!("{file_path}:{}:", current + 1)
                    } else {
                        format!("{file_path}:")
                    };
                    content_lines.push(GrepContentLine {
                        file_path: file_path.clone(),
                        rendered: format!("{prefix}{line}"),
                    });
                }
            }
        }
    }

    if output_mode == GrepOutputMode::Content {
        return Ok(grep_content_output(
            output_mode,
            content_lines,
            input.head_limit,
            input.offset,
        ));
    }

    let (filenames, applied_limit, applied_offset) =
        apply_limit(filenames, input.head_limit, input.offset);

    Ok(GrepSearchOutput {
        mode: Some(output_mode.as_str().to_string()),
        num_files: filenames.len(),
        filenames,
        content: None,
        num_lines: None,
        num_matches: (output_mode == GrepOutputMode::Count).then_some(total_matches),
        applied_limit,
        applied_offset,
    })
}

fn grep_content_output(
    output_mode: GrepOutputMode,
    content_lines: Vec<GrepContentLine>,
    head_limit: Option<usize>,
    offset: Option<usize>,
) -> GrepSearchOutput {
    let (lines, applied_limit, applied_offset) = apply_limit(content_lines, head_limit, offset);
    let filenames = unique_grep_content_filenames(&lines);
    let num_lines = lines.len();
    let rendered_content = lines
        .into_iter()
        .map(|line| line.rendered)
        .collect::<Vec<_>>()
        .join("\n");

    GrepSearchOutput {
        mode: Some(output_mode.as_str().to_string()),
        num_files: filenames.len(),
        filenames,
        num_lines: Some(num_lines),
        content: Some(rendered_content),
        num_matches: None,
        applied_limit,
        applied_offset,
    }
}

fn unique_grep_content_filenames(lines: &[GrepContentLine]) -> Vec<String> {
    let mut filenames = Vec::new();
    for line in lines {
        if !filenames.contains(&line.file_path) {
            filenames.push(line.file_path.clone());
        }
    }
    filenames
}

fn collect_search_files(base_path: &Path, explicit_path: bool) -> io::Result<Vec<PathBuf>> {
    if base_path.is_file() {
        return Ok(vec![base_path.to_path_buf()]);
    }

    let mut files = Vec::new();
    let mut builder = WalkBuilder::new(base_path);
    builder
        .hidden(!explicit_path)
        .ignore(!explicit_path)
        .git_ignore(!explicit_path)
        .git_global(!explicit_path)
        .git_exclude(!explicit_path)
        .parents(!explicit_path);

    for entry in builder.build() {
        let entry = entry.map_err(|error| io::Error::other(error.to_string()))?;
        let path = entry.path();
        if entry
            .file_type()
            .is_some_and(|file_type| file_type.is_file())
            && (explicit_path || is_default_search_visible(path, base_path))
        {
            files.push(entry.path().to_path_buf());
        }
    }
    files.sort();
    Ok(files)
}

fn matches_optional_filters(
    path: &Path,
    glob_filter: Option<&Pattern>,
    file_type: Option<&str>,
) -> bool {
    if let Some(glob_filter) = glob_filter {
        let path_string = path.to_string_lossy();
        if !glob_filter.matches(&path_string) && !glob_filter.matches_path(path) {
            return false;
        }
    }

    if let Some(file_type) = file_type {
        let extension = path.extension().and_then(|extension| extension.to_str());
        if extension != Some(file_type) {
            return false;
        }
    }

    true
}

fn apply_limit<T>(
    items: Vec<T>,
    limit: Option<usize>,
    offset: Option<usize>,
) -> (Vec<T>, Option<usize>, Option<usize>) {
    let offset_value = offset.unwrap_or(0);
    let mut items = items.into_iter().skip(offset_value).collect::<Vec<_>>();
    let explicit_limit = limit.unwrap_or(250);
    if explicit_limit == 0 {
        return (items, None, (offset_value > 0).then_some(offset_value));
    }

    let truncated = items.len() > explicit_limit;
    items.truncate(explicit_limit);
    (
        items,
        truncated.then_some(explicit_limit),
        (offset_value > 0).then_some(offset_value),
    )
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PatchPlan {
    path: PathBuf,
    display_path: String,
    operation: PatchOperation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum PatchOperation {
    Unified { hunks: Vec<UnifiedHunk> },
    Add { content: String },
    Delete,
    OpenAiUpdate { hunks: Vec<Vec<PatchLine>> },
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct UnifiedHunk {
    old_start: usize,
    old_lines: usize,
    new_start: usize,
    new_lines: usize,
    lines: Vec<PatchLine>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PatchLine {
    kind: char,
    text: String,
    no_newline: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TextLines {
    lines: Vec<String>,
    final_newline: bool,
    newline: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PendingPatchChange {
    path: PathBuf,
    display_path: String,
    before: Option<String>,
    after: Option<String>,
}

fn parse_patch_plans(patch: &str, cwd: &Path) -> io::Result<Vec<PatchPlan>> {
    let trimmed = patch.trim_start();
    if trimmed.starts_with("*** Begin Patch") {
        parse_openai_patch(trimmed, cwd)
    } else {
        parse_unified_patch(patch, cwd)
    }
}

#[allow(clippy::too_many_lines)]
fn parse_unified_patch(patch: &str, cwd: &Path) -> io::Result<Vec<PatchPlan>> {
    let lines = patch
        .lines()
        .map(|line| line.trim_end_matches('\r').to_string())
        .collect::<Vec<_>>();
    let mut plans = Vec::new();
    let mut index = 0usize;

    while index < lines.len() {
        while index < lines.len()
            && !lines[index].starts_with("diff --git ")
            && !lines[index].starts_with("--- ")
        {
            if is_binary_patch_marker(&lines[index]) {
                return Err(invalid_patch(
                    "binary patches are not supported by apply_patch",
                ));
            }
            index += 1;
        }
        if index >= lines.len() {
            break;
        }

        if lines[index].starts_with("diff --git ") {
            index += 1;
            while index < lines.len() && !lines[index].starts_with("--- ") {
                if is_binary_patch_marker(&lines[index]) {
                    return Err(invalid_patch(
                        "binary patches are not supported by apply_patch",
                    ));
                }
                index += 1;
            }
            if index >= lines.len() {
                return Err(invalid_patch("missing --- file header after diff --git"));
            }
        }

        let old_path = parse_diff_header_path(&lines[index], "--- ")?;
        index += 1;
        if index >= lines.len() || !lines[index].starts_with("+++ ") {
            return Err(invalid_patch("missing +++ file header"));
        }
        let new_path = parse_diff_header_path(&lines[index], "+++ ")?;
        index += 1;

        let display_path = patch_target_path(&old_path, &new_path)?;
        let absolute_path = validate_patch_path(cwd, &display_path)?;
        let mut hunks = Vec::new();
        while index < lines.len()
            && !lines[index].starts_with("diff --git ")
            && !lines[index].starts_with("--- ")
        {
            if lines[index].trim().is_empty() {
                index += 1;
                continue;
            }
            if !lines[index].starts_with("@@ ") {
                return Err(invalid_patch(format!(
                    "expected hunk header for `{display_path}`, got `{}`",
                    lines[index]
                )));
            }
            let (old_start, old_lines, new_start, new_lines) = parse_hunk_header(&lines[index])?;
            index += 1;
            let mut hunk_lines: Vec<PatchLine> = Vec::new();
            let mut actual_old = 0usize;
            let mut actual_new = 0usize;
            while actual_old < old_lines || actual_new < new_lines {
                if index >= lines.len() {
                    return Err(invalid_patch(format!(
                        "hunk for `{display_path}` ended before its declared line counts"
                    )));
                }
                let line = &lines[index];
                if line.starts_with("\\ No newline at end of file") {
                    if let Some(previous) = hunk_lines.last_mut() {
                        previous.no_newline = true;
                    }
                    index += 1;
                    continue;
                }
                if line.starts_with("@@ ") || line.starts_with("diff --git ") {
                    return Err(invalid_patch(format!(
                        "hunk for `{display_path}` ended before its declared line counts"
                    )));
                }
                let Some(kind) = line.chars().next() else {
                    return Err(invalid_patch("empty line in hunk"));
                };
                if !matches!(kind, ' ' | '-' | '+') {
                    return Err(invalid_patch(format!("invalid hunk line `{line}`")));
                }
                if matches!(kind, ' ' | '-') {
                    actual_old += 1;
                }
                if matches!(kind, ' ' | '+') {
                    actual_new += 1;
                }
                if actual_old > old_lines || actual_new > new_lines {
                    return Err(invalid_patch(format!(
                        "hunk for `{display_path}` has more lines than declared"
                    )));
                }
                hunk_lines.push(PatchLine {
                    kind,
                    text: line[1..].to_string(),
                    no_newline: false,
                });
                index += 1;
            }
            validate_hunk_counts(old_lines, new_lines, &hunk_lines)?;
            while index < lines.len() && lines[index].starts_with("\\ No newline at end of file") {
                if let Some(previous) = hunk_lines.last_mut() {
                    previous.no_newline = true;
                }
                index += 1;
            }
            hunks.push(UnifiedHunk {
                old_start,
                old_lines,
                new_start,
                new_lines,
                lines: hunk_lines,
            });
        }

        if hunks.is_empty() {
            return Err(invalid_patch(format!(
                "patch for `{display_path}` has no hunks"
            )));
        }

        plans.push(PatchPlan {
            path: absolute_path,
            display_path,
            operation: PatchOperation::Unified { hunks },
        });
    }

    Ok(plans)
}

fn parse_openai_patch(patch: &str, cwd: &Path) -> io::Result<Vec<PatchPlan>> {
    let lines = patch
        .lines()
        .map(|line| line.trim_end_matches('\r').to_string())
        .collect::<Vec<_>>();
    if lines.first().map(String::as_str) != Some("*** Begin Patch")
        || lines.last().map(String::as_str) != Some("*** End Patch")
    {
        return Err(invalid_patch(
            "OpenAI-style patches must start with *** Begin Patch and end with *** End Patch",
        ));
    }

    let mut plans = Vec::new();
    let mut index = 1usize;
    while index + 1 < lines.len() {
        let line = &lines[index];
        if let Some(path) = line.strip_prefix("*** Add File: ") {
            index += 1;
            let mut content_lines = Vec::new();
            while index + 1 < lines.len() && !lines[index].starts_with("*** ") {
                let Some(content) = lines[index].strip_prefix('+') else {
                    return Err(invalid_patch(format!(
                        "add file `{path}` contains a line that does not start with +"
                    )));
                };
                content_lines.push(content.to_string());
                index += 1;
            }
            let content = if content_lines.is_empty() {
                String::new()
            } else {
                format!("{}\n", content_lines.join("\n"))
            };
            plans.push(openai_plan(cwd, path, PatchOperation::Add { content })?);
            continue;
        }

        if let Some(path) = line.strip_prefix("*** Delete File: ") {
            plans.push(openai_plan(cwd, path, PatchOperation::Delete)?);
            index += 1;
            continue;
        }

        if let Some(path) = line.strip_prefix("*** Update File: ") {
            index += 1;
            let mut hunks = Vec::new();
            let mut current = Vec::new();
            while index + 1 < lines.len() && !lines[index].starts_with("*** ") {
                if lines[index].starts_with("@@") {
                    if !current.is_empty() {
                        hunks.push(current);
                        current = Vec::new();
                    }
                    index += 1;
                    continue;
                }
                let Some(kind) = lines[index].chars().next() else {
                    index += 1;
                    continue;
                };
                if !matches!(kind, ' ' | '-' | '+') {
                    return Err(invalid_patch(format!(
                        "update file `{path}` contains invalid line `{}`",
                        lines[index]
                    )));
                }
                current.push(PatchLine {
                    kind,
                    text: lines[index][1..].to_string(),
                    no_newline: false,
                });
                index += 1;
            }
            if !current.is_empty() {
                hunks.push(current);
            }
            if hunks.is_empty() {
                return Err(invalid_patch(format!("update file `{path}` has no hunks")));
            }
            plans.push(openai_plan(
                cwd,
                path,
                PatchOperation::OpenAiUpdate { hunks },
            )?);
            continue;
        }

        return Err(invalid_patch(format!(
            "unsupported patch directive `{line}`"
        )));
    }

    Ok(plans)
}

fn openai_plan(cwd: &Path, path: &str, operation: PatchOperation) -> io::Result<PatchPlan> {
    let path = path.trim();
    Ok(PatchPlan {
        path: validate_patch_path(cwd, path)?,
        display_path: path.to_string(),
        operation,
    })
}

fn parse_diff_header_path(line: &str, prefix: &str) -> io::Result<String> {
    let raw = line
        .strip_prefix(prefix)
        .ok_or_else(|| invalid_patch(format!("expected `{prefix}` header")))?;
    parse_patch_path_token(raw)
}

fn patch_target_path(old_path: &str, new_path: &str) -> io::Result<String> {
    let target = if new_path == "/dev/null" {
        old_path
    } else {
        new_path
    };
    let target = target
        .strip_prefix("a/")
        .or_else(|| target.strip_prefix("b/"))
        .unwrap_or(target);
    if target == "/dev/null" || target.trim().is_empty() {
        return Err(invalid_patch("patch did not identify a target path"));
    }
    Ok(target.to_string())
}

fn parse_patch_path_token(raw: &str) -> io::Result<String> {
    let trimmed = raw.trim();
    if trimmed.starts_with('"') {
        let (path, _end) = parse_quoted_patch_path(trimmed)?;
        return Ok(path);
    }
    let path = trimmed.split('\t').next().unwrap_or(trimmed).trim();
    Ok(path.to_string())
}

fn parse_quoted_patch_path(input: &str) -> io::Result<(String, usize)> {
    let mut output = String::new();
    let mut chars = input.char_indices();
    if chars.next().map(|(_, ch)| ch) != Some('"') {
        return Err(invalid_patch("quoted patch path must start with a quote"));
    }
    while let Some((index, ch)) = chars.next() {
        match ch {
            '"' => return Ok((output, index + ch.len_utf8())),
            '\\' => {
                let Some((_escape_index, escaped)) = chars.next() else {
                    return Err(invalid_patch("unterminated escape in quoted patch path"));
                };
                match escaped {
                    'n' => output.push('\n'),
                    'r' => output.push('\r'),
                    't' => output.push('\t'),
                    '\\' => output.push('\\'),
                    '"' => output.push('"'),
                    '0'..='7' => {
                        let mut value = escaped.to_digit(8).unwrap_or(0);
                        for _ in 0..2 {
                            let mut clone = chars.clone();
                            if let Some((_, next)) = clone.next() {
                                if let Some(digit) = next.to_digit(8) {
                                    value = value * 8 + digit;
                                    chars = clone;
                                    continue;
                                }
                            }
                            break;
                        }
                        if let Some(byte) = char::from_u32(value) {
                            output.push(byte);
                        }
                    }
                    other => output.push(other),
                }
            }
            other => output.push(other),
        }
    }
    Err(invalid_patch("unterminated quoted patch path"))
}

fn is_binary_patch_marker(line: &str) -> bool {
    line.starts_with("Binary files ") || line.starts_with("GIT binary patch")
}

fn validate_patch_path(cwd: &Path, path: &str) -> io::Result<PathBuf> {
    let relative = Path::new(path);
    if relative.is_absolute() {
        return Err(invalid_patch(format!(
            "patch path `{path}` must be relative to the workspace"
        )));
    }

    let mut clean = PathBuf::new();
    for component in relative.components() {
        match component {
            std::path::Component::Normal(part) => clean.push(part),
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                return Err(invalid_patch(format!(
                    "patch path `{path}` must not contain .."
                )));
            }
            std::path::Component::RootDir | std::path::Component::Prefix(_) => {
                return Err(invalid_patch(format!("invalid patch path `{path}`")));
            }
        }
    }
    if clean.as_os_str().is_empty() {
        return Err(invalid_patch("patch path must not be empty"));
    }
    normalize_missing_workspace_path_in_workspace(&clean, cwd)
        .map_err(|error| invalid_patch(error.to_string()))
}

fn parse_hunk_header(header: &str) -> io::Result<(usize, usize, usize, usize)> {
    let body = header
        .strip_prefix("@@ ")
        .and_then(|value| value.split(" @@").next())
        .ok_or_else(|| invalid_patch(format!("invalid hunk header `{header}`")))?;
    let mut parts = body.split_whitespace();
    let old = parts
        .next()
        .ok_or_else(|| invalid_patch(format!("invalid hunk header `{header}`")))?;
    let new = parts
        .next()
        .ok_or_else(|| invalid_patch(format!("invalid hunk header `{header}`")))?;
    let (old_start, old_lines) = parse_hunk_range(old, '-')?;
    let (new_start, new_lines) = parse_hunk_range(new, '+')?;
    Ok((old_start, old_lines, new_start, new_lines))
}

fn parse_hunk_range(value: &str, marker: char) -> io::Result<(usize, usize)> {
    let range = value
        .strip_prefix(marker)
        .ok_or_else(|| invalid_patch(format!("invalid hunk range `{value}`")))?;
    if let Some((start, count)) = range.split_once(',') {
        Ok((
            start
                .parse()
                .map_err(|_| invalid_patch(format!("invalid hunk start `{start}`")))?,
            count
                .parse()
                .map_err(|_| invalid_patch(format!("invalid hunk count `{count}`")))?,
        ))
    } else {
        Ok((
            range
                .parse()
                .map_err(|_| invalid_patch(format!("invalid hunk start `{range}`")))?,
            1,
        ))
    }
}

fn validate_hunk_counts(old_lines: usize, new_lines: usize, lines: &[PatchLine]) -> io::Result<()> {
    let actual_old = lines
        .iter()
        .filter(|line| matches!(line.kind, ' ' | '-'))
        .count();
    let actual_new = lines
        .iter()
        .filter(|line| matches!(line.kind, ' ' | '+'))
        .count();
    if actual_old != old_lines || actual_new != new_lines {
        return Err(invalid_patch(format!(
            "hunk line counts do not match header: expected -{old_lines} +{new_lines}, got -{actual_old} +{actual_new}"
        )));
    }
    Ok(())
}

fn read_optional_text_file(path: &Path) -> io::Result<Option<String>> {
    match fs::read(path) {
        Ok(bytes) => String::from_utf8(bytes).map(Some).map_err(|_| {
            invalid_patch(format!(
                "cannot patch non-UTF-8 or binary file `{}`",
                path.display()
            ))
        }),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error),
    }
}

fn write_pending_patch_changes(changes: &[PendingPatchChange]) -> io::Result<()> {
    let mut applied = Vec::new();
    for change in changes {
        if let Err(error) = write_patch_file_state(&change.path, change.after.as_deref()) {
            let rollback_error = rollback_patch_changes(&applied);
            return Err(match rollback_error {
                Some(rollback_error) => io::Error::new(
                    error.kind(),
                    format!(
                        "failed to apply patch: {error}; rollback also failed: {rollback_error}"
                    ),
                ),
                None => io::Error::new(
                    error.kind(),
                    format!("failed to apply patch: {error}; changes were rolled back"),
                ),
            });
        }
        applied.push(change);
    }
    Ok(())
}

fn rollback_patch_changes(changes: &[&PendingPatchChange]) -> Option<io::Error> {
    let mut rollback_error = None;
    for change in changes.iter().rev() {
        if let Err(error) = write_patch_file_state(&change.path, change.before.as_deref()) {
            rollback_error = Some(error);
        }
    }
    rollback_error
}

fn write_patch_file_state(path: &Path, content: Option<&str>) -> io::Result<()> {
    match content {
        Some(content) => {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            write_atomic(path, content)
        }
        None => fs::remove_file(path),
    }
}

fn apply_unified_hunks(
    plan: &PatchPlan,
    before: Option<&str>,
    hunks: &[UnifiedHunk],
) -> io::Result<Option<String>> {
    let creates_file = hunks.first().is_some_and(|hunk| hunk.old_start == 0);
    let deletes_file = hunks.last().is_some_and(|hunk| hunk.new_start == 0);
    if creates_file && before.is_some() {
        return Err(invalid_patch(format!(
            "cannot create `{}` because it already exists",
            plan.display_path
        )));
    }
    if !creates_file && before.is_none() {
        return Err(invalid_patch(format!(
            "cannot patch `{}` because it does not exist",
            plan.display_path
        )));
    }

    let before_content = before.unwrap_or("");
    let old_text = split_text_lines(before_content);
    let mut output = Vec::new();
    let mut cursor = 0usize;

    for hunk in hunks {
        let old_index = hunk.old_start.saturating_sub(1);
        if old_index < cursor {
            return Err(invalid_patch(format!(
                "overlapping hunks in `{}`",
                plan.display_path
            )));
        }
        if old_index > old_text.lines.len() {
            return Err(invalid_patch(format!(
                "hunk for `{}` starts beyond end of file",
                plan.display_path
            )));
        }
        output.extend_from_slice(&old_text.lines[cursor..old_index]);
        cursor = old_index;

        for line in &hunk.lines {
            match line.kind {
                ' ' => {
                    expect_line(&plan.display_path, &old_text.lines, cursor, &line.text)?;
                    output.push(old_text.lines[cursor].clone());
                    cursor += 1;
                }
                '-' => {
                    expect_line(&plan.display_path, &old_text.lines, cursor, &line.text)?;
                    cursor += 1;
                }
                '+' => output.push(line.text.clone()),
                _ => unreachable!("validated hunk line kind"),
            }
        }
    }
    output.extend_from_slice(&old_text.lines[cursor..]);

    if deletes_file && output.is_empty() {
        return Ok(None);
    }
    if deletes_file {
        return Err(invalid_patch(format!(
            "delete patch for `{}` did not remove all file contents",
            plan.display_path
        )));
    }
    Ok(Some(join_text_lines(
        &output,
        patch_final_newline(hunks).unwrap_or(old_text.final_newline || creates_file),
        old_text.newline,
    )))
}

fn apply_openai_update_hunks(
    display_path: &str,
    before_content: &str,
    hunks: &[Vec<PatchLine>],
) -> io::Result<String> {
    let mut text = split_text_lines(before_content);
    for hunk in hunks {
        let old_sequence = hunk
            .iter()
            .filter(|line| matches!(line.kind, ' ' | '-'))
            .map(|line| line.text.clone())
            .collect::<Vec<_>>();
        let new_sequence = hunk
            .iter()
            .filter(|line| matches!(line.kind, ' ' | '+'))
            .map(|line| line.text.clone())
            .collect::<Vec<_>>();
        if old_sequence.is_empty() {
            return Err(invalid_patch(format!(
                "update hunk for `{display_path}` has no context to match"
            )));
        }
        let Some(start) = find_unique_subsequence(&text.lines, &old_sequence) else {
            return Err(invalid_patch(format!(
                "update hunk for `{display_path}` did not match uniquely"
            )));
        };
        text.lines
            .splice(start..start + old_sequence.len(), new_sequence);
    }
    Ok(join_text_lines(
        &text.lines,
        text.final_newline,
        text.newline,
    ))
}

fn split_text_lines(content: &str) -> TextLines {
    if content.is_empty() {
        return TextLines {
            lines: Vec::new(),
            final_newline: false,
            newline: "\n",
        };
    }
    let newline = if content.contains("\r\n") {
        "\r\n"
    } else {
        "\n"
    };
    let final_newline = content.ends_with('\n');
    let mut lines = content
        .split('\n')
        .map(|line| line.strip_suffix('\r').unwrap_or(line).to_string())
        .collect::<Vec<_>>();
    if final_newline {
        lines.pop();
    }
    TextLines {
        lines,
        final_newline,
        newline,
    }
}

fn join_text_lines(lines: &[String], final_newline: bool, newline: &str) -> String {
    let mut content = lines.join(newline);
    if final_newline && (!content.is_empty() || !lines.is_empty()) {
        content.push_str(newline);
    }
    content
}

fn patch_final_newline(hunks: &[UnifiedHunk]) -> Option<bool> {
    hunks
        .iter()
        .flat_map(|hunk| hunk.lines.iter())
        .rev()
        .find(|line| matches!(line.kind, ' ' | '+'))
        .map(|line| !line.no_newline)
}

fn expect_line(path: &str, lines: &[String], index: usize, expected: &str) -> io::Result<()> {
    let Some(actual) = lines.get(index) else {
        return Err(invalid_patch(format!(
            "hunk for `{path}` expected `{expected}` beyond end of file"
        )));
    };
    if actual != expected {
        return Err(invalid_patch(format!(
            "hunk for `{path}` did not match at line {}: expected `{expected}`, found `{actual}`",
            index + 1
        )));
    }
    Ok(())
}

fn find_unique_subsequence(lines: &[String], wanted: &[String]) -> Option<usize> {
    if wanted.is_empty() || wanted.len() > lines.len() {
        return None;
    }
    let mut matches = lines
        .windows(wanted.len())
        .enumerate()
        .filter_map(|(index, window)| (window == wanted).then_some(index));
    let first = matches.next()?;
    matches.next().is_none().then_some(first)
}

fn format_patch_summary(dry_run: bool, files: &[ApplyPatchFileChange]) -> String {
    let mut creates = 0usize;
    let mut updates = 0usize;
    let mut deletes = 0usize;
    for file in files {
        match file.action.as_str() {
            "create" => creates += 1,
            "update" => updates += 1,
            "delete" => deletes += 1,
            _ => {}
        }
    }
    let verb = if dry_run { "would change" } else { "changed" };
    format!(
        "{verb} {} files ({} create, {} update, {} delete)",
        files.len(),
        creates,
        updates,
        deletes
    )
}

fn invalid_patch(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn make_patch(original: &str, updated: &str) -> Vec<StructuredPatchHunk> {
    let mut lines = Vec::new();
    for line in original.lines() {
        lines.push(format!("-{line}"));
    }
    for line in updated.lines() {
        lines.push(format!("+{line}"));
    }

    vec![StructuredPatchHunk {
        old_start: 1,
        old_lines: original.lines().count(),
        new_start: 1,
        new_lines: updated.lines().count(),
        lines,
    }]
}

fn workspace_root() -> io::Result<PathBuf> {
    std::env::current_dir()?.canonicalize()
}

fn normalize_existing_workspace_path(path: &str) -> io::Result<PathBuf> {
    let workspace = workspace_root()?;
    normalize_existing_workspace_path_in_workspace(Path::new(path), &workspace)
}

fn normalize_existing_workspace_path_in_workspace(
    path: &Path,
    workspace: &Path,
) -> io::Result<PathBuf> {
    let candidate = resolve_candidate_path(path, workspace);
    let lexical = normalize_path_lexically(&candidate);
    ensure_workspace_path(&lexical, workspace)?;
    let canonical = candidate.canonicalize()?;
    ensure_workspace_path(&canonical, workspace)
}

fn normalize_missing_workspace_path(path: &str) -> io::Result<PathBuf> {
    let workspace = workspace_root()?;
    normalize_missing_workspace_path_in_workspace(Path::new(path), &workspace)
}

fn normalize_missing_workspace_path_in_workspace(
    path: &Path,
    workspace: &Path,
) -> io::Result<PathBuf> {
    let candidate = resolve_candidate_path(path, workspace);
    let lexical = normalize_path_lexically(&candidate);
    ensure_workspace_path(&lexical, workspace)?;

    if let Ok(canonical) = lexical.canonicalize() {
        return ensure_workspace_path(&canonical, workspace);
    }

    let mut existing = lexical.as_path();
    let mut missing_components = Vec::new();
    loop {
        match existing.canonicalize() {
            Ok(canonical_existing) => {
                let mut resolved = ensure_workspace_path(&canonical_existing, workspace)?;
                for component in missing_components.iter().rev() {
                    resolved.push(component);
                }
                return ensure_workspace_path(&resolved, workspace);
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                let Some(file_name) = existing.file_name() else {
                    return Err(io::Error::new(
                        io::ErrorKind::NotFound,
                        format!("path has no existing parent: {}", lexical.display()),
                    ));
                };
                missing_components.push(file_name.to_os_string());
                existing = existing.parent().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::NotFound,
                        format!("path has no existing parent: {}", lexical.display()),
                    )
                })?;
            }
            Err(error) => return Err(error),
        }
    }
}

fn resolve_candidate_path(path: &Path, workspace: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        workspace.join(path)
    }
}

fn normalize_path_lexically(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
            Component::RootDir => normalized.push(component.as_os_str()),
            Component::CurDir => {}
            Component::Normal(part) => normalized.push(part),
            Component::ParentDir => {
                normalized.pop();
            }
        }
    }
    normalized
}

fn ensure_workspace_path(path: &Path, workspace: &Path) -> io::Result<PathBuf> {
    if path.starts_with(workspace) {
        Ok(path.to_path_buf())
    } else {
        Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            format!("path escapes workspace: {}", path.display()),
        ))
    }
}

fn is_default_search_visible(path: &Path, base_path: &Path) -> bool {
    let relative = path.strip_prefix(base_path).unwrap_or(path);
    !is_hidden_path(relative) && !is_default_skip_path(relative)
}

fn is_hidden_path(path: &Path) -> bool {
    path.components().any(|component| {
        let Component::Normal(part) = component else {
            return false;
        };
        part.to_str()
            .is_some_and(|part| part.starts_with('.') && part != "." && part != "..")
    })
}

fn is_default_skip_path(path: &Path) -> bool {
    DEFAULT_SEARCH_SKIP_DIRS
        .iter()
        .map(Path::new)
        .any(|skip| path == skip || path.starts_with(skip))
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::{
        apply_patch, edit_file, glob_search, grep_search, read_file, write_file, GrepSearchInput,
    };

    fn temp_path(name: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should move forward")
            .as_nanos();
        std::env::temp_dir().join(format!("pebble-native-{name}-{unique}"))
    }

    fn with_temp_workspace<T>(name: &str, test: impl FnOnce(&Path) -> T) -> T {
        let _guard = crate::test_env_lock();
        let dir = temp_path(name);
        std::fs::create_dir_all(&dir).expect("workspace directory should be created");
        let canonical_dir = dir
            .canonicalize()
            .expect("workspace directory should canonicalize");
        let original_dir = std::env::current_dir().expect("cwd should exist");
        std::env::set_current_dir(&canonical_dir).expect("set cwd");

        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| test(&canonical_dir)));
        std::env::set_current_dir(original_dir).expect("restore cwd");
        let _ = std::fs::remove_dir_all(&dir);

        match result {
            Ok(value) => value,
            Err(payload) => std::panic::resume_unwind(payload),
        }
    }

    fn grep_input(pattern: &str, path: Option<&str>, output_mode: &str) -> GrepSearchInput {
        GrepSearchInput {
            pattern: pattern.to_string(),
            path: path.map(str::to_string),
            glob: None,
            output_mode: Some(output_mode.to_string()),
            before: None,
            after: None,
            context_short: None,
            context: None,
            line_numbers: Some(true),
            case_insensitive: Some(false),
            file_type: None,
            head_limit: Some(100),
            offset: Some(0),
            multiline: Some(false),
        }
    }

    fn assert_error_contains(error: &std::io::Error, expected: &str) {
        let message = error.to_string();
        assert!(
            message.contains(expected),
            "expected error `{message}` to contain `{expected}`"
        );
    }

    fn assert_contains_relative_path(paths: &[String], workspace: &Path, relative: &str) {
        let expected = workspace.join(relative).to_string_lossy().into_owned();
        assert!(
            paths.iter().any(|path| path == &expected),
            "expected paths to contain `{expected}`, got {paths:?}"
        );
    }

    fn assert_excludes_relative_path(paths: &[String], workspace: &Path, relative: &str) {
        let expected = workspace.join(relative).to_string_lossy().into_owned();
        assert!(
            paths.iter().all(|path| path != &expected),
            "expected paths to exclude `{expected}`, got {paths:?}"
        );
    }

    #[test]
    fn reads_and_writes_files_inside_workspace() {
        with_temp_workspace("read-write", |_| {
            let write_output =
                write_file("read-write.txt", "one\ntwo\nthree").expect("write should succeed");
            assert_eq!(write_output.kind, "create");

            let read_output =
                read_file("read-write.txt", Some(1), Some(1)).expect("read should succeed");
            assert_eq!(read_output.file.content, "two");
        });
    }

    #[test]
    fn edits_file_contents_inside_workspace() {
        with_temp_workspace("edit", |_| {
            write_file("edit.txt", "alpha beta alpha").expect("initial write should succeed");
            let output =
                edit_file("edit.txt", "alpha", "omega", true).expect("edit should succeed");
            assert!(output.replace_all);
            assert_eq!(
                std::fs::read_to_string("edit.txt").expect("read edited file"),
                "omega beta omega"
            );
        });
    }

    #[test]
    fn rejects_absolute_paths_outside_workspace() {
        with_temp_workspace("outside-path", |_| {
            let outside = temp_path("outside-read.txt");
            std::fs::write(&outside, "secret").expect("outside fixture write");

            let error = read_file(outside.to_string_lossy().as_ref(), None, None)
                .expect_err("outside read should fail");
            assert_error_contains(&error, "path escapes workspace:");

            let error = write_file(outside.to_string_lossy().as_ref(), "changed")
                .expect_err("outside write should fail");
            assert_error_contains(&error, "path escapes workspace:");
            assert_eq!(
                std::fs::read_to_string(&outside).expect("outside fixture read"),
                "secret"
            );

            let _ = std::fs::remove_file(outside);
        });
    }

    #[test]
    fn rejects_parent_directory_escape() {
        with_temp_workspace("parent-escape", |_| {
            let error = write_file("../escape.txt", "nope").expect_err("parent escape should fail");
            assert_error_contains(&error, "path escapes workspace:");

            let error = glob_search("../*.txt", None).expect_err("glob escape should fail");
            assert_error_contains(&error, "path escapes workspace:");
        });
    }

    #[test]
    fn creates_missing_files_inside_workspace() {
        with_temp_workspace("missing-create", |_| {
            let output = write_file("nested/new.txt", "created\n")
                .expect("missing file create should succeed");
            assert_eq!(output.kind, "create");
            assert_eq!(
                std::fs::read_to_string("nested/new.txt").expect("created file read"),
                "created\n"
            );
        });
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlink_escape_for_native_file_tools() {
        with_temp_workspace("symlink-escape", |_| {
            let outside = temp_path("symlink-outside.txt");
            std::fs::write(&outside, "secret").expect("outside fixture write");
            std::os::unix::fs::symlink(&outside, "link.txt").expect("symlink fixture");

            let error =
                read_file("link.txt", None, None).expect_err("symlink read escape should fail");
            assert_error_contains(&error, "path escapes workspace:");

            let error =
                write_file("link.txt", "changed").expect_err("symlink write escape should fail");
            assert_error_contains(&error, "path escapes workspace:");

            let error = edit_file("link.txt", "secret", "changed", false)
                .expect_err("symlink edit escape should fail");
            assert_error_contains(&error, "path escapes workspace:");

            assert_eq!(
                std::fs::read_to_string(&outside).expect("outside fixture read"),
                "secret"
            );
            let _ = std::fs::remove_file(outside);
        });
    }

    #[cfg(unix)]
    #[test]
    fn rejects_patch_targets_that_escape_through_symlink() {
        with_temp_workspace("patch-symlink-escape", |_| {
            let outside = temp_path("patch-symlink-outside.txt");
            std::fs::write(&outside, "secret\n").expect("outside fixture write");
            std::os::unix::fs::symlink(&outside, "link.txt").expect("symlink fixture");

            let patch = "\
*** Begin Patch
*** Update File: link.txt
@@
-secret
+changed
*** End Patch";
            let error = apply_patch(patch, true).expect_err("symlink patch should fail");
            assert_error_contains(&error, "path escapes workspace:");
            assert_eq!(
                std::fs::read_to_string(&outside).expect("outside fixture read"),
                "secret\n"
            );
            let _ = std::fs::remove_file(outside);
        });
    }

    #[test]
    fn applies_unified_patch_with_dry_run() {
        with_temp_workspace("patch-dir", |_| {
            write_file("demo.txt", "alpha\nbeta\ngamma\n").expect("initial write should succeed");

            let patch = "\
--- a/demo.txt
+++ b/demo.txt
@@ -1,3 +1,3 @@
 alpha
-beta
+omega
 gamma
";
            let checked = apply_patch(patch, true).expect("dry run should succeed");
            assert!(checked.dry_run);
            assert_eq!(checked.changed_files[0].action, "update");
            assert_eq!(
                std::fs::read_to_string("demo.txt").expect("read file"),
                "alpha\nbeta\ngamma\n"
            );

            let applied = apply_patch(patch, false).expect("apply should succeed");
            assert!(!applied.dry_run);
            assert_eq!(
                std::fs::read_to_string("demo.txt").expect("read file"),
                "alpha\nomega\ngamma\n"
            );
        });
    }

    #[test]
    fn applies_openai_style_patch_and_rejects_unsafe_paths() {
        with_temp_workspace("openai-patch-dir", |_| {
            let patch = "\
*** Begin Patch
*** Add File: nested/new.txt
+one
+two
*** End Patch";
            apply_patch(patch, false).expect("add file patch should succeed");
            assert_eq!(
                std::fs::read_to_string("nested/new.txt").expect("read file"),
                "one\ntwo\n"
            );

            let unsafe_patch = "\
*** Begin Patch
*** Add File: ../escape.txt
+nope
*** End Patch";
            let error = apply_patch(unsafe_patch, true).expect_err("unsafe path should fail");
            assert!(error.to_string().contains("must not contain .."));
        });
    }

    #[test]
    fn rejects_patch_context_mismatch_without_writing() {
        with_temp_workspace("patch-mismatch-dir", |_| {
            write_file("demo.txt", "alpha\nbeta\n").expect("initial write should succeed");

            let patch = "\
--- a/demo.txt
+++ b/demo.txt
@@ -1,2 +1,2 @@
 alpha
-missing
+omega
";
            let error = apply_patch(patch, false).expect_err("mismatch should fail");
            assert!(error.to_string().contains("did not match"));
            assert_eq!(
                std::fs::read_to_string("demo.txt").expect("read file"),
                "alpha\nbeta\n"
            );
        });
    }

    #[test]
    fn applies_patch_fixture_with_quoted_space_path() {
        with_temp_workspace("patch-quoted-space-dir", |_| {
            std::fs::create_dir_all("dir").expect("directory should be created");
            write_file("dir/file with spaces.txt", "alpha\nbeta\ngamma\n")
                .expect("initial write should succeed");

            let patch = include_str!("../tests/fixtures/patches/quoted-space-path.patch");
            apply_patch(patch, false).expect("quoted path patch should succeed");
            assert_eq!(
                std::fs::read_to_string("dir/file with spaces.txt").expect("read file"),
                "alpha\nomega\ngamma\n"
            );
        });
    }

    #[test]
    fn preserves_crlf_and_no_final_newline() {
        with_temp_workspace("patch-line-ending-dir", |_| {
            write_file("crlf.txt", "alpha\r\nbeta\r\ngamma\r\n")
                .expect("initial write should succeed");
            let crlf_patch = "\
--- a/crlf.txt
+++ b/crlf.txt
@@ -1,3 +1,3 @@
 alpha
-beta
+omega
 gamma
";
            apply_patch(crlf_patch, false).expect("crlf patch should succeed");
            assert_eq!(
                std::fs::read_to_string("crlf.txt").expect("read file"),
                "alpha\r\nomega\r\ngamma\r\n"
            );

            write_file("no-newline.txt", "old").expect("initial write should succeed");
            let no_newline_patch = include_str!("../tests/fixtures/patches/no-final-newline.patch");
            apply_patch(no_newline_patch, false).expect("no-final-newline patch should succeed");
            assert_eq!(
                std::fs::read_to_string("no-newline.txt").expect("read file"),
                "new"
            );
        });
    }

    #[test]
    fn rejects_binary_patch_and_binary_target_with_clear_errors() {
        with_temp_workspace("patch-binary-dir", |_| {
            let binary_patch = include_str!("../tests/fixtures/patches/binary.patch");
            let error = apply_patch(binary_patch, true).expect_err("binary patch should fail");
            assert!(error
                .to_string()
                .contains("binary patches are not supported"));

            std::fs::write("image.bin", [0xff, 0x00, 0xfe]).expect("binary fixture write");
            let text_patch = "\
--- a/image.bin
+++ b/image.bin
@@ -1 +1 @@
-old
+new
";
            let error = apply_patch(text_patch, true).expect_err("binary target should fail");
            assert!(error.to_string().contains("non-UTF-8 or binary file"));
        });
    }

    #[test]
    fn broad_search_skips_repo_state_hidden_dirs_and_gitignored_files() {
        with_temp_workspace("repo-aware-search", |workspace| {
            write_file("src/visible.txt", "visible needle\n").expect("visible fixture write");
            for dir in super::DEFAULT_SEARCH_SKIP_DIRS {
                write_file(&format!("{dir}/skipped.txt"), "skipped needle\n")
                    .expect("skipped fixture write");
            }
            write_file(".gitignore", "ignored.txt\nignored-dir/\n")
                .expect("gitignore fixture write");
            write_file("ignored.txt", "ignored needle\n").expect("ignored file write");
            write_file("ignored-dir/file.txt", "ignored dir needle\n")
                .expect("ignored dir file write");

            let grep_output = grep_search(&grep_input("needle", None, "files_with_matches"))
                .expect("broad grep should succeed");
            assert_contains_relative_path(&grep_output.filenames, workspace, "src/visible.txt");
            assert_excludes_relative_path(&grep_output.filenames, workspace, "ignored.txt");
            assert_excludes_relative_path(
                &grep_output.filenames,
                workspace,
                "ignored-dir/file.txt",
            );
            for dir in super::DEFAULT_SEARCH_SKIP_DIRS {
                assert_excludes_relative_path(
                    &grep_output.filenames,
                    workspace,
                    &format!("{dir}/skipped.txt"),
                );
            }

            let glob_output = glob_search("**/*.txt", None).expect("broad glob should succeed");
            assert_contains_relative_path(&glob_output.filenames, workspace, "src/visible.txt");
            assert_excludes_relative_path(&glob_output.filenames, workspace, "ignored.txt");
            assert_excludes_relative_path(
                &glob_output.filenames,
                workspace,
                "ignored-dir/file.txt",
            );
            for dir in super::DEFAULT_SEARCH_SKIP_DIRS {
                assert_excludes_relative_path(
                    &glob_output.filenames,
                    workspace,
                    &format!("{dir}/skipped.txt"),
                );
            }

            let dot_grep = grep_search(&grep_input("needle", Some("."), "files_with_matches"))
                .expect("dot-path grep should keep broad-search defaults");
            assert_contains_relative_path(&dot_grep.filenames, workspace, "src/visible.txt");
            assert_excludes_relative_path(&dot_grep.filenames, workspace, "ignored.txt");

            let dot_glob =
                glob_search("**/*.txt", Some(".")).expect("dot-path glob should succeed");
            assert_contains_relative_path(&dot_glob.filenames, workspace, "src/visible.txt");
            assert_excludes_relative_path(&dot_glob.filenames, workspace, "ignored.txt");
        });
    }

    #[test]
    fn explicit_search_paths_can_target_default_skipped_directories() {
        with_temp_workspace("explicit-skipped-search", |workspace| {
            for dir in super::DEFAULT_SEARCH_SKIP_DIRS {
                write_file(&format!("{dir}/explicit.txt"), "explicit needle\n")
                    .expect("explicit skipped fixture write");
            }

            for dir in super::DEFAULT_SEARCH_SKIP_DIRS {
                let grep_output =
                    grep_search(&grep_input("needle", Some(dir), "files_with_matches"))
                        .expect("explicit grep should succeed");
                assert_contains_relative_path(
                    &grep_output.filenames,
                    workspace,
                    &format!("{dir}/explicit.txt"),
                );

                let glob_output =
                    glob_search("**/*.txt", Some(dir)).expect("explicit glob should succeed");
                assert_contains_relative_path(
                    &glob_output.filenames,
                    workspace,
                    &format!("{dir}/explicit.txt"),
                );
            }
        });
    }

    #[test]
    fn grep_rejects_unknown_output_mode() {
        with_temp_workspace("grep-invalid-mode", |_| {
            write_file("demo.txt", "alpha\n").expect("fixture write should succeed");

            let error = grep_search(&grep_input("alpha", None, "bad_mode"))
                .expect_err("invalid output mode should fail");
            assert_error_contains(
                &error,
                "unsupported grep_search output_mode `bad_mode` (expected content, files_with_matches, or count)",
            );
        });
    }

    #[test]
    fn grep_content_offset_counts_files_from_returned_lines() {
        with_temp_workspace("grep-content-offset", |workspace| {
            write_file("a.txt", "alpha one\nalpha two\n").expect("fixture a write");
            write_file("b.txt", "alpha three\n").expect("fixture b write");

            let mut input = grep_input("alpha", None, "content");
            input.head_limit = Some(2);
            input.offset = Some(1);
            let output = grep_search(&input).expect("content grep should succeed");

            assert_eq!(output.mode.as_deref(), Some("content"));
            assert_eq!(output.num_lines, Some(2));
            assert_eq!(output.num_files, 2);
            assert_eq!(output.applied_offset, Some(1));
            assert_eq!(output.applied_limit, None);
            assert_contains_relative_path(&output.filenames, workspace, "a.txt");
            assert_contains_relative_path(&output.filenames, workspace, "b.txt");
            let content = output.content.expect("content output should be present");
            assert!(content.contains("alpha two"));
            assert!(content.contains("alpha three"));
            assert!(!content.contains("alpha one"));
        });
    }

    #[test]
    fn grep_count_and_files_with_matches_apply_file_limits() {
        with_temp_workspace("grep-file-limits", |workspace| {
            write_file("a.txt", "alpha alpha\n").expect("fixture a write");
            write_file("b.txt", "alpha\n").expect("fixture b write");

            let count_output = grep_search(&grep_input("alpha", None, "count"))
                .expect("count grep should succeed");
            assert_eq!(count_output.mode.as_deref(), Some("count"));
            assert_eq!(count_output.num_matches, Some(3));
            assert_contains_relative_path(&count_output.filenames, workspace, "a.txt");
            assert_contains_relative_path(&count_output.filenames, workspace, "b.txt");

            let mut files_input = grep_input("alpha", None, "files_with_matches");
            files_input.head_limit = Some(1);
            files_input.offset = Some(1);
            let files_output = grep_search(&files_input).expect("files grep should succeed");
            assert_eq!(files_output.mode.as_deref(), Some("files_with_matches"));
            assert_eq!(files_output.num_files, 1);
            assert_eq!(files_output.applied_offset, Some(1));
            assert_eq!(files_output.applied_limit, None);
            assert_contains_relative_path(&files_output.filenames, workspace, "b.txt");
            assert_excludes_relative_path(&files_output.filenames, workspace, "a.txt");
        });
    }

    #[test]
    fn globs_and_greps_directory() {
        with_temp_workspace("search-dir", |_| {
            write_file("src/demo.rs", "fn main() {\n println!(\"hello\");\n}\n")
                .expect("file write should succeed");

            let globbed = glob_search("**/*.rs", Some("src")).expect("glob should succeed");
            assert_eq!(globbed.num_files, 1);

            let mut input = grep_input("hello", Some("src"), "content");
            input.glob = Some(String::from("**/*.rs"));
            input.head_limit = Some(10);
            let grep_output = grep_search(&input).expect("grep should succeed");
            assert!(grep_output.content.unwrap_or_default().contains("hello"));
        });
    }
}
