use std::collections::BTreeMap;

use runtime::{
    get_compact_continuation_message, get_tool_result_context_output, ContentBlock,
    ConversationMessage, MessageRole, PermissionMode,
};
use serde_json::{Map, Number, Value};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeToolSpec {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
    pub required_permission: PermissionMode,
}

impl From<tools::ToolSpec> for RuntimeToolSpec {
    fn from(value: tools::ToolSpec) -> Self {
        Self {
            name: value.name.to_string(),
            description: value.description.to_string(),
            input_schema: value.input_schema,
            required_permission: value.required_permission,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProxyCommand {
    Toggle,
    Enable,
    Disable,
    Status,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProxySegment {
    Text(String),
    ToolUse {
        id: String,
        name: String,
        input: String,
    },
}

pub fn parse_proxy_value(value: Option<&str>) -> Result<ProxyCommand, String> {
    match value.map(str::trim).filter(|value| !value.is_empty()) {
        None => Ok(ProxyCommand::Toggle),
        Some("on" | "enable" | "enabled") => Ok(ProxyCommand::Enable),
        Some("off" | "disable" | "disabled") => Ok(ProxyCommand::Disable),
        Some("status") => Ok(ProxyCommand::Status),
        Some(other) => Err(format!(
            "proxy accepts one optional argument: on, off, or status (got {other})"
        )),
    }
}

pub fn build_proxy_system_prompt(tool_specs: &[RuntimeToolSpec]) -> String {
    let mut lines = vec![
        "# XML Tool Proxy".to_string(),
        "Native tool calling is disabled for this session.".to_string(),
        "When you need a tool, emit XML <tool_call> blocks instead of native or JSON tool calls."
            .to_string(),
        "If tool use is needed, do not narrate your intent first. Start your next response with the <tool_call> block immediately.".to_string(),
        "Do not say things like \"I'll inspect the project\" or \"Let me explore\" before the tool call.".to_string(),
        "Use this exact shape:".to_string(),
        "<tool_call name=\"read_file\">".to_string(),
        "  <arg name=\"path\">src/main.rs</arg>".to_string(),
        "  <arg name=\"offset\" type=\"integer\">0</arg>".to_string(),
        "</tool_call>".to_string(),
        "Rules:".to_string(),
        " - Use one <tool_call> block per tool invocation.".to_string(),
        " - Put arguments in <arg name=\"...\">value</arg> children.".to_string(),
        " - Add type=\"integer\", type=\"number\", type=\"boolean\", or type=\"json\" when the value is not a plain string.".to_string(),
        " - Escape XML special characters inside argument values.".to_string(),
        " - Do not emit native tool calls or JSON function-call envelopes.".to_string(),
        " - Tool results will come back as <tool_result ...>...</tool_result> blocks in later user messages.".to_string(),
        String::new(),
        "Available tools:".to_string(),
    ];

    for spec in tool_specs {
        lines.push(format!(" - {}: {}", spec.name, spec.description));
        if let Some(properties) = spec
            .input_schema
            .get("properties")
            .and_then(Value::as_object)
        {
            for (name, property) in properties {
                let type_name = schema_type_name(property).unwrap_or("string");
                lines.push(format!("   - arg `{name}` ({type_name})"));
            }
        }
    }

    lines.join("\n")
}

pub fn convert_messages_for_proxy(
    messages: &[ConversationMessage],
) -> Result<Vec<ProxyMessage>, String> {
    let mut converted = Vec::new();

    for message in messages {
        let role = match message.role {
            MessageRole::Assistant => "assistant",
            MessageRole::System | MessageRole::User | MessageRole::Tool => "user",
        };
        let content = render_proxy_message_content(message)?;
        if !content.trim().is_empty() {
            converted.push(ProxyMessage {
                role: role.to_string(),
                content,
            });
        }
    }

    Ok(converted)
}

pub fn parse_proxy_response(
    text: &str,
    tool_specs: &[RuntimeToolSpec],
) -> Result<Vec<ProxySegment>, String> {
    let mut segments = Vec::new();
    let mut cursor = 0;
    let mut ordinal = 0usize;

    while let Some((start, block_kind)) = find_next_proxy_block(text, cursor, tool_specs) {
        if start > cursor {
            segments.push(ProxySegment::Text(text[cursor..start].to_string()));
        }
        let open_end = if let Ok(open_end) = find_tag_end(text, start) {
            open_end
        } else {
            segments.push(ProxySegment::Text(text[start..].to_string()));
            return Ok(segments);
        };
        let Some((block, consumed_end)) =
            recover_proxy_block(text, start, open_end, &block_kind, tool_specs)
        else {
            segments.push(ProxySegment::Text(text[start..].to_string()));
            return Ok(segments);
        };
        ordinal += 1;
        match parse_proxy_block(&block, &block_kind, tool_specs, ordinal) {
            Ok(Some(segment)) => segments.push(segment),
            Ok(None) => {}
            Err(_) => segments.push(ProxySegment::Text(block)),
        }
        cursor = consumed_end;
    }

    if cursor < text.len() {
        segments.push(ProxySegment::Text(text[cursor..].to_string()));
    }

    Ok(segments)
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ProxyBlockKind {
    ToolCall,
    ToolResult,
    DirectTool(String),
}

fn find_next_proxy_block(
    text: &str,
    cursor: usize,
    tool_specs: &[RuntimeToolSpec],
) -> Option<(usize, ProxyBlockKind)> {
    let mut best =
        find_tag_start(text, cursor, "tool_call").map(|start| (start, ProxyBlockKind::ToolCall));
    if let Some(start) = find_tag_start(text, cursor, "tool_result") {
        match &best {
            Some((best_start, _)) if *best_start <= start => {}
            _ => best = Some((start, ProxyBlockKind::ToolResult)),
        }
    }
    for tool_spec in tool_specs {
        if let Some(start) = find_tag_start(text, cursor, &tool_spec.name) {
            let candidate = (start, ProxyBlockKind::DirectTool(tool_spec.name.clone()));
            match &best {
                Some((best_start, _)) if *best_start <= start => {}
                _ => best = Some(candidate),
            }
        }
    }
    best
}

fn find_tag_start(text: &str, cursor: usize, tag_name: &str) -> Option<usize> {
    let needle = format!("<{tag_name}");
    let mut search_from = cursor;
    while let Some(offset) = text[search_from..].find(&needle) {
        let start = search_from + offset;
        let next_char = text[start + needle.len()..].chars().next();
        if next_char.is_none_or(|ch| ch.is_whitespace() || ch == '>') {
            return Some(start);
        }
        search_from = start + needle.len();
    }
    None
}

fn recover_proxy_block(
    text: &str,
    start: usize,
    open_end: usize,
    block_kind: &ProxyBlockKind,
    tool_specs: &[RuntimeToolSpec],
) -> Option<(String, usize)> {
    let tag_name = match block_kind {
        ProxyBlockKind::ToolCall => "tool_call",
        ProxyBlockKind::ToolResult => "tool_result",
        ProxyBlockKind::DirectTool(name) => name.as_str(),
    };
    recover_named_block(text, start, open_end, tag_name, tool_specs)
}

fn recover_named_block(
    text: &str,
    start: usize,
    open_end: usize,
    tag_name: &str,
    tool_specs: &[RuntimeToolSpec],
) -> Option<(String, usize)> {
    let exact_close = format!("</{tag_name}>");
    if let Some(close_start) = text[open_end..]
        .find(&exact_close)
        .map(|offset| open_end + offset)
    {
        let consumed_end = close_start + exact_close.len();
        return Some((text[start..consumed_end].to_string(), consumed_end));
    }

    let truncated_close = format!("</{tag_name}");
    if let Some(truncated_close_start) = text[open_end..]
        .find(&truncated_close)
        .map(|offset| open_end + offset)
    {
        let next_block_start = find_next_proxy_block_start(
            text,
            truncated_close_start + truncated_close.len(),
            tool_specs,
        );
        let consumed_end = next_block_start.unwrap_or(text.len());
        let malformed_close = text[truncated_close_start..consumed_end].trim_end();
        let remainder = &malformed_close[truncated_close.len()..];
        if !remainder.contains('<') {
            return Some((
                format!("{}{}</{tag_name}>", &text[start..truncated_close_start], ""),
                consumed_end,
            ));
        }
    }

    if let Some(next_block_start) = find_next_proxy_block_start(text, open_end, tool_specs) {
        let trailing = text[open_end..next_block_start].trim_end();
        if let Some(arg_end) = trailing.rfind("</arg>") {
            let suffix = trailing[arg_end + "</arg>".len()..].trim();
            if is_recoverable_suffix_after_last_arg(suffix) {
                return Some((
                    format!("{}{}</{tag_name}>", &text[start..next_block_start], ""),
                    next_block_start,
                ));
            }
        }
    } else {
        let trailing = text[open_end..].trim_end();
        if let Some(arg_end) = trailing.rfind("</arg>") {
            let suffix = trailing[arg_end + "</arg>".len()..].trim();
            if is_recoverable_suffix_after_last_arg(suffix) {
                return Some((
                    format!("{}{}</{tag_name}>", &text[start..text.len()], ""),
                    text.len(),
                ));
            }
        }
    }
    None
}

fn is_arg_trailing_noise(input: &str) -> bool {
    if input.is_empty() {
        return true;
    }
    if input.contains('<') {
        return false;
    }
    input.chars().all(|ch| {
        ch.is_whitespace()
            || matches!(
                ch,
                '(' | ')'
                    | '['
                    | ']'
                    | '{'
                    | '}'
                    | '>'
                    | '〉'
                    | '＞'
                    | ','
                    | '.'
                    | ':'
                    | ';'
                    | '!'
                    | '?'
                    | '_'
                    | '-'
                    | '/'
                    | '\\'
                    | '|'
            )
    })
}

fn find_next_proxy_block_start(
    text: &str,
    cursor: usize,
    tool_specs: &[RuntimeToolSpec],
) -> Option<usize> {
    find_next_proxy_block(text, cursor, tool_specs).map(|(start, _)| start)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProxyMessage {
    pub role: String,
    pub content: String,
}

fn render_proxy_message_content(message: &ConversationMessage) -> Result<String, String> {
    let mut chunks = Vec::new();
    for block in &message.blocks {
        match block {
            ContentBlock::Text { text } => chunks.push(text.clone()),
            ContentBlock::Thinking { .. } => {}
            ContentBlock::ToolUse { id, name, input } => {
                chunks.push(render_tool_call_xml(id, name, input)?);
            }
            ContentBlock::ToolResult {
                tool_use_id,
                tool_name,
                output,
                is_error,
                compacted,
                ..
            } => chunks.push(render_tool_result_xml(
                tool_use_id,
                tool_name,
                get_tool_result_context_output(output, *compacted).as_ref(),
                *is_error,
            )),
            ContentBlock::CompactionSummary {
                summary,
                recent_messages_preserved,
                ..
            } => chunks.push(get_compact_continuation_message(
                summary,
                true,
                *recent_messages_preserved,
            )),
        }
    }
    Ok(chunks.join("\n\n"))
}

fn render_tool_call_xml(id: &str, name: &str, input: &str) -> Result<String, String> {
    let parsed_input = serde_json::from_str::<Value>(input)
        .map_err(|error| format!("invalid tool input JSON: {error}"))?;
    let Value::Object(object) = parsed_input else {
        return Err("tool input JSON must be an object".to_string());
    };

    let mut xml = format!(
        "<tool_call name=\"{}\" id=\"{}\">",
        escape_xml_attr(name),
        escape_xml_attr(id)
    );
    if !object.is_empty() {
        xml.push('\n');
    }
    for (key, value) in object {
        xml.push_str("  <arg name=\"");
        xml.push_str(&escape_xml_attr(&key));
        xml.push('"');
        if let Some(type_name) = json_type_name(&value) {
            xml.push_str(" type=\"");
            xml.push_str(type_name);
            xml.push('"');
        }
        xml.push('>');
        xml.push_str(&escape_xml_text(&json_value_as_xml_text(&value)));
        xml.push_str("</arg>\n");
    }
    xml.push_str("</tool_call>");
    Ok(xml)
}

fn render_tool_result_xml(
    tool_use_id: &str,
    tool_name: &str,
    output: &str,
    is_error: bool,
) -> String {
    format!(
        "<tool_result id=\"{}\" name=\"{}\" error=\"{}\">{}</tool_result>",
        escape_xml_attr(tool_use_id),
        escape_xml_attr(tool_name),
        if is_error { "true" } else { "false" },
        escape_xml_text(output)
    )
}

fn parse_proxy_block(
    block: &str,
    block_kind: &ProxyBlockKind,
    tool_specs: &[RuntimeToolSpec],
    ordinal: usize,
) -> Result<Option<ProxySegment>, String> {
    match block_kind {
        ProxyBlockKind::ToolCall => parse_tool_call_block(block, tool_specs, ordinal).map(Some),
        ProxyBlockKind::ToolResult => Ok(None),
        ProxyBlockKind::DirectTool(name) => {
            parse_direct_tool_block(block, name, tool_specs, ordinal).map(Some)
        }
    }
}

fn parse_tool_call_block(
    block: &str,
    tool_specs: &[RuntimeToolSpec],
    ordinal: usize,
) -> Result<ProxySegment, String> {
    let open_end = find_tag_end(block, 0)?;
    let open_tag = &block["<tool_call".len()..open_end - 1];
    let attributes = parse_attributes(open_tag)?;
    let name = attributes
        .get("name")
        .cloned()
        .ok_or_else(|| "tool_call is missing name=\"...\"".to_string())?;
    let id = attributes
        .get("id")
        .cloned()
        .unwrap_or_else(|| format!("proxy-tool-{ordinal}"));
    let tool_spec = tool_specs
        .iter()
        .find(|spec| spec.name == name)
        .ok_or_else(|| format!("unknown proxy tool: {name}"))?;
    let body = &block[open_end..block.len() - "</tool_call>".len()];
    let input = parse_tool_args(body, tool_spec)?;
    Ok(ProxySegment::ToolUse {
        id,
        name,
        input: Value::Object(input).to_string(),
    })
}

fn parse_direct_tool_block(
    block: &str,
    tool_name: &str,
    tool_specs: &[RuntimeToolSpec],
    ordinal: usize,
) -> Result<ProxySegment, String> {
    let open_end = find_tag_end(block, 0)?;
    let open_tag = &block[1 + tool_name.len()..open_end - 1];
    let attributes = parse_attributes(open_tag)?;
    let id = attributes
        .get("id")
        .cloned()
        .unwrap_or_else(|| format!("proxy-tool-{ordinal}"));
    let tool_spec = tool_specs
        .iter()
        .find(|spec| spec.name == tool_name)
        .ok_or_else(|| format!("unknown proxy tool: {tool_name}"))?;
    let close_tag = format!("</{tool_name}>");
    let body = &block[open_end..block.len() - close_tag.len()];
    let input = parse_tool_args(body, tool_spec)?;
    Ok(ProxySegment::ToolUse {
        id,
        name: tool_name.to_string(),
        input: Value::Object(input).to_string(),
    })
}

fn parse_tool_args(body: &str, tool_spec: &RuntimeToolSpec) -> Result<Map<String, Value>, String> {
    let mut result = Map::new();
    let mut cursor = 0;

    while cursor < body.len() {
        let remainder = &body[cursor..];
        let trimmed = remainder.trim_start();
        cursor += remainder.len() - trimmed.len();
        if trimmed.is_empty() {
            break;
        }
        if !trimmed.starts_with("<arg") {
            if let Some(consumed) = orphan_closing_tag_len(trimmed) {
                cursor += consumed;
                continue;
            }
            if is_arg_trailing_noise(trimmed) {
                break;
            }
            return Err("proxy tool_call body may only contain <arg> children".to_string());
        }

        let arg_start = cursor;
        let open_end = find_tag_end(body, arg_start)?;
        let open_tag = &body[arg_start + "<arg".len()..open_end - 1];
        let attributes = parse_attributes(open_tag)?;
        let name = attributes
            .get("name")
            .cloned()
            .ok_or_else(|| "arg is missing name=\"...\"".to_string())?;
        let declared_type = attributes.get("type").map(String::as_str);
        let close_start = body[open_end..]
            .find("</arg>")
            .map(|offset| open_end + offset)
            .ok_or_else(|| format!("unterminated <arg> for {name}"))?;
        let raw_value = unescape_xml_text(&body[open_end..close_start])?;
        let value = coerce_proxy_arg_value(tool_spec, &name, declared_type, raw_value.trim())?;
        result.insert(name, value);
        cursor = close_start + "</arg>".len();
    }

    Ok(result)
}

fn is_recoverable_suffix_after_last_arg(input: &str) -> bool {
    let mut remainder = input.trim();
    loop {
        if remainder.is_empty() || is_arg_trailing_noise(remainder) {
            return true;
        }
        let Some(consumed) = orphan_closing_tag_len(remainder) else {
            return false;
        };
        remainder = remainder[consumed..].trim_start();
    }
}

fn orphan_closing_tag_len(input: &str) -> Option<usize> {
    if !input.starts_with("</") {
        return None;
    }
    input.find('>').map(|index| index + 1)
}

fn coerce_proxy_arg_value(
    tool_spec: &RuntimeToolSpec,
    name: &str,
    declared_type: Option<&str>,
    raw: &str,
) -> Result<Value, String> {
    let inferred_type =
        declared_type.or_else(|| schema_property_type_name(&tool_spec.input_schema, name));
    match inferred_type {
        Some("integer") => raw
            .parse::<i64>()
            .map(Number::from)
            .map(Value::Number)
            .map_err(|error| format!("invalid integer for {name}: {error}")),
        Some("number") => {
            let parsed = raw
                .parse::<f64>()
                .map_err(|error| format!("invalid number for {name}: {error}"))?;
            let number = Number::from_f64(parsed)
                .ok_or_else(|| format!("invalid finite number for {name}"))?;
            Ok(Value::Number(number))
        }
        Some("boolean") => raw
            .parse::<bool>()
            .map(Value::Bool)
            .map_err(|error| format!("invalid boolean for {name}: {error}")),
        Some("json" | "object" | "array") => {
            serde_json::from_str(raw).map_err(|error| format!("invalid JSON for {name}: {error}"))
        }
        _ => Ok(Value::String(raw.to_string())),
    }
}

fn schema_property_type_name<'a>(schema: &'a Value, name: &str) -> Option<&'a str> {
    schema
        .get("properties")
        .and_then(Value::as_object)
        .and_then(|properties| properties.get(name))
        .and_then(schema_type_name)
}

fn schema_type_name(schema: &Value) -> Option<&str> {
    match schema.get("type") {
        Some(Value::String(kind)) => Some(kind.as_str()),
        Some(Value::Array(items)) => items.iter().find_map(Value::as_str),
        _ => None,
    }
}

fn json_type_name(value: &Value) -> Option<&'static str> {
    match value {
        Value::Null => Some("json"),
        Value::Bool(_) => Some("boolean"),
        Value::Number(number) if number.is_i64() || number.is_u64() => Some("integer"),
        Value::Number(_) => Some("number"),
        Value::String(_) => None,
        Value::Array(_) | Value::Object(_) => Some("json"),
    }
}

fn json_value_as_xml_text(value: &Value) -> String {
    match value {
        Value::Null => "null".to_string(),
        Value::Bool(value) => value.to_string(),
        Value::Number(value) => value.to_string(),
        Value::String(value) => value.clone(),
        Value::Array(_) | Value::Object(_) => value.to_string(),
    }
}

fn find_tag_end(input: &str, start: usize) -> Result<usize, String> {
    input[start..]
        .find('>')
        .map(|offset| start + offset + 1)
        .ok_or_else(|| "unterminated XML tag".to_string())
}

fn parse_attributes(input: &str) -> Result<BTreeMap<String, String>, String> {
    let mut attributes = BTreeMap::new();
    let bytes = input.as_bytes();
    let mut index = 0;

    while index < bytes.len() {
        while index < bytes.len() && bytes[index].is_ascii_whitespace() {
            index += 1;
        }
        if index >= bytes.len() {
            break;
        }

        let key_start = index;
        while index < bytes.len() && !bytes[index].is_ascii_whitespace() && bytes[index] != b'=' {
            index += 1;
        }
        let key = input[key_start..index].trim();
        while index < bytes.len() && bytes[index].is_ascii_whitespace() {
            index += 1;
        }
        if index >= bytes.len() || bytes[index] != b'=' {
            return Err(format!("invalid XML attribute near {key}"));
        }
        index += 1;
        while index < bytes.len() && bytes[index].is_ascii_whitespace() {
            index += 1;
        }
        if index >= bytes.len() || bytes[index] != b'"' {
            return Err(format!("XML attribute {key} must use double quotes"));
        }
        index += 1;
        let value_start = index;
        while index < bytes.len() && bytes[index] != b'"' {
            index += 1;
        }
        if index >= bytes.len() {
            return Err(format!("unterminated XML attribute {key}"));
        }
        let value = unescape_xml_text(&input[value_start..index])?;
        attributes.insert(key.to_string(), value);
        index += 1;
    }

    Ok(attributes)
}

fn escape_xml_attr(value: &str) -> String {
    escape_xml_text(value)
}

fn escape_xml_text(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

fn unescape_xml_text(value: &str) -> Result<String, String> {
    let mut output = String::new();
    let mut cursor = 0;

    while let Some(rel) = value[cursor..].find('&') {
        let start = cursor + rel;
        output.push_str(&value[cursor..start]);
        let tail = &value[start..];
        if let Some(rest) = tail.strip_prefix("&lt;") {
            output.push('<');
            cursor = value.len() - rest.len();
        } else if let Some(rest) = tail.strip_prefix("&gt;") {
            output.push('>');
            cursor = value.len() - rest.len();
        } else if let Some(rest) = tail.strip_prefix("&amp;") {
            output.push('&');
            cursor = value.len() - rest.len();
        } else if let Some(rest) = tail.strip_prefix("&quot;") {
            output.push('"');
            cursor = value.len() - rest.len();
        } else if let Some(rest) = tail.strip_prefix("&apos;") {
            output.push('\'');
            cursor = value.len() - rest.len();
        } else {
            return Err("unsupported XML entity in proxy tool call".to_string());
        }
    }

    output.push_str(&value[cursor..]);
    Ok(output)
}

#[cfg(test)]
mod tests {
    use runtime::{ContentBlock, ConversationMessage};
    use serde_json::json;

    use super::{
        build_proxy_system_prompt, convert_messages_for_proxy, parse_proxy_response,
        parse_proxy_value, ProxyCommand, ProxySegment, RuntimeToolSpec,
    };

    fn specs() -> Vec<RuntimeToolSpec> {
        tools::mvp_tool_specs()
            .into_iter()
            .map(RuntimeToolSpec::from)
            .collect()
    }

    #[test]
    fn parses_proxy_command_values() {
        assert_eq!(
            parse_proxy_value(None).expect("toggle"),
            ProxyCommand::Toggle
        );
        assert_eq!(
            parse_proxy_value(Some("on")).expect("enable"),
            ProxyCommand::Enable
        );
        assert_eq!(
            parse_proxy_value(Some("off")).expect("disable"),
            ProxyCommand::Disable
        );
        assert_eq!(
            parse_proxy_value(Some("status")).expect("status"),
            ProxyCommand::Status
        );
    }

    #[test]
    fn renders_proxy_tool_protocol_prompt() {
        let prompt = build_proxy_system_prompt(&specs());
        assert!(prompt.contains("# XML Tool Proxy"));
        assert!(prompt.contains("<tool_call name=\"read_file\">"));
        assert!(prompt.contains("bash: Execute a shell command"));
    }

    #[test]
    fn converts_session_messages_to_proxy_text() {
        let messages = vec![
            ConversationMessage::assistant(vec![ContentBlock::ToolUse {
                id: "tool-1".to_string(),
                name: "bash".to_string(),
                input: json!({"command":"pwd","timeout":1}).to_string(),
            }]),
            ConversationMessage::tool_result("tool-1", "bash", "{\"ok\":true}", false),
        ];

        let converted =
            convert_messages_for_proxy(&messages).expect("proxy conversion should work");
        assert_eq!(converted.len(), 2);
        assert!(converted[0]
            .content
            .contains("<tool_call name=\"bash\" id=\"tool-1\">"));
        assert!(converted[0]
            .content
            .contains("<arg name=\"command\">pwd</arg>"));
        assert!(converted[1]
            .content
            .contains("<tool_result id=\"tool-1\" name=\"bash\" error=\"false\">"));
    }

    #[test]
    fn renders_compacted_tool_results_as_placeholder() {
        let messages = vec![ConversationMessage::compacted_tool_result(
            "tool-1",
            "bash",
            "sensitive output",
            false,
        )];

        let converted =
            convert_messages_for_proxy(&messages).expect("proxy conversion should work");

        assert_eq!(converted.len(), 1);
        assert!(converted[0]
            .content
            .contains("[Old tool result content cleared]"));
        assert!(!converted[0].content.contains("sensitive output"));
    }

    #[test]
    fn parses_proxy_tool_call_blocks_into_tool_use_segments() {
        let text = "I will inspect that.\n\n<tool_call name=\"read_file\" id=\"call-1\">\n  <arg name=\"path\">src/main.rs</arg>\n  <arg name=\"offset\" type=\"integer\">0</arg>\n</tool_call>";
        let segments = parse_proxy_response(text, &specs()).expect("proxy response should parse");
        assert_eq!(
            segments,
            vec![
                ProxySegment::Text("I will inspect that.\n\n".to_string()),
                ProxySegment::ToolUse {
                    id: "call-1".to_string(),
                    name: "read_file".to_string(),
                    input: json!({"path":"src/main.rs","offset":0}).to_string(),
                }
            ]
        );
    }

    #[test]
    fn treats_unterminated_tool_call_blocks_as_plain_text() {
        let text =
            "I will inspect that.\n\n<tool_call name=\"read_file\"><arg name=\"path\">src/main.rs</arg>";
        let segments = parse_proxy_response(text, &specs()).expect("proxy response should parse");
        assert_eq!(
            segments,
            vec![
                ProxySegment::Text("I will inspect that.\n\n".to_string()),
                ProxySegment::ToolUse {
                    id: "proxy-tool-1".to_string(),
                    name: "read_file".to_string(),
                    input: json!({"path":"src/main.rs"}).to_string(),
                },
            ]
        );
    }

    #[test]
    fn recovers_tool_call_when_closing_tag_is_missing_final_bracket() {
        let text = "<tool_call name=\"bash\">\n  <arg name=\"command\">pwd</arg>\n</tool_call";
        let segments = parse_proxy_response(text, &specs()).expect("proxy response should parse");
        assert_eq!(
            segments,
            vec![ProxySegment::ToolUse {
                id: "proxy-tool-1".to_string(),
                name: "bash".to_string(),
                input: json!({"command":"pwd"}).to_string(),
            }]
        );
    }

    #[test]
    fn recovers_tool_call_when_closing_tag_has_extra_characters() {
        let text = "<tool_call name=\"bash\">\n  <arg name=\"command\">pwd</arg>\n</tool_call_>";
        let segments = parse_proxy_response(text, &specs()).expect("proxy response should parse");
        assert_eq!(
            segments,
            vec![ProxySegment::ToolUse {
                id: "proxy-tool-1".to_string(),
                name: "bash".to_string(),
                input: json!({"command":"pwd"}).to_string(),
            }]
        );
    }

    #[test]
    fn recovers_tool_call_when_closing_tag_uses_slash_before_bracket() {
        let text = "<tool_call name=\"write_file\">\n  <arg name=\"path\">test.md</arg>\n  <arg name=\"content\">hello</arg>\n</tool_call />";
        let segments = parse_proxy_response(text, &specs()).expect("proxy response should parse");
        assert_eq!(
            segments,
            vec![ProxySegment::ToolUse {
                id: "proxy-tool-1".to_string(),
                name: "write_file".to_string(),
                input: json!({"path":"test.md","content":"hello"}).to_string(),
            }]
        );
    }

    #[test]
    fn recovers_tool_call_when_closing_tag_has_named_suffix() {
        let text =
            "<tool_call name=\"read_file\" id=\"proxy-tool-3\">\n  <arg name=\"path\">README.md</arg>\n</tool_call_Name>";
        let segments = parse_proxy_response(text, &specs()).expect("proxy response should parse");
        assert_eq!(
            segments,
            vec![ProxySegment::ToolUse {
                id: "proxy-tool-3".to_string(),
                name: "read_file".to_string(),
                input: json!({"path":"README.md"}).to_string(),
            }]
        );
    }

    #[test]
    fn recovers_multiple_tool_calls_with_unicode_close_tag() {
        let text = "<tool_call name=\"read_file\" id=\"proxy-tool-2\">\n  <arg name=\"path\">Cargo.toml</arg>\n</tool_call〉 <tool_call name=\"read_file\" id=\"proxy-tool-3\">\n  <arg name=\"path\">crates/pebble/src/main.rs</arg>\n</tool_call〉";
        let segments = parse_proxy_response(text, &specs()).expect("proxy response should parse");
        assert_eq!(
            segments,
            vec![
                ProxySegment::ToolUse {
                    id: "proxy-tool-2".to_string(),
                    name: "read_file".to_string(),
                    input: json!({"path":"Cargo.toml"}).to_string(),
                },
                ProxySegment::ToolUse {
                    id: "proxy-tool-3".to_string(),
                    name: "read_file".to_string(),
                    input: json!({"path":"crates/pebble/src/main.rs"}).to_string(),
                },
            ]
        );
    }

    #[test]
    fn parses_direct_tool_tag_blocks_into_tool_use_segments() {
        let text = "<read_file>\n<arg name=\"path\">Cargo.toml</arg>\n</read_file>";
        let segments = parse_proxy_response(text, &specs()).expect("proxy response should parse");
        assert_eq!(
            segments,
            vec![ProxySegment::ToolUse {
                id: "proxy-tool-1".to_string(),
                name: "read_file".to_string(),
                input: json!({"path":"Cargo.toml"}).to_string(),
            }]
        );
    }

    #[test]
    fn recovers_multiple_tool_calls_when_separator_noise_replaces_close_tag() {
        let text = "<tool_call name=\"read_file\" id=\"proxy-tool-10\">\n  <arg name=\"limit\" type=\"integer\">50</arg>\n  <arg name=\"offset\" type=\"integer\">80</arg>\n  <arg name=\"path\">crates/pebble/src/main.rs</arg>\n()>\n<tool_call name=\"glob_search\" id=\"proxy-tool-11\">\n  <arg name=\"pattern\">crates/api/src/**/*.rs</arg>\n()>";
        let segments = parse_proxy_response(text, &specs()).expect("proxy response should parse");
        assert_eq!(
            segments,
            vec![
                ProxySegment::ToolUse {
                    id: "proxy-tool-10".to_string(),
                    name: "read_file".to_string(),
                    input: json!({
                        "limit": 50,
                        "offset": 80,
                        "path": "crates/pebble/src/main.rs"
                    })
                    .to_string(),
                },
                ProxySegment::ToolUse {
                    id: "proxy-tool-11".to_string(),
                    name: "glob_search".to_string(),
                    input: json!({"pattern":"crates/api/src/**/*.rs"}).to_string(),
                },
            ]
        );
    }

    #[test]
    fn ignores_orphan_arg_closers_inside_tool_call_body() {
        let text = "<tool_call name=\"write_file\">\n  <arg name=\"path\">pebble-project-summary.md</arg>\n  <arg name=\"content\">hello</arg>\n</arg>\n</tool_call>";
        let segments = parse_proxy_response(text, &specs()).expect("proxy response should parse");
        assert_eq!(
            segments,
            vec![ProxySegment::ToolUse {
                id: "proxy-tool-1".to_string(),
                name: "write_file".to_string(),
                input: json!({
                    "path": "pebble-project-summary.md",
                    "content": "hello"
                })
                .to_string(),
            }]
        );
    }

    #[test]
    fn ignores_orphan_non_arg_closers_inside_tool_call_body() {
        let text =
            "<tool_call name=\"read_file\">\n  <arg name=\"path\">readme.md</arg>\n</parameter>\n</tool_call>";
        let segments = parse_proxy_response(text, &specs()).expect("proxy response should parse");
        assert_eq!(
            segments,
            vec![ProxySegment::ToolUse {
                id: "proxy-tool-1".to_string(),
                name: "read_file".to_string(),
                input: json!({"path":"readme.md"}).to_string(),
            }]
        );
    }

    #[test]
    fn ignores_assistant_emitted_tool_result_blocks() {
        let text = "<tool_call name=\"write_file\"><arg name=\"path\">test.md</arg><arg name=\"content\">hello</arg></tool_call><tool_result id=\"proxy-tool-1\" name=\"write_file\" error=\"false\">{\"path\":\"test.md\"}</tool_result>\nDone.";
        let segments = parse_proxy_response(text, &specs()).expect("proxy response should parse");
        assert_eq!(
            segments,
            vec![
                ProxySegment::ToolUse {
                    id: "proxy-tool-1".to_string(),
                    name: "write_file".to_string(),
                    input: json!({"path":"test.md","content":"hello"}).to_string(),
                },
                ProxySegment::Text("\nDone.".to_string()),
            ]
        );
    }

    #[test]
    fn recovers_tool_call_with_orphan_non_arg_closer_and_missing_tool_close() {
        let text =
            "<tool_call name=\"read_file\">\n  <arg name=\"path\">README.md</arg>\n</parameter>";
        let segments = parse_proxy_response(text, &specs()).expect("proxy response should parse");
        assert_eq!(
            segments,
            vec![ProxySegment::ToolUse {
                id: "proxy-tool-1".to_string(),
                name: "read_file".to_string(),
                input: json!({"path":"README.md"}).to_string(),
            }]
        );
    }
}
