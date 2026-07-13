use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use api::{
    ApiError, ApiService, ContentBlockDelta, ContentBlockDeltaEvent, ContentBlockStartEvent,
    ImageSource, InputContentBlock, InputMessage, MessageDeltaEvent, MessageRequest, NanoGptClient,
    OutputContentBlock, StreamEvent, ToolChoice, ToolDefinition,
};
use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::Mutex;

#[tokio::test]
async fn send_message_posts_json_and_parses_response() {
    let state = Arc::new(Mutex::new(Vec::<CapturedRequest>::new()));
    let body = concat!(
        "{",
        "\"id\":\"msg_test\",",
        "\"type\":\"message\",",
        "\"role\":\"assistant\",",
        "\"content\":[{\"type\":\"text\",\"text\":\"Hello from Pebble\"}],",
        "\"model\":\"openai/gpt-5.2\",",
        "\"stop_reason\":\"end_turn\",",
        "\"stop_sequence\":null,",
        "\"usage\":{\"input_tokens\":12,\"output_tokens\":4},",
        "\"request_id\":\"req_body_123\"",
        "}"
    );
    let server = spawn_server(
        state.clone(),
        vec![http_response("200 OK", "application/json", body)],
    )
    .await;

    let client = NanoGptClient::new("test-key").with_base_url(server.base_url());
    let response = client
        .send_message(&sample_request(false))
        .await
        .expect("request should succeed");

    assert_eq!(response.id, "msg_test");
    assert_eq!(response.total_tokens(), 16);
    assert_eq!(response.request_id.as_deref(), Some("req_body_123"));
    assert_eq!(
        response.content,
        vec![OutputContentBlock::Text {
            text: "Hello from Pebble".to_string(),
        }]
    );

    let captured = state.lock().await;
    let request = captured.first().expect("server should capture request");
    assert_eq!(request.method, "POST");
    assert_eq!(request.path, "/v1/messages");
    assert_eq!(
        request.headers.get("x-api-key").map(String::as_str),
        Some("test-key")
    );
    assert_eq!(
        request.headers.get("authorization").map(String::as_str),
        Some("Bearer test-key")
    );
    let body: serde_json::Value =
        serde_json::from_str(&request.body).expect("request body should be json");
    assert_eq!(
        body.get("model").and_then(serde_json::Value::as_str),
        Some("openai/gpt-5.2")
    );
    assert!(body.get("stream").is_none());
    assert_eq!(body["tools"][0]["name"], json!("get_weather"));
    assert_eq!(body["tool_choice"]["type"], json!("auto"));
}

#[tokio::test]
async fn opencode_go_chat_models_translate_messages_to_chat_completions() {
    let state = Arc::new(Mutex::new(Vec::<CapturedRequest>::new()));
    let body = concat!(
        "{",
        "\"id\":\"chatcmpl_test\",",
        "\"object\":\"chat.completion\",",
        "\"created\":1774656000,",
        "\"choices\":[{",
        "\"index\":0,",
        "\"message\":{",
        "\"role\":\"assistant\",",
        "\"content\":\"Hello from OpenCode Go\",",
        "\"reasoning_content\":\"Internal reasoning\",",
        "\"tool_calls\":[{",
        "\"id\":\"call_123\",",
        "\"type\":\"function\",",
        "\"function\":{\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\":\\\"Paris\\\"}\"}",
        "}]",
        "},",
        "\"finish_reason\":\"tool_calls\"",
        "}],",
        "\"usage\":{\"prompt_tokens\":9,\"completion_tokens\":3,\"total_tokens\":12}",
        "}"
    );
    let server = spawn_server(
        state.clone(),
        vec![http_response_with_headers(
            "200 OK",
            "application/json",
            body,
            &[("request-id", "req_go_123")],
        )],
    )
    .await;

    let client = NanoGptClient::new("go-key")
        .with_service(ApiService::OpencodeGo)
        .with_base_url(server.base_url());
    let response = client
        .send_message(&MessageRequest {
            model: "opencode-go/kimi-k2.6".to_string(),
            max_tokens: 128,
            messages: vec![InputMessage::user_text("hello")],
            system: Some("system".to_string()),
            tools: Some(vec![ToolDefinition {
                name: "get_weather".to_string(),
                description: Some("Get weather".to_string()),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"}
                    }
                }),
            }]),
            tool_choice: Some(ToolChoice::Auto),
            thinking: None,
            reasoning_effort: None,
            fast_mode: false,
            stream: false,
        })
        .await
        .expect("request should succeed");

    assert_eq!(response.request_id.as_deref(), Some("req_go_123"));
    assert_eq!(response.model, "opencode-go/kimi-k2.6");
    assert_eq!(response.usage.input_tokens, 9);
    assert_eq!(response.usage.output_tokens, 3);
    assert_eq!(
        response.content,
        vec![
            OutputContentBlock::Thinking {
                thinking: "Internal reasoning".to_string(),
                signature: None,
            },
            OutputContentBlock::Text {
                text: "Hello from OpenCode Go".to_string(),
            },
            OutputContentBlock::ToolUse {
                id: "call_123".to_string(),
                name: "get_weather".to_string(),
                input: json!({"city":"Paris"}),
            }
        ]
    );

    let captured = state.lock().await;
    let request = captured.first().expect("server should capture request");
    assert_eq!(request.path, "/v1/chat/completions");
    assert_eq!(
        request.headers.get("authorization").map(String::as_str),
        Some("Bearer go-key")
    );
    assert_eq!(
        request.headers.get("x-api-key").map(String::as_str),
        Some("go-key")
    );
    let body: serde_json::Value =
        serde_json::from_str(&request.body).expect("request body should be json");
    assert_eq!(body["model"], json!("kimi-k2.6"));
    assert_eq!(body["messages"][0]["role"], json!("system"));
    assert_eq!(body["messages"][1]["role"], json!("user"));
    assert_eq!(body["messages"][1]["content"], json!("hello"));
    assert_eq!(body["tool_choice"], json!("auto"));
    assert_eq!(body["thinking"]["type"], json!("disabled"));
}

#[tokio::test]
async fn openai_compatible_providers_use_chat_completions_and_bearer_auth() {
    for (service, model, expected_model) in [
        (
            ApiService::Neuralwatt,
            "neuralwatt/zai-org/glm-5.2",
            "zai-org/glm-5.2",
        ),
        (
            ApiService::Lilac,
            "lilac/moonshotai/kimi-k2.6",
            "moonshotai/kimi-k2.6",
        ),
    ] {
        let state = Arc::new(Mutex::new(Vec::<CapturedRequest>::new()));
        let server = spawn_server(
            state.clone(),
            vec![http_response(
                "200 OK",
                "application/json",
                r#"{"id":"chatcmpl_provider","object":"chat.completion","created":1,"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}"#,
            )],
        )
        .await;
        let response = NanoGptClient::new("provider-key")
            .with_service(service)
            .with_base_url(server.base_url())
            .send_message(&MessageRequest {
                model: model.to_string(),
                max_tokens: 32,
                messages: vec![InputMessage::user_text("hello")],
                system: Some("system".to_string()),
                tools: None,
                tool_choice: None,
                thinking: None,
                reasoning_effort: None,
                fast_mode: false,
                stream: false,
            })
            .await
            .expect("provider request should succeed");
        assert_eq!(
            response.content[0],
            OutputContentBlock::Text {
                text: "ok".to_string()
            }
        );

        let captured = state.lock().await;
        let request = captured.first().expect("request should be captured");
        assert_eq!(request.path, "/v1/chat/completions");
        assert_eq!(
            request.headers.get("authorization").map(String::as_str),
            Some("Bearer provider-key")
        );
        assert!(!request.headers.contains_key("x-api-key"));
        let body: serde_json::Value = serde_json::from_str(&request.body).expect("json body");
        assert_eq!(body["model"], json!(expected_model));
    }
}

#[tokio::test]
async fn openai_compatible_streams_text_tools_usage_and_vision_payloads() {
    let state = Arc::new(Mutex::new(Vec::<CapturedRequest>::new()));
    let body = concat!(
        "data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"read_file\",\"arguments\":\"{\\\"path\\\":\"}}]}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"Cargo.toml\\\"}\"}}]}}]}\n\n",
        "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":3,\"total_tokens\":10}}\n\n",
        "data: [DONE]\n\n"
    );
    let server = spawn_server(
        state.clone(),
        vec![http_response("200 OK", "text/event-stream", body)],
    )
    .await;
    let client = NanoGptClient::new("lilac-key")
        .with_service(ApiService::Lilac)
        .with_base_url(server.base_url());
    let mut stream = client
        .stream_message(&MessageRequest {
            model: "lilac/moonshotai/kimi-k2.6".to_string(),
            max_tokens: 64,
            messages: vec![InputMessage {
                role: "user".to_string(),
                content: vec![
                    InputContentBlock::Text {
                        text: "describe".to_string(),
                    },
                    InputContentBlock::Image {
                        source: ImageSource {
                            kind: "base64".to_string(),
                            media_type: "image/png".to_string(),
                            data: "aGVsbG8=".to_string(),
                        },
                    },
                ],
                reasoning_content: None,
                reasoning: None,
            }],
            system: None,
            tools: None,
            tool_choice: None,
            thinking: None,
            reasoning_effort: None,
            fast_mode: false,
            stream: true,
        })
        .await
        .expect("stream should start");
    let mut events = Vec::new();
    while let Some(event) = stream.next_event().await.expect("stream event") {
        events.push(event);
    }
    assert!(events.iter().any(|event| matches!(event, StreamEvent::ContentBlockDelta(delta) if delta.delta == ContentBlockDelta::TextDelta { text: "Hel".to_string() })));
    assert!(events.iter().any(|event| matches!(event, StreamEvent::ContentBlockStart(start) if matches!(&start.content_block, OutputContentBlock::ToolUse { id, name, .. } if id == "call_1" && name == "read_file"))));
    assert!(events.iter().any(|event| matches!(event, StreamEvent::MessageDelta(delta) if delta.usage.input_tokens == 7 && delta.usage.output_tokens == 3)));

    let captured = state.lock().await;
    let request = captured.first().expect("request should be captured");
    let body: serde_json::Value = serde_json::from_str(&request.body).expect("json body");
    assert_eq!(body["stream"], json!(true));
    assert_eq!(body["stream_options"]["include_usage"], json!(true));
    assert_eq!(
        body["messages"][0]["content"][1]["type"],
        json!("image_url")
    );
    assert_eq!(
        body["messages"][0]["content"][1]["image_url"]["url"],
        json!("data:image/png;base64,aGVsbG8=")
    );
}

#[tokio::test]
async fn opencode_go_stream_message_emits_buffered_content_blocks() {
    let state = Arc::new(Mutex::new(Vec::<CapturedRequest>::new()));
    let body = concat!(
        "{",
        "\"id\":\"chatcmpl_test\",",
        "\"object\":\"chat.completion\",",
        "\"created\":1774656000,",
        "\"choices\":[{",
        "\"index\":0,",
        "\"message\":{",
        "\"role\":\"assistant\",",
        "\"content\":\"Hello from OpenCode Go\",",
        "\"tool_calls\":[{",
        "\"id\":\"call_123\",",
        "\"type\":\"function\",",
        "\"function\":{\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\":\\\"Paris\\\"}\"}",
        "}]",
        "},",
        "\"finish_reason\":\"tool_calls\"",
        "}],",
        "\"usage\":{\"prompt_tokens\":9,\"completion_tokens\":3,\"total_tokens\":12}",
        "}"
    );
    let server = spawn_server(
        state,
        vec![http_response("200 OK", "application/json", body)],
    )
    .await;

    let client = NanoGptClient::new("go-key")
        .with_service(ApiService::OpencodeGo)
        .with_base_url(server.base_url());
    let mut stream = client
        .stream_message(&MessageRequest {
            model: "opencode-go/glm-5.1".to_string(),
            max_tokens: 128,
            messages: vec![InputMessage::user_text("hello")],
            system: None,
            tools: None,
            tool_choice: None,
            thinking: None,
            reasoning_effort: None,
            fast_mode: false,
            stream: true,
        })
        .await
        .expect("stream should start");

    let mut events = Vec::new();
    while let Some(event) = stream
        .next_event()
        .await
        .expect("stream event should parse")
    {
        events.push(event);
    }

    assert!(matches!(events[0], StreamEvent::MessageStart(_)));
    assert!(matches!(
        events[1],
        StreamEvent::ContentBlockStart(ContentBlockStartEvent {
            content_block: OutputContentBlock::Text { .. },
            ..
        })
    ));
    assert!(matches!(events[2], StreamEvent::ContentBlockStop(_)));
    assert!(matches!(
        events[3],
        StreamEvent::ContentBlockStart(ContentBlockStartEvent {
            content_block: OutputContentBlock::ToolUse { .. },
            ..
        })
    ));
    assert!(matches!(events[4], StreamEvent::ContentBlockStop(_)));
    assert!(matches!(events[5], StreamEvent::MessageDelta(_)));
    assert!(matches!(events[6], StreamEvent::MessageStop(_)));
}

#[test]
fn image_content_blocks_serialize_with_base64_source() {
    let request = MessageRequest {
        model: "openai/gpt-5.2".to_string(),
        max_tokens: 64,
        messages: vec![InputMessage {
            role: "user".to_string(),
            content: vec![InputContentBlock::Image {
                source: ImageSource {
                    kind: "base64".to_string(),
                    media_type: "image/png".to_string(),
                    data: "AQID".to_string(),
                },
            }],
            reasoning_content: None,
            reasoning: None,
        }],
        system: None,
        tools: None,
        tool_choice: None,
        thinking: None,
        reasoning_effort: None,
        fast_mode: false,
        stream: false,
    };

    let json = serde_json::to_value(request).expect("request should serialize");
    assert_eq!(json["messages"][0]["content"][0]["type"], json!("image"));
    assert_eq!(
        json["messages"][0]["content"][0]["source"],
        json!({
            "type": "base64",
            "media_type": "image/png",
            "data": "AQID"
        })
    );
}

#[tokio::test]
async fn stream_message_parses_sse_events_with_tool_use() {
    let state = Arc::new(Mutex::new(Vec::<CapturedRequest>::new()));
    let sse = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_stream\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"openai/gpt-5.2\",\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":8,\"output_tokens\":0}}}\n\n",
        "event: content_block_start\n",
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_123\",\"name\":\"get_weather\",\"input\":{}}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"city\\\":\\\"Paris\\\"}\"}}\n\n",
        "event: content_block_stop\n",
        "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "event: message_delta\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\",\"stop_sequence\":null},\"usage\":{\"input_tokens\":8,\"output_tokens\":1}}\n\n",
        "event: message_stop\n",
        "data: {\"type\":\"message_stop\"}\n\n",
        "data: [DONE]\n\n"
    );
    let server = spawn_server(
        state.clone(),
        vec![http_response_with_headers(
            "200 OK",
            "text/event-stream",
            sse,
            &[("request-id", "req_stream_456")],
        )],
    )
    .await;

    let client = NanoGptClient::new("test-key").with_base_url(server.base_url());
    let mut stream = client
        .stream_message(&sample_request(false))
        .await
        .expect("stream should start");

    assert_eq!(stream.request_id(), Some("req_stream_456"));

    let mut events = Vec::new();
    while let Some(event) = stream
        .next_event()
        .await
        .expect("stream event should parse")
    {
        events.push(event);
    }

    assert_eq!(events.len(), 6);
    assert!(matches!(events[0], StreamEvent::MessageStart(_)));
    assert!(matches!(
        events[1],
        StreamEvent::ContentBlockStart(ContentBlockStartEvent {
            content_block: OutputContentBlock::ToolUse { .. },
            ..
        })
    ));
    assert!(matches!(
        events[2],
        StreamEvent::ContentBlockDelta(ContentBlockDeltaEvent {
            delta: ContentBlockDelta::InputJsonDelta { .. },
            ..
        })
    ));
    assert!(matches!(events[3], StreamEvent::ContentBlockStop(_)));
    assert!(matches!(
        events[4],
        StreamEvent::MessageDelta(MessageDeltaEvent { .. })
    ));
    assert!(matches!(events[5], StreamEvent::MessageStop(_)));

    match &events[1] {
        StreamEvent::ContentBlockStart(ContentBlockStartEvent {
            content_block: OutputContentBlock::ToolUse { name, input, .. },
            ..
        }) => {
            assert_eq!(name, "get_weather");
            assert_eq!(input, &json!({}));
        }
        other => panic!("expected tool_use block, got {other:?}"),
    }

    let captured = state.lock().await;
    let request = captured.first().expect("server should capture request");
    assert!(request.body.contains("\"stream\":true"));
}

#[tokio::test]
async fn stream_message_surfaces_nanogpt_error_events() {
    let state = Arc::new(Mutex::new(Vec::<CapturedRequest>::new()));
    let sse = concat!(
        "event: error\n",
        "data: {\"type\":\"error\",\"error\":{\"type\":\"invalid_request_error\",\"message\":\"provider override is not valid for this model\"}}\n\n"
    );
    let server = spawn_server(
        state,
        vec![http_response("200 OK", "text/event-stream", sse)],
    )
    .await;

    let client = NanoGptClient::new("test-key").with_base_url(server.base_url());
    let mut stream = client
        .stream_message(&sample_request(false))
        .await
        .expect("stream should start");

    let error = stream
        .next_event()
        .await
        .expect_err("stream error event should fail the stream");
    assert!(matches!(
        error,
        ApiError::StreamApi {
            error_type: Some(ref error_type),
            message: Some(ref message),
            ..
        } if error_type == "invalid_request_error"
            && message == "provider override is not valid for this model"
    ));
}

#[tokio::test]
async fn retries_retryable_failures_before_succeeding() {
    let state = Arc::new(Mutex::new(Vec::<CapturedRequest>::new()));
    let server = spawn_server(
        state.clone(),
        vec![
            http_response(
                "429 Too Many Requests",
                "application/json",
                "{\"type\":\"error\",\"error\":{\"type\":\"rate_limit_error\",\"message\":\"slow down\"}}",
            ),
            http_response(
                "200 OK",
                "application/json",
                "{\"id\":\"msg_retry\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"text\",\"text\":\"Recovered\"}],\"model\":\"openai/gpt-5.2\",\"stop_reason\":\"end_turn\",\"stop_sequence\":null,\"usage\":{\"input_tokens\":3,\"output_tokens\":2}}",
            ),
        ],
    )
    .await;

    let client = NanoGptClient::new("test-key")
        .with_base_url(server.base_url())
        .with_retry_policy(2, Duration::from_millis(1), Duration::from_millis(2));

    let response = client
        .send_message(&sample_request(false))
        .await
        .expect("retry should eventually succeed");

    assert_eq!(response.total_tokens(), 5);
    assert_eq!(state.lock().await.len(), 2);
}

#[tokio::test]
async fn fetch_models_parses_live_nanogpt_catalog_shape() {
    let state = Arc::new(Mutex::new(Vec::<CapturedRequest>::new()));
    let body = concat!(
        "{",
        "\"object\":\"list\",",
        "\"data\":[{",
        "\"id\":\"zai-org/glm-5.1\",",
        "\"object\":\"model\",",
        "\"created\":1774656000,",
        "\"owned_by\":\"zhipu\",",
        "\"name\":\"GLM 5.1\",",
        "\"description\":\"Example\",",
        "\"context_length\":200000,",
        "\"max_output_tokens\":131072,",
        "\"architecture\":{\"modality\":\"text->text\",\"input_modalities\":[\"text\"],\"output_modalities\":[\"text\"]},",
        "\"capabilities\":{\"reasoning\":true,\"tool_calling\":true,\"structured_output\":true},",
        "\"pricing\":{\"prompt\":1.7,\"completion\":5.1,\"currency\":\"USD\",\"unit\":\"per_million_tokens\"},",
        "\"cost_estimate\":0.0034,",
        "\"icon_url\":\"/icons/Zhipu.svg\",",
        "\"subscription\":{\"included\":true,\"note\":\"Included in subscription\"}",
        "}]",
        "}"
    );
    let server = spawn_server(
        state.clone(),
        vec![http_response("200 OK", "application/json", body)],
    )
    .await;

    let client = NanoGptClient::new("test-key").with_base_url(server.base_url());
    let response = client
        .fetch_models(true)
        .await
        .expect("models request should deserialize");

    assert_eq!(response.object, "list");
    assert_eq!(response.data.len(), 1);
    assert_eq!(response.data[0].id, "zai-org/glm-5.1");

    let captured = state.lock().await;
    let request = captured.first().expect("server should capture request");
    assert_eq!(request.method, "GET");
    assert_eq!(request.path, "/v1/models?detailed=true");
}

#[tokio::test]
async fn provider_override_forces_paygo_headers() {
    let state = Arc::new(Mutex::new(Vec::<CapturedRequest>::new()));
    let body = concat!(
        "{",
        "\"id\":\"msg_test\",",
        "\"type\":\"message\",",
        "\"role\":\"assistant\",",
        "\"content\":[{\"type\":\"text\",\"text\":\"ok\"}],",
        "\"model\":\"zai-org/glm-5.1\",",
        "\"stop_reason\":\"end_turn\",",
        "\"stop_sequence\":null,",
        "\"usage\":{\"input_tokens\":1,\"output_tokens\":1}",
        "}"
    );
    let server = spawn_server(
        state.clone(),
        vec![http_response("200 OK", "application/json", body)],
    )
    .await;

    let client = NanoGptClient::new("test-key")
        .with_base_url(server.base_url())
        .with_provider(Some("fireworks".to_string()));
    client
        .send_message(&MessageRequest {
            model: "zai-org/glm-5.1".to_string(),
            max_tokens: 16,
            messages: vec![InputMessage::user_text("hello")],
            system: None,
            tools: None,
            tool_choice: None,
            thinking: None,
            reasoning_effort: None,
            fast_mode: false,
            stream: false,
        })
        .await
        .expect("request should succeed");

    let captured = state.lock().await;
    let request = captured.first().expect("server should capture request");
    assert_eq!(
        request.headers.get("x-provider").map(String::as_str),
        Some("fireworks")
    );
    assert_eq!(
        request.headers.get("x-billing-mode").map(String::as_str),
        Some("paygo")
    );
}

#[tokio::test]
async fn surfaces_retry_exhaustion_for_persistent_retryable_errors() {
    let state = Arc::new(Mutex::new(Vec::<CapturedRequest>::new()));
    let server = spawn_server(
        state.clone(),
        vec![
            http_response(
                "503 Service Unavailable",
                "application/json",
                "{\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"busy\"}}",
            ),
            http_response(
                "503 Service Unavailable",
                "application/json",
                "{\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"still busy\"}}",
            ),
        ],
    )
    .await;

    let client = NanoGptClient::new("test-key")
        .with_base_url(server.base_url())
        .with_retry_policy(1, Duration::from_millis(1), Duration::from_millis(2));

    let error = client
        .send_message(&sample_request(false))
        .await
        .expect_err("persistent 503 should fail");

    match error {
        ApiError::RetriesExhausted {
            attempts,
            last_error,
        } => {
            assert_eq!(attempts, 2);
            assert!(matches!(
                *last_error,
                ApiError::Api {
                    status: reqwest::StatusCode::SERVICE_UNAVAILABLE,
                    retryable: true,
                    ..
                }
            ));
        }
        other => panic!("expected retries exhausted, got {other:?}"),
    }
}

#[tokio::test]
#[ignore = "requires NANOGPT_API_KEY and network access"]
async fn live_stream_smoke_test() {
    let client = NanoGptClient::from_env().expect("NANOGPT_API_KEY must be set");
    let mut stream = client
        .stream_message(&MessageRequest {
            model: std::env::var("NANOGPT_MODEL").unwrap_or_else(|_| "openai/gpt-5.2".to_string()),
            max_tokens: 32,
            messages: vec![InputMessage::user_text(
                "Reply with exactly: hello from rust",
            )],
            system: None,
            tools: None,
            tool_choice: None,
            thinking: None,
            reasoning_effort: None,
            fast_mode: false,
            stream: false,
        })
        .await
        .expect("live stream should start");

    while let Some(_event) = stream
        .next_event()
        .await
        .expect("live stream should yield events")
    {}
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CapturedRequest {
    method: String,
    path: String,
    headers: HashMap<String, String>,
    body: String,
}

struct TestServer {
    base_url: String,
    join_handle: tokio::task::JoinHandle<()>,
}

impl TestServer {
    fn base_url(&self) -> String {
        self.base_url.clone()
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        self.join_handle.abort();
    }
}

async fn spawn_server(
    state: Arc<Mutex<Vec<CapturedRequest>>>,
    responses: Vec<String>,
) -> TestServer {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("listener should bind");
    let address = listener
        .local_addr()
        .expect("listener should have local addr");
    let join_handle = tokio::spawn(async move {
        for response in responses {
            let (mut socket, _) = listener.accept().await.expect("server should accept");
            let mut buffer = Vec::new();
            let mut header_end = None;

            loop {
                let mut chunk = [0_u8; 1024];
                let read = socket
                    .read(&mut chunk)
                    .await
                    .expect("request read should succeed");
                if read == 0 {
                    break;
                }
                buffer.extend_from_slice(&chunk[..read]);
                if let Some(position) = find_header_end(&buffer) {
                    header_end = Some(position);
                    break;
                }
            }

            let header_end = header_end.expect("request should include headers");
            let (header_bytes, remaining) = buffer.split_at(header_end);
            let header_text =
                String::from_utf8(header_bytes.to_vec()).expect("headers should be utf8");
            let mut lines = header_text.split("\r\n");
            let request_line = lines.next().expect("request line should exist");
            let mut parts = request_line.split_whitespace();
            let method = parts.next().expect("method should exist").to_string();
            let path = parts.next().expect("path should exist").to_string();
            let mut headers = HashMap::new();
            let mut content_length = 0_usize;
            for line in lines {
                if line.is_empty() {
                    continue;
                }
                let (name, value) = line.split_once(':').expect("header should have colon");
                let value = value.trim().to_string();
                if name.eq_ignore_ascii_case("content-length") {
                    content_length = value.parse().expect("content length should parse");
                }
                headers.insert(name.to_ascii_lowercase(), value);
            }

            let mut body = remaining[4..].to_vec();
            while body.len() < content_length {
                let mut chunk = vec![0_u8; content_length - body.len()];
                let read = socket
                    .read(&mut chunk)
                    .await
                    .expect("body read should succeed");
                if read == 0 {
                    break;
                }
                body.extend_from_slice(&chunk[..read]);
            }

            state.lock().await.push(CapturedRequest {
                method,
                path,
                headers,
                body: String::from_utf8(body).expect("body should be utf8"),
            });

            socket
                .write_all(response.as_bytes())
                .await
                .expect("response write should succeed");
        }
    });

    TestServer {
        base_url: format!("http://{address}"),
        join_handle,
    }
}

fn find_header_end(bytes: &[u8]) -> Option<usize> {
    bytes.windows(4).position(|window| window == b"\r\n\r\n")
}

fn http_response(status: &str, content_type: &str, body: &str) -> String {
    http_response_with_headers(status, content_type, body, &[])
}

fn http_response_with_headers(
    status: &str,
    content_type: &str,
    body: &str,
    headers: &[(&str, &str)],
) -> String {
    let mut extra_headers = String::new();
    for (name, value) in headers {
        use std::fmt::Write as _;
        write!(&mut extra_headers, "{name}: {value}\r\n").expect("header write should succeed");
    }
    format!(
        "HTTP/1.1 {status}\r\ncontent-type: {content_type}\r\n{extra_headers}content-length: {}\r\nconnection: close\r\n\r\n{body}",
        body.len()
    )
}

fn sample_request(stream: bool) -> MessageRequest {
    MessageRequest {
        model: "openai/gpt-5.2".to_string(),
        max_tokens: 64,
        messages: vec![InputMessage {
            role: "user".to_string(),
            content: vec![
                InputContentBlock::Text {
                    text: "Say hello".to_string(),
                },
                InputContentBlock::ToolResult {
                    tool_use_id: "toolu_prev".to_string(),
                    content: vec![api::ToolResultContentBlock::Json {
                        value: json!({"forecast": "sunny"}),
                    }],
                    is_error: false,
                },
            ],
            reasoning_content: None,
            reasoning: None,
        }],
        system: Some("Use tools when needed".to_string()),
        tools: Some(vec![ToolDefinition {
            name: "get_weather".to_string(),
            description: Some("Fetches the weather".to_string()),
            input_schema: json!({
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }),
        }]),
        tool_choice: Some(ToolChoice::Auto),
        thinking: None,
        reasoning_effort: None,
        fast_mode: false,
        stream,
    }
}
