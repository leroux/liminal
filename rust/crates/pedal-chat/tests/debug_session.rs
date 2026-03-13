//! Debug test: use claudewire CliSession directly to see what happens.
#![cfg(feature = "integration")]

use claudewire::config::Config;
use claudewire::session::CliSession;

#[tokio::test]
async fn raw_session_round_trip() {
    let config = Config {
        model: "sonnet".into(),
        append_system_prompt: Some("Reply with one word.".into()),
        permission_mode: "plan".into(),
        ..Default::default()
    };

    eprintln!("Spawning session...");
    let mut session = CliSession::spawn(
        &config,
        "debug-test".into(),
        Some(Box::new(|line: &str| {
            eprintln!("[stderr] {line}");
        })),
    )
    .expect("spawn");

    eprintln!("Waiting for messages...");

    // Read until system.init
    let mut got_init = false;
    for i in 0..20 {
        match session.read_message().await {
            Some(m) => {
                let t = m.get("type").and_then(|v| v.as_str()).unwrap_or("?");
                eprintln!("[msg {i}] type={t}");
                if t == "system" {
                    got_init = true;
                    break;
                }
            }
            None => {
                panic!("Session ended before system.init at message {i}");
            }
        }
    }

    assert!(got_init, "Should have received system.init");
    eprintln!("Got system.init! Sending user message...");

    let user_msg = serde_json::json!({
        "type": "user",
        "session_id": "",
        "message": {"role": "user", "content": "Say hello"},
        "parent_tool_use_id": null,
    });
    session.write(&user_msg.to_string()).await.expect("write");
    eprintln!("Sent. Reading responses...");

    let mut got_result = false;
    for i in 0..100 {
        match session.read_message().await {
            Some(m) => {
                let t = m.get("type").and_then(|v| v.as_str()).unwrap_or("?");

                if t == "result" {
                    let is_error = m.get("is_error").and_then(|v| v.as_bool()).unwrap_or(false);
                    let result_text = m.get("result").and_then(|v| v.as_str()).unwrap_or("");
                    eprintln!("[resp {i}] result is_error={is_error} text={result_text:.200}");
                    got_result = true;
                    break;
                } else if t == "stream_event" {
                    if let Some(event) = m.get("event") {
                        let et = event.get("type").and_then(|v| v.as_str()).unwrap_or("?");
                        if et == "content_block_delta" {
                            if let Some(delta) = event.get("delta") {
                                if let Some(text) = delta.get("text").and_then(|v| v.as_str()) {
                                    eprint!("{text}");
                                }
                            }
                        } else {
                            eprintln!("[resp {i}] stream_event/{et}");
                        }
                    }
                } else if t == "control_request" {
                    let rid = m.get("request_id").and_then(|v| v.as_str()).unwrap_or("");
                    let subtype = m.get("request")
                        .and_then(|r| r.get("subtype"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("?");
                    eprintln!("[resp {i}] control_request/{subtype} id={rid}");

                    let resp = serde_json::json!({
                        "type": "control_response",
                        "response": {
                            "subtype": "success",
                            "request_id": rid,
                            "response": {},
                        },
                    });
                    session.write(&resp.to_string()).await.ok();
                } else if t == "assistant" {
                    eprintln!("[resp {i}] assistant (assembled message)");
                } else {
                    eprintln!("[resp {i}] type={t}");
                }
            }
            None => {
                eprintln!("[resp {i}] stream ended (None)");
                break;
            }
        }
    }

    eprintln!();
    assert!(got_result, "Should have received result");
    session.stop().await;
}
