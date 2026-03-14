//! Rust SDK for Claude Code CLI.
//!
//! Async client built on top of claudewire's `CliSession`.
//! Handles env cleanup, system.init handshake, message construction,
//! response parsing, and control request auto-approval.
//!
//! ```no_run
//! use claudesdk::{Client, ClientConfig, Event};
//!
//! # #[tokio::main]
//! # async fn main() -> anyhow::Result<()> {
//! let mut client = Client::start(ClientConfig {
//!     model: "sonnet".into(),
//!     system_prompt: Some("You are helpful.".into()),
//!     ..Default::default()
//! }).await?;
//!
//! // Streaming
//! client.send("Say hello").await?;
//! while let Some(event) = client.recv().await {
//!     match event {
//!         Event::TextDelta(s) => print!("{s}"),
//!         Event::Message(s) => println!("\n{s}"),
//!         Event::TurnDone { cost_usd, .. } => break,
//!         Event::Error(e) => { eprintln!("{e}"); break; }
//!         _ => {}
//!     }
//! }
//!
//! // Or one-shot convenience
//! let turn = client.message("What is 2+2?").await?;
//! println!("{}", turn.text);
//!
//! client.stop().await;
//! # Ok(())
//! # }
//! ```

use anyhow::{Context, Result};
use claudewire::config::Config;
use claudewire::session::CliSession;
use tokio::process::Command;
use tracing::debug;

/// Client configuration.
#[derive(Debug, Clone, Default)]
pub struct ClientConfig {
    pub model: String,
    pub system_prompt: Option<String>,
    pub permission_mode: String,
    /// Tools to explicitly block (e.g. `["Bash", "Edit", "WebFetch"]`).
    pub disallowed_tools: Vec<String>,
    /// Tools to explicitly allow (empty = no allowlist filtering).
    pub allowed_tools: Vec<String>,
    /// Auto-approve all control requests (tool permissions).
    /// If false, control requests are emitted as `Event::ControlRequest` and
    /// must be answered with `Client::approve()` / `Client::deny()`.
    pub auto_approve: bool,
}

/// Events emitted during a conversational turn.
#[derive(Debug, Clone)]
pub enum Event {
    /// Incremental text fragment (from `content_block_delta`).
    TextDelta(String),
    /// Full assembled assistant message text.
    Message(String),
    /// Turn is complete.
    TurnDone {
        cost_usd: Option<f64>,
        duration_ms: Option<i64>,
        session_id: String,
        is_error: bool,
    },
    /// Error from the CLI.
    Error(String),
    /// Permission prompt from CLI (only if `auto_approve` is false).
    ControlRequest {
        request_id: String,
        subtype: String,
        raw: serde_json::Value,
    },
}

/// Result of a completed conversational turn.
#[derive(Debug, Clone)]
pub struct Turn {
    pub text: String,
    pub cost_usd: Option<f64>,
    pub duration_ms: Option<i64>,
    pub session_id: String,
    pub is_error: bool,
}

/// Async Claude Code client.
pub struct Client {
    session: CliSession,
    auto_approve: bool,
    /// Accumulated text for current turn.
    accumulated: String,
    /// Whether we're mid-turn (between send and TurnDone).
    in_turn: bool,
    /// Whether we've seen system.init yet.
    init_done: bool,
}

impl Client {
    /// Start a new client. Spawns the Claude CLI process.
    ///
    /// The CLI only emits `system.init` after receiving the first stdin message,
    /// so initialization completes lazily on the first `send()` / `message()`.
    pub async fn start(config: ClientConfig) -> Result<Self> {
        let wire_config = Config {
            model: config.model,
            append_system_prompt: config.system_prompt,
            permission_mode: if config.permission_mode.is_empty() {
                "plan".into()
            } else {
                config.permission_mode
            },
            disallowed_tools: config.disallowed_tools,
            allowed_tools: config.allowed_tools,
            ..Default::default()
        };

        let args = wire_config.to_cli_args();
        let env = wire_config.to_env();

        // Resolve absolute path to claude binary — DAWs and plugin hosts
        // often have a minimal PATH that doesn't include /opt/homebrew/bin.
        let claude_bin = resolve_claude_path(&args[0]);
        let mut cmd = Command::new(&claude_bin);
        cmd.args(&args[1..]);
        cmd.envs(env);
        // Remove env vars that trigger nested-session detection
        cmd.env_remove("CLAUDECODE");
        cmd.env_remove("CLAUDE_CODE_SSE_PORT");

        let session = CliSession::from_command(cmd, "claudesdk".into(), None)
            .context("failed to spawn Claude CLI")?;

        debug!("[claudesdk] spawned");

        Ok(Self {
            session,
            auto_approve: config.auto_approve,
            accumulated: String::new(),
            in_turn: false,
            init_done: false,
        })
    }

    /// Send a user message. Call `recv()` to get response events.
    pub async fn send(&mut self, text: &str) -> Result<()> {
        self.accumulated.clear();
        self.in_turn = true;

        let user_msg = serde_json::json!({
            "type": "user",
            "message": {
                "role": "user",
                "content": text,
            }
        });

        self.session
            .write(&user_msg.to_string())
            .await
            .context("failed to write user message")?;

        Ok(())
    }

    /// Read the next event. Returns `None` when the turn is complete or session ends.
    pub async fn recv(&mut self) -> Option<Event> {
        if !self.in_turn {
            return None;
        }

        loop {
            match self.session.read_message().await {
                Some(msg) => {
                    let t = msg_type_str(&msg);

                    match t {
                        "stream_event" => {
                            if let Some(text) = extract_text_delta(&msg) {
                                self.accumulated.push_str(&text);
                                return Some(Event::TextDelta(text));
                            }
                            // Other stream events (thinking, tool_use, etc.) — skip
                        }
                        "assistant" => {
                            if let Some(text) = extract_assistant_text(&msg) {
                                if !text.is_empty() {
                                    self.accumulated = text.clone();
                                    return Some(Event::Message(text));
                                }
                            }
                        }
                        "result" => {
                            self.in_turn = false;
                            let is_error = msg
                                .get("is_error")
                                .and_then(|v| v.as_bool())
                                .unwrap_or(false);

                            if is_error {
                                let err = msg
                                    .get("result")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("unknown error")
                                    .to_string();
                                return Some(Event::Error(err));
                            }

                            return Some(Event::TurnDone {
                                cost_usd: msg
                                    .get("total_cost_usd")
                                    .and_then(|v| v.as_f64()),
                                duration_ms: msg
                                    .get("duration_ms")
                                    .and_then(|v| v.as_i64()),
                                session_id: msg
                                    .get("session_id")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .to_string(),
                                is_error: false,
                            });
                        }
                        "control_request" => {
                            if self.auto_approve {
                                auto_approve_control(&mut self.session, &msg).await;
                            } else {
                                let request_id = msg
                                    .get("request_id")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .to_string();
                                let subtype = msg
                                    .get("request")
                                    .and_then(|r| r.get("subtype"))
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .to_string();
                                return Some(Event::ControlRequest {
                                    request_id,
                                    subtype,
                                    raw: msg,
                                });
                            }
                        }
                        "system" => {
                            debug!("[claudesdk] got system.init");
                            self.init_done = true;
                        }
                        _ => {
                            debug!("[claudesdk] ignoring type={t}");
                        }
                    }
                }
                None => {
                    self.in_turn = false;
                    return None;
                }
            }
        }
    }

    /// Approve a control request.
    pub async fn approve(&mut self, request_id: &str) -> Result<()> {
        let response = serde_json::json!({
            "type": "control_response",
            "response": {
                "subtype": "success",
                "request_id": request_id,
                "response": {},
            },
        });
        self.session.write(&response.to_string()).await?;
        Ok(())
    }

    /// Deny a control request.
    pub async fn deny(&mut self, request_id: &str) -> Result<()> {
        let response = serde_json::json!({
            "type": "control_response",
            "response": {
                "subtype": "permissions_response",
                "request_id": request_id,
                "response": {"allowed": false},
            },
        });
        self.session.write(&response.to_string()).await?;
        Ok(())
    }

    /// Convenience: send a message and collect the full response.
    pub async fn message(&mut self, text: &str) -> Result<Turn> {
        self.send(text).await?;

        let mut result_text = String::new();
        let mut cost_usd = None;
        let mut duration_ms = None;
        let mut session_id = String::new();
        let mut is_error = false;

        while let Some(event) = self.recv().await {
            match event {
                Event::TextDelta(_) => {}
                Event::Message(t) => result_text = t,
                Event::TurnDone {
                    cost_usd: c,
                    duration_ms: d,
                    session_id: s,
                    is_error: e,
                } => {
                    cost_usd = c;
                    duration_ms = d;
                    session_id = s;
                    is_error = e;
                    break;
                }
                Event::Error(e) => {
                    is_error = true;
                    result_text = e;
                    break;
                }
                Event::ControlRequest { .. } => {}
            }
        }

        // If no assembled message, use accumulated deltas
        if result_text.is_empty() {
            result_text = self.accumulated.clone();
        }

        Ok(Turn {
            text: result_text,
            cost_usd,
            duration_ms,
            session_id,
            is_error,
        })
    }

    /// Get the accumulated text from the current/last turn.
    pub fn accumulated_text(&self) -> &str {
        &self.accumulated
    }

    /// Stop the CLI session.
    pub async fn stop(&mut self) {
        self.session.stop().await;
    }
}

// ── Helpers ──

/// Resolve the claude binary to an absolute path.
/// Checks common install locations if the bare name isn't found on PATH.
fn resolve_claude_path(name: &str) -> String {
    // If already absolute, use as-is
    if name.starts_with('/') {
        return name.to_string();
    }

    // Try PATH first (works in terminals, may fail in DAWs)
    if let Ok(path) = std::process::Command::new("which")
        .arg(name)
        .output()
    {
        let out = String::from_utf8_lossy(&path.stdout).trim().to_string();
        if !out.is_empty() && std::path::Path::new(&out).exists() {
            return out;
        }
    }

    // Common install locations on macOS
    for candidate in &[
        "/opt/homebrew/bin/claude",
        "/usr/local/bin/claude",
    ] {
        if std::path::Path::new(candidate).exists() {
            return candidate.to_string();
        }
    }

    // Fall back to bare name (will fail with a clear error)
    name.to_string()
}

fn msg_type_str(msg: &serde_json::Value) -> &str {
    msg.get("type").and_then(|v| v.as_str()).unwrap_or("")
}

fn extract_text_delta(msg: &serde_json::Value) -> Option<String> {
    let event = msg.get("event")?;
    if event.get("type")?.as_str()? != "content_block_delta" {
        return None;
    }
    let delta = event.get("delta")?;
    if delta.get("type")?.as_str()? != "text_delta" {
        return None;
    }
    delta.get("text")?.as_str().map(String::from)
}

fn extract_assistant_text(msg: &serde_json::Value) -> Option<String> {
    let content = msg.get("message")?.get("content")?.as_array()?;
    let mut text = String::new();
    for block in content {
        if block.get("type").and_then(|v| v.as_str()) == Some("text") {
            if let Some(t) = block.get("text").and_then(|v| v.as_str()) {
                text.push_str(t);
            }
        }
    }
    Some(text)
}

async fn auto_approve_control(session: &mut CliSession, msg: &serde_json::Value) {
    let request_id = msg
        .get("request_id")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    debug!("[claudesdk] auto-approving {request_id}");
    let response = serde_json::json!({
        "type": "control_response",
        "response": {
            "subtype": "success",
            "request_id": request_id,
            "response": {},
        },
    });
    session.write(&response.to_string()).await.ok();
}

// ── Unit tests ──

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_text_delta_works() {
        let msg = serde_json::json!({
            "type": "stream_event",
            "uuid": "u1", "session_id": "s1",
            "event": {
                "type": "content_block_delta", "index": 0,
                "delta": {"type": "text_delta", "text": "Hello"}
            }
        });
        assert_eq!(extract_text_delta(&msg), Some("Hello".into()));
    }

    #[test]
    fn extract_text_delta_ignores_thinking() {
        let msg = serde_json::json!({
            "type": "stream_event",
            "uuid": "u1", "session_id": "s1",
            "event": {
                "type": "content_block_delta", "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "hmm"}
            }
        });
        assert_eq!(extract_text_delta(&msg), None);
    }

    #[test]
    fn extract_assistant_text_works() {
        let msg = serde_json::json!({
            "type": "assistant", "session_id": "s1", "uuid": "u1",
            "message": {
                "model": "claude-sonnet-4-6", "id": "m1", "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Hello "},
                    {"type": "text", "text": "world"}
                ]
            }
        });
        assert_eq!(extract_assistant_text(&msg), Some("Hello world".into()));
    }

    #[test]
    fn extract_assistant_text_skips_tool_use() {
        let msg = serde_json::json!({
            "type": "assistant", "session_id": "s1", "uuid": "u1",
            "message": {
                "model": "m", "id": "m1", "type": "message", "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "Bash", "input": {}},
                    {"type": "text", "text": "Done."}
                ]
            }
        });
        assert_eq!(extract_assistant_text(&msg), Some("Done.".into()));
    }

    #[test]
    fn extract_assistant_text_no_message() {
        assert_eq!(extract_assistant_text(&serde_json::json!({"type": "result"})), None);
    }

    #[test]
    fn msg_type_str_works() {
        assert_eq!(msg_type_str(&serde_json::json!({"type": "result"})), "result");
        assert_eq!(msg_type_str(&serde_json::json!({})), "");
    }

    #[test]
    fn default_config() {
        let c = ClientConfig::default();
        assert!(c.model.is_empty());
        assert!(c.system_prompt.is_none());
        assert!(!c.auto_approve);
    }
}
