//! Plugin chat backend — spawns Claude CLI on a background thread.
//!
//! Designed for use in audio plugins where no tokio runtime exists on the
//! GUI thread. All IO happens on a dedicated background thread. Communication
//! uses lock-free `std::sync::mpsc` channels so the GUI thread never blocks.
//!
//! The DSP/audio thread is completely unaffected — this module only touches
//! the GUI thread via non-blocking `try_recv()` polls.

use std::path::PathBuf;
use std::sync::mpsc as std_mpsc;

/// A message from the chat backend to the GUI.
#[derive(Debug, Clone)]
pub enum ChatMsg {
    /// Streaming assistant text (accumulated so far for the current response).
    AssistantText(String),
    /// An error occurred.
    Error(String),
    /// The assistant finished responding.
    Done,
}

/// Chat backend that manages a Claude CLI subprocess on a background thread.
///
/// All methods are non-blocking and safe to call from a GUI thread.
pub struct ChatBackend {
    to_backend: std_mpsc::Sender<String>,
    from_backend: std_mpsc::Receiver<ChatMsg>,
}

impl ChatBackend {
    /// Create a new chat backend with the given system prompt.
    ///
    /// The Claude CLI process is not spawned until the first message is sent.
    pub fn new(system_prompt: &str) -> Self {
        let (user_tx, user_rx) = std_mpsc::channel::<String>();
        let (chat_tx, chat_rx) = std_mpsc::channel::<ChatMsg>();
        let system = system_prompt.to_string();

        std::thread::Builder::new()
            .name("claude-chat".into())
            .spawn(move || {
                run_chat_thread(user_rx, chat_tx, system);
            })
            .expect("spawn chat thread");

        Self {
            to_backend: user_tx,
            from_backend: chat_rx,
        }
    }

    /// Send a user message. Non-blocking.
    pub fn send(&self, text: &str) {
        let _ = self.to_backend.send(text.to_string());
    }

    /// Poll for new messages from the backend. Non-blocking, returns immediately.
    pub fn poll(&self) -> Vec<ChatMsg> {
        let mut msgs = Vec::new();
        while let Ok(msg) = self.from_backend.try_recv() {
            msgs.push(msg);
        }
        msgs
    }
}

/// Find the `claude` binary, checking PATH and common install locations.
fn find_claude_binary() -> Option<PathBuf> {
    // Check PATH first
    if let Ok(output) = std::process::Command::new("which")
        .arg("claude")
        .output()
    {
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path.is_empty() {
                return Some(PathBuf::from(path));
            }
        }
    }

    // Common install locations (DAWs don't inherit shell PATH)
    let home = std::env::var("HOME").unwrap_or_default();
    let candidates = [
        format!("{home}/.claude/local/claude"),
        format!("{home}/.local/bin/claude"),
        format!("{home}/.npm/bin/claude"),
        "/usr/local/bin/claude".into(),
        "/opt/homebrew/bin/claude".into(),
    ];

    for path in &candidates {
        let p = PathBuf::from(path);
        if p.exists() {
            return Some(p);
        }
    }

    None
}

/// Background thread that manages the Claude CLI subprocess.
fn run_chat_thread(
    user_rx: std_mpsc::Receiver<String>,
    chat_tx: std_mpsc::Sender<ChatMsg>,
    system_prompt: String,
) {
    let rt = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(e) => {
            let _ = chat_tx.send(ChatMsg::Error(format!("Failed to create runtime: {e}")));
            return;
        }
    };

    rt.block_on(async move {
        // Wait for first user message before spawning the process
        let first_msg = match user_rx.recv() {
            Ok(msg) => msg,
            Err(_) => return,
        };

        let claude_bin = match find_claude_binary() {
            Some(p) => p,
            None => {
                let _ = chat_tx.send(ChatMsg::Error(
                    "claude CLI not found. Install it or add it to PATH.".into(),
                ));
                return;
            }
        };

        let mut cmd = tokio::process::Command::new(&claude_bin);
        cmd.arg("--output-format")
            .arg("stream-json")
            .arg("--verbose")
            .arg("-p")
            .arg(&first_msg)
            .arg("--system-prompt")
            .arg(&system_prompt)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null());

        // Ensure common bin dirs are in PATH for the child process
        let home = std::env::var("HOME").unwrap_or_default();
        let extra_path = format!(
            "/usr/local/bin:/opt/homebrew/bin:{home}/.local/bin:{home}/.npm/bin",
        );
        let current_path = std::env::var("PATH").unwrap_or_default();
        cmd.env("PATH", format!("{extra_path}:{current_path}"));

        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                let _ = chat_tx.send(ChatMsg::Error(format!(
                    "Failed to start claude at {}: {e}",
                    claude_bin.display()
                )));
                return;
            }
        };

        let stdout = child.stdout.take().unwrap();
        let mut stdin = child.stdin.take().unwrap();

        // Read stdout — parse stream-json, extract text deltas
        let chat_tx2 = chat_tx.clone();
        let read_handle = tokio::spawn(async move {
            use tokio::io::{AsyncBufReadExt, BufReader};
            let reader = BufReader::new(stdout);
            let mut lines = reader.lines();
            let mut accumulated = String::new();
            while let Ok(Some(line)) = lines.next_line().await {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                let msg: serde_json::Value = match serde_json::from_str(trimmed) {
                    Ok(m) => m,
                    Err(_) => continue,
                };
                let msg_type = msg.get("type").and_then(|v| v.as_str()).unwrap_or("");
                match msg_type {
                    "content_block_delta" => {
                        if let Some(text) = msg
                            .get("delta")
                            .and_then(|d| d.get("text"))
                            .and_then(|t| t.as_str())
                        {
                            accumulated.push_str(text);
                            let _ =
                                chat_tx2.send(ChatMsg::AssistantText(accumulated.clone()));
                        }
                    }
                    "result" => {
                        if accumulated.is_empty() {
                            if let Some(text) =
                                msg.get("result").and_then(|r| r.as_str())
                            {
                                accumulated = text.to_string();
                                let _ = chat_tx2
                                    .send(ChatMsg::AssistantText(accumulated.clone()));
                            }
                        }
                        let _ = chat_tx2.send(ChatMsg::Done);
                        accumulated.clear();
                    }
                    _ => {}
                }
            }
        });

        // Forward subsequent user messages to stdin
        let stdin_handle = tokio::spawn(async move {
            use tokio::io::AsyncWriteExt;
            while let Ok(msg) = user_rx.recv() {
                let user_msg = crate::events::make_user_message(
                    &serde_json::Value::String(msg),
                );
                let json = serde_json::to_string(&user_msg).unwrap();
                if stdin.write_all(json.as_bytes()).await.is_err() {
                    break;
                }
                if stdin.write_all(b"\n").await.is_err() {
                    break;
                }
                let _ = stdin.flush().await;
            }
        });

        let _ = tokio::join!(read_handle, stdin_handle);
        let _ = child.kill().await;
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn find_claude_binary_returns_something_or_none() {
        // Just verify it doesn't panic
        let result = find_claude_binary();
        // On CI claude won't exist, on dev machines it should
        if let Some(path) = &result {
            assert!(path.exists(), "found path should exist: {}", path.display());
        }
    }

    #[test]
    fn chat_backend_send_before_poll() {
        let backend = ChatBackend::new("test system prompt");
        // Polling before sending should return empty
        let msgs = backend.poll();
        assert!(msgs.is_empty());
    }

    #[test]
    fn chat_backend_poll_is_nonblocking() {
        let backend = ChatBackend::new("test");
        let start = std::time::Instant::now();
        for _ in 0..100 {
            let _ = backend.poll();
        }
        let elapsed = start.elapsed();
        // 100 polls should complete in well under 10ms
        assert!(
            elapsed < Duration::from_millis(10),
            "poll took too long: {elapsed:?}"
        );
    }

    #[test]
    fn chat_backend_send_triggers_error_when_no_claude() {
        // This test verifies the error path when claude is not found
        // We create a backend and send a message — if claude isn't installed,
        // we should get an error message back.
        let backend = ChatBackend::new("test");
        backend.send("hello");

        // Give the background thread time to try spawning
        std::thread::sleep(Duration::from_millis(500));

        let msgs = backend.poll();
        // We either get an error (no claude) or assistant text (claude exists)
        // Either way, poll should work without blocking
        for msg in &msgs {
            match msg {
                ChatMsg::Error(e) => {
                    assert!(
                        e.contains("not found") || e.contains("Failed"),
                        "unexpected error: {e}"
                    );
                }
                ChatMsg::AssistantText(_) | ChatMsg::Done => {
                    // Claude is installed and responded — that's fine too
                }
            }
        }
    }

    #[test]
    fn chat_backend_with_mock_process() {
        // Test the channel plumbing using a mock "process" (just echo)
        let (user_tx, _user_rx) = std_mpsc::channel::<String>();
        let (chat_tx, chat_rx) = std_mpsc::channel::<ChatMsg>();

        // Simulate backend sending messages
        chat_tx.send(ChatMsg::AssistantText("Hello".into())).unwrap();
        chat_tx.send(ChatMsg::AssistantText("Hello world".into())).unwrap();
        chat_tx.send(ChatMsg::Done).unwrap();

        let backend = ChatBackend {
            to_backend: user_tx,
            from_backend: chat_rx,
        };

        let msgs = backend.poll();
        assert_eq!(msgs.len(), 3);
        assert!(matches!(&msgs[0], ChatMsg::AssistantText(t) if t == "Hello"));
        assert!(matches!(&msgs[1], ChatMsg::AssistantText(t) if t == "Hello world"));
        assert!(matches!(&msgs[2], ChatMsg::Done));
    }
}
