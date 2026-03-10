//! `DirectTransport` — spawns and manages a `claude` CLI subprocess directly.
//!
//! Unlike `BridgeTransport` which requires an external process manager,
//! `DirectTransport` owns the child process and communicates via stdin/stdout.

use std::process::Stdio;

use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::mpsc;
use tracing::debug;

use crate::schema::is_bare_stream_type;

/// A transport that spawns `claude` directly and communicates via stream-json.
pub struct DirectTransport {
    child: Option<Child>,
    stdin_tx: Option<mpsc::UnboundedSender<String>>,
    stdout_rx: Option<mpsc::UnboundedReceiver<serde_json::Value>>,
    stderr_rx: Option<mpsc::UnboundedReceiver<String>>,
    exited: bool,
    exit_code: Option<i32>,
}

impl DirectTransport {
    /// Spawn a new `claude` process with the given arguments.
    ///
    /// The process is started with `--output-format stream-json` automatically.
    /// Additional args (e.g. `--model`, `--system-prompt`, `-p`) can be passed.
    pub fn spawn(extra_args: &[&str]) -> anyhow::Result<Self> {
        let mut cmd = Command::new("claude");
        cmd.arg("--output-format").arg("stream-json");
        for arg in extra_args {
            cmd.arg(arg);
        }
        cmd.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let mut child = cmd.spawn()?;

        let stdin = child.stdin.take().expect("stdin piped");
        let stdout = child.stdout.take().expect("stdout piped");
        let stderr = child.stderr.take().expect("stderr piped");

        // Stdin writer task
        let (stdin_tx, mut stdin_rx) = mpsc::unbounded_channel::<String>();
        tokio::spawn(async move {
            let mut stdin = stdin;
            while let Some(line) = stdin_rx.recv().await {
                if stdin.write_all(line.as_bytes()).await.is_err() {
                    break;
                }
                if stdin.write_all(b"\n").await.is_err() {
                    break;
                }
                let _ = stdin.flush().await;
            }
        });

        // Stdout reader task — parse JSON lines, filter bare stream duplicates
        let (stdout_tx, stdout_rx) = mpsc::unbounded_channel();
        tokio::spawn(async move {
            let reader = BufReader::new(stdout);
            let mut lines = reader.lines();
            while let Ok(Some(line)) = lines.next_line().await {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                match serde_json::from_str::<serde_json::Value>(trimmed) {
                    Ok(msg) => {
                        let msg_type = msg
                            .get("type")
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        if is_bare_stream_type(msg_type) {
                            continue;
                        }
                        if stdout_tx.send(msg).is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        debug!("stdout parse error: {e} — line: {trimmed:.200}");
                    }
                }
            }
        });

        // Stderr reader task
        let (stderr_tx, stderr_rx) = mpsc::unbounded_channel();
        tokio::spawn(async move {
            let reader = BufReader::new(stderr);
            let mut lines = reader.lines();
            while let Ok(Some(line)) = lines.next_line().await {
                let _ = stderr_tx.send(line);
            }
        });

        Ok(Self {
            child: Some(child),
            stdin_tx: Some(stdin_tx),
            stdout_rx: Some(stdout_rx),
            stderr_rx: Some(stderr_rx),
            exited: false,
            exit_code: None,
        })
    }

    /// Send a JSON message to the CLI's stdin.
    pub fn send(&self, msg: &serde_json::Value) -> anyhow::Result<()> {
        let tx = self
            .stdin_tx
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("stdin closed"))?;
        let json = serde_json::to_string(msg)?;
        tx.send(json)
            .map_err(|_| anyhow::anyhow!("stdin channel closed"))?;
        Ok(())
    }

    /// Send a user message (convenience wrapper around `events::make_user_message`).
    pub fn send_user_message(&self, text: &str) -> anyhow::Result<()> {
        let msg = crate::events::make_user_message(&serde_json::Value::String(text.to_string()));
        self.send(&msg)
    }

    /// Read the next JSON message from stdout. Returns `None` when the stream ends.
    pub async fn read_message(&mut self) -> Option<serde_json::Value> {
        self.stdout_rx.as_mut()?.recv().await
    }

    /// Drain any available stderr lines (non-blocking).
    pub fn drain_stderr(&mut self) -> Vec<String> {
        let mut lines = Vec::new();
        if let Some(rx) = &mut self.stderr_rx {
            while let Ok(line) = rx.try_recv() {
                lines.push(line);
            }
        }
        lines
    }

    /// Kill the subprocess.
    pub async fn kill(&mut self) {
        if let Some(child) = &mut self.child {
            let _ = child.kill().await;
            if let Ok(status) = child.wait().await {
                self.exit_code = status.code();
            }
        }
        self.exited = true;
        self.stdin_tx = None;
    }

    pub const fn exited(&self) -> bool {
        self.exited
    }

    pub const fn exit_code(&self) -> Option<i32> {
        self.exit_code
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn spawn_echo_test() {
        // Test with `echo` instead of `claude` to verify process management
        let mut cmd = Command::new("echo");
        cmd.arg(r#"{"type":"result","result":"hello"}"#)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let mut child = cmd.spawn().unwrap();
        let stdout = child.stdout.take().unwrap();

        let (tx, mut rx) = mpsc::unbounded_channel();
        tokio::spawn(async move {
            let reader = BufReader::new(stdout);
            let mut lines = reader.lines();
            while let Ok(Some(line)) = lines.next_line().await {
                if let Ok(msg) = serde_json::from_str::<serde_json::Value>(&line) {
                    let _ = tx.send(msg);
                }
            }
        });

        let msg = rx.recv().await.unwrap();
        assert_eq!(msg["type"], "result");
        assert_eq!(msg["result"], "hello");
    }
}
