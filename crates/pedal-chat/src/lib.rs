//! Sync-friendly chat backend for effect pedal GUIs.
//!
//! Thin wrapper around [`claudesdk::Client`] that bridges async → sync
//! with a background thread. Safe to use from a GUI event loop.
//!
//! ```no_run
//! use pedal_chat::{ChatBackend, ChatMsg};
//!
//! let backend = ChatBackend::new("You are a reverb expert.");
//! backend.send("Make it darker");
//!
//! // Poll on a timer (e.g. every 200ms):
//! for msg in backend.poll() {
//!     match msg {
//!         ChatMsg::Text(t) => println!("streaming: {t}"),
//!         ChatMsg::Done => println!("turn complete"),
//!         ChatMsg::Error(e) => eprintln!("error: {e}"),
//!     }
//! }
//! ```

use std::sync::mpsc;

pub use claudesdk::{ClientConfig, Event};

/// Messages returned by [`ChatBackend::poll`].
#[derive(Debug, Clone)]
pub enum ChatMsg {
    /// Accumulated assistant text so far (grows with each delta).
    Text(String),
    /// The assistant turn is complete.
    Done,
    /// An error occurred.
    Error(String),
}

/// Non-blocking chat backend that drives a Claude CLI subprocess.
pub struct ChatBackend {
    cmd_tx: mpsc::Sender<String>,
    msg_rx: mpsc::Receiver<ChatMsg>,
}

impl ChatBackend {
    /// All tools that the Claude CLI could invoke. We block every single one
    /// because the pedal chat is pure text-in/text-out — it never needs to
    /// read files, run commands, or access the network.
    const DISALLOWED_TOOLS: &'static [&'static str] = &[
        "Bash", "Edit", "Write", "Read", "Glob", "Grep",
        "WebFetch", "WebSearch", "Agent",
        "NotebookEdit", "TodoRead", "TodoWrite",
        "TaskCreate", "TaskUpdate", "TaskGet", "TaskList",
    ];

    /// Create a backend with just a system prompt (uses defaults for model etc).
    ///
    /// Security: all tools are disallowed, auto_approve is false, and any
    /// stray control requests are automatically denied.
    pub fn new(system_prompt: &str) -> Self {
        Self::with_config(ClientConfig {
            model: "sonnet".into(),
            system_prompt: Some(system_prompt.to_string()),
            permission_mode: "plan".into(),
            disallowed_tools: Self::DISALLOWED_TOOLS.iter().map(|s| s.to_string()).collect(),
            auto_approve: false,
            ..Default::default()
        })
    }

    /// Create a backend with full config control.
    pub fn with_config(config: ClientConfig) -> Self {
        let (cmd_tx, cmd_rx) = mpsc::channel::<String>();
        let (msg_tx, msg_rx) = mpsc::channel::<ChatMsg>();

        std::thread::Builder::new()
            .name("pedal-chat".into())
            .spawn(move || {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("tokio runtime");
                rt.block_on(run(config, cmd_rx, msg_tx));
            })
            .expect("spawn pedal-chat thread");

        Self { cmd_tx, msg_rx }
    }

    /// Queue a user message. Non-blocking.
    pub fn send(&self, message: &str) {
        self.cmd_tx.send(message.to_string()).ok();
    }

    /// Drain all pending messages. Non-blocking, returns empty vec if nothing new.
    pub fn poll(&self) -> Vec<ChatMsg> {
        let mut msgs = Vec::new();
        while let Ok(msg) = self.msg_rx.try_recv() {
            msgs.push(msg);
        }
        msgs
    }
}

async fn run(
    config: ClientConfig,
    cmd_rx: mpsc::Receiver<String>,
    msg_tx: mpsc::Sender<ChatMsg>,
) {
    let mut client = match claudesdk::Client::start(config).await {
        Ok(c) => c,
        Err(e) => {
            msg_tx
                .send(ChatMsg::Error(format!("Failed to start: {e}")))
                .ok();
            return;
        }
    };

    while let Ok(user_text) = cmd_rx.recv() {
        if let Err(e) = client.send(&user_text).await {
            msg_tx
                .send(ChatMsg::Error(format!("Send failed: {e}")))
                .ok();
            continue;
        }

        let mut accumulated = String::new();

        while let Some(event) = client.recv().await {
            match event {
                Event::TextDelta(delta) => {
                    accumulated.push_str(&delta);
                    msg_tx.send(ChatMsg::Text(accumulated.clone())).ok();
                }
                Event::Message(text) => {
                    accumulated = text;
                    msg_tx.send(ChatMsg::Text(accumulated.clone())).ok();
                }
                Event::TurnDone { .. } => {
                    msg_tx.send(ChatMsg::Done).ok();
                    break;
                }
                Event::Error(e) => {
                    msg_tx.send(ChatMsg::Error(e)).ok();
                    msg_tx.send(ChatMsg::Done).ok();
                    break;
                }
                Event::ControlRequest { request_id, .. } => {
                    // Defense in depth: deny any tool permission request.
                    // All tools are already disallowed via CLI flags, but if
                    // one somehow arrives, reject it.
                    if let Err(e) = client.deny(&request_id).await {
                        msg_tx.send(ChatMsg::Error(format!("Deny failed: {e}"))).ok();
                    }
                }
            }
        }
    }

    client.stop().await;
}
