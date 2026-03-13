//! AI sound design tuner — manages Claude-backed parameter tuning sessions.
//!
//! Wraps `claudewire::ChatBackend` with domain logic: system prompt construction,
//! JSON param extraction, validation/clamping, autonomous iteration, undo, and
//! audio metrics context.
//!
//! All methods are non-blocking and safe to call from a GUI thread.

use claudewire::chat::{ChatBackend, ChatMsg};

use crate::metrics::{format_features, AudioMetrics};
use crate::params::ParamSchema;

/// Maximum silent iteration rounds per user request.
const MAX_ITERATE: u32 = 5;

/// Messages from the tuner to the GUI.
#[derive(Debug, Clone)]
pub enum TunerMsg {
    /// Streaming assistant text (delta, not accumulated).
    Text(String),
    /// Claude produced validated parameters. Apply these to the UI.
    Params(serde_json::Map<String, serde_json::Value>),
    /// Claude wants a silent iteration: render with these params, then call
    /// [`SoundDesigner::continue_iteration`] with the resulting metrics.
    Iterate(serde_json::Map<String, serde_json::Value>),
    /// The response is complete.
    Done,
    /// An error occurred.
    Error(String),
}

/// State of the tuner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Idle,
    WaitingForResponse,
    WaitingForIteration,
}

/// AI sound design tuner.
///
/// Manages a Claude chat session for parameter tuning. The tuner:
/// - Builds system prompts from a guide text and parameter schema
/// - Sends user prompts with current params, audio metrics, and spectrogram paths
/// - Extracts JSON parameter blocks from Claude responses
/// - Validates and clamps parameters against the schema
/// - Supports autonomous iteration (up to N silent renders)
/// - Tracks undo state for reverting LLM changes
pub struct SoundDesigner {
    guide_text: String,
    schema: ParamSchema,
    backend: Option<ChatBackend>,
    state: State,
    undo_params: Option<serde_json::Map<String, serde_json::Value>>,
    prev_metrics: Option<AudioMetrics>,
    source_sent: bool,
    iterate_count: u32,
    stop_iterating: bool,
    /// Accumulated text from the current response (for JSON extraction).
    accumulated_text: String,
    /// Current params during an active request (for merging).
    current_params: serde_json::Map<String, serde_json::Value>,
}

impl SoundDesigner {
    /// Create a new sound designer with the given guide text and parameter schema.
    pub fn new(guide_text: String, schema: ParamSchema) -> Self {
        Self {
            guide_text,
            schema,
            backend: None,
            state: State::Idle,
            undo_params: None,
            prev_metrics: None,
            source_sent: false,
            iterate_count: 0,
            stop_iterating: false,
            accumulated_text: String::new(),
            current_params: serde_json::Map::new(),
        }
    }

    /// Create with a pre-built backend (for testing).
    pub fn with_backend(guide_text: String, schema: ParamSchema, backend: ChatBackend) -> Self {
        Self {
            backend: Some(backend),
            ..Self::new(guide_text, schema)
        }
    }

    /// Whether the tuner is currently processing a request.
    pub fn is_busy(&self) -> bool {
        self.state != State::Idle
    }

    /// Send a user prompt. Returns `Err` if already busy.
    ///
    /// After calling this, poll [`Self::poll`] on each GUI frame to receive
    /// [`TunerMsg`] messages.
    ///
    /// - `user_text`: The user's message.
    /// - `current_params`: Current parameter values.
    /// - `metrics`: Output audio metrics from the current render (optional).
    /// - `source_metrics`: Input/source audio metrics (optional, sent once per session).
    /// - `spectrogram_path`: Path to output spectrogram PNG (optional).
    /// - `source_spectrogram_path`: Path to source spectrogram PNG (optional).
    pub fn send_prompt(
        &mut self,
        user_text: &str,
        current_params: &serde_json::Map<String, serde_json::Value>,
        metrics: Option<&AudioMetrics>,
        source_metrics: Option<&AudioMetrics>,
        spectrogram_path: Option<&str>,
        source_spectrogram_path: Option<&str>,
    ) -> Result<(), String> {
        if self.state != State::Idle {
            return Err("Already processing a request".into());
        }

        self.undo_params = Some(current_params.clone());
        self.current_params = current_params.clone();
        self.iterate_count = 0;
        self.stop_iterating = false;
        self.accumulated_text.clear();
        self.state = State::WaitingForResponse;

        // Ensure backend exists
        if self.backend.is_none() {
            let prompt = self.build_system_prompt();
            self.backend = Some(ChatBackend::new(&prompt));
        }

        // Build the full prompt
        let full_prompt = self.build_user_prompt(
            user_text,
            current_params,
            metrics,
            source_metrics,
            spectrogram_path,
            source_spectrogram_path,
        );

        self.backend.as_ref().unwrap().send(&full_prompt);
        Ok(())
    }

    /// Continue an iteration cycle. Called by the GUI after a silent render completes.
    ///
    /// - `current_params`: The params that were just rendered.
    /// - `metrics`: Audio metrics from the render.
    /// - `spectrogram_path`: Path to the updated spectrogram PNG (optional).
    pub fn continue_iteration(
        &mut self,
        current_params: &serde_json::Map<String, serde_json::Value>,
        metrics: &AudioMetrics,
        spectrogram_path: Option<&str>,
    ) {
        self.current_params = current_params.clone();
        self.accumulated_text.clear();

        let src: Option<&AudioMetrics> = None;
        let features_text = format_features(Some(metrics), self.prev_metrics.as_ref(), src);
        self.prev_metrics = Some(metrics.clone());

        let mut prompt = format!(
            "[Iteration {}/{}] Render complete.\n\n{}",
            self.iterate_count, MAX_ITERATE, features_text
        );

        if let Some(path) = spectrogram_path {
            prompt.push_str(&format!("\n\nUpdated output spectrogram: {path}"));
        }

        if self.iterate_count >= MAX_ITERATE {
            prompt.push_str("\n\n[Max iterations reached — do NOT include _iterate.]");
        }

        self.state = State::WaitingForResponse;
        if let Some(backend) = &self.backend {
            backend.send(&prompt);
        }
    }

    /// Stop the iteration chain after the current render completes.
    pub fn stop_iterating(&mut self) {
        self.stop_iterating = true;
    }

    /// Poll for messages. Call this on every GUI frame.
    ///
    /// Returns a list of [`TunerMsg`] to process. The list may be empty.
    pub fn poll(&mut self) -> Vec<TunerMsg> {
        if self.state == State::Idle || self.state == State::WaitingForIteration {
            return Vec::new();
        }

        let backend = match &self.backend {
            Some(b) => b,
            None => return Vec::new(),
        };

        let raw_msgs = backend.poll();
        if raw_msgs.is_empty() {
            return Vec::new();
        }

        let mut out = Vec::new();

        for msg in raw_msgs {
            match msg {
                ChatMsg::AssistantText(accumulated) => {
                    // ChatBackend sends accumulated text. Compute the delta.
                    if accumulated.len() > self.accumulated_text.len() {
                        let delta = &accumulated[self.accumulated_text.len()..];
                        out.push(TunerMsg::Text(delta.to_string()));
                    }
                    self.accumulated_text = accumulated;
                }
                ChatMsg::Done => {
                    // Response complete — extract params from accumulated text
                    let (raw_params, iterate) = extract_json_block(&self.accumulated_text);

                    if let Some(raw) = raw_params {
                        let validated = self.schema.validate_and_clamp(&raw);
                        let merged = merge_params(&self.current_params, &validated);

                        if iterate
                            && self.iterate_count < MAX_ITERATE
                            && !self.stop_iterating
                        {
                            self.iterate_count += 1;
                            self.state = State::WaitingForIteration;
                            out.push(TunerMsg::Iterate(merged));
                        } else {
                            self.state = State::Idle;
                            out.push(TunerMsg::Params(merged));
                            out.push(TunerMsg::Done);
                        }
                    } else {
                        // No params in response — just a text reply
                        self.state = State::Idle;
                        out.push(TunerMsg::Done);
                    }

                    self.accumulated_text.clear();
                }
                ChatMsg::Error(e) => {
                    self.state = State::Idle;
                    out.push(TunerMsg::Error(e));
                }
            }
        }

        out
    }

    /// Returns the pre-LLM params for undo, or `None` if nothing to undo.
    pub fn undo(&self) -> Option<&serde_json::Map<String, serde_json::Value>> {
        self.undo_params.as_ref()
    }

    /// Reset the session. The next `send_prompt` creates a fresh Claude session.
    pub fn reset_session(&mut self) {
        self.prev_metrics = None;
        self.source_sent = false;
        self.backend = None;
        self.state = State::Idle;
        self.accumulated_text.clear();
    }

    fn build_system_prompt(&self) -> String {
        format!(
            "{}\n\n\
RULES:\n\
- You can have a conversation with the user to understand what they want before\n\
  committing parameter changes. Ask clarifying questions if needed.\n\
- When you're ready to apply changes, include a ```json code block with a JSON\n\
  object of parameter key-value pairs. This will be parsed and applied to the UI.\n\
- If the user is just chatting or asking questions, respond normally without\n\
  a JSON code block. Not every message needs parameter changes.\n\
- You may include all parameters or just the ones you want to change.\n\
- Missing keys will keep their current values.\n\
- Stay within the documented ranges.\n\
- For per-node parameters (arrays of 8), always provide all 8 values.\n\
- For integer choice params (mode, waveform, etc), use the integer value.\n\
- Include a brief text explanation of what you're changing and why.\n\
\n\
AUTONOMOUS TUNING:\n\
- To iterate silently (render + check metrics without the user listening), include\n\
  \"_iterate\": true in your JSON block alongside the parameters.\n\
- You have up to {MAX_ITERATE} silent iterations per user request. Use them to converge before\n\
  presenting the result.\n\
- When satisfied (or on final iteration), omit \"_iterate\" — the system will play\n\
  for the user to hear.\n\
- Each iteration renders in 1-2 seconds. Make meaningful changes each round.\n\
- Always explain your reasoning in text, even during silent iterations.",
            self.guide_text
        )
    }

    fn build_user_prompt(
        &mut self,
        user_text: &str,
        current_params: &serde_json::Map<String, serde_json::Value>,
        metrics: Option<&AudioMetrics>,
        source_metrics: Option<&AudioMetrics>,
        spectrogram_path: Option<&str>,
        source_spectrogram_path: Option<&str>,
    ) -> String {
        let params_json = serde_json::to_string_pretty(
            &serde_json::Value::Object(current_params.clone()),
        )
        .unwrap_or_default();

        let mut prompt = format!(
            "Current parameters:\n{params_json}\n\nUser request: {user_text}"
        );

        // Audio metrics with A/B delta (skip source if already sent)
        let src = if self.source_sent {
            None
        } else {
            source_metrics
        };
        if metrics.is_some() || src.is_some() {
            let features_text =
                format_features(metrics, self.prev_metrics.as_ref(), src);
            if !features_text.is_empty() {
                prompt.push_str(&format!("\n\n{features_text}"));
            }
            if let Some(m) = metrics {
                self.prev_metrics = Some(m.clone());
            }
        }

        // Spectrogram paths for Claude to Read
        if !self.source_sent {
            if let Some(path) = source_spectrogram_path {
                prompt.push_str(&format!(
                    "\n\nInput audio spectrogram saved at: {path}\nUse the Read tool to view it."
                ));
            }
        }
        if let Some(path) = spectrogram_path {
            prompt.push_str(&format!(
                "\n\nOutput audio spectrogram saved at: {path}\nUse the Read tool to view it."
            ));
        }

        self.source_sent = true;
        prompt
    }
}

/// Extract a JSON object from a ```json code block.
///
/// Returns `(params, iterate)` where `iterate` is the `_iterate` flag value.
fn extract_json_block(
    text: &str,
) -> (Option<serde_json::Map<String, serde_json::Value>>, bool) {
    // Find ```json ... ``` block
    let start_marker = "```json";
    let end_marker = "```";

    let start = match text.find(start_marker) {
        Some(i) => i + start_marker.len(),
        None => return (None, false),
    };

    // Skip to newline after ```json
    let start = text[start..].find('\n').map(|i| start + i + 1).unwrap_or(start);

    let end = match text[start..].find(end_marker) {
        Some(i) => start + i,
        None => return (None, false),
    };

    let json_str = &text[start..end];

    match serde_json::from_str::<serde_json::Value>(json_str) {
        Ok(serde_json::Value::Object(mut obj)) => {
            let iterate = obj
                .remove("_iterate")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            (Some(obj), iterate)
        }
        _ => (None, false),
    }
}

/// Merge validated params into current params (missing keys keep current values).
fn merge_params(
    current: &serde_json::Map<String, serde_json::Value>,
    validated: &serde_json::Map<String, serde_json::Value>,
) -> serde_json::Map<String, serde_json::Value> {
    let mut merged = current.clone();
    for (k, v) in validated {
        merged.insert(k.clone(), v.clone());
    }
    merged
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::{ParamDef, ParamSchema};

    #[test]
    fn extract_json_block_basic() {
        let text = "I'll adjust the decay.\n```json\n{\"decay\": 5.0, \"mix\": 0.8}\n```\nShould sound longer.";
        let (params, iterate) = extract_json_block(text);
        let params = params.unwrap();
        assert_eq!(params["decay"], 5.0);
        assert_eq!(params["mix"], 0.8);
        assert!(!iterate);
    }

    #[test]
    fn extract_json_block_with_iterate() {
        let text = "```json\n{\"decay\": 3.0, \"_iterate\": true}\n```";
        let (params, iterate) = extract_json_block(text);
        let params = params.unwrap();
        assert_eq!(params["decay"], 3.0);
        assert!(!params.contains_key("_iterate")); // removed
        assert!(iterate);
    }

    #[test]
    fn extract_json_block_none() {
        let text = "No params here, just a chat message.";
        let (params, iterate) = extract_json_block(text);
        assert!(params.is_none());
        assert!(!iterate);
    }

    #[test]
    fn merge_params_preserves_existing() {
        let mut current = serde_json::Map::new();
        current.insert("a".into(), serde_json::json!(1.0));
        current.insert("b".into(), serde_json::json!(2.0));

        let mut validated = serde_json::Map::new();
        validated.insert("a".into(), serde_json::json!(5.0));

        let merged = merge_params(&current, &validated);
        assert_eq!(merged["a"], 5.0);
        assert_eq!(merged["b"], 2.0); // preserved
    }

    fn test_schema() -> ParamSchema {
        ParamSchema::new(vec![
            ParamDef::float("decay", 2.0, (0.1, 30.0), "reverb"),
            ParamDef::float("mix", 0.5, (0.0, 1.0), "reverb"),
        ])
    }

    #[test]
    fn sound_designer_not_busy_initially() {
        let sd = SoundDesigner::new("Guide".into(), test_schema());
        assert!(!sd.is_busy());
    }

    #[test]
    fn sound_designer_undo_none_initially() {
        let sd = SoundDesigner::new("Guide".into(), test_schema());
        assert!(sd.undo().is_none());
    }

    #[test]
    fn sound_designer_reset_clears_state() {
        let mut sd = SoundDesigner::new("Guide".into(), test_schema());
        sd.source_sent = true;
        sd.prev_metrics = Some(AudioMetrics::default());
        sd.reset_session();
        assert!(!sd.source_sent);
        assert!(sd.prev_metrics.is_none());
        assert!(!sd.is_busy());
    }
}
