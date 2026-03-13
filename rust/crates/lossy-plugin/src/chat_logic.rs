//! Pure logic for the chat feature — JSON extraction, param merging, prompt building.
//!
//! Separated from gui.rs so it can be tested without a GUI framework.

use crate::capture::AudioMetrics;

/// Extract a JSON object from a ```json code block in assistant text.
/// Strips `_iterate` key if present. Returns None if no valid JSON block found.
pub fn extract_json_block(text: &str) -> Option<serde_json::Value> {
    // Find opening marker (case-insensitive for robustness)
    let lower = text.to_lowercase();
    let start = lower.find("```json")?;
    let after_marker = start + "```json".len();

    // Skip optional whitespace/newline after marker
    let json_start = text[after_marker..]
        .find(|c: char| !c.is_whitespace() || c == '{' || c == '[')
        .map(|i| after_marker + i)
        .unwrap_or(after_marker);

    // Find closing ``` — must be the NEXT occurrence after the opening
    let end = text[json_start..].find("\n```").or_else(|| text[json_start..].find("\r\n```"))?;
    let json_str = text[json_start..json_start + end].trim();

    let mut val: serde_json::Value = serde_json::from_str(json_str).ok()?;

    // Remove non-param keys
    if let Some(obj) = val.as_object_mut() {
        obj.remove("_iterate");
    }

    Some(val)
}

/// Strip ```json code blocks from text for cleaner chat display.
pub fn strip_json_block(text: &str) -> String {
    let lower = text.to_lowercase();
    if let Some(start) = lower.find("```json") {
        // Find the closing ```
        let after_marker = start + "```json".len();
        if let Some(end_offset) = text[after_marker..].find("\n```") {
            let end = after_marker + end_offset + "\n```".len();
            // Skip trailing newline
            let end = if text.len() > end && text[end..].starts_with('\n') {
                end + 1
            } else {
                end
            };
            let mut result = text[..start].to_string();
            result.push_str(&text[end..]);
            return result.trim().to_string();
        }
    }
    text.to_string()
}

/// Merge extracted JSON params on top of current LossyParams.
/// Only updates keys present in `extracted`, keeps all others at current values.
/// Returns error string on failure for display to user.
pub fn merge_params(
    current: &lossy_dsp::LossyParams,
    extracted: &serde_json::Value,
) -> Result<lossy_dsp::LossyParams, String> {
    let current_json = serde_json::to_value(current)
        .map_err(|e| format!("Failed to serialize current params: {e}"))?;

    let mut merged = current_json;

    match (merged.as_object_mut(), extracted.as_object()) {
        (Some(base), Some(overlay)) => {
            for (k, v) in overlay {
                base.insert(k.clone(), v.clone());
            }
        }
        _ => return Err("Expected JSON objects for merge".into()),
    }

    serde_json::from_value::<lossy_dsp::LossyParams>(merged)
        .map_err(|e| format!("Failed to apply params: {e}"))
}

/// Build the enriched prompt sent to Claude (current params + metrics + user text).
pub fn build_enriched_prompt(
    user_text: &str,
    params: &lossy_dsp::LossyParams,
    metrics: Option<&AudioMetrics>,
) -> String {
    let params_json =
        serde_json::to_string_pretty(params).unwrap_or_else(|_| "{}".into());

    let mut prompt = format!("Current parameters:\n{params_json}");

    if let Some(m) = metrics {
        prompt.push_str(&format!(
            "\n\nAudio metrics (last {:.1}s):\
             \n- RMS: {:.1} dB\
             \n- Peak: {:.1} dB\
             \n- Crest factor: {:.1} dB\
             \n- Spectral centroid: {:.0} Hz",
            m.duration_secs, m.rms_db, m.peak_db, m.crest_factor_db,
            m.spectral_centroid_hz,
        ));
    }

    prompt.push_str(&format!("\n\nUser request: {user_text}"));

    prompt
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── extract_json_block ──

    #[test]
    fn extract_simple_json_block() {
        let text = "I'll increase the loss.\n```json\n{\"loss\": 0.8}\n```\nEnjoy!";
        let val = extract_json_block(text).unwrap();
        assert_eq!(val["loss"], 0.8);
    }

    #[test]
    fn extract_json_block_with_iterate() {
        let text = "Adjusting...\n```json\n{\"loss\": 0.9, \"_iterate\": true}\n```";
        let val = extract_json_block(text).unwrap();
        assert_eq!(val["loss"], 0.9);
        assert!(val.get("_iterate").is_none(), "_iterate should be stripped");
    }

    #[test]
    fn extract_json_block_multiple_params() {
        let text = "Here:\n```json\n{\n  \"loss\": 0.7,\n  \"crush\": 0.3,\n  \"window_size\": 1024\n}\n```";
        let val = extract_json_block(text).unwrap();
        assert_eq!(val["loss"], 0.7);
        assert_eq!(val["crush"], 0.3);
        assert_eq!(val["window_size"], 1024);
    }

    #[test]
    fn extract_json_block_none_when_no_block() {
        let text = "Just chatting, no params to change.";
        assert!(extract_json_block(text).is_none());
    }

    #[test]
    fn extract_json_block_none_for_invalid_json() {
        let text = "```json\n{invalid json here}\n```";
        assert!(extract_json_block(text).is_none());
    }

    #[test]
    fn extract_json_block_case_insensitive_marker() {
        let text = "```JSON\n{\"loss\": 0.5}\n```";
        let val = extract_json_block(text).unwrap();
        assert_eq!(val["loss"], 0.5);
    }

    #[test]
    fn extract_ignores_non_json_code_blocks() {
        let text = "```rust\nfn main() {}\n```\nNo JSON here.";
        assert!(extract_json_block(text).is_none());
    }

    #[test]
    fn extract_json_block_with_trailing_text() {
        let text = "Setting a lo-fi radio sound:\n```json\n{\"loss\": 0.4, \"filter_type\": 1, \"filter_freq\": 1200.0, \"verb\": 0.2}\n```\nThis bandpasses the signal around 1200Hz for that radio effect.";
        let val = extract_json_block(text).unwrap();
        assert_eq!(val["loss"], 0.4);
        assert_eq!(val["filter_type"], 1);
        assert_eq!(val["filter_freq"], 1200.0);
        assert_eq!(val["verb"], 0.2);
    }

    // ── strip_json_block ──

    #[test]
    fn strip_removes_json_block() {
        let text = "Here's the change:\n```json\n{\"loss\": 0.8}\n```\nDone.";
        let stripped = strip_json_block(text);
        assert!(!stripped.contains("```json"));
        assert!(!stripped.contains("\"loss\""));
        assert!(stripped.contains("Here's the change:"));
        assert!(stripped.contains("Done."));
    }

    #[test]
    fn strip_no_json_returns_original() {
        let text = "Just a normal message.";
        assert_eq!(strip_json_block(text), text);
    }

    // ── merge_params ──

    #[test]
    fn merge_partial_params_preserves_existing() {
        let current = lossy_dsp::LossyParams::default();
        let extracted: serde_json::Value =
            serde_json::json!({"loss": 0.9, "crush": 0.4});
        let merged = merge_params(&current, &extracted).unwrap();

        // Changed values
        assert_eq!(merged.loss, 0.9);
        assert_eq!(merged.crush, 0.4);
        // Preserved values
        assert_eq!(merged.window_size, current.window_size);
        assert_eq!(merged.wet_dry, current.wet_dry);
        assert_eq!(merged.wet_dry, 1.0);
    }

    #[test]
    fn merge_integer_params() {
        let current = lossy_dsp::LossyParams::default();
        let extracted: serde_json::Value =
            serde_json::json!({"window_size": 512, "n_bands": 8});
        let merged = merge_params(&current, &extracted).unwrap();
        assert_eq!(merged.window_size, 512);
        assert_eq!(merged.n_bands, 8);
    }

    #[test]
    fn merge_filter_slope_actual_values() {
        // filter_slope stores actual dB values: 6, 24, 96
        let current = lossy_dsp::LossyParams::default();
        let extracted: serde_json::Value = serde_json::json!({"filter_slope": 96});
        let merged = merge_params(&current, &extracted).unwrap();
        assert_eq!(merged.filter_slope, 96);
    }

    #[test]
    fn merge_float_as_int_accepted() {
        // Claude might send 1024.0 instead of 1024 for integer fields
        let current = lossy_dsp::LossyParams::default();
        let extracted: serde_json::Value =
            serde_json::json!({"window_size": 1024.0});
        let merged = merge_params(&current, &extracted).unwrap();
        assert_eq!(merged.window_size, 1024);
    }

    #[test]
    fn merge_empty_extracted_returns_current() {
        let current = lossy_dsp::LossyParams {
            loss: 0.75,
            crush: 0.3,
            ..Default::default()
        };
        let extracted: serde_json::Value = serde_json::json!({});
        let merged = merge_params(&current, &extracted).unwrap();
        assert_eq!(merged.loss, 0.75);
        assert_eq!(merged.crush, 0.3);
    }

    #[test]
    fn merge_unknown_field_ignored() {
        // Unknown fields are silently ignored (no deny_unknown_fields)
        let current = lossy_dsp::LossyParams::default();
        let extracted: serde_json::Value =
            serde_json::json!({"loss": 0.9, "nonexistent_param": 42});
        let merged = merge_params(&current, &extracted).unwrap();
        assert_eq!(merged.loss, 0.9);
    }

    // ── round-trip: serialize → merge → deserialize ──

    #[test]
    fn full_round_trip_preserves_all_fields() {
        let original = lossy_dsp::LossyParams {
            loss: 0.42,
            crush: 0.15,
            window_size: 4096,
            filter_slope: 96,
            freeze: 1,
            wet_dry: 0.8,
            ..Default::default()
        };

        // Simulate: serialize current, Claude sends partial, merge, deserialize
        let claude_json: serde_json::Value =
            serde_json::json!({"loss": 0.99, "verb": 0.3});
        let merged = merge_params(&original, &claude_json).unwrap();

        // Claude's changes applied
        assert_eq!(merged.loss, 0.99);
        assert_eq!(merged.verb, 0.3);
        // Original values preserved
        assert_eq!(merged.crush, 0.15);
        assert_eq!(merged.window_size, 4096);
        assert_eq!(merged.filter_slope, 96);
        assert_eq!(merged.freeze, 1);
        assert_eq!(merged.wet_dry, 0.8);
    }

    // ── build_enriched_prompt ──

    #[test]
    fn enriched_prompt_contains_params_and_user_text() {
        let params = lossy_dsp::LossyParams::default();
        let prompt = build_enriched_prompt("make it more destroyed", &params, None);

        assert!(prompt.contains("Current parameters:"));
        assert!(prompt.contains("\"loss\""));
        assert!(prompt.contains("User request: make it more destroyed"));
    }

    #[test]
    fn enriched_prompt_includes_metrics() {
        let params = lossy_dsp::LossyParams::default();
        let metrics = AudioMetrics {
            rms_db: -18.5,
            peak_db: -3.2,
            crest_factor_db: 15.3,
            spectral_centroid_hz: 2400.0,
            duration_secs: 2.1,
        };
        let prompt =
            build_enriched_prompt("what do you hear?", &params, Some(&metrics));

        assert!(prompt.contains("RMS: -18.5 dB"));
        assert!(prompt.contains("Peak: -3.2 dB"));
        assert!(prompt.contains("Spectral centroid: 2400 Hz"));
    }

    #[test]
    fn enriched_prompt_no_metrics_when_none() {
        let params = lossy_dsp::LossyParams::default();
        let prompt = build_enriched_prompt("hi", &params, None);
        assert!(!prompt.contains("Audio metrics"));
    }

    // ── end-to-end: extract from realistic Claude response → merge → verify ──

    #[test]
    fn e2e_realistic_claude_response() {
        let response = r#"I'll set up a lo-fi radio effect for you. This uses a bandpass filter around 1kHz with some spectral loss and a touch of reverb.

```json
{
  "loss": 0.4,
  "filter_type": 1,
  "filter_freq": 1000.0,
  "filter_width": 0.3,
  "filter_slope": 24,
  "verb": 0.15,
  "decimate": 0.2,
  "wet_dry": 1.0
}
```

The bandpass at 1kHz with moderate width removes the extreme lows and highs, giving that radio frequency range. The slight decimation adds aliasing artifacts."#;

        let current = lossy_dsp::LossyParams::default();

        // Step 1: Extract
        let extracted = extract_json_block(response).expect("should find JSON block");
        assert_eq!(extracted["loss"], 0.4);
        assert_eq!(extracted["filter_type"], 1);

        // Step 2: Merge
        let merged = merge_params(&current, &extracted).unwrap();
        assert_eq!(merged.loss, 0.4);
        assert_eq!(merged.filter_type, 1);
        assert_eq!(merged.filter_freq, 1000.0);
        assert_eq!(merged.filter_slope, 24);
        assert_eq!(merged.verb, 0.15);
        assert_eq!(merged.decimate, 0.2);
        // Unchanged
        assert_eq!(merged.crush, 0.0);
        assert_eq!(merged.window_size, 2048);

        // Step 3: Strip for display
        let display = strip_json_block(response);
        assert!(display.contains("lo-fi radio effect"));
        assert!(display.contains("bandpass at 1kHz"));
        assert!(!display.contains("```json"));
    }

    #[test]
    fn e2e_chat_only_no_params() {
        let response = "Your current settings look pretty good for a glitchy effect. \
                        The window size of 2048 with loss at 0.5 gives a nice degraded sound. \
                        Would you like me to push it further or change the character?";

        assert!(extract_json_block(response).is_none());
        assert_eq!(strip_json_block(response), response);
    }
}
