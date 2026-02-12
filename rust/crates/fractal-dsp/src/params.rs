//! Parameter schema for the Fractal audio fractalization effect.
//!
//! All callers (CLI, plugin) use the same `FractalParams` struct.
//! Mirrors `fractal/engine/params.py` exactly.

use serde::{Deserialize, Deserializer, Serialize};

pub const SR: f64 = 44100.0;

/// Accept both `4096` and `4096.0` from JSON, truncate to i32.
fn as_i32<'de, D: Deserializer<'de>>(d: D) -> Result<i32, D::Error> {
    let v: serde_json::Value = Deserialize::deserialize(d)?;
    match &v {
        serde_json::Value::Number(n) => n
            .as_i64()
            .map(|i| i as i32)
            .or_else(|| n.as_f64().map(|f| f as i32))
            .ok_or_else(|| serde::de::Error::custom(format!("cannot convert {n} to i32"))),
        _ => Err(serde::de::Error::custom(format!("expected number, got {v}"))),
    }
}

/// Bounce target parameter names (subset of params that can be modulated).
pub const BOUNCE_TARGETS: &[&str] = &[
    "scale_ratio",
    "amplitude_decay",
    "num_scales",
    "saturation",
    "filter_freq",
    "crush",
    "spectral",
];

/// Parameter ranges for continuous parameters (min, max).
pub fn param_range(key: &str) -> Option<(f64, f64)> {
    match key {
        "num_scales" => Some((2.0, 8.0)),
        "scale_ratio" => Some((0.1, 0.9)),
        "amplitude_decay" => Some((0.1, 1.0)),
        "scale_offset" => Some((0.0, 1.0)),
        "iterations" => Some((1.0, 4.0)),
        "iter_decay" => Some((0.3, 1.0)),
        "saturation" => Some((0.0, 1.0)),
        "spectral" => Some((0.0, 1.0)),
        "window_size" => Some((256.0, 8192.0)),
        "filter_freq" => Some((20.0, 20000.0)),
        "filter_q" => Some((0.1, 10.0)),
        "post_filter_freq" => Some((20.0, 20000.0)),
        "gate" => Some((0.0, 1.0)),
        "crush" => Some((0.0, 1.0)),
        "decimate" => Some((0.0, 1.0)),
        "bounce_rate" => Some((0.0, 1.0)),
        "bounce_lfo_min" => Some((0.01, 50.0)),
        "bounce_lfo_max" => Some((0.01, 50.0)),
        "wet_dry" => Some((0.0, 1.0)),
        "output_gain" => Some((0.0, 1.0)),
        "threshold" => Some((0.0, 1.0)),
        "layer_gain_1" => Some((0.0, 2.0)),
        "layer_gain_2" => Some((0.0, 2.0)),
        "layer_gain_3" => Some((0.0, 2.0)),
        "layer_gain_4" => Some((0.0, 2.0)),
        "layer_gain_5" => Some((0.0, 2.0)),
        "layer_gain_6" => Some((0.0, 2.0)),
        "layer_gain_7" => Some((0.0, 2.0)),
        "layer_spread" => Some((0.0, 1.0)),
        "layer_detune" => Some((0.0, 1.0)),
        "layer_delay" => Some((0.0, 1.0)),
        "layer_tilt" => Some((-1.0, 1.0)),
        "feedback" => Some((0.0, 0.95)),
        _ => None,
    }
}

/// All fractal parameters.
///
/// Uses `#[serde(default)]` so sparse preset JSON loads correctly —
/// missing keys get default values.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct FractalParams {
    // --- Core fractal ---
    #[serde(deserialize_with = "as_i32")]
    pub num_scales: i32,
    pub scale_ratio: f64,
    pub amplitude_decay: f64,
    #[serde(deserialize_with = "as_i32")]
    pub interp: i32,
    #[serde(deserialize_with = "as_i32")]
    pub reverse_scales: i32,
    pub scale_offset: f64,

    // --- Iteration / feedback ---
    #[serde(deserialize_with = "as_i32")]
    pub iterations: i32,
    pub iter_decay: f64,
    pub saturation: f64,

    // --- Spectral fractal ---
    pub spectral: f64,
    #[serde(deserialize_with = "as_i32")]
    pub window_size: i32,

    // --- Pre-filter ---
    #[serde(deserialize_with = "as_i32")]
    pub filter_type: i32,
    pub filter_freq: f64,
    pub filter_q: f64,

    // --- Post-filter ---
    #[serde(deserialize_with = "as_i32")]
    pub post_filter_type: i32,
    pub post_filter_freq: f64,

    // --- Effects ---
    pub gate: f64,
    pub crush: f64,
    pub decimate: f64,

    // --- Layers ---
    pub layer_gain_1: f64,
    pub layer_gain_2: f64,
    pub layer_gain_3: f64,
    pub layer_gain_4: f64,
    pub layer_gain_5: f64,
    pub layer_gain_6: f64,
    pub layer_gain_7: f64,
    #[serde(deserialize_with = "as_i32")]
    pub fractal_only_wet: i32,
    pub layer_spread: f64,
    pub layer_detune: f64,
    pub layer_delay: f64,
    pub layer_tilt: f64,

    // --- Feedback ---
    pub feedback: f64,

    // --- Bounce ---
    #[serde(deserialize_with = "as_i32")]
    pub bounce: i32,
    #[serde(deserialize_with = "as_i32")]
    pub bounce_target: i32,
    pub bounce_rate: f64,
    pub bounce_lfo_min: f64,
    pub bounce_lfo_max: f64,

    // --- Output ---
    pub wet_dry: f64,
    pub output_gain: f64,
    pub threshold: f64,

    // --- Internal ---
    #[serde(deserialize_with = "as_i32")]
    pub seed: i32,

    // --- Metadata (ignored for DSP, present in presets) ---
    #[serde(rename = "_meta", default, skip_serializing)]
    pub meta: Option<serde_json::Value>,
}

impl Default for FractalParams {
    fn default() -> Self {
        Self {
            // Core fractal
            num_scales: 3,
            scale_ratio: 0.5,
            amplitude_decay: 0.707,
            interp: 0,
            reverse_scales: 0,
            scale_offset: 0.0,
            // Iteration / feedback
            iterations: 1,
            iter_decay: 0.8,
            saturation: 0.0,
            // Spectral fractal
            spectral: 0.0,
            window_size: 2048,
            // Pre-filter
            filter_type: 0,
            filter_freq: 2000.0,
            filter_q: 0.707,
            // Post-filter
            post_filter_type: 0,
            post_filter_freq: 8000.0,
            // Effects
            gate: 0.0,
            crush: 0.0,
            decimate: 0.0,
            // Layers
            layer_gain_1: 1.0,
            layer_gain_2: 1.0,
            layer_gain_3: 1.0,
            layer_gain_4: 1.0,
            layer_gain_5: 1.0,
            layer_gain_6: 1.0,
            layer_gain_7: 1.0,
            fractal_only_wet: 0,
            layer_spread: 0.0,
            layer_detune: 0.0,
            layer_delay: 0.0,
            layer_tilt: 0.0,
            // Feedback
            feedback: 0.0,
            // Bounce
            bounce: 0,
            bounce_target: 0,
            bounce_rate: 0.3,
            bounce_lfo_min: 0.1,
            bounce_lfo_max: 5.0,
            // Output
            wet_dry: 1.0,
            output_gain: 0.5,
            threshold: 0.5,
            // Internal
            seed: 42,
            // Metadata
            meta: None,
        }
    }
}

impl FractalParams {
    /// Parse from JSON string. Missing fields get default values.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Get the value of a named parameter for bounce modulation.
    pub fn get_bounce_target_value(&self, key: &str) -> f64 {
        match key {
            "scale_ratio" => self.scale_ratio,
            "amplitude_decay" => self.amplitude_decay,
            "num_scales" => self.num_scales as f64,
            "saturation" => self.saturation,
            "filter_freq" => self.filter_freq,
            "crush" => self.crush,
            "spectral" => self.spectral,
            _ => 0.5,
        }
    }

    /// Get per-layer gain for scale index s (1..7).
    pub fn layer_gain(&self, s: i32) -> f64 {
        match s {
            1 => self.layer_gain_1,
            2 => self.layer_gain_2,
            3 => self.layer_gain_3,
            4 => self.layer_gain_4,
            5 => self.layer_gain_5,
            6 => self.layer_gain_6,
            7 => self.layer_gain_7,
            _ => 1.0,
        }
    }

    /// Set the value of a named parameter for bounce modulation.
    pub fn set_bounce_target_value(&mut self, key: &str, value: f64) {
        match key {
            "scale_ratio" => self.scale_ratio = value,
            "amplitude_decay" => self.amplitude_decay = value,
            "num_scales" => self.num_scales = value as i32,
            "saturation" => self.saturation = value,
            "filter_freq" => self.filter_freq = value,
            "crush" => self.crush = value,
            "spectral" => self.spectral = value,
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_params() {
        let p = FractalParams::default();
        assert_eq!(p.num_scales, 3);
        assert_eq!(p.scale_ratio, 0.5);
        assert_eq!(p.amplitude_decay, 0.707);
        assert_eq!(p.wet_dry, 1.0);
        assert_eq!(p.seed, 42);
    }

    #[test]
    fn test_sparse_json_load() {
        let json = r#"{"num_scales": 5, "scale_ratio": 0.3}"#;
        let p = FractalParams::from_json(json).unwrap();
        assert_eq!(p.num_scales, 5);
        assert_eq!(p.scale_ratio, 0.3);
        // Missing fields should get defaults
        assert_eq!(p.amplitude_decay, 0.707);
        assert_eq!(p.wet_dry, 1.0);
    }

    #[test]
    fn test_preset_with_meta() {
        let json = r#"{
            "num_scales": 7,
            "scale_ratio": 0.137,
            "_meta": {"category": "Pure Fractal", "description": "test"}
        }"#;
        let p = FractalParams::from_json(json).unwrap();
        assert_eq!(p.num_scales, 7);
        assert!(p.meta.is_some());
    }

    #[test]
    fn test_param_ranges() {
        assert_eq!(param_range("scale_ratio"), Some((0.1, 0.9)));
        assert_eq!(param_range("filter_freq"), Some((20.0, 20000.0)));
        assert_eq!(param_range("nonexistent"), None);
    }

    #[test]
    fn test_bounce_target_get_set() {
        let mut p = FractalParams::default();
        p.set_bounce_target_value("scale_ratio", 0.3);
        assert_eq!(p.get_bounce_target_value("scale_ratio"), 0.3);
    }
}
