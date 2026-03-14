//! Parameter schema for Lossy codec emulation.
//!
//! All callers (CLI, plugin, Python bindings) use the same `LossyParams` struct.
//! Mirrors `lossy/engine/params.py` exactly.

use serde::{Deserialize, Deserializer, Serialize};

pub const SR: f64 = 44100.0;

// --- Fixed constants (removed from user-facing params after PCA analysis) ---
// These 8 params are at their defaults in 89+ of 98 hand-designed presets.
// Fixing them reduces the param count from 41 to 33 with negligible perceptual impact.

/// RNG seed for spectral and packet processing.
pub const SEED: i32 = 42;
/// Pre-echo artifact amount (always off).
pub const PRE_ECHO: f64 = 0.0;
/// Transient detection threshold for pre-echo (unused since pre_echo=0).
pub const TRANSIENT_RATIO: f64 = 4.0;
/// Envelope-following noise shaping (always off).
pub const NOISE_SHAPE: f64 = 0.0;
/// Freeze slushy drift rate.
pub const SLUSHY_RATE: f64 = 0.03;
/// Freeze mode: 0=Slushy (drift enabled), 1=Solid.
pub const FREEZE_MODE: i32 = 0;
/// Reverb position: 0=Pre (before spectral), 1=Post (after filter).
pub const VERB_POSITION: i32 = 0;
/// Master intensity multiplier (always 1.0 = full).
pub const GLOBAL_AMOUNT: f64 = 1.0;

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
    "loss",
    "window_size",
    "crush",
    "decimate",
    "verb",
    "filter_freq",
    "gate",
];

/// Filter slope options (dB/oct).
pub const SLOPE_OPTIONS: &[i32] = &[6, 24, 96];

/// Biquad cascade counts for each slope.
pub fn slope_sections(slope: i32) -> usize {
    match slope {
        6 => 1,
        24 => 2,
        96 => 8,
        _ => 1,
    }
}

/// Parameter ranges for continuous parameters (min, max).
pub fn param_range(key: &str) -> Option<(f64, f64)> {
    match key {
        "jitter" => Some((0.0, 1.0)),
        "loss" => Some((0.0, 1.0)),
        "window_size" => Some((64.0, 16384.0)),
        "hop_divisor" => Some((1.0, 8.0)),
        "n_bands" => Some((2.0, 64.0)),
        "phase_loss" => Some((0.0, 1.0)),
        "weighting" => Some((0.0, 1.0)),
        "hf_threshold" => Some((0.0, 1.0)),
        "crush" => Some((0.0, 1.0)),
        "decimate" => Some((0.0, 1.0)),
        "packet_rate" => Some((0.0, 1.0)),
        "packet_size" => Some((5.0, 200.0)),
        "filter_freq" => Some((20.0, 20000.0)),
        "filter_width" => Some((0.0, 1.0)),
        "verb" => Some((0.0, 1.0)),
        "decay" => Some((0.0, 1.0)),
        "freezer" => Some((0.0, 1.0)),
        "gate" => Some((0.0, 1.0)),
        "threshold" => Some((0.0, 1.0)),
        "auto_gain" => Some((0.0, 1.0)),
        "loss_gain" => Some((0.0, 1.0)),
        "bounce_rate" => Some((0.0, 1.0)),
        "bounce_lfo_min" => Some((0.01, 50.0)),
        "bounce_lfo_max" => Some((0.01, 50.0)),
        "wet_dry" => Some((0.0, 1.0)),
        _ => None,
    }
}

/// All lossy parameters.
///
/// Uses `#[serde(default)]` so sparse preset JSON loads correctly —
/// missing keys get default values.
/// All lossy parameters (33 user-facing + metadata).
///
/// 8 former params are now fixed constants (see `SEED`, `PRE_ECHO`, etc.).
/// Uses `#[serde(default)]` so sparse preset JSON loads correctly —
/// missing keys get defaults, and legacy keys (pre_echo, etc.) are ignored.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LossyParams {
    // --- Spectral loss ---
    #[serde(deserialize_with = "as_i32")]
    pub inverse: i32,
    pub jitter: f64,
    pub loss: f64,
    #[serde(deserialize_with = "as_i32")]
    pub window_size: i32,
    #[serde(deserialize_with = "as_i32")]
    pub hop_divisor: i32,
    #[serde(deserialize_with = "as_i32")]
    pub n_bands: i32,
    pub phase_loss: f64,
    #[serde(deserialize_with = "as_i32")]
    pub quantizer: i32,
    pub weighting: f64,
    pub hf_threshold: f64,

    // --- Crush ---
    pub crush: f64,
    pub decimate: f64,

    // --- Packets ---
    #[serde(deserialize_with = "as_i32")]
    pub packets: i32,
    pub packet_rate: f64,
    pub packet_size: f64,

    // --- Filter ---
    #[serde(deserialize_with = "as_i32")]
    pub filter_type: i32,
    pub filter_freq: f64,
    pub filter_width: f64,
    #[serde(deserialize_with = "as_i32")]
    pub filter_slope: i32,

    // --- Reverb ---
    pub verb: f64,
    pub decay: f64,

    // --- Freeze ---
    #[serde(deserialize_with = "as_i32")]
    pub freeze: i32,
    pub freezer: f64,

    // --- Gate ---
    pub gate: f64,

    // --- Hidden ---
    pub threshold: f64,
    pub auto_gain: f64,
    pub loss_gain: f64,

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

    // --- Metadata (ignored for DSP, present in presets) ---
    #[serde(rename = "_meta", default, skip_serializing)]
    pub meta: Option<serde_json::Value>,
}

impl Default for LossyParams {
    fn default() -> Self {
        Self {
            // Spectral loss
            inverse: 0,
            jitter: 0.0,
            loss: 0.5,
            window_size: 2048,
            hop_divisor: 4,
            n_bands: 21,
            phase_loss: 0.0,
            quantizer: 0,
            weighting: 1.0,
            hf_threshold: 0.3,
            // Crush
            crush: 0.0,
            decimate: 0.0,
            // Packets
            packets: 0,
            packet_rate: 0.3,
            packet_size: 30.0,
            // Filter
            filter_type: 0,
            filter_freq: 1000.0,
            filter_width: 0.5,
            filter_slope: 1,
            // Reverb
            verb: 0.0,
            decay: 0.5,
            // Freeze
            freeze: 0,
            freezer: 1.0,
            // Gate
            gate: 0.0,
            // Hidden
            threshold: 0.5,
            auto_gain: 0.0,
            loss_gain: 0.5,
            // Bounce
            bounce: 0,
            bounce_target: 0,
            bounce_rate: 0.3,
            bounce_lfo_min: 0.1,
            bounce_lfo_max: 5.0,
            // Output
            wet_dry: 1.0,
            // Metadata
            meta: None,
        }
    }
}

impl LossyParams {
    /// Parse from JSON string. Missing fields get default values.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Parse from JSON, falling back to defaults for any missing/invalid fields.
    pub fn from_json_with_defaults(json: &str) -> Self {
        serde_json::from_str(json)
            .unwrap_or_else(|e| panic!("Failed to parse LossyParams JSON: {e}\nInput: {json}"))
    }

    /// Get the value of a named parameter for bounce modulation.
    pub fn get_bounce_target_value(&self, key: &str) -> f64 {
        match key {
            "loss" => self.loss,
            "window_size" => self.window_size as f64,
            "crush" => self.crush,
            "decimate" => self.decimate,
            "verb" => self.verb,
            "filter_freq" => self.filter_freq,
            "gate" => self.gate,
            _ => 0.5,
        }
    }

    /// Set the value of a named parameter for bounce modulation.
    pub fn set_bounce_target_value(&mut self, key: &str, value: f64) {
        match key {
            "loss" => self.loss = value,
            "window_size" => self.window_size = value as i32,
            "crush" => self.crush = value,
            "decimate" => self.decimate = value,
            "verb" => self.verb = value,
            "filter_freq" => self.filter_freq = value,
            "gate" => self.gate = value,
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mt19937_matches_numpy() {
        // Verify rand_mt produces same sequence as numpy.random.RandomState(42)
        // numpy uses genrand_res53: (a >> 5) * 2^26 + (b >> 6) / 2^53
        use rand_mt::Mt;
        let mut rng = Mt::new(42);
        let expected = [
            0.37454011884736249094_f64,
            0.95071430640991616556,
            0.73199394181140509108,
            0.59865848419703659999,
            0.15601864044243651808,
        ];
        for (i, &exp) in expected.iter().enumerate() {
            let a = rng.next_u32() >> 5;
            let b = rng.next_u32() >> 6;
            let val = (a as f64 * 67108864.0 + b as f64) / 9007199254740992.0;
            assert!(
                (val - exp).abs() < 1e-18,
                "mismatch at [{i}]: rust={val:.20} numpy={exp:.20}"
            );
        }
    }

    #[test]
    fn test_default_params() {
        let p = LossyParams::default();
        assert_eq!(p.loss, 0.5);
        assert_eq!(p.window_size, 2048);
        assert_eq!(p.wet_dry, 1.0);
    }

    #[test]
    fn test_sparse_json_load() {
        let json = r#"{"loss": 0.8, "crush": 0.5}"#;
        let p = LossyParams::from_json(json).unwrap();
        assert_eq!(p.loss, 0.8);
        assert_eq!(p.crush, 0.5);
        // Missing fields should get defaults
        assert_eq!(p.window_size, 2048);
        assert_eq!(p.wet_dry, 1.0);
    }

    #[test]
    fn test_legacy_preset_with_removed_fields() {
        // Old presets with the 8 removed fields should still deserialize
        let json = r#"{
            "loss": 0.4,
            "pre_echo": 0.3,
            "noise_shape": 0.5,
            "transient_ratio": 8.0,
            "slushy_rate": 0.1,
            "freeze_mode": 1,
            "verb_position": 1,
            "global_amount": 0.8,
            "seed": 123
        }"#;
        let p = LossyParams::from_json(json).unwrap();
        assert_eq!(p.loss, 0.4);
        // Removed fields are silently ignored
        assert_eq!(p.window_size, 2048); // default
    }

    #[test]
    fn test_preset_with_meta() {
        let json = r#"{
            "loss": 0.4,
            "window_size": 1024,
            "_meta": {"category": "Communication", "description": "test"}
        }"#;
        let p = LossyParams::from_json(json).unwrap();
        assert_eq!(p.loss, 0.4);
        assert_eq!(p.window_size, 1024);
        assert!(p.meta.is_some());
    }

    #[test]
    fn test_param_ranges() {
        assert_eq!(param_range("loss"), Some((0.0, 1.0)));
        assert_eq!(param_range("filter_freq"), Some((20.0, 20000.0)));
        assert_eq!(param_range("nonexistent"), None);
    }

    #[test]
    fn test_bounce_target_get_set() {
        let mut p = LossyParams::default();
        p.set_bounce_target_value("loss", 0.9);
        assert_eq!(p.get_bounce_target_value("loss"), 0.9);
    }
}
