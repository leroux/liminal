//! Parameter schema for the FDN reverb.
//!
//! All callers (CLI, plugin, Python bindings) use the same `ReverbParams` struct.
//! Mirrors `reverb/engine/params.py` exactly.

use serde::{Deserialize, Deserializer, Serialize};

pub const SR: f64 = 44100.0;
pub const N: usize = 8;

// --- Early Reflections ---
pub const N_ER_TAPS: usize = 12;

/// ER tap delay ratios relative to the first FDN delay time.
const ER_TAP_RATIOS: [f64; N_ER_TAPS] = [
    0.07, 0.11, 0.15, 0.20, 0.26, 0.33, 0.41, 0.50, 0.60, 0.71, 0.83, 0.96,
];

/// ER tap gains (approximate 1/sqrt(distance) falloff).
const ER_TAP_GAINS: [f64; N_ER_TAPS] = [
    0.84, 0.71, 0.63, 0.56, 0.50, 0.45, 0.40, 0.36, 0.32, 0.28, 0.25, 0.22,
];

/// ER tap pan positions (alternating L/R pattern).
pub const ER_TAP_PANS: [f64; N_ER_TAPS] = [
    -0.8, 0.6, -0.4, 0.9, -0.2, 0.7, -0.6, 0.3, -0.9, 0.5, -0.1, 0.8,
];

/// Generate ER delay times (samples) from the base FDN delay.
pub fn generate_er_delays(base_delay: i32) -> Vec<i32> {
    ER_TAP_RATIOS
        .iter()
        .map(|&r| (r * base_delay as f64).max(1.0) as i32)
        .collect()
}

/// Generate ER gains.
pub fn generate_er_gains() -> Vec<f64> {
    ER_TAP_GAINS.to_vec()
}

/// Generate ER pan positions.
pub fn generate_er_pans() -> Vec<f64> {
    ER_TAP_PANS.to_vec()
}

fn default_sr() -> f64 {
    44100.0
}

/// Accept both `42` and `42.0` from JSON, truncate to i32.
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

/// Accept both ints and floats in JSON arrays, convert to Vec<i32>.
fn as_i32_vec<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<i32>, D::Error> {
    let arr: Vec<serde_json::Value> = Deserialize::deserialize(d)?;
    arr.iter()
        .enumerate()
        .map(|(i, v)| match v {
            serde_json::Value::Number(n) => n
                .as_i64()
                .map(|x| x as i32)
                .or_else(|| n.as_f64().map(|f| f as i32))
                .ok_or_else(|| serde::de::Error::custom(format!("cannot convert [{i}]={n} to i32"))),
            _ => Err(serde::de::Error::custom(format!("expected number at [{i}], got {v}"))),
        })
        .collect()
}

/// Matrix type names (order matches Python CHOICE_RANGES).
pub const MATRIX_TYPES: &[&str] = &[
    "householder",
    "hadamard",
    "diagonal",
    "random_orthogonal",
    "circulant",
    "stautner_puckette",
];

/// Deserialize matrix_type from either string name or integer index.
fn as_matrix_type<'de, D: Deserializer<'de>>(d: D) -> Result<String, D::Error> {
    let v: serde_json::Value = Deserialize::deserialize(d)?;
    match &v {
        serde_json::Value::String(s) => Ok(s.clone()),
        serde_json::Value::Number(n) => {
            let idx = n.as_u64().unwrap_or(0) as usize;
            Ok(MATRIX_TYPES.get(idx).unwrap_or(&"householder").to_string())
        }
        _ => Ok("householder".to_string()),
    }
}

/// Internal FDN parameters — the full per-node parameter set consumed by the DSP.
///
/// Users should prefer `ReverbParams` (the simplified macro controls) and call
/// `to_fdn_params()` to get this struct. Uses `#[serde(default)]` for sparse JSON.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct FdnParams {
    // --- Per-node arrays (8 nodes) ---
    #[serde(deserialize_with = "as_i32_vec")]
    pub delay_times: Vec<i32>,
    pub damping_coeffs: Vec<f64>,
    pub input_gains: Vec<f64>,
    pub output_gains: Vec<f64>,
    pub node_pans: Vec<f64>,

    // --- Global ---
    pub feedback_gain: f64,
    pub wet_dry: f64,
    pub diffusion: f64,
    #[serde(deserialize_with = "as_i32")]
    pub diffusion_stages: i32,
    #[serde(deserialize_with = "as_i32_vec")]
    pub diffusion_delays: Vec<i32>,
    pub saturation: f64,
    #[serde(deserialize_with = "as_i32")]
    pub pre_delay: i32,
    pub stereo_width: f64,

    // --- Matrix ---
    #[serde(deserialize_with = "as_matrix_type")]
    pub matrix_type: String,
    #[serde(deserialize_with = "as_i32")]
    pub matrix_seed: i32,
    /// Optional custom 8x8 matrix (flattened row-major or nested).
    #[serde(default)]
    pub matrix_custom: Option<Vec<Vec<f64>>>,

    // --- Modulation ---
    pub mod_master_rate: f64,
    pub mod_node_rate_mult: Vec<f64>,
    pub mod_correlation: f64,
    #[serde(deserialize_with = "as_i32")]
    pub mod_waveform: i32,
    pub mod_depth_delay: Vec<f64>,
    pub mod_depth_damping: Vec<f64>,
    pub mod_depth_output: Vec<f64>,
    pub mod_depth_matrix: f64,
    pub mod_rate_scale_delay: f64,
    pub mod_rate_scale_damping: f64,
    pub mod_rate_scale_output: f64,
    pub mod_rate_matrix: f64,
    #[serde(deserialize_with = "as_matrix_type")]
    pub mod_matrix2_type: String,
    #[serde(deserialize_with = "as_i32")]
    pub mod_matrix2_seed: i32,

    // --- Runtime (not serialized) ---
    #[serde(skip, default = "default_sr")]
    pub sample_rate: f64,

    // --- Early Reflections ---
    #[serde(deserialize_with = "as_i32_vec")]
    pub er_delays: Vec<i32>,
    pub er_gains: Vec<f64>,
    pub er_pans: Vec<f64>,
    pub er_level: f64,

    // --- Output EQ ---
    pub wet_low_cut_hz: f64,
    pub wet_high_cut_hz: f64,

    // --- Metadata (ignored for DSP, present in presets) ---
    #[serde(rename = "_meta", default, skip_serializing)]
    pub meta: Option<serde_json::Value>,
}

impl Default for FdnParams {
    fn default() -> Self {
        let delay_ms = [29.7, 37.1, 41.3, 47.9, 53.1, 59.3, 67.7, 73.1];
        let diffusion_ms = [5.3, 7.9, 11.7, 16.1];

        Self {
            delay_times: delay_ms.iter().map(|ms| (ms / 1000.0 * SR) as i32).collect(),
            damping_coeffs: vec![0.3; N],
            input_gains: vec![1.0 / N as f64; N],
            output_gains: vec![1.0; N],
            node_pans: vec![-1.0, -0.714, -0.429, -0.143, 0.143, 0.429, 0.714, 1.0],

            feedback_gain: 0.85,
            wet_dry: 0.5,
            diffusion: 0.5,
            diffusion_stages: 4,
            diffusion_delays: diffusion_ms.iter().map(|ms| (ms / 1000.0 * SR) as i32).collect(),
            saturation: 0.0,
            pre_delay: (10.0 / 1000.0 * SR) as i32,
            stereo_width: 1.0,

            matrix_type: "householder".to_string(),
            matrix_seed: 42,
            matrix_custom: None,

            mod_master_rate: 0.0,
            mod_node_rate_mult: vec![1.0; N],
            mod_correlation: 1.0,
            mod_waveform: 0,
            mod_depth_delay: vec![0.0; N],
            mod_depth_damping: vec![0.0; N],
            mod_depth_output: vec![0.0; N],
            mod_depth_matrix: 0.0,
            mod_rate_scale_delay: 1.0,
            mod_rate_scale_damping: 1.0,
            mod_rate_scale_output: 1.0,
            mod_rate_matrix: 0.0,
            mod_matrix2_type: "random_orthogonal".to_string(),
            mod_matrix2_seed: 137,

            sample_rate: 44100.0,
            er_delays: generate_er_delays(delay_ms.iter().map(|ms| (ms / 1000.0 * SR) as i32).next().unwrap_or(1310)),
            er_gains: generate_er_gains(),
            er_pans: generate_er_pans(),
            er_level: 0.0,

            wet_low_cut_hz: 20.0,
            wet_high_cut_hz: 20000.0,

            meta: None,
        }
    }
}

impl FdnParams {
    /// Parse from JSON string. Missing fields get default values.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Check if any modulation is active.
    pub fn has_modulation(&self) -> bool {
        self.mod_master_rate > 0.0
            && (self.mod_depth_delay.iter().any(|&d| d > 0.0)
                || self.mod_depth_damping.iter().any(|&d| d > 0.0)
                || self.mod_depth_output.iter().any(|&d| d > 0.0)
                || self.mod_depth_matrix > 0.0)
    }

    /// Ensure all per-node arrays have exactly N elements, padding/truncating.
    pub fn normalize(&mut self) {
        let dt_fill = self.delay_times.first().copied().unwrap_or(1310);
        pad_or_truncate_i32(&mut self.delay_times, N, dt_fill);
        pad_or_truncate_f64(&mut self.damping_coeffs, N, 0.3);
        pad_or_truncate_f64(&mut self.input_gains, N, 1.0 / N as f64);
        pad_or_truncate_f64(&mut self.output_gains, N, 1.0);
        pad_or_truncate_f64(&mut self.node_pans, N, 0.0);
        pad_or_truncate_f64(&mut self.mod_node_rate_mult, N, 1.0);
        pad_or_truncate_f64(&mut self.mod_depth_delay, N, 0.0);
        pad_or_truncate_f64(&mut self.mod_depth_damping, N, 0.0);
        pad_or_truncate_f64(&mut self.mod_depth_output, N, 0.0);
        pad_or_truncate_i32(&mut self.diffusion_delays, 4, 441);
        pad_or_truncate_i32(&mut self.er_delays, N_ER_TAPS, 100);
        pad_or_truncate_f64(&mut self.er_gains, N_ER_TAPS, 0.3);
        pad_or_truncate_f64(&mut self.er_pans, N_ER_TAPS, 0.0);
        if self.sample_rate <= 0.0 {
            self.sample_rate = 44100.0;
        }
    }
}

fn pad_or_truncate_f64(v: &mut Vec<f64>, n: usize, fill: f64) {
    v.truncate(n);
    while v.len() < n {
        v.push(fill);
    }
}

fn pad_or_truncate_i32(v: &mut Vec<i32>, n: usize, fill: i32) {
    v.truncate(n);
    while v.len() < n {
        v.push(fill);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_params() {
        let p = FdnParams::default();
        assert_eq!(p.delay_times.len(), N);
        assert_eq!(p.damping_coeffs.len(), N);
        assert_eq!(p.feedback_gain, 0.85);
        assert_eq!(p.wet_dry, 0.5);
        assert_eq!(p.matrix_type, "householder");
    }

    #[test]
    fn test_sparse_json_load() {
        let json = r#"{"feedback_gain": 0.9, "wet_dry": 0.7}"#;
        let p = FdnParams::from_json(json).unwrap();
        assert_eq!(p.feedback_gain, 0.9);
        assert_eq!(p.wet_dry, 0.7);
        assert_eq!(p.delay_times.len(), N);
    }

    #[test]
    fn test_float_to_int_coercion() {
        let json = r#"{"pre_delay": 441.0, "delay_times": [1310.0, 1637.0, 1821.0, 2112.0, 2342.0, 2615.0, 2986.0, 3223.0]}"#;
        let p = FdnParams::from_json(json).unwrap();
        assert_eq!(p.pre_delay, 441);
        assert_eq!(p.delay_times[0], 1310);
    }

    #[test]
    fn test_matrix_type_as_string() {
        let json = r#"{"matrix_type": "hadamard"}"#;
        let p = FdnParams::from_json(json).unwrap();
        assert_eq!(p.matrix_type, "hadamard");
    }

    #[test]
    fn test_matrix_type_as_int() {
        let json = r#"{"matrix_type": 3}"#;
        let p = FdnParams::from_json(json).unwrap();
        assert_eq!(p.matrix_type, "random_orthogonal");
    }

    #[test]
    fn test_has_modulation() {
        let mut p = FdnParams::default();
        assert!(!p.has_modulation());
        p.mod_master_rate = 1.0;
        p.mod_depth_delay[0] = 10.0;
        assert!(p.has_modulation());
    }

    #[test]
    fn test_preset_with_meta() {
        let json = r#"{
            "feedback_gain": 0.9,
            "_meta": {"category": "Large", "description": "test"}
        }"#;
        let p = FdnParams::from_json(json).unwrap();
        assert_eq!(p.feedback_gain, 0.9);
        assert!(p.meta.is_some());
    }

    #[test]
    fn test_normalize() {
        let mut p = FdnParams::default();
        p.delay_times = vec![1000, 2000]; // too short
        p.normalize();
        assert_eq!(p.delay_times.len(), N);
        assert_eq!(p.delay_times[2], 1000); // padded with first element
    }
}
