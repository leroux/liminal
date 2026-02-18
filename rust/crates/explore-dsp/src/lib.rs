pub mod effects;
pub mod primitives;
pub mod stft;

use serde_json::Value;
use std::collections::HashMap;

/// Output from an effect: either mono or stereo.
pub enum AudioOutput {
    Mono(Vec<f32>),
    Stereo(Vec<[f32; 2]>),
}

/// A registered effect with its processing function and optional variants.
pub struct EffectEntry {
    pub id: &'static str,
    pub process: fn(&[f32], u32, &HashMap<String, Value>) -> AudioOutput,
    pub variants: fn() -> Vec<HashMap<String, Value>>,
    pub category: &'static str,
}

/// Get all registered effects.
pub fn discover_effects() -> Vec<EffectEntry> {
    let mut all = Vec::new();
    all.extend(effects::a_delay::register());
    all.extend(effects::b_reverb::register());
    all.extend(effects::c_modulation::register());
    all.extend(effects::d_distortion::register());
    all.extend(effects::e_filter::register());
    all.extend(effects::f_dynamics::register());
    all.extend(effects::g_pitch_time::register());
    all.extend(effects::h_spectral::register());
    all.extend(effects::i_granular::register());
    all.extend(effects::j_chaos_math::register());
    all.extend(effects::k_neural::register());
    all.extend(effects::l_convolution::register());
    all.extend(effects::m_physical::register());
    all.extend(effects::n_lofi::register());
    all.extend(effects::o_spatial::register());
    all.extend(effects::p_envelope::register());
    all.extend(effects::q_combo::register());
    all.extend(effects::r_misc::register());
    all
}

/// Helper to get f32 param with default.
pub fn pf(params: &HashMap<String, Value>, key: &str, default: f32) -> f32 {
    params.get(key).and_then(|v| v.as_f64()).map(|v| v as f32).unwrap_or(default)
}

/// Helper to get i32 param with default.
pub fn pi(params: &HashMap<String, Value>, key: &str, default: i32) -> i32 {
    params.get(key).and_then(|v| v.as_i64()).map(|v| v as i32).unwrap_or(default)
}

/// Helper to get string param with default.
pub fn ps<'a>(params: &'a HashMap<String, Value>, key: &str, default: &'a str) -> &'a str {
    params.get(key).and_then(|v| v.as_str()).unwrap_or(default)
}

/// Helper to get u64 param with default.
pub fn pu(params: &HashMap<String, Value>, key: &str, default: u64) -> u64 {
    params.get(key).and_then(|v| v.as_u64()).unwrap_or(default)
}

/// Build a params map from key-value pairs.
#[macro_export]
macro_rules! params {
    ($($key:expr => $val:expr),* $(,)?) => {{
        let mut m = std::collections::HashMap::new();
        $(m.insert($key.to_string(), serde_json::json!($val));)*
        m
    }};
}
