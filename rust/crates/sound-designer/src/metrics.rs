//! Format audio metrics for LLM consumption.
//!
//! Used by the tuner to send readable, interpretive audio analysis to Claude.
//! Ported from Python `shared/audio_features.py`.

use std::collections::HashMap;
use std::fmt::Write;

/// Audio metrics from an analysis pass.
///
/// All fields are optional — analysis may not compute every metric.
#[derive(Debug, Clone, Default)]
pub struct AudioMetrics {
    pub rms_db: Option<f64>,
    pub peak_db: Option<f64>,
    pub rt60: Option<f64>,
    pub edt: Option<f64>,
    pub spectral_centroid: Option<f64>,
    pub echo_density: Option<f64>,
    pub crest_factor: Option<f64>,
    pub c50: Option<f64>,
    pub c80: Option<f64>,
    pub spectral_flatness: Option<f64>,
    pub bandwidth: Option<f64>,
    /// Comparison metrics (lossy pedal).
    pub energy_ratio_db: Option<f64>,
    pub thd_n_percent: Option<f64>,
    pub spectral_centroid_shift: Option<f64>,
    pub bandwidth_change: Option<f64>,
    /// Band RT60: frequency label -> seconds.
    pub rt60_bands: HashMap<String, f64>,
}

impl AudioMetrics {
    /// Create from a JSON map (as returned by analysis functions).
    pub fn from_json(map: &serde_json::Map<String, serde_json::Value>) -> Self {
        let f = |key: &str| map.get(key).and_then(|v| v.as_f64());
        let mut m = Self {
            rms_db: f("rms_db"),
            peak_db: f("peak_db"),
            rt60: f("rt60"),
            edt: f("edt"),
            spectral_centroid: f("spectral_centroid"),
            echo_density: f("echo_density"),
            crest_factor: f("crest_factor"),
            c50: f("c50"),
            c80: f("c80"),
            spectral_flatness: f("spectral_flatness"),
            bandwidth: f("bandwidth"),
            energy_ratio_db: f("energy_ratio_db"),
            thd_n_percent: f("thd_n_percent"),
            spectral_centroid_shift: f("spectral_centroid_shift"),
            bandwidth_change: f("bandwidth_change"),
            rt60_bands: HashMap::new(),
        };
        if let Some(bands) = map.get("rt60_bands").and_then(|v| v.as_object()) {
            for (freq, val) in bands {
                if let Some(v) = val.as_f64() {
                    m.rt60_bands.insert(freq.clone(), v);
                }
            }
        }
        m
    }
}

/// Format metrics for inclusion in an LLM prompt.
///
/// - `metrics`: Current output audio metrics.
/// - `prev_metrics`: Previous output metrics for A/B delta.
/// - `source_metrics`: Input/source audio metrics.
pub fn format_features(
    metrics: Option<&AudioMetrics>,
    prev_metrics: Option<&AudioMetrics>,
    source_metrics: Option<&AudioMetrics>,
) -> String {
    let mut parts = Vec::new();

    if let Some(src) = source_metrics {
        parts.push("INPUT AUDIO:".to_string());
        parts.push(format_common(src));
        parts.push(String::new());
    }

    if let Some(m) = metrics {
        parts.push("OUTPUT AUDIO (current render):".to_string());
        parts.push(format_output(m));

        if let Some(prev) = prev_metrics {
            let delta = format_delta(prev, m);
            if !delta.is_empty() {
                parts.push(String::new());
                parts.push("CHANGES FROM PREVIOUS RENDER:".to_string());
                parts.push(delta);
            }
        }
    }

    parts.join("\n")
}

fn format_common(m: &AudioMetrics) -> String {
    let mut lines = Vec::new();
    if let Some(v) = m.rms_db {
        lines.push(format!("  RMS level: {v:.1} dB"));
    }
    if let Some(v) = m.peak_db {
        lines.push(format!("  Peak level: {v:.1} dB"));
    }
    if let Some(v) = m.rt60 {
        lines.push(format!("  RT60: {v:.3}s"));
    }
    if let Some(v) = m.edt {
        lines.push(format!("  EDT: {v:.3}s"));
    }
    if let Some(v) = m.spectral_centroid {
        lines.push(format!("  Spectral centroid: {v:.0} Hz"));
    }
    if let Some(v) = m.echo_density {
        lines.push(format!("  Echo density: {v:.2}"));
    }
    if let Some(v) = m.crest_factor {
        lines.push(format!("  Crest factor: {v:.1} dB"));
    }
    if let Some(v) = m.c50 {
        lines.push(format!("  C50: {v:.1} dB"));
    }
    if let Some(v) = m.c80 {
        lines.push(format!("  C80: {v:.1} dB"));
    }
    if let Some(v) = m.spectral_flatness {
        lines.push(format!("  Spectral flatness: {v:.3}"));
    }
    if let Some(v) = m.bandwidth {
        lines.push(format!("  Bandwidth: {v:.0} Hz"));
    }
    lines.join("\n")
}

fn format_output(m: &AudioMetrics) -> String {
    let mut lines = Vec::new();
    // Common metrics
    lines.push(format_common(m));
    // Comparison metrics (lossy pedal)
    if let Some(v) = m.energy_ratio_db {
        lines.push(format!("  Energy ratio: {v:.1} dB"));
    }
    if let Some(v) = m.thd_n_percent {
        lines.push(format!("  THD+N: {v:.1}%"));
    }
    if let Some(v) = m.spectral_centroid_shift {
        lines.push(format!("  Centroid shift: {v:.0} Hz"));
    }
    if let Some(v) = m.bandwidth_change {
        lines.push(format!("  Bandwidth change: {v:.0} Hz"));
    }
    // Band RT60
    if !m.rt60_bands.is_empty() {
        let mut band_strs: Vec<String> = Vec::new();
        let mut bands: Vec<_> = m.rt60_bands.iter().collect();
        bands.sort_by(|a, b| {
            a.0.parse::<f64>()
                .unwrap_or(0.0)
                .partial_cmp(&b.0.parse::<f64>().unwrap_or(0.0))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for (freq, val) in bands {
            band_strs.push(format!("{freq}Hz={val:.2}s"));
        }
        lines.push(format!("  Band RT60: {}", band_strs.join(", ")));
    }
    lines.join("\n")
}

/// Delta definitions: (unit, precision, hint_up, hint_down).
struct DeltaDef {
    key: &'static str,
    unit: &'static str,
    precision: usize,
    hint_up: &'static str,
    hint_down: &'static str,
}

const DELTA_DEFS: &[DeltaDef] = &[
    DeltaDef { key: "rt60", unit: "s", precision: 3, hint_up: "longer tail", hint_down: "shorter tail" },
    DeltaDef { key: "edt", unit: "s", precision: 3, hint_up: "longer perceived", hint_down: "shorter perceived" },
    DeltaDef { key: "spectral_centroid", unit: "Hz", precision: 0, hint_up: "brighter", hint_down: "darker" },
    DeltaDef { key: "echo_density", unit: "", precision: 2, hint_up: "denser", hint_down: "sparser" },
    DeltaDef { key: "crest_factor", unit: "dB", precision: 1, hint_up: "more dynamic", hint_down: "more compressed" },
    DeltaDef { key: "c50", unit: "dB", precision: 1, hint_up: "clearer", hint_down: "more diffuse" },
    DeltaDef { key: "c80", unit: "dB", precision: 1, hint_up: "clearer", hint_down: "more diffuse" },
    DeltaDef { key: "spectral_flatness", unit: "", precision: 3, hint_up: "noisier", hint_down: "more tonal" },
    DeltaDef { key: "bandwidth", unit: "Hz", precision: 0, hint_up: "wider", hint_down: "narrower" },
    DeltaDef { key: "energy_ratio_db", unit: "dB", precision: 1, hint_up: "louder", hint_down: "quieter" },
    DeltaDef { key: "thd_n_percent", unit: "%", precision: 1, hint_up: "more distorted", hint_down: "cleaner" },
];

fn get_metric(m: &AudioMetrics, key: &str) -> Option<f64> {
    match key {
        "rt60" => m.rt60,
        "edt" => m.edt,
        "spectral_centroid" => m.spectral_centroid,
        "echo_density" => m.echo_density,
        "crest_factor" => m.crest_factor,
        "c50" => m.c50,
        "c80" => m.c80,
        "spectral_flatness" => m.spectral_flatness,
        "bandwidth" => m.bandwidth,
        "energy_ratio_db" => m.energy_ratio_db,
        "thd_n_percent" => m.thd_n_percent,
        _ => None,
    }
}

fn format_delta(prev: &AudioMetrics, curr: &AudioMetrics) -> String {
    let mut lines = Vec::new();

    for def in DELTA_DEFS {
        let old = match get_metric(prev, def.key) {
            Some(v) => v,
            None => continue,
        };
        let new = match get_metric(curr, def.key) {
            Some(v) => v,
            None => continue,
        };
        let diff = new - old;

        // Skip insignificant changes: >5% relative or meaningful absolute
        if old.abs() > 1e-6 {
            if (diff / old).abs() < 0.05 {
                continue;
            }
        } else if diff.abs() < 0.01 {
            continue;
        }

        let hint = if diff > 0.0 { def.hint_up } else { def.hint_down };
        let sign = if diff > 0.0 { "+" } else { "" };
        let p = def.precision;
        let u = def.unit;
        let mut line = String::new();
        write!(
            line,
            "  {}: {old:.p$} -> {new:.p$}{u} ({sign}{diff:.p$}{u}, {hint})",
            def.key
        )
        .unwrap();
        lines.push(line);
    }

    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_features_empty() {
        assert_eq!(format_features(None, None, None), "");
    }

    #[test]
    fn format_features_output_only() {
        let m = AudioMetrics {
            rms_db: Some(-12.0),
            peak_db: Some(-1.5),
            rt60: Some(2.345),
            ..Default::default()
        };
        let text = format_features(Some(&m), None, None);
        assert!(text.contains("OUTPUT AUDIO"));
        assert!(text.contains("RMS level: -12.0 dB"));
        assert!(text.contains("RT60: 2.345s"));
    }

    #[test]
    fn format_features_with_source() {
        let src = AudioMetrics {
            rms_db: Some(-20.0),
            ..Default::default()
        };
        let out = AudioMetrics {
            rms_db: Some(-12.0),
            ..Default::default()
        };
        let text = format_features(Some(&out), None, Some(&src));
        assert!(text.contains("INPUT AUDIO"));
        assert!(text.contains("OUTPUT AUDIO"));
    }

    #[test]
    fn format_delta_significant_change() {
        let prev = AudioMetrics {
            rt60: Some(1.0),
            spectral_centroid: Some(2000.0),
            ..Default::default()
        };
        let curr = AudioMetrics {
            rt60: Some(2.0),
            spectral_centroid: Some(3000.0),
            ..Default::default()
        };
        let text = format_features(Some(&curr), Some(&prev), None);
        assert!(text.contains("CHANGES FROM PREVIOUS RENDER"));
        assert!(text.contains("longer tail"));
        assert!(text.contains("brighter"));
    }

    #[test]
    fn format_delta_insignificant_filtered() {
        let prev = AudioMetrics {
            rt60: Some(1.0),
            ..Default::default()
        };
        let curr = AudioMetrics {
            rt60: Some(1.01), // <5% change
            ..Default::default()
        };
        let text = format_features(Some(&curr), Some(&prev), None);
        assert!(!text.contains("CHANGES"));
    }

    #[test]
    fn from_json_roundtrip() {
        let mut map = serde_json::Map::new();
        map.insert("rms_db".into(), serde_json::json!(-12.5));
        map.insert("rt60".into(), serde_json::json!(2.3));
        map.insert(
            "rt60_bands".into(),
            serde_json::json!({"250": 1.5, "1000": 2.0}),
        );

        let m = AudioMetrics::from_json(&map);
        assert_eq!(m.rms_db, Some(-12.5));
        assert_eq!(m.rt60, Some(2.3));
        assert_eq!(m.rt60_bands.len(), 2);
        assert_eq!(m.rt60_bands["250"], 1.5);
    }
}
