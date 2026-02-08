//! Preset loading for the Lossy plugin.
//!
//! Reads JSON preset files from the lossy/gui/presets/ directory
//! and applies them to nih-plug parameters.

use crate::params::*;
use nih_plug::prelude::*;
use std::path::{Path, PathBuf};

include!(concat!(env!("OUT_DIR"), "/embedded_presets.rs"));

/// A loaded preset with name and parsed params.
#[derive(Debug, Clone)]
pub struct Preset {
    pub name: String,
    pub category: String,
    pub description: String,
    pub params: lossy_dsp::LossyParams,
}

/// Find the preset directory. Searches for lossy/gui/presets/ relative to
/// the executable or well-known locations.
pub fn find_preset_dir() -> Option<PathBuf> {
    // Try relative to executable
    if let Ok(exe) = std::env::current_exe() {
        // Navigate up from the binary location to find the project root
        for ancestor in exe.ancestors().skip(1) {
            let candidate = ancestor.join("lossy/gui/presets");
            if candidate.is_dir() {
                return Some(candidate);
            }
        }
    }

    // Try relative to current directory
    let cwd_candidates = [
        "lossy/gui/presets",
        "../lossy/gui/presets",
        "../../lossy/gui/presets",
        "../../../lossy/gui/presets",
    ];
    for c in &cwd_candidates {
        let p = PathBuf::from(c);
        if p.is_dir() {
            return Some(p);
        }
    }

    // Try an environment variable
    if let Ok(dir) = std::env::var("LOSSY_PRESET_DIR") {
        let p = PathBuf::from(dir);
        if p.is_dir() {
            return Some(p);
        }
    }

    None
}

/// Load all presets from a directory.
pub fn load_presets(dir: &Path) -> Vec<Preset> {
    let mut presets = Vec::new();
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return presets,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        let json = match std::fs::read_to_string(&path) {
            Ok(s) => s,
            Err(_) => continue,
        };

        // Extract _meta before parsing params
        let (category, description) = extract_meta(&json);

        let params = match lossy_dsp::LossyParams::from_json(&json) {
            Ok(p) => p,
            Err(_) => continue,
        };

        presets.push(Preset {
            name,
            category,
            description,
            params,
        });
    }

    presets.sort_by(|a, b| a.category.cmp(&b.category).then(a.name.cmp(&b.name)));
    presets
}

/// Load presets embedded at compile time.
pub fn load_embedded_presets() -> Vec<Preset> {
    let mut presets = Vec::new();
    for (name, json) in EMBEDDED_PRESETS {
        let (category, description) = extract_meta(json);
        let params = match lossy_dsp::LossyParams::from_json(json) {
            Ok(p) => p,
            Err(_) => continue,
        };
        presets.push(Preset {
            name: name.to_string(),
            category,
            description,
            params,
        });
    }
    presets.sort_by(|a, b| a.category.cmp(&b.category).then(a.name.cmp(&b.name)));
    presets
}

fn extract_meta(json: &str) -> (String, String) {
    if let Ok(val) = serde_json::from_str::<serde_json::Value>(json) {
        let meta = val.get("_meta").cloned().unwrap_or_default();
        let cat = meta
            .get("category")
            .and_then(|v| v.as_str())
            .unwrap_or("Uncategorized")
            .to_string();
        let desc = meta
            .get("description")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        (cat, desc)
    } else {
        ("Uncategorized".to_string(), String::new())
    }
}

/// Apply a preset's DSP params to the nih-plug parameter set.
pub fn apply_preset(preset: &Preset, plugin_params: &LossyPluginParams, setter: &ParamSetter) {
    let p = &preset.params;

    set_enum(setter, &plugin_params.mode, p.inverse);
    set_float(setter, &plugin_params.jitter, p.jitter);
    set_float(setter, &plugin_params.loss, p.loss);
    set_int(setter, &plugin_params.window_size, p.window_size);
    set_int(setter, &plugin_params.hop_divisor, p.hop_divisor);
    set_int(setter, &plugin_params.n_bands, p.n_bands);
    set_float(setter, &plugin_params.global_amount, p.global_amount);
    set_float(setter, &plugin_params.phase_loss, p.phase_loss);
    set_enum(setter, &plugin_params.quantizer, p.quantizer);
    set_float(setter, &plugin_params.pre_echo, p.pre_echo);
    set_float(setter, &plugin_params.noise_shape, p.noise_shape);
    set_float(setter, &plugin_params.weighting, p.weighting);
    set_float(setter, &plugin_params.hf_threshold, p.hf_threshold);
    set_float(setter, &plugin_params.transient_ratio, p.transient_ratio);
    set_float(setter, &plugin_params.slushy_rate, p.slushy_rate);
    set_float(setter, &plugin_params.crush, p.crush);
    set_float(setter, &plugin_params.decimate, p.decimate);
    set_enum(setter, &plugin_params.packets, p.packets);
    set_float(setter, &plugin_params.packet_rate, p.packet_rate);
    set_float(setter, &plugin_params.packet_size, p.packet_size);
    set_enum(setter, &plugin_params.filter_type, p.filter_type);
    set_float(setter, &plugin_params.filter_freq, p.filter_freq);
    set_float(setter, &plugin_params.filter_width, p.filter_width);
    set_filter_slope(setter, &plugin_params.filter_slope, p.filter_slope);
    set_float(setter, &plugin_params.verb, p.verb);
    set_float(setter, &plugin_params.decay, p.decay);
    set_enum(setter, &plugin_params.verb_position, p.verb_position);
    set_bool(setter, &plugin_params.freeze, p.freeze != 0);
    set_enum(setter, &plugin_params.freeze_mode, p.freeze_mode);
    set_float(setter, &plugin_params.freezer, p.freezer);
    set_float(setter, &plugin_params.gate, p.gate);
    set_float(setter, &plugin_params.threshold, p.threshold);
    set_float(setter, &plugin_params.auto_gain, p.auto_gain);
    set_float(setter, &plugin_params.loss_gain, p.loss_gain);
    set_float(setter, &plugin_params.wet_dry, p.wet_dry);
}

fn set_float(setter: &ParamSetter, param: &FloatParam, value: f64) {
    setter.begin_set_parameter(param);
    setter.set_parameter(param, value as f32);
    setter.end_set_parameter(param);
}

fn set_int(setter: &ParamSetter, param: &IntParam, value: i32) {
    setter.begin_set_parameter(param);
    setter.set_parameter(param, value);
    setter.end_set_parameter(param);
}

fn set_bool(setter: &ParamSetter, param: &BoolParam, value: bool) {
    setter.begin_set_parameter(param);
    setter.set_parameter(param, value);
    setter.end_set_parameter(param);
}

fn set_enum<T: Enum + PartialEq>(setter: &ParamSetter, param: &EnumParam<T>, index: i32) {
    setter.begin_set_parameter(param);
    setter.set_parameter(param, T::from_index(index as usize));
    setter.end_set_parameter(param);
}

fn set_filter_slope(setter: &ParamSetter, param: &EnumParam<FilterSlope>, value: i32) {
    let slope = match value {
        6 => FilterSlope::Slope6,
        96 => FilterSlope::Slope96,
        _ => FilterSlope::Slope24,
    };
    setter.begin_set_parameter(param);
    setter.set_parameter(param, slope);
    setter.end_set_parameter(param);
}
