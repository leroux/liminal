//! Preset loading for the Fractal plugin.
//!
//! Reads JSON preset files from the fractal/gui/presets/ directory
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
    pub params: fractal_dsp::FractalParams,
}

/// Find the preset directory. Searches for fractal/gui/presets/ relative to
/// the executable or well-known locations.
pub fn find_preset_dir() -> Option<PathBuf> {
    // Try relative to executable
    if let Ok(exe) = std::env::current_exe() {
        for ancestor in exe.ancestors().skip(1) {
            let candidate = ancestor.join("fractal/gui/presets");
            if candidate.is_dir() {
                return Some(candidate);
            }
        }
    }

    // Try relative to current directory
    let cwd_candidates = [
        "fractal/gui/presets",
        "../fractal/gui/presets",
        "../../fractal/gui/presets",
        "../../../fractal/gui/presets",
    ];
    for c in &cwd_candidates {
        let p = PathBuf::from(c);
        if p.is_dir() {
            return Some(p);
        }
    }

    // Try an environment variable
    if let Ok(dir) = std::env::var("FRACTAL_PRESET_DIR") {
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

        let (category, description) = extract_meta(&json);

        let params = match fractal_dsp::FractalParams::from_json(&json) {
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
        let params = match fractal_dsp::FractalParams::from_json(json) {
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
pub fn apply_preset(preset: &Preset, plugin_params: &FractalPluginParams, setter: &ParamSetter) {
    let p = &preset.params;

    set_int(setter, &plugin_params.num_scales, p.num_scales);
    set_float(setter, &plugin_params.scale_ratio, p.scale_ratio);
    set_float(setter, &plugin_params.amplitude_decay, p.amplitude_decay);
    set_enum(setter, &plugin_params.interp, p.interp);
    set_bool(setter, &plugin_params.reverse_scales, p.reverse_scales != 0);
    set_float(setter, &plugin_params.scale_offset, p.scale_offset);
    set_int(setter, &plugin_params.iterations, p.iterations);
    set_float(setter, &plugin_params.iter_decay, p.iter_decay);
    set_float(setter, &plugin_params.saturation, p.saturation);
    set_float(setter, &plugin_params.spectral, p.spectral);
    set_int(setter, &plugin_params.window_size, p.window_size);
    set_enum(setter, &plugin_params.filter_type, p.filter_type);
    set_float(setter, &plugin_params.filter_freq, p.filter_freq);
    set_float(setter, &plugin_params.filter_q, p.filter_q);
    set_enum(setter, &plugin_params.post_filter_type, p.post_filter_type);
    set_float(setter, &plugin_params.post_filter_freq, p.post_filter_freq);
    set_float(setter, &plugin_params.gate, p.gate);
    set_float(setter, &plugin_params.crush, p.crush);
    set_float(setter, &plugin_params.decimate, p.decimate);
    set_float(setter, &plugin_params.layer_gain_1, p.layer_gain_1);
    set_float(setter, &plugin_params.layer_gain_2, p.layer_gain_2);
    set_float(setter, &plugin_params.layer_gain_3, p.layer_gain_3);
    set_float(setter, &plugin_params.layer_gain_4, p.layer_gain_4);
    set_float(setter, &plugin_params.layer_gain_5, p.layer_gain_5);
    set_float(setter, &plugin_params.layer_gain_6, p.layer_gain_6);
    set_float(setter, &plugin_params.layer_gain_7, p.layer_gain_7);
    set_bool(setter, &plugin_params.fractal_only_wet, p.fractal_only_wet != 0);
    set_float(setter, &plugin_params.layer_spread, p.layer_spread);
    set_float(setter, &plugin_params.layer_detune, p.layer_detune);
    set_float(setter, &plugin_params.layer_delay, p.layer_delay);
    set_float(setter, &plugin_params.layer_tilt, p.layer_tilt);
    set_float(setter, &plugin_params.feedback, p.feedback);
    set_float(setter, &plugin_params.wet_dry, p.wet_dry);
    set_float(setter, &plugin_params.output_gain, p.output_gain);
    set_float(setter, &plugin_params.threshold, p.threshold);
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
