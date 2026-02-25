//! Preset loading for the Reverb plugin.
//!
//! Reads JSON preset files from the reverb/gui/presets/ directory
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
    pub params: reverb_dsp::ReverbParams,
}

/// Find the preset directory. Searches for reverb/gui/presets/ relative to
/// the executable or well-known locations.
pub fn find_preset_dir() -> Option<PathBuf> {
    // Try relative to executable
    if let Ok(exe) = std::env::current_exe() {
        for ancestor in exe.ancestors().skip(1) {
            let candidate = ancestor.join("reverb/gui/presets");
            if candidate.is_dir() {
                return Some(candidate);
            }
        }
    }

    // Try relative to current directory
    let cwd_candidates = [
        "reverb/gui/presets",
        "../reverb/gui/presets",
        "../../reverb/gui/presets",
        "../../../reverb/gui/presets",
    ];
    for c in &cwd_candidates {
        let p = PathBuf::from(c);
        if p.is_dir() {
            return Some(p);
        }
    }

    // Try an environment variable
    if let Ok(dir) = std::env::var("REVERB_PRESET_DIR") {
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

        let params = match reverb_dsp::ReverbParams::from_json(&json) {
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
        let params = match reverb_dsp::ReverbParams::from_json(json) {
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

/// Helper: get array element or default.
fn arr_f64(arr: &[f64], i: usize, default: f64) -> f64 {
    arr.get(i).copied().unwrap_or(default)
}

fn arr_i32(arr: &[i32], i: usize, default: i32) -> i32 {
    arr.get(i).copied().unwrap_or(default)
}

/// Apply a preset's DSP params to the nih-plug parameter set.
pub fn apply_preset(preset: &Preset, plugin_params: &ReverbPluginParams, setter: &ParamSetter) {
    let p = &preset.params;

    // Global
    set_float(setter, &plugin_params.feedback_gain, p.feedback_gain);
    set_float(setter, &plugin_params.wet_dry, p.wet_dry);
    set_float(setter, &plugin_params.diffusion, p.diffusion);
    set_int(setter, &plugin_params.diffusion_stages, p.diffusion_stages);
    set_float(setter, &plugin_params.saturation, p.saturation);
    set_int(setter, &plugin_params.pre_delay, p.pre_delay);
    set_float(setter, &plugin_params.stereo_width, p.stereo_width);

    // Matrix
    set_matrix_type(setter, &plugin_params.matrix_type, &p.matrix_type);
    set_int(setter, &plugin_params.matrix_seed, p.matrix_seed);

    // Delay times
    let dt = &p.delay_times;
    set_int(setter, &plugin_params.delay_time_1, arr_i32(dt, 0, 1310));
    set_int(setter, &plugin_params.delay_time_2, arr_i32(dt, 1, 1637));
    set_int(setter, &plugin_params.delay_time_3, arr_i32(dt, 2, 1821));
    set_int(setter, &plugin_params.delay_time_4, arr_i32(dt, 3, 2112));
    set_int(setter, &plugin_params.delay_time_5, arr_i32(dt, 4, 2342));
    set_int(setter, &plugin_params.delay_time_6, arr_i32(dt, 5, 2615));
    set_int(setter, &plugin_params.delay_time_7, arr_i32(dt, 6, 2986));
    set_int(setter, &plugin_params.delay_time_8, arr_i32(dt, 7, 3223));

    // Damping
    let dc = &p.damping_coeffs;
    set_float(setter, &plugin_params.damping_1, arr_f64(dc, 0, 0.3));
    set_float(setter, &plugin_params.damping_2, arr_f64(dc, 1, 0.3));
    set_float(setter, &plugin_params.damping_3, arr_f64(dc, 2, 0.3));
    set_float(setter, &plugin_params.damping_4, arr_f64(dc, 3, 0.3));
    set_float(setter, &plugin_params.damping_5, arr_f64(dc, 4, 0.3));
    set_float(setter, &plugin_params.damping_6, arr_f64(dc, 5, 0.3));
    set_float(setter, &plugin_params.damping_7, arr_f64(dc, 6, 0.3));
    set_float(setter, &plugin_params.damping_8, arr_f64(dc, 7, 0.3));

    // Input gains
    let ig = &p.input_gains;
    set_float(setter, &plugin_params.input_gain_1, arr_f64(ig, 0, 0.125));
    set_float(setter, &plugin_params.input_gain_2, arr_f64(ig, 1, 0.125));
    set_float(setter, &plugin_params.input_gain_3, arr_f64(ig, 2, 0.125));
    set_float(setter, &plugin_params.input_gain_4, arr_f64(ig, 3, 0.125));
    set_float(setter, &plugin_params.input_gain_5, arr_f64(ig, 4, 0.125));
    set_float(setter, &plugin_params.input_gain_6, arr_f64(ig, 5, 0.125));
    set_float(setter, &plugin_params.input_gain_7, arr_f64(ig, 6, 0.125));
    set_float(setter, &plugin_params.input_gain_8, arr_f64(ig, 7, 0.125));

    // Output gains
    let og = &p.output_gains;
    set_float(setter, &plugin_params.output_gain_1, arr_f64(og, 0, 1.0));
    set_float(setter, &plugin_params.output_gain_2, arr_f64(og, 1, 1.0));
    set_float(setter, &plugin_params.output_gain_3, arr_f64(og, 2, 1.0));
    set_float(setter, &plugin_params.output_gain_4, arr_f64(og, 3, 1.0));
    set_float(setter, &plugin_params.output_gain_5, arr_f64(og, 4, 1.0));
    set_float(setter, &plugin_params.output_gain_6, arr_f64(og, 5, 1.0));
    set_float(setter, &plugin_params.output_gain_7, arr_f64(og, 6, 1.0));
    set_float(setter, &plugin_params.output_gain_8, arr_f64(og, 7, 1.0));

    // Node pans
    let np = &p.node_pans;
    set_float(setter, &plugin_params.node_pan_1, arr_f64(np, 0, -1.0));
    set_float(setter, &plugin_params.node_pan_2, arr_f64(np, 1, -0.714));
    set_float(setter, &plugin_params.node_pan_3, arr_f64(np, 2, -0.429));
    set_float(setter, &plugin_params.node_pan_4, arr_f64(np, 3, -0.143));
    set_float(setter, &plugin_params.node_pan_5, arr_f64(np, 4, 0.143));
    set_float(setter, &plugin_params.node_pan_6, arr_f64(np, 5, 0.429));
    set_float(setter, &plugin_params.node_pan_7, arr_f64(np, 6, 0.714));
    set_float(setter, &plugin_params.node_pan_8, arr_f64(np, 7, 1.0));

    // Modulation
    set_float(setter, &plugin_params.mod_master_rate, p.mod_master_rate);
    set_float(setter, &plugin_params.mod_correlation, p.mod_correlation);
    set_enum(setter, &plugin_params.mod_waveform, p.mod_waveform);
    // For uniform mod depths, take the average of the per-node array
    set_float(
        setter,
        &plugin_params.mod_depth_delay,
        avg_f64(&p.mod_depth_delay),
    );
    set_float(
        setter,
        &plugin_params.mod_depth_damping,
        avg_f64(&p.mod_depth_damping),
    );
    set_float(
        setter,
        &plugin_params.mod_depth_output,
        avg_f64(&p.mod_depth_output),
    );
    set_float(setter, &plugin_params.mod_depth_matrix, p.mod_depth_matrix);
    set_float(
        setter,
        &plugin_params.mod_rate_scale_delay,
        p.mod_rate_scale_delay,
    );
    set_float(
        setter,
        &plugin_params.mod_rate_scale_damping,
        p.mod_rate_scale_damping,
    );
    set_float(
        setter,
        &plugin_params.mod_rate_scale_output,
        p.mod_rate_scale_output,
    );
    set_float(setter, &plugin_params.mod_rate_matrix, p.mod_rate_matrix);
    set_matrix_type(setter, &plugin_params.mod_matrix2_type, &p.mod_matrix2_type);
    set_int(setter, &plugin_params.mod_matrix2_seed, p.mod_matrix2_seed);
}

fn avg_f64(arr: &[f64]) -> f64 {
    if arr.is_empty() {
        0.0
    } else {
        arr.iter().sum::<f64>() / arr.len() as f64
    }
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

fn set_enum<T: Enum + PartialEq>(setter: &ParamSetter, param: &EnumParam<T>, index: i32) {
    setter.begin_set_parameter(param);
    setter.set_parameter(param, T::from_index(index as usize));
    setter.end_set_parameter(param);
}

fn set_matrix_type(setter: &ParamSetter, param: &EnumParam<MatrixType>, name: &str) {
    let mt = match name {
        "householder" => MatrixType::Householder,
        "hadamard" => MatrixType::Hadamard,
        "diagonal" => MatrixType::Diagonal,
        "random_orthogonal" => MatrixType::RandomOrthogonal,
        "circulant" => MatrixType::Circulant,
        "stautner_puckette" => MatrixType::StautnerPuckette,
        _ => MatrixType::Householder,
    };
    setter.begin_set_parameter(param);
    setter.set_parameter(param, mt);
    setter.end_set_parameter(param);
}
