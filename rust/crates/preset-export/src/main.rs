//! Export JSON presets to .vstpreset files for Ableton / other VST3 hosts.
//!
//! Reads the JSON preset directories and writes .vstpreset files to
//! ~/Library/Audio/Presets/reverb-project/<Plugin Name>/

use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// VST3 class IDs (must match the plugins' Vst3Plugin::VST3_CLASS_ID)
// ---------------------------------------------------------------------------

const REVERB_CID: [u8; 16] = *b"ReverbFDNPlugin_";
const LOSSY_CID: [u8; 16] = *b"LossyCodecEmul!_";
const FRACTAL_CID: [u8; 16] = *b"FractalAudioFx!_";

const PLUGIN_VERSION: &str = "0.1.0";

// ---------------------------------------------------------------------------
// vstpreset binary writer
// ---------------------------------------------------------------------------

fn write_vstpreset(class_id: &[u8; 16], state_json: &[u8]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(48 + state_json.len() + 8 + 20);

    // Header (48 bytes)
    buf.extend_from_slice(b"VST3"); // magic
    buf.extend_from_slice(&1i32.to_le_bytes()); // format version
    // Class ID: 16 raw bytes → 32 uppercase hex ASCII chars
    for &b in class_id {
        buf.push(hex_nibble(b >> 4));
        buf.push(hex_nibble(b & 0x0F));
    }
    // Placeholder for chunk list offset (will be patched)
    let list_offset_pos = buf.len();
    buf.extend_from_slice(&0i64.to_le_bytes());

    assert_eq!(buf.len(), 48);

    // Comp chunk data (starts at byte 48)
    let comp_offset: i64 = 48;
    let comp_size = state_json.len() as i64;
    buf.extend_from_slice(state_json);

    // Chunk list
    let chunk_list_offset = buf.len() as i64;
    buf.extend_from_slice(b"List"); // list magic
    buf.extend_from_slice(&1i32.to_le_bytes()); // 1 entry

    // Entry: Comp
    buf.extend_from_slice(b"Comp");
    buf.extend_from_slice(&comp_offset.to_le_bytes());
    buf.extend_from_slice(&comp_size.to_le_bytes());

    // Patch the chunk list offset in the header
    buf[list_offset_pos..list_offset_pos + 8]
        .copy_from_slice(&chunk_list_offset.to_le_bytes());

    buf
}

fn hex_nibble(n: u8) -> u8 {
    match n {
        0..=9 => b'0' + n,
        10..=15 => b'A' + (n - 10),
        _ => b'0',
    }
}

// ---------------------------------------------------------------------------
// nih-plug PluginState JSON builder
// ---------------------------------------------------------------------------

fn make_plugin_state(params: BTreeMap<String, Value>) -> Vec<u8> {
    let mut fields = BTreeMap::new();
    fields.insert(
        "editor-state".to_string(),
        Value::String("{\"scale_factor\":1.0}".to_string()),
    );

    let state = json!({
        "version": PLUGIN_VERSION,
        "params": params,
        "fields": fields,
    });

    serde_json::to_vec(&state).expect("serialize state")
}

fn pv_f32(v: f64) -> Value {
    json!({"f32": v as f32})
}

fn pv_i32(v: i32) -> Value {
    json!({"i32": v})
}

fn pv_bool(v: bool) -> Value {
    json!({"bool": v})
}

fn arr_f64(arr: &[f64], i: usize, def: f64) -> f64 {
    arr.get(i).copied().unwrap_or(def)
}

fn arr_i32(arr: &[i32], i: usize, def: i32) -> i32 {
    arr.get(i).copied().unwrap_or(def)
}

fn avg_f64(arr: &[f64]) -> f64 {
    if arr.is_empty() {
        0.0
    } else {
        arr.iter().sum::<f64>() / arr.len() as f64
    }
}

// ---------------------------------------------------------------------------
// Reverb: DSP params → nih-plug param map
// ---------------------------------------------------------------------------

fn reverb_params_to_nih(p: &reverb_dsp::ReverbParams) -> BTreeMap<String, Value> {
    let mut m = BTreeMap::new();

    // Global
    m.insert("feedback_gain".into(), pv_f32(p.feedback_gain));
    m.insert("wet_dry".into(), pv_f32(p.wet_dry));
    m.insert("diffusion".into(), pv_f32(p.diffusion));
    m.insert("diffusion_stages".into(), pv_i32(p.diffusion_stages));
    m.insert("saturation".into(), pv_f32(p.saturation));
    m.insert("pre_delay".into(), pv_i32(p.pre_delay));
    m.insert("stereo_width".into(), pv_f32(p.stereo_width));

    // Matrix
    m.insert("matrix_type".into(), pv_i32(matrix_type_index(&p.matrix_type)));
    m.insert("matrix_seed".into(), pv_i32(p.matrix_seed));

    // Delay times
    let dt = &p.delay_times;
    for (i, (def, name)) in [
        (1310, "delay_time_1"),
        (1637, "delay_time_2"),
        (1821, "delay_time_3"),
        (2112, "delay_time_4"),
        (2342, "delay_time_5"),
        (2615, "delay_time_6"),
        (2986, "delay_time_7"),
        (3223, "delay_time_8"),
    ]
    .iter()
    .enumerate()
    {
        m.insert(name.to_string(), pv_i32(arr_i32(dt, i, *def)));
    }

    // Damping
    let dc = &p.damping_coeffs;
    for i in 0..8 {
        m.insert(
            format!("damping_{}", i + 1),
            pv_f32(arr_f64(dc, i, 0.3)),
        );
    }

    // Input gains
    let ig = &p.input_gains;
    for i in 0..8 {
        m.insert(
            format!("input_gain_{}", i + 1),
            pv_f32(arr_f64(ig, i, 0.125)),
        );
    }

    // Output gains
    let og = &p.output_gains;
    for i in 0..8 {
        m.insert(
            format!("output_gain_{}", i + 1),
            pv_f32(arr_f64(og, i, 1.0)),
        );
    }

    // Node pans
    let np = &p.node_pans;
    let pan_defaults = [-1.0, -0.714, -0.429, -0.143, 0.143, 0.429, 0.714, 1.0];
    for i in 0..8 {
        m.insert(
            format!("node_pan_{}", i + 1),
            pv_f32(arr_f64(np, i, pan_defaults[i])),
        );
    }

    // Modulation
    m.insert("mod_master_rate".into(), pv_f32(p.mod_master_rate));
    m.insert("mod_correlation".into(), pv_f32(p.mod_correlation));
    m.insert("mod_waveform".into(), pv_i32(p.mod_waveform));
    m.insert("mod_depth_delay".into(), pv_f32(avg_f64(&p.mod_depth_delay)));
    m.insert("mod_depth_damping".into(), pv_f32(avg_f64(&p.mod_depth_damping)));
    m.insert("mod_depth_output".into(), pv_f32(avg_f64(&p.mod_depth_output)));
    m.insert("mod_depth_matrix".into(), pv_f32(p.mod_depth_matrix));
    m.insert("mod_rate_scale_delay".into(), pv_f32(p.mod_rate_scale_delay));
    m.insert("mod_rate_scale_damping".into(), pv_f32(p.mod_rate_scale_damping));
    m.insert("mod_rate_scale_output".into(), pv_f32(p.mod_rate_scale_output));
    m.insert("mod_rate_matrix".into(), pv_f32(p.mod_rate_matrix));
    m.insert(
        "mod_matrix2_type".into(),
        pv_i32(matrix_type_index(&p.mod_matrix2_type)),
    );
    m.insert("mod_matrix2_seed".into(), pv_i32(p.mod_matrix2_seed));

    m
}

fn matrix_type_index(name: &str) -> i32 {
    match name {
        "householder" => 0,
        "hadamard" => 1,
        "diagonal" => 2,
        "random_orthogonal" => 3,
        "circulant" => 4,
        "stautner_puckette" => 5,
        _ => 0,
    }
}

// ---------------------------------------------------------------------------
// Lossy: DSP params → nih-plug param map
// ---------------------------------------------------------------------------

fn lossy_params_to_nih(p: &lossy_dsp::LossyParams) -> BTreeMap<String, Value> {
    let mut m = BTreeMap::new();

    // Spectral loss
    m.insert("mode".into(), pv_i32(p.inverse));
    m.insert("jitter".into(), pv_f32(p.jitter));
    m.insert("loss".into(), pv_f32(p.loss));
    m.insert("window_size".into(), pv_i32(p.window_size));
    m.insert("hop_divisor".into(), pv_i32(p.hop_divisor));
    m.insert("n_bands".into(), pv_i32(p.n_bands));
    m.insert("global_amount".into(), pv_f32(p.global_amount));
    m.insert("phase_loss".into(), pv_f32(p.phase_loss));
    m.insert("quantizer".into(), pv_i32(p.quantizer));
    m.insert("pre_echo".into(), pv_f32(p.pre_echo));
    m.insert("noise_shape".into(), pv_f32(p.noise_shape));
    m.insert("weighting".into(), pv_f32(p.weighting));
    m.insert("hf_threshold".into(), pv_f32(p.hf_threshold));
    m.insert("transient_ratio".into(), pv_f32(p.transient_ratio));
    m.insert("slushy_rate".into(), pv_f32(p.slushy_rate));

    // Crush
    m.insert("crush".into(), pv_f32(p.crush));
    m.insert("decimate".into(), pv_f32(p.decimate));

    // Packets
    m.insert("packets".into(), pv_i32(p.packets));
    m.insert("packet_rate".into(), pv_f32(p.packet_rate));
    m.insert("packet_size".into(), pv_f32(p.packet_size));

    // Filter
    m.insert("filter_type".into(), pv_i32(p.filter_type));
    m.insert("filter_freq".into(), pv_f32(p.filter_freq));
    m.insert("filter_width".into(), pv_f32(p.filter_width));
    // filter_slope: DSP uses literal dB/oct (6, 24, 96) → enum index
    m.insert("filter_slope".into(), pv_i32(filter_slope_index(p.filter_slope)));

    // Reverb
    m.insert("verb".into(), pv_f32(p.verb));
    m.insert("decay".into(), pv_f32(p.decay));
    m.insert("verb_position".into(), pv_i32(p.verb_position));

    // Freeze
    m.insert("freeze".into(), pv_bool(p.freeze != 0));
    m.insert("freeze_mode".into(), pv_i32(p.freeze_mode));
    m.insert("freezer".into(), pv_f32(p.freezer));

    // Gate / hidden / output
    m.insert("gate".into(), pv_f32(p.gate));
    m.insert("threshold".into(), pv_f32(p.threshold));
    m.insert("auto_gain".into(), pv_f32(p.auto_gain));
    m.insert("loss_gain".into(), pv_f32(p.loss_gain));
    m.insert("wet_dry".into(), pv_f32(p.wet_dry));

    m
}

fn filter_slope_index(slope: i32) -> i32 {
    match slope {
        6 => 0,  // FilterSlope::Slope6
        96 => 2, // FilterSlope::Slope96
        _ => 1,  // FilterSlope::Slope24 (default)
    }
}

// ---------------------------------------------------------------------------
// Fractal: DSP params → nih-plug param map
// ---------------------------------------------------------------------------

fn fractal_params_to_nih(p: &fractal_dsp::FractalParams) -> BTreeMap<String, Value> {
    let mut m = BTreeMap::new();

    // Core fractal
    m.insert("num_scales".into(), pv_i32(p.num_scales));
    m.insert("scale_ratio".into(), pv_f32(p.scale_ratio));
    m.insert("amplitude_decay".into(), pv_f32(p.amplitude_decay));
    m.insert("interp".into(), pv_i32(p.interp));
    m.insert("reverse_scales".into(), pv_bool(p.reverse_scales != 0));
    m.insert("scale_offset".into(), pv_f32(p.scale_offset));

    // Iteration / feedback
    m.insert("iterations".into(), pv_i32(p.iterations));
    m.insert("iter_decay".into(), pv_f32(p.iter_decay));
    m.insert("saturation".into(), pv_f32(p.saturation));

    // Spectral
    m.insert("spectral".into(), pv_f32(p.spectral));
    m.insert("window_size".into(), pv_i32(p.window_size));

    // Pre-filter
    m.insert("filter_type".into(), pv_i32(p.filter_type));
    m.insert("filter_freq".into(), pv_f32(p.filter_freq));
    m.insert("filter_q".into(), pv_f32(p.filter_q));

    // Post-filter
    m.insert("post_filter_type".into(), pv_i32(p.post_filter_type));
    m.insert("post_filter_freq".into(), pv_f32(p.post_filter_freq));

    // Effects
    m.insert("gate".into(), pv_f32(p.gate));
    m.insert("crush".into(), pv_f32(p.crush));
    m.insert("decimate".into(), pv_f32(p.decimate));

    // Layers
    m.insert("layer_gain_1".into(), pv_f32(p.layer_gain_1));
    m.insert("layer_gain_2".into(), pv_f32(p.layer_gain_2));
    m.insert("layer_gain_3".into(), pv_f32(p.layer_gain_3));
    m.insert("layer_gain_4".into(), pv_f32(p.layer_gain_4));
    m.insert("layer_gain_5".into(), pv_f32(p.layer_gain_5));
    m.insert("layer_gain_6".into(), pv_f32(p.layer_gain_6));
    m.insert("layer_gain_7".into(), pv_f32(p.layer_gain_7));
    m.insert("fractal_only_wet".into(), pv_bool(p.fractal_only_wet != 0));
    m.insert("layer_spread".into(), pv_f32(p.layer_spread));
    m.insert("layer_detune".into(), pv_f32(p.layer_detune));
    m.insert("layer_delay".into(), pv_f32(p.layer_delay));
    m.insert("layer_tilt".into(), pv_f32(p.layer_tilt));

    // Feedback / output
    m.insert("feedback".into(), pv_f32(p.feedback));
    m.insert("wet_dry".into(), pv_f32(p.wet_dry));
    m.insert("output_gain".into(), pv_f32(p.output_gain));
    m.insert("threshold".into(), pv_f32(p.threshold));

    m
}

// ---------------------------------------------------------------------------
// Preset loading (reuses the DSP crates' from_json)
// ---------------------------------------------------------------------------

struct PresetInfo {
    name: String,
    json: String,
}

fn load_preset_jsons(dir: &Path) -> Vec<PresetInfo> {
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
        presets.push(PresetInfo { name, json });
    }
    presets.sort_by(|a, b| a.name.cmp(&b.name));
    presets
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let home = std::env::var("HOME").expect("HOME not set");
    let base = PathBuf::from(&home).join("Library/Audio/Presets/reverb-project");

    // Find preset source directories
    let repo_root = find_repo_root().expect("Could not find repo root");

    let plugins: Vec<(&str, &[u8; 16], PathBuf, fn(&str) -> Option<BTreeMap<String, Value>>)> = vec![
        ("Reverb", &REVERB_CID, repo_root.join("reverb/gui/presets"), |json| {
            reverb_dsp::ReverbParams::from_json(json).ok().map(|p| reverb_params_to_nih(&p))
        }),
        ("Lossy", &LOSSY_CID, repo_root.join("lossy/gui/presets"), |json| {
            lossy_dsp::LossyParams::from_json(json).ok().map(|p| lossy_params_to_nih(&p))
        }),
        ("Fractal", &FRACTAL_CID, repo_root.join("fractal/gui/presets"), |json| {
            fractal_dsp::FractalParams::from_json(json).ok().map(|p| fractal_params_to_nih(&p))
        }),
    ];

    let mut total = 0;

    for (name, cid, preset_dir, converter) in &plugins {
        let out_dir = base.join(name);
        std::fs::create_dir_all(&out_dir).expect("create output dir");

        if !preset_dir.is_dir() {
            eprintln!("Warning: preset dir not found: {}", preset_dir.display());
            continue;
        }

        let presets = load_preset_jsons(preset_dir);
        let mut count = 0;

        for preset in &presets {
            let params = match converter(&preset.json) {
                Some(p) => p,
                None => {
                    eprintln!("  Skip (parse error): {}", preset.name);
                    continue;
                }
            };

            let state_json = make_plugin_state(params);
            let vstpreset = write_vstpreset(cid, &state_json);

            let out_path = out_dir.join(format!("{}.vstpreset", preset.name));
            std::fs::write(&out_path, &vstpreset).expect("write vstpreset");
            count += 1;
        }

        println!("{name}: exported {count} presets → {}", out_dir.display());
        total += count;
    }

    println!("\nTotal: {total} presets exported.");
}

fn find_repo_root() -> Option<PathBuf> {
    // Try relative to current dir
    let candidates = [
        ".",
        "..",
        "../..",
    ];
    for c in &candidates {
        let p = PathBuf::from(c);
        if p.join("reverb/gui/presets").is_dir()
            && p.join("lossy/gui/presets").is_dir()
            && p.join("fractal/gui/presets").is_dir()
        {
            return Some(std::fs::canonicalize(p).ok()?);
        }
    }

    // Try relative to executable
    if let Ok(exe) = std::env::current_exe() {
        for ancestor in exe.ancestors().skip(1) {
            if ancestor.join("reverb/gui/presets").is_dir() {
                return Some(ancestor.to_path_buf());
            }
        }
    }

    None
}
