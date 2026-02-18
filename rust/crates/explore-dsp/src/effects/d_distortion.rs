//! D-series: Distortion effects (D001-D016).

use std::collections::HashMap;
use serde_json::Value;
use crate::{AudioOutput, EffectEntry, pf, pi, ps, params};

// ---------------------------------------------------------------------------
// D001 -- Hard Clipping
// ---------------------------------------------------------------------------

fn process_d001(samples: &[f32], _sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let threshold = pf(params, "threshold", 0.5);
    let pre_gain = pf(params, "pre_gain", 4.0);
    let out: Vec<f32> = samples
        .iter()
        .map(|&s| {
            let x = s * pre_gain;
            x.clamp(-threshold, threshold)
        })
        .collect();
    AudioOutput::Mono(out)
}

fn variants_d001() -> Vec<HashMap<String, Value>> {
    vec![
        params!("threshold" => 0.8, "pre_gain" => 1.5),
        params!("threshold" => 0.5, "pre_gain" => 4.0),
        params!("threshold" => 0.3, "pre_gain" => 8.0),
        params!("threshold" => 0.1, "pre_gain" => 15.0),
        params!("threshold" => 0.05, "pre_gain" => 20.0),
    ]
}

// ---------------------------------------------------------------------------
// D002 -- Soft Clipping (Tanh)
// ---------------------------------------------------------------------------

fn process_d002(samples: &[f32], _sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let drive = pf(params, "drive", 3.0);
    let out: Vec<f32> = samples
        .iter()
        .map(|&s| (s * drive).tanh())
        .collect();
    AudioOutput::Mono(out)
}

fn variants_d002() -> Vec<HashMap<String, Value>> {
    vec![
        params!("drive" => 1.2),
        params!("drive" => 3.0),
        params!("drive" => 7.0),
        params!("drive" => 12.0),
        params!("drive" => 20.0),
    ]
}

// ---------------------------------------------------------------------------
// D003 -- Tube Saturation
// ---------------------------------------------------------------------------

fn process_d003(samples: &[f32], _sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let drive = pf(params, "drive", 3.0);
    let asymmetry = pf(params, "asymmetry", 0.1);
    let d_pos = drive * (1.0 + asymmetry);
    let d_neg = drive * (1.0 - asymmetry);
    let out: Vec<f32> = samples
        .iter()
        .map(|&x| {
            if x >= 0.0 {
                1.0 - (-d_pos * x).exp()
            } else {
                -(1.0 - (d_neg * x).exp())
            }
        })
        .collect();
    AudioOutput::Mono(out)
}

fn variants_d003() -> Vec<HashMap<String, Value>> {
    vec![
        params!("drive" => 1.5, "asymmetry" => 0.0),
        params!("drive" => 3.0, "asymmetry" => 0.1),
        params!("drive" => 6.0, "asymmetry" => 0.25),
        params!("drive" => 10.0, "asymmetry" => 0.0),
        params!("drive" => 10.0, "asymmetry" => 0.5),
        params!("drive" => 2.0, "asymmetry" => 0.4),
    ]
}

// ---------------------------------------------------------------------------
// D004 -- Foldback Distortion
// ---------------------------------------------------------------------------

fn process_d004(samples: &[f32], _sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let threshold = pf(params, "threshold", 0.5);
    let pre_gain = pf(params, "pre_gain", 5.0);
    let out: Vec<f32> = samples
        .iter()
        .map(|&s| {
            let mut x = s * pre_gain;
            for _ in 0..20 {
                if x > threshold {
                    x = threshold - (x - threshold);
                } else if x < -threshold {
                    x = -threshold - (x + threshold);
                } else {
                    break;
                }
            }
            x
        })
        .collect();
    AudioOutput::Mono(out)
}

fn variants_d004() -> Vec<HashMap<String, Value>> {
    vec![
        params!("threshold" => 0.8, "pre_gain" => 2.0),
        params!("threshold" => 0.5, "pre_gain" => 5.0),
        params!("threshold" => 0.3, "pre_gain" => 10.0),
        params!("threshold" => 0.2, "pre_gain" => 20.0),
        params!("threshold" => 0.1, "pre_gain" => 30.0),
        params!("threshold" => 0.6, "pre_gain" => 8.0),
    ]
}

// ---------------------------------------------------------------------------
// D005 -- Chebyshev Polynomial Waveshaper
// ---------------------------------------------------------------------------

fn process_d005(samples: &[f32], _sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let order = pi(params, "order", 5) as usize;
    let default_coeffs: Vec<f32> = vec![0.5, 0.3, 0.1, 0.0, 0.0, 0.0];
    let coeffs: Vec<f32> = params
        .get("coefficients")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                .collect()
        })
        .unwrap_or(default_coeffs);

    let out: Vec<f32> = samples
        .iter()
        .map(|&s| {
            let x = s.clamp(-1.0, 1.0);
            // T_0 = 1, T_1 = x
            let mut t_prev2: f32 = 1.0; // T_0
            let mut t_prev1: f32 = x;   // T_1
            // Always pass through fundamental
            let mut y: f32 = x;
            for k in 2..=order {
                let t_curr = 2.0 * x * t_prev1 - t_prev2;
                let idx = k - 2;
                if idx < coeffs.len() {
                    y += coeffs[idx] * t_curr;
                }
                t_prev2 = t_prev1;
                t_prev1 = t_curr;
            }
            y
        })
        .collect();
    AudioOutput::Mono(out)
}

fn variants_d005() -> Vec<HashMap<String, Value>> {
    vec![
        params!("order" => 2, "coefficients" => vec![0.3]),
        params!("order" => 3, "coefficients" => vec![0.0, 0.5]),
        params!("order" => 5, "coefficients" => vec![0.5, 0.3, 0.2, 0.1]),
        params!("order" => 8, "coefficients" => vec![0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
        params!("order" => 4, "coefficients" => vec![0.8, 0.0, 0.4]),
        params!("order" => 6, "coefficients" => vec![0.0, 0.6, 0.0, 0.4, 0.0]),
    ]
}

// ---------------------------------------------------------------------------
// D006 -- Polynomial Waveshaper
// ---------------------------------------------------------------------------

fn process_d006(samples: &[f32], _sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let a1 = pf(params, "a1", 1.0);
    let a2 = pf(params, "a2", 0.0);
    let a3 = pf(params, "a3", -0.3);
    let a4 = pf(params, "a4", 0.0);
    let a5 = pf(params, "a5", 0.1);
    let out: Vec<f32> = samples
        .iter()
        .map(|&x| {
            let x2 = x * x;
            let x3 = x2 * x;
            let x4 = x2 * x2;
            let x5 = x4 * x;
            a1 * x + a2 * x2 + a3 * x3 + a4 * x4 + a5 * x5
        })
        .collect();
    AudioOutput::Mono(out)
}

fn variants_d006() -> Vec<HashMap<String, Value>> {
    vec![
        params!("a1" => 1.0, "a2" => 0.0, "a3" => -0.3, "a4" => 0.0, "a5" => 0.0),
        params!("a1" => 1.0, "a2" => 0.5, "a3" => 0.0, "a4" => 0.0, "a5" => 0.0),
        params!("a1" => 0.8, "a2" => 0.0, "a3" => -0.5, "a4" => 0.0, "a5" => 0.3),
        params!("a1" => 1.5, "a2" => -1.0, "a3" => 0.5, "a4" => 0.3, "a5" => -0.2),
        params!("a1" => 0.5, "a2" => 1.5, "a3" => -1.0, "a4" => 0.5, "a5" => 0.5),
        params!("a1" => 2.0, "a2" => -2.0, "a3" => 2.0, "a4" => -2.0, "a5" => 2.0),
    ]
}

// ---------------------------------------------------------------------------
// D007 -- Sigmoid Family
// ---------------------------------------------------------------------------

fn process_d007(samples: &[f32], _sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let drive = pf(params, "drive", 5.0);
    let sig_type = ps(params, "type", "atan");
    let out: Vec<f32> = match sig_type {
        "atan" => {
            let scale = 2.0_f32 / std::f32::consts::PI;
            samples
                .iter()
                .map(|&s| scale * (s * drive).atan())
                .collect()
        }
        "algebraic" => samples
            .iter()
            .map(|&s| {
                let x = s * drive;
                x / (1.0 + x * x).sqrt()
            })
            .collect(),
        "erf" => samples
            .iter()
            .map(|&s| {
                let x = s * drive;
                // Abramowitz & Stegun approximation for erf
                let ax = x.abs();
                let t = 1.0 / (1.0 + 0.3275911 * ax);
                let poly = t
                    * (0.254829592
                        + t * (-0.284496736
                            + t * (1.421413741
                                + t * (-1.453152027 + t * 1.061405429))));
                let e = 1.0 - poly * (-ax * ax).exp();
                if x < 0.0 {
                    -e
                } else {
                    e
                }
            })
            .collect(),
        _ => {
            let scale = 2.0_f32 / std::f32::consts::PI;
            samples
                .iter()
                .map(|&s| scale * (s * drive).atan())
                .collect()
        }
    };
    AudioOutput::Mono(out)
}

fn variants_d007() -> Vec<HashMap<String, Value>> {
    vec![
        params!("drive" => 2.0, "type" => "atan"),
        params!("drive" => 10.0, "type" => "atan"),
        params!("drive" => 50.0, "type" => "atan"),
        params!("drive" => 5.0, "type" => "erf"),
        params!("drive" => 20.0, "type" => "erf"),
        params!("drive" => 3.0, "type" => "algebraic"),
        params!("drive" => 30.0, "type" => "algebraic"),
    ]
}

// ---------------------------------------------------------------------------
// D008 -- Bit Crusher
// ---------------------------------------------------------------------------

fn process_d008(samples: &[f32], _sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let bits = pi(params, "bits", 8);
    let levels = (1_i64 << bits) as f32;
    let out: Vec<f32> = samples
        .iter()
        .map(|&s| (s * levels + 0.5).floor() / levels)
        .collect();
    AudioOutput::Mono(out)
}

fn variants_d008() -> Vec<HashMap<String, Value>> {
    vec![
        params!("bits" => 12),
        params!("bits" => 8),
        params!("bits" => 6),
        params!("bits" => 4),
        params!("bits" => 2),
        params!("bits" => 1),
    ]
}

// ---------------------------------------------------------------------------
// D009 -- Sample Rate Reduction
// ---------------------------------------------------------------------------

fn process_d009(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let target_sr = pi(params, "target_sr", 8000) as u32;
    let factor = (sr / target_sr).max(1) as usize;
    let mut out = Vec::with_capacity(samples.len());
    let mut held: f32 = 0.0;
    let mut counter: usize = 0;
    for &s in samples {
        if counter == 0 {
            held = s;
            counter = factor;
        }
        out.push(held);
        counter -= 1;
    }
    AudioOutput::Mono(out)
}

fn variants_d009() -> Vec<HashMap<String, Value>> {
    vec![
        params!("target_sr" => 16000),
        params!("target_sr" => 8000),
        params!("target_sr" => 4000),
        params!("target_sr" => 2000),
        params!("target_sr" => 1000),
        params!("target_sr" => 500),
    ]
}

// ---------------------------------------------------------------------------
// D010 -- Slew Rate Limiter
// ---------------------------------------------------------------------------

fn process_d010(samples: &[f32], _sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let max_slew = pf(params, "max_slew", 0.05);
    if samples.is_empty() {
        return AudioOutput::Mono(vec![]);
    }
    let mut out = Vec::with_capacity(samples.len());
    out.push(samples[0]);
    for i in 1..samples.len() {
        let prev = out[i - 1];
        let diff = samples[i] - prev;
        if diff > max_slew {
            out.push(prev + max_slew);
        } else if diff < -max_slew {
            out.push(prev - max_slew);
        } else {
            out.push(samples[i]);
        }
    }
    AudioOutput::Mono(out)
}

fn variants_d010() -> Vec<HashMap<String, Value>> {
    vec![
        params!("max_slew" => 0.3),
        params!("max_slew" => 0.1),
        params!("max_slew" => 0.05),
        params!("max_slew" => 0.01),
        params!("max_slew" => 0.005),
        params!("max_slew" => 0.001),
    ]
}

// ---------------------------------------------------------------------------
// D011 -- Diode Clipper
// ---------------------------------------------------------------------------

fn process_d011(samples: &[f32], _sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let forward_voltage = pf(params, "forward_voltage", 0.5);
    let num_diodes = pi(params, "num_diodes", 2) as f32;
    let pre_gain = pf(params, "pre_gain", 3.0);
    let vt = forward_voltage * num_diodes;
    let out: Vec<f32> = samples
        .iter()
        .map(|&s| {
            let x = s * pre_gain;
            let ax = x.abs();
            if ax <= vt {
                // Below threshold: cubic nonlinearity
                x - (x * x * x) / (3.0 * vt * vt)
            } else {
                // Above threshold: hard saturate at 2/3 * vt
                let sign = if x >= 0.0 { 1.0_f32 } else { -1.0_f32 };
                sign * (2.0 / 3.0 * vt)
            }
        })
        .collect();
    AudioOutput::Mono(out)
}

fn variants_d011() -> Vec<HashMap<String, Value>> {
    vec![
        params!("forward_voltage" => 0.7, "num_diodes" => 1, "pre_gain" => 2.0),
        params!("forward_voltage" => 0.5, "num_diodes" => 2, "pre_gain" => 3.0),
        params!("forward_voltage" => 0.3, "num_diodes" => 2, "pre_gain" => 5.0),
        params!("forward_voltage" => 0.2, "num_diodes" => 4, "pre_gain" => 4.0),
        params!("forward_voltage" => 0.7, "num_diodes" => 4, "pre_gain" => 8.0),
        params!("forward_voltage" => 0.3, "num_diodes" => 1, "pre_gain" => 10.0),
    ]
}

// ---------------------------------------------------------------------------
// D012 -- Rectification
// ---------------------------------------------------------------------------

fn process_d012(samples: &[f32], _sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let rect_type = ps(params, "type", "full");
    let bias = pf(params, "bias", 0.0);
    let out: Vec<f32> = match rect_type {
        "full" => samples.iter().map(|&s| (s + bias).abs()).collect(),
        "half" => samples
            .iter()
            .map(|&s| {
                let x = s + bias;
                if x > 0.0 {
                    x
                } else {
                    0.0
                }
            })
            .collect(),
        "biased" => samples
            .iter()
            .map(|&s| {
                let x = s + bias;
                if x > 0.0 {
                    x
                } else {
                    x * 0.1
                }
            })
            .collect(),
        _ => samples.iter().map(|&s| (s + bias).abs()).collect(),
    };
    AudioOutput::Mono(out)
}

fn variants_d012() -> Vec<HashMap<String, Value>> {
    vec![
        params!("type" => "full", "bias" => 0.0),
        params!("type" => "half", "bias" => 0.0),
        params!("type" => "full", "bias" => 0.3),
        params!("type" => "half", "bias" => -0.2),
        params!("type" => "biased", "bias" => 0.0),
        params!("type" => "biased", "bias" => 0.4),
        params!("type" => "half", "bias" => 0.5),
    ]
}

// ---------------------------------------------------------------------------
// D013 -- Dynamic Waveshaping
// ---------------------------------------------------------------------------

fn process_d013(samples: &[f32], _sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let base_drive = pf(params, "base_drive", 2.0);
    let env_drive = pf(params, "env_drive", 8.0);
    let env_speed = pf(params, "env_speed", 0.01);
    let mut out = Vec::with_capacity(samples.len());
    let mut env: f32 = 0.0;
    for &s in samples {
        // Envelope follower
        let inp = s.abs();
        if inp > env {
            env += env_speed * (inp - env);
        } else {
            env -= env_speed * 0.25 * (env - inp);
        }
        // Drive depends on envelope
        let drive = base_drive + env_drive * env;
        out.push((s * drive).tanh());
    }
    AudioOutput::Mono(out)
}

fn variants_d013() -> Vec<HashMap<String, Value>> {
    vec![
        params!("base_drive" => 1.0, "env_drive" => 5.0, "env_speed" => 0.01),
        params!("base_drive" => 1.0, "env_drive" => 15.0, "env_speed" => 0.005),
        params!("base_drive" => 3.0, "env_drive" => 10.0, "env_speed" => 0.05),
        params!("base_drive" => 5.0, "env_drive" => 0.0, "env_speed" => 0.01),
        params!("base_drive" => 1.0, "env_drive" => 20.0, "env_speed" => 0.1),
        params!("base_drive" => 2.0, "env_drive" => 10.0, "env_speed" => 0.001),
    ]
}

// ---------------------------------------------------------------------------
// D014 -- Bitwise Distortion (XOR / AND / OR)
// ---------------------------------------------------------------------------

fn process_d014(samples: &[f32], _sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let bit_depth = pi(params, "bit_depth", 12);
    let operation = ps(params, "operation", "xor");
    let pattern = pi(params, "pattern", 0xAA) as i64;

    let max_val = ((1_i64) << bit_depth) - 1;
    let pat = pattern & max_val;

    let out: Vec<f32> = samples
        .iter()
        .map(|&s| {
            let x = s.clamp(-1.0, 1.0);
            // Quantize float to integer
            let quantized = ((x as f64 * 0.5 + 0.5) * max_val as f64) as i64 & max_val;
            // Apply bitwise operation
            let result = match operation {
                "xor" => quantized ^ pat,
                "and" => quantized & pat,
                "or" => quantized | pat,
                _ => quantized ^ pat,
            };
            // Convert back to float [-1, 1]
            (result as f64 / max_val as f64 * 2.0 - 1.0) as f32
        })
        .collect();
    AudioOutput::Mono(out)
}

fn variants_d014() -> Vec<HashMap<String, Value>> {
    vec![
        params!("bit_depth" => 16, "operation" => "xor", "pattern" => 0x00FF_i64),
        params!("bit_depth" => 12, "operation" => "xor", "pattern" => 0x0AAA_i64),
        params!("bit_depth" => 8, "operation" => "xor", "pattern" => 0xFF_i64),
        params!("bit_depth" => 8, "operation" => "and", "pattern" => 0xF0_i64),
        params!("bit_depth" => 10, "operation" => "or", "pattern" => 0x155_i64),
        params!("bit_depth" => 8, "operation" => "xor", "pattern" => 0x55_i64),
        params!("bit_depth" => 12, "operation" => "and", "pattern" => 0xFC0_i64),
    ]
}

// ---------------------------------------------------------------------------
// D015 -- Modular Arithmetic Distortion
// ---------------------------------------------------------------------------

fn process_d015(samples: &[f32], _sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let scale = pf(params, "scale", 3.0);
    let modulus = pf(params, "modulus", 1.0);
    let offset = pf(params, "offset", 0.0);
    let half_mod = modulus * 0.5;
    let out: Vec<f32> = samples
        .iter()
        .map(|&s| {
            let x = s * scale + offset;
            // Modular wrap: bring into [0, modulus) then center
            let y = x - modulus * (x / modulus).floor();
            y - half_mod
        })
        .collect();
    AudioOutput::Mono(out)
}

fn variants_d015() -> Vec<HashMap<String, Value>> {
    vec![
        params!("scale" => 2.0, "modulus" => 1.5, "offset" => 0.0),
        params!("scale" => 4.0, "modulus" => 1.0, "offset" => 0.0),
        params!("scale" => 8.0, "modulus" => 0.5, "offset" => 0.0),
        params!("scale" => 3.0, "modulus" => 0.8, "offset" => 0.5),
        params!("scale" => 10.0, "modulus" => 0.3, "offset" => 0.0),
        params!("scale" => 1.5, "modulus" => 2.0, "offset" => 0.0),
        params!("scale" => 6.0, "modulus" => 0.2, "offset" => 0.3),
    ]
}

// ---------------------------------------------------------------------------
// D016 -- Serge-Style Wavefolder
// ---------------------------------------------------------------------------

fn process_d016(samples: &[f32], _sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let pre_gain = pf(params, "pre_gain", 3.0);
    let fold_type_str = ps(params, "fold_type", "sine");
    let stages = pi(params, "stages", 1).max(1) as usize;
    let asymmetry = pf(params, "asymmetry", 0.0);

    let fold_type: i32 = match fold_type_str {
        "sine" => 0,
        "triangle" => 1,
        "tanh" => 2,
        _ => 0,
    };

    let pi_val = std::f32::consts::PI;

    let out: Vec<f32> = samples
        .iter()
        .map(|&s| {
            let mut x = s * pre_gain + asymmetry;
            for _ in 0..stages {
                x = match fold_type {
                    0 => {
                        // Sine fold
                        (pi_val * x).sin()
                    }
                    1 => {
                        // Triangle fold
                        2.0 * (x * 0.5 - (x * 0.5 + 0.5).floor()).abs() * 2.0 - 1.0
                    }
                    _ => {
                        // Tanh-fold
                        (x * pi_val * 0.5).sin().tanh()
                    }
                };
            }
            x
        })
        .collect();
    AudioOutput::Mono(out)
}

fn variants_d016() -> Vec<HashMap<String, Value>> {
    vec![
        params!("pre_gain" => 2.0, "fold_type" => "sine", "stages" => 1, "asymmetry" => 0.0),
        params!("pre_gain" => 4.0, "fold_type" => "sine", "stages" => 2, "asymmetry" => 0.0),
        params!("pre_gain" => 8.0, "fold_type" => "sine", "stages" => 3, "asymmetry" => 0.2),
        params!("pre_gain" => 3.0, "fold_type" => "triangle", "stages" => 1, "asymmetry" => 0.0),
        params!("pre_gain" => 6.0, "fold_type" => "triangle", "stages" => 2, "asymmetry" => 0.1),
        params!("pre_gain" => 10.0, "fold_type" => "triangle", "stages" => 3, "asymmetry" => 0.3),
        params!("pre_gain" => 3.0, "fold_type" => "tanh", "stages" => 1, "asymmetry" => 0.0),
        params!("pre_gain" => 7.0, "fold_type" => "tanh", "stages" => 2, "asymmetry" => 0.15),
        params!("pre_gain" => 5.0, "fold_type" => "sine", "stages" => 1, "asymmetry" => 0.4),
    ]
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

pub fn register() -> Vec<EffectEntry> {
    vec![
        EffectEntry {
            id: "D001",
            process: process_d001,
            variants: variants_d001,
            category: "distortion",
        },
        EffectEntry {
            id: "D002",
            process: process_d002,
            variants: variants_d002,
            category: "distortion",
        },
        EffectEntry {
            id: "D003",
            process: process_d003,
            variants: variants_d003,
            category: "distortion",
        },
        EffectEntry {
            id: "D004",
            process: process_d004,
            variants: variants_d004,
            category: "distortion",
        },
        EffectEntry {
            id: "D005",
            process: process_d005,
            variants: variants_d005,
            category: "distortion",
        },
        EffectEntry {
            id: "D006",
            process: process_d006,
            variants: variants_d006,
            category: "distortion",
        },
        EffectEntry {
            id: "D007",
            process: process_d007,
            variants: variants_d007,
            category: "distortion",
        },
        EffectEntry {
            id: "D008",
            process: process_d008,
            variants: variants_d008,
            category: "distortion",
        },
        EffectEntry {
            id: "D009",
            process: process_d009,
            variants: variants_d009,
            category: "distortion",
        },
        EffectEntry {
            id: "D010",
            process: process_d010,
            variants: variants_d010,
            category: "distortion",
        },
        EffectEntry {
            id: "D011",
            process: process_d011,
            variants: variants_d011,
            category: "distortion",
        },
        EffectEntry {
            id: "D012",
            process: process_d012,
            variants: variants_d012,
            category: "distortion",
        },
        EffectEntry {
            id: "D013",
            process: process_d013,
            variants: variants_d013,
            category: "distortion",
        },
        EffectEntry {
            id: "D014",
            process: process_d014,
            variants: variants_d014,
            category: "distortion",
        },
        EffectEntry {
            id: "D015",
            process: process_d015,
            variants: variants_d015,
            category: "distortion",
        },
        EffectEntry {
            id: "D016",
            process: process_d016,
            variants: variants_d016,
            category: "distortion",
        },
    ]
}
