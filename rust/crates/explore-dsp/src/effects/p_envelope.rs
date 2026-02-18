//! P-series: Envelope effects (P001-P005).
//!
//! P001 -- Envelope Reshaping
//! P002 -- Envelope Inversion
//! P003 -- Noise Gate with Decay
//! P004 -- Rhythmic Gain Sequencer
//! P005 -- Live Buffer Freeze / Stutter Hold

use std::collections::HashMap;
use serde_json::Value;
use crate::{AudioOutput, EffectEntry, pf, pi, pu, params};
use crate::primitives::*;

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Extract amplitude envelope using a one-pole follower.
/// Returns an envelope array the same length as samples.
fn extract_envelope(samples: &[f32], smoothing_samples: usize) -> Vec<f32> {
    let n = samples.len();
    let mut env = vec![0.0f32; n];
    let sm = if smoothing_samples < 1 { 1 } else { smoothing_samples };
    let coeff: f32 = 1.0 - 1.0 / sm as f32;
    let mut prev: f32 = 0.0;
    for i in 0..n {
        let rect = samples[i].abs();
        if rect > prev {
            prev = 0.6 * prev + 0.4 * rect; // fast attack
        } else {
            prev = coeff * prev + (1.0 - coeff) * rect;
        }
        env[i] = prev;
    }
    env
}

// ---------------------------------------------------------------------------
// P001 -- Envelope Reshaping
// ---------------------------------------------------------------------------

/// Generate a target envelope shape of length n.
///
/// shape_id mapping:
///   0 = ramp_up
///   1 = ramp_down
///   2 = triangle
///   3 = pulse
///   4 = adsr
fn generate_shape(n: usize, shape_id: i32) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    match shape_id {
        0 => {
            // ramp_up
            let denom = (n as i64 - 1).max(1) as f32;
            for i in 0..n {
                out[i] = i as f32 / denom;
            }
        }
        1 => {
            // ramp_down
            let denom = (n as i64 - 1).max(1) as f32;
            for i in 0..n {
                out[i] = 1.0 - i as f32 / denom;
            }
        }
        2 => {
            // triangle
            let mid = n / 2;
            for i in 0..n {
                if i <= mid {
                    out[i] = i as f32 / mid.max(1) as f32;
                } else {
                    let denom = (n as i64 - mid as i64 - 1).max(1) as f32;
                    out[i] = 1.0 - (i - mid) as f32 / denom;
                }
            }
        }
        3 => {
            // pulse: 50% duty cycle
            let half = n / 2;
            for i in 0..n {
                out[i] = if i < half { 1.0 } else { 0.0 };
            }
        }
        4 => {
            // adsr: attack 10%, decay 15%, sustain level 0.6 for 50%, release 25%
            let a_end = n / 10;
            let d_end = a_end + (n * 15) / 100;
            let r_start = n - n / 4;
            let sustain_level: f32 = 0.6;
            for i in 0..n {
                if i < a_end {
                    out[i] = i as f32 / a_end.max(1) as f32;
                } else if i < d_end {
                    let frac = (i - a_end) as f32 / (d_end - a_end).max(1) as f32;
                    out[i] = 1.0 - frac * (1.0 - sustain_level);
                } else if i < r_start {
                    out[i] = sustain_level;
                } else {
                    let frac = (i - r_start) as f32 / (n as i64 - r_start as i64 - 1).max(1) as f32;
                    out[i] = sustain_level * (1.0 - frac);
                }
            }
        }
        _ => {
            // default to triangle (shape_id 2)
            let mid = n / 2;
            for i in 0..n {
                if i <= mid {
                    out[i] = i as f32 / mid.max(1) as f32;
                } else {
                    let denom = (n as i64 - mid as i64 - 1).max(1) as f32;
                    out[i] = 1.0 - (i - mid) as f32 / denom;
                }
            }
        }
    }
    out
}

/// Normalize original envelope then replace with target shape.
fn apply_envelope_reshape(samples: &[f32], env: &[f32], target_shape: &[f32]) -> Vec<f32> {
    let n = samples.len();
    let mut out = vec![0.0f32; n];

    // Find peak of original envelope for normalization
    let mut peak: f32 = 0.0;
    for i in 0..n {
        if env[i] > peak {
            peak = env[i];
        }
    }
    if peak < 1e-10 {
        peak = 1e-10;
    }

    for i in 0..n {
        // Divide out old envelope, multiply by new shape
        let mut norm_env = env[i] / peak;
        if norm_env < 1e-10 {
            norm_env = 1e-10;
        }
        let mut gain = target_shape[i] / norm_env;
        // Clamp gain to avoid extreme amplification
        if gain > 50.0 {
            gain = 50.0;
        }
        out[i] = samples[i] * gain;
    }
    out
}

fn shape_name_to_id(name: &str) -> i32 {
    match name {
        "ramp_up" => 0,
        "ramp_down" => 1,
        "triangle" => 2,
        "pulse" => 3,
        "adsr" => 4,
        _ => 2, // default to triangle
    }
}

fn process_p001(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let new_shape = params
        .get("new_shape")
        .and_then(|v| v.as_str())
        .unwrap_or("triangle");
    let shape_id = shape_name_to_id(new_shape);

    let n = samples.len();
    let smoothing_samples = (0.01 * sr as f32) as usize;
    let smoothing_samples = smoothing_samples.max(1);
    let env = extract_envelope(samples, smoothing_samples);
    let target = generate_shape(n, shape_id);

    AudioOutput::Mono(apply_envelope_reshape(samples, &env, &target))
}

fn variants_p001() -> Vec<HashMap<String, Value>> {
    vec![
        params!("new_shape" => "ramp_up"),
        params!("new_shape" => "ramp_down"),
        params!("new_shape" => "triangle"),
        params!("new_shape" => "pulse"),
        params!("new_shape" => "adsr"),
    ]
}

// ---------------------------------------------------------------------------
// P002 -- Envelope Inversion
// ---------------------------------------------------------------------------

/// Loud becomes quiet and quiet becomes loud.
/// gain = (1 - normalized_envelope) * max_gain, clamped.
fn invert_envelope_kernel(samples: &[f32], env: &[f32], max_gain: f32) -> Vec<f32> {
    let n = samples.len();
    let mut out = vec![0.0f32; n];

    // Find peak of envelope
    let mut peak: f32 = 0.0;
    for i in 0..n {
        if env[i] > peak {
            peak = env[i];
        }
    }
    if peak < 1e-10 {
        peak = 1e-10;
    }

    for i in 0..n {
        let norm = env[i] / peak;
        // Invert: when norm is high, gain is low and vice versa
        let mut gain = (1.0 - norm) * max_gain;
        if gain > max_gain {
            gain = max_gain;
        }
        if gain < 0.0 {
            gain = 0.0;
        }
        out[i] = samples[i] * gain;
    }
    out
}

fn process_p002(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let smoothing_ms = pf(params, "smoothing_ms", 20.0);
    let max_gain = pf(params, "max_gain", 30.0);

    let smoothing_samples = (smoothing_ms * sr as f32 / 1000.0) as usize;
    let smoothing_samples = smoothing_samples.max(1);
    let env = extract_envelope(samples, smoothing_samples);

    AudioOutput::Mono(invert_envelope_kernel(samples, &env, max_gain))
}

fn variants_p002() -> Vec<HashMap<String, Value>> {
    vec![
        // Fast follower, moderate gain
        params!("smoothing_ms" => 5.0, "max_gain" => 20.0),
        // Default
        params!("smoothing_ms" => 20.0, "max_gain" => 30.0),
        // Slow follower, high gain -- dreamy swells
        params!("smoothing_ms" => 50.0, "max_gain" => 60.0),
        // Very fast, extreme gain -- noisy artifacts
        params!("smoothing_ms" => 5.0, "max_gain" => 100.0),
        // Slow, gentle inversion
        params!("smoothing_ms" => 40.0, "max_gain" => 10.0),
        // Medium speed, moderate boost
        params!("smoothing_ms" => 15.0, "max_gain" => 40.0),
    ]
}

// ---------------------------------------------------------------------------
// P003 -- Noise Gate with Decay
// ---------------------------------------------------------------------------

/// Gate with hysteresis and exponential decay below threshold.
///
/// When envelope drops below threshold_lin, apply exponential decay.
/// Gate reopens when envelope exceeds threshold_open_lin (threshold + hysteresis).
fn noise_gate_kernel(
    samples: &[f32],
    env: &[f32],
    threshold_lin: f32,
    threshold_open_lin: f32,
    decay_coeff: f32,
) -> Vec<f32> {
    let n = samples.len();
    let mut out = vec![0.0f32; n];
    let mut gate_open = true;
    let mut decay_gain: f32 = 1.0;

    for i in 0..n {
        let level = env[i];
        if gate_open {
            if level < threshold_lin {
                gate_open = false;
                decay_gain = 1.0;
            }
        } else if level > threshold_open_lin {
            gate_open = true;
            decay_gain = 1.0;
        }

        if gate_open {
            out[i] = samples[i];
        } else {
            out[i] = samples[i] * decay_gain;
            decay_gain *= decay_coeff;
            if decay_gain < 1e-8 {
                decay_gain = 1e-8;
            }
        }
    }
    out
}

fn process_p003(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let threshold_db = pf(params, "threshold_db", -30.0);
    let decay_ms = pf(params, "decay_ms", 200.0);
    let hysteresis_db = pf(params, "hysteresis_db", 3.0);

    // Envelope extraction (5 ms smoothing for fast response)
    let smoothing_samples = (0.005 * sr as f32) as usize;
    let smoothing_samples = smoothing_samples.max(1);
    let env = extract_envelope(samples, smoothing_samples);

    let threshold_lin = 10.0f32.powf(threshold_db / 20.0);
    let threshold_open_lin = 10.0f32.powf((threshold_db + hysteresis_db) / 20.0);

    // Decay coefficient: per-sample multiplier so that amplitude halves in decay_ms
    // exp(-ln(2) / (decay_ms * 0.001 * sr))
    let decay_coeff = if decay_ms > 0.0 {
        (-(2.0f32.ln()) / (decay_ms * 0.001 * sr as f32)).exp()
    } else {
        0.0
    };

    AudioOutput::Mono(noise_gate_kernel(
        samples,
        &env,
        threshold_lin,
        threshold_open_lin,
        decay_coeff,
    ))
}

fn variants_p003() -> Vec<HashMap<String, Value>> {
    vec![
        // Gentle gate, slow decay
        params!("threshold_db" => -40.0, "decay_ms" => 400.0, "hysteresis_db" => 4.0),
        // Standard gate
        params!("threshold_db" => -30.0, "decay_ms" => 200.0, "hysteresis_db" => 3.0),
        // Aggressive gate, fast cutoff
        params!("threshold_db" => -20.0, "decay_ms" => 50.0, "hysteresis_db" => 2.0),
        // Sensitive gate, very slow decay (tail-like)
        params!("threshold_db" => -45.0, "decay_ms" => 500.0, "hysteresis_db" => 5.0),
        // Tight percussive gate
        params!("threshold_db" => -25.0, "decay_ms" => 80.0, "hysteresis_db" => 6.0),
        // Wide hysteresis, medium decay
        params!("threshold_db" => -35.0, "decay_ms" => 300.0, "hysteresis_db" => 6.0),
    ]
}

// ---------------------------------------------------------------------------
// P004 -- Rhythmic Gain Sequencer
// ---------------------------------------------------------------------------

/// Apply a 16-step gain pattern at the given tempo.
///
/// Each step lasts step_samples.  Pattern repeats.  Short crossfade
/// (64 samples) between steps to avoid clicks.
fn rhythmic_gain_kernel(
    samples: &[f32],
    step_samples: usize,
    pattern: &[f32; 16],
    num_steps: usize,
) -> Vec<f32> {
    let n = samples.len();
    let mut out = vec![0.0f32; n];
    let mut fade_len: usize = 64;
    if fade_len > step_samples / 2 {
        fade_len = (step_samples / 2).max(1);
    }

    for i in 0..n {
        let step_idx = (i / step_samples) % num_steps;
        let pos_in_step = i % step_samples;

        let current_gain = pattern[step_idx];
        let next_step_idx = (step_idx + 1) % num_steps;
        let next_gain = pattern[next_step_idx];

        // Crossfade at the end of each step
        let gain = if pos_in_step >= step_samples - fade_len {
            let fade_pos = pos_in_step - (step_samples - fade_len);
            let frac = fade_pos as f32 / fade_len as f32;
            current_gain * (1.0 - frac) + next_gain * frac
        } else {
            current_gain
        };

        out[i] = samples[i] * gain;
    }
    out
}

fn process_p004(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let step_ms = pf(params, "step_ms", 125.0);

    // Parse pattern from params
    let mut pattern = [0.0f32; 16];
    let pattern_input = params.get("pattern").and_then(|v| v.as_array());

    match pattern_input {
        Some(arr) => {
            // Fill from array, pad with 1.0 if shorter than 16, truncate if longer
            for i in 0..16 {
                if i < arr.len() {
                    pattern[i] = arr[i].as_f64().unwrap_or(1.0) as f32;
                } else {
                    pattern[i] = 1.0;
                }
            }
        }
        None => {
            // Default: alternating 1.0 and 0.0
            for i in 0..16 {
                pattern[i] = if i % 2 == 0 { 1.0 } else { 0.0 };
            }
        }
    }

    let step_samples = (step_ms * sr as f32 / 1000.0) as usize;
    let step_samples = step_samples.max(1);

    AudioOutput::Mono(rhythmic_gain_kernel(samples, step_samples, &pattern, 16))
}

fn variants_p004() -> Vec<HashMap<String, Value>> {
    vec![
        // Classic trance gate (alternating on/off)
        params!(
            "step_ms" => 125.0,
            "pattern" => vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
                              1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        ),
        // Funky syncopation
        params!(
            "step_ms" => 100.0,
            "pattern" => vec![1.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.8,
                              1.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0, 0.5]
        ),
        // Slow swell and duck
        params!(
            "step_ms" => 200.0,
            "pattern" => vec![0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.0,
                              1.0, 0.9, 0.7, 0.5, 0.3, 0.1, 0.0, 0.0]
        ),
        // Staccato bursts
        params!(
            "step_ms" => 60.0,
            "pattern" => vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                              1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        ),
        // Half-time pulse
        params!(
            "step_ms" => 250.0,
            "pattern" => vec![1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
                              1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
        ),
        // Crescendo pattern
        params!(
            "step_ms" => 125.0,
            "pattern" => vec![0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                              0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.0, 1.0]
        ),
        // Random-feel accents
        params!(
            "step_ms" => 150.0,
            "pattern" => vec![1.0, 0.2, 0.0, 0.8, 0.0, 0.5, 1.0, 0.0,
                              0.3, 1.0, 0.0, 0.0, 0.7, 0.0, 1.0, 0.4]
        ),
    ]
}

// ---------------------------------------------------------------------------
// P005 -- Live Buffer Freeze / Stutter Hold
// ---------------------------------------------------------------------------

/// Capture a loop at freeze_pos, crossfade in, hold, crossfade out.
fn buffer_freeze_kernel(
    samples: &[f32],
    sr: u32,
    freeze_pos: f32,
    loop_ms: f32,
    fade_ms: f32,
    hold_ms: f32,
) -> Vec<f32> {
    let n = samples.len();
    let mut out = vec![0.0f32; n];

    let loop_samps = (loop_ms * 0.001 * sr as f32) as usize;
    let loop_samps = loop_samps.max(64);
    let fade_samps = (fade_ms * 0.001 * sr as f32) as usize;
    let fade_samps = fade_samps.max(1).min(loop_samps / 2);
    let hold_samps = (hold_ms * 0.001 * sr as f32) as usize;
    let hold_samps = hold_samps.max(1);

    let mut freeze_start = (freeze_pos * n as f32) as usize;
    if freeze_start + loop_samps > n {
        freeze_start = if n > loop_samps { n - loop_samps } else { 0 };
    }

    // Extract loop buffer with crossfade at boundaries for seamless looping
    let mut loop_buf = vec![0.0f32; loop_samps];
    for j in 0..loop_samps {
        let idx = freeze_start + j;
        if idx < n {
            loop_buf[j] = samples[idx];
        }
    }

    // Apply crossfade at loop boundary for seamless repeat
    for j in 0..fade_samps {
        let frac = j as f32 / fade_samps as f32;
        // Fade end of loop
        loop_buf[loop_samps - 1 - j] *= frac;
        // This creates a smooth loop point
    }

    // Build output: original -> crossfade into freeze -> hold -> crossfade out
    let fade_in_start = freeze_start;
    let hold_end = fade_in_start + fade_samps + hold_samps;
    let fade_out_end = hold_end + fade_samps;

    for i in 0..n {
        if i < fade_in_start {
            // Before freeze: pass through original
            out[i] = samples[i];
        } else if i < fade_in_start + fade_samps {
            // Crossfade from original to frozen loop
            let frac = (i - fade_in_start) as f32 / fade_samps as f32;
            let loop_idx = (i - fade_in_start) % loop_samps;
            out[i] = (1.0 - frac) * samples[i] + frac * loop_buf[loop_idx];
        } else if i < hold_end {
            // Frozen loop region
            let loop_idx = (i - fade_in_start) % loop_samps;
            out[i] = loop_buf[loop_idx];
        } else if i < fade_out_end {
            // Crossfade back to original
            let frac = (i - hold_end) as f32 / fade_samps as f32;
            let loop_idx = (i - fade_in_start) % loop_samps;
            let orig = if i < n { samples[i] } else { 0.0 };
            out[i] = (1.0 - frac) * loop_buf[loop_idx] + frac * orig;
        } else {
            // After freeze: pass through original
            if i < n {
                out[i] = samples[i];
            }
        }
    }

    out
}

fn process_p005(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let freeze_pos = pf(params, "freeze_pos", 0.3);
    let loop_ms = pf(params, "loop_ms", 100.0);
    let fade_ms = pf(params, "fade_ms", 20.0);
    let hold_ms = pf(params, "hold_ms", 500.0);

    AudioOutput::Mono(buffer_freeze_kernel(
        samples, sr, freeze_pos, loop_ms, fade_ms, hold_ms,
    ))
}

fn variants_p005() -> Vec<HashMap<String, Value>> {
    vec![
        params!("freeze_pos" => 0.2, "loop_ms" => 50.0, "fade_ms" => 10.0, "hold_ms" => 300.0),
        params!("freeze_pos" => 0.3, "loop_ms" => 100.0, "fade_ms" => 20.0, "hold_ms" => 500.0),
        params!("freeze_pos" => 0.5, "loop_ms" => 200.0, "fade_ms" => 30.0, "hold_ms" => 800.0),
        params!("freeze_pos" => 0.4, "loop_ms" => 30.0, "fade_ms" => 5.0, "hold_ms" => 1000.0),
        params!("freeze_pos" => 0.7, "loop_ms" => 150.0, "fade_ms" => 50.0, "hold_ms" => 400.0),
        params!("freeze_pos" => 0.1, "loop_ms" => 80.0, "fade_ms" => 15.0, "hold_ms" => 2000.0),
    ]
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

pub fn register() -> Vec<EffectEntry> {
    vec![
        EffectEntry {
            id: "P001",
            process: process_p001,
            variants: variants_p001,
            category: "envelope",
        },
        EffectEntry {
            id: "P002",
            process: process_p002,
            variants: variants_p002,
            category: "envelope",
        },
        EffectEntry {
            id: "P003",
            process: process_p003,
            variants: variants_p003,
            category: "envelope",
        },
        EffectEntry {
            id: "P004",
            process: process_p004,
            variants: variants_p004,
            category: "envelope",
        },
        EffectEntry {
            id: "P005",
            process: process_p005,
            variants: variants_p005,
            category: "envelope",
        },
    ]
}
