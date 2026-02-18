//! F-series: Dynamics effects (F001-F015).
//!
//! F001 -- Compressor
//! F002 -- Expander/Gate
//! F003 -- Transient Shaper
//! F004 -- Multiband Dynamics (3-band)
//! F005 -- Sidechain Ducker
//! F006 -- RMS Compressor
//! F007 -- Feedback Compressor
//! F008 -- Opto Compressor
//! F009 -- FET Compressor
//! F010 -- Soft-Knee Compressor
//! F011 -- Parallel (NY) Compressor
//! F012 -- Upward Compressor
//! F013 -- Program-Dependent Compressor
//! F014 -- Lookahead Limiter
//! F015 -- Spectral Compressor

use std::collections::HashMap;
use serde_json::Value;
use crate::{AudioOutput, EffectEntry, pf, pi, ps, params};
use crate::primitives::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Peak envelope follower with separate attack/release coefficients.
fn envelope_follow(samples: &[f32], attack_coeff: f32, release_coeff: f32) -> Vec<f32> {
    let n = samples.len();
    let mut env = vec![0.0f32; n];
    let mut prev = 0.0f32;
    for i in 0..n {
        let inp = samples[i].abs();
        if inp > prev {
            prev = attack_coeff * prev + (1.0 - attack_coeff) * inp;
        } else {
            prev = release_coeff * prev + (1.0 - release_coeff) * inp;
        }
        env[i] = prev;
    }
    env
}

/// Compress kernel: apply gain reduction based on envelope and threshold.
fn compress_kernel(
    samples: &[f32],
    env: &[f32],
    threshold_lin: f32,
    ratio: f32,
    makeup_gain: f32,
) -> Vec<f32> {
    let n = samples.len();
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        let level = env[i];
        let gain = if level > threshold_lin && level > 1e-12 {
            let over_db = 20.0 * (level / threshold_lin).log10();
            let reduction_db = over_db * (1.0 - 1.0 / ratio);
            10.0f32.powf(-reduction_db / 20.0)
        } else {
            1.0
        };
        out[i] = samples[i] * gain * makeup_gain;
    }
    out
}

// ===================================================================
// F001 -- Compressor
// ===================================================================

fn process_f001(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let threshold_db = pf(params, "threshold_db", -20.0);
    let ratio = pf(params, "ratio", 4.0);
    let attack_ms = pf(params, "attack_ms", 10.0);
    let release_ms = pf(params, "release_ms", 100.0);

    let attack_coeff = ms_to_coeff(attack_ms, sr);
    let release_coeff = ms_to_coeff(release_ms, sr);
    let env = envelope_follow(samples, attack_coeff, release_coeff);
    let threshold_lin = 10.0f32.powf(threshold_db / 20.0);
    // Makeup gain: compensate for average gain reduction
    let makeup_db = (threshold_db.abs() * (1.0 - 1.0 / ratio)) * 0.5;
    let makeup_gain = 10.0f32.powf(makeup_db / 20.0);
    AudioOutput::Mono(compress_kernel(samples, &env, threshold_lin, ratio, makeup_gain))
}

fn variants_f001() -> Vec<HashMap<String, Value>> {
    vec![
        // Gentle bus glue
        params!("threshold_db" => -10.0, "ratio" => 2.0, "attack_ms" => 30.0, "release_ms" => 200.0),
        // Moderate vocal compression
        params!("threshold_db" => -20.0, "ratio" => 4.0, "attack_ms" => 10.0, "release_ms" => 100.0),
        // Heavy squash
        params!("threshold_db" => -30.0, "ratio" => 10.0, "attack_ms" => 5.0, "release_ms" => 150.0),
        // Brick-wall limiting
        params!("threshold_db" => -15.0, "ratio" => 20.0, "attack_ms" => 0.1, "release_ms" => 50.0),
        // Slow breathing compressor
        params!("threshold_db" => -25.0, "ratio" => 6.0, "attack_ms" => 80.0, "release_ms" => 800.0),
        // Fast transient tamer
        params!("threshold_db" => -18.0, "ratio" => 8.0, "attack_ms" => 0.5, "release_ms" => 60.0),
    ]
}

// ===================================================================
// F002 -- Expander / Gate
// ===================================================================

fn expand_kernel(
    samples: &[f32],
    env: &[f32],
    threshold_lin: f32,
    ratio: f32,
) -> Vec<f32> {
    let n = samples.len();
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        let level = env[i];
        let gain = if level < threshold_lin && level > 1e-12 {
            let under_db = 20.0 * (threshold_lin / level).log10();
            let attenuation_db = under_db * (ratio - 1.0);
            10.0f32.powf(-attenuation_db / 20.0)
        } else if level <= 1e-12 {
            0.0
        } else {
            1.0
        };
        out[i] = samples[i] * gain;
    }
    out
}

fn process_f002(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let threshold_db = pf(params, "threshold_db", -30.0);
    let ratio = pf(params, "ratio", 4.0);
    let attack_ms = pf(params, "attack_ms", 5.0);
    let release_ms = pf(params, "release_ms", 50.0);

    let attack_coeff = ms_to_coeff(attack_ms, sr);
    let release_coeff = ms_to_coeff(release_ms, sr);
    let env = envelope_follow(samples, attack_coeff, release_coeff);
    let threshold_lin = 10.0f32.powf(threshold_db / 20.0);
    AudioOutput::Mono(expand_kernel(samples, &env, threshold_lin, ratio))
}

fn variants_f002() -> Vec<HashMap<String, Value>> {
    vec![
        // Gentle noise reduction
        params!("threshold_db" => -45.0, "ratio" => 3.0, "attack_ms" => 5.0, "release_ms" => 100.0),
        // Medium gate
        params!("threshold_db" => -30.0, "ratio" => 8.0, "attack_ms" => 2.0, "release_ms" => 80.0),
        // Hard gate (tight)
        params!("threshold_db" => -25.0, "ratio" => 20.0, "attack_ms" => 0.5, "release_ms" => 30.0),
        // Slow expander
        params!("threshold_db" => -35.0, "ratio" => 4.0, "attack_ms" => 20.0, "release_ms" => 300.0),
        // Aggressive chop
        params!("threshold_db" => -20.0, "ratio" => 15.0, "attack_ms" => 1.0, "release_ms" => 20.0),
    ]
}

// ===================================================================
// F003 -- Transient Shaper
// ===================================================================

fn transient_shaper_kernel(
    samples: &[f32],
    attack_gain: f32,
    sustain_gain: f32,
    fast_coeff_a: f32,
    fast_coeff_r: f32,
    slow_coeff_a: f32,
    slow_coeff_r: f32,
) -> Vec<f32> {
    let n = samples.len();
    let mut out = vec![0.0f32; n];
    let mut fast_env = 0.0f32;
    let mut slow_env = 0.0f32;
    for i in 0..n {
        let x = samples[i];
        let inp = x.abs();

        // Fast envelope (tracks transients)
        if inp > fast_env {
            fast_env = fast_coeff_a * fast_env + (1.0 - fast_coeff_a) * inp;
        } else {
            fast_env = fast_coeff_r * fast_env + (1.0 - fast_coeff_r) * inp;
        }

        // Slow envelope (tracks sustain)
        if inp > slow_env {
            slow_env = slow_coeff_a * slow_env + (1.0 - slow_coeff_a) * inp;
        } else {
            slow_env = slow_coeff_r * slow_env + (1.0 - slow_coeff_r) * inp;
        }

        // Transient detection: fast - slow
        let diff = fast_env - slow_env;
        let mut gain = if diff > 0.0 {
            // Transient region: apply attack gain
            1.0 + attack_gain * (diff / (fast_env + 1e-12))
        } else {
            // Sustain region: apply sustain gain
            1.0 + sustain_gain * (-diff / (slow_env + 1e-12))
        };

        // Clamp gain to avoid negative amplitudes
        if gain < 0.0 {
            gain = 0.0;
        }

        out[i] = x * gain;
    }
    out
}

fn process_f003(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let attack_gain = pf(params, "attack_gain", 0.5);
    let sustain_gain = pf(params, "sustain_gain", 0.0);

    // Fast envelope: ~0.3ms attack, ~5ms release
    let fast_a = ms_to_coeff(0.3, sr);
    let fast_r = ms_to_coeff(5.0, sr);
    // Slow envelope: ~20ms attack, ~200ms release
    let slow_a = ms_to_coeff(20.0, sr);
    let slow_r = ms_to_coeff(200.0, sr);

    AudioOutput::Mono(transient_shaper_kernel(
        samples,
        attack_gain,
        sustain_gain,
        fast_a,
        fast_r,
        slow_a,
        slow_r,
    ))
}

fn variants_f003() -> Vec<HashMap<String, Value>> {
    vec![
        // Punch up transients
        params!("attack_gain" => 1.0, "sustain_gain" => 0.0),
        // Soften transients (smoother)
        params!("attack_gain" => -0.8, "sustain_gain" => 0.0),
        // Enhance sustain only
        params!("attack_gain" => 0.0, "sustain_gain" => 1.5),
        // Reduce sustain (tighter)
        params!("attack_gain" => 0.0, "sustain_gain" => -0.8),
        // Max snap: big transients, less sustain
        params!("attack_gain" => 2.0, "sustain_gain" => -0.5),
        // Opposite: smoothed attack, blooming sustain
        params!("attack_gain" => -0.6, "sustain_gain" => 1.0),
        // Subtle presence
        params!("attack_gain" => 0.3, "sustain_gain" => 0.3),
    ]
}

// ===================================================================
// F004 -- Multiband Dynamics (3-band)
// ===================================================================

/// Compress a single band: inline envelope follow + gain reduction.
fn compress_band(
    samples: &[f32],
    attack_coeff: f32,
    release_coeff: f32,
    threshold_lin: f32,
    ratio: f32,
) -> Vec<f32> {
    let n = samples.len();
    let mut out = vec![0.0f32; n];
    let mut env_prev = 0.0f32;
    for i in 0..n {
        let x = samples[i];
        let inp = x.abs();

        // Envelope follower
        if inp > env_prev {
            env_prev = attack_coeff * env_prev + (1.0 - attack_coeff) * inp;
        } else {
            env_prev = release_coeff * env_prev + (1.0 - release_coeff) * inp;
        }

        let level = env_prev;
        let gain = if level > threshold_lin && level > 1e-12 {
            let over_db = 20.0 * (level / threshold_lin).log10();
            let reduction_db = over_db * (1.0 - 1.0 / ratio);
            10.0f32.powf(-reduction_db / 20.0)
        } else {
            1.0
        };
        out[i] = x * gain;
    }
    out
}

fn process_f004(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let low_xover = pf(params, "low_xover", 250.0);
    let high_xover = pf(params, "high_xover", 4000.0);
    let low_threshold_db = pf(params, "low_threshold_db", -15.0);
    let mid_threshold_db = pf(params, "mid_threshold_db", -15.0);
    let high_threshold_db = pf(params, "high_threshold_db", -15.0);
    let ratio = pf(params, "ratio", 4.0);

    let q = 0.7071067811865476f32; // 1/sqrt(2)

    // Compute crossover filter coefficients
    let (lp1_b0, lp1_b1, lp1_b2, lp1_a1, lp1_a2) = biquad_coeffs_lpf(low_xover, sr, q);
    let (hp1_b0, hp1_b1, hp1_b2, hp1_a1, hp1_a2) = biquad_coeffs_hpf(low_xover, sr, q);
    let (lp2_b0, lp2_b1, lp2_b2, lp2_a1, lp2_a2) = biquad_coeffs_lpf(high_xover, sr, q);
    let (hp2_b0, hp2_b1, hp2_b2, hp2_a1, hp2_a2) = biquad_coeffs_hpf(high_xover, sr, q);

    // Split into 3 bands
    // Low band: lowpass at low_xover
    let low_band = biquad_filter(samples, lp1_b0, lp1_b1, lp1_b2, lp1_a1, lp1_a2);

    // High-passed at low_xover (everything above low_xover)
    let above_low = biquad_filter(samples, hp1_b0, hp1_b1, hp1_b2, hp1_a1, hp1_a2);

    // Mid band: the above_low signal, lowpassed at high_xover
    let mid_band = biquad_filter(&above_low, lp2_b0, lp2_b1, lp2_b2, lp2_a1, lp2_a2);

    // High band: the above_low signal, highpassed at high_xover
    let high_band = biquad_filter(&above_low, hp2_b0, hp2_b1, hp2_b2, hp2_a1, hp2_a2);

    // Compress each band independently
    let attack_coeff = ms_to_coeff(10.0, sr);
    let release_coeff = ms_to_coeff(100.0, sr);

    let low_thresh_lin = 10.0f32.powf(low_threshold_db / 20.0);
    let mid_thresh_lin = 10.0f32.powf(mid_threshold_db / 20.0);
    let high_thresh_lin = 10.0f32.powf(high_threshold_db / 20.0);

    let low_comp = compress_band(&low_band, attack_coeff, release_coeff, low_thresh_lin, ratio);
    let mid_comp = compress_band(&mid_band, attack_coeff, release_coeff, mid_thresh_lin, ratio);
    let high_comp = compress_band(&high_band, attack_coeff, release_coeff, high_thresh_lin, ratio);

    // Sum bands to reconstruct
    let n = samples.len();
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        out[i] = low_comp[i] + mid_comp[i] + high_comp[i];
    }
    AudioOutput::Mono(out)
}

fn variants_f004() -> Vec<HashMap<String, Value>> {
    vec![
        // Balanced gentle multiband
        params!("low_xover" => 200.0, "high_xover" => 4000.0,
                "low_threshold_db" => -15.0, "mid_threshold_db" => -15.0,
                "high_threshold_db" => -15.0, "ratio" => 3.0),
        // Heavy bass control, light highs
        params!("low_xover" => 150.0, "high_xover" => 3000.0,
                "low_threshold_db" => -25.0, "mid_threshold_db" => -12.0,
                "high_threshold_db" => -8.0, "ratio" => 6.0),
        // Tame harsh highs
        params!("low_xover" => 300.0, "high_xover" => 5000.0,
                "low_threshold_db" => -8.0, "mid_threshold_db" => -10.0,
                "high_threshold_db" => -25.0, "ratio" => 5.0),
        // Aggressive full-range squash
        params!("low_xover" => 250.0, "high_xover" => 4000.0,
                "low_threshold_db" => -28.0, "mid_threshold_db" => -28.0,
                "high_threshold_db" => -28.0, "ratio" => 10.0),
        // Wide mid scoop compression
        params!("low_xover" => 400.0, "high_xover" => 2000.0,
                "low_threshold_db" => -10.0, "mid_threshold_db" => -25.0,
                "high_threshold_db" => -10.0, "ratio" => 4.0),
        // Broadcast style
        params!("low_xover" => 120.0, "high_xover" => 6000.0,
                "low_threshold_db" => -20.0, "mid_threshold_db" => -18.0,
                "high_threshold_db" => -22.0, "ratio" => 5.0),
    ]
}

// ===================================================================
// F005 -- Sidechain Ducker
// ===================================================================

/// Generate synthetic kick sidechain signal: sine bursts at regular intervals.
fn generate_kick_pattern(n: usize, sr: u32, beat_ms: f32, kick_freq: f32, kick_dur_ms: f32) -> Vec<f32> {
    let mut sidechain = vec![0.0f32; n];
    let beat_samples = ((beat_ms * 0.001 * sr as f32) as usize).max(1);
    let kick_dur_samples = ((kick_dur_ms * 0.001 * sr as f32) as usize).max(1);

    let mut pos = 0usize;
    while pos < n {
        for j in 0..kick_dur_samples {
            let idx = pos + j;
            if idx >= n {
                break;
            }
            // Sine burst with exponential decay
            let t = j as f32 / sr as f32;
            let decay = (-(j as f32) / (kick_dur_samples as f32 * 0.3)).exp();
            sidechain[idx] = decay * (2.0 * std::f32::consts::PI * kick_freq * t).sin();
        }
        pos += beat_samples;
    }
    sidechain
}

/// Apply ducking based on sidechain envelope.
fn ducker_kernel(samples: &[f32], sc_env: &[f32], duck_amount: f32) -> Vec<f32> {
    let n = samples.len();
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        let gain = (1.0 - duck_amount * sc_env[i]).max(0.0);
        out[i] = samples[i] * gain;
    }
    out
}

fn process_f005(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let beat_ms = pf(params, "beat_ms", 500.0);
    let duck_amount = pf(params, "duck_amount", 0.8);
    let attack_ms = pf(params, "attack_ms", 2.0);
    let release_ms = pf(params, "release_ms", 100.0);

    let n = samples.len();

    // Generate synthetic kick sidechain: 50 Hz sine burst, ~20ms duration
    let sidechain = generate_kick_pattern(n, sr, beat_ms, 50.0, 20.0);

    // Envelope follow the sidechain signal
    let attack_coeff = ms_to_coeff(attack_ms, sr);
    let release_coeff = ms_to_coeff(release_ms, sr);
    let mut sc_env = envelope_follow(&sidechain, attack_coeff, release_coeff);

    // Normalize sidechain envelope to [0, 1]
    let peak = sc_env.iter().cloned().fold(0.0f32, f32::max);
    if peak > 1e-12 {
        for v in sc_env.iter_mut() {
            *v /= peak;
        }
    }

    AudioOutput::Mono(ducker_kernel(samples, &sc_env, duck_amount))
}

fn variants_f005() -> Vec<HashMap<String, Value>> {
    vec![
        // Classic four-on-the-floor pump (120 BPM)
        params!("beat_ms" => 500.0, "duck_amount" => 0.9, "attack_ms" => 2.0, "release_ms" => 150.0),
        // Fast techno pump (140 BPM)
        params!("beat_ms" => 428.0, "duck_amount" => 1.0, "attack_ms" => 1.0, "release_ms" => 100.0),
        // Slow half-time (80 BPM, beats on halves)
        params!("beat_ms" => 375.0, "duck_amount" => 0.7, "attack_ms" => 5.0, "release_ms" => 250.0),
        // Subtle rhythmic ducking
        params!("beat_ms" => 500.0, "duck_amount" => 0.4, "attack_ms" => 3.0, "release_ms" => 200.0),
        // Extreme choppy gate effect
        params!("beat_ms" => 250.0, "duck_amount" => 1.0, "attack_ms" => 1.0, "release_ms" => 60.0),
    ]
}

// ===================================================================
// F006 -- RMS Compressor
// ===================================================================

/// Compute running RMS envelope.
fn rms_envelope(samples: &[f32], window_samples: usize) -> Vec<f32> {
    let n = samples.len();
    let mut env = vec![0.0f32; n];
    let mut sum_sq = 0.0f32;
    let w = window_samples.max(1);
    for i in 0..n {
        sum_sq += samples[i] * samples[i];
        if i >= w {
            sum_sq -= samples[i - w] * samples[i - w];
            if sum_sq < 0.0 {
                sum_sq = 0.0;
            }
        }
        let count = (i + 1).min(w);
        env[i] = (sum_sq / count as f32).sqrt();
    }
    env
}

/// Compress based on smoothed RMS envelope.
fn rms_compress_kernel(
    samples: &[f32],
    env: &[f32],
    threshold_lin: f32,
    ratio: f32,
    makeup_gain: f32,
    attack_coeff: f32,
    release_coeff: f32,
) -> Vec<f32> {
    let n = samples.len();
    let mut out = vec![0.0f32; n];
    let mut smooth_env = 0.0f32;
    for i in 0..n {
        let level = env[i];
        if level > smooth_env {
            smooth_env = attack_coeff * smooth_env + (1.0 - attack_coeff) * level;
        } else {
            smooth_env = release_coeff * smooth_env + (1.0 - release_coeff) * level;
        }
        let gain = if smooth_env > threshold_lin && smooth_env > 1e-12 {
            let over_db = 20.0 * (smooth_env / threshold_lin).log10();
            let reduction_db = over_db * (1.0 - 1.0 / ratio);
            10.0f32.powf(-reduction_db / 20.0)
        } else {
            1.0
        };
        out[i] = samples[i] * gain * makeup_gain;
    }
    out
}

fn process_f006(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let threshold_db = pf(params, "threshold_db", -20.0);
    let ratio = pf(params, "ratio", 4.0);
    let attack_ms = pf(params, "attack_ms", 10.0);
    let release_ms = pf(params, "release_ms", 100.0);
    let rms_window_ms = pf(params, "rms_window_ms", 50.0);

    let window_samples = (rms_window_ms * 0.001 * sr as f32) as usize;
    let env = rms_envelope(samples, window_samples);
    let attack_coeff = ms_to_coeff(attack_ms, sr);
    let release_coeff = ms_to_coeff(release_ms, sr);
    let threshold_lin = 10.0f32.powf(threshold_db / 20.0);
    let makeup_db = (threshold_db.abs() * (1.0 - 1.0 / ratio)) * 0.5;
    let makeup_gain = 10.0f32.powf(makeup_db / 20.0);
    AudioOutput::Mono(rms_compress_kernel(
        samples,
        &env,
        threshold_lin,
        ratio,
        makeup_gain,
        attack_coeff,
        release_coeff,
    ))
}

fn variants_f006() -> Vec<HashMap<String, Value>> {
    vec![
        // Gentle vocal leveler
        params!("threshold_db" => -18.0, "ratio" => 3.0, "attack_ms" => 20.0, "release_ms" => 200.0, "rms_window_ms" => 80.0),
        // Mix bus glue
        params!("threshold_db" => -14.0, "ratio" => 2.0, "attack_ms" => 30.0, "release_ms" => 300.0, "rms_window_ms" => 100.0),
        // Heavy sustain boost
        params!("threshold_db" => -28.0, "ratio" => 8.0, "attack_ms" => 10.0, "release_ms" => 150.0, "rms_window_ms" => 50.0),
        // Pad leveler (very slow)
        params!("threshold_db" => -22.0, "ratio" => 4.0, "attack_ms" => 60.0, "release_ms" => 600.0, "rms_window_ms" => 150.0),
        // Tight RMS squash
        params!("threshold_db" => -24.0, "ratio" => 12.0, "attack_ms" => 5.0, "release_ms" => 80.0, "rms_window_ms" => 20.0),
        // Bass smoothing
        params!("threshold_db" => -20.0, "ratio" => 5.0, "attack_ms" => 15.0, "release_ms" => 120.0, "rms_window_ms" => 60.0),
    ]
}

// ===================================================================
// F007 -- Feedback Compressor
// ===================================================================

/// Feedback topology: envelope follows the output, not the input.
fn feedback_compress_kernel(
    samples: &[f32],
    threshold_lin: f32,
    ratio: f32,
    makeup_gain: f32,
    attack_coeff: f32,
    release_coeff: f32,
) -> Vec<f32> {
    let n = samples.len();
    let mut out = vec![0.0f32; n];
    let mut env = 0.0f32;
    let mut prev_gain = 1.0f32;
    for i in 0..n {
        // Envelope follows the previous output sample
        let out_level = (samples[i] * prev_gain).abs();
        if out_level > env {
            env = attack_coeff * env + (1.0 - attack_coeff) * out_level;
        } else {
            env = release_coeff * env + (1.0 - release_coeff) * out_level;
        }
        let gain = if env > threshold_lin && env > 1e-12 {
            let over_db = 20.0 * (env / threshold_lin).log10();
            let reduction_db = over_db * (1.0 - 1.0 / ratio);
            10.0f32.powf(-reduction_db / 20.0)
        } else {
            1.0
        };
        prev_gain = gain;
        out[i] = samples[i] * gain * makeup_gain;
    }
    out
}

fn process_f007(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let threshold_db = pf(params, "threshold_db", -20.0);
    let ratio = pf(params, "ratio", 4.0);
    let attack_ms = pf(params, "attack_ms", 10.0);
    let release_ms = pf(params, "release_ms", 100.0);

    let attack_coeff = ms_to_coeff(attack_ms, sr);
    let release_coeff = ms_to_coeff(release_ms, sr);
    let threshold_lin = 10.0f32.powf(threshold_db / 20.0);
    let makeup_db = (threshold_db.abs() * (1.0 - 1.0 / ratio)) * 0.4;
    let makeup_gain = 10.0f32.powf(makeup_db / 20.0);
    AudioOutput::Mono(feedback_compress_kernel(
        samples,
        threshold_lin,
        ratio,
        makeup_gain,
        attack_coeff,
        release_coeff,
    ))
}

fn variants_f007() -> Vec<HashMap<String, Value>> {
    vec![
        // Subtle vintage leveling
        params!("threshold_db" => -16.0, "ratio" => 3.0, "attack_ms" => 20.0, "release_ms" => 200.0),
        // Aggressive pumping
        params!("threshold_db" => -24.0, "ratio" => 8.0, "attack_ms" => 5.0, "release_ms" => 80.0),
        // Bass fattener
        params!("threshold_db" => -20.0, "ratio" => 6.0, "attack_ms" => 15.0, "release_ms" => 150.0),
        // Gentle mix bus
        params!("threshold_db" => -12.0, "ratio" => 2.0, "attack_ms" => 30.0, "release_ms" => 300.0),
        // Vocal warmth
        params!("threshold_db" => -22.0, "ratio" => 4.0, "attack_ms" => 8.0, "release_ms" => 120.0),
        // Hard feedback slam
        params!("threshold_db" => -30.0, "ratio" => 15.0, "attack_ms" => 2.0, "release_ms" => 60.0),
    ]
}

// ===================================================================
// F008 -- Opto Compressor
// ===================================================================

/// Opto-style: two-stage release models photocell behavior.
///
/// The release has a fast initial component and a slow exponential tail.
/// Louder signals drive the slow component harder (program-dependent).
fn opto_compress_kernel(
    samples: &[f32],
    threshold_lin: f32,
    ratio: f32,
    makeup_gain: f32,
    attack_coeff: f32,
    fast_rel_coeff: f32,
    slow_rel_coeff: f32,
    slow_blend: f32,
) -> Vec<f32> {
    let n = samples.len();
    let mut out = vec![0.0f32; n];
    let mut fast_env = 0.0f32;
    let mut slow_env = 0.0f32;
    for i in 0..n {
        let inp = samples[i].abs();
        // Fast envelope
        if inp > fast_env {
            fast_env = attack_coeff * fast_env + (1.0 - attack_coeff) * inp;
        } else {
            fast_env *= fast_rel_coeff;
        }
        // Slow envelope -- driven harder by loud signals
        if inp > slow_env {
            slow_env = attack_coeff * slow_env + (1.0 - attack_coeff) * inp;
        } else {
            slow_env *= slow_rel_coeff;
        }
        // Blend the two envelopes
        let env = (1.0 - slow_blend) * fast_env + slow_blend * slow_env;
        let gain = if env > threshold_lin && env > 1e-12 {
            let over_db = 20.0 * (env / threshold_lin).log10();
            let reduction_db = over_db * (1.0 - 1.0 / ratio);
            10.0f32.powf(-reduction_db / 20.0)
        } else {
            1.0
        };
        out[i] = samples[i] * gain * makeup_gain;
    }
    out
}

fn process_f008(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let threshold_db = pf(params, "threshold_db", -20.0);
    let ratio = pf(params, "ratio", 4.0);
    let attack_ms = pf(params, "attack_ms", 10.0);
    let fast_release_ms = pf(params, "fast_release_ms", 60.0);
    let slow_release_ms = pf(params, "slow_release_ms", 500.0);
    let slow_blend = pf(params, "slow_blend", 0.5);

    let attack_coeff = ms_to_coeff(attack_ms, sr);
    let fast_rel_coeff = ms_to_coeff(fast_release_ms, sr);
    let slow_rel_coeff = ms_to_coeff(slow_release_ms, sr);
    let threshold_lin = 10.0f32.powf(threshold_db / 20.0);
    let makeup_db = (threshold_db.abs() * (1.0 - 1.0 / ratio)) * 0.45;
    let makeup_gain = 10.0f32.powf(makeup_db / 20.0);
    AudioOutput::Mono(opto_compress_kernel(
        samples,
        threshold_lin,
        ratio,
        makeup_gain,
        attack_coeff,
        fast_rel_coeff,
        slow_rel_coeff,
        slow_blend,
    ))
}

fn variants_f008() -> Vec<HashMap<String, Value>> {
    vec![
        // Classic LA-2A vocal
        params!("threshold_db" => -20.0, "ratio" => 4.0, "attack_ms" => 10.0,
                "fast_release_ms" => 60.0, "slow_release_ms" => 500.0, "slow_blend" => 0.6),
        // Fingerpicked guitar
        params!("threshold_db" => -18.0, "ratio" => 3.0, "attack_ms" => 15.0,
                "fast_release_ms" => 80.0, "slow_release_ms" => 800.0, "slow_blend" => 0.4),
        // Pad leveling (mostly slow)
        params!("threshold_db" => -24.0, "ratio" => 5.0, "attack_ms" => 20.0,
                "fast_release_ms" => 100.0, "slow_release_ms" => 1500.0, "slow_blend" => 0.8),
        // Snappy opto (mostly fast)
        params!("threshold_db" => -16.0, "ratio" => 6.0, "attack_ms" => 5.0,
                "fast_release_ms" => 40.0, "slow_release_ms" => 300.0, "slow_blend" => 0.2),
        // Heavy opto squash
        params!("threshold_db" => -28.0, "ratio" => 10.0, "attack_ms" => 8.0,
                "fast_release_ms" => 50.0, "slow_release_ms" => 600.0, "slow_blend" => 0.5),
        // Gentle acoustic smoothing
        params!("threshold_db" => -15.0, "ratio" => 2.5, "attack_ms" => 12.0,
                "fast_release_ms" => 70.0, "slow_release_ms" => 1000.0, "slow_blend" => 0.7),
    ]
}

// ===================================================================
// F009 -- FET Compressor
// ===================================================================

/// FET-style: ultra-fast, can push into harmonic distortion.
fn fet_compress_kernel(
    samples: &[f32],
    threshold_lin: f32,
    ratio: f32,
    makeup_gain: f32,
    attack_coeff: f32,
    release_coeff: f32,
    drive: f32,
) -> Vec<f32> {
    let n = samples.len();
    let mut out = vec![0.0f32; n];
    let mut env = 0.0f32;
    for i in 0..n {
        // Drive the input
        let driven = samples[i] * drive;
        let inp = driven.abs();
        if inp > env {
            env = attack_coeff * env + (1.0 - attack_coeff) * inp;
        } else {
            env = release_coeff * env + (1.0 - release_coeff) * inp;
        }
        let gain = if env > threshold_lin && env > 1e-12 {
            let over_db = 20.0 * (env / threshold_lin).log10();
            let reduction_db = over_db * (1.0 - 1.0 / ratio);
            10.0f32.powf(-reduction_db / 20.0)
        } else {
            1.0
        };
        // FET-style saturation on heavy gain reduction
        let mut out_sample = driven * gain * makeup_gain;
        // Soft clip at extremes
        if out_sample > 1.0 || out_sample < -1.0 {
            out_sample = out_sample.tanh();
        }
        out[i] = out_sample;
    }
    out
}

fn process_f009(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let threshold_db = pf(params, "threshold_db", -20.0);
    let ratio = pf(params, "ratio", 8.0);
    let attack_ms = pf(params, "attack_ms", 0.2);
    let release_ms = pf(params, "release_ms", 50.0);
    let input_drive = pf(params, "input_drive", 1.0);

    let attack_coeff = ms_to_coeff(attack_ms, sr);
    let release_coeff = ms_to_coeff(release_ms, sr);
    let threshold_lin = 10.0f32.powf(threshold_db / 20.0);
    let makeup_db = (threshold_db.abs() * (1.0 - 1.0 / ratio)) * 0.4;
    let makeup_gain = 10.0f32.powf(makeup_db / 20.0);
    AudioOutput::Mono(fet_compress_kernel(
        samples,
        threshold_lin,
        ratio,
        makeup_gain,
        attack_coeff,
        release_coeff,
        input_drive,
    ))
}

fn variants_f009() -> Vec<HashMap<String, Value>> {
    vec![
        // All-buttons-in (1176 trick: everything fast, max ratio)
        params!("threshold_db" => -20.0, "ratio" => 20.0, "attack_ms" => 0.05, "release_ms" => 30.0, "input_drive" => 2.0),
        // Drum smash
        params!("threshold_db" => -24.0, "ratio" => 12.0, "attack_ms" => 0.1, "release_ms" => 50.0, "input_drive" => 1.5),
        // Bass crunch
        params!("threshold_db" => -18.0, "ratio" => 8.0, "attack_ms" => 0.5, "release_ms" => 80.0, "input_drive" => 1.8),
        // Vocal bite
        params!("threshold_db" => -22.0, "ratio" => 8.0, "attack_ms" => 0.2, "release_ms" => 60.0, "input_drive" => 1.2),
        // Gentle FET color
        params!("threshold_db" => -14.0, "ratio" => 4.0, "attack_ms" => 0.8, "release_ms" => 100.0, "input_drive" => 1.0),
        // Extreme destruction
        params!("threshold_db" => -30.0, "ratio" => 20.0, "attack_ms" => 0.02, "release_ms" => 20.0, "input_drive" => 4.0),
    ]
}

// ===================================================================
// F010 -- Soft-Knee Compressor
// ===================================================================

/// Soft-knee: gradual onset of compression around the threshold.
fn softknee_compress_kernel(
    samples: &[f32],
    env: &[f32],
    threshold_db: f32,
    ratio: f32,
    knee_db: f32,
    makeup_gain: f32,
) -> Vec<f32> {
    let n = samples.len();
    let mut out = vec![0.0f32; n];
    let half_knee = knee_db * 0.5;
    for i in 0..n {
        let level = env[i];
        if level < 1e-12 {
            out[i] = samples[i] * makeup_gain;
            continue;
        }
        let level_db = 20.0 * level.log10();
        let gain_db = if level_db < threshold_db - half_knee {
            // Below knee: no compression
            0.0
        } else if level_db > threshold_db + half_knee {
            // Above knee: full compression
            let over_db = level_db - threshold_db;
            -over_db * (1.0 - 1.0 / ratio)
        } else {
            // In the knee: quadratic interpolation
            let x = level_db - threshold_db + half_knee;
            -(1.0 - 1.0 / ratio) * x * x / (2.0 * knee_db)
        };
        let gain = 10.0f32.powf(gain_db / 20.0);
        out[i] = samples[i] * gain * makeup_gain;
    }
    out
}

fn process_f010(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let threshold_db = pf(params, "threshold_db", -20.0);
    let ratio = pf(params, "ratio", 4.0);
    let knee_db = pf(params, "knee_db", 10.0).max(0.01);
    let attack_ms = pf(params, "attack_ms", 10.0);
    let release_ms = pf(params, "release_ms", 100.0);

    let attack_coeff = ms_to_coeff(attack_ms, sr);
    let release_coeff = ms_to_coeff(release_ms, sr);
    let env = envelope_follow(samples, attack_coeff, release_coeff);
    let makeup_db = (threshold_db.abs() * (1.0 - 1.0 / ratio)) * 0.4;
    let makeup_gain = 10.0f32.powf(makeup_db / 20.0);
    AudioOutput::Mono(softknee_compress_kernel(
        samples,
        &env,
        threshold_db,
        ratio,
        knee_db,
        makeup_gain,
    ))
}

fn variants_f010() -> Vec<HashMap<String, Value>> {
    vec![
        // Wide knee transparent (mastering)
        params!("threshold_db" => -16.0, "ratio" => 2.5, "knee_db" => 16.0, "attack_ms" => 20.0, "release_ms" => 200.0),
        // Narrow knee punchy
        params!("threshold_db" => -22.0, "ratio" => 6.0, "knee_db" => 3.0, "attack_ms" => 5.0, "release_ms" => 80.0),
        // Mastering limiter
        params!("threshold_db" => -8.0, "ratio" => 15.0, "knee_db" => 6.0, "attack_ms" => 1.0, "release_ms" => 50.0),
        // Gentle bus glue (wide knee)
        params!("threshold_db" => -14.0, "ratio" => 2.0, "knee_db" => 20.0, "attack_ms" => 30.0, "release_ms" => 300.0),
        // Vocal polish
        params!("threshold_db" => -20.0, "ratio" => 4.0, "knee_db" => 10.0, "attack_ms" => 10.0, "release_ms" => 120.0),
        // Medium knee drum tamer
        params!("threshold_db" => -18.0, "ratio" => 8.0, "knee_db" => 8.0, "attack_ms" => 2.0, "release_ms" => 60.0),
    ]
}

// ===================================================================
// F011 -- Parallel (NY) Compressor
// ===================================================================

fn process_f011(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let threshold_db = pf(params, "threshold_db", -30.0);
    let ratio = pf(params, "ratio", 10.0);
    let attack_ms = pf(params, "attack_ms", 5.0);
    let release_ms = pf(params, "release_ms", 100.0);
    let wet_mix = pf(params, "wet_mix", 0.5);

    let attack_coeff = ms_to_coeff(attack_ms, sr);
    let release_coeff = ms_to_coeff(release_ms, sr);
    let env = envelope_follow(samples, attack_coeff, release_coeff);
    let threshold_lin = 10.0f32.powf(threshold_db / 20.0);
    // Heavy makeup for the compressed signal
    let makeup_db = (threshold_db.abs() * (1.0 - 1.0 / ratio)) * 0.7;
    let makeup_gain = 10.0f32.powf(makeup_db / 20.0);
    let compressed = compress_kernel(samples, &env, threshold_lin, ratio, makeup_gain);
    let dry = 1.0 - wet_mix;
    let n = samples.len();
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        out[i] = dry * samples[i] + wet_mix * compressed[i];
    }
    AudioOutput::Mono(out)
}

fn variants_f011() -> Vec<HashMap<String, Value>> {
    vec![
        // Drum punch (fast, 50% wet)
        params!("threshold_db" => -28.0, "ratio" => 12.0, "attack_ms" => 2.0, "release_ms" => 80.0, "wet_mix" => 0.5),
        // Vocal thickness (slower, 30% wet)
        params!("threshold_db" => -24.0, "ratio" => 8.0, "attack_ms" => 15.0, "release_ms" => 150.0, "wet_mix" => 0.3),
        // Full smash blend
        params!("threshold_db" => -35.0, "ratio" => 100.0, "attack_ms" => 1.0, "release_ms" => 50.0, "wet_mix" => 0.4),
        // Subtle body
        params!("threshold_db" => -20.0, "ratio" => 6.0, "attack_ms" => 10.0, "release_ms" => 120.0, "wet_mix" => 0.25),
        // Ambient room lift
        params!("threshold_db" => -32.0, "ratio" => 20.0, "attack_ms" => 20.0, "release_ms" => 300.0, "wet_mix" => 0.6),
        // Bass parallel thickening
        params!("threshold_db" => -26.0, "ratio" => 10.0, "attack_ms" => 8.0, "release_ms" => 100.0, "wet_mix" => 0.45),
    ]
}

// ===================================================================
// F012 -- Upward Compressor
// ===================================================================

/// Boost signal below threshold -- upward compression.
fn upward_compress_kernel(
    samples: &[f32],
    env: &[f32],
    threshold_lin: f32,
    ratio: f32,
    max_boost_lin: f32,
    attack_coeff: f32,
    release_coeff: f32,
) -> Vec<f32> {
    let n = samples.len();
    let mut out = vec![0.0f32; n];
    let mut smooth = 0.0f32;
    for i in 0..n {
        let level = env[i];
        if level > smooth {
            smooth = attack_coeff * smooth + (1.0 - attack_coeff) * level;
        } else {
            smooth = release_coeff * smooth + (1.0 - release_coeff) * level;
        }
        let gain = if smooth < threshold_lin && smooth > 1e-12 {
            let under_db = 20.0 * (threshold_lin / smooth).log10();
            let boost_db = under_db * (1.0 - 1.0 / ratio);
            let g = 10.0f32.powf(boost_db / 20.0);
            g.min(max_boost_lin)
        } else {
            1.0
        };
        out[i] = samples[i] * gain;
    }
    out
}

fn process_f012(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let threshold_db = pf(params, "threshold_db", -30.0);
    let ratio = pf(params, "ratio", 3.0);
    let attack_ms = pf(params, "attack_ms", 10.0);
    let release_ms = pf(params, "release_ms", 150.0);
    let max_boost_db = pf(params, "max_boost_db", 20.0);

    let attack_coeff = ms_to_coeff(attack_ms, sr);
    let release_coeff = ms_to_coeff(release_ms, sr);
    let env = envelope_follow(samples, attack_coeff, release_coeff);
    let threshold_lin = 10.0f32.powf(threshold_db / 20.0);
    let max_boost_lin = 10.0f32.powf(max_boost_db / 20.0);
    AudioOutput::Mono(upward_compress_kernel(
        samples,
        &env,
        threshold_lin,
        ratio,
        max_boost_lin,
        attack_coeff,
        release_coeff,
    ))
}

fn variants_f012() -> Vec<HashMap<String, Value>> {
    vec![
        // Room tone lifter
        params!("threshold_db" => -35.0, "ratio" => 3.0, "attack_ms" => 15.0, "release_ms" => 200.0, "max_boost_db" => 18.0),
        // Reverb tail enhancer
        params!("threshold_db" => -40.0, "ratio" => 4.0, "attack_ms" => 20.0, "release_ms" => 400.0, "max_boost_db" => 24.0),
        // Subtle detail boost
        params!("threshold_db" => -28.0, "ratio" => 2.0, "attack_ms" => 10.0, "release_ms" => 150.0, "max_boost_db" => 12.0),
        // Aggressive uplift
        params!("threshold_db" => -25.0, "ratio" => 8.0, "attack_ms" => 5.0, "release_ms" => 100.0, "max_boost_db" => 20.0),
        // Ghost note reveal
        params!("threshold_db" => -32.0, "ratio" => 5.0, "attack_ms" => 3.0, "release_ms" => 80.0, "max_boost_db" => 15.0),
        // Ambient texture lift
        params!("threshold_db" => -38.0, "ratio" => 3.5, "attack_ms" => 30.0, "release_ms" => 500.0, "max_boost_db" => 22.0),
    ]
}

// ===================================================================
// F013 -- Program-Dependent Compressor
// ===================================================================

/// Release adapts based on crest factor: transients get fast release,
/// sustained signals get slow release.
fn program_dependent_kernel(
    samples: &[f32],
    threshold_lin: f32,
    ratio: f32,
    makeup_gain: f32,
    attack_coeff: f32,
    min_rel_coeff: f32,
    max_rel_coeff: f32,
) -> Vec<f32> {
    let n = samples.len();
    let mut out = vec![0.0f32; n];
    let mut env = 0.0f32;
    let mut rms_acc = 0.0f32;
    let rms_coeff = 0.999f32;
    for i in 0..n {
        let inp = samples[i].abs();
        // Running RMS estimate
        rms_acc = rms_coeff * rms_acc + (1.0 - rms_coeff) * (samples[i] * samples[i]);
        let rms = rms_acc.sqrt();
        // Crest factor: peak / rms (high = transient, low = sustained)
        let crest = if rms > 1e-12 { inp / rms } else { 1.0 };
        // Map crest to release coefficient: high crest -> fast release (min_rel_coeff)
        // Clamp crest to [1, 10] range for mapping
        let crest_norm = (crest.clamp(1.0, 10.0) - 1.0) / 9.0;
        let rel_coeff = max_rel_coeff + (min_rel_coeff - max_rel_coeff) * crest_norm;
        if inp > env {
            env = attack_coeff * env + (1.0 - attack_coeff) * inp;
        } else {
            env = rel_coeff * env + (1.0 - rel_coeff) * inp;
        }
        let gain = if env > threshold_lin && env > 1e-12 {
            let over_db = 20.0 * (env / threshold_lin).log10();
            let reduction_db = over_db * (1.0 - 1.0 / ratio);
            10.0f32.powf(-reduction_db / 20.0)
        } else {
            1.0
        };
        out[i] = samples[i] * gain * makeup_gain;
    }
    out
}

fn process_f013(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let threshold_db = pf(params, "threshold_db", -20.0);
    let ratio = pf(params, "ratio", 4.0);
    let attack_ms = pf(params, "attack_ms", 5.0);
    let min_release_ms = pf(params, "min_release_ms", 20.0);
    let max_release_ms = pf(params, "max_release_ms", 500.0);

    let attack_coeff = ms_to_coeff(attack_ms, sr);
    let min_rel_coeff = ms_to_coeff(min_release_ms, sr);
    let max_rel_coeff = ms_to_coeff(max_release_ms, sr);
    let threshold_lin = 10.0f32.powf(threshold_db / 20.0);
    let makeup_db = (threshold_db.abs() * (1.0 - 1.0 / ratio)) * 0.5;
    let makeup_gain = 10.0f32.powf(makeup_db / 20.0);
    AudioOutput::Mono(program_dependent_kernel(
        samples,
        threshold_lin,
        ratio,
        makeup_gain,
        attack_coeff,
        min_rel_coeff,
        max_rel_coeff,
    ))
}

fn variants_f013() -> Vec<HashMap<String, Value>> {
    vec![
        // Adaptive drums
        params!("threshold_db" => -22.0, "ratio" => 6.0, "attack_ms" => 2.0, "min_release_ms" => 15.0, "max_release_ms" => 300.0),
        // Adaptive full mix
        params!("threshold_db" => -16.0, "ratio" => 3.0, "attack_ms" => 10.0, "min_release_ms" => 30.0, "max_release_ms" => 500.0),
        // Aggressive auto-release
        params!("threshold_db" => -26.0, "ratio" => 10.0, "attack_ms" => 1.0, "min_release_ms" => 10.0, "max_release_ms" => 200.0),
        // Gentle vocal riding
        params!("threshold_db" => -20.0, "ratio" => 4.0, "attack_ms" => 8.0, "min_release_ms" => 40.0, "max_release_ms" => 800.0),
        // Fast transient preserve
        params!("threshold_db" => -18.0, "ratio" => 5.0, "attack_ms" => 5.0, "min_release_ms" => 5.0, "max_release_ms" => 400.0),
        // Wide auto range
        params!("threshold_db" => -24.0, "ratio" => 8.0, "attack_ms" => 3.0, "min_release_ms" => 10.0, "max_release_ms" => 1500.0),
    ]
}

// ===================================================================
// F014 -- Lookahead Limiter
// ===================================================================

/// Lookahead limiter: sliding window max + delayed audio.
fn lookahead_limiter_kernel(
    samples: &[f32],
    threshold_lin: f32,
    release_coeff: f32,
    lookahead: usize,
) -> Vec<f32> {
    let n = samples.len();
    let mut out = vec![0.0f32; n];
    let la = lookahead.max(1);

    // Sliding window max using monotone deque
    // We store (index, value) pairs; front is always the current max.
    let cap = la + 1;
    let mut ring_idx = vec![0usize; cap];
    let mut ring_val = vec![0.0f32; cap];
    let mut head: usize = 0;
    let mut tail: usize = 0; // tail is one past the last valid entry

    let mut peak_env = vec![0.0f32; n];
    for i in 0..n {
        // The window covers samples[i .. i+la]
        let right = (i + la).min(n - 1);
        let v = samples[right].abs();
        // Remove from back anything smaller than v
        while tail > head && ring_val[tail - 1] <= v {
            tail -= 1;
        }
        ring_idx[tail % cap] = right;
        ring_val[tail % cap] = v;
        tail += 1;
        // Remove from front anything outside the window
        while head < tail && ring_idx[head % cap] < i {
            head += 1;
        }
        peak_env[i] = ring_val[head % cap];
    }

    // Smooth the envelope (release only -- attack is instant due to lookahead)
    let mut smooth = peak_env[0];
    for i in 0..n {
        if peak_env[i] > smooth {
            smooth = peak_env[i];
        } else {
            smooth = release_coeff * smooth + (1.0 - release_coeff) * peak_env[i];
        }
        peak_env[i] = smooth;
    }

    // Apply gain reduction to delayed audio
    for i in 0..n {
        let sample = if i >= la {
            samples[i - la]
        } else {
            0.0
        };
        let level = peak_env[i];
        let gain = if level > threshold_lin {
            threshold_lin / level
        } else {
            1.0
        };
        out[i] = sample * gain;
    }
    out
}

fn process_f014(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let threshold_db = pf(params, "threshold_db", -1.0);
    let release_ms = pf(params, "release_ms", 50.0);
    let lookahead_ms = pf(params, "lookahead_ms", 5.0);

    let threshold_lin = 10.0f32.powf(threshold_db / 20.0);
    let release_coeff = ms_to_coeff(release_ms, sr);
    let lookahead_samples = (lookahead_ms * 0.001 * sr as f32) as usize;
    AudioOutput::Mono(lookahead_limiter_kernel(
        samples,
        threshold_lin,
        release_coeff,
        lookahead_samples,
    ))
}

fn variants_f014() -> Vec<HashMap<String, Value>> {
    vec![
        // Transparent brickwall (mastering)
        params!("threshold_db" => -1.0, "release_ms" => 50.0, "lookahead_ms" => 5.0),
        // Punchy short lookahead
        params!("threshold_db" => -2.0, "release_ms" => 30.0, "lookahead_ms" => 1.0),
        // Aggressive 5ms lookahead
        params!("threshold_db" => -3.0, "release_ms" => 40.0, "lookahead_ms" => 5.0),
        // Heavy ceiling
        params!("threshold_db" => -6.0, "release_ms" => 60.0, "lookahead_ms" => 3.0),
        // Long lookahead, slow release
        params!("threshold_db" => -1.5, "release_ms" => 100.0, "lookahead_ms" => 10.0),
        // Broadcast ceiling
        params!("threshold_db" => -0.5, "release_ms" => 80.0, "lookahead_ms" => 5.0),
    ]
}

// ===================================================================
// F015 -- Spectral Compressor
// ===================================================================

fn process_f015(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    use crate::stft::{stft, istft};
    use num_complex::Complex;

    let threshold_db = pf(params, "threshold_db", -20.0);
    let ratio = pf(params, "ratio", 4.0);
    let fft_size = pi(params, "fft_size", 2048) as usize;
    let attack_ms = pf(params, "attack_ms", 10.0);
    let release_ms = pf(params, "release_ms", 100.0);

    let n = samples.len();
    let threshold_lin = 10.0f32.powf(threshold_db / 20.0);
    let inv_ratio = 1.0 - 1.0 / ratio;
    let hop_size = fft_size / 4;

    // Forward STFT
    let mut frames = stft(samples, fft_size, hop_size);
    let n_bins = fft_size / 2 + 1;
    let n_frames = frames.len();

    // Per-bin envelope smoothing across frames
    let attack_c = if attack_ms > 0.0 {
        (-1.0 / (attack_ms * 0.001 * sr as f32 / hop_size as f32)).exp()
    } else {
        0.0
    };
    let release_c = if release_ms > 0.0 {
        (-1.0 / (release_ms * 0.001 * sr as f32 / hop_size as f32)).exp()
    } else {
        0.0
    };

    let mut bin_env = vec![0.0f32; n_bins];

    for t in 0..n_frames {
        for b in 0..n_bins.min(frames[t].len()) {
            let frame_mag = frames[t][b].norm();
            // Envelope follow per-bin
            if frame_mag > bin_env[b] {
                bin_env[b] = attack_c * bin_env[b] + (1.0 - attack_c) * frame_mag;
            } else {
                bin_env[b] = release_c * bin_env[b] + (1.0 - release_c) * frame_mag;
            }
            // Compress if above threshold
            if bin_env[b] > threshold_lin && bin_env[b] > 1e-12 {
                let over_db = 20.0 * (bin_env[b] / threshold_lin).log10();
                let reduction_db = over_db * inv_ratio;
                let gain = 10.0f32.powf(-reduction_db / 20.0);
                frames[t][b] = frames[t][b] * gain;
            }
        }
    }

    // Inverse STFT
    let mut out = istft(&frames, fft_size, hop_size, Some(n));

    // Match original length
    out.truncate(n);
    while out.len() < n {
        out.push(0.0);
    }

    AudioOutput::Mono(out)
}

fn variants_f015() -> Vec<HashMap<String, Value>> {
    vec![
        // Resonance tamer
        params!("threshold_db" => -18.0, "ratio" => 6.0, "fft_size" => 2048, "attack_ms" => 5.0, "release_ms" => 60.0),
        // Spectral leveler
        params!("threshold_db" => -24.0, "ratio" => 4.0, "fft_size" => 2048, "attack_ms" => 15.0, "release_ms" => 150.0),
        // De-harshener (smaller FFT, faster)
        params!("threshold_db" => -16.0, "ratio" => 3.0, "fft_size" => 1024, "attack_ms" => 8.0, "release_ms" => 80.0),
        // Fine spectral control
        params!("threshold_db" => -22.0, "ratio" => 8.0, "fft_size" => 4096, "attack_ms" => 10.0, "release_ms" => 100.0),
        // Aggressive spectral squash
        params!("threshold_db" => -30.0, "ratio" => 12.0, "fft_size" => 2048, "attack_ms" => 3.0, "release_ms" => 40.0),
        // Gentle spectral evening
        params!("threshold_db" => -14.0, "ratio" => 2.0, "fft_size" => 2048, "attack_ms" => 20.0, "release_ms" => 200.0),
    ]
}

// ===================================================================
// Registration
// ===================================================================

pub fn register() -> Vec<EffectEntry> {
    vec![
        EffectEntry {
            id: "F001",
            process: process_f001,
            variants: variants_f001,
            category: "dynamics",
        },
        EffectEntry {
            id: "F002",
            process: process_f002,
            variants: variants_f002,
            category: "dynamics",
        },
        EffectEntry {
            id: "F003",
            process: process_f003,
            variants: variants_f003,
            category: "dynamics",
        },
        EffectEntry {
            id: "F004",
            process: process_f004,
            variants: variants_f004,
            category: "dynamics",
        },
        EffectEntry {
            id: "F005",
            process: process_f005,
            variants: variants_f005,
            category: "dynamics",
        },
        EffectEntry {
            id: "F006",
            process: process_f006,
            variants: variants_f006,
            category: "dynamics",
        },
        EffectEntry {
            id: "F007",
            process: process_f007,
            variants: variants_f007,
            category: "dynamics",
        },
        EffectEntry {
            id: "F008",
            process: process_f008,
            variants: variants_f008,
            category: "dynamics",
        },
        EffectEntry {
            id: "F009",
            process: process_f009,
            variants: variants_f009,
            category: "dynamics",
        },
        EffectEntry {
            id: "F010",
            process: process_f010,
            variants: variants_f010,
            category: "dynamics",
        },
        EffectEntry {
            id: "F011",
            process: process_f011,
            variants: variants_f011,
            category: "dynamics",
        },
        EffectEntry {
            id: "F012",
            process: process_f012,
            variants: variants_f012,
            category: "dynamics",
        },
        EffectEntry {
            id: "F013",
            process: process_f013,
            variants: variants_f013,
            category: "dynamics",
        },
        EffectEntry {
            id: "F014",
            process: process_f014,
            variants: variants_f014,
            category: "dynamics",
        },
        EffectEntry {
            id: "F015",
            process: process_f015,
            variants: variants_f015,
            category: "dynamics",
        },
    ]
}
