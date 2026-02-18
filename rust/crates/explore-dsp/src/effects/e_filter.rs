//! E-series effects: Filters (E001-E012).
//!
//! Biquad parametric EQ, state variable filter, Moog ladder, comb filter,
//! formant filter, vowel morph, auto-wah, resonant sweep, multi-mode
//! crossfade, detuned resonators, allpass lattice, pitch-tracking resonator.

use std::collections::HashMap;
use std::f32::consts::PI;

use serde_json::Value;

use crate::primitives::*;
use crate::{params, pf, pi, ps, AudioOutput, EffectEntry};

// ---------------------------------------------------------------------------
// E001 -- Biquad Filter (Parametric EQ)
// ---------------------------------------------------------------------------

fn process_e001(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let filter_type = ps(params, "filter_type", "lpf");
    let freq_hz = pf(params, "freq_hz", 1000.0).clamp(20.0, sr as f32 * 0.499);
    let q = pf(params, "Q", 1.0).clamp(0.1, 20.0);
    let gain_db = pf(params, "gain_db", 0.0);

    let (b0, b1, b2, a1, a2) = match filter_type {
        "lpf" => biquad_coeffs_lpf(freq_hz, sr, q),
        "hpf" => biquad_coeffs_hpf(freq_hz, sr, q),
        "bpf" => biquad_coeffs_bpf(freq_hz, sr, q),
        "notch" => biquad_coeffs_notch(freq_hz, sr, q),
        "peak" => biquad_coeffs_peak(freq_hz, sr, q, gain_db),
        _ => biquad_coeffs_lpf(freq_hz, sr, q),
    };

    AudioOutput::Mono(biquad_filter(samples, b0, b1, b2, a1, a2))
}

fn variants_e001() -> Vec<HashMap<String, Value>> {
    vec![
        params! { "filter_type" => "lpf", "freq_hz" => 400.0, "Q" => 0.707 },
        params! { "filter_type" => "lpf", "freq_hz" => 2000.0, "Q" => 5.0 },
        params! { "filter_type" => "hpf", "freq_hz" => 800.0, "Q" => 1.0 },
        params! { "filter_type" => "bpf", "freq_hz" => 1200.0, "Q" => 8.0 },
        params! { "filter_type" => "notch", "freq_hz" => 3000.0, "Q" => 10.0 },
        params! { "filter_type" => "peak", "freq_hz" => 2500.0, "Q" => 2.0, "gain_db" => 12.0 },
        params! { "filter_type" => "peak", "freq_hz" => 500.0, "Q" => 4.0, "gain_db" => -18.0 },
        params! { "filter_type" => "lpf", "freq_hz" => 150.0, "Q" => 15.0 },
    ]
}

// ---------------------------------------------------------------------------
// E002 -- State Variable Filter
// ---------------------------------------------------------------------------

/// State variable filter kernel.
/// Returns (lp, hp, bp, notch) output vectors.
fn svf_kernel(samples: &[f32], f_coeff: f32, q_inv: f32) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = samples.len();
    let mut lp_out = vec![0.0f32; n];
    let mut hp_out = vec![0.0f32; n];
    let mut bp_out = vec![0.0f32; n];
    let mut notch_out = vec![0.0f32; n];

    let mut lp = 0.0f32;
    let mut bp = 0.0f32;

    for i in 0..n {
        let x = samples[i];
        let hp = x - lp - q_inv * bp;
        bp = bp + f_coeff * hp;
        lp = lp + f_coeff * bp;
        let notch = x - q_inv * bp;

        // Clamp to avoid instability
        bp = bp.clamp(-10.0, 10.0);
        lp = lp.clamp(-10.0, 10.0);

        lp_out[i] = lp;
        hp_out[i] = hp;
        bp_out[i] = bp;
        notch_out[i] = notch;
    }

    (lp_out, hp_out, bp_out, notch_out)
}

fn process_e002(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let cutoff_hz = pf(params, "cutoff_hz", 1000.0).clamp(20.0, sr as f32 * 0.499);
    let q = pf(params, "Q", 2.0).clamp(0.5, 30.0);
    let output_type = ps(params, "output_type", "lp");

    let f_coeff = 2.0 * (PI * cutoff_hz / sr as f32).sin();
    let q_inv = 1.0 / q;

    let (lp, hp, bp, notch) = svf_kernel(samples, f_coeff, q_inv);

    let out = match output_type {
        "hp" => hp,
        "bp" => bp,
        "notch" => notch,
        _ => lp,
    };

    AudioOutput::Mono(out)
}

fn variants_e002() -> Vec<HashMap<String, Value>> {
    vec![
        params! { "cutoff_hz" => 500.0, "Q" => 2.0, "output_type" => "lp" },
        params! { "cutoff_hz" => 3000.0, "Q" => 8.0, "output_type" => "lp" },
        params! { "cutoff_hz" => 1000.0, "Q" => 1.0, "output_type" => "hp" },
        params! { "cutoff_hz" => 1500.0, "Q" => 12.0, "output_type" => "bp" },
        params! { "cutoff_hz" => 2000.0, "Q" => 15.0, "output_type" => "notch" },
        params! { "cutoff_hz" => 200.0, "Q" => 25.0, "output_type" => "bp" },
    ]
}

// ---------------------------------------------------------------------------
// E003 -- Moog Ladder Filter
// ---------------------------------------------------------------------------

/// 4-cascaded one-pole stages with tanh nonlinearity and feedback.
fn moog_ladder_kernel(samples: &[f32], cutoff_norm: f32, resonance: f32) -> Vec<f32> {
    let n = samples.len();
    let mut out = vec![0.0f32; n];
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;

    for i in 0..n {
        let mut x = samples[i];
        // Feedback: subtract resonance * stage-4 output
        x -= resonance * s3;
        // tanh saturation on input
        x = x.tanh();

        // Stage 1
        s0 += cutoff_norm * (x - s0);
        // Stage 2
        s1 += cutoff_norm * (s0 - s1);
        // Stage 3
        s2 += cutoff_norm * (s1 - s2);
        // Stage 4
        s3 += cutoff_norm * (s2 - s3);

        out[i] = s3;
    }

    out
}

fn process_e003(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let cutoff_hz = pf(params, "cutoff_hz", 1000.0).clamp(20.0, sr as f32 * 0.499);
    let resonance = pf(params, "resonance", 2.0).clamp(0.0, 4.0);

    let cutoff_norm = 2.0 * (PI * cutoff_hz / sr as f32).sin();

    AudioOutput::Mono(moog_ladder_kernel(samples, cutoff_norm, resonance))
}

fn variants_e003() -> Vec<HashMap<String, Value>> {
    vec![
        params! { "cutoff_hz" => 300.0, "resonance" => 0.0 },
        params! { "cutoff_hz" => 800.0, "resonance" => 2.0 },
        params! { "cutoff_hz" => 1500.0, "resonance" => 3.5 },
        params! { "cutoff_hz" => 5000.0, "resonance" => 1.0 },
        params! { "cutoff_hz" => 200.0, "resonance" => 3.8 },
        params! { "cutoff_hz" => 3000.0, "resonance" => 0.5 },
    ]
}

// ---------------------------------------------------------------------------
// E004 -- Comb Filter
// ---------------------------------------------------------------------------

/// Feedback comb filter: y[n] = x[n] + g * y[n - delay].
fn comb_feedback_kernel(samples: &[f32], delay_samples: usize, g: f32) -> Vec<f32> {
    let n = samples.len();
    let mut out = vec![0.0f32; n];
    let buf_len = delay_samples.max(1);
    let mut buf = vec![0.0f32; buf_len];
    let mut write_pos: usize = 0;

    for i in 0..n {
        let read_pos = (write_pos + buf_len - delay_samples) % buf_len;
        let delayed = buf[read_pos];
        let mut y = samples[i] + g * delayed;
        // Clamp to prevent blowup
        y = y.clamp(-10.0, 10.0);
        buf[write_pos] = y;
        out[i] = y;
        write_pos = (write_pos + 1) % buf_len;
    }

    out
}

/// Feedforward comb filter: y[n] = x[n] + g * x[n - delay].
fn comb_feedforward_kernel(samples: &[f32], delay_samples: usize, g: f32) -> Vec<f32> {
    let n = samples.len();
    let mut out = vec![0.0f32; n];
    let buf_len = delay_samples.max(1);
    let mut buf = vec![0.0f32; buf_len];
    let mut write_pos: usize = 0;

    for i in 0..n {
        let read_pos = (write_pos + buf_len - delay_samples) % buf_len;
        let delayed = buf[read_pos];
        out[i] = samples[i] + g * delayed;
        buf[write_pos] = samples[i];
        write_pos = (write_pos + 1) % buf_len;
    }

    out
}

fn process_e004(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let freq_hz = pf(params, "freq_hz", 200.0).clamp(50.0, 2000.0);
    let g = pf(params, "g", 0.7).clamp(-0.99, 0.99);
    let mode = ps(params, "mode", "feedback");

    let delay_samples = (sr as f32 / freq_hz).round() as usize;
    let delay_samples = delay_samples.max(1);

    let out = match mode {
        "feedforward" => comb_feedforward_kernel(samples, delay_samples, g),
        _ => comb_feedback_kernel(samples, delay_samples, g),
    };

    AudioOutput::Mono(out)
}

fn variants_e004() -> Vec<HashMap<String, Value>> {
    vec![
        params! { "freq_hz" => 200.0, "g" => 0.8, "mode" => "feedback" },
        params! { "freq_hz" => 500.0, "g" => 0.6, "mode" => "feedback" },
        params! { "freq_hz" => 100.0, "g" => 0.9, "mode" => "feedback" },
        params! { "freq_hz" => 300.0, "g" => 0.5, "mode" => "feedforward" },
        params! { "freq_hz" => 800.0, "g" => -0.7, "mode" => "feedforward" },
        params! { "freq_hz" => 150.0, "g" => -0.85, "mode" => "feedback" },
    ]
}

// ---------------------------------------------------------------------------
// E005 -- Formant Filter
// ---------------------------------------------------------------------------

/// Vowel formant frequencies (F1, F2, F3) -- standard male voice approximations.
fn formant_freqs(vowel: &str) -> (f32, f32, f32) {
    match vowel {
        "a" => (800.0, 1200.0, 2800.0),
        "e" => (400.0, 2250.0, 2800.0),
        "i" => (280.0, 2600.0, 3500.0),
        "o" => (500.0, 800.0, 2800.0),
        "u" => (320.0, 800.0, 2500.0),
        _ => (800.0, 1200.0, 2800.0), // default to 'a'
    }
}

/// Relative amplitudes for each formant (linear).
fn formant_amps(vowel: &str) -> (f32, f32, f32) {
    match vowel {
        "a" => (1.0, 0.5, 0.25),
        "e" => (1.0, 0.6, 0.3),
        "i" => (1.0, 0.4, 0.2),
        "o" => (1.0, 0.5, 0.2),
        "u" => (1.0, 0.4, 0.15),
        _ => (1.0, 0.5, 0.25), // default to 'a'
    }
}

fn process_e005(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let vowel = ps(params, "vowel", "a");
    let q = pf(params, "Q", 10.0).clamp(5.0, 20.0);

    let (f1, f2, f3) = formant_freqs(vowel);
    let (a1, a2, a3) = formant_amps(vowel);

    let n = samples.len();
    let mut out = vec![0.0f32; n];

    let freqs = [f1, f2, f3];
    let amps = [a1, a2, a3];

    for (&freq, &amp) in freqs.iter().zip(amps.iter()) {
        let (b0, b1, b2, ca1, ca2) = biquad_coeffs_bpf(freq.clamp(20.0, sr as f32 * 0.499), sr, q);
        let filtered = biquad_filter(samples, b0, b1, b2, ca1, ca2);
        for j in 0..n {
            out[j] += amp * filtered[j];
        }
    }

    AudioOutput::Mono(out)
}

fn variants_e005() -> Vec<HashMap<String, Value>> {
    vec![
        params! { "vowel" => "a", "Q" => 10.0 },
        params! { "vowel" => "e", "Q" => 10.0 },
        params! { "vowel" => "i", "Q" => 12.0 },
        params! { "vowel" => "o", "Q" => 8.0 },
        params! { "vowel" => "u", "Q" => 15.0 },
        params! { "vowel" => "a", "Q" => 20.0 },
        params! { "vowel" => "i", "Q" => 5.0 },
    ]
}

// ---------------------------------------------------------------------------
// E006 -- Vowel Morph
// ---------------------------------------------------------------------------

fn process_e006(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let vowel_from = ps(params, "vowel_from", "a");
    let vowel_to = ps(params, "vowel_to", "o");
    let morph_rate_hz = pf(params, "morph_rate_hz", 0.5).clamp(0.1, 5.0);
    let q = pf(params, "Q", 10.0).clamp(5.0, 20.0);

    let (ff1, ff2, ff3) = formant_freqs(vowel_from);
    let (af1, af2, af3) = formant_amps(vowel_from);
    let (tf1, tf2, tf3) = formant_freqs(vowel_to);
    let (at1, at2, at3) = formant_amps(vowel_to);

    let n = samples.len();

    // Run 6 bandpass filters (3 per vowel)
    let from_freqs = [ff1, ff2, ff3];
    let from_amps = [af1, af2, af3];
    let to_freqs = [tf1, tf2, tf3];
    let to_amps = [at1, at2, at3];

    let mut bp_from = [vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]];
    let mut bp_to = [vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]];

    for k in 0..3 {
        let freq = from_freqs[k].clamp(20.0, sr as f32 * 0.499);
        let (b0, b1, b2, a1, a2) = biquad_coeffs_bpf(freq, sr, q);
        bp_from[k] = biquad_filter(samples, b0, b1, b2, a1, a2);
    }

    for k in 0..3 {
        let freq = to_freqs[k].clamp(20.0, sr as f32 * 0.499);
        let (b0, b1, b2, a1, a2) = biquad_coeffs_bpf(freq, sr, q);
        bp_to[k] = biquad_filter(samples, b0, b1, b2, a1, a2);
    }

    // Crossfade with LFO
    let mut out = vec![0.0f32; n];
    let phase_inc = morph_rate_hz / sr as f32;
    let mut phase = 0.0f32;

    for i in 0..n {
        // Raised cosine morph: 0..1
        let morph = 0.5 * (1.0 - (2.0 * PI * phase).cos());

        let val_from = from_amps[0] * bp_from[0][i]
            + from_amps[1] * bp_from[1][i]
            + from_amps[2] * bp_from[2][i];

        let val_to = to_amps[0] * bp_to[0][i]
            + to_amps[1] * bp_to[1][i]
            + to_amps[2] * bp_to[2][i];

        out[i] = (1.0 - morph) * val_from + morph * val_to;

        phase += phase_inc;
        if phase >= 1.0 {
            phase -= 1.0;
        }
    }

    AudioOutput::Mono(out)
}

fn variants_e006() -> Vec<HashMap<String, Value>> {
    vec![
        params! { "vowel_from" => "a", "vowel_to" => "o", "morph_rate_hz" => 0.5, "Q" => 10.0 },
        params! { "vowel_from" => "e", "vowel_to" => "i", "morph_rate_hz" => 1.0, "Q" => 12.0 },
        params! { "vowel_from" => "a", "vowel_to" => "u", "morph_rate_hz" => 0.2, "Q" => 8.0 },
        params! { "vowel_from" => "i", "vowel_to" => "o", "morph_rate_hz" => 2.0, "Q" => 15.0 },
        params! { "vowel_from" => "u", "vowel_to" => "e", "morph_rate_hz" => 0.3, "Q" => 6.0 },
        params! { "vowel_from" => "o", "vowel_to" => "a", "morph_rate_hz" => 4.0, "Q" => 10.0 },
    ]
}

// ---------------------------------------------------------------------------
// E007 -- Auto-Wah
// ---------------------------------------------------------------------------

/// Envelope follower drives SVF bandpass cutoff.
fn autowah_kernel(
    samples: &[f32],
    min_freq: f32,
    max_freq: f32,
    sensitivity: f32,
    q: f32,
    sr: u32,
) -> Vec<f32> {
    let n = samples.len();
    let mut out = vec![0.0f32; n];

    let mut env = 0.0f32;
    let attack = 0.001f32;
    let release = 0.9995f32;

    let mut lp = 0.0f32;
    let mut bp = 0.0f32;
    let q_inv = 1.0 / q;

    let pi_over_sr = PI / sr as f32;

    for i in 0..n {
        let x = samples[i];

        // Envelope follower
        let abs_x = x.abs();
        if abs_x > env {
            env += (1.0 - attack) * (abs_x - env);
        } else {
            env *= release;
        }

        // Map envelope to cutoff frequency
        let mut cutoff = min_freq + sensitivity * env * (max_freq - min_freq) / 0.01;
        cutoff = cutoff.clamp(min_freq, max_freq);

        // SVF coefficients
        let f_coeff = (2.0 * (cutoff * pi_over_sr).sin()).min(1.8);

        // SVF step
        let hp = x - lp - q_inv * bp;
        bp += f_coeff * hp;
        lp += f_coeff * bp;

        // Clamp
        bp = bp.clamp(-10.0, 10.0);
        lp = lp.clamp(-10.0, 10.0);

        out[i] = bp; // bandpass output for wah character
    }

    out
}

fn process_e007(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let min_freq = pf(params, "min_freq", 200.0).clamp(100.0, 500.0);
    let max_freq = pf(params, "max_freq", 3000.0).clamp(1000.0, 8000.0);
    let sensitivity = pf(params, "sensitivity", 0.01).clamp(0.001, 0.05);
    let q = pf(params, "Q", 5.0).clamp(2.0, 15.0);

    AudioOutput::Mono(autowah_kernel(samples, min_freq, max_freq, sensitivity, q, sr))
}

fn variants_e007() -> Vec<HashMap<String, Value>> {
    vec![
        params! { "min_freq" => 200.0, "max_freq" => 3000.0, "sensitivity" => 0.01, "Q" => 5.0 },
        params! { "min_freq" => 100.0, "max_freq" => 5000.0, "sensitivity" => 0.03, "Q" => 8.0 },
        params! { "min_freq" => 300.0, "max_freq" => 2000.0, "sensitivity" => 0.005, "Q" => 3.0 },
        params! { "min_freq" => 150.0, "max_freq" => 8000.0, "sensitivity" => 0.02, "Q" => 12.0 },
        params! { "min_freq" => 400.0, "max_freq" => 4000.0, "sensitivity" => 0.04, "Q" => 10.0 },
    ]
}

// ---------------------------------------------------------------------------
// E008 -- Resonant Filter Sweep
// ---------------------------------------------------------------------------

/// Sweep a resonant SVF from start_freq to end_freq (log-linear) over the signal duration.
fn resonant_sweep_kernel(
    samples: &[f32],
    start_freq: f32,
    end_freq: f32,
    q: f32,
    sr: u32,
    use_bpf: bool,
) -> Vec<f32> {
    let n = samples.len();
    let mut out = vec![0.0f32; n];

    let log_start = start_freq.ln();
    let log_end = end_freq.ln();
    let q_inv = 1.0 / q;
    let pi_over_sr = PI / sr as f32;

    let mut lp = 0.0f32;
    let mut bp = 0.0f32;

    let n_f = n as f32;

    for i in 0..n {
        // Log-linear sweep
        let t = i as f32 / n_f;
        let log_freq = log_start + t * (log_end - log_start);
        let cutoff = log_freq.exp();

        let f_coeff = (2.0 * (cutoff * pi_over_sr).sin()).min(1.8);

        let x = samples[i];
        let hp = x - lp - q_inv * bp;
        bp += f_coeff * hp;
        lp += f_coeff * bp;

        // Clamp
        bp = bp.clamp(-10.0, 10.0);
        lp = lp.clamp(-10.0, 10.0);

        out[i] = if use_bpf { bp } else { lp };
    }

    out
}

fn process_e008(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let start_freq = pf(params, "start_freq", 200.0).clamp(100.0, 1000.0);
    let end_freq = pf(params, "end_freq", 6000.0).clamp(2000.0, 12000.0f32.min(sr as f32 * 0.499));
    let q = pf(params, "Q", 10.0).clamp(5.0, 30.0);
    let filter_type = ps(params, "filter_type", "lpf");
    let use_bpf = filter_type == "bpf";

    AudioOutput::Mono(resonant_sweep_kernel(
        samples, start_freq, end_freq, q, sr, use_bpf,
    ))
}

fn variants_e008() -> Vec<HashMap<String, Value>> {
    vec![
        params! { "start_freq" => 200.0, "end_freq" => 6000.0, "Q" => 10.0, "filter_type" => "lpf" },
        params! { "start_freq" => 100.0, "end_freq" => 10000.0, "Q" => 20.0, "filter_type" => "lpf" },
        params! { "start_freq" => 500.0, "end_freq" => 4000.0, "Q" => 5.0, "filter_type" => "bpf" },
        params! { "start_freq" => 300.0, "end_freq" => 8000.0, "Q" => 25.0, "filter_type" => "bpf" },
        params! { "start_freq" => 800.0, "end_freq" => 3000.0, "Q" => 15.0, "filter_type" => "lpf" },
        params! { "start_freq" => 150.0, "end_freq" => 12000.0, "Q" => 8.0, "filter_type" => "lpf" },
    ]
}

// ---------------------------------------------------------------------------
// E009 -- Multi-Mode Filter Crossfade
// ---------------------------------------------------------------------------

/// LFO morphs between LP, BP, HP outputs of an SVF.
///
/// LFO phase 0..0.33: LP -> BP
/// LFO phase 0.33..0.67: BP -> HP
/// LFO phase 0.67..1.0: HP -> LP
fn multimode_crossfade_kernel(
    samples: &[f32],
    cutoff_hz: f32,
    q: f32,
    morph_rate_hz: f32,
    sr: u32,
) -> Vec<f32> {
    let n = samples.len();
    let mut out = vec![0.0f32; n];

    let f_coeff = (2.0 * (PI * cutoff_hz / sr as f32).sin()).min(1.8);
    let q_inv = 1.0 / q;

    let mut lp = 0.0f32;
    let mut bp = 0.0f32;

    let phase_inc = morph_rate_hz / sr as f32;
    let mut phase = 0.0f32;
    let third = 1.0f32 / 3.0;

    for i in 0..n {
        let x = samples[i];

        // SVF step
        let hp = x - lp - q_inv * bp;
        bp += f_coeff * hp;
        lp += f_coeff * bp;

        // Clamp
        bp = bp.clamp(-10.0, 10.0);
        lp = lp.clamp(-10.0, 10.0);

        // Three-way crossfade based on LFO phase
        out[i] = if phase < third {
            // LP -> BP
            let t = phase / third;
            (1.0 - t) * lp + t * bp
        } else if phase < 2.0 * third {
            // BP -> HP
            let t = (phase - third) / third;
            (1.0 - t) * bp + t * hp
        } else {
            // HP -> LP
            let t = (phase - 2.0 * third) / third;
            (1.0 - t) * hp + t * lp
        };

        phase += phase_inc;
        if phase >= 1.0 {
            phase -= 1.0;
        }
    }

    out
}

fn process_e009(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let morph_rate_hz = pf(params, "morph_rate_hz", 0.5).clamp(0.1, 5.0);
    let q = pf(params, "Q", 5.0).clamp(2.0, 15.0);
    let cutoff_hz = pf(params, "cutoff_hz", 1500.0).clamp(500.0, 5000.0f32.min(sr as f32 * 0.499));

    AudioOutput::Mono(multimode_crossfade_kernel(
        samples, cutoff_hz, q, morph_rate_hz, sr,
    ))
}

fn variants_e009() -> Vec<HashMap<String, Value>> {
    vec![
        params! { "morph_rate_hz" => 0.3, "Q" => 5.0, "cutoff_hz" => 1500.0 },
        params! { "morph_rate_hz" => 1.0, "Q" => 8.0, "cutoff_hz" => 2000.0 },
        params! { "morph_rate_hz" => 0.1, "Q" => 3.0, "cutoff_hz" => 800.0 },
        params! { "morph_rate_hz" => 2.5, "Q" => 12.0, "cutoff_hz" => 3000.0 },
        params! { "morph_rate_hz" => 5.0, "Q" => 10.0, "cutoff_hz" => 1000.0 },
        params! { "morph_rate_hz" => 0.5, "Q" => 15.0, "cutoff_hz" => 4000.0 },
    ]
}

// ---------------------------------------------------------------------------
// E010 -- Cascade of Detuned Resonators
// ---------------------------------------------------------------------------

fn process_e010(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let base_freq = pf(params, "base_freq", 500.0).clamp(200.0, 2000.0);
    let num_resonators = pi(params, "num_resonators", 5).clamp(3, 8) as usize;
    let detune = pf(params, "detune", 0.05).clamp(0.01, 0.1);
    let q = pf(params, "Q", 20.0).clamp(10.0, 50.0);

    let n = samples.len();
    let mut out = vec![0.0f32; n];

    // Spread resonators symmetrically around base_freq
    let gain = 1.0 / num_resonators as f32;

    for r in 0..num_resonators {
        let offset = if num_resonators == 1 {
            0.0
        } else {
            -detune + 2.0 * detune * r as f32 / (num_resonators - 1) as f32
        };
        let freq = (base_freq * (1.0 + offset)).clamp(20.0, sr as f32 * 0.499);
        let (b0, b1, b2, a1, a2) = biquad_coeffs_bpf(freq, sr, q);
        let filtered = biquad_filter(samples, b0, b1, b2, a1, a2);
        for j in 0..n {
            out[j] += gain * filtered[j];
        }
    }

    AudioOutput::Mono(out)
}

fn variants_e010() -> Vec<HashMap<String, Value>> {
    vec![
        params! { "base_freq" => 500.0, "num_resonators" => 5, "detune" => 0.05, "Q" => 20.0 },
        params! { "base_freq" => 300.0, "num_resonators" => 8, "detune" => 0.08, "Q" => 30.0 },
        params! { "base_freq" => 1000.0, "num_resonators" => 3, "detune" => 0.02, "Q" => 40.0 },
        params! { "base_freq" => 800.0, "num_resonators" => 6, "detune" => 0.1, "Q" => 15.0 },
        params! { "base_freq" => 200.0, "num_resonators" => 4, "detune" => 0.03, "Q" => 50.0 },
        params! { "base_freq" => 1500.0, "num_resonators" => 7, "detune" => 0.06, "Q" => 25.0 },
    ]
}

// ---------------------------------------------------------------------------
// E011 -- Allpass Lattice Filter
// ---------------------------------------------------------------------------

/// Lattice allpass filter: each stage is a first-order allpass with
/// coupled forward/backward paths.
fn allpass_lattice_kernel(samples: &[f32], coeffs: &[f32], num_stages: usize) -> Vec<f32> {
    let n = samples.len();
    let mut out = vec![0.0f32; n];
    let mut state = vec![0.0f32; num_stages];

    for i in 0..n {
        let mut x = samples[i];

        for s in 0..num_stages {
            let k = coeffs[s];
            // Lattice allpass: y = k*x + state; state_next = x - k*y
            let y = k * x + state[s];
            state[s] = x - k * y;
            x = y;
        }

        out[i] = x;
    }

    out
}

fn process_e011(samples: &[f32], _sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let num_stages = pi(params, "num_stages", 6).clamp(2, 16) as usize;
    let base_coeff = pf(params, "base_coeff", 0.5);
    let spread = pf(params, "spread", 0.3);
    let mix = pf(params, "mix", 0.7);

    let mut coeffs = vec![0.0f32; num_stages];
    for s in 0..num_stages {
        let frac = s as f32 / (num_stages - 1).max(1) as f32;
        coeffs[s] = (base_coeff + spread * (frac - 0.5)).clamp(-0.99, 0.99);
    }

    let filtered = allpass_lattice_kernel(samples, &coeffs, num_stages);

    // Mix original with allpass output to create notches
    let n = samples.len();
    let mut out = vec![0.0f32; n];
    let dry = 1.0 - mix;
    for i in 0..n {
        out[i] = dry * samples[i] + mix * filtered[i];
    }

    AudioOutput::Mono(out)
}

fn variants_e011() -> Vec<HashMap<String, Value>> {
    vec![
        params! { "num_stages" => 4, "base_coeff" => 0.3, "spread" => 0.2, "mix" => 0.7 },
        params! { "num_stages" => 6, "base_coeff" => 0.5, "spread" => 0.3, "mix" => 0.7 },
        params! { "num_stages" => 8, "base_coeff" => 0.7, "spread" => 0.4, "mix" => 0.8 },
        params! { "num_stages" => 12, "base_coeff" => 0.4, "spread" => 0.5, "mix" => 0.6 },
        params! { "num_stages" => 16, "base_coeff" => 0.6, "spread" => 0.2, "mix" => 0.9 },
        params! { "num_stages" => 6, "base_coeff" => -0.5, "spread" => 0.6, "mix" => 0.7 },
    ]
}

// ---------------------------------------------------------------------------
// E012 -- Pitch-Tracking Resonator
// ---------------------------------------------------------------------------

/// Simple autocorrelation pitch detection on a frame.
fn autocorr_pitch(frame: &[f32], sr: u32, min_freq: f32, max_freq: f32) -> f32 {
    let n = frame.len();
    let min_lag = (sr as f32 / max_freq) as usize;
    let min_lag = min_lag.max(1);
    let max_lag = (sr as f32 / min_freq) as usize;
    let max_lag = max_lag.min(n - 1);

    if min_lag >= max_lag {
        return 0.0;
    }

    let mut best_lag = min_lag;
    let mut best_corr = -1.0f32;

    for lag in min_lag..=max_lag {
        let mut corr = 0.0f32;
        let mut energy = 0.0f32;
        for j in 0..(n - lag) {
            corr += frame[j] * frame[j + lag];
            energy += frame[j] * frame[j];
        }
        if energy > 1e-10 {
            let norm_corr = corr / energy;
            if norm_corr > best_corr {
                best_corr = norm_corr;
                best_lag = lag;
            }
        }
    }

    if best_corr < 0.2 {
        return 0.0; // no clear pitch
    }
    sr as f32 / best_lag as f32
}

/// Apply a bank of biquad bandpass resonators at given frequencies.
fn resonator_bank_kernel(
    samples: &[f32],
    sr: u32,
    freqs: &[f32],
    num_harmonics: usize,
    q: f64,
    wet: f32,
) -> Vec<f32> {
    let n = samples.len();
    let mut resonated = vec![0.0f32; n];
    let nyq = sr as f32 * 0.499;

    for h in 0..num_harmonics {
        let freq = freqs[h];
        if freq < 20.0 || freq > nyq {
            continue;
        }

        // Biquad BPF coefficients (matching Python: computed in f64 for precision)
        let w0 = 2.0 * std::f64::consts::PI * freq as f64 / sr as f64;
        let alpha = w0.sin() / (2.0 * q);
        let a0 = 1.0 + alpha;
        let b0 = (alpha / a0) as f32;
        let b1 = 0.0f32;
        let b2 = (-alpha / a0) as f32;
        let a1 = (-2.0 * w0.cos() / a0) as f32;
        let a2 = ((1.0 - alpha) / a0) as f32;

        let gain = 1.0 / num_harmonics as f32;

        let mut z1 = 0.0f32;
        let mut z2 = 0.0f32;
        for i in 0..n {
            let x = samples[i];
            let y = b0 * x + z1;
            z1 = b1 * x - a1 * y + z2;
            z2 = b2 * x - a2 * y;
            resonated[i] += y * gain;
        }
    }

    // Wet/dry mix
    let dry = 1.0 - wet;
    let mut result = vec![0.0f32; n];
    for i in 0..n {
        result[i] = dry * samples[i] + wet * resonated[i];
    }
    result
}

fn process_e012(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let num_harmonics = pi(params, "num_harmonics", 8).clamp(1, 32) as usize;
    let q = pf(params, "Q", 15.0) as f64;
    let wet = pf(params, "wet", 0.6);

    let n = samples.len();

    // Detect pitch from a chunk near the beginning (50ms)
    let analysis_len = n.min((0.05 * sr as f32) as usize);
    let analysis_start = (n / 4).min(n.saturating_sub(analysis_len));
    let frame = &samples[analysis_start..analysis_start + analysis_len];

    let mut fundamental = autocorr_pitch(frame, sr, 50.0, 2000.0);
    if fundamental < 20.0 {
        fundamental = 220.0; // fallback to A3
    }

    // Build harmonic frequencies
    let mut freqs = vec![0.0f32; num_harmonics];
    for h in 0..num_harmonics {
        freqs[h] = fundamental * (h + 1) as f32;
    }

    AudioOutput::Mono(resonator_bank_kernel(
        samples,
        sr,
        &freqs,
        num_harmonics,
        q,
        wet,
    ))
}

fn variants_e012() -> Vec<HashMap<String, Value>> {
    vec![
        params! { "num_harmonics" => 4, "Q" => 10.0, "wet" => 0.4 },
        params! { "num_harmonics" => 8, "Q" => 15.0, "wet" => 0.5 },
        params! { "num_harmonics" => 12, "Q" => 20.0, "wet" => 0.6 },
        params! { "num_harmonics" => 6, "Q" => 30.0, "wet" => 0.7 },
        params! { "num_harmonics" => 16, "Q" => 10.0, "wet" => 0.5 },
        params! { "num_harmonics" => 8, "Q" => 50.0, "wet" => 0.8 },
    ]
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

pub fn register() -> Vec<EffectEntry> {
    vec![
        EffectEntry {
            id: "E001",
            process: process_e001,
            variants: variants_e001,
            category: "filter",
        },
        EffectEntry {
            id: "E002",
            process: process_e002,
            variants: variants_e002,
            category: "filter",
        },
        EffectEntry {
            id: "E003",
            process: process_e003,
            variants: variants_e003,
            category: "filter",
        },
        EffectEntry {
            id: "E004",
            process: process_e004,
            variants: variants_e004,
            category: "filter",
        },
        EffectEntry {
            id: "E005",
            process: process_e005,
            variants: variants_e005,
            category: "filter",
        },
        EffectEntry {
            id: "E006",
            process: process_e006,
            variants: variants_e006,
            category: "filter",
        },
        EffectEntry {
            id: "E007",
            process: process_e007,
            variants: variants_e007,
            category: "filter",
        },
        EffectEntry {
            id: "E008",
            process: process_e008,
            variants: variants_e008,
            category: "filter",
        },
        EffectEntry {
            id: "E009",
            process: process_e009,
            variants: variants_e009,
            category: "filter",
        },
        EffectEntry {
            id: "E010",
            process: process_e010,
            variants: variants_e010,
            category: "filter",
        },
        EffectEntry {
            id: "E011",
            process: process_e011,
            variants: variants_e011,
            category: "filter",
        },
        EffectEntry {
            id: "E012",
            process: process_e012,
            variants: variants_e012,
            category: "filter",
        },
    ]
}
