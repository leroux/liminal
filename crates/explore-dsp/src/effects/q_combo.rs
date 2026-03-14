//! Q-series: Meta-effects that chain other effects (Q001-Q005).
//!
//! These combo effects compose primitive DSP operations inline rather than
//! dispatching through the registry, keeping them self-contained.

use std::collections::HashMap;
use serde_json::Value;
use crate::{AudioOutput, EffectEntry, pf, pi, ps, params};
use crate::primitives::*;

// ===========================================================================
// Inline DSP helpers (ported from sibling modules to keep combos standalone)
// ===========================================================================

/// Soft clip via tanh (D002).
fn dsp_soft_clip_tanh(samples: &[f32], drive: f32) -> Vec<f32> {
    samples.iter().map(|&s| (s * drive).tanh()).collect()
}

/// Simple feedback delay (A001).
fn dsp_simple_delay(samples: &[f32], sr: u32, delay_ms: f32, feedback: f32) -> Vec<f32> {
    let delay_samples = ((delay_ms * sr as f32 / 1000.0).round() as usize).max(1);
    let n = samples.len();
    let buf_len = (delay_samples + 1).max(1);
    let mut buf = vec![0.0f32; buf_len];
    let mut out = vec![0.0f32; n];
    let mut write_pos: usize = 0;
    for i in 0..n {
        let read_pos = (write_pos + buf_len - delay_samples) % buf_len;
        let y = samples[i] + feedback * buf[read_pos];
        buf[write_pos] = y;
        out[i] = y;
        write_pos = (write_pos + 1) % buf_len;
    }
    out
}

/// Bit crusher (D008).
fn dsp_bit_crusher(samples: &[f32], bits: i32) -> Vec<f32> {
    let levels = (1_i64 << bits) as f32;
    samples
        .iter()
        .map(|&s| (s * levels + 0.5).floor() / levels)
        .collect()
}

/// Tube saturation (D003).
fn dsp_tube_saturation(samples: &[f32], drive: f32, asymmetry: f32) -> Vec<f32> {
    let d_pos = drive * (1.0 + asymmetry);
    let d_neg = drive * (1.0 - asymmetry);
    samples
        .iter()
        .map(|&x| {
            if x >= 0.0 {
                1.0 - (-d_pos * x).exp()
            } else {
                -(1.0 - (d_neg * x).exp())
            }
        })
        .collect()
}

/// Foldback distortion (D004).
fn dsp_foldback(samples: &[f32], threshold: f32, pre_gain: f32) -> Vec<f32> {
    samples
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
        .collect()
}

/// Sample rate reduction (D009).
fn dsp_sample_rate_reduction(samples: &[f32], sr: u32, target_sr: u32) -> Vec<f32> {
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
    out
}

/// Slew rate limiter (D010).
fn dsp_slew_rate_limiter(samples: &[f32], max_slew: f32) -> Vec<f32> {
    if samples.is_empty() {
        return vec![];
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
    out
}

/// Tape delay emulation (A005).
fn dsp_tape_delay(
    samples: &[f32],
    sr: u32,
    delay_ms: f32,
    feedback: f32,
    wow_rate_hz: f32,
) -> Vec<f32> {
    let wow_depth: f32 = 3.0;
    let filter_cutoff: f32 = 3500.0;
    let delay_samples = ((delay_ms * sr as f32 / 1000.0).round() as usize).max(1);

    let dt = 1.0 / sr as f32;
    let rc = 1.0 / (2.0 * std::f32::consts::PI * filter_cutoff);
    let filter_coeff = (-dt / rc).exp();

    let n = samples.len();
    let buf_len = delay_samples + wow_depth as usize + 4;
    let mut buf = vec![0.0f32; buf_len];
    let mut out = vec![0.0f32; n];
    let mut write_pos: usize = 0;
    let mut lp_state = 0.0f32;
    let two_pi = 2.0 * std::f32::consts::PI;

    for i in 0..n {
        let phase = two_pi * wow_rate_hz * i as f32 / sr as f32;
        let modulation = wow_depth * phase.sin();
        let frac_delay = delay_samples as f32 + modulation;
        let int_delay = frac_delay as usize;
        let frac = frac_delay - int_delay as f32;

        let read_pos_0 = (write_pos + buf_len - int_delay) % buf_len;
        let read_pos_1 = (write_pos + buf_len - int_delay - 1) % buf_len;
        let delayed = (1.0 - frac) * buf[read_pos_0] + frac * buf[read_pos_1];

        lp_state = filter_coeff * lp_state + (1.0 - filter_coeff) * delayed;

        let y = samples[i] + feedback * lp_state;
        buf[write_pos] = y;
        out[i] = y;
        write_pos = (write_pos + 1) % buf_len;
    }
    out
}

/// Allpass delay diffuser (A007).
fn dsp_allpass_diffuser(
    samples: &[f32],
    sr: u32,
    num_stages: usize,
    delay_range_ms: f32,
    g: f32,
) -> Vec<f32> {
    let primes = [2usize, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
    let mut delays = vec![0usize; num_stages];
    for i in 0..num_stages {
        let prime_idx = i % primes.len();
        let denom_idx = (num_stages - 1).min(primes.len() - 1);
        let frac = primes[prime_idx] as f32 / primes[denom_idx] as f32;
        let d_ms = 1.0 + frac * (delay_range_ms - 1.0);
        delays[i] = ((d_ms * sr as f32 / 1000.0).round() as usize).max(1);
    }

    let n = samples.len();
    let mut current = samples.to_vec();

    for s in 0..num_stages {
        let d = delays[s];
        let buf_len = (d + 1).max(1);
        let mut x_buf = vec![0.0f32; buf_len];
        let mut y_buf = vec![0.0f32; buf_len];
        let mut stage_out = vec![0.0f32; n];
        let mut write_pos: usize = 0;

        for i in 0..n {
            let read_pos = (write_pos + buf_len - d) % buf_len;
            let x_delayed = x_buf[read_pos];
            let y_delayed = y_buf[read_pos];
            let y_val = -g * current[i] + x_delayed + g * y_delayed;
            x_buf[write_pos] = current[i];
            y_buf[write_pos] = y_val;
            stage_out[i] = y_val;
            write_pos = (write_pos + 1) % buf_len;
        }

        current = stage_out;
    }
    current
}

/// Stutter / Retrigger (A010).
fn dsp_stutter(
    samples: &[f32],
    sr: u32,
    window_ms: f32,
    repeats: usize,
    decay: f32,
) -> Vec<f32> {
    let window_size = ((window_ms * sr as f32 / 1000.0).round() as usize).max(1);
    let n = samples.len();
    let mut out = vec![0.0f32; n];
    let num_windows = n / window_size;

    for w in 0..num_windows {
        let src_start = w * window_size;
        let out_pos = w * window_size;

        // Capture the window
        let mut window = vec![0.0f32; window_size];
        for j in 0..window_size {
            if src_start + j < n {
                window[j] = samples[src_start + j];
            }
        }

        // Place original
        for j in 0..window_size {
            if out_pos + j < n {
                out[out_pos + j] = window[j];
            }
        }

        // Place repeats within the same window duration
        let total_repeat_space = window_size;
        let repeat_len = (total_repeat_space / repeats.max(1)).max(1);

        let mut gain = 1.0f32;
        for r in 1..repeats {
            gain *= decay;
            for j in 0..repeat_len {
                let src_j = j; // no pitch drift for combo use
                if src_j >= window_size {
                    break;
                }
                let out_idx = out_pos + r * repeat_len + j;
                if out_idx < n {
                    out[out_idx] += gain * window[src_j];
                }
            }
        }
    }

    // Copy remaining samples
    let remaining_start = num_windows * window_size;
    for i in remaining_start..n {
        out[i] = samples[i];
    }
    out
}

/// Reverse chunks (A012).
fn dsp_reverse_chunks(samples: &[f32], sr: u32, chunk_ms: f32, reverse_probability: f32) -> Vec<f32> {
    let n = samples.len();
    let chunk_size = ((chunk_ms * sr as f32 / 1000.0).round() as usize).max(1);
    let num_chunks = n / chunk_size;
    let fade_samples = ((0.005 * sr as f32) as usize).min(chunk_size / 4);

    let mut out = vec![0.0f32; n];

    // Simple LCG for deterministic random (matching Python's seed=123)
    let mut rng_state: u64 = 123;

    for c in 0..num_chunks {
        let start = c * chunk_size;
        let mut chunk: Vec<f32> = samples[start..start + chunk_size].to_vec();

        // Pseudo-random decision
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let rand_val = ((rng_state >> 33) & 0x7FFFFFFF) as f32 / 2147483647.0;
        let should_reverse = rand_val < reverse_probability;

        if should_reverse {
            chunk.reverse();
        }

        // Apply crossfade
        for j in 0..fade_samples {
            let fade = j as f32 / fade_samples as f32;
            chunk[j] *= fade;
            chunk[chunk_size - 1 - j] *= fade;
        }

        out[start..start + chunk_size].copy_from_slice(&chunk);
    }

    // Copy remainder
    let remainder_start = num_chunks * chunk_size;
    if remainder_start < n {
        out[remainder_start..n].copy_from_slice(&samples[remainder_start..n]);
    }
    out
}

/// Spectral blur (H002) -- uniform filter across frequency bins on STFT magnitudes.
fn dsp_spectral_blur(samples: &[f32], sr: u32, blur_width: usize) -> Vec<f32> {
    let fft_size = 2048;
    let hop_size = fft_size / 4;

    let frames = crate::stft::stft(samples, fft_size, hop_size);
    if frames.is_empty() {
        return samples.to_vec();
    }

    let mut result_frames = Vec::with_capacity(frames.len());
    let blur_w = blur_width.max(1);

    for frame in &frames {
        let num_bins = frame.len();
        let mag: Vec<f32> = frame.iter().map(|c| c.norm()).collect();
        let phase: Vec<f32> = frame.iter().map(|c| c.im.atan2(c.re)).collect();

        // Uniform filter along frequency axis
        let mut blurred_mag = vec![0.0f32; num_bins];
        for b in 0..num_bins {
            let lo = if b >= blur_w / 2 { b - blur_w / 2 } else { 0 };
            let hi = (b + blur_w / 2 + 1).min(num_bins);
            let count = (hi - lo) as f32;
            let sum: f32 = mag[lo..hi].iter().sum();
            blurred_mag[b] = sum / count;
        }

        // Reconstruct complex frame
        let new_frame: Vec<num_complex::Complex<f32>> = blurred_mag
            .iter()
            .zip(phase.iter())
            .map(|(&m, &p)| num_complex::Complex::new(m * p.cos(), m * p.sin()))
            .collect();
        result_frames.push(new_frame);
    }

    crate::stft::istft(&result_frames, fft_size, hop_size, Some(samples.len()))
}

/// Phase randomization (H005) -- blend original phases with random phases.
fn dsp_phase_randomization(samples: &[f32], _sr: u32, amount: f32) -> Vec<f32> {
    let fft_size = 2048;
    let hop_size = fft_size / 4;

    let frames = crate::stft::stft(samples, fft_size, hop_size);
    if frames.is_empty() {
        return samples.to_vec();
    }

    let mut rng = Lcg::new(42);
    let pi_val = std::f32::consts::PI;

    let mut result_frames = Vec::with_capacity(frames.len());
    for frame in &frames {
        let num_bins = frame.len();
        let mag: Vec<f32> = frame.iter().map(|c| c.norm()).collect();
        let orig_phase: Vec<f32> = frame.iter().map(|c| c.im.atan2(c.re)).collect();

        let mut new_frame = Vec::with_capacity(num_bins);
        for b in 0..num_bins {
            let random_phase = rng.next_bipolar() * pi_val;
            let blended_phase = (1.0 - amount) * orig_phase[b] + amount * random_phase;
            new_frame.push(num_complex::Complex::new(
                mag[b] * blended_phase.cos(),
                mag[b] * blended_phase.sin(),
            ));
        }
        result_frames.push(new_frame);
    }

    crate::stft::istft(&result_frames, fft_size, hop_size, Some(samples.len()))
}

/// Spectral freeze (H001) -- freeze magnitudes at a chosen position.
fn dsp_spectral_freeze(samples: &[f32], _sr: u32, freeze_position: f32) -> Vec<f32> {
    let fft_size = 2048;
    let hop_size = fft_size / 4;

    let frames = crate::stft::stft(samples, fft_size, hop_size);
    if frames.is_empty() {
        return samples.to_vec();
    }

    let num_frames = frames.len();
    let freeze_frame = (freeze_position.clamp(0.0, 1.0) * (num_frames - 1) as f32) as usize;
    let frozen_mag: Vec<f32> = frames[freeze_frame].iter().map(|c| c.norm()).collect();

    let mut result_frames = Vec::with_capacity(num_frames);
    for frame in &frames {
        let phase: Vec<f32> = frame.iter().map(|c| c.im.atan2(c.re)).collect();
        let new_frame: Vec<num_complex::Complex<f32>> = frozen_mag
            .iter()
            .zip(phase.iter())
            .map(|(&m, &p)| num_complex::Complex::new(m * p.cos(), m * p.sin()))
            .collect();
        result_frames.push(new_frame);
    }

    crate::stft::istft(&result_frames, fft_size, hop_size, Some(samples.len()))
}

/// Spectral shift (H004) -- shift all frequency bins up or down.
fn dsp_spectral_shift(samples: &[f32], _sr: u32, shift_bins: i32) -> Vec<f32> {
    let fft_size = 2048;
    let hop_size = fft_size / 4;

    let frames = crate::stft::stft(samples, fft_size, hop_size);
    if frames.is_empty() {
        return samples.to_vec();
    }

    let mut result_frames = Vec::with_capacity(frames.len());
    for frame in &frames {
        let num_bins = frame.len();
        let mut new_frame = vec![num_complex::Complex::new(0.0f32, 0.0f32); num_bins];
        for b in 0..num_bins {
            let src = b as i32 - shift_bins;
            if src >= 0 && (src as usize) < num_bins {
                new_frame[b] = frame[src as usize];
            }
        }
        result_frames.push(new_frame);
    }

    crate::stft::istft(&result_frames, fft_size, hop_size, Some(samples.len()))
}

/// Ensure mono f32 output from AudioOutput (utility for chaining).
fn to_mono(output: AudioOutput) -> Vec<f32> {
    match output {
        AudioOutput::Mono(v) => v,
        AudioOutput::Stereo(v) => v.iter().map(|s| s[0]).collect(),
    }
}

// ---------------------------------------------------------------------------
// Q001 -- Serial Chain (2 effects)
// ---------------------------------------------------------------------------

fn process_q001(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    // Read which effects to chain (string IDs for variant documentation,
    // but the actual DSP is dispatched below via known effect identifiers).
    let effect_a = ps(params, "effect_a", "d002_soft_clipping_tanh");
    let effect_b = ps(params, "effect_b", "a001_simple_delay");

    // Run effect A
    let out_a = run_inline_effect(effect_a, samples, sr, params, "params_a");

    // Run effect B on the output of A
    let out_b = run_inline_effect(effect_b, &out_a, sr, params, "params_b");

    AudioOutput::Mono(out_b)
}

fn variants_q001() -> Vec<HashMap<String, Value>> {
    vec![
        // Distortion into delay -- classic guitar pedal order
        params! {
            "effect_a" => "d002_soft_clipping_tanh",
            "params_a" => serde_json::json!({"drive": 5.0}),
            "effect_b" => "a001_simple_delay",
            "params_b" => serde_json::json!({"delay_ms": 300, "feedback": 0.5})
        },
        // Delay into distortion -- each echo gets more distorted
        params! {
            "effect_a" => "a001_simple_delay",
            "params_a" => serde_json::json!({"delay_ms": 200, "feedback": 0.6}),
            "effect_b" => "d002_soft_clipping_tanh",
            "params_b" => serde_json::json!({"drive": 8.0})
        },
        // Bit crusher into spectral blur
        params! {
            "effect_a" => "d008_bit_crusher",
            "params_a" => serde_json::json!({"bits": 6}),
            "effect_b" => "h002_spectral_blur",
            "params_b" => serde_json::json!({"blur_width": 20})
        },
        // Phase randomization into tube saturation
        params! {
            "effect_a" => "h005_phase_randomization",
            "params_a" => serde_json::json!({"amount": 0.5}),
            "effect_b" => "d003_tube_saturation",
            "params_b" => serde_json::json!({"drive": 4.0, "asymmetry": 0.2})
        },
        // Reverse chunks into allpass diffuser
        params! {
            "effect_a" => "a012_reverse_chunks",
            "params_a" => serde_json::json!({"chunk_ms": 100, "reverse_probability": 0.7}),
            "effect_b" => "a007_allpass_diffuser",
            "params_b" => serde_json::json!({"num_stages": 8, "delay_range_ms": 20, "g": 0.6})
        },
    ]
}

// ---------------------------------------------------------------------------
// Q002 -- Serial Chain (3 effects)
// ---------------------------------------------------------------------------

fn process_q002(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let effect_a = ps(params, "effect_a", "d002_soft_clipping_tanh");
    let effect_b = ps(params, "effect_b", "a001_simple_delay");
    let effect_c = ps(params, "effect_c", "h005_phase_randomization");

    let out_a = run_inline_effect(effect_a, samples, sr, params, "params_a");
    let out_b = run_inline_effect(effect_b, &out_a, sr, params, "params_b");
    let out_c = run_inline_effect(effect_c, &out_b, sr, params, "params_c");

    AudioOutput::Mono(out_c)
}

fn variants_q002() -> Vec<HashMap<String, Value>> {
    vec![
        // Distortion -> delay -> spectral blur (ambient distortion wash)
        params! {
            "effect_a" => "d002_soft_clipping_tanh",
            "params_a" => serde_json::json!({"drive": 6.0}),
            "effect_b" => "a001_simple_delay",
            "params_b" => serde_json::json!({"delay_ms": 400, "feedback": 0.6}),
            "effect_c" => "h002_spectral_blur",
            "params_c" => serde_json::json!({"blur_width": 15})
        },
        // Bit crush -> foldback -> sample rate reduce (lo-fi chain)
        params! {
            "effect_a" => "d008_bit_crusher",
            "params_a" => serde_json::json!({"bits": 8}),
            "effect_b" => "d004_foldback_distortion",
            "params_b" => serde_json::json!({"threshold": 0.5, "pre_gain": 3.0}),
            "effect_c" => "d009_sample_rate_reduction",
            "params_c" => serde_json::json!({"target_sr": 8000})
        },
        // Stutter -> reverse chunks -> spectral freeze
        params! {
            "effect_a" => "a010_stutter",
            "params_a" => serde_json::json!({"window_ms": 80, "repeats": 4, "decay": 0.9}),
            "effect_b" => "a012_reverse_chunks",
            "params_b" => serde_json::json!({"chunk_ms": 150, "reverse_probability": 0.6}),
            "effect_c" => "h001_spectral_freeze",
            "params_c" => serde_json::json!({"freeze_position": 0.5})
        },
        // Allpass diffuser -> tape delay -> tube saturation (warm ambient)
        params! {
            "effect_a" => "a007_allpass_diffuser",
            "params_a" => serde_json::json!({"num_stages": 6, "delay_range_ms": 15, "g": 0.55}),
            "effect_b" => "a005_tape_delay",
            "params_b" => serde_json::json!({"delay_ms": 300, "feedback": 0.5, "wow_rate_hz": 1.0}),
            "effect_c" => "d003_tube_saturation",
            "params_c" => serde_json::json!({"drive": 2.0, "asymmetry": 0.1})
        },
        // Phase randomize -> spectral shift -> slew rate limit (alien textures)
        params! {
            "effect_a" => "h005_phase_randomization",
            "params_a" => serde_json::json!({"amount": 0.7}),
            "effect_b" => "h004_spectral_shift",
            "params_b" => serde_json::json!({"shift_bins": 15}),
            "effect_c" => "d010_slew_rate_limiter",
            "params_c" => serde_json::json!({"max_slew": 0.02})
        },
    ]
}

// ---------------------------------------------------------------------------
// Q003 -- Parallel Mix (2 effects)
// ---------------------------------------------------------------------------

fn process_q003(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let effect_a = ps(params, "effect_a", "d002_soft_clipping_tanh");
    let effect_b = ps(params, "effect_b", "a001_simple_delay");
    let mix_a = pf(params, "mix_a", 0.5);
    let mix_b = pf(params, "mix_b", 0.5);

    let out_a = run_inline_effect(effect_a, samples, sr, params, "params_a");
    let out_b = run_inline_effect(effect_b, samples, sr, params, "params_b");

    // Match lengths
    let min_len = out_a.len().min(out_b.len());
    let out: Vec<f32> = (0..min_len)
        .map(|i| mix_a * out_a[i] + mix_b * out_b[i])
        .collect();

    AudioOutput::Mono(out)
}

fn variants_q003() -> Vec<HashMap<String, Value>> {
    vec![
        // Distortion + clean delay blended equally
        params! {
            "effect_a" => "d002_soft_clipping_tanh",
            "params_a" => serde_json::json!({"drive": 5.0}),
            "effect_b" => "a001_simple_delay",
            "params_b" => serde_json::json!({"delay_ms": 250, "feedback": 0.5}),
            "mix_a" => 0.5,
            "mix_b" => 0.5
        },
        // Heavy distortion blended with spectral blur
        params! {
            "effect_a" => "d004_foldback_distortion",
            "params_a" => serde_json::json!({"threshold": 0.3, "pre_gain": 10.0}),
            "effect_b" => "h002_spectral_blur",
            "params_b" => serde_json::json!({"blur_width": 30}),
            "mix_a" => 0.3,
            "mix_b" => 0.7
        },
        // Phase randomization + spectral freeze layered
        params! {
            "effect_a" => "h005_phase_randomization",
            "params_a" => serde_json::json!({"amount": 0.8}),
            "effect_b" => "h001_spectral_freeze",
            "params_b" => serde_json::json!({"freeze_position": 0.4}),
            "mix_a" => 0.6,
            "mix_b" => 0.4
        },
        // Two delays at different times for rhythmic pattern
        params! {
            "effect_a" => "a001_simple_delay",
            "params_a" => serde_json::json!({"delay_ms": 150, "feedback": 0.4}),
            "effect_b" => "a001_simple_delay",
            "params_b" => serde_json::json!({"delay_ms": 375, "feedback": 0.5}),
            "mix_a" => 0.5,
            "mix_b" => 0.5
        },
        // Bit crusher + tube saturation for textural layering
        params! {
            "effect_a" => "d008_bit_crusher",
            "params_a" => serde_json::json!({"bits": 4}),
            "effect_b" => "d003_tube_saturation",
            "params_b" => serde_json::json!({"drive": 6.0, "asymmetry": 0.3}),
            "mix_a" => 0.4,
            "mix_b" => 0.6
        },
    ]
}

// ---------------------------------------------------------------------------
// Q004 -- Wet/Dry Crossfade Over Time
// ---------------------------------------------------------------------------

fn process_q004(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let effect_id = ps(params, "effect_id", "h005_phase_randomization");
    let direction = ps(params, "direction", "dry_to_wet");

    let wet = run_inline_effect(effect_id, samples, sr, params, "effect_params");

    let dry = samples;
    let min_len = dry.len().min(wet.len());

    let out: Vec<f32> = (0..min_len)
        .map(|i| {
            let t = i as f32 / (min_len - 1).max(1) as f32;
            let wet_amount = if direction == "dry_to_wet" { t } else { 1.0 - t };
            (1.0 - wet_amount) * dry[i] + wet_amount * wet[i]
        })
        .collect();

    AudioOutput::Mono(out)
}

fn variants_q004() -> Vec<HashMap<String, Value>> {
    vec![
        // Fade into phase randomization
        params! {
            "effect_id" => "h005_phase_randomization",
            "effect_params" => serde_json::json!({"amount": 1.0}),
            "direction" => "dry_to_wet"
        },
        // Fade out of spectral freeze
        params! {
            "effect_id" => "h001_spectral_freeze",
            "effect_params" => serde_json::json!({"freeze_position": 0.3}),
            "direction" => "wet_to_dry"
        },
        // Gradually introduce distortion
        params! {
            "effect_id" => "d002_soft_clipping_tanh",
            "effect_params" => serde_json::json!({"drive": 10.0}),
            "direction" => "dry_to_wet"
        },
        // Fade out of bit crusher
        params! {
            "effect_id" => "d008_bit_crusher",
            "effect_params" => serde_json::json!({"bits": 4}),
            "direction" => "wet_to_dry"
        },
        // Fade into spectral blur
        params! {
            "effect_id" => "h002_spectral_blur",
            "effect_params" => serde_json::json!({"blur_width": 40}),
            "direction" => "dry_to_wet"
        },
    ]
}

// ---------------------------------------------------------------------------
// Q005 -- Feedback Through Effect
// ---------------------------------------------------------------------------

fn process_q005(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let effect_id = ps(params, "effect_id", "d002_soft_clipping_tanh");
    let block_size_ms = pf(params, "block_size_ms", 50.0).clamp(10.0, 100.0);
    let feedback = pf(params, "feedback", 0.4).clamp(0.1, 0.8);

    let block_size = ((block_size_ms * sr as f32 / 1000.0) as usize).max(1);
    let n = samples.len();
    let mut out = vec![0.0f32; n];
    let mut prev_block = vec![0.0f32; block_size];

    let num_blocks = (n + block_size - 1) / block_size;

    // Extract sub-params for the inner effect
    let sub_params = extract_sub_params(params, "effect_params");

    for b in 0..num_blocks {
        let start = b * block_size;
        let end = (start + block_size).min(n);
        let current_len = end - start;

        // Process previous block through the effect
        let effect_out = run_inline_effect_with_params(effect_id, &prev_block, sr, &sub_params);

        // Trim/pad to block_size
        let effect_block: Vec<f32> = if effect_out.len() < block_size {
            let mut padded = vec![0.0f32; block_size];
            for i in 0..effect_out.len() {
                padded[i] = effect_out[i];
            }
            padded
        } else {
            effect_out[..block_size].to_vec()
        };

        // y = x + feedback * effect(y_prev_block)
        for i in 0..current_len {
            out[start + i] = samples[start + i] + feedback * effect_block[i];
        }

        // Store current output block as prev_block for next iteration
        prev_block = vec![0.0f32; block_size];
        for i in 0..current_len {
            prev_block[i] = out[start + i];
        }
    }

    AudioOutput::Mono(out)
}

fn variants_q005() -> Vec<HashMap<String, Value>> {
    vec![
        // Feedback through soft clip -- self-limiting feedback distortion
        params! {
            "effect_id" => "d002_soft_clipping_tanh",
            "effect_params" => serde_json::json!({"drive": 3.0}),
            "block_size_ms" => 50,
            "feedback" => 0.4
        },
        // Feedback through bit crusher -- increasingly degraded echoes
        params! {
            "effect_id" => "d008_bit_crusher",
            "effect_params" => serde_json::json!({"bits": 6}),
            "block_size_ms" => 80,
            "feedback" => 0.5
        },
        // Feedback through slew limiter -- smoothing recirculation
        params! {
            "effect_id" => "d010_slew_rate_limiter",
            "effect_params" => serde_json::json!({"max_slew": 0.05}),
            "block_size_ms" => 30,
            "feedback" => 0.6
        },
        // Feedback through foldback -- chaotic harmonics buildup
        params! {
            "effect_id" => "d004_foldback_distortion",
            "effect_params" => serde_json::json!({"threshold": 0.5, "pre_gain": 3.0}),
            "block_size_ms" => 40,
            "feedback" => 0.3
        },
        // Higher feedback for longer sustain
        params! {
            "effect_id" => "d002_soft_clipping_tanh",
            "effect_params" => serde_json::json!({"drive": 5.0}),
            "block_size_ms" => 100,
            "feedback" => 0.7
        },
    ]
}

// ===========================================================================
// Inline effect dispatcher
// ===========================================================================

/// Extract a nested params object from a parent params map.
fn extract_sub_params(params: &HashMap<String, Value>, key: &str) -> HashMap<String, Value> {
    params
        .get(key)
        .and_then(|v| v.as_object())
        .map(|obj| {
            obj.iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect()
        })
        .unwrap_or_default()
}

/// Run an effect by its string ID, reading sub-parameters from the given key.
fn run_inline_effect(
    effect_id: &str,
    samples: &[f32],
    sr: u32,
    params: &HashMap<String, Value>,
    sub_params_key: &str,
) -> Vec<f32> {
    let sub = extract_sub_params(params, sub_params_key);
    run_inline_effect_with_params(effect_id, samples, sr, &sub)
}

/// Run an effect by its string ID with an explicit params map.
fn run_inline_effect_with_params(
    effect_id: &str,
    samples: &[f32],
    sr: u32,
    sub: &HashMap<String, Value>,
) -> Vec<f32> {
    match effect_id {
        "d002_soft_clipping_tanh" => {
            let drive = pf(sub, "drive", 3.0);
            dsp_soft_clip_tanh(samples, drive)
        }
        "a001_simple_delay" => {
            let delay_ms = pf(sub, "delay_ms", 300.0);
            let feedback = pf(sub, "feedback", 0.5);
            dsp_simple_delay(samples, sr, delay_ms, feedback)
        }
        "d008_bit_crusher" => {
            let bits = pi(sub, "bits", 8);
            dsp_bit_crusher(samples, bits)
        }
        "d003_tube_saturation" => {
            let drive = pf(sub, "drive", 3.0);
            let asymmetry = pf(sub, "asymmetry", 0.1);
            dsp_tube_saturation(samples, drive, asymmetry)
        }
        "d004_foldback_distortion" => {
            let threshold = pf(sub, "threshold", 0.5);
            let pre_gain = pf(sub, "pre_gain", 5.0);
            dsp_foldback(samples, threshold, pre_gain)
        }
        "d009_sample_rate_reduction" => {
            let target_sr = pi(sub, "target_sr", 8000) as u32;
            dsp_sample_rate_reduction(samples, sr, target_sr)
        }
        "d010_slew_rate_limiter" => {
            let max_slew = pf(sub, "max_slew", 0.05);
            dsp_slew_rate_limiter(samples, max_slew)
        }
        "a005_tape_delay" => {
            let delay_ms = pf(sub, "delay_ms", 300.0);
            let feedback = pf(sub, "feedback", 0.5);
            let wow_rate_hz = pf(sub, "wow_rate_hz", 1.5);
            dsp_tape_delay(samples, sr, delay_ms, feedback, wow_rate_hz)
        }
        "a007_allpass_diffuser" => {
            let num_stages = pi(sub, "num_stages", 6) as usize;
            let delay_range_ms = pf(sub, "delay_range_ms", 20.0);
            let g = pf(sub, "g", 0.6);
            dsp_allpass_diffuser(samples, sr, num_stages, delay_range_ms, g)
        }
        "a010_stutter" => {
            let window_ms = pf(sub, "window_ms", 80.0);
            let repeats = pi(sub, "repeats", 8) as usize;
            let decay = pf(sub, "decay", 0.9);
            dsp_stutter(samples, sr, window_ms, repeats, decay)
        }
        "a012_reverse_chunks" => {
            let chunk_ms = pf(sub, "chunk_ms", 150.0);
            let reverse_probability = pf(sub, "reverse_probability", 0.5);
            dsp_reverse_chunks(samples, sr, chunk_ms, reverse_probability)
        }
        "h001_spectral_freeze" => {
            let freeze_position = pf(sub, "freeze_position", 0.3);
            dsp_spectral_freeze(samples, sr, freeze_position)
        }
        "h002_spectral_blur" => {
            let blur_width = pi(sub, "blur_width", 10) as usize;
            dsp_spectral_blur(samples, sr, blur_width)
        }
        "h004_spectral_shift" => {
            let shift_bins = pi(sub, "shift_bins", 10);
            dsp_spectral_shift(samples, sr, shift_bins)
        }
        "h005_phase_randomization" => {
            let amount = pf(sub, "amount", 0.5);
            dsp_phase_randomization(samples, sr, amount)
        }
        // Fallback: pass through unchanged
        _ => samples.to_vec(),
    }
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

pub fn register() -> Vec<EffectEntry> {
    vec![
        EffectEntry {
            id: "Q001",
            process: process_q001,
            variants: variants_q001,
            category: "combo",
        },
        EffectEntry {
            id: "Q002",
            process: process_q002,
            variants: variants_q002,
            category: "combo",
        },
        EffectEntry {
            id: "Q003",
            process: process_q003,
            variants: variants_q003,
            category: "combo",
        },
        EffectEntry {
            id: "Q004",
            process: process_q004,
            variants: variants_q004,
            category: "combo",
        },
        EffectEntry {
            id: "Q005",
            process: process_q005,
            variants: variants_q005,
            category: "combo",
        },
    ]
}
