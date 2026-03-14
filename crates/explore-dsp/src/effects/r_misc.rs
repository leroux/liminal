//! R-series: Miscellaneous / experimental effects (R001-R016).

use std::collections::HashMap;
use serde_json::Value;
use crate::{AudioOutput, EffectEntry, pf, pi, ps, pu, params};
use crate::primitives::*;
use crate::stft::{stft, istft};
use realfft::RealFftPlanner;
use num_complex::Complex;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand::Rng;

// ---------------------------------------------------------------------------
// R001 -- Audio Fractalization
// ---------------------------------------------------------------------------

fn process_r001(samples: &[f32], _sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let num_scales = (pi(params, "num_scales", 3) as usize).clamp(2, 5);
    let scale_ratio = pf(params, "scale_ratio", 0.5).clamp(0.3, 0.7);
    let amplitude_per_scale = pf(params, "amplitude_per_scale", 0.5).clamp(0.3, 0.8);

    let n = samples.len();
    let mut out: Vec<f32> = samples.to_vec();

    for s in 1..num_scales {
        // Each scale compresses the entire signal into a shorter version
        let compressed_len = (n as f64 * (scale_ratio as f64).powi(s as i32)) as usize;
        let compressed_len = compressed_len.max(1);

        // Resample by nearest-neighbor
        let mut compressed = vec![0.0f32; compressed_len];
        for i in 0..compressed_len {
            let idx = ((i as f64 / (compressed_len - 1).max(1) as f64) * (n - 1) as f64) as usize;
            compressed[i] = samples[idx.min(n - 1)];
        }

        // Tile the compressed signal to fill the original length
        let gain = amplitude_per_scale.powi(s as i32);
        for i in 0..n {
            out[i] += gain * compressed[i % compressed_len];
        }
    }

    // Normalize to prevent clipping
    let peak = out.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    if peak > 0.0 {
        let in_peak = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        if peak > in_peak {
            let scale = in_peak / peak;
            for s in out.iter_mut() {
                *s *= scale;
            }
        }
    }

    AudioOutput::Mono(out)
}

fn variants_r001() -> Vec<HashMap<String, Value>> {
    vec![
        params!("num_scales" => 2, "scale_ratio" => 0.5, "amplitude_per_scale" => 0.6),
        params!("num_scales" => 3, "scale_ratio" => 0.5, "amplitude_per_scale" => 0.5),
        params!("num_scales" => 4, "scale_ratio" => 0.4, "amplitude_per_scale" => 0.4),
        params!("num_scales" => 5, "scale_ratio" => 0.3, "amplitude_per_scale" => 0.5),
        params!("num_scales" => 3, "scale_ratio" => 0.7, "amplitude_per_scale" => 0.7),
    ]
}

// ---------------------------------------------------------------------------
// R002 -- Spectral Peak Tracking + Resynthesis
// ---------------------------------------------------------------------------

fn process_r002(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let num_peaks = (pi(params, "num_peaks", 20) as usize).clamp(5, 50);
    let vibrato_depth = pf(params, "vibrato_depth", 0.0);
    let vibrato_rate = pf(params, "vibrato_rate", 0.0);

    let fft_size = 2048;
    let hop_size = 512;
    let two_pi = 2.0 * std::f32::consts::PI;

    let frames = stft(samples, fft_size, hop_size);
    let num_frames = frames.len();
    let num_bins = frames[0].len();
    let bin_freq = sr as f32 / fft_size as f32;

    let n = samples.len();
    let mut out = vec![0.0f32; n];

    for frame_idx in 0..num_frames {
        let mag: Vec<f32> = frames[frame_idx].iter().map(|c| c.norm()).collect();

        // Find top N peaks by sorting indices by magnitude
        let k = num_peaks.min(num_bins);
        let mut indices: Vec<usize> = (0..num_bins).collect();
        indices.sort_unstable_by(|&a, &b| mag[b].partial_cmp(&mag[a]).unwrap());
        let peak_bins = &indices[..k];

        let frame_start = frame_idx * hop_size;
        let frame_end = (frame_start + hop_size).min(n);
        let frame_len = frame_end - frame_start;

        for &b in peak_bins {
            let freq = b as f32 * bin_freq;
            let amp = mag[b] / fft_size as f32;
            if freq < 20.0 || amp < 1e-8 {
                continue;
            }
            let phase_offset = frames[frame_idx][b].arg();

            for j in 0..frame_len {
                let t = (frame_start + j) as f32 / sr as f32;
                let vib = vibrato_depth * (two_pi * vibrato_rate * t).sin();
                out[frame_start + j] +=
                    amp * (two_pi * freq * (1.0 + vib) * t + phase_offset).sin();
            }
        }
    }

    // Normalize by RMS matching
    let in_rms = (samples.iter().map(|s| s * s).sum::<f32>() / n.max(1) as f32).sqrt() + 1e-10;
    let out_rms = (out.iter().map(|s| s * s).sum::<f32>() / n.max(1) as f32).sqrt() + 1e-10;
    let scale = in_rms / out_rms;
    for s in out.iter_mut() {
        *s *= scale;
    }

    AudioOutput::Mono(out)
}

fn variants_r002() -> Vec<HashMap<String, Value>> {
    vec![
        params!("num_peaks" => 5, "vibrato_depth" => 0, "vibrato_rate" => 0),
        params!("num_peaks" => 10, "vibrato_depth" => 0, "vibrato_rate" => 0),
        params!("num_peaks" => 20, "vibrato_depth" => 0, "vibrato_rate" => 0),
        params!("num_peaks" => 50, "vibrato_depth" => 0, "vibrato_rate" => 0),
        params!("num_peaks" => 15, "vibrato_depth" => 1.0, "vibrato_rate" => 3.0),
        params!("num_peaks" => 20, "vibrato_depth" => 2.0, "vibrato_rate" => 5.0),
    ]
}

// ---------------------------------------------------------------------------
// R003 -- Autoregressive Model Resynthesis (LPC)
// ---------------------------------------------------------------------------

/// Levinson-Durbin recursion for LPC coefficients from autocorrelation.
fn levinson_durbin(r: &[f64], order: usize) -> (Vec<f64>, f64) {
    let mut a = vec![0.0f64; order + 1];
    let mut e = r[0];
    a[0] = 1.0;

    for i in 1..=order {
        let mut acc = 0.0f64;
        for j in 1..i {
            acc += a[j] * r[i - j];
        }
        let k = -(r[i] + acc) / (e + 1e-30);

        let a_old = a.clone();
        a[i] = k;
        for j in 1..i {
            a[j] = a_old[j] + k * a_old[i - j];
        }

        e *= 1.0 - k * k;
        if e <= 0.0 {
            e = 1e-10;
        }
    }

    (a[1..].to_vec(), e)
}

fn process_r003(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let lpc_order = (pi(params, "lpc_order", 20) as usize).clamp(10, 50);
    let modification = ps(params, "modification", "scale");
    let mod_amount = pf(params, "mod_amount", 1.2) as f64;

    let x: Vec<f64> = samples.iter().map(|&s| s as f64).collect();
    let n = x.len();

    let frame_size = 1024;
    let hop = 512;
    let num_frames = if n >= frame_size { (n - frame_size) / hop + 1 } else { 1 };

    let mut out = vec![0.0f64; n];
    let window: Vec<f64> = (0..frame_size)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (frame_size - 1) as f64).cos()))
        .collect();

    let mut rng = ChaCha8Rng::seed_from_u64(42);

    for f in 0..num_frames {
        let start = f * hop;
        let end = (start + frame_size).min(n);
        let frame_len = end - start;

        let mut frame = vec![0.0f64; frame_size];
        for i in 0..frame_len {
            frame[i] = x[start + i] * window[i];
        }

        // Autocorrelation
        let mut r = vec![0.0f64; lpc_order + 1];
        for lag in 0..=lpc_order {
            let mut sum = 0.0f64;
            for i in 0..frame_size - lag {
                sum += frame[i] * frame[i + lag];
            }
            r[lag] = sum;
        }

        if r[0] < 1e-10 {
            // Silent frame -- pass through
            for i in 0..frame_len {
                out[start + i] += frame[i];
            }
            continue;
        }

        let (mut coeffs, error) = levinson_durbin(&r, lpc_order);

        // Modify coefficients
        match modification {
            "scale" => {
                for c in coeffs.iter_mut() {
                    *c *= mod_amount;
                }
            }
            "jitter" => {
                for c in coeffs.iter_mut() {
                    let jitter: f64 = (rng.gen::<f64>() * 2.0 - 1.0) * (mod_amount - 1.0) * 0.1;
                    *c += jitter;
                }
            }
            _ => {
                for c in coeffs.iter_mut() {
                    *c *= mod_amount;
                }
            }
        }

        // Stability check: scale coefficients if any root magnitude >= 1
        // Quick heuristic: check if sum of absolute coefficients >= 1
        let abs_sum: f64 = coeffs.iter().map(|c| c.abs()).sum();
        if abs_sum >= 1.0 {
            let scale_factor = 0.98 / abs_sum;
            for (idx, c) in coeffs.iter_mut().enumerate() {
                *c *= scale_factor.powi((idx + 1) as i32);
            }
        }

        // Excitation: white noise scaled by prediction error
        let excitation_gain = error.max(1e-10).sqrt();

        // All-pole synthesis: y[n] = excitation[n] - sum(a[k]*y[n-k])
        let mut synth = vec![0.0f64; frame_size];
        for i in 0..frame_size {
            let exc = (rng.gen::<f64>() * 2.0 - 1.0) * excitation_gain;
            let mut val = exc;
            for k in 0..lpc_order.min(i) {
                val -= coeffs[k] * synth[i - 1 - k];
            }
            // Clip to prevent runaway
            val = val.clamp(-10.0, 10.0);
            synth[i] = val;
        }

        for i in 0..frame_size {
            synth[i] *= window[i];
        }
        let out_end = (start + frame_size).min(n);
        for i in start..out_end {
            out[i] += synth[i - start];
        }
    }

    // Replace NaN/inf with zeros
    for s in out.iter_mut() {
        if !s.is_finite() {
            *s = 0.0;
        }
    }

    // Normalize by RMS matching
    let in_rms = (x.iter().map(|s| s * s).sum::<f64>() / n.max(1) as f64).sqrt() + 1e-10;
    let out_rms = (out.iter().map(|s| s * s).sum::<f64>() / n.max(1) as f64).sqrt() + 1e-10;
    let scale = in_rms / out_rms;
    let result: Vec<f32> = out.iter().map(|&s| (s * scale) as f32).collect();

    AudioOutput::Mono(result)
}

fn variants_r003() -> Vec<HashMap<String, Value>> {
    vec![
        params!("lpc_order" => 10, "modification" => "scale", "mod_amount" => 1.0),
        params!("lpc_order" => 20, "modification" => "scale", "mod_amount" => 1.2),
        params!("lpc_order" => 30, "modification" => "scale", "mod_amount" => 1.5),
        params!("lpc_order" => 50, "modification" => "scale", "mod_amount" => 2.0),
        params!("lpc_order" => 20, "modification" => "jitter", "mod_amount" => 1.2),
        params!("lpc_order" => 30, "modification" => "jitter", "mod_amount" => 1.5),
    ]
}

// ---------------------------------------------------------------------------
// R004 -- Spectral Painting
// ---------------------------------------------------------------------------

fn process_r004(samples: &[f32], _sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let shape_type = ps(params, "shape_type", "gaussian_peaks");
    let evolution_rate = pf(params, "evolution_rate", 0.3);

    let fft_size = 2048;
    let hop_size = 512;
    let two_pi = 2.0 * std::f32::consts::PI;

    let frames = stft(samples, fft_size, hop_size);
    let num_frames = frames.len();
    let num_bins = frames[0].len();

    let bins: Vec<f32> = (0..num_bins).map(|i| i as f32 / num_bins as f32).collect();

    let mut modified_frames = Vec::with_capacity(num_frames);

    for i in 0..num_frames {
        let t = if num_frames > 1 { i as f32 / (num_frames - 1) as f32 } else { 0.0 };
        let phase_offset = t * evolution_rate * two_pi;

        let mask: Vec<f32> = match shape_type {
            "sine_wave" => {
                bins.iter()
                    .map(|&b| 0.5 + 0.5 * (two_pi * 3.0 * b + phase_offset).sin())
                    .collect()
            }
            "gaussian_peaks" => {
                let num_gaussians = 5;
                let mut m = vec![0.0f32; num_bins];
                for g in 0..num_gaussians {
                    let center = (g as f32 / num_gaussians as f32
                        + t * evolution_rate * 0.1)
                        % 1.0;
                    let sigma = 0.05f32;
                    for (idx, &b) in bins.iter().enumerate() {
                        let diff = b - center;
                        m[idx] += (-0.5 * (diff / sigma) * (diff / sigma)).exp();
                    }
                }
                let max_m = m.iter().cloned().fold(0.0f32, f32::max) + 1e-10;
                m.iter().map(|&v| v / max_m).collect()
            }
            "sawtooth" => {
                bins.iter()
                    .map(|&b| (b * 5.0 + phase_offset / two_pi) % 1.0)
                    .collect()
            }
            "random_curve" => {
                // Deterministic pseudo-random per frame using seed based on frame index
                let mut rng = ChaCha8Rng::seed_from_u64((i * 7 + 42) as u64);
                let mut m = vec![0.0f32; num_bins];
                for _h in 0..8 {
                    let freq: f32 = rng.gen::<f32>() * 9.0 + 1.0;
                    let amp: f32 = rng.gen::<f32>() * 0.2 + 0.1;
                    let ph: f32 = rng.gen::<f32>() * two_pi;
                    for (idx, &b) in bins.iter().enumerate() {
                        m[idx] += amp * (two_pi * freq * b + ph + phase_offset).sin();
                    }
                }
                let max_abs = m.iter().map(|v| v.abs()).fold(0.0f32, f32::max) + 1e-10;
                m.iter().map(|&v| 0.5 + 0.5 * v / max_abs).collect()
            }
            _ => vec![1.0f32; num_bins],
        };

        let mut frame = frames[i].clone();
        for b in 0..num_bins {
            let mag = frame[b].norm() * mask[b];
            let phase = frame[b].arg();
            frame[b] = Complex::from_polar(mag, phase);
        }
        modified_frames.push(frame);
    }

    let out = istft(&modified_frames, fft_size, hop_size, Some(samples.len()));
    AudioOutput::Mono(out)
}

fn variants_r004() -> Vec<HashMap<String, Value>> {
    vec![
        params!("shape_type" => "sine_wave", "evolution_rate" => 0.0),
        params!("shape_type" => "sine_wave", "evolution_rate" => 0.5),
        params!("shape_type" => "gaussian_peaks", "evolution_rate" => 0.3),
        params!("shape_type" => "gaussian_peaks", "evolution_rate" => 1.0),
        params!("shape_type" => "sawtooth", "evolution_rate" => 0.2),
        params!("shape_type" => "random_curve", "evolution_rate" => 0.5),
    ]
}

// ---------------------------------------------------------------------------
// R005 -- Phase Gradient Manipulation
// ---------------------------------------------------------------------------

fn process_r005(samples: &[f32], _sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let gradient_scale = pf(params, "gradient_scale", 2.0);

    let fft_size = 2048;
    let hop_size = 512;

    let frames = stft(samples, fft_size, hop_size);
    let num_frames = frames.len();
    let num_bins = frames[0].len();

    let mut modified_frames = Vec::with_capacity(num_frames);

    for i in 0..num_frames {
        let mag: Vec<f32> = frames[i].iter().map(|c| c.norm()).collect();
        let phase: Vec<f32> = frames[i].iter().map(|c| c.arg()).collect();

        // Compute phase differences between adjacent bins (gradient)
        let mut phase_diff = vec![0.0f32; num_bins - 1];
        for b in 0..(num_bins - 1) {
            phase_diff[b] = phase[b + 1] - phase[b];
        }

        // Scale the gradient
        for d in phase_diff.iter_mut() {
            *d *= gradient_scale;
        }

        // Reconstruct phase from scaled gradient
        let mut new_phase = vec![0.0f32; num_bins];
        new_phase[0] = phase[0];
        for b in 1..num_bins {
            new_phase[b] = new_phase[b - 1] + phase_diff[b - 1];
        }

        let mut frame = vec![Complex::new(0.0f32, 0.0); num_bins];
        for b in 0..num_bins {
            frame[b] = Complex::from_polar(mag[b], new_phase[b]);
        }
        modified_frames.push(frame);
    }

    let out = istft(&modified_frames, fft_size, hop_size, Some(samples.len()));
    AudioOutput::Mono(out)
}

fn variants_r005() -> Vec<HashMap<String, Value>> {
    vec![
        params!("gradient_scale" => 0.1),
        params!("gradient_scale" => 0.5),
        params!("gradient_scale" => 1.0),
        params!("gradient_scale" => 2.0),
        params!("gradient_scale" => 3.0),
        params!("gradient_scale" => 5.0),
    ]
}

// ---------------------------------------------------------------------------
// R006 -- Spectral Entropy Gate
// ---------------------------------------------------------------------------

fn process_r006(samples: &[f32], _sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let entropy_threshold = pf(params, "entropy_threshold", 0.7).clamp(0.3, 0.9);
    let mode = ps(params, "mode", "keep_tonal");

    let fft_size = 2048;
    let hop_size = 512;

    let frames = stft(samples, fft_size, hop_size);
    let num_frames = frames.len();
    let num_bins = frames[0].len();

    let max_entropy = (num_bins as f32).log2();

    let mut modified_frames = Vec::with_capacity(num_frames);

    for i in 0..num_frames {
        let mag: Vec<f32> = frames[i].iter().map(|c| c.norm()).collect();

        // Compute spectral probability distribution
        let total: f32 = mag.iter().sum::<f32>() + 1e-10;
        let p: Vec<f32> = mag.iter().map(|&m| m / total).collect();

        // Shannon entropy
        let mut entropy = 0.0f32;
        for &prob in &p {
            if prob > 1e-10 {
                entropy -= prob * prob.log2();
            }
        }
        let normalized_entropy = entropy / max_entropy;

        let mut frame = frames[i].clone();

        let should_zero = match mode {
            "keep_tonal" => normalized_entropy > entropy_threshold,
            _ => normalized_entropy < entropy_threshold, // keep_noisy
        };

        if should_zero {
            for c in frame.iter_mut() {
                *c = Complex::new(0.0, 0.0);
            }
        }

        modified_frames.push(frame);
    }

    let out = istft(&modified_frames, fft_size, hop_size, Some(samples.len()));
    AudioOutput::Mono(out)
}

fn variants_r006() -> Vec<HashMap<String, Value>> {
    vec![
        params!("entropy_threshold" => 0.5, "mode" => "keep_tonal"),
        params!("entropy_threshold" => 0.7, "mode" => "keep_tonal"),
        params!("entropy_threshold" => 0.9, "mode" => "keep_tonal"),
        params!("entropy_threshold" => 0.5, "mode" => "keep_noisy"),
        params!("entropy_threshold" => 0.7, "mode" => "keep_noisy"),
        params!("entropy_threshold" => 0.9, "mode" => "keep_noisy"),
    ]
}

// ---------------------------------------------------------------------------
// R007 -- Wavelet Decomposition (Haar)
// ---------------------------------------------------------------------------

/// Haar wavelet decomposition: returns list of (approx, detail) at each level.
fn haar_decompose(signal: &[f32], num_levels: usize) -> Vec<(Vec<f32>, Vec<f32>)> {
    let mut coeffs = Vec::new();
    let mut current = signal.to_vec();

    for _ in 0..num_levels {
        let n = current.len();
        if n < 2 {
            break;
        }
        let half = n / 2;
        let mut approx = vec![0.0f32; half];
        let mut detail = vec![0.0f32; half];
        for i in 0..half {
            approx[i] = (current[2 * i] + current[2 * i + 1]) * 0.5;
            detail[i] = (current[2 * i] - current[2 * i + 1]) * 0.5;
        }
        coeffs.push((approx.clone(), detail));
        current = approx;
    }

    coeffs
}

/// Reconstruct signal from Haar wavelet coefficients.
fn haar_reconstruct(coeffs: &[(Vec<f32>, Vec<f32>)], original_length: usize) -> Vec<f32> {
    if coeffs.is_empty() {
        return vec![0.0f32; original_length];
    }

    // Start from the deepest level approximation
    let mut current = coeffs.last().unwrap().0.clone();

    for level in (0..coeffs.len()).rev() {
        let detail_level = &coeffs[level].1;
        let n = detail_level.len();
        let mut reconstructed = vec![0.0f32; 2 * n];
        for i in 0..n {
            reconstructed[2 * i] = current[i] + detail_level[i];
            reconstructed[2 * i + 1] = current[i] - detail_level[i];
        }
        current = reconstructed;
    }

    // Trim or pad to original length
    if current.len() < original_length {
        current.resize(original_length, 0.0);
    } else {
        current.truncate(original_length);
    }

    current
}

fn process_r007(samples: &[f32], _sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let num_levels = (pi(params, "num_levels", 5) as usize).clamp(3, 8);
    let modification = ps(params, "modification", "amplify");
    let mod_amount = pf(params, "mod_amount", 2.0);

    let n = samples.len();

    // Pad to nearest power of 2 for clean decomposition
    let mut pow2 = 1usize;
    while pow2 < n {
        pow2 *= 2;
    }
    let mut padded = vec![0.0f32; pow2];
    padded[..n].copy_from_slice(samples);

    let mut coeffs = haar_decompose(&padded, num_levels);

    // Modify detail coefficients
    for level in 0..coeffs.len() {
        match modification {
            "zero" => {
                // Remove detail (smooth)
                let len = coeffs[level].1.len();
                coeffs[level].1 = vec![0.0f32; len];
            }
            "amplify" => {
                // Boost detail (enhance texture)
                for d in coeffs[level].1.iter_mut() {
                    *d *= mod_amount;
                }
            }
            "threshold" => {
                // Hard threshold: zero out small details
                let max_abs = coeffs[level]
                    .1
                    .iter()
                    .map(|d| d.abs())
                    .fold(0.0f32, f32::max);
                let thresh = max_abs * (1.0 / mod_amount);
                for d in coeffs[level].1.iter_mut() {
                    if d.abs() < thresh {
                        *d = 0.0;
                    }
                }
            }
            _ => {
                for d in coeffs[level].1.iter_mut() {
                    *d *= mod_amount;
                }
            }
        }
    }

    let mut out = haar_reconstruct(&coeffs, pow2);
    out.truncate(n);

    // Normalize to match input level
    let in_rms = (samples.iter().map(|s| s * s).sum::<f32>() / n.max(1) as f32).sqrt() + 1e-10;
    let out_rms = (out.iter().map(|s| s * s).sum::<f32>() / n.max(1) as f32).sqrt() + 1e-10;
    let scale = in_rms / out_rms;
    for s in out.iter_mut() {
        *s *= scale;
    }

    AudioOutput::Mono(out)
}

fn variants_r007() -> Vec<HashMap<String, Value>> {
    vec![
        params!("num_levels" => 3, "modification" => "amplify", "mod_amount" => 1.5),
        params!("num_levels" => 5, "modification" => "amplify", "mod_amount" => 2.0),
        params!("num_levels" => 8, "modification" => "amplify", "mod_amount" => 3.0),
        params!("num_levels" => 5, "modification" => "zero", "mod_amount" => 1.0),
        params!("num_levels" => 5, "modification" => "threshold", "mod_amount" => 1.5),
        params!("num_levels" => 5, "modification" => "threshold", "mod_amount" => 3.0),
    ]
}

// ---------------------------------------------------------------------------
// R008 -- Hilbert Envelope + Fine Structure Swap
// ---------------------------------------------------------------------------

fn process_r008(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let fine_structure_source = ps(params, "fine_structure_source", "noise");

    let x: Vec<f64> = samples.iter().map(|&s| s as f64).collect();
    let n = x.len();

    if n == 0 {
        return AudioOutput::Mono(vec![]);
    }

    // Hilbert transform via FFT
    // Pad to power of 2 for efficient FFT
    let mut fft_len = 1;
    while fft_len < n {
        fft_len *= 2;
    }

    let mut planner = RealFftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(fft_len);
    let ifft = planner.plan_fft_inverse(fft_len);

    let mut input = vec![0.0f64; fft_len];
    input[..n].copy_from_slice(&x);

    let mut spectrum = fft.make_output_vec();
    let mut scratch_fwd = fft.make_scratch_vec();
    fft.process_with_scratch(&mut input, &mut spectrum, &mut scratch_fwd).unwrap();

    // Build the analytic signal filter: h[0]=1, h[N/2]=1, h[1..N/2]=2, rest 0
    // For rfft output (len = fft_len/2+1), DC stays, Nyquist stays, others *2
    let spec_len = spectrum.len();
    for i in 1..spec_len - 1 {
        spectrum[i] = spectrum[i].scale(2.0);
    }
    // DC (index 0) and Nyquist (last) remain as-is

    let mut analytic_real = ifft.make_output_vec();
    let mut scratch_inv = ifft.make_scratch_vec();
    ifft.process_with_scratch(&mut spectrum, &mut analytic_real, &mut scratch_inv).unwrap();

    // Compute envelope (magnitude of analytic signal)
    // Since we only have the real part from real IFFT, we need both real & imag.
    // Actually with real FFT: after zeroing negative frequencies and inverse FFT,
    // we get the real part. We need to do a full complex FFT approach instead.
    // Simplified approach: compute envelope from the doubled-spectrum trick.

    // Alternative: use the relationship that for analytic signal z = x + j*hilbert(x),
    // |z|^2 = x^2 + hilbert(x)^2. We can get hilbert(x) by phase-shifting.

    // Let's redo with a simpler approach: compute analytic signal components.
    // Re-do FFT with full complex approach conceptually via real FFT:

    // The analytic_real we got is actually the real part of the analytic signal
    // (unnormalized). We need the imaginary part too.
    // Actually, for the Hilbert envelope, we can use: the IFFT of the one-sided
    // spectrum gives us the analytic signal (real = original, imag = hilbert transform).
    // But realfft only gives us real output from IFFT.

    // Simpler robust approach: compute envelope via low-pass of abs signal
    // or use the original Python method with manual complex FFT.

    // Use manual DFT-based approach with f64 for accuracy:
    // Compute full FFT manually
    let two_pi = 2.0 * std::f64::consts::PI;

    // For efficiency, use the realfft result to reconstruct.
    // The analytic signal's envelope can be computed as:
    // 1. Take FFT of x
    // 2. Zero negative frequencies
    // 3. IFFT -> complex analytic signal
    // 4. |analytic| = envelope

    // Since we have realfft, let's do this differently:
    // Use overlapping DFT frames for the envelope, or use a simpler method.

    // Simplest accurate method: compute Hilbert transform in frequency domain
    // using our full spectrum data.

    // We'll reconstruct envelope using: env^2 = x^2 + H(x)^2
    // where H(x) is obtained by: shift all positive freq components by -pi/2.

    // Recompute: forward FFT, apply -j to positive freqs, +j to negative freqs, IFFT
    let mut planner2 = RealFftPlanner::<f64>::new();
    let fft2 = planner2.plan_fft_forward(fft_len);
    let ifft2 = planner2.plan_fft_inverse(fft_len);

    let mut input2 = vec![0.0f64; fft_len];
    input2[..n].copy_from_slice(&x);

    let mut spec2 = fft2.make_output_vec();
    let mut scratch2 = fft2.make_scratch_vec();
    fft2.process_with_scratch(&mut input2, &mut spec2, &mut scratch2).unwrap();

    // Multiply positive frequencies by -j (rotate by -pi/2):
    // -j * (a + jb) = b - ja
    let spec2_len = spec2.len();
    // DC stays zero for Hilbert transform
    let dc = spec2[0];
    let nyq = spec2[spec2_len - 1];
    spec2[0] = num_complex::Complex::new(0.0, 0.0);
    spec2[spec2_len - 1] = num_complex::Complex::new(0.0, 0.0);
    for i in 1..spec2_len - 1 {
        let a = spec2[i].re;
        let b = spec2[i].im;
        // For positive frequencies in rfft: multiply by -j
        spec2[i] = num_complex::Complex::new(b, -a);
    }

    let mut hilbert_out = ifft2.make_output_vec();
    let mut scratch_inv2 = ifft2.make_scratch_vec();
    ifft2.process_with_scratch(&mut spec2, &mut hilbert_out, &mut scratch_inv2).unwrap();

    let norm = 1.0 / fft_len as f64;

    // Also get normalized original from IFFT or just use input directly
    // Compute envelope: sqrt(x^2 + hilbert(x)^2)
    let mut envelope = vec![0.0f32; n];
    for i in 0..n {
        let h = hilbert_out[i] * norm;
        let env = ((x[i] * x[i] + h * h) as f64).sqrt();
        envelope[i] = env as f32;
    }

    // Generate replacement fine structure
    let fine: Vec<f32> = match fine_structure_source {
        "noise" => {
            let mut rng = ChaCha8Rng::seed_from_u64(42);
            let mut f: Vec<f32> = (0..n).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
            let max_abs = f.iter().map(|v| v.abs()).fold(0.0f32, f32::max) + 1e-10;
            for v in f.iter_mut() {
                *v /= max_abs;
            }
            f
        }
        "sine" => {
            let two_pi_f32 = 2.0 * std::f32::consts::PI;
            (0..n)
                .map(|i| (two_pi_f32 * 440.0 * i as f32 / sr as f32).sin())
                .collect()
        }
        _ => {
            // "original": use original fine structure
            // fine = real(analytic / (envelope + eps))
            // analytic real part = x, so fine = x / envelope
            (0..n)
                .map(|i| {
                    let env = envelope[i] + 1e-10;
                    (x[i] as f32) / env
                })
                .collect()
        }
    };

    let out: Vec<f32> = (0..n).map(|i| envelope[i] * fine[i]).collect();
    AudioOutput::Mono(out)
}

fn variants_r008() -> Vec<HashMap<String, Value>> {
    vec![
        params!("fine_structure_source" => "original"),
        params!("fine_structure_source" => "noise"),
        params!("fine_structure_source" => "sine"),
    ]
}

// ---------------------------------------------------------------------------
// R009 -- Spectral Freeze with Drift
// ---------------------------------------------------------------------------

fn process_r009(samples: &[f32], _sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let freeze_point = pf(params, "freeze_point", 0.3).clamp(0.1, 0.9);
    let drift_rate = pf(params, "drift_rate", 0.01).clamp(0.001, 0.1);

    let fft_size = 2048;
    let hop_size = 512;

    let frames = stft(samples, fft_size, hop_size);
    let num_frames = frames.len();

    if num_frames == 0 {
        return AudioOutput::Mono(samples.to_vec());
    }

    let num_bins = frames[0].len();
    let freeze_frame = (freeze_point * (num_frames - 1) as f32) as usize;

    let frozen_mag: Vec<f32> = frames[freeze_frame].iter().map(|c| c.norm()).collect();
    let mut current_mag = frozen_mag.clone();

    let mut modified_frames = Vec::with_capacity(num_frames);

    for i in 0..num_frames {
        if i <= freeze_frame {
            // Before freeze point: pass through
            modified_frames.push(frames[i].clone());
        } else {
            // After freeze: drift from frozen toward current
            let target_mag: Vec<f32> = frames[i].iter().map(|c| c.norm()).collect();
            for b in 0..num_bins {
                current_mag[b] = current_mag[b] + drift_rate * (target_mag[b] - current_mag[b]);
            }

            let phase: Vec<f32> = frames[i].iter().map(|c| c.arg()).collect();
            let frame: Vec<Complex<f32>> = (0..num_bins)
                .map(|b| Complex::from_polar(current_mag[b], phase[b]))
                .collect();
            modified_frames.push(frame);
        }
    }

    let out = istft(&modified_frames, fft_size, hop_size, Some(samples.len()));
    AudioOutput::Mono(out)
}

fn variants_r009() -> Vec<HashMap<String, Value>> {
    vec![
        params!("freeze_point" => 0.1, "drift_rate" => 0.01),
        params!("freeze_point" => 0.3, "drift_rate" => 0.01),
        params!("freeze_point" => 0.5, "drift_rate" => 0.005),
        params!("freeze_point" => 0.3, "drift_rate" => 0.05),
        params!("freeze_point" => 0.3, "drift_rate" => 0.1),
        params!("freeze_point" => 0.7, "drift_rate" => 0.001),
    ]
}

// ---------------------------------------------------------------------------
// R010 -- Sample-Level Markov Chain
// ---------------------------------------------------------------------------

fn process_r010(samples: &[f32], _sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let num_levels = (pi(params, "num_levels", 64) as usize).clamp(16, 256);
    let _order = (pi(params, "order", 1) as usize).clamp(1, 3);

    let n = samples.len();
    if n < 2 {
        return AudioOutput::Mono(samples.to_vec());
    }

    // Quantize samples to discrete levels
    let mut quantized = vec![0usize; n];
    for i in 0..n {
        let val = ((samples[i] + 1.0) * 0.5).clamp(0.0, 1.0);
        quantized[i] = (val * num_levels as f32) as usize;
        if quantized[i] >= num_levels {
            quantized[i] = num_levels - 1;
        }
    }

    // Build transition table (order 1)
    let mut transitions = vec![vec![0.0f32; num_levels]; num_levels];
    for i in 0..n - 1 {
        transitions[quantized[i]][quantized[i + 1]] += 1.0;
    }

    // Normalize rows to probabilities
    for s in 0..num_levels {
        let row_sum: f32 = transitions[s].iter().sum();
        if row_sum > 0.0 {
            for t in 0..num_levels {
                transitions[s][t] /= row_sum;
            }
        } else {
            for t in 0..num_levels {
                transitions[s][t] = 1.0 / num_levels as f32;
            }
        }
    }

    // Generate audio from Markov chain
    let mut out = vec![0.0f32; n];
    let mut state = quantized[n / 2];

    // Simple LCG random number generator (matching Python)
    let mut rng_state: i64 = 42;

    for i in 0..n {
        out[i] = state as f32 / num_levels as f32 * 2.0 - 1.0;

        // Choose next state based on transition probabilities
        rng_state = (rng_state.wrapping_mul(1103515245).wrapping_add(12345)) & 0x7FFFFFFF;
        let rand_val = (rng_state & 0xFFFF) as f32 / 0xFFFF as f32;

        let mut cumsum = 0.0f32;
        let mut next_state = state;
        for s in 0..num_levels {
            cumsum += transitions[state][s];
            if rand_val <= cumsum {
                next_state = s;
                break;
            }
        }
        state = next_state;
    }

    // Light smoothing: 3-sample moving average
    let mut smoothed = vec![0.0f32; n];
    smoothed[0] = out[0];
    if n > 1 {
        smoothed[n - 1] = out[n - 1];
    }
    for i in 1..n.saturating_sub(1) {
        smoothed[i] = (out[i - 1] + out[i] + out[i + 1]) / 3.0;
    }

    AudioOutput::Mono(smoothed)
}

fn variants_r010() -> Vec<HashMap<String, Value>> {
    vec![
        params!("num_levels" => 16, "order" => 1),
        params!("num_levels" => 32, "order" => 1),
        params!("num_levels" => 64, "order" => 1),
        params!("num_levels" => 128, "order" => 1),
        params!("num_levels" => 256, "order" => 1),
        params!("num_levels" => 64, "order" => 2),
    ]
}

// ---------------------------------------------------------------------------
// R011 -- Frequency Domain Convolution
// ---------------------------------------------------------------------------

fn process_r011(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let synthetic_type = ps(params, "synthetic_type", "sawtooth");

    let fft_size = 2048;
    let hop_size = 512;

    let frames = stft(samples, fft_size, hop_size);
    let num_frames = frames.len();
    let num_bins = frames[0].len();

    // Generate synthetic magnitude spectrum
    let bin_freq = sr as f32 / fft_size as f32;
    let mut synth_mag = vec![0.0f32; num_bins];

    match synthetic_type {
        "sawtooth" => {
            // Sawtooth harmonic series: 1/k amplitude for harmonics of 100 Hz
            let fundamental = 100.0f32;
            for k in 1..50 {
                let harmonic_freq = fundamental * k as f32;
                let bin_idx = (harmonic_freq / bin_freq) as usize;
                if bin_idx < num_bins {
                    synth_mag[bin_idx] = 1.0 / k as f32;
                    if bin_idx > 0 {
                        synth_mag[bin_idx - 1] += 0.3 / k as f32;
                    }
                    if bin_idx + 1 < num_bins {
                        synth_mag[bin_idx + 1] += 0.3 / k as f32;
                    }
                }
            }
        }
        "harmonic_series" => {
            // Pure harmonic series with equal amplitude
            let fundamental = 200.0f32;
            for k in 1..30 {
                let harmonic_freq = fundamental * k as f32;
                let bin_idx = (harmonic_freq / bin_freq) as usize;
                if bin_idx < num_bins {
                    synth_mag[bin_idx] = 1.0;
                    if bin_idx > 0 {
                        synth_mag[bin_idx - 1] += 0.5;
                    }
                    if bin_idx + 1 < num_bins {
                        synth_mag[bin_idx + 1] += 0.5;
                    }
                }
            }
        }
        "noise_shaped" => {
            // Shaped noise: pink-ish spectrum (1/f)
            for b in 0..num_bins {
                let f = (b as f32 * bin_freq).max(1.0);
                synth_mag[b] = 1.0 / f.sqrt();
            }
        }
        _ => {
            for s in synth_mag.iter_mut() {
                *s = 1.0;
            }
        }
    }

    // Normalize synthetic spectrum
    let synth_max = synth_mag.iter().cloned().fold(0.0f32, f32::max) + 1e-10;
    for s in synth_mag.iter_mut() {
        *s /= synth_max;
    }

    // Apply: pointwise multiply magnitude by synthetic spectrum
    let mut modified_frames = Vec::with_capacity(num_frames);
    for i in 0..num_frames {
        let frame: Vec<Complex<f32>> = frames[i]
            .iter()
            .enumerate()
            .map(|(b, &c)| {
                let mag = c.norm() * synth_mag[b];
                let phase = c.arg();
                Complex::from_polar(mag, phase)
            })
            .collect();
        modified_frames.push(frame);
    }

    let out = istft(&modified_frames, fft_size, hop_size, Some(samples.len()));
    AudioOutput::Mono(out)
}

fn variants_r011() -> Vec<HashMap<String, Value>> {
    vec![
        params!("synthetic_type" => "sawtooth"),
        params!("synthetic_type" => "harmonic_series"),
        params!("synthetic_type" => "noise_shaped"),
    ]
}

// ---------------------------------------------------------------------------
// R012 -- Audio Quine
// ---------------------------------------------------------------------------

fn process_r012(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let chunk_start_ms = pf(params, "chunk_start_ms", 100.0).clamp(0.0, 500.0);
    let chunk_length_ms = pf(params, "chunk_length_ms", 100.0).clamp(50.0, 200.0);

    let n = samples.len();
    let chunk_start = ((chunk_start_ms * sr as f32 / 1000.0) as usize).min(n.saturating_sub(1));
    let chunk_length = (chunk_length_ms * sr as f32 / 1000.0) as usize;

    let chunk_start = chunk_start.min(n.saturating_sub(chunk_length));
    let chunk_end = (chunk_start + chunk_length).min(n);

    if chunk_end <= chunk_start {
        return AudioOutput::Mono(samples.to_vec());
    }

    let mut chunk: Vec<f32> = samples[chunk_start..chunk_end].to_vec();

    // Normalize chunk to prevent explosion
    let chunk_peak = chunk.iter().map(|s| s.abs()).fold(0.0f32, f32::max) + 1e-10;
    for c in chunk.iter_mut() {
        *c /= chunk_peak;
    }

    let chunk_len = chunk.len();

    // FFT convolution
    let conv_len = n + chunk_len - 1;
    let mut fft_len = 1;
    while fft_len < conv_len {
        fft_len *= 2;
    }

    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(fft_len);
    let ifft = planner.plan_fft_inverse(fft_len);

    // Forward FFT of input signal
    let mut sig_buf = vec![0.0f32; fft_len];
    sig_buf[..n].copy_from_slice(samples);
    let mut sig_spec = fft.make_output_vec();
    let mut scratch = fft.make_scratch_vec();
    fft.process_with_scratch(&mut sig_buf, &mut sig_spec, &mut scratch).unwrap();

    // Forward FFT of chunk
    let mut chunk_buf = vec![0.0f32; fft_len];
    chunk_buf[..chunk_len].copy_from_slice(&chunk);
    let mut chunk_spec = fft.make_output_vec();
    fft.process_with_scratch(&mut chunk_buf, &mut chunk_spec, &mut scratch).unwrap();

    // Pointwise multiply
    let spec_len = sig_spec.len();
    let mut result_spec: Vec<Complex<f32>> = (0..spec_len)
        .map(|i| sig_spec[i] * chunk_spec[i])
        .collect();

    // Inverse FFT
    let mut result_buf = ifft.make_output_vec();
    let mut scratch_inv = ifft.make_scratch_vec();
    ifft.process_with_scratch(&mut result_spec, &mut result_buf, &mut scratch_inv).unwrap();

    let norm = 1.0 / fft_len as f32;
    let mut out: Vec<f32> = result_buf[..n].iter().map(|&s| s * norm).collect();

    // Normalize output to match input level
    let in_rms = (samples.iter().map(|s| s * s).sum::<f32>() / n.max(1) as f32).sqrt() + 1e-10;
    let out_rms = (out.iter().map(|s| s * s).sum::<f32>() / n.max(1) as f32).sqrt() + 1e-10;
    let scale = in_rms / out_rms;
    for s in out.iter_mut() {
        *s *= scale;
    }

    AudioOutput::Mono(out)
}

fn variants_r012() -> Vec<HashMap<String, Value>> {
    vec![
        params!("chunk_start_ms" => 0, "chunk_length_ms" => 50),
        params!("chunk_start_ms" => 100, "chunk_length_ms" => 100),
        params!("chunk_start_ms" => 200, "chunk_length_ms" => 150),
        params!("chunk_start_ms" => 0, "chunk_length_ms" => 200),
        params!("chunk_start_ms" => 500, "chunk_length_ms" => 100),
    ]
}

// ---------------------------------------------------------------------------
// R013 -- Spectral Phase Vocoder Modified Hop
// ---------------------------------------------------------------------------

fn process_r013(samples: &[f32], _sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let analysis_hop = (pi(params, "analysis_hop", 512) as usize).clamp(256, 1024);
    let synthesis_hop = (pi(params, "synthesis_hop", 1024) as usize).clamp(128, 2048);

    let fft_size = 2048;
    let n = samples.len();

    // Analysis STFT with analysis_hop
    let frames = stft(samples, fft_size, analysis_hop);
    let num_frames = frames.len();

    if num_frames < 2 {
        return AudioOutput::Mono(samples.to_vec());
    }

    let num_bins = frames[0].len();
    let two_pi = 2.0 * std::f64::consts::PI;

    // Compute magnitudes and phases
    let mag: Vec<Vec<f32>> = frames
        .iter()
        .map(|f| f.iter().map(|c| c.norm()).collect())
        .collect();
    let phase: Vec<Vec<f64>> = frames
        .iter()
        .map(|f| f.iter().map(|c| (c.arg() as f64)).collect())
        .collect();

    // Expected phase advance per analysis hop
    let omega: Vec<f64> = (0..num_bins)
        .map(|b| two_pi * b as f64 * analysis_hop as f64 / fft_size as f64)
        .collect();
    let hop_ratio = synthesis_hop as f64 / analysis_hop as f64;

    let mut synth_phase = vec![vec![0.0f64; num_bins]; num_frames];
    synth_phase[0] = phase[0].clone();

    for i in 1..num_frames {
        for b in 0..num_bins {
            let dphi = phase[i][b] - phase[i - 1][b] - omega[b];
            // Wrap to [-pi, pi]
            let dphi_wrapped = dphi - two_pi * (dphi / two_pi).round();
            let freq_dev = dphi_wrapped / analysis_hop as f64;
            synth_phase[i][b] =
                synth_phase[i - 1][b] + (omega[b] + freq_dev * analysis_hop as f64) * hop_ratio;
        }
    }

    // Build synthesis frames
    let synth_frames: Vec<Vec<Complex<f32>>> = (0..num_frames)
        .map(|i| {
            (0..num_bins)
                .map(|b| Complex::from_polar(mag[i][b], synth_phase[i][b] as f32))
                .collect()
        })
        .collect();

    // Output length adjusted by hop ratio
    let out_length = (n as f64 * hop_ratio) as usize;
    let raw_out = istft(&synth_frames, fft_size, synthesis_hop, Some(out_length));

    // Resample back to original length via linear interpolation
    if raw_out.len() != n && !raw_out.is_empty() {
        let mut result = vec![0.0f32; n];
        let out_len = raw_out.len();
        for i in 0..n {
            let idx_f = i as f64 / n.max(1) as f64 * (out_len - 1) as f64;
            let idx0 = (idx_f as usize).min(out_len.saturating_sub(2));
            let frac = (idx_f - idx0 as f64) as f32;
            result[i] = (1.0 - frac) * raw_out[idx0] + frac * raw_out[idx0 + 1];
        }
        AudioOutput::Mono(result)
    } else {
        AudioOutput::Mono(raw_out)
    }
}

fn variants_r013() -> Vec<HashMap<String, Value>> {
    vec![
        params!("analysis_hop" => 512, "synthesis_hop" => 256),
        params!("analysis_hop" => 512, "synthesis_hop" => 512),
        params!("analysis_hop" => 512, "synthesis_hop" => 1024),
        params!("analysis_hop" => 512, "synthesis_hop" => 2048),
        params!("analysis_hop" => 256, "synthesis_hop" => 1024),
        params!("analysis_hop" => 1024, "synthesis_hop" => 256),
    ]
}

// ---------------------------------------------------------------------------
// R014 -- Karplus-Strong Cloud
// ---------------------------------------------------------------------------

fn process_r014(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let num_strings = (pi(params, "num_strings", 15) as usize).clamp(10, 30);
    let min_freq = pf(params, "min_freq", 80.0).clamp(50.0, 200.0);
    let max_freq = pf(params, "max_freq", 1000.0).clamp(500.0, 2000.0);
    let decay = pf(params, "decay", 0.99).clamp(0.95, 0.999);

    let n = samples.len();

    // Generate random frequencies (log-spaced)
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let log_min = min_freq.ln();
    let log_max = max_freq.ln();

    let mut delays = vec![0usize; num_strings];
    let mut gains = vec![0.0f32; num_strings];
    for i in 0..num_strings {
        let log_freq = rng.gen::<f32>() * (log_max - log_min) + log_min;
        let freq = log_freq.exp();
        delays[i] = (sr as f32 / freq) as usize;
        delays[i] = delays[i].max(2);
        gains[i] = 1.0 / num_strings as f32;
    }

    let mut out = vec![0.0f32; n];

    for s_idx in 0..num_strings {
        let delay = delays[s_idx];
        let gain = gains[s_idx];
        let buf_len = (delay + 1).max(2);
        let mut buf = vec![0.0f32; buf_len];
        let mut write_pos: usize = 0;

        // Initialize buffer with the input signal (noise burst excitation)
        for i in 0..delay.min(n) {
            buf[i] = samples[i] * gain;
        }

        for i in 0..n {
            let read_pos = (write_pos + buf_len - delay) % buf_len;
            let read_pos_next = (read_pos + 1) % buf_len;
            // KS averaging filter
            let mut y = decay * 0.5 * (buf[read_pos] + buf[read_pos_next]);

            // Add input excitation
            if i < delay {
                y += samples[i] * gain;
            }

            buf[write_pos] = y;
            out[i] += y;
            write_pos = (write_pos + 1) % buf_len;
        }
    }

    // Normalize
    let in_rms = (samples.iter().map(|s| s * s).sum::<f32>() / n.max(1) as f32).sqrt() + 1e-10;
    let out_rms = (out.iter().map(|s| s * s).sum::<f32>() / n.max(1) as f32).sqrt() + 1e-10;
    let scale = in_rms / out_rms;
    for s in out.iter_mut() {
        *s *= scale;
    }

    AudioOutput::Mono(out)
}

fn variants_r014() -> Vec<HashMap<String, Value>> {
    vec![
        params!("num_strings" => 10, "min_freq" => 80, "max_freq" => 500, "decay" => 0.99),
        params!("num_strings" => 15, "min_freq" => 80, "max_freq" => 1000, "decay" => 0.99),
        params!("num_strings" => 20, "min_freq" => 100, "max_freq" => 1500, "decay" => 0.995),
        params!("num_strings" => 30, "min_freq" => 50, "max_freq" => 2000, "decay" => 0.998),
        params!("num_strings" => 10, "min_freq" => 200, "max_freq" => 800, "decay" => 0.95),
        params!("num_strings" => 20, "min_freq" => 100, "max_freq" => 600, "decay" => 0.999),
    ]
}

// ---------------------------------------------------------------------------
// R015 -- Feedback FM Synthesis
// ---------------------------------------------------------------------------

fn process_r015(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let carrier_freq = pf(params, "carrier_freq", 440.0).clamp(50.0, 2000.0);
    let mod_index = pf(params, "mod_index", 3.0).clamp(0.5, 10.0);
    let feedback = pf(params, "feedback", 0.4).clamp(0.1, 0.9);

    let n = samples.len();
    let two_pi = 2.0 * std::f32::consts::PI;
    let mut out = vec![0.0f32; n];
    let mut y_prev = 0.0f32;

    for i in 0..n {
        let phase = two_pi * carrier_freq * i as f32 / sr as f32;
        let fm_signal = (phase + mod_index * y_prev).sin();
        let mut y = samples[i] + feedback * fm_signal;
        // Soft clip to prevent runaway
        if y > 1.0 || y < -1.0 {
            y = y.tanh();
        }
        out[i] = y;
        y_prev = y;
    }

    AudioOutput::Mono(out)
}

fn variants_r015() -> Vec<HashMap<String, Value>> {
    vec![
        params!("carrier_freq" => 200, "mod_index" => 1.0, "feedback" => 0.2),
        params!("carrier_freq" => 440, "mod_index" => 3.0, "feedback" => 0.4),
        params!("carrier_freq" => 880, "mod_index" => 5.0, "feedback" => 0.3),
        params!("carrier_freq" => 100, "mod_index" => 7.0, "feedback" => 0.5),
        params!("carrier_freq" => 1500, "mod_index" => 2.0, "feedback" => 0.6),
        params!("carrier_freq" => 300, "mod_index" => 10.0, "feedback" => 0.9),
        params!("carrier_freq" => 660, "mod_index" => 4.0, "feedback" => 0.15),
    ]
}

// ---------------------------------------------------------------------------
// R016 -- Feedback AM Synthesis
// ---------------------------------------------------------------------------

fn process_r016(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let carrier_freq = pf(params, "carrier_freq", 200.0).clamp(20.0, 2000.0);
    let feedback = pf(params, "feedback", 0.3).clamp(0.0, 0.95);
    let depth = pf(params, "depth", 0.8).clamp(0.1, 1.0);

    let n = samples.len();
    let two_pi = 2.0 * std::f32::consts::PI;
    let mut out = vec![0.0f32; n];
    let mut y_prev = 0.0f32;

    for i in 0..n {
        // Carrier phase modulated by feedback
        let phase = two_pi * carrier_freq * i as f32 / sr as f32 + feedback * y_prev;
        let carrier = phase.sin();

        // AM: multiply input by (1 - depth + depth*carrier) to preserve some dry signal
        let mut modulated = samples[i] * (1.0 - depth + depth * carrier);

        // Soft clip
        if modulated > 1.0 || modulated < -1.0 {
            modulated = modulated.tanh();
        }

        out[i] = modulated;
        y_prev = modulated;
    }

    AudioOutput::Mono(out)
}

fn variants_r016() -> Vec<HashMap<String, Value>> {
    vec![
        params!("carrier_freq" => 100, "feedback" => 0.0, "depth" => 0.5),
        params!("carrier_freq" => 200, "feedback" => 0.2, "depth" => 0.7),
        params!("carrier_freq" => 440, "feedback" => 0.4, "depth" => 0.8),
        params!("carrier_freq" => 800, "feedback" => 0.3, "depth" => 1.0),
        params!("carrier_freq" => 150, "feedback" => 0.7, "depth" => 0.9),
        params!("carrier_freq" => 1200, "feedback" => 0.6, "depth" => 0.6),
        params!("carrier_freq" => 300, "feedback" => 0.95, "depth" => 1.0),
    ]
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

pub fn register() -> Vec<EffectEntry> {
    vec![
        EffectEntry {
            id: "r001_audio_fractalization",
            process: process_r001,
            variants: variants_r001,
            category: "r_misc",
        },
        EffectEntry {
            id: "r002_spectral_peak_resynthesis",
            process: process_r002,
            variants: variants_r002,
            category: "r_misc",
        },
        EffectEntry {
            id: "r003_lpc_resynthesis",
            process: process_r003,
            variants: variants_r003,
            category: "r_misc",
        },
        EffectEntry {
            id: "r004_spectral_painting",
            process: process_r004,
            variants: variants_r004,
            category: "r_misc",
        },
        EffectEntry {
            id: "r005_phase_gradient",
            process: process_r005,
            variants: variants_r005,
            category: "r_misc",
        },
        EffectEntry {
            id: "r006_spectral_entropy_gate",
            process: process_r006,
            variants: variants_r006,
            category: "r_misc",
        },
        EffectEntry {
            id: "r007_wavelet_decomposition",
            process: process_r007,
            variants: variants_r007,
            category: "r_misc",
        },
        EffectEntry {
            id: "r008_hilbert_envelope_swap",
            process: process_r008,
            variants: variants_r008,
            category: "r_misc",
        },
        EffectEntry {
            id: "r009_spectral_freeze_drift",
            process: process_r009,
            variants: variants_r009,
            category: "r_misc",
        },
        EffectEntry {
            id: "r010_markov_chain",
            process: process_r010,
            variants: variants_r010,
            category: "r_misc",
        },
        EffectEntry {
            id: "r011_freq_domain_convolution",
            process: process_r011,
            variants: variants_r011,
            category: "r_misc",
        },
        EffectEntry {
            id: "r012_audio_quine",
            process: process_r012,
            variants: variants_r012,
            category: "r_misc",
        },
        EffectEntry {
            id: "r013_modified_hop_vocoder",
            process: process_r013,
            variants: variants_r013,
            category: "r_misc",
        },
        EffectEntry {
            id: "r014_karplus_strong_cloud",
            process: process_r014,
            variants: variants_r014,
            category: "r_misc",
        },
        EffectEntry {
            id: "r015_feedback_fm",
            process: process_r015,
            variants: variants_r015,
            category: "r_misc",
        },
        EffectEntry {
            id: "r016_feedback_am",
            process: process_r016,
            variants: variants_r016,
            category: "r_misc",
        },
    ]
}
