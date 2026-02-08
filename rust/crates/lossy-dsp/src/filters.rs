//! Time-domain processing: biquad filter, lo-fi reverb, noise gate, limiter.
//!
//! Port of `lossy/engine/filters.py`.

use crate::params::{LossyParams, SR, SLOPE_OPTIONS};

// ---------------------------------------------------------------------------
// Biquad filter (bandpass / notch)
// ---------------------------------------------------------------------------

struct BiquadCoeffs {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
}

fn compute_biquad_coeffs(filter_type: i32, freq: f64, width: f64, slope_idx: i32) -> (BiquadCoeffs, usize) {
    let idx = (slope_idx as usize).min(SLOPE_OPTIONS.len() - 1);
    let slope = SLOPE_OPTIONS[idx];
    let n_sections = crate::params::slope_sections(slope);

    // Width -> Q (log mapping: 0 -> Q=20, 1 -> Q=0.3)
    let q_base = 0.3 * (20.0_f64 / 0.3).powf(1.0 - width);
    let q_boost = match slope {
        6 => 1.0,
        24 => 1.5,
        96 => 3.0,
        _ => 1.0,
    };
    let q = q_base * q_boost;

    let w0 = 2.0 * std::f64::consts::PI * freq / SR;
    let alpha = w0.sin() / (2.0 * q);
    let cos_w0 = w0.cos();

    let (b, a) = if filter_type == 1 {
        // Bandpass (0 dB peak)
        ([alpha, 0.0, -alpha], [1.0 + alpha, -2.0 * cos_w0, 1.0 - alpha])
    } else {
        // Notch
        ([1.0, -2.0 * cos_w0, 1.0], [1.0 + alpha, -2.0 * cos_w0, 1.0 - alpha])
    };

    // Normalize by a[0]
    let a0 = a[0];
    let coeffs = BiquadCoeffs {
        b0: b[0] / a0,
        b1: b[1] / a0,
        b2: b[2] / a0,
        a1: a[1] / a0,
        a2: a[2] / a0,
    };

    (coeffs, n_sections)
}

/// Direct-form II biquad, one pass.
fn biquad_process(audio: &[f64], c: &BiquadCoeffs) -> Vec<f64> {
    let n = audio.len();
    let mut out = vec![0.0; n];
    let mut w1 = 0.0_f64;
    let mut w2 = 0.0_f64;
    for i in 0..n {
        let w0 = audio[i] - c.a1 * w1 - c.a2 * w2;
        out[i] = c.b0 * w0 + c.b1 * w1 + c.b2 * w2;
        w2 = w1;
        w1 = w0;
    }
    out
}

/// Run biquad filter chain on audio.
pub fn apply_filter(audio: &[f64], params: &LossyParams) -> Vec<f64> {
    if params.filter_type == 0 {
        return audio.to_vec();
    }

    let freq = params.filter_freq.clamp(20.0, SR / 2.0 - 1.0);
    let (coeffs, n_sections) = compute_biquad_coeffs(
        params.filter_type,
        freq,
        params.filter_width,
        params.filter_slope,
    );

    let mut out = audio.to_vec();
    for _ in 0..n_sections {
        out = biquad_process(&out, &coeffs);
    }
    out
}

// ---------------------------------------------------------------------------
// Lo-fi reverb (4 comb filters + allpass, deliberately cheap & metallic)
// ---------------------------------------------------------------------------

/// Blend in a lo-fi Schroeder reverb.
pub fn lofi_reverb(audio: &[f64], params: &LossyParams) -> Vec<f64> {
    let g = params.global_amount;
    let mix = params.verb * g;
    if mix <= 0.0 {
        return audio.to_vec();
    }
    let fb = 0.4 + 0.55 * params.decay; // range 0.4 - 0.95
    comb_reverb(audio, mix, fb)
}

fn comb_reverb(audio: &[f64], mix: f64, fb: f64) -> Vec<f64> {
    let n = audio.len();
    let mut wet = vec![0.0_f64; n];

    // Short prime-number delays -> metallic, lo-fi
    let delays = [1031_usize, 1327, 1657, 1973];
    let damp = 0.45_f64;

    for &d in &delays {
        let mut buf = vec![0.0_f64; d];
        let mut y1 = 0.0_f64;
        for i in 0..n {
            let idx = i % d;
            let rd = buf[idx];
            y1 = damp * rd + (1.0 - damp) * y1;
            wet[i] += rd * 0.25;
            buf[idx] = audio[i] * 0.25 + y1 * fb;
        }
    }

    // Single allpass diffuser
    let ap_d = 379_usize;
    let mut ap_buf = vec![0.0_f64; ap_d];
    let ap_g = 0.6_f64;
    for i in 0..n {
        let idx = i % ap_d;
        let delayed = ap_buf[idx];
        let inp = wet[i];
        wet[i] = delayed - ap_g * inp;
        ap_buf[idx] = inp + ap_g * wet[i];
    }

    // Mix
    let mut out = vec![0.0_f64; n];
    for i in 0..n {
        out[i] = audio[i] * (1.0 - mix) + wet[i] * mix;
    }
    out
}

// ---------------------------------------------------------------------------
// Noise gate
// ---------------------------------------------------------------------------

/// Simple RMS-based noise gate.
pub fn noise_gate(audio: &[f64], params: &LossyParams) -> Vec<f64> {
    let g = params.global_amount;
    let threshold = params.gate * g;
    if threshold <= 0.0 {
        return audio.to_vec();
    }
    gate_process(audio, threshold)
}

fn gate_process(audio: &[f64], threshold: f64) -> Vec<f64> {
    let n = audio.len();
    let mut out = audio.to_vec();
    let win = 512_usize; // ~11 ms gate window
    let mut start = 0;
    while start < n {
        let end = (start + win).min(n);
        let mut s = 0.0_f64;
        for i in start..end {
            s += audio[i] * audio[i];
        }
        let rms = (s / (end - start) as f64).sqrt();
        if rms < threshold {
            let gain = rms / threshold;
            for i in start..end {
                out[i] *= gain;
            }
        }
        start = end;
    }
    out
}

// ---------------------------------------------------------------------------
// Limiter
// ---------------------------------------------------------------------------

/// Simple peak limiter.
///
/// If params is provided, reads `threshold` to set the ceiling:
/// threshold 0 = heavy limiting (ceiling 0.1), threshold 1 = light (ceiling 0.95).
pub fn limiter(audio: &[f64], params: &LossyParams) -> Vec<f64> {
    let ceiling = 0.1 + 0.85 * params.threshold;

    let peak = audio.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
    if peak <= 0.0 {
        return audio.to_vec();
    }
    if peak > ceiling {
        let scale = ceiling / peak;
        return audio.iter().map(|x| x * scale).collect();
    }
    audio.to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_bypass() {
        let audio: Vec<f64> = (0..1000).map(|i| (i as f64 * 0.01).sin()).collect();
        let params = LossyParams::default(); // filter_type=0 (bypass)
        let out = apply_filter(&audio, &params);
        assert_eq!(out.len(), audio.len());
        for (a, b) in audio.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
    }

    #[test]
    fn test_bandpass_attenuates() {
        // Generate DC + high freq. Bandpass at 1000Hz should attenuate both.
        let n = 44100;
        let audio: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / SR;
                0.5 + (2.0 * std::f64::consts::PI * 10000.0 * t).sin() * 0.5
            })
            .collect();
        let mut params = LossyParams::default();
        params.filter_type = 1; // bandpass
        params.filter_freq = 1000.0;
        params.filter_width = 0.3;
        let out = apply_filter(&audio, &params);
        // RMS should be lower than input (DC and HF attenuated)
        let rms_in: f64 = (audio.iter().map(|x| x * x).sum::<f64>() / n as f64).sqrt();
        let rms_out: f64 = (out.iter().map(|x| x * x).sum::<f64>() / n as f64).sqrt();
        assert!(rms_out < rms_in);
    }

    #[test]
    fn test_reverb_passthrough_when_zero() {
        let audio: Vec<f64> = (0..1000).map(|i| (i as f64 * 0.01).sin()).collect();
        let params = LossyParams::default(); // verb=0.0
        let out = lofi_reverb(&audio, &params);
        for (a, b) in audio.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
    }

    #[test]
    fn test_reverb_adds_tail() {
        let n = 44100;
        let mut audio = vec![0.0_f64; n];
        // Impulse at start
        audio[0] = 1.0;
        let mut params = LossyParams::default();
        params.verb = 1.0;
        params.decay = 0.8;
        let out = lofi_reverb(&audio, &params);
        // Should have energy after the impulse
        let tail_energy: f64 = out[1000..].iter().map(|x| x * x).sum();
        assert!(tail_energy > 0.001);
    }

    #[test]
    fn test_gate_silences_quiet() {
        let audio = vec![0.001_f64; 1000]; // very quiet
        let mut params = LossyParams::default();
        params.gate = 0.5;
        let out = noise_gate(&audio, &params);
        let rms: f64 = (out.iter().map(|x| x * x).sum::<f64>() / out.len() as f64).sqrt();
        assert!(rms < 0.001); // should be attenuated
    }

    #[test]
    fn test_limiter_caps_peak() {
        let audio = vec![2.0_f64; 100]; // way above ceiling
        let params = LossyParams::default(); // threshold=0.5 -> ceiling=0.525
        let out = limiter(&audio, &params);
        let peak = out.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
        assert!(peak <= 0.526); // within floating point tolerance
    }
}
