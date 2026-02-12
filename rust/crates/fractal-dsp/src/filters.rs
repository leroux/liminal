//! Filters, noise gate, limiter, bitcrusher for the Fractal effect.
//!
//! Port of `fractal/engine/filters.py`.

use crate::params::{FractalParams, SR};

// ---------------------------------------------------------------------------
// Biquad filter (lowpass / highpass / bandpass)
// ---------------------------------------------------------------------------

struct BiquadCoeffs {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
}

/// Compute biquad coefficients for the given filter type.
/// filter_type: 1=lowpass, 2=highpass, 3=bandpass
fn compute_biquad_coeffs(filter_type: i32, freq: f64, q: f64) -> BiquadCoeffs {
    let w0 = 2.0 * std::f64::consts::PI * freq / SR;
    let alpha = w0.sin() / (2.0 * q);
    let cos_w0 = w0.cos();

    let (b, a) = match filter_type {
        1 => {
            // Lowpass
            let half = (1.0 - cos_w0) / 2.0;
            (
                [half, 1.0 - cos_w0, half],
                [1.0 + alpha, -2.0 * cos_w0, 1.0 - alpha],
            )
        }
        2 => {
            // Highpass
            let half = (1.0 + cos_w0) / 2.0;
            (
                [half, -(1.0 + cos_w0), half],
                [1.0 + alpha, -2.0 * cos_w0, 1.0 - alpha],
            )
        }
        _ => {
            // Bandpass (type 3)
            (
                [alpha, 0.0, -alpha],
                [1.0 + alpha, -2.0 * cos_w0, 1.0 - alpha],
            )
        }
    };

    let a0 = a[0];
    BiquadCoeffs {
        b0: b[0] / a0,
        b1: b[1] / a0,
        b2: b[2] / a0,
        a1: a[1] / a0,
        a2: a[2] / a0,
    }
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

/// Apply pre-fractal filter.
pub fn apply_pre_filter(audio: &[f64], params: &FractalParams) -> Vec<f64> {
    if params.filter_type == 0 {
        return audio.to_vec();
    }

    let freq = params.filter_freq.clamp(20.0, SR / 2.0 - 1.0);
    let q = params.filter_q.max(0.1);
    let coeffs = compute_biquad_coeffs(params.filter_type, freq, q);
    biquad_process(audio, &coeffs)
}

/// Apply post-fractal filter.
pub fn apply_post_filter(audio: &[f64], params: &FractalParams) -> Vec<f64> {
    if params.post_filter_type == 0 {
        return audio.to_vec();
    }

    let freq = params.post_filter_freq.clamp(20.0, SR / 2.0 - 1.0);
    // Post-filter uses gentle Q for taming
    let coeffs = compute_biquad_coeffs(params.post_filter_type, freq, 0.707);
    biquad_process(audio, &coeffs)
}

// ---------------------------------------------------------------------------
// Bitcrusher + sample rate reducer
// ---------------------------------------------------------------------------

/// Apply bitcrusher and/or sample rate reducer.
pub fn crush_and_decimate(audio: &[f64], params: &FractalParams) -> Vec<f64> {
    let crush = params.crush;
    let decimate = params.decimate;

    if crush <= 0.0 && decimate <= 0.0 {
        return audio.to_vec();
    }

    let n = audio.len();
    let mut out = vec![0.0_f64; n];

    if crush > 0.0 {
        let bits = 16.0 - 12.0 * crush;
        let quant = (2.0_f64).powf(bits - 1.0);
        for i in 0..n {
            out[i] = (audio[i] * quant + 0.5).floor() / quant;
        }
    } else {
        out.copy_from_slice(audio);
    }

    if decimate > 0.0 {
        let rate_factor = 1.0 + 31.0 * decimate;
        let mut phase = 0.0_f64;
        let mut held = 0.0_f64;
        for i in 0..n {
            phase += 1.0;
            if phase >= rate_factor {
                held = out[i];
                phase -= rate_factor;
            }
            out[i] = held;
        }
    }

    out
}

// ---------------------------------------------------------------------------
// Noise gate
// ---------------------------------------------------------------------------

/// Simple RMS-based noise gate.
pub fn noise_gate(audio: &[f64], params: &FractalParams) -> Vec<f64> {
    let threshold = params.gate;
    if threshold <= 0.0 {
        return audio.to_vec();
    }

    let n = audio.len();
    let mut out = audio.to_vec();
    let win = 512_usize;
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
// One-pole filters for layer tilt
// ---------------------------------------------------------------------------

/// In-place one-pole lowpass filter.
pub fn apply_one_pole_lp(audio: &mut [f64], cutoff_hz: f64) {
    let cutoff = cutoff_hz.clamp(20.0, SR / 2.0 - 1.0);
    let rc = 1.0 / (2.0 * std::f64::consts::PI * cutoff);
    let dt = 1.0 / SR;
    let alpha = dt / (rc + dt);
    let mut prev = audio[0];
    for s in audio.iter_mut() {
        prev += alpha * (*s - prev);
        *s = prev;
    }
}

/// In-place one-pole highpass filter.
pub fn apply_one_pole_hp(audio: &mut [f64], cutoff_hz: f64) {
    let cutoff = cutoff_hz.clamp(20.0, SR / 2.0 - 1.0);
    let rc = 1.0 / (2.0 * std::f64::consts::PI * cutoff);
    let dt = 1.0 / SR;
    let alpha = rc / (rc + dt);
    let mut prev_in = audio[0];
    let mut prev_out = audio[0];
    for s in audio.iter_mut() {
        let inp = *s;
        prev_out = alpha * (prev_out + inp - prev_in);
        prev_in = inp;
        *s = prev_out;
    }
}

// ---------------------------------------------------------------------------
// Limiter
// ---------------------------------------------------------------------------

/// Simple peak limiter.
///
/// threshold 0 = heavy limiting (ceiling 0.1), threshold 1 = light (ceiling 0.95).
pub fn limiter(audio: &[f64], params: &FractalParams) -> Vec<f64> {
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
        let params = FractalParams::default(); // filter_type=0 (bypass)
        let out = apply_pre_filter(&audio, &params);
        assert_eq!(out.len(), audio.len());
        for (a, b) in audio.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
    }

    #[test]
    fn test_lowpass_attenuates_hf() {
        let n = 44100;
        let audio: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / SR;
                (2.0 * std::f64::consts::PI * 10000.0 * t).sin()
            })
            .collect();
        let mut params = FractalParams::default();
        params.filter_type = 1; // lowpass
        params.filter_freq = 1000.0;
        params.filter_q = 0.707;
        let out = apply_pre_filter(&audio, &params);
        let rms_in: f64 = (audio.iter().map(|x| x * x).sum::<f64>() / n as f64).sqrt();
        let rms_out: f64 = (out.iter().map(|x| x * x).sum::<f64>() / n as f64).sqrt();
        assert!(rms_out < rms_in * 0.5);
    }

    #[test]
    fn test_post_filter_bypass() {
        let audio: Vec<f64> = (0..1000).map(|i| (i as f64 * 0.01).sin()).collect();
        let params = FractalParams::default(); // post_filter_type=0 (bypass)
        let out = apply_post_filter(&audio, &params);
        for (a, b) in audio.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
    }

    #[test]
    fn test_crush_and_decimate() {
        let audio: Vec<f64> = (0..1000).map(|i| (i as f64 * 0.01).sin() * 0.5).collect();
        let mut params = FractalParams::default();
        params.crush = 0.5;
        params.decimate = 0.3;
        let out = crush_and_decimate(&audio, &params);
        assert_eq!(out.len(), audio.len());
        assert!(out.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_gate_silences_quiet() {
        let audio = vec![0.001_f64; 1000];
        let mut params = FractalParams::default();
        params.gate = 0.5;
        let out = noise_gate(&audio, &params);
        let rms: f64 = (out.iter().map(|x| x * x).sum::<f64>() / out.len() as f64).sqrt();
        assert!(rms < 0.001);
    }

    #[test]
    fn test_limiter_caps_peak() {
        let audio = vec![2.0_f64; 100];
        let params = FractalParams::default(); // threshold=0.5 -> ceiling=0.525
        let out = limiter(&audio, &params);
        let peak = out.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
        assert!(peak <= 0.526);
    }
}
