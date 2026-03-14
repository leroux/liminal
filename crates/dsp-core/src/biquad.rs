//! Direct-form II biquad filter.

use crate::SR;

/// Biquad filter coefficients (normalized, a0 = 1).
#[derive(Clone, Debug)]
pub struct BiquadCoeffs {
    pub b0: f64,
    pub b1: f64,
    pub b2: f64,
    pub a1: f64,
    pub a2: f64,
}

/// Filter type for `compute_biquad_coeffs`.
/// 0 = bypass, 1 = lowpass, 2 = highpass, 3 = bandpass, 4 = notch.
pub fn compute_biquad_coeffs(filter_type: i32, freq: f64, q: f64) -> BiquadCoeffs {
    let freq = freq.clamp(1.0, SR / 2.0 - 1.0);
    let q = q.max(0.001);
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
        3 => {
            // Bandpass (0 dB peak)
            (
                [alpha, 0.0, -alpha],
                [1.0 + alpha, -2.0 * cos_w0, 1.0 - alpha],
            )
        }
        4 => {
            // Notch
            (
                [1.0, -2.0 * cos_w0, 1.0],
                [1.0 + alpha, -2.0 * cos_w0, 1.0 - alpha],
            )
        }
        _ => {
            // Bypass (pass-through coefficients)
            return BiquadCoeffs {
                b0: 1.0,
                b1: 0.0,
                b2: 0.0,
                a1: 0.0,
                a2: 0.0,
            };
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

/// Direct-form II biquad, one pass, allocating.
pub fn biquad_process(audio: &[f64], c: &BiquadCoeffs) -> Vec<f64> {
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

/// Direct-form II biquad, one pass, in-place.
pub fn biquad_process_inplace(audio: &mut [f64], c: &BiquadCoeffs) {
    let mut w1 = 0.0_f64;
    let mut w2 = 0.0_f64;
    for s in audio.iter_mut() {
        let w0 = *s - c.a1 * w1 - c.a2 * w2;
        *s = c.b0 * w0 + c.b1 * w1 + c.b2 * w2;
        w2 = w1;
        w1 = w0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bypass_is_identity() {
        let audio: Vec<f64> = (0..100).map(|i| (i as f64 * 0.01).sin()).collect();
        let c = compute_biquad_coeffs(0, 1000.0, 0.707);
        let out = biquad_process(&audio, &c);
        for (a, b) in audio.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
    }

    #[test]
    fn lowpass_attenuates_high_freq() {
        let n = 44100;
        let audio: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / SR;
                (2.0 * std::f64::consts::PI * 10000.0 * t).sin()
            })
            .collect();
        let c = compute_biquad_coeffs(1, 1000.0, 0.707);
        let out = biquad_process(&audio, &c);
        let rms_in: f64 = (audio.iter().map(|x| x * x).sum::<f64>() / n as f64).sqrt();
        let rms_out: f64 = (out.iter().map(|x| x * x).sum::<f64>() / n as f64).sqrt();
        assert!(rms_out < rms_in * 0.5);
    }

    #[test]
    fn inplace_matches_allocating() {
        let audio: Vec<f64> = (0..1000).map(|i| (i as f64 * 0.01).sin()).collect();
        let c = compute_biquad_coeffs(1, 2000.0, 1.0);
        let out = biquad_process(&audio, &c);
        let mut inplace = audio.clone();
        biquad_process_inplace(&mut inplace, &c);
        for (a, b) in out.iter().zip(inplace.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
    }

    /// Helper: generate a sine wave at `freq_hz` for `n` samples.
    fn sine(freq_hz: f64, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let t = i as f64 / SR;
                (2.0 * std::f64::consts::PI * freq_hz * t).sin()
            })
            .collect()
    }

    /// Helper: RMS of a signal.
    fn rms(signal: &[f64]) -> f64 {
        (signal.iter().map(|x| x * x).sum::<f64>() / signal.len() as f64).sqrt()
    }

    #[test]
    fn highpass_attenuates_low_freq() {
        let n = 44100;
        let low = sine(100.0, n);
        let c = compute_biquad_coeffs(2, 5000.0, 0.707);
        let out = biquad_process(&low, &c);
        // 100 Hz should be heavily attenuated by a 5 kHz highpass
        assert!(rms(&out) < rms(&low) * 0.1);
    }

    #[test]
    fn highpass_passes_high_freq() {
        let n = 44100;
        let high = sine(10000.0, n);
        let c = compute_biquad_coeffs(2, 5000.0, 0.707);
        let out = biquad_process(&high, &c);
        // 10 kHz should pass through a 5 kHz highpass with minimal loss
        assert!(rms(&out) > rms(&high) * 0.5);
    }

    #[test]
    fn bandpass_passes_center_freq() {
        let n = 44100;
        let center = sine(2000.0, n);
        let c = compute_biquad_coeffs(3, 2000.0, 1.0);
        let out = biquad_process(&center, &c);
        // Center frequency should pass through with reasonable level
        assert!(rms(&out) > rms(&center) * 0.3);
    }

    #[test]
    fn bandpass_attenuates_far_freq() {
        let n = 44100;
        let far = sine(15000.0, n);
        let c = compute_biquad_coeffs(3, 2000.0, 1.0);
        let out = biquad_process(&far, &c);
        // 15 kHz should be attenuated by a 2 kHz bandpass
        assert!(rms(&out) < rms(&far) * 0.3);
    }

    #[test]
    fn notch_attenuates_center_freq() {
        let n = 44100;
        let center = sine(2000.0, n);
        let c = compute_biquad_coeffs(4, 2000.0, 5.0); // narrow Q for deeper notch
        let out = biquad_process(&center, &c);
        // Center frequency should be deeply attenuated
        assert!(rms(&out) < rms(&center) * 0.15);
    }

    #[test]
    fn notch_passes_distant_freq() {
        let n = 44100;
        let distant = sine(10000.0, n);
        let c = compute_biquad_coeffs(4, 2000.0, 5.0);
        let out = biquad_process(&distant, &c);
        // 10 kHz should pass through a 2 kHz notch virtually unchanged
        assert!(rms(&out) > rms(&distant) * 0.8);
    }

    #[test]
    fn all_filter_types_preserve_length() {
        let audio: Vec<f64> = (0..500).map(|i| (i as f64 * 0.01).sin()).collect();
        for ft in 0..=4 {
            let c = compute_biquad_coeffs(ft, 1000.0, 0.707);
            let out = biquad_process(&audio, &c);
            assert_eq!(out.len(), audio.len(), "filter_type {ft} changed length");
        }
    }

    #[test]
    fn zero_length_input_does_not_panic() {
        let empty: Vec<f64> = vec![];
        for ft in 0..=4 {
            let c = compute_biquad_coeffs(ft, 1000.0, 0.707);
            let out = biquad_process(&empty, &c);
            assert!(out.is_empty());

            let mut inplace: Vec<f64> = vec![];
            biquad_process_inplace(&mut inplace, &c);
            assert!(inplace.is_empty());
        }
    }
}
