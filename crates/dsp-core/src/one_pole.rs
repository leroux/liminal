//! One-pole lowpass and highpass filters (in-place).

use crate::SR;

/// In-place one-pole lowpass filter.
pub fn lowpass(audio: &mut [f64], cutoff_hz: f64) {
    if audio.is_empty() {
        return;
    }
    let cutoff = cutoff_hz.clamp(20.0, SR / 2.0 - 1.0);
    let rc = 1.0 / (2.0 * std::f64::consts::PI * cutoff);
    let dt = 1.0 / SR;
    let alpha = dt / (rc + dt);
    let mut prev = 0.0_f64;
    for s in audio.iter_mut() {
        prev += alpha * (*s - prev);
        *s = prev;
    }
}

/// In-place one-pole highpass filter.
pub fn highpass(audio: &mut [f64], cutoff_hz: f64) {
    if audio.is_empty() {
        return;
    }
    let cutoff = cutoff_hz.clamp(20.0, SR / 2.0 - 1.0);
    let rc = 1.0 / (2.0 * std::f64::consts::PI * cutoff);
    let dt = 1.0 / SR;
    let alpha = rc / (rc + dt);
    let mut prev_in = 0.0_f64;
    let mut prev_out = 0.0_f64;
    for s in audio.iter_mut() {
        let inp = *s;
        prev_out = alpha * (prev_out + inp - prev_in);
        prev_in = inp;
        *s = prev_out;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lowpass_attenuates_high_freq() {
        let n = 44100;
        let audio: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / SR;
                (2.0 * std::f64::consts::PI * 10000.0 * t).sin()
            })
            .collect();
        let rms_in: f64 = (audio.iter().map(|x| x * x).sum::<f64>() / n as f64).sqrt();
        let mut filtered = audio;
        lowpass(&mut filtered, 1000.0);
        let rms_out: f64 = (filtered.iter().map(|x| x * x).sum::<f64>() / n as f64).sqrt();
        assert!(rms_out < rms_in * 0.3);
    }

    #[test]
    fn highpass_attenuates_low_freq() {
        let n = 44100;
        let audio: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / SR;
                (2.0 * std::f64::consts::PI * 100.0 * t).sin()
            })
            .collect();
        let rms_in: f64 = (audio.iter().map(|x| x * x).sum::<f64>() / n as f64).sqrt();
        let mut filtered = audio;
        highpass(&mut filtered, 5000.0);
        let rms_out: f64 = (filtered.iter().map(|x| x * x).sum::<f64>() / n as f64).sqrt();
        assert!(rms_out < rms_in * 0.3);
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
    fn lowpass_passes_low_freq() {
        let n = 44100;
        let low = sine(100.0, n);
        let rms_in = rms(&low);
        let mut filtered = low;
        lowpass(&mut filtered, 1000.0);
        let rms_out = rms(&filtered);
        // 100 Hz through a 1 kHz lowpass: should have minimal attenuation
        assert!(
            rms_out > rms_in * 0.9,
            "expected minimal attenuation, got ratio {}",
            rms_out / rms_in
        );
    }

    #[test]
    fn highpass_passes_high_freq() {
        let n = 44100;
        let high = sine(10000.0, n);
        let rms_in = rms(&high);
        let mut filtered = high;
        highpass(&mut filtered, 1000.0);
        let rms_out = rms(&filtered);
        // 10 kHz through a 1 kHz highpass: should have minimal attenuation
        assert!(
            rms_out > rms_in * 0.8,
            "expected minimal attenuation, got ratio {}",
            rms_out / rms_in
        );
    }

    #[test]
    fn lowpass_cutoff_at_minimum() {
        // cutoff clamped to 20 Hz
        let n = 44100;
        let high = sine(5000.0, n);
        let rms_in = rms(&high);
        let mut filtered = high;
        lowpass(&mut filtered, 5.0); // will be clamped to 20 Hz
        let rms_out = rms(&filtered);
        // 5 kHz through a 20 Hz lowpass should be heavily attenuated
        assert!(rms_out < rms_in * 0.05);
    }

    #[test]
    fn highpass_cutoff_at_minimum() {
        // cutoff clamped to 20 Hz — a 20 Hz highpass should pass nearly everything
        let n = 44100;
        let audio = sine(1000.0, n);
        let rms_in = rms(&audio);
        let mut filtered = audio;
        highpass(&mut filtered, 5.0); // clamped to 20 Hz
        let rms_out = rms(&filtered);
        assert!(
            rms_out > rms_in * 0.9,
            "20 Hz highpass should barely touch 1 kHz signal"
        );
    }

    #[test]
    fn lowpass_cutoff_at_maximum() {
        // cutoff at SR/2 - 1: should be near-bypass
        let n = 44100;
        let audio = sine(1000.0, n);
        let rms_in = rms(&audio);
        let mut filtered = audio;
        lowpass(&mut filtered, SR / 2.0 - 1.0);
        let rms_out = rms(&filtered);
        assert!(
            rms_out > rms_in * 0.95,
            "max-cutoff lowpass should be near-bypass, got ratio {}",
            rms_out / rms_in
        );
    }

    #[test]
    fn highpass_cutoff_at_maximum() {
        // cutoff at SR/2 - 1: should attenuate everything
        let n = 44100;
        let audio = sine(1000.0, n);
        let rms_in = rms(&audio);
        let mut filtered = audio;
        highpass(&mut filtered, SR / 2.0 - 1.0);
        let rms_out = rms(&filtered);
        assert!(
            rms_out < rms_in * 0.3,
            "max-cutoff highpass should attenuate 1 kHz, got ratio {}",
            rms_out / rms_in
        );
    }

    #[test]
    fn single_sample_does_not_panic() {
        let mut one_sample = vec![0.5];
        lowpass(&mut one_sample, 1000.0);
        // After one sample of lowpass, prev starts at 0 so output = alpha * 0.5
        assert!(one_sample[0].is_finite());
        assert!(one_sample[0] > 0.0);

        let mut one_sample = vec![0.5];
        highpass(&mut one_sample, 1000.0);
        // First sample: prev_out = alpha*(0 + 0.5 - 0) = alpha * 0.5
        assert!(one_sample[0].is_finite());
    }

    #[test]
    fn empty_input_does_not_panic() {
        let mut empty: Vec<f64> = vec![];
        lowpass(&mut empty, 1000.0);
        assert!(empty.is_empty());

        let mut empty: Vec<f64> = vec![];
        highpass(&mut empty, 1000.0);
        assert!(empty.is_empty());
    }
}
