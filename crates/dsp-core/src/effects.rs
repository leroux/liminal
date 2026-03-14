//! Common audio effects: noise gate, peak limiter, bitcrusher, decimator.
//!
//! All functions take raw numeric parameters (not param structs).

/// RMS-based noise gate, in-place.
pub fn noise_gate_inplace(audio: &mut [f64], threshold: f64) {
    if threshold <= 0.0 {
        return;
    }
    let n = audio.len();
    let win = 512_usize; // ~11 ms gate window
    let mut start = 0;
    while start < n {
        let end = (start + win).min(n);
        let mut s = 0.0_f64;
        for sample in audio.iter().take(end).skip(start) {
            s += sample * sample;
        }
        let rms = (s / (end - start) as f64).sqrt();
        if rms < threshold {
            let gain = rms / threshold;
            for sample in audio.iter_mut().take(end).skip(start) {
                *sample *= gain;
            }
        }
        start = end;
    }
}

/// RMS-based noise gate, allocating.
pub fn noise_gate(audio: &[f64], threshold: f64) -> Vec<f64> {
    let mut out = audio.to_vec();
    noise_gate_inplace(&mut out, threshold);
    out
}

/// Peak limiter, in-place. Scales signal so peak <= ceiling.
pub fn peak_limiter_inplace(audio: &mut [f64], ceiling: f64) {
    let peak = audio.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
    if peak > ceiling && peak > 0.0 {
        let scale = ceiling / peak;
        for s in audio.iter_mut() {
            *s *= scale;
        }
    }
}

/// Peak limiter, allocating.
pub fn peak_limiter(audio: &[f64], ceiling: f64) -> Vec<f64> {
    let mut out = audio.to_vec();
    peak_limiter_inplace(&mut out, ceiling);
    out
}

/// Bitcrusher in-place. `crush` 0..1: 0=off, 1=extreme (16 down to 4 bits).
pub fn crush_inplace(audio: &mut [f64], crush: f64) {
    if crush <= 0.0 {
        return;
    }
    let bits = 16.0 - 12.0 * crush;
    let quant = (2.0_f64).powf(bits - 1.0);
    for s in audio.iter_mut() {
        *s = (*s * quant + 0.5).floor() / quant;
    }
}

/// Zero-order hold decimator in-place. `decimate` 0..1: 0=off, 1=extreme (hold 32 samples).
pub fn decimate_inplace(audio: &mut [f64], decimate: f64) {
    if decimate <= 0.0 {
        return;
    }
    let rate_factor = 1.0 + 31.0 * decimate;
    let mut phase = 0.0_f64;
    let mut held = 0.0_f64;
    for s in audio.iter_mut() {
        phase += 1.0;
        if phase >= rate_factor {
            held = *s;
            phase -= rate_factor;
        }
        *s = held;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_silences_quiet() {
        let audio = vec![0.001_f64; 1000];
        let out = noise_gate(&audio, 0.5);
        let rms: f64 = (out.iter().map(|x| x * x).sum::<f64>() / out.len() as f64).sqrt();
        assert!(rms < 0.001);
    }

    #[test]
    fn limiter_caps_peak() {
        let audio = vec![2.0_f64; 100];
        let ceiling = 0.525;
        let out = peak_limiter(&audio, ceiling);
        let peak = out.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
        assert!(peak <= ceiling + 1e-10);
    }

    #[test]
    fn crush_reduces_precision() {
        let audio: Vec<f64> = (0..1000).map(|i| (i as f64 * 0.01).sin() * 0.5).collect();
        let mut out = audio.clone();
        crush_inplace(&mut out, 1.0);
        let unique: std::collections::HashSet<u64> = out.iter().map(|x| x.to_bits()).collect();
        assert!(unique.len() < audio.len());
    }

    #[test]
    fn decimate_holds_values() {
        let audio: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let mut out = audio.clone();
        decimate_inplace(&mut out, 1.0);
        let runs: usize = (1..out.len()).filter(|&i| out[i] == out[i - 1]).count();
        assert!(runs > 50);
    }

    #[test]
    fn noise_gate_threshold_zero_is_noop() {
        let audio = vec![0.5, -0.3, 0.8, -0.1, 0.0];
        let mut out = audio.clone();
        noise_gate_inplace(&mut out, 0.0);
        assert_eq!(out, audio);
    }

    #[test]
    fn noise_gate_loud_signal_passes_through() {
        // Signal well above threshold should be unchanged
        let audio: Vec<f64> = (0..1024).map(|i| (i as f64 * 0.1).sin() * 0.9).collect();
        let out = noise_gate(&audio, 0.01);
        // The RMS of a 0.9-amplitude sine is ~0.636, well above 0.01
        for (a, b) in audio.iter().zip(out.iter()) {
            assert!(
                (a - b).abs() < 1e-15,
                "Loud signal should pass through gate unchanged"
            );
        }
    }

    #[test]
    fn peak_limiter_below_ceiling_is_noop() {
        let audio = vec![0.1, -0.2, 0.3, -0.15];
        let mut out = audio.clone();
        peak_limiter_inplace(&mut out, 1.0);
        assert_eq!(out, audio);
    }

    #[test]
    fn peak_limiter_zero_ceiling() {
        let audio = vec![0.5, -0.3, 0.8, -0.1];
        let mut out = audio.clone();
        peak_limiter_inplace(&mut out, 0.0);
        // peak=0.8 > ceiling=0.0 but ceiling/peak = 0, so all should be 0
        for s in &out {
            assert_eq!(*s, 0.0);
        }
    }

    #[test]
    fn crush_zero_is_noop() {
        let audio: Vec<f64> = (0..100).map(|i| (i as f64 * 0.07).sin()).collect();
        let mut out = audio.clone();
        crush_inplace(&mut out, 0.0);
        assert_eq!(out, audio);
    }

    #[test]
    fn crush_intermediate_value() {
        let audio: Vec<f64> = (0..1000).map(|i| (i as f64 * 0.01).sin() * 0.5).collect();
        let mut out = audio.clone();
        crush_inplace(&mut out, 0.5);
        // Intermediate crush should quantize somewhat — fewer unique values than original
        let unique_orig: std::collections::HashSet<u64> =
            audio.iter().map(|x| x.to_bits()).collect();
        let unique_crushed: std::collections::HashSet<u64> =
            out.iter().map(|x| x.to_bits()).collect();
        assert!(unique_crushed.len() < unique_orig.len());
        // But less aggressively than crush=1.0
        let mut extreme = audio.clone();
        crush_inplace(&mut extreme, 1.0);
        let unique_extreme: std::collections::HashSet<u64> =
            extreme.iter().map(|x| x.to_bits()).collect();
        assert!(unique_crushed.len() > unique_extreme.len());
    }

    #[test]
    fn decimate_zero_is_noop() {
        let audio: Vec<f64> = (0..100).map(|i| i as f64 * 0.01).collect();
        let mut out = audio.clone();
        decimate_inplace(&mut out, 0.0);
        assert_eq!(out, audio);
    }

    #[test]
    fn empty_input_no_panic() {
        let mut empty: Vec<f64> = vec![];

        noise_gate_inplace(&mut empty, 0.5);
        assert!(empty.is_empty());

        let alloc_gate = noise_gate(&[], 0.5);
        assert!(alloc_gate.is_empty());

        peak_limiter_inplace(&mut empty, 0.5);
        assert!(empty.is_empty());

        let alloc_lim = peak_limiter(&[], 0.5);
        assert!(alloc_lim.is_empty());

        crush_inplace(&mut empty, 0.5);
        assert!(empty.is_empty());

        decimate_inplace(&mut empty, 0.5);
        assert!(empty.is_empty());
    }
}
