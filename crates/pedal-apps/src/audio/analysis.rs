/// Audio quality metrics — port of shared/analysis.py.
///
/// Computes RT60, EDT, spectral centroid, echo density, C50/C80,
/// crest factor, octave-band RT60, spectral flatness, bandwidth,
/// and optional dry-vs-wet comparison metrics.
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AudioMetrics {
    pub rms_db: Option<f64>,
    pub peak_db: Option<f64>,
    pub rt60: Option<f64>,
    pub edt: Option<f64>,
    pub spectral_centroid: Option<f64>,
    pub echo_density: Option<f64>,
    pub c50: Option<f64>,
    pub c80: Option<f64>,
    pub crest_factor: Option<f64>,
    pub rt60_bands: HashMap<String, Option<f64>>,
    pub spectral_flatness: Option<f64>,
    pub bandwidth: Option<f64>,
    // Comparison metrics (only when reference provided)
    pub spectral_centroid_shift: Option<f64>,
    pub energy_ratio_db: Option<f64>,
    pub thd_n_percent: Option<f64>,
    pub bandwidth_change: Option<f64>,
}

/// Analyze audio quality metrics.
///
/// Port of shared/analysis.py analyze().
pub fn analyze(audio: &[f64], sr: u32, reference: Option<&[f64]>) -> AudioMetrics {
    let mono = to_mono_slice(audio);
    if mono.len() < 256 {
        return AudioMetrics::default();
    }

    let rms = rms_value(&mono);
    let peak = peak_value(&mono);

    let mut result = AudioMetrics {
        rms_db: if rms > 1e-12 {
            Some(round1(20.0 * rms.log10()))
        } else {
            None
        },
        peak_db: if peak > 1e-12 {
            Some(round1(20.0 * peak.log10()))
        } else {
            None
        },
        rt60: find_rt60(&mono, sr),
        edt: find_edt(&mono, sr),
        spectral_centroid: spectral_centroid(&mono, sr),
        echo_density: echo_density(&mono, sr),
        crest_factor: crest_factor(&mono),
        rt60_bands: octave_band_rt60(&mono, sr),
        spectral_flatness: spectral_flatness_value(&mono),
        bandwidth: bandwidth_value(&mono, sr),
        ..Default::default()
    };
    let (c50, c80) = clarity(&mono, sr);
    result.c50 = c50;
    result.c80 = c80;

    if let Some(ref_audio) = reference {
        let ref_mono = to_mono_slice(ref_audio);
        if ref_mono.len() >= 256 {
            let ref_centroid = spectral_centroid(&ref_mono, sr);
            result.spectral_centroid_shift = match (result.spectral_centroid, ref_centroid) {
                (Some(a), Some(b)) => Some(a - b),
                _ => None,
            };

            let ref_rms = rms_value(&ref_mono);
            let wet_rms = rms;
            result.energy_ratio_db = if ref_rms > 1e-12 {
                Some(round2(20.0 * (wet_rms / ref_rms).log10()))
            } else {
                None
            };

            let min_len = mono.len().min(ref_mono.len());
            let diff_rms = {
                let mut sum = 0.0;
                for i in 0..min_len {
                    let d = mono[i] - ref_mono[i];
                    sum += d * d;
                }
                (sum / min_len as f64).sqrt()
            };
            result.thd_n_percent = if ref_rms > 1e-12 {
                Some(round2(100.0 * diff_rms / ref_rms))
            } else {
                None
            };

            let ref_bw = bandwidth_value(&ref_mono, sr);
            result.bandwidth_change = match (result.bandwidth, ref_bw) {
                (Some(a), Some(b)) => Some(round1(a - b)),
                _ => None,
            };
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

fn to_mono_slice(audio: &[f64]) -> Vec<f64> {
    // Assume input is already mono (callers should use AudioBuffer::to_mono)
    audio.to_vec()
}

fn rms_value(mono: &[f64]) -> f64 {
    let sum: f64 = mono.iter().map(|s| s * s).sum();
    (sum / mono.len() as f64).sqrt()
}

fn peak_value(mono: &[f64]) -> f64 {
    mono.iter().map(|s| s.abs()).fold(0.0_f64, f64::max)
}

fn round1(v: f64) -> f64 {
    (v * 10.0).round() / 10.0
}

fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

fn round3(v: f64) -> f64 {
    (v * 1000.0).round() / 1000.0
}

fn round4(v: f64) -> f64 {
    (v * 10000.0).round() / 10000.0
}

/// Backward-integrated energy decay curve (dB), normalized to 0 dB start.
fn schroeder_decay(mono: &[f64]) -> Vec<f64> {
    let n = mono.len();
    let mut energy: Vec<f64> = mono.iter().map(|s| s * s).collect();

    // Cumulative sum from end to start
    for i in (0..n - 1).rev() {
        energy[i] += energy[i + 1];
    }

    let max_e = energy[0].max(1e-30);
    energy
        .iter()
        .map(|e| 10.0 * (e.max(1e-30) / max_e).log10())
        .collect()
}

/// T30 extrapolation: fit line from -5 to -35 dB, extrapolate to -60 dB.
fn find_rt60(mono: &[f64], sr: u32) -> Option<f64> {
    let db = schroeder_decay(mono);

    // Find indices where decay crosses -5 dB and -35 dB
    let i5 = db.iter().position(|&v| v <= -5.0)?;
    let i35 = db.iter().position(|&v| v <= -35.0)?;

    if i5 >= i35 {
        return None;
    }

    let (slope, _intercept) = linregress(&db, i5, i35, sr)?;
    if slope >= 0.0 {
        return None;
    }

    let rt60 = -60.0 / slope;
    Some(round3(rt60.max(0.0)))
}

/// Early Decay Time: fit from 0 to -10 dB, extrapolate to -60 dB.
fn find_edt(mono: &[f64], sr: u32) -> Option<f64> {
    let db = schroeder_decay(mono);
    let i10 = db.iter().position(|&v| v <= -10.0)?;

    if i10 == 0 {
        return None;
    }

    let (slope, _intercept) = linregress(&db, 0, i10, sr)?;
    if slope >= 0.0 {
        return None;
    }

    let edt = -60.0 / slope;
    Some(round3(edt.max(0.0)))
}

/// Simple linear regression on decay curve segment.
/// Returns (slope, intercept) where x is in seconds.
fn linregress(db: &[f64], i_start: usize, i_end: usize, sr: u32) -> Option<(f64, f64)> {
    let n = i_end - i_start;
    if n < 3 {
        return None;
    }

    let sr_f = sr as f64;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xy = 0.0;
    let mut sum_xx = 0.0;
    let nf = n as f64;

    for (i, &y) in db.iter().enumerate().take(i_end).skip(i_start) {
        let x = i as f64 / sr_f;
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_xx += x * x;
    }

    let denom = nf * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-30 {
        return None;
    }

    let slope = (nf * sum_xy - sum_x * sum_y) / denom;
    let intercept = (sum_y - slope * sum_x) / nf;
    Some((slope, intercept))
}

/// Brightness: weighted mean of FFT frequencies (Hz).
fn spectral_centroid(mono: &[f64], sr: u32) -> Option<f64> {
    let n = mono.len();
    let windowed = apply_hann(mono);
    let spectrum = rfft_magnitude(&windowed);
    let freqs = rfft_freqs(n, sr);

    let total: f64 = spectrum.iter().sum();
    if total < 1e-12 {
        return None;
    }

    let centroid: f64 = freqs
        .iter()
        .zip(spectrum.iter())
        .map(|(f, m)| f * m)
        .sum::<f64>()
        / total;
    Some(round1(centroid))
}

/// Normalized echo density (0-1) via sliding-window std-dev.
fn echo_density(mono: &[f64], sr: u32) -> Option<f64> {
    let win_samples = (sr as f64 * 0.001).max(1.0) as usize;
    let n = mono.len();
    if n < win_samples * 10 {
        return None;
    }

    let n_windows = n / win_samples;
    let mut stds = Vec::with_capacity(n_windows);
    for w in 0..n_windows {
        let start = w * win_samples;
        let end = start + win_samples;
        let mean: f64 = mono[start..end].iter().sum::<f64>() / win_samples as f64;
        let var: f64 =
            mono[start..end].iter().map(|s| (s - mean).powi(2)).sum::<f64>() / win_samples as f64;
        stds.push(var.sqrt());
    }

    let max_std = stds.iter().cloned().fold(0.0_f64, f64::max);
    let threshold = max_std * 0.05;
    if threshold < 1e-12 {
        return Some(0.0);
    }

    let count = stds.iter().filter(|&&s| s > threshold).count();
    Some(round3(count as f64 / stds.len() as f64))
}

/// C50 and C80: early-to-late energy ratio in dB.
fn clarity(mono: &[f64], sr: u32) -> (Option<f64>, Option<f64>) {
    let energy: Vec<f64> = mono.iter().map(|s| s * s).collect();

    let n50 = ((sr as f64 * 0.050) as usize).min(energy.len());
    let n80 = ((sr as f64 * 0.080) as usize).min(energy.len());

    let early50: f64 = energy[..n50].iter().sum();
    let late50: f64 = energy[n50..].iter().sum();
    let early80: f64 = energy[..n80].iter().sum();
    let late80: f64 = energy[n80..].iter().sum();

    let c50 = if late50 > 1e-30 {
        Some(round2(10.0 * (early50 / late50).log10()))
    } else {
        None
    };
    let c80 = if late80 > 1e-30 {
        Some(round2(10.0 * (early80 / late80).log10()))
    } else {
        None
    };

    (c50, c80)
}

/// Peak-to-RMS ratio in dB.
fn crest_factor(mono: &[f64]) -> Option<f64> {
    let rms = rms_value(mono);
    let peak = peak_value(mono);
    if rms < 1e-12 {
        return None;
    }
    Some(round2(20.0 * (peak / rms).log10()))
}

/// RT60 at standard octave bands: 125, 250, 500, 1k, 2k, 4k, 8k Hz.
fn octave_band_rt60(mono: &[f64], sr: u32) -> HashMap<String, Option<f64>> {
    let bands = [125, 250, 500, 1000, 2000, 4000, 8000];
    let nyquist = sr as f64 / 2.0;
    let mut result = HashMap::new();

    for &fc in &bands {
        let lo = fc as f64 / std::f64::consts::SQRT_2;
        let hi = fc as f64 * std::f64::consts::SQRT_2;

        if hi >= nyquist * 0.95 {
            continue;
        }

        let filtered = bandpass_filter(mono, lo, hi, sr, 4);
        let rt = find_rt60(&filtered, sr);
        result.insert(fc.to_string(), rt);
    }

    result
}

/// 2nd-order Butterworth biquad bandpass filter (cascaded for higher order).
fn bandpass_filter(input: &[f64], lo: f64, hi: f64, sr: u32, order: usize) -> Vec<f64> {
    let sections = order / 2;
    let mut data = input.to_vec();

    for _ in 0..sections {
        data = apply_biquad_bandpass(&data, lo, hi, sr);
    }

    data
}

/// Single 2nd-order bandpass biquad (Butterworth-like).
fn apply_biquad_bandpass(input: &[f64], lo: f64, hi: f64, sr: u32) -> Vec<f64> {
    let fs = sr as f64;
    let f0 = (lo * hi).sqrt();
    let bw = (hi / lo).ln() / (2.0_f64.ln()) * 0.5; // approximate bandwidth in octaves

    let w0 = 2.0 * std::f64::consts::PI * f0 / fs;
    let alpha = (w0 / 2.0).sin() * (2.0_f64.ln() / 2.0 * bw * w0 / w0.sin()).sinh();

    let b0 = alpha;
    let b1 = 0.0;
    let b2 = -alpha;
    let a0 = 1.0 + alpha;
    let a1 = -2.0 * w0.cos();
    let a2 = 1.0 - alpha;

    // Normalize
    let b0 = b0 / a0;
    let b1 = b1 / a0;
    let b2 = b2 / a0;
    let a1 = a1 / a0;
    let a2 = a2 / a0;

    let mut output = Vec::with_capacity(input.len());
    let mut x1 = 0.0;
    let mut x2 = 0.0;
    let mut y1 = 0.0;
    let mut y2 = 0.0;

    for &x in input {
        let y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
        output.push(y);
        x2 = x1;
        x1 = x;
        y2 = y1;
        y1 = y;
    }

    output
}

/// Wiener entropy: geometric/arithmetic mean of power spectrum. 0=tonal, 1=noise.
fn spectral_flatness_value(mono: &[f64]) -> Option<f64> {
    let spectrum = rfft_magnitude(mono);
    // Skip DC, use power spectrum
    let power: Vec<f64> = spectrum[1..]
        .iter()
        .map(|m| (m * m).max(1e-30))
        .collect();

    if power.is_empty() {
        return Some(0.0);
    }

    let log_mean: f64 = power.iter().map(|p| p.ln()).sum::<f64>() / power.len() as f64;
    let geo_mean = log_mean.exp();
    let arith_mean: f64 = power.iter().sum::<f64>() / power.len() as f64;

    if arith_mean < 1e-30 {
        return Some(0.0);
    }

    let flatness = (geo_mean / arith_mean).clamp(0.0, 1.0);
    Some(round4(flatness))
}

/// Frequency at -3dB from spectral peak (Hz).
fn bandwidth_value(mono: &[f64], sr: u32) -> Option<f64> {
    let n = mono.len();
    let spectrum = rfft_magnitude(mono);
    let freqs = rfft_freqs(n, sr);

    if spectrum.len() < 2 {
        return None;
    }

    let peak_mag = spectrum.iter().cloned().fold(0.0_f64, f64::max);
    if peak_mag < 1e-12 {
        return None;
    }

    let threshold = peak_mag * 10.0_f64.powf(-3.0 / 20.0); // -3 dB below peak

    // Find highest frequency above threshold
    let last_above = spectrum
        .iter()
        .enumerate()
        .rev()
        .find(|(_, &m)| m >= threshold);

    last_above.map(|(idx, _)| round1(freqs[idx]))
}

// ---------------------------------------------------------------------------
// FFT helpers
// ---------------------------------------------------------------------------

fn apply_hann(data: &[f64]) -> Vec<f64> {
    let n = data.len();
    data.iter()
        .enumerate()
        .map(|(i, &s)| {
            let w = 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos());
            s * w
        })
        .collect()
}

/// Compute |FFT| magnitudes for real input.
fn rfft_magnitude(data: &[f64]) -> Vec<f64> {
    use realfft::RealFftPlanner;

    let n = data.len();
    if n == 0 {
        return vec![];
    }

    let mut planner = RealFftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);

    let mut input = data.to_vec();
    let mut spectrum = fft.make_output_vec();
    fft.process(&mut input, &mut spectrum).ok();

    spectrum.iter().map(|c| c.norm()).collect()
}

/// Compute FFT frequency bins for real input.
fn rfft_freqs(n: usize, sr: u32) -> Vec<f64> {
    let n_bins = n / 2 + 1;
    let df = sr as f64 / n as f64;
    (0..n_bins).map(|i| i as f64 * df).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_impulse(sr: u32, seconds: f64) -> Vec<f64> {
        let n = (sr as f64 * seconds) as usize;
        let mut data = vec![0.0; n];
        data[0] = 1.0;
        data
    }

    #[test]
    fn test_analyze_impulse() {
        let impulse = make_test_impulse(44100, 1.0);
        let metrics = analyze(&impulse, 44100, None);
        // An impulse should have measurable peak/rms
        assert!(metrics.peak_db.is_some());
        assert!(metrics.rms_db.is_some());
        // Single-sample impulse has flat spectrum -> spectral_centroid is valid
        // but the energy is concentrated in one sample, so centroid may be None
        // if total magnitude is too low after windowing
    }

    #[test]
    fn test_analyze_sine() {
        let sr = 44100;
        let n = sr * 2;
        let freq = 440.0;
        let sine: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / sr as f64).sin() * 0.5)
            .collect();
        let metrics = analyze(&sine, sr as u32, None);
        // A sine wave should have low spectral flatness (tonal)
        assert!(metrics.spectral_flatness.unwrap() < 0.1);
        assert!(metrics.spectral_centroid.is_some());
    }

    #[test]
    fn test_analyze_with_reference() {
        let impulse = make_test_impulse(44100, 1.0);
        let scaled: Vec<f64> = impulse.iter().map(|s| s * 0.5).collect();
        let metrics = analyze(&scaled, 44100, Some(&impulse));
        assert!(metrics.energy_ratio_db.is_some());
        assert!(metrics.thd_n_percent.is_some());
    }

    #[test]
    fn test_schroeder_decay() {
        let impulse = make_test_impulse(44100, 0.1);
        let db = schroeder_decay(&impulse);
        assert!((db[0] - 0.0).abs() < 0.01);
        // Decay should be negative after the impulse
        assert!(db[100] < 0.0);
    }

    #[test]
    fn test_clarity() {
        // A pure impulse has zero late energy, so clarity is None.
        // Use a signal with energy in both early and late portions.
        let sr = 44100_u32;
        let n = sr as usize;
        let mut signal = vec![0.0; n];
        signal[0] = 1.0; // early energy
        // Add some late energy (noise floor)
        for i in (sr as usize / 10)..n {
            signal[i] = 0.001 * ((i as f64 * 0.1).sin());
        }
        let (c50, c80) = clarity(&signal, sr);
        // Now both early and late have energy, so clarity should be defined
        assert!(c50.is_some());
        assert!(c80.is_some());
        assert!(c50.unwrap() > 0.0);
    }

    #[test]
    fn test_spectral_flatness() {
        // White noise should have high flatness
        let n = 44100;
        let noise: Vec<f64> = (0..n)
            .map(|i| {
                // Simple pseudo-random
                let x = (i as f64 * 0.123456789).sin() * 43758.5453;
                x - x.floor() - 0.5
            })
            .collect();
        let flatness = spectral_flatness_value(&noise).unwrap();
        assert!(flatness > 0.3, "noise flatness={flatness}");
    }

    #[test]
    fn test_octave_band_rt60() {
        let impulse = make_test_impulse(44100, 0.5);
        let bands = octave_band_rt60(&impulse, 44100);
        // Should have entries for the standard bands below Nyquist
        assert!(bands.contains_key("1000"));
        assert!(bands.contains_key("4000"));
    }
}
