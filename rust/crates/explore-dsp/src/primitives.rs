//! Shared DSP primitives and post-processing utilities.

use std::f32::consts::PI;

/// Remove DC offset by subtracting mean.
pub fn dc_remove(signal: &[f32]) -> Vec<f32> {
    let mean: f32 = signal.iter().sum::<f32>() / signal.len() as f32;
    signal.iter().map(|&s| s - mean).collect()
}

/// Peak normalize to specified dB level.
pub fn normalize(signal: &mut [f32], peak_db: f32) {
    let peak = signal.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    if peak < 1e-10 {
        return;
    }
    let target = 10.0f32.powf(peak_db / 20.0);
    let scale = target / peak;
    for s in signal.iter_mut() {
        *s *= scale;
    }
}

/// Soft limiter using tanh curve above threshold.
pub fn soft_limit(signal: &mut [f32], threshold: f32) {
    for s in signal.iter_mut() {
        let ax = s.abs();
        if ax > threshold {
            *s = s.signum() * (threshold + (ax - threshold).tanh() * (1.0 - threshold));
        }
    }
}

/// 2nd order Butterworth lowpass (direct form II transposed biquad).
pub fn gentle_lowpass(signal: &mut [f32], sr: u32, cutoff: f32) {
    let nyq = sr as f32 / 2.0;
    if cutoff >= nyq {
        return;
    }
    let (b0, b1, b2, a1, a2) = biquad_coeffs_lpf(cutoff, sr, 0.707);
    let filtered = biquad_filter(signal, b0, b1, b2, a1, a2);
    signal.copy_from_slice(&filtered);
}

/// Linear fade at start and end to avoid clicks.
pub fn fade_in_out(signal: &mut [f32], fade_samples: usize) {
    let n = signal.len();
    let fade_in = fade_samples.min(n);
    let fade_out = fade_samples.min(n);
    for i in 0..fade_in {
        signal[i] *= i as f32 / fade_in as f32;
    }
    for i in 0..fade_out {
        signal[n - 1 - i] *= i as f32 / fade_out as f32;
    }
}

/// Full post-processing chain: dc_remove -> gentle_lowpass -> soft_limit -> normalize -> fade_in_out
pub fn post_process(signal: &[f32], sr: u32) -> Vec<f32> {
    let mut s = dc_remove(signal);
    gentle_lowpass(&mut s, sr, 16000.0);
    soft_limit(&mut s, 0.9);
    normalize(&mut s, -1.0);
    fade_in_out(&mut s, 256);
    s
}

/// Post-process stereo signal.
pub fn post_process_stereo(signal: &[[f32; 2]], sr: u32) -> Vec<[f32; 2]> {
    let left: Vec<f32> = signal.iter().map(|s| s[0]).collect();
    let right: Vec<f32> = signal.iter().map(|s| s[1]).collect();
    let left = post_process(&left, sr);
    let right = post_process(&right, sr);
    left.into_iter().zip(right).map(|(l, r)| [l, r]).collect()
}

/// Crossfade between dry and wet signal.
pub fn mix_wet_dry(dry: &[f32], wet: &[f32], mix: f32) -> Vec<f32> {
    let n = dry.len().min(wet.len());
    (0..n).map(|i| (1.0 - mix) * dry[i] + mix * wet[i]).collect()
}

/// Direct Form II Transposed biquad filter.
pub fn biquad_filter(samples: &[f32], b0: f32, b1: f32, b2: f32, a1: f32, a2: f32) -> Vec<f32> {
    let n = samples.len();
    let mut out = vec![0.0f32; n];
    let mut z1 = 0.0f32;
    let mut z2 = 0.0f32;
    for i in 0..n {
        let x = samples[i];
        let y = b0 * x + z1;
        z1 = b1 * x - a1 * y + z2;
        z2 = b2 * x - a2 * y;
        out[i] = y;
    }
    out
}

/// Compute biquad coefficients for lowpass filter.
pub fn biquad_coeffs_lpf(freq_hz: f32, sr: u32, q: f32) -> (f32, f32, f32, f32, f32) {
    let w0 = 2.0 * PI * freq_hz / sr as f32;
    let alpha = w0.sin() / (2.0 * q);
    let cos_w0 = w0.cos();
    let b0 = (1.0 - cos_w0) / 2.0;
    let b1 = 1.0 - cos_w0;
    let b2 = (1.0 - cos_w0) / 2.0;
    let a0 = 1.0 + alpha;
    let a1 = -2.0 * cos_w0;
    let a2 = 1.0 - alpha;
    (b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)
}

/// Compute biquad coefficients for highpass filter.
pub fn biquad_coeffs_hpf(freq_hz: f32, sr: u32, q: f32) -> (f32, f32, f32, f32, f32) {
    let w0 = 2.0 * PI * freq_hz / sr as f32;
    let alpha = w0.sin() / (2.0 * q);
    let cos_w0 = w0.cos();
    let b0 = (1.0 + cos_w0) / 2.0;
    let b1 = -(1.0 + cos_w0);
    let b2 = (1.0 + cos_w0) / 2.0;
    let a0 = 1.0 + alpha;
    let a1 = -2.0 * cos_w0;
    let a2 = 1.0 - alpha;
    (b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)
}

/// Compute biquad coefficients for bandpass filter.
pub fn biquad_coeffs_bpf(freq_hz: f32, sr: u32, q: f32) -> (f32, f32, f32, f32, f32) {
    let w0 = 2.0 * PI * freq_hz / sr as f32;
    let alpha = w0.sin() / (2.0 * q);
    let cos_w0 = w0.cos();
    let b0 = alpha;
    let b1 = 0.0;
    let b2 = -alpha;
    let a0 = 1.0 + alpha;
    let a1 = -2.0 * cos_w0;
    let a2 = 1.0 - alpha;
    (b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)
}

/// Compute biquad coefficients for notch filter.
pub fn biquad_coeffs_notch(freq_hz: f32, sr: u32, q: f32) -> (f32, f32, f32, f32, f32) {
    let w0 = 2.0 * PI * freq_hz / sr as f32;
    let alpha = w0.sin() / (2.0 * q);
    let cos_w0 = w0.cos();
    let a0 = 1.0 + alpha;
    let a1 = -2.0 * cos_w0;
    let a2 = 1.0 - alpha;
    (1.0 / a0, a1 / a0, 1.0 / a0, a1 / a0, a2 / a0)
}

/// Compute biquad coefficients for peaking EQ filter.
pub fn biquad_coeffs_peak(freq_hz: f32, sr: u32, q: f32, gain_db: f32) -> (f32, f32, f32, f32, f32) {
    let a_lin = 10.0f32.powf(gain_db / 40.0);
    let w0 = 2.0 * PI * freq_hz / sr as f32;
    let alpha = w0.sin() / (2.0 * q);
    let cos_w0 = w0.cos();
    let b0 = 1.0 + alpha * a_lin;
    let b1 = -2.0 * cos_w0;
    let b2 = 1.0 - alpha * a_lin;
    let a0 = 1.0 + alpha / a_lin;
    let a1 = -2.0 * cos_w0;
    let a2 = 1.0 - alpha / a_lin;
    (b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)
}

/// One-pole lowpass filter. coeff in [0, 1), higher = more smoothing.
pub fn one_pole_lp(samples: &[f32], coeff: f32) -> Vec<f32> {
    let n = samples.len();
    let mut out = vec![0.0f32; n];
    let mut prev = 0.0f32;
    for i in 0..n {
        prev = coeff * prev + (1.0 - coeff) * samples[i];
        out[i] = prev;
    }
    out
}

/// Envelope follower with separate attack and release.
pub fn envelope_follower(samples: &[f32], attack_coeff: f32, release_coeff: f32) -> Vec<f32> {
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

/// Convert milliseconds to one-pole filter coefficient.
pub fn ms_to_coeff(ms: f32, sr: u32) -> f32 {
    if ms <= 0.0 {
        return 0.0;
    }
    (-1.0 / (ms * 0.001 * sr as f32)).exp()
}

/// Simple LCG pseudo-random number generator.
pub struct Lcg {
    state: u64,
}

impl Lcg {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Returns value in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let val = ((self.state >> 33) & 0x7FFFFFFF) as f32 / 2147483647.0;
        val
    }

    /// Returns value in [-1, 1).
    pub fn next_bipolar(&mut self) -> f32 {
        self.next_f32() * 2.0 - 1.0
    }
}

/// Generate a Hann window of given size.
pub fn hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (size - 1) as f32).cos()))
        .collect()
}

/// Generate the first `count` prime numbers.
pub fn gen_primes(count: usize) -> Vec<usize> {
    let mut primes = Vec::with_capacity(count);
    let mut candidate = 2;
    while primes.len() < count {
        let is_prime = primes.iter().all(|&p| {
            if p * p > candidate { return true; }
            candidate % p != 0
        });
        if is_prime {
            primes.push(candidate);
        }
        candidate += 1;
    }
    primes
}
