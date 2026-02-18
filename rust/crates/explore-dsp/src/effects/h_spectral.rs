//! H-series: FFT/spectral effects (H001-H019).

use std::collections::HashMap;
use serde_json::Value;
use crate::{AudioOutput, EffectEntry, pf, pi, ps, params};
use crate::primitives::*;
use crate::stft::{stft, istft};
use num_complex::Complex;

// ---------------------------------------------------------------------------
// Inline filter helpers (replacing scipy.ndimage)
// ---------------------------------------------------------------------------

/// Sliding window average (box filter) along axis=1 (frequency) of a 2D array.
/// `data` is row-major: data[frame][bin].
fn uniform_filter1d_axis1(data: &[Vec<f32>], size: usize) -> Vec<Vec<f32>> {
    let size = size.max(1);
    let half = (size / 2) as i32;
    data.iter()
        .map(|row| {
            let n = row.len();
            let mut out = vec![0.0f32; n];
            // Running sum approach
            let mut sum = 0.0f64;
            let mut count = 0u32;
            // Initialize window for index 0
            for j in 0..=(half as usize).min(n - 1) {
                sum += row[j] as f64;
                count += 1;
            }
            out[0] = (sum / count as f64) as f32;
            for i in 1..n {
                // Add new right element
                let right = i as i32 + half;
                if right >= 0 && (right as usize) < n {
                    sum += row[right as usize] as f64;
                    count += 1;
                }
                // Remove old left element
                let old_left = i as i32 - half - 1;
                if old_left >= 0 && (old_left as usize) < n {
                    sum -= row[old_left as usize] as f64;
                    count -= 1;
                }
                out[i] = (sum / count as f64) as f32;
            }
            out
        })
        .collect()
}

/// Sliding window median along a given axis of a 2D array.
/// For axis=0 (time), `kernel` is (filter_length, 1).
/// For axis=1 (frequency), `kernel` is (1, filter_length).
fn median_filter_2d(data: &[Vec<f32>], kernel: (usize, usize)) -> Vec<Vec<f32>> {
    let num_frames = data.len();
    if num_frames == 0 {
        return vec![];
    }
    let num_bins = data[0].len();
    let half_t = (kernel.0 / 2) as i32;
    let half_f = (kernel.1 / 2) as i32;

    let mut result = vec![vec![0.0f32; num_bins]; num_frames];

    for i in 0..num_frames {
        for j in 0..num_bins {
            let mut window = Vec::new();
            let t_start = (i as i32 - half_t).max(0) as usize;
            let t_end = ((i as i32 + half_t) as usize + 1).min(num_frames);
            let f_start = (j as i32 - half_f).max(0) as usize;
            let f_end = ((j as i32 + half_f) as usize + 1).min(num_bins);
            for ti in t_start..t_end {
                for fi in f_start..f_end {
                    window.push(data[ti][fi]);
                }
            }
            window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            result[i][j] = window[window.len() / 2];
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Simple PRNG for deterministic noise (matching numpy default_rng(42))
// ---------------------------------------------------------------------------

struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        // Use LCG seeded deterministically
        Self {
            state: seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407),
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }

    /// Uniform f32 in [0, 1)
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Uniform f32 in [lo, hi)
    fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_f32()
    }

    /// Approximate standard normal using Box-Muller
    fn standard_normal(&mut self) -> f32 {
        let u1 = self.next_f32().max(1e-10);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }

    /// Generate a permutation of 0..n
    fn permutation(&mut self, n: usize) -> Vec<usize> {
        let mut perm: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = (self.next_u64() as usize) % (i + 1);
            perm.swap(i, j);
        }
        perm
    }
}

// ---------------------------------------------------------------------------
// STFT helpers: extract magnitude/phase, reconstruct
// ---------------------------------------------------------------------------

fn mag_phase(frames: &[Vec<Complex<f32>>]) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mag: Vec<Vec<f32>> = frames
        .iter()
        .map(|row| row.iter().map(|c| c.norm()).collect())
        .collect();
    let phase: Vec<Vec<f32>> = frames
        .iter()
        .map(|row| row.iter().map(|c| c.arg()).collect())
        .collect();
    (mag, phase)
}

fn from_mag_phase(mag: &[Vec<f32>], phase: &[Vec<f32>]) -> Vec<Vec<Complex<f32>>> {
    mag.iter()
        .zip(phase.iter())
        .map(|(m_row, p_row)| {
            m_row
                .iter()
                .zip(p_row.iter())
                .map(|(&m, &p)| Complex::new(m * p.cos(), m * p.sin()))
                .collect()
        })
        .collect()
}

fn from_mag_zero_phase(mag: &[Vec<f32>]) -> Vec<Vec<Complex<f32>>> {
    mag.iter()
        .map(|m_row| m_row.iter().map(|&m| Complex::new(m, 0.0)).collect())
        .collect()
}

// ---------------------------------------------------------------------------
// Carrier generation for cross-synthesis / vocoder / transfer effects
// ---------------------------------------------------------------------------

fn make_carrier(source_type: &str, length: usize, sr: u32) -> Vec<f32> {
    let sr_f = sr as f32;
    match source_type {
        "noise" => {
            let mut rng = SimpleRng::new(42);
            (0..length).map(|_| rng.standard_normal() * 0.5).collect()
        }
        "sine_sweep" => {
            let duration = length as f32 / sr_f;
            let log_ratio = (8000.0f32 / 100.0).ln();
            (0..length)
                .map(|i| {
                    let t = i as f32 / sr_f;
                    let phase = 2.0 * std::f32::consts::PI * 100.0 * duration
                        * ((t / duration * log_ratio).exp() - 1.0)
                        / log_ratio;
                    0.5 * phase.sin()
                })
                .collect()
        }
        "sawtooth" | "saw" => {
            let freq = 100.0f32;
            (0..length)
                .map(|i| {
                    let t = i as f32 / sr_f;
                    let v = t * freq;
                    0.5 * 2.0 * (v - (v + 0.5).floor())
                })
                .collect()
        }
        "chirp" => {
            let duration = length as f32 / sr_f;
            let log_ratio = (6000.0f32 / 80.0).ln();
            (0..length)
                .map(|i| {
                    let t = i as f32 / sr_f;
                    let phase = 2.0 * std::f32::consts::PI * 80.0 * duration
                        * ((t / duration * log_ratio).exp() - 1.0)
                        / log_ratio;
                    0.5 * phase.sin()
                })
                .collect()
        }
        "pulse" => {
            let mut carrier = vec![0.0f32; length];
            let period = (sr_f / 100.0) as usize;
            if period > 0 {
                let mut i = 0;
                while i < length {
                    carrier[i] = 1.0;
                    i += period;
                }
            }
            carrier
        }
        "input_self" => {
            // Caller should handle this case; return silence as fallback
            vec![0.0f32; length]
        }
        _ => {
            let mut rng = SimpleRng::new(42);
            (0..length).map(|_| rng.standard_normal() * 0.5).collect()
        }
    }
}

// ---------------------------------------------------------------------------
// Percentile helper
// ---------------------------------------------------------------------------

fn percentile(data: &[f32], pct: f32) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f32> = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((pct / 100.0) * (sorted.len() - 1) as f32).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Percentile along axis=0 (per-bin across all frames) of a 2D array.
fn percentile_axis0(data: &[Vec<f32>], pct: f32) -> Vec<f32> {
    if data.is_empty() {
        return vec![];
    }
    let num_bins = data[0].len();
    let num_frames = data.len();
    let mut result = vec![0.0f32; num_bins];
    for b in 0..num_bins {
        let col: Vec<f32> = (0..num_frames).map(|f| data[f][b]).collect();
        result[b] = percentile(&col, pct);
    }
    result
}

// ---------------------------------------------------------------------------
// H001 -- Spectral Freeze
// ---------------------------------------------------------------------------

fn process_h001(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let freeze_position = pf(params, "freeze_position", 0.3);
    let fft_size = pi(params, "fft_size", 2048) as usize;
    let hop_size = fft_size / 4;

    let x = stft(samples, fft_size, hop_size);
    let num_frames = x.len();
    if num_frames == 0 {
        return AudioOutput::Mono(samples.to_vec());
    }

    let freeze_frame = (freeze_position.clamp(0.0, 1.0) * (num_frames - 1) as f32).round() as usize;
    let frozen_mag: Vec<f32> = x[freeze_frame].iter().map(|c| c.norm()).collect();

    let y: Vec<Vec<Complex<f32>>> = x
        .iter()
        .map(|frame| {
            frame
                .iter()
                .zip(frozen_mag.iter())
                .map(|(c, &fm)| {
                    let phase = c.arg();
                    Complex::new(fm * phase.cos(), fm * phase.sin())
                })
                .collect()
        })
        .collect();

    let out = istft(&y, fft_size, hop_size, Some(samples.len()));
    AudioOutput::Mono(post_process(&out, sr))
}

fn variants_h001() -> Vec<HashMap<String, Value>> {
    vec![
        params!("freeze_position" => 0.0, "fft_size" => 2048),
        params!("freeze_position" => 0.25, "fft_size" => 2048),
        params!("freeze_position" => 0.5, "fft_size" => 2048),
        params!("freeze_position" => 0.75, "fft_size" => 2048),
        params!("freeze_position" => 1.0, "fft_size" => 2048),
        params!("freeze_position" => 0.5, "fft_size" => 4096),
        params!("freeze_position" => 0.5, "fft_size" => 1024),
    ]
}

// ---------------------------------------------------------------------------
// H002 -- Spectral Blur
// ---------------------------------------------------------------------------

fn process_h002(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let blur_width = pi(params, "blur_width", 10).max(1) as usize;
    let fft_size = pi(params, "fft_size", 2048) as usize;
    let hop_size = fft_size / 4;

    let x = stft(samples, fft_size, hop_size);
    let (mag, phase) = mag_phase(&x);

    let mag_blurred = uniform_filter1d_axis1(&mag, blur_width);

    let y = from_mag_phase(&mag_blurred, &phase);
    let out = istft(&y, fft_size, hop_size, Some(samples.len()));
    AudioOutput::Mono(post_process(&out, sr))
}

fn variants_h002() -> Vec<HashMap<String, Value>> {
    vec![
        params!("blur_width" => 1),
        params!("blur_width" => 5),
        params!("blur_width" => 10),
        params!("blur_width" => 20),
        params!("blur_width" => 35),
        params!("blur_width" => 50),
    ]
}

// ---------------------------------------------------------------------------
// H003 -- Spectral Gate
// ---------------------------------------------------------------------------

fn process_h003(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let gate_percentile = pf(params, "gate_percentile", 75.0);
    let fft_size = pi(params, "fft_size", 2048) as usize;
    let hop_size = fft_size / 4;

    let x = stft(samples, fft_size, hop_size);
    let (mag, _phase) = mag_phase(&x);

    let mut y = x.clone();
    for i in 0..y.len() {
        let threshold = percentile(&mag[i], gate_percentile);
        for (j, c) in y[i].iter_mut().enumerate() {
            if mag[i][j] < threshold {
                *c = Complex::new(0.0, 0.0);
            }
        }
    }

    let out = istft(&y, fft_size, hop_size, Some(samples.len()));
    AudioOutput::Mono(post_process(&out, sr))
}

fn variants_h003() -> Vec<HashMap<String, Value>> {
    vec![
        params!("gate_percentile" => 50),
        params!("gate_percentile" => 65),
        params!("gate_percentile" => 75),
        params!("gate_percentile" => 85),
        params!("gate_percentile" => 92),
        params!("gate_percentile" => 99),
    ]
}

// ---------------------------------------------------------------------------
// H004 -- Spectral Shift
// ---------------------------------------------------------------------------

fn process_h004(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let shift_bins = pi(params, "shift_bins", 10);
    let fft_size = pi(params, "fft_size", 2048) as usize;
    let hop_size = fft_size / 4;

    let x = stft(samples, fft_size, hop_size);
    let num_frames = x.len();
    if num_frames == 0 {
        return AudioOutput::Mono(samples.to_vec());
    }
    let num_bins = x[0].len();

    let mut y: Vec<Vec<Complex<f32>>> = vec![vec![Complex::new(0.0, 0.0); num_bins]; num_frames];
    for i in 0..num_frames {
        for b in 0..num_bins {
            let src = b as i32 - shift_bins;
            if src >= 0 && (src as usize) < num_bins {
                y[i][b] = x[i][src as usize];
            }
        }
    }

    let out = istft(&y, fft_size, hop_size, Some(samples.len()));
    AudioOutput::Mono(post_process(&out, sr))
}

fn variants_h004() -> Vec<HashMap<String, Value>> {
    vec![
        params!("shift_bins" => -100),
        params!("shift_bins" => -30),
        params!("shift_bins" => -10),
        params!("shift_bins" => 5),
        params!("shift_bins" => 10),
        params!("shift_bins" => 30),
        params!("shift_bins" => 100),
    ]
}

// ---------------------------------------------------------------------------
// H005 -- Phase Randomization
// ---------------------------------------------------------------------------

fn process_h005(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let amount = pf(params, "amount", 0.5);
    let fft_size = pi(params, "fft_size", 2048) as usize;
    let hop_size = fft_size / 4;

    let x = stft(samples, fft_size, hop_size);
    let (mag, orig_phase) = mag_phase(&x);

    let mut rng = SimpleRng::new(42);

    let y: Vec<Vec<Complex<f32>>> = mag
        .iter()
        .zip(orig_phase.iter())
        .map(|(m_row, p_row)| {
            m_row
                .iter()
                .zip(p_row.iter())
                .map(|(&m, &p)| {
                    let random_phase = rng.uniform(
                        -std::f32::consts::PI,
                        std::f32::consts::PI,
                    );
                    let blended = (1.0 - amount) * p + amount * random_phase;
                    Complex::new(m * blended.cos(), m * blended.sin())
                })
                .collect()
        })
        .collect();

    let out = istft(&y, fft_size, hop_size, Some(samples.len()));
    AudioOutput::Mono(post_process(&out, sr))
}

fn variants_h005() -> Vec<HashMap<String, Value>> {
    vec![
        params!("amount" => 0.0),
        params!("amount" => 0.2),
        params!("amount" => 0.5),
        params!("amount" => 0.8),
        params!("amount" => 1.0),
    ]
}

// ---------------------------------------------------------------------------
// H006 -- Robotization
// ---------------------------------------------------------------------------

fn process_h006(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let fft_size = pi(params, "fft_size", 2048) as usize;
    let hop_size = fft_size / 4;

    let x = stft(samples, fft_size, hop_size);
    let (mag, _phase) = mag_phase(&x);

    // All phases zero: complex values are just the magnitudes (real part)
    let y = from_mag_zero_phase(&mag);

    let out = istft(&y, fft_size, hop_size, Some(samples.len()));
    AudioOutput::Mono(post_process(&out, sr))
}

fn variants_h006() -> Vec<HashMap<String, Value>> {
    vec![
        params!("fft_size" => 512),
        params!("fft_size" => 1024),
        params!("fft_size" => 2048),
        params!("fft_size" => 4096),
        params!("fft_size" => 8192),
    ]
}

// ---------------------------------------------------------------------------
// H007 -- Spectral Bin Sorting
// ---------------------------------------------------------------------------

fn process_h007(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let order = ps(params, "order", "descending");
    let partial_sort = pf(params, "partial_sort", 1.0).clamp(0.1, 1.0);
    let fft_size = pi(params, "fft_size", 2048) as usize;
    let hop_size = fft_size / 4;

    let x = stft(samples, fft_size, hop_size);
    let (mut mag, phase) = mag_phase(&x);
    let ascending = order == "ascending";

    for i in 0..mag.len() {
        let num_bins = mag[i].len();
        let n_sort = (num_bins as f32 * partial_sort).round().max(1.0) as usize;

        // Build index array for the first n_sort bins, sorted by magnitude
        let mut indices: Vec<usize> = (0..n_sort).collect();
        indices.sort_by(|&a, &b| {
            mag[i][a]
                .partial_cmp(&mag[i][b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if !ascending {
            indices.reverse();
        }

        // Apply sorted magnitudes
        let orig: Vec<f32> = mag[i][..n_sort].to_vec();
        for (dst, &src) in indices.iter().enumerate() {
            mag[i][dst] = orig[src];
        }
    }

    let y = from_mag_phase(&mag, &phase);
    let out = istft(&y, fft_size, hop_size, Some(samples.len()));
    AudioOutput::Mono(post_process(&out, sr))
}

fn variants_h007() -> Vec<HashMap<String, Value>> {
    vec![
        params!("order" => "ascending", "partial_sort" => 1.0),
        params!("order" => "descending", "partial_sort" => 1.0),
        params!("order" => "ascending", "partial_sort" => 0.3),
        params!("order" => "descending", "partial_sort" => 0.3),
        params!("order" => "ascending", "partial_sort" => 0.1),
        params!("order" => "descending", "partial_sort" => 0.6),
    ]
}

// ---------------------------------------------------------------------------
// H008 -- Spectral Bin Permutation
// ---------------------------------------------------------------------------

fn process_h008(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let seed = pi(params, "seed", 42) as u64;
    let permutation_amount = pf(params, "permutation_amount", 0.5).clamp(0.0, 1.0);
    let fft_size = pi(params, "fft_size", 2048) as usize;
    let hop_size = fft_size / 4;

    let x = stft(samples, fft_size, hop_size);
    let num_frames = x.len();
    if num_frames == 0 {
        return AudioOutput::Mono(samples.to_vec());
    }
    let num_bins = x[0].len();

    let mut rng = SimpleRng::new(seed);
    let perm = rng.permutation(num_bins);

    let y: Vec<Vec<Complex<f32>>> = x
        .iter()
        .map(|frame| {
            frame
                .iter()
                .enumerate()
                .map(|(j, &orig)| {
                    let permuted = frame[perm[j]];
                    let re = (1.0 - permutation_amount) * orig.re + permutation_amount * permuted.re;
                    let im = (1.0 - permutation_amount) * orig.im + permutation_amount * permuted.im;
                    Complex::new(re, im)
                })
                .collect()
        })
        .collect();

    let out = istft(&y, fft_size, hop_size, Some(samples.len()));
    AudioOutput::Mono(post_process(&out, sr))
}

fn variants_h008() -> Vec<HashMap<String, Value>> {
    vec![
        params!("seed" => 42, "permutation_amount" => 0.2),
        params!("seed" => 42, "permutation_amount" => 0.5),
        params!("seed" => 42, "permutation_amount" => 1.0),
        params!("seed" => 123, "permutation_amount" => 0.5),
        params!("seed" => 7, "permutation_amount" => 0.8),
        params!("seed" => 999, "permutation_amount" => 1.0),
    ]
}

// ---------------------------------------------------------------------------
// H009 -- Spectral Cross-Synthesis
// ---------------------------------------------------------------------------

fn process_h009(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let source_type = ps(params, "source_type", "noise");
    let blend = pf(params, "blend", 0.5);
    let fft_size = pi(params, "fft_size", 2048) as usize;
    let hop_size = fft_size / 4;

    let carrier = make_carrier(source_type, samples.len(), sr);

    let x_input = stft(samples, fft_size, hop_size);
    let x_carrier = stft(&carrier, fft_size, hop_size);

    let (mag_input, phase_input) = mag_phase(&x_input);
    let (_mag_carrier, phase_carrier) = mag_phase(&x_carrier);

    // Cross-synthesis: input magnitudes, blended phases
    let y: Vec<Vec<Complex<f32>>> = mag_input
        .iter()
        .zip(phase_input.iter())
        .zip(phase_carrier.iter())
        .map(|((m_row, pi_row), pc_row)| {
            m_row
                .iter()
                .zip(pi_row.iter())
                .zip(pc_row.iter())
                .map(|((&m, &p_in), &p_car)| {
                    let blended_phase = (1.0 - blend) * p_in + blend * p_car;
                    Complex::new(m * blended_phase.cos(), m * blended_phase.sin())
                })
                .collect()
        })
        .collect();

    let out = istft(&y, fft_size, hop_size, Some(samples.len()));
    AudioOutput::Mono(post_process(&out, sr))
}

fn variants_h009() -> Vec<HashMap<String, Value>> {
    vec![
        params!("source_type" => "noise", "blend" => 0.3),
        params!("source_type" => "noise", "blend" => 0.7),
        params!("source_type" => "noise", "blend" => 1.0),
        params!("source_type" => "sine_sweep", "blend" => 0.5),
        params!("source_type" => "sine_sweep", "blend" => 1.0),
        params!("source_type" => "sawtooth", "blend" => 0.5),
        params!("source_type" => "sawtooth", "blend" => 1.0),
    ]
}

// ---------------------------------------------------------------------------
// H010 -- Classic Channel Vocoder
// ---------------------------------------------------------------------------

fn process_h010(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let num_bands = (pi(params, "num_bands", 16) as usize).clamp(8, 64);
    let carrier_type = ps(params, "carrier_type", "noise");
    let _fft_size = pi(params, "fft_size", 2048) as usize;

    let n = samples.len();
    let sr_f = sr as f64;

    // Generate carrier
    let carrier: Vec<f32> = if carrier_type == "input_self" {
        samples.to_vec()
    } else {
        make_carrier(carrier_type, n, sr)
    };

    // Logarithmically spaced band center frequencies
    let low_freq: f64 = 80.0;
    let high_freq: f64 = (sr_f / 2.0 - 100.0).min(12000.0);
    let log_low = low_freq.ln();
    let log_high = high_freq.ln();
    let band_freqs: Vec<f32> = (0..num_bands)
        .map(|i| {
            let t = i as f64 / (num_bands - 1).max(1) as f64;
            (log_low + t * (log_high - log_low)).exp() as f32
        })
        .collect();

    // Envelope follower coefficients: ~5 ms attack, ~20 ms release
    let attack_coeff = (-1.0f32 / (0.005 * sr as f32)).exp();
    let release_coeff = (-1.0f32 / (0.020 * sr as f32)).exp();
    let q = 2.0f32;

    let mut output = vec![0.0f32; n];

    for &freq in &band_freqs {
        // Biquad bandpass coefficients
        let (b0, b1, b2, a1, a2) = biquad_coeffs_bpf(freq, sr, q);

        // Filter analysis signal through bandpass
        let analysis_band = biquad_filter(samples, b0, b1, b2, a1, a2);

        // Extract envelope
        let env = envelope_follower(&analysis_band, attack_coeff, release_coeff);

        // Filter carrier through same bandpass
        let carrier_band = biquad_filter(&carrier, b0, b1, b2, a1, a2);

        // Modulate carrier band by envelope
        for i in 0..n {
            output[i] += carrier_band[i] * env[i];
        }
    }

    // Normalize to match input level
    let in_rms = (samples.iter().map(|&s| (s * s) as f64).sum::<f64>() / n as f64).sqrt() as f32 + 1e-10;
    let out_rms = (output.iter().map(|&s| (s * s) as f64).sum::<f64>() / n as f64).sqrt() as f32 + 1e-10;
    let scale = in_rms / out_rms;
    for s in output.iter_mut() {
        *s *= scale;
    }

    AudioOutput::Mono(post_process(&output, sr))
}

fn variants_h010() -> Vec<HashMap<String, Value>> {
    vec![
        params!("num_bands" => 8, "carrier_type" => "noise"),
        params!("num_bands" => 16, "carrier_type" => "noise"),
        params!("num_bands" => 32, "carrier_type" => "noise"),
        params!("num_bands" => 64, "carrier_type" => "noise"),
        params!("num_bands" => 16, "carrier_type" => "saw"),
        params!("num_bands" => 32, "carrier_type" => "saw"),
        params!("num_bands" => 16, "carrier_type" => "input_self"),
    ]
}

// ---------------------------------------------------------------------------
// H011 -- Harmonic/Percussive Separation
// ---------------------------------------------------------------------------

fn process_h011(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let mut filter_length = pi(params, "filter_length", 17).max(3) as usize;
    let output_mode = ps(params, "output", "harmonic");
    let fft_size = pi(params, "fft_size", 2048) as usize;
    let hop_size = fft_size / 4;

    // Ensure odd filter length
    if filter_length % 2 == 0 {
        filter_length += 1;
    }

    let x = stft(samples, fft_size, hop_size);
    let (mag, phase) = mag_phase(&x);

    // Harmonic: median filter along time axis (axis=0) -- kernel (filter_length, 1)
    let harmonic_mag = median_filter_2d(&mag, (filter_length, 1));
    // Percussive: median filter along frequency axis (axis=1) -- kernel (1, filter_length)
    let percussive_mag = median_filter_2d(&mag, (1, filter_length));

    // Soft masks (Wiener-style)
    let num_frames = mag.len();
    let num_bins = if num_frames > 0 { mag[0].len() } else { 0 };

    let mut result_mag = vec![vec![0.0f32; num_bins]; num_frames];

    for i in 0..num_frames {
        for j in 0..num_bins {
            let total = harmonic_mag[i][j] + percussive_mag[i][j] + 1e-10;
            let h_mask = harmonic_mag[i][j] / total;
            let p_mask = percussive_mag[i][j] / total;

            result_mag[i][j] = match output_mode {
                "harmonic" => mag[i][j] * h_mask,
                "percussive" => mag[i][j] * p_mask,
                _ => mag[i][j] * (0.7 * h_mask + 1.3 * p_mask), // remix
            };
        }
    }

    let y = from_mag_phase(&result_mag, &phase);
    let out = istft(&y, fft_size, hop_size, Some(samples.len()));
    AudioOutput::Mono(post_process(&out, sr))
}

fn variants_h011() -> Vec<HashMap<String, Value>> {
    vec![
        params!("filter_length" => 7, "output" => "harmonic"),
        params!("filter_length" => 17, "output" => "harmonic"),
        params!("filter_length" => 31, "output" => "harmonic"),
        params!("filter_length" => 7, "output" => "percussive"),
        params!("filter_length" => 17, "output" => "percussive"),
        params!("filter_length" => 31, "output" => "percussive"),
        params!("filter_length" => 17, "output" => "remix"),
    ]
}

// ---------------------------------------------------------------------------
// H012 -- Spectral Mirror
// ---------------------------------------------------------------------------

fn process_h012(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let mirror_center_hz = pf(params, "mirror_center_hz", 2000.0);
    let fft_size = pi(params, "fft_size", 2048) as usize;
    let hop_size = fft_size / 4;

    let x = stft(samples, fft_size, hop_size);
    let (mag, phase) = mag_phase(&x);
    let num_frames = mag.len();
    if num_frames == 0 {
        return AudioOutput::Mono(samples.to_vec());
    }
    let num_bins = mag[0].len();

    let bin_hz = sr as f32 / fft_size as f32;
    let center_bin = (mirror_center_hz / bin_hz).round() as i32;
    let center_bin = center_bin.clamp(1, num_bins as i32 - 2) as usize;

    let mut mirrored_mag = vec![vec![0.0f32; num_bins]; num_frames];
    for b in 0..num_bins {
        let src = 2 * center_bin as i32 - b as i32;
        if src >= 0 && (src as usize) < num_bins {
            for i in 0..num_frames {
                mirrored_mag[i][b] = mag[i][src as usize];
            }
        }
    }

    let y = from_mag_phase(&mirrored_mag, &phase);
    let out = istft(&y, fft_size, hop_size, Some(samples.len()));
    AudioOutput::Mono(post_process(&out, sr))
}

fn variants_h012() -> Vec<HashMap<String, Value>> {
    vec![
        params!("mirror_center_hz" => 500),
        params!("mirror_center_hz" => 1000),
        params!("mirror_center_hz" => 2000),
        params!("mirror_center_hz" => 3000),
        params!("mirror_center_hz" => 5000),
    ]
}

// ---------------------------------------------------------------------------
// H013 -- Spectral Stretch/Compress
// ---------------------------------------------------------------------------

fn process_h013(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let spectral_stretch = pf(params, "spectral_stretch", 1.5) as f64;
    let fft_size = pi(params, "fft_size", 2048) as usize;
    let hop_size = fft_size / 4;

    let x = stft(samples, fft_size, hop_size);
    let (mag, phase) = mag_phase(&x);
    let num_frames = mag.len();
    if num_frames == 0 {
        return AudioOutput::Mono(samples.to_vec());
    }
    let num_bins = mag[0].len();

    let mut stretched_mag = vec![vec![0.0f32; num_bins]; num_frames];

    for b in 0..num_bins {
        let src = b as f64 / spectral_stretch;
        let src_int = src.floor() as i32;
        let frac = (src - src_int as f64) as f32;

        if src_int < 0 || src_int as usize >= num_bins {
            continue;
        }

        if (src_int + 1) >= 0 && ((src_int + 1) as usize) < num_bins {
            for i in 0..num_frames {
                stretched_mag[i][b] =
                    (1.0 - frac) * mag[i][src_int as usize] + frac * mag[i][(src_int + 1) as usize];
            }
        } else {
            for i in 0..num_frames {
                stretched_mag[i][b] = mag[i][src_int as usize];
            }
        }
    }

    let y = from_mag_phase(&stretched_mag, &phase);
    let out = istft(&y, fft_size, hop_size, Some(samples.len()));
    AudioOutput::Mono(post_process(&out, sr))
}

fn variants_h013() -> Vec<HashMap<String, Value>> {
    vec![
        params!("spectral_stretch" => 0.5),
        params!("spectral_stretch" => 0.75),
        params!("spectral_stretch" => 1.0),
        params!("spectral_stretch" => 1.25),
        params!("spectral_stretch" => 1.5),
        params!("spectral_stretch" => 2.0),
    ]
}

// ---------------------------------------------------------------------------
// H014 -- Cepstral Processing
// ---------------------------------------------------------------------------

/// Simple real-to-real FFT and inverse for cepstral domain work.
/// We use the STFT infrastructure with a single frame.
fn rfft_real(data: &[f32]) -> Vec<Complex<f32>> {
    use realfft::RealFftPlanner;
    let n = data.len();
    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n);
    let mut input = data.to_vec();
    let mut output = fft.make_output_vec();
    let mut scratch = fft.make_scratch_vec();
    fft.process_with_scratch(&mut input, &mut output, &mut scratch).unwrap();
    output
}

fn irfft_real(data: &[Complex<f32>], n: usize) -> Vec<f32> {
    use realfft::RealFftPlanner;
    let mut planner = RealFftPlanner::<f32>::new();
    let ifft = planner.plan_fft_inverse(n);
    let mut input = data.to_vec();
    let mut output = ifft.make_output_vec();
    let mut scratch = ifft.make_scratch_vec();
    ifft.process_with_scratch(&mut input, &mut output, &mut scratch).unwrap();
    let norm = 1.0 / n as f32;
    for s in output.iter_mut() {
        *s *= norm;
    }
    output
}

fn process_h014(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let lifter_cutoff = pi(params, "lifter_cutoff", 30).max(1) as usize;
    let operation = ps(params, "operation", "smooth_envelope");
    let fft_size = pi(params, "fft_size", 2048) as usize;
    let hop_size = fft_size / 4;

    let x = stft(samples, fft_size, hop_size);
    let num_frames = x.len();
    if num_frames == 0 {
        return AudioOutput::Mono(samples.to_vec());
    }
    let num_bins = x[0].len();

    let mut y: Vec<Vec<Complex<f32>>> = Vec::with_capacity(num_frames);

    // We need a real-valued FFT size for cepstral processing.
    // The log-magnitude has num_bins values (fft_size/2 + 1).
    // We'll do irfft of log_mag to get cepstrum, then rfft to get back.
    // irfft expects num_bins complex values and produces fft_size real values.
    // But log_mag is real-valued, so we treat it as real part of complex with zero imag.

    for i in 0..num_frames {
        let frame = &x[i];
        // log magnitude
        let log_mag: Vec<f32> = frame.iter().map(|c| (c.norm() + 1e-10).ln()).collect();

        // Compute cepstrum: irfft of log_mag
        // We pass log_mag as complex values (real part only)
        let log_mag_complex: Vec<Complex<f32>> =
            log_mag.iter().map(|&v| Complex::new(v, 0.0)).collect();
        let cepstrum = irfft_real(&log_mag_complex, fft_size);

        // Lifter
        let mut liftered = if operation == "smooth_envelope" {
            // Low-pass: keep only first lifter_cutoff cepstral coefficients
            let mut l = vec![0.0f32; cepstrum.len()];
            let cutoff = lifter_cutoff.min(cepstrum.len());
            l[..cutoff].copy_from_slice(&cepstrum[..cutoff]);
            l
        } else {
            // High-pass (remove_pitch): zero out low quefrency
            let mut l = cepstrum.clone();
            let cutoff = lifter_cutoff.min(l.len());
            for j in 0..cutoff {
                l[j] = 0.0;
            }
            l
        };

        // Back to log-magnitude domain
        let smoothed_complex = rfft_real(&liftered);
        let smoothed_log_mag: Vec<f32> = smoothed_complex.iter().map(|c| c.re).collect();
        let smoothed_mag: Vec<f32> = smoothed_log_mag.iter().map(|&v| v.exp()).collect();

        // Reconstruct with original phase
        let frame_out: Vec<Complex<f32>> = smoothed_mag
            .iter()
            .zip(frame.iter())
            .map(|(&m, c)| {
                let phase = c.arg();
                Complex::new(m * phase.cos(), m * phase.sin())
            })
            .collect();
        y.push(frame_out);
    }

    let out = istft(&y, fft_size, hop_size, Some(samples.len()));
    AudioOutput::Mono(post_process(&out, sr))
}

fn variants_h014() -> Vec<HashMap<String, Value>> {
    vec![
        params!("lifter_cutoff" => 10, "operation" => "smooth_envelope"),
        params!("lifter_cutoff" => 30, "operation" => "smooth_envelope"),
        params!("lifter_cutoff" => 60, "operation" => "smooth_envelope"),
        params!("lifter_cutoff" => 100, "operation" => "smooth_envelope"),
        params!("lifter_cutoff" => 10, "operation" => "remove_pitch"),
        params!("lifter_cutoff" => 30, "operation" => "remove_pitch"),
        params!("lifter_cutoff" => 60, "operation" => "remove_pitch"),
    ]
}

// ---------------------------------------------------------------------------
// H015 -- Spectral Delay
// ---------------------------------------------------------------------------

fn process_h015(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let base_delay_ms = pf(params, "base_delay_ms", 10.0);
    let delay_slope_ms_per_bin = pf(params, "delay_slope_ms_per_bin", 0.1);
    let fft_size = pi(params, "fft_size", 2048) as usize;
    let hop_size = fft_size / 4;

    let x = stft(samples, fft_size, hop_size);
    let num_frames = x.len();
    if num_frames == 0 {
        return AudioOutput::Mono(samples.to_vec());
    }
    let num_bins = x[0].len();

    let hop_duration_ms = (hop_size as f32 / sr as f32) * 1000.0;

    let mut y: Vec<Vec<Complex<f32>>> = vec![vec![Complex::new(0.0, 0.0); num_bins]; num_frames];

    for b in 0..num_bins {
        let delay_ms = base_delay_ms + delay_slope_ms_per_bin * b as f32;
        let delay_frames = (delay_ms / hop_duration_ms) as i32;
        for t in 0..num_frames {
            let src_t = t as i32 - delay_frames;
            if src_t >= 0 && (src_t as usize) < num_frames {
                y[t][b] = x[src_t as usize][b];
            }
        }
    }

    let out = istft(&y, fft_size, hop_size, Some(samples.len()));
    AudioOutput::Mono(post_process(&out, sr))
}

fn variants_h015() -> Vec<HashMap<String, Value>> {
    vec![
        params!("base_delay_ms" => 0, "delay_slope_ms_per_bin" => 0.1),
        params!("base_delay_ms" => 0, "delay_slope_ms_per_bin" => 0.5),
        params!("base_delay_ms" => 0, "delay_slope_ms_per_bin" => 1.0),
        params!("base_delay_ms" => 20, "delay_slope_ms_per_bin" => 0.0),
        params!("base_delay_ms" => 50, "delay_slope_ms_per_bin" => 0.2),
        params!("base_delay_ms" => 100, "delay_slope_ms_per_bin" => 0.5),
    ]
}

// ---------------------------------------------------------------------------
// H016 -- Spectral Compressor
// ---------------------------------------------------------------------------

fn process_h016(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let threshold_db = pf(params, "threshold_db", -30.0);
    let ratio = pf(params, "ratio", 4.0);
    let fft_size = pi(params, "fft_size", 2048) as usize;
    let hop_size = fft_size / 4;

    let x = stft(samples, fft_size, hop_size);
    let (mag, phase) = mag_phase(&x);
    let num_frames = mag.len();
    if num_frames == 0 {
        return AudioOutput::Mono(samples.to_vec());
    }
    let num_bins = mag[0].len();

    // Convert to dB, compress, convert back
    let mut compressed_mag = vec![vec![0.0f32; num_bins]; num_frames];
    let mut sum_reduction = 0.0f64;
    let mut count = 0u64;

    for i in 0..num_frames {
        for j in 0..num_bins {
            let mag_db = 20.0 * (mag[i][j] + 1e-10).log10();
            let compressed_db = if mag_db > threshold_db {
                threshold_db + (mag_db - threshold_db) / ratio
            } else {
                mag_db
            };
            sum_reduction += (mag_db - compressed_db) as f64;
            count += 1;
            compressed_mag[i][j] = 10.0f32.powf(compressed_db / 20.0);
        }
    }

    // Make-up gain: half the average reduction
    let avg_reduction = sum_reduction / count.max(1) as f64;
    let makeup_linear = 10.0f64.powf(avg_reduction / 40.0) as f32;
    for i in 0..num_frames {
        for j in 0..num_bins {
            compressed_mag[i][j] *= makeup_linear;
        }
    }

    let y = from_mag_phase(&compressed_mag, &phase);
    let out = istft(&y, fft_size, hop_size, Some(samples.len()));
    AudioOutput::Mono(post_process(&out, sr))
}

fn variants_h016() -> Vec<HashMap<String, Value>> {
    vec![
        params!("threshold_db" => -10, "ratio" => 2),
        params!("threshold_db" => -20, "ratio" => 2),
        params!("threshold_db" => -30, "ratio" => 4),
        params!("threshold_db" => -40, "ratio" => 4),
        params!("threshold_db" => -60, "ratio" => 10),
        params!("threshold_db" => -30, "ratio" => 20),
    ]
}

// ---------------------------------------------------------------------------
// H017 -- Spectral Reassignment
// ---------------------------------------------------------------------------

fn process_h017(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let sharpening_amount = pf(params, "sharpening_amount", 0.5).clamp(0.0, 1.0);
    let fft_size = pi(params, "fft_size", 2048) as usize;
    let hop_size = fft_size / 4;

    let n = samples.len();
    let mut x_padded;
    let signal = if n < fft_size {
        x_padded = vec![0.0f32; fft_size];
        x_padded[..n].copy_from_slice(samples);
        &x_padded[..]
    } else {
        samples
    };
    let sig_len = signal.len();
    let num_frames = if sig_len >= fft_size {
        1 + (sig_len - fft_size) / hop_size
    } else {
        1
    };
    let num_bins = fft_size / 2 + 1;
    let sr_f = sr as f32;

    // Standard STFT
    let x = stft(signal, fft_size, hop_size);
    let (mag, phase) = mag_phase(&x);

    // Time-weighted window for reassignment
    let window = hann_window(fft_size);
    let t_window: Vec<f32> = (0..fft_size)
        .map(|i| (i as f32 - fft_size as f32 / 2.0) / sr_f)
        .collect();
    let dwindow: Vec<f32> = t_window
        .iter()
        .zip(window.iter())
        .map(|(&t, &w)| t * w)
        .collect();

    // Time-derivative STFT using time-ramped window
    let x_d = {
        use realfft::RealFftPlanner;
        let mut planner = RealFftPlanner::<f32>::new();
        let fft_plan = planner.plan_fft_forward(fft_size);

        let mut result = Vec::with_capacity(num_frames);
        for i in 0..num_frames {
            let start = i * hop_size;
            let end = (start + fft_size).min(sig_len);
            let mut frame = vec![0.0f32; fft_size];
            for j in 0..(end - start) {
                frame[j] = signal[start + j] * dwindow[j];
            }
            let mut spectrum = fft_plan.make_output_vec();
            let mut scratch = fft_plan.make_scratch_vec();
            fft_plan
                .process_with_scratch(&mut frame, &mut spectrum, &mut scratch)
                .unwrap();
            result.push(spectrum);
        }
        result
    };

    // Compute reassignment shifts
    let eps = 1e-10f32;
    let mut reassign_shift = vec![vec![0.0f32; num_bins]; num_frames];
    for i in 0..num_frames {
        for b in 0..num_bins.min(x[i].len()).min(x_d[i].len()) {
            if mag[i][b] > eps {
                // Frequency reassignment: -Im(X_d / X)
                let x_val = x[i][b];
                let xd_val = x_d[i][b];
                let denom = x_val + Complex::new(eps, 0.0);
                let ratio = xd_val / denom;
                reassign_shift[i][b] = -ratio.im;
            }
        }
    }

    // Build sharpened spectrogram by reassigning energy
    let mut sharpened_mag = vec![vec![0.0f32; num_bins]; num_frames];
    for i in 0..num_frames {
        for b in 0..num_bins.min(mag[i].len()) {
            if mag[i][b] < eps {
                continue;
            }
            let target_b =
                b as f32 + sharpening_amount * reassign_shift[i][b] * fft_size as f32 / sr_f;
            let target_b_int = target_b.round() as i32;
            if target_b_int >= 0 && (target_b_int as usize) < num_bins {
                sharpened_mag[i][target_b_int as usize] += mag[i][b];
            }
        }
    }

    let y = from_mag_phase(&sharpened_mag, &phase);
    let out = istft(&y, fft_size, hop_size, Some(samples.len()));
    AudioOutput::Mono(post_process(&out, sr))
}

fn variants_h017() -> Vec<HashMap<String, Value>> {
    vec![
        params!("sharpening_amount" => 0.0),
        params!("sharpening_amount" => 0.2),
        params!("sharpening_amount" => 0.5),
        params!("sharpening_amount" => 0.8),
        params!("sharpening_amount" => 1.0),
    ]
}

// ---------------------------------------------------------------------------
// H018 -- Spectral Subtraction
// ---------------------------------------------------------------------------

fn process_h018(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let subtraction_factor = pf(params, "subtraction_factor", 2.0);
    let noise_percentile = pf(params, "noise_percentile", 10.0);
    let fft_size = pi(params, "fft_size", 2048) as usize;
    let hop_size = fft_size / 4;

    let x = stft(samples, fft_size, hop_size);
    let (mag, phase) = mag_phase(&x);
    let num_frames = mag.len();
    if num_frames == 0 {
        return AudioOutput::Mono(samples.to_vec());
    }
    let num_bins = mag[0].len();

    // Estimate noise floor: percentile across frames per bin
    let noise_floor = percentile_axis0(&mag, noise_percentile);

    // Subtract scaled noise floor
    let mut cleaned_mag = vec![vec![0.0f32; num_bins]; num_frames];
    for i in 0..num_frames {
        for j in 0..num_bins {
            cleaned_mag[i][j] = (mag[i][j] - subtraction_factor * noise_floor[j]).max(0.0);
        }
    }

    let y = from_mag_phase(&cleaned_mag, &phase);
    let out = istft(&y, fft_size, hop_size, Some(samples.len()));
    AudioOutput::Mono(post_process(&out, sr))
}

fn variants_h018() -> Vec<HashMap<String, Value>> {
    vec![
        params!("subtraction_factor" => 1.0, "noise_percentile" => 10),
        params!("subtraction_factor" => 2.0, "noise_percentile" => 10),
        params!("subtraction_factor" => 4.0, "noise_percentile" => 15),
        params!("subtraction_factor" => 8.0, "noise_percentile" => 20),
        params!("subtraction_factor" => 15.0, "noise_percentile" => 30),
        params!("subtraction_factor" => 3.0, "noise_percentile" => 50),
    ]
}

// ---------------------------------------------------------------------------
// H019 -- Spectral Transfer / Timbre Stamp
// ---------------------------------------------------------------------------

fn process_h019(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let carrier_type = ps(params, "carrier_type", "noise");
    let envelope_order = pi(params, "envelope_order", 30).max(1) as usize;
    let blend = pf(params, "blend", 0.7);
    let fft_size = pi(params, "fft_size", 2048) as usize;
    let hop_size = fft_size / 4;

    let n = samples.len();

    // Generate carrier
    let carrier = make_carrier(carrier_type, n, sr);

    let x_input = stft(samples, fft_size, hop_size);
    let x_carrier = stft(&carrier, fft_size, hop_size);

    let num_frames = x_input.len();
    let mut y: Vec<Vec<Complex<f32>>> = Vec::with_capacity(num_frames);

    for i in 0..num_frames {
        let input_frame = &x_input[i];
        let carrier_frame = if i < x_carrier.len() {
            &x_carrier[i]
        } else {
            // Fallback: use last carrier frame
            x_carrier.last().unwrap()
        };
        let num_bins = input_frame.len();

        // Extract spectral envelope via cepstral smoothing
        let log_mag: Vec<f32> = input_frame.iter().map(|c| (c.norm() + 1e-10).ln()).collect();
        let log_mag_complex: Vec<Complex<f32>> =
            log_mag.iter().map(|&v| Complex::new(v, 0.0)).collect();
        let cepstrum = irfft_real(&log_mag_complex, fft_size);

        // Low-pass lifter: keep only first envelope_order cepstral coefficients
        let mut liftered = vec![0.0f32; cepstrum.len()];
        let cutoff = envelope_order.min(cepstrum.len());
        liftered[..cutoff].copy_from_slice(&cepstrum[..cutoff]);

        let envelope_complex = rfft_real(&liftered);
        let envelope: Vec<f32> = envelope_complex.iter().map(|c| c.re.exp()).collect();

        // Carrier magnitudes and phases
        let carrier_mag: Vec<f32> = carrier_frame.iter().map(|c| c.norm()).collect();
        let carrier_phase: Vec<f32> = carrier_frame.iter().map(|c| c.arg()).collect();
        let carrier_peak = carrier_mag.iter().cloned().fold(0.0f32, f32::max) + 1e-10;

        // Original magnitudes and phases
        let orig_mag: Vec<f32> = input_frame.iter().map(|c| c.norm()).collect();
        let orig_phase: Vec<f32> = input_frame.iter().map(|c| c.arg()).collect();

        let frame_out: Vec<Complex<f32>> = (0..num_bins)
            .map(|j| {
                // Normalize carrier magnitude, shape with input envelope
                let shaped_mag = if j < envelope.len() && j < carrier_mag.len() {
                    envelope[j] * (carrier_mag[j] / carrier_peak)
                } else {
                    0.0
                };

                // Blend magnitudes
                let blended_mag = (1.0 - blend) * orig_mag[j] + blend * shaped_mag;

                // Blend phases
                let cp = if j < carrier_phase.len() {
                    carrier_phase[j]
                } else {
                    0.0
                };
                let blended_phase = (1.0 - blend) * orig_phase[j] + blend * cp;

                Complex::new(
                    blended_mag * blended_phase.cos(),
                    blended_mag * blended_phase.sin(),
                )
            })
            .collect();

        y.push(frame_out);
    }

    let out = istft(&y, fft_size, hop_size, Some(n));
    AudioOutput::Mono(post_process(&out, sr))
}

fn variants_h019() -> Vec<HashMap<String, Value>> {
    vec![
        params!("carrier_type" => "noise", "envelope_order" => 20, "blend" => 0.5),
        params!("carrier_type" => "noise", "envelope_order" => 40, "blend" => 0.8),
        params!("carrier_type" => "noise", "envelope_order" => 10, "blend" => 1.0),
        params!("carrier_type" => "chirp", "envelope_order" => 30, "blend" => 0.6),
        params!("carrier_type" => "chirp", "envelope_order" => 30, "blend" => 1.0),
        params!("carrier_type" => "pulse", "envelope_order" => 30, "blend" => 0.7),
        params!("carrier_type" => "pulse", "envelope_order" => 15, "blend" => 1.0),
    ]
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

pub fn register() -> Vec<EffectEntry> {
    vec![
        EffectEntry {
            id: "H001",
            process: process_h001,
            variants: variants_h001,
            category: "spectral",
        },
        EffectEntry {
            id: "H002",
            process: process_h002,
            variants: variants_h002,
            category: "spectral",
        },
        EffectEntry {
            id: "H003",
            process: process_h003,
            variants: variants_h003,
            category: "spectral",
        },
        EffectEntry {
            id: "H004",
            process: process_h004,
            variants: variants_h004,
            category: "spectral",
        },
        EffectEntry {
            id: "H005",
            process: process_h005,
            variants: variants_h005,
            category: "spectral",
        },
        EffectEntry {
            id: "H006",
            process: process_h006,
            variants: variants_h006,
            category: "spectral",
        },
        EffectEntry {
            id: "H007",
            process: process_h007,
            variants: variants_h007,
            category: "spectral",
        },
        EffectEntry {
            id: "H008",
            process: process_h008,
            variants: variants_h008,
            category: "spectral",
        },
        EffectEntry {
            id: "H009",
            process: process_h009,
            variants: variants_h009,
            category: "spectral",
        },
        EffectEntry {
            id: "H010",
            process: process_h010,
            variants: variants_h010,
            category: "spectral",
        },
        EffectEntry {
            id: "H011",
            process: process_h011,
            variants: variants_h011,
            category: "spectral",
        },
        EffectEntry {
            id: "H012",
            process: process_h012,
            variants: variants_h012,
            category: "spectral",
        },
        EffectEntry {
            id: "H013",
            process: process_h013,
            variants: variants_h013,
            category: "spectral",
        },
        EffectEntry {
            id: "H014",
            process: process_h014,
            variants: variants_h014,
            category: "spectral",
        },
        EffectEntry {
            id: "H015",
            process: process_h015,
            variants: variants_h015,
            category: "spectral",
        },
        EffectEntry {
            id: "H016",
            process: process_h016,
            variants: variants_h016,
            category: "spectral",
        },
        EffectEntry {
            id: "H017",
            process: process_h017,
            variants: variants_h017,
            category: "spectral",
        },
        EffectEntry {
            id: "H018",
            process: process_h018,
            variants: variants_h018,
            category: "spectral",
        },
        EffectEntry {
            id: "H019",
            process: process_h019,
            variants: variants_h019,
            category: "spectral",
        },
    ]
}
