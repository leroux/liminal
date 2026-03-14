//! L-series: Convolution-based algorithms (L001-L006).

use std::collections::HashMap;
use serde_json::Value;
use crate::{AudioOutput, EffectEntry, pf, pi, ps, params};
use crate::primitives::*;
use crate::stft::{stft, istft};
use realfft::RealFftPlanner;
use num_complex::Complex;

/// Return the smallest power of 2 >= n.
fn next_pow2(n: usize) -> usize {
    let mut p = 1;
    while p < n {
        p <<= 1;
    }
    p
}

/// Convolve x with h using FFT, padded to next power of 2. Returns first len(x) samples.
fn fft_convolve(x: &[f32], h: &[f32]) -> Vec<f32> {
    let n = x.len() + h.len() - 1;
    let fft_size = next_pow2(n);

    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(fft_size);
    let ifft = planner.plan_fft_inverse(fft_size);

    // Forward FFT of x
    let mut x_buf = vec![0.0f32; fft_size];
    x_buf[..x.len()].copy_from_slice(x);
    let mut x_spec = fft.make_output_vec();
    let mut scratch = fft.make_scratch_vec();
    fft.process_with_scratch(&mut x_buf, &mut x_spec, &mut scratch).unwrap();

    // Forward FFT of h
    let mut h_buf = vec![0.0f32; fft_size];
    h_buf[..h.len()].copy_from_slice(h);
    let mut h_spec = fft.make_output_vec();
    fft.process_with_scratch(&mut h_buf, &mut h_spec, &mut scratch).unwrap();

    // Multiply spectra
    for i in 0..x_spec.len() {
        x_spec[i] = x_spec[i] * h_spec[i];
    }

    // Inverse FFT
    let mut out_buf = ifft.make_output_vec();
    let mut iscratch = ifft.make_scratch_vec();
    ifft.process_with_scratch(&mut x_spec, &mut out_buf, &mut iscratch).unwrap();

    // realfft inverse is unnormalized
    let norm = 1.0 / fft_size as f32;
    out_buf.iter().take(x.len()).map(|&s| s * norm).collect()
}

// ---------------------------------------------------------------------------
// L001 -- Convolve with Mathematical IR
// ---------------------------------------------------------------------------

/// Generate an impulse response from a mathematical function.
fn generate_ir(ir_type: &str, ir_length_samples: usize, decay_rate: f32, freq_hz: f32, sr: u32) -> Vec<f32> {
    let sr_f = sr as f64;
    let decay_rate = decay_rate as f64;
    let freq_hz = freq_hz as f64;

    let ir: Vec<f64> = match ir_type {
        "sinc" => {
            let mid = ir_length_samples as f64 / 2.0;
            (0..ir_length_samples)
                .map(|i| {
                    let t_centered = (i as f64 - mid) / sr_f;
                    let arg = freq_hz * t_centered;
                    // sinc(x) = sin(pi*x) / (pi*x)
                    if arg.abs() < 1e-12 {
                        1.0
                    } else {
                        (std::f64::consts::PI * arg).sin() / (std::f64::consts::PI * arg)
                    }
                })
                .collect()
        }
        "chirp" => {
            let t_vec: Vec<f64> = (0..ir_length_samples).map(|i| i as f64 / sr_f).collect();
            let f0 = freq_hz * 0.25;
            let f1 = freq_hz;
            let t_end = t_vec.last().copied().unwrap_or(0.0) + 1e-12;
            t_vec.iter()
                .map(|&t| {
                    let phase = 2.0 * std::f64::consts::PI * (f0 * t + (f1 - f0) / (2.0 * t_end) * t * t);
                    phase.sin() * (-decay_rate * 0.5 * t).exp()
                })
                .collect()
        }
        "gaussian" => {
            let mid = ir_length_samples as f64 / 2.0;
            let sigma = ir_length_samples as f64 / (2.0 * decay_rate + 1e-12);
            (0..ir_length_samples)
                .map(|i| {
                    let idx = i as f64;
                    let t = i as f64 / sr_f;
                    let gauss = (-0.5 * ((idx - mid) / sigma).powi(2)).exp();
                    gauss * (2.0 * std::f64::consts::PI * freq_hz * t).cos()
                })
                .collect()
        }
        // "exponential" or default
        _ => {
            (0..ir_length_samples)
                .map(|i| {
                    let t = i as f64 / sr_f;
                    (-decay_rate * t).exp()
                })
                .collect()
        }
    };

    // Normalize IR energy
    let energy = (ir.iter().map(|&x| x * x).sum::<f64>() + 1e-12).sqrt();
    ir.iter().map(|&x| (x / energy) as f32).collect()
}

fn process_l001(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let ir_type = ps(params, "ir_type", "exponential");
    let ir_length_ms = pf(params, "ir_length_ms", 100.0);
    let decay_rate = pf(params, "decay_rate", 5.0);
    let freq_hz = pf(params, "freq_hz", 500.0);

    let ir_length_samples = (ir_length_ms * sr as f32 / 1000.0) as usize;
    let ir_length_samples = ir_length_samples.max(1);
    let ir = generate_ir(ir_type, ir_length_samples, decay_rate, freq_hz, sr);
    AudioOutput::Mono(fft_convolve(samples, &ir))
}

fn variants_l001() -> Vec<HashMap<String, Value>> {
    vec![
        params!("ir_type" => "exponential", "ir_length_ms" => 200, "decay_rate" => 3.0, "freq_hz" => 500),
        params!("ir_type" => "exponential", "ir_length_ms" => 50, "decay_rate" => 15.0, "freq_hz" => 500),
        params!("ir_type" => "sinc", "ir_length_ms" => 100, "decay_rate" => 5.0, "freq_hz" => 400),
        params!("ir_type" => "sinc", "ir_length_ms" => 200, "decay_rate" => 5.0, "freq_hz" => 1500),
        params!("ir_type" => "chirp", "ir_length_ms" => 150, "decay_rate" => 4.0, "freq_hz" => 800),
        params!("ir_type" => "gaussian", "ir_length_ms" => 80, "decay_rate" => 6.0, "freq_hz" => 600),
        params!("ir_type" => "gaussian", "ir_length_ms" => 300, "decay_rate" => 2.0, "freq_hz" => 1200),
    ]
}

// ---------------------------------------------------------------------------
// L002 -- Auto-Convolution
// ---------------------------------------------------------------------------

fn process_l002(samples: &[f32], _sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let num_iterations = pi(params, "num_iterations", 1).max(1).min(4) as usize;
    let orig_len = samples.len();

    let mut x: Vec<f32> = samples.to_vec();

    let mut planner = RealFftPlanner::<f32>::new();

    for _ in 0..num_iterations {
        let n = 2 * x.len() - 1;
        let fft_size = next_pow2(n);

        let fft = planner.plan_fft_forward(fft_size);
        let ifft = planner.plan_fft_inverse(fft_size);

        // Forward FFT of x
        let mut x_buf = vec![0.0f32; fft_size];
        let copy_len = x.len().min(fft_size);
        x_buf[..copy_len].copy_from_slice(&x[..copy_len]);
        let mut x_spec = fft.make_output_vec();
        let mut scratch = fft.make_scratch_vec();
        fft.process_with_scratch(&mut x_buf, &mut x_spec, &mut scratch).unwrap();

        // Square the spectrum (auto-convolution)
        for i in 0..x_spec.len() {
            x_spec[i] = x_spec[i] * x_spec[i];
        }

        // Inverse FFT
        let mut out_buf = ifft.make_output_vec();
        let mut iscratch = ifft.make_scratch_vec();
        ifft.process_with_scratch(&mut x_spec, &mut out_buf, &mut iscratch).unwrap();

        // Normalize by fft_size (realfft unnormalized inverse)
        let norm = 1.0 / fft_size as f32;

        // Trim back to original length
        x = out_buf.iter().take(orig_len).map(|&s| s * norm).collect();

        // Normalize to prevent explosion
        let peak = x.iter().map(|s| s.abs()).fold(0.0f32, f32::max) + 1e-12;
        for s in x.iter_mut() {
            *s /= peak;
        }
    }

    AudioOutput::Mono(x)
}

fn variants_l002() -> Vec<HashMap<String, Value>> {
    vec![
        params!("num_iterations" => 1),
        params!("num_iterations" => 2),
        params!("num_iterations" => 3),
        params!("num_iterations" => 4),
    ]
}

// ---------------------------------------------------------------------------
// L003 -- Deconvolution
// ---------------------------------------------------------------------------

fn process_l003(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let ir_type = ps(params, "ir_type", "exponential");
    let ir_length_ms = pf(params, "ir_length_ms", 50.0);
    let epsilon = pf(params, "epsilon", 0.01);

    let ir_length_samples = (ir_length_ms * sr as f32 / 1000.0) as usize;
    let ir_length_samples = ir_length_samples.max(1);
    let sr_f = sr as f64;

    // Generate IR for deconvolution
    let ir: Vec<f64> = match ir_type {
        "gaussian" => {
            let mid = ir_length_samples as f64 / 2.0;
            let sigma = ir_length_samples as f64 / 6.0;
            (0..ir_length_samples)
                .map(|i| {
                    let idx = i as f64;
                    (-0.5 * ((idx - mid) / sigma).powi(2)).exp()
                })
                .collect()
        }
        // "exponential" or default
        _ => {
            (0..ir_length_samples)
                .map(|i| {
                    let t = i as f64 / sr_f;
                    (-5.0 * t).exp()
                })
                .collect()
        }
    };

    // Normalize IR
    let energy = (ir.iter().map(|&x| x * x).sum::<f64>() + 1e-12).sqrt();
    let ir_f32: Vec<f32> = ir.iter().map(|&x| (x / energy) as f32).collect();

    let x_len = samples.len();
    let n = x_len + ir_length_samples - 1;
    let fft_size = next_pow2(n);

    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(fft_size);
    let ifft = planner.plan_fft_inverse(fft_size);

    // Forward FFT of x
    let mut x_buf = vec![0.0f32; fft_size];
    x_buf[..x_len].copy_from_slice(samples);
    let mut x_spec = fft.make_output_vec();
    let mut scratch = fft.make_scratch_vec();
    fft.process_with_scratch(&mut x_buf, &mut x_spec, &mut scratch).unwrap();

    // Forward FFT of IR
    let mut h_buf = vec![0.0f32; fft_size];
    h_buf[..ir_f32.len()].copy_from_slice(&ir_f32);
    let mut h_spec = fft.make_output_vec();
    fft.process_with_scratch(&mut h_buf, &mut h_spec, &mut scratch).unwrap();

    // Deconvolution with regularization: Y = X / (H + eps)
    let eps = Complex::new(epsilon, 0.0);
    for i in 0..x_spec.len() {
        x_spec[i] = x_spec[i] / (h_spec[i] + eps);
    }

    // Inverse FFT
    let mut out_buf = ifft.make_output_vec();
    let mut iscratch = ifft.make_scratch_vec();
    ifft.process_with_scratch(&mut x_spec, &mut out_buf, &mut iscratch).unwrap();

    let norm = 1.0 / fft_size as f32;
    let mut out: Vec<f32> = out_buf.iter().take(x_len).map(|&s| s * norm).collect();

    // Normalize output
    let peak = out.iter().map(|s| s.abs()).fold(0.0f32, f32::max) + 1e-12;
    for s in out.iter_mut() {
        *s /= peak;
    }

    AudioOutput::Mono(out)
}

fn variants_l003() -> Vec<HashMap<String, Value>> {
    vec![
        params!("ir_type" => "exponential", "ir_length_ms" => 30, "epsilon" => 0.01),
        params!("ir_type" => "exponential", "ir_length_ms" => 100, "epsilon" => 0.005),
        params!("ir_type" => "exponential", "ir_length_ms" => 50, "epsilon" => 0.1),
        params!("ir_type" => "gaussian", "ir_length_ms" => 50, "epsilon" => 0.01),
        params!("ir_type" => "gaussian", "ir_length_ms" => 150, "epsilon" => 0.05),
        params!("ir_type" => "exponential", "ir_length_ms" => 200, "epsilon" => 0.001),
    ]
}

// ---------------------------------------------------------------------------
// L004 -- Spectral Morphing
// ---------------------------------------------------------------------------

/// Generate a target magnitude spectrum for morphing.
fn generate_target_spectrum(target_type: &str, num_bins: usize, sr: u32, fft_size: usize) -> Vec<f32> {
    let freqs: Vec<f32> = (0..num_bins)
        .map(|i| i as f32 * sr as f32 / fft_size as f32)
        .collect();

    match target_type {
        "noise" => {
            // Flat white noise spectrum
            vec![1.0f32; num_bins]
        }
        "sawtooth" => {
            // Sawtooth: 1/k harmonic series based on 100 Hz fundamental
            let mut mag = vec![0.0f32; num_bins];
            let f0 = 100.0f32;
            for k in 1..num_bins {
                let harmonic_freq = f0 * k as f32;
                let bin_idx = (harmonic_freq * fft_size as f32 / sr as f32) as usize;
                if bin_idx < num_bins {
                    mag[bin_idx] += 1.0 / k as f32;
                }
            }
            // Smooth slightly with [0.25, 0.5, 0.25] kernel
            let mut smoothed = vec![0.0f32; num_bins];
            // Pad mag with zeros at edges
            for i in 0..num_bins {
                let left = if i > 0 { mag[i - 1] } else { 0.0 };
                let center = mag[i];
                let right = if i + 1 < num_bins { mag[i + 1] } else { 0.0 };
                smoothed[i] = 0.25 * left + 0.5 * center + 0.25 * right;
            }
            // Add small floor
            for s in smoothed.iter_mut() {
                *s += 0.01;
            }
            smoothed
        }
        "formant" => {
            // Vocal formant: peaks at ~700, 1200, 2500 Hz
            let formant_freqs = [700.0f32, 1200.0, 2500.0];
            let formant_bws = [130.0f32, 70.0, 160.0];
            let formant_amps = [1.0f32, 0.6, 0.3];
            let mut mag = vec![0.0f32; num_bins];
            for fi in 0..formant_freqs.len() {
                let fc = formant_freqs[fi];
                let bw = formant_bws[fi];
                let amp = formant_amps[fi];
                for i in 0..num_bins {
                    mag[i] += amp * (-0.5 * ((freqs[i] - fc) / bw).powi(2)).exp();
                }
            }
            // Add small floor
            for s in mag.iter_mut() {
                *s += 0.01;
            }
            mag
        }
        _ => {
            vec![1.0f32; num_bins]
        }
    }
}

fn process_l004(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let target_type = ps(params, "target_type", "noise");
    let alpha = pf(params, "alpha", 0.5).clamp(0.0, 1.0);

    let n = samples.len();
    let fft_size = 2048;
    let hop_size = fft_size / 4;

    // Analysis via STFT
    let spec = stft(samples, fft_size, hop_size);
    let num_bins = fft_size / 2 + 1;
    let num_frames = spec.len();

    // Generate target magnitude spectrum
    let mut target_mag = generate_target_spectrum(target_type, num_bins, sr, fft_size);

    // Compute average magnitude across frames
    let mut avg_mag = vec![0.0f32; num_bins];
    for frame in &spec {
        for (i, c) in frame.iter().enumerate() {
            avg_mag[i] += (c.re * c.re + c.im * c.im).sqrt();
        }
    }
    if num_frames > 0 {
        for m in avg_mag.iter_mut() {
            *m /= num_frames as f32;
            *m += 1e-12;
        }
    }

    // Normalize target to similar energy as average frame
    let avg_mag_mean: f32 = avg_mag.iter().sum::<f32>() / num_bins as f32;
    let target_mag_mean: f32 = target_mag.iter().sum::<f32>() / num_bins as f32 + 1e-12;
    let scale = avg_mag_mean / target_mag_mean;
    for t in target_mag.iter_mut() {
        *t *= scale;
    }

    // Morph each frame: interpolate magnitude, preserve phase
    let mut morphed: Vec<Vec<Complex<f32>>> = Vec::with_capacity(num_frames);
    for frame in &spec {
        let mut new_frame = Vec::with_capacity(num_bins);
        for (i, c) in frame.iter().enumerate() {
            let mag = (c.re * c.re + c.im * c.im).sqrt();
            let phase = c.im.atan2(c.re);
            let new_mag = (1.0 - alpha) * mag + alpha * target_mag[i];
            new_frame.push(Complex::new(new_mag * phase.cos(), new_mag * phase.sin()));
        }
        morphed.push(new_frame);
    }

    // Synthesis via ISTFT
    let out = istft(&morphed, fft_size, hop_size, Some(n));

    AudioOutput::Mono(out)
}

fn variants_l004() -> Vec<HashMap<String, Value>> {
    vec![
        params!("target_type" => "noise", "alpha" => 0.3),
        params!("target_type" => "noise", "alpha" => 0.7),
        params!("target_type" => "noise", "alpha" => 1.0),
        params!("target_type" => "sawtooth", "alpha" => 0.5),
        params!("target_type" => "sawtooth", "alpha" => 0.9),
        params!("target_type" => "formant", "alpha" => 0.4),
        params!("target_type" => "formant", "alpha" => 0.8),
    ]
}

// ---------------------------------------------------------------------------
// L005 -- Convolution with Chirp IR
// ---------------------------------------------------------------------------

fn process_l005(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let f_start = pf(params, "f_start", 100.0) as f64;
    let f_end = pf(params, "f_end", 8000.0) as f64;
    let chirp_duration_ms = pf(params, "chirp_duration_ms", 100.0);

    let chirp_samples = ((chirp_duration_ms * sr as f32 / 1000.0) as usize).max(1);
    let sr_f = sr as f64;

    let t: Vec<f64> = (0..chirp_samples).map(|i| i as f64 / sr_f).collect();
    let duration = t.last().copied().unwrap_or(0.0) + 1e-12;

    // Linear chirp: instantaneous frequency sweeps linearly from f_start to f_end
    let ir: Vec<f64> = t.iter()
        .map(|&t_val| {
            let phase = 2.0 * std::f64::consts::PI
                * (f_start * t_val + (f_end - f_start) / (2.0 * duration) * t_val * t_val);
            let envelope = (-3.0 * t_val / duration).exp();
            phase.sin() * envelope
        })
        .collect();

    // Normalize IR
    let energy = (ir.iter().map(|&x| x * x).sum::<f64>() + 1e-12).sqrt();
    let ir_f32: Vec<f32> = ir.iter().map(|&x| (x / energy) as f32).collect();

    AudioOutput::Mono(fft_convolve(samples, &ir_f32))
}

fn variants_l005() -> Vec<HashMap<String, Value>> {
    vec![
        params!("f_start" => 100, "f_end" => 4000, "chirp_duration_ms" => 50),
        params!("f_start" => 50, "f_end" => 10000, "chirp_duration_ms" => 100),
        params!("f_start" => 200, "f_end" => 2000, "chirp_duration_ms" => 200),
        params!("f_start" => 500, "f_end" => 15000, "chirp_duration_ms" => 30),
        params!("f_start" => 20, "f_end" => 8000, "chirp_duration_ms" => 500),
        params!("f_start" => 1000, "f_end" => 1500, "chirp_duration_ms" => 150),
    ]
}

// ---------------------------------------------------------------------------
// L006 -- Morphological Audio Processing
// ---------------------------------------------------------------------------

/// Dilation: sliding maximum.
fn morpho_dilate(samples: &[f32], kernel_size: usize) -> Vec<f32> {
    let n = samples.len();
    let half = kernel_size / 2;
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        let mut max_val = f32::NEG_INFINITY;
        let k_start = if i >= half { i - half } else { 0 };
        let k_end = (i + half + 1).min(n);
        for idx in k_start..k_end {
            if samples[idx] > max_val {
                max_val = samples[idx];
            }
        }
        out[i] = max_val;
    }
    out
}

/// Erosion: sliding minimum.
fn morpho_erode(samples: &[f32], kernel_size: usize) -> Vec<f32> {
    let n = samples.len();
    let half = kernel_size / 2;
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        let mut min_val = f32::INFINITY;
        let k_start = if i >= half { i - half } else { 0 };
        let k_end = (i + half + 1).min(n);
        for idx in k_start..k_end {
            if samples[idx] < min_val {
                min_val = samples[idx];
            }
        }
        out[i] = min_val;
    }
    out
}

fn process_l006(samples: &[f32], _sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let mut kernel_size = pi(params, "kernel_size", 11) as usize;
    let operation = ps(params, "operation", "dilate");

    // Force odd kernel size
    if kernel_size % 2 == 0 {
        kernel_size += 1;
    }
    kernel_size = kernel_size.clamp(3, 51);

    match operation {
        "dilate" => AudioOutput::Mono(morpho_dilate(samples, kernel_size)),
        "erode" => AudioOutput::Mono(morpho_erode(samples, kernel_size)),
        "open" => {
            // Opening = erosion followed by dilation
            let eroded = morpho_erode(samples, kernel_size);
            AudioOutput::Mono(morpho_dilate(&eroded, kernel_size))
        }
        "close" => {
            // Closing = dilation followed by erosion
            let dilated = morpho_dilate(samples, kernel_size);
            AudioOutput::Mono(morpho_erode(&dilated, kernel_size))
        }
        _ => AudioOutput::Mono(morpho_dilate(samples, kernel_size)),
    }
}

fn variants_l006() -> Vec<HashMap<String, Value>> {
    vec![
        params!("kernel_size" => 5, "operation" => "dilate"),
        params!("kernel_size" => 21, "operation" => "dilate"),
        params!("kernel_size" => 5, "operation" => "erode"),
        params!("kernel_size" => 21, "operation" => "erode"),
        params!("kernel_size" => 11, "operation" => "open"),
        params!("kernel_size" => 11, "operation" => "close"),
        params!("kernel_size" => 31, "operation" => "open"),
        params!("kernel_size" => 31, "operation" => "close"),
    ]
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

pub fn register() -> Vec<EffectEntry> {
    vec![
        EffectEntry {
            id: "L001",
            process: process_l001,
            variants: variants_l001,
            category: "convolution",
        },
        EffectEntry {
            id: "L002",
            process: process_l002,
            variants: variants_l002,
            category: "convolution",
        },
        EffectEntry {
            id: "L003",
            process: process_l003,
            variants: variants_l003,
            category: "convolution",
        },
        EffectEntry {
            id: "L004",
            process: process_l004,
            variants: variants_l004,
            category: "convolution",
        },
        EffectEntry {
            id: "L005",
            process: process_l005,
            variants: variants_l005,
            category: "convolution",
        },
        EffectEntry {
            id: "L006",
            process: process_l006,
            variants: variants_l006,
            category: "convolution",
        },
    ]
}
