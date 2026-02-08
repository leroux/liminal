//! STFT-based spectral degradation engine.
//!
//! Port of `lossy/engine/spectral.py`.
//!
//! Core algorithm: window -> FFT -> degrade spectral content -> IFFT -> overlap-add.
//!
//! Magnitude processing (controlled by loss):
//!   Standard -- spectral quantization + psychoacoustic band gating
//!   Inverse  -- output the spectral residual (everything Standard discards)
//!
//! Phase processing:
//!   Phase Loss -- deterministic quantization of phase angles
//!   Jitter     -- random phase perturbation

use crate::params::{LossyParams, SR};
use crate::rng::NumpyRng;
use num_complex::Complex;
use realfft::RealFftPlanner;

/// Run STFT spectral degradation on input audio.
pub fn spectral_process(input_audio: &[f64], params: &LossyParams) -> Vec<f64> {
    let g = params.global_amount;
    let loss = params.loss * g;
    let inverse = params.inverse != 0;
    let jitter = params.jitter * g;
    let seed = params.seed;
    let freeze = params.freeze != 0;
    let freeze_mode = params.freeze_mode;
    let freezer_blend = params.freezer;
    let phase_loss = params.phase_loss * g;
    let quantizer_type = params.quantizer;
    let pre_echo_amount = params.pre_echo * g;
    let noise_shape = params.noise_shape;
    let weighting = params.weighting;
    let hf_threshold = params.hf_threshold;
    let transient_ratio = params.transient_ratio;
    let slushy_rate_param = params.slushy_rate;

    if loss <= 0.0 && !freeze && phase_loss <= 0.0 && jitter <= 0.0 {
        return input_audio.to_vec();
    }

    // Window and hop from params
    let window_size = (params.window_size.max(2)) as usize;
    let hop_divisor = (params.hop_divisor.max(1)) as usize;
    let hop_size = (window_size / hop_divisor).max(1);
    let n_bins = window_size / 2 + 1;
    let n_bands_param = (params.n_bands.max(2)) as usize;

    let window = hann_window(window_size);

    // Pad input so edges are handled cleanly (reflection or zero padding)
    let pad = window_size;
    let padded = pad_reflect(input_audio, pad);
    let n_samples = padded.len();

    let mut output = vec![0.0_f64; n_samples];
    let mut win_sum = vec![0.0_f64; n_samples];

    let n_frames = if n_samples >= window_size {
        (n_samples - window_size) / hop_size + 1
    } else {
        0
    };

    let mut rng = NumpyRng::new(seed as u32);

    // Bark-like log-spaced band edges
    let (band_edges, n_bands) = compute_band_edges(n_bins, n_bands_param);

    // ATH weighting per band
    let ath_weights = compute_ath_weights(&band_edges, n_bands, n_bins, window_size);

    // Pre-echo detection pass
    let transient_flags = if pre_echo_amount > 0.0 && n_frames > 1 {
        let mut energies = vec![0.0_f64; n_frames];
        for fi in 0..n_frames {
            let start = fi * hop_size;
            let mut e = 0.0;
            for j in 0..window_size {
                let s = padded[start + j] * window[j];
                e += s * s;
            }
            energies[fi] = e;
        }
        let mut flags = vec![false; n_frames];
        for fi in 1..n_frames {
            if energies[fi - 1] > 1e-12 && energies[fi] / energies[fi - 1] > transient_ratio {
                flags[fi] = true;
            }
        }
        Some(flags)
    } else {
        None
    };

    // Set up FFT planner
    let mut planner = RealFftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(window_size);
    let ifft = planner.plan_fft_inverse(window_size);

    let mut fft_input = vec![0.0_f64; window_size];
    let mut spectrum = vec![Complex::new(0.0, 0.0); n_bins];
    let mut ifft_output = vec![0.0_f64; window_size];

    let mut frozen_spectrum: Option<Vec<f64>> = None;

    for fi in 0..n_frames {
        let start = fi * hop_size;

        // Window the frame
        for j in 0..window_size {
            fft_input[j] = padded[start + j] * window[j];
        }

        // Forward FFT
        fft.process(&mut fft_input, &mut spectrum).unwrap();

        // Extract magnitude and phase
        let magnitudes: Vec<f64> = spectrum.iter().map(|c| c.norm()).collect();
        let mut phases: Vec<f64> = spectrum.iter().map(|c| c.arg()).collect();

        // Pre-echo: boost loss for frames right before a transient
        let mut frame_loss = loss;
        if let Some(ref flags) = transient_flags {
            if fi < n_frames - 1 && flags[fi + 1] {
                frame_loss = (loss + pre_echo_amount * 0.5).min(1.0);
            }
        }

        // ---------- magnitude processing ----------
        let mut proc_mag = if frame_loss > 0.0 {
            standard_degrade(
                &magnitudes,
                frame_loss,
                &mut rng,
                &band_edges,
                n_bands,
                &ath_weights,
                quantizer_type,
                noise_shape,
                weighting,
            )
        } else {
            magnitudes.clone()
        };

        if inverse {
            for i in 0..proc_mag.len() {
                proc_mag[i] = (magnitudes[i] - proc_mag[i]).max(0.0);
            }
        }

        // ---------- phase processing ----------
        if phase_loss > 0.0 {
            let n_levels = (64.0 * (1.0 - phase_loss)).max(4.0) as i32;
            let step = 2.0 * std::f64::consts::PI / n_levels as f64;
            for p in phases.iter_mut() {
                *p = step * (*p / step).round();
            }
        }
        if jitter > 0.0 {
            let pi = std::f64::consts::PI;
            for p in phases.iter_mut() {
                let noise = rng.uniform(-pi, pi) * jitter;
                *p += noise;
            }
        }

        // ---------- bandwidth limiting (like low-bitrate MP3) ----------
        if frame_loss > hf_threshold {
            let cutoff = ((n_bins as f64 * (1.0 - 0.6 * frame_loss)) as usize).max(n_bins / 8);
            let hf_range = if hf_threshold < 1.0 {
                1.0 - hf_threshold
            } else {
                1.0
            };
            let mult = ((1.0 - (frame_loss - hf_threshold) / hf_range).max(0.0)).min(1.0);
            for i in cutoff..proc_mag.len() {
                proc_mag[i] *= mult;
            }
        }

        // ---------- freeze ----------
        if freeze {
            if frozen_spectrum.is_none() {
                frozen_spectrum = Some(proc_mag.clone());
            }
            let frozen = frozen_spectrum.as_mut().unwrap();
            if freeze_mode == 1 {
                // Solid: use frozen spectrum directly
                // frozen_spectrum stays unchanged
            } else {
                // Slushy: drift frozen spectrum toward live signal
                for i in 0..frozen.len() {
                    frozen[i] = (1.0 - slushy_rate_param) * frozen[i]
                        + slushy_rate_param * proc_mag[i];
                }
            }
            // Blend
            for i in 0..proc_mag.len() {
                proc_mag[i] =
                    freezer_blend * frozen[i] + (1.0 - freezer_blend) * proc_mag[i];
            }
        }

        // ---------- reconstruct ----------
        for i in 0..n_bins {
            spectrum[i] = Complex::from_polar(proc_mag[i], phases[i]);
        }

        // realfft requires DC and Nyquist bins to have zero imaginary part
        spectrum[0] = Complex::new(spectrum[0].re, 0.0);
        if n_bins > 1 {
            let last = n_bins - 1;
            spectrum[last] = Complex::new(spectrum[last].re, 0.0);
        }

        // Inverse FFT
        ifft.process(&mut spectrum, &mut ifft_output).unwrap();

        // realfft inverse does NOT normalize — divide by window_size
        let norm = 1.0 / window_size as f64;

        // Overlap-add
        for j in 0..window_size {
            output[start + j] += ifft_output[j] * norm * window[j];
            win_sum[start + j] += window[j] * window[j];
        }
    }

    // Normalize overlap-add
    for i in 0..n_samples {
        if win_sum[i] < 1e-8 {
            win_sum[i] = 1.0;
        }
        output[i] /= win_sum[i];
    }

    // Remove padding
    output[pad..pad + input_audio.len()].to_vec()
}

// ---------- internal helpers ----------

fn hann_window(size: usize) -> Vec<f64> {
    (0..size)
        .map(|i| {
            0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / size as f64).cos())
        })
        .collect()
}

fn pad_reflect(audio: &[f64], pad: usize) -> Vec<f64> {
    let n = audio.len();
    if n > pad {
        // Reflection padding
        let mut padded = Vec::with_capacity(n + 2 * pad);
        // Left reflection
        for i in (0..pad).rev() {
            let idx = (i + 1) % n;
            padded.push(audio[idx]);
        }
        padded.extend_from_slice(audio);
        // Right reflection
        for i in 0..pad {
            let idx = n.saturating_sub(2).saturating_sub(i % n.max(1));
            padded.push(audio[idx.min(n - 1)]);
        }
        padded
    } else {
        // Zero padding for very short audio
        let mut padded = vec![0.0; pad];
        padded.extend_from_slice(audio);
        padded.extend(vec![0.0; pad]);
        padded
    }
}

fn compute_band_edges(n_bins: usize, n_bands_param: usize) -> (Vec<usize>, usize) {
    // Log-spaced edges mimicking Bark scale
    let mut edges: Vec<usize> = (0..=n_bands_param)
        .map(|i| {
            let t = i as f64 / n_bands_param as f64;
            let v = (10.0_f64).powf(t * (n_bins as f64).log10());
            (v as usize).min(n_bins)
        })
        .collect();

    // Deduplicate and sort
    edges.sort();
    edges.dedup();

    let n_bands = if edges.len() > 1 {
        edges.len() - 1
    } else {
        0
    };
    (edges, n_bands)
}

/// Absolute Threshold of Hearing weighting per band (Terhardt's approximation).
fn compute_ath_weights(
    band_edges: &[usize],
    n_bands: usize,
    n_bins: usize,
    window_size: usize,
) -> Vec<f64> {
    // Per-bin ATH
    let bin_ath: Vec<f64> = (0..n_bins)
        .map(|i| {
            let freq = i as f64 * SR / window_size as f64;
            let f_khz = (freq / 1000.0).clamp(0.02, 20.0);
            3.64 * f_khz.powf(-0.8) - 6.5 * (-0.6 * (f_khz - 3.3).powi(2)).exp()
                + 1e-3 * f_khz.powi(4)
        })
        .collect();

    // Normalize to 0-1
    let min_ath = bin_ath.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_range = bin_ath.iter().cloned().fold(0.0_f64, f64::max) - min_ath;
    let ath_norm: Vec<f64> = if max_range > 0.0 {
        bin_ath.iter().map(|&a| (a - min_ath) / max_range).collect()
    } else {
        vec![0.0; n_bins]
    };

    // Average per band
    let mut weights = vec![0.0_f64; n_bands];
    for b in 0..n_bands {
        let lo = band_edges[b];
        let hi = band_edges[b + 1];
        if hi > lo {
            let sum: f64 = ath_norm[lo..hi].iter().sum();
            weights[b] = sum / (hi - lo) as f64;
        }
    }
    weights
}

/// Standard mode: quantization + psychoacoustic band gating.
fn standard_degrade(
    magnitudes: &[f64],
    loss: f64,
    rng: &mut NumpyRng,
    band_edges: &[usize],
    n_bands: usize,
    ath_weights: &[f64],
    quantizer_type: i32,
    noise_shape: f64,
    weighting: f64,
) -> Vec<f64> {
    if loss <= 0.0 {
        return magnitudes.to_vec();
    }

    let mut proc = magnitudes.to_vec();

    // ---- Quantization ----
    let max_mag = proc.iter().cloned().fold(0.0_f64, f64::max);
    if max_mag > 0.0 {
        let bits = 16.0 - 14.0 * loss; // loss 0 -> 16 bits, loss 1 -> 2 bits
        let n_levels = (2.0_f64).powf(bits);

        if quantizer_type == 1 {
            // Compand (power-law, MP3-style)
            let mut compressed: Vec<f64> = proc.iter().map(|&x| x.powf(0.75)).collect();
            let max_c = compressed.iter().cloned().fold(0.0_f64, f64::max);
            if max_c > 0.0 {
                if noise_shape > 0.0 {
                    let base_delta = 2.0 * max_c / n_levels;
                    let shaped = shape_delta(&compressed, base_delta, noise_shape);
                    for i in 0..compressed.len() {
                        let d = shaped[i].max(1e-20);
                        compressed[i] = d * (compressed[i] / d).round();
                    }
                } else {
                    let delta = 2.0 * max_c / n_levels;
                    for c in compressed.iter_mut() {
                        *c = delta * (*c / delta).round();
                    }
                }
            }
            for i in 0..proc.len() {
                proc[i] = compressed[i].max(0.0).powf(4.0 / 3.0);
            }
        } else {
            // Uniform (mid-tread)
            if noise_shape > 0.0 {
                let base_delta = 2.0 * max_mag / n_levels;
                let shaped = shape_delta(&proc, base_delta, noise_shape);
                let old = proc.clone();
                for i in 0..proc.len() {
                    let d = shaped[i].max(1e-20);
                    proc[i] = d * (old[i] / d).round();
                }
            } else {
                let delta = 2.0 * max_mag / n_levels;
                for p in proc.iter_mut() {
                    *p = delta * (*p / delta).round();
                }
            }
        }
    }

    // ---- Psychoacoustic band gating ----
    let mut band_energy = vec![0.0_f64; n_bands];
    for b in 0..n_bands {
        let lo = band_edges[b];
        let hi = band_edges[b + 1];
        if hi > lo {
            let sum: f64 = proc[lo..hi].iter().map(|x| x * x).sum();
            band_energy[b] = sum / (hi - lo) as f64;
        }
    }

    let mean_energy = band_energy.iter().sum::<f64>() / n_bands.max(1) as f64 + 1e-12;

    for b in 0..n_bands {
        // Signal-dependent: low-energy bands more likely to be gated
        let relative = (band_energy[b] / mean_energy).min(2.0) / 2.0;
        // ATH-weighted
        let ath_factor = (1.0 - weighting) * 0.75 + weighting * (0.5 + 0.5 * ath_weights[b]);
        // Combined gating probability
        let mut gate_prob = loss * 0.6 * (1.0 - relative) * ath_factor;
        // Random perturbation
        gate_prob += rng.random() * loss * 0.2;
        if rng.random() < gate_prob {
            let lo = band_edges[b];
            let hi = band_edges[b + 1];
            for i in lo..hi {
                proc[i] = 0.0;
            }
        }
    }

    proc
}

/// Envelope-following noise shaping.
fn shape_delta(magnitudes: &[f64], base_delta: f64, amount: f64) -> Vec<f64> {
    let n = magnitudes.len();
    if n == 0 {
        return vec![];
    }

    // 7-sample moving average for envelope
    let kernel_size = 7;
    let mut envelope = vec![0.0_f64; n];
    for i in 0..n {
        let lo = i.saturating_sub(kernel_size / 2);
        let hi = (i + kernel_size / 2 + 1).min(n);
        let sum: f64 = magnitudes[lo..hi].iter().sum();
        envelope[i] = (sum / (hi - lo) as f64).max(1e-12);
    }

    // Inverse envelope, normalized
    let mut inv_env: Vec<f64> = envelope.iter().map(|&e| 1.0 / e).collect();
    let max_inv = inv_env.iter().cloned().fold(0.0_f64, f64::max);
    if max_inv > 0.0 {
        for v in inv_env.iter_mut() {
            *v /= max_inv;
        }
    }

    // amount=0 -> uniform delta, amount=1 -> 4x coarser in valleys
    inv_env
        .iter()
        .map(|&v| base_delta * (1.0 + amount * v * 3.0))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_passthrough_when_no_effect() {
        let audio: Vec<f64> = (0..4096).map(|i| (i as f64 * 0.01).sin()).collect();
        let params = LossyParams {
            loss: 0.0,
            freeze: 0,
            phase_loss: 0.0,
            jitter: 0.0,
            ..Default::default()
        };
        let out = spectral_process(&audio, &params);
        assert_eq!(out.len(), audio.len());
        // Should be nearly identical (passthrough)
        for (a, b) in audio.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-10, "diff at sample: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_loss_degrades_signal() {
        let audio: Vec<f64> = (0..44100)
            .map(|i| (2.0 * std::f64::consts::PI * 440.0 * i as f64 / SR).sin())
            .collect();
        let mut params = LossyParams::default();
        params.loss = 0.8;
        let out = spectral_process(&audio, &params);
        assert_eq!(out.len(), audio.len());
        // Output should differ from input
        let diff: f64 = audio
            .iter()
            .zip(out.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 1.0);
    }

    #[test]
    fn test_output_length_matches() {
        for &len in &[100, 1000, 4096, 10000, 44100] {
            let audio: Vec<f64> = (0..len).map(|i| (i as f64 * 0.01).sin()).collect();
            let params = LossyParams::default();
            let out = spectral_process(&audio, &params);
            assert_eq!(out.len(), len, "length mismatch for input size {}", len);
        }
    }

    #[test]
    fn test_hann_window() {
        let w = hann_window(1024);
        assert_eq!(w.len(), 1024);
        assert!(w[0].abs() < 1e-10); // starts at 0
        assert!((w[512] - 1.0).abs() < 1e-10); // peak at middle
    }

    #[test]
    fn test_inverse_mode() {
        let audio: Vec<f64> = (0..8192)
            .map(|i| (2.0 * std::f64::consts::PI * 1000.0 * i as f64 / SR).sin())
            .collect();
        let mut params = LossyParams::default();
        params.loss = 0.5;
        params.inverse = 0;
        let standard = spectral_process(&audio, &params);

        params.inverse = 1;
        let residual = spectral_process(&audio, &params);

        // Standard + residual should roughly reconstruct original
        // (Not exact due to quantization and gating randomness, but energy should be similar)
        let energy_in: f64 = audio.iter().map(|x| x * x).sum();
        let energy_std: f64 = standard.iter().map(|x| x * x).sum();
        let energy_res: f64 = residual.iter().map(|x| x * x).sum();
        // Total energy of parts should be in the right ballpark
        assert!(energy_std + energy_res > energy_in * 0.1);
    }
}
