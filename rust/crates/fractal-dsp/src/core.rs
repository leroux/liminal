//! Core fractal algorithm.
//!
//! Port of `fractal/engine/core.py`.
//!
//! Compress audio into progressively shorter copies, tile them to fill the
//! original length, and sum with decaying gains to create self-similar
//! texture at multiple timescales.

use crate::filters::{apply_one_pole_hp, apply_one_pole_lp};
use crate::params::FractalParams;
use num_complex::Complex64;
use realfft::RealFftPlanner;

// ---------------------------------------------------------------------------
// Time-domain fractalization
// ---------------------------------------------------------------------------

/// Nearest-neighbor resampling variant (aliased, gritty).
fn fractalize_time_nearest(
    samples: &[f64],
    num_scales: i32,
    scale_ratio: f64,
    amplitude_decay: f64,
    reverse_scales: bool,
    scale_offset: f64,
) -> Vec<f64> {
    let n = samples.len();
    let mut out = samples.to_vec();

    for s in 1..num_scales {
        let compressed_len = (n as f64 * scale_ratio.powi(s)).max(1.0) as usize;
        let gain = amplitude_decay.powi(s);

        // Build compressed version via nearest-neighbor
        let mut compressed = vec![0.0_f64; compressed_len];
        for i in 0..compressed_len {
            let idx = if compressed_len <= 1 {
                0
            } else {
                ((i * (n - 1)) / (compressed_len - 1)).min(n - 1)
            };
            compressed[i] = samples[idx];
        }

        // Optionally reverse
        if reverse_scales {
            compressed.reverse();
        }

        // Tile with offset
        let offset_samples = (scale_offset * compressed_len as f64) as usize;
        for i in 0..n {
            let src = (i + offset_samples) % compressed_len;
            out[i] += gain * compressed[src];
        }
    }

    out
}

/// Linear interpolation resampling variant (smoother).
fn fractalize_time_linear(
    samples: &[f64],
    num_scales: i32,
    scale_ratio: f64,
    amplitude_decay: f64,
    reverse_scales: bool,
    scale_offset: f64,
) -> Vec<f64> {
    let n = samples.len();
    let mut out = samples.to_vec();

    for s in 1..num_scales {
        let compressed_len = (n as f64 * scale_ratio.powi(s)).max(2.0) as usize;
        let gain = amplitude_decay.powi(s);

        // Build compressed version via linear interpolation
        let mut compressed = vec![0.0_f64; compressed_len];
        for i in 0..compressed_len {
            let pos = if compressed_len <= 1 {
                0.0
            } else {
                i as f64 * (n - 1) as f64 / (compressed_len - 1) as f64
            };
            let idx0 = pos as usize;
            let idx1 = (idx0 + 1).min(n - 1);
            let frac = pos - idx0 as f64;
            compressed[i] = samples[idx0] * (1.0 - frac) + samples[idx1] * frac;
        }

        // Optionally reverse
        if reverse_scales {
            compressed.reverse();
        }

        // Tile with offset
        let offset_samples = (scale_offset * compressed_len as f64) as usize;
        for i in 0..n {
            let src = (i + offset_samples) % compressed_len;
            out[i] += gain * compressed[src];
        }
    }

    out
}

/// Apply time-domain fractalization to mono audio.
pub fn fractalize_time(samples: &[f64], params: &FractalParams) -> Vec<f64> {
    let num_scales = params.num_scales.clamp(2, 8);
    let scale_ratio = params.scale_ratio.clamp(0.1, 0.9);
    let amplitude_decay = params.amplitude_decay.clamp(0.1, 1.0);
    let reverse_scales = params.reverse_scales != 0;
    let scale_offset = params.scale_offset.clamp(0.0, 1.0);

    let out = if params.interp == 1 {
        fractalize_time_linear(
            samples,
            num_scales,
            scale_ratio,
            amplitude_decay,
            reverse_scales,
            scale_offset,
        )
    } else {
        fractalize_time_nearest(
            samples,
            num_scales,
            scale_ratio,
            amplitude_decay,
            reverse_scales,
            scale_offset,
        )
    };

    // Normalize to input peak to prevent clipping
    normalize_to_input_peak(samples, out)
}

// ---------------------------------------------------------------------------
// Per-layer fractalization (features 1, 2, 4, 5, 7)
// ---------------------------------------------------------------------------

/// Produce per-layer buffers instead of accumulating into one.
/// Returns Vec<Vec<f64>> where index 0 is the original (or silence if only_wet).
pub fn fractalize_time_layers(samples: &[f64], params: &FractalParams) -> Vec<Vec<f64>> {
    let n = samples.len();
    let num_scales = params.num_scales.clamp(2, 8);
    let scale_ratio = params.scale_ratio.clamp(0.1, 0.9);
    let amplitude_decay = params.amplitude_decay.clamp(0.1, 1.0);
    let reverse_scales = params.reverse_scales != 0;
    let scale_offset = params.scale_offset.clamp(0.0, 1.0);
    let layer_detune = params.layer_detune.clamp(0.0, 1.0);
    let linear = params.interp == 1;

    let mut layers = Vec::with_capacity(num_scales as usize);

    // Layer 0: original signal or silence (feature 2: fractal_only_wet)
    if params.fractal_only_wet == 0 {
        layers.push(samples.to_vec());
    } else {
        layers.push(vec![0.0; n]);
    }

    for s in 1..num_scales {
        // Base compressed length with optional detune (feature 4)
        let mut cl = n as f64 * scale_ratio.powi(s);
        if layer_detune > 0.0 {
            let sign = if s % 2 == 0 { 1.0 } else { -1.0 };
            cl *= 1.0 + layer_detune * 0.1 * sign;
        }
        let compressed_len = if linear { cl.max(2.0) } else { cl.max(1.0) } as usize;

        let gain = amplitude_decay.powi(s) * params.layer_gain(s);

        // Build compressed version
        let mut compressed = vec![0.0_f64; compressed_len];
        if linear {
            for i in 0..compressed_len {
                let pos = if compressed_len <= 1 {
                    0.0
                } else {
                    i as f64 * (n - 1) as f64 / (compressed_len - 1) as f64
                };
                let idx0 = pos as usize;
                let idx1 = (idx0 + 1).min(n - 1);
                let frac = pos - idx0 as f64;
                compressed[i] = samples[idx0] * (1.0 - frac) + samples[idx1] * frac;
            }
        } else {
            for i in 0..compressed_len {
                let idx = if compressed_len <= 1 {
                    0
                } else {
                    ((i * (n - 1)) / (compressed_len - 1)).min(n - 1)
                };
                compressed[i] = samples[idx];
            }
        }

        if reverse_scales {
            compressed.reverse();
        }

        // Tile with offset and apply gain
        let offset_samples = (scale_offset * compressed_len as f64) as usize;
        let mut layer = vec![0.0_f64; n];
        for i in 0..n {
            let src = (i + offset_samples) % compressed_len;
            layer[i] = gain * compressed[src];
        }

        layers.push(layer);
    }

    layers
}

/// Shift a layer forward in time (feature 5: layer_delay).
fn apply_layer_delay(layer: &mut [f64], delay_samples: usize) {
    if delay_samples == 0 || delay_samples >= layer.len() {
        return;
    }
    layer.rotate_right(delay_samples);
    for s in layer[..delay_samples].iter_mut() {
        *s = 0.0;
    }
}

/// Apply per-layer tilt filtering (feature 7).
fn apply_layer_tilt(layer: &mut [f64], tilt: f64, s: i32, num_scales: i32) {
    if tilt.abs() < 0.001 || s == 0 {
        return;
    }
    let ratio = s as f64 / (num_scales - 1).max(1) as f64;
    if tilt > 0.0 {
        // Positive tilt: darken higher layers
        let cutoff = 20000.0 * (1.0 - tilt * ratio);
        apply_one_pole_lp(layer, cutoff.max(200.0));
    } else {
        // Negative tilt: thin out higher layers
        let cutoff = 20.0 + 5000.0 * tilt.abs() * ratio;
        apply_one_pole_hp(layer, cutoff);
    }
}

// ---------------------------------------------------------------------------
// Spectral-domain fractalization
// ---------------------------------------------------------------------------

/// Apply fractalization to STFT magnitude frames.
pub fn fractalize_spectral(samples: &[f64], params: &FractalParams) -> Vec<f64> {
    let num_scales = params.num_scales.clamp(2, 8);
    let scale_ratio = params.scale_ratio.clamp(0.1, 0.9);
    let amplitude_decay = params.amplitude_decay.clamp(0.1, 1.0);
    let window_size = (params.window_size as usize).clamp(256, 8192);
    let hop_size = window_size / 4;

    let out_len = samples.len();

    // Pad if needed
    let x: Vec<f64> = if samples.len() < window_size {
        let mut padded = samples.to_vec();
        padded.resize(window_size, 0.0);
        padded
    } else {
        samples.to_vec()
    };
    let n = x.len();

    // Hann window
    let window: Vec<f64> = (0..window_size)
        .map(|i| {
            0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / window_size as f64).cos())
        })
        .collect();

    let num_frames = if n >= window_size {
        1 + (n - window_size) / hop_size
    } else {
        0
    };
    if num_frames == 0 {
        return samples.to_vec();
    }

    let num_bins = window_size / 2 + 1;

    // Set up FFT
    let mut planner = RealFftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(window_size);
    let ifft = planner.plan_fft_inverse(window_size);

    // Process each frame
    let mut output = vec![0.0_f64; n.max(window_size + (num_frames - 1) * hop_size)];

    for fi in 0..num_frames {
        let start = fi * hop_size;

        // Window the frame
        let mut frame = vec![0.0_f64; window_size];
        for i in 0..window_size {
            frame[i] = x[start + i] * window[i];
        }

        // Forward FFT
        let mut spectrum = vec![Complex64::new(0.0, 0.0); num_bins];
        fft.process(&mut frame, &mut spectrum).unwrap();

        // Extract magnitude and phase
        let mut mag: Vec<f64> = spectrum.iter().map(|c| c.norm()).collect();
        let phase: Vec<f64> = spectrum.iter().map(|c| c.arg()).collect();
        let orig_mag = mag.clone();

        // Fractalize magnitude
        for s in 1..num_scales {
            let compressed_len = (num_bins as f64 * scale_ratio.powi(s)).max(1.0) as usize;
            let gain = amplitude_decay.powi(s);

            // Downsample magnitude via linear interpolation
            let mut compressed = vec![0.0_f64; compressed_len];
            for i in 0..compressed_len {
                let pos = if compressed_len <= 1 {
                    0.0
                } else {
                    i as f64 * (num_bins - 1) as f64 / (compressed_len - 1) as f64
                };
                let idx0 = pos as usize;
                let idx1 = (idx0 + 1).min(num_bins - 1);
                let frac = pos - idx0 as f64;
                compressed[i] = orig_mag[idx0] * (1.0 - frac) + orig_mag[idx1] * frac;
            }

            // Tile to fill
            for i in 0..num_bins {
                mag[i] += gain * compressed[i % compressed_len];
            }
        }

        // Reconstruct complex spectrum
        let mut recon: Vec<Complex64> = mag
            .iter()
            .zip(phase.iter())
            .map(|(&m, &p)| Complex64::from_polar(m, p))
            .collect();

        // realfft requires DC and Nyquist bins to be purely real
        recon[0] = Complex64::new(recon[0].re, 0.0);
        if num_bins > 1 {
            let last = num_bins - 1;
            recon[last] = Complex64::new(recon[last].re, 0.0);
        }

        // Inverse FFT
        let mut time_buf = vec![0.0_f64; window_size];
        ifft.process(&mut recon, &mut time_buf).unwrap();

        // Apply window and normalize (realfft scales by N)
        let inv_n = 1.0 / window_size as f64;
        for i in 0..window_size {
            let pos = start + i;
            if pos < output.len() {
                output[pos] += time_buf[i] * inv_n * window[i];
            }
        }
    }

    let mut result = output[..out_len].to_vec();

    // Normalize
    normalize_to_input_peak_inplace(samples, &mut result);

    result
}

// ---------------------------------------------------------------------------
// Combined entry point
// ---------------------------------------------------------------------------

/// tanh saturation: blend between clean and saturated.
pub fn apply_saturation(audio: &mut [f64], amount: f64) {
    if amount <= 0.0 {
        return;
    }
    for s in audio.iter_mut() {
        let clean = *s;
        let sat = clean.tanh();
        *s = (1.0 - amount) * clean + amount * sat;
    }
}

/// Apply fractalization with iteration and spectral blend.
/// Uses per-layer processing to support gain, detune, delay, tilt, and only-wet.
pub fn render_fractal_core(samples: &[f64], params: &FractalParams) -> Vec<f64> {
    let spectral_blend = params.spectral.clamp(0.0, 1.0);
    let iterations = params.iterations.clamp(1, 4);
    let iter_decay = params.iter_decay.clamp(0.3, 1.0);
    let saturation = params.saturation.clamp(0.0, 1.0);
    let layer_delay = params.layer_delay.clamp(0.0, 1.0);
    let layer_tilt = params.layer_tilt;
    let num_scales = params.num_scales.clamp(2, 8);

    let mut current = samples.to_vec();

    for it in 0..iterations {
        // Time-domain fractalization via per-layer approach
        let time_out = if spectral_blend < 1.0 {
            let mut layers = fractalize_time_layers(&current, params);
            let n = current.len();

            // Apply per-layer delay (feature 5)
            if layer_delay > 0.0 {
                let max_delay = (n as f64 * 0.1) as usize; // up to 10% of buffer
                for (s, layer) in layers.iter_mut().enumerate() {
                    if s == 0 {
                        continue;
                    }
                    let delay_samples = (layer_delay * max_delay as f64 * s as f64
                        / (num_scales - 1).max(1) as f64) as usize;
                    apply_layer_delay(layer, delay_samples);
                }
            }

            // Apply per-layer tilt (feature 7)
            if layer_tilt.abs() > 0.001 {
                for (s, layer) in layers.iter_mut().enumerate() {
                    apply_layer_tilt(layer, layer_tilt, s as i32, num_scales);
                }
            }

            // Sum all layers
            let mut sum = vec![0.0_f64; n];
            for layer in &layers {
                for (i, &v) in layer.iter().enumerate() {
                    sum[i] += v;
                }
            }

            normalize_to_input_peak(&current, sum)
        } else {
            current.clone()
        };

        // Spectral-domain fractalization
        let spec_out = if spectral_blend > 0.0 {
            fractalize_spectral(&current, params)
        } else {
            current.clone()
        };

        // Blend
        let mut result = if spectral_blend <= 0.0 {
            time_out
        } else if spectral_blend >= 1.0 {
            spec_out
        } else {
            time_out
                .iter()
                .zip(spec_out.iter())
                .map(|(&t, &s)| (1.0 - spectral_blend) * t + spectral_blend * s)
                .collect()
        };

        // Apply saturation between iterations
        if saturation > 0.0 {
            apply_saturation(&mut result, saturation);
        }

        // Apply iteration decay for subsequent passes
        if it < iterations - 1 {
            for s in result.iter_mut() {
                *s *= iter_decay;
            }
        }

        current = result;
    }

    current
}

/// Stereo core with per-layer constant-power pan (feature 3: layer_spread).
pub fn render_fractal_core_stereo(
    left: &[f64],
    right: &[f64],
    params: &FractalParams,
) -> (Vec<f64>, Vec<f64>) {
    let layer_spread = params.layer_spread.clamp(0.0, 1.0);

    if layer_spread <= 0.0 {
        // No spread: process channels independently
        let out_l = render_fractal_core(left, params);
        let out_r = render_fractal_core(right, params);
        return (out_l, out_r);
    }

    let spectral_blend = params.spectral.clamp(0.0, 1.0);
    let iterations = params.iterations.clamp(1, 4);
    let iter_decay = params.iter_decay.clamp(0.3, 1.0);
    let saturation = params.saturation.clamp(0.0, 1.0);
    let layer_delay_param = params.layer_delay.clamp(0.0, 1.0);
    let layer_tilt = params.layer_tilt;
    let num_scales = params.num_scales.clamp(2, 8);
    let n = left.len();

    let mut cur_l = left.to_vec();
    let mut cur_r = right.to_vec();

    for it in 0..iterations {
        let (time_l, time_r) = if spectral_blend < 1.0 {
            let mut layers_l = fractalize_time_layers(&cur_l, params);
            let mut layers_r = fractalize_time_layers(&cur_r, params);

            // Per-layer delay
            if layer_delay_param > 0.0 {
                let max_delay = (n as f64 * 0.1) as usize;
                for s in 1..layers_l.len() {
                    let ds = (layer_delay_param * max_delay as f64 * s as f64
                        / (num_scales - 1).max(1) as f64) as usize;
                    apply_layer_delay(&mut layers_l[s], ds);
                    apply_layer_delay(&mut layers_r[s], ds);
                }
            }

            // Per-layer tilt
            if layer_tilt.abs() > 0.001 {
                for s in 0..layers_l.len() {
                    apply_layer_tilt(&mut layers_l[s], layer_tilt, s as i32, num_scales);
                    apply_layer_tilt(&mut layers_r[s], layer_tilt, s as i32, num_scales);
                }
            }

            // Combine with per-layer constant-power pan
            let mut out_l = vec![0.0_f64; n];
            let mut out_r = vec![0.0_f64; n];

            for (s, (ll, lr)) in layers_l.iter().zip(layers_r.iter()).enumerate() {
                if s == 0 {
                    // Layer 0 stays center
                    for i in 0..n {
                        out_l[i] += ll[i];
                        out_r[i] += lr[i];
                    }
                } else {
                    // Alternate layers left/right
                    let sign = if s % 2 == 0 { 1.0 } else { -1.0 };
                    let pan = layer_spread * sign * s as f64
                        / (num_scales - 1).max(1) as f64;
                    let theta = (pan + 1.0) * std::f64::consts::FRAC_PI_4;
                    let gain_l = theta.cos();
                    let gain_r = theta.sin();
                    for i in 0..n {
                        out_l[i] += ll[i] * gain_l + lr[i] * gain_l;
                        out_r[i] += ll[i] * gain_r + lr[i] * gain_r;
                    }
                }
            }

            let norm_l = normalize_to_input_peak(&cur_l, out_l);
            let norm_r = normalize_to_input_peak(&cur_r, out_r);
            (norm_l, norm_r)
        } else {
            (cur_l.clone(), cur_r.clone())
        };

        // Spectral blend per channel
        let (spec_l, spec_r) = if spectral_blend > 0.0 {
            (
                fractalize_spectral(&cur_l, params),
                fractalize_spectral(&cur_r, params),
            )
        } else {
            (cur_l.clone(), cur_r.clone())
        };

        let (mut res_l, mut res_r) = if spectral_blend <= 0.0 {
            (time_l, time_r)
        } else if spectral_blend >= 1.0 {
            (spec_l, spec_r)
        } else {
            let bl: Vec<f64> = time_l
                .iter()
                .zip(spec_l.iter())
                .map(|(&t, &s)| (1.0 - spectral_blend) * t + spectral_blend * s)
                .collect();
            let br: Vec<f64> = time_r
                .iter()
                .zip(spec_r.iter())
                .map(|(&t, &s)| (1.0 - spectral_blend) * t + spectral_blend * s)
                .collect();
            (bl, br)
        };

        if saturation > 0.0 {
            apply_saturation(&mut res_l, saturation);
            apply_saturation(&mut res_r, saturation);
        }

        if it < iterations - 1 {
            for s in res_l.iter_mut() {
                *s *= iter_decay;
            }
            for s in res_r.iter_mut() {
                *s *= iter_decay;
            }
        }

        cur_l = res_l;
        cur_r = res_r;
    }

    (cur_l, cur_r)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn normalize_to_input_peak(input: &[f64], mut output: Vec<f64>) -> Vec<f64> {
    normalize_to_input_peak_inplace(input, &mut output);
    output
}

fn normalize_to_input_peak_inplace(input: &[f64], output: &mut [f64]) {
    let peak_in = input.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
    let peak_out = output.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
    if peak_out > 0.0 && peak_in > 0.0 && peak_out > peak_in {
        let scale = peak_in / peak_out;
        for s in output.iter_mut() {
            *s *= scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fractalize_time_nearest() {
        let audio: Vec<f64> = (0..4096)
            .map(|i| (2.0 * std::f64::consts::PI * 440.0 * i as f64 / 44100.0).sin())
            .collect();
        let params = FractalParams::default();
        let out = fractalize_time(&audio, &params);
        assert_eq!(out.len(), audio.len());
        assert!(out.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_fractalize_time_linear() {
        let audio: Vec<f64> = (0..4096)
            .map(|i| (2.0 * std::f64::consts::PI * 440.0 * i as f64 / 44100.0).sin())
            .collect();
        let mut params = FractalParams::default();
        params.interp = 1;
        let out = fractalize_time(&audio, &params);
        assert_eq!(out.len(), audio.len());
        assert!(out.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_fractalize_spectral() {
        let audio: Vec<f64> = (0..4096)
            .map(|i| (2.0 * std::f64::consts::PI * 440.0 * i as f64 / 44100.0).sin())
            .collect();
        let params = FractalParams::default();
        let out = fractalize_spectral(&audio, &params);
        assert_eq!(out.len(), audio.len());
        assert!(out.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_render_fractal_core() {
        let audio: Vec<f64> = (0..4096)
            .map(|i| (2.0 * std::f64::consts::PI * 440.0 * i as f64 / 44100.0).sin())
            .collect();
        let params = FractalParams::default();
        let out = render_fractal_core(&audio, &params);
        assert_eq!(out.len(), audio.len());
        assert!(out.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_saturation() {
        let mut audio = vec![0.5, -0.5, 1.0, -1.0];
        apply_saturation(&mut audio, 1.0);
        // Fully saturated: should be tanh(x)
        assert!((audio[0] - 0.5_f64.tanh()).abs() < 1e-10);
        assert!((audio[1] - (-0.5_f64).tanh()).abs() < 1e-10);
    }

    #[test]
    fn test_iterations() {
        let audio: Vec<f64> = (0..4096)
            .map(|i| (2.0 * std::f64::consts::PI * 440.0 * i as f64 / 44100.0).sin())
            .collect();
        let mut params = FractalParams::default();
        params.iterations = 3;
        params.iter_decay = 0.8;
        params.saturation = 0.3;
        let out = render_fractal_core(&audio, &params);
        assert_eq!(out.len(), audio.len());
        assert!(out.iter().all(|x| x.is_finite()));
    }
}
