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
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Cached FFT plan + Hann window for spectral processing
// ---------------------------------------------------------------------------

struct SpectralPlan {
    window: Vec<f64>,
    fft: Arc<dyn realfft::RealToComplex<f64>>,
    ifft: Arc<dyn realfft::ComplexToReal<f64>>,
}

impl SpectralPlan {
    fn new(window_size: usize) -> Self {
        let mut planner = RealFftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(window_size);
        let ifft = planner.plan_fft_inverse(window_size);
        let window: Vec<f64> = (0..window_size)
            .map(|i| {
                0.5 * (1.0
                    - (2.0 * std::f64::consts::PI * i as f64 / window_size as f64).cos())
            })
            .collect();
        Self { window, fft, ifft }
    }
}

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

        // Tile with offset (modulo-free for auto-vectorization)
        let offset_samples = (scale_offset * compressed_len as f64) as usize;
        tile_scaled_add(&compressed, offset_samples, gain, &mut out);
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

        // Tile with offset (modulo-free for auto-vectorization)
        let offset_samples = (scale_offset * compressed_len as f64) as usize;
        tile_scaled_add(&compressed, offset_samples, gain, &mut out);
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

        // Tile with offset and apply gain (modulo-free for auto-vectorization)
        let offset_samples = (scale_offset * compressed_len as f64) as usize;
        let mut layer = vec![0.0_f64; n];
        tile_scaled(&compressed, offset_samples, gain, &mut layer);

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
    let window_size = (params.window_size as usize).clamp(256, 8192);
    let plan = SpectralPlan::new(window_size);
    fractalize_spectral_with_plan(samples, params, &plan)
}

/// Apply fractalization to STFT magnitude frames, reusing a cached FFT plan.
fn fractalize_spectral_with_plan(
    samples: &[f64],
    params: &FractalParams,
    plan: &SpectralPlan,
) -> Vec<f64> {
    let num_scales = params.num_scales.clamp(2, 8);
    let scale_ratio = params.scale_ratio.clamp(0.1, 0.9);
    let amplitude_decay = params.amplitude_decay.clamp(0.1, 1.0);
    let window_size = plan.window.len();
    let hop_size = window_size / 4;
    let window = &plan.window;

    let out_len = samples.len();

    // Use samples directly when possible, only pad when too short
    let needs_pad = samples.len() < window_size;
    let mut padded_storage;
    let x: &[f64] = if needs_pad {
        padded_storage = samples.to_vec();
        padded_storage.resize(window_size, 0.0);
        &padded_storage
    } else {
        samples
    };
    let n = x.len();

    let num_frames = if n >= window_size {
        1 + (n - window_size) / hop_size
    } else {
        0
    };
    if num_frames == 0 {
        return samples.to_vec();
    }

    let num_bins = window_size / 2 + 1;

    // Pre-allocate per-frame buffers (reused across all frames)
    let mut frame = vec![0.0_f64; window_size];
    let mut spectrum = vec![Complex64::new(0.0, 0.0); num_bins];
    let mut mag = vec![0.0_f64; num_bins];
    let mut orig_mag = vec![0.0_f64; num_bins];
    let mut phase = vec![0.0_f64; num_bins];
    let mut time_buf = vec![0.0_f64; window_size];

    // Pre-allocate compressed buffers for each scale
    let mut compressed_bufs: Vec<Vec<f64>> = (1..num_scales)
        .map(|s| {
            let cl = (num_bins as f64 * scale_ratio.powi(s)).max(1.0) as usize;
            vec![0.0_f64; cl]
        })
        .collect();

    // Size output exactly to out_len (avoid extra copy at end)
    let ola_len = window_size + (num_frames - 1) * hop_size;
    let mut output = vec![0.0_f64; ola_len.max(out_len)];

    for fi in 0..num_frames {
        let start = fi * hop_size;

        // Window the frame
        for i in 0..window_size {
            frame[i] = x[start + i] * window[i];
        }

        // Forward FFT
        plan.fft.process(&mut frame, &mut spectrum).unwrap();

        // Extract magnitude and phase; copy mag to orig_mag in one pass
        for i in 0..num_bins {
            let m = spectrum[i].norm();
            mag[i] = m;
            orig_mag[i] = m;
            phase[i] = spectrum[i].arg();
        }

        // Fractalize magnitude
        for (si, s) in (1..num_scales).enumerate() {
            let compressed = &mut compressed_bufs[si];
            let compressed_len = compressed.len();
            let gain = amplitude_decay.powi(s);

            // Downsample magnitude via linear interpolation
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

            // Tile to fill (modulo-free for auto-vectorization)
            tile_scaled_add(compressed, 0, gain, &mut mag[..num_bins]);
        }

        // Reconstruct complex spectrum in-place
        for i in 0..num_bins {
            spectrum[i] = Complex64::from_polar(mag[i], phase[i]);
        }
        spectrum[0] = Complex64::new(spectrum[0].re, 0.0);
        if num_bins > 1 {
            let last = num_bins - 1;
            spectrum[last] = Complex64::new(spectrum[last].re, 0.0);
        }

        // Inverse FFT
        plan.ifft.process(&mut spectrum, &mut time_buf).unwrap();

        // Apply window and normalize (realfft scales by N)
        let inv_n = 1.0 / window_size as f64;
        for i in 0..window_size {
            let pos = start + i;
            if pos < output.len() {
                output[pos] += time_buf[i] * inv_n * window[i];
            }
        }
    }

    // Truncate in-place (no extra allocation)
    output.truncate(out_len);

    // Normalize
    normalize_to_input_peak_inplace(samples, &mut output);

    output
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
    let n = samples.len();

    let mut current = samples.to_vec();
    // Pre-allocate a scratch buffer for summing layers (reused across iterations)
    let mut sum_buf = vec![0.0_f64; n];

    // Cache FFT plan + window if spectral processing is needed (once for all iterations)
    let plan = if spectral_blend > 0.0 {
        let ws = (params.window_size as usize).clamp(256, 8192);
        Some(SpectralPlan::new(ws))
    } else {
        None
    };

    for it in 0..iterations {
        // Time-domain fractalization via per-layer approach
        let has_time = spectral_blend < 1.0;
        let has_spec = spectral_blend > 0.0;

        if has_time {
            let mut layers = fractalize_time_layers(&current, params);

            // Apply per-layer delay (feature 5)
            if layer_delay > 0.0 {
                let max_delay = (n as f64 * 0.1) as usize;
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

            // Sum all layers into sum_buf
            sum_buf[..n].fill(0.0);
            for layer in &layers {
                for (i, &v) in layer.iter().enumerate() {
                    sum_buf[i] += v;
                }
            }

            normalize_to_input_peak_inplace(&current, &mut sum_buf[..n]);

            if !has_spec {
                // Pure time-domain: swap sum_buf into current
                current.copy_from_slice(&sum_buf[..n]);
            } else {
                // Need to blend with spectral — compute spectral, then blend in-place
                let spec_out =
                    fractalize_spectral_with_plan(&current, params, plan.as_ref().unwrap());
                for i in 0..n {
                    current[i] =
                        (1.0 - spectral_blend) * sum_buf[i] + spectral_blend * spec_out[i];
                }
            }
        } else {
            // Pure spectral
            let spec_out =
                fractalize_spectral_with_plan(&current, params, plan.as_ref().unwrap());
            current.copy_from_slice(&spec_out[..n]);
        }

        // Apply saturation between iterations
        if saturation > 0.0 {
            apply_saturation(&mut current, saturation);
        }

        // Apply iteration decay for subsequent passes
        if it < iterations - 1 {
            for s in current.iter_mut() {
                *s *= iter_decay;
            }
        }
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

    // Cache FFT plan + window if spectral processing is needed
    let plan = if spectral_blend > 0.0 {
        let ws = (params.window_size as usize).clamp(256, 8192);
        Some(SpectralPlan::new(ws))
    } else {
        None
    };

    // Precompute per-layer pan gains (constant across iterations)
    let mut pan_gains_l = [0.0_f64; 8];
    let mut pan_gains_r = [0.0_f64; 8];
    for s in 1..num_scales as usize {
        let sign = if s % 2 == 0 { 1.0 } else { -1.0 };
        let pan = layer_spread * sign * s as f64 / (num_scales - 1).max(1) as f64;
        let theta = (pan + 1.0) * std::f64::consts::FRAC_PI_4;
        pan_gains_l[s] = theta.cos();
        pan_gains_r[s] = theta.sin();
    }

    for it in 0..iterations {
        // Restructured: compute only what's needed, avoid unnecessary clones
        let (mut res_l, mut res_r) = if spectral_blend <= 0.0 {
            // Pure time-domain
            let mut layers_l = fractalize_time_layers(&cur_l, params);
            let mut layers_r = fractalize_time_layers(&cur_r, params);

            apply_stereo_layer_effects(
                &mut layers_l,
                &mut layers_r,
                layer_delay_param,
                layer_tilt,
                num_scales,
                n,
            );

            let (out_l, out_r) = combine_stereo_layers(
                &layers_l,
                &layers_r,
                &pan_gains_l,
                &pan_gains_r,
                n,
            );

            (
                normalize_to_input_peak(&cur_l, out_l),
                normalize_to_input_peak(&cur_r, out_r),
            )
        } else if spectral_blend >= 1.0 {
            // Pure spectral
            let p = plan.as_ref().unwrap();
            (
                fractalize_spectral_with_plan(&cur_l, params, p),
                fractalize_spectral_with_plan(&cur_r, params, p),
            )
        } else {
            // Blend time + spectral
            let mut layers_l = fractalize_time_layers(&cur_l, params);
            let mut layers_r = fractalize_time_layers(&cur_r, params);

            apply_stereo_layer_effects(
                &mut layers_l,
                &mut layers_r,
                layer_delay_param,
                layer_tilt,
                num_scales,
                n,
            );

            let (out_l, out_r) = combine_stereo_layers(
                &layers_l,
                &layers_r,
                &pan_gains_l,
                &pan_gains_r,
                n,
            );

            let mut tl = normalize_to_input_peak(&cur_l, out_l);
            let mut tr = normalize_to_input_peak(&cur_r, out_r);

            let p = plan.as_ref().unwrap();
            let spec_l = fractalize_spectral_with_plan(&cur_l, params, p);
            let spec_r = fractalize_spectral_with_plan(&cur_r, params, p);

            // Blend in-place (no extra allocation)
            for i in 0..n {
                tl[i] = (1.0 - spectral_blend) * tl[i] + spectral_blend * spec_l[i];
                tr[i] = (1.0 - spectral_blend) * tr[i] + spectral_blend * spec_r[i];
            }
            (tl, tr)
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

/// Tile `src` (starting at `offset`) into `dst`, multiplying by `gain` and assigning.
/// Modulo-free: uses sequential segment copies for auto-vectorization.
fn tile_scaled(src: &[f64], offset: usize, gain: f64, dst: &mut [f64]) {
    let cl = src.len();
    if cl == 0 {
        return;
    }
    let n = dst.len();
    let start_in_src = offset % cl;
    let mut pos = 0;

    // First partial tile (from start_in_src to end of src)
    let first_len = (cl - start_in_src).min(n);
    for i in 0..first_len {
        dst[i] = gain * src[start_in_src + i];
    }
    pos += first_len;

    // Full tiles
    while pos + cl <= n {
        for i in 0..cl {
            dst[pos + i] = gain * src[i];
        }
        pos += cl;
    }

    // Final partial tile
    let remaining = n - pos;
    for i in 0..remaining {
        dst[pos + i] = gain * src[i];
    }
}

/// Tile `src` (starting at `offset`) into `dst`, multiplying by `gain` and adding.
/// Modulo-free: uses sequential segment copies for auto-vectorization.
fn tile_scaled_add(src: &[f64], offset: usize, gain: f64, dst: &mut [f64]) {
    let cl = src.len();
    if cl == 0 {
        return;
    }
    let n = dst.len();
    let start_in_src = offset % cl;
    let mut pos = 0;

    // First partial tile
    let first_len = (cl - start_in_src).min(n);
    for i in 0..first_len {
        dst[i] += gain * src[start_in_src + i];
    }
    pos += first_len;

    // Full tiles
    while pos + cl <= n {
        for i in 0..cl {
            dst[pos + i] += gain * src[i];
        }
        pos += cl;
    }

    // Final partial tile
    let remaining = n - pos;
    for i in 0..remaining {
        dst[pos + i] += gain * src[i];
    }
}

/// Apply delay and tilt effects to stereo layer pairs.
fn apply_stereo_layer_effects(
    layers_l: &mut [Vec<f64>],
    layers_r: &mut [Vec<f64>],
    layer_delay_param: f64,
    layer_tilt: f64,
    num_scales: i32,
    n: usize,
) {
    if layer_delay_param > 0.0 {
        let max_delay = (n as f64 * 0.1) as usize;
        for s in 1..layers_l.len() {
            let ds = (layer_delay_param * max_delay as f64 * s as f64
                / (num_scales - 1).max(1) as f64) as usize;
            apply_layer_delay(&mut layers_l[s], ds);
            apply_layer_delay(&mut layers_r[s], ds);
        }
    }
    if layer_tilt.abs() > 0.001 {
        for s in 0..layers_l.len() {
            apply_layer_tilt(&mut layers_l[s], layer_tilt, s as i32, num_scales);
            apply_layer_tilt(&mut layers_r[s], layer_tilt, s as i32, num_scales);
        }
    }
}

/// Combine stereo layers with per-layer constant-power panning.
fn combine_stereo_layers(
    layers_l: &[Vec<f64>],
    layers_r: &[Vec<f64>],
    pan_gains_l: &[f64; 8],
    pan_gains_r: &[f64; 8],
    n: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut out_l = vec![0.0_f64; n];
    let mut out_r = vec![0.0_f64; n];

    for (s, (ll, lr)) in layers_l.iter().zip(layers_r.iter()).enumerate() {
        if s == 0 {
            for i in 0..n {
                out_l[i] += ll[i];
                out_r[i] += lr[i];
            }
        } else {
            let gl = pan_gains_l[s];
            let gr = pan_gains_r[s];
            for i in 0..n {
                out_l[i] += ll[i] * gl + lr[i] * gl;
                out_r[i] += ll[i] * gr + lr[i] * gr;
            }
        }
    }

    (out_l, out_r)
}

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
