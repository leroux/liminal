//! I-series: Granular effects (I001-I007).

use std::collections::HashMap;
use serde_json::Value;
use crate::{AudioOutput, EffectEntry, pf, params};
use crate::primitives::*;

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Overlap-add all grains into `out`.
///
/// * `grain_starts`  : output sample index where each grain begins
/// * `grain_positions`: source sample index where each grain reads from
/// * `grain_lens`    : length of each grain in samples
/// * `grain_ratios`  : playback rate for each grain (1.0 = original pitch)
/// * `grain_amplitudes`: per-grain amplitude
/// * `window`        : Hann window (shared, scaled per grain via w_idx)
fn overlap_add_grains(
    samples: &[f32],
    out: &mut [f32],
    grain_starts: &[i64],
    grain_positions: &[i64],
    grain_lens: &[i64],
    grain_ratios: &[f64],
    grain_amplitudes: &[f32],
    window: &[f32],
) {
    let n_grains = grain_starts.len();
    let n_src = samples.len() as i64;
    let out_len = out.len() as i64;
    let w_len = window.len() as i64;

    for g in 0..n_grains {
        let g_start = grain_starts[g];
        let g_pos = grain_positions[g];
        let g_len = grain_lens[g];
        let g_ratio = grain_ratios[g];
        let g_amp = grain_amplitudes[g];

        for i in 0..g_len {
            let oi = g_start + i;
            if oi < 0 || oi >= out_len {
                continue;
            }
            let src_pos = g_pos as f64 + (i as f64) * g_ratio;
            let mut idx = src_pos.floor() as i64;
            let mut frac = (src_pos - idx as f64) as f32;
            if idx < 0 {
                idx = 0;
                frac = 0.0;
            }
            if idx >= n_src - 1 {
                idx = n_src - 2;
                frac = 0.0;
                if idx < 0 {
                    continue;
                }
            }
            // Window index: scale i into window length
            let mut w_idx = ((i as f64) / (g_len as f64) * ((w_len - 1) as f64)) as i64;
            if w_idx >= w_len {
                w_idx = w_len - 1;
            }
            let val = samples[idx as usize] * (1.0 - frac)
                + samples[(idx + 1) as usize] * frac;
            out[oi as usize] += val * window[w_idx as usize] * g_amp;
        }
    }
}

/// Like `overlap_add_grains` but with a per-grain reverse flag.
fn overlap_add_grains_reversible(
    samples: &[f32],
    out: &mut [f32],
    grain_starts: &[i64],
    grain_positions: &[i64],
    grain_lens: &[i64],
    grain_ratios: &[f64],
    grain_amplitudes: &[f32],
    grain_reversed: &[bool],
    window: &[f32],
) {
    let n_grains = grain_starts.len();
    let n_src = samples.len() as i64;
    let out_len = out.len() as i64;
    let w_len = window.len() as i64;

    for g in 0..n_grains {
        let g_start = grain_starts[g];
        let g_pos = grain_positions[g];
        let g_len = grain_lens[g];
        let g_ratio = grain_ratios[g];
        let g_amp = grain_amplitudes[g];
        let is_rev = grain_reversed[g];

        for i in 0..g_len {
            let oi = g_start + i;
            if oi < 0 || oi >= out_len {
                continue;
            }
            let src_pos = if is_rev {
                g_pos as f64 + ((g_len - 1 - i) as f64) * g_ratio
            } else {
                g_pos as f64 + (i as f64) * g_ratio
            };
            let mut idx = src_pos.floor() as i64;
            let mut frac = (src_pos - idx as f64) as f32;
            if idx < 0 {
                idx = 0;
                frac = 0.0;
            }
            if idx >= n_src - 1 {
                idx = n_src - 2;
                frac = 0.0;
                if idx < 0 {
                    continue;
                }
            }
            let mut w_idx = ((i as f64) / (g_len as f64) * ((w_len - 1) as f64)) as i64;
            if w_idx >= w_len {
                w_idx = w_len - 1;
            }
            let val = samples[idx as usize] * (1.0 - frac)
                + samples[(idx + 1) as usize] * frac;
            out[oi as usize] += val * window[w_idx as usize] * g_amp;
        }
    }
}

/// Schedule grain onset sample-indices using Poisson process.
/// `density` is grains per second. Returns sorted onset times.
fn schedule_grains_poisson(n_samples: usize, sr: u32, density: f32, rng: &mut Lcg) -> Vec<i64> {
    let avg_interval = sr as f64 / (density as f64).max(0.01);
    let mut times = Vec::new();
    let mut t: f64 = 0.0;
    while t < n_samples as f64 {
        times.push(t as i64);
        // Exponential distribution: -ln(1 - U) * mean, where U in [0, 1)
        let u = (rng.next_f32() as f64).max(1e-12);
        t += -u.ln() * avg_interval;
    }
    times
}

/// Schedule grains with density ramping from `start_density` to `end_density`.
fn schedule_grains_density_ramp(
    n_samples: usize,
    sr: u32,
    start_density: f32,
    end_density: f32,
    exponential: bool,
    rng: &mut Lcg,
) -> Vec<i64> {
    let mut times = Vec::new();
    let mut t: f64 = 0.0;
    let total = n_samples as f64;
    while t < total {
        times.push(t as i64);
        let progress = t / total;
        let local_density = if exponential {
            // Exponential interpolation in density space
            let base = (end_density as f64) / (start_density as f64).max(0.01);
            (start_density as f64) * base.powf(progress)
        } else {
            (start_density as f64) + ((end_density - start_density) as f64) * progress
        };
        let local_density = local_density.max(0.01);
        let avg_interval = (sr as f64) / local_density;
        let u = (rng.next_f32() as f64).max(1e-12);
        t += -u.ln() * avg_interval;
    }
    times
}

// ---------------------------------------------------------------------------
// I001 -- Granular Cloud
// ---------------------------------------------------------------------------

fn process_i001(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let grain_size_ms = pf(params, "grain_size_ms", 50.0);
    let density = pf(params, "density", 20.0);
    let position_spread = pf(params, "position_spread", 0.5);
    let pitch_spread_st = pf(params, "pitch_spread_semitones", 0.0);
    let amplitude_spread = pf(params, "amplitude_spread", 0.1);

    let n = samples.len();
    let grain_len = ((grain_size_ms * sr as f32 / 1000.0) as usize).max(1);
    let window = hann_window(grain_len);

    let mut rng = Lcg::new(42);
    let grain_starts = schedule_grains_poisson(n, sr, density, &mut rng);
    let n_grains = grain_starts.len();
    if n_grains == 0 {
        return AudioOutput::Mono(vec![0.0; n]);
    }

    // Random source positions: center at grain_start, spread by position_spread
    let half_spread = (position_spread * n as f32 * 0.5) as i64;
    let mut grain_positions = Vec::with_capacity(n_grains);
    for i in 0..n_grains {
        let offset = if half_spread > 0 {
            let r = rng.next_f32();
            (r * 2.0 * half_spread as f32) as i64 - half_spread
        } else {
            0
        };
        let pos = grain_starts[i] + offset;
        let pos = pos.max(0).min((n as i64) - (grain_len as i64));
        grain_positions.push(pos);
    }

    // Pitch ratios
    let mut grain_ratios = Vec::with_capacity(n_grains);
    for _ in 0..n_grains {
        if pitch_spread_st > 0.0 {
            let semitones = (rng.next_f32() * 2.0 - 1.0) * pitch_spread_st;
            grain_ratios.push(2.0f64.powf(semitones as f64 / 12.0));
        } else {
            grain_ratios.push(1.0);
        }
    }

    // Amplitudes
    let mut grain_amplitudes = Vec::with_capacity(n_grains);
    for _ in 0..n_grains {
        let amp = 1.0 - amplitude_spread + rng.next_f32() * amplitude_spread;
        grain_amplitudes.push(amp);
    }

    let grain_lens_arr: Vec<i64> = vec![grain_len as i64; n_grains];

    let mut out = vec![0.0f32; n];
    overlap_add_grains(
        samples,
        &mut out,
        &grain_starts,
        &grain_positions,
        &grain_lens_arr,
        &grain_ratios,
        &grain_amplitudes,
        &window,
    );
    AudioOutput::Mono(post_process(&out, sr))
}

fn variants_i001() -> Vec<HashMap<String, Value>> {
    vec![
        params!("grain_size_ms" => 15, "density" => 80, "position_spread" => 0.3, "pitch_spread_semitones" => 0, "amplitude_spread" => 0.05),
        params!("grain_size_ms" => 50, "density" => 20, "position_spread" => 0.5, "pitch_spread_semitones" => 0, "amplitude_spread" => 0.1),
        params!("grain_size_ms" => 100, "density" => 10, "position_spread" => 0.8, "pitch_spread_semitones" => 0, "amplitude_spread" => 0.2),
        params!("grain_size_ms" => 30, "density" => 40, "position_spread" => 0.2, "pitch_spread_semitones" => 5, "amplitude_spread" => 0.1),
        params!("grain_size_ms" => 200, "density" => 5, "position_spread" => 1.0, "pitch_spread_semitones" => 12, "amplitude_spread" => 0.4),
        params!("grain_size_ms" => 10, "density" => 100, "position_spread" => 0.05, "pitch_spread_semitones" => 0, "amplitude_spread" => 0.0),
    ]
}

// ---------------------------------------------------------------------------
// I002 -- Granular Freeze
// ---------------------------------------------------------------------------

fn process_i002(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let freeze_position = pf(params, "freeze_position", 0.5);
    let position_jitter_ms = pf(params, "position_jitter_ms", 10.0);
    let pitch_jitter = pf(params, "pitch_jitter", 0.5);
    let density = pf(params, "density", 30.0);
    let grain_size_ms = pf(params, "grain_size_ms", 40.0);

    let n = samples.len();
    let grain_len = ((grain_size_ms * sr as f32 / 1000.0) as usize).max(1);
    let window = hann_window(grain_len);

    let mut rng = Lcg::new(42);
    let grain_starts = schedule_grains_poisson(n, sr, density, &mut rng);
    let n_grains = grain_starts.len();
    if n_grains == 0 {
        return AudioOutput::Mono(vec![0.0; n]);
    }

    // Fixed source position with jitter
    let center_pos = (freeze_position * (n as f32 - grain_len as f32)) as i64;
    let jitter_samples = (position_jitter_ms * sr as f32 / 1000.0) as i64;

    let mut grain_positions = Vec::with_capacity(n_grains);
    for _ in 0..n_grains {
        let jit = if jitter_samples > 0 {
            let r = rng.next_f32();
            (r * 2.0 * jitter_samples as f32) as i64 - jitter_samples
        } else {
            0
        };
        let pos = center_pos + jit;
        let pos = pos.max(0).min((n as i64) - (grain_len as i64));
        grain_positions.push(pos);
    }

    // Pitch with jitter around 1.0
    let mut grain_ratios = Vec::with_capacity(n_grains);
    for _ in 0..n_grains {
        if pitch_jitter > 0.0 {
            let semitones = (rng.next_f32() * 2.0 - 1.0) * pitch_jitter;
            grain_ratios.push(2.0f64.powf(semitones as f64 / 12.0));
        } else {
            grain_ratios.push(1.0);
        }
    }

    let grain_amplitudes = vec![1.0f32; n_grains];
    let grain_lens_arr = vec![grain_len as i64; n_grains];

    let mut out = vec![0.0f32; n];
    overlap_add_grains(
        samples,
        &mut out,
        &grain_starts,
        &grain_positions,
        &grain_lens_arr,
        &grain_ratios,
        &grain_amplitudes,
        &window,
    );
    AudioOutput::Mono(post_process(&out, sr))
}

fn variants_i002() -> Vec<HashMap<String, Value>> {
    vec![
        params!("freeze_position" => 0.5, "position_jitter_ms" => 0, "pitch_jitter" => 0, "density" => 30, "grain_size_ms" => 40),
        params!("freeze_position" => 0.5, "position_jitter_ms" => 10, "pitch_jitter" => 0.5, "density" => 30, "grain_size_ms" => 40),
        params!("freeze_position" => 0.25, "position_jitter_ms" => 30, "pitch_jitter" => 1.0, "density" => 50, "grain_size_ms" => 20),
        params!("freeze_position" => 0.75, "position_jitter_ms" => 50, "pitch_jitter" => 2.0, "density" => 20, "grain_size_ms" => 80),
        params!("freeze_position" => 0.1, "position_jitter_ms" => 5, "pitch_jitter" => 0, "density" => 100, "grain_size_ms" => 30),
        params!("freeze_position" => 0.5, "position_jitter_ms" => 40, "pitch_jitter" => 1.5, "density" => 10, "grain_size_ms" => 100),
    ]
}

// ---------------------------------------------------------------------------
// I003 -- Granular Time Stretch
// ---------------------------------------------------------------------------

fn process_i003(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let stretch_factor = pf(params, "stretch_factor", 4.0);
    let grain_size_ms = pf(params, "grain_size_ms", 40.0);
    let density = pf(params, "density", 30.0);

    let n_src = samples.len();
    let n_out = (n_src as f64 * stretch_factor as f64) as usize;
    let grain_len = ((grain_size_ms * sr as f32 / 1000.0) as usize).max(1);
    let window = hann_window(grain_len);

    let mut rng = Lcg::new(42);
    let grain_starts = schedule_grains_poisson(n_out, sr, density, &mut rng);
    let n_grains = grain_starts.len();
    if n_grains == 0 {
        return AudioOutput::Mono(vec![0.0; n_out]);
    }

    // Source read position moves slower: map output time to source time
    let mut grain_positions = Vec::with_capacity(n_grains);
    for i in 0..n_grains {
        let src_pos = (grain_starts[i] as f64 / stretch_factor as f64) as i64;
        let src_pos = src_pos.max(0).min((n_src as i64) - (grain_len as i64));
        grain_positions.push(src_pos);
    }

    let grain_ratios = vec![1.0f64; n_grains];
    let grain_amplitudes = vec![1.0f32; n_grains];
    let grain_lens_arr = vec![grain_len as i64; n_grains];

    let mut out = vec![0.0f32; n_out];
    overlap_add_grains(
        samples,
        &mut out,
        &grain_starts,
        &grain_positions,
        &grain_lens_arr,
        &grain_ratios,
        &grain_amplitudes,
        &window,
    );
    AudioOutput::Mono(post_process(&out, sr))
}

fn variants_i003() -> Vec<HashMap<String, Value>> {
    vec![
        params!("stretch_factor" => 1.5, "grain_size_ms" => 30, "density" => 40),
        params!("stretch_factor" => 4, "grain_size_ms" => 40, "density" => 30),
        params!("stretch_factor" => 10, "grain_size_ms" => 60, "density" => 25),
        params!("stretch_factor" => 2, "grain_size_ms" => 20, "density" => 60),
        params!("stretch_factor" => 20, "grain_size_ms" => 80, "density" => 20),
        params!("stretch_factor" => 50, "grain_size_ms" => 100, "density" => 15),
    ]
}

// ---------------------------------------------------------------------------
// I004 -- Granular Reverse Scatter
// ---------------------------------------------------------------------------

fn process_i004(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let reverse_probability = pf(params, "reverse_probability", 0.5);
    let grain_size_ms = pf(params, "grain_size_ms", 40.0);
    let density = pf(params, "density", 25.0);

    let n = samples.len();
    let grain_len = ((grain_size_ms * sr as f32 / 1000.0) as usize).max(1);
    let window = hann_window(grain_len);

    let mut rng = Lcg::new(42);
    let grain_starts = schedule_grains_poisson(n, sr, density, &mut rng);
    let n_grains = grain_starts.len();
    if n_grains == 0 {
        return AudioOutput::Mono(vec![0.0; n]);
    }

    // Source positions follow output positions
    let mut grain_positions = Vec::with_capacity(n_grains);
    for i in 0..n_grains {
        let pos = grain_starts[i].max(0).min((n as i64) - (grain_len as i64));
        grain_positions.push(pos);
    }

    let grain_ratios = vec![1.0f64; n_grains];
    let grain_amplitudes = vec![1.0f32; n_grains];
    let grain_lens_arr = vec![grain_len as i64; n_grains];

    // Decide which grains are reversed
    let mut grain_reversed = Vec::with_capacity(n_grains);
    for _ in 0..n_grains {
        grain_reversed.push(rng.next_f32() < reverse_probability);
    }

    let mut out = vec![0.0f32; n];
    overlap_add_grains_reversible(
        samples,
        &mut out,
        &grain_starts,
        &grain_positions,
        &grain_lens_arr,
        &grain_ratios,
        &grain_amplitudes,
        &grain_reversed,
        &window,
    );
    AudioOutput::Mono(post_process(&out, sr))
}

fn variants_i004() -> Vec<HashMap<String, Value>> {
    vec![
        params!("reverse_probability" => 0.1, "grain_size_ms" => 40, "density" => 25),
        params!("reverse_probability" => 0.5, "grain_size_ms" => 40, "density" => 25),
        params!("reverse_probability" => 1.0, "grain_size_ms" => 40, "density" => 25),
        params!("reverse_probability" => 0.5, "grain_size_ms" => 80, "density" => 15),
        params!("reverse_probability" => 0.5, "grain_size_ms" => 20, "density" => 60),
        params!("reverse_probability" => 0.3, "grain_size_ms" => 100, "density" => 10),
    ]
}

// ---------------------------------------------------------------------------
// I005 -- Granular Pitch Cloud
// ---------------------------------------------------------------------------

fn process_i005(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let center_semitones = pf(params, "center_semitones", 0.0);
    let spread_semitones = pf(params, "spread_semitones", 7.0);
    let grain_size_ms = pf(params, "grain_size_ms", 50.0);
    let density = pf(params, "density", 20.0);

    let n = samples.len();
    let grain_len = ((grain_size_ms * sr as f32 / 1000.0) as usize).max(1);
    let window = hann_window(grain_len);

    let mut rng = Lcg::new(42);
    let grain_starts = schedule_grains_poisson(n, sr, density, &mut rng);
    let n_grains = grain_starts.len();
    if n_grains == 0 {
        return AudioOutput::Mono(vec![0.0; n]);
    }

    // All grains read from same position as their output position
    let mut grain_positions = Vec::with_capacity(n_grains);
    for i in 0..n_grains {
        let pos = grain_starts[i].max(0).min((n as i64) - (grain_len as i64));
        grain_positions.push(pos);
    }

    // Random pitches around center
    let mut grain_ratios = Vec::with_capacity(n_grains);
    for _ in 0..n_grains {
        let semitones = center_semitones + (rng.next_f32() * 2.0 - 1.0) * spread_semitones;
        grain_ratios.push(2.0f64.powf(semitones as f64 / 12.0));
    }

    let grain_amplitudes = vec![1.0f32; n_grains];
    let grain_lens_arr = vec![grain_len as i64; n_grains];

    let mut out = vec![0.0f32; n];
    overlap_add_grains(
        samples,
        &mut out,
        &grain_starts,
        &grain_positions,
        &grain_lens_arr,
        &grain_ratios,
        &grain_amplitudes,
        &window,
    );
    AudioOutput::Mono(post_process(&out, sr))
}

fn variants_i005() -> Vec<HashMap<String, Value>> {
    vec![
        params!("center_semitones" => 0, "spread_semitones" => 2, "grain_size_ms" => 50, "density" => 20),
        params!("center_semitones" => 0, "spread_semitones" => 7, "grain_size_ms" => 50, "density" => 20),
        params!("center_semitones" => 0, "spread_semitones" => 12, "grain_size_ms" => 40, "density" => 30),
        params!("center_semitones" => 7, "spread_semitones" => 3, "grain_size_ms" => 60, "density" => 15),
        params!("center_semitones" => -12, "spread_semitones" => 5, "grain_size_ms" => 80, "density" => 10),
        params!("center_semitones" => 0, "spread_semitones" => 24, "grain_size_ms" => 30, "density" => 40),
        params!("center_semitones" => 12, "spread_semitones" => 1, "grain_size_ms" => 20, "density" => 60),
    ]
}

// ---------------------------------------------------------------------------
// I006 -- Granular Density Ramp
// ---------------------------------------------------------------------------

fn process_i006(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let start_density = pf(params, "start_density", 2.0);
    let end_density = pf(params, "end_density", 100.0);
    let grain_size_ms = pf(params, "grain_size_ms", 30.0);
    let ramp_curve = params
        .get("ramp_curve")
        .and_then(|v| v.as_str())
        .unwrap_or("exponential");
    let exponential = ramp_curve == "exponential";

    let n = samples.len();
    let grain_len = ((grain_size_ms * sr as f32 / 1000.0) as usize).max(1);
    let window = hann_window(grain_len);

    let mut rng = Lcg::new(42);
    let grain_starts =
        schedule_grains_density_ramp(n, sr, start_density, end_density, exponential, &mut rng);
    let n_grains = grain_starts.len();
    if n_grains == 0 {
        return AudioOutput::Mono(vec![0.0; n]);
    }

    let mut grain_positions = Vec::with_capacity(n_grains);
    for i in 0..n_grains {
        let pos = grain_starts[i].max(0).min((n as i64) - (grain_len as i64));
        grain_positions.push(pos);
    }

    let grain_ratios = vec![1.0f64; n_grains];
    let grain_amplitudes = vec![1.0f32; n_grains];
    let grain_lens_arr = vec![grain_len as i64; n_grains];

    let mut out = vec![0.0f32; n];
    overlap_add_grains(
        samples,
        &mut out,
        &grain_starts,
        &grain_positions,
        &grain_lens_arr,
        &grain_ratios,
        &grain_amplitudes,
        &window,
    );
    AudioOutput::Mono(post_process(&out, sr))
}

fn variants_i006() -> Vec<HashMap<String, Value>> {
    vec![
        params!("start_density" => 2, "end_density" => 50, "grain_size_ms" => 30, "ramp_curve" => "linear"),
        params!("start_density" => 2, "end_density" => 100, "grain_size_ms" => 30, "ramp_curve" => "exponential"),
        params!("start_density" => 1, "end_density" => 200, "grain_size_ms" => 20, "ramp_curve" => "exponential"),
        params!("start_density" => 10, "end_density" => 80, "grain_size_ms" => 50, "ramp_curve" => "linear"),
        params!("start_density" => 1, "end_density" => 150, "grain_size_ms" => 80, "ramp_curve" => "exponential"),
        params!("start_density" => 5, "end_density" => 50, "grain_size_ms" => 40, "ramp_curve" => "linear"),
    ]
}

// ---------------------------------------------------------------------------
// I007 -- Microsound Particles
// ---------------------------------------------------------------------------

fn process_i007(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let grain_size_ms = pf(params, "grain_size_ms", 3.0);
    let density = pf(params, "density", 300.0);
    let pitch_range = pf(params, "pitch_range", 2.0);

    let n = samples.len();
    let grain_len = ((grain_size_ms * sr as f32 / 1000.0) as usize).max(1);
    let window = hann_window(grain_len);

    let mut rng = Lcg::new(42);
    let grain_starts = schedule_grains_poisson(n, sr, density, &mut rng);
    let n_grains = grain_starts.len();
    if n_grains == 0 {
        return AudioOutput::Mono(vec![0.0; n]);
    }

    let mut grain_positions = Vec::with_capacity(n_grains);
    for i in 0..n_grains {
        let pos = grain_starts[i].max(0).min((n as i64) - (grain_len as i64));
        grain_positions.push(pos);
    }

    // Random pitch within range
    let mut grain_ratios = Vec::with_capacity(n_grains);
    for _ in 0..n_grains {
        if pitch_range > 0.0 {
            let semitones = (rng.next_f32() * 2.0 - 1.0) * pitch_range;
            grain_ratios.push(2.0f64.powf(semitones as f64 / 12.0));
        } else {
            grain_ratios.push(1.0);
        }
    }

    let grain_amplitudes = vec![1.0f32; n_grains];
    let grain_lens_arr = vec![grain_len as i64; n_grains];

    let mut out = vec![0.0f32; n];
    overlap_add_grains(
        samples,
        &mut out,
        &grain_starts,
        &grain_positions,
        &grain_lens_arr,
        &grain_ratios,
        &grain_amplitudes,
        &window,
    );
    AudioOutput::Mono(post_process(&out, sr))
}

fn variants_i007() -> Vec<HashMap<String, Value>> {
    vec![
        params!("grain_size_ms" => 3, "density" => 300, "pitch_range" => 0),
        params!("grain_size_ms" => 3, "density" => 300, "pitch_range" => 2),
        params!("grain_size_ms" => 1, "density" => 1000, "pitch_range" => 1),
        params!("grain_size_ms" => 10, "density" => 100, "pitch_range" => 5),
        params!("grain_size_ms" => 2, "density" => 500, "pitch_range" => 12),
        params!("grain_size_ms" => 5, "density" => 200, "pitch_range" => 0),
        params!("grain_size_ms" => 1, "density" => 800, "pitch_range" => 8),
    ]
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

pub fn register() -> Vec<EffectEntry> {
    vec![
        EffectEntry {
            id: "I001",
            process: process_i001,
            variants: variants_i001,
            category: "granular",
        },
        EffectEntry {
            id: "I002",
            process: process_i002,
            variants: variants_i002,
            category: "granular",
        },
        EffectEntry {
            id: "I003",
            process: process_i003,
            variants: variants_i003,
            category: "granular",
        },
        EffectEntry {
            id: "I004",
            process: process_i004,
            variants: variants_i004,
            category: "granular",
        },
        EffectEntry {
            id: "I005",
            process: process_i005,
            variants: variants_i005,
            category: "granular",
        },
        EffectEntry {
            id: "I006",
            process: process_i006,
            variants: variants_i006,
            category: "granular",
        },
        EffectEntry {
            id: "I007",
            process: process_i007,
            variants: variants_i007,
            category: "granular",
        },
    ]
}
