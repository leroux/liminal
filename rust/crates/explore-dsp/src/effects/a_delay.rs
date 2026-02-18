//! A-series effects: Delay-based algorithms (A001-A013).

use std::collections::HashMap;
use serde_json::Value;
use crate::{AudioOutput, EffectEntry, pf, pi, params};
use crate::primitives::*;

// ---------------------------------------------------------------------------
// A001 -- Simple Feedback Delay
// ---------------------------------------------------------------------------

fn process_a001(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let delay_ms = pf(params, "delay_ms", 300.0);
    let feedback = pf(params, "feedback", 0.5);
    let delay_samples = (delay_ms * sr as f32 / 1000.0).round() as usize;
    let delay_samples = delay_samples.max(1);

    let n = samples.len();
    let buf_len = (delay_samples + 1).max(1);
    let mut buf = vec![0.0f32; buf_len];
    let mut out = vec![0.0f32; n];
    let mut write_pos: usize = 0;

    for i in 0..n {
        let read_pos = (write_pos + buf_len - delay_samples) % buf_len;
        let y = samples[i] + feedback * buf[read_pos];
        buf[write_pos] = y;
        out[i] = y;
        write_pos = (write_pos + 1) % buf_len;
    }

    AudioOutput::Mono(out)
}

fn variants_a001() -> Vec<HashMap<String, Value>> {
    vec![
        params!{"delay_ms" => 100, "feedback" => 0.3},
        params!{"delay_ms" => 250, "feedback" => 0.5},
        params!{"delay_ms" => 500, "feedback" => 0.7},
        params!{"delay_ms" => 75, "feedback" => 0.85},
        params!{"delay_ms" => 1000, "feedback" => 0.4},
        params!{"delay_ms" => 150, "feedback" => 0.0},
    ]
}

// ---------------------------------------------------------------------------
// A002 -- Multi-Tap Delay
// ---------------------------------------------------------------------------

fn process_a002(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let num_taps = pi(params, "num_taps", 4) as usize;
    let base_ms = pf(params, "base_ms", 100.0);
    let decay = pf(params, "decay", 0.7);

    let phi: f32 = 1.6180339887;
    let mut tap_delays = vec![0usize; num_taps];
    let mut tap_gains = vec![0.0f32; num_taps];
    for k in 0..num_taps {
        let delay_ms = base_ms * phi.powi(k as i32);
        tap_delays[k] = (delay_ms * sr as f32 / 1000.0) as usize;
        tap_delays[k] = tap_delays[k].max(1);
        tap_gains[k] = decay.powi(k as i32);
    }

    let max_delay = *tap_delays.iter().max().unwrap_or(&1);
    let buf_len = (max_delay + 1).max(1);
    let mut buf = vec![0.0f32; buf_len];
    let mut out = vec![0.0f32; samples.len()];
    let mut write_pos: usize = 0;

    for i in 0..samples.len() {
        let mut y = samples[i];
        for t in 0..num_taps {
            let read_pos = (write_pos + buf_len - tap_delays[t]) % buf_len;
            y += tap_gains[t] * buf[read_pos];
        }
        buf[write_pos] = samples[i];
        out[i] = y;
        write_pos = (write_pos + 1) % buf_len;
    }

    AudioOutput::Mono(out)
}

fn variants_a002() -> Vec<HashMap<String, Value>> {
    vec![
        params!{"num_taps" => 3, "base_ms" => 80, "decay" => 0.7},
        params!{"num_taps" => 5, "base_ms" => 50, "decay" => 0.8},
        params!{"num_taps" => 8, "base_ms" => 120, "decay" => 0.6},
        params!{"num_taps" => 2, "base_ms" => 400, "decay" => 0.9},
        params!{"num_taps" => 6, "base_ms" => 200, "decay" => 0.5},
    ]
}

// ---------------------------------------------------------------------------
// A003 -- Ping-Pong Stereo Delay
// ---------------------------------------------------------------------------

fn process_a003(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let delay_ms = pf(params, "delay_ms", 300.0);
    let feedback = pf(params, "feedback", 0.5);
    let delay_samples = ((delay_ms * sr as f32 / 1000.0).round() as usize).max(1);

    let n = samples.len();
    let buf_len = (delay_samples + 1).max(1);
    let mut buf_l = vec![0.0f32; buf_len];
    let mut buf_r = vec![0.0f32; buf_len];
    let mut out = vec![[0.0f32; 2]; n];
    let mut write_pos: usize = 0;

    for i in 0..n {
        let read_pos = (write_pos + buf_len - delay_samples) % buf_len;
        let l_val = samples[i] + feedback * buf_r[read_pos];
        let r_val = feedback * buf_l[read_pos];
        buf_l[write_pos] = l_val;
        buf_r[write_pos] = r_val;
        out[i][0] = l_val;
        out[i][1] = r_val;
        write_pos = (write_pos + 1) % buf_len;
    }

    AudioOutput::Stereo(out)
}

fn variants_a003() -> Vec<HashMap<String, Value>> {
    vec![
        params!{"delay_ms" => 150, "feedback" => 0.4},
        params!{"delay_ms" => 300, "feedback" => 0.6},
        params!{"delay_ms" => 500, "feedback" => 0.75},
        params!{"delay_ms" => 100, "feedback" => 0.85},
        params!{"delay_ms" => 800, "feedback" => 0.3},
    ]
}

// ---------------------------------------------------------------------------
// A004 -- Reverse Delay
// ---------------------------------------------------------------------------

fn process_a004(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let delay_ms = pf(params, "delay_ms", 250.0);
    let feedback = pf(params, "feedback", 0.5);
    let chunk_size = ((delay_ms * sr as f32 / 1000.0).round() as usize).max(1);

    let n = samples.len();
    let mut out = vec![0.0f32; n];
    let mut buf = vec![0.0f32; chunk_size];
    let mut rev = vec![0.0f32; chunk_size];

    let num_passes = 4;
    let mut gain = 1.0f32;

    for _p in 0..num_passes {
        let num_chunks = n / chunk_size;
        for c in 0..num_chunks {
            let start = c * chunk_size;
            // Fill buf from samples (like fb_accum in Python, first pass uses input)
            for j in 0..chunk_size {
                buf[j] = samples[start + j];
            }
            // Reverse
            for j in 0..chunk_size {
                rev[j] = buf[chunk_size - 1 - j];
            }
            // Add reversed chunk to output, delayed by one chunk
            let out_start = start + chunk_size;
            if out_start + chunk_size <= n {
                for j in 0..chunk_size {
                    out[out_start + j] += gain * rev[j];
                }
            }
        }
        gain *= feedback;
    }

    // Mix in dry
    for i in 0..n {
        out[i] += samples[i];
    }

    AudioOutput::Mono(out)
}

fn variants_a004() -> Vec<HashMap<String, Value>> {
    vec![
        params!{"delay_ms" => 100, "feedback" => 0.4},
        params!{"delay_ms" => 200, "feedback" => 0.6},
        params!{"delay_ms" => 500, "feedback" => 0.3},
        params!{"delay_ms" => 150, "feedback" => 0.8},
        params!{"delay_ms" => 350, "feedback" => 0.5},
    ]
}

// ---------------------------------------------------------------------------
// A005 -- Tape Delay Emulation
// ---------------------------------------------------------------------------

fn process_a005(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let delay_ms = pf(params, "delay_ms", 300.0);
    let feedback = pf(params, "feedback", 0.5);
    let wow_rate_hz = pf(params, "wow_rate_hz", 1.5);
    let wow_depth = pf(params, "wow_depth_samples", 3.0);
    let filter_cutoff = pf(params, "filter_cutoff", 3500.0);

    let delay_samples = ((delay_ms * sr as f32 / 1000.0).round() as usize).max(1);

    // Compute one-pole coefficient from cutoff
    let dt = 1.0 / sr as f32;
    let rc = 1.0 / (2.0 * std::f32::consts::PI * filter_cutoff);
    let filter_coeff = (-dt / rc).exp();

    let n = samples.len();
    let buf_len = delay_samples + wow_depth as usize + 4; // extra headroom for modulation
    let mut buf = vec![0.0f32; buf_len];
    let mut out = vec![0.0f32; n];
    let mut write_pos: usize = 0;
    let mut lp_state = 0.0f32;
    let two_pi = 2.0 * std::f32::consts::PI;

    for i in 0..n {
        // Modulated delay with wow
        let phase = two_pi * wow_rate_hz * i as f32 / sr as f32;
        let modulation = wow_depth * phase.sin();
        let frac_delay = delay_samples as f32 + modulation;
        let int_delay = frac_delay as usize;
        let frac = frac_delay - int_delay as f32;

        // Linear interpolation for fractional read
        let read_pos_0 = (write_pos + buf_len - int_delay) % buf_len;
        let read_pos_1 = (write_pos + buf_len - int_delay - 1) % buf_len;
        let delayed = (1.0 - frac) * buf[read_pos_0] + frac * buf[read_pos_1];

        // One-pole lowpass in feedback path
        lp_state = filter_coeff * lp_state + (1.0 - filter_coeff) * delayed;

        let y = samples[i] + feedback * lp_state;
        buf[write_pos] = y;
        out[i] = y;
        write_pos = (write_pos + 1) % buf_len;
    }

    AudioOutput::Mono(out)
}

fn variants_a005() -> Vec<HashMap<String, Value>> {
    vec![
        params!{"delay_ms" => 200, "feedback" => 0.5, "wow_rate_hz" => 0.5, "wow_depth_samples" => 2, "filter_cutoff" => 4000},
        params!{"delay_ms" => 400, "feedback" => 0.6, "wow_rate_hz" => 1.0, "wow_depth_samples" => 5, "filter_cutoff" => 3000},
        params!{"delay_ms" => 150, "feedback" => 0.75, "wow_rate_hz" => 2.5, "wow_depth_samples" => 8, "filter_cutoff" => 2000},
        params!{"delay_ms" => 600, "feedback" => 0.3, "wow_rate_hz" => 0.3, "wow_depth_samples" => 1, "filter_cutoff" => 5000},
        params!{"delay_ms" => 100, "feedback" => 0.8, "wow_rate_hz" => 3.0, "wow_depth_samples" => 4, "filter_cutoff" => 2500},
        params!{"delay_ms" => 500, "feedback" => 0.45, "wow_rate_hz" => 1.8, "wow_depth_samples" => 6, "filter_cutoff" => 3500},
    ]
}

// ---------------------------------------------------------------------------
// A006 -- Granular Delay
// ---------------------------------------------------------------------------

fn process_a006(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let delay_ms = pf(params, "delay_ms", 300.0);
    let grain_size_ms = pf(params, "grain_size_ms", 50.0);
    let scatter_ms = pf(params, "scatter_ms", 50.0);
    let density = pf(params, "density", 20.0);
    let feedback = pf(params, "feedback", 0.3);

    let delay_samples = ((delay_ms * sr as f32 / 1000.0).round() as usize).max(1);
    let grain_size = ((grain_size_ms * sr as f32 / 1000.0).round() as usize).max(2);
    let scatter_samples = (scatter_ms * sr as f32 / 1000.0).round() as usize;

    let n = samples.len();
    let buf_len = (delay_samples + scatter_samples + grain_size + 1).max(grain_size + 1);
    let mut buf = vec![0.0f32; buf_len];
    let mut out = vec![0.0f32; n];
    let mut write_pos: usize = 0;

    let window = hann_window(grain_size);

    // Grain scheduling: interval between grains
    let grain_interval = (sr as f32 / density).round() as usize;
    let grain_interval = grain_interval.max(1);

    // Simple pseudo-random via LCG
    let mut rng_state: i64 = 42;

    for i in 0..n {
        // Write input + feedback into delay buffer
        if i >= delay_samples {
            buf[write_pos] = samples[i] + feedback * out[i - delay_samples];
        } else {
            buf[write_pos] = samples[i];
        }

        // Spawn grain at scheduled intervals
        if i % grain_interval == 0 && i >= delay_samples {
            // Pseudo-random scatter offset
            rng_state = (rng_state.wrapping_mul(1103515245).wrapping_add(12345)) & 0x7FFFFFFF;
            let scatter_range = (scatter_samples * 2 + 1).max(1) as i64;
            let scatter_offset = (rng_state % scatter_range) as isize - scatter_samples as isize;

            for j in 0..grain_size {
                let out_idx = i + j;
                if out_idx >= n {
                    break;
                }
                let raw_pos = write_pos as isize - delay_samples as isize + scatter_offset - j as isize;
                let read_pos = ((raw_pos % buf_len as isize) + buf_len as isize) as usize % buf_len;
                out[out_idx] += window[j] * buf[read_pos];
            }
        }

        write_pos = (write_pos + 1) % buf_len;
    }

    // Mix in dry signal
    for i in 0..n {
        out[i] += samples[i];
    }

    AudioOutput::Mono(out)
}

fn variants_a006() -> Vec<HashMap<String, Value>> {
    vec![
        params!{"delay_ms" => 200, "grain_size_ms" => 30, "scatter_ms" => 10, "density" => 15, "feedback" => 0.2},
        params!{"delay_ms" => 500, "grain_size_ms" => 80, "scatter_ms" => 100, "density" => 30, "feedback" => 0.5},
        params!{"delay_ms" => 1000, "grain_size_ms" => 100, "scatter_ms" => 200, "density" => 50, "feedback" => 0.0},
        params!{"delay_ms" => 150, "grain_size_ms" => 10, "scatter_ms" => 0, "density" => 40, "feedback" => 0.6},
        params!{"delay_ms" => 300, "grain_size_ms" => 50, "scatter_ms" => 50, "density" => 10, "feedback" => 0.7},
        params!{"delay_ms" => 700, "grain_size_ms" => 60, "scatter_ms" => 150, "density" => 5, "feedback" => 0.4},
    ]
}

// ---------------------------------------------------------------------------
// A007 -- Allpass Delay Diffuser
// ---------------------------------------------------------------------------

fn process_a007(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let num_stages = pi(params, "num_stages", 6) as usize;
    let delay_range_ms = pf(params, "delay_range_ms", 20.0);
    let g = pf(params, "g", 0.6);

    // Distribute delays across range using prime-like spacing
    let primes = [2usize, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
    let mut delays = vec![0usize; num_stages];
    for i in 0..num_stages {
        let prime_idx = i % primes.len();
        let denom_idx = (num_stages - 1).min(primes.len() - 1);
        let frac = primes[prime_idx] as f32 / primes[denom_idx] as f32;
        let d_ms = 1.0 + frac * (delay_range_ms - 1.0);
        delays[i] = ((d_ms * sr as f32 / 1000.0).round() as usize).max(1);
    }

    let n = samples.len();
    let mut current = samples.to_vec();

    for s in 0..num_stages {
        let d = delays[s];
        let buf_len = (d + 1).max(1);
        let mut x_buf = vec![0.0f32; buf_len];
        let mut y_buf = vec![0.0f32; buf_len];
        let mut stage_out = vec![0.0f32; n];
        let mut write_pos: usize = 0;

        for i in 0..n {
            let read_pos = (write_pos + buf_len - d) % buf_len;
            let x_delayed = x_buf[read_pos];
            let y_delayed = y_buf[read_pos];
            let y_val = -g * current[i] + x_delayed + g * y_delayed;
            x_buf[write_pos] = current[i];
            y_buf[write_pos] = y_val;
            stage_out[i] = y_val;
            write_pos = (write_pos + 1) % buf_len;
        }

        current = stage_out;
    }

    AudioOutput::Mono(current)
}

fn variants_a007() -> Vec<HashMap<String, Value>> {
    vec![
        params!{"num_stages" => 4, "delay_range_ms" => 5, "g" => 0.5},
        params!{"num_stages" => 8, "delay_range_ms" => 20, "g" => 0.6},
        params!{"num_stages" => 12, "delay_range_ms" => 50, "g" => 0.7},
        params!{"num_stages" => 6, "delay_range_ms" => 10, "g" => 0.55},
        params!{"num_stages" => 10, "delay_range_ms" => 35, "g" => 0.65},
    ]
}

// ---------------------------------------------------------------------------
// A008 -- Fibonacci Delay Network
// ---------------------------------------------------------------------------

fn process_a008(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let base_ms = pf(params, "base_ms", 20.0);
    let num_fibs = pi(params, "num_fibs", 8) as usize;
    let decay = pf(params, "decay", 0.8);

    // Generate fibonacci sequence
    let mut fibs = vec![1usize; num_fibs.max(2)];
    fibs[0] = 1;
    fibs[1] = 1;
    for i in 2..num_fibs {
        fibs[i] = fibs[i - 1] + fibs[i - 2];
    }

    let mut tap_delays = vec![0usize; num_fibs];
    let mut tap_gains = vec![0.0f32; num_fibs];
    for k in 0..num_fibs {
        let d_ms = base_ms * fibs[k] as f32;
        tap_delays[k] = ((d_ms * sr as f32 / 1000.0).round() as usize).max(1);
        tap_gains[k] = decay.powi((k + 1) as i32);
    }

    // Normalize gains so their sum stays below 0.95 for feedback stability
    let total_gain: f32 = tap_gains.iter().sum();
    if total_gain > 0.95 {
        let scale = 0.95 / total_gain;
        for g in tap_gains.iter_mut() {
            *g *= scale;
        }
    }

    let max_delay = *tap_delays.iter().max().unwrap_or(&1);
    let buf_len = (max_delay + 1).max(1);
    let mut buf = vec![0.0f32; buf_len];
    let mut out = vec![0.0f32; samples.len()];
    let mut write_pos: usize = 0;

    for i in 0..samples.len() {
        let mut y = samples[i];
        for t in 0..num_fibs {
            let read_pos = (write_pos + buf_len - tap_delays[t]) % buf_len;
            y += tap_gains[t] * buf[read_pos];
        }
        buf[write_pos] = y;
        out[i] = y;
        write_pos = (write_pos + 1) % buf_len;
    }

    AudioOutput::Mono(out)
}

fn variants_a008() -> Vec<HashMap<String, Value>> {
    vec![
        params!{"base_ms" => 5, "num_fibs" => 8, "decay" => 0.8},
        params!{"base_ms" => 15, "num_fibs" => 6, "decay" => 0.9},
        params!{"base_ms" => 50, "num_fibs" => 5, "decay" => 0.7},
        params!{"base_ms" => 10, "num_fibs" => 12, "decay" => 0.6},
        params!{"base_ms" => 30, "num_fibs" => 10, "decay" => 0.95},
        params!{"base_ms" => 8, "num_fibs" => 7, "decay" => 0.85},
    ]
}

// ---------------------------------------------------------------------------
// A009 -- Prime Number Delay
// ---------------------------------------------------------------------------

fn process_a009(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let base_ms = pf(params, "base_ms", 5.0);
    let num_primes = pi(params, "num_primes", 8) as usize;
    let feedback = pf(params, "feedback", 0.5);

    let primes = gen_primes(num_primes);
    let mut tap_delays = vec![0usize; num_primes];
    for k in 0..num_primes {
        let d_ms = base_ms * primes[k] as f32;
        tap_delays[k] = ((d_ms * sr as f32 / 1000.0).round() as usize).max(1);
    }

    let max_delay = *tap_delays.iter().max().unwrap_or(&1);
    let buf_len = (max_delay + 1).max(1);
    let mut buf = vec![0.0f32; buf_len];
    let mut out = vec![0.0f32; samples.len()];
    let mut write_pos: usize = 0;

    // Equal gain per tap scaled by feedback
    let tap_gain = feedback / num_primes as f32;

    for i in 0..samples.len() {
        let mut y = samples[i];
        for t in 0..num_primes {
            let read_pos = (write_pos + buf_len - tap_delays[t]) % buf_len;
            y += tap_gain * buf[read_pos];
        }
        buf[write_pos] = y;
        out[i] = y;
        write_pos = (write_pos + 1) % buf_len;
    }

    AudioOutput::Mono(out)
}

fn variants_a009() -> Vec<HashMap<String, Value>> {
    vec![
        params!{"base_ms" => 2, "num_primes" => 8, "feedback" => 0.5},
        params!{"base_ms" => 5, "num_primes" => 12, "feedback" => 0.6},
        params!{"base_ms" => 10, "num_primes" => 5, "feedback" => 0.8},
        params!{"base_ms" => 1, "num_primes" => 15, "feedback" => 0.3},
        params!{"base_ms" => 20, "num_primes" => 6, "feedback" => 0.7},
        params!{"base_ms" => 15, "num_primes" => 10, "feedback" => 0.4},
    ]
}

// ---------------------------------------------------------------------------
// A010 -- Stutter / Retrigger
// ---------------------------------------------------------------------------

fn process_a010(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let window_ms = pf(params, "window_ms", 80.0);
    let repeats = pi(params, "repeats", 8) as usize;
    let decay = pf(params, "decay", 0.9);
    let pitch_drift = pf(params, "pitch_drift", 0.0);

    let window_size = ((window_ms * sr as f32 / 1000.0).round() as usize).max(1);
    let n = samples.len();
    let mut out = vec![0.0f32; n];

    // Process in blocks of window_size
    let num_windows = n / window_size;

    for w in 0..num_windows {
        let src_start = w * window_size;
        let out_pos = w * window_size;

        // Capture the window
        let mut window = vec![0.0f32; window_size];
        for j in 0..window_size {
            if src_start + j < n {
                window[j] = samples[src_start + j];
            }
        }

        // Place original
        for j in 0..window_size {
            if out_pos + j < n {
                out[out_pos + j] = window[j];
            }
        }

        // Place repeats within the same window duration
        let total_repeat_space = window_size;
        let repeat_len = (total_repeat_space / repeats.max(1)).max(1);

        let mut gain = 1.0f32;
        for r in 1..repeats {
            gain *= decay;
            let drift_factor = 1.0 + pitch_drift * r as f32;
            for j in 0..repeat_len {
                let src_j = (j as f32 * drift_factor) as usize;
                if src_j >= window_size {
                    break;
                }
                let out_idx = out_pos + r * repeat_len + j;
                if out_idx < n {
                    out[out_idx] += gain * window[src_j];
                }
            }
        }
    }

    // Copy any remaining samples
    let remaining_start = num_windows * window_size;
    for i in remaining_start..n {
        out[i] = samples[i];
    }

    AudioOutput::Mono(out)
}

fn variants_a010() -> Vec<HashMap<String, Value>> {
    vec![
        params!{"window_ms" => 50, "repeats" => 4, "decay" => 0.9, "pitch_drift" => 0.0},
        params!{"window_ms" => 100, "repeats" => 8, "decay" => 0.85, "pitch_drift" => 0.02},
        params!{"window_ms" => 200, "repeats" => 16, "decay" => 0.95, "pitch_drift" => -0.05},
        params!{"window_ms" => 30, "repeats" => 32, "decay" => 0.8, "pitch_drift" => 0.0},
        params!{"window_ms" => 150, "repeats" => 6, "decay" => 1.0, "pitch_drift" => 0.1},
        params!{"window_ms" => 20, "repeats" => 12, "decay" => 0.92, "pitch_drift" => -0.1},
    ]
}

// ---------------------------------------------------------------------------
// A011 -- Buffer Shuffle
// ---------------------------------------------------------------------------

fn process_a011(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    use rand::seq::SliceRandom;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    let chunk_ms = pf(params, "chunk_ms", 200.0);
    let seed = pi(params, "seed", 42) as u64;

    let n = samples.len();
    let chunk_size = ((chunk_ms * sr as f32 / 1000.0).round() as usize).max(1);
    let num_chunks = n / chunk_size;

    if num_chunks < 2 {
        return AudioOutput::Mono(samples.to_vec());
    }

    // Create chunks
    let mut chunks: Vec<Vec<f32>> = Vec::with_capacity(num_chunks);
    for c in 0..num_chunks {
        let start = c * chunk_size;
        chunks.push(samples[start..start + chunk_size].to_vec());
    }

    // Permute chunk order
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut perm: Vec<usize> = (0..num_chunks).collect();
    perm.shuffle(&mut rng);

    // Reassemble with crossfade
    let fade_samples = ((0.005 * sr as f32) as usize).min(chunk_size / 4);
    let mut out = vec![0.0f32; n];

    for idx in 0..num_chunks {
        let src_chunk = &chunks[perm[idx]];
        let start = idx * chunk_size;

        // Apply fade in/out to each chunk
        let mut chunk_copy = src_chunk.clone();
        for j in 0..fade_samples {
            let fade = j as f32 / fade_samples as f32;
            chunk_copy[j] *= fade;
            chunk_copy[chunk_size - 1 - j] *= fade;
        }

        out[start..start + chunk_size].copy_from_slice(&chunk_copy);
    }

    // Copy remainder
    let remainder_start = num_chunks * chunk_size;
    if remainder_start < n {
        out[remainder_start..n].copy_from_slice(&samples[remainder_start..n]);
    }

    AudioOutput::Mono(out)
}

fn variants_a011() -> Vec<HashMap<String, Value>> {
    vec![
        params!{"chunk_ms" => 100, "seed" => 1},
        params!{"chunk_ms" => 250, "seed" => 17},
        params!{"chunk_ms" => 500, "seed" => 42},
        params!{"chunk_ms" => 50, "seed" => 99},
        params!{"chunk_ms" => 150, "seed" => 7},
    ]
}

// ---------------------------------------------------------------------------
// A012 -- Reverse Chunks
// ---------------------------------------------------------------------------

fn process_a012(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    use rand::Rng;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    let chunk_ms = pf(params, "chunk_ms", 150.0);
    let reverse_probability = pf(params, "reverse_probability", 0.5);

    let n = samples.len();
    let chunk_size = ((chunk_ms * sr as f32 / 1000.0).round() as usize).max(1);
    let num_chunks = n / chunk_size;
    let fade_samples = ((0.005 * sr as f32) as usize).min(chunk_size / 4);

    let mut out = vec![0.0f32; n];
    let mut rng = ChaCha8Rng::seed_from_u64(123);

    for c in 0..num_chunks {
        let start = c * chunk_size;
        let mut chunk = samples[start..start + chunk_size].to_vec();

        // Decide whether to reverse
        let should_reverse: bool = rng.gen::<f32>() < reverse_probability;

        if should_reverse {
            chunk.reverse();
        }

        // Apply crossfade
        for j in 0..fade_samples {
            let fade = j as f32 / fade_samples as f32;
            chunk[j] *= fade;
            chunk[chunk_size - 1 - j] *= fade;
        }

        out[start..start + chunk_size].copy_from_slice(&chunk);
    }

    // Copy remainder
    let remainder_start = num_chunks * chunk_size;
    if remainder_start < n {
        out[remainder_start..n].copy_from_slice(&samples[remainder_start..n]);
    }

    AudioOutput::Mono(out)
}

fn variants_a012() -> Vec<HashMap<String, Value>> {
    vec![
        params!{"chunk_ms" => 50, "reverse_probability" => 0.5},
        params!{"chunk_ms" => 100, "reverse_probability" => 0.7},
        params!{"chunk_ms" => 200, "reverse_probability" => 1.0},
        params!{"chunk_ms" => 300, "reverse_probability" => 0.3},
        params!{"chunk_ms" => 80, "reverse_probability" => 0.9},
        params!{"chunk_ms" => 150, "reverse_probability" => 0.6},
    ]
}

// ---------------------------------------------------------------------------
// A013 -- Bouncing Ball Delay
// ---------------------------------------------------------------------------

fn process_a013(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let initial_delay_ms = pf(params, "initial_delay_ms", 400.0);
    let decay = pf(params, "decay", 0.7);
    let num_bounces = pi(params, "num_bounces", 15) as usize;
    let damping = pf(params, "damping", 0.65);

    let initial_delay_samples = ((initial_delay_ms * sr as f32 / 1000.0).round() as usize).max(1);
    let n = samples.len();

    // Copy dry signal
    let mut out = samples.to_vec();

    // Each bounce: delay halves (like gravity), amplitude decays
    let mut cumulative_delay: usize = 0;
    let mut gain = 1.0f32;
    let mut current_delay = initial_delay_samples;

    for _b in 0..num_bounces {
        cumulative_delay += current_delay;
        gain *= decay;

        // Add delayed copy
        for i in 0..n {
            if i >= cumulative_delay {
                let src = i - cumulative_delay;
                if src < n {
                    out[i] += gain * samples[src];
                }
            }
        }

        // Next bounce: shorter interval (simulating gravity)
        current_delay = ((current_delay as f32 * damping).round() as usize).max(1);
        if current_delay < 1 {
            break;
        }
    }

    AudioOutput::Mono(out)
}

fn variants_a013() -> Vec<HashMap<String, Value>> {
    vec![
        params!{"initial_delay_ms" => 500, "decay" => 0.7, "num_bounces" => 12, "damping" => 0.6},
        params!{"initial_delay_ms" => 300, "decay" => 0.8, "num_bounces" => 20, "damping" => 0.7},
        params!{"initial_delay_ms" => 200, "decay" => 0.6, "num_bounces" => 10, "damping" => 0.5},
        params!{"initial_delay_ms" => 800, "decay" => 0.75, "num_bounces" => 8, "damping" => 0.65},
        params!{"initial_delay_ms" => 150, "decay" => 0.85, "num_bounces" => 25, "damping" => 0.75},
        params!{"initial_delay_ms" => 1000, "decay" => 0.5, "num_bounces" => 6, "damping" => 0.55},
    ]
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

pub fn register() -> Vec<EffectEntry> {
    vec![
        EffectEntry { id: "a001_simple_delay", process: process_a001, variants: variants_a001, category: "a_delay" },
        EffectEntry { id: "a002_multi_tap_delay", process: process_a002, variants: variants_a002, category: "a_delay" },
        EffectEntry { id: "a003_ping_pong_delay", process: process_a003, variants: variants_a003, category: "a_delay" },
        EffectEntry { id: "a004_reverse_delay", process: process_a004, variants: variants_a004, category: "a_delay" },
        EffectEntry { id: "a005_tape_delay", process: process_a005, variants: variants_a005, category: "a_delay" },
        EffectEntry { id: "a006_granular_delay", process: process_a006, variants: variants_a006, category: "a_delay" },
        EffectEntry { id: "a007_allpass_diffuser", process: process_a007, variants: variants_a007, category: "a_delay" },
        EffectEntry { id: "a008_fibonacci_delay", process: process_a008, variants: variants_a008, category: "a_delay" },
        EffectEntry { id: "a009_prime_delay", process: process_a009, variants: variants_a009, category: "a_delay" },
        EffectEntry { id: "a010_stutter", process: process_a010, variants: variants_a010, category: "a_delay" },
        EffectEntry { id: "a011_buffer_shuffle", process: process_a011, variants: variants_a011, category: "a_delay" },
        EffectEntry { id: "a012_reverse_chunks", process: process_a012, variants: variants_a012, category: "a_delay" },
        EffectEntry { id: "a013_bouncing_ball", process: process_a013, variants: variants_a013, category: "a_delay" },
    ]
}
