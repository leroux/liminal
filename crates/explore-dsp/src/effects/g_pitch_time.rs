//! G-series: Pitch & Time effects (G001-G007).

use std::collections::HashMap;
use std::f32::consts::PI;
use std::f64::consts::PI as PI64;

use num_complex::Complex;
use serde_json::Value;

use crate::primitives::*;
use crate::stft::{istft, stft};
use crate::{params, pf, pi, AudioOutput, EffectEntry};

// ---------------------------------------------------------------------------
// G001 -- Phase Vocoder Pitch Shift
// ---------------------------------------------------------------------------

fn process_g001(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let semitones = pf(params, "semitones", 7.0);
    let fft_size = pi(params, "fft_size", 2048) as usize;
    let hop_size = pi(params, "hop_size", 512) as usize;

    let ratio = 2.0_f64.powf(semitones as f64 / 12.0);
    let n = samples.len();

    // Analysis
    let frames = stft(samples, fft_size, hop_size);
    let num_frames = frames.len();
    let num_bins = fft_size / 2 + 1;

    let mut magnitudes = vec![vec![0.0f64; num_bins]; num_frames];
    let mut phases = vec![vec![0.0f64; num_bins]; num_frames];

    for (i, frame) in frames.iter().enumerate() {
        for j in 0..num_bins.min(frame.len()) {
            magnitudes[i][j] = frame[j].norm() as f64;
            phases[i][j] = frame[j].arg() as f64;
        }
    }

    // Phase vocoder with proper phase accumulation
    let mut synth_magnitudes = vec![vec![0.0f64; num_bins]; num_frames];
    let mut synth_phases = vec![vec![0.0f64; num_bins]; num_frames];

    let mut prev_phase = vec![0.0f64; num_bins];
    let mut prev_synth_phase = vec![0.0f64; num_bins];

    let two_pi = 2.0 * PI64;

    for frame_idx in 0..num_frames {
        for bin_idx in 0..num_bins {
            // Expected phase advance for this bin
            let expected_phase =
                prev_phase[bin_idx] + two_pi * hop_size as f64 * bin_idx as f64 / fft_size as f64;
            // Phase deviation (instantaneous frequency deviation)
            let mut deviation = phases[frame_idx][bin_idx] - expected_phase;
            // Wrap deviation to [-pi, pi]
            deviation -= two_pi * (deviation / two_pi).round();
            // True frequency for this bin
            let true_freq =
                (two_pi * bin_idx as f64 / fft_size as f64) + deviation / hop_size as f64;

            // Map this bin to the shifted output bin
            let new_bin = (bin_idx as f64 * ratio).round() as isize;
            if new_bin >= 0 && (new_bin as usize) < num_bins {
                let nb = new_bin as usize;
                synth_magnitudes[frame_idx][nb] += magnitudes[frame_idx][bin_idx];
                // Synthesis phase: accumulate using shifted frequency
                synth_phases[frame_idx][nb] =
                    prev_synth_phase[nb] + hop_size as f64 * true_freq * ratio;
            }

            prev_phase[bin_idx] = phases[frame_idx][bin_idx];
        }

        // Update previous synthesis phases for all bins
        for bin_idx in 0..num_bins {
            prev_synth_phase[bin_idx] = synth_phases[frame_idx][bin_idx];
        }
    }

    // Reconstruct complex spectrogram
    let mut synth_frames: Vec<Vec<Complex<f32>>> = Vec::with_capacity(num_frames);
    for frame_idx in 0..num_frames {
        let mut frame = vec![Complex::new(0.0f32, 0.0f32); num_bins];
        for bin_idx in 0..num_bins {
            let mag = synth_magnitudes[frame_idx][bin_idx] as f32;
            let ph = synth_phases[frame_idx][bin_idx] as f32;
            frame[bin_idx] = Complex::from_polar(mag, ph);
        }
        synth_frames.push(frame);
    }

    // Synthesis
    let mut output = istft(&synth_frames, fft_size, hop_size, Some(n));

    // Normalize to avoid clipping
    let peak = output.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    if peak > 0.0 {
        let input_peak = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        if input_peak > 0.0 {
            let scale = input_peak / peak;
            for s in output.iter_mut() {
                *s *= scale;
            }
        }
    }

    AudioOutput::Mono(output)
}

fn variants_g001() -> Vec<HashMap<String, Value>> {
    vec![
        params!("semitones" => 7.0, "fft_size" => 2048, "hop_size" => 512),    // perfect fifth up
        params!("semitones" => 12.0, "fft_size" => 2048, "hop_size" => 512),   // octave up
        params!("semitones" => -12.0, "fft_size" => 2048, "hop_size" => 512),  // octave down
        params!("semitones" => -5.0, "fft_size" => 2048, "hop_size" => 512),   // fourth down
        params!("semitones" => 3.0, "fft_size" => 4096, "hop_size" => 1024),   // minor third up, larger window
        params!("semitones" => -24.0, "fft_size" => 2048, "hop_size" => 256),  // two octaves down, fine hop
        params!("semitones" => 1.0, "fft_size" => 2048, "hop_size" => 512),    // subtle half-step up (detuned)
    ]
}

// ---------------------------------------------------------------------------
// G002 -- Granular Pitch Shift
// ---------------------------------------------------------------------------

fn granular_pitch_shift_kernel(samples: &[f32], ratio: f32, grain_size: usize, hop: usize) -> Vec<f32> {
    let n = samples.len();
    let mut out = vec![0.0f32; n];

    // Hanning window for grain
    let window = hann_window(grain_size);

    let mut pos: usize = 0;
    while pos < n {
        for i in 0..grain_size {
            // Read position in original signal, resampled by ratio
            let read_pos = pos as f64 + i as f64 * ratio as f64;
            let read_idx = read_pos.floor() as isize;
            let frac = (read_pos - read_idx as f64) as f32;
            let val = if read_idx < 0 || read_idx as usize >= n - 1 {
                0.0f32
            } else {
                let ri = read_idx as usize;
                // Linear interpolation
                samples[ri] * (1.0 - frac) + samples[ri + 1] * frac
            };

            let out_idx = pos + i;
            if out_idx < n {
                out[out_idx] += val * window[i];
            }
        }

        pos += hop;
    }
    out
}

fn process_g002(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let semitones = pf(params, "semitones", -5.0);
    let grain_size_ms = pf(params, "grain_size_ms", 50.0);

    let ratio = 2.0f32.powf(semitones / 12.0);
    let grain_size = (grain_size_ms * sr as f32 / 1000.0) as usize;
    let grain_size = grain_size.max(4);
    let hop = grain_size / 2; // 50% overlap

    let mut output = granular_pitch_shift_kernel(samples, ratio, grain_size, hop);

    // Normalize
    let peak = output.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    if peak > 0.0 {
        let input_peak = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        if input_peak > 0.0 {
            let scale = input_peak / peak;
            for s in output.iter_mut() {
                *s *= scale;
            }
        }
    }

    AudioOutput::Mono(output)
}

fn variants_g002() -> Vec<HashMap<String, Value>> {
    vec![
        params!("semitones" => -5.0, "grain_size_ms" => 50.0),   // fourth down, medium grains
        params!("semitones" => 7.0, "grain_size_ms" => 30.0),    // fifth up, small grains (more artifacts)
        params!("semitones" => -12.0, "grain_size_ms" => 80.0),  // octave down, large grains (smoother)
        params!("semitones" => 12.0, "grain_size_ms" => 40.0),   // octave up
        params!("semitones" => -3.0, "grain_size_ms" => 100.0),  // minor third down, very smooth
        params!("semitones" => 5.0, "grain_size_ms" => 20.0),    // fourth up, tiny grains (glitchy)
    ]
}

// ---------------------------------------------------------------------------
// G003 -- Harmonizer
// ---------------------------------------------------------------------------

/// Internal helper: phase vocoder pitch shift returning raw Vec<f32>.
fn phase_vocoder_pitch_shift(samples: &[f32], sr: u32, semitones: f32, fft_size: usize, hop_size: usize) -> Vec<f32> {
    let mut p = HashMap::new();
    p.insert("semitones".to_string(), serde_json::json!(semitones));
    p.insert("fft_size".to_string(), serde_json::json!(fft_size as i64));
    p.insert("hop_size".to_string(), serde_json::json!(hop_size as i64));
    match process_g001(samples, sr, &p) {
        AudioOutput::Mono(v) => v,
        AudioOutput::Stereo(v) => v.iter().map(|s| (s[0] + s[1]) * 0.5).collect(),
    }
}

fn process_g003(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let intervals: Vec<f32> = params
        .get("intervals_semitones")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_f64().map(|f| f as f32))
                .collect()
        })
        .unwrap_or_else(|| vec![7.0, 12.0]);
    let wet_mix = pf(params, "wet_mix", 0.5);

    let n = samples.len();
    let fft_size = 2048usize;
    let hop_size = 512usize;

    // Start with dry signal
    let mut output: Vec<f32> = samples.iter().map(|&s| s * (1.0 - wet_mix)).collect();

    // Add each pitch-shifted voice
    let num_intervals = intervals.len().max(1);
    let voice_gain = wet_mix / num_intervals as f32;
    for &semitones in &intervals {
        let shifted = phase_vocoder_pitch_shift(samples, sr, semitones, fft_size, hop_size);
        let len = n.min(shifted.len());
        for i in 0..len {
            output[i] += shifted[i] * voice_gain;
        }
    }

    // Normalize
    let peak = output.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    if peak > 1.0 {
        let inv = 1.0 / peak;
        for s in output.iter_mut() {
            *s *= inv;
        }
    }

    AudioOutput::Mono(output)
}

fn variants_g003() -> Vec<HashMap<String, Value>> {
    vec![
        params!("intervals_semitones" => vec![7.0], "wet_mix" => 0.5),                      // parallel fifth
        params!("intervals_semitones" => vec![12.0], "wet_mix" => 0.4),                     // octave doubler
        params!("intervals_semitones" => vec![7.0, 12.0], "wet_mix" => 0.5),               // fifth + octave (power chord)
        params!("intervals_semitones" => vec![4.0, 7.0], "wet_mix" => 0.5),                // major triad
        params!("intervals_semitones" => vec![3.0, 7.0], "wet_mix" => 0.5),                // minor triad
        params!("intervals_semitones" => vec![-12.0, 12.0], "wet_mix" => 0.6),             // sub-octave + octave up
        params!("intervals_semitones" => vec![5.0, 7.0, 12.0], "wet_mix" => 0.6),         // fourth + fifth + octave
        params!("intervals_semitones" => vec![7.0], "wet_mix" => 0.8),                      // heavy wet fifth, shimmery
    ]
}

// ---------------------------------------------------------------------------
// G004 -- Octave Up
// ---------------------------------------------------------------------------

fn fullwave_rectify(samples: &[f32]) -> Vec<f32> {
    samples.iter().map(|&s| s.abs()).collect()
}

fn biquad_bandpass_filter(samples: &[f32], b0: f32, b1: f32, b2: f32, a1: f32, a2: f32) -> Vec<f32> {
    let n = samples.len();
    let mut out = vec![0.0f32; n];
    let mut x1 = 0.0f32;
    let mut x2 = 0.0f32;
    let mut y1 = 0.0f32;
    let mut y2 = 0.0f32;
    for i in 0..n {
        let x0 = samples[i];
        let y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
        out[i] = y0;
        x2 = x1;
        x1 = x0;
        y2 = y1;
        y1 = y0;
    }
    out
}

fn process_g004(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let fundamental_hz = pf(params, "fundamental_hz", 220.0);
    let filter_q = pf(params, "filter_Q", 5.0);
    let wet_mix = pf(params, "wet_mix", 0.6);

    // Full-wave rectification
    let mut rectified = fullwave_rectify(samples);

    // Remove DC offset from rectification
    let mean: f32 = rectified.iter().sum::<f32>() / rectified.len() as f32;
    for s in rectified.iter_mut() {
        *s -= mean;
    }

    // Bandpass filter centered at 2 * fundamental (the octave)
    let target_freq = fundamental_hz * 2.0;
    let omega = 2.0 * PI * target_freq / sr as f32;
    let sin_w = omega.sin();
    let cos_w = omega.cos();
    let alpha = sin_w / (2.0 * filter_q);

    // Bandpass coefficients (constant-0dB-peak-gain)
    let a0 = 1.0 + alpha;
    let b0 = alpha / a0;
    let b1 = 0.0f32;
    let b2 = -alpha / a0;
    let a1 = -2.0 * cos_w / a0;
    let a2 = (1.0 - alpha) / a0;

    let mut filtered = biquad_bandpass_filter(&rectified, b0, b1, b2, a1, a2);

    // Normalize filtered to match input level
    let peak_f = filtered.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    let peak_x = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    if peak_f > 0.0 && peak_x > 0.0 {
        let scale = peak_x / peak_f;
        for s in filtered.iter_mut() {
            *s *= scale;
        }
    }

    // Mix
    let n = samples.len();
    let mut output = vec![0.0f32; n];
    for i in 0..n {
        output[i] = samples[i] * (1.0 - wet_mix) + filtered[i] * wet_mix;
    }

    AudioOutput::Mono(output)
}

fn variants_g004() -> Vec<HashMap<String, Value>> {
    vec![
        params!("fundamental_hz" => 220.0, "filter_Q" => 5.0, "wet_mix" => 0.6),   // guitar A3 fundamental
        params!("fundamental_hz" => 110.0, "filter_Q" => 4.0, "wet_mix" => 0.5),   // bass A2 fundamental
        params!("fundamental_hz" => 330.0, "filter_Q" => 6.0, "wet_mix" => 0.7),   // higher voice, tight filter
        params!("fundamental_hz" => 440.0, "filter_Q" => 8.0, "wet_mix" => 0.5),   // A4, very narrow filter
        params!("fundamental_hz" => 220.0, "filter_Q" => 2.0, "wet_mix" => 0.8),   // wide filter, more harmonics bleed
        params!("fundamental_hz" => 150.0, "filter_Q" => 10.0, "wet_mix" => 0.4),  // narrow isolation, subtle
    ]
}

// ---------------------------------------------------------------------------
// G005 -- Time Stretch (WSOLA)
// ---------------------------------------------------------------------------

fn wsola_cross_corr_search(
    x: &[f32],
    target_pos: isize,
    prev_end: usize,
    window_size: usize,
    search_range: isize,
    n: usize,
) -> isize {
    let mut best_offset: isize = 0;
    let mut best_corr = -1e30f32;

    // The segment that ended previous grain (for cross-correlation matching)
    let ref_start_raw = prev_end as isize - window_size as isize / 4;
    let ref_start = ref_start_raw.max(0) as usize;
    let mut ref_len = window_size / 4;
    if ref_start + ref_len > n {
        ref_len = n.saturating_sub(ref_start);
    }
    if ref_len == 0 {
        return 0;
    }

    for offset in -search_range..=search_range {
        let cand_start = target_pos + offset;
        if cand_start < 0 || (cand_start as usize) + ref_len > n {
            continue;
        }
        let cs = cand_start as usize;
        // Cross-correlation
        let mut corr = 0.0f32;
        for j in 0..ref_len {
            corr += x[ref_start + j] * x[cs + j];
        }
        if corr > best_corr {
            best_corr = corr;
            best_offset = offset;
        }
    }

    best_offset
}

fn wsola_kernel(x: &[f32], stretch_factor: f32, window_size: usize) -> Vec<f32> {
    let n = x.len();
    let hop_in = window_size / 2;
    let hop_out = (hop_in as f32 * stretch_factor) as usize;
    let out_len = (n as f32 * stretch_factor) as usize;
    let mut output = vec![0.0f32; out_len];

    // Hanning window
    let window = hann_window(window_size);

    let search_range = (hop_in / 4) as isize;
    let mut read_pos: isize = 0;
    let mut write_pos: usize = 0;
    let mut prev_end: usize = 0;

    while write_pos + window_size <= out_len && (read_pos as usize) + window_size <= n {
        // Find best alignment near expected read position
        let aligned_pos = if write_pos > 0 {
            let offset =
                wsola_cross_corr_search(x, read_pos, prev_end, window_size, search_range, n);
            let ap = read_pos + offset;
            ap
        } else {
            read_pos
        };

        let mut aligned_pos = aligned_pos.max(0) as usize;
        if aligned_pos + window_size > n {
            if n >= window_size {
                aligned_pos = n - window_size;
            } else {
                break;
            }
        }

        // Overlap-add
        for i in 0..window_size {
            if write_pos + i < out_len {
                output[write_pos + i] += x[aligned_pos + i] * window[i];
            }
        }

        prev_end = aligned_pos + window_size;
        write_pos += hop_out;
        read_pos += hop_in as isize;
    }

    output
}

fn process_g005(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let stretch_factor = pf(params, "stretch_factor", 1.5);
    let window_ms = pf(params, "window_ms", 40.0);

    let n = samples.len();
    let mut window_size = (window_ms * sr as f32 / 1000.0) as usize;
    window_size = window_size.max(4);
    // Ensure window_size is even
    if window_size % 2 != 0 {
        window_size += 1;
    }

    let mut output = wsola_kernel(samples, stretch_factor, window_size);

    // Normalize
    let peak = output.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    if peak > 0.0 {
        let input_peak = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        if input_peak > 0.0 {
            let scale = input_peak / peak;
            for s in output.iter_mut() {
                *s *= scale;
            }
        }
    }

    let _ = n; // used only conceptually
    AudioOutput::Mono(output)
}

fn variants_g005() -> Vec<HashMap<String, Value>> {
    vec![
        params!("stretch_factor" => 1.5, "window_ms" => 40.0),   // moderate stretch, standard window
        params!("stretch_factor" => 2.0, "window_ms" => 50.0),   // double length
        params!("stretch_factor" => 0.5, "window_ms" => 30.0),   // half speed (compress time)
        params!("stretch_factor" => 0.75, "window_ms" => 40.0),  // slight speedup
        params!("stretch_factor" => 3.0, "window_ms" => 60.0),   // extreme stretch
        params!("stretch_factor" => 1.25, "window_ms" => 80.0),  // subtle stretch, large window (smooth)
        params!("stretch_factor" => 4.0, "window_ms" => 20.0),   // extreme stretch, small window (grainy)
    ]
}

// ---------------------------------------------------------------------------
// G006 -- Paulstretch
// ---------------------------------------------------------------------------

fn process_g006(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let stretch_factor = pf(params, "stretch_factor", 8.0);
    let window_size = pi(params, "window_size", 4096) as usize;

    let mut x = samples.to_vec();
    let n_orig = x.len();

    // Pad input if shorter than window
    if x.len() < window_size {
        x.resize(window_size, 0.0);
    }
    let n = x.len();

    let out_length = (n as f32 * stretch_factor) as usize;
    let hop_in = window_size / 4;
    let hop_out = (hop_in as f32 * stretch_factor) as usize;

    // Number of input frames
    let num_frames = if n >= window_size {
        1 + (n - window_size) / hop_in
    } else {
        1
    };

    // Build window
    let window = hann_window(window_size);

    // Output buffer
    let mut output = vec![0.0f32; out_length];

    // Deterministic RNG (seed 42)
    let mut rng = Lcg::new(42);

    // We need per-frame FFT with random phases
    // Use realfft for forward/inverse
    let mut planner = realfft::RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(window_size);
    let ifft = planner.plan_fft_inverse(window_size);
    let mut fwd_scratch = fft.make_scratch_vec();
    let mut inv_scratch = ifft.make_scratch_vec();

    for frame_idx in 0..num_frames {
        // Extract frame
        let start = frame_idx * hop_in;
        let mut frame = vec![0.0f32; window_size];
        let end = (start + window_size).min(n);
        let valid = end - start;
        frame[..valid].copy_from_slice(&x[start..start + valid]);

        // Apply window
        for j in 0..window_size {
            frame[j] *= window[j];
        }

        // FFT
        let mut spectrum = fft.make_output_vec();
        fft.process_with_scratch(&mut frame, &mut spectrum, &mut fwd_scratch)
            .unwrap();

        let num_bins = spectrum.len();

        // Get magnitudes
        let magnitudes: Vec<f32> = spectrum.iter().map(|c| c.norm()).collect();

        // Randomize phases
        for j in 0..num_bins {
            let random_phase = rng.next_f32() * 2.0 * PI - PI; // uniform in [-pi, pi)
            spectrum[j] = Complex::from_polar(magnitudes[j], random_phase);
        }

        // IFFT
        let mut grain = ifft.make_output_vec();
        ifft.process_with_scratch(&mut spectrum, &mut grain, &mut inv_scratch)
            .unwrap();

        // realfft inverse is unnormalized, divide by fft_size
        let norm = 1.0 / window_size as f32;

        // Apply window to grain
        for j in 0..window_size {
            grain[j] *= norm * window[j];
        }

        // Place grain in output
        let out_start = (frame_idx as f64 * hop_out as f64) as usize;
        let out_end = (out_start + window_size).min(out_length);
        let valid_len = out_end - out_start;
        if valid_len > 0 {
            for j in 0..valid_len {
                output[out_start + j] += grain[j];
            }
        }
    }

    // Normalize
    let peak = output.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    if peak > 0.0 {
        let input_peak = x.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        if input_peak > 0.0 {
            let scale = input_peak / peak;
            for s in output.iter_mut() {
                *s *= scale;
            }
        }
    }

    let _ = (sr, n_orig);
    AudioOutput::Mono(output)
}

fn variants_g006() -> Vec<HashMap<String, Value>> {
    vec![
        params!("stretch_factor" => 8.0, "window_size" => 4096),     // classic paulstretch, ambient wash
        params!("stretch_factor" => 2.0, "window_size" => 2048),     // mild stretch, still recognizable
        params!("stretch_factor" => 20.0, "window_size" => 8192),    // extreme stretch, glacial drone
        params!("stretch_factor" => 50.0, "window_size" => 16384),   // ultra stretch, pure texture
        params!("stretch_factor" => 100.0, "window_size" => 65536),  // maximal stretch, frozen sound
        params!("stretch_factor" => 4.0, "window_size" => 2048),     // moderate stretch, smaller window
        params!("stretch_factor" => 10.0, "window_size" => 32768),   // large window, very smooth
    ]
}

// ---------------------------------------------------------------------------
// G007 -- Formant-Preserving Pitch Shift
// ---------------------------------------------------------------------------

/// Extract spectral envelope via cepstral method.
///
/// Takes log-magnitude spectrum, computes cepstrum, windows it to keep
/// only the low-quefrency components (spectral envelope), transforms back.
fn cepstral_envelope(spectrum: &[Complex<f32>], lpc_order: usize) -> Vec<f32> {
    let num_bins = spectrum.len();

    // log magnitude
    let log_mag: Vec<f32> = spectrum
        .iter()
        .map(|c| c.norm().max(1e-10).ln())
        .collect();

    // Real cepstrum: IRFFT of log magnitude
    // We need to treat log_mag as a half-spectrum (rfft output) and do irfft
    // This gives us a real cepstrum of length fft_size = (num_bins - 1) * 2
    let cepstrum_len = (num_bins - 1) * 2;

    let mut planner = realfft::RealFftPlanner::<f32>::new();
    let ifft = planner.plan_fft_inverse(cepstrum_len);
    let mut inv_scratch = ifft.make_scratch_vec();

    // Build complex spectrum from log_mag (zero imaginary)
    let mut spec: Vec<Complex<f32>> = log_mag
        .iter()
        .map(|&m| Complex::new(m, 0.0))
        .collect();

    let mut cepstrum = ifft.make_output_vec();
    ifft.process_with_scratch(&mut spec, &mut cepstrum, &mut inv_scratch)
        .unwrap();

    // Normalize (realfft inverse is unnormalized)
    let norm = 1.0 / cepstrum_len as f32;
    for c in cepstrum.iter_mut() {
        *c *= norm;
    }

    // Lifter: keep only first lpc_order coefficients (spectral envelope)
    let mut liftered = vec![0.0f32; cepstrum_len];
    let order = lpc_order.min(cepstrum_len / 2);
    liftered[0] = cepstrum[0];
    for i in 1..order {
        liftered[i] = cepstrum[i] * 2.0; // double for one-sided
    }
    // DC stays as-is, Nyquist if present
    if cepstrum_len % 2 == 0 && order >= cepstrum_len / 2 {
        liftered[cepstrum_len / 2] = cepstrum[cepstrum_len / 2];
    }

    // Transform back to get spectral envelope: rfft of liftered, take real, exp
    let fft = planner.plan_fft_forward(cepstrum_len);
    let mut fwd_scratch = fft.make_scratch_vec();
    let mut output_spec = fft.make_output_vec();
    fft.process_with_scratch(&mut liftered, &mut output_spec, &mut fwd_scratch)
        .unwrap();

    // The envelope is exp(real part of forward FFT), normalized
    let fft_norm = 1.0; // forward FFT in realfft is already normalized
    let envelope: Vec<f32> = output_spec
        .iter()
        .map(|c| (c.re * fft_norm).exp())
        .collect();

    envelope
}

fn process_g007(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let semitones = pf(params, "semitones", 5.0);
    let lpc_order = pi(params, "lpc_order", 30) as usize;
    let fft_size = 2048usize;
    let hop_size = 512usize;

    let ratio = 2.0_f64.powf(semitones as f64 / 12.0);
    let n = samples.len();

    // Analysis STFT
    let frames = stft(samples, fft_size, hop_size);
    let num_frames = frames.len();
    let num_bins = fft_size / 2 + 1;

    let mut magnitudes = vec![vec![0.0f64; num_bins]; num_frames];
    let mut phases = vec![vec![0.0f64; num_bins]; num_frames];

    for (i, frame) in frames.iter().enumerate() {
        for j in 0..num_bins.min(frame.len()) {
            magnitudes[i][j] = frame[j].norm() as f64;
            phases[i][j] = frame[j].arg() as f64;
        }
    }

    // For each frame: separate envelope from fine structure, shift fine, reapply envelope
    let mut y_mag = vec![vec![0.0f64; num_bins]; num_frames];
    let mut y_phase = vec![vec![0.0f64; num_bins]; num_frames];

    let mut prev_phase = vec![0.0f64; num_bins];
    let mut prev_synth_phase = vec![0.0f64; num_bins];
    let two_pi = 2.0 * PI64;

    for frame_idx in 0..num_frames {
        let frame_spectrum = &frames[frame_idx];

        // Extract spectral envelope
        let mut envelope = cepstral_envelope(frame_spectrum, lpc_order);
        // Ensure envelope matches num_bins
        if envelope.len() > num_bins {
            envelope.truncate(num_bins);
        }
        while envelope.len() < num_bins {
            envelope.push(1.0);
        }

        // Fine structure = magnitude / envelope
        let fine_structure: Vec<f64> = (0..num_bins)
            .map(|j| magnitudes[frame_idx][j] / (envelope[j] as f64).max(1e-10))
            .collect();

        // Pitch shift the fine structure (shift bins)
        let mut shifted_fine = vec![0.0f64; num_bins];
        for bin_idx in 0..num_bins {
            let new_bin = (bin_idx as f64 * ratio).round() as isize;
            if new_bin >= 0 && (new_bin as usize) < num_bins {
                shifted_fine[new_bin as usize] += fine_structure[bin_idx];
            }
        }

        // Reapply original envelope to shifted fine structure
        for j in 0..num_bins {
            y_mag[frame_idx][j] = shifted_fine[j] * envelope[j] as f64;
        }

        // Phase vocoder with proper accumulation for shifted bins
        for bin_idx in 0..num_bins {
            let expected_phase = prev_phase[bin_idx]
                + two_pi * hop_size as f64 * bin_idx as f64 / fft_size as f64;
            let mut deviation = phases[frame_idx][bin_idx] - expected_phase;
            deviation -= two_pi * (deviation / two_pi).round();
            let true_freq =
                (two_pi * bin_idx as f64 / fft_size as f64) + deviation / hop_size as f64;

            let new_bin = (bin_idx as f64 * ratio).round() as isize;
            if new_bin >= 0 && (new_bin as usize) < num_bins {
                let nb = new_bin as usize;
                y_phase[frame_idx][nb] =
                    prev_synth_phase[nb] + hop_size as f64 * true_freq * ratio;
            }

            prev_phase[bin_idx] = phases[frame_idx][bin_idx];
        }

        for bin_idx in 0..num_bins {
            prev_synth_phase[bin_idx] = y_phase[frame_idx][bin_idx];
        }
    }

    // Reconstruct complex spectrogram
    let mut synth_frames: Vec<Vec<Complex<f32>>> = Vec::with_capacity(num_frames);
    for frame_idx in 0..num_frames {
        let mut frame = vec![Complex::new(0.0f32, 0.0f32); num_bins];
        for bin_idx in 0..num_bins {
            let mag = y_mag[frame_idx][bin_idx] as f32;
            let ph = y_phase[frame_idx][bin_idx] as f32;
            frame[bin_idx] = Complex::from_polar(mag, ph);
        }
        synth_frames.push(frame);
    }

    // Synthesis
    let mut output = istft(&synth_frames, fft_size, hop_size, Some(n));

    // Normalize
    let peak = output.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    if peak > 0.0 {
        let input_peak = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        if input_peak > 0.0 {
            let scale = input_peak / peak;
            for s in output.iter_mut() {
                *s *= scale;
            }
        }
    }

    let _ = sr;
    AudioOutput::Mono(output)
}

fn variants_g007() -> Vec<HashMap<String, Value>> {
    vec![
        params!("semitones" => 5.0, "lpc_order" => 30),    // fourth up, natural voice preservation
        params!("semitones" => -5.0, "lpc_order" => 30),   // fourth down, retains formants
        params!("semitones" => 12.0, "lpc_order" => 20),   // octave up without chipmunk effect
        params!("semitones" => -12.0, "lpc_order" => 20),  // octave down without muddiness
        params!("semitones" => 7.0, "lpc_order" => 40),    // fifth up, high-order envelope (precise formants)
        params!("semitones" => -3.0, "lpc_order" => 10),   // minor third down, coarse envelope (colored)
        params!("semitones" => 2.0, "lpc_order" => 30),    // whole step up, subtle pitch correction feel
    ]
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

pub fn register() -> Vec<EffectEntry> {
    vec![
        EffectEntry {
            id: "G001",
            process: process_g001,
            variants: variants_g001,
            category: "Pitch & Time",
        },
        EffectEntry {
            id: "G002",
            process: process_g002,
            variants: variants_g002,
            category: "Pitch & Time",
        },
        EffectEntry {
            id: "G003",
            process: process_g003,
            variants: variants_g003,
            category: "Pitch & Time",
        },
        EffectEntry {
            id: "G004",
            process: process_g004,
            variants: variants_g004,
            category: "Pitch & Time",
        },
        EffectEntry {
            id: "G005",
            process: process_g005,
            variants: variants_g005,
            category: "Pitch & Time",
        },
        EffectEntry {
            id: "G006",
            process: process_g006,
            variants: variants_g006,
            category: "Pitch & Time",
        },
        EffectEntry {
            id: "G007",
            process: process_g007,
            variants: variants_g007,
            category: "Pitch & Time",
        },
    ]
}
