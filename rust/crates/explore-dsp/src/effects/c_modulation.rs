//! C-series: Modulation effects (C001-C012).
//!
//! Chorus, flanger, phaser, vibrato, tremolo, ring modulation variants,
//! frequency shifting, barber pole flanger, stereo auto-pan, doppler.

use std::collections::HashMap;
use std::f32::consts::PI;

use serde_json::Value;

use crate::{AudioOutput, EffectEntry, pf, pi, ps, params};
use crate::primitives::*;

const TWO_PI: f32 = 2.0 * PI;

// ---------------------------------------------------------------------------
// C001 -- Chorus: modulated delay with multiple voices
// ---------------------------------------------------------------------------

fn process_c001(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let base_delay_ms = pf(params, "base_delay_ms", 15.0);
    let depth_ms = pf(params, "depth_ms", 3.0);
    let rate_hz = pf(params, "rate_hz", 1.5);
    let voices = pi(params, "voices", 2) as usize;

    let n = samples.len();
    let sr_f = sr as f32;
    let base_delay_samp = base_delay_ms * 0.001 * sr_f;
    let depth_samp = depth_ms * 0.001 * sr_f;
    let max_delay = (base_delay_samp + depth_samp + 2.0) as usize;
    let buf_size = max_delay + 1;
    let mut buf = vec![0.0f32; buf_size];
    let mut out = vec![0.0f32; n];
    let mut write_pos: usize = 0;
    let inv_voices = 1.0 / voices as f32;

    for i in 0..n {
        buf[write_pos] = samples[i];
        let mut wet = 0.0f32;
        for v in 0..voices {
            let phase_offset = TWO_PI * v as f32 / voices as f32;
            let modv = (TWO_PI * rate_hz * i as f32 / sr_f + phase_offset).sin();
            let delay = base_delay_samp + depth_samp * modv;
            let mut read_pos_f = write_pos as f32 - delay;
            if read_pos_f < 0.0 {
                read_pos_f += buf_size as f32;
            }
            let idx = read_pos_f as usize;
            let frac = read_pos_f - idx as f32;
            let idx0 = idx % buf_size;
            let idx1 = (idx + 1) % buf_size;
            wet += buf[idx0] * (1.0 - frac) + buf[idx1] * frac;
        }
        out[i] = samples[i] * 0.5 + wet * inv_voices * 0.5;
        write_pos = (write_pos + 1) % buf_size;
    }
    AudioOutput::Mono(post_process(&out, sr))
}

fn variants_c001() -> Vec<HashMap<String, Value>> {
    vec![
        params!{"base_delay_ms" => 7.0, "depth_ms" => 1.5, "rate_hz" => 0.8, "voices" => 1},
        params!{"base_delay_ms" => 12.0, "depth_ms" => 3.0, "rate_hz" => 1.2, "voices" => 2},
        params!{"base_delay_ms" => 20.0, "depth_ms" => 5.0, "rate_hz" => 0.3, "voices" => 3},
        params!{"base_delay_ms" => 25.0, "depth_ms" => 8.0, "rate_hz" => 4.5, "voices" => 4},
        params!{"base_delay_ms" => 5.0, "depth_ms" => 2.0, "rate_hz" => 2.0, "voices" => 2},
        params!{"base_delay_ms" => 30.0, "depth_ms" => 10.0, "rate_hz" => 0.1, "voices" => 4},
    ]
}

// ---------------------------------------------------------------------------
// C002 -- Flanger: short modulated delay with feedback
// ---------------------------------------------------------------------------

fn process_c002(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let base_delay_ms = pf(params, "base_delay_ms", 2.0);
    let depth_ms = pf(params, "depth_ms", 2.0);
    let rate_hz = pf(params, "rate_hz", 0.3);
    let feedback = pf(params, "feedback", 0.7);

    let n = samples.len();
    let sr_f = sr as f32;
    let base_delay_samp = base_delay_ms * 0.001 * sr_f;
    let depth_samp = depth_ms * 0.001 * sr_f;
    let max_delay = (base_delay_samp + depth_samp + 2.0) as usize;
    let buf_size = max_delay + 1;
    let mut buf = vec![0.0f32; buf_size];
    let mut out = vec![0.0f32; n];
    let mut write_pos: usize = 0;
    let mut fb_sample = 0.0f32;

    for i in 0..n {
        let inp = samples[i] + feedback * fb_sample;
        buf[write_pos] = inp;
        let modv = (TWO_PI * rate_hz * i as f32 / sr_f).sin();
        let delay = base_delay_samp + depth_samp * modv;
        let mut read_pos_f = write_pos as f32 - delay;
        if read_pos_f < 0.0 {
            read_pos_f += buf_size as f32;
        }
        let idx = read_pos_f as usize;
        let frac = read_pos_f - idx as f32;
        let idx0 = idx % buf_size;
        let idx1 = (idx + 1) % buf_size;
        fb_sample = buf[idx0] * (1.0 - frac) + buf[idx1] * frac;
        out[i] = samples[i] * 0.5 + fb_sample * 0.5;
        write_pos = (write_pos + 1) % buf_size;
    }
    AudioOutput::Mono(post_process(&out, sr))
}

fn variants_c002() -> Vec<HashMap<String, Value>> {
    vec![
        params!{"base_delay_ms" => 0.5, "depth_ms" => 0.5, "rate_hz" => 0.1, "feedback" => 0.3},
        params!{"base_delay_ms" => 1.5, "depth_ms" => 1.5, "rate_hz" => 0.25, "feedback" => 0.7},
        params!{"base_delay_ms" => 3.0, "depth_ms" => 3.0, "rate_hz" => 0.5, "feedback" => -0.8},
        params!{"base_delay_ms" => 5.0, "depth_ms" => 5.0, "rate_hz" => 2.0, "feedback" => 0.95},
        params!{"base_delay_ms" => 1.0, "depth_ms" => 1.0, "rate_hz" => 0.05, "feedback" => -0.95},
        params!{"base_delay_ms" => 2.5, "depth_ms" => 4.0, "rate_hz" => 1.0, "feedback" => 0.0},
    ]
}

// ---------------------------------------------------------------------------
// C003 -- Phaser: chain of allpass filters with swept cutoff
// ---------------------------------------------------------------------------

fn process_c003(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let num_stages = pi(params, "num_stages", 6) as usize;
    let f_min = pf(params, "f_min", 200.0);
    let f_max = pf(params, "f_max", 4000.0);
    let rate_hz = pf(params, "rate_hz", 0.3);
    let feedback = pf(params, "feedback", 0.5);
    let depth = pf(params, "depth", 0.8);

    let n = samples.len();
    let sr_f = sr as f32;
    let log_f_min = f_min.ln();
    let log_f_max = f_max.ln();
    let mut ap_z = vec![0.0f32; num_stages];
    let mut fb_sample = 0.0f32;
    let mut out = vec![0.0f32; n];

    for i in 0..n {
        // LFO: 0..1
        let lfo = 0.5 * (1.0 + (TWO_PI * rate_hz * i as f32 / sr_f).sin());
        // Swept frequency in log space
        let log_f = log_f_min + (log_f_max - log_f_min) * lfo;
        let mut f = log_f.exp();
        // Clamp to avoid instability
        let max_f = sr_f * 0.49;
        if f > max_f {
            f = max_f;
        }
        // Allpass coefficient
        let tan_val = (PI * f / sr_f).tan();
        let a = (1.0 - tan_val) / (1.0 + tan_val);

        // Input with feedback
        let mut x = samples[i] + feedback * fb_sample;

        // Chain of first-order allpass filters
        for s in 0..num_stages {
            let inp = x;
            x = a * inp + ap_z[s];
            ap_z[s] = inp - a * x;
        }

        fb_sample = x;
        // Mix: dry + depth * wet
        out[i] = samples[i] + depth * x;
    }
    AudioOutput::Mono(post_process(&out, sr))
}

fn variants_c003() -> Vec<HashMap<String, Value>> {
    vec![
        params!{"num_stages" => 4, "f_min" => 100, "f_max" => 2000, "rate_hz" => 0.2, "feedback" => 0.3, "depth" => 0.7},
        params!{"num_stages" => 6, "f_min" => 200, "f_max" => 4000, "rate_hz" => 0.5, "feedback" => 0.6, "depth" => 0.9},
        params!{"num_stages" => 8, "f_min" => 300, "f_max" => 6000, "rate_hz" => 1.0, "feedback" => 0.8, "depth" => 1.0},
        params!{"num_stages" => 12, "f_min" => 400, "f_max" => 8000, "rate_hz" => 0.05, "feedback" => 0.9, "depth" => 0.5},
        params!{"num_stages" => 4, "f_min" => 150, "f_max" => 1000, "rate_hz" => 2.0, "feedback" => 0.0, "depth" => 1.0},
        params!{"num_stages" => 8, "f_min" => 100, "f_max" => 5000, "rate_hz" => 0.1, "feedback" => 0.7, "depth" => 0.6},
    ]
}

// ---------------------------------------------------------------------------
// C004 -- Vibrato: pure pitch modulation via modulated delay, no dry mix
// ---------------------------------------------------------------------------

fn process_c004(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let rate_hz = pf(params, "rate_hz", 5.0);
    let depth_ms = pf(params, "depth_ms", 3.0);

    let n = samples.len();
    let sr_f = sr as f32;
    let depth_samp = depth_ms * 0.001 * sr_f;
    let base_delay_samp = depth_samp + 1.0;
    let max_delay = (base_delay_samp + depth_samp + 2.0) as usize;
    let buf_size = max_delay + 1;
    let mut buf = vec![0.0f32; buf_size];
    let mut out = vec![0.0f32; n];
    let mut write_pos: usize = 0;

    for i in 0..n {
        buf[write_pos] = samples[i];
        let modv = (TWO_PI * rate_hz * i as f32 / sr_f).sin();
        let delay = base_delay_samp + depth_samp * modv;
        let mut read_pos_f = write_pos as f32 - delay;
        if read_pos_f < 0.0 {
            read_pos_f += buf_size as f32;
        }
        let idx = read_pos_f as usize;
        let frac = read_pos_f - idx as f32;
        let idx0 = idx % buf_size;
        let idx1 = (idx + 1) % buf_size;
        out[i] = buf[idx0] * (1.0 - frac) + buf[idx1] * frac;
        write_pos = (write_pos + 1) % buf_size;
    }
    AudioOutput::Mono(post_process(&out, sr))
}

fn variants_c004() -> Vec<HashMap<String, Value>> {
    vec![
        params!{"rate_hz" => 1.0, "depth_ms" => 1.0},
        params!{"rate_hz" => 3.0, "depth_ms" => 2.0},
        params!{"rate_hz" => 5.0, "depth_ms" => 4.0},
        params!{"rate_hz" => 7.0, "depth_ms" => 7.0},
        params!{"rate_hz" => 8.0, "depth_ms" => 10.0},
        params!{"rate_hz" => 2.0, "depth_ms" => 5.0},
    ]
}

// ---------------------------------------------------------------------------
// C005 -- Tremolo: amplitude modulation with LFO shapes
// ---------------------------------------------------------------------------

fn process_c005(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let rate_hz = pf(params, "rate_hz", 5.0);
    let depth = pf(params, "depth", 0.7);
    let shape = ps(params, "shape", "sin");
    let shape_id: i32 = match shape {
        "sin" => 0,
        "tri" => 1,
        "square" => 2,
        "sh" => 3,
        _ => 0,
    };

    let n = samples.len();
    let sr_f = sr as f32;
    let period = sr_f / rate_hz;
    let mut out = vec![0.0f32; n];

    // For sample-and-hold
    let mut sh_val = 1.0f32;
    let mut sh_counter: i32 = 0;
    let sh_period = (period as i32).max(1);
    // Simple LCG for S&H noise
    let mut rng_state: u32 = 12345;

    for i in 0..n {
        let phase = i as f32 / period;
        let frac_phase = phase - phase.floor();

        let lfo = if shape_id == 0 {
            // Sine
            0.5 * (1.0 + (TWO_PI * rate_hz * i as f32 / sr_f).sin())
        } else if shape_id == 1 {
            // Triangle
            let raw = if frac_phase < 0.5 {
                4.0 * frac_phase - 1.0
            } else {
                3.0 - 4.0 * frac_phase
            };
            0.5 * (raw + 1.0)
        } else if shape_id == 2 {
            // Square
            if frac_phase < 0.5 { 1.0 } else { 0.0 }
        } else {
            // Sample and hold
            if sh_counter <= 0 {
                rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
                sh_val = ((rng_state >> 16) & 0x7FFF) as f32 / 32767.0;
                sh_counter = sh_period;
            }
            sh_counter -= 1;
            sh_val
        };

        // Modulation: 1 - depth + depth * lfo  (ranges from 1-depth to 1)
        let modv = 1.0 - depth * (1.0 - lfo);
        out[i] = samples[i] * modv;
    }
    AudioOutput::Mono(post_process(&out, sr))
}

fn variants_c005() -> Vec<HashMap<String, Value>> {
    vec![
        params!{"rate_hz" => 3.0, "depth" => 0.5, "shape" => "sin"},
        params!{"rate_hz" => 8.0, "depth" => 0.9, "shape" => "sin"},
        params!{"rate_hz" => 5.0, "depth" => 0.7, "shape" => "tri"},
        params!{"rate_hz" => 4.0, "depth" => 1.0, "shape" => "square"},
        params!{"rate_hz" => 12.0, "depth" => 0.6, "shape" => "sh"},
        params!{"rate_hz" => 20.0, "depth" => 0.3, "shape" => "sin"},
        params!{"rate_hz" => 1.0, "depth" => 1.0, "shape" => "tri"},
    ]
}

// ---------------------------------------------------------------------------
// C006 -- Ring Modulation: y = x * sin(2*pi*freq*n/sr)
// ---------------------------------------------------------------------------

fn process_c006(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let carrier_freq_hz = pf(params, "carrier_freq_hz", 200.0);

    let n = samples.len();
    let sr_f = sr as f32;
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        let carrier = (TWO_PI * carrier_freq_hz * i as f32 / sr_f).sin();
        out[i] = samples[i] * carrier;
    }
    AudioOutput::Mono(post_process(&out, sr))
}

fn variants_c006() -> Vec<HashMap<String, Value>> {
    vec![
        params!{"carrier_freq_hz" => 20.0},
        params!{"carrier_freq_hz" => 80.0},
        params!{"carrier_freq_hz" => 200.0},
        params!{"carrier_freq_hz" => 440.0},
        params!{"carrier_freq_hz" => 1000.0},
        params!{"carrier_freq_hz" => 2000.0},
    ]
}

// ---------------------------------------------------------------------------
// C007 -- Ring Mod with Noise Carrier: bandpass filtered noise
// ---------------------------------------------------------------------------

fn process_c007(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let center_freq = pf(params, "center_freq", 1000.0);
    let bandwidth_hz = pf(params, "bandwidth_hz", 500.0);

    let n = samples.len();

    // Generate white noise with deterministic seed
    let mut rng = Lcg::new(42);
    let noise: Vec<f32> = (0..n).map(|_| rng.next_bipolar()).collect();

    // Compute bandpass biquad coefficients
    let q = center_freq / bandwidth_hz.max(1.0);
    let (b0, b1, b2, a1, a2) = biquad_coeffs_bpf(center_freq, sr, q);

    // Apply bandpass filter
    let mut carrier = biquad_filter(&noise, b0, b1, b2, a1, a2);

    // Normalize carrier to roughly unit amplitude
    let peak = carrier.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    if peak > 1e-10 {
        let inv_peak = 1.0 / peak;
        for s in carrier.iter_mut() {
            *s *= inv_peak;
        }
    }

    // Ring modulate
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        out[i] = samples[i] * carrier[i];
    }
    AudioOutput::Mono(post_process(&out, sr))
}

fn variants_c007() -> Vec<HashMap<String, Value>> {
    vec![
        params!{"center_freq" => 100.0, "bandwidth_hz" => 50.0},
        params!{"center_freq" => 500.0, "bandwidth_hz" => 200.0},
        params!{"center_freq" => 1000.0, "bandwidth_hz" => 500.0},
        params!{"center_freq" => 2000.0, "bandwidth_hz" => 100.0},
        params!{"center_freq" => 3000.0, "bandwidth_hz" => 1000.0},
        params!{"center_freq" => 5000.0, "bandwidth_hz" => 2000.0},
    ]
}

// ---------------------------------------------------------------------------
// C008 -- Ring Mod with Chaos Carrier: logistic map
// ---------------------------------------------------------------------------

fn process_c008(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let r = pf(params, "r", 3.9);
    let chaos_speed = pi(params, "chaos_speed", 1).max(1);

    let n = samples.len();
    let mut out = vec![0.0f32; n];
    let mut x = 0.5f32; // initial state of logistic map
    let mut carrier_val = 0.0f32;
    let mut counter: i32 = 0;

    for i in 0..n {
        if counter <= 0 {
            // Iterate logistic map
            x = r * x * (1.0 - x);
            // Map from [0,1] to [-1,1]
            carrier_val = 2.0 * x - 1.0;
            counter = chaos_speed;
        }
        counter -= 1;
        out[i] = samples[i] * carrier_val;
    }
    AudioOutput::Mono(post_process(&out, sr))
}

fn variants_c008() -> Vec<HashMap<String, Value>> {
    vec![
        params!{"r" => 3.5, "chaos_speed" => 1},
        params!{"r" => 3.7, "chaos_speed" => 1},
        params!{"r" => 3.85, "chaos_speed" => 2},
        params!{"r" => 3.95, "chaos_speed" => 1},
        params!{"r" => 4.0, "chaos_speed" => 4},
        params!{"r" => 3.99, "chaos_speed" => 10},
        params!{"r" => 3.6, "chaos_speed" => 50},
    ]
}

// ---------------------------------------------------------------------------
// C009 -- Frequency Shifting (Hilbert): analytic signal approach
// ---------------------------------------------------------------------------

fn process_c009(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    use realfft::RealFftPlanner;
    use num_complex::Complex;

    let shift_hz = pf(params, "shift_hz", 50.0) as f64;
    let n = samples.len();

    // We need to do the FFT at a power-of-2 or any size realfft supports.
    // realfft works with arbitrary even sizes; pad to even if needed.
    let fft_size = if n % 2 == 0 { n } else { n + 1 };

    let mut planner = RealFftPlanner::<f64>::new();
    let fft_fwd = planner.plan_fft_forward(fft_size);
    let fft_inv = planner.plan_fft_inverse(fft_size);

    // Prepare input (zero-padded if odd)
    let mut time_buf = vec![0.0f64; fft_size];
    for i in 0..n {
        time_buf[i] = samples[i] as f64;
    }

    // Forward FFT
    let mut spectrum = fft_fwd.make_output_vec(); // size = fft_size/2 + 1
    let mut scratch = fft_fwd.make_scratch_vec();
    fft_fwd.process_with_scratch(&mut time_buf, &mut spectrum, &mut scratch).unwrap();

    // Build analytic signal: zero negative frequencies, double positive.
    // For realfft, spectrum holds bins 0..=fft_size/2.
    // Bin 0 (DC) and bin fft_size/2 (Nyquist) keep their values (multiply by 1).
    // Bins 1..fft_size/2-1 are doubled (these represent positive frequencies;
    // the negative frequencies are implicitly zeroed since we won't reconstruct them
    // via the real FFT inverse -- instead we use a complex inverse).
    //
    // To properly do Hilbert via realfft, we need to reconstruct the full complex
    // spectrum, apply the H filter, then do a complex IFFT. Since realfft only
    // gives us the positive half, we'll build the full spectrum manually.

    let mut full_spectrum = vec![Complex::<f64>::new(0.0, 0.0); fft_size];

    // Fill positive frequencies from realfft output
    for i in 0..spectrum.len() {
        full_spectrum[i] = Complex::new(spectrum[i].re, spectrum[i].im);
    }
    // Fill negative frequencies (conjugate mirror)
    for i in 1..fft_size / 2 {
        full_spectrum[fft_size - i] = Complex::new(spectrum[i].re, -spectrum[i].im);
    }

    // Apply analytic signal filter H
    // H[0] = 1, H[N/2] = 1, H[1..N/2-1] = 2, H[N/2+1..N-1] = 0
    // DC stays
    // full_spectrum[0] stays as is
    for i in 1..fft_size / 2 {
        full_spectrum[i] = full_spectrum[i].scale(2.0);
    }
    // Nyquist stays as is (full_spectrum[fft_size/2])
    for i in (fft_size / 2 + 1)..fft_size {
        full_spectrum[i] = Complex::new(0.0, 0.0);
    }

    // Complex IFFT (manual DIT or use a simple approach)
    // Since we don't have a complex FFT planner readily, we can do the IFFT
    // by conjugating, doing FFT, conjugating, and dividing by N.
    // But we only have realfft. Let's do the complex IFFT manually via
    // splitting into real and imaginary parts.
    //
    // Alternative: compute analytic signal directly.
    // analytic[i] = (1/N) * sum_k full_spectrum[k] * exp(j*2*pi*k*i/N)
    // Then shift: result[i] = Re(analytic[i] * exp(j*2*pi*shift*i/sr))
    //
    // For efficiency, do the complex IFFT via two real IFFTs:
    // Split full_spectrum into even/odd or real/imag channels.
    // Pack: channel1[k] = Re(full_spectrum[k]), channel2[k] = Im(full_spectrum[k])
    // Then: f(t) = IFFT(full_spectrum) = IFFT(Re) + j*IFFT(Im)
    // But realfft inverse expects real-valued output from conjugate-symmetric input.
    //
    // Simplest correct approach: manual complex IFFT using the Cooley-Tukey
    // algorithm would be complex. Instead, let's use the relationship:
    //
    // We can compute two real IFFTs:
    //   Put Re(full_spectrum) into one real-format spectrum
    //   Put Im(full_spectrum) into another real-format spectrum
    //   IFFT each -> get time-domain real and imaginary parts
    //
    // For a real IFFT, the input spectrum must be conjugate symmetric.
    // Re(full_spectrum) IS conjugate symmetric (since Re is even if we had
    // a real signal, but after H multiplication it's not necessarily).
    //
    // Better approach: Use the DFT definition directly for moderate lengths,
    // or implement a simple radix-2 complex FFT.

    // Implement complex IFFT via definition for correctness.
    // For large N this is O(N^2), but audio buffers are typically manageable.
    // If performance is needed, a proper complex FFT crate could be used.
    //
    // Actually, let's be smarter. We can do complex IFFT using two real IFFTs
    // by packing two complex sequences into one:
    //   Let a[k] = Re(S[k]), b[k] = Im(S[k])
    //   Form c[k] = a[k] + j*b[k]  (this IS our full_spectrum)
    //   But we need IFFT of c[k].
    //
    // We can use the trick: pack two real signals into one complex FFT.
    // Given full_spectrum S[k], form:
    //   X[k] = S[k] for the forward direction... this is circular.
    //
    // Let's just do a Bluestein or split-radix. Actually, the simplest
    // approach that works with realfft:
    //
    // analytic_real[i] = Re(IFFT(full_spectrum))
    // analytic_imag[i] = Im(IFFT(full_spectrum))
    //
    // We can compute these using:
    // Pack: z[k] = full_spectrum[k]
    // Define: re_spec[k] = Re(z[k]), im_spec[k] = Im(z[k])
    //
    // Note that IFFT(z)[n] = (1/N) * sum_k z[k] * exp(j*2pi*k*n/N)
    //   = (1/N) * sum_k (re_spec[k] + j*im_spec[k]) * (cos(2pi*k*n/N) + j*sin(2pi*k*n/N))
    //
    // Real part: (1/N) * sum_k (re_spec[k]*cos - im_spec[k]*sin)
    // Imag part: (1/N) * sum_k (re_spec[k]*sin + im_spec[k]*cos)
    //
    // We can compute sum_k re_spec[k]*exp(j*2pi*k*n/N) via IFFT of re_spec
    // and sum_k im_spec[k]*exp(j*2pi*k*n/N) via IFFT of im_spec.
    //
    // But re_spec and im_spec are NOT conjugate-symmetric in general,
    // so we can't use realfft inverse directly on them.
    //
    // Final approach: just do the complex IFFT with a simple O(N log N)
    // implementation using the standard Cooley-Tukey, or for simplicity
    // (and since this only runs once per effect call, not real-time),
    // use an O(N * sqrt(N)) Bluestein or just brute-force for short signals.
    //
    // For correctness and simplicity, let's implement a basic radix-2
    // complex FFT. We'll zero-pad to next power of 2 if needed.

    let analytic = complex_ifft(&full_spectrum);

    // Frequency shift: multiply by exp(j * 2pi * shift_hz * i / sr)
    let sr_f64 = sr as f64;
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        let phase = TWO_PI as f64 * shift_hz * i as f64 / sr_f64;
        let shift_re = phase.cos();
        let shift_im = phase.sin();
        // analytic[i] * (shift_re + j*shift_im) -> take real part
        let re = analytic[i].re * shift_re - analytic[i].im * shift_im;
        out[i] = re as f32;
    }
    AudioOutput::Mono(post_process(&out, sr))
}

/// Radix-2 decimation-in-time complex FFT (forward).
/// Input length must be a power of 2.
fn complex_fft_radix2(buf: &mut [Complex<f64>], inverse: bool) {
    let n = buf.len();
    if n <= 1 {
        return;
    }
    debug_assert!(n.is_power_of_two(), "FFT size must be power of 2");

    // Bit-reversal permutation
    let mut j: usize = 0;
    for i in 0..n {
        if i < j {
            buf.swap(i, j);
        }
        let mut m = n >> 1;
        while m >= 1 && j >= m {
            j -= m;
            m >>= 1;
        }
        j += m;
    }

    // Cooley-Tukey butterfly
    let sign = if inverse { 1.0 } else { -1.0 };
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let angle = sign * std::f64::consts::TAU / len as f64;
        let wn = Complex::new(angle.cos(), angle.sin());
        let mut start = 0;
        while start < n {
            let mut w = Complex::new(1.0, 0.0);
            for k in 0..half {
                let u = buf[start + k];
                let t = w * buf[start + k + half];
                buf[start + k] = u + t;
                buf[start + k + half] = u - t;
                w = w * wn;
            }
            start += len;
        }
        len <<= 1;
    }

    if inverse {
        let inv_n = 1.0 / n as f64;
        for val in buf.iter_mut() {
            *val = val.scale(inv_n);
        }
    }
}

/// Complex IFFT for arbitrary-length input, zero-pads to next power of 2.
fn complex_ifft(spectrum: &[Complex<f64>]) -> Vec<Complex<f64>> {
    use num_complex::Complex;

    let n = spectrum.len();
    let fft_size = n.next_power_of_two();

    let mut buf = vec![Complex::<f64>::new(0.0, 0.0); fft_size];
    for i in 0..n {
        buf[i] = spectrum[i];
    }

    complex_fft_radix2(&mut buf, true);

    // Return only the first n samples
    buf.truncate(n);
    buf
}

fn variants_c009() -> Vec<HashMap<String, Value>> {
    vec![
        params!{"shift_hz" => -500.0},
        params!{"shift_hz" => -100.0},
        params!{"shift_hz" => -20.0},
        params!{"shift_hz" => 5.0},
        params!{"shift_hz" => 50.0},
        params!{"shift_hz" => 200.0},
        params!{"shift_hz" => 500.0},
    ]
}

// ---------------------------------------------------------------------------
// C010 -- Barber Pole Flanger: staggered flangers with fade in/out
// ---------------------------------------------------------------------------

fn process_c010(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let num_voices = pi(params, "num_voices", 4) as usize;
    let rate_hz = pf(params, "rate_hz", 0.15);
    let depth_ms = pf(params, "depth_ms", 3.0);
    let feedback = pf(params, "feedback", 0.5);

    let n = samples.len();
    let sr_f = sr as f32;
    let depth_samp = depth_ms * 0.001 * sr_f;
    let base_delay_samp = depth_samp + 2.0;
    let max_delay = (base_delay_samp + depth_samp + 2.0) as usize;
    let buf_size = max_delay + 1;
    let mut buf = vec![0.0f32; buf_size];
    let mut out = vec![0.0f32; n];
    let mut write_pos: usize = 0;
    let inv_voices = 1.0 / num_voices as f32;
    let mut fb_accum = 0.0f32;

    for i in 0..n {
        let inp = samples[i] + feedback * fb_accum;
        buf[write_pos] = inp;
        let mut wet = 0.0f32;

        for v in 0..num_voices {
            // Each voice has a staggered phase
            let raw_phase = rate_hz * i as f32 / sr_f + v as f32 / num_voices as f32;
            // Use triangle wave for smooth ramp
            let frac_phase = raw_phase - raw_phase.floor();
            let modv = if frac_phase < 0.5 {
                4.0 * frac_phase - 1.0
            } else {
                3.0 - 4.0 * frac_phase
            };

            // Fade envelope: sine-based crossfade on the fractional phase
            let fade = (PI * frac_phase).sin();

            let delay = base_delay_samp + depth_samp * modv;
            let mut read_pos_f = write_pos as f32 - delay;
            if read_pos_f < 0.0 {
                read_pos_f += buf_size as f32;
            }
            let idx = read_pos_f as usize;
            let frac = read_pos_f - idx as f32;
            let idx0 = idx % buf_size;
            let idx1 = (idx + 1) % buf_size;
            let voice_out = buf[idx0] * (1.0 - frac) + buf[idx1] * frac;
            wet += voice_out * fade;
        }

        fb_accum = wet * inv_voices;
        out[i] = samples[i] * 0.5 + fb_accum * 0.5;
        write_pos = (write_pos + 1) % buf_size;
    }
    AudioOutput::Mono(post_process(&out, sr))
}

fn variants_c010() -> Vec<HashMap<String, Value>> {
    vec![
        params!{"num_voices" => 3, "rate_hz" => 0.05, "depth_ms" => 1.0, "feedback" => 0.3},
        params!{"num_voices" => 4, "rate_hz" => 0.15, "depth_ms" => 3.0, "feedback" => 0.5},
        params!{"num_voices" => 5, "rate_hz" => 0.3, "depth_ms" => 5.0, "feedback" => 0.7},
        params!{"num_voices" => 6, "rate_hz" => 0.5, "depth_ms" => 2.0, "feedback" => -0.5},
        params!{"num_voices" => 3, "rate_hz" => 0.1, "depth_ms" => 4.0, "feedback" => -0.7},
        params!{"num_voices" => 4, "rate_hz" => 0.08, "depth_ms" => 1.5, "feedback" => 0.0},
    ]
}

// ---------------------------------------------------------------------------
// C011 -- Stereo Auto-Pan: L=cos, R=sin panning with LFO. Returns stereo.
// ---------------------------------------------------------------------------

fn process_c011(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let rate_hz = pf(params, "rate_hz", 2.0);
    let depth = pf(params, "depth", 0.8);
    let lfo_shape = ps(params, "lfo_shape", "sin");
    let shape_id: i32 = match lfo_shape {
        "sin" => 0,
        "tri" => 1,
        "square" => 2,
        _ => 0,
    };

    let n = samples.len();
    let sr_f = sr as f32;
    let period = sr_f / rate_hz;
    let mut out = vec![[0.0f32; 2]; n];

    for i in 0..n {
        let phase = TWO_PI * rate_hz * i as f32 / sr_f;

        let lfo = if shape_id == 0 {
            // Sine LFO
            phase.sin()
        } else if shape_id == 1 {
            // Triangle
            let frac_phase = i as f32 / period;
            let frac_phase = frac_phase - frac_phase.floor();
            if frac_phase < 0.5 {
                4.0 * frac_phase - 1.0
            } else {
                3.0 - 4.0 * frac_phase
            }
        } else {
            // Square
            let frac_phase = i as f32 / period;
            let frac_phase = frac_phase - frac_phase.floor();
            if frac_phase < 0.5 { 1.0 } else { -1.0 }
        };

        // Pan angle: center (0) + depth * lfo * pi/4
        // At center: L and R both ~0.707
        // Full depth: sweeps from hard left to hard right
        let angle = (PI * 0.25) * (1.0 + depth * lfo);
        let gain_l = angle.cos();
        let gain_r = angle.sin();

        out[i][0] = samples[i] * gain_l;
        out[i][1] = samples[i] * gain_r;
    }
    AudioOutput::Stereo(post_process_stereo(&out, sr))
}

fn variants_c011() -> Vec<HashMap<String, Value>> {
    vec![
        params!{"rate_hz" => 0.5, "depth" => 0.5, "lfo_shape" => "sin"},
        params!{"rate_hz" => 2.0, "depth" => 1.0, "lfo_shape" => "sin"},
        params!{"rate_hz" => 5.0, "depth" => 0.8, "lfo_shape" => "tri"},
        params!{"rate_hz" => 10.0, "depth" => 0.3, "lfo_shape" => "sin"},
        params!{"rate_hz" => 1.0, "depth" => 1.0, "lfo_shape" => "square"},
        params!{"rate_hz" => 0.1, "depth" => 1.0, "lfo_shape" => "tri"},
    ]
}

// ---------------------------------------------------------------------------
// C012 -- Doppler Effect: moving source with variable delay, amplitude, filtering
// ---------------------------------------------------------------------------

fn process_c012(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let speed_mps = pf(params, "speed_mps", 30.0);
    let closest_distance_m = pf(params, "closest_distance_m", 5.0);
    let path = ps(params, "path", "flyby");
    let path_id: i32 = match path {
        "flyby" => 0,
        "orbit" => 1,
        "approach" => 2,
        _ => 0,
    };

    let n = samples.len();
    let sr_f = sr as f32;
    let speed_of_sound = 343.0f32;
    let duration = n as f32 / sr_f;

    // Delay buffer for variable-delay read
    let max_delay_samp = (100.0 / speed_of_sound * sr_f) as usize + (sr_f * 0.5) as usize + 2;
    let buf_size = max_delay_samp + 1;
    let mut buf = vec![0.0f32; buf_size];
    let mut out = vec![0.0f32; n];
    let mut write_pos: usize = 0;

    // One-pole lowpass state for distance-based filtering
    let mut lp_state = 0.0f32;

    for i in 0..n {
        let t = i as f32 / sr_f;

        let distance = if path_id == 0 {
            // Flyby: source moves along x-axis, listener at origin
            let x_pos = speed_mps * (t - duration * 0.5);
            (x_pos * x_pos + closest_distance_m * closest_distance_m).sqrt()
        } else if path_id == 1 {
            // Orbit: circular path around listener
            let orbit_radius = closest_distance_m;
            let angle = TWO_PI * speed_mps * t / (TWO_PI * orbit_radius);
            // Add slight variation for realism
            orbit_radius + orbit_radius * 0.1 * (angle * 3.0).sin()
        } else {
            // Approach: source moves toward listener from far to close
            let start_dist = closest_distance_m + speed_mps * duration;
            let d = start_dist - speed_mps * t;
            d.max(closest_distance_m)
        };

        // Delay based on distance (propagation time)
        let delay_sec = distance / speed_of_sound;
        let mut delay_samp = delay_sec * sr_f;

        // Clamp delay
        if delay_samp < 1.0 {
            delay_samp = 1.0;
        }
        if delay_samp > (buf_size - 2) as f32 {
            delay_samp = (buf_size - 2) as f32;
        }

        buf[write_pos] = samples[i];

        // Read with fractional delay (linear interpolation)
        let mut read_pos_f = write_pos as f32 - delay_samp;
        if read_pos_f < 0.0 {
            read_pos_f += buf_size as f32;
        }
        let idx = read_pos_f as usize;
        let frac = read_pos_f - idx as f32;
        let idx0 = idx % buf_size;
        let idx1 = (idx + 1) % buf_size;
        let delayed = buf[idx0] * (1.0 - frac) + buf[idx1] * frac;

        // Amplitude: inverse distance law (normalized to closest distance)
        let mut amp = closest_distance_m / distance.max(0.1);
        if amp > 2.0 {
            amp = 2.0;
        }

        let sig = delayed * amp;

        // Distance-based lowpass: further = duller
        let dist_ratio = distance / closest_distance_m;
        let lp_coeff = 1.0 - 1.0 / (1.0 + dist_ratio * 0.3);
        lp_state = lp_coeff * lp_state + (1.0 - lp_coeff) * sig;

        out[i] = lp_state;
        write_pos = (write_pos + 1) % buf_size;
    }
    AudioOutput::Mono(post_process(&out, sr))
}

fn variants_c012() -> Vec<HashMap<String, Value>> {
    vec![
        params!{"speed_mps" => 10.0, "closest_distance_m" => 3.0, "path" => "flyby"},
        params!{"speed_mps" => 50.0, "closest_distance_m" => 5.0, "path" => "flyby"},
        params!{"speed_mps" => 100.0, "closest_distance_m" => 10.0, "path" => "flyby"},
        params!{"speed_mps" => 20.0, "closest_distance_m" => 8.0, "path" => "orbit"},
        params!{"speed_mps" => 60.0, "closest_distance_m" => 3.0, "path" => "orbit"},
        params!{"speed_mps" => 30.0, "closest_distance_m" => 20.0, "path" => "approach"},
        params!{"speed_mps" => 80.0, "closest_distance_m" => 1.0, "path" => "approach"},
        params!{"speed_mps" => 5.0, "closest_distance_m" => 2.0, "path" => "flyby"},
    ]
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

pub fn register() -> Vec<EffectEntry> {
    vec![
        EffectEntry {
            id: "C001",
            process: process_c001,
            variants: variants_c001,
            category: "modulation",
        },
        EffectEntry {
            id: "C002",
            process: process_c002,
            variants: variants_c002,
            category: "modulation",
        },
        EffectEntry {
            id: "C003",
            process: process_c003,
            variants: variants_c003,
            category: "modulation",
        },
        EffectEntry {
            id: "C004",
            process: process_c004,
            variants: variants_c004,
            category: "modulation",
        },
        EffectEntry {
            id: "C005",
            process: process_c005,
            variants: variants_c005,
            category: "modulation",
        },
        EffectEntry {
            id: "C006",
            process: process_c006,
            variants: variants_c006,
            category: "modulation",
        },
        EffectEntry {
            id: "C007",
            process: process_c007,
            variants: variants_c007,
            category: "modulation",
        },
        EffectEntry {
            id: "C008",
            process: process_c008,
            variants: variants_c008,
            category: "modulation",
        },
        EffectEntry {
            id: "C009",
            process: process_c009,
            variants: variants_c009,
            category: "modulation",
        },
        EffectEntry {
            id: "C010",
            process: process_c010,
            variants: variants_c010,
            category: "modulation",
        },
        EffectEntry {
            id: "C011",
            process: process_c011,
            variants: variants_c011,
            category: "modulation",
        },
        EffectEntry {
            id: "C012",
            process: process_c012,
            variants: variants_c012,
            category: "modulation",
        },
    ]
}
