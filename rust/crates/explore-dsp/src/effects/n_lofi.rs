//! N-series: Lo-fi effects (N001-N007).
//!
//! Vinyl crackle, tape hiss, tape wow/flutter, telephone, radio tuning,
//! underwater, and AM radio simulation.

use std::collections::HashMap;
use serde_json::Value;
use crate::{AudioOutput, EffectEntry, pf, pu, params};
use crate::primitives::*;

// ---------------------------------------------------------------------------
// N001 -- Vinyl Crackle Overlay
// ---------------------------------------------------------------------------

fn process_n001(samples: &[f32], sr: u32, params_map: &HashMap<String, Value>) -> AudioOutput {
    let density = pf(params_map, "crackle_density", 30.0);
    let amplitude = pf(params_map, "crackle_amplitude", 0.03);
    let seed = pu(params_map, "seed", 42);

    let n = samples.len();
    let mut out = vec![0.0f32; n];
    let avg_spacing = sr as f32 / density;
    let decay_rate = 1.0f32 / (0.002f32 * sr as f32);
    let mut rng = Lcg::new(seed);
    let mut crackle_val = 0.0f32;
    let mut next_crackle: i64 = 0;

    for i in 0..n {
        if i as i64 >= next_crackle {
            // Generate random sign
            let rand_val = rng.next_f32();
            let sign = if rand_val > 0.5 { 1.0f32 } else { -1.0f32 };
            // Amplitude variation: 0.5x to 1.5x
            let amp_var = 0.5f32 + rng.next_f32();
            crackle_val = sign * amplitude * amp_var;
            // Next crackle at random interval (exponential distribution approximation)
            let mut spacing_rand = rng.next_f32();
            if spacing_rand < 0.001 {
                spacing_rand = 0.001;
            }
            next_crackle = i as i64 + (-spacing_rand.ln() * avg_spacing) as i64;
            if next_crackle <= i as i64 {
                next_crackle = i as i64 + 1;
            }
        } else {
            // Exponential decay of current crackle
            crackle_val *= 1.0 - decay_rate;
        }
        out[i] = samples[i] + crackle_val;
    }
    AudioOutput::Mono(out)
}

fn variants_n001() -> Vec<HashMap<String, Value>> {
    vec![
        params!("crackle_density" => 5, "crackle_amplitude" => 0.02),     // rare, quiet pops
        params!("crackle_density" => 15, "crackle_amplitude" => 0.03),    // gentle vintage
        params!("crackle_density" => 30, "crackle_amplitude" => 0.03),    // standard vinyl character
        params!("crackle_density" => 60, "crackle_amplitude" => 0.05),    // well-worn record
        params!("crackle_density" => 100, "crackle_amplitude" => 0.08),   // heavily damaged vinyl
        params!("crackle_density" => 80, "crackle_amplitude" => 0.01),    // dense but subtle texture
    ]
}

// ---------------------------------------------------------------------------
// N002 -- Tape Hiss
// ---------------------------------------------------------------------------

fn process_n002(samples: &[f32], sr: u32, params_map: &HashMap<String, Value>) -> AudioOutput {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use rand::Rng;

    let hiss_level_db = pf(params_map, "hiss_level_db", -25.0).clamp(-40.0, -15.0);
    let color = crate::ps(params_map, "color", "warm");
    let seed = pu(params_map, "seed", 42);
    let level_linear = 10.0f32.powf(hiss_level_db / 20.0);

    let n = samples.len();

    // Generate white noise using rand_chacha
    let mut chacha_rng = ChaCha8Rng::seed_from_u64(seed);
    let noise: Vec<f32> = (0..n)
        .map(|_| {
            // Box-Muller-like: use standard normal approximation via 12-sample sum
            // Or just use the Rng trait's gen_range for uniform and approximate
            // Actually, use a simple normal approximation: sum of 12 uniforms - 6
            let mut sum = 0.0f32;
            for _ in 0..12 {
                sum += chacha_rng.random::<f32>();
            }
            sum - 6.0
        })
        .collect();

    // Select filter frequencies based on color mode
    let (hp_freq, lp_freq) = if color == "bright" {
        (2000.0f32, 12000.0f32)
    } else {
        (800.0f32, 6000.0f32)
    };

    // Apply highpass then lowpass to shape noise
    let (hp_b0, hp_b1, hp_b2, hp_a1, hp_a2) = biquad_coeffs_hpf(hp_freq, sr, 0.707);
    let filtered = biquad_filter(&noise, hp_b0, hp_b1, hp_b2, hp_a1, hp_a2);
    let (lp_b0, lp_b1, lp_b2, lp_a1, lp_a2) = biquad_coeffs_lpf(lp_freq, sr, 0.707);
    let filtered = biquad_filter(&filtered, lp_b0, lp_b1, lp_b2, lp_a1, lp_a2);

    // Mix hiss with signal
    let out: Vec<f32> = samples
        .iter()
        .zip(filtered.iter())
        .map(|(&s, &h)| s + h * level_linear)
        .collect();

    AudioOutput::Mono(out)
}

fn variants_n002() -> Vec<HashMap<String, Value>> {
    vec![
        params!("hiss_level_db" => -35, "color" => "warm"),      // barely perceptible warmth
        params!("hiss_level_db" => -25, "color" => "warm"),      // classic warm tape hiss
        params!("hiss_level_db" => -20, "color" => "warm"),      // noticeable warm hiss
        params!("hiss_level_db" => -30, "color" => "bright"),    // subtle bright tape character
        params!("hiss_level_db" => -20, "color" => "bright"),    // prominent bright hiss
        params!("hiss_level_db" => -15, "color" => "warm"),      // heavy worn tape noise
    ]
}

// ---------------------------------------------------------------------------
// N003 -- Tape Wow and Flutter
// ---------------------------------------------------------------------------

fn process_n003(samples: &[f32], sr: u32, params_map: &HashMap<String, Value>) -> AudioOutput {
    let wow_rate = pf(params_map, "wow_rate", 1.5);
    let wow_depth = pf(params_map, "wow_depth", 0.003);
    let flutter_rate = pf(params_map, "flutter_rate", 10.0);
    let flutter_depth = pf(params_map, "flutter_depth", 0.0005);

    let n = samples.len();
    let sr_f = sr as f32;
    let two_pi = 2.0f32 * std::f32::consts::PI;

    // Maximum additional delay in samples from modulation
    let max_mod_samples = (wow_depth + flutter_depth) * sr_f;
    // Base delay so we can modulate around it
    let base_delay = max_mod_samples + 2.0;
    let buf_size = (base_delay + max_mod_samples + 4.0) as usize;
    let mut buf = vec![0.0f32; buf_size];
    let mut out = vec![0.0f32; n];
    let mut write_pos: usize = 0;

    for i in 0..n {
        buf[write_pos] = samples[i];
        let t = i as f32 / sr_f;

        // Wow modulation (slow)
        let wow_mod = wow_depth * (two_pi * wow_rate * t).sin();
        // Flutter modulation (faster)
        let flutter_mod = flutter_depth * (two_pi * flutter_rate * t).sin();
        // Total modulation in samples
        let mod_samples = (wow_mod + flutter_mod) * sr_f;
        let mut delay = base_delay + mod_samples;

        // Clamp delay
        if delay < 1.0 {
            delay = 1.0;
        }
        if delay > (buf_size as f32 - 2.0) {
            delay = buf_size as f32 - 2.0;
        }

        // Fractional delay read with linear interpolation
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

    AudioOutput::Mono(out)
}

fn variants_n003() -> Vec<HashMap<String, Value>> {
    vec![
        params!("wow_rate" => 0.5, "wow_depth" => 0.001, "flutter_rate" => 6, "flutter_depth" => 0.0001),    // subtle, well-maintained tape
        params!("wow_rate" => 1.0, "wow_depth" => 0.003, "flutter_rate" => 10, "flutter_depth" => 0.0003),   // gentle cassette wobble
        params!("wow_rate" => 1.5, "wow_depth" => 0.003, "flutter_rate" => 10, "flutter_depth" => 0.0005),   // standard tape character
        params!("wow_rate" => 2.5, "wow_depth" => 0.006, "flutter_rate" => 15, "flutter_depth" => 0.001),    // worn-out mechanism
        params!("wow_rate" => 3.0, "wow_depth" => 0.01, "flutter_rate" => 20, "flutter_depth" => 0.002),     // broken deck, extreme wobble
        params!("wow_rate" => 0.7, "wow_depth" => 0.008, "flutter_rate" => 5, "flutter_depth" => 0.0002),    // slow deep wow, minimal flutter
        params!("wow_rate" => 2.0, "wow_depth" => 0.002, "flutter_rate" => 18, "flutter_depth" => 0.0015),   // dominant flutter, rapid shimmer
    ]
}

// ---------------------------------------------------------------------------
// N004 -- Telephone Effect
// ---------------------------------------------------------------------------

fn process_n004(samples: &[f32], sr: u32, params_map: &HashMap<String, Value>) -> AudioOutput {
    let low_cut = pf(params_map, "low_cut", 300.0).clamp(200.0, 500.0);
    let high_cut = pf(params_map, "high_cut", 3400.0).clamp(2500.0, 4000.0f32.min(sr as f32 * 0.499));
    let distortion_amount = pf(params_map, "distortion_amount", 0.2).clamp(0.0, 0.5);
    let noise_level = pf(params_map, "noise_level", 0.01).clamp(0.0, 0.05);
    let seed = pu(params_map, "seed", 42);

    // Highpass filter (remove low frequencies)
    let (hp_b0, hp_b1, hp_b2, hp_a1, hp_a2) = biquad_coeffs_hpf(low_cut, sr, 0.707);
    let filtered = biquad_filter(samples, hp_b0, hp_b1, hp_b2, hp_a1, hp_a2);

    // Lowpass filter (remove high frequencies)
    let (lp_b0, lp_b1, lp_b2, lp_a1, lp_a2) = biquad_coeffs_lpf(high_cut, sr, 0.707);
    let filtered = biquad_filter(&filtered, lp_b0, lp_b1, lp_b2, lp_a1, lp_a2);

    // Second pass of each for steeper rolloff
    let filtered = biquad_filter(&filtered, hp_b0, hp_b1, hp_b2, hp_a1, hp_a2);
    let filtered = biquad_filter(&filtered, lp_b0, lp_b1, lp_b2, lp_a1, lp_a2);

    // Apply distortion and noise
    let n = filtered.len();
    let mut out = vec![0.0f32; n];
    let mut rng = Lcg::new(seed);

    for i in 0..n {
        let mut x = filtered[i];

        // Subtle distortion via soft clipping
        if distortion_amount > 0.0 {
            let drive = 1.0 + distortion_amount * 10.0;
            x = (x * drive).tanh() / drive.tanh();
        }

        // Add noise
        if noise_level > 0.0 {
            let noise_val = rng.next_bipolar() * noise_level;
            x += noise_val;
        }

        out[i] = x;
    }

    AudioOutput::Mono(out)
}

fn variants_n004() -> Vec<HashMap<String, Value>> {
    vec![
        params!("low_cut" => 300, "high_cut" => 3400, "distortion_amount" => 0.1, "noise_level" => 0.005),    // clean landline
        params!("low_cut" => 300, "high_cut" => 3400, "distortion_amount" => 0.2, "noise_level" => 0.01),     // standard telephone
        params!("low_cut" => 400, "high_cut" => 2800, "distortion_amount" => 0.3, "noise_level" => 0.02),     // poor connection
        params!("low_cut" => 500, "high_cut" => 2500, "distortion_amount" => 0.4, "noise_level" => 0.03),     // very narrow, distorted
        params!("low_cut" => 200, "high_cut" => 4000, "distortion_amount" => 0.0, "noise_level" => 0.0),      // wideband ISDN, clean
        params!("low_cut" => 350, "high_cut" => 3000, "distortion_amount" => 0.5, "noise_level" => 0.05),     // bad mobile connection
    ]
}

// ---------------------------------------------------------------------------
// N005 -- Radio Tuning Effect
// ---------------------------------------------------------------------------

fn process_n005(samples: &[f32], sr: u32, params_map: &HashMap<String, Value>) -> AudioOutput {
    let sweep_rate = pf(params_map, "sweep_rate", 0.5).clamp(0.1, 2.0);
    let noise_level = pf(params_map, "noise_level", 0.05).clamp(0.01, 0.1);
    let signal_clarity = pf(params_map, "signal_clarity", 0.7).clamp(0.3, 1.0);
    let seed = pu(params_map, "seed", 42);

    let n = samples.len();
    let mut out = vec![0.0f32; n];
    let two_pi = 2.0f32 * std::f32::consts::PI;
    let pi_over_sr = std::f32::consts::PI / sr as f32;
    let sr_f = sr as f32;
    let mut rng = Lcg::new(seed);

    // SVF state for sweeping bandpass
    let mut lp = 0.0f32;
    let mut bp = 0.0f32;
    let q_inv = 1.0f32 / 3.0; // moderate Q for bandpass

    let log_min = 200.0f32.ln();
    let log_max = 6000.0f32.ln();
    let log_mid = 0.5 * (log_min + log_max);
    let log_half_range = 0.5 * (log_max - log_min);

    for i in 0..n {
        let t = i as f32 / sr_f;

        // Sweep the bandpass center frequency (200Hz - 6000Hz in log space)
        let sweep_phase = (two_pi * sweep_rate * t).sin();
        let log_f = log_mid + log_half_range * sweep_phase;
        let cutoff = log_f.exp();

        let mut f_coeff = 2.0 * (cutoff * pi_over_sr).sin();
        if f_coeff > 1.8 {
            f_coeff = 1.8;
        }

        // SVF step
        let x = samples[i];
        let hp = x - lp - q_inv * bp;
        bp += f_coeff * hp;
        lp += f_coeff * bp;

        // Clamp SVF states
        bp = bp.clamp(-10.0, 10.0);
        lp = lp.clamp(-10.0, 10.0);

        // AM modulation: signal clarity varies with sweep position
        // When sweep is near center (sine near 0), clarity is highest
        let clarity_mod = signal_clarity * (1.0 - 0.5 * sweep_phase.abs());

        // Generate noise
        let noise_val = rng.next_bipolar();

        // Mix: filtered signal * clarity + noise * (1 - clarity)
        out[i] = bp * clarity_mod + noise_val * noise_level * (1.0 - clarity_mod);
    }

    AudioOutput::Mono(out)
}

fn variants_n005() -> Vec<HashMap<String, Value>> {
    vec![
        params!("sweep_rate" => 0.1, "noise_level" => 0.02, "signal_clarity" => 0.9),    // slow scan, mostly clear
        params!("sweep_rate" => 0.3, "noise_level" => 0.05, "signal_clarity" => 0.7),    // gentle dial turning
        params!("sweep_rate" => 0.5, "noise_level" => 0.05, "signal_clarity" => 0.7),    // standard radio tuning
        params!("sweep_rate" => 1.0, "noise_level" => 0.08, "signal_clarity" => 0.5),    // frantic channel surfing
        params!("sweep_rate" => 2.0, "noise_level" => 0.1, "signal_clarity" => 0.3),     // chaotic dial spinning
        params!("sweep_rate" => 0.2, "noise_level" => 0.03, "signal_clarity" => 1.0),    // almost locked on station
    ]
}

// ---------------------------------------------------------------------------
// N006 -- Underwater Effect
// ---------------------------------------------------------------------------

fn process_n006(samples: &[f32], sr: u32, params_map: &HashMap<String, Value>) -> AudioOutput {
    let depth = pf(params_map, "depth", 0.5).clamp(0.1, 1.0);
    let bubble_density = pf(params_map, "bubble_density", 5.0).clamp(0.0, 20.0);
    let chorus_rate = pf(params_map, "chorus_rate", 0.5).clamp(0.3, 2.0);
    let seed = pu(params_map, "seed", 42);

    let n = samples.len();
    let mut out = vec![0.0f32; n];
    let two_pi = 2.0f32 * std::f32::consts::PI;
    let sr_f = sr as f32;

    // One-pole lowpass coefficient: deeper = lower cutoff
    // Map depth [0.1, 1.0] to cutoff [2000Hz, 200Hz]
    let cutoff_hz = (2000.0 - depth * 1800.0).max(100.0);
    let lp_coeff = (-two_pi * cutoff_hz / sr_f).exp();
    let mut lp_state = 0.0f32;
    // Second one-pole for steeper rolloff
    let mut lp_state2 = 0.0f32;

    // Chorus delay line for underwater warble
    let chorus_depth_ms = 3.0 + depth * 5.0;
    let chorus_depth_samp = chorus_depth_ms * 0.001 * sr_f;
    let base_delay_samp = chorus_depth_samp + 2.0;
    let buf_size = (base_delay_samp + chorus_depth_samp + 4.0) as usize;
    let mut buf = vec![0.0f32; buf_size];
    let mut write_pos: usize = 0;

    // Pitch wobble LFO: slow random pitch variation
    let wobble_rate = 0.3 + depth * 0.5;

    // Bubble RNG
    let mut rng = Lcg::new(seed);
    let bubble_avg_spacing = sr_f / bubble_density.max(0.01);
    let mut next_bubble: i64 = 0;
    let mut bubble_val = 0.0f32;
    let bubble_decay = 1.0 / (0.005 * sr_f); // 5ms decay

    for i in 0..n {
        let x = samples[i];

        // Two-pole lowpass (cascaded one-pole)
        lp_state = lp_coeff * lp_state + (1.0 - lp_coeff) * x;
        lp_state2 = lp_coeff * lp_state2 + (1.0 - lp_coeff) * lp_state;
        let filtered = lp_state2;

        // Write to chorus buffer
        buf[write_pos] = filtered;
        let t = i as f32 / sr_f;

        // Chorus modulation with wobble
        let modv = (two_pi * chorus_rate * t).sin();
        let wobble = (two_pi * wobble_rate * t).sin() * 0.3;
        let mut delay = base_delay_samp + chorus_depth_samp * (modv + wobble);

        if delay < 1.0 {
            delay = 1.0;
        }
        if delay > (buf_size as f32 - 2.0) {
            delay = buf_size as f32 - 2.0;
        }

        let mut read_pos_f = write_pos as f32 - delay;
        if read_pos_f < 0.0 {
            read_pos_f += buf_size as f32;
        }
        let idx = read_pos_f as usize;
        let frac = read_pos_f - idx as f32;
        let idx0 = idx % buf_size;
        let idx1 = (idx + 1) % buf_size;
        let chorus_out = buf[idx0] * (1.0 - frac) + buf[idx1] * frac;

        // Mix dry lowpassed with chorus wet
        let mut mixed = filtered * 0.6 + chorus_out * 0.4;

        // Bubble pops
        if bubble_density > 0.0 {
            if i as i64 >= next_bubble {
                let rand_val = rng.next_f32();
                bubble_val = (rand_val - 0.5) * 0.1 * depth;
                let mut spacing = rng.next_f32();
                if spacing < 0.001 {
                    spacing = 0.001;
                }
                next_bubble = i as i64 + (-spacing.ln() * bubble_avg_spacing) as i64;
                if next_bubble <= i as i64 {
                    next_bubble = i as i64 + 1;
                }
            } else {
                bubble_val *= 1.0 - bubble_decay;
            }
            mixed += bubble_val;
        }

        out[i] = mixed;
        write_pos = (write_pos + 1) % buf_size;
    }

    AudioOutput::Mono(out)
}

fn variants_n006() -> Vec<HashMap<String, Value>> {
    vec![
        params!("depth" => 0.2, "bubble_density" => 2, "chorus_rate" => 0.3),     // shallow pool, subtle muffling
        params!("depth" => 0.4, "bubble_density" => 5, "chorus_rate" => 0.5),     // swimming pool depth
        params!("depth" => 0.5, "bubble_density" => 5, "chorus_rate" => 0.5),     // standard underwater
        params!("depth" => 0.7, "bubble_density" => 10, "chorus_rate" => 0.8),    // deep dive, many bubbles
        params!("depth" => 1.0, "bubble_density" => 15, "chorus_rate" => 1.5),    // deep ocean, heavy warble
        params!("depth" => 0.3, "bubble_density" => 0, "chorus_rate" => 0.4),     // muffled, no bubbles (tank)
        params!("depth" => 0.8, "bubble_density" => 20, "chorus_rate" => 2.0),    // agitated water, dense bubbles
    ]
}

// ---------------------------------------------------------------------------
// N007 -- AM Radio Effect
// ---------------------------------------------------------------------------

fn process_n007(samples: &[f32], sr: u32, params_map: &HashMap<String, Value>) -> AudioOutput {
    let mod_index = pf(params_map, "modulation_index", 0.7).clamp(0.3, 1.0);
    let noise_level = pf(params_map, "noise_level", 0.03).clamp(0.01, 0.1);
    let hum_level = pf(params_map, "hum_level", 0.02).clamp(0.0, 0.05);
    let seed = pu(params_map, "seed", 42);

    let n = samples.len();
    let mut out = vec![0.0f32; n];
    let two_pi = 2.0f32 * std::f32::consts::PI;
    let sr_f = sr as f32;
    let mut rng = Lcg::new(seed);

    // One-pole lowpass for 5kHz bandwidth limiting (cascaded for steeper rolloff)
    let lp_coeff = (-two_pi * 5000.0 / sr_f).exp();
    let mut lp_state1 = 0.0f32;
    let mut lp_state2 = 0.0f32;

    // AM carrier frequency (audible artifact of imperfect demodulation)
    let am_carrier_hz = 1000.0f32;

    // Crackle parameters
    let crackle_spacing = sr_f / 15.0; // ~15 crackles/sec
    let mut next_crackle: i64 = 0;
    let mut crackle_val = 0.0f32;
    let crackle_decay = 1.0 / (0.001 * sr_f);

    for i in 0..n {
        let x = samples[i];
        let t = i as f32 / sr_f;

        // Bandlimit: two-pole lowpass at 5kHz
        lp_state1 = lp_coeff * lp_state1 + (1.0 - lp_coeff) * x;
        lp_state2 = lp_coeff * lp_state2 + (1.0 - lp_coeff) * lp_state1;
        let bandlimited = lp_state2;

        // AM modulation artifact: slight amplitude modulation at carrier-related frequency
        // Imperfect demodulation leaves residual carrier modulation
        let am_mod = 1.0 - mod_index * 0.1 * (1.0 - (two_pi * am_carrier_hz * t).cos());
        let modulated = bandlimited * am_mod;

        // 60Hz power line hum
        let mut hum = hum_level * (two_pi * 60.0 * t).sin();
        // Add 2nd harmonic (120Hz) for realism
        hum += hum_level * 0.3 * (two_pi * 120.0 * t).sin();

        // Background noise (static)
        let noise_val = rng.next_bipolar() * noise_level;

        // Crackle pops
        if i as i64 >= next_crackle {
            let sign_val = rng.next_f32();
            let sign = if sign_val > 0.5 { 1.0f32 } else { -1.0f32 };
            crackle_val = sign * noise_level * 2.0;
            let mut spacing = rng.next_f32();
            if spacing < 0.001 {
                spacing = 0.001;
            }
            next_crackle = i as i64 + (-spacing.ln() * crackle_spacing) as i64;
            if next_crackle <= i as i64 {
                next_crackle = i as i64 + 1;
            }
        } else {
            crackle_val *= 1.0 - crackle_decay;
        }

        out[i] = modulated + hum + noise_val + crackle_val;
    }

    AudioOutput::Mono(out)
}

fn variants_n007() -> Vec<HashMap<String, Value>> {
    vec![
        params!("modulation_index" => 0.3, "noise_level" => 0.01, "hum_level" => 0.0),     // clean AM reception
        params!("modulation_index" => 0.5, "noise_level" => 0.02, "hum_level" => 0.01),    // decent AM station
        params!("modulation_index" => 0.7, "noise_level" => 0.03, "hum_level" => 0.02),    // standard AM radio
        params!("modulation_index" => 0.8, "noise_level" => 0.06, "hum_level" => 0.03),    // weak signal reception
        params!("modulation_index" => 1.0, "noise_level" => 0.1, "hum_level" => 0.05),     // barely tuned in, heavy artifacts
        params!("modulation_index" => 0.5, "noise_level" => 0.04, "hum_level" => 0.04),    // ground loop hum dominant
    ]
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

pub fn register() -> Vec<EffectEntry> {
    vec![
        EffectEntry {
            id: "N001_vinyl_crackle",
            process: process_n001,
            variants: variants_n001,
            category: "lofi",
        },
        EffectEntry {
            id: "N002_tape_hiss",
            process: process_n002,
            variants: variants_n002,
            category: "lofi",
        },
        EffectEntry {
            id: "N003_tape_wow_flutter",
            process: process_n003,
            variants: variants_n003,
            category: "lofi",
        },
        EffectEntry {
            id: "N004_telephone",
            process: process_n004,
            variants: variants_n004,
            category: "lofi",
        },
        EffectEntry {
            id: "N005_radio_tuning",
            process: process_n005,
            variants: variants_n005,
            category: "lofi",
        },
        EffectEntry {
            id: "N006_underwater",
            process: process_n006,
            variants: variants_n006,
            category: "lofi",
        },
        EffectEntry {
            id: "N007_am_radio",
            process: process_n007,
            variants: variants_n007,
            category: "lofi",
        },
    ]
}
