//! O-series: Spatial effects (O001-O004).
//!
//! O001 -- Haas Effect (Stereo Widener)
//! O002 -- Mid-Side Processing
//! O003 -- Binaural Panning
//! O004 -- Distance Simulation

use std::collections::HashMap;
use serde_json::Value;
use crate::{AudioOutput, EffectEntry, pf, pi, ps, params};
use crate::primitives::*;

// ---------------------------------------------------------------------------
// O001 -- Haas Effect (Stereo Widener)
// ---------------------------------------------------------------------------

fn process_o001(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let haas_delay_ms = pf(params, "haas_delay_ms", 10.0);
    let gain = pf(params, "gain", 0.8);

    let delay_samples = ((haas_delay_ms * sr as f32 / 1000.0) as usize).max(1);
    let n = samples.len();
    let buf_len = (delay_samples + 1).max(1);
    let mut buf = vec![0.0f32; buf_len];
    let mut write_pos: usize = 0;
    let mut out = Vec::with_capacity(n);

    for i in 0..n {
        buf[write_pos] = samples[i];
        let read_pos = (write_pos + buf_len - delay_samples) % buf_len;
        let left = samples[i];
        let right = buf[read_pos] * gain;
        out.push([left, right]);
        write_pos = (write_pos + 1) % buf_len;
    }

    AudioOutput::Stereo(out)
}

fn variants_o001() -> Vec<HashMap<String, Value>> {
    vec![
        // Subtle widening
        params!("haas_delay_ms" => 3.0, "gain" => 0.9),
        // Classic Haas
        params!("haas_delay_ms" => 10.0, "gain" => 0.8),
        // Wide stereo
        params!("haas_delay_ms" => 20.0, "gain" => 0.7),
        // Maximum width, near-echo territory
        params!("haas_delay_ms" => 30.0, "gain" => 1.0),
        // Tight, quiet side
        params!("haas_delay_ms" => 1.0, "gain" => 0.5),
        // Mid-range delay, full gain
        params!("haas_delay_ms" => 15.0, "gain" => 1.0),
    ]
}

// ---------------------------------------------------------------------------
// O002 -- Mid-Side Processing
// ---------------------------------------------------------------------------

fn process_o002(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let mid_gain_db = pf(params, "mid_gain_db", 0.0);
    let side_gain_db = pf(params, "side_gain_db", 3.0);

    let mid_gain = 10.0f32.powf(mid_gain_db / 20.0);
    let side_gain = 10.0f32.powf(side_gain_db / 20.0);

    let n = samples.len();

    // Create pseudo-stereo: L = original, R = delayed by 5 ms
    let delay_samples = ((0.005 * sr as f32) as usize).max(1);
    let mut samples_l = Vec::with_capacity(n);
    let mut samples_r = vec![0.0f32; n];

    for i in 0..n {
        samples_l.push(samples[i]);
        let j = i as isize - delay_samples as isize;
        if j >= 0 {
            samples_r[i] = samples[j as usize];
        }
    }

    // M/S encode, adjust gains, decode back to L/R
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mid = (samples_l[i] + samples_r[i]) * 0.5;
        let side = (samples_l[i] - samples_r[i]) * 0.5;
        let mid = mid * mid_gain;
        let side = side * side_gain;
        let left = mid + side;
        let right = mid - side;
        out.push([left, right]);
    }

    AudioOutput::Stereo(out)
}

fn variants_o002() -> Vec<HashMap<String, Value>> {
    vec![
        // Wide: boost sides
        params!("mid_gain_db" => 0.0, "side_gain_db" => 6.0),
        // Narrow: cut sides
        params!("mid_gain_db" => 0.0, "side_gain_db" => -6.0),
        // Mid boost, subtle widening
        params!("mid_gain_db" => 4.0, "side_gain_db" => 3.0),
        // Extreme width
        params!("mid_gain_db" => -3.0, "side_gain_db" => 12.0),
        // Mono-ish: kill sides
        params!("mid_gain_db" => 6.0, "side_gain_db" => -6.0),
        // Balanced default
        params!("mid_gain_db" => 0.0, "side_gain_db" => 3.0),
    ]
}

// ---------------------------------------------------------------------------
// O003 -- Binaural Panning
// ---------------------------------------------------------------------------

fn process_o003(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let azimuth_deg = pf(params, "azimuth_degrees", 45.0);
    let head_size_cm = pf(params, "head_size_cm", 17.0);

    let n = samples.len();

    // Physics
    let c: f32 = 343.0; // speed of sound in m/s
    let d: f32 = head_size_cm / 100.0; // head diameter in metres
    let theta_rad = azimuth_deg.abs() * std::f32::consts::PI / 180.0;

    // ITD in seconds, then convert to samples
    let itd_sec = (d / c) * theta_rad.sin();
    let itd_samples = (itd_sec * sr as f32).round().max(0.0) as usize;

    // ILD model: simple sine-law level difference
    // At 90 degrees the difference is roughly 8 dB max
    let ild_db = 8.0 * theta_rad.sin();
    let ild_gain_near = 10.0f32.powf(ild_db / 2.0 / 20.0);
    let ild_gain_far = 10.0f32.powf(-ild_db / 2.0 / 20.0);

    // Determine which ear is near
    // Positive azimuth = right, so right ear is near
    let left_is_near = azimuth_deg < 0.0;

    let buf_len = (itd_samples + 1).max(1);
    let mut buf = vec![0.0f32; buf_len];
    let mut write_pos: usize = 0;
    let mut out = Vec::with_capacity(n);

    for i in 0..n {
        buf[write_pos] = samples[i];
        let read_pos = (write_pos + buf_len - itd_samples) % buf_len;

        let near_val = samples[i] * ild_gain_near;
        let far_val = buf[read_pos] * ild_gain_far;

        if left_is_near {
            out.push([near_val, far_val]);
        } else {
            out.push([far_val, near_val]);
        }

        write_pos = (write_pos + 1) % buf_len;
    }

    AudioOutput::Stereo(out)
}

fn variants_o003() -> Vec<HashMap<String, Value>> {
    vec![
        // Centre (no panning)
        params!("azimuth_degrees" => 0.0, "head_size_cm" => 17.0),
        // Moderate right
        params!("azimuth_degrees" => 45.0, "head_size_cm" => 17.0),
        // Full right
        params!("azimuth_degrees" => 90.0, "head_size_cm" => 17.0),
        // Moderate left
        params!("azimuth_degrees" => -45.0, "head_size_cm" => 17.0),
        // Full left
        params!("azimuth_degrees" => -90.0, "head_size_cm" => 17.0),
        // Slight right, large head (more ITD)
        params!("azimuth_degrees" => 30.0, "head_size_cm" => 20.0),
        // Slight left, small head (less ITD)
        params!("azimuth_degrees" => -30.0, "head_size_cm" => 15.0),
    ]
}

// ---------------------------------------------------------------------------
// O004 -- Distance Simulation
// ---------------------------------------------------------------------------

fn process_o004(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let distance = pf(params, "distance", 10.0).max(0.5);

    let n = samples.len();

    // 1. Amplitude: inverse-distance law, normalised so distance=1 => gain=1
    let amplitude = 1.0 / distance;
    let mut attenuated = Vec::with_capacity(n);
    for &s in samples {
        attenuated.push(s * amplitude);
    }

    // 2. Air absorption lowpass: cutoff decreases with distance
    //    At 1m -> 20 kHz (essentially bypass), at 100m -> ~1 kHz
    let cutoff_hz = (20000.0 / (1.0 + (distance - 1.0) * 0.2))
        .max(200.0)
        .min(sr as f32 * 0.499);
    let dt = 1.0 / sr as f32;
    let rc = 1.0 / (2.0 * std::f32::consts::PI * cutoff_hz);
    let coeff = (-dt / rc).exp();

    // One-pole lowpass filter for air absorption
    let mut filtered = Vec::with_capacity(n);
    let mut y: f32 = 0.0;
    for i in 0..n {
        y = coeff * y + (1.0 - coeff) * attenuated[i];
        filtered.push(y);
    }

    // 3. Reverb mix: increases with distance (more diffuse at distance)
    let reverb_mix = (0.05 + (distance - 1.0) * 0.008).min(0.8).max(0.0);

    // Multi-tap delays using prime-ish spacing for diffusion
    let tap_delays_ms: [f32; 6] = [11.0, 23.0, 37.0, 53.0, 71.0, 97.0];
    let num_taps = tap_delays_ms.len();
    let mut tap_delays = Vec::with_capacity(num_taps);
    let mut tap_gains = Vec::with_capacity(num_taps);

    for t in 0..num_taps {
        let delay = ((tap_delays_ms[t] * sr as f32 / 1000.0) as usize).max(1);
        tap_delays.push(delay);
        // Decaying gains: 0.7^(t+1)
        tap_gains.push(0.7f32.powi((t + 1) as i32));
    }

    let max_delay = *tap_delays.iter().max().unwrap_or(&1);
    let buf_len = (max_delay + 1).max(1);
    let mut buf = vec![0.0f32; buf_len];
    let mut write_pos: usize = 0;
    let mut out = Vec::with_capacity(n);

    for i in 0..n {
        buf[write_pos] = filtered[i];
        let mut wet: f32 = 0.0;
        for t in 0..num_taps {
            let read_pos = (write_pos + buf_len - tap_delays[t]) % buf_len;
            wet += tap_gains[t] * buf[read_pos];
        }
        let sample = (1.0 - reverb_mix) * filtered[i] + reverb_mix * wet;
        out.push(sample);
        write_pos = (write_pos + 1) % buf_len;
    }

    AudioOutput::Mono(out)
}

fn variants_o004() -> Vec<HashMap<String, Value>> {
    vec![
        // Intimate / close-up
        params!("distance" => 0.5),
        // Arm's length
        params!("distance" => 2.0),
        // Across the room
        params!("distance" => 10.0),
        // Down the hall
        params!("distance" => 30.0),
        // Distant outdoor
        params!("distance" => 60.0),
        // Far away
        params!("distance" => 100.0),
    ]
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

pub fn register() -> Vec<EffectEntry> {
    vec![
        EffectEntry {
            id: "O001",
            process: process_o001,
            variants: variants_o001,
            category: "spatial",
        },
        EffectEntry {
            id: "O002",
            process: process_o002,
            variants: variants_o002,
            category: "spatial",
        },
        EffectEntry {
            id: "O003",
            process: process_o003,
            variants: variants_o003,
            category: "spatial",
        },
        EffectEntry {
            id: "O004",
            process: process_o004,
            variants: variants_o004,
            category: "spatial",
        },
    ]
}
