//! J-series: Chaos and mathematical algorithm effects (J001-J020).

use std::collections::HashMap;
use serde_json::Value;
use crate::{AudioOutput, EffectEntry, pf, pi, ps, pu, params};
use crate::primitives::*;
use crate::stft::{stft, istft};
use num_complex::Complex;

// ---------------------------------------------------------------------------
// J001 -- Logistic Map Modulator
// ---------------------------------------------------------------------------

fn process_j001(samples: &[f32], _sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let r = pf(params, "r", 3.8);
    let step_rate = pi(params, "step_rate", 100).max(1) as usize;
    let mod_target_str = ps(params, "mod_target", "amplitude");
    let mod_depth = pf(params, "mod_depth", 0.5);

    let mod_target: i32 = if mod_target_str == "amplitude" { 0 } else { 1 };

    let n = samples.len();
    let mut out = vec![0.0f32; n];

    let mut x: f32 = 0.5;
    let mut mod_val: f32 = 0.0;
    let mut step_counter: usize = 0;
    let mut lp_state: f32 = 0.0;

    for i in 0..n {
        step_counter += 1;
        if step_counter >= step_rate {
            step_counter = 0;
            x = r * x * (1.0 - x);
            if x < 0.0 {
                x = 0.01;
            } else if x > 1.0 {
                x = 0.99;
            }
            mod_val = x;
        }

        if mod_target == 0 {
            // Amplitude modulation
            let gain = 1.0 - mod_depth + mod_depth * mod_val;
            out[i] = samples[i] * gain;
        } else {
            // Cutoff modulation (one-pole lowpass)
            let coeff = 0.05 + 0.94 * mod_depth * mod_val;
            lp_state = coeff * lp_state + (1.0 - coeff) * samples[i];
            out[i] = lp_state;
        }
    }

    AudioOutput::Mono(out)
}

fn variants_j001() -> Vec<HashMap<String, Value>> {
    vec![
        params!("r" => 3.57, "step_rate" => 200, "mod_target" => "amplitude", "mod_depth" => 0.3),
        params!("r" => 3.8, "step_rate" => 100, "mod_target" => "amplitude", "mod_depth" => 0.6),
        params!("r" => 3.99, "step_rate" => 50, "mod_target" => "amplitude", "mod_depth" => 1.0),
        params!("r" => 3.7, "step_rate" => 500, "mod_target" => "cutoff", "mod_depth" => 0.5),
        params!("r" => 3.95, "step_rate" => 30, "mod_target" => "cutoff", "mod_depth" => 0.8),
        params!("r" => 3.6, "step_rate" => 150, "mod_target" => "amplitude", "mod_depth" => 0.2),
    ]
}

// ---------------------------------------------------------------------------
// J002 -- Logistic Map Waveshaper
// ---------------------------------------------------------------------------

fn process_j002(samples: &[f32], _sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let r = pf(params, "r", 3.8);
    let num_iterations = pi(params, "num_iterations", 5).clamp(1, 20) as usize;

    let n = samples.len();
    let mut out = vec![0.0f32; n];

    for i in 0..n {
        let mut x = samples[i].abs();
        if x >= 1.0 {
            x = 0.999;
        }
        if x <= 0.0 {
            x = 0.001;
        }

        for _ in 0..num_iterations {
            x = r * x * (1.0 - x);
            if x < 0.0 {
                x = 0.01;
            } else if x > 1.0 {
                x = 0.99;
            }
        }

        let sign: f32 = if samples[i] >= 0.0 { 1.0 } else { -1.0 };
        out[i] = sign * (2.0 * x - 1.0);
    }

    AudioOutput::Mono(out)
}

fn variants_j002() -> Vec<HashMap<String, Value>> {
    vec![
        params!("r" => 3.5, "num_iterations" => 2),
        params!("r" => 3.7, "num_iterations" => 5),
        params!("r" => 3.85, "num_iterations" => 8),
        params!("r" => 3.95, "num_iterations" => 3),
        params!("r" => 4.0, "num_iterations" => 12),
        params!("r" => 3.6, "num_iterations" => 20),
    ]
}

// ---------------------------------------------------------------------------
// J003 -- Lorenz Attractor Modulation
// ---------------------------------------------------------------------------

fn process_j003(samples: &[f32], _sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let sigma: f64 = 10.0;
    let beta: f64 = 8.0 / 3.0;
    let rho = pf(params, "rho", 28.0) as f64;
    let integration_speed = pf(params, "integration_speed", 0.005) as f64;
    let mod_depth = pf(params, "mod_depth", 0.5);

    let n = samples.len();
    let mut out = vec![0.0f32; n];

    let mut x: f64 = 1.0;
    let mut y: f64 = 1.0;
    let mut z: f64 = 1.0;
    let dt = integration_speed;

    for i in 0..n {
        let dx = sigma * (y - x);
        let dy = x * (rho - z) - y;
        let dz = x * y - beta * z;
        x += dx * dt;
        y += dy * dt;
        z += dz * dt;

        // Normalize x to roughly [-1, 1] (Lorenz x typically in [-20, 20])
        let mut mod_val = (x / 20.0) as f32;
        mod_val = mod_val.clamp(-1.0, 1.0);

        // Modulate amplitude
        let gain = (1.0 + mod_depth * mod_val).max(0.0);
        out[i] = samples[i] * gain;
    }

    AudioOutput::Mono(out)
}

fn variants_j003() -> Vec<HashMap<String, Value>> {
    vec![
        params!("rho" => 20.0, "integration_speed" => 0.001, "mod_depth" => 0.3),
        params!("rho" => 28.0, "integration_speed" => 0.005, "mod_depth" => 0.5),
        params!("rho" => 28.0, "integration_speed" => 0.01, "mod_depth" => 0.8),
        params!("rho" => 35.0, "integration_speed" => 0.003, "mod_depth" => 1.0),
        params!("rho" => 24.0, "integration_speed" => 0.008, "mod_depth" => 0.4),
        params!("rho" => 32.0, "integration_speed" => 0.002, "mod_depth" => 0.7),
    ]
}

// ---------------------------------------------------------------------------
// J004 -- Lorenz as Audio-Rate Signal
// ---------------------------------------------------------------------------

fn process_j004(samples: &[f32], _sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let sigma: f64 = 10.0;
    let beta: f64 = 8.0 / 3.0;
    let rho = pf(params, "rho", 28.0) as f64;
    let mix_mode_str = ps(params, "mix_mode", "ring_mod");
    let mix_amount = pf(params, "mix_amount", 0.5);

    let mix_mode: i32 = match mix_mode_str {
        "ring_mod" => 0,
        "add" => 1,
        "am" => 2,
        _ => 0,
    };

    let n = samples.len();
    let mut out = vec![0.0f32; n];

    let mut x: f64 = 0.1;
    let mut y: f64 = 0.0;
    let mut z: f64 = 0.0;
    let dt: f64 = 0.001;

    for i in 0..n {
        let dx = sigma * (y - x);
        let dy = x * (rho - z) - y;
        let dz = x * y - beta * z;
        x += dx * dt;
        y += dy * dt;
        z += dz * dt;

        // Normalize to [-1, 1]
        let lorenz_val = (x / 25.0) as f32;
        let lorenz_val = lorenz_val.clamp(-1.0, 1.0);

        out[i] = match mix_mode {
            0 => {
                // Ring modulation
                samples[i] * lorenz_val * mix_amount + samples[i] * (1.0 - mix_amount)
            }
            1 => {
                // Additive
                samples[i] + lorenz_val * mix_amount
            }
            _ => {
                // AM: (1 + lorenz) * signal
                let am = (1.0 + lorenz_val * mix_amount) * 0.5;
                samples[i] * am
            }
        };
    }

    AudioOutput::Mono(out)
}

fn variants_j004() -> Vec<HashMap<String, Value>> {
    vec![
        params!("rho" => 28.0, "mix_mode" => "ring_mod", "mix_amount" => 0.3),
        params!("rho" => 28.0, "mix_mode" => "ring_mod", "mix_amount" => 0.7),
        params!("rho" => 35.0, "mix_mode" => "add", "mix_amount" => 0.4),
        params!("rho" => 22.0, "mix_mode" => "am", "mix_amount" => 0.5),
        params!("rho" => 28.0, "mix_mode" => "am", "mix_amount" => 0.8),
        params!("rho" => 30.0, "mix_mode" => "add", "mix_amount" => 0.2),
    ]
}

// ---------------------------------------------------------------------------
// J005 -- Henon Map Distortion
// ---------------------------------------------------------------------------

fn process_j005(samples: &[f32], _sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let a = pf(params, "a", 1.2);
    let b = pf(params, "b", 0.3);

    let n = samples.len();
    let mut out = vec![0.0f32; n];

    let mut hx: f32 = 0.0;
    let mut hy: f32 = 0.0;

    for i in 0..n {
        let inp = samples[i];
        let mut new_hx = 1.0 - a * hx * hx + hy + inp * 0.1;
        let mut new_hy = b * hx;

        new_hx = new_hx.clamp(-2.0, 2.0);
        new_hy = new_hy.clamp(-2.0, 2.0);

        hx = new_hx;
        hy = new_hy;

        out[i] = hx.tanh();
    }

    AudioOutput::Mono(out)
}

fn variants_j005() -> Vec<HashMap<String, Value>> {
    vec![
        params!("a" => 1.0, "b" => 0.2),
        params!("a" => 1.2, "b" => 0.3),
        params!("a" => 1.3, "b" => 0.3),
        params!("a" => 1.4, "b" => 0.3),
        params!("a" => 1.1, "b" => 0.4),
        params!("a" => 1.35, "b" => 0.2),
    ]
}

// ---------------------------------------------------------------------------
// J006 -- Duffing Oscillator
// ---------------------------------------------------------------------------

fn process_j006(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let delta = pf(params, "delta", 0.3) as f64;
    let alpha = pf(params, "alpha", -1.0) as f64;
    let beta = pf(params, "beta", 1.0) as f64;
    let gamma = pf(params, "gamma", 0.5) as f64;
    let omega_hz = pf(params, "omega_hz", 200.0) as f64;

    let n = samples.len();
    let mut out = vec![0.0f32; n];

    let mut x: f64 = 0.1;
    let mut v: f64 = 0.0;
    let dt = 1.0 / sr as f64;
    let two_pi: f64 = std::f64::consts::TAU;

    for i in 0..n {
        let t = i as f64 * dt;
        let forcing = gamma * (two_pi * omega_hz * t).cos() + samples[i] as f64;

        let accel = forcing - delta * v - alpha * x - beta * x * x * x;

        v += accel * dt;
        x += v * dt;

        x = x.clamp(-10.0, 10.0);
        v = v.clamp(-100.0, 100.0);

        out[i] = (x * 0.2).tanh() as f32;
    }

    AudioOutput::Mono(out)
}

fn variants_j006() -> Vec<HashMap<String, Value>> {
    vec![
        params!("delta" => 0.1, "alpha" => -1.0, "beta" => 1.0, "gamma" => 0.3, "omega_hz" => 100),
        params!("delta" => 0.3, "alpha" => -1.0, "beta" => 1.0, "gamma" => 0.5, "omega_hz" => 200),
        params!("delta" => 0.5, "alpha" => 1.0, "beta" => 0.5, "gamma" => 1.0, "omega_hz" => 300),
        params!("delta" => 0.2, "alpha" => -0.5, "beta" => 2.0, "gamma" => 1.5, "omega_hz" => 80),
        params!("delta" => 0.15, "alpha" => 0.0, "beta" => 1.5, "gamma" => 0.8, "omega_hz" => 500),
        params!("delta" => 0.4, "alpha" => -1.0, "beta" => 1.0, "gamma" => 0.1, "omega_hz" => 50),
    ]
}

// ---------------------------------------------------------------------------
// J007 -- Double Pendulum Modulation
// ---------------------------------------------------------------------------

/// Compute double pendulum angular accelerations given state.
fn dp_accels(
    theta1: f64, theta2: f64, omega1: f64, omega2: f64,
    l1: f64, l2: f64, m1: f64, m2: f64, g: f64,
) -> (f64, f64) {
    let d_theta = theta1 - theta2;
    let sin_d = d_theta.sin();
    let cos_d = d_theta.cos();

    let mut denom1 = (m1 + m2) * l1 - m2 * l1 * cos_d * cos_d;
    if denom1.abs() < 1e-10 {
        denom1 = 1e-10;
    }

    let alpha1 = (-g * (m1 + m2) * theta1.sin()
        - m2 * g * (theta1 - 2.0 * theta2).sin() * 0.5
        - m2 * sin_d * (omega2 * omega2 * l2 + omega1 * omega1 * l1 * cos_d))
        / denom1;

    let mut denom2 = l2 * (m1 + m2 - m2 * cos_d * cos_d);
    if denom2.abs() < 1e-10 {
        denom2 = 1e-10;
    }

    let alpha2 = (sin_d
        * ((m1 + m2) * (omega1 * omega1 * l1 + g * theta1.cos())
            + omega2 * omega2 * l2 * m2 * cos_d))
        / denom2;

    (alpha1, alpha2)
}

fn process_j007(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let l1 = pf(params, "l1", 1.5) as f64;
    let l2 = pf(params, "l2", 1.0) as f64;
    let initial_angle1 = pf(params, "initial_angle1", 2.0) as f64;
    let initial_angle2 = pf(params, "initial_angle2", 2.0) as f64;
    let mod_depth = pf(params, "mod_depth", 0.5);

    let m1: f64 = 1.0;
    let m2: f64 = 1.0;
    let g: f64 = 9.81;

    let n = samples.len();
    let mut out = vec![0.0f32; n];

    let mut theta1 = initial_angle1;
    let mut theta2 = initial_angle2;
    let mut omega1: f64 = 0.0;
    let mut omega2: f64 = 0.0;

    let steps_per_sample = 1;
    let dt = 1.0 / (sr as f64 * steps_per_sample as f64);

    for i in 0..n {
        for _ in 0..steps_per_sample {
            // k1
            let (a1_k1, a2_k1) = dp_accels(theta1, theta2, omega1, omega2, l1, l2, m1, m2, g);
            let k1_t1 = omega1;
            let k1_t2 = omega2;

            // k2
            let t1m = theta1 + 0.5 * dt * k1_t1;
            let t2m = theta2 + 0.5 * dt * k1_t2;
            let o1m = omega1 + 0.5 * dt * a1_k1;
            let o2m = omega2 + 0.5 * dt * a2_k1;
            let (a1_k2, a2_k2) = dp_accels(t1m, t2m, o1m, o2m, l1, l2, m1, m2, g);
            let k2_t1 = o1m;
            let k2_t2 = o2m;

            // k3
            let t1m2 = theta1 + 0.5 * dt * k2_t1;
            let t2m2 = theta2 + 0.5 * dt * k2_t2;
            let o1m2 = omega1 + 0.5 * dt * a1_k2;
            let o2m2 = omega2 + 0.5 * dt * a2_k2;
            let (a1_k3, a2_k3) = dp_accels(t1m2, t2m2, o1m2, o2m2, l1, l2, m1, m2, g);
            let k3_t1 = o1m2;
            let k3_t2 = o2m2;

            // k4
            let t1e = theta1 + dt * k3_t1;
            let t2e = theta2 + dt * k3_t2;
            let o1e = omega1 + dt * a1_k3;
            let o2e = omega2 + dt * a2_k3;
            let (a1_k4, a2_k4) = dp_accels(t1e, t2e, o1e, o2e, l1, l2, m1, m2, g);
            let k4_t1 = o1e;
            let k4_t2 = o2e;

            // Update state
            theta1 += dt / 6.0 * (k1_t1 + 2.0 * k2_t1 + 2.0 * k3_t1 + k4_t1);
            theta2 += dt / 6.0 * (k1_t2 + 2.0 * k2_t2 + 2.0 * k3_t2 + k4_t2);
            omega1 += dt / 6.0 * (a1_k1 + 2.0 * a1_k2 + 2.0 * a1_k3 + a1_k4);
            omega2 += dt / 6.0 * (a2_k1 + 2.0 * a2_k2 + 2.0 * a2_k3 + a2_k4);
        }

        // Use angular velocity of second pendulum as modulation source
        let mod_val = (omega2 * 0.1).tanh() as f32;

        let gain = (1.0 + mod_depth * mod_val).max(0.0);
        out[i] = samples[i] * gain;
    }

    AudioOutput::Mono(out)
}

fn variants_j007() -> Vec<HashMap<String, Value>> {
    vec![
        params!("l1" => 1.0, "l2" => 1.0, "initial_angle1" => 1.5, "initial_angle2" => 1.5, "mod_depth" => 0.3),
        params!("l1" => 1.5, "l2" => 1.0, "initial_angle1" => 2.0, "initial_angle2" => 2.0, "mod_depth" => 0.5),
        params!("l1" => 2.0, "l2" => 2.0, "initial_angle1" => 3.0, "initial_angle2" => 1.0, "mod_depth" => 0.8),
        params!("l1" => 1.0, "l2" => 2.0, "initial_angle1" => 2.5, "initial_angle2" => 3.0, "mod_depth" => 1.0),
        params!("l1" => 1.5, "l2" => 1.5, "initial_angle1" => 1.0, "initial_angle2" => 3.0, "mod_depth" => 0.6),
        params!("l1" => 2.0, "l2" => 1.0, "initial_angle1" => 2.8, "initial_angle2" => 1.5, "mod_depth" => 0.4),
    ]
}

// ---------------------------------------------------------------------------
// J008 -- Cellular Automaton Rhythm Gate
// ---------------------------------------------------------------------------

fn process_j008(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let rule = pi(params, "rule", 30).clamp(0, 255) as u8;
    let num_cells = pi(params, "num_cells", 32).clamp(4, 128) as usize;
    let cell_duration_ms = pf(params, "cell_duration_ms", 50.0);
    let cell_duration_samples = ((cell_duration_ms * sr as f32 / 1000.0) as usize).max(1);

    let n = samples.len();
    let mut out = vec![0.0f32; n];

    // Initialize CA state: single cell in center
    let mut state = vec![0i32; num_cells];
    state[num_cells / 2] = 1;

    let row_len = num_cells * cell_duration_samples;
    let num_rows = (n + row_len - 1) / row_len;

    let mut sample_idx = 0;
    for _row in 0..num_rows {
        // Apply gate pattern from current state
        for cell in 0..num_cells {
            for _s in 0..cell_duration_samples {
                if sample_idx >= n {
                    break;
                }
                if state[cell] == 1 {
                    out[sample_idx] = samples[sample_idx];
                } else {
                    out[sample_idx] = samples[sample_idx] * 0.05;
                }
                sample_idx += 1;
            }
        }

        // Evolve CA to next generation
        let mut new_state = vec![0i32; num_cells];
        for c in 0..num_cells {
            let left = state[(c + num_cells - 1) % num_cells];
            let center = state[c];
            let right = state[(c + 1) % num_cells];
            let neighborhood = ((left << 2) | (center << 1) | right) as u8;
            if (rule >> neighborhood) & 1 != 0 {
                new_state[c] = 1;
            }
        }
        state = new_state;
    }

    AudioOutput::Mono(out)
}

fn variants_j008() -> Vec<HashMap<String, Value>> {
    vec![
        params!("rule" => 30, "num_cells" => 32, "cell_duration_ms" => 50),
        params!("rule" => 110, "num_cells" => 64, "cell_duration_ms" => 30),
        params!("rule" => 90, "num_cells" => 16, "cell_duration_ms" => 100),
        params!("rule" => 150, "num_cells" => 48, "cell_duration_ms" => 20),
        params!("rule" => 60, "num_cells" => 128, "cell_duration_ms" => 10),
        params!("rule" => 184, "num_cells" => 24, "cell_duration_ms" => 80),
    ]
}

// ---------------------------------------------------------------------------
// J009 -- Cellular Automaton Spectral Mask
// ---------------------------------------------------------------------------

fn process_j009(samples: &[f32], _sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let rule = pi(params, "rule", 90).clamp(0, 255) as u8;
    let initial_density = pf(params, "initial_density", 0.5);

    let fft_size = 2048;
    let hop_size = 512;

    let mut frames = stft(samples, fft_size, hop_size);
    let num_frames = frames.len();
    if num_frames == 0 {
        return AudioOutput::Mono(samples.to_vec());
    }
    let num_bins = frames[0].len();

    // Initialize CA state from initial_density using deterministic LCG
    let mut rng = Lcg::new(42);
    let mut state = vec![0i32; num_bins];
    for b in 0..num_bins {
        if rng.next_f32() < initial_density {
            state[b] = 1;
        }
    }

    for frame in 0..num_frames {
        // Apply mask: bins where state==0 are attenuated
        for b in 0..num_bins {
            if state[b] == 0 {
                frames[frame][b] *= 0.05;
            }
        }

        // Evolve CA
        let mut new_state = vec![0i32; num_bins];
        for c in 0..num_bins {
            let left = state[(c + num_bins - 1) % num_bins];
            let center = state[c];
            let right = state[(c + 1) % num_bins];
            let neighborhood = ((left << 2) | (center << 1) | right) as u8;
            if (rule >> neighborhood) & 1 != 0 {
                new_state[c] = 1;
            }
        }
        state = new_state;
    }

    let out = istft(&frames, fft_size, hop_size, Some(samples.len()));
    AudioOutput::Mono(out)
}

fn variants_j009() -> Vec<HashMap<String, Value>> {
    vec![
        params!("rule" => 30, "initial_density" => 0.5),
        params!("rule" => 90, "initial_density" => 0.4),
        params!("rule" => 110, "initial_density" => 0.6),
        params!("rule" => 150, "initial_density" => 0.3),
        params!("rule" => 60, "initial_density" => 0.7),
        params!("rule" => 184, "initial_density" => 0.5),
    ]
}

// ---------------------------------------------------------------------------
// J010 -- Reaction-Diffusion Spectrogram
// ---------------------------------------------------------------------------

fn process_j010(samples: &[f32], _sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let f_param = pf(params, "F", 0.04) as f64;
    let k = pf(params, "k", 0.06) as f64;
    let num_iterations = pi(params, "num_iterations", 50).max(1) as usize;

    let fft_size = 2048;
    let hop_size = 512;

    let frames = stft(samples, fft_size, hop_size);
    let num_frames = frames.len();
    if num_frames == 0 {
        return AudioOutput::Mono(samples.to_vec());
    }
    let num_bins = frames[0].len();

    // Extract magnitude and phase
    let mut mag = vec![vec![0.0f64; num_bins]; num_frames];
    let mut phase = vec![vec![0.0f32; num_bins]; num_frames];
    let mut mag_max: f64 = 0.0;

    for r in 0..num_frames {
        for c in 0..num_bins {
            let m = frames[r][c].norm() as f64;
            mag[r][c] = m;
            phase[r][c] = frames[r][c].arg();
            if m > mag_max {
                mag_max = m;
            }
        }
    }

    if mag_max < 1e-10 {
        return AudioOutput::Mono(samples.to_vec());
    }

    // Normalize magnitudes to [0, 1]
    for r in 0..num_frames {
        for c in 0..num_bins {
            mag[r][c] /= mag_max;
        }
    }

    // Initialize U and V grids
    let mut u_grid = vec![vec![1.0f64; num_bins]; num_frames];
    let mut v_grid = mag.clone();

    let du: f64 = 0.16;
    let dv: f64 = 0.08;
    let dt_rd: f64 = 1.0;

    for _ in 0..num_iterations {
        // Laplacian via simple 2D convolution
        let mut lap_u = vec![vec![0.0f64; num_bins]; num_frames];
        let mut lap_v = vec![vec![0.0f64; num_bins]; num_frames];

        for r in 1..(num_frames.saturating_sub(1)) {
            for c in 1..(num_bins.saturating_sub(1)) {
                lap_u[r][c] = u_grid[r - 1][c] + u_grid[r + 1][c]
                    + u_grid[r][c - 1] + u_grid[r][c + 1] - 4.0 * u_grid[r][c];
                lap_v[r][c] = v_grid[r - 1][c] + v_grid[r + 1][c]
                    + v_grid[r][c - 1] + v_grid[r][c + 1] - 4.0 * v_grid[r][c];
            }
        }

        // Gray-Scott update
        for r in 0..num_frames {
            for c in 0..num_bins {
                let uvv = u_grid[r][c] * v_grid[r][c] * v_grid[r][c];
                u_grid[r][c] += dt_rd * (du * lap_u[r][c] - uvv + f_param * (1.0 - u_grid[r][c]));
                v_grid[r][c] += dt_rd * (dv * lap_v[r][c] + uvv - (f_param + k) * v_grid[r][c]);

                u_grid[r][c] = u_grid[r][c].clamp(0.0, 1.0);
                v_grid[r][c] = v_grid[r][c].clamp(0.0, 1.0);
            }
        }
    }

    // Use V as new magnitude scaling
    let mut out_frames = vec![vec![Complex::new(0.0f32, 0.0); num_bins]; num_frames];
    for r in 0..num_frames {
        for c in 0..num_bins {
            let new_mag = (v_grid[r][c] * mag_max) as f32;
            let p = phase[r][c];
            out_frames[r][c] = Complex::new(new_mag * p.cos(), new_mag * p.sin());
        }
    }

    let out = istft(&out_frames, fft_size, hop_size, Some(samples.len()));
    AudioOutput::Mono(out)
}

fn variants_j010() -> Vec<HashMap<String, Value>> {
    vec![
        params!("F" => 0.02, "k" => 0.05, "num_iterations" => 30),
        params!("F" => 0.04, "k" => 0.06, "num_iterations" => 50),
        params!("F" => 0.03, "k" => 0.055, "num_iterations" => 100),
        params!("F" => 0.06, "k" => 0.07, "num_iterations" => 20),
        params!("F" => 0.025, "k" => 0.045, "num_iterations" => 150),
        params!("F" => 0.05, "k" => 0.065, "num_iterations" => 80),
    ]
}

// ---------------------------------------------------------------------------
// J011 -- L-System Parameter Sequencer
// ---------------------------------------------------------------------------

fn expand_lsystem(axiom: &str, rules: &HashMap<char, String>, iterations: usize) -> String {
    let mut current = axiom.to_string();
    for _ in 0..iterations {
        let mut next = String::with_capacity(current.len() * 2);
        for ch in current.chars() {
            if let Some(replacement) = rules.get(&ch) {
                next.push_str(replacement);
            } else {
                next.push(ch);
            }
        }
        current = next;
    }
    current
}

fn process_j011(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let axiom = ps(params, "axiom", "F");
    let rules_str = ps(params, "rules_str", "F->F+F-F");
    let iterations = pi(params, "iterations", 4).clamp(1, 8) as usize;
    let step_duration_ms = pf(params, "step_duration_ms", 50.0);
    let step_duration_samples = ((step_duration_ms * sr as f32 / 1000.0) as usize).max(1);

    // Parse rules
    let mut rules: HashMap<char, String> = HashMap::new();
    for part in rules_str.split(',') {
        let part = part.trim();
        if let Some(arrow_pos) = part.find("->") {
            let lhs = part[..arrow_pos].trim();
            let rhs = part[arrow_pos + 2..].trim();
            if let Some(ch) = lhs.chars().next() {
                rules.insert(ch, rhs.to_string());
            }
        }
    }

    // Expand
    let expanded = expand_lsystem(axiom, &rules, iterations);

    // Map symbols to cutoff coefficients
    let mut cutoff_val: f32 = 0.5;
    let mut cutoff_list: Vec<f32> = Vec::new();
    for ch in expanded.chars() {
        match ch {
            'F' => cutoff_val = (cutoff_val + 0.05).min(0.98),
            '+' => cutoff_val = (cutoff_val + 0.1).min(0.98),
            '-' => cutoff_val = (cutoff_val - 0.1).max(0.05),
            _ => cutoff_val = 0.5,
        }
        cutoff_list.push(cutoff_val);
    }

    if cutoff_list.is_empty() {
        cutoff_list.push(0.5);
    }

    // Limit sequence length
    let max_steps = 10000;
    if cutoff_list.len() > max_steps {
        cutoff_list.truncate(max_steps);
    }

    // Apply one-pole lowpass with cutoff changing per step from L-system
    let n = samples.len();
    let mut out = vec![0.0f32; n];
    let num_steps = cutoff_list.len();

    let mut lp_state: f32 = 0.0;
    let mut step_idx: usize = 0;
    let mut step_counter: usize = 0;

    for i in 0..n {
        step_counter += 1;
        if step_counter >= step_duration_samples && step_idx < num_steps - 1 {
            step_counter = 0;
            step_idx += 1;
        }

        let coeff = cutoff_list[step_idx];
        lp_state = coeff * lp_state + (1.0 - coeff) * samples[i];
        out[i] = lp_state;
    }

    AudioOutput::Mono(out)
}

fn variants_j011() -> Vec<HashMap<String, Value>> {
    vec![
        params!("axiom" => "F", "rules_str" => "F->F+F-F", "iterations" => 3, "step_duration_ms" => 50),
        params!("axiom" => "F", "rules_str" => "F->F+F-F", "iterations" => 5, "step_duration_ms" => 20),
        params!("axiom" => "F", "rules_str" => "F->FF+F-", "iterations" => 4, "step_duration_ms" => 40),
        params!("axiom" => "F", "rules_str" => "F->F-F+F+F-F", "iterations" => 3, "step_duration_ms" => 30),
        params!("axiom" => "F", "rules_str" => "F->F+F--F+F", "iterations" => 4, "step_duration_ms" => 80),
        params!("axiom" => "F", "rules_str" => "F->+F-F+", "iterations" => 6, "step_duration_ms" => 100),
    ]
}

// ---------------------------------------------------------------------------
// J012 -- IFS Audio
// ---------------------------------------------------------------------------

fn process_j012(samples: &[f32], _sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let num_transforms = pi(params, "num_transforms", 3).clamp(2, 5) as usize;
    let contraction = pf(params, "contraction", 0.5);
    let iterations = pi(params, "iterations", 5).clamp(1, 10) as usize;

    let n = samples.len();
    let mut out = vec![0.0f32; n];

    let chunk_size = (n / 16).max(64);

    // Generate transform parameters deterministically
    let mut scales = vec![0.0f32; num_transforms];
    let mut offsets = vec![0.0f32; num_transforms];
    for t in 0..num_transforms {
        scales[t] = contraction;
        offsets[t] = t as f32 / num_transforms as f32;
    }

    let num_chunks = (n / chunk_size).max(1);

    for c in 0..num_chunks {
        let start = c * chunk_size;
        let end = (start + chunk_size).min(n);
        let actual_len = end - start;

        // Copy chunk
        let mut result = vec![0.0f32; actual_len];
        for j in 0..actual_len {
            result[j] = samples[start + j];
        }

        // Apply IFS iterations
        for _it in 0..iterations {
            let mut new_result = vec![0.0f32; actual_len];
            for t in 0..num_transforms {
                for j in 0..actual_len {
                    let src_pos = j as f32 * scales[t] + offsets[t] * actual_len as f32;
                    let src_idx = (src_pos as usize) % actual_len;
                    new_result[j] += result[src_idx] * scales[t];
                }
            }
            // Normalize
            let max_val = new_result.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            if max_val > 0.001 {
                for j in 0..actual_len {
                    new_result[j] /= max_val;
                }
            }
            result = new_result;
        }

        for j in 0..actual_len {
            out[start + j] = result[j];
        }
    }

    AudioOutput::Mono(out)
}

fn variants_j012() -> Vec<HashMap<String, Value>> {
    vec![
        params!("num_transforms" => 2, "contraction" => 0.5, "iterations" => 3),
        params!("num_transforms" => 3, "contraction" => 0.5, "iterations" => 5),
        params!("num_transforms" => 4, "contraction" => 0.4, "iterations" => 7),
        params!("num_transforms" => 5, "contraction" => 0.3, "iterations" => 10),
        params!("num_transforms" => 2, "contraction" => 0.7, "iterations" => 4),
        params!("num_transforms" => 3, "contraction" => 0.6, "iterations" => 6),
    ]
}

// ---------------------------------------------------------------------------
// J013 -- Fibonacci Rhythmic Gate
// ---------------------------------------------------------------------------

fn fibonacci_word(num_generations: usize) -> Vec<i32> {
    if num_generations == 0 {
        return vec![1];
    }
    if num_generations == 1 {
        return vec![0];
    }

    let mut prev2 = vec![1i32];
    let mut prev1 = vec![0i32];
    for _ in 2..=num_generations {
        let mut current = prev1.clone();
        current.extend_from_slice(&prev2);
        prev2 = prev1;
        prev1 = current;
    }
    prev1
}

fn process_j013(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let base_ms = pf(params, "base_ms", 80.0);
    let num_generations = pi(params, "num_generations", 8).clamp(2, 15) as usize;

    let base_samples = ((base_ms * sr as f32 / 1000.0) as usize).max(1);

    let mut pattern = fibonacci_word(num_generations);
    if pattern.len() > 5000 {
        pattern.truncate(5000);
    }

    let n = samples.len();
    let mut out = vec![0.0f32; n];
    let pat_len = pattern.len();

    if pat_len == 0 {
        return AudioOutput::Mono(samples.to_vec());
    }

    let mut sample_idx = 0;
    let mut pat_idx = 0;

    while sample_idx < n {
        let gate_on = pattern[pat_idx % pat_len];

        for _s in 0..base_samples {
            if sample_idx >= n {
                break;
            }
            if gate_on == 1 {
                out[sample_idx] = samples[sample_idx];
            } else {
                out[sample_idx] = samples[sample_idx] * 0.02;
            }
            sample_idx += 1;
        }

        pat_idx += 1;
    }

    AudioOutput::Mono(out)
}

fn variants_j013() -> Vec<HashMap<String, Value>> {
    vec![
        params!("base_ms" => 50, "num_generations" => 6),
        params!("base_ms" => 80, "num_generations" => 8),
        params!("base_ms" => 120, "num_generations" => 10),
        params!("base_ms" => 200, "num_generations" => 5),
        params!("base_ms" => 20, "num_generations" => 12),
        params!("base_ms" => 30, "num_generations" => 9),
    ]
}

// ---------------------------------------------------------------------------
// J014 -- Brownian Motion Walk
// ---------------------------------------------------------------------------

fn process_j014(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let sigma = pf(params, "sigma", 0.01);
    let smoothing = pf(params, "smoothing", 0.995);
    let min_freq = pf(params, "min_freq", 400.0);
    let max_freq = pf(params, "max_freq", 6000.0);

    let two_pi: f32 = std::f32::consts::TAU;
    let min_coeff = (-two_pi * min_freq / sr as f32).exp();
    let max_coeff = (-two_pi * max_freq / sr as f32).exp();

    // For one-pole y = a*y + (1-a)*x, higher a = more filtering (lower cutoff)
    // So min_freq -> max_coeff, max_freq -> min_coeff. Swap for walk mapping.
    let coeff_low = max_coeff;  // corresponds to higher freq (less filtering)
    let coeff_high = min_coeff; // corresponds to lower freq (more filtering)

    let n = samples.len();
    let mut out = vec![0.0f32; n];

    let mut lp_state: f32 = 0.0;
    let mut walk_val: f32 = 0.5;

    // Simple LCG for deterministic random
    let mut rng_state: i64 = 12345;

    for i in 0..n {
        rng_state = (rng_state.wrapping_mul(1103515245).wrapping_add(12345)) & 0x7FFFFFFF;
        let rand_val = rng_state as f32 / 0x7FFFFFFF_u32 as f32 * 2.0 - 1.0;

        let step = sigma * rand_val;
        walk_val = smoothing * walk_val + (1.0 - smoothing) * (walk_val + step);

        walk_val = walk_val.clamp(0.0, 1.0);

        let coeff = coeff_low + walk_val * (coeff_high - coeff_low);

        lp_state = coeff * lp_state + (1.0 - coeff) * samples[i];
        out[i] = lp_state;
    }

    AudioOutput::Mono(out)
}

fn variants_j014() -> Vec<HashMap<String, Value>> {
    vec![
        params!("sigma" => 0.005, "smoothing" => 0.999, "min_freq" => 300, "max_freq" => 5000),
        params!("sigma" => 0.01, "smoothing" => 0.995, "min_freq" => 400, "max_freq" => 8000),
        params!("sigma" => 0.03, "smoothing" => 0.99, "min_freq" => 200, "max_freq" => 10000),
        params!("sigma" => 0.05, "smoothing" => 0.9, "min_freq" => 500, "max_freq" => 4000),
        params!("sigma" => 0.001, "smoothing" => 0.999, "min_freq" => 1000, "max_freq" => 3000),
        params!("sigma" => 0.02, "smoothing" => 0.98, "min_freq" => 600, "max_freq" => 7000),
    ]
}

// ---------------------------------------------------------------------------
// J015 -- Strange Attractor Spectral Curve
// ---------------------------------------------------------------------------

fn process_j015(samples: &[f32], _sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let trajectory_width_bins = pi(params, "trajectory_width_bins", 5) as usize;
    let boost_db = pf(params, "boost_db", 10.0);

    let fft_size = 2048;
    let hop_size = 512;

    let mut frames = stft(samples, fft_size, hop_size);
    let num_frames = frames.len();
    if num_frames == 0 {
        return AudioOutput::Mono(samples.to_vec());
    }
    let num_bins = frames[0].len();

    // Generate Lorenz trajectory
    let sigma: f64 = 10.0;
    let rho: f64 = 28.0;
    let beta: f64 = 8.0 / 3.0;
    let mut lx: f64 = 1.0;
    let mut ly: f64 = 1.0;
    let mut lz: f64 = 1.0;
    let dt_lorenz: f64 = 0.005;

    let boost_linear = 10.0f32.powf(boost_db / 20.0);

    for frame in 0..num_frames {
        // Integrate Lorenz for several steps per frame
        for _ in 0..10 {
            let dx = sigma * (ly - lx);
            let dy = lx * (rho - lz) - ly;
            let dz = lx * ly - beta * lz;
            lx += dx * dt_lorenz;
            ly += dy * dt_lorenz;
            lz += dz * dt_lorenz;
        }

        // Map x-coordinate to bin index (x typically [-20, 20])
        let norm_x = ((lx + 20.0) / 40.0).clamp(0.0, 1.0);
        let center_bin = (norm_x * (num_bins - 1) as f64) as usize;

        // Apply boost around center_bin
        let start_b = if center_bin >= trajectory_width_bins {
            center_bin - trajectory_width_bins
        } else {
            0
        };
        let end_b = (center_bin + trajectory_width_bins + 1).min(num_bins);

        for b in start_b..end_b {
            let dist = if b >= center_bin { b - center_bin } else { center_bin - b };
            let half_width = (trajectory_width_bins as f32 * 0.5).max(1.0);
            let weight = (-0.5 * (dist as f32 / half_width).powi(2)).exp();
            let gain = 1.0 + (boost_linear - 1.0) * weight;
            frames[frame][b] *= gain;
        }
    }

    let out = istft(&frames, fft_size, hop_size, Some(samples.len()));
    AudioOutput::Mono(out)
}

fn variants_j015() -> Vec<HashMap<String, Value>> {
    vec![
        params!("trajectory_width_bins" => 3, "boost_db" => 6),
        params!("trajectory_width_bins" => 5, "boost_db" => 10),
        params!("trajectory_width_bins" => 8, "boost_db" => 15),
        params!("trajectory_width_bins" => 2, "boost_db" => 20),
        params!("trajectory_width_bins" => 10, "boost_db" => 5),
        params!("trajectory_width_bins" => 6, "boost_db" => 12),
    ]
}

// ---------------------------------------------------------------------------
// J016 -- Mobius Transform on Spectrum
// ---------------------------------------------------------------------------

fn process_j016(samples: &[f32], _sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let a_real = pf(params, "a_real", 1.0);
    let b_real = pf(params, "b_real", 0.0);
    let c_real = pf(params, "c_real", 0.1);
    let d_real = pf(params, "d_real", 1.0);

    let fft_size = 2048;
    let hop_size = 512;

    let mut frames = stft(samples, fft_size, hop_size);
    let num_frames = frames.len();
    if num_frames == 0 {
        return AudioOutput::Mono(samples.to_vec());
    }
    let num_bins = frames[0].len();

    let a = Complex::new(a_real, 0.0f32);
    let b = Complex::new(b_real, 0.0f32);
    let c = Complex::new(c_real, 0.0f32);
    let d = Complex::new(d_real, 0.0f32);

    for frame in 0..num_frames {
        for bin_idx in 0..num_bins {
            let z = frames[frame][bin_idx];
            let mut denom = c * z + d;
            if denom.norm() < 1e-10 {
                denom = Complex::new(1e-10, 0.0);
            }
            let w = (a * z + b) / denom;
            frames[frame][bin_idx] = w;
        }
    }

    let out = istft(&frames, fft_size, hop_size, Some(samples.len()));
    AudioOutput::Mono(out)
}

fn variants_j016() -> Vec<HashMap<String, Value>> {
    vec![
        params!("a_real" => 1.0, "b_real" => 0.0, "c_real" => 0.1, "d_real" => 1.0),
        params!("a_real" => 1.5, "b_real" => 0.5, "c_real" => 0.2, "d_real" => 1.5),
        params!("a_real" => 2.0, "b_real" => -1.0, "c_real" => 0.3, "d_real" => 0.5),
        params!("a_real" => 0.5, "b_real" => 1.0, "c_real" => 0.0, "d_real" => 2.0),
        params!("a_real" => 1.0, "b_real" => -0.5, "c_real" => 0.5, "d_real" => 1.0),
        params!("a_real" => 1.8, "b_real" => 0.3, "c_real" => 0.05, "d_real" => 1.2),
    ]
}

// ---------------------------------------------------------------------------
// J017 -- Fractal Delay Network
// ---------------------------------------------------------------------------

fn process_j017(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let base_delay_ms = pf(params, "base_delay_ms", 100.0);
    let ratio = pf(params, "ratio", 2.0);
    let num_levels = pi(params, "num_levels", 5).clamp(2, 7) as usize;
    let feedback = pf(params, "feedback", 0.5);

    let n = samples.len();

    let mut delays = vec![0usize; num_levels];
    let mut gains = vec![0.0f32; num_levels];

    for lev in 0..num_levels {
        let delay_ms = base_delay_ms * ratio.powi(lev as i32);
        delays[lev] = ((delay_ms * sr as f32 / 1000.0) as usize).max(1);
        gains[lev] = 1.0 / (lev as f32 + 1.0);
    }

    let max_delay = *delays.iter().max().unwrap_or(&1);
    let buf_len = (max_delay + 1).max(1);
    let mut buf = vec![0.0f32; buf_len];
    let mut out = vec![0.0f32; n];
    let mut write_pos: usize = 0;

    for i in 0..n {
        let mut fb_sum: f32 = 0.0;
        for lev in 0..num_levels {
            let read_pos = (write_pos + buf_len - delays[lev]) % buf_len;
            fb_sum += gains[lev] * buf[read_pos];
        }

        let mut y = samples[i] + feedback * fb_sum;
        y = y.clamp(-2.0, 2.0);

        buf[write_pos] = y;
        out[i] = y;
        write_pos = (write_pos + 1) % buf_len;
    }

    AudioOutput::Mono(out)
}

fn variants_j017() -> Vec<HashMap<String, Value>> {
    vec![
        params!("base_delay_ms" => 50, "ratio" => 2.0, "num_levels" => 4, "feedback" => 0.4),
        params!("base_delay_ms" => 100, "ratio" => 2.0, "num_levels" => 5, "feedback" => 0.5),
        params!("base_delay_ms" => 200, "ratio" => 1.5, "num_levels" => 6, "feedback" => 0.6),
        params!("base_delay_ms" => 500, "ratio" => 3.0, "num_levels" => 3, "feedback" => 0.7),
        params!("base_delay_ms" => 80, "ratio" => 2.5, "num_levels" => 5, "feedback" => 0.3),
        params!("base_delay_ms" => 150, "ratio" => 1.618, "num_levels" => 7, "feedback" => 0.8),
    ]
}

// ---------------------------------------------------------------------------
// J018 -- Audio Boids
// ---------------------------------------------------------------------------

fn process_j018(samples: &[f32], _sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let num_boids = pi(params, "num_boids", 10).max(1) as usize;
    let speed = pf(params, "speed", 2.0) as f64;

    let fft_size = 2048;
    let hop_size = 512;

    let frames = stft(samples, fft_size, hop_size);
    let num_frames = frames.len();
    if num_frames == 0 {
        return AudioOutput::Mono(samples.to_vec());
    }
    let num_bins = frames[0].len();

    // Initialize boid positions deterministically
    let mut rng = Lcg::new(42);
    let mut positions = vec![0.0f64; num_boids];
    let mut velocities = vec![0.0f64; num_boids];
    for b in 0..num_boids {
        positions[b] = rng.next_f32() as f64 * (num_bins - 1) as f64;
        velocities[b] = (rng.next_f32() as f64 * 2.0 - 1.0) * speed;
    }

    let mut out_frames = frames.clone();

    for frame in 0..num_frames {
        let mag: Vec<f32> = frames[frame].iter().map(|c| c.norm()).collect();

        // Boid flocking rules
        for b in 0..num_boids {
            // Cohesion: move toward center of flock
            let center: f64 = positions.iter().sum::<f64>() / num_boids as f64;
            let cohesion_force = (center - positions[b]) * 0.01;

            // Separation: avoid nearby boids
            let mut sep_force: f64 = 0.0;
            for other in 0..num_boids {
                if other == b {
                    continue;
                }
                let diff = positions[b] - positions[other];
                let dist = diff.abs();
                if dist < 5.0 && dist > 0.01 {
                    sep_force += diff / (dist * dist);
                }
            }
            sep_force *= 0.5;

            // Alignment: match average velocity
            let avg_vel: f64 = velocities.iter().sum::<f64>() / num_boids as f64;
            let align_force = (avg_vel - velocities[b]) * 0.05;

            // Attraction to spectral energy
            let bin_idx = (positions[b] as usize) % num_bins;
            let mut attract_force: f64 = 0.0;
            for offset in -10i32..=10 {
                let check_bin = ((bin_idx as i32 + offset).rem_euclid(num_bins as i32)) as usize;
                if mag[check_bin] > mag[bin_idx] {
                    attract_force += offset as f64 * mag[check_bin] as f64 * 0.001;
                }
            }

            velocities[b] += cohesion_force + sep_force + align_force + attract_force;
            velocities[b] = velocities[b].clamp(-speed, speed);

            positions[b] += velocities[b];
            // Wrap around
            positions[b] = positions[b].rem_euclid(num_bins as f64);
        }

        // Apply boid positions: each boid boosts spectral energy at its position
        let mut boost_mask = vec![1.0f64; num_bins];
        for b in 0..num_boids {
            let center = (positions[b] as usize) % num_bins;
            for offset in -3i32..=3 {
                let idx = ((center as i32 + offset).rem_euclid(num_bins as i32)) as usize;
                let dist_factor = 1.0 - offset.unsigned_abs() as f64 / 4.0;
                boost_mask[idx] += dist_factor * 0.5;
            }
        }

        for b_idx in 0..num_bins {
            out_frames[frame][b_idx] = frames[frame][b_idx] * boost_mask[b_idx] as f32;
        }
    }

    let out = istft(&out_frames, fft_size, hop_size, Some(samples.len()));
    AudioOutput::Mono(out)
}

fn variants_j018() -> Vec<HashMap<String, Value>> {
    vec![
        params!("num_boids" => 5, "speed" => 1.0),
        params!("num_boids" => 10, "speed" => 2.0),
        params!("num_boids" => 20, "speed" => 3.0),
        params!("num_boids" => 30, "speed" => 1.5),
        params!("num_boids" => 8, "speed" => 5.0),
        params!("num_boids" => 15, "speed" => 0.5),
    ]
}

// ---------------------------------------------------------------------------
// J019 -- Stochastic Resonance
// ---------------------------------------------------------------------------

fn process_j019(samples: &[f32], _sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let noise_amplitude = pf(params, "noise_amplitude", 0.1);
    let threshold = pf(params, "threshold", 0.15);

    let n = samples.len();
    let mut out = vec![0.0f32; n];

    let mut rng_state: i64 = 54321;

    for i in 0..n {
        rng_state = (rng_state.wrapping_mul(1103515245).wrapping_add(12345)) & 0x7FFFFFFF;
        let noise = (rng_state as f32 / 0x7FFFFFFF_u32 as f32 * 2.0 - 1.0) * noise_amplitude;

        let noisy = samples[i] + noise;

        if noisy.abs() > threshold {
            out[i] = noisy;
        } else {
            out[i] = noisy * 0.3;
        }
    }

    AudioOutput::Mono(out)
}

fn variants_j019() -> Vec<HashMap<String, Value>> {
    vec![
        params!("noise_amplitude" => 0.02, "threshold" => 0.05),
        params!("noise_amplitude" => 0.05, "threshold" => 0.1),
        params!("noise_amplitude" => 0.1, "threshold" => 0.15),
        params!("noise_amplitude" => 0.2, "threshold" => 0.2),
        params!("noise_amplitude" => 0.3, "threshold" => 0.25),
        params!("noise_amplitude" => 0.5, "threshold" => 0.3),
    ]
}

// ---------------------------------------------------------------------------
// J020 -- Chua's Circuit
// ---------------------------------------------------------------------------

fn process_j020(samples: &[f32], _sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let alpha = pf(params, "alpha", 10.0) as f64;
    let beta = pf(params, "beta", 14.87) as f64;
    let drive_amount = pf(params, "drive_amount", 0.5);

    let n = samples.len();
    let mut out = vec![0.0f32; n];

    let mut x: f64 = 0.1;
    let mut y: f64 = 0.0;
    let mut z: f64 = 0.0;

    // Chua diode parameters (piecewise-linear)
    let m0: f64 = -1.143;
    let m1: f64 = -0.714;
    let bp: f64 = 1.0;

    let dt: f64 = 0.01;

    for i in 0..n {
        // Chua diode function h(x) = m1*x + 0.5*(m0-m1)*(|x+bp| - |x-bp|)
        let h = m1 * x + 0.5 * (m0 - m1) * ((x + bp).abs() - (x - bp).abs());

        let dx = alpha * (y - x - h) + samples[i] as f64 * drive_amount as f64 * 0.5;
        let dy = x - y + z;
        let dz = -beta * y;

        x += dx * dt;
        y += dy * dt;
        z += dz * dt;

        x = x.clamp(-5.0, 5.0);
        y = y.clamp(-5.0, 5.0);
        z = z.clamp(-50.0, 50.0);

        let chua_norm = (x * 0.3).tanh() as f32;
        out[i] = samples[i] * (1.0 - drive_amount) + chua_norm * drive_amount;
    }

    AudioOutput::Mono(out)
}

fn variants_j020() -> Vec<HashMap<String, Value>> {
    vec![
        params!("alpha" => 9.0, "beta" => 14.0, "drive_amount" => 0.3),
        params!("alpha" => 10.0, "beta" => 14.87, "drive_amount" => 0.5),
        params!("alpha" => 12.0, "beta" => 14.5, "drive_amount" => 0.6),
        params!("alpha" => 15.0, "beta" => 15.0, "drive_amount" => 0.4),
        params!("alpha" => 16.0, "beta" => 14.2, "drive_amount" => 0.8),
        params!("alpha" => 11.0, "beta" => 14.87, "drive_amount" => 0.2),
    ]
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

pub fn register() -> Vec<EffectEntry> {
    vec![
        EffectEntry {
            id: "J001",
            process: process_j001,
            variants: variants_j001,
            category: "chaos_math",
        },
        EffectEntry {
            id: "J002",
            process: process_j002,
            variants: variants_j002,
            category: "chaos_math",
        },
        EffectEntry {
            id: "J003",
            process: process_j003,
            variants: variants_j003,
            category: "chaos_math",
        },
        EffectEntry {
            id: "J004",
            process: process_j004,
            variants: variants_j004,
            category: "chaos_math",
        },
        EffectEntry {
            id: "J005",
            process: process_j005,
            variants: variants_j005,
            category: "chaos_math",
        },
        EffectEntry {
            id: "J006",
            process: process_j006,
            variants: variants_j006,
            category: "chaos_math",
        },
        EffectEntry {
            id: "J007",
            process: process_j007,
            variants: variants_j007,
            category: "chaos_math",
        },
        EffectEntry {
            id: "J008",
            process: process_j008,
            variants: variants_j008,
            category: "chaos_math",
        },
        EffectEntry {
            id: "J009",
            process: process_j009,
            variants: variants_j009,
            category: "chaos_math",
        },
        EffectEntry {
            id: "J010",
            process: process_j010,
            variants: variants_j010,
            category: "chaos_math",
        },
        EffectEntry {
            id: "J011",
            process: process_j011,
            variants: variants_j011,
            category: "chaos_math",
        },
        EffectEntry {
            id: "J012",
            process: process_j012,
            variants: variants_j012,
            category: "chaos_math",
        },
        EffectEntry {
            id: "J013",
            process: process_j013,
            variants: variants_j013,
            category: "chaos_math",
        },
        EffectEntry {
            id: "J014",
            process: process_j014,
            variants: variants_j014,
            category: "chaos_math",
        },
        EffectEntry {
            id: "J015",
            process: process_j015,
            variants: variants_j015,
            category: "chaos_math",
        },
        EffectEntry {
            id: "J016",
            process: process_j016,
            variants: variants_j016,
            category: "chaos_math",
        },
        EffectEntry {
            id: "J017",
            process: process_j017,
            variants: variants_j017,
            category: "chaos_math",
        },
        EffectEntry {
            id: "J018",
            process: process_j018,
            variants: variants_j018,
            category: "chaos_math",
        },
        EffectEntry {
            id: "J019",
            process: process_j019,
            variants: variants_j019,
            category: "chaos_math",
        },
        EffectEntry {
            id: "J020",
            process: process_j020,
            variants: variants_j020,
            category: "chaos_math",
        },
    ]
}
