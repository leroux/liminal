//! B-series effects: Reverb algorithms (B001-B011).

use std::collections::HashMap;
use serde_json::Value;
use crate::{AudioOutput, EffectEntry, pf, pi, params};
use crate::primitives::*;

// ---------------------------------------------------------------------------
// B001 — Schroeder Reverb
// 4 parallel comb filters -> 2 series allpass filters
// ---------------------------------------------------------------------------

fn process_b001(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let rt60 = pf(params, "rt60", 1.5);
    let wet_mix = pf(params, "wet_mix", 0.5);
    let n = samples.len();

    // Comb filter delays in seconds -> samples
    let comb_delays_ms: [f64; 4] = [29.7, 37.1, 41.1, 43.7];
    let mut comb_delays = [0usize; 4];
    let mut comb_g = [0.0f32; 4];
    for i in 0..4 {
        let d = (comb_delays_ms[i] * 0.001 * sr as f64) as usize;
        let d = d.max(1);
        comb_delays[i] = d;
        comb_g[i] = 10.0f32.powf(-3.0 * d as f32 / (sr as f32 * rt60));
    }

    // Allpass delays
    let ap_delays_ms: [f64; 2] = [5.0, 1.7];
    let mut ap_delays = [0usize; 2];
    for i in 0..2 {
        let d = (ap_delays_ms[i] * 0.001 * sr as f64) as usize;
        ap_delays[i] = d.max(1);
    }
    let ap_g: [f32; 2] = [0.7, 0.7];

    // Allocate comb buffers
    let max_comb = *comb_delays.iter().max().unwrap();
    let mut comb_bufs = vec![vec![0.0f32; max_comb + 1]; 4];
    let mut comb_idx = [0usize; 4];

    // Allocate allpass buffers
    let max_ap = *ap_delays.iter().max().unwrap();
    let mut ap_bufs = vec![vec![0.0f32; max_ap + 1]; 2];
    let mut ap_idx = [0usize; 2];

    let mut out = vec![0.0f32; n];

    for i in 0..n {
        let x = samples[i];

        // 4 parallel comb filters summed
        let mut comb_sum = 0.0f32;
        for c in 0..4 {
            let dl = comb_delays[c];
            let read_pos = (comb_idx[c] + dl - dl) % dl; // (idx - dl) % dl
            // Match Python: read_pos = (comb_idx[c] - dl) % dl
            // In Python with unsigned, (comb_idx[c] wrapping sub dl) mod dl
            let read_pos = if comb_idx[c] >= dl {
                (comb_idx[c] - dl) % dl
            } else {
                dl - ((dl - comb_idx[c]) % dl)
            };
            let read_pos = read_pos % dl;
            let delayed = comb_bufs[c][read_pos];
            let val = x + comb_g[c] * delayed;
            comb_bufs[c][comb_idx[c] % dl] = val;
            comb_idx[c] += 1;
            comb_sum += delayed;
        }

        // Scale comb output
        let mut y = comb_sum * 0.25;

        // 2 series allpass filters
        for a in 0..2 {
            let dl = ap_delays[a];
            let read_pos = if ap_idx[a] >= dl {
                (ap_idx[a] - dl) % dl
            } else {
                dl - ((dl - ap_idx[a]) % dl)
            };
            let read_pos = read_pos % dl;
            let delayed = ap_bufs[a][read_pos];
            let g = ap_g[a];
            let v = y - g * delayed;
            ap_bufs[a][ap_idx[a] % dl] = v;
            ap_idx[a] += 1;
            y = delayed + g * v;
        }

        out[i] = (1.0 - wet_mix) * x + wet_mix * y;
    }

    let out = post_process(&out, sr);
    AudioOutput::Mono(out)
}

fn variants_b001() -> Vec<HashMap<String, Value>> {
    vec![
        params!{"rt60" => 0.5, "wet_mix" => 0.3},
        params!{"rt60" => 1.0, "wet_mix" => 0.5},
        params!{"rt60" => 2.0, "wet_mix" => 0.5},
        params!{"rt60" => 3.5, "wet_mix" => 0.6},
        params!{"rt60" => 5.0, "wet_mix" => 0.8},
    ]
}

// ---------------------------------------------------------------------------
// B002 — Moorer Reverb
// 6 comb filters with one-pole lowpass in feedback -> allpass chain
// ---------------------------------------------------------------------------

fn process_b002(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let rt60 = pf(params, "rt60", 2.0);
    let damping = pf(params, "damping", 0.5);
    let wet_mix = pf(params, "wet_mix", 0.5);
    let n = samples.len();

    let comb_delays_ms: [f64; 6] = [50.0, 56.0, 61.0, 68.0, 72.0, 78.0];
    let num_combs = 6;
    let mut comb_delays = [0usize; 6];
    let mut comb_g = [0.0f32; 6];
    for i in 0..num_combs {
        let d = (comb_delays_ms[i] * 0.001 * sr as f64) as usize;
        let d = d.max(1);
        comb_delays[i] = d;
        comb_g[i] = 10.0f32.powf(-3.0 * d as f32 / (sr as f32 * rt60));
    }

    let max_comb = *comb_delays.iter().max().unwrap();
    let buf_size = max_comb + 1;
    let mut comb_bufs = vec![vec![0.0f32; buf_size]; num_combs];
    let mut comb_idx = [0usize; 6];
    let mut comb_lp_state = [0.0f32; 6];

    let lp_coeff = damping;

    // 1 allpass for diffusion after combs
    let ap_delay = ((0.005 * sr as f64) as usize).max(1);
    let mut ap_buf = vec![0.0f32; ap_delay];
    let mut ap_idx = 0usize;
    let ap_g = 0.7f32;

    let mut out = vec![0.0f32; n];

    for i in 0..n {
        let x = samples[i];
        let mut comb_sum = 0.0f32;

        for c in 0..num_combs {
            let dl = comb_delays[c];
            let read_pos = (comb_idx[c] + buf_size - dl) % buf_size;
            let delayed = comb_bufs[c][read_pos];

            // One-pole lowpass in feedback
            comb_lp_state[c] = lp_coeff * comb_lp_state[c] + (1.0 - lp_coeff) * delayed;
            let filtered = comb_lp_state[c];

            let val = x + comb_g[c] * filtered;
            let write_pos = comb_idx[c] % buf_size;
            comb_bufs[c][write_pos] = val;
            comb_idx[c] += 1;
            comb_sum += delayed;
        }

        let mut y = comb_sum / num_combs as f32;

        // Allpass
        let read_pos = ap_idx % ap_delay;
        let delayed_ap = ap_buf[read_pos];
        let v = y - ap_g * delayed_ap;
        ap_buf[ap_idx % ap_delay] = v;
        ap_idx += 1;
        y = delayed_ap + ap_g * v;

        out[i] = (1.0 - wet_mix) * x + wet_mix * y;
    }

    let out = post_process(&out, sr);
    AudioOutput::Mono(out)
}

fn variants_b002() -> Vec<HashMap<String, Value>> {
    vec![
        params!{"rt60" => 0.5, "damping" => 0.3, "wet_mix" => 0.3},
        params!{"rt60" => 1.5, "damping" => 0.5, "wet_mix" => 0.5},
        params!{"rt60" => 3.0, "damping" => 0.7, "wet_mix" => 0.5},
        params!{"rt60" => 5.0, "damping" => 0.9, "wet_mix" => 0.6},
        params!{"rt60" => 8.0, "damping" => 0.4, "wet_mix" => 0.8},
    ]
}

// ---------------------------------------------------------------------------
// B003 — FDN Reverb
// N delay lines with NxN Hadamard feedback matrix, lowpass in feedback
// ---------------------------------------------------------------------------

fn hadamard4() -> [[f32; 4]; 4] {
    let h = [
        [1.0, 1.0, 1.0, 1.0],
        [1.0, -1.0, 1.0, -1.0],
        [1.0, 1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0, 1.0],
    ];
    let mut out = [[0.0f32; 4]; 4];
    for r in 0..4 {
        for c in 0..4 {
            out[r][c] = h[r][c] * 0.5;
        }
    }
    out
}

fn hadamard8() -> [[f32; 8]; 8] {
    let h4 = [
        [1.0f32, 1.0, 1.0, 1.0],
        [1.0, -1.0, 1.0, -1.0],
        [1.0, 1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0, 1.0],
    ];
    let mut h8 = [[0.0f32; 8]; 8];
    for i in 0..4 {
        for j in 0..4 {
            h8[i][j] = h4[i][j];
            h8[i][j + 4] = h4[i][j];
            h8[i + 4][j] = h4[i][j];
            h8[i + 4][j + 4] = -h4[i][j];
        }
    }
    let scale = 1.0 / (8.0f32).sqrt();
    for i in 0..8 {
        for j in 0..8 {
            h8[i][j] *= scale;
        }
    }
    h8
}

fn process_b003(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let n_delays_param = pi(params, "n_delays", 4) as usize;
    let rt60 = pf(params, "rt60", 2.0);
    let damping = pf(params, "damping", 0.3);
    let wet_mix = pf(params, "wet_mix", 0.5);
    let n = samples.len();

    let delay_ms_8: [f64; 8] = [29.7, 37.1, 41.1, 43.7, 53.0, 59.9, 67.3, 73.1];
    let delay_ms_4: [f64; 4] = [29.7, 37.1, 41.1, 43.7];

    let n_delays = if n_delays_param == 8 { 8 } else { 4 };

    // Build delay and gain arrays (up to 8)
    let mut delays = [0usize; 8];
    let mut gains = [0.0f32; 8];
    for i in 0..n_delays {
        let ms = if n_delays == 8 { delay_ms_8[i] } else { delay_ms_4[i] };
        let d = (ms * 0.001 * sr as f64) as usize;
        let d = d.max(1);
        delays[i] = d;
        gains[i] = 10.0f32.powf(-3.0 * d as f32 / (sr as f32 * rt60));
    }

    let max_delay = delays[..n_delays].iter().copied().max().unwrap();
    let buf_size = max_delay + 1;
    let mut bufs = vec![vec![0.0f32; buf_size]; n_delays];
    let mut write_idx = vec![0usize; n_delays];
    let mut lp_state = vec![0.0f32; n_delays];
    let lp_coeff = damping;

    // Get matrix values -- use flat Vec for dynamic size
    let mat8 = hadamard8();
    let mat4 = hadamard4();

    let mut out = vec![0.0f32; n];

    for i in 0..n {
        let x = samples[i];

        // Read from delay lines
        let mut delayed = [0.0f32; 8];
        for d in 0..n_delays {
            let dl = delays[d];
            let read_pos = (write_idx[d] + buf_size - dl) % buf_size;
            delayed[d] = bufs[d][read_pos];
        }

        // Apply feedback matrix
        let mut feedback = [0.0f32; 8];
        for r in 0..n_delays {
            let mut s = 0.0f32;
            for c in 0..n_delays {
                let mat_val = if n_delays == 8 { mat8[r][c] } else { mat4[r][c] };
                s += mat_val * delayed[c];
            }
            feedback[r] = s;
        }

        // Apply gain and lowpass, write back
        let mut out_sum = 0.0f32;
        for d in 0..n_delays {
            lp_state[d] = lp_coeff * lp_state[d] + (1.0 - lp_coeff) * feedback[d];
            let val = x + gains[d] * lp_state[d];
            let write_pos = write_idx[d] % buf_size;
            bufs[d][write_pos] = val;
            write_idx[d] += 1;
            out_sum += delayed[d];
        }

        let y = out_sum / n_delays as f32;
        out[i] = (1.0 - wet_mix) * x + wet_mix * y;
    }

    let out = post_process(&out, sr);
    AudioOutput::Mono(out)
}

fn variants_b003() -> Vec<HashMap<String, Value>> {
    vec![
        params!{"n_delays" => 4, "rt60" => 1.0, "damping" => 0.2, "wet_mix" => 0.4},
        params!{"n_delays" => 4, "rt60" => 3.0, "damping" => 0.5, "wet_mix" => 0.5},
        params!{"n_delays" => 8, "rt60" => 2.0, "damping" => 0.3, "wet_mix" => 0.5},
        params!{"n_delays" => 8, "rt60" => 5.0, "damping" => 0.6, "wet_mix" => 0.6},
        params!{"n_delays" => 8, "rt60" => 10.0, "damping" => 0.8, "wet_mix" => 0.7},
        params!{"n_delays" => 8, "rt60" => 20.0, "damping" => 0.9, "wet_mix" => 0.8},
    ]
}

// ---------------------------------------------------------------------------
// B004 — Plate Reverb
// Nested allpass + delay sections with modulation
// ---------------------------------------------------------------------------

fn process_b004(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let decay = pf(params, "decay", 2.0);
    let damping = pf(params, "damping", 0.4);
    let mod_rate = pf(params, "mod_rate", 1.0);
    let mod_depth = pf(params, "mod_depth", 4.0);
    let pre_delay_ms = pf(params, "pre_delay_ms", 20.0);
    let n = samples.len();

    // Pre-delay
    let pre_delay_samps = ((pre_delay_ms * 0.001 * sr as f32) as usize).max(1);
    let mut pre_buf = vec![0.0f32; pre_delay_samps];
    let mut pre_idx = 0usize;

    // Allpass delays for diffusion (4 allpass filters in series)
    let ap_delays_ms: [f64; 4] = [4.77, 3.60, 12.73, 9.31];
    let num_ap = 4;
    let mut ap_delays = [0usize; 4];
    for i in 0..num_ap {
        ap_delays[i] = ((ap_delays_ms[i] * 0.001 * sr as f64) as usize).max(1);
    }
    let max_ap = *ap_delays.iter().max().unwrap();
    let ap_buf_size = max_ap + 1;
    let mut ap_bufs = vec![vec![0.0f32; ap_buf_size]; num_ap];
    let mut ap_idx = [0usize; 4];
    let ap_g = 0.6f32;

    // Two delay lines for the tank
    let tank_delay_ms: [f64; 2] = [30.51, 22.58];
    let num_tank = 2;
    let mut tank_delays = [0usize; 2];
    for i in 0..num_tank {
        let d = (tank_delay_ms[i] * 0.001 * sr as f64) as usize + mod_depth as usize + 2;
        tank_delays[i] = d.max(2);
    }
    let max_tank = *tank_delays.iter().max().unwrap();
    let tank_buf_size = max_tank + 1;
    let mut tank_bufs = vec![vec![0.0f32; tank_buf_size]; num_tank];
    let mut tank_idx = [0usize; 2];

    // Decay gain
    let decay_g = 10.0f32.powf(-3.0 * 0.03 / decay);
    let lp_coeff = damping;
    let mut lp_state = [0.0f32; 2];

    // Two tank allpasses
    let tank_ap_delays_ms: [f64; 2] = [8.93, 6.28];
    let mut tank_ap_delays = [0usize; 2];
    for i in 0..num_tank {
        tank_ap_delays[i] = ((tank_ap_delays_ms[i] * 0.001 * sr as f64) as usize).max(1);
    }
    let max_tap = *tank_ap_delays.iter().max().unwrap();
    let tap_buf_size = max_tap + 1;
    let mut tank_ap_bufs = vec![vec![0.0f32; tap_buf_size]; num_tank];
    let mut tank_ap_idx = [0usize; 2];
    let tank_ap_g = 0.5f32;

    let mut out = vec![0.0f32; n];

    // Mod phase
    let mut mod_phase = 0.0f32;
    let mod_inc = 2.0 * std::f32::consts::PI * mod_rate / sr as f32;

    // Tank feedback state
    let mut tank_fb = [0.0f32; 2];

    for i in 0..n {
        let x = samples[i];

        // Pre-delay
        let pre_out = pre_buf[pre_idx % pre_delay_samps];
        pre_buf[pre_idx % pre_delay_samps] = x;
        pre_idx += 1;

        // Input diffusion: 4 series allpass
        let mut y = pre_out;
        for a in 0..num_ap {
            let dl = ap_delays[a];
            let rd = (ap_idx[a] + ap_buf_size - dl) % ap_buf_size;
            let delayed = ap_bufs[a][rd];
            let v = y - ap_g * delayed;
            ap_bufs[a][ap_idx[a] % ap_buf_size] = v;
            ap_idx[a] += 1;
            y = delayed + ap_g * v;
        }

        // Feed into tank with cross-feedback
        let tank_in_0 = y + decay_g * tank_fb[1];
        let tank_in_1 = y + decay_g * tank_fb[0];

        let mut out_sum = 0.0f32;
        for t in 0..num_tank {
            let tank_in = if t == 0 { tank_in_0 } else { tank_in_1 };

            // Tank allpass
            let dl_tap = tank_ap_delays[t];
            let rd = (tank_ap_idx[t] + tap_buf_size - dl_tap) % tap_buf_size;
            let delayed = tank_ap_bufs[t][rd];
            let v = tank_in - tank_ap_g * delayed;
            tank_ap_bufs[t][tank_ap_idx[t] % tap_buf_size] = v;
            tank_ap_idx[t] += 1;
            let ap_out = delayed + tank_ap_g * v;

            // Modulated delay
            let base_dl = (tank_delay_ms[t] * 0.001 * sr as f64) as i32;
            let mod_offset = mod_depth * (mod_phase + t as f32 * std::f32::consts::FRAC_PI_2).sin();
            let mut mod_dl = base_dl + mod_offset as i32;
            if mod_dl < 1 { mod_dl = 1; }
            if mod_dl >= tank_delays[t] as i32 { mod_dl = tank_delays[t] as i32 - 1; }
            let mod_dl = mod_dl as usize;

            let rd = (tank_idx[t] + tank_buf_size - mod_dl) % tank_buf_size;
            let delayed_tank = tank_bufs[t][rd];

            // Lowpass in feedback
            lp_state[t] = lp_coeff * lp_state[t] + (1.0 - lp_coeff) * delayed_tank;
            tank_fb[t] = lp_state[t];

            tank_bufs[t][tank_idx[t] % tank_buf_size] = ap_out;
            tank_idx[t] += 1;

            out_sum += delayed_tank;
        }

        mod_phase += mod_inc;
        if mod_phase > 2.0 * std::f32::consts::PI {
            mod_phase -= 2.0 * std::f32::consts::PI;
        }

        let wet = out_sum * 0.5;
        out[i] = 0.5 * x + 0.5 * wet;
    }

    let out = post_process(&out, sr);
    AudioOutput::Mono(out)
}

fn variants_b004() -> Vec<HashMap<String, Value>> {
    vec![
        params!{"decay" => 0.5, "damping" => 0.3, "mod_rate" => 0.5, "mod_depth" => 2.0, "pre_delay_ms" => 0.0},
        params!{"decay" => 1.5, "damping" => 0.4, "mod_rate" => 1.0, "mod_depth" => 4.0, "pre_delay_ms" => 10.0},
        params!{"decay" => 3.0, "damping" => 0.5, "mod_rate" => 1.5, "mod_depth" => 6.0, "pre_delay_ms" => 30.0},
        params!{"decay" => 5.0, "damping" => 0.6, "mod_rate" => 0.8, "mod_depth" => 3.0, "pre_delay_ms" => 50.0},
        params!{"decay" => 8.0, "damping" => 0.8, "mod_rate" => 2.0, "mod_depth" => 8.0, "pre_delay_ms" => 80.0},
        params!{"decay" => 10.0, "damping" => 0.2, "mod_rate" => 1.2, "mod_depth" => 1.0, "pre_delay_ms" => 100.0},
    ]
}

// ---------------------------------------------------------------------------
// B005 — Spring Reverb
// Allpass chain with frequency-dependent delay + nonlinearity
// ---------------------------------------------------------------------------

fn process_b005(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let num_springs = pi(params, "num_springs", 2) as usize;
    let tension = pf(params, "tension", 0.6);
    let damping = pf(params, "damping", 0.5);
    let chaos = pf(params, "chaos", 0.1);
    let n = samples.len();

    let ap_per_spring = 8usize;
    let total_ap = num_springs * ap_per_spring;

    // Compute delays based on tension
    let base_delay_ms = 3.0f32 * (1.0 - tension) + 0.5;
    let max_delay_samps = ((base_delay_ms * 2.0 * 0.001 * sr as f32) as usize + 4).max(4);

    let mut ap_delays = vec![0usize; total_ap];
    let mut ap_g_vals = vec![0.0f32; total_ap];

    for s in 0..num_springs {
        for a in 0..ap_per_spring {
            let idx = s * ap_per_spring + a;
            let frac = a as f32 / ap_per_spring as f32;
            let mut delay_ms = base_delay_ms * (1.0 + frac * 1.5);
            delay_ms += s as f32 * 0.7;
            let mut d = (delay_ms * 0.001 * sr as f32) as usize;
            d = d.max(1);
            if d >= max_delay_samps {
                d = max_delay_samps - 1;
            }
            ap_delays[idx] = d;
            ap_g_vals[idx] = 0.5 + 0.3 * (1.0 - damping) * (1.0 - frac * 0.5);
        }
    }

    let mut ap_bufs = vec![vec![0.0f32; max_delay_samps]; total_ap];
    let mut ap_idx = vec![0usize; total_ap];

    // Lowpass state per spring
    let mut lp_state = vec![0.0f32; num_springs];
    let lp_coeff = damping;

    let mut out = vec![0.0f32; n];

    for i in 0..n {
        let x = samples[i];
        let mut spring_sum = 0.0f32;

        for s in 0..num_springs {
            let mut y = x;

            // Allpass chain for this spring
            for a in 0..ap_per_spring {
                let idx = s * ap_per_spring + a;
                let dl = ap_delays[idx];
                let g = ap_g_vals[idx];
                let rd = (ap_idx[idx] + max_delay_samps - dl) % max_delay_samps;
                let delayed = ap_bufs[idx][rd];
                let v = y - g * delayed;
                ap_bufs[idx][ap_idx[idx] % max_delay_samps] = v;
                ap_idx[idx] += 1;
                y = delayed + g * v;
            }

            // Nonlinearity (soft clip for spring bounce character)
            if chaos > 0.0 {
                y = y + chaos * ((y * 3.0).tanh() - y);
            }

            // Damping lowpass
            lp_state[s] = lp_coeff * lp_state[s] + (1.0 - lp_coeff) * y;
            spring_sum += lp_state[s];
        }

        let spring_out = spring_sum / num_springs.max(1) as f32;
        out[i] = 0.5 * x + 0.5 * spring_out;
    }

    let out = post_process(&out, sr);
    AudioOutput::Mono(out)
}

fn variants_b005() -> Vec<HashMap<String, Value>> {
    vec![
        params!{"num_springs" => 1, "tension" => 0.3, "damping" => 0.5, "chaos" => 0.0},
        params!{"num_springs" => 2, "tension" => 0.6, "damping" => 0.5, "chaos" => 0.1},
        params!{"num_springs" => 3, "tension" => 0.9, "damping" => 0.3, "chaos" => 0.0},
        params!{"num_springs" => 2, "tension" => 0.4, "damping" => 0.8, "chaos" => 0.3},
        params!{"num_springs" => 3, "tension" => 0.7, "damping" => 0.6, "chaos" => 0.5},
        params!{"num_springs" => 1, "tension" => 0.5, "damping" => 0.4, "chaos" => 0.2},
    ]
}

// ---------------------------------------------------------------------------
// B006 — Shimmer Reverb
// FDN reverb with pitch-shifted feedback (+12 semitones via simple resampling)
// ---------------------------------------------------------------------------

fn process_b006(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let rt60 = pf(params, "rt60", 3.0);
    let pitch_shift_semitones = pf(params, "pitch_shift_semitones", 12.0);
    let shimmer_amount = pf(params, "shimmer_amount", 0.3);
    let wet_mix = pf(params, "wet_mix", 0.5);
    let n = samples.len();

    // Pitch shift ratio
    let ratio = 2.0f64.powf(pitch_shift_semitones as f64 / 12.0);

    // FDN with 4 delay lines
    let n_delays = 4usize;
    let delay_ms: [f64; 4] = [29.7, 37.1, 41.1, 43.7];
    let mut delays = [0usize; 4];
    let mut gains = [0.0f32; 4];
    for i in 0..n_delays {
        let d = (delay_ms[i] * 0.001 * sr as f64) as usize;
        let d = d.max(1);
        delays[i] = d;
        gains[i] = 10.0f32.powf(-3.0 * d as f32 / (sr as f32 * rt60));
    }

    let max_delay = *delays.iter().max().unwrap();
    let buf_size = max_delay + 1;
    let mut bufs = vec![vec![0.0f32; buf_size]; n_delays];
    let mut write_idx = [0usize; 4];

    // Hadamard/2 matrix
    let mut mat = [
        [1.0f32, 1.0, 1.0, 1.0],
        [1.0, -1.0, 1.0, -1.0],
        [1.0, 1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0, 1.0],
    ];
    for r in 0..4 {
        for c in 0..4 {
            mat[r][c] *= 0.5;
        }
    }

    // Shimmer pitch shift buffer
    let shimmer_buf_len = (max_delay * 4).max(1024);
    let mut shimmer_buf = vec![0.0f32; shimmer_buf_len];
    let mut shimmer_write = 0usize;
    let mut shimmer_read_pos = 0.0f64;

    let mut lp_state = [0.0f32; 4];
    let lp_coeff = 0.3f32;

    let mut out = vec![0.0f32; n];

    for i in 0..n {
        let x = samples[i];

        // Read from delay lines
        let mut delayed = [0.0f32; 4];
        for d in 0..n_delays {
            let dl = delays[d];
            let read_pos = (write_idx[d] + buf_size - dl) % buf_size;
            delayed[d] = bufs[d][read_pos];
        }

        // Sum delayed for output
        let mut out_sum = 0.0f32;
        for d in 0..n_delays {
            out_sum += delayed[d];
        }

        // Write current output sum to shimmer buffer for pitch shifting
        let wet_out = out_sum / n_delays as f32;
        shimmer_buf[shimmer_write % shimmer_buf_len] = wet_out;
        shimmer_write += 1;

        // Read pitch-shifted sample from shimmer buffer
        let rd_idx0 = (shimmer_read_pos as usize) % shimmer_buf_len;
        let rd_idx1 = (rd_idx0 + 1) % shimmer_buf_len;
        let frac = (shimmer_read_pos - (shimmer_read_pos as usize as f64)) as f32;
        let pitched = (1.0 - frac) * shimmer_buf[rd_idx0] + frac * shimmer_buf[rd_idx1];
        shimmer_read_pos += ratio;
        // Keep read pos from getting too far behind
        if shimmer_write as f64 - shimmer_read_pos > shimmer_buf_len as f64 * 0.5 {
            shimmer_read_pos = shimmer_write as f64 - shimmer_buf_len as f64 * 0.25;
        }

        // Apply feedback matrix
        let mut feedback = [0.0f32; 4];
        for r in 0..n_delays {
            let mut s = 0.0f32;
            for c in 0..n_delays {
                s += mat[r][c] * delayed[c];
            }
            feedback[r] = s;
        }

        // Mix shimmer into feedback
        for d in 0..n_delays {
            let fb = feedback[d];
            let shimmer_contribution = shimmer_amount * pitched;
            let fb_mixed = (1.0 - shimmer_amount) * fb + shimmer_contribution;

            // Lowpass
            lp_state[d] = lp_coeff * lp_state[d] + (1.0 - lp_coeff) * fb_mixed;
            let val = x + gains[d] * lp_state[d];
            let write_pos = write_idx[d] % buf_size;
            bufs[d][write_pos] = val;
            write_idx[d] += 1;
        }

        out[i] = (1.0 - wet_mix) * x + wet_mix * wet_out;
    }

    let out = post_process(&out, sr);
    AudioOutput::Mono(out)
}

fn variants_b006() -> Vec<HashMap<String, Value>> {
    vec![
        params!{"rt60" => 1.0, "pitch_shift_semitones" => 12.0, "shimmer_amount" => 0.2, "wet_mix" => 0.4},
        params!{"rt60" => 3.0, "pitch_shift_semitones" => 12.0, "shimmer_amount" => 0.3, "wet_mix" => 0.5},
        params!{"rt60" => 5.0, "pitch_shift_semitones" => 7.0, "shimmer_amount" => 0.4, "wet_mix" => 0.6},
        params!{"rt60" => 7.0, "pitch_shift_semitones" => 5.0, "shimmer_amount" => 0.2, "wet_mix" => 0.5},
        params!{"rt60" => 10.0, "pitch_shift_semitones" => 12.0, "shimmer_amount" => 0.6, "wet_mix" => 0.8},
        params!{"rt60" => 4.0, "pitch_shift_semitones" => 9.0, "shimmer_amount" => 0.5, "wet_mix" => 0.7},
    ]
}

// ---------------------------------------------------------------------------
// B007 — Convolution Reverb with Synthetic IR
// Generate noise*exp decay IR + early reflection spikes, convolve via FFT
// ---------------------------------------------------------------------------

use realfft::RealFftPlanner;

fn generate_synthetic_ir(sr: u32, ir_length_ms: f32, decay_rate: f32, num_early_reflections: usize, er_spacing_ms: f32) -> Vec<f32> {
    let ir_length_samps = ((ir_length_ms * 0.001 * sr as f32) as usize).max(1);

    // Exponential decay envelope applied to noise
    let mut rng = Lcg::new(42);
    let mut ir = vec![0.0f32; ir_length_samps];
    for pos in 0..ir_length_samps {
        let t = pos as f32 / sr as f32;
        let envelope = (-decay_rate * t).exp();
        let noise = rng.next_bipolar();
        ir[pos] = noise * envelope;
    }

    // Early reflections: discrete spikes
    let er_spacing_samps = ((er_spacing_ms * 0.001 * sr as f32) as usize).max(1);
    for r in 0..num_early_reflections {
        let pos = (r + 1) * er_spacing_samps;
        if pos < ir_length_samps {
            let mut amp = 0.8f32 * 0.7f32.powi(r as i32);
            if r % 2 == 1 {
                amp = -amp;
            }
            ir[pos] += amp;
        }
    }

    // Normalize IR
    let peak = ir.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    if peak > 1e-10 {
        for s in ir.iter_mut() {
            *s /= peak;
        }
    }

    ir
}

fn fft_convolve(signal: &[f32], ir: &[f32]) -> Vec<f32> {
    let n = signal.len();
    let ir_len = ir.len();
    let conv_len = n + ir_len - 1;

    // Next power of 2
    let mut fft_size = 1;
    while fft_size < conv_len {
        fft_size *= 2;
    }

    let mut planner = RealFftPlanner::<f32>::new();
    let fft_fwd = planner.plan_fft_forward(fft_size);
    let fft_inv = planner.plan_fft_inverse(fft_size);

    // Pad signal
    let mut sig_buf = vec![0.0f32; fft_size];
    sig_buf[..n].copy_from_slice(signal);

    // Pad IR
    let mut ir_buf = vec![0.0f32; fft_size];
    ir_buf[..ir_len].copy_from_slice(ir);

    // Forward FFTs
    let mut sig_spec = fft_fwd.make_output_vec();
    let mut ir_spec = fft_fwd.make_output_vec();
    let mut scratch = fft_fwd.make_scratch_vec();
    fft_fwd.process_with_scratch(&mut sig_buf, &mut sig_spec, &mut scratch).unwrap();
    fft_fwd.process_with_scratch(&mut ir_buf, &mut ir_spec, &mut scratch).unwrap();

    // Multiply spectra
    for i in 0..sig_spec.len() {
        sig_spec[i] = sig_spec[i] * ir_spec[i];
    }

    // Inverse FFT
    let mut result = fft_inv.make_output_vec();
    let mut inv_scratch = fft_inv.make_scratch_vec();
    fft_inv.process_with_scratch(&mut sig_spec, &mut result, &mut inv_scratch).unwrap();

    // Normalize (realfft inverse is unnormalized)
    let norm = 1.0 / fft_size as f32;
    for s in result.iter_mut() {
        *s *= norm;
    }

    let out_len = conv_len.min(fft_size);
    result.truncate(out_len);
    result
}

fn process_b007(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let ir_length_ms = pf(params, "ir_length_ms", 1500.0);
    let decay_rate = pf(params, "decay_rate", 2.0);
    let num_early_reflections = pi(params, "num_early_reflections", 8) as usize;
    let er_spacing_ms = pf(params, "er_spacing_ms", 15.0);
    let wet_mix = pf(params, "wet_mix", 0.5);

    let ir = generate_synthetic_ir(sr, ir_length_ms, decay_rate, num_early_reflections, er_spacing_ms);
    let convolved = fft_convolve(samples, &ir);

    let n = samples.len();
    let out_len = convolved.len();

    // Wet/dry mix
    let mut result = vec![0.0f32; out_len];
    for i in 0..out_len {
        let dry = if i < n { samples[i] } else { 0.0 };
        result[i] = (1.0 - wet_mix) * dry + wet_mix * convolved[i];
    }

    let result = post_process(&result, sr);
    AudioOutput::Mono(result)
}

fn variants_b007() -> Vec<HashMap<String, Value>> {
    vec![
        params!{"ir_length_ms" => 200.0, "decay_rate" => 5.0, "num_early_reflections" => 3, "er_spacing_ms" => 5.0},
        params!{"ir_length_ms" => 800.0, "decay_rate" => 3.0, "num_early_reflections" => 6, "er_spacing_ms" => 10.0},
        params!{"ir_length_ms" => 1500.0, "decay_rate" => 2.0, "num_early_reflections" => 8, "er_spacing_ms" => 15.0},
        params!{"ir_length_ms" => 3000.0, "decay_rate" => 1.0, "num_early_reflections" => 12, "er_spacing_ms" => 20.0},
        params!{"ir_length_ms" => 5000.0, "decay_rate" => 0.5, "num_early_reflections" => 15, "er_spacing_ms" => 30.0},
        params!{"ir_length_ms" => 1000.0, "decay_rate" => 4.0, "num_early_reflections" => 5, "er_spacing_ms" => 8.0},
    ]
}

// ---------------------------------------------------------------------------
// B008 — Metallic Resonator
// Bank of short comb filters (1-5ms) in parallel with high feedback
// ---------------------------------------------------------------------------

fn process_b008(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let num_resonators = pi(params, "num_resonators", 6) as usize;
    let base_freq_hz = pf(params, "base_freq_hz", 500.0);
    let freq_spread = pf(params, "freq_spread", 1.0);
    let feedback = pf(params, "feedback", 0.95);
    let n = samples.len();

    // Compute delay lengths from frequencies
    let mut delays = vec![0usize; num_resonators];
    for r in 0..num_resonators {
        let freq_ratio = 1.0 + freq_spread * r as f32 / (num_resonators - 1).max(1) as f32;
        let freq = base_freq_hz * freq_ratio;
        let mut d = (sr as f32 / freq) as usize;
        d = d.max(1);
        // Clamp to 1-5ms range
        let min_d = (0.001 * sr as f32) as usize;
        let max_d = (0.005 * sr as f32) as usize;
        d = d.clamp(min_d, max_d);
        delays[r] = d;
    }

    let max_delay = *delays.iter().max().unwrap();
    let buf_size = max_delay + 1;
    let mut bufs = vec![vec![0.0f32; buf_size]; num_resonators];
    let mut write_idx = vec![0usize; num_resonators];

    let fb = feedback;
    let mut out = vec![0.0f32; n];

    for i in 0..n {
        let x = samples[i];
        let mut res_sum = 0.0f32;

        for r in 0..num_resonators {
            let dl = delays[r];
            let read_pos = (write_idx[r] + buf_size - dl) % buf_size;
            let delayed = bufs[r][read_pos];

            // Comb filter: input + feedback * delayed
            let val = x + fb * delayed;
            let write_pos = write_idx[r] % buf_size;
            bufs[r][write_pos] = val;
            write_idx[r] += 1;

            res_sum += delayed;
        }

        let mut res_out = res_sum / num_resonators.max(1) as f32;

        // Soft clip to prevent blowup with high feedback
        if res_out > 1.0 {
            res_out = 1.0 - 1.0 / (res_out + 1.0);
        } else if res_out < -1.0 {
            res_out = -1.0 + 1.0 / (-res_out + 1.0);
        }

        out[i] = 0.4 * x + 0.6 * res_out;
    }

    let out = post_process(&out, sr);
    AudioOutput::Mono(out)
}

fn variants_b008() -> Vec<HashMap<String, Value>> {
    vec![
        params!{"num_resonators" => 4, "base_freq_hz" => 200.0, "freq_spread" => 0.5, "feedback" => 0.92},
        params!{"num_resonators" => 6, "base_freq_hz" => 500.0, "freq_spread" => 1.0, "feedback" => 0.95},
        params!{"num_resonators" => 8, "base_freq_hz" => 800.0, "freq_spread" => 1.5, "feedback" => 0.97},
        params!{"num_resonators" => 10, "base_freq_hz" => 1200.0, "freq_spread" => 2.0, "feedback" => 0.99},
        params!{"num_resonators" => 12, "base_freq_hz" => 2000.0, "freq_spread" => 0.8, "feedback" => 0.993},
        params!{"num_resonators" => 5, "base_freq_hz" => 350.0, "freq_spread" => 1.2, "feedback" => 0.98},
        params!{"num_resonators" => 8, "base_freq_hz" => 1000.0, "freq_spread" => 0.5, "feedback" => 0.995},
    ]
}

// ---------------------------------------------------------------------------
// B009 — Dattorro Plate Reverb
// Distinct from B004: uses Dattorro's specific topology with two nested
// allpass chains feeding a figure-eight tank with modulated delays.
// ---------------------------------------------------------------------------

fn process_b009(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let decay = pf(params, "decay", 0.7);
    let damping = pf(params, "damping", 0.4);
    let bandwidth = pf(params, "bandwidth", 0.7);
    let pre_delay_ms = pf(params, "pre_delay_ms", 10.0);
    let n = samples.len();

    // Pre-delay
    let pre_delay_samps = ((pre_delay_ms * 0.001 * sr as f32) as usize).max(1);
    let mut pre_buf = vec![0.0f32; pre_delay_samps];
    let mut pre_idx = 0usize;

    // Input diffusion: 4 allpass filters in series
    let in_ap_delays_ms: [f64; 4] = [4.77, 3.60, 12.73, 9.31];
    let in_ap_g: [f32; 4] = [0.75, 0.75, 0.625, 0.625];
    let num_in_ap = 4;
    let mut in_ap_delays = [0usize; 4];
    for i in 0..num_in_ap {
        in_ap_delays[i] = ((in_ap_delays_ms[i] * 0.001 * sr as f64) as usize).max(1);
    }
    let max_in_ap = *in_ap_delays.iter().max().unwrap();
    let in_ap_buf_size = max_in_ap + 1;
    let mut in_ap_bufs = vec![vec![0.0f32; in_ap_buf_size]; num_in_ap];
    let mut in_ap_idx = [0usize; 4];

    // Tank: two halves, each has allpass -> delay -> lowpass -> decay
    let tank_ap_delays_ms: [f64; 2] = [22.58, 30.51];
    let tank_delay_ms: [f64; 2] = [149.63, 125.0];
    let tank_ap_g = 0.5f32;

    let mut tank_ap_delays = [0usize; 2];
    let mut tank_delays = [0usize; 2];
    for i in 0..2 {
        tank_ap_delays[i] = ((tank_ap_delays_ms[i] * 0.001 * sr as f64) as usize).max(1);
        tank_delays[i] = ((tank_delay_ms[i] * 0.001 * sr as f64) as usize).max(1);
    }

    let max_tank_ap = *tank_ap_delays.iter().max().unwrap();
    let max_tank_dl = *tank_delays.iter().max().unwrap();

    let tank_ap_buf_size = max_tank_ap + 1;
    let tank_dl_buf_size = max_tank_dl + 1;

    let mut tank_ap_bufs = vec![vec![0.0f32; tank_ap_buf_size]; 2];
    let mut tank_ap_idx = [0usize; 2];
    let mut tank_dl_bufs = vec![vec![0.0f32; tank_dl_buf_size]; 2];
    let mut tank_dl_idx = [0usize; 2];
    let mut tank_lp = [0.0f32; 2];
    let mut tank_fb = [0.0f32; 2];

    let bw_coeff = bandwidth;
    let lp_coeff = damping;
    let decay_g = decay;

    // Input bandwidth lowpass
    let mut bw_state = 0.0f32;

    let mut out = vec![0.0f32; n];

    for i in 0..n {
        let x = samples[i];

        // Input bandwidth control (one-pole lowpass)
        bw_state = bw_coeff * bw_state + (1.0 - bw_coeff) * x;
        let mut y = bw_state;

        // Pre-delay
        let pre_out = pre_buf[pre_idx % pre_delay_samps];
        pre_buf[pre_idx % pre_delay_samps] = y;
        pre_idx += 1;
        y = pre_out;

        // Input diffusion allpass chain
        for a in 0..num_in_ap {
            let dl = in_ap_delays[a];
            let rd = (in_ap_idx[a] + in_ap_buf_size - dl) % in_ap_buf_size;
            let delayed = in_ap_bufs[a][rd];
            let g = in_ap_g[a];
            let v = y - g * delayed;
            in_ap_bufs[a][in_ap_idx[a] % in_ap_buf_size] = v;
            in_ap_idx[a] += 1;
            y = delayed + g * v;
        }

        // Feed into tank with cross-feedback (figure-eight)
        let tank_in_0 = y + decay_g * tank_fb[1];
        let tank_in_1 = y + decay_g * tank_fb[0];

        let mut out_sum = 0.0f32;
        for t in 0..2 {
            let tank_in = if t == 0 { tank_in_0 } else { tank_in_1 };

            // Tank allpass
            let dl = tank_ap_delays[t];
            let rd = (tank_ap_idx[t] + tank_ap_buf_size - dl) % tank_ap_buf_size;
            let delayed = tank_ap_bufs[t][rd];
            let v = tank_in - tank_ap_g * delayed;
            tank_ap_bufs[t][tank_ap_idx[t] % tank_ap_buf_size] = v;
            tank_ap_idx[t] += 1;
            let ap_out = delayed + tank_ap_g * v;

            // Tank delay line
            let dl2 = tank_delays[t];
            let rd2 = (tank_dl_idx[t] + tank_dl_buf_size - dl2) % tank_dl_buf_size;
            let delayed2 = tank_dl_bufs[t][rd2];
            tank_dl_bufs[t][tank_dl_idx[t] % tank_dl_buf_size] = ap_out;
            tank_dl_idx[t] += 1;

            // Damping lowpass
            tank_lp[t] = lp_coeff * tank_lp[t] + (1.0 - lp_coeff) * delayed2;
            tank_fb[t] = tank_lp[t];

            out_sum += delayed2;
        }

        let wet = out_sum * 0.5;
        out[i] = 0.5 * x + 0.5 * wet;
    }

    let out = post_process(&out, sr);
    AudioOutput::Mono(out)
}

fn variants_b009() -> Vec<HashMap<String, Value>> {
    vec![
        params!{"decay" => 0.3, "damping" => 0.2, "bandwidth" => 0.9, "pre_delay_ms" => 0.0},
        params!{"decay" => 0.5, "damping" => 0.4, "bandwidth" => 0.7, "pre_delay_ms" => 10.0},
        params!{"decay" => 0.7, "damping" => 0.5, "bandwidth" => 0.7, "pre_delay_ms" => 20.0},
        params!{"decay" => 0.85, "damping" => 0.6, "bandwidth" => 0.5, "pre_delay_ms" => 40.0},
        params!{"decay" => 0.95, "damping" => 0.8, "bandwidth" => 0.3, "pre_delay_ms" => 60.0},
        params!{"decay" => 0.99, "damping" => 0.3, "bandwidth" => 0.8, "pre_delay_ms" => 100.0},
    ]
}

// ---------------------------------------------------------------------------
// B010 — Freeverb (Jezar)
// 8 parallel comb filters -> 4 series allpass, distinct from Schroeder/Moorer
// ---------------------------------------------------------------------------

fn process_b010(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let room_size = pf(params, "room_size", 0.7);
    let damping = pf(params, "damping", 0.5);
    let wet_mix = pf(params, "wet_mix", 0.5);
    let n = samples.len();

    // Jezar's original comb filter delays (scaled by sr/44100)
    let scale = sr as f64 / 44100.0;
    let comb_delays_base: [i64; 8] = [1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617];
    let num_combs = 8;
    let mut comb_delays = [0usize; 8];
    for i in 0..num_combs {
        comb_delays[i] = ((comb_delays_base[i] as f64 * scale) as usize).max(1);
    }

    let max_comb = *comb_delays.iter().max().unwrap();
    let comb_buf_size = max_comb + 1;
    let mut comb_bufs = vec![vec![0.0f32; comb_buf_size]; num_combs];
    let mut comb_idx = [0usize; 8];
    let mut comb_filter_state = [0.0f32; 8];

    let fb = room_size;
    let damp1 = damping;
    let damp2 = 1.0 - damp1;

    // 4 series allpass filters
    let ap_delays_base: [i64; 4] = [556, 441, 341, 225];
    let num_ap = 4;
    let mut ap_delays = [0usize; 4];
    for i in 0..num_ap {
        ap_delays[i] = ((ap_delays_base[i] as f64 * scale) as usize).max(1);
    }

    let max_ap = *ap_delays.iter().max().unwrap();
    let ap_buf_size = max_ap + 1;
    let mut ap_bufs = vec![vec![0.0f32; ap_buf_size]; num_ap];
    let mut ap_idx = [0usize; 4];
    let ap_g = 0.5f32;

    let mut out = vec![0.0f32; n];

    for i in 0..n {
        let x = samples[i];

        // 8 parallel Lowpass-Feedback-Comb filters summed
        let mut comb_sum = 0.0f32;
        for c in 0..num_combs {
            let dl = comb_delays[c];
            let rd = (comb_idx[c] + comb_buf_size - dl) % comb_buf_size;
            let delayed = comb_bufs[c][rd];

            // One-pole lowpass in feedback (Jezar's damping)
            comb_filter_state[c] = delayed * damp2 + comb_filter_state[c] * damp1;

            let val = x + fb * comb_filter_state[c];
            comb_bufs[c][comb_idx[c] % comb_buf_size] = val;
            comb_idx[c] += 1;
            comb_sum += delayed;
        }

        let mut y = comb_sum * 0.125; // scale by 1/8

        // 4 series allpass filters
        for a in 0..num_ap {
            let dl = ap_delays[a];
            let rd = (ap_idx[a] + ap_buf_size - dl) % ap_buf_size;
            let delayed = ap_bufs[a][rd];
            let v = y + ap_g * delayed;
            ap_bufs[a][ap_idx[a] % ap_buf_size] = y;
            ap_idx[a] += 1;
            y = delayed - ap_g * v;
        }

        out[i] = (1.0 - wet_mix) * x + wet_mix * y;
    }

    let out = post_process(&out, sr);
    AudioOutput::Mono(out)
}

fn variants_b010() -> Vec<HashMap<String, Value>> {
    vec![
        params!{"room_size" => 0.3, "damping" => 0.3, "wet_mix" => 0.3},
        params!{"room_size" => 0.5, "damping" => 0.5, "wet_mix" => 0.4},
        params!{"room_size" => 0.7, "damping" => 0.5, "wet_mix" => 0.5},
        params!{"room_size" => 0.85, "damping" => 0.7, "wet_mix" => 0.6},
        params!{"room_size" => 0.95, "damping" => 0.2, "wet_mix" => 0.7},
        params!{"room_size" => 0.99, "damping" => 0.9, "wet_mix" => 0.8},
    ]
}

// ---------------------------------------------------------------------------
// B011 — Velvet Noise Reverb
// Sparse random +1/-1 impulse sequence as FIR, convolved via FFT.
// Very efficient, perceptually distinct smooth character.
// ---------------------------------------------------------------------------

fn generate_velvet_ir(sr: u32, ir_length_ms: f32, density: f32, decay_rate: f32, seed: u64) -> Vec<f32> {
    let ir_length_samps = ((ir_length_ms * 0.001 * sr as f32) as usize).max(1);
    let mut ir = vec![0.0f32; ir_length_samps];

    let mut rng = Lcg::new(seed);

    // Average spacing between impulses
    let avg_spacing = (sr as f32 / density) as usize;
    let avg_spacing = avg_spacing.max(1);

    let mut pos = 0usize;
    while pos < ir_length_samps {
        // Place impulse with random +1/-1 polarity
        let polarity: f32 = if rng.next_f32() > 0.5 { 1.0 } else { -1.0 };
        // Decay envelope
        let t = pos as f32 / sr as f32;
        let amp = (-decay_rate * t).exp();
        ir[pos] = polarity * amp;

        // Next position: jittered spacing
        // rng.integers(1, max(2, avg_spacing * 2))  =>  1..avg_spacing*2
        let upper = (avg_spacing * 2).max(2);
        let jitter = 1 + (rng.next_f32() * (upper - 1) as f32) as usize;
        pos += jitter;
    }

    // Normalize
    let peak = ir.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    if peak > 1e-10 {
        for s in ir.iter_mut() {
            *s /= peak;
        }
    }

    ir
}

fn process_b011(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let ir_length_ms = pf(params, "ir_length_ms", 1500.0);
    let density = pf(params, "density", 2000.0);
    let decay_rate = pf(params, "decay_rate", 2.0);
    let wet_mix = pf(params, "wet_mix", 0.5);
    let seed = crate::pu(params, "seed", 42);

    let ir = generate_velvet_ir(sr, ir_length_ms, density, decay_rate, seed);
    let convolved = fft_convolve(samples, &ir);

    let n = samples.len();
    let out_len = convolved.len();

    let mut result = vec![0.0f32; out_len];
    for i in 0..out_len {
        let dry = if i < n { samples[i] } else { 0.0 };
        result[i] = (1.0 - wet_mix) * dry + wet_mix * convolved[i];
    }

    let result = post_process(&result, sr);
    AudioOutput::Mono(result)
}

fn variants_b011() -> Vec<HashMap<String, Value>> {
    vec![
        params!{"ir_length_ms" => 500, "density" => 1000, "decay_rate" => 4.0, "wet_mix" => 0.3, "seed" => 42},
        params!{"ir_length_ms" => 1000, "density" => 2000, "decay_rate" => 2.5, "wet_mix" => 0.4, "seed" => 42},
        params!{"ir_length_ms" => 1500, "density" => 2000, "decay_rate" => 2.0, "wet_mix" => 0.5, "seed" => 42},
        params!{"ir_length_ms" => 2500, "density" => 3000, "decay_rate" => 1.5, "wet_mix" => 0.5, "seed" => 7},
        params!{"ir_length_ms" => 4000, "density" => 4000, "decay_rate" => 1.0, "wet_mix" => 0.6, "seed" => 99},
        params!{"ir_length_ms" => 1500, "density" => 500, "decay_rate" => 2.0, "wet_mix" => 0.5, "seed" => 17},
    ]
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

pub fn register() -> Vec<EffectEntry> {
    vec![
        EffectEntry {
            id: "B001",
            process: process_b001,
            variants: variants_b001,
            category: "reverb",
        },
        EffectEntry {
            id: "B002",
            process: process_b002,
            variants: variants_b002,
            category: "reverb",
        },
        EffectEntry {
            id: "B003",
            process: process_b003,
            variants: variants_b003,
            category: "reverb",
        },
        EffectEntry {
            id: "B004",
            process: process_b004,
            variants: variants_b004,
            category: "reverb",
        },
        EffectEntry {
            id: "B005",
            process: process_b005,
            variants: variants_b005,
            category: "reverb",
        },
        EffectEntry {
            id: "B006",
            process: process_b006,
            variants: variants_b006,
            category: "reverb",
        },
        EffectEntry {
            id: "B007",
            process: process_b007,
            variants: variants_b007,
            category: "reverb",
        },
        EffectEntry {
            id: "B008",
            process: process_b008,
            variants: variants_b008,
            category: "reverb",
        },
        EffectEntry {
            id: "B009",
            process: process_b009,
            variants: variants_b009,
            category: "reverb",
        },
        EffectEntry {
            id: "B010",
            process: process_b010,
            variants: variants_b010,
            category: "reverb",
        },
        EffectEntry {
            id: "B011",
            process: process_b011,
            variants: variants_b011,
            category: "reverb",
        },
    ]
}
