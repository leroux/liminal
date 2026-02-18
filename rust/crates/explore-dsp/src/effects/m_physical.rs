//! M-series: Physical modeling effects (M001-M005).
//!
//! Karplus-Strong, waveguide string, mass-spring chain, membrane/drum
//! resonator, tube resonator. All use physical simulation to color the input.

use std::collections::HashMap;
use serde_json::Value;
use crate::{AudioOutput, EffectEntry, pf, pi, params};
#[allow(unused_imports)]
use crate::primitives::*;

// ---------------------------------------------------------------------------
// M001 -- Karplus-Strong as Effect
// ---------------------------------------------------------------------------

fn process_m001(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let freq_hz = pf(params, "freq_hz", 220.0);
    let decay_factor = pf(params, "decay_factor", 0.99);
    let delay_len = (sr as f32 / freq_hz).max(1.0) as usize;

    let n = samples.len();
    let buf_len = (delay_len + 2).max(2);
    let mut buf = vec![0.0f32; buf_len];
    let mut out = vec![0.0f32; n];

    // Initialize buffer with input (first delay_len samples)
    let init_len = delay_len.min(n);
    for i in 0..init_len {
        buf[i] = samples[i];
    }

    let mut write_pos: usize = 0;
    for i in 0..n {
        // Read from delay_len samples back
        let read_pos_0 = (write_pos + buf_len - delay_len) % buf_len;
        let read_pos_1 = (write_pos + buf_len - delay_len - 1) % buf_len;
        // KS averaging filter
        let mut y = decay_factor * 0.5 * (buf[read_pos_0] + buf[read_pos_1]);
        // Mix input with KS output
        y += samples[i];
        // Clamp to prevent explosion
        y = y.clamp(-10.0, 10.0);
        buf[write_pos] = y;
        out[i] = y;
        write_pos = (write_pos + 1) % buf_len;
    }
    AudioOutput::Mono(out)
}

fn variants_m001() -> Vec<HashMap<String, Value>> {
    vec![
        params!("freq_hz" => 80, "decay_factor" => 0.995),     // deep bass drone
        params!("freq_hz" => 150, "decay_factor" => 0.99),      // low pluck coloring
        params!("freq_hz" => 220, "decay_factor" => 0.98),       // mid-range metallic ring
        params!("freq_hz" => 440, "decay_factor" => 0.97),       // bright pluck resonance
        params!("freq_hz" => 880, "decay_factor" => 0.95),       // high-pitched ping
        params!("freq_hz" => 1500, "decay_factor" => 0.92),      // very bright, short decay
        params!("freq_hz" => 55, "decay_factor" => 0.999),       // sub-bass, long sustain
    ]
}

// ---------------------------------------------------------------------------
// M002 -- Waveguide String Resonator
// ---------------------------------------------------------------------------

fn process_m002(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let freq_hz = pf(params, "freq_hz", 220.0);
    let decay = pf(params, "decay", 0.995);
    let brightness = pf(params, "brightness", 0.5);
    let excitation_position = pf(params, "excitation_position", 0.3);

    // Total delay = sr / freq_hz, split between upper and lower rails
    let total_delay = (sr as f32 / freq_hz).max(4.0) as usize;
    let upper_len = (total_delay / 2).max(2);
    let lower_len = (total_delay - upper_len).max(2);

    // Excitation position as fraction of each rail
    let excite_pos_upper = (excitation_position * upper_len as f32).max(1.0) as usize;
    let excite_pos_lower = (excitation_position * lower_len as f32).max(1.0) as usize;

    let n = samples.len();
    let mut upper_buf = vec![0.0f32; upper_len];
    let mut lower_buf = vec![0.0f32; lower_len];
    let mut out = vec![0.0f32; n];

    // One-pole lowpass coefficient for brightness at nut reflection
    let lp_coeff = brightness;
    let mut lp_state_upper = 0.0f32;
    let mut lp_state_lower = 0.0f32;

    let mut upper_write: usize = 0;
    let mut lower_write: usize = 0;

    for i in 0..n {
        // Read from end of upper rail (nut end)
        let upper_read = (upper_write + 1) % upper_len;
        let upper_end = upper_buf[upper_read];

        // Read from end of lower rail (bridge end)
        let lower_read = (lower_write + 1) % lower_len;
        let lower_end = lower_buf[lower_read];

        // Nut reflection: invert + lowpass (brightness control)
        lp_state_upper = lp_coeff * upper_end + (1.0 - lp_coeff) * lp_state_upper;
        let nut_reflect = -decay * lp_state_upper;

        // Bridge reflection: invert + slight lowpass
        lp_state_lower = lp_coeff * lower_end + (1.0 - lp_coeff) * lp_state_lower;
        let bridge_reflect = -decay * lp_state_lower;

        // Feed reflections back: nut reflection -> lower rail, bridge -> upper rail
        lower_buf[lower_write] = nut_reflect;
        upper_buf[upper_write] = bridge_reflect;

        // Inject input at excitation position on both rails
        let excite_upper_write = (upper_write + excite_pos_upper) % upper_len;
        let excite_lower_write = (lower_write + excite_pos_lower) % lower_len;
        upper_buf[excite_upper_write] += samples[i] * 0.5;
        lower_buf[excite_lower_write] += samples[i] * 0.5;

        // Output from bridge (sum of both rails at bridge)
        let y = (upper_end + lower_end).clamp(-10.0, 10.0);
        out[i] = y;

        upper_write = (upper_write + 1) % upper_len;
        lower_write = (lower_write + 1) % lower_len;
    }
    AudioOutput::Mono(out)
}

fn variants_m002() -> Vec<HashMap<String, Value>> {
    vec![
        params!("freq_hz" => 110, "decay" => 0.998, "brightness" => 0.3, "excitation_position" => 0.5),   // dark low string
        params!("freq_hz" => 220, "decay" => 0.995, "brightness" => 0.5, "excitation_position" => 0.3),   // warm mid string
        params!("freq_hz" => 330, "decay" => 0.99, "brightness" => 0.7, "excitation_position" => 0.15),    // bright string, near bridge
        params!("freq_hz" => 440, "decay" => 0.985, "brightness" => 0.8, "excitation_position" => 0.1),    // brilliant high string
        params!("freq_hz" => 82, "decay" => 0.999, "brightness" => 0.2, "excitation_position" => 0.5),     // deep bass string, center excite
        params!("freq_hz" => 660, "decay" => 0.97, "brightness" => 0.9, "excitation_position" => 0.25),    // high, very bright, short
        params!("freq_hz" => 150, "decay" => 0.997, "brightness" => 0.4, "excitation_position" => 0.7),    // low, near nut excitation
    ]
}

// ---------------------------------------------------------------------------
// M003 -- Mass-Spring Damper Chain
// ---------------------------------------------------------------------------

fn process_m003(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let num_masses = pi(params, "num_masses", 8) as usize;
    let stiffness = pf(params, "stiffness", 1.0);
    let damping = pf(params, "damping", 0.1);
    let mass = pf(params, "mass", 0.5);

    let n = samples.len();
    let mut out = vec![0.0f32; n];
    let mut pos = vec![0.0f32; num_masses];
    let mut vel = vec![0.0f32; num_masses];
    let mut force = vec![0.0f32; num_masses];

    // Scale stiffness to get audio-rate resonances.
    let sr_f = sr as f32;
    let k_scaled = stiffness * sr_f * sr_f * 0.001;
    let d_scaled = damping * sr_f * 2.0;
    let inv_mass = 1.0 / mass;

    let dt = 1.0 / sr_f;

    // Substeps for stability: CFL condition
    let omega_max = (k_scaled * 4.0 * inv_mass).sqrt();
    let num_substeps = ((omega_max as f64 * dt as f64 * 1.5) as usize).max(1).min(32);
    let sub_dt = dt / num_substeps as f32;

    for i in 0..n {
        // Drive first mass with input
        let drive_force = samples[i] * k_scaled;

        for _ in 0..num_substeps {
            // Compute forces on all masses
            for m in 0..num_masses {
                let left_pos = if m == 0 { 0.0 } else { pos[m - 1] };
                let right_pos = if m == num_masses - 1 { 0.0 } else { pos[m + 1] };

                let spring_f = k_scaled * (left_pos - pos[m]) + k_scaled * (right_pos - pos[m]);
                let damp_f = -d_scaled * vel[m];
                let ext_f = if m == 0 { drive_force } else { 0.0 };
                force[m] = spring_f + damp_f + ext_f;
            }

            // Update velocities (symplectic Euler)
            for m in 0..num_masses {
                vel[m] += force[m] * inv_mass * sub_dt;
                vel[m] = vel[m].clamp(-10.0, 10.0);
            }

            // Update positions
            for m in 0..num_masses {
                pos[m] += vel[m] * sub_dt;
                pos[m] = pos[m].clamp(-10.0, 10.0);
            }
        }

        // Output from last mass
        out[i] = pos[num_masses - 1];
    }

    // Auto-gain: normalize output to match input peak level
    let out_peak = out.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    let in_peak = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    if out_peak > 1e-10 && in_peak > 1e-10 {
        let scale = in_peak / out_peak;
        for s in out.iter_mut() {
            *s *= scale;
        }
    }

    AudioOutput::Mono(out)
}

fn variants_m003() -> Vec<HashMap<String, Value>> {
    vec![
        params!("num_masses" => 3, "stiffness" => 0.5, "damping" => 0.05, "mass" => 0.3),   // short chain, low stiffness, ringy
        params!("num_masses" => 8, "stiffness" => 1.0, "damping" => 0.1, "mass" => 0.5),     // medium chain, balanced
        params!("num_masses" => 15, "stiffness" => 2.0, "damping" => 0.05, "mass" => 0.2),   // long chain, high stiffness, metallic
        params!("num_masses" => 5, "stiffness" => 4.0, "damping" => 0.3, "mass" => 1.0),     // stiff, heavily damped, thud
        params!("num_masses" => 20, "stiffness" => 0.3, "damping" => 0.02, "mass" => 1.5),   // long, floppy, slow propagation
        params!("num_masses" => 10, "stiffness" => 3.0, "damping" => 0.5, "mass" => 0.1),    // light masses, fast, damped
        params!("num_masses" => 6, "stiffness" => 0.8, "damping" => 0.01, "mass" => 0.4),    // low damping, long sustain
    ]
}

// ---------------------------------------------------------------------------
// M004 -- Membrane / Drum Resonator (2D waveguide mesh)
// ---------------------------------------------------------------------------

fn process_m004(samples: &[f32], _sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let mut grid_size = pi(params, "grid_size", 8) as usize;
    let tension = pf(params, "tension", 0.3);
    let damping = pf(params, "damping", 0.01);

    // Clamp grid_size to [5, 20] for performance
    grid_size = grid_size.clamp(5, 20);

    let n = samples.len();
    let mut out = vec![0.0f32; n];

    // Courant number: c = tension, must be <= 0.5 for 2D stability
    let c = tension.min(0.5);
    let c2 = c * c;

    // Two time steps of the 2D grid (current and previous)
    // Using flat arrays for the grids: index = gx * grid_size + gy
    let gs = grid_size;
    let grid_total = gs * gs;
    let mut grid_curr = vec![0.0f32; grid_total];
    let mut grid_prev = vec![0.0f32; grid_total];
    let mut grid_next = vec![0.0f32; grid_total];

    // Input injection at center
    let cx = gs / 2;
    let cy = gs / 2;
    // Output read from edge
    let ox = gs - 2;
    let oy = gs / 2;

    let damp = 1.0 - damping;

    for i in 0..n {
        // Inject input at center
        grid_curr[cx * gs + cy] += samples[i] * 0.5;

        // Update interior grid points using finite difference wave equation
        for gx in 1..(gs - 1) {
            for gy in 1..(gs - 1) {
                let idx = gx * gs + gy;
                let laplacian = grid_curr[(gx + 1) * gs + gy]
                    + grid_curr[(gx - 1) * gs + gy]
                    + grid_curr[gx * gs + (gy + 1)]
                    + grid_curr[gx * gs + (gy - 1)]
                    - 4.0 * grid_curr[idx];
                grid_next[idx] = (2.0 * grid_curr[idx] - grid_prev[idx] + c2 * laplacian) * damp;
            }
        }

        // Boundary: fixed edges (Dirichlet) - grid_next edges stay 0

        // Clamp values to prevent explosion
        for val in grid_next.iter_mut() {
            *val = val.clamp(-10.0, 10.0);
        }

        // Read output from edge point
        out[i] = grid_next[ox * gs + oy];

        // Swap grids: prev = curr, curr = next, next = 0
        for idx in 0..grid_total {
            grid_prev[idx] = grid_curr[idx];
            grid_curr[idx] = grid_next[idx];
            grid_next[idx] = 0.0;
        }
    }

    AudioOutput::Mono(out)
}

fn variants_m004() -> Vec<HashMap<String, Value>> {
    vec![
        params!("grid_size" => 5, "tension" => 0.4, "damping" => 0.005),    // small tight membrane, high pitch
        params!("grid_size" => 8, "tension" => 0.3, "damping" => 0.01),      // medium drum, balanced
        params!("grid_size" => 12, "tension" => 0.2, "damping" => 0.008),    // large membrane, lower pitch
        params!("grid_size" => 6, "tension" => 0.45, "damping" => 0.002),    // small, very tight, long ring
        params!("grid_size" => 15, "tension" => 0.15, "damping" => 0.03),    // large, loose, short decay
        params!("grid_size" => 10, "tension" => 0.35, "damping" => 0.05),    // medium, heavily damped, thud
        params!("grid_size" => 7, "tension" => 0.1, "damping" => 0.005),     // small, low tension, flabby
    ]
}

// ---------------------------------------------------------------------------
// M005 -- Tube Resonator (1D waveguide with scattering junctions)
// ---------------------------------------------------------------------------

fn process_m005(samples: &[f32], sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let mut num_segments = pi(params, "num_segments", 5) as usize;
    let tube_length_ms = pf(params, "tube_length_ms", 20.0);

    // Clamp num_segments
    num_segments = num_segments.clamp(3, 10);

    // Total delay in samples
    let total_delay = (tube_length_ms * sr as f32 / 1000.0)
        .max((num_segments * 2) as f32) as usize;
    let base_seg_len = (total_delay / num_segments).max(2);

    // Create varying cross-sections (conical-ish profile)
    // Cross-section varies from narrow at input to wider at bell
    let mut cross_sections = vec![0.0f32; num_segments];
    for s in 0..num_segments {
        let t = if num_segments > 1 {
            s as f32 / (num_segments - 1) as f32
        } else {
            0.5
        };
        cross_sections[s] = 1.0 + 3.0 * t * t;
    }

    // Compute segment lengths (slight variation for richer modes)
    let mut segment_lengths = vec![0usize; num_segments];
    for s in 0..num_segments {
        segment_lengths[s] = (base_seg_len as f32 * (0.8 + 0.4 * cross_sections[s] / 4.0))
            .max(2.0) as usize;
    }

    // Compute scattering coefficients from cross-section ratios
    // k = (S_{i+1} - S_i) / (S_{i+1} + S_i)
    let mut scattering_coeffs = vec![0.0f32; num_segments];
    for s in 0..(num_segments - 1) {
        let s1 = cross_sections[s];
        let s2 = cross_sections[s + 1];
        scattering_coeffs[s] = (s2 - s1) / (s2 + s1);
    }

    // Damping: per-junction loss
    let damping = 0.005f32;

    let n = samples.len();
    let mut out = vec![0.0f32; n];

    // Find max segment length for flat buffer allocation
    let max_seg_len = segment_lengths.iter().copied().max().unwrap_or(2).max(1);

    // Forward and backward traveling wave delay buffers
    // Flat arrays: segment s occupies [s * max_seg_len .. (s+1) * max_seg_len]
    let mut fwd_bufs = vec![0.0f32; num_segments * max_seg_len];
    let mut bwd_bufs = vec![0.0f32; num_segments * max_seg_len];
    let mut fwd_write = vec![0usize; num_segments];
    let mut bwd_write = vec![0usize; num_segments];

    let damp = 1.0 - damping;

    for i in 0..n {
        // Inject input into forward wave of first segment
        let seg0_len = segment_lengths[0];
        fwd_bufs[fwd_write[0]] += samples[i];

        // Process each junction between segments
        for s in 0..(num_segments - 1) {
            let seg_offset = s * max_seg_len;
            let seg_len = segment_lengths[s];
            let next_seg_offset = (s + 1) * max_seg_len;
            let next_seg_len = segment_lengths[s + 1];

            // Read from end of forward delay of segment s
            let fwd_read = (fwd_write[s] + seg_len - seg_len + 1) % seg_len;
            // Equivalent to (fwd_write[s] - seg_len + 1) % seg_len with wrapping
            let fwd_read = if fwd_write[s] + 1 >= seg_len {
                fwd_write[s] + 1 - seg_len
            } else {
                fwd_write[s] + 1 + seg_len - seg_len
                // This simplifies: we need (fwd_write[s] - seg_len + 1) mod seg_len
            };
            let fwd_read = (fwd_write[s] + 1) % seg_len;
            let p_plus = fwd_bufs[seg_offset + fwd_read];

            // Read from beginning of backward delay of segment s+1
            let bwd_read = (bwd_write[s + 1] + 1) % next_seg_len;
            let p_minus = bwd_bufs[next_seg_offset + bwd_read];

            // Scattering junction
            let k = scattering_coeffs[s];
            let half_1pk = 0.5 * (1.0 + k);
            let half_1mk = 0.5 * (1.0 - k);

            let mut fwd_out = half_1pk * p_plus + half_1mk * p_minus;
            let mut bwd_out = half_1mk * p_plus + half_1pk * p_minus;

            // Apply damping
            fwd_out *= damp;
            bwd_out *= damp;

            // Clamp
            fwd_out = fwd_out.clamp(-10.0, 10.0);
            bwd_out = bwd_out.clamp(-10.0, 10.0);

            // Write transmitted forward into next segment
            fwd_bufs[next_seg_offset + fwd_write[s + 1]] = fwd_out;
            // Write reflected backward into current segment
            bwd_bufs[seg_offset + bwd_write[s]] = bwd_out;
        }

        // End reflections
        // Open end at output (last segment): partial reflection, inverted
        let last_s = num_segments - 1;
        let last_offset = last_s * max_seg_len;
        let last_len = segment_lengths[last_s];
        let last_fwd_read = (fwd_write[last_s] + 1) % last_len;
        let end_out = fwd_bufs[last_offset + last_fwd_read];

        // Output: radiated wave from open end
        out[i] = end_out;

        // Reflect back (open end: inversion, partial reflection)
        let reflected = -end_out * 0.6 * damp;
        bwd_bufs[last_offset + bwd_write[last_s]] = reflected;

        // Closed end at input (first segment): reflect backward wave with no inversion
        let first_len = segment_lengths[0];
        let first_bwd_read = (bwd_write[0] + 1) % first_len;
        let closed_reflect = bwd_bufs[first_bwd_read] * 0.8 * damp;
        fwd_bufs[fwd_write[0]] += closed_reflect;

        // Advance write positions
        for s in 0..num_segments {
            fwd_write[s] = (fwd_write[s] + 1) % segment_lengths[s];
            bwd_write[s] = (bwd_write[s] + 1) % segment_lengths[s];
        }
    }

    // Normalize to prevent excessive amplitude from resonance buildup
    let peak = out.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    let in_peak = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    if peak > 0.0 && in_peak > 0.0 {
        let target_peak = in_peak * 2.0; // allow up to 2x gain
        if peak > target_peak {
            let scale = target_peak / peak;
            for s in out.iter_mut() {
                *s *= scale;
            }
        }
    }

    AudioOutput::Mono(out)
}

fn variants_m005() -> Vec<HashMap<String, Value>> {
    vec![
        params!("num_segments" => 3, "tube_length_ms" => 10),    // short tube, bright, nasal
        params!("num_segments" => 5, "tube_length_ms" => 20),     // medium tube, balanced resonance
        params!("num_segments" => 8, "tube_length_ms" => 35),     // long tube, rich harmonics
        params!("num_segments" => 4, "tube_length_ms" => 5),      // very short, high-pitched resonance
        params!("num_segments" => 10, "tube_length_ms" => 50),    // long complex tube, deep formants
        params!("num_segments" => 6, "tube_length_ms" => 15),     // medium-short, tight resonance
    ]
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

pub fn register() -> Vec<EffectEntry> {
    vec![
        EffectEntry {
            id: "M001",
            process: process_m001,
            variants: variants_m001,
            category: "physical",
        },
        EffectEntry {
            id: "M002",
            process: process_m002,
            variants: variants_m002,
            category: "physical",
        },
        EffectEntry {
            id: "M003",
            process: process_m003,
            variants: variants_m003,
            category: "physical",
        },
        EffectEntry {
            id: "M004",
            process: process_m004,
            variants: variants_m004,
            category: "physical",
        },
        EffectEntry {
            id: "M005",
            process: process_m005,
            variants: variants_m005,
            category: "physical",
        },
    ]
}
