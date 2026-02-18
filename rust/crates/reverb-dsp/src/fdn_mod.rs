//! Modulated FDN inner loop.
//!
//! Extends the static FDN with per-sample LFO modulation of delay times,
//! damping coefficients, output gains, and feedback matrix blending.
//! Uses fractional delay interpolation for smooth delay-time modulation.
//!
//! Port of `reverb/engine/numba_fdn_mod.py`.

use crate::fdn::build_matrix;
use crate::matrix;
use crate::params::{ReverbParams, SR, N};

// ---------------------------------------------------------------------------
// LFO Waveforms
// ---------------------------------------------------------------------------

/// LFO value for phase (0-1) and waveform type. Returns [-1, +1].
/// Waveform codes: 0=sine, 1=triangle, 2=sample-and-hold.
#[inline(always)]
fn lfo_value(phase: f64, waveform: i32) -> f64 {
    match waveform {
        0 => (2.0 * std::f64::consts::PI * phase).sin(),
        1 => {
            if phase < 0.25 {
                phase * 4.0
            } else if phase < 0.75 {
                2.0 - phase * 4.0
            } else {
                phase * 4.0 - 4.0
            }
        }
        _ => {
            // Sample-and-hold
            let n = (phase * 1000.0) as i64;
            let h = ((n.wrapping_mul(1103515245).wrapping_add(12345)) >> 16) & 0x7FFF;
            (h as f64 / 16383.5) - 1.0
        }
    }
}

/// Read from a delay line with linear interpolation (fractional delay).
#[inline(always)]
fn read_delay_frac(buf: &[f64], write_idx: usize, delay_frac: f64, buf_len: usize) -> f64 {
    let delay_int = delay_frac as usize;
    let frac = delay_frac - delay_int as f64;

    let idx0 = (write_idx + buf_len - 1 - delay_int) % buf_len;
    let idx1 = (write_idx + buf_len - 2 - delay_int) % buf_len;

    buf[idx0] * (1.0 - frac) + buf[idx1] * frac
}

/// Render mono input through the modulated FDN, returning interleaved stereo.
pub fn render_fdn_mod(input: &[f64], params: &ReverbParams) -> Vec<f64> {
    let n_samples = input.len();

    // --- Build feedback matrices ---
    let (mat, _) = build_matrix(params);
    let mat2 = matrix::get_matrix(
        &params.mod_matrix2_type,
        N,
        params.mod_matrix2_seed,
    );
    let matrix_type_flag_is_householder = params.matrix_type == "householder";

    // --- Pre-delay ---
    let pre_delay_samples = params.pre_delay.max(1) as usize;
    let pre_delay_len = pre_delay_samples + 1;
    let mut pre_delay_buf = vec![0.0; pre_delay_len];
    let mut pd_wi: usize = 0;

    // --- Diffusion allpasses ---
    let n_diff_stages = (params.diffusion_stages as usize)
        .min(params.diffusion_delays.len());
    let max_diff_len = if n_diff_stages > 0 {
        params.diffusion_delays[..n_diff_stages]
            .iter()
            .map(|&d| d as usize)
            .max()
            .unwrap_or(1)
    } else {
        1
    };
    let mut diff_bufs = vec![vec![0.0; max_diff_len]; n_diff_stages.max(1)];
    let diff_lens: Vec<usize> = if n_diff_stages > 0 {
        params.diffusion_delays[..n_diff_stages]
            .iter()
            .map(|&d| d as usize)
            .collect()
    } else {
        vec![1]
    };
    let diff_gain = params.diffusion;
    let mut diff_idxs = vec![0usize; n_diff_stages.max(1)];

    // --- FDN delay lines (enlarged for modulation excursion) ---
    let delay_times_base: Vec<f64> = params.delay_times.iter().map(|&d| d as f64).collect();
    let mod_depth_delay = &params.mod_depth_delay;
    let max_delay = delay_times_base
        .iter()
        .zip(mod_depth_delay.iter())
        .map(|(d, m)| (*d + m.abs()) as usize)
        .max()
        .unwrap_or(1)
        + 4;
    let delay_buf_len = max_delay + 1;
    let mut delay_bufs = vec![vec![0.0; delay_buf_len]; N];
    let mut delay_write_idxs = vec![0usize; N];

    // --- Damping ---
    let damping_coeffs_base = &params.damping_coeffs;
    let mut damping_y1 = vec![0.0; N];

    // --- Gains ---
    let feedback_gain = params.feedback_gain;
    let input_gains = &params.input_gains;
    let output_gains_base = &params.output_gains;
    let wet_dry = params.wet_dry;
    let saturation = params.saturation;
    let dry_gain = 1.0 - wet_dry;

    // --- DC blocker ---
    let dc_r = 1.0 - 2.0 * std::f64::consts::PI * 5.0 / SR;
    let mut dc_x1 = vec![0.0; N];
    let mut dc_y1 = vec![0.0; N];

    // --- Stereo panning ---
    let mut pan_gain_l = vec![0.0; N];
    let mut pan_gain_r = vec![0.0; N];
    for i in 0..N {
        let pan = params.node_pans[i];
        let angle = (pan * params.stereo_width + 1.0) * std::f64::consts::FRAC_PI_4;
        pan_gain_l[i] = angle.cos();
        pan_gain_r[i] = angle.sin();
    }

    // --- Modulation parameters ---
    let master_rate = params.mod_master_rate;
    let mod_waveform = params.mod_waveform;
    let correlation = params.mod_correlation;
    let mod_depth_damping = &params.mod_depth_damping;
    let mod_depth_output = &params.mod_depth_output;
    let mod_depth_matrix = params.mod_depth_matrix;

    // Compute per-node rates
    let mut mod_rate_delay = [0.0; N];
    let mut mod_rate_damping = [0.0; N];
    let mut mod_rate_output = [0.0; N];
    for i in 0..N {
        mod_rate_delay[i] = master_rate * params.mod_node_rate_mult[i] * params.mod_rate_scale_delay;
        mod_rate_damping[i] = master_rate * params.mod_node_rate_mult[i] * params.mod_rate_scale_damping;
        mod_rate_output[i] = master_rate * params.mod_node_rate_mult[i] * params.mod_rate_scale_output;
    }
    let mod_rate_matrix = params.mod_rate_matrix;

    // Phase offsets: correlated (all same) vs independent (spread evenly)
    let mut phase_offsets = [0.0; N];
    for i in 0..N {
        phase_offsets[i] = (i as f64 / N as f64) * (1.0 - correlation);
    }

    // Phase increments (rate / sample_rate)
    let mut phase_inc_delay = [0.0; N];
    let mut phase_inc_damping = [0.0; N];
    let mut phase_inc_output = [0.0; N];
    for i in 0..N {
        phase_inc_delay[i] = mod_rate_delay[i] / SR;
        phase_inc_damping[i] = mod_rate_damping[i] / SR;
        phase_inc_output[i] = mod_rate_output[i] / SR;
    }
    let phase_inc_matrix = mod_rate_matrix / SR;

    // Active masks
    let any_delay_mod = mod_depth_delay.iter().any(|&d| d > 0.0);
    let any_damping_mod = mod_depth_damping.iter().any(|&d| d > 0.0);
    let any_output_mod = mod_depth_output.iter().any(|&d| d > 0.0);

    // Initial phases
    let mut cur_phase_delay = [0.0; N];
    let mut cur_phase_damping = [0.0; N];
    let mut cur_phase_output = [0.0; N];
    for i in 0..N {
        cur_phase_delay[i] = phase_offsets[i];
        cur_phase_damping[i] = phase_offsets[i];
        cur_phase_output[i] = phase_offsets[i];
    }
    let mut cur_phase_matrix = 0.0;

    // --- Output buffer ---
    let mut output = vec![0.0; n_samples * 2];

    // --- Scratch ---
    let mut reads = [0.0; N];
    let mut mixed = [0.0; N];

    // --- Per-sample loop ---
    for n in 0..n_samples {
        let x = input[n];

        // Pre-delay
        pre_delay_buf[pd_wi] = x;
        pd_wi = (pd_wi + 1) % pre_delay_len;
        let rd_idx = (pd_wi + pre_delay_len - 1 - pre_delay_samples) % pre_delay_len;
        let x_delayed = pre_delay_buf[rd_idx];

        // Input diffusion (allpass chain)
        let mut diffused = x_delayed;
        for s in 0..n_diff_stages {
            let idx = diff_idxs[s];
            let delayed = diff_bufs[s][idx];
            let g = diff_gain;
            let v = diffused + g * delayed;
            diffused = -g * v + delayed;
            diff_bufs[s][idx] = v;
            diff_idxs[s] = (idx + 1) % diff_lens[s];
        }

        // Matrix modulation LFO
        let mat_blend = if mod_depth_matrix > 0.0 {
            let lfo_mat = lfo_value(cur_phase_matrix, mod_waveform);
            cur_phase_matrix = (cur_phase_matrix + phase_inc_matrix) % 1.0;
            0.5 + 0.5 * lfo_mat * mod_depth_matrix
        } else {
            0.0
        };

        // Read from delay lines (with fractional delay modulation)
        let mut wet_l = 0.0;
        let mut wet_r = 0.0;
        for i in 0..N {
            let wi = delay_write_idxs[i];

            // Modulated delay time
            let current_delay = if any_delay_mod && mod_depth_delay[i] > 0.0 {
                let lfo_d = lfo_value(cur_phase_delay[i], mod_waveform);
                (delay_times_base[i] + mod_depth_delay[i] * lfo_d).max(1.0)
            } else {
                delay_times_base[i]
            };

            reads[i] = read_delay_frac(&delay_bufs[i], wi, current_delay, delay_buf_len);

            // Modulated output gain
            let current_out_gain = if any_output_mod && mod_depth_output[i] > 0.0 {
                let lfo_o = lfo_value(cur_phase_output[i], mod_waveform);
                (output_gains_base[i] * (1.0 + mod_depth_output[i] * lfo_o)).max(0.0)
            } else {
                output_gains_base[i]
            };

            let tap = reads[i] * current_out_gain;
            wet_l += tap * pan_gain_l[i];
            wet_r += tap * pan_gain_r[i];
        }

        // Damping with modulated coefficients
        for i in 0..N {
            let current_damp = if any_damping_mod && mod_depth_damping[i] > 0.0 {
                let lfo_da = lfo_value(cur_phase_damping[i], mod_waveform);
                (damping_coeffs_base[i] + mod_depth_damping[i] * lfo_da).clamp(0.0, 0.999)
            } else {
                damping_coeffs_base[i]
            };
            damping_y1[i] = (1.0 - current_damp) * reads[i] + current_damp * damping_y1[i];
            reads[i] = damping_y1[i];
        }

        // Advance LFO phases
        for i in 0..N {
            cur_phase_delay[i] = (cur_phase_delay[i] + phase_inc_delay[i]) % 1.0;
            cur_phase_damping[i] = (cur_phase_damping[i] + phase_inc_damping[i]) % 1.0;
            cur_phase_output[i] = (cur_phase_output[i] + phase_inc_output[i]) % 1.0;
        }

        // Feedback matrix multiply (with optional blending)
        if mat_blend > 0.0 {
            let inv_blend = 1.0 - mat_blend;
            for i in 0..N {
                let mut s = 0.0;
                for j in 0..N {
                    let m = mat[i * N + j] * inv_blend + mat2[i * N + j] * mat_blend;
                    s += m * reads[j];
                }
                mixed[i] = s;
            }
        } else if matrix_type_flag_is_householder {
            let mut s = 0.0;
            for i in 0..N {
                s += reads[i];
            }
            s *= 2.0 / N as f64;
            for i in 0..N {
                mixed[i] = reads[i] - s;
            }
        } else {
            for i in 0..N {
                let mut s = 0.0;
                for j in 0..N {
                    s += mat[i * N + j] * reads[j];
                }
                mixed[i] = s;
            }
        }

        // Write back (with saturation + DC blocker)
        for i in 0..N {
            let wi = delay_write_idxs[i];
            let mut val = feedback_gain * mixed[i] + input_gains[i] * diffused;
            if saturation > 0.0 {
                val = (1.0 - saturation) * val + saturation * val.tanh();
            }
            let dc_y = val - dc_x1[i] + dc_r * dc_y1[i];
            dc_x1[i] = val;
            dc_y1[i] = dc_y;
            delay_bufs[i][wi] = dc_y;
            delay_write_idxs[i] = (wi + 1) % delay_buf_len;
        }

        // Wet/dry mix (stereo interleaved)
        output[n * 2] = dry_gain * x + wet_dry * wet_l;
        output[n * 2 + 1] = dry_gain * x + wet_dry * wet_r;
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lfo_sine_range() {
        for i in 0..100 {
            let phase = i as f64 / 100.0;
            let v = lfo_value(phase, 0);
            assert!(v >= -1.0 && v <= 1.0, "Sine LFO out of range: {v}");
        }
    }

    #[test]
    fn test_lfo_triangle_range() {
        for i in 0..100 {
            let phase = i as f64 / 100.0;
            let v = lfo_value(phase, 1);
            assert!(v >= -1.0 && v <= 1.0, "Triangle LFO out of range: {v} at phase {phase}");
        }
    }

    #[test]
    fn test_lfo_sah_range() {
        for i in 0..100 {
            let phase = i as f64 / 100.0;
            let v = lfo_value(phase, 2);
            assert!(v >= -1.0 && v <= 1.0, "S&H LFO out of range: {v}");
        }
    }

    #[test]
    fn test_modulated_impulse() {
        let mut input = vec![0.0; 44100];
        input[0] = 1.0;
        let mut params = ReverbParams::default();
        params.mod_master_rate = 2.0;
        params.mod_depth_delay = vec![5.0; N];
        let output = render_fdn_mod(&input, &params);
        assert_eq!(output.len(), 44100 * 2);
        let energy: f64 = output.iter().map(|s| s * s).sum();
        assert!(energy > 0.01, "Modulated impulse response should have energy");
        for &s in &output {
            assert!(s.is_finite(), "Output contains non-finite value");
        }
    }

    #[test]
    fn test_matrix_blend() {
        let mut input = vec![0.0; 44100];
        input[0] = 1.0;
        let mut params = ReverbParams::default();
        params.mod_master_rate = 1.0;
        params.mod_depth_matrix = 0.5;
        params.mod_rate_matrix = 1.0;
        let output = render_fdn_mod(&input, &params);
        for &s in &output {
            assert!(s.is_finite(), "Output contains non-finite value");
        }
    }
}
