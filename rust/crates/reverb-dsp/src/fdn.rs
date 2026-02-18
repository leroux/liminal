//! Static (non-modulated) FDN inner loop.
//!
//! Port of `reverb/engine/numba_fdn.py`.

use crate::matrix;
use crate::params::{ReverbParams, SR, N};

/// Render mono input through the static FDN, returning stereo (interleaved L,R pairs).
///
/// Output length = input_len * 2 (interleaved stereo).
pub fn render_fdn_static(input: &[f64], params: &ReverbParams) -> Vec<f64> {
    let n_samples = input.len();

    // --- Build feedback matrix ---
    let (mat, is_householder) = build_matrix(params);

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

    // --- FDN delay lines ---
    let delay_times: Vec<usize> = params.delay_times.iter().map(|&d| d as usize).collect();
    let delay_buf_len = delay_times.iter().copied().max().unwrap_or(1) + 1;
    let mut delay_bufs = vec![vec![0.0; delay_buf_len]; N];
    let mut delay_write_idxs = vec![0usize; N];

    // --- Damping ---
    let damping_coeffs = &params.damping_coeffs;
    let mut damping_y1 = vec![0.0; N];

    // --- Gains ---
    let feedback_gain = params.feedback_gain;
    let input_gains = &params.input_gains;
    let output_gains = &params.output_gains;
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
        let angle = (pan * params.stereo_width + 1.0) * (std::f64::consts::FRAC_PI_4);
        pan_gain_l[i] = angle.cos();
        pan_gain_r[i] = angle.sin();
    }

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

        // Read from delay lines + output taps
        let mut wet_l = 0.0;
        let mut wet_r = 0.0;
        for i in 0..N {
            let wi = delay_write_idxs[i];
            let rd = (wi + delay_buf_len - 1 - delay_times[i]) % delay_buf_len;
            reads[i] = delay_bufs[i][rd];
            let tap = reads[i] * output_gains[i];
            wet_l += tap * pan_gain_l[i];
            wet_r += tap * pan_gain_r[i];
        }

        // Damping (one-pole lowpass)
        for i in 0..N {
            let a = damping_coeffs[i];
            damping_y1[i] = (1.0 - a) * reads[i] + a * damping_y1[i];
            reads[i] = damping_y1[i];
        }

        // Feedback matrix multiply
        if is_householder {
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

        // Write back to delay lines (with saturation + DC blocker)
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

/// Build the feedback matrix from params. Returns (flat_matrix, is_householder).
pub fn build_matrix(params: &ReverbParams) -> (Vec<f64>, bool) {
    if params.matrix_type == "custom" {
        if let Some(ref custom) = params.matrix_custom {
            let mut flat = Vec::with_capacity(N * N);
            for row in custom {
                for &v in row {
                    flat.push(v);
                }
            }
            // Pad if needed
            while flat.len() < N * N {
                flat.push(0.0);
            }
            return (flat, false);
        }
    }

    let mat = matrix::get_matrix(&params.matrix_type, N, params.matrix_seed);
    let is_hh = params.matrix_type == "householder";
    (mat, is_hh)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_silence_in_silence_out() {
        let input = vec![0.0; 1000];
        let params = ReverbParams::default();
        let output = render_fdn_static(&input, &params);
        assert_eq!(output.len(), 2000);
        for &s in &output {
            assert!(s.abs() < 1e-10, "Expected silence, got {s}");
        }
    }

    #[test]
    fn test_impulse_response() {
        let mut input = vec![0.0; 44100];
        input[0] = 1.0;
        let params = ReverbParams::default();
        let output = render_fdn_static(&input, &params);
        assert_eq!(output.len(), 44100 * 2);
        // Should have some energy after the impulse
        let energy: f64 = output.iter().map(|s| s * s).sum();
        assert!(energy > 0.01, "Impulse response should have energy, got {energy}");
        // Should decay (energy in last quarter < first quarter)
        let q_len = 44100 / 2; // stereo samples per quarter
        let e_first: f64 = output[..q_len].iter().map(|s| s * s).sum();
        let e_last: f64 = output[output.len() - q_len..].iter().map(|s| s * s).sum();
        assert!(e_last < e_first, "Reverb should decay: first={e_first}, last={e_last}");
    }

    #[test]
    fn test_bypass_silence() {
        let mut input = vec![0.0; 1000];
        input[0] = 1.0;
        let mut params = ReverbParams::default();
        params.feedback_gain = 0.0;
        params.wet_dry = 0.0;
        let output = render_fdn_static(&input, &params);
        // With wet_dry=0, output should be dry only
        assert!((output[0] - 1.0).abs() < 1e-10); // Left
        assert!((output[1] - 1.0).abs() < 1e-10); // Right
    }

    #[test]
    fn test_output_is_finite() {
        let mut input = vec![0.0; 44100];
        input[0] = 1.0;
        let mut params = ReverbParams::default();
        params.feedback_gain = 1.5; // high feedback
        params.saturation = 0.8;    // saturation should keep it bounded
        let output = render_fdn_static(&input, &params);
        for &s in &output {
            assert!(s.is_finite(), "Output contains non-finite value");
        }
    }
}
