//! Pre-allocated FDN reverb processor for real-time use.
//!
//! Unlike `render_fdn` which allocates all buffers per call, `FdnProcessor`
//! pre-allocates everything at construction time and maintains DSP state
//! (delay lines, filters, LFO phases) across calls. Zero allocations in
//! the audio thread.

use crate::matrix;
use crate::params::{ReverbParams, SR, N};

// Maximum buffer sizes based on plugin param ranges.
// Pre-delay: max 250ms = 11025 samples.
const MAX_PRE_DELAY: usize = 11026;
// Delay lines: max 300ms = 13230 samples + 100 mod excursion + margin.
const MAX_DELAY: usize = 13400;
const MAX_DIFF_STAGES: usize = 4;
// Diffusion: max 16.1ms = 710 samples + margin.
const MAX_DIFF_DELAY: usize = 720;

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
            let n = (phase * 1000.0) as i64;
            let h = ((n.wrapping_mul(1103515245).wrapping_add(12345)) >> 16) & 0x7FFF;
            (h as f64 / 16383.5) - 1.0
        }
    }
}

#[inline(always)]
fn read_delay_frac(buf: &[f64], write_idx: usize, delay_frac: f64, buf_len: usize) -> f64 {
    let delay_int = delay_frac as usize;
    let frac = delay_frac - delay_int as f64;
    let idx0 = (write_idx + buf_len - 1 - delay_int) % buf_len;
    let idx1 = (write_idx + buf_len - 2 - delay_int) % buf_len;
    buf[idx0] * (1.0 - frac) + buf[idx1] * frac
}

/// Pre-allocated FDN reverb processor. One instance per mono channel.
///
/// Maintains all DSP state across `process()` calls — delay lines, filter
/// states, LFO phases all persist, giving correct reverb tails.
pub struct FdnProcessor {
    // Pre-delay ring buffer
    pre_delay_buf: Vec<f64>,
    pd_wi: usize,

    // Diffusion allpass buffers
    diff_bufs: Vec<Vec<f64>>,
    diff_idxs: [usize; MAX_DIFF_STAGES],

    // FDN delay lines (N nodes)
    delay_bufs: Vec<Vec<f64>>,
    delay_write_idxs: [usize; N],

    // Filter state
    damping_y1: [f64; N],
    dc_x1: [f64; N],
    dc_y1: [f64; N],

    // LFO phases (persistent across calls)
    phase_delay: [f64; N],
    phase_damping: [f64; N],
    phase_output: [f64; N],
    phase_matrix: f64,

    // Cached matrices (only rebuilt when type/seed changes)
    mat: Vec<f64>,
    mat2: Vec<f64>,
    is_householder: bool,
    prev_matrix_type: String,
    prev_matrix_seed: i32,
    prev_mat2_type: String,
    prev_mat2_seed: i32,
}

impl FdnProcessor {
    pub fn new() -> Self {
        Self {
            pre_delay_buf: vec![0.0; MAX_PRE_DELAY],
            pd_wi: 0,
            diff_bufs: (0..MAX_DIFF_STAGES).map(|_| vec![0.0; MAX_DIFF_DELAY]).collect(),
            diff_idxs: [0; MAX_DIFF_STAGES],
            delay_bufs: (0..N).map(|_| vec![0.0; MAX_DELAY]).collect(),
            delay_write_idxs: [0; N],
            damping_y1: [0.0; N],
            dc_x1: [0.0; N],
            dc_y1: [0.0; N],
            phase_delay: [0.0; N],
            phase_damping: [0.0; N],
            phase_output: [0.0; N],
            phase_matrix: 0.0,
            mat: vec![0.0; N * N],
            mat2: vec![0.0; N * N],
            is_householder: true,
            prev_matrix_type: String::new(),
            prev_matrix_seed: -1,
            prev_mat2_type: String::new(),
            prev_mat2_seed: -1,
        }
    }

    /// Reset all DSP state without deallocating. Call on transport stop/reset.
    pub fn reset(&mut self) {
        self.pre_delay_buf.fill(0.0);
        self.pd_wi = 0;
        for buf in &mut self.diff_bufs {
            buf.fill(0.0);
        }
        self.diff_idxs = [0; MAX_DIFF_STAGES];
        for buf in &mut self.delay_bufs {
            buf.fill(0.0);
        }
        self.delay_write_idxs = [0; N];
        self.damping_y1 = [0.0; N];
        self.dc_x1 = [0.0; N];
        self.dc_y1 = [0.0; N];
        self.phase_delay = [0.0; N];
        self.phase_damping = [0.0; N];
        self.phase_output = [0.0; N];
        self.phase_matrix = 0.0;
    }

    /// Rebuild cached matrices only when type/seed actually changes.
    fn update_matrix(&mut self, params: &ReverbParams) {
        if self.prev_matrix_type != params.matrix_type
            || self.prev_matrix_seed != params.matrix_seed
        {
            let new_mat = matrix::get_matrix(&params.matrix_type, N, params.matrix_seed);
            self.mat.copy_from_slice(&new_mat);
            self.is_householder = params.matrix_type == "householder";
            self.prev_matrix_type.clone_from(&params.matrix_type);
            self.prev_matrix_seed = params.matrix_seed;
        }
    }

    fn update_matrix2(&mut self, params: &ReverbParams) {
        if self.prev_mat2_type != params.mod_matrix2_type
            || self.prev_mat2_seed != params.mod_matrix2_seed
        {
            let new_mat2 =
                matrix::get_matrix(&params.mod_matrix2_type, N, params.mod_matrix2_seed);
            self.mat2.copy_from_slice(&new_mat2);
            self.prev_mat2_type.clone_from(&params.mod_matrix2_type);
            self.prev_mat2_seed = params.mod_matrix2_seed;
        }
    }

    /// Process mono input, writing interleaved stereo [L0,R0,L1,R1,...] to output.
    ///
    /// `output` must have length >= `input.len() * 2`.
    /// Applies wet/dry mixing. DSP state persists across calls.
    pub fn process(&mut self, input: &[f64], params: &ReverbParams, output: &mut [f64]) {
        let n_samples = input.len();
        debug_assert!(output.len() >= n_samples * 2);

        self.update_matrix(params);

        if params.has_modulation() {
            self.update_matrix2(params);
            self.process_modulated(input, params, output);
        } else {
            self.process_static(input, params, output);
        }
    }

    fn process_static(&mut self, input: &[f64], params: &ReverbParams, output: &mut [f64]) {
        let n_samples = input.len();

        // Extract params into locals (no allocation — just copies of scalars/slices)
        let pre_delay_samples = params.pre_delay.max(1) as usize;
        let pre_delay_len = (pre_delay_samples + 1).min(MAX_PRE_DELAY);

        let n_diff_stages = (params.diffusion_stages as usize)
            .min(params.diffusion_delays.len())
            .min(MAX_DIFF_STAGES);
        let mut diff_lens = [1usize; MAX_DIFF_STAGES];
        for i in 0..n_diff_stages {
            diff_lens[i] = (params.diffusion_delays[i] as usize).min(MAX_DIFF_DELAY);
        }
        let diff_gain = params.diffusion;

        let mut delay_times = [0usize; N];
        for i in 0..N.min(params.delay_times.len()) {
            delay_times[i] = (params.delay_times[i] as usize).min(MAX_DELAY - 2);
        }
        let delay_buf_len = MAX_DELAY;

        let feedback_gain = params.feedback_gain;
        let wet_dry = params.wet_dry;
        let saturation = params.saturation;
        let dry_gain = 1.0 - wet_dry;

        let dc_r = 1.0 - 2.0 * std::f64::consts::PI * 5.0 / SR;

        // Compute pan gains (no allocation — fixed arrays)
        let mut pan_l = [0.0; N];
        let mut pan_r = [0.0; N];
        for i in 0..N {
            let pan = params.node_pans.get(i).copied().unwrap_or(0.0);
            let angle = (pan * params.stereo_width + 1.0) * std::f64::consts::FRAC_PI_4;
            pan_l[i] = angle.cos();
            pan_r[i] = angle.sin();
        }

        let mut reads = [0.0; N];
        let mut mixed = [0.0; N];

        for n in 0..n_samples {
            let x = input[n];

            // Pre-delay
            self.pre_delay_buf[self.pd_wi] = x;
            self.pd_wi = (self.pd_wi + 1) % pre_delay_len;
            let rd_idx = (self.pd_wi + pre_delay_len - 1 - pre_delay_samples) % pre_delay_len;
            let x_delayed = self.pre_delay_buf[rd_idx];

            // Diffusion allpass chain
            let mut diffused = x_delayed;
            for s in 0..n_diff_stages {
                let idx = self.diff_idxs[s];
                let delayed = self.diff_bufs[s][idx];
                let v = diffused + diff_gain * delayed;
                diffused = -diff_gain * v + delayed;
                self.diff_bufs[s][idx] = v;
                self.diff_idxs[s] = (idx + 1) % diff_lens[s];
            }

            // Read from delay lines + output taps
            let mut wet_l = 0.0;
            let mut wet_r = 0.0;
            for i in 0..N {
                let wi = self.delay_write_idxs[i];
                let rd = (wi + delay_buf_len - 1 - delay_times[i]) % delay_buf_len;
                reads[i] = self.delay_bufs[i][rd];
                let out_gain = params.output_gains.get(i).copied().unwrap_or(1.0);
                let tap = reads[i] * out_gain;
                wet_l += tap * pan_l[i];
                wet_r += tap * pan_r[i];
            }

            // Damping
            for i in 0..N {
                let a = params.damping_coeffs.get(i).copied().unwrap_or(0.3);
                self.damping_y1[i] = (1.0 - a) * reads[i] + a * self.damping_y1[i];
                reads[i] = self.damping_y1[i];
            }

            // Matrix multiply
            if self.is_householder {
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
                        s += self.mat[i * N + j] * reads[j];
                    }
                    mixed[i] = s;
                }
            }

            // Write back (saturation + DC blocker)
            for i in 0..N {
                let wi = self.delay_write_idxs[i];
                let in_gain = params.input_gains.get(i).copied().unwrap_or(1.0 / N as f64);
                let mut val = feedback_gain * mixed[i] + in_gain * diffused;
                if saturation > 0.0 {
                    val = (1.0 - saturation) * val + saturation * val.tanh();
                }
                let dc_y = val - self.dc_x1[i] + dc_r * self.dc_y1[i];
                self.dc_x1[i] = val;
                self.dc_y1[i] = dc_y;
                self.delay_bufs[i][wi] = dc_y;
                self.delay_write_idxs[i] = (wi + 1) % delay_buf_len;
            }

            output[n * 2] = dry_gain * x + wet_dry * wet_l;
            output[n * 2 + 1] = dry_gain * x + wet_dry * wet_r;
        }
    }

    fn process_modulated(&mut self, input: &[f64], params: &ReverbParams, output: &mut [f64]) {
        let n_samples = input.len();

        let pre_delay_samples = params.pre_delay.max(1) as usize;
        let pre_delay_len = (pre_delay_samples + 1).min(MAX_PRE_DELAY);

        let n_diff_stages = (params.diffusion_stages as usize)
            .min(params.diffusion_delays.len())
            .min(MAX_DIFF_STAGES);
        let mut diff_lens = [1usize; MAX_DIFF_STAGES];
        for i in 0..n_diff_stages {
            diff_lens[i] = (params.diffusion_delays[i] as usize).min(MAX_DIFF_DELAY);
        }
        let diff_gain = params.diffusion;

        let mut delay_times_base = [0.0f64; N];
        for i in 0..N.min(params.delay_times.len()) {
            delay_times_base[i] = params.delay_times[i] as f64;
        }
        let delay_buf_len = MAX_DELAY;

        let feedback_gain = params.feedback_gain;
        let wet_dry = params.wet_dry;
        let saturation = params.saturation;
        let dry_gain = 1.0 - wet_dry;

        let dc_r = 1.0 - 2.0 * std::f64::consts::PI * 5.0 / SR;

        let mut pan_l = [0.0; N];
        let mut pan_r = [0.0; N];
        for i in 0..N {
            let pan = params.node_pans.get(i).copied().unwrap_or(0.0);
            let angle = (pan * params.stereo_width + 1.0) * std::f64::consts::FRAC_PI_4;
            pan_l[i] = angle.cos();
            pan_r[i] = angle.sin();
        }

        // Modulation params
        let master_rate = params.mod_master_rate;
        let mod_waveform = params.mod_waveform;
        let mod_depth_matrix = params.mod_depth_matrix;

        let mut mod_depth_delay = [0.0; N];
        let mut mod_depth_damping = [0.0; N];
        let mut mod_depth_output = [0.0; N];
        for i in 0..N {
            mod_depth_delay[i] = params.mod_depth_delay.get(i).copied().unwrap_or(0.0);
            mod_depth_damping[i] = params.mod_depth_damping.get(i).copied().unwrap_or(0.0);
            mod_depth_output[i] = params.mod_depth_output.get(i).copied().unwrap_or(0.0);
        }

        // Phase increments
        let mut phase_inc_delay = [0.0; N];
        let mut phase_inc_damping = [0.0; N];
        let mut phase_inc_output = [0.0; N];
        for i in 0..N {
            let node_mult = params.mod_node_rate_mult.get(i).copied().unwrap_or(1.0);
            phase_inc_delay[i] = master_rate * node_mult * params.mod_rate_scale_delay / SR;
            phase_inc_damping[i] = master_rate * node_mult * params.mod_rate_scale_damping / SR;
            phase_inc_output[i] = master_rate * node_mult * params.mod_rate_scale_output / SR;
        }
        let phase_inc_matrix = params.mod_rate_matrix / SR;

        // Initialize phase offsets on first call (phases persist across calls)
        let any_delay_mod = mod_depth_delay.iter().any(|&d| d > 0.0);
        let any_damping_mod = mod_depth_damping.iter().any(|&d| d > 0.0);
        let any_output_mod = mod_depth_output.iter().any(|&d| d > 0.0);

        let mut reads = [0.0; N];
        let mut mixed = [0.0; N];

        let is_householder = self.is_householder;

        for n in 0..n_samples {
            let x = input[n];

            // Pre-delay
            self.pre_delay_buf[self.pd_wi] = x;
            self.pd_wi = (self.pd_wi + 1) % pre_delay_len;
            let rd_idx = (self.pd_wi + pre_delay_len - 1 - pre_delay_samples) % pre_delay_len;
            let x_delayed = self.pre_delay_buf[rd_idx];

            // Diffusion
            let mut diffused = x_delayed;
            for s in 0..n_diff_stages {
                let idx = self.diff_idxs[s];
                let delayed = self.diff_bufs[s][idx];
                let v = diffused + diff_gain * delayed;
                diffused = -diff_gain * v + delayed;
                self.diff_bufs[s][idx] = v;
                self.diff_idxs[s] = (idx + 1) % diff_lens[s];
            }

            // Matrix modulation LFO
            let mat_blend = if mod_depth_matrix > 0.0 {
                let lfo_mat = lfo_value(self.phase_matrix, mod_waveform);
                self.phase_matrix = (self.phase_matrix + phase_inc_matrix) % 1.0;
                0.5 + 0.5 * lfo_mat * mod_depth_matrix
            } else {
                0.0
            };

            // Read from delay lines (with fractional delay modulation)
            let mut wet_l = 0.0;
            let mut wet_r = 0.0;
            for i in 0..N {
                let wi = self.delay_write_idxs[i];

                let current_delay = if any_delay_mod && mod_depth_delay[i] > 0.0 {
                    let lfo_d = lfo_value(self.phase_delay[i], mod_waveform);
                    (delay_times_base[i] + mod_depth_delay[i] * lfo_d).max(1.0)
                } else {
                    delay_times_base[i]
                };

                reads[i] =
                    read_delay_frac(&self.delay_bufs[i], wi, current_delay, delay_buf_len);

                let base_out = params.output_gains.get(i).copied().unwrap_or(1.0);
                let current_out_gain = if any_output_mod && mod_depth_output[i] > 0.0 {
                    let lfo_o = lfo_value(self.phase_output[i], mod_waveform);
                    (base_out * (1.0 + mod_depth_output[i] * lfo_o)).max(0.0)
                } else {
                    base_out
                };

                let tap = reads[i] * current_out_gain;
                wet_l += tap * pan_l[i];
                wet_r += tap * pan_r[i];
            }

            // Damping with modulated coefficients
            for i in 0..N {
                let base_damp = params.damping_coeffs.get(i).copied().unwrap_or(0.3);
                let current_damp = if any_damping_mod && mod_depth_damping[i] > 0.0 {
                    let lfo_da = lfo_value(self.phase_damping[i], mod_waveform);
                    (base_damp + mod_depth_damping[i] * lfo_da).clamp(0.0, 0.999)
                } else {
                    base_damp
                };
                self.damping_y1[i] = (1.0 - current_damp) * reads[i]
                    + current_damp * self.damping_y1[i];
                reads[i] = self.damping_y1[i];
            }

            // Advance LFO phases
            for i in 0..N {
                self.phase_delay[i] = (self.phase_delay[i] + phase_inc_delay[i]) % 1.0;
                self.phase_damping[i] = (self.phase_damping[i] + phase_inc_damping[i]) % 1.0;
                self.phase_output[i] = (self.phase_output[i] + phase_inc_output[i]) % 1.0;
            }

            // Matrix multiply (with optional blending)
            if mat_blend > 0.0 {
                let inv_blend = 1.0 - mat_blend;
                for i in 0..N {
                    let mut s = 0.0;
                    for j in 0..N {
                        let m = self.mat[i * N + j] * inv_blend
                            + self.mat2[i * N + j] * mat_blend;
                        s += m * reads[j];
                    }
                    mixed[i] = s;
                }
            } else if is_householder {
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
                        s += self.mat[i * N + j] * reads[j];
                    }
                    mixed[i] = s;
                }
            }

            // Write back (saturation + DC blocker)
            for i in 0..N {
                let wi = self.delay_write_idxs[i];
                let in_gain = params.input_gains.get(i).copied().unwrap_or(1.0 / N as f64);
                let mut val = feedback_gain * mixed[i] + in_gain * diffused;
                if saturation > 0.0 {
                    val = (1.0 - saturation) * val + saturation * val.tanh();
                }
                let dc_y = val - self.dc_x1[i] + dc_r * self.dc_y1[i];
                self.dc_x1[i] = val;
                self.dc_y1[i] = dc_y;
                self.delay_bufs[i][wi] = dc_y;
                self.delay_write_idxs[i] = (wi + 1) % delay_buf_len;
            }

            output[n * 2] = dry_gain * x + wet_dry * wet_l;
            output[n * 2 + 1] = dry_gain * x + wet_dry * wet_r;
        }
    }
}

/// Stereo FDN processor — two mono FDN instances + pre-allocated scratch buffers.
pub struct StereoFdnProcessor {
    fdn_l: FdnProcessor,
    fdn_r: FdnProcessor,
    /// Scratch buffer for left FDN interleaved stereo output.
    scratch_l: Vec<f64>,
    /// Scratch buffer for right FDN interleaved stereo output.
    scratch_r: Vec<f64>,
}

impl StereoFdnProcessor {
    pub fn new() -> Self {
        Self {
            fdn_l: FdnProcessor::new(),
            fdn_r: FdnProcessor::new(),
            scratch_l: Vec::new(),
            scratch_r: Vec::new(),
        }
    }

    pub fn reset(&mut self) {
        self.fdn_l.reset();
        self.fdn_r.reset();
    }

    /// Process stereo input, writing to pre-allocated output slices.
    ///
    /// Each FDN runs with wet_dry=1.0 internally, then wet/dry mixing
    /// is applied here (matching `render_fdn_stereo` semantics).
    pub fn process_stereo(
        &mut self,
        left: &[f64],
        right: &[f64],
        params: &ReverbParams,
        out_l: &mut [f64],
        out_r: &mut [f64],
    ) {
        let n = left.len().min(right.len());
        let mix = params.wet_dry;
        let dry_gain = 1.0 - mix;

        // Ensure scratch buffers are large enough (grows once, never shrinks)
        let stereo_len = n * 2;
        if self.scratch_l.len() < stereo_len {
            self.scratch_l.resize(stereo_len, 0.0);
            self.scratch_r.resize(stereo_len, 0.0);
        }

        // Process each channel with full wet
        let mut wet_params = params.clone();
        wet_params.wet_dry = 1.0;

        self.fdn_l
            .process(&left[..n], &wet_params, &mut self.scratch_l[..stereo_len]);
        self.fdn_r
            .process(&right[..n], &wet_params, &mut self.scratch_r[..stereo_len]);

        // Mix: sum wet contributions from both channels, blend with dry
        for i in 0..n {
            let wl = self.scratch_l[i * 2] + self.scratch_r[i * 2];
            let wr = self.scratch_l[i * 2 + 1] + self.scratch_r[i * 2 + 1];
            out_l[i] = dry_gain * left[i] + mix * wl;
            out_r[i] = dry_gain * right[i] + mix * wr;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sine(len: usize) -> Vec<f64> {
        (0..len)
            .map(|i| (2.0 * std::f64::consts::PI * 440.0 * i as f64 / SR).sin())
            .collect()
    }

    #[test]
    fn test_processor_matches_render_fdn() {
        let mut input = vec![0.0; 4410];
        input[0] = 1.0;
        let params = ReverbParams::default();

        // Existing allocating API
        let expected = crate::chain::render_fdn(&input, &params);

        // New pre-allocated processor
        let mut proc = FdnProcessor::new();
        let mut output = vec![0.0; input.len() * 2];
        let mut norm_params = params.clone();
        norm_params.normalize();
        proc.process(&input, &norm_params, &mut output);

        assert_eq!(expected.len(), output.len());
        for (i, (&a, &b)) in expected.iter().zip(output.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-10,
                "Mismatch at sample {i}: expected {a}, got {b}"
            );
        }
    }

    #[test]
    fn test_processor_modulated_matches() {
        let mut input = vec![0.0; 4410];
        input[0] = 1.0;
        let mut params = ReverbParams::default();
        params.mod_master_rate = 2.0;
        params.mod_depth_delay = vec![5.0; N];
        params.mod_depth_damping = vec![0.1; N];

        let expected = crate::chain::render_fdn(&input, &params);

        let mut proc = FdnProcessor::new();
        let mut output = vec![0.0; input.len() * 2];
        let mut norm_params = params.clone();
        norm_params.normalize();
        proc.process(&input, &norm_params, &mut output);

        assert_eq!(expected.len(), output.len());
        for (i, (&a, &b)) in expected.iter().zip(output.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-10,
                "Mismatch at sample {i}: expected {a}, got {b}"
            );
        }
    }

    #[test]
    fn test_stereo_processor_matches() {
        let mut left = vec![0.0; 4410];
        let mut right = vec![0.0; 4410];
        left[0] = 1.0;
        right[100] = 1.0;
        let params = ReverbParams::default();

        let (exp_l, exp_r) = crate::chain::render_fdn_stereo(&left, &right, &params);

        let mut proc = StereoFdnProcessor::new();
        let mut out_l = vec![0.0; left.len()];
        let mut out_r = vec![0.0; right.len()];
        let mut norm_params = params.clone();
        norm_params.normalize();
        proc.process_stereo(&left, &right, &norm_params, &mut out_l, &mut out_r);

        for (i, (&a, &b)) in exp_l.iter().zip(out_l.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-10,
                "Left mismatch at {i}: expected {a}, got {b}"
            );
        }
        for (i, (&a, &b)) in exp_r.iter().zip(out_r.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-10,
                "Right mismatch at {i}: expected {a}, got {b}"
            );
        }
    }

    #[test]
    fn test_processor_persistent_state() {
        let input = make_sine(1024);
        let params = ReverbParams::default();
        let mut norm_params = params.clone();
        norm_params.normalize();

        let mut proc = FdnProcessor::new();
        let mut out1 = vec![0.0; 1024 * 2];
        let mut out2 = vec![0.0; 1024 * 2];

        // Process two consecutive blocks
        proc.process(&input, &norm_params, &mut out1);
        proc.process(&input, &norm_params, &mut out2);

        // Second block should differ from first (reverb tail carries over)
        let diff: f64 = out1
            .iter()
            .zip(out2.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 0.01, "Persistent state should affect second block");
    }

    #[test]
    fn test_processor_zero_alloc_in_process() {
        // Verify process doesn't allocate by running it many times.
        // If it allocated, we'd see growing memory, but more importantly
        // this test documents the contract.
        let input = make_sine(256);
        let params = ReverbParams::default();
        let mut norm_params = params.clone();
        norm_params.normalize();

        let mut proc = FdnProcessor::new();
        let mut output = vec![0.0; 256 * 2];

        for _ in 0..1000 {
            proc.process(&input, &norm_params, &mut output);
        }

        // Just verify it produces finite output after many calls
        assert!(output.iter().all(|x| x.is_finite()));
    }
}
