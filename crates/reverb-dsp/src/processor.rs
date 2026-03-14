//! Pre-allocated FDN reverb processor for real-time use.
//!
//! Unlike `render_fdn` which allocates all buffers per call, `FdnProcessor`
//! pre-allocates everything at construction time and maintains DSP state
//! (delay lines, filters, LFO phases) across calls. Zero allocations in
//! the audio thread.
//!
//! Improvements over the allocating path:
//! - Thiran allpass interpolation for modulated delays (no HF loss)
//! - Early reflections (ER) multi-tap delay line
//! - Parameter smoothing (~5ms time constant) to prevent zipper noise
//! - Feedback gain safety clamp when saturation is too low
//! - Output EQ (high-cut + low-cut biquads on wet signal)

use crate::matrix;
use crate::params::{FdnParams, N, N_ER_TAPS};

// Maximum buffer sizes — sized for up to 192kHz sample rate.
// Pre-delay: max 250ms @ 192kHz = 48000 samples.
const MAX_PRE_DELAY: usize = 48100;
// Delay lines: max 300ms @ 192kHz = 57600 + 100 mod excursion + margin.
const MAX_DELAY: usize = 58000;
const MAX_DIFF_STAGES: usize = 4;
// Diffusion: max 16.1ms @ 192kHz = ~3100 samples + margin.
const MAX_DIFF_DELAY: usize = 3200;
// Early reflections: max ER delay ≈ 0.96 * max_delay_time.
const MAX_ER_DELAY: usize = 19500;

/// Exponential smoothing coefficient for ~5ms time constant at 44.1kHz.
#[inline]
fn smooth_coeff(sample_rate: f64) -> f64 {
    (-1.0 / (0.005 * sample_rate)).exp()
}

// ---------------------------------------------------------------------------
// LFO
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Thiran allpass fractional delay
// ---------------------------------------------------------------------------

/// Read from a delay line with Thiran first-order allpass interpolation.
/// Flat magnitude response — no HF loss during modulation.
#[inline(always)]
fn read_delay_frac(
    buf: &[f64],
    write_idx: usize,
    delay_frac: f64,
    buf_len: usize,
    ap_state: &mut f64,
) -> f64 {
    let delay_int = delay_frac as usize;
    let frac = delay_frac - delay_int as f64;

    let idx0 = (write_idx + buf_len - 1 - delay_int) % buf_len;
    let idx1 = (write_idx + buf_len - 2 - delay_int) % buf_len;
    let x0 = buf[idx0];
    let x1 = buf[idx1];

    if frac < 1e-10 {
        *ap_state = x0;
        x0
    } else {
        let eta = (1.0 - frac) / (1.0 + frac);
        let out = eta * x0 + x1 - eta * *ap_state;
        *ap_state = out;
        out
    }
}

// ---------------------------------------------------------------------------
// Biquad filter for output EQ
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct BiquadState {
    x1: f64,
    x2: f64,
    y1: f64,
    y2: f64,
}

impl BiquadState {
    fn new() -> Self {
        Self {
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }

    #[inline(always)]
    fn process(&mut self, x: f64, c: &BiquadCoeffs) -> f64 {
        let y = c.b0 * x + c.b1 * self.x1 + c.b2 * self.x2
            - c.a1 * self.y1
            - c.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }
}

#[derive(Clone)]
struct BiquadCoeffs {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
}

impl BiquadCoeffs {
    fn bypass() -> Self {
        Self {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
        }
    }
}

/// Compute 2nd-order Butterworth lowpass coefficients.
fn compute_lowpass_coeffs(freq: f64, sr: f64) -> BiquadCoeffs {
    if freq >= sr * 0.499 {
        return BiquadCoeffs::bypass();
    }
    let w0 = 2.0 * std::f64::consts::PI * freq / sr;
    let cos_w0 = w0.cos();
    let sin_w0 = w0.sin();
    let alpha = sin_w0 / std::f64::consts::SQRT_2; // Q = 1/sqrt(2) for Butterworth
    let a0 = 1.0 + alpha;
    BiquadCoeffs {
        b0: ((1.0 - cos_w0) / 2.0) / a0,
        b1: (1.0 - cos_w0) / a0,
        b2: ((1.0 - cos_w0) / 2.0) / a0,
        a1: (-2.0 * cos_w0) / a0,
        a2: (1.0 - alpha) / a0,
    }
}

/// Compute 2nd-order Butterworth highpass coefficients.
fn compute_highpass_coeffs(freq: f64, sr: f64) -> BiquadCoeffs {
    if freq <= 1.0 {
        return BiquadCoeffs::bypass();
    }
    let w0 = 2.0 * std::f64::consts::PI * freq / sr;
    let cos_w0 = w0.cos();
    let sin_w0 = w0.sin();
    let alpha = sin_w0 / std::f64::consts::SQRT_2;
    let a0 = 1.0 + alpha;
    BiquadCoeffs {
        b0: ((1.0 + cos_w0) / 2.0) / a0,
        b1: (-(1.0 + cos_w0)) / a0,
        b2: ((1.0 + cos_w0) / 2.0) / a0,
        a1: (-2.0 * cos_w0) / a0,
        a2: (1.0 - alpha) / a0,
    }
}

// ---------------------------------------------------------------------------
// FdnProcessor
// ---------------------------------------------------------------------------

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

    // Thiran allpass interpolation state (for modulated delays)
    allpass_state: [f64; N],

    // Early reflections delay buffer
    er_buf: Vec<f64>,
    er_wi: usize,

    // Parameter smoothing state (feedback + saturation only — they're in the loop)
    smooth_feedback: f64,
    smooth_saturation: f64,

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
        let defaults = FdnParams::default();
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
            allpass_state: [0.0; N],
            er_buf: vec![0.0; MAX_ER_DELAY],
            er_wi: 0,
            smooth_feedback: defaults.feedback_gain,
            smooth_saturation: defaults.saturation,
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
        self.allpass_state = [0.0; N];
        self.er_buf.fill(0.0);
        self.er_wi = 0;
        self.phase_delay = [0.0; N];
        self.phase_damping = [0.0; N];
        self.phase_output = [0.0; N];
        self.phase_matrix = 0.0;
        let defaults = FdnParams::default();
        self.smooth_feedback = defaults.feedback_gain;
        self.smooth_saturation = defaults.saturation;
    }

    /// Rebuild cached matrices only when type/seed actually changes.
    fn update_matrix(&mut self, params: &FdnParams) {
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

    fn update_matrix2(&mut self, params: &FdnParams) {
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

    /// Apply feedback safety clamp: when feedback > 1.0 and saturation is too low,
    /// soft-limit feedback to prevent blowup.
    #[inline]
    fn safe_feedback(feedback: f64, saturation: f64) -> f64 {
        if feedback > 1.0 && saturation < 0.01 {
            1.0 + (feedback - 1.0) * (0.5_f64).powf(feedback - 1.0)
        } else {
            feedback
        }
    }

    /// Process mono input, writing interleaved stereo [L0,R0,L1,R1,...] to output.
    ///
    /// `output` must have length >= `input.len() * 2`.
    /// Applies wet/dry mixing. DSP state persists across calls.
    pub fn process(&mut self, input: &[f64], params: &FdnParams, output: &mut [f64]) {
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

    /// Process with wet_dry forced to 1.0 (avoids cloning params).
    /// Also sets the smoothed wet_dry to 1.0 to prevent ramp artifacts
    /// when used by StereoFdnProcessor (which does its own mixing).
    pub fn process_wet(&mut self, input: &[f64], params: &FdnParams, output: &mut [f64]) {
        let n_samples = input.len();
        debug_assert!(output.len() >= n_samples * 2);

        self.update_matrix(params);

        if params.has_modulation() {
            self.update_matrix2(params);
            self.process_modulated_inner(input, params, 1.0, output);
        } else {
            self.process_static_inner(input, params, 1.0, output);
        }
    }

    fn process_static(&mut self, input: &[f64], params: &FdnParams, output: &mut [f64]) {
        self.process_static_inner(input, params, params.wet_dry, output);
    }

    fn process_static_inner(
        &mut self,
        input: &[f64],
        params: &FdnParams,
        wet_dry: f64,
        output: &mut [f64],
    ) {
        let n_samples = input.len();
        let sr = params.sample_rate;
        let sc = smooth_coeff(sr);

        // Extract params into locals
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

        let target_feedback = params.feedback_gain;
        let target_saturation = params.saturation;
        let dry_gain = 1.0 - wet_dry;

        let dc_r: f64 = 1.0 - 2.0 * std::f64::consts::PI * 5.0 / sr;

        // ER params
        let er_level = params.er_level;
        let er_buf_len = MAX_ER_DELAY;
        let n_er_taps = params.er_delays.len().min(N_ER_TAPS);
        let mut er_delays = [0usize; N_ER_TAPS];
        let mut er_gains = [0.0_f64; N_ER_TAPS];
        let mut er_pan_l = [0.0_f64; N_ER_TAPS];
        let mut er_pan_r = [0.0_f64; N_ER_TAPS];
        if er_level > 0.0 {
            for i in 0..n_er_taps {
                er_delays[i] = (params.er_delays[i] as usize).min(MAX_ER_DELAY - 2);
                er_gains[i] = params.er_gains.get(i).copied().unwrap_or(0.3);
                let pan = params.er_pans.get(i).copied().unwrap_or(0.0);
                let angle = (pan + 1.0) * std::f64::consts::FRAC_PI_4;
                er_pan_l[i] = angle.cos();
                er_pan_r[i] = angle.sin();
            }
        }

        // Hoist per-node gains to stack arrays
        let mut output_gains = [1.0_f64; N];
        let mut damping_coeffs = [0.3_f64; N];
        let mut input_gains = [1.0 / N as f64; N];
        let mut pan_l = [0.0; N];
        let mut pan_r = [0.0; N];
        for i in 0..N {
            output_gains[i] = params.output_gains.get(i).copied().unwrap_or(1.0);
            damping_coeffs[i] = params.damping_coeffs.get(i).copied().unwrap_or(0.3);
            input_gains[i] = params.input_gains.get(i).copied().unwrap_or(1.0 / N as f64);
            let pan = params.node_pans.get(i).copied().unwrap_or(0.0);
            let angle = (pan * params.stereo_width + 1.0) * std::f64::consts::FRAC_PI_4;
            pan_l[i] = angle.cos();
            pan_r[i] = angle.sin();
        }

        let mut reads = [0.0; N];
        let mut mixed = [0.0; N];

        for n in 0..n_samples {
            let x = input[n];

            // Parameter smoothing
            self.smooth_feedback = sc * self.smooth_feedback + (1.0 - sc) * target_feedback;
            self.smooth_saturation = sc * self.smooth_saturation + (1.0 - sc) * target_saturation;

            let feedback_gain = Self::safe_feedback(self.smooth_feedback, self.smooth_saturation);
            let saturation = self.smooth_saturation;

            // Pre-delay
            self.pre_delay_buf[self.pd_wi] = x;
            self.pd_wi = (self.pd_wi + 1) % pre_delay_len;
            let rd_idx = (self.pd_wi + pre_delay_len - 1 - pre_delay_samples) % pre_delay_len;
            let x_delayed = self.pre_delay_buf[rd_idx];

            // Write to ER buffer (before diffusion, from pre-delayed signal)
            if er_level > 0.0 {
                self.er_buf[self.er_wi] = x_delayed;
                self.er_wi = (self.er_wi + 1) % er_buf_len;
            }

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
            let mut wl = 0.0;
            let mut wr = 0.0;
            for i in 0..N {
                let wi = self.delay_write_idxs[i];
                let rd = (wi + delay_buf_len - 1 - delay_times[i]) % delay_buf_len;
                reads[i] = self.delay_bufs[i][rd];
                let tap = reads[i] * output_gains[i];
                wl += tap * pan_l[i];
                wr += tap * pan_r[i];
            }

            // Early reflections tap reads
            if er_level > 0.0 {
                for i in 0..n_er_taps {
                    let er_rd = (self.er_wi + er_buf_len - 1 - er_delays[i]) % er_buf_len;
                    let er_tap = self.er_buf[er_rd] * er_gains[i] * er_level;
                    wl += er_tap * er_pan_l[i];
                    wr += er_tap * er_pan_r[i];
                }
            }

            // Damping
            for i in 0..N {
                self.damping_y1[i] =
                    (1.0 - damping_coeffs[i]) * reads[i] + damping_coeffs[i] * self.damping_y1[i];
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
                let mut val = feedback_gain * mixed[i] + input_gains[i] * diffused;
                if saturation > 0.0 {
                    val = (1.0 - saturation) * val + saturation * val.tanh();
                }
                let dc_y = val - self.dc_x1[i] + dc_r * self.dc_y1[i];
                self.dc_x1[i] = val;
                self.dc_y1[i] = dc_y;
                self.delay_bufs[i][wi] = dc_y;
                self.delay_write_idxs[i] = (wi + 1) % delay_buf_len;
            }

            output[n * 2] = dry_gain * x + wet_dry * wl;
            output[n * 2 + 1] = dry_gain * x + wet_dry * wr;
        }
    }

    fn process_modulated(&mut self, input: &[f64], params: &FdnParams, output: &mut [f64]) {
        self.process_modulated_inner(input, params, params.wet_dry, output);
    }

    fn process_modulated_inner(
        &mut self,
        input: &[f64],
        params: &FdnParams,
        wet_dry: f64,
        output: &mut [f64],
    ) {
        let n_samples = input.len();
        let sr = params.sample_rate;
        let sc = smooth_coeff(sr);

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

        let target_feedback = params.feedback_gain;
        let target_saturation = params.saturation;
        let dry_gain = 1.0 - wet_dry;

        let dc_r: f64 = 1.0 - 2.0 * std::f64::consts::PI * 5.0 / sr;

        // ER params
        let er_level = params.er_level;
        let er_buf_len = MAX_ER_DELAY;
        let n_er_taps = params.er_delays.len().min(N_ER_TAPS);
        let mut er_delays = [0usize; N_ER_TAPS];
        let mut er_gains = [0.0_f64; N_ER_TAPS];
        let mut er_pan_l = [0.0_f64; N_ER_TAPS];
        let mut er_pan_r = [0.0_f64; N_ER_TAPS];
        if er_level > 0.0 {
            for i in 0..n_er_taps {
                er_delays[i] = (params.er_delays[i] as usize).min(MAX_ER_DELAY - 2);
                er_gains[i] = params.er_gains.get(i).copied().unwrap_or(0.3);
                let pan = params.er_pans.get(i).copied().unwrap_or(0.0);
                let angle = (pan + 1.0) * std::f64::consts::FRAC_PI_4;
                er_pan_l[i] = angle.cos();
                er_pan_r[i] = angle.sin();
            }
        }

        // Hoist per-node gains to stack arrays
        let mut base_output_gains = [1.0_f64; N];
        let mut base_damping_coeffs = [0.3_f64; N];
        let mut input_gains = [1.0 / N as f64; N];
        let mut pan_l = [0.0; N];
        let mut pan_r = [0.0; N];
        for i in 0..N {
            base_output_gains[i] = params.output_gains.get(i).copied().unwrap_or(1.0);
            base_damping_coeffs[i] = params.damping_coeffs.get(i).copied().unwrap_or(0.3);
            input_gains[i] = params.input_gains.get(i).copied().unwrap_or(1.0 / N as f64);
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
            phase_inc_delay[i] = master_rate * node_mult * params.mod_rate_scale_delay / sr;
            phase_inc_damping[i] = master_rate * node_mult * params.mod_rate_scale_damping / sr;
            phase_inc_output[i] = master_rate * node_mult * params.mod_rate_scale_output / sr;
        }
        let phase_inc_matrix = params.mod_rate_matrix / sr;

        let any_delay_mod = mod_depth_delay.iter().any(|&d| d > 0.0);
        let any_damping_mod = mod_depth_damping.iter().any(|&d| d > 0.0);
        let any_output_mod = mod_depth_output.iter().any(|&d| d > 0.0);

        let mut reads = [0.0; N];
        let mut mixed = [0.0; N];

        let is_householder = self.is_householder;

        let mut blended_mat = [0.0_f64; N * N];

        for n in 0..n_samples {
            let x = input[n];

            // Parameter smoothing
            self.smooth_feedback = sc * self.smooth_feedback + (1.0 - sc) * target_feedback;
            self.smooth_saturation = sc * self.smooth_saturation + (1.0 - sc) * target_saturation;

            let feedback_gain = Self::safe_feedback(self.smooth_feedback, self.smooth_saturation);
            let saturation = self.smooth_saturation;

            // Pre-delay
            self.pre_delay_buf[self.pd_wi] = x;
            self.pd_wi = (self.pd_wi + 1) % pre_delay_len;
            let rd_idx = (self.pd_wi + pre_delay_len - 1 - pre_delay_samples) % pre_delay_len;
            let x_delayed = self.pre_delay_buf[rd_idx];

            // Write to ER buffer
            if er_level > 0.0 {
                self.er_buf[self.er_wi] = x_delayed;
                self.er_wi = (self.er_wi + 1) % er_buf_len;
            }

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

            // Read from delay lines (with Thiran allpass fractional delay)
            let mut wl = 0.0;
            let mut wr = 0.0;
            for i in 0..N {
                let wi = self.delay_write_idxs[i];

                let current_delay = if any_delay_mod && mod_depth_delay[i] > 0.0 {
                    let lfo_d = lfo_value(self.phase_delay[i], mod_waveform);
                    (delay_times_base[i] + mod_depth_delay[i] * lfo_d).max(1.0)
                } else {
                    delay_times_base[i]
                };

                reads[i] = read_delay_frac(
                    &self.delay_bufs[i],
                    wi,
                    current_delay,
                    delay_buf_len,
                    &mut self.allpass_state[i],
                );

                let current_out_gain = if any_output_mod && mod_depth_output[i] > 0.0 {
                    let lfo_o = lfo_value(self.phase_output[i], mod_waveform);
                    (base_output_gains[i] * (1.0 + mod_depth_output[i] * lfo_o)).max(0.0)
                } else {
                    base_output_gains[i]
                };

                let tap = reads[i] * current_out_gain;
                wl += tap * pan_l[i];
                wr += tap * pan_r[i];
            }

            // ER tap reads
            if er_level > 0.0 {
                for i in 0..n_er_taps {
                    let er_rd = (self.er_wi + er_buf_len - 1 - er_delays[i]) % er_buf_len;
                    let er_tap = self.er_buf[er_rd] * er_gains[i] * er_level;
                    wl += er_tap * er_pan_l[i];
                    wr += er_tap * er_pan_r[i];
                }
            }

            // Damping with modulated coefficients
            for i in 0..N {
                let current_damp = if any_damping_mod && mod_depth_damping[i] > 0.0 {
                    let lfo_da = lfo_value(self.phase_damping[i], mod_waveform);
                    (base_damping_coeffs[i] + mod_depth_damping[i] * lfo_da).clamp(0.0, 0.999)
                } else {
                    base_damping_coeffs[i]
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
                for k in 0..N * N {
                    blended_mat[k] = self.mat[k] * inv_blend + self.mat2[k] * mat_blend;
                }
                for i in 0..N {
                    let mut s = 0.0;
                    for j in 0..N {
                        s += blended_mat[i * N + j] * reads[j];
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
                let mut val = feedback_gain * mixed[i] + input_gains[i] * diffused;
                if saturation > 0.0 {
                    val = (1.0 - saturation) * val + saturation * val.tanh();
                }
                let dc_y = val - self.dc_x1[i] + dc_r * self.dc_y1[i];
                self.dc_x1[i] = val;
                self.dc_y1[i] = dc_y;
                self.delay_bufs[i][wi] = dc_y;
                self.delay_write_idxs[i] = (wi + 1) % delay_buf_len;
            }

            output[n * 2] = dry_gain * x + wet_dry * wl;
            output[n * 2 + 1] = dry_gain * x + wet_dry * wr;
        }
    }
}

// ---------------------------------------------------------------------------
// StereoFdnProcessor
// ---------------------------------------------------------------------------

/// Stereo FDN processor — two mono FDN instances + output EQ + scratch buffers.
pub struct StereoFdnProcessor {
    fdn_l: FdnProcessor,
    fdn_r: FdnProcessor,
    /// Scratch buffer for left FDN interleaved stereo output.
    scratch_l: Vec<f64>,
    /// Scratch buffer for right FDN interleaved stereo output.
    scratch_r: Vec<f64>,
    // Output EQ biquads (wet signal only)
    hc_state_l: BiquadState,
    hc_state_r: BiquadState,
    lc_state_l: BiquadState,
    lc_state_r: BiquadState,
    hc_coeffs: BiquadCoeffs,
    lc_coeffs: BiquadCoeffs,
    prev_high_cut: f64,
    prev_low_cut: f64,
    prev_sr: f64,
}

impl StereoFdnProcessor {
    pub fn new() -> Self {
        Self {
            fdn_l: FdnProcessor::new(),
            fdn_r: FdnProcessor::new(),
            scratch_l: vec![0.0; 8192 * 2],
            scratch_r: vec![0.0; 8192 * 2],
            hc_state_l: BiquadState::new(),
            hc_state_r: BiquadState::new(),
            lc_state_l: BiquadState::new(),
            lc_state_r: BiquadState::new(),
            hc_coeffs: BiquadCoeffs::bypass(),
            lc_coeffs: BiquadCoeffs::bypass(),
            prev_high_cut: 20000.0,
            prev_low_cut: 20.0,
            prev_sr: 44100.0,
        }
    }

    pub fn reset(&mut self) {
        self.fdn_l.reset();
        self.fdn_r.reset();
        self.hc_state_l.reset();
        self.hc_state_r.reset();
        self.lc_state_l.reset();
        self.lc_state_r.reset();
    }

    /// Process stereo input, writing to pre-allocated output slices.
    ///
    /// Each FDN runs with wet_dry=1.0 internally, then output EQ is applied
    /// to the wet signal, and wet/dry mixing is applied here.
    pub fn process_stereo(
        &mut self,
        left: &[f64],
        right: &[f64],
        params: &FdnParams,
        out_l: &mut [f64],
        out_r: &mut [f64],
    ) {
        let n = left.len().min(right.len());
        let mix = params.wet_dry;
        let dry_gain = 1.0 - mix;
        let sr = params.sample_rate;

        // Update EQ coefficients if changed
        if params.wet_high_cut_hz != self.prev_high_cut || sr != self.prev_sr {
            self.hc_coeffs = compute_lowpass_coeffs(params.wet_high_cut_hz, sr);
            self.prev_high_cut = params.wet_high_cut_hz;
        }
        if params.wet_low_cut_hz != self.prev_low_cut || sr != self.prev_sr {
            self.lc_coeffs = compute_highpass_coeffs(params.wet_low_cut_hz, sr);
            self.prev_low_cut = params.wet_low_cut_hz;
        }
        self.prev_sr = sr;

        let eq_active = params.wet_high_cut_hz < 19999.0 || params.wet_low_cut_hz > 21.0;

        // Ensure scratch buffers are large enough (grows once, never shrinks)
        let stereo_len = n * 2;
        if self.scratch_l.len() < stereo_len {
            self.scratch_l.resize(stereo_len, 0.0);
            self.scratch_r.resize(stereo_len, 0.0);
        }

        // Process each channel with full wet
        self.fdn_l
            .process_wet(&left[..n], params, &mut self.scratch_l[..stereo_len]);
        self.fdn_r
            .process_wet(&right[..n], params, &mut self.scratch_r[..stereo_len]);

        // Mix: sum wet contributions from both channels, apply EQ, blend with dry
        for i in 0..n {
            let mut wl = self.scratch_l[i * 2] + self.scratch_r[i * 2];
            let mut wr = self.scratch_l[i * 2 + 1] + self.scratch_r[i * 2 + 1];

            // Output EQ on wet signal
            if eq_active {
                wl = self.hc_state_l.process(wl, &self.hc_coeffs);
                wl = self.lc_state_l.process(wl, &self.lc_coeffs);
                wr = self.hc_state_r.process(wr, &self.hc_coeffs);
                wr = self.lc_state_r.process(wr, &self.lc_coeffs);
            }

            out_l[i] = dry_gain * left[i] + mix * wl;
            out_r[i] = dry_gain * right[i] + mix * wr;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::SR;

    fn make_sine(len: usize) -> Vec<f64> {
        (0..len)
            .map(|i| (2.0 * std::f64::consts::PI * 440.0 * i as f64 / SR).sin())
            .collect()
    }

    #[test]
    fn test_processor_matches_render_fdn() {
        // Note: processor now has parameter smoothing, but since default params
        // match the initial smooth values, smoothing is a no-op → exact match.
        let mut input = vec![0.0; 4410];
        input[0] = 1.0;
        let params = FdnParams::default();

        let expected = crate::chain::render_fdn(&input, &params);

        let mut proc = FdnProcessor::new();
        let mut output = vec![0.0; input.len() * 2];
        let mut norm_params = params.clone();
        norm_params.normalize();
        proc.process(&input, &norm_params, &mut output);

        assert_eq!(expected.len(), output.len());
        for (i, (&a, &b)) in expected.iter().zip(output.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "Mismatch at sample {i}: expected {a}, got {b}"
            );
        }
    }

    #[test]
    fn test_processor_modulated_matches() {
        let mut input = vec![0.0; 4410];
        input[0] = 1.0;
        let mut params = FdnParams::default();
        params.mod_master_rate = 2.0;
        params.mod_depth_delay = vec![5.0; N];
        params.mod_depth_damping = vec![0.1; N];

        // Processor uses Thiran allpass, allocating path uses Thiran allpass too
        // (both updated). Should match.
        let expected = crate::chain::render_fdn(&input, &params);

        let mut proc = FdnProcessor::new();
        let mut output = vec![0.0; input.len() * 2];
        let mut norm_params = params.clone();
        norm_params.normalize();
        proc.process(&input, &norm_params, &mut output);

        assert_eq!(expected.len(), output.len());
        for (i, (&a, &b)) in expected.iter().zip(output.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
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
        let params = FdnParams::default();

        let (exp_l, exp_r) = crate::chain::render_fdn_stereo(&left, &right, &params);

        let mut proc = StereoFdnProcessor::new();
        let mut out_l = vec![0.0; left.len()];
        let mut out_r = vec![0.0; right.len()];
        let mut norm_params = params.clone();
        norm_params.normalize();
        proc.process_stereo(&left, &right, &norm_params, &mut out_l, &mut out_r);

        for (i, (&a, &b)) in exp_l.iter().zip(out_l.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "Left mismatch at {i}: expected {a}, got {b}"
            );
        }
        for (i, (&a, &b)) in exp_r.iter().zip(out_r.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "Right mismatch at {i}: expected {a}, got {b}"
            );
        }
    }

    #[test]
    fn test_processor_persistent_state() {
        let input = make_sine(1024);
        let params = FdnParams::default();
        let mut norm_params = params.clone();
        norm_params.normalize();

        let mut proc = FdnProcessor::new();
        let mut out1 = vec![0.0; 1024 * 2];
        let mut out2 = vec![0.0; 1024 * 2];

        proc.process(&input, &norm_params, &mut out1);
        proc.process(&input, &norm_params, &mut out2);

        let diff: f64 = out1
            .iter()
            .zip(out2.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 0.01, "Persistent state should affect second block");
    }

    #[test]
    fn test_processor_zero_alloc_in_process() {
        let input = make_sine(256);
        let params = FdnParams::default();
        let mut norm_params = params.clone();
        norm_params.normalize();

        let mut proc = FdnProcessor::new();
        let mut output = vec![0.0; 256 * 2];

        for _ in 0..1000 {
            proc.process(&input, &norm_params, &mut output);
        }

        assert!(output.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_feedback_safety_no_blowup() {
        let mut input = vec![0.0; 44100];
        input[0] = 1.0;
        let mut params = FdnParams::default();
        params.feedback_gain = 1.5;
        params.saturation = 0.0; // No saturation — safety clamp should engage
        params.normalize();

        let mut proc = FdnProcessor::new();
        // Snap smoothing to target so safety engages immediately
        proc.smooth_feedback = params.feedback_gain;
        proc.smooth_saturation = params.saturation;
        let mut output = vec![0.0; 44100 * 2];
        proc.process(&input, &params, &mut output);

        for &s in &output {
            assert!(s.is_finite(), "Output should be finite with safety clamp");
            assert!(s.abs() < 100.0, "Output should be bounded with safety clamp, got {s}");
        }
    }

    #[test]
    fn test_early_reflections() {
        let mut input = vec![0.0; 4410];
        input[0] = 1.0;
        let mut params = FdnParams::default();
        params.er_level = 0.5;
        params.normalize();

        let mut proc = FdnProcessor::new();
        proc.smooth_feedback = params.feedback_gain;
        proc.smooth_saturation = params.saturation;
        let mut output_er = vec![0.0; 4410 * 2];
        proc.process(&input, &params, &mut output_er);

        // With ER, output after pre-delay + ER taps should have more energy.
        // Pre-delay is ~441 samples, earliest ER tap ~92 samples → ~533 samples.
        // Output is interleaved stereo, so check indices 1000..3000 (samples 500-1500).
        let early_energy: f64 = output_er[1000..3000].iter().map(|s| s * s).sum();

        let mut proc2 = FdnProcessor::new();
        proc2.smooth_feedback = params.feedback_gain;
        proc2.smooth_saturation = params.saturation;
        let mut params_no_er = params.clone();
        params_no_er.er_level = 0.0;
        let mut output_no_er = vec![0.0; 4410 * 2];
        proc2.process(&input, &params_no_er, &mut output_no_er);

        let early_energy_no_er: f64 = output_no_er[1000..3000].iter().map(|s| s * s).sum();
        assert!(
            early_energy > early_energy_no_er,
            "ER should add early energy: with={early_energy}, without={early_energy_no_er}"
        );
    }

    #[test]
    fn test_output_eq() {
        let mut input = vec![0.0; 4410];
        input[0] = 1.0;
        let mut params = FdnParams::default();
        params.wet_high_cut_hz = 2000.0; // Low high-cut → darker sound
        params.normalize();

        let mut proc = StereoFdnProcessor::new();
        let mut out_l = vec![0.0; 4410];
        let mut out_r = vec![0.0; 4410];
        proc.process_stereo(&input, &input, &params, &mut out_l, &mut out_r);

        // Compare with bypass EQ
        let mut params_bypass = params.clone();
        params_bypass.wet_high_cut_hz = 20000.0;
        let mut proc2 = StereoFdnProcessor::new();
        let mut out_l2 = vec![0.0; 4410];
        let mut out_r2 = vec![0.0; 4410];
        proc2.process_stereo(&input, &input, &params_bypass, &mut out_l2, &mut out_r2);

        // EQ'd output should differ from bypass
        let diff: f64 = out_l.iter().zip(out_l2.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 0.01, "Output EQ should change the signal, diff={diff}");
    }
}
