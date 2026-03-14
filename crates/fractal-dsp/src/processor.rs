//! Block-buffering stereo fractal processor for real-time use.
//!
//! The fractal algorithm is inherently block-based (time-domain stretching,
//! layering, FFT-based spectral processing), so it cannot process sample-by-sample.
//! This processor handles input accumulation, block dispatch, feedback mixing,
//! and output draining — moving all that logic out of the plugin layer.
//!
//! Latency: `BLOCK_SIZE` samples (8192 = ~186ms at 44100 Hz).

use crate::chain::render_fractal_stereo;
use crate::params::FractalParams;

/// Processing block size. Input accumulates until this many samples arrive,
/// then the entire chain processes. At 44100 Hz, 8192 samples = ~186ms.
pub const BLOCK_SIZE: usize = 8192;

/// Pre-allocated stereo fractal processor with block buffering and feedback.
pub struct StereoFractalProcessor {
    /// Per-channel input accumulation buffers.
    input_buf: [Vec<f64>; 2],
    /// Per-channel output drain buffers (processed audio waiting to be sent).
    output_buf: [Vec<f64>; 2],
    /// Per-channel feedback buffers from previous block.
    feedback_buf: [Vec<f64>; 2],
    /// Write position into input_buf.
    in_pos: usize,
    /// Read position from output_buf.
    out_pos: usize,
    /// How many valid samples in output_buf from out_pos onward.
    out_avail: usize,
    /// True once we've filled the first block (latency compensation).
    primed: bool,
}

impl Default for StereoFractalProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl StereoFractalProcessor {
    pub fn new() -> Self {
        Self {
            input_buf: [vec![0.0; BLOCK_SIZE], vec![0.0; BLOCK_SIZE]],
            output_buf: [vec![0.0; BLOCK_SIZE], vec![0.0; BLOCK_SIZE]],
            feedback_buf: [vec![0.0; BLOCK_SIZE], vec![0.0; BLOCK_SIZE]],
            in_pos: 0,
            out_pos: 0,
            out_avail: 0,
            primed: false,
        }
    }

    /// Returns the latency in samples.
    pub fn latency(&self) -> usize {
        BLOCK_SIZE
    }

    /// Reset all state without deallocating.
    pub fn reset(&mut self) {
        for buf in &mut self.input_buf {
            buf.fill(0.0);
        }
        for buf in &mut self.output_buf {
            buf.fill(0.0);
        }
        for buf in &mut self.feedback_buf {
            buf.fill(0.0);
        }
        self.in_pos = 0;
        self.out_pos = 0;
        self.out_avail = 0;
        self.primed = false;
    }

    /// Process a stereo buffer. Reads from `left_in`/`right_in`, writes to `left_out`/`right_out`.
    /// All slices must have the same length. Outputs silence until the first full block is processed.
    pub fn process(
        &mut self,
        left_in: &[f64],
        right_in: &[f64],
        params: &FractalParams,
        left_out: &mut [f64],
        right_out: &mut [f64],
    ) {
        let num_samples = left_in.len();
        debug_assert_eq!(left_in.len(), right_in.len());
        debug_assert!(left_out.len() >= num_samples);
        debug_assert!(right_out.len() >= num_samples);

        for i in 0..num_samples {
            // Accumulate input
            self.input_buf[0][self.in_pos] = left_in[i];
            self.input_buf[1][self.in_pos] = right_in[i];
            self.in_pos += 1;

            // Fire when block is full
            if self.in_pos >= BLOCK_SIZE {
                self.process_block(params);
                self.in_pos = 0;
                self.primed = true;
            }

            // Drain output
            if self.primed && self.out_avail > 0 {
                left_out[i] = self.output_buf[0][self.out_pos];
                right_out[i] = self.output_buf[1][self.out_pos];
                self.out_pos += 1;
                self.out_avail -= 1;
            } else {
                left_out[i] = 0.0;
                right_out[i] = 0.0;
            }
        }
    }

    /// Process the accumulated block: mix feedback, run chain, store output + feedback.
    fn process_block(&mut self, params: &FractalParams) {
        let mut dsp_params = params.clone();
        let feedback = dsp_params.feedback.clamp(0.0, 0.95);

        // Mix feedback into input
        if feedback > 0.001 {
            for ch in 0..2 {
                for i in 0..BLOCK_SIZE {
                    self.input_buf[ch][i] += feedback * self.feedback_buf[ch][i];
                }
            }
            // Prevent double-application in DSP chain
            dsp_params.feedback = 0.0;
        }

        let (out_l, out_r) = render_fractal_stereo(
            &self.input_buf[0],
            &self.input_buf[1],
            &dsp_params,
        );

        // Store output as feedback for next block (copy, not clone)
        if feedback > 0.001 {
            self.feedback_buf[0][..BLOCK_SIZE].copy_from_slice(&out_l[..BLOCK_SIZE]);
            self.feedback_buf[1][..BLOCK_SIZE].copy_from_slice(&out_r[..BLOCK_SIZE]);
        }

        self.output_buf[0][..BLOCK_SIZE].copy_from_slice(&out_l[..BLOCK_SIZE]);
        self.output_buf[1][..BLOCK_SIZE].copy_from_slice(&out_r[..BLOCK_SIZE]);
        self.out_pos = 0;
        self.out_avail = BLOCK_SIZE;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::SR;

    fn make_stereo_sine(len: usize) -> (Vec<f64>, Vec<f64>) {
        let left: Vec<f64> = (0..len)
            .map(|i| (2.0 * std::f64::consts::PI * 440.0 * i as f64 / SR).sin())
            .collect();
        let right: Vec<f64> = (0..len)
            .map(|i| (2.0 * std::f64::consts::PI * 660.0 * i as f64 / SR).sin())
            .collect();
        (left, right)
    }

    #[test]
    fn test_outputs_silence_before_primed() {
        let mut proc = StereoFractalProcessor::new();
        let params = FractalParams::default();
        let n = 256;
        let left_in = vec![1.0; n];
        let right_in = vec![1.0; n];
        let mut left_out = vec![0.0; n];
        let mut right_out = vec![0.0; n];

        proc.process(&left_in, &right_in, &params, &mut left_out, &mut right_out);

        // Before a full block, output should be silence
        assert!(left_out.iter().all(|&x| x == 0.0));
        assert!(right_out.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_produces_output_after_full_block() {
        let mut proc = StereoFractalProcessor::new();
        let params = FractalParams::default();
        let (left_in, right_in) = make_stereo_sine(BLOCK_SIZE + 256);
        let mut left_out = vec![0.0; left_in.len()];
        let mut right_out = vec![0.0; right_in.len()];

        proc.process(&left_in, &right_in, &params, &mut left_out, &mut right_out);

        // After BLOCK_SIZE samples, the next 256 should have output
        let tail_l = &left_out[BLOCK_SIZE..];
        let has_output = tail_l.iter().any(|&x| x != 0.0);
        assert!(has_output, "Should produce output after first full block");
    }

    #[test]
    fn test_output_is_finite() {
        let mut proc = StereoFractalProcessor::new();
        let params = FractalParams::default();
        let (left_in, right_in) = make_stereo_sine(BLOCK_SIZE * 2);
        let mut left_out = vec![0.0; left_in.len()];
        let mut right_out = vec![0.0; right_in.len()];

        proc.process(&left_in, &right_in, &params, &mut left_out, &mut right_out);

        assert!(left_out.iter().all(|x| x.is_finite()));
        assert!(right_out.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_reset_clears_state() {
        let mut proc = StereoFractalProcessor::new();
        let params = FractalParams::default();
        let (left_in, right_in) = make_stereo_sine(BLOCK_SIZE * 2);
        let mut left_out = vec![0.0; left_in.len()];
        let mut right_out = vec![0.0; right_in.len()];

        // Process some audio
        proc.process(&left_in, &right_in, &params, &mut left_out, &mut right_out);
        assert!(proc.primed);

        // Reset
        proc.reset();
        assert!(!proc.primed);
        assert_eq!(proc.in_pos, 0);
        assert_eq!(proc.out_pos, 0);
        assert_eq!(proc.out_avail, 0);
    }

    #[test]
    fn test_incremental_processing() {
        // Process in small chunks vs one big chunk — should produce same output
        let mut proc_big = StereoFractalProcessor::new();
        let mut proc_small = StereoFractalProcessor::new();
        let params = FractalParams::default();
        let total = BLOCK_SIZE * 2;
        let (left_in, right_in) = make_stereo_sine(total);

        // Big: process all at once
        let mut big_l = vec![0.0; total];
        let mut big_r = vec![0.0; total];
        proc_big.process(&left_in, &right_in, &params, &mut big_l, &mut big_r);

        // Small: process in 256-sample chunks
        let mut small_l = vec![0.0; total];
        let mut small_r = vec![0.0; total];
        let chunk = 256;
        for start in (0..total).step_by(chunk) {
            let end = (start + chunk).min(total);
            proc_small.process(
                &left_in[start..end],
                &right_in[start..end],
                &params,
                &mut small_l[start..end],
                &mut small_r[start..end],
            );
        }

        // Should be bit-identical
        assert_eq!(big_l, small_l);
        assert_eq!(big_r, small_r);
    }

    #[test]
    fn test_feedback_affects_output() {
        let total = BLOCK_SIZE * 3;
        let (left_in, right_in) = make_stereo_sine(total);

        let mut proc_no_fb = StereoFractalProcessor::new();
        let mut params_no_fb = FractalParams::default();
        params_no_fb.feedback = 0.0;
        let mut out_l_no = vec![0.0; total];
        let mut out_r_no = vec![0.0; total];
        proc_no_fb.process(&left_in, &right_in, &params_no_fb, &mut out_l_no, &mut out_r_no);

        let mut proc_fb = StereoFractalProcessor::new();
        let mut params_fb = FractalParams::default();
        params_fb.feedback = 0.5;
        let mut out_l_fb = vec![0.0; total];
        let mut out_r_fb = vec![0.0; total];
        proc_fb.process(&left_in, &right_in, &params_fb, &mut out_l_fb, &mut out_r_fb);

        // Outputs should differ when feedback is active (at least after 2nd block)
        let diff: f64 = out_l_no[BLOCK_SIZE * 2..]
            .iter()
            .zip(out_l_fb[BLOCK_SIZE * 2..].iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 0.0, "Feedback should change the output");
    }
}
