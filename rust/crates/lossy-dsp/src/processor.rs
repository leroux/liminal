//! Pre-allocated lossy processor for real-time use.
//!
//! Unlike `render_lossy` which allocates all buffers per call, `LossyProcessor`
//! pre-allocates everything at construction time. Zero allocations in the audio
//! thread. The lofi reverb and freeze states persist across calls.

use crate::params::{LossyParams, SR, SLOPE_OPTIONS};
use crate::rng::NumpyRng;
use num_complex::Complex;
use realfft::RealFftPlanner;
use std::sync::Arc;

// Maximum sizes based on param ranges and PROCESS_BLOCK=8192.
const MAX_BLOCK: usize = 8192;
const MAX_WINDOW: usize = 16384;
const MAX_PADDED: usize = MAX_BLOCK + 2 * MAX_WINDOW;
const MAX_BINS: usize = MAX_WINDOW / 2 + 1;
const MAX_BANDS: usize = 65;
const MAX_FRAMES: usize = 1100;
const MAX_PACKET_SAMPLES: usize = 8820; // 200ms at 44100

// Comb filter delays for lofi reverb.
const COMB_DELAYS: [usize; 4] = [1031, 1327, 1657, 1973];
const ALLPASS_DELAY: usize = 379;

/// Pre-allocated lossy processor. Processes mono audio blocks.
pub struct LossyProcessor {
    // Chain working buffer
    wet: Vec<f64>,

    // Spectral STFT buffers
    planner: RealFftPlanner<f64>,
    cached_fft: Option<Arc<dyn realfft::RealToComplex<f64>>>,
    cached_ifft: Option<Arc<dyn realfft::ComplexToReal<f64>>>,
    cached_window_size: usize,
    window: Vec<f64>,
    padded: Vec<f64>,
    stft_output: Vec<f64>,
    win_sum: Vec<f64>,
    fft_input: Vec<f64>,
    ifft_output: Vec<f64>,
    spectrum: Vec<Complex<f64>>,
    magnitudes: Vec<f64>,
    phases: Vec<f64>,
    proc_mag: Vec<f64>,
    compressed: Vec<f64>,
    band_edges: Vec<usize>,
    ath_weights: Vec<f64>,
    band_energy: Vec<f64>,
    energies: Vec<f64>,
    transient_flags: Vec<bool>,
    envelope: Vec<f64>,
    inv_env: Vec<f64>,
    shaped: Vec<f64>,
    frozen_spectrum: Option<Vec<f64>>,
    cached_stft_window_size: i32,
    cached_n_bands: i32,
    /// Actual number of edges stored (after dedup, may be < n_bands_param + 1).
    actual_n_edges: usize,

    // Scratch buffer for POST verb path (avoids allocation)
    verb_scratch: Vec<f64>,

    // Lofi reverb persistent state
    comb_bufs: [Vec<f64>; 4],
    comb_y1: [f64; 4],
    allpass_buf: Vec<f64>,
    reverb_wet: Vec<f64>,

    // Packet processing
    packet_last_good: Vec<f64>,
    packet_fade_in: Vec<f64>,
    packet_fade_out: Vec<f64>,

    // Filter biquad state (per-section)
    biquad_w1: [f64; 8],
    biquad_w2: [f64; 8],
}

impl LossyProcessor {
    pub fn new() -> Self {
        Self {
            wet: vec![0.0; MAX_BLOCK],

            planner: RealFftPlanner::new(),
            cached_fft: None,
            cached_ifft: None,
            cached_window_size: 0,
            window: vec![0.0; MAX_WINDOW],
            padded: vec![0.0; MAX_PADDED],
            stft_output: vec![0.0; MAX_PADDED],
            win_sum: vec![0.0; MAX_PADDED],
            fft_input: vec![0.0; MAX_WINDOW],
            ifft_output: vec![0.0; MAX_WINDOW],
            spectrum: vec![Complex::new(0.0, 0.0); MAX_BINS],
            magnitudes: vec![0.0; MAX_BINS],
            phases: vec![0.0; MAX_BINS],
            proc_mag: vec![0.0; MAX_BINS],
            compressed: vec![0.0; MAX_BINS],
            band_edges: vec![0; MAX_BANDS + 1],
            ath_weights: vec![0.0; MAX_BANDS],
            band_energy: vec![0.0; MAX_BANDS],
            energies: vec![0.0; MAX_FRAMES],
            transient_flags: vec![false; MAX_FRAMES],
            envelope: vec![0.0; MAX_BINS],
            inv_env: vec![0.0; MAX_BINS],
            shaped: vec![0.0; MAX_BINS],
            frozen_spectrum: None,
            cached_stft_window_size: 0,
            cached_n_bands: 0,
            actual_n_edges: 0,

            verb_scratch: vec![0.0; MAX_BLOCK],

            comb_bufs: [
                vec![0.0; COMB_DELAYS[0]],
                vec![0.0; COMB_DELAYS[1]],
                vec![0.0; COMB_DELAYS[2]],
                vec![0.0; COMB_DELAYS[3]],
            ],
            comb_y1: [0.0; 4],
            allpass_buf: vec![0.0; ALLPASS_DELAY],
            reverb_wet: vec![0.0; MAX_BLOCK],

            packet_last_good: vec![0.0; MAX_PACKET_SAMPLES],
            packet_fade_in: vec![0.0; 256],
            packet_fade_out: vec![0.0; 256],

            biquad_w1: [0.0; 8],
            biquad_w2: [0.0; 8],
        }
    }

    pub fn reset(&mut self) {
        for buf in &mut self.comb_bufs {
            buf.fill(0.0);
        }
        self.comb_y1 = [0.0; 4];
        self.allpass_buf.fill(0.0);
        self.frozen_spectrum = None;
        self.biquad_w1 = [0.0; 8];
        self.biquad_w2 = [0.0; 8];
    }

    /// Process a mono block. `output` must have length >= `input.len()`.
    /// Wet/dry mixing is applied.
    pub fn process(&mut self, input: &[f64], params: &LossyParams, output: &mut [f64]) {
        let n = input.len();
        debug_assert!(n <= MAX_BLOCK);
        debug_assert!(output.len() >= n);

        // Bounce is not supported in processor (plugin doesn't expose it)
        self.render_chain(input, params, n);

        // Wet/dry mix
        let mix = params.wet_dry;
        for i in 0..n {
            output[i] = input[i] * (1.0 - mix) + self.wet[i] * mix;
        }
    }

    fn render_chain(&mut self, dry: &[f64], params: &LossyParams, n: usize) {
        let verb_pos = params.verb_position;

        // PRE verb
        if verb_pos == 0 {
            self.lofi_reverb(dry, params, n);
        } else {
            self.wet[..n].copy_from_slice(&dry[..n]);
        }

        // RMS before spectral (for auto_gain)
        let rms_before = if params.auto_gain > 0.0 {
            rms(&self.wet[..n])
        } else {
            0.0
        };

        // Spectral processing
        self.spectral_process(n, params);

        // Auto gain
        if params.auto_gain > 0.0 && rms_before > 1e-8 {
            let rms_after = rms(&self.wet[..n]);
            if rms_after > 1e-8 {
                let ratio = rms_before / rms_after;
                let gain = (1.0 + params.auto_gain * (ratio - 1.0)).min(10.0);
                for i in 0..n {
                    self.wet[i] *= gain;
                }
            }
        }

        // Loss gain
        if params.loss_gain != 0.5 {
            let db = (params.loss_gain - 0.5) * 72.0;
            let gain = (10.0_f64).powf(db / 20.0);
            for i in 0..n {
                self.wet[i] *= gain;
            }
        }

        // Crush + decimate (in-place)
        self.crush_and_decimate(n, params);

        // Packets (in-place)
        self.packet_process(n, params);

        // Filter (in-place)
        self.apply_filter(n, params);

        // POST verb
        if verb_pos == 1 {
            // Copy wet to scratch, apply reverb from scratch into wet
            self.verb_scratch[..n].copy_from_slice(&self.wet[..n]);
            self.lofi_reverb_from_scratch(params, n);
        }

        // Gate (in-place)
        self.noise_gate(n, params);

        // Limiter (in-place)
        self.limiter(n, params);
    }

    // -----------------------------------------------------------------------
    // Lofi reverb (persistent state)
    // -----------------------------------------------------------------------
    fn lofi_reverb(&mut self, input: &[f64], params: &LossyParams, n: usize) {
        let g = params.global_amount;
        let mix = params.verb * g;
        if mix <= 0.0 {
            self.wet[..n].copy_from_slice(&input[..n]);
            return;
        }
        let fb = 0.4 + 0.55 * params.decay;
        let damp = 0.45_f64;

        self.reverb_wet[..n].fill(0.0);
        for c in 0..4 {
            let d = COMB_DELAYS[c];
            for i in 0..n {
                let idx = i % d;
                let rd = self.comb_bufs[c][idx];
                self.comb_y1[c] = damp * rd + (1.0 - damp) * self.comb_y1[c];
                self.reverb_wet[i] += rd * 0.25;
                self.comb_bufs[c][idx] = input[i] * 0.25 + self.comb_y1[c] * fb;
            }
        }

        let ap_g = 0.6_f64;
        for i in 0..n {
            let idx = i % ALLPASS_DELAY;
            let delayed = self.allpass_buf[idx];
            let inp = self.reverb_wet[i];
            self.reverb_wet[i] = delayed - ap_g * inp;
            self.allpass_buf[idx] = inp + ap_g * self.reverb_wet[i];
        }

        for i in 0..n {
            self.wet[i] = input[i] * (1.0 - mix) + self.reverb_wet[i] * mix;
        }
    }

    /// Like `lofi_reverb` but reads from `self.verb_scratch` (POST verb path).
    fn lofi_reverb_from_scratch(&mut self, params: &LossyParams, n: usize) {
        let g = params.global_amount;
        let mix = params.verb * g;
        if mix <= 0.0 {
            // wet already has the data from verb_scratch copy
            return;
        }
        let fb = 0.4 + 0.55 * params.decay;
        let damp = 0.45_f64;

        self.reverb_wet[..n].fill(0.0);
        for c in 0..4 {
            let d = COMB_DELAYS[c];
            for i in 0..n {
                let idx = i % d;
                let rd = self.comb_bufs[c][idx];
                self.comb_y1[c] = damp * rd + (1.0 - damp) * self.comb_y1[c];
                self.reverb_wet[i] += rd * 0.25;
                self.comb_bufs[c][idx] = self.verb_scratch[i] * 0.25 + self.comb_y1[c] * fb;
            }
        }

        let ap_g = 0.6_f64;
        for i in 0..n {
            let idx = i % ALLPASS_DELAY;
            let delayed = self.allpass_buf[idx];
            let inp = self.reverb_wet[i];
            self.reverb_wet[i] = delayed - ap_g * inp;
            self.allpass_buf[idx] = inp + ap_g * self.reverb_wet[i];
        }

        for i in 0..n {
            self.wet[i] = self.verb_scratch[i] * (1.0 - mix) + self.reverb_wet[i] * mix;
        }
    }

    // -----------------------------------------------------------------------
    // Spectral processing (STFT)
    // -----------------------------------------------------------------------
    fn spectral_process(&mut self, n: usize, params: &LossyParams) {
        let g = params.global_amount;
        let loss = params.loss * g;
        let inverse = params.inverse != 0;
        let jitter = params.jitter * g;
        let seed = params.seed;
        let freeze = params.freeze != 0;
        let freeze_mode = params.freeze_mode;
        let freezer_blend = params.freezer;
        let phase_loss = params.phase_loss * g;
        let quantizer_type = params.quantizer;
        let pre_echo_amount = params.pre_echo * g;
        let noise_shape = params.noise_shape;
        let weighting = params.weighting;
        let hf_threshold = params.hf_threshold;
        let transient_ratio = params.transient_ratio;
        let slushy_rate_param = params.slushy_rate;

        if loss <= 0.0 && !freeze && phase_loss <= 0.0 && jitter <= 0.0 {
            // wet already contains input, nothing to do
            return;
        }

        let window_size = (params.window_size.max(2)) as usize;
        let hop_divisor = (params.hop_divisor.max(1)) as usize;
        let hop_size = (window_size / hop_divisor).max(1);
        let n_bins = window_size / 2 + 1;
        let n_bands_param = (params.n_bands.max(2)) as usize;

        // Update cached window and FFT plans
        if window_size != self.cached_window_size {
            self.cached_window_size = window_size;
            for i in 0..window_size {
                self.window[i] =
                    0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / window_size as f64).cos());
            }
            self.cached_fft = Some(self.planner.plan_fft_forward(window_size));
            self.cached_ifft = Some(self.planner.plan_fft_inverse(window_size));
        }

        // Update cached band edges and ATH weights
        if params.window_size != self.cached_stft_window_size
            || params.n_bands != self.cached_n_bands
        {
            self.cached_stft_window_size = params.window_size;
            self.cached_n_bands = params.n_bands;
            let (edges, n_b) = compute_band_edges(n_bins, n_bands_param);
            let n_edge = edges.len().min(self.band_edges.len());
            self.band_edges[..n_edge].copy_from_slice(&edges[..n_edge]);
            self.actual_n_edges = n_edge;
            let ath = compute_ath_weights(&edges, n_b, n_bins, window_size);
            let n_ath = ath.len().min(self.ath_weights.len());
            self.ath_weights[..n_ath].copy_from_slice(&ath[..n_ath]);
        }

        let fft = self.cached_fft.as_ref().unwrap().clone();
        let ifft = self.cached_ifft.as_ref().unwrap().clone();

        // Pad input (reflection)
        let pad = window_size;
        let input = &self.wet[..n]; // wet currently holds the input
        let padded_len = pad_reflect_into(input, pad, &mut self.padded);

        // Zero output and win_sum
        self.stft_output[..padded_len].fill(0.0);
        self.win_sum[..padded_len].fill(0.0);

        let n_frames = if padded_len >= window_size {
            (padded_len - window_size) / hop_size + 1
        } else {
            0
        };

        let mut rng = NumpyRng::new(seed as u32);

        // Band edges info (use actual count after dedup, not n_bands_param)
        let mut band_edges_copy = [0usize; MAX_BANDS + 1];
        let n_edge_count = self.actual_n_edges.min(band_edges_copy.len());
        band_edges_copy[..n_edge_count].copy_from_slice(&self.band_edges[..n_edge_count]);
        let n_bands = if n_edge_count > 1 { n_edge_count - 1 } else { 0 };

        // Pre-echo detection
        let has_transients = pre_echo_amount > 0.0 && n_frames > 1;
        if has_transients {
            for fi in 0..n_frames.min(MAX_FRAMES) {
                let start = fi * hop_size;
                let mut e = 0.0;
                for j in 0..window_size {
                    if start + j < padded_len {
                        let s = self.padded[start + j] * self.window[j];
                        e += s * s;
                    }
                }
                self.energies[fi] = e;
            }
            self.transient_flags[0] = false;
            for fi in 1..n_frames.min(MAX_FRAMES) {
                self.transient_flags[fi] =
                    self.energies[fi - 1] > 1e-12
                        && self.energies[fi] / self.energies[fi - 1] > transient_ratio;
            }
        }

        for fi in 0..n_frames {
            let start = fi * hop_size;

            // Window the frame
            for j in 0..window_size {
                self.fft_input[j] = if start + j < padded_len {
                    self.padded[start + j] * self.window[j]
                } else {
                    0.0
                };
            }

            // Forward FFT
            fft.process(&mut self.fft_input[..window_size], &mut self.spectrum[..n_bins])
                .unwrap();

            // Extract magnitude and phase
            for i in 0..n_bins {
                self.magnitudes[i] = self.spectrum[i].norm();
                self.phases[i] = self.spectrum[i].arg();
            }

            // Pre-echo
            let mut frame_loss = loss;
            if has_transients && fi < n_frames - 1 && fi + 1 < MAX_FRAMES {
                if self.transient_flags[fi + 1] {
                    frame_loss = (loss + pre_echo_amount * 0.5).min(1.0);
                }
            }

            // Magnitude processing
            if frame_loss > 0.0 {
                self.standard_degrade(
                    n_bins,
                    frame_loss,
                    &mut rng,
                    &band_edges_copy,
                    n_bands,
                    quantizer_type,
                    noise_shape,
                    weighting,
                );
            } else {
                self.proc_mag[..n_bins].copy_from_slice(&self.magnitudes[..n_bins]);
            }

            if inverse {
                for i in 0..n_bins {
                    self.proc_mag[i] = (self.magnitudes[i] - self.proc_mag[i]).max(0.0);
                }
            }

            // Phase processing
            if phase_loss > 0.0 {
                let n_levels = (64.0 * (1.0 - phase_loss)).max(4.0) as i32;
                let step = 2.0 * std::f64::consts::PI / n_levels as f64;
                for i in 0..n_bins {
                    self.phases[i] = step * (self.phases[i] / step).round();
                }
            }
            if jitter > 0.0 {
                let pi = std::f64::consts::PI;
                for i in 0..n_bins {
                    let noise = rng.uniform(-pi, pi) * jitter;
                    self.phases[i] += noise;
                }
            }

            // HF limiting
            if frame_loss > hf_threshold {
                let cutoff = ((n_bins as f64 * (1.0 - 0.6 * frame_loss)) as usize).max(n_bins / 8);
                let hf_range = if hf_threshold < 1.0 {
                    1.0 - hf_threshold
                } else {
                    1.0
                };
                let mult = ((1.0 - (frame_loss - hf_threshold) / hf_range).max(0.0)).min(1.0);
                for i in cutoff..n_bins {
                    self.proc_mag[i] *= mult;
                }
            }

            // Freeze
            if freeze {
                if self.frozen_spectrum.is_none() {
                    self.frozen_spectrum = Some(self.proc_mag[..n_bins].to_vec());
                }
                let frozen = self.frozen_spectrum.as_mut().unwrap();
                if freeze_mode != 1 {
                    // Slushy
                    for i in 0..n_bins.min(frozen.len()) {
                        frozen[i] = (1.0 - slushy_rate_param) * frozen[i]
                            + slushy_rate_param * self.proc_mag[i];
                    }
                }
                for i in 0..n_bins.min(frozen.len()) {
                    self.proc_mag[i] =
                        freezer_blend * frozen[i] + (1.0 - freezer_blend) * self.proc_mag[i];
                }
            }

            // Reconstruct
            for i in 0..n_bins {
                self.spectrum[i] = Complex::from_polar(self.proc_mag[i], self.phases[i]);
            }
            self.spectrum[0] = Complex::new(self.spectrum[0].re, 0.0);
            if n_bins > 1 {
                let last = n_bins - 1;
                self.spectrum[last] = Complex::new(self.spectrum[last].re, 0.0);
            }

            // Inverse FFT
            ifft.process(
                &mut self.spectrum[..n_bins],
                &mut self.ifft_output[..window_size],
            )
            .unwrap();

            let norm = 1.0 / window_size as f64;
            for j in 0..window_size {
                if start + j < padded_len {
                    self.stft_output[start + j] += self.ifft_output[j] * norm * self.window[j];
                    self.win_sum[start + j] += self.window[j] * self.window[j];
                }
            }
        }

        // Normalize and remove padding
        for i in 0..padded_len {
            if self.win_sum[i] < 1e-8 {
                self.win_sum[i] = 1.0;
            }
            self.stft_output[i] /= self.win_sum[i];
        }

        // Copy result back to wet (remove padding)
        for i in 0..n {
            self.wet[i] = self.stft_output[pad + i];
        }
    }

    fn standard_degrade(
        &mut self,
        n_bins: usize,
        loss: f64,
        rng: &mut NumpyRng,
        band_edges: &[usize],
        n_bands: usize,
        quantizer_type: i32,
        noise_shape: f64,
        weighting: f64,
    ) {
        self.proc_mag[..n_bins].copy_from_slice(&self.magnitudes[..n_bins]);

        let max_mag = self.proc_mag[..n_bins]
            .iter()
            .cloned()
            .fold(0.0_f64, f64::max);
        if max_mag > 0.0 {
            let bits = 16.0 - 14.0 * loss;
            let n_levels = (2.0_f64).powf(bits);

            if quantizer_type == 1 {
                // Compand
                for i in 0..n_bins {
                    self.compressed[i] = self.proc_mag[i].powf(0.75);
                }
                let max_c = self.compressed[..n_bins]
                    .iter()
                    .cloned()
                    .fold(0.0_f64, f64::max);
                if max_c > 0.0 {
                    if noise_shape > 0.0 {
                        let base_delta = 2.0 * max_c / n_levels;
                        self.shape_delta_into(n_bins, base_delta, noise_shape, true);
                        for i in 0..n_bins {
                            let d = self.shaped[i].max(1e-20);
                            self.compressed[i] = d * (self.compressed[i] / d).round();
                        }
                    } else {
                        let delta = 2.0 * max_c / n_levels;
                        for i in 0..n_bins {
                            self.compressed[i] =
                                delta * (self.compressed[i] / delta).round();
                        }
                    }
                }
                for i in 0..n_bins {
                    self.proc_mag[i] = self.compressed[i].max(0.0).powf(4.0 / 3.0);
                }
            } else {
                // Uniform
                if noise_shape > 0.0 {
                    let base_delta = 2.0 * max_mag / n_levels;
                    self.shape_delta_into(n_bins, base_delta, noise_shape, false);
                    for i in 0..n_bins {
                        let d = self.shaped[i].max(1e-20);
                        self.proc_mag[i] = d * (self.magnitudes[i] / d).round();
                    }
                } else {
                    let delta = 2.0 * max_mag / n_levels;
                    for i in 0..n_bins {
                        self.proc_mag[i] = delta * (self.proc_mag[i] / delta).round();
                    }
                }
            }
        }

        // Band gating
        for b in 0..n_bands {
            let lo = band_edges[b];
            let hi = band_edges[b + 1];
            if hi > lo {
                let sum: f64 = self.proc_mag[lo..hi].iter().map(|x| x * x).sum();
                self.band_energy[b] = sum / (hi - lo) as f64;
            } else {
                self.band_energy[b] = 0.0;
            }
        }

        let mean_energy =
            self.band_energy[..n_bands].iter().sum::<f64>() / n_bands.max(1) as f64 + 1e-12;

        for b in 0..n_bands {
            let relative = (self.band_energy[b] / mean_energy).min(2.0) / 2.0;
            let ath_factor =
                (1.0 - weighting) * 0.75 + weighting * (0.5 + 0.5 * self.ath_weights[b]);
            let mut gate_prob = loss * 0.6 * (1.0 - relative) * ath_factor;
            gate_prob += rng.random() * loss * 0.2;
            if rng.random() < gate_prob {
                let lo = band_edges[b];
                let hi = band_edges[b + 1];
                for i in lo..hi {
                    self.proc_mag[i] = 0.0;
                }
            }
        }
    }

    fn shape_delta_into(
        &mut self,
        n: usize,
        base_delta: f64,
        amount: f64,
        use_compressed: bool,
    ) {
        let src = if use_compressed {
            &self.compressed
        } else {
            &self.proc_mag
        };

        let kernel_size = 7;
        for i in 0..n {
            let lo = i.saturating_sub(kernel_size / 2);
            let hi = (i + kernel_size / 2 + 1).min(n);
            let sum: f64 = src[lo..hi].iter().sum();
            self.envelope[i] = (sum / (hi - lo) as f64).max(1e-12);
        }

        let mut max_inv = 0.0_f64;
        for i in 0..n {
            self.inv_env[i] = 1.0 / self.envelope[i];
            if self.inv_env[i] > max_inv {
                max_inv = self.inv_env[i];
            }
        }
        if max_inv > 0.0 {
            for i in 0..n {
                self.inv_env[i] /= max_inv;
            }
        }

        for i in 0..n {
            self.shaped[i] = base_delta * (1.0 + amount * self.inv_env[i] * 3.0);
        }
    }

    // -----------------------------------------------------------------------
    // Crush + decimate (in-place on self.wet)
    // -----------------------------------------------------------------------
    fn crush_and_decimate(&mut self, n: usize, params: &LossyParams) {
        let g = params.global_amount;
        let crush = params.crush * g;
        let decimate = params.decimate * g;

        if crush <= 0.0 && decimate <= 0.0 {
            return;
        }

        if crush > 0.0 {
            let bits = 16.0 - 12.0 * crush;
            let quant = (2.0_f64).powf(bits - 1.0);
            for i in 0..n {
                self.wet[i] = (self.wet[i] * quant + 0.5).floor() / quant;
            }
        }

        if decimate > 0.0 {
            let rate_factor = 1.0 + 31.0 * decimate;
            let mut phase = 0.0_f64;
            let mut held = 0.0_f64;
            for i in 0..n {
                phase += 1.0;
                if phase >= rate_factor {
                    held = self.wet[i];
                    phase -= rate_factor;
                }
                self.wet[i] = held;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Packet processing (in-place on self.wet)
    // -----------------------------------------------------------------------
    fn packet_process(&mut self, n: usize, params: &LossyParams) {
        let mode = params.packets;
        if mode == 0 {
            return;
        }

        let g = params.global_amount;
        let rate = params.packet_rate * g;
        let packet_ms = params.packet_size;
        let seed = params.seed;

        if rate <= 0.0 {
            return;
        }

        let packet_samples = ((packet_ms * SR / 1000.0).max(1.0) as usize).min(MAX_PACKET_SAMPLES);
        let mut rng = NumpyRng::new((seed + 1000) as u32);

        let p_g2b = rate * 0.3;
        let p_b2g = 0.4;
        let mut in_bad = false;
        let mut prev_bad = false;

        // Save original for crossfade
        // We need a copy of the input for crossfade blending
        // Use stft_output as scratch (it's not being used here)
        self.stft_output[..n].copy_from_slice(&self.wet[..n]);

        let xfade = (0.003 * SR) as usize;
        let xfade = xfade.min(packet_samples / 4);
        // Compute fade windows into pre-allocated buffers
        for i in 0..xfade {
            let full_len = xfade * 2;
            self.packet_fade_in[i] =
                0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / full_len as f64).cos());
            self.packet_fade_out[i] = 0.5
                * (1.0
                    - (2.0 * std::f64::consts::PI * (xfade + i) as f64 / full_len as f64).cos());
        }

        let mut start = 0;
        while start < n {
            let end = (start + packet_samples).min(n);
            let chunk_len = end - start;

            if in_bad {
                if mode == 1 {
                    for i in start..end {
                        self.wet[i] = 0.0;
                    }
                } else if mode == 2 {
                    for i in 0..chunk_len {
                        self.wet[start + i] = self.packet_last_good[i];
                    }
                }

                if !prev_bad && xfade > 0 && start > 0 {
                    let xf = xfade.min(start).min(chunk_len);
                    for i in 0..xf {
                        self.wet[start + i] *= self.packet_fade_in[i];
                        self.wet[start + i] +=
                            self.stft_output[start + i] * self.packet_fade_out[xfade - xf + i];
                    }
                }

                if rng.random() < p_b2g {
                    prev_bad = true;
                    in_bad = false;
                } else {
                    prev_bad = true;
                }
            } else {
                if prev_bad && xfade > 0 {
                    let xf = xfade.min(chunk_len);
                    for i in 0..xf {
                        self.wet[start + i] *= self.packet_fade_in[i];
                    }
                }

                for i in 0..chunk_len.min(MAX_PACKET_SAMPLES) {
                    self.packet_last_good[i] = self.stft_output[start + i];
                }
                prev_bad = false;
                if rng.random() < p_g2b {
                    in_bad = true;
                }
            }

            start = end;
        }
    }

    // -----------------------------------------------------------------------
    // Biquad filter (in-place on self.wet)
    // -----------------------------------------------------------------------
    fn apply_filter(&mut self, n: usize, params: &LossyParams) {
        if params.filter_type == 0 {
            return;
        }

        let freq = params.filter_freq.clamp(20.0, SR / 2.0 - 1.0);
        let idx = (params.filter_slope as usize).min(SLOPE_OPTIONS.len() - 1);
        let slope = SLOPE_OPTIONS[idx];
        let n_sections = crate::params::slope_sections(slope);

        let width = params.filter_width;
        let q_base = 0.3 * (20.0_f64 / 0.3).powf(1.0 - width);
        let q_boost = match slope {
            6 => 1.0,
            24 => 1.5,
            96 => 3.0,
            _ => 1.0,
        };
        let q = q_base * q_boost;

        let w0 = 2.0 * std::f64::consts::PI * freq / SR;
        let alpha = w0.sin() / (2.0 * q);
        let cos_w0 = w0.cos();

        let (b, a) = if params.filter_type == 1 {
            ([alpha, 0.0, -alpha], [1.0 + alpha, -2.0 * cos_w0, 1.0 - alpha])
        } else {
            ([1.0, -2.0 * cos_w0, 1.0], [1.0 + alpha, -2.0 * cos_w0, 1.0 - alpha])
        };

        let a0 = a[0];
        let b0 = b[0] / a0;
        let b1 = b[1] / a0;
        let b2 = b[2] / a0;
        let a1 = a[1] / a0;
        let a2 = a[2] / a0;

        for sec in 0..n_sections {
            let mut w1 = self.biquad_w1[sec];
            let mut w2 = self.biquad_w2[sec];
            for i in 0..n {
                let w0_val = self.wet[i] - a1 * w1 - a2 * w2;
                self.wet[i] = b0 * w0_val + b1 * w1 + b2 * w2;
                w2 = w1;
                w1 = w0_val;
            }
            self.biquad_w1[sec] = w1;
            self.biquad_w2[sec] = w2;
        }
    }

    // -----------------------------------------------------------------------
    // Noise gate (in-place on self.wet)
    // -----------------------------------------------------------------------
    fn noise_gate(&mut self, n: usize, params: &LossyParams) {
        let g = params.global_amount;
        let threshold = params.gate * g;
        if threshold <= 0.0 {
            return;
        }

        let win = 512usize;
        let mut start = 0;
        while start < n {
            let end = (start + win).min(n);
            let mut s = 0.0_f64;
            for i in start..end {
                s += self.wet[i] * self.wet[i];
            }
            let rms_val = (s / (end - start) as f64).sqrt();
            if rms_val < threshold {
                let gain = rms_val / threshold;
                for i in start..end {
                    self.wet[i] *= gain;
                }
            }
            start = end;
        }
    }

    // -----------------------------------------------------------------------
    // Limiter (in-place on self.wet)
    // -----------------------------------------------------------------------
    fn limiter(&mut self, n: usize, params: &LossyParams) {
        let ceiling = 0.1 + 0.85 * params.threshold;
        let peak = self.wet[..n]
            .iter()
            .map(|x| x.abs())
            .fold(0.0_f64, f64::max);
        if peak > ceiling && peak > 0.0 {
            let scale = ceiling / peak;
            for i in 0..n {
                self.wet[i] *= scale;
            }
        }
    }
}

/// Stereo lossy processor.
pub struct StereoLossyProcessor {
    proc_l: LossyProcessor,
    proc_r: LossyProcessor,
}

impl StereoLossyProcessor {
    pub fn new() -> Self {
        Self {
            proc_l: LossyProcessor::new(),
            proc_r: LossyProcessor::new(),
        }
    }

    pub fn reset(&mut self) {
        self.proc_l.reset();
        self.proc_r.reset();
    }

    pub fn process_stereo(
        &mut self,
        left: &[f64],
        right: &[f64],
        params: &LossyParams,
        out_l: &mut [f64],
        out_r: &mut [f64],
    ) {
        self.proc_l.process(left, params, out_l);
        self.proc_r.process(right, params, out_r);
    }
}

// -----------------------------------------------------------------------
// Helpers (no allocation)
// -----------------------------------------------------------------------

fn rms(audio: &[f64]) -> f64 {
    if audio.is_empty() {
        return 0.0;
    }
    let sum: f64 = audio.iter().map(|x| x * x).sum();
    (sum / audio.len() as f64).sqrt()
}

/// Reflection-pad `audio` into `out`, returns padded length.
fn pad_reflect_into(audio: &[f64], pad: usize, out: &mut [f64]) -> usize {
    let n = audio.len();
    let total = n + 2 * pad;
    debug_assert!(out.len() >= total);

    if n > pad {
        for i in 0..pad {
            let idx = (pad - 1 - i + 1) % n;
            out[i] = audio[idx];
        }
        out[pad..pad + n].copy_from_slice(audio);
        for i in 0..pad {
            let idx = n.saturating_sub(2).saturating_sub(i % n.max(1));
            out[pad + n + i] = audio[idx.min(n - 1)];
        }
    } else {
        out[..pad].fill(0.0);
        out[pad..pad + n].copy_from_slice(audio);
        out[pad + n..total].fill(0.0);
    }
    total
}

fn compute_band_edges(n_bins: usize, n_bands_param: usize) -> (Vec<usize>, usize) {
    let mut edges: Vec<usize> = (0..=n_bands_param)
        .map(|i| {
            let t = i as f64 / n_bands_param as f64;
            let v = (10.0_f64).powf(t * (n_bins as f64).log10());
            (v as usize).min(n_bins)
        })
        .collect();
    edges.sort();
    edges.dedup();
    let n_bands = if edges.len() > 1 { edges.len() - 1 } else { 0 };
    (edges, n_bands)
}

fn compute_ath_weights(
    band_edges: &[usize],
    n_bands: usize,
    n_bins: usize,
    window_size: usize,
) -> Vec<f64> {
    let mut bin_ath = vec![0.0; n_bins];
    for i in 0..n_bins {
        let freq = i as f64 * SR / window_size as f64;
        let f_khz = (freq / 1000.0).clamp(0.02, 20.0);
        bin_ath[i] = 3.64 * f_khz.powf(-0.8) - 6.5 * (-0.6 * (f_khz - 3.3).powi(2)).exp()
            + 1e-3 * f_khz.powi(4);
    }

    let min_ath = bin_ath.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_range = bin_ath.iter().cloned().fold(0.0_f64, f64::max) - min_ath;

    let mut weights = vec![0.0_f64; n_bands];
    for b in 0..n_bands {
        let lo = band_edges[b];
        let hi = band_edges[b + 1];
        if hi > lo && max_range > 0.0 {
            let sum: f64 = bin_ath[lo..hi]
                .iter()
                .map(|&a| (a - min_ath) / max_range)
                .sum();
            weights[b] = sum / (hi - lo) as f64;
        }
    }
    weights
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
    fn test_processor_matches_render() {
        let audio = make_sine(4096);
        let params = LossyParams::default();

        let expected = crate::chain::render_lossy(&audio, &params);

        let mut proc = LossyProcessor::new();
        let mut output = vec![0.0; 4096];
        proc.process(&audio, &params, &mut output);

        assert_eq!(expected.len(), output.len());
        let max_diff: f64 = expected
            .iter()
            .zip(output.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff < 1e-10,
            "Processor output differs from render_lossy: max_diff={max_diff}"
        );
    }

    #[test]
    fn test_processor_bypass() {
        let audio: Vec<f64> = (0..4096).map(|i| (i as f64 * 0.01).sin() * 0.3).collect();
        let mut params = LossyParams::default();
        params.loss = 0.0;
        params.crush = 0.0;
        params.packets = 0;
        params.filter_type = 0;
        params.verb = 0.0;
        params.gate = 0.0;
        params.threshold = 1.0;
        params.wet_dry = 1.0;

        let mut proc = LossyProcessor::new();
        let mut output = vec![0.0; 4096];
        proc.process(&audio, &params, &mut output);

        let max_diff: f64 = audio
            .iter()
            .zip(output.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_diff < 0.01, "Bypass should be near-passthrough: max_diff={max_diff}");
    }

    #[test]
    fn test_stereo_processor() {
        let left = make_sine(4096);
        let right = make_sine(4096);
        let params = LossyParams::default();

        let mut proc = StereoLossyProcessor::new();
        let mut out_l = vec![0.0; 4096];
        let mut out_r = vec![0.0; 4096];
        proc.process_stereo(&left, &right, &params, &mut out_l, &mut out_r);

        assert!(out_l.iter().all(|x| x.is_finite()));
        assert!(out_r.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_processor_no_alloc_repeated() {
        let audio = make_sine(256);
        let params = LossyParams::default();
        let mut proc = LossyProcessor::new();
        let mut output = vec![0.0; 256];

        for _ in 0..100 {
            proc.process(&audio, &params, &mut output);
        }
        assert!(output.iter().all(|x| x.is_finite()));
    }
}
