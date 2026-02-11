//! Main render entry point for the Fractal engine.
//!
//! Port of `fractal/engine/fractal.py`.
//!
//! Signal chain:
//!     Input -> Pre-Filter -> Fractalize (with iterations) -> Output Gain
//!           -> Crush/Decimate -> Post-Filter -> Gate -> Limiter -> Mix -> Output

use crate::core::render_fractal_core;
use crate::filters::{apply_post_filter, apply_pre_filter, crush_and_decimate, limiter, noise_gate};
use crate::params::{param_range, FractalParams, BOUNCE_TARGETS, SR};

/// Core signal chain without bounce modulation.
fn render_chain(dry: &[f64], params: &FractalParams) -> Vec<f64> {
    // 1) Pre-filter
    let mut wet = apply_pre_filter(dry, params);

    // 2) Fractalize (core algorithm with iterations)
    wet = render_fractal_core(&wet, params);

    // 3) Output gain (-36 to +36 dB)
    if params.output_gain != 0.5 {
        let db = (params.output_gain - 0.5) * 72.0;
        let gain = (10.0_f64).powf(db / 20.0);
        for s in wet.iter_mut() {
            *s *= gain;
        }
    }

    // 4) Bitcrusher + sample rate reducer
    wet = crush_and_decimate(&wet, params);

    // 5) Post-filter
    wet = apply_post_filter(&wet, params);

    // 6) Noise gate
    wet = noise_gate(&wet, params);

    // 7) Limiter
    wet = limiter(&wet, params);

    wet
}

/// Render a single mono channel.
fn render_mono(dry: &[f64], params: &FractalParams) -> Vec<f64> {
    let wet = if params.bounce != 0 {
        render_with_bounce(dry, params)
    } else {
        render_chain(dry, params)
    };

    let mix = params.wet_dry;
    dry.iter()
        .zip(wet.iter())
        .map(|(&d, &w)| d * (1.0 - mix) + w * mix)
        .collect()
}

/// Block-based render with LFO modulation of a target parameter.
fn render_with_bounce(dry: &[f64], params: &FractalParams) -> Vec<f64> {
    let bounce_target_idx = (params.bounce_target as usize).min(BOUNCE_TARGETS.len() - 1);
    let bounce_rate_param = params.bounce_rate;

    let lfo_min = params.bounce_lfo_min;
    let lfo_max = params.bounce_lfo_max;
    let lfo_hz = lfo_min + (lfo_max - lfo_min) * bounce_rate_param;

    let target_key = BOUNCE_TARGETS[bounce_target_idx];
    let base_value = params.get_bounce_target_value(target_key);

    let (lo, hi) = param_range(target_key).unwrap_or((0.0, 1.0));

    // Block size ~50ms
    let block_samples = (SR * 0.05) as usize;
    let n = dry.len();
    let mut wet = vec![0.0_f64; n];

    let mut start = 0;
    while start < n {
        let end = (start + block_samples).min(n);
        let block_mid = (start + end) as f64 / 2.0;

        // LFO value at block midpoint (sine, 0 to 1 range)
        let t = block_mid / SR;
        let lfo = 0.5 + 0.5 * (2.0 * std::f64::consts::PI * lfo_hz * t).sin();

        // Modulate: sweep between lo and base_value
        let mod_value = (lo + lfo * (base_value - lo)).clamp(lo, hi);

        // Create modified params for this block
        let mut block_params = params.clone();
        block_params.set_bounce_target_value(target_key, mod_value);
        block_params.bounce = 0; // prevent recursion

        let block_wet = render_chain(&dry[start..end], &block_params);
        wet[start..end].copy_from_slice(&block_wet);

        start = end;
    }

    wet
}

/// Process audio through the fractal engine (mono).
///
/// This is the main entry point matching `render_fractal` in Python.
pub fn render_fractal(input_audio: &[f64], params: &FractalParams) -> Vec<f64> {
    render_mono(input_audio, params)
}

/// Process stereo audio (two separate channels).
pub fn render_fractal_stereo(
    left: &[f64],
    right: &[f64],
    params: &FractalParams,
) -> (Vec<f64>, Vec<f64>) {
    let out_l = render_mono(left, params);
    let out_r = render_mono(right, params);
    (out_l, out_r)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render_default_params() {
        let audio: Vec<f64> = (0..44100)
            .map(|i| (2.0 * std::f64::consts::PI * 440.0 * i as f64 / SR).sin())
            .collect();
        let params = FractalParams::default();
        let out = render_fractal(&audio, &params);
        assert_eq!(out.len(), audio.len());
        assert!(out.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_stereo_render() {
        let left: Vec<f64> = (0..4096).map(|i| (i as f64 * 0.01).sin()).collect();
        let right: Vec<f64> = (0..4096).map(|i| (i as f64 * 0.02).cos()).collect();
        let params = FractalParams::default();
        let (out_l, out_r) = render_fractal_stereo(&left, &right, &params);
        assert_eq!(out_l.len(), left.len());
        assert_eq!(out_r.len(), right.len());
    }

    #[test]
    fn test_wet_dry_mix() {
        let audio: Vec<f64> = (0..4096).map(|i| (i as f64 * 0.01).sin()).collect();
        let mut params = FractalParams::default();

        // Fully dry
        params.wet_dry = 0.0;
        let out_dry = render_fractal(&audio, &params);
        let diff_dry: f64 = audio
            .iter()
            .zip(out_dry.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff_dry < 1e-10);
    }

    #[test]
    fn test_bounce_modulation() {
        let audio: Vec<f64> = (0..44100)
            .map(|i| (2.0 * std::f64::consts::PI * 440.0 * i as f64 / SR).sin())
            .collect();
        let mut params = FractalParams::default();
        params.bounce = 1;
        params.bounce_target = 0; // scale_ratio
        params.bounce_rate = 0.5;
        let out = render_fractal(&audio, &params);
        assert_eq!(out.len(), audio.len());
        assert!(out.iter().all(|x| x.is_finite()));
    }
}
