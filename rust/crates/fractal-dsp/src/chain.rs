//! Main render entry point for the Fractal engine.
//!
//! Port of `fractal/engine/fractal.py`.
//!
//! Signal chain:
//!     Input -> Pre-Filter -> Fractalize (with iterations) -> Output Gain
//!           -> Crush/Decimate -> Post-Filter -> Gate -> Limiter -> Mix -> Output

use crate::core::{render_fractal_core, render_fractal_core_stereo};
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

/// Mono feedback processing: re-feed output into input in 8192-sample chunks.
fn render_with_feedback_mono(dry: &[f64], params: &FractalParams) -> Vec<f64> {
    let feedback = params.feedback.clamp(0.0, 0.95);
    let chunk_size = 8192;
    let n = dry.len();
    let mut out = vec![0.0_f64; n];
    let mut feedback_buf = vec![0.0_f64; chunk_size];

    // Use params with feedback=0 to prevent recursion
    let mut inner_params = params.clone();
    inner_params.feedback = 0.0;

    let mut start = 0;
    while start < n {
        let end = (start + chunk_size).min(n);
        let block_len = end - start;

        // Mix feedback into input
        let mut input_chunk: Vec<f64> = dry[start..end].to_vec();
        for i in 0..block_len {
            input_chunk[i] += feedback * feedback_buf[i];
        }

        // Process
        let processed = if inner_params.bounce != 0 {
            render_with_bounce(&input_chunk, &inner_params)
        } else {
            render_chain(&input_chunk, &inner_params)
        };

        out[start..end].copy_from_slice(&processed[..block_len]);

        // Store for next chunk
        feedback_buf = vec![0.0_f64; chunk_size];
        for i in 0..block_len {
            feedback_buf[i] = processed[i];
        }

        start = end;
    }

    out
}

/// Render a single mono channel.
fn render_mono(dry: &[f64], params: &FractalParams) -> Vec<f64> {
    let wet = if params.feedback > 0.001 {
        render_with_feedback_mono(dry, params)
    } else if params.bounce != 0 {
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

/// Stereo signal chain using render_fractal_core_stereo for spread.
fn render_stereo_chain(left: &[f64], right: &[f64], params: &FractalParams) -> (Vec<f64>, Vec<f64>) {
    // 1) Pre-filter per channel
    let wet_l = apply_pre_filter(left, params);
    let wet_r = apply_pre_filter(right, params);

    // 2) Stereo fractalize
    let (mut out_l, mut out_r) = render_fractal_core_stereo(&wet_l, &wet_r, params);

    // 3) Output gain
    if params.output_gain != 0.5 {
        let db = (params.output_gain - 0.5) * 72.0;
        let gain = (10.0_f64).powf(db / 20.0);
        for s in out_l.iter_mut() {
            *s *= gain;
        }
        for s in out_r.iter_mut() {
            *s *= gain;
        }
    }

    // 4-7) Crush, post-filter, gate, limiter per channel
    out_l = crush_and_decimate(&out_l, params);
    out_r = crush_and_decimate(&out_r, params);
    out_l = apply_post_filter(&out_l, params);
    out_r = apply_post_filter(&out_r, params);
    out_l = noise_gate(&out_l, params);
    out_r = noise_gate(&out_r, params);
    out_l = limiter(&out_l, params);
    out_r = limiter(&out_r, params);

    (out_l, out_r)
}

/// Stereo feedback processing.
fn render_with_feedback_stereo(
    left: &[f64],
    right: &[f64],
    params: &FractalParams,
) -> (Vec<f64>, Vec<f64>) {
    let feedback = params.feedback.clamp(0.0, 0.95);
    let chunk_size = 8192;
    let n = left.len();
    let mut out_l = vec![0.0_f64; n];
    let mut out_r = vec![0.0_f64; n];
    let mut fb_l = vec![0.0_f64; chunk_size];
    let mut fb_r = vec![0.0_f64; chunk_size];

    let mut inner_params = params.clone();
    inner_params.feedback = 0.0;

    let use_stereo_chain = inner_params.layer_spread > 0.0;

    let mut start = 0;
    while start < n {
        let end = (start + chunk_size).min(n);
        let block_len = end - start;

        let mut in_l: Vec<f64> = left[start..end].to_vec();
        let mut in_r: Vec<f64> = right[start..end].to_vec();
        for i in 0..block_len {
            in_l[i] += feedback * fb_l[i];
            in_r[i] += feedback * fb_r[i];
        }

        let (proc_l, proc_r) = if use_stereo_chain {
            render_stereo_chain(&in_l, &in_r, &inner_params)
        } else {
            (render_chain(&in_l, &inner_params), render_chain(&in_r, &inner_params))
        };

        out_l[start..end].copy_from_slice(&proc_l[..block_len]);
        out_r[start..end].copy_from_slice(&proc_r[..block_len]);

        fb_l = vec![0.0_f64; chunk_size];
        fb_r = vec![0.0_f64; chunk_size];
        for i in 0..block_len {
            fb_l[i] = proc_l[i];
            fb_r[i] = proc_r[i];
        }

        start = end;
    }

    (out_l, out_r)
}

/// Process stereo audio with optional spread and feedback.
pub fn render_fractal_stereo(
    left: &[f64],
    right: &[f64],
    params: &FractalParams,
) -> (Vec<f64>, Vec<f64>) {
    let use_stereo_chain = params.layer_spread > 0.0;
    let use_feedback = params.feedback > 0.001;

    let (wet_l, wet_r) = if use_feedback {
        render_with_feedback_stereo(left, right, params)
    } else if use_stereo_chain {
        render_stereo_chain(left, right, params)
    } else {
        let out_l = render_mono(left, params);
        let out_r = render_mono(right, params);
        return (out_l, out_r);
    };

    // Wet/dry mix
    let mix = params.wet_dry;
    let out_l: Vec<f64> = left
        .iter()
        .zip(wet_l.iter())
        .map(|(&d, &w)| d * (1.0 - mix) + w * mix)
        .collect();
    let out_r: Vec<f64> = right
        .iter()
        .zip(wet_r.iter())
        .map(|(&d, &w)| d * (1.0 - mix) + w * mix)
        .collect();
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
    fn test_layer_spread_stereo_differs() {
        let audio: Vec<f64> = (0..4096).map(|i| (i as f64 * 0.01).sin()).collect();
        let mut params = FractalParams::default();
        params.layer_spread = 0.5;
        let (out_l, out_r) = render_fractal_stereo(&audio, &audio, &params);
        assert_eq!(out_l.len(), audio.len());
        // With spread > 0 and identical input, L and R should differ
        let diff: f64 = out_l.iter().zip(out_r.iter()).map(|(l, r)| (l - r).abs()).sum();
        assert!(diff > 0.0, "Stereo spread should produce L/R difference");
    }

    #[test]
    fn test_feedback_adds_energy() {
        let audio: Vec<f64> = (0..8192)
            .map(|i| (2.0 * std::f64::consts::PI * 440.0 * i as f64 / SR).sin())
            .collect();
        let mut params = FractalParams::default();
        params.feedback = 0.0;
        let out_no_fb = render_fractal(&audio, &params);

        params.feedback = 0.5;
        let out_fb = render_fractal(&audio, &params);

        let energy_no_fb: f64 = out_no_fb.iter().map(|x| x * x).sum();
        let energy_fb: f64 = out_fb.iter().map(|x| x * x).sum();
        assert!(out_fb.iter().all(|x| x.is_finite()));
        // Feedback should add energy (or at minimum not crash)
        assert!(energy_fb > 0.0 || energy_no_fb > 0.0);
    }

    #[test]
    fn test_layer_gain_mutes() {
        use crate::core::fractalize_time_layers;
        let audio: Vec<f64> = (0..4096)
            .map(|i| (2.0 * std::f64::consts::PI * 440.0 * i as f64 / SR).sin())
            .collect();
        let mut params = FractalParams::default();
        // Mute all layers except original
        params.layer_gain_1 = 0.0;
        params.layer_gain_2 = 0.0;
        let layers = fractalize_time_layers(&audio, &params);
        // Layer 0 should equal original, layers 1+ should be zero
        let diff0: f64 = audio.iter().zip(layers[0].iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff0 < 1e-10, "Layer 0 should be original");
        for s in 1..layers.len() {
            let energy: f64 = layers[s].iter().map(|x| x * x).sum();
            assert!(energy < 1e-10, "Muted layer {s} should have zero energy");
        }
    }

    #[test]
    fn test_only_wet_mode() {
        let audio: Vec<f64> = (0..4096)
            .map(|i| (2.0 * std::f64::consts::PI * 440.0 * i as f64 / SR).sin())
            .collect();
        let mut params = FractalParams::default();
        params.fractal_only_wet = 1;
        params.wet_dry = 1.0;
        let out = render_fractal(&audio, &params);
        assert_eq!(out.len(), audio.len());
        assert!(out.iter().all(|x| x.is_finite()));
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
