//! Main render entry point for the Lossy engine.
//!
//! Port of `lossy/engine/lossy.py`.
//!
//! Signal chain (verb_position=0, PRE — default):
//!     Input -> Verb -> Spectral Loss -> Auto Gain -> Loss Gain -> Crush/Decimate
//!           -> Packets -> Filter -> Gate -> Limiter -> Mix -> Output
//!
//! Signal chain (verb_position=1, POST):
//!     Input -> Spectral Loss -> Auto Gain -> Loss Gain -> Crush/Decimate
//!           -> Packets -> Filter -> Verb -> Gate -> Limiter -> Mix -> Output

use crate::bitcrush::crush_and_decimate;
use crate::filters::{apply_filter, limiter, lofi_reverb, noise_gate};
use crate::packets::packet_process;
use crate::params::{param_range, LossyParams, BOUNCE_TARGETS, SR};
use crate::spectral::spectral_process;

/// Core signal chain without bounce modulation.
fn render_chain(dry: &[f64], params: &LossyParams) -> Vec<f64> {
    let verb_pos = params.verb_position;

    // PRE verb: reverb runs on dry signal before spectral processing
    let mut wet = if verb_pos == 0 {
        lofi_reverb(dry, params)
    } else {
        dry.to_vec()
    };

    // Measure input RMS before spectral loss (for auto_gain)
    let rms_before = if params.auto_gain > 0.0 {
        rms(&wet)
    } else {
        0.0
    };

    // 1) Spectral loss (STFT-based)
    wet = spectral_process(&wet, params);

    // 2) Auto gain compensation
    if params.auto_gain > 0.0 && rms_before > 1e-8 {
        let rms_after = rms(&wet);
        if rms_after > 1e-8 {
            let ratio = rms_before / rms_after;
            let gain = (1.0 + params.auto_gain * (ratio - 1.0)).min(10.0);
            for s in wet.iter_mut() {
                *s *= gain;
            }
        }
    }

    // 3) Loss gain (wet signal volume: 0->-36dB, 0.5->0dB, 1->+36dB)
    if params.loss_gain != 0.5 {
        let db = (params.loss_gain - 0.5) * 72.0;
        let gain = (10.0_f64).powf(db / 20.0);
        for s in wet.iter_mut() {
            *s *= gain;
        }
    }

    // 4) Bitcrusher + sample rate reducer
    wet = crush_and_decimate(&wet, params);

    // 5) Packet loss / repeat
    wet = packet_process(&wet, params);

    // 6) Biquad filter (bandpass / notch)
    wet = apply_filter(&wet, params);

    // POST verb: reverb runs after filter
    if verb_pos == 1 {
        wet = lofi_reverb(&wet, params);
    }

    // 7) Noise gate
    wet = noise_gate(&wet, params);

    // 8) Limiter (threshold-aware)
    wet = limiter(&wet, params);

    wet
}

/// Render a single mono channel.
fn render_mono(dry: &[f64], params: &LossyParams) -> Vec<f64> {
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
fn render_with_bounce(dry: &[f64], params: &LossyParams) -> Vec<f64> {
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

/// Process audio through the lossy engine (mono).
///
/// This is the main entry point matching `render_lossy` in Python.
pub fn render_lossy(input_audio: &[f64], params: &LossyParams) -> Vec<f64> {
    render_mono(input_audio, params)
}

/// Process stereo audio (two separate channels).
pub fn render_lossy_stereo(
    left: &[f64],
    right: &[f64],
    params: &LossyParams,
) -> (Vec<f64>, Vec<f64>) {
    let out_l = render_mono(left, params);
    let out_r = render_mono(right, params);
    (out_l, out_r)
}

fn rms(audio: &[f64]) -> f64 {
    if audio.is_empty() {
        return 0.0;
    }
    let sum: f64 = audio.iter().map(|x| x * x).sum();
    (sum / audio.len() as f64).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render_default_params() {
        let audio: Vec<f64> = (0..44100)
            .map(|i| (2.0 * std::f64::consts::PI * 440.0 * i as f64 / SR).sin())
            .collect();
        let params = LossyParams::default();
        let out = render_lossy(&audio, &params);
        assert_eq!(out.len(), audio.len());
        // Should produce finite output
        assert!(out.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_bypass_when_clean() {
        // Use small amplitude to avoid limiter
        let audio: Vec<f64> = (0..4096).map(|i| (i as f64 * 0.01).sin() * 0.3).collect();
        let mut params = LossyParams::default();
        params.loss = 0.0;
        params.crush = 0.0;
        params.packets = 0;
        params.filter_type = 0;
        params.verb = 0.0;
        params.gate = 0.0;
        params.threshold = 1.0; // light limiting (ceiling=0.95)
        params.wet_dry = 1.0;
        let out = render_lossy(&audio, &params);
        // Should be very close to input
        let max_diff: f64 = audio
            .iter()
            .zip(out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_diff < 0.01, "max diff: {}", max_diff);
    }

    #[test]
    fn test_stereo_render() {
        let left: Vec<f64> = (0..4096).map(|i| (i as f64 * 0.01).sin()).collect();
        let right: Vec<f64> = (0..4096).map(|i| (i as f64 * 0.02).cos()).collect();
        let params = LossyParams::default();
        let (out_l, out_r) = render_lossy_stereo(&left, &right, &params);
        assert_eq!(out_l.len(), left.len());
        assert_eq!(out_r.len(), right.len());
    }

    #[test]
    fn test_wet_dry_mix() {
        let audio: Vec<f64> = (0..4096).map(|i| (i as f64 * 0.01).sin()).collect();
        let mut params = LossyParams::default();
        params.loss = 0.8;

        // Fully dry
        params.wet_dry = 0.0;
        let out_dry = render_lossy(&audio, &params);
        let diff_dry: f64 = audio
            .iter()
            .zip(out_dry.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff_dry < 1e-10); // should be identical to input

        // Fully wet
        params.wet_dry = 1.0;
        let _out_wet = render_lossy(&audio, &params);
        // Just check it runs without panic
    }

    #[test]
    fn test_bounce_modulation() {
        let audio: Vec<f64> = (0..44100)
            .map(|i| (2.0 * std::f64::consts::PI * 440.0 * i as f64 / SR).sin())
            .collect();
        let mut params = LossyParams::default();
        params.bounce = 1;
        params.bounce_target = 0; // loss
        params.bounce_rate = 0.5;
        params.loss = 0.5;
        let out = render_lossy(&audio, &params);
        assert_eq!(out.len(), audio.len());
        assert!(out.iter().all(|x| x.is_finite()));
    }
}
