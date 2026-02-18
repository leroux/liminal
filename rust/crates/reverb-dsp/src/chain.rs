//! Entry points for the reverb engine.
//!
//! Routes between static and modulated FDN based on parameter state.
//! Handles stereo input by processing channels independently.

use crate::fdn::render_fdn_static;
use crate::fdn_mod::render_fdn_mod;
use crate::params::ReverbParams;

/// Render mono input through the FDN reverb.
///
/// Automatically routes to static or modulated engine based on params.
/// Returns interleaved stereo output [L0, R0, L1, R1, ...].
pub fn render_fdn(input: &[f64], params: &ReverbParams) -> Vec<f64> {
    let mut params = params.clone();
    params.normalize();

    if params.has_modulation() {
        render_fdn_mod(input, &params)
    } else {
        render_fdn_static(input, &params)
    }
}

/// Render stereo input through the FDN reverb.
///
/// Processes each channel independently with wet_dry=1.0, then mixes.
/// Returns (left_out, right_out) vectors.
pub fn render_fdn_stereo(left: &[f64], right: &[f64], params: &ReverbParams) -> (Vec<f64>, Vec<f64>) {
    let mut params = params.clone();
    params.normalize();

    let n_samples = left.len().min(right.len());
    let mix = params.wet_dry;

    // Process each channel with full wet to get reverb contribution
    let mut wet_params = params.clone();
    wet_params.wet_dry = 1.0;

    let wet_l = if wet_params.has_modulation() {
        render_fdn_mod(&left[..n_samples], &wet_params)
    } else {
        render_fdn_static(&left[..n_samples], &wet_params)
    };
    let wet_r = if wet_params.has_modulation() {
        render_fdn_mod(&right[..n_samples], &wet_params)
    } else {
        render_fdn_static(&right[..n_samples], &wet_params)
    };

    // Mix: sum wet contributions from both channels, blend with dry
    let mut out_l = Vec::with_capacity(n_samples);
    let mut out_r = Vec::with_capacity(n_samples);
    let dry_gain = 1.0 - mix;

    for i in 0..n_samples {
        // wet_l/wet_r are interleaved stereo [L, R, L, R, ...]
        let wl = wet_l[i * 2] + wet_r[i * 2];
        let wr = wet_l[i * 2 + 1] + wet_r[i * 2 + 1];
        out_l.push(dry_gain * left[i] + mix * wl);
        out_r.push(dry_gain * right[i] + mix * wr);
    }

    (out_l, out_r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::N;

    #[test]
    fn test_render_fdn_mono() {
        let mut input = vec![0.0; 4410];
        input[0] = 1.0;
        let params = ReverbParams::default();
        let output = render_fdn(&input, &params);
        assert_eq!(output.len(), 4410 * 2);
    }

    #[test]
    fn test_render_fdn_stereo() {
        let mut left = vec![0.0; 4410];
        let mut right = vec![0.0; 4410];
        left[0] = 1.0;
        right[100] = 1.0;
        let params = ReverbParams::default();
        let (out_l, out_r) = render_fdn_stereo(&left, &right, &params);
        assert_eq!(out_l.len(), 4410);
        assert_eq!(out_r.len(), 4410);
    }

    #[test]
    fn test_modulation_routing() {
        let mut input = vec![0.0; 4410];
        input[0] = 1.0;
        let mut params = ReverbParams::default();
        // No modulation -> static path
        let out_static = render_fdn(&input, &params);
        // Add modulation -> mod path
        params.mod_master_rate = 2.0;
        params.mod_depth_delay = vec![5.0; N];
        let out_mod = render_fdn(&input, &params);
        // Both should produce output, but different
        assert_eq!(out_static.len(), out_mod.len());
        let diff: f64 = out_static
            .iter()
            .zip(out_mod.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 0.001, "Static and modulated should differ");
    }
}
