//! Simplified macro controls for the FDN reverb.
//!
//! Maps ~14 intuitive macro parameters to the full 30+ `FdnParams`.
//! The DSP code is unchanged — this is a pure mapping layer.
//!
//! Designed from PCA analysis of the 65-preset bank, which showed that
//! 8 components explain 91% of variance in hand-designed presets.

use serde::{Deserialize, Serialize};

use crate::params::{FdnParams, SR, N, MATRIX_TYPES};

// --- Constants for size mapping ---

/// Minimum base delay in ms (size=0 → tiny resonator).
const MIN_DELAY_MS: f64 = 2.0;
/// Ratio of max to min delay (used in exponential mapping). 300ms / 2ms = 150.
const DELAY_RATIO: f64 = 150.0;

/// Default delay time ratios relative to the base delay.
/// Derived from FdnParams::default() delay_ms values.
const DELAY_RATIOS: [f64; N] = [1.0, 1.249, 1.391, 1.613, 1.788, 1.997, 2.279, 2.461];

/// Default diffusion delay ratios relative to a scaled base.
const DIFFUSION_RATIOS: [f64; 4] = [1.0, 1.491, 2.208, 3.038];

/// Diffusion base as a fraction of the main base delay.
const DIFFUSION_SCALE: f64 = 0.1785; // 5.3 / 29.7

/// Maximum damping coefficient (brightness=0 → maximum damping).
const MAX_DAMPING: f64 = 0.95;

/// Default node pans (evenly spaced L to R).
const DEFAULT_PANS: [f64; N] = [-1.0, -0.714, -0.429, -0.143, 0.143, 0.429, 0.714, 1.0];

// --- Modulation mapping constants ---

/// Max delay modulation depth (samples) at character=0 (chorus mode).
const MOD_DELAY_MAX: f64 = 50.0;
/// Min delay modulation depth (samples) at character=1 (glitch mode).
const MOD_DELAY_MIN: f64 = 5.0;
/// Max damping modulation at character=1.
const MOD_DAMP_MAX: f64 = 0.5;
/// Max output modulation at character=1.
const MOD_OUTPUT_MAX: f64 = 1.0;
/// Max matrix modulation at character=1.
const MOD_MATRIX_MAX: f64 = 0.8;

/// Node rate multiplier template for spread (even nodes stay at 1x).
const SPREAD_TEMPLATE: [f64; N] = [0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 3.0];

/// Simplified macro controls for the reverb.
///
/// ~14 controls vs. 30+ raw params. Maps forward to `FdnParams` losslessly
/// for the hand-designed preset space; reverse mapping is a best-fit approximation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ReverbParams {
    // --- Core controls (always active) ---
    /// Room size (0–1). Exponential mapping to delay times.
    /// 0 = tiny resonator (~2ms), 0.5 = medium room (~25ms), 1.0 = huge space (~300ms).
    pub size: f64,

    /// Decay time (0–1.15). Maps directly to feedback_gain.
    /// 0 = very short, 0.85 = default, >1.0 = infinite/growing.
    pub decay: f64,

    /// Tonal brightness (0–1). Inverse of damping.
    /// 0 = very dark (heavy damping), 1.0 = bright (no damping).
    pub brightness: f64,

    /// Diffusion amount (0–0.7). Controls smearing of early reflections.
    pub diffusion: f64,

    /// Wet/dry mix (0–1). 0 = fully dry, 1.0 = fully wet.
    pub mix: f64,

    /// Saturation in the feedback loop (0–1).
    pub saturation: f64,

    /// Pre-delay in milliseconds (0–250).
    pub pre_delay_ms: f64,

    /// Stereo width (0–1). Also derives node panning.
    pub stereo_width: f64,

    /// Feedback matrix type. One of: householder, hadamard, diagonal,
    /// random_orthogonal, circulant, stautner_puckette.
    pub matrix_type: String,

    // --- Modulation controls (conditional on mod_rate > 0) ---
    /// Master modulation rate in Hz (0–20).
    pub mod_rate: f64,

    /// Overall modulation depth (0–1). Distributed across targets by mod_character.
    pub mod_depth: f64,

    /// Modulation character (0–1).
    /// 0 = chorus (delay-only), 0.5 = balanced, 1.0 = glitchy (matrix-heavy).
    pub mod_character: f64,

    /// Modulation decorrelation between nodes (0–1).
    /// 0 = fully correlated (sync), 1.0 = fully decorrelated (spread).
    pub mod_spread: f64,

    /// Modulation waveform. 0 = sine, 1 = triangle, 2 = sample-and-hold.
    pub mod_waveform: i32,
}

impl Default for ReverbParams {
    fn default() -> Self {
        Self {
            size: 0.5,
            decay: 0.85,
            brightness: 0.7,
            diffusion: 0.5,
            mix: 0.5,
            saturation: 0.0,
            pre_delay_ms: 10.0,
            stereo_width: 1.0,
            matrix_type: "householder".to_string(),
            mod_rate: 0.0,
            mod_depth: 0.0,
            mod_character: 0.0,
            mod_spread: 0.0,
            mod_waveform: 0,
        }
    }
}

impl ReverbParams {
    /// Parse from JSON string. Missing fields get default values.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Convert to full `FdnParams` for DSP processing.
    pub fn to_fdn_params(&self) -> FdnParams {
        let mut p = FdnParams::default();

        // --- Size → delay_times + diffusion_delays ---
        let base_ms = MIN_DELAY_MS * DELAY_RATIO.powf(self.size.clamp(0.0, 1.0));
        for i in 0..N {
            p.delay_times[i] = (base_ms * DELAY_RATIOS[i] / 1000.0 * SR) as i32;
        }
        let diff_base_ms = base_ms * DIFFUSION_SCALE;
        for i in 0..4 {
            p.diffusion_delays[i] = (diff_base_ms * DIFFUSION_RATIOS[i] / 1000.0 * SR).max(1.0) as i32;
        }

        // --- Decay → feedback_gain ---
        p.feedback_gain = self.decay.clamp(0.0, 1.15);

        // --- Brightness → damping_coeffs (uniform across nodes) ---
        let damping = MAX_DAMPING * (1.0 - self.brightness.clamp(0.0, 1.0));
        p.damping_coeffs = vec![damping; N];

        // --- Direct scalar mappings ---
        p.diffusion = self.diffusion.clamp(0.0, 0.7);
        p.wet_dry = self.mix.clamp(0.0, 1.0);
        p.saturation = self.saturation.clamp(0.0, 1.0);
        p.pre_delay = (self.pre_delay_ms.clamp(0.0, 250.0) / 1000.0 * SR) as i32;
        p.stereo_width = self.stereo_width.clamp(0.0, 1.0);

        // --- Stereo width → node_pans ---
        let w = p.stereo_width;
        p.node_pans = DEFAULT_PANS.iter().map(|&pan| pan * w).collect();

        // --- Matrix type ---
        p.matrix_type = if MATRIX_TYPES.contains(&self.matrix_type.as_str()) {
            self.matrix_type.clone()
        } else {
            "householder".to_string()
        };

        // --- Fixed values (eliminated from simplified layer) ---
        p.input_gains = vec![1.0 / N as f64; N];
        p.output_gains = vec![1.0; N];
        p.diffusion_stages = 4;
        p.matrix_seed = 42;

        // --- Modulation ---
        p.mod_master_rate = self.mod_rate.max(0.0);
        p.mod_waveform = self.mod_waveform.clamp(0, 2);

        if self.mod_rate > 0.0 && self.mod_depth > 0.0 {
            let d = self.mod_depth.clamp(0.0, 1.0);
            let c = self.mod_character.clamp(0.0, 1.0);

            // Distribute depth across targets based on character
            let delay_depth = d * lerp(MOD_DELAY_MAX, MOD_DELAY_MIN, c);
            let damp_depth = d * lerp(0.0, MOD_DAMP_MAX, c);
            let output_depth = d * lerp(0.0, MOD_OUTPUT_MAX, c);
            let matrix_depth = d * lerp(0.0, MOD_MATRIX_MAX, c);

            p.mod_depth_delay = vec![delay_depth; N];
            p.mod_depth_damping = vec![damp_depth; N];
            p.mod_depth_output = vec![output_depth; N];
            p.mod_depth_matrix = matrix_depth;

            // Spread → correlation + node rate multipliers
            let s = self.mod_spread.clamp(0.0, 1.0);
            p.mod_correlation = 1.0 - s;
            p.mod_node_rate_mult = SPREAD_TEMPLATE
                .iter()
                .map(|&t| 1.0 + s * t)
                .collect();
        }

        p
    }

    /// Best-fit reverse mapping from full `FdnParams` (lossy).
    ///
    /// Reconstructs the closest simplified params. Information that doesn't
    /// fit the simplified model (per-node variation, custom matrices) is lost.
    pub fn from_fdn_params(p: &FdnParams) -> Self {
        // --- Size: invert exponential from mean delay time ---
        let mean_delay_ms = if p.delay_times.is_empty() {
            29.7
        } else {
            let sum: f64 = p.delay_times.iter().map(|&s| s as f64 / SR * 1000.0).sum();
            sum / p.delay_times.len() as f64
        };
        // mean_delay_ms ≈ base_ms * mean(DELAY_RATIOS)
        let mean_ratio: f64 = DELAY_RATIOS.iter().sum::<f64>() / N as f64;
        let base_ms = (mean_delay_ms / mean_ratio).max(MIN_DELAY_MS);
        let size = (base_ms / MIN_DELAY_MS).ln() / DELAY_RATIO.ln();

        // --- Decay: direct from feedback_gain ---
        let decay = p.feedback_gain;

        // --- Brightness: from mean damping ---
        let mean_damping = if p.damping_coeffs.is_empty() {
            0.3
        } else {
            p.damping_coeffs.iter().sum::<f64>() / p.damping_coeffs.len() as f64
        };
        let brightness = 1.0 - (mean_damping / MAX_DAMPING);

        // --- Direct scalar mappings ---
        let diffusion = p.diffusion;
        let mix = p.wet_dry;
        let saturation = p.saturation;
        let pre_delay_ms = p.pre_delay as f64 / SR * 1000.0;
        let stereo_width = p.stereo_width;

        // --- Matrix type ---
        let matrix_type = if MATRIX_TYPES.contains(&p.matrix_type.as_str()) {
            p.matrix_type.clone()
        } else {
            "householder".to_string()
        };

        // --- Modulation reverse mapping ---
        let mod_rate = p.mod_master_rate;
        let mod_waveform = p.mod_waveform;

        let (mod_depth, mod_character, mod_spread) = if mod_rate > 0.0 {
            // Recover depth and character by solving the forward equations:
            //   delay_depth = d * (MOD_DELAY_MAX - (MOD_DELAY_MAX - MOD_DELAY_MIN) * c)
            //   output_depth = d * MOD_OUTPUT_MAX * c
            // Solving:
            //   d = (delay + (MAX-MIN) * output/OUTPUT_MAX) / MAX
            //   c = output / (d * OUTPUT_MAX)
            let mean_delay_d = mean_of(&p.mod_depth_delay);
            let mean_out_d = mean_of(&p.mod_depth_output);
            let delay_range = MOD_DELAY_MAX - MOD_DELAY_MIN;

            let (depth, character) = if mean_out_d > 1e-6 {
                let d = (mean_delay_d + delay_range * mean_out_d / MOD_OUTPUT_MAX)
                    / MOD_DELAY_MAX;
                let c = if d > 1e-6 {
                    (mean_out_d / (d * MOD_OUTPUT_MAX)).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                (d.clamp(0.0, 1.0), c)
            } else if mean_delay_d > 1e-6 {
                // Pure chorus (character=0): d = delay / MOD_DELAY_MAX
                ((mean_delay_d / MOD_DELAY_MAX).clamp(0.0, 1.0), 0.0)
            } else {
                (0.0, 0.0)
            };

            // Spread from correlation
            let spread = 1.0 - p.mod_correlation.clamp(0.0, 1.0);

            (depth, character, spread)
        } else {
            (0.0, 0.0, 0.0)
        };

        ReverbParams {
            size: size.clamp(0.0, 1.0),
            decay: decay.clamp(0.0, 1.15),
            brightness: brightness.clamp(0.0, 1.0),
            diffusion,
            mix,
            saturation,
            pre_delay_ms,
            stereo_width,
            matrix_type,
            mod_rate,
            mod_depth,
            mod_character,
            mod_spread,
            mod_waveform,
        }
    }
}

/// Linear interpolation.
#[inline]
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

/// Mean of a slice, or 0.0 if empty.
#[inline]
fn mean_of(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_roundtrip() {
        let simplified = ReverbParams::default();
        let full = simplified.to_fdn_params();

        assert_eq!(full.delay_times.len(), N);
        assert_eq!(full.damping_coeffs.len(), N);
        assert_eq!(full.input_gains.len(), N);
        assert_eq!(full.output_gains.len(), N);
        assert_eq!(full.diffusion_delays.len(), 4);
        assert!((full.feedback_gain - 0.85).abs() < 1e-6);
        assert!((full.wet_dry - 0.5).abs() < 1e-6);
        assert_eq!(full.matrix_type, "householder");
    }

    #[test]
    fn test_size_extremes() {
        // Size 0 → tiny delays
        let mut s = ReverbParams::default();
        s.size = 0.0;
        let p = s.to_fdn_params();
        let min_delay_ms = p.delay_times[0] as f64 / SR * 1000.0;
        assert!(min_delay_ms < 3.0, "size=0 should give tiny delays, got {min_delay_ms}ms");

        // Size 1 → huge delays
        s.size = 1.0;
        let p = s.to_fdn_params();
        let max_delay_ms = p.delay_times[0] as f64 / SR * 1000.0;
        assert!(max_delay_ms > 250.0, "size=1 should give huge delays, got {max_delay_ms}ms");
    }

    #[test]
    fn test_brightness_mapping() {
        let mut s = ReverbParams::default();

        // Brightness 1 → no damping
        s.brightness = 1.0;
        let p = s.to_fdn_params();
        assert!(p.damping_coeffs[0].abs() < 1e-6);

        // Brightness 0 → max damping
        s.brightness = 0.0;
        let p = s.to_fdn_params();
        assert!((p.damping_coeffs[0] - MAX_DAMPING).abs() < 1e-6);
    }

    #[test]
    fn test_modulation_chorus() {
        let mut s = ReverbParams::default();
        s.mod_rate = 3.0;
        s.mod_depth = 0.5;
        s.mod_character = 0.0; // chorus = delay-only
        let p = s.to_fdn_params();

        assert!(p.mod_depth_delay[0] > 0.0, "chorus should have delay modulation");
        assert!(p.mod_depth_damping[0].abs() < 1e-6, "chorus should have no damping mod");
        assert!(p.mod_depth_matrix.abs() < 1e-6, "chorus should have no matrix mod");
    }

    #[test]
    fn test_modulation_glitch() {
        let mut s = ReverbParams::default();
        s.mod_rate = 5.0;
        s.mod_depth = 1.0;
        s.mod_character = 1.0; // glitchy = matrix-heavy
        let p = s.to_fdn_params();

        assert!(p.mod_depth_delay[0] > 0.0, "glitch still has some delay mod");
        assert!(p.mod_depth_matrix > 0.5, "glitch should have strong matrix mod");
    }

    #[test]
    fn test_reverse_mapping_scalars() {
        let p = FdnParams::default();
        let s = ReverbParams::from_fdn_params(&p);

        assert!((s.decay - 0.85).abs() < 1e-6);
        assert!((s.mix - 0.5).abs() < 1e-6);
        assert!((s.diffusion - 0.5).abs() < 1e-6);
        assert!((s.saturation - 0.0).abs() < 1e-6);
        assert!((s.stereo_width - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_roundtrip_approximate() {
        // Forward then reverse should approximately recover simplified params
        let original = ReverbParams {
            size: 0.6,
            decay: 0.9,
            brightness: 0.8,
            diffusion: 0.4,
            mix: 0.7,
            saturation: 0.1,
            pre_delay_ms: 20.0,
            stereo_width: 0.8,
            matrix_type: "hadamard".to_string(),
            mod_rate: 0.0,
            mod_depth: 0.0,
            mod_character: 0.0,
            mod_spread: 0.0,
            mod_waveform: 0,
        };

        let full = original.to_fdn_params();
        let recovered = ReverbParams::from_fdn_params(&full);

        assert!((recovered.size - original.size).abs() < 0.05,
                "size: {} vs {}", recovered.size, original.size);
        assert!((recovered.decay - original.decay).abs() < 1e-6);
        assert!((recovered.brightness - original.brightness).abs() < 0.05,
                "brightness: {} vs {}", recovered.brightness, original.brightness);
        assert!((recovered.mix - original.mix).abs() < 1e-6);
        assert!((recovered.saturation - original.saturation).abs() < 1e-6);
        assert_eq!(recovered.matrix_type, original.matrix_type);
    }

    #[test]
    fn test_from_json() {
        let json = r#"{"size": 0.7, "decay": 0.9, "mix": 0.6}"#;
        let s = ReverbParams::from_json(json).unwrap();
        assert!((s.size - 0.7).abs() < 1e-6);
        assert!((s.decay - 0.9).abs() < 1e-6);
        assert!((s.mix - 0.6).abs() < 1e-6);
        // Defaults for unspecified fields
        assert!((s.brightness - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_spread_node_rate_mult() {
        let mut s = ReverbParams::default();
        s.mod_rate = 2.0;
        s.mod_depth = 0.5;
        s.mod_spread = 1.0;

        let p = s.to_fdn_params();
        assert!((p.mod_correlation - 0.0).abs() < 1e-6);
        assert!((p.mod_node_rate_mult[0] - 1.0).abs() < 1e-6); // even nodes stay at 1
        assert!((p.mod_node_rate_mult[1] - 2.0).abs() < 1e-6); // odd nodes get multiplied
        assert!((p.mod_node_rate_mult[7] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_pre_delay_conversion() {
        let mut s = ReverbParams::default();
        s.pre_delay_ms = 100.0;
        let p = s.to_fdn_params();
        let expected_samples = (100.0 / 1000.0 * SR) as i32;
        assert_eq!(p.pre_delay, expected_samples);

        // Reverse
        let s2 = ReverbParams::from_fdn_params(&p);
        assert!((s2.pre_delay_ms - 100.0).abs() < 0.1);
    }

    #[test]
    fn test_node_pans_from_width() {
        let mut s = ReverbParams::default();
        s.stereo_width = 0.5;
        let p = s.to_fdn_params();
        // Pans should be half the default spread
        assert!((p.node_pans[0] - (-0.5)).abs() < 1e-3);
        assert!((p.node_pans[7] - 0.5).abs() < 1e-3);
    }
}
