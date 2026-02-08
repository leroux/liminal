//! Time-domain degradation: bitcrusher + sample rate reducer.
//!
//! Port of `lossy/engine/bitcrush.py`.
//!
//! Crush    -- reduce amplitude quantization levels (16-bit down to ~4-bit).
//! Decimate -- zero-order hold sample rate reduction with phase accumulator.

use crate::params::LossyParams;

/// Apply bitcrusher and/or sample rate reducer.
pub fn crush_and_decimate(audio: &[f64], params: &LossyParams) -> Vec<f64> {
    let g = params.global_amount;
    let crush = params.crush * g;
    let decimate = params.decimate * g;

    if crush <= 0.0 && decimate <= 0.0 {
        return audio.to_vec();
    }

    let mut out = Vec::with_capacity(audio.len());

    // Bitcrusher: crush 0 -> off, 1 -> extreme (16 down to 4 bits)
    if crush > 0.0 {
        let bits = 16.0 - 12.0 * crush;
        let quant = (2.0_f64).powf(bits - 1.0);
        for &s in audio {
            out.push((s * quant + 0.5).floor() / quant);
        }
    } else {
        out.extend_from_slice(audio);
    }

    // Sample rate reducer (zero-order hold with fractional phase accumulator)
    if decimate > 0.0 {
        let rate_factor = 1.0 + 31.0 * decimate; // hold for 1..32 samples
        let mut phase = 0.0_f64;
        let mut held = 0.0_f64;
        for s in out.iter_mut() {
            phase += 1.0;
            if phase >= rate_factor {
                held = *s;
                phase -= rate_factor;
            }
            *s = held;
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_passthrough_when_zero() {
        let audio: Vec<f64> = (0..100).map(|i| (i as f64 / 100.0).sin()).collect();
        let params = LossyParams::default(); // crush=0, decimate=0
        let out = crush_and_decimate(&audio, &params);
        assert_eq!(out.len(), audio.len());
        for (a, b) in audio.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
    }

    #[test]
    fn test_crush_reduces_precision() {
        let audio: Vec<f64> = (0..1000).map(|i| (i as f64 * 0.01).sin() * 0.5).collect();
        let mut params = LossyParams::default();
        params.crush = 1.0; // extreme crush
        let out = crush_and_decimate(&audio, &params);
        assert_eq!(out.len(), audio.len());
        // With extreme crush, many values should be quantized to the same level
        let unique_values: std::collections::HashSet<u64> =
            out.iter().map(|x| x.to_bits()).collect();
        assert!(unique_values.len() < audio.len());
    }

    #[test]
    fn test_decimate_holds_values() {
        let audio: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let mut params = LossyParams::default();
        params.decimate = 1.0; // extreme decimation (hold for 32 samples)
        let out = crush_and_decimate(&audio, &params);
        // Consecutive samples should be identical (held)
        let mut run_count = 0;
        for i in 1..out.len() {
            if out[i] == out[i - 1] {
                run_count += 1;
            }
        }
        assert!(run_count > 50); // most samples should be held
    }
}
