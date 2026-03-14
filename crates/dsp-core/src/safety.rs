//! Safety checks and normalization for rendered audio.

/// Check output audio for divergence or explosion.
/// Returns Ok(()) if safe, Err with description otherwise.
pub fn safety_check(output: &[f64]) -> Result<(), String> {
    for &s in output {
        if !s.is_finite() {
            return Err("output diverged (non-finite values)".to_string());
        }
    }
    let peak = output.iter().map(|s| s.abs()).fold(0.0_f64, f64::max);
    if peak > 1e6 {
        return Err(format!("output exploded (peak={peak:.0e})"));
    }
    Ok(())
}

/// RMS limiter + headroom normalization.
/// Returns (normalized_output, warning_string).
pub fn normalize_output(output: &[f64], target_rms: f64, headroom: f64) -> (Vec<f64>, String) {
    let peak = output.iter().map(|s| s.abs()).fold(0.0_f64, f64::max);

    let mut normalized: Vec<f64> = if peak > 0.0 {
        output.iter().map(|s| s / peak).collect()
    } else {
        output.to_vec()
    };

    let rms = {
        let sum: f64 = normalized.iter().map(|s| s * s).sum();
        (sum / normalized.len().max(1) as f64).sqrt()
    };

    let warning = if rms > target_rms {
        let gain = target_rms / rms;
        for s in &mut normalized {
            *s *= gain;
        }
        format!(" (loud — reduced {:.0}x)", 1.0 / gain)
    } else {
        String::new()
    };

    for s in &mut normalized {
        *s *= headroom;
    }

    (normalized, warning)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safety_check_ok() {
        assert!(safety_check(&[0.5, -0.3, 0.8, -0.1]).is_ok());
    }

    #[test]
    fn safety_check_nan() {
        assert!(safety_check(&[0.5, f64::NAN, 0.8]).is_err());
    }

    #[test]
    fn safety_check_inf() {
        assert!(safety_check(&[0.5, f64::INFINITY]).is_err());
    }

    #[test]
    fn safety_check_exploded() {
        assert!(safety_check(&[0.5, 2e6]).is_err());
    }

    #[test]
    fn normalize_output_basic() {
        let audio = vec![1.0, -1.0, 0.5, -0.5];
        let (norm, _) = normalize_output(&audio, 0.2, 0.9);
        for s in &norm {
            assert!(s.abs() <= 0.9 + 1e-10);
        }
    }

    #[test]
    fn normalize_silent() {
        let (norm, warning) = normalize_output(&[0.0, 0.0, 0.0], 0.2, 0.9);
        assert_eq!(norm, vec![0.0, 0.0, 0.0]);
        assert!(warning.is_empty());
    }

    #[test]
    fn safety_check_empty_is_ok() {
        assert!(safety_check(&[]).is_ok());
    }

    #[test]
    fn safety_check_exactly_at_threshold() {
        // 1e6 is exactly the threshold — should be OK (only > 1e6 fails)
        assert!(safety_check(&[1e6]).is_ok());
    }

    #[test]
    fn safety_check_just_above_threshold() {
        assert!(safety_check(&[1e6 + 1.0]).is_err());
    }

    #[test]
    fn safety_check_negative_infinity() {
        let result = safety_check(&[0.5, f64::NEG_INFINITY]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("non-finite"));
    }

    #[test]
    fn normalize_loud_signal_warns_reduced() {
        // All 1.0 values -> peak=1.0, normalized peak=1.0, RMS=1.0
        // RMS (1.0) > target_rms (0.2) => triggers reduction
        let audio = vec![1.0; 100];
        let (norm, warning) = normalize_output(&audio, 0.2, 0.9);
        assert!(
            warning.contains("reduced"),
            "Expected warning containing 'reduced', got: {warning}"
        );
        // All output samples should be scaled down and within bounds
        for s in &norm {
            assert!(s.abs() <= 0.9 + 1e-10);
        }
    }

    #[test]
    fn normalize_moderate_signal_no_warning() {
        // A sine wave with low amplitude: peak=0.1, normalized to 1.0,
        // RMS of a full sine at amplitude 1.0 is ~0.707 which is > 0.2,
        // so we need a signal whose RMS after peak-normalization is <= target_rms.
        // Use a sparse signal: one nonzero sample among many zeros.
        let mut audio = vec![0.0; 1000];
        audio[0] = 0.1;
        let (_, warning) = normalize_output(&audio, 0.2, 0.9);
        // After peak-norm: audio[0]=1.0, rest=0.0, RMS = sqrt(1/1000) ≈ 0.0316 < 0.2
        assert!(
            warning.is_empty(),
            "Expected no warning for moderate signal, got: {warning}"
        );
    }

    #[test]
    fn safety_check_all_zeros() {
        assert!(safety_check(&[0.0, 0.0, 0.0]).is_ok());
    }

    #[test]
    fn safety_check_exploded_error_message() {
        let result = safety_check(&[2e6]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("exploded"));
    }
}
