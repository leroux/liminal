/// Safety checks and normalization — port of shared/streaming.py.

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
    fn test_safety_check_ok() {
        let audio = vec![0.5, -0.3, 0.8, -0.1];
        assert!(safety_check(&audio).is_ok());
    }

    #[test]
    fn test_safety_check_nan() {
        let audio = vec![0.5, f64::NAN, 0.8];
        assert!(safety_check(&audio).is_err());
    }

    #[test]
    fn test_safety_check_inf() {
        let audio = vec![0.5, f64::INFINITY];
        assert!(safety_check(&audio).is_err());
    }

    #[test]
    fn test_safety_check_exploded() {
        let audio = vec![0.5, 2e6];
        assert!(safety_check(&audio).is_err());
    }

    #[test]
    fn test_normalize_output() {
        let audio = vec![1.0, -1.0, 0.5, -0.5];
        let (norm, warning) = normalize_output(&audio, 0.2, 0.9);
        // All values should be within headroom
        for s in &norm {
            assert!(s.abs() <= 0.9 + 1e-10);
        }
        assert!(warning.is_empty() || warning.contains("loud"));
    }

    #[test]
    fn test_normalize_silent() {
        let audio = vec![0.0, 0.0, 0.0];
        let (norm, warning) = normalize_output(&audio, 0.2, 0.9);
        assert_eq!(norm, vec![0.0, 0.0, 0.0]);
        assert!(warning.is_empty());
    }
}
