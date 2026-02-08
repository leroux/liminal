//! Lock-free parameter smoothing for real-time audio.
//!
//! Provides exponential ramp between current and target values,
//! avoiding zipper noise when parameters change.

/// Smoothed parameter with exponential ramp.
pub struct SmoothedParam {
    current: f64,
    target: f64,
    /// Coefficient per sample: `current = current + coeff * (target - current)`
    coeff: f64,
}

impl SmoothedParam {
    /// Create a new smoothed parameter.
    ///
    /// `ramp_ms` — time to reach ~63% of target (one time constant).
    /// `sample_rate` — audio sample rate in Hz.
    pub fn new(initial: f64, ramp_ms: f64, sample_rate: f64) -> Self {
        let samples = (ramp_ms / 1000.0) * sample_rate;
        Self {
            current: initial,
            target: initial,
            coeff: 1.0 - (-1.0_f64 / samples).exp(),
        }
    }

    /// Set a new target value (called from param-change thread).
    pub fn set_target(&mut self, target: f64) {
        self.target = target;
    }

    /// Get next smoothed value (called per sample from audio thread).
    #[inline]
    pub fn next(&mut self) -> f64 {
        self.current += self.coeff * (self.target - self.current);
        self.current
    }

    /// Snap to target immediately (e.g. on reset).
    pub fn reset(&mut self, value: f64) {
        self.current = value;
        self.target = value;
    }

    /// Check if smoothing is still active.
    pub fn is_smoothing(&self) -> bool {
        (self.current - self.target).abs() > 1e-8
    }

    /// Update ramp time (e.g. if sample rate changes).
    pub fn set_ramp(&mut self, ramp_ms: f64, sample_rate: f64) {
        let samples = (ramp_ms / 1000.0) * sample_rate;
        self.coeff = 1.0 - (-1.0_f64 / samples).exp();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snap_on_reset() {
        let mut p = SmoothedParam::new(0.0, 10.0, 44100.0);
        p.reset(1.0);
        assert_eq!(p.next(), 1.0);
    }

    #[test]
    fn ramps_toward_target() {
        let mut p = SmoothedParam::new(0.0, 10.0, 44100.0);
        p.set_target(1.0);
        // After many samples, should be close to target
        for _ in 0..44100 {
            p.next();
        }
        assert!((p.next() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn reaches_63_percent_at_one_tau() {
        let ramp_ms = 10.0;
        let sr = 44100.0;
        let mut p = SmoothedParam::new(0.0, ramp_ms, sr);
        p.set_target(1.0);
        let tau_samples = (ramp_ms / 1000.0 * sr) as usize;
        for _ in 0..tau_samples {
            p.next();
        }
        let val = p.next();
        // Should be ~63.2% of the way there
        assert!((val - 0.632).abs() < 0.02, "val={val}");
    }
}
