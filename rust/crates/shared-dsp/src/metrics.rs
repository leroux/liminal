//! Running audio metrics computed in real time.
//!
//! Tracks windowed RMS, peak, and DC offset for display and
//! OSC reporting.

/// Running audio metrics over a fixed window.
pub struct RunningMetrics {
    window: Vec<f64>,
    pos: usize,
    filled: bool,
    peak: f64,
    sum_sq: f64,
}

impl RunningMetrics {
    /// Create metrics tracker with given window size in samples.
    pub fn new(window_size: usize) -> Self {
        Self {
            window: vec![0.0; window_size],
            pos: 0,
            filled: false,
            peak: 0.0,
            sum_sq: 0.0,
        }
    }

    /// Push one sample and update running stats.
    #[inline]
    pub fn push(&mut self, sample: f64) {
        let old = self.window[self.pos];
        self.window[self.pos] = sample;
        self.sum_sq += sample * sample - old * old;
        // Prevent negative accumulation from floating point drift
        if self.sum_sq < 0.0 {
            self.sum_sq = 0.0;
        }

        let abs = sample.abs();
        if abs > self.peak {
            self.peak = abs;
        }

        self.pos += 1;
        if self.pos >= self.window.len() {
            self.pos = 0;
            self.filled = true;
            // Decay peak slowly
            self.peak *= 0.9995;
        }
    }

    /// Current RMS level.
    pub fn rms(&self) -> f64 {
        let n = if self.filled {
            self.window.len()
        } else {
            self.pos.max(1)
        };
        (self.sum_sq / n as f64).sqrt()
    }

    /// Current peak level.
    pub fn peak(&self) -> f64 {
        self.peak
    }

    /// Snapshot of current metrics.
    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            rms: self.rms(),
            peak: self.peak(),
        }
    }

    /// Reset all state.
    pub fn reset(&mut self) {
        self.window.fill(0.0);
        self.pos = 0;
        self.filled = false;
        self.peak = 0.0;
        self.sum_sq = 0.0;
    }
}

/// Point-in-time metrics reading.
#[derive(Debug, Clone, Copy)]
pub struct MetricsSnapshot {
    pub rms: f64,
    pub peak: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silence_is_zero() {
        let m = RunningMetrics::new(512);
        assert_eq!(m.rms(), 0.0);
        assert_eq!(m.peak(), 0.0);
    }

    #[test]
    fn dc_signal_rms() {
        let mut m = RunningMetrics::new(100);
        for _ in 0..100 {
            m.push(0.5);
        }
        assert!((m.rms() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn peak_tracks_max() {
        let mut m = RunningMetrics::new(100);
        m.push(0.3);
        m.push(0.8);
        m.push(0.2);
        assert!((m.peak() - 0.8).abs() < 1e-10);
    }
}
