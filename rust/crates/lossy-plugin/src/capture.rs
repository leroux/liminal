//! Lock-free audio capture for GUI analysis.
//!
//! The audio thread writes output samples via `try_lock()` — if the GUI is
//! reading, the audio thread skips (losing a few ms of capture is fine for
//! analysis). The GUI reads a snapshot only when composing a chat message.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// Shared audio capture buffer between audio thread and GUI.
#[derive(Clone)]
pub struct AudioCapture {
    inner: Arc<Mutex<CaptureState>>,
}

struct CaptureState {
    /// Interleaved stereo samples [L0, R0, L1, R1, ...]
    data: VecDeque<f32>,
    /// Maximum number of samples (stereo interleaved) to keep.
    max_samples: usize,
    sample_rate: f32,
}

/// Basic audio metrics computed from captured samples.
pub struct AudioMetrics {
    pub rms_db: f32,
    pub peak_db: f32,
    pub crest_factor_db: f32,
    pub spectral_centroid_hz: f32,
    pub duration_secs: f32,
}

impl AudioCapture {
    /// Create a capture buffer that holds `duration_secs` of stereo audio.
    pub fn new(sample_rate: f32, duration_secs: f32) -> Self {
        let max_samples = (sample_rate * duration_secs * 2.0) as usize; // stereo
        Self {
            inner: Arc::new(Mutex::new(CaptureState {
                data: VecDeque::with_capacity(max_samples),
                max_samples,
                sample_rate,
            })),
        }
    }

    /// Called from the audio thread. Uses `try_lock` — never blocks.
    /// Writes interleaved stereo output samples.
    pub fn write_audio(&self, left: &[f64], right: &[f64], num_samples: usize) {
        if let Ok(mut state) = self.inner.try_lock() {
            for i in 0..num_samples {
                state.data.push_back(left[i] as f32);
                state.data.push_back(right[i] as f32);
            }
            // Trim to max capacity
            while state.data.len() > state.max_samples {
                state.data.pop_front();
                state.data.pop_front(); // keep stereo pairs aligned
            }
        }
        // If lock is held by GUI, we silently skip — acceptable for analysis
    }

    /// Read the last N seconds of captured audio as mono (L+R average).
    /// Called from the GUI thread.
    pub fn read_mono(&self, max_secs: f32) -> (Vec<f32>, f32) {
        let state = self.inner.lock().unwrap();
        let sr = state.sample_rate;
        let max_frames = (sr * max_secs) as usize;
        let total_frames = state.data.len() / 2;
        let start_frame = total_frames.saturating_sub(max_frames);

        let mut mono = Vec::with_capacity(total_frames - start_frame);
        for i in start_frame..total_frames {
            let l = state.data[i * 2];
            let r = state.data[i * 2 + 1];
            mono.push((l + r) * 0.5);
        }
        (mono, sr)
    }

    /// Compute basic audio metrics from the captured buffer.
    pub fn compute_metrics(&self) -> Option<AudioMetrics> {
        let (mono, sr) = self.read_mono(3.0);
        if mono.is_empty() {
            return None;
        }

        let n = mono.len() as f32;
        let duration_secs = n / sr;

        // RMS
        let sum_sq: f32 = mono.iter().map(|s| s * s).sum();
        let rms = (sum_sq / n).sqrt();
        let rms_db = if rms > 1e-10 {
            20.0 * rms.log10()
        } else {
            -100.0
        };

        // Peak
        let peak = mono.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        let peak_db = if peak > 1e-10 {
            20.0 * peak.log10()
        } else {
            -100.0
        };

        // Crest factor
        let crest_factor_db = peak_db - rms_db;

        // Spectral centroid (simple FFT-free estimate using zero-crossing rate
        // weighted by amplitude — good enough for chat context)
        let spectral_centroid_hz = estimate_spectral_centroid(&mono, sr);

        Some(AudioMetrics {
            rms_db,
            peak_db,
            crest_factor_db,
            spectral_centroid_hz,
            duration_secs,
        })
    }
}

/// Estimate spectral centroid via zero-crossing rate (fast approximation).
/// ZCR correlates with spectral centroid for broadband signals.
fn estimate_spectral_centroid(mono: &[f32], sr: f32) -> f32 {
    if mono.len() < 2 {
        return 0.0;
    }
    // Use last ~8192 samples
    let n = 8192.min(mono.len());
    let start = mono.len() - n;
    let segment = &mono[start..];

    let mut crossings = 0u32;
    for i in 1..segment.len() {
        if (segment[i] >= 0.0) != (segment[i - 1] >= 0.0) {
            crossings += 1;
        }
    }

    // ZCR to Hz: each crossing is half a cycle
    let zcr = crossings as f32 / (segment.len() - 1) as f32;
    zcr * sr * 0.5
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_write_and_read_mono() {
        let cap = AudioCapture::new(44100.0, 1.0);
        let n = 256;
        let left: Vec<f64> = (0..n).map(|i| (i as f64 * 0.01).sin()).collect();
        let right: Vec<f64> = (0..n).map(|i| (i as f64 * 0.01).cos()).collect();

        cap.write_audio(&left, &right, n);

        let (mono, sr) = cap.read_mono(1.0);
        assert_eq!(sr, 44100.0);
        assert_eq!(mono.len(), n);

        // Verify mono = (L+R) / 2
        for i in 0..n {
            let expected = (left[i] as f32 + right[i] as f32) * 0.5;
            assert!(
                (mono[i] - expected).abs() < 1e-6,
                "sample {i}: got {} expected {}",
                mono[i],
                expected
            );
        }
    }

    #[test]
    fn capture_respects_max_duration() {
        // Buffer holds 0.1s = 4410 stereo samples = 8820 interleaved
        let cap = AudioCapture::new(44100.0, 0.1);
        let n = 8820; // way more than 4410 frames
        let left = vec![0.5_f64; n];
        let right = vec![-0.5_f64; n];

        cap.write_audio(&left, &right, n);

        let (mono, _sr) = cap.read_mono(10.0); // request way more than available
        // Should be capped at ~4410 frames
        assert!(
            mono.len() <= 4410,
            "expected <= 4410 frames, got {}",
            mono.len()
        );
    }

    #[test]
    fn capture_read_mono_limited_duration() {
        let cap = AudioCapture::new(44100.0, 3.0);
        // Write 2 seconds of audio (88200 frames)
        let n = 88200;
        let left = vec![0.1_f64; n];
        let right = vec![0.1_f64; n];
        cap.write_audio(&left, &right, n);

        // Read only last 0.5s
        let (mono, _sr) = cap.read_mono(0.5);
        let expected_frames = (44100.0 * 0.5) as usize;
        assert_eq!(mono.len(), expected_frames);
    }

    #[test]
    fn capture_empty_returns_none_metrics() {
        let cap = AudioCapture::new(44100.0, 1.0);
        assert!(cap.compute_metrics().is_none());
    }

    #[test]
    fn capture_compute_metrics_silence() {
        let cap = AudioCapture::new(44100.0, 1.0);
        let n = 1024;
        let silence = vec![0.0_f64; n];
        cap.write_audio(&silence, &silence, n);

        let m = cap.compute_metrics().unwrap();
        assert!(m.rms_db < -90.0, "RMS of silence should be very low: {}", m.rms_db);
        assert!(m.peak_db < -90.0, "Peak of silence should be very low: {}", m.peak_db);
    }

    #[test]
    fn capture_compute_metrics_loud_signal() {
        let cap = AudioCapture::new(44100.0, 1.0);
        let n = 4096;
        // Full-scale sine wave
        let left: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * 440.0 * i as f64 / 44100.0).sin())
            .collect();
        let right = left.clone();
        cap.write_audio(&left, &right, n);

        let m = cap.compute_metrics().unwrap();
        // Sine wave RMS ~= -3 dB
        assert!(m.rms_db > -5.0 && m.rms_db < -1.0, "RMS: {}", m.rms_db);
        // Peak ~= 0 dB
        assert!(m.peak_db > -1.0 && m.peak_db < 1.0, "Peak: {}", m.peak_db);
        // Crest factor of sine ~= 3 dB
        assert!(
            m.crest_factor_db > 2.0 && m.crest_factor_db < 4.0,
            "Crest: {}",
            m.crest_factor_db
        );
        // 440 Hz spectral centroid
        assert!(
            m.spectral_centroid_hz > 300.0 && m.spectral_centroid_hz < 600.0,
            "Centroid: {}",
            m.spectral_centroid_hz
        );
    }

    #[test]
    fn capture_try_lock_doesnt_block() {
        let cap = AudioCapture::new(44100.0, 1.0);

        // Simulate GUI holding the lock
        let state = cap.inner.lock().unwrap();

        // Audio thread write should silently skip (not deadlock)
        let left = vec![1.0_f64; 64];
        let right = vec![1.0_f64; 64];
        cap.write_audio(&left, &right, 64);

        // Release lock
        drop(state);

        // Buffer should be empty — write was skipped
        let (mono, _) = cap.read_mono(1.0);
        assert!(mono.is_empty());
    }

    #[test]
    fn capture_multiple_writes_accumulate() {
        let cap = AudioCapture::new(44100.0, 3.0);

        for _ in 0..10 {
            let left = vec![0.5_f64; 100];
            let right = vec![0.5_f64; 100];
            cap.write_audio(&left, &right, 100);
        }

        let (mono, _) = cap.read_mono(3.0);
        assert_eq!(mono.len(), 1000); // 10 * 100 frames
    }

    #[test]
    fn zcr_silent_signal() {
        let mono = vec![0.0f32; 100];
        let hz = estimate_spectral_centroid(&mono, 44100.0);
        assert_eq!(hz, 0.0);
    }

    #[test]
    fn zcr_single_sample() {
        let mono = vec![1.0f32];
        let hz = estimate_spectral_centroid(&mono, 44100.0);
        assert_eq!(hz, 0.0);
    }
}
