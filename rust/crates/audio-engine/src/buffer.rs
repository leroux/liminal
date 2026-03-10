/// Multi-channel audio buffer.
#[derive(Debug, Clone)]
pub struct AudioBuffer {
    /// Interleaved sample data (channel-interleaved).
    /// For stereo: [L0, R0, L1, R1, ...]
    /// For mono: [S0, S1, S2, ...]
    pub data: Vec<f64>,
    pub channels: usize,
    pub sample_rate: u32,
}

impl AudioBuffer {
    pub fn new(data: Vec<f64>, channels: usize, sample_rate: u32) -> Self {
        Self {
            data,
            channels,
            sample_rate,
        }
    }

    /// Number of sample frames (total samples / channels).
    pub fn num_frames(&self) -> usize {
        if self.channels == 0 {
            0
        } else {
            self.data.len() / self.channels
        }
    }

    /// Duration in seconds.
    pub fn duration_seconds(&self) -> f64 {
        self.num_frames() as f64 / self.sample_rate as f64
    }

    /// Mix to mono (average channels). Returns a new Vec.
    pub fn to_mono(&self) -> Vec<f64> {
        if self.channels == 1 {
            return self.data.clone();
        }
        let n = self.num_frames();
        let mut mono = Vec::with_capacity(n);
        for i in 0..n {
            let mut sum = 0.0;
            for ch in 0..self.channels {
                sum += self.data[i * self.channels + ch];
            }
            mono.push(sum / self.channels as f64);
        }
        mono
    }

    /// Get left channel (channel 0). Returns a new Vec.
    pub fn left(&self) -> Vec<f64> {
        self.channel(0)
    }

    /// Get right channel (channel 1, or channel 0 if mono). Returns a new Vec.
    pub fn right(&self) -> Vec<f64> {
        if self.channels < 2 {
            self.channel(0)
        } else {
            self.channel(1)
        }
    }

    /// Extract a single channel by index.
    pub fn channel(&self, ch: usize) -> Vec<f64> {
        let n = self.num_frames();
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            out.push(self.data[i * self.channels + ch]);
        }
        out
    }

    /// Create stereo buffer from separate L/R vectors.
    pub fn from_stereo(left: &[f64], right: &[f64], sample_rate: u32) -> Self {
        let n = left.len().min(right.len());
        let mut data = Vec::with_capacity(n * 2);
        for i in 0..n {
            data.push(left[i]);
            data.push(right[i]);
        }
        Self {
            data,
            channels: 2,
            sample_rate,
        }
    }

    /// Create mono buffer from a single channel.
    pub fn from_mono(samples: Vec<f64>, sample_rate: u32) -> Self {
        Self {
            data: samples,
            channels: 1,
            sample_rate,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mono_buffer() {
        let buf = AudioBuffer::from_mono(vec![1.0, 0.5, -0.5], 44100);
        assert_eq!(buf.num_frames(), 3);
        assert_eq!(buf.channels, 1);
        assert!((buf.duration_seconds() - 3.0 / 44100.0).abs() < 1e-10);
    }

    #[test]
    fn test_stereo_buffer() {
        let buf = AudioBuffer::from_stereo(&[1.0, 0.5], &[0.0, -0.5], 44100);
        assert_eq!(buf.num_frames(), 2);
        assert_eq!(buf.channels, 2);
        assert_eq!(buf.left(), vec![1.0, 0.5]);
        assert_eq!(buf.right(), vec![0.0, -0.5]);
    }

    #[test]
    fn test_to_mono() {
        let buf = AudioBuffer::from_stereo(&[1.0, 0.0], &[0.0, 1.0], 44100);
        let mono = buf.to_mono();
        assert_eq!(mono, vec![0.5, 0.5]);
    }
}
