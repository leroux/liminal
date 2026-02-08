//! Fixed-size circular audio buffer for capture snapshots.
//!
//! Used by the plugin to keep a rolling window of recent audio
//! that can be dumped to WAV on request (e.g. for spectrogram analysis).

/// Circular buffer storing the most recent `capacity` samples.
pub struct RingBuffer {
    data: Vec<f64>,
    write_pos: usize,
    filled: bool,
}

impl RingBuffer {
    /// Create a ring buffer with given capacity in samples.
    pub fn new(capacity: usize) -> Self {
        Self {
            data: vec![0.0; capacity],
            write_pos: 0,
            filled: false,
        }
    }

    /// Create a ring buffer sized for `duration_secs` at `sample_rate`.
    pub fn with_duration(duration_secs: f64, sample_rate: u32) -> Self {
        Self::new((duration_secs * sample_rate as f64) as usize)
    }

    /// Write one sample.
    #[inline]
    pub fn push(&mut self, sample: f64) {
        self.data[self.write_pos] = sample;
        self.write_pos += 1;
        if self.write_pos >= self.data.len() {
            self.write_pos = 0;
            self.filled = true;
        }
    }

    /// Write a slice of samples.
    pub fn push_slice(&mut self, samples: &[f64]) {
        for &s in samples {
            self.push(s);
        }
    }

    /// Number of valid samples available.
    pub fn len(&self) -> usize {
        if self.filled {
            self.data.len()
        } else {
            self.write_pos
        }
    }

    pub fn is_empty(&self) -> bool {
        !self.filled && self.write_pos == 0
    }

    /// Read all valid samples in chronological order.
    pub fn read(&self) -> Vec<f64> {
        if self.filled {
            let mut out = Vec::with_capacity(self.data.len());
            out.extend_from_slice(&self.data[self.write_pos..]);
            out.extend_from_slice(&self.data[..self.write_pos]);
            out
        } else {
            self.data[..self.write_pos].to_vec()
        }
    }

    /// Dump contents to a WAV file.
    pub fn dump_to_wav(&self, path: &str, sample_rate: u32) -> Result<(), hound::Error> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut writer = hound::WavWriter::create(path, spec)?;
        for sample in self.read() {
            writer.write_sample(sample as f32)?;
        }
        writer.finalize()
    }

    /// Clear the buffer.
    pub fn clear(&mut self) {
        self.write_pos = 0;
        self.filled = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_push_read() {
        let mut rb = RingBuffer::new(4);
        rb.push(1.0);
        rb.push(2.0);
        rb.push(3.0);
        assert_eq!(rb.len(), 3);
        assert_eq!(rb.read(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn wraps_around() {
        let mut rb = RingBuffer::new(4);
        for i in 0..6 {
            rb.push(i as f64);
        }
        assert_eq!(rb.len(), 4);
        assert_eq!(rb.read(), vec![2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn push_slice() {
        let mut rb = RingBuffer::new(4);
        rb.push_slice(&[10.0, 20.0, 30.0, 40.0, 50.0]);
        assert_eq!(rb.read(), vec![20.0, 30.0, 40.0, 50.0]);
    }

    #[test]
    fn clear_resets() {
        let mut rb = RingBuffer::new(4);
        rb.push_slice(&[1.0, 2.0, 3.0]);
        rb.clear();
        assert!(rb.is_empty());
        assert_eq!(rb.len(), 0);
    }
}
