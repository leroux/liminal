/// WAV I/O — port of shared/audio.py.
use crate::audio::buffer::AudioBuffer;
use std::path::Path;

/// Load a WAV file and optionally resample to target sample rate.
pub fn load_wav(path: &Path, target_sr: u32) -> Result<AudioBuffer, WavError> {
    let reader = hound::WavReader::open(path).map_err(WavError::Hound)?;
    let spec = reader.spec();
    let file_sr = spec.sample_rate;
    let channels = spec.channels as usize;

    let samples: Vec<f64> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max_val = (1u64 << (spec.bits_per_sample - 1)) as f64;
            reader
                .into_samples::<i32>()
                .map(|s| s.map(|v| v as f64 / max_val))
                .collect::<Result<Vec<_>, _>>()
                .map_err(WavError::Hound)?
        }
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .map(|s| s.map(|v| v as f64))
            .collect::<Result<Vec<_>, _>>()
            .map_err(WavError::Hound)?,
    };

    if file_sr == target_sr {
        return Ok(AudioBuffer::new(samples, channels, target_sr));
    }

    // Resample using rubato
    resample_buffer(&samples, channels, file_sr, target_sr)
}

/// Save audio to a 16-bit WAV file with peak normalization.
/// Port of shared/audio.py save_wav.
pub fn save_wav(path: &Path, audio: &AudioBuffer) -> Result<(), WavError> {
    let spec = hound::WavSpec {
        channels: audio.channels as u16,
        sample_rate: audio.sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec).map_err(WavError::Hound)?;

    // Peak normalize
    let peak = audio
        .data
        .iter()
        .map(|s| s.abs())
        .fold(0.0_f64, f64::max);
    let gain = if peak > 1.0 {
        0.95 / peak
    } else if peak > 0.0 && peak < 0.1 {
        0.9 / peak
    } else {
        1.0
    };

    let max_i16 = i16::MAX as f64;
    for &sample in &audio.data {
        let normalized = (sample * gain).clamp(-1.0, 1.0);
        writer
            .write_sample((normalized * max_i16) as i16)
            .map_err(WavError::Hound)?;
    }
    writer.finalize().map_err(WavError::Hound)?;
    Ok(())
}

/// Generate a unit impulse (click) for testing.
/// Port of shared/audio.py make_impulse.
pub fn make_impulse(sr: u32, seconds: f64) -> AudioBuffer {
    let n = (sr as f64 * seconds) as usize;
    let mut data = vec![0.0; n];
    if !data.is_empty() {
        data[0] = 1.0;
    }
    AudioBuffer::from_mono(data, sr)
}

/// Resample interleaved audio data using rubato.
fn resample_buffer(
    samples: &[f64],
    channels: usize,
    from_sr: u32,
    to_sr: u32,
) -> Result<AudioBuffer, WavError> {
    use rubato::{FftFixedInOut, Resampler};

    let chunk_size = 1024;
    let mut resampler = FftFixedInOut::<f64>::new(from_sr as usize, to_sr as usize, chunk_size, channels)
        .map_err(|e| WavError::Resample(e.to_string()))?;

    let frames = samples.len() / channels;

    // De-interleave into per-channel buffers
    let mut channel_data: Vec<Vec<f64>> = (0..channels)
        .map(|ch| {
            (0..frames)
                .map(|i| samples[i * channels + ch])
                .collect()
        })
        .collect();

    // Pad to multiple of chunk_size
    let input_chunk_size = resampler.input_frames_next();
    let remainder = frames % input_chunk_size;
    if remainder != 0 {
        let pad = input_chunk_size - remainder;
        for ch in &mut channel_data {
            ch.extend(std::iter::repeat_n(0.0, pad));
        }
    }

    let padded_frames = channel_data[0].len();
    let mut output_channels: Vec<Vec<f64>> = vec![Vec::new(); channels];

    let mut pos = 0;
    while pos + input_chunk_size <= padded_frames {
        let chunk_in: Vec<&[f64]> = channel_data
            .iter()
            .map(|ch| &ch[pos..pos + input_chunk_size])
            .collect();
        let out = resampler
            .process(&chunk_in, None)
            .map_err(|e| WavError::Resample(e.to_string()))?;
        for (ch_idx, ch_out) in out.iter().enumerate() {
            output_channels[ch_idx].extend_from_slice(ch_out);
        }
        pos += input_chunk_size;
    }

    // Trim output to expected length
    let expected_frames = (frames as f64 * to_sr as f64 / from_sr as f64).ceil() as usize;
    for ch in &mut output_channels {
        ch.truncate(expected_frames);
    }
    let actual_frames = output_channels[0].len();

    // Re-interleave
    let mut interleaved = Vec::with_capacity(actual_frames * channels);
    for i in 0..actual_frames {
        for ch in &output_channels {
            interleaved.push(ch[i]);
        }
    }

    Ok(AudioBuffer::new(interleaved, channels, to_sr))
}

#[derive(Debug)]
pub enum WavError {
    Hound(hound::Error),
    Resample(String),
}

impl std::fmt::Display for WavError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WavError::Hound(e) => write!(f, "WAV error: {e}"),
            WavError::Resample(e) => write!(f, "Resample error: {e}"),
        }
    }
}

impl std::error::Error for WavError {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_make_impulse() {
        let buf = make_impulse(44100, 0.5);
        assert_eq!(buf.num_frames(), 22050);
        assert_eq!(buf.data[0], 1.0);
        assert_eq!(buf.data[1], 0.0);
        assert_eq!(buf.channels, 1);
    }

    #[test]
    fn test_wav_round_trip() {
        let dir = std::env::temp_dir();
        let path = dir.join("audio_engine_test_round_trip.wav");

        let original = AudioBuffer::from_stereo(
            &[0.5, -0.3, 0.8, -0.1],
            &[-0.5, 0.3, -0.8, 0.1],
            44100,
        );
        save_wav(&path, &original).unwrap();

        let loaded = load_wav(&path, 44100).unwrap();
        assert_eq!(loaded.channels, 2);
        assert_eq!(loaded.num_frames(), 4);
        assert_eq!(loaded.sample_rate, 44100);

        // 16-bit quantization means ~1/32768 error
        for (a, b) in original.data.iter().zip(loaded.data.iter()) {
            // The original gets peak-normalized, so compare normalized values
            assert!((a.abs() - b.abs()).abs() < 0.01, "a={a}, b={b}");
        }

        std::fs::remove_file(&path).ok();
    }
}
