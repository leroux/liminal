/// Audio playback via cpal — port of shared/streaming.py StreamPlayer.
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

/// Information about an audio output device.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub name: String,
    pub is_default: bool,
}

/// Enumerate available output devices.
pub fn enumerate_devices() -> Vec<DeviceInfo> {
    let host = cpal::default_host();
    let default_name = host
        .default_output_device()
        .and_then(|d| d.name().ok())
        .unwrap_or_default();

    let mut devices = Vec::new();
    if let Ok(output_devices) = host.output_devices() {
        for device in output_devices {
            if let Ok(name) = device.name() {
                let is_default = name == default_name;
                devices.push(DeviceInfo { name, is_default });
            }
        }
    }
    devices
}

/// Handle to a playing audio stream. Use to query state and stop playback.
pub struct PlayHandle {
    _stream: cpal::Stream,
    stop_flag: Arc<AtomicBool>,
    position: Arc<AtomicU64>,
    total_frames: u64,
    sample_rate: u32,
}

impl PlayHandle {
    pub fn stop(&self) {
        self.stop_flag.store(true, Ordering::Relaxed);
    }

    pub fn is_playing(&self) -> bool {
        !self.stop_flag.load(Ordering::Relaxed)
    }

    pub fn position_seconds(&self) -> f64 {
        let frames = self.position.load(Ordering::Relaxed);
        frames as f64 / self.sample_rate as f64
    }

    pub fn position_fraction(&self) -> f64 {
        if self.total_frames == 0 {
            return 0.0;
        }
        let frames = self.position.load(Ordering::Relaxed);
        frames as f64 / self.total_frames as f64
    }
}

/// Play stereo f64 audio on the default (or named) output device.
/// Returns a PlayHandle for controlling playback.
pub fn play(
    left: &[f64],
    right: &[f64],
    sample_rate: u32,
    device_name: Option<&str>,
) -> Result<PlayHandle, PlaybackError> {
    let host = cpal::default_host();

    let device = match device_name {
        Some(name) => host
            .output_devices()
            .map_err(|e| PlaybackError(e.to_string()))?
            .find(|d| d.name().map(|n| n == name).unwrap_or(false))
            .ok_or_else(|| PlaybackError(format!("device not found: {name}")))?,
        None => host
            .default_output_device()
            .ok_or_else(|| PlaybackError("no default output device".to_string()))?,
    };

    let config = cpal::StreamConfig {
        channels: 2,
        sample_rate: cpal::SampleRate(sample_rate),
        buffer_size: cpal::BufferSize::Default,
    };

    let total_frames = left.len().min(right.len()) as u64;
    let stop_flag = Arc::new(AtomicBool::new(false));
    let position = Arc::new(AtomicU64::new(0));

    // Copy audio data for the callback thread
    let audio_l: Vec<f32> = left.iter().map(|&s| s as f32).collect();
    let audio_r: Vec<f32> = right.iter().map(|&s| s as f32).collect();

    let stop_clone = stop_flag.clone();
    let pos_clone = position.clone();
    let n_frames = total_frames as usize;

    let stream = device
        .build_output_stream(
            &config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                let mut frame_idx = pos_clone.load(Ordering::Relaxed) as usize;

                for chunk in data.chunks_mut(2) {
                    if frame_idx >= n_frames || stop_clone.load(Ordering::Relaxed) {
                        chunk.fill(0.0);
                        if frame_idx >= n_frames {
                            stop_clone.store(true, Ordering::Relaxed);
                        }
                    } else {
                        chunk[0] = audio_l[frame_idx];
                        chunk[1] = if frame_idx < audio_r.len() {
                            audio_r[frame_idx]
                        } else {
                            audio_l[frame_idx]
                        };
                        frame_idx += 1;
                    }
                }
                pos_clone.store(frame_idx as u64, Ordering::Relaxed);
            },
            move |err| {
                eprintln!("audio playback error: {err}");
            },
            None,
        )
        .map_err(|e| PlaybackError(e.to_string()))?;

    stream.play().map_err(|e| PlaybackError(e.to_string()))?;

    Ok(PlayHandle {
        _stream: stream,
        stop_flag,
        position,
        total_frames,
        sample_rate,
    })
}

#[derive(Debug)]
pub struct PlaybackError(pub String);

impl std::fmt::Display for PlaybackError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Playback error: {}", self.0)
    }
}

impl std::error::Error for PlaybackError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enumerate_devices() {
        // Should not panic even if no audio devices are available
        let devices = enumerate_devices();
        // Just verify it returns without error
        let _ = devices;
    }
}
