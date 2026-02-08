//! Packet loss and repeat simulation.
//!
//! Port of `lossy/engine/packets.py`.
//!
//! Uses a Gilbert-Elliott two-state Markov model for bursty dropout patterns.
//! Short Hann crossfades at packet boundaries prevent clicks.

use crate::params::{LossyParams, SR};
use crate::rng::NumpyRng;

/// Crossfade length at packet boundaries (~3 ms at 44.1 kHz).
const XFADE_SAMPLES: usize = (0.003 * SR) as usize;

/// Apply packet loss or repeat simulation.
pub fn packet_process(audio: &[f64], params: &LossyParams) -> Vec<f64> {
    let mode = params.packets;
    if mode == 0 {
        return audio.to_vec();
    }

    let g = params.global_amount;
    let rate = params.packet_rate * g;
    let packet_ms = params.packet_size;
    let seed = params.seed;

    if rate <= 0.0 {
        return audio.to_vec();
    }

    let packet_samples = (packet_ms * SR / 1000.0).max(1.0) as usize;
    let mut rng = NumpyRng::new((seed + 1000) as u32);

    let mut output = audio.to_vec();
    let n = output.len();

    // Gilbert-Elliott: Good <-> Bad
    let p_g2b = rate * 0.3;
    let p_b2g = 0.4;
    let mut in_bad = false;
    let mut prev_bad = false;

    let mut last_good = vec![0.0_f64; packet_samples];

    // Pre-compute crossfade windows (Hann halves)
    let xfade = XFADE_SAMPLES.min(packet_samples / 4);
    let fade_in = hann_fade_in(xfade);
    let fade_out = hann_fade_out(xfade);

    let mut start = 0;
    while start < n {
        let end = (start + packet_samples).min(n);
        let chunk_len = end - start;

        if in_bad {
            if mode == 1 {
                // Packet loss -> silence
                for i in start..end {
                    output[i] = 0.0;
                }
            } else if mode == 2 {
                // Packet repeat -> stutter
                for i in 0..chunk_len {
                    output[start + i] = last_good[i];
                }
            }

            // Crossfade at boundary entering bad state
            if !prev_bad && xfade > 0 && start > 0 {
                let xf = xfade.min(start).min(chunk_len);
                for i in 0..xf {
                    output[start + i] *= fade_in[i];
                    output[start + i] += audio[start + i] * fade_out[fade_out.len() - xf + i];
                }
            }

            if rng.random() < p_b2g {
                prev_bad = true;
                in_bad = false;
            } else {
                prev_bad = true;
            }
        } else {
            // Crossfade at boundary leaving bad state
            if prev_bad && xfade > 0 {
                let xf = xfade.min(chunk_len);
                for i in 0..xf {
                    output[start + i] *= fade_in[i];
                }
            }

            for i in 0..chunk_len {
                last_good[i] = audio[start + i];
            }
            prev_bad = false;
            if rng.random() < p_g2b {
                in_bad = true;
            }
        }

        start = end;
    }

    output
}

/// First half of Hann window (fade in: 0 -> 1).
fn hann_fade_in(len: usize) -> Vec<f64> {
    if len == 0 {
        return vec![];
    }
    let full_len = len * 2;
    (0..len)
        .map(|i| {
            0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / full_len as f64).cos())
        })
        .collect()
}

/// Second half of Hann window (fade out: 1 -> 0).
fn hann_fade_out(len: usize) -> Vec<f64> {
    if len == 0 {
        return vec![];
    }
    let full_len = len * 2;
    (len..full_len)
        .map(|i| {
            0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / full_len as f64).cos())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clean_passthrough() {
        let audio: Vec<f64> = (0..10000).map(|i| (i as f64 * 0.01).sin()).collect();
        let params = LossyParams::default(); // packets=0
        let out = packet_process(&audio, &params);
        for (a, b) in audio.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
    }

    #[test]
    fn test_packet_loss_creates_silence() {
        let audio = vec![1.0_f64; 44100]; // 1 second of DC
        let mut params = LossyParams::default();
        params.packets = 1; // loss
        params.packet_rate = 0.8; // high rate
        let out = packet_process(&audio, &params);
        // Should have some zeros
        let zeros = out.iter().filter(|&&x| x.abs() < 1e-10).count();
        assert!(zeros > 0);
    }

    #[test]
    fn test_packet_repeat_creates_stutters() {
        // Ramp signal - repeats will show as non-monotonic
        let audio: Vec<f64> = (0..44100).map(|i| i as f64 / 44100.0).collect();
        let mut params = LossyParams::default();
        params.packets = 2; // repeat
        params.packet_rate = 0.8;
        let out = packet_process(&audio, &params);
        // Should have some backwards jumps (stutters)
        let mut backwards = 0;
        for i in 1..out.len() {
            if out[i] < out[i - 1] - 0.01 {
                backwards += 1;
            }
        }
        assert!(backwards > 0);
    }

    #[test]
    fn test_hann_windows() {
        let fi = hann_fade_in(100);
        assert_eq!(fi.len(), 100);
        assert!(fi[0] < 0.01); // starts near 0
        assert!(fi[99] > 0.95); // ends near 1

        let fo = hann_fade_out(100);
        assert_eq!(fo.len(), 100);
        assert!(fo[0] > 0.95); // starts near 1
        assert!(fo[99] < 0.01); // ends near 0
    }
}
