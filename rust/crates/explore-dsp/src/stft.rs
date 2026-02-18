//! STFT / ISTFT implementation using realfft.

use realfft::RealFftPlanner;
use num_complex::Complex;
use std::f32::consts::PI;

/// Forward STFT. Returns complex frames of shape (num_frames, fft_size/2+1).
pub fn stft(x: &[f32], fft_size: usize, hop_size: usize) -> Vec<Vec<Complex<f32>>> {
    let window = hann(fft_size);
    let mut padded;
    let signal = if x.len() < fft_size {
        padded = vec![0.0f32; fft_size];
        padded[..x.len()].copy_from_slice(x);
        &padded
    } else {
        x
    };
    let n = signal.len();
    let num_frames = if n >= fft_size { 1 + (n - fft_size) / hop_size } else { 1 };

    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(fft_size);
    let mut scratch = fft.make_scratch_vec();

    let mut result = Vec::with_capacity(num_frames);
    for i in 0..num_frames {
        let start = i * hop_size;
        let end = (start + fft_size).min(n);
        let mut frame = vec![0.0f32; fft_size];
        for j in 0..(end - start) {
            frame[j] = signal[start + j] * window[j];
        }
        let mut spectrum = fft.make_output_vec();
        fft.process_with_scratch(&mut frame, &mut spectrum, &mut scratch).unwrap();
        result.push(spectrum);
    }
    result
}

/// Inverse STFT with overlap-add.
pub fn istft(frames: &[Vec<Complex<f32>>], fft_size: usize, hop_size: usize, length: Option<usize>) -> Vec<f32> {
    let window = hann(fft_size);
    let num_frames = frames.len();
    let out_len = length.unwrap_or(fft_size + (num_frames.saturating_sub(1)) * hop_size);

    let mut planner = RealFftPlanner::<f32>::new();
    let ifft = planner.plan_fft_inverse(fft_size);
    let mut scratch = ifft.make_scratch_vec();

    let mut output = vec![0.0f32; out_len];
    for (i, spectrum) in frames.iter().enumerate() {
        let mut spec = spectrum.clone();
        let mut frame = ifft.make_output_vec();
        ifft.process_with_scratch(&mut spec, &mut frame, &mut scratch).unwrap();
        // realfft inverse is unnormalized, divide by fft_size
        let norm = 1.0 / fft_size as f32;
        let start = i * hop_size;
        for j in 0..fft_size {
            let out_idx = start + j;
            if out_idx < out_len {
                output[out_idx] += frame[j] * norm * window[j];
            }
        }
    }
    output
}

fn hann(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (size - 1) as f32).cos()))
        .collect()
}
