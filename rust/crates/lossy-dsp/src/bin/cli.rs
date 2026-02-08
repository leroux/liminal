//! CLI tool for testing the lossy DSP engine.
//!
//! Usage: lossy-cli <input.wav> <output.wav> [preset.json]
//!
//! Reads WAV, applies lossy processing, writes output WAV.
//! If no preset given, uses default params.

use lossy_dsp::{render_lossy, render_lossy_stereo, LossyParams};
use std::env;
use std::fs;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: lossy-cli <input.wav> <output.wav> [preset.json]");
        std::process::exit(1);
    }

    let input_path = &args[1];
    let output_path = &args[2];
    let preset_path = args.get(3);

    // Load params
    let params = if let Some(path) = preset_path {
        let json = fs::read_to_string(path).unwrap_or_else(|e| {
            eprintln!("Failed to read preset {}: {}", path, e);
            std::process::exit(1);
        });
        LossyParams::from_json_with_defaults(&json)
    } else {
        LossyParams::default()
    };

    // Read input WAV
    let reader = hound::WavReader::open(input_path).unwrap_or_else(|e| {
        eprintln!("Failed to open {}: {}", input_path, e);
        std::process::exit(1);
    });

    let spec = reader.spec();
    let channels = spec.channels as usize;
    let sample_rate = spec.sample_rate;
    let bits = spec.bits_per_sample;

    eprintln!(
        "Input: {} ch, {} Hz, {}-bit, {} samples/ch",
        channels,
        sample_rate,
        bits,
        reader.len() as usize / channels
    );

    // Read samples as f64
    let samples: Vec<f64> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max_val = (1_i64 << (bits - 1)) as f64;
            reader
                .into_samples::<i32>()
                .map(|s| s.unwrap() as f64 / max_val)
                .collect()
        }
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .map(|s| s.unwrap() as f64)
            .collect(),
    };

    let n_samples = samples.len() / channels;

    // Process
    let output_samples = if channels == 1 {
        eprintln!("Processing mono...");
        render_lossy(&samples, &params)
    } else if channels == 2 {
        // Deinterleave
        let mut left = Vec::with_capacity(n_samples);
        let mut right = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            left.push(samples[i * 2]);
            right.push(samples[i * 2 + 1]);
        }

        eprintln!("Processing stereo...");
        let (out_l, out_r) = render_lossy_stereo(&left, &right, &params);

        // Interleave
        let mut interleaved = Vec::with_capacity(out_l.len() * 2);
        for i in 0..out_l.len() {
            interleaved.push(out_l[i]);
            interleaved.push(out_r[i]);
        }
        interleaved
    } else {
        eprintln!("Unsupported channel count: {}", channels);
        std::process::exit(1);
    };

    // Write output WAV (same format as input but always 32-bit float)
    let out_spec = hound::WavSpec {
        channels: channels as u16,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };

    let mut writer = hound::WavWriter::create(output_path, out_spec).unwrap_or_else(|e| {
        eprintln!("Failed to create {}: {}", output_path, e);
        std::process::exit(1);
    });

    for &s in &output_samples {
        writer.write_sample(s as f32).unwrap();
    }
    writer.finalize().unwrap();

    eprintln!("Written {} ({} samples)", output_path, output_samples.len() / channels);
}
