//! Real-time performance tests for reverb DSP.
//!
//! These tests assert that processing completes faster than real-time,
//! ensuring the DSP can run in a live audio context. Each test processes
//! audio and asserts the Real-Time Factor (RTF) is below a threshold.
//!
//! RTF = processing_time / audio_duration (< 1.0 means faster than real-time)

use reverb_dsp::chain::{render_fdn, render_fdn_stereo};
use reverb_dsp::params::{ReverbParams, N, SR};
use reverb_dsp::processor::{FdnProcessor, StereoFdnProcessor};
use std::time::Instant;

const AUDIO_DURATION_SECS: f64 = 5.0;
const N_SAMPLES: usize = (SR * AUDIO_DURATION_SECS) as usize;

/// Maximum allowed RTF. Well under 1.0 to leave headroom for the host,
/// other plugins, and system overhead.
const MAX_RTF: f64 = 0.5;

fn make_sine(len: usize) -> Vec<f64> {
    (0..len)
        .map(|i| (2.0 * std::f64::consts::PI * 440.0 * i as f64 / SR).sin())
        .collect()
}

fn measure_rtf<F: FnOnce() -> Vec<f64>>(n_samples: usize, f: F) -> f64 {
    let audio_secs = n_samples as f64 / SR;
    let start = Instant::now();
    let result = f();
    let elapsed = start.elapsed().as_secs_f64();
    // Ensure the result isn't optimized away
    assert!(!result.is_empty());
    let rtf = elapsed / audio_secs;
    eprintln!(
        "  {n_samples} samples ({audio_secs:.2}s audio): {elapsed:.4}s processing, RTF = {rtf:.4} ({:.0}x realtime)",
        1.0 / rtf
    );
    rtf
}

fn measure_rtf_stereo<F: FnOnce() -> (Vec<f64>, Vec<f64>)>(n_samples: usize, f: F) -> f64 {
    let audio_secs = n_samples as f64 / SR;
    let start = Instant::now();
    let (l, r) = f();
    let elapsed = start.elapsed().as_secs_f64();
    assert!(!l.is_empty() && !r.is_empty());
    let rtf = elapsed / audio_secs;
    eprintln!(
        "  {n_samples} samples ({audio_secs:.2}s audio): {elapsed:.4}s processing, RTF = {rtf:.4} ({:.0}x realtime)",
        1.0 / rtf
    );
    rtf
}

#[test]
fn realtime_static_householder() {
    let input = make_sine(N_SAMPLES);
    let params = ReverbParams::default();
    let rtf = measure_rtf(N_SAMPLES, || render_fdn(&input, &params));
    assert!(
        rtf < MAX_RTF,
        "Static Householder RTF {rtf:.4} exceeds {MAX_RTF}"
    );
}

#[test]
fn realtime_static_random_orthogonal() {
    let input = make_sine(N_SAMPLES);
    let mut params = ReverbParams::default();
    params.matrix_type = "random_orthogonal".to_string();
    let rtf = measure_rtf(N_SAMPLES, || render_fdn(&input, &params));
    assert!(
        rtf < MAX_RTF,
        "Static random_orthogonal RTF {rtf:.4} exceeds {MAX_RTF}"
    );
}

#[test]
fn realtime_modulated() {
    let input = make_sine(N_SAMPLES);
    let mut params = ReverbParams::default();
    params.mod_master_rate = 2.0;
    params.mod_depth_delay = vec![5.0; N];
    params.mod_depth_damping = vec![0.1; N];
    params.mod_depth_output = vec![0.05; N];
    let rtf = measure_rtf(N_SAMPLES, || render_fdn(&input, &params));
    assert!(
        rtf < MAX_RTF,
        "Modulated RTF {rtf:.4} exceeds {MAX_RTF}"
    );
}

#[test]
fn realtime_heavy_modulated() {
    let input = make_sine(N_SAMPLES);
    let mut params = ReverbParams::default();
    params.matrix_type = "random_orthogonal".to_string();
    params.saturation = 0.8;
    params.diffusion = 0.7;
    params.diffusion_stages = 4;
    params.mod_master_rate = 3.0;
    params.mod_depth_delay = vec![8.0; N];
    params.mod_depth_damping = vec![0.2; N];
    params.mod_depth_output = vec![0.1; N];
    params.mod_depth_matrix = 0.5;
    let rtf = measure_rtf(N_SAMPLES, || render_fdn(&input, &params));
    assert!(
        rtf < MAX_RTF,
        "Heavy modulated RTF {rtf:.4} exceeds {MAX_RTF}"
    );
}

#[test]
fn realtime_stereo() {
    let left = make_sine(N_SAMPLES);
    let right = make_sine(N_SAMPLES);
    let params = ReverbParams::default();
    let rtf = measure_rtf_stereo(N_SAMPLES, || {
        render_fdn_stereo(&left, &right, &params)
    });
    assert!(
        rtf < MAX_RTF,
        "Stereo RTF {rtf:.4} exceeds {MAX_RTF}"
    );
}

#[test]
fn realtime_stereo_modulated() {
    let left = make_sine(N_SAMPLES);
    let right = make_sine(N_SAMPLES);
    let mut params = ReverbParams::default();
    params.mod_master_rate = 2.0;
    params.mod_depth_delay = vec![5.0; N];
    params.mod_depth_matrix = 0.3;
    let rtf = measure_rtf_stereo(N_SAMPLES, || {
        render_fdn_stereo(&left, &right, &params)
    });
    // Stereo modulated processes 2 channels, so allow slightly higher RTF
    assert!(
        rtf < MAX_RTF * 2.0,
        "Stereo modulated RTF {rtf:.4} exceeds {}", MAX_RTF * 2.0
    );
}

// --- Pre-allocated processor tests (plugin path) ---

#[test]
fn realtime_processor_small_buffers() {
    // Simulate DAW calling with 256-sample buffers for 5 seconds of audio.
    // This is the actual plugin path — must be fast with no spikes.
    let input = make_sine(256);
    let mut params = ReverbParams::default();
    params.normalize();
    let mut proc = FdnProcessor::new();
    let mut output = vec![0.0; 256 * 2];

    let n_calls = N_SAMPLES / 256;
    let audio_secs = AUDIO_DURATION_SECS;
    let start = Instant::now();
    for _ in 0..n_calls {
        proc.process(&input, &params, &mut output);
    }
    let elapsed = start.elapsed().as_secs_f64();
    let rtf = elapsed / audio_secs;
    eprintln!(
        "  Processor 256×{n_calls}: {elapsed:.4}s processing, RTF = {rtf:.4} ({:.0}x realtime)",
        1.0 / rtf
    );
    assert!(
        rtf < MAX_RTF,
        "Processor small-buffer RTF {rtf:.4} exceeds {MAX_RTF}"
    );
}

#[test]
fn realtime_processor_stereo() {
    let left = make_sine(N_SAMPLES);
    let right = make_sine(N_SAMPLES);
    let mut params = ReverbParams::default();
    params.normalize();
    let mut proc = StereoFdnProcessor::new();
    let mut out_l = vec![0.0; N_SAMPLES];
    let mut out_r = vec![0.0; N_SAMPLES];

    let audio_secs = N_SAMPLES as f64 / SR;
    let start = Instant::now();
    proc.process_stereo(&left, &right, &params, &mut out_l, &mut out_r);
    let elapsed = start.elapsed().as_secs_f64();
    let rtf = elapsed / audio_secs;
    eprintln!(
        "  StereoProcessor: {elapsed:.4}s processing, RTF = {rtf:.4} ({:.0}x realtime)",
        1.0 / rtf
    );
    assert!(
        rtf < MAX_RTF,
        "Stereo processor RTF {rtf:.4} exceeds {MAX_RTF}"
    );
}

/// Verify that output is always finite and within safe bounds.
#[test]
fn output_safety_all_configs() {
    let input = make_sine(44100);

    let configs: Vec<(&str, ReverbParams)> = vec![
        ("default", ReverbParams::default()),
        ("high_feedback", {
            let mut p = ReverbParams::default();
            p.feedback_gain = 0.99;
            p
        }),
        ("max_saturation", {
            let mut p = ReverbParams::default();
            p.saturation = 1.0;
            p
        }),
        ("all_modulation", {
            let mut p = ReverbParams::default();
            p.mod_master_rate = 5.0;
            p.mod_depth_delay = vec![10.0; N];
            p.mod_depth_damping = vec![0.5; N];
            p.mod_depth_output = vec![0.5; N];
            p.mod_depth_matrix = 1.0;
            p
        }),
    ];

    for (name, params) in &configs {
        let output = render_fdn(&input, params);
        assert!(
            output.iter().all(|x| x.is_finite()),
            "{name}: output contains non-finite values"
        );
        let peak = output.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
        assert!(
            peak < 1e6,
            "{name}: peak {peak:.2} exceeds safety limit"
        );
    }
}
