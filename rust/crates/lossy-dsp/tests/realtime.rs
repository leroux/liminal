//! Real-time performance tests for lossy DSP.
//!
//! These tests assert that processing completes faster than real-time,
//! ensuring the DSP can run in a live audio context.
//!
//! RTF = processing_time / audio_duration (< 1.0 means faster than real-time)

use lossy_dsp::chain::{render_lossy, render_lossy_stereo};
use lossy_dsp::params::{LossyParams, SR};
use std::time::Instant;

const AUDIO_DURATION_SECS: f64 = 5.0;
const N_SAMPLES: usize = (SR * AUDIO_DURATION_SECS) as usize;

/// Maximum allowed RTF.
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
fn realtime_default() {
    let input = make_sine(N_SAMPLES);
    let params = LossyParams::default();
    let rtf = measure_rtf(N_SAMPLES, || render_lossy(&input, &params));
    assert!(
        rtf < MAX_RTF,
        "Default RTF {rtf:.4} exceeds {MAX_RTF}"
    );
}

#[test]
fn realtime_spectral_heavy() {
    let input = make_sine(N_SAMPLES);
    let mut params = LossyParams::default();
    params.loss = 0.9;
    params.jitter = 0.5;
    params.phase_loss = 0.5;
    params.noise_shape = 0.5;
    params.pre_echo = 0.3;
    let rtf = measure_rtf(N_SAMPLES, || render_lossy(&input, &params));
    assert!(
        rtf < MAX_RTF,
        "Spectral heavy RTF {rtf:.4} exceeds {MAX_RTF}"
    );
}

#[test]
fn realtime_full_chain() {
    let input = make_sine(N_SAMPLES);
    let mut params = LossyParams::default();
    params.loss = 0.6;
    params.crush = 0.3;
    params.decimate = 0.2;
    params.verb = 0.4;
    params.decay = 0.5;
    params.filter_type = 1;
    params.filter_freq = 2000.0;
    params.packets = 1;
    params.packet_rate = 0.2;
    params.gate = 0.1;
    let rtf = measure_rtf(N_SAMPLES, || render_lossy(&input, &params));
    assert!(
        rtf < MAX_RTF,
        "Full chain RTF {rtf:.4} exceeds {MAX_RTF}"
    );
}

#[test]
fn realtime_large_window() {
    let input = make_sine(N_SAMPLES);
    let mut params = LossyParams::default();
    params.window_size = 8192;
    params.loss = 0.5;
    let rtf = measure_rtf(N_SAMPLES, || render_lossy(&input, &params));
    assert!(
        rtf < MAX_RTF,
        "Large window (8192) RTF {rtf:.4} exceeds {MAX_RTF}"
    );
}

#[test]
fn realtime_small_window() {
    let input = make_sine(N_SAMPLES);
    let mut params = LossyParams::default();
    params.window_size = 512;
    params.loss = 0.5;
    let rtf = measure_rtf(N_SAMPLES, || render_lossy(&input, &params));
    assert!(
        rtf < MAX_RTF,
        "Small window (512) RTF {rtf:.4} exceeds {MAX_RTF}"
    );
}

#[test]
fn realtime_bounce() {
    let input = make_sine(N_SAMPLES);
    let mut params = LossyParams::default();
    params.bounce = 1;
    params.bounce_target = 0;
    params.bounce_rate = 0.5;
    params.loss = 0.5;
    let rtf = measure_rtf(N_SAMPLES, || render_lossy(&input, &params));
    // Bounce processes in ~50ms blocks with repeated spectral analysis, allow more headroom
    assert!(
        rtf < MAX_RTF * 2.0,
        "Bounce RTF {rtf:.4} exceeds {}", MAX_RTF * 2.0
    );
}

#[test]
fn realtime_stereo() {
    let left = make_sine(N_SAMPLES);
    let right = make_sine(N_SAMPLES);
    let params = LossyParams::default();
    let rtf = measure_rtf_stereo(N_SAMPLES, || {
        render_lossy_stereo(&left, &right, &params)
    });
    // Stereo processes 2 independent channels
    assert!(
        rtf < MAX_RTF * 2.0,
        "Stereo RTF {rtf:.4} exceeds {}", MAX_RTF * 2.0
    );
}

/// Verify that output is always finite across various configurations.
#[test]
fn output_safety_all_configs() {
    let input = make_sine(44100);

    let configs: Vec<(&str, LossyParams)> = vec![
        ("default", LossyParams::default()),
        ("max_loss", {
            let mut p = LossyParams::default();
            p.loss = 1.0;
            p
        }),
        ("max_crush", {
            let mut p = LossyParams::default();
            p.crush = 1.0;
            p.decimate = 1.0;
            p
        }),
        ("heavy_verb", {
            let mut p = LossyParams::default();
            p.verb = 1.0;
            p.decay = 1.0;
            p
        }),
        ("inverse_mode", {
            let mut p = LossyParams::default();
            p.inverse = 1;
            p.loss = 0.5;
            p
        }),
        ("full_effects", {
            let mut p = LossyParams::default();
            p.loss = 0.8;
            p.crush = 0.5;
            p.decimate = 0.3;
            p.verb = 0.5;
            p.filter_type = 2; // notch
            p.filter_freq = 5000.0;
            p.packets = 2; // repeat
            p.packet_rate = 0.3;
            p.gate = 0.2;
            p
        }),
    ];

    for (name, params) in &configs {
        let output = render_lossy(&input, params);
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
