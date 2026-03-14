//! Performance tests for fractal DSP.
//!
//! Fractal can't run in true real-time (needs full buffer for resampling),
//! but we still measure RTF to track performance and catch regressions.

use fractal_dsp::chain::{render_fractal, render_fractal_stereo};
use fractal_dsp::params::{FractalParams, SR};
use std::time::Instant;

const AUDIO_DURATION_SECS: f64 = 5.0;
const N_SAMPLES: usize = (SR * AUDIO_DURATION_SECS) as usize;

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
fn perf_default() {
    let input = make_sine(N_SAMPLES);
    let params = FractalParams::default();
    let rtf = measure_rtf(N_SAMPLES, || render_fractal(&input, &params));
    assert!(rtf < 1.0, "Default RTF {rtf:.4} exceeds 1.0");
}

#[test]
fn perf_many_scales() {
    let input = make_sine(N_SAMPLES);
    let mut params = FractalParams::default();
    params.num_scales = 7;
    let rtf = measure_rtf(N_SAMPLES, || render_fractal(&input, &params));
    assert!(rtf < 1.0, "7-scale RTF {rtf:.4} exceeds 1.0");
}

#[test]
fn perf_iterations() {
    let input = make_sine(N_SAMPLES);
    let mut params = FractalParams::default();
    params.iterations = 4;
    params.saturation = 0.5;
    let rtf = measure_rtf(N_SAMPLES, || render_fractal(&input, &params));
    assert!(rtf < 2.0, "4-iteration RTF {rtf:.4} exceeds 2.0");
}

#[test]
fn perf_spectral() {
    let input = make_sine(N_SAMPLES);
    let mut params = FractalParams::default();
    params.spectral = 1.0;
    let rtf = measure_rtf(N_SAMPLES, || render_fractal(&input, &params));
    assert!(rtf < 1.0, "Spectral RTF {rtf:.4} exceeds 1.0");
}

#[test]
fn perf_heavy() {
    let input = make_sine(N_SAMPLES);
    let mut params = FractalParams::default();
    params.num_scales = 7;
    params.iterations = 3;
    params.saturation = 0.5;
    params.spectral = 0.3;
    let rtf = measure_rtf(N_SAMPLES, || render_fractal(&input, &params));
    assert!(rtf < 3.0, "Heavy RTF {rtf:.4} exceeds 3.0");
}

#[test]
fn perf_stereo() {
    let left = make_sine(N_SAMPLES);
    let right = make_sine(N_SAMPLES);
    let params = FractalParams::default();
    let rtf = measure_rtf_stereo(N_SAMPLES, || {
        render_fractal_stereo(&left, &right, &params)
    });
    assert!(rtf < 2.0, "Stereo RTF {rtf:.4} exceeds 2.0");
}

#[test]
fn perf_stereo_spread() {
    let left = make_sine(N_SAMPLES);
    let right = make_sine(N_SAMPLES);
    let mut params = FractalParams::default();
    params.layer_spread = 0.5;
    let rtf = measure_rtf_stereo(N_SAMPLES, || {
        render_fractal_stereo(&left, &right, &params)
    });
    assert!(rtf < 2.0, "Stereo spread RTF {rtf:.4} exceeds 2.0");
}

/// Verify that output is always finite across various configurations.
#[test]
fn output_safety_all_configs() {
    let input = make_sine(44100);

    let configs: Vec<(&str, FractalParams)> = vec![
        ("default", FractalParams::default()),
        ("max_scales", {
            let mut p = FractalParams::default();
            p.num_scales = 8;
            p
        }),
        ("max_saturation", {
            let mut p = FractalParams::default();
            p.saturation = 1.0;
            p
        }),
        ("spectral_only", {
            let mut p = FractalParams::default();
            p.spectral = 1.0;
            p
        }),
        ("full_effects", {
            let mut p = FractalParams::default();
            p.num_scales = 5;
            p.iterations = 2;
            p.crush = 0.5;
            p.filter_type = 1;
            p.filter_freq = 2000.0;
            p.gate = 0.2;
            p
        }),
    ];

    for (name, params) in &configs {
        let output = render_fractal(&input, params);
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
