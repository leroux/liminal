//! Tolerance tests: verify optimized code produces output matching golden values.
//! Golden values captured from the pre-optimization implementation.

use fractal_dsp::chain::{render_fractal, render_fractal_stereo};
use fractal_dsp::params::{FractalParams, SR};

fn make_sine(len: usize) -> Vec<f64> {
    (0..len)
        .map(|i| (2.0 * std::f64::consts::PI * 440.0 * i as f64 / SR).sin())
        .collect()
}

struct Golden {
    rms: f64,
    peak: f64,
    head: f64,
    tail: f64,
}

fn check(signal: &[f64], g: &Golden, tol: f64, label: &str) {
    let rms = (signal.iter().map(|x| x * x).sum::<f64>() / signal.len() as f64).sqrt();
    let peak = signal.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
    let head: f64 = signal[..100].iter().sum();
    let tail: f64 = signal[signal.len() - 100..].iter().sum();

    assert!(
        (rms - g.rms).abs() < tol,
        "{label}: RMS {rms:.15e} vs golden {:.15e}, diff={:.2e}",
        g.rms,
        (rms - g.rms).abs()
    );
    assert!(
        (peak - g.peak).abs() < tol,
        "{label}: peak {peak:.15e} vs golden {:.15e}, diff={:.2e}",
        g.peak,
        (peak - g.peak).abs()
    );
    assert!(
        (head - g.head).abs() < tol * 100.0,
        "{label}: head {head:.15e} vs golden {:.15e}, diff={:.2e}",
        g.head,
        (head - g.head).abs()
    );
    assert!(
        (tail - g.tail).abs() < tol * 100.0,
        "{label}: tail {tail:.15e} vs golden {:.15e}, diff={:.2e}",
        g.tail,
        (tail - g.tail).abs()
    );
}

const TOL: f64 = 1e-6;

#[test]
fn golden_default() {
    let input = make_sine(44100);
    let out = render_fractal(&input, &FractalParams::default());
    check(&out, &Golden { rms: 3.059965315158805e-1, peak: 5.250000000000000e-1, head: 1.261318957669161e-2, tail: 2.792420070029268e-2 }, TOL, "default");
}

#[test]
fn golden_7_scales() {
    let input = make_sine(44100);
    let mut p = FractalParams::default();
    p.num_scales = 7;
    let out = render_fractal(&input, &p);
    check(&out, &Golden { rms: 2.405178122737932e-1, peak: 5.250000000000000e-1, head: 3.751561640282164e-2, tail: -3.380982997551660e-2 }, TOL, "7_scales");
}

#[test]
fn golden_spectral_full() {
    let input = make_sine(44100);
    let mut p = FractalParams::default();
    p.spectral = 1.0;
    let out = render_fractal(&input, &p);
    check(&out, &Golden { rms: 2.218430303677447e-1, peak: 5.250000000000000e-1, head: 8.197480617543869e-3, tail: -3.446694091080942e-4 }, TOL, "spectral_full");
}

#[test]
fn golden_spectral_blend() {
    let input = make_sine(44100);
    let mut p = FractalParams::default();
    p.spectral = 0.5;
    let out = render_fractal(&input, &p);
    check(&out, &Golden { rms: 2.699210570100337e-1, peak: 5.250000000000000e-1, head: 1.159476629799338e-2, tail: 1.536607023919985e-2 }, TOL, "spectral_blend");
}

#[test]
fn golden_3_iterations() {
    let input = make_sine(44100);
    let mut p = FractalParams::default();
    p.iterations = 3;
    p.saturation = 0.3;
    p.iter_decay = 0.8;
    let out = render_fractal(&input, &p);
    check(&out, &Golden { rms: 3.108087679026595e-1, peak: 5.250000000000000e-1, head: 6.211569017112917e-2, tail: 2.650848943316547e-1 }, TOL, "3_iterations");
}

#[test]
fn golden_heavy() {
    let input = make_sine(44100);
    let mut p = FractalParams::default();
    p.num_scales = 7;
    p.iterations = 3;
    p.saturation = 0.5;
    p.spectral = 0.3;
    let out = render_fractal(&input, &p);
    check(&out, &Golden { rms: 1.427574888292305e-1, peak: 4.229283302699187e-1, head: -2.399976187198230e-2, tail: 1.280098709879346e-1 }, TOL, "heavy");
}

#[test]
fn golden_with_filters() {
    let input = make_sine(44100);
    let mut p = FractalParams::default();
    p.filter_type = 1;
    p.filter_freq = 2000.0;
    p.post_filter_type = 1;
    p.crush = 0.3;
    p.gate = 0.1;
    let out = render_fractal(&input, &p);
    check(&out, &Golden { rms: 2.701424311980486e-1, peak: 5.250000000000000e-1, head: 8.223893637026948e-1, tail: 6.995515974995009e-2 }, TOL, "with_filters");
}

#[test]
fn golden_linear_interp() {
    let input = make_sine(44100);
    let mut p = FractalParams::default();
    p.interp = 1;
    let out = render_fractal(&input, &p);
    check(&out, &Golden { rms: 3.036516566268557e-1, peak: 5.250000000000000e-1, head: 1.232038729850482e-2, tail: -2.227444701486107e-3 }, TOL, "linear_interp");
}

#[test]
fn golden_reverse_scales() {
    let input = make_sine(44100);
    let mut p = FractalParams::default();
    p.reverse_scales = 1;
    p.scale_offset = 0.3;
    let out = render_fractal(&input, &p);
    check(&out, &Golden { rms: 3.082077370338357e-1, peak: 5.250000000000000e-1, head: -3.055452656455342e-4, tail: 2.255984875027584e-3 }, TOL, "reverse_scales");
}

#[test]
fn golden_only_wet() {
    let input = make_sine(44100);
    let mut p = FractalParams::default();
    p.fractal_only_wet = 1;
    let out = render_fractal(&input, &p);
    check(&out, &Golden { rms: 2.995072571208333e-1, peak: 5.250000000000000e-1, head: 1.458338479692636e-2, tail: 3.906038866743899e-2 }, TOL, "only_wet");
}

#[test]
fn golden_layer_features() {
    let input = make_sine(44100);
    let mut p = FractalParams::default();
    p.num_scales = 5;
    p.layer_detune = 0.5;
    p.layer_delay = 0.3;
    p.layer_tilt = 0.5;
    p.layer_gain_1 = 0.5;
    p.layer_gain_2 = 1.5;
    let out = render_fractal(&input, &p);
    check(&out, &Golden { rms: 1.976911845920581e-1, peak: 5.250000000000000e-1, head: 1.829549265002391e-3, tail: -1.898411615473751e-1 }, TOL, "layer_features");
}

#[test]
fn golden_dry_mix() {
    let input = make_sine(44100);
    let mut p = FractalParams::default();
    p.wet_dry = 0.0;
    let out = render_fractal(&input, &p);
    check(&out, &Golden { rms: 7.071067811865486e-1, peak: 9.999997462578870e-1, head: 8.742038041825814e-3, tail: 5.505065665694198e-3 }, TOL, "dry_mix");
}

#[test]
fn golden_stereo_default() {
    let left = make_sine(44100);
    let right: Vec<f64> = (0..44100)
        .map(|i| (2.0 * std::f64::consts::PI * 660.0 * i as f64 / SR).sin())
        .collect();
    let (out_l, out_r) = render_fractal_stereo(&left, &right, &FractalParams::default());
    check(&out_l, &Golden { rms: 3.059965315158805e-1, peak: 5.250000000000000e-1, head: 1.261318957669161e-2, tail: 2.792420070029268e-2 }, TOL, "stereo_default_L");
    check(&out_r, &Golden { rms: 3.021400687253670e-1, peak: 5.250000000000000e-1, head: 6.875615640101582e0, tail: -6.829613111494343e0 }, TOL, "stereo_default_R");
}

#[test]
fn golden_stereo_spread() {
    let left = make_sine(44100);
    let right: Vec<f64> = (0..44100)
        .map(|i| (2.0 * std::f64::consts::PI * 660.0 * i as f64 / SR).sin())
        .collect();
    let mut p = FractalParams::default();
    p.layer_spread = 0.5;
    let (out_l, out_r) = render_fractal_stereo(&left, &right, &p);
    check(&out_l, &Golden { rms: 2.725776647329530e-1, peak: 5.250000000000000e-1, head: 1.483069435838480e-2, tail: 4.054544944430984e-2 }, TOL, "stereo_spread_L");
    check(&out_r, &Golden { rms: 2.382427668259373e-1, peak: 5.250000000000000e-1, head: 5.447293232585010e0, tail: -5.403241838225580e0 }, TOL, "stereo_spread_R");
}

#[test]
fn golden_stereo_spectral() {
    let left = make_sine(44100);
    let right: Vec<f64> = (0..44100)
        .map(|i| (2.0 * std::f64::consts::PI * 660.0 * i as f64 / SR).sin())
        .collect();
    let mut p = FractalParams::default();
    p.spectral = 0.5;
    p.layer_spread = 0.5;
    let (out_l, out_r) = render_fractal_stereo(&left, &right, &p);
    check(&out_l, &Golden { rms: 2.573963336914262e-1, peak: 5.250000000000000e-1, head: 1.309141241466193e-2, tail: 2.285396004599457e-2 }, TOL, "stereo_spectral_L");
    check(&out_r, &Golden { rms: 2.484067772350437e-1, peak: 5.250000000000000e-1, head: 3.075891071169772e0, tail: -3.033954159633021e0 }, TOL, "stereo_spectral_R");
}

/// Verify output is always finite and peak-bounded across all configs.
#[test]
fn output_safety_comprehensive() {
    let input = make_sine(44100);
    let configs: Vec<(&str, FractalParams)> = vec![
        ("default", FractalParams::default()),
        ("max_scales", { let mut p = FractalParams::default(); p.num_scales = 8; p }),
        ("max_iterations", { let mut p = FractalParams::default(); p.iterations = 4; p.saturation = 1.0; p }),
        ("full_spectral", { let mut p = FractalParams::default(); p.spectral = 1.0; p }),
        ("heavy", { let mut p = FractalParams::default(); p.num_scales = 7; p.iterations = 3; p.spectral = 0.5; p.saturation = 0.5; p }),
    ];
    for (name, params) in &configs {
        let out = render_fractal(&input, params);
        assert_eq!(out.len(), input.len(), "{name}: length mismatch");
        assert!(out.iter().all(|x| x.is_finite()), "{name}: non-finite output");
        let peak = out.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
        assert!(peak < 1e6, "{name}: peak {peak} too large");
    }
}

/// Dry-only mix must return exact input.
#[test]
fn dry_only_is_exact_passthrough() {
    let input = make_sine(4096);
    let mut p = FractalParams::default();
    p.wet_dry = 0.0;
    let out = render_fractal(&input, &p);
    for (i, (&a, &b)) in input.iter().zip(out.iter()).enumerate() {
        assert!((a - b).abs() < 1e-15, "Sample {i}: {a} vs {b}");
    }
}

/// Determinism: same input + params always produces identical output.
#[test]
fn deterministic_output() {
    let input = make_sine(8192);
    let mut p = FractalParams::default();
    p.spectral = 0.5;
    p.iterations = 2;
    let out1 = render_fractal(&input, &p);
    let out2 = render_fractal(&input, &p);
    for (i, (&a, &b)) in out1.iter().zip(out2.iter()).enumerate() {
        assert_eq!(a, b, "Non-deterministic at sample {i}");
    }
}
