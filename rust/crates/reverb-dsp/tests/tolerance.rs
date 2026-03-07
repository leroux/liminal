//! Tolerance tests for reverb-dsp optimizations.

use reverb_dsp::chain::render_fdn;
use reverb_dsp::params::{ReverbParams, SR};

fn make_sine(len: usize) -> Vec<f64> {
    (0..len)
        .map(|i| (2.0 * std::f64::consts::PI * 440.0 * i as f64 / SR).sin())
        .collect()
}

struct Golden { rms: f64, peak: f64, head: f64, tail: f64 }

fn check(signal: &[f64], g: &Golden, tol: f64, label: &str) {
    let rms = (signal.iter().map(|x| x * x).sum::<f64>() / signal.len() as f64).sqrt();
    let peak = signal.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
    let head: f64 = signal[..100].iter().sum();
    let tail: f64 = signal[signal.len() - 100..].iter().sum();
    assert!((rms - g.rms).abs() < tol, "{label}: RMS {rms:.15e} vs {:.15e}", g.rms);
    assert!((peak - g.peak).abs() < tol, "{label}: peak {peak:.15e} vs {:.15e}", g.peak);
    assert!((head - g.head).abs() < tol * 100.0, "{label}: head {head:.15e} vs {:.15e}", g.head);
    assert!((tail - g.tail).abs() < tol * 100.0, "{label}: tail {tail:.15e} vs {:.15e}", g.tail);
}

const TOL: f64 = 1e-6;

#[test]
fn golden_default() {
    let input = make_sine(4410);
    let out = render_fdn(&input, &ReverbParams::default());
    check(&out, &Golden { rms: 3.604326840177436e-1, peak: 6.307209438728478e-1, head: 3.188891636739190e1, tail: -3.809016336292755e1 }, TOL, "default");
}

#[test]
fn golden_high_feedback() {
    let input = make_sine(4410);
    let mut p = ReverbParams::default();
    p.feedback_gain = 0.95;
    let out = render_fdn(&input, &p);
    check(&out, &Golden { rms: 3.606259155428777e-1, peak: 6.362778525907040e-1, head: 3.188891636739190e1, tail: -3.829076910300675e1 }, TOL, "high_feedback");
}

#[test]
fn golden_low_feedback() {
    let input = make_sine(4410);
    let mut p = ReverbParams::default();
    p.feedback_gain = 0.3;
    let out = render_fdn(&input, &p);
    check(&out, &Golden { rms: 3.593997093881169e-1, peak: 5.995908237994825e-1, head: 3.188891636739190e1, tail: -3.696610757498586e1 }, TOL, "low_feedback");
}

#[test]
fn golden_with_diffusion() {
    let input = make_sine(4410);
    let mut p = ReverbParams::default();
    p.diffusion = 0.8;
    let out = render_fdn(&input, &p);
    check(&out, &Golden { rms: 3.321597298624784e-1, peak: 4.999998731289434e-1, head: 3.188891636739190e1, tail: -2.793533592930980e1 }, TOL, "with_diffusion");
}

#[test]
fn golden_with_saturation() {
    let input = make_sine(4410);
    let mut p = ReverbParams::default();
    p.saturation = 0.5;
    let out = render_fdn(&input, &p);
    check(&out, &Golden { rms: 3.604246862492364e-1, peak: 6.305460842208919e-1, head: 3.188891636739190e1, tail: -3.808328540778420e1 }, TOL, "with_saturation");
}

#[test]
fn golden_high_damping() {
    let input = make_sine(4410);
    let mut p = ReverbParams::default();
    p.damping_coeffs = vec![0.8; 8];
    let out = render_fdn(&input, &p);
    check(&out, &Golden { rms: 3.601055885211507e-1, peak: 6.284377015838321e-1, head: 3.188891636739190e1, tail: -3.798767916816816e1 }, TOL, "high_damping");
}

#[test]
fn golden_modulated() {
    let input = make_sine(4410);
    let mut p = ReverbParams::default();
    p.mod_master_rate = 1.0;
    p.mod_depth_delay = vec![0.5; 8];
    let out = render_fdn(&input, &p);
    check(&out, &Golden { rms: 3.602214381959780e-1, peak: 6.305771050388289e-1, head: 3.188891636739190e1, tail: -3.806142149590873e1 }, TOL, "modulated");
}

#[test]
fn golden_full_effects() {
    let input = make_sine(4410);
    let mut p = ReverbParams::default();
    p.feedback_gain = 0.7;
    p.diffusion = 0.6;
    p.saturation = 0.3;
    p.damping_coeffs = vec![0.5; 8];
    p.mod_master_rate = 0.5;
    p.mod_depth_delay = vec![0.3; 8];
    let out = render_fdn(&input, &p);
    check(&out, &Golden { rms: 3.543092203516079e-1, peak: 5.975640284306118e-1, head: 3.188891636739190e1, tail: -3.606783351039059e1 }, TOL, "full_effects");
}

#[test]
fn output_safety() {
    let input = make_sine(4410);
    let params = ReverbParams::default();
    let out = render_fdn(&input, &params);
    assert_eq!(out.len(), input.len() * 2);
    assert!(out.iter().all(|x| x.is_finite()));
}

#[test]
fn deterministic() {
    let input = make_sine(4410);
    let params = ReverbParams::default();
    let out1 = render_fdn(&input, &params);
    let out2 = render_fdn(&input, &params);
    for (i, (&a, &b)) in out1.iter().zip(out2.iter()).enumerate() {
        assert_eq!(a, b, "Non-deterministic at {i}");
    }
}
