//! Tolerance tests for lossy-dsp optimizations.

use lossy_dsp::chain::render_lossy;
use lossy_dsp::params::{LossyParams, SR};

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
    let input = make_sine(44100);
    let out = render_lossy(&input, &LossyParams::default());
    check(&out, &Golden { rms: 3.254304338173923e-1, peak: 5.250000000000000e-1, head: -3.610193137373197e0, tail: -4.296662808099290e-1 }, TOL, "default");
}

#[test]
fn golden_high_loss() {
    let input = make_sine(44100);
    let mut p = LossyParams::default();
    p.loss = 0.8;
    let out = render_lossy(&input, &p);
    check(&out, &Golden { rms: 3.368499107993096e-1, peak: 5.250000000000000e-1, head: -2.010832550966926e0, tail: 2.201041445550764e0 }, TOL, "high_loss");
}

#[test]
fn golden_with_reverb() {
    let input = make_sine(44100);
    let mut p = LossyParams::default();
    p.verb = 0.5;
    let out = render_lossy(&input, &p);
    check(&out, &Golden { rms: 3.248314268557942e-1, peak: 5.250000000000000e-1, head: -3.191431072613696e0, tail: -4.885158560335749e-1 }, TOL, "with_reverb");
}

#[test]
fn golden_with_filter() {
    let input = make_sine(44100);
    let mut p = LossyParams::default();
    p.filter_type = 1;
    p.filter_freq = 2000.0;
    let out = render_lossy(&input, &p);
    check(&out, &Golden { rms: 2.740665369671132e-3, peak: 2.437264865571566e-2, head: 4.638822999755415e-2, tail: 3.975703579821091e-4 }, TOL, "with_filter");
}

#[test]
fn golden_crush_decimate() {
    let input = make_sine(44100);
    let mut p = LossyParams::default();
    p.crush = 0.5;
    p.decimate = 0.3;
    let out = render_lossy(&input, &p);
    check(&out, &Golden { rms: 3.380651883273232e-1, peak: 5.250000000000000e-1, head: -2.995923913043483e0, tail: -6.552989130434744e-1 }, TOL, "crush_decimate");
}

#[test]
fn golden_full_effects() {
    let input = make_sine(44100);
    let mut p = LossyParams::default();
    p.loss = 0.5;
    p.verb = 0.3;
    p.filter_type = 1;
    p.filter_freq = 3000.0;
    p.crush = 0.2;
    p.gate = 0.1;
    let out = render_lossy(&input, &p);
    check(&out, &Golden { rms: 1.106076186745633e-5, peak: 3.423866858553111e-4, head: 7.156318899274813e-4, tail: 6.849419381035979e-4 }, TOL, "full_effects");
}

#[test]
fn golden_packets() {
    let input = make_sine(44100);
    let mut p = LossyParams::default();
    p.packets = 1;
    p.packet_rate = 0.3;
    p.packet_size = 0.5;
    let out = render_lossy(&input, &p);
    check(&out, &Golden { rms: 2.919434445228257e-1, peak: 5.250000000000000e-1, head: -3.610193137373197e0, tail: 4.170013158739235e-1 }, TOL, "packets");
}

#[test]
fn golden_freeze() {
    let input = make_sine(44100);
    let mut p = LossyParams::default();
    p.loss = 0.5;
    p.freeze = 1;
    let out = render_lossy(&input, &p);
    check(&out, &Golden { rms: 3.307019003607372e-1, peak: 5.250000000000000e-1, head: -2.538733834998983e0, tail: 2.402755105908047e0 }, TOL, "freeze");
}

#[test]
fn output_safety() {
    let input = make_sine(44100);
    let params = LossyParams::default();
    let out = render_lossy(&input, &params);
    assert_eq!(out.len(), input.len());
    assert!(out.iter().all(|x| x.is_finite()));
}

#[test]
fn deterministic() {
    let input = make_sine(44100);
    let params = LossyParams::default();
    let out1 = render_lossy(&input, &params);
    let out2 = render_lossy(&input, &params);
    for (i, (&a, &b)) in out1.iter().zip(out2.iter()).enumerate() {
        assert_eq!(a, b, "Non-deterministic at {i}");
    }
}
