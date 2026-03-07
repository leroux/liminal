use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use lossy_dsp::chain::{render_lossy, render_lossy_stereo};
use lossy_dsp::params::LossyParams;

fn make_sine(len: usize) -> Vec<f64> {
    (0..len)
        .map(|i| (2.0 * std::f64::consts::PI * 440.0 * i as f64 / 44100.0).sin())
        .collect()
}

fn default_params() -> LossyParams {
    LossyParams::default()
}

fn spectral_heavy_params() -> LossyParams {
    let mut p = LossyParams::default();
    p.loss = 0.9;
    p.jitter = 0.5;
    p.phase_loss = 0.5;
    p.noise_shape = 0.5;
    p.weighting = 1.0;
    p.pre_echo = 0.3;
    p
}

fn full_chain_params() -> LossyParams {
    let mut p = LossyParams::default();
    p.loss = 0.6;
    p.crush = 0.3;
    p.decimate = 0.2;
    p.verb = 0.4;
    p.decay = 0.5;
    p.filter_type = 1; // bandpass
    p.filter_freq = 2000.0;
    p.filter_width = 0.5;
    p.packets = 1; // packet loss
    p.packet_rate = 0.2;
    p.gate = 0.1;
    p
}

fn bypass_params() -> LossyParams {
    let mut p = LossyParams::default();
    p.loss = 0.0;
    p.crush = 0.0;
    p.decimate = 0.0;
    p.verb = 0.0;
    p.filter_type = 0;
    p.packets = 0;
    p.gate = 0.0;
    p
}

// --- Buffer size benchmarks ---

fn bench_buffer_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("lossy_buffer_size");
    let params = default_params();

    for &size in &[2048, 8192, 44100, 88200] {
        let input = make_sine(size);
        group.bench_with_input(BenchmarkId::new("default", size), &size, |b, _| {
            b.iter(|| render_lossy(black_box(&input), black_box(&params)))
        });
    }
    group.finish();
}

// --- Window size benchmarks ---

fn bench_window_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("lossy_window_size");
    let input = make_sine(44100);

    for &ws in &[512, 1024, 2048, 4096, 8192] {
        let mut p = default_params();
        p.window_size = ws;
        group.bench_with_input(BenchmarkId::new("spectral", ws), &ws, |b, _| {
            b.iter(|| render_lossy(black_box(&input), black_box(&p)))
        });
    }
    group.finish();
}

// --- Engine configurations ---

fn bench_engine_configs(c: &mut Criterion) {
    let mut group = c.benchmark_group("lossy_engine");
    let input = make_sine(44100);

    group.bench_function("bypass", |b| {
        let p = bypass_params();
        b.iter(|| render_lossy(black_box(&input), black_box(&p)))
    });

    group.bench_function("default", |b| {
        let p = default_params();
        b.iter(|| render_lossy(black_box(&input), black_box(&p)))
    });

    group.bench_function("spectral_heavy", |b| {
        let p = spectral_heavy_params();
        b.iter(|| render_lossy(black_box(&input), black_box(&p)))
    });

    group.bench_function("full_chain", |b| {
        let p = full_chain_params();
        b.iter(|| render_lossy(black_box(&input), black_box(&p)))
    });

    group.bench_function("bounce", |b| {
        let mut p = default_params();
        p.bounce = 1;
        p.bounce_target = 0;
        p.bounce_rate = 0.5;
        b.iter(|| render_lossy(black_box(&input), black_box(&p)))
    });

    group.finish();
}

// --- Stereo ---

fn bench_stereo(c: &mut Criterion) {
    let mut group = c.benchmark_group("lossy_stereo");
    let left = make_sine(44100);
    let right = make_sine(44100);

    group.bench_function("default_1s", |b| {
        let p = default_params();
        b.iter(|| render_lossy_stereo(black_box(&left), black_box(&right), black_box(&p)))
    });

    group.bench_function("full_chain_1s", |b| {
        let p = full_chain_params();
        b.iter(|| render_lossy_stereo(black_box(&left), black_box(&right), black_box(&p)))
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_buffer_sizes,
    bench_window_sizes,
    bench_engine_configs,
    bench_stereo,
);
criterion_main!(benches);
