use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use fractal_dsp::chain::{render_fractal, render_fractal_stereo};
use fractal_dsp::params::FractalParams;

fn make_sine(len: usize) -> Vec<f64> {
    (0..len)
        .map(|i| (2.0 * std::f64::consts::PI * 440.0 * i as f64 / 44100.0).sin())
        .collect()
}

fn default_params() -> FractalParams {
    FractalParams::default()
}

fn heavy_params() -> FractalParams {
    let mut p = FractalParams::default();
    p.num_scales = 7;
    p.iterations = 3;
    p.saturation = 0.5;
    p.spectral = 0.3;
    p
}

// --- Buffer size benchmarks ---

fn bench_buffer_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("fractal_buffer_size");
    let params = default_params();

    for &size in &[2048, 8192, 44100] {
        let input = make_sine(size);
        group.bench_with_input(BenchmarkId::new("default", size), &size, |b, _| {
            b.iter(|| render_fractal(black_box(&input), black_box(&params)))
        });
    }
    group.finish();
}

// --- Engine configurations ---

fn bench_engine_configs(c: &mut Criterion) {
    let mut group = c.benchmark_group("fractal_engine");
    let input = make_sine(44100);

    group.bench_function("default", |b| {
        let p = default_params();
        b.iter(|| render_fractal(black_box(&input), black_box(&p)))
    });

    group.bench_function("7_scales", |b| {
        let mut p = default_params();
        p.num_scales = 7;
        b.iter(|| render_fractal(black_box(&input), black_box(&p)))
    });

    group.bench_function("3_iterations", |b| {
        let mut p = default_params();
        p.iterations = 3;
        b.iter(|| render_fractal(black_box(&input), black_box(&p)))
    });

    group.bench_function("spectral", |b| {
        let mut p = default_params();
        p.spectral = 1.0;
        b.iter(|| render_fractal(black_box(&input), black_box(&p)))
    });

    group.bench_function("spectral_blend", |b| {
        let mut p = default_params();
        p.spectral = 0.5;
        b.iter(|| render_fractal(black_box(&input), black_box(&p)))
    });

    group.bench_function("heavy", |b| {
        let p = heavy_params();
        b.iter(|| render_fractal(black_box(&input), black_box(&p)))
    });

    group.bench_function("with_filters", |b| {
        let mut p = default_params();
        p.filter_type = 1;
        p.filter_freq = 2000.0;
        p.post_filter_type = 1;
        p.crush = 0.3;
        p.gate = 0.1;
        b.iter(|| render_fractal(black_box(&input), black_box(&p)))
    });

    group.finish();
}

// --- Stereo ---

fn bench_stereo(c: &mut Criterion) {
    let mut group = c.benchmark_group("fractal_stereo");
    let left = make_sine(44100);
    let right = make_sine(44100);

    group.bench_function("default_1s", |b| {
        let p = default_params();
        b.iter(|| render_fractal_stereo(black_box(&left), black_box(&right), black_box(&p)))
    });

    group.bench_function("with_spread", |b| {
        let mut p = default_params();
        p.layer_spread = 0.5;
        b.iter(|| render_fractal_stereo(black_box(&left), black_box(&right), black_box(&p)))
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_buffer_sizes,
    bench_engine_configs,
    bench_stereo,
);
criterion_main!(benches);
