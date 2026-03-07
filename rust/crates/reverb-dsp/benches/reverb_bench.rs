use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use reverb_dsp::chain::{render_fdn, render_fdn_stereo};
use reverb_dsp::params::{ReverbParams, N};
use reverb_dsp::processor::{FdnProcessor, StereoFdnProcessor};

fn make_sine(len: usize) -> Vec<f64> {
    (0..len)
        .map(|i| (2.0 * std::f64::consts::PI * 440.0 * i as f64 / 44100.0).sin())
        .collect()
}

fn default_params() -> ReverbParams {
    let mut p = ReverbParams::default();
    p.normalize();
    p
}

fn modulated_params() -> ReverbParams {
    let mut p = ReverbParams::default();
    p.mod_master_rate = 2.0;
    p.mod_depth_delay = vec![5.0; N];
    p.mod_depth_damping = vec![0.1; N];
    p.mod_depth_output = vec![0.05; N];
    p.mod_depth_matrix = 0.3;
    p.normalize();
    p
}

fn random_ortho_params() -> ReverbParams {
    let mut p = ReverbParams::default();
    p.matrix_type = "random_orthogonal".to_string();
    p.normalize();
    p
}

fn heavy_params() -> ReverbParams {
    let mut p = ReverbParams::default();
    p.matrix_type = "random_orthogonal".to_string();
    p.saturation = 0.8;
    p.diffusion = 0.7;
    p.diffusion_stages = 4;
    p.mod_master_rate = 3.0;
    p.mod_depth_delay = vec![8.0; N];
    p.mod_depth_damping = vec![0.2; N];
    p.mod_depth_output = vec![0.1; N];
    p.mod_depth_matrix = 0.5;
    p.normalize();
    p
}

// --- Buffer size benchmarks ---

fn bench_buffer_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("reverb_buffer_size");
    let params = default_params();

    for &size in &[128, 512, 2048, 8192, 44100] {
        let input = make_sine(size);
        group.bench_with_input(BenchmarkId::new("static", size), &size, |b, _| {
            b.iter(|| render_fdn(black_box(&input), black_box(&params)))
        });
    }
    group.finish();
}

// --- Static vs modulated ---

fn bench_static_vs_mod(c: &mut Criterion) {
    let mut group = c.benchmark_group("reverb_engine");
    let input = make_sine(44100); // 1 second

    group.bench_function("static_householder", |b| {
        let p = default_params();
        b.iter(|| render_fdn(black_box(&input), black_box(&p)))
    });

    group.bench_function("static_random_ortho", |b| {
        let p = random_ortho_params();
        b.iter(|| render_fdn(black_box(&input), black_box(&p)))
    });

    group.bench_function("modulated", |b| {
        let p = modulated_params();
        b.iter(|| render_fdn(black_box(&input), black_box(&p)))
    });

    group.bench_function("heavy_all_mod", |b| {
        let p = heavy_params();
        b.iter(|| render_fdn(black_box(&input), black_box(&p)))
    });

    group.finish();
}

// --- Stereo ---

fn bench_stereo(c: &mut Criterion) {
    let mut group = c.benchmark_group("reverb_stereo");
    let left = make_sine(44100);
    let right = make_sine(44100);

    group.bench_function("static_1s", |b| {
        let p = default_params();
        b.iter(|| render_fdn_stereo(black_box(&left), black_box(&right), black_box(&p)))
    });

    group.bench_function("modulated_1s", |b| {
        let p = modulated_params();
        b.iter(|| render_fdn_stereo(black_box(&left), black_box(&right), black_box(&p)))
    });

    group.finish();
}

// --- Matrix types ---

fn bench_matrix_types(c: &mut Criterion) {
    let mut group = c.benchmark_group("reverb_matrix");
    let input = make_sine(44100);

    for matrix_type in &["householder", "hadamard", "diagonal", "random_orthogonal", "circulant", "stautner_puckette"] {
        let mut p = default_params();
        p.matrix_type = matrix_type.to_string();
        group.bench_function(*matrix_type, |b| {
            b.iter(|| render_fdn(black_box(&input), black_box(&p)))
        });
    }
    group.finish();
}

// --- Pre-allocated processor vs allocating API ---

fn bench_processor_vs_allocating(c: &mut Criterion) {
    let mut group = c.benchmark_group("reverb_alloc_vs_prealloc");
    let input = make_sine(44100);
    let params = default_params();

    group.bench_function("allocating_1s", |b| {
        b.iter(|| render_fdn(black_box(&input), black_box(&params)))
    });

    group.bench_function("prealloc_1s", |b| {
        let mut proc = FdnProcessor::new();
        let mut output = vec![0.0; 44100 * 2];
        b.iter(|| proc.process(black_box(&input), black_box(&params), black_box(&mut output)))
    });

    let mod_params = modulated_params();

    group.bench_function("allocating_mod_1s", |b| {
        b.iter(|| render_fdn(black_box(&input), black_box(&mod_params)))
    });

    group.bench_function("prealloc_mod_1s", |b| {
        let mut proc = FdnProcessor::new();
        let mut output = vec![0.0; 44100 * 2];
        b.iter(|| {
            proc.process(
                black_box(&input),
                black_box(&mod_params),
                black_box(&mut output),
            )
        })
    });

    group.finish();
}

// --- Plugin-realistic: small buffers, repeated calls ---

fn bench_plugin_realistic(c: &mut Criterion) {
    let mut group = c.benchmark_group("reverb_plugin_realistic");
    let params = default_params();

    // Simulate a DAW calling process() with 256-sample buffers (5.8ms at 44100)
    // Processing 1 second = ~172 calls
    group.bench_function("prealloc_256x172", |b| {
        let mut proc = FdnProcessor::new();
        let input = make_sine(256);
        let mut output = vec![0.0; 256 * 2];
        b.iter(|| {
            for _ in 0..172 {
                proc.process(black_box(&input), black_box(&params), black_box(&mut output));
            }
        })
    });

    group.bench_function("allocating_256x172", |b| {
        let input = make_sine(256);
        b.iter(|| {
            for _ in 0..172 {
                render_fdn(black_box(&input), black_box(&params));
            }
        })
    });

    group.finish();
}

// --- Stereo processor ---

fn bench_stereo_processor(c: &mut Criterion) {
    let mut group = c.benchmark_group("reverb_stereo_prealloc");
    let left = make_sine(44100);
    let right = make_sine(44100);

    group.bench_function("allocating_1s", |b| {
        let p = default_params();
        b.iter(|| render_fdn_stereo(black_box(&left), black_box(&right), black_box(&p)))
    });

    group.bench_function("prealloc_1s", |b| {
        let p = default_params();
        let mut proc = StereoFdnProcessor::new();
        let mut out_l = vec![0.0; 44100];
        let mut out_r = vec![0.0; 44100];
        b.iter(|| {
            proc.process_stereo(
                black_box(&left),
                black_box(&right),
                black_box(&p),
                black_box(&mut out_l),
                black_box(&mut out_r),
            )
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_buffer_sizes,
    bench_static_vs_mod,
    bench_stereo,
    bench_matrix_types,
    bench_processor_vs_allocating,
    bench_plugin_realistic,
    bench_stereo_processor,
);
criterion_main!(benches);
