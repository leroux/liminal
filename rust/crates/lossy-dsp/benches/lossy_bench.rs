use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use lossy_dsp::chain::{render_lossy, render_lossy_stereo};
use lossy_dsp::params::LossyParams;
use lossy_dsp::processor::{LossyProcessor, StereoLossyProcessor};

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

// --- Pre-allocated processor vs allocating API ---

fn bench_processor_vs_allocating(c: &mut Criterion) {
    let mut group = c.benchmark_group("lossy_alloc_vs_prealloc");
    let input = make_sine(44100);
    let params = default_params();

    group.bench_function("allocating_1s", |b| {
        b.iter(|| render_lossy(black_box(&input), black_box(&params)))
    });

    group.bench_function("prealloc_256x172", |b| {
        let mut proc = LossyProcessor::new();
        let chunk = make_sine(256);
        let mut output = vec![0.0; 256];
        b.iter(|| {
            for _ in 0..172 {
                proc.process(black_box(&chunk), black_box(&params), black_box(&mut output));
            }
        })
    });

    group.finish();
}

// --- Plugin-realistic: small buffers, repeated calls ---

fn bench_plugin_realistic(c: &mut Criterion) {
    let mut group = c.benchmark_group("lossy_plugin_realistic");
    let params = default_params();

    // Simulate a DAW calling process() with 256-sample buffers
    // Processing 1 second = ~172 calls
    group.bench_function("prealloc_256x172", |b| {
        let mut proc = LossyProcessor::new();
        let input = make_sine(256);
        let mut output = vec![0.0; 256];
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
                render_lossy(black_box(&input), black_box(&params));
            }
        })
    });

    group.finish();
}

// --- Stereo processor ---

fn bench_stereo_processor(c: &mut Criterion) {
    let mut group = c.benchmark_group("lossy_stereo_prealloc");
    let left = make_sine(44100);
    let right = make_sine(44100);

    group.bench_function("allocating_1s", |b| {
        let p = default_params();
        b.iter(|| render_lossy_stereo(black_box(&left), black_box(&right), black_box(&p)))
    });

    group.bench_function("prealloc_256x172_stereo", |b| {
        let p = default_params();
        let mut proc = StereoLossyProcessor::new();
        let chunk_l = make_sine(256);
        let chunk_r = make_sine(256);
        let mut out_l = vec![0.0; 256];
        let mut out_r = vec![0.0; 256];
        b.iter(|| {
            for _ in 0..172 {
                proc.process_stereo(
                    black_box(&chunk_l),
                    black_box(&chunk_r),
                    black_box(&p),
                    black_box(&mut out_l),
                    black_box(&mut out_r),
                );
            }
        })
    });

    group.finish();
}

// --- Conditional code-path benchmarks (streaming processor) ---

fn freeze_params() -> LossyParams {
    let mut p = default_params();
    p.freeze = 1;
    p.freeze_mode = 0; // slushy
    p.freezer = 1.0;
    p.slushy_rate = 0.03;
    p
}

fn freeze_solid_params() -> LossyParams {
    let mut p = default_params();
    p.freeze = 1;
    p.freeze_mode = 1; // solid
    p.freezer = 1.0;
    p
}

fn post_verb_params() -> LossyParams {
    let mut p = default_params();
    p.verb = 0.5;
    p.decay = 0.6;
    p.verb_position = 1; // post
    p
}

fn notch_steep_params() -> LossyParams {
    let mut p = default_params();
    p.filter_type = 2; // notch
    p.filter_freq = 3000.0;
    p.filter_width = 0.3;
    p.filter_slope = 96;
    p
}

fn packet_repeat_params() -> LossyParams {
    let mut p = default_params();
    p.packets = 2; // repeat mode
    p.packet_rate = 0.3;
    p.packet_size = 50.0;
    p
}

fn auto_gain_params() -> LossyParams {
    let mut p = default_params();
    p.auto_gain = 0.8;
    p
}

fn bench_conditional_paths(c: &mut Criterion) {
    let mut group = c.benchmark_group("lossy_conditionals");
    let input = make_sine(256);
    let mut output = vec![0.0; 256];

    let configs: &[(&str, LossyParams)] = &[
        ("freeze_slushy", freeze_params()),
        ("freeze_solid", freeze_solid_params()),
        ("post_verb", post_verb_params()),
        ("notch_96db", notch_steep_params()),
        ("packet_repeat", packet_repeat_params()),
        ("auto_gain", auto_gain_params()),
        ("spectral_heavy", spectral_heavy_params()),
        ("full_chain", full_chain_params()),
        ("bypass", bypass_params()),
    ];

    for (name, params) in configs {
        group.bench_function(*name, |b| {
            let mut proc = LossyProcessor::new();
            b.iter(|| {
                for _ in 0..172 {
                    proc.process(black_box(&input), black_box(params), black_box(&mut output));
                }
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_buffer_sizes,
    bench_window_sizes,
    bench_engine_configs,
    bench_stereo,
    bench_processor_vs_allocating,
    bench_plugin_realistic,
    bench_stereo_processor,
    bench_conditional_paths,
);
criterion_main!(benches);
