//! CLI runner for the explore effects framework.
//!
//! Generates synthetic test inputs, then runs all registered effects
//! with their variants, writing WAV outputs organized by category.

use clap::Parser;
use explore_dsp::{discover_effects, AudioOutput, EffectEntry};
use explore_dsp::primitives::{post_process, post_process_stereo};
use hound::{WavSpec, WavWriter, SampleFormat};
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

#[derive(Parser)]
#[command(name = "explore", about = "Audio effects explorer")]
struct Cli {
    /// Input WAV file (omit to use generated test signals)
    #[arg()]
    input_wav: Option<String>,

    /// Output directory
    #[arg(short, long, default_value = "explore_output")]
    output: String,

    /// Maximum number of variants to process
    #[arg(long, default_value_t = 1200)]
    max_variants: usize,

    /// Number of worker threads (0 = auto)
    #[arg(long, default_value_t = 0)]
    workers: usize,

    /// Comma-separated glob patterns to include (e.g. "a*,j*")
    #[arg(long)]
    include: Option<String>,

    /// Comma-separated glob patterns to exclude
    #[arg(long)]
    exclude: Option<String>,

    /// Only generate input WAVs
    #[arg(long)]
    inputs_only: bool,

    /// Per-variant timeout in seconds
    #[arg(long, default_value_t = 120)]
    timeout: u64,
}

const SR: u32 = 44100;

// ---------------------------------------------------------------------------
// Test signal generators
// ---------------------------------------------------------------------------

fn gen_click(sr: u32, duration: f32) -> Vec<f32> {
    let n = (sr as f32 * duration) as usize;
    let mut signal = vec![0.0f32; n];
    signal[0] = 1.0;
    signal
}

fn gen_sparse_hits(sr: u32, duration: f32) -> Vec<f32> {
    let n = (sr as f32 * duration) as usize;
    let mut signal = vec![0.0f32; n];

    let add_hit = |signal: &mut Vec<f32>, offset_s: f32, freq: f32, decay_ms: f32, noise_mix: f32, noise_decay_ms: f32, seed: u64| {
        let start = (offset_s * sr as f32) as usize;
        let length = (0.3 * sr as f32) as usize;
        let mut rng = explore_dsp::primitives::Lcg::new(seed);
        for j in 0..length {
            let t = j as f32 / sr as f32;
            let tone_env = (-t / (decay_ms * 0.001)).exp();
            let tone = (2.0 * std::f32::consts::PI * freq * t).sin() * tone_env;
            let mut val = tone;
            if noise_mix > 0.0 {
                let noise_env = (-t / (noise_decay_ms * 0.001)).exp();
                let noise = rng.next_bipolar() * noise_env;
                val = (1.0 - noise_mix) * tone + noise_mix * noise;
            }
            let idx = start + j;
            if idx < signal.len() {
                signal[idx] += val;
            }
        }
    };

    add_hit(&mut signal, 0.5, 60.0, 80.0, 0.1, 10.0, 42);
    add_hit(&mut signal, 1.5, 200.0, 30.0, 0.6, 20.0, 43);
    add_hit(&mut signal, 2.5, 800.0, 5.0, 0.9, 8.0, 44);
    add_hit(&mut signal, 3.5, 150.0, 50.0, 0.4, 40.0, 45);

    let peak = signal.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    if peak > 0.0 {
        let scale = 0.9 / peak;
        for s in signal.iter_mut() {
            *s *= scale;
        }
    }
    signal
}

fn gen_sustained_chord(sr: u32, duration: f32) -> Vec<f32> {
    let n = (sr as f32 * duration) as usize;
    let mut signal = vec![0.0f64; n];

    let fundamentals = [130.81f64, 164.81, 196.00];
    let detune_cents = [-3.0f64, 0.0, 2.0];

    for (fund, cents) in fundamentals.iter().zip(detune_cents.iter()) {
        let freq = fund * 2.0f64.powf(cents / 1200.0);
        for harmonic in 1..=4 {
            let amplitude = 1.0 / harmonic as f64;
            for i in 0..n {
                let t = i as f64 / sr as f64;
                signal[i] += amplitude * (2.0 * std::f64::consts::PI * freq * harmonic as f64 * t).sin();
            }
        }
    }

    let fade_in = (0.05 * sr as f32) as usize;
    let fade_out = (0.3 * sr as f32) as usize;
    for i in 0..fade_in.min(n) {
        signal[i] *= i as f64 / fade_in as f64;
    }
    for i in 0..fade_out.min(n) {
        signal[n - 1 - i] *= i as f64 / fade_out as f64;
    }

    let peak = signal.iter().map(|s| s.abs()).fold(0.0f64, f64::max);
    if peak > 0.0 {
        let scale = 0.9 / peak;
        signal.iter().map(|&s| (s * scale) as f32).collect()
    } else {
        signal.iter().map(|&s| s as f32).collect()
    }
}

fn gen_pink_noise_burst(sr: u32, duration: f32, burst_duration: f32) -> Vec<f32> {
    let n = (sr as f32 * duration) as usize;
    let burst_n = (sr as f32 * burst_duration) as usize;

    // Voss-McCartney pink noise
    let mut rng = explore_dsp::primitives::Lcg::new(123);
    let num_rows = 16;
    let mut rows = vec![vec![0.0f32; num_rows]; burst_n];
    for j in 0..burst_n {
        for col in 0..num_rows {
            rows[j][col] = rng.next_bipolar();
        }
    }
    for col in 1..num_rows {
        let step = 1 << col;
        for j in 1..burst_n {
            if j % step != 0 {
                rows[j][col] = rows[j - 1][col];
            }
        }
    }

    let mut pink: Vec<f32> = rows.iter().map(|row| row.iter().sum::<f32>()).collect();

    let fade = (0.005 * sr as f32) as usize;
    for i in 0..fade.min(burst_n) {
        pink[i] *= i as f32 / fade as f32;
    }
    for i in 0..fade.min(burst_n) {
        pink[burst_n - 1 - i] *= i as f32 / fade as f32;
    }

    let mut signal = vec![0.0f32; n];
    for i in 0..burst_n.min(n) {
        signal[i] = pink[i];
    }

    let peak = signal.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    if peak > 0.0 {
        let scale = 0.9 / peak;
        for s in signal.iter_mut() {
            *s *= scale;
        }
    }
    signal
}

fn generate_inputs(inputs_dir: &Path) {
    fs::create_dir_all(inputs_dir).unwrap();

    let signals: Vec<(&str, Vec<f32>)> = vec![
        ("click", gen_click(SR, 3.0)),
        ("sparse_hits", gen_sparse_hits(SR, 5.0)),
        ("sustained_chord", gen_sustained_chord(SR, 5.0)),
        ("pink_noise_burst", gen_pink_noise_burst(SR, 3.0, 1.0)),
    ];

    for (name, signal) in &signals {
        let path = inputs_dir.join(format!("{}.wav", name));
        write_wav(&path, signal, SR);
        let dur = signal.len() as f32 / SR as f32;
        println!("  {} ({:.1}s, {} samples)", path.display(), dur, signal.len());
    }
}

// ---------------------------------------------------------------------------
// WAV I/O
// ---------------------------------------------------------------------------

fn write_wav(path: &Path, samples: &[f32], sr: u32) {
    let spec = WavSpec {
        channels: 1,
        sample_rate: sr,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };
    let mut writer = WavWriter::create(path, spec).unwrap();
    for &s in samples {
        writer.write_sample(s).unwrap();
    }
    writer.finalize().unwrap();
}

fn write_wav_stereo(path: &Path, samples: &[[f32; 2]], sr: u32) {
    let spec = WavSpec {
        channels: 2,
        sample_rate: sr,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };
    let mut writer = WavWriter::create(path, spec).unwrap();
    for s in samples {
        writer.write_sample(s[0]).unwrap();
        writer.write_sample(s[1]).unwrap();
    }
    writer.finalize().unwrap();
}

fn load_wav(path: &str) -> (Vec<f32>, u32) {
    let mut reader = hound::WavReader::open(path).unwrap();
    let spec = reader.spec();
    let sr = spec.sample_rate;
    let samples: Vec<f32> = match spec.sample_format {
        SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
        SampleFormat::Int => {
            let max_val = (1 << (spec.bits_per_sample - 1)) as f32;
            reader.samples::<i32>().map(|s| s.unwrap() as f32 / max_val).collect()
        }
    };
    // Convert to mono if stereo
    if spec.channels == 2 {
        let mono: Vec<f32> = samples.chunks(2).map(|c| (c[0] + c.get(1).copied().unwrap_or(0.0)) * 0.5).collect();
        (mono, sr)
    } else {
        (samples, sr)
    }
}

// ---------------------------------------------------------------------------
// Filename / category helpers
// ---------------------------------------------------------------------------

fn safe_filename(s: &str, max_len: usize) -> String {
    let s = s.replace(' ', "_").replace('/', "_").replace('\\', "_");
    let safe: String = s.chars().filter(|c| c.is_alphanumeric() || *c == '_' || *c == '-' || *c == '.').collect();
    safe.chars().take(max_len).collect()
}

fn make_param_description(params: &HashMap<String, serde_json::Value>, max_len: usize) -> String {
    let mut parts: Vec<String> = Vec::new();
    let mut keys: Vec<&String> = params.keys().collect();
    keys.sort();
    for k in keys {
        let v = &params[k];
        if let Some(f) = v.as_f64() {
            parts.push(format!("{}{:.2}", &k[..k.len().min(6)], f).replace('.', ""));
        } else if let Some(i) = v.as_i64() {
            parts.push(format!("{}{}", &k[..k.len().min(6)], i));
        } else if let Some(s) = v.as_str() {
            parts.push(format!("{}_{}", &k[..k.len().min(4)], &s[..s.len().min(6)]));
        }
    }
    let desc = parts.join("_");
    desc.chars().take(max_len).collect::<String>().trim_end_matches('_').to_string()
}

fn matches_patterns(id: &str, patterns: &[&str]) -> bool {
    for pat in patterns {
        if pat.contains('*') {
            let prefix = pat.trim_end_matches('*');
            if id.starts_with(prefix) {
                return true;
            }
        } else if id == *pat {
            return true;
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Work item
// ---------------------------------------------------------------------------

struct WorkItem {
    effect_id: String,
    params: HashMap<String, serde_json::Value>,
    category: String,
}

fn main() {
    let cli = Cli::parse();

    if cli.workers > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(cli.workers)
            .build_global()
            .unwrap();
    }

    let explore_dir = PathBuf::from("explore");
    let inputs_dir = explore_dir.join("inputs");
    let output_base = PathBuf::from(&cli.output);

    // Step 1: Generate or load inputs
    if cli.input_wav.is_none() {
        println!("=== Generating synthetic inputs ===");
        generate_inputs(&inputs_dir);
        println!();
    }

    if cli.inputs_only {
        return;
    }

    // Collect input files
    let input_files: Vec<(String, PathBuf)> = if let Some(ref input) = cli.input_wav {
        let p = PathBuf::from(input);
        let name = p.file_stem().unwrap().to_string_lossy().to_string();
        vec![(name, p)]
    } else {
        let mut files = Vec::new();
        if inputs_dir.exists() {
            for entry in fs::read_dir(&inputs_dir).unwrap() {
                let entry = entry.unwrap();
                let path = entry.path();
                if path.extension().map_or(false, |e| e == "wav") {
                    let name = path.file_stem().unwrap().to_string_lossy().to_string();
                    files.push((name, path));
                }
            }
        }
        files.sort();
        files
    };

    let effects = discover_effects();
    println!("Discovered {} effects", effects.len());

    // Step 2: Process each input
    for (name, wav_path) in &input_files {
        let out_dir = output_base.join(name);
        println!("\n{}", "=".repeat(60));
        println!("  Input: {}", name);
        println!("  Output: {}", out_dir.display());
        println!("{}", "=".repeat(60));

        let (samples, sr) = load_wav(wav_path.to_str().unwrap());
        println!("Input: {}, {} samples, {} Hz, {:.2}s",
                 wav_path.display(), samples.len(), sr, samples.len() as f32 / sr as f32);

        // Build work items
        let mut work_items: Vec<WorkItem> = Vec::new();
        let mut skipped = 0usize;

        // Filter effects
        let include_patterns: Vec<&str> = cli.include.as_ref()
            .map(|s| s.split(',').map(|p| p.trim()).collect())
            .unwrap_or_default();
        let exclude_patterns: Vec<&str> = cli.exclude.as_ref()
            .map(|s| s.split(',').map(|p| p.trim()).collect())
            .unwrap_or_default();

        for effect in &effects {
            if !include_patterns.is_empty() && !matches_patterns(effect.id, &include_patterns) {
                continue;
            }
            if !exclude_patterns.is_empty() && matches_patterns(effect.id, &exclude_patterns) {
                continue;
            }

            let variants = (effect.variants)();
            let variant_params = if variants.is_empty() {
                vec![HashMap::new()]
            } else {
                variants
            };

            for params in variant_params {
                if work_items.len() + skipped >= cli.max_variants {
                    break;
                }

                let param_desc = if params.is_empty() { "default".to_string() } else { make_param_description(&params, 50) };
                let code = effect.id.split('_').next().unwrap_or(effect.id);
                let fname = safe_filename(&format!("{}_{}.wav", code, param_desc), 90);
                let fpath = out_dir.join(effect.category).join(&fname);

                if fpath.exists() {
                    skipped += 1;
                    continue;
                }

                work_items.push(WorkItem {
                    effect_id: effect.id.to_string(),
                    params,
                    category: effect.category.to_string(),
                });

                if work_items.len() + skipped >= cli.max_variants {
                    break;
                }
            }
        }

        if skipped > 0 {
            println!("Skipped {} already-generated variants", skipped);
        }

        println!("Processing {} variants with rayon", work_items.len());

        let total_start = Instant::now();
        let ok_count = AtomicUsize::new(0);
        let fail_count = AtomicUsize::new(0);
        let done_count = AtomicUsize::new(0);
        let total = work_items.len();

        // Build effect lookup
        let effect_map: HashMap<&str, &EffectEntry> = effects.iter().map(|e| (e.id, e)).collect();

        work_items.par_iter().for_each(|item| {
            let effect = match effect_map.get(item.effect_id.as_str()) {
                Some(e) => e,
                None => {
                    fail_count.fetch_add(1, Ordering::Relaxed);
                    return;
                }
            };

            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                (effect.process)(&samples, sr, &item.params)
            }));

            match result {
                Ok(output) => {
                    // Post-process and validate
                    let (final_data, is_stereo) = match output {
                        AudioOutput::Mono(data) => {
                            if data.is_empty() || data.iter().any(|s| !s.is_finite()) || data.iter().map(|s| s.abs()).fold(0.0f32, f32::max) > 1e6 {
                                fail_count.fetch_add(1, Ordering::Relaxed);
                                done_count.fetch_add(1, Ordering::Relaxed);
                                return;
                            }
                            (AudioOutput::Mono(post_process(&data, sr)), false)
                        }
                        AudioOutput::Stereo(data) => {
                            if data.is_empty() || data.iter().any(|s| !s[0].is_finite() || !s[1].is_finite()) {
                                fail_count.fetch_add(1, Ordering::Relaxed);
                                done_count.fetch_add(1, Ordering::Relaxed);
                                return;
                            }
                            (AudioOutput::Stereo(post_process_stereo(&data, sr)), true)
                        }
                    };

                    let param_desc = if item.params.is_empty() { "default".to_string() } else { make_param_description(&item.params, 50) };
                    let code = item.effect_id.split('_').next().unwrap_or(&item.effect_id);
                    let fname = safe_filename(&format!("{}_{}.wav", code, param_desc), 90);
                    let cat_dir = out_dir.join(&item.category);
                    fs::create_dir_all(&cat_dir).ok();
                    let fpath = cat_dir.join(&fname);
                    let tmp_path = fpath.with_extension("wav.part");

                    let write_ok = match final_data {
                        AudioOutput::Mono(ref data) => {
                            write_wav(&tmp_path, data, sr);
                            true
                        }
                        AudioOutput::Stereo(ref data) => {
                            write_wav_stereo(&tmp_path, data, sr);
                            true
                        }
                    };

                    if write_ok {
                        fs::rename(&tmp_path, &fpath).ok();
                        ok_count.fetch_add(1, Ordering::Relaxed);
                    } else {
                        fail_count.fetch_add(1, Ordering::Relaxed);
                    }
                }
                Err(_) => {
                    fail_count.fetch_add(1, Ordering::Relaxed);
                }
            }

            let done = done_count.fetch_add(1, Ordering::Relaxed) + 1;
            if done % 20 == 0 || done == total {
                let elapsed = total_start.elapsed().as_secs_f32();
                let rate = done as f32 / elapsed;
                let eta = if rate > 0.0 { (total - done) as f32 / rate } else { 0.0 };
                let ok = ok_count.load(Ordering::Relaxed);
                let fail = fail_count.load(Ordering::Relaxed);
                eprintln!("  [{}/{}] {} ok, {} failed, {:.1}s elapsed, ~{:.0}s remaining",
                         done, total, ok, fail, elapsed, eta);
            }
        });

        let elapsed = total_start.elapsed().as_secs_f32();
        let ok = ok_count.load(Ordering::Relaxed);
        let fail = fail_count.load(Ordering::Relaxed);
        println!("\nGenerated {} outputs in {:.1}s ({:.1} variants/sec, {} failed)",
                 ok, elapsed, ok as f32 / elapsed, fail);
    }

    println!("\n=== Done. Outputs in {}/ ===", output_base.display());
}
