//! Golden/regression tests for `StereoFractalProcessor`.
//!
//! These tests ensure the processor produces identical output to the allocating
//! API (`render_fractal_stereo`), so any future refactoring that changes behavior
//! will be caught.

use fractal_dsp::chain::render_fractal_stereo;
use fractal_dsp::params::{FractalParams, SR};
use fractal_dsp::processor::{StereoFractalProcessor, BLOCK_SIZE};

// ---------------------------------------------------------------------------
// Helpers (mirroring tolerance.rs pattern)
// ---------------------------------------------------------------------------

fn make_stereo_sine(len: usize) -> (Vec<f64>, Vec<f64>) {
    let left: Vec<f64> = (0..len)
        .map(|i| (2.0 * std::f64::consts::PI * 440.0 * i as f64 / SR).sin())
        .collect();
    let right: Vec<f64> = (0..len)
        .map(|i| (2.0 * std::f64::consts::PI * 660.0 * i as f64 / SR).sin())
        .collect();
    (left, right)
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

// ---------------------------------------------------------------------------
// Helper: run processor over input, return full output (including latency).
//
// The processor fires process_block when in_pos reaches BLOCK_SIZE and
// immediately starts draining on the same loop iteration. This means the
// first output sample appears at index BLOCK_SIZE - 1 (not BLOCK_SIZE),
// giving an effective latency of BLOCK_SIZE - 1 samples of silence.
// ---------------------------------------------------------------------------

fn run_processor(
    left_in: &[f64],
    right_in: &[f64],
    params: &FractalParams,
) -> (Vec<f64>, Vec<f64>) {
    let mut proc = StereoFractalProcessor::new();
    let total = left_in.len();
    let mut left_out = vec![0.0; total];
    let mut right_out = vec![0.0; total];
    proc.process(left_in, right_in, params, &mut left_out, &mut right_out);
    (left_out, right_out)
}

// ===========================================================================
// Test 1: processor_matches_allocating_default
// ===========================================================================

/// Process 2*BLOCK_SIZE samples through the processor and compare output to
/// calling render_fractal_stereo directly on the first block. With default
/// params (no feedback, no spread), the processor just wraps the allocating API.
#[test]
fn processor_matches_allocating_default() {
    let total = 2 * BLOCK_SIZE;
    let (left_in, right_in) = make_stereo_sine(total);
    let params = FractalParams::default();

    // --- Processor path ---
    let (proc_l, proc_r) = run_processor(&left_in, &right_in, &params);

    // --- Allocating API path: process first block ---
    let (alloc_l, alloc_r) =
        render_fractal_stereo(&left_in[..BLOCK_SIZE], &right_in[..BLOCK_SIZE], &params);

    // First BLOCK_SIZE - 1 samples are silence (latency).
    for i in 0..(BLOCK_SIZE - 1) {
        assert_eq!(proc_l[i], 0.0, "Left latency sample {i} should be zero");
        assert_eq!(proc_r[i], 0.0, "Right latency sample {i} should be zero");
    }

    // Starting at index BLOCK_SIZE - 1, the processor drains block 1's output.
    for i in 0..BLOCK_SIZE {
        assert!(
            (proc_l[BLOCK_SIZE - 1 + i] - alloc_l[i]).abs() < 1e-15,
            "Left sample {i}: proc={:.15e} vs alloc={:.15e}",
            proc_l[BLOCK_SIZE - 1 + i],
            alloc_l[i]
        );
        assert!(
            (proc_r[BLOCK_SIZE - 1 + i] - alloc_r[i]).abs() < 1e-15,
            "Right sample {i}: proc={:.15e} vs alloc={:.15e}",
            proc_r[BLOCK_SIZE - 1 + i],
            alloc_r[i]
        );
    }
}

// ===========================================================================
// Test 2: processor_golden_default
// ===========================================================================

/// Process a known sine wave through the processor with default params and
/// assert golden values (rms, peak, head, tail) match.
#[test]
fn processor_golden_default() {
    let total = 2 * BLOCK_SIZE;
    let (left_in, right_in) = make_stereo_sine(total);
    let params = FractalParams::default();

    let (proc_l, proc_r) = run_processor(&left_in, &right_in, &params);

    // Extract meaningful output (after latency: starts at BLOCK_SIZE - 1).
    let out_l = &proc_l[BLOCK_SIZE - 1..2 * BLOCK_SIZE - 1];
    let out_r = &proc_r[BLOCK_SIZE - 1..2 * BLOCK_SIZE - 1];

    check(
        out_l,
        &Golden {
            rms: 2.223783374603610e-1,
            peak: 5.250000000000000e-1,
            head: 9.183608707259955e-3,
            tail: 1.152180083164316e-1,
        },
        TOL,
        "proc_default_L",
    );
    check(
        out_r,
        &Golden {
            rms: 2.381604857642720e-1,
            peak: 5.250000000000000e-1,
            head: 5.413033084227743e0,
            tail: 4.478754970263103e0,
        },
        TOL,
        "proc_default_R",
    );
}

// ===========================================================================
// Test 3: processor_golden_with_effects
// ===========================================================================

/// Process with filter_type=1, crush=0.3, gate=0.1 and assert golden values.
#[test]
fn processor_golden_with_effects() {
    let total = 2 * BLOCK_SIZE;
    let (left_in, right_in) = make_stereo_sine(total);
    let mut params = FractalParams::default();
    params.filter_type = 1;
    params.crush = 0.3;
    params.gate = 0.1;

    let (proc_l, proc_r) = run_processor(&left_in, &right_in, &params);

    let out_l = &proc_l[BLOCK_SIZE - 1..2 * BLOCK_SIZE - 1];
    let out_r = &proc_r[BLOCK_SIZE - 1..2 * BLOCK_SIZE - 1];

    check(
        out_l,
        &Golden {
            rms: 2.266710429156199e-1,
            peak: 5.250000000000000e-1,
            head: 4.198445020362849e-1,
            tail: 1.012680488707891e-1,
        },
        TOL,
        "proc_effects_L",
    );
    check(
        out_r,
        &Golden {
            rms: 2.406578816836237e-1,
            peak: 5.250000000000000e-1,
            head: 5.354335690360997e0,
            tail: 5.407675846669148e0,
        },
        TOL,
        "proc_effects_R",
    );
}

// ===========================================================================
// Test 4: processor_golden_stereo_spread
// ===========================================================================

/// Process with layer_spread=0.5 and assert golden values.
#[test]
fn processor_golden_stereo_spread() {
    let total = 2 * BLOCK_SIZE;
    let (left_in, right_in) = make_stereo_sine(total);
    let mut params = FractalParams::default();
    params.layer_spread = 0.5;

    let (proc_l, proc_r) = run_processor(&left_in, &right_in, &params);

    let out_l = &proc_l[BLOCK_SIZE - 1..2 * BLOCK_SIZE - 1];
    let out_r = &proc_r[BLOCK_SIZE - 1..2 * BLOCK_SIZE - 1];

    check(
        out_l,
        &Golden {
            rms: 1.919421823158007e-1,
            peak: 5.250000000000000e-1,
            head: 1.237265435392675e-2,
            tail: 8.958205460997321e-2,
        },
        TOL,
        "proc_spread_L",
    );
    check(
        out_r,
        &Golden {
            rms: 2.048642521119305e-1,
            peak: 5.250000000000000e-1,
            head: 4.649722757079067e0,
            tail: 3.881368482771526e0,
        },
        TOL,
        "proc_spread_R",
    );
}

// ===========================================================================
// Test 5: processor_deterministic
// ===========================================================================

/// Running twice with the same input and params gives identical output.
#[test]
fn processor_deterministic() {
    let total = 2 * BLOCK_SIZE;
    let (left_in, right_in) = make_stereo_sine(total);
    let params = FractalParams::default();

    let (out_l_1, out_r_1) = run_processor(&left_in, &right_in, &params);
    let (out_l_2, out_r_2) = run_processor(&left_in, &right_in, &params);

    for i in 0..total {
        assert_eq!(
            out_l_1[i], out_l_2[i],
            "Left non-deterministic at sample {i}"
        );
        assert_eq!(
            out_r_1[i], out_r_2[i],
            "Right non-deterministic at sample {i}"
        );
    }
}

// ===========================================================================
// Test 6: processor_matches_chain_output
// ===========================================================================

/// Process N blocks through the processor. Separately, call render_fractal_stereo
/// block by block. Verify both produce identical output. This ensures the
/// processor's block buffering logic matches the expected behavior.
///
/// With default params (feedback=0, spread=0), the processor simply calls
/// render_fractal_stereo on each accumulated block with no feedback mixing.
#[test]
fn processor_matches_chain_output() {
    let num_blocks = 3;
    let total = num_blocks * BLOCK_SIZE;
    let (left_in, right_in) = make_stereo_sine(total);
    let params = FractalParams::default();

    // --- Processor path ---
    let (proc_l, proc_r) = run_processor(&left_in, &right_in, &params);

    // --- Manual chain path: replicate the processor's block logic ---
    let mut chain_blocks_l: Vec<Vec<f64>> = Vec::new();
    let mut chain_blocks_r: Vec<Vec<f64>> = Vec::new();

    for blk in 0..num_blocks {
        let start = blk * BLOCK_SIZE;
        let end = start + BLOCK_SIZE;
        let block_l = &left_in[start..end];
        let block_r = &right_in[start..end];

        let (out_l, out_r) = render_fractal_stereo(block_l, block_r, &params);
        chain_blocks_l.push(out_l);
        chain_blocks_r.push(out_r);
    }

    // The processor fires at in_pos == BLOCK_SIZE and immediately starts
    // draining on the same sample, so block N's output starts at sample
    // N * BLOCK_SIZE + (BLOCK_SIZE - 1) = (N+1) * BLOCK_SIZE - 1.
    for blk in 0..num_blocks {
        let proc_start = blk * BLOCK_SIZE + (BLOCK_SIZE - 1);
        for i in 0..BLOCK_SIZE {
            if proc_start + i >= total {
                break;
            }
            assert!(
                (proc_l[proc_start + i] - chain_blocks_l[blk][i]).abs() < 1e-15,
                "Block {blk} Left sample {i}: proc={:.15e} vs chain={:.15e}",
                proc_l[proc_start + i],
                chain_blocks_l[blk][i]
            );
            assert!(
                (proc_r[proc_start + i] - chain_blocks_r[blk][i]).abs() < 1e-15,
                "Block {blk} Right sample {i}: proc={:.15e} vs chain={:.15e}",
                proc_r[proc_start + i],
                chain_blocks_r[blk][i]
            );
        }
    }
}
