# Plugin Real-Time Audio Guidelines

Rules for writing DSP code that runs in audio plugin contexts (VST3/CLAP).

## The Golden Rule: No Allocations in the Audio Thread

The `process()` callback runs on the host's real-time audio thread. Any heap
allocation (`Vec::new`, `String`, `Box`, `clone()` of heap types) can cause
intermittent CPU spikes due to:

- Allocator mutex contention with the GUI thread
- Page faults when the OS maps fresh memory
- Allocator bookkeeping (coalescing, free-list traversal)

These spikes are **intermittent** — the allocator is usually fast, but
occasionally takes 10-100x longer. This manifests as random audio dropouts
and 100% CPU spikes that are impossible to reproduce reliably.

### What to do instead

1. **Pre-allocate all buffers at plugin init** (`initialize()` or `new()`).
   Size them for the worst case based on parameter ranges.

2. **Create a processor struct** that holds all pre-allocated state:
   - Delay line buffers
   - Filter state (y1, x1 for IIR filters)
   - Scratch/working arrays
   - Output buffers
   - Cached matrices or lookup tables

3. **Maintain DSP state across process calls.** For effects like reverb,
   delay lines and filter state must persist — resetting every buffer
   boundary creates audible artifacts and adds unnecessary latency.

4. **Cache expensive computations.** Matrix construction (e.g., QR
   decomposition for random orthogonal matrices) should only run when
   the relevant parameter actually changes, not every process call.

5. **Update parameters in-place.** Instead of building a new params struct
   with `vec![...]` every call, mutate existing fields:
   ```rust
   // BAD: allocates 8 Vecs every process call
   fn build_params(&self) -> DspParams {
       DspParams { delay_times: vec![...], ... }
   }

   // GOOD: mutates existing Vecs in-place (no allocation)
   fn update_params(&mut self) {
       self.cached_params.delay_times[0] = self.param1.value();
       ...
   }
   ```

6. **Never call `.clone()` on types containing Vecs/Strings** in the audio
   thread. If you need a copy, pre-allocate the destination and copy into it.

## Pre-allocated Processor Pattern

See `reverb-dsp/src/processor.rs` for the reference implementation:

```rust
pub struct FdnProcessor {
    // All buffers allocated once in new()
    delay_bufs: Vec<Vec<f64>>,     // [N][MAX_DELAY]
    pre_delay_buf: Vec<f64>,       // [MAX_PRE_DELAY]
    diff_bufs: Vec<Vec<f64>>,      // [STAGES][MAX_DIFF_DELAY]

    // Filter state persists across calls
    damping_y1: [f64; N],
    dc_x1: [f64; N],
    dc_y1: [f64; N],

    // Cached (only rebuilt on param change)
    mat: Vec<f64>,
    prev_matrix_type: String,
    prev_matrix_seed: i32,
}

impl FdnProcessor {
    pub fn new() -> Self { /* allocate everything */ }
    pub fn reset(&mut self) { /* zero state, don't dealloc */ }
    pub fn process(&mut self, input: &[f64], params: &Params, output: &mut [f64]) {
        // Zero allocations in here
    }
}
```

The plugin stores the processor and calls `process()` directly:

```rust
fn process(&mut self, buffer: &mut Buffer, ...) {
    self.update_dsp_params();  // in-place, no alloc
    self.processor.process_stereo(
        &self.input_l[..n], &self.input_r[..n],  // pre-allocated
        &self.cached_params,
        &mut self.output_l[..n], &mut self.output_r[..n],  // pre-allocated
    );
}
```

## Latency

With persistent DSP state, there is no need to buffer input samples before
processing. The plugin can process whatever buffer size the host provides
directly, reporting zero added latency. The old collect-and-process pattern
(accumulating 8192 samples = 186ms latency) was only necessary because DSP
state was recreated from scratch each block.

## Testing for Real-Time Safety

### RTF (Real-Time Factor) Tests

Every DSP crate must have integration tests asserting RTF < 0.5 (processing
takes less than half the audio duration). This leaves headroom for the host,
other plugins, and system overhead. See `tests/realtime.rs` in each crate.

```rust
#[test]
fn realtime_processor_small_buffers() {
    let mut proc = FdnProcessor::new();
    let mut output = vec![0.0; 256 * 2];
    let start = Instant::now();
    for _ in 0..n_calls {
        proc.process(&input, &params, &mut output);
    }
    let rtf = elapsed / audio_secs;
    assert!(rtf < 0.5);
}
```

Key test scenarios:
- **Small buffer sizes** (128-512 samples) — realistic plugin buffers
- **Worst-case parameters** — max modulation, expensive matrix types
- **Stereo processing** — 2x the work
- **Repeated calls** — catches state-dependent slowdowns

### Criterion Benchmarks

Use `cargo bench` for detailed profiling and regression detection:
```bash
cargo bench -p reverb-dsp --bench reverb_bench
cargo bench -p lossy-dsp --bench lossy_bench
```

The `reverb_plugin_realistic` benchmark group specifically tests the
allocating vs pre-allocated paths with small buffer sizes.

### Output Safety Tests

Every configuration must produce finite output with bounded peaks:
```rust
assert!(output.iter().all(|x| x.is_finite()));
assert!(peak < 1e6);
```

## Checklist for New Plugins

- [ ] DSP processor struct with `new()`, `reset()`, `process()`
- [ ] All buffers pre-allocated in `new()` at max parameter range sizes
- [ ] Matrix/table caching with dirty-checking on relevant params
- [ ] Parameter update via in-place mutation, not reconstruction
- [ ] No `Vec::new`, `clone()`, `collect()`, `String` in process path
- [ ] RTF assertion tests with small buffer sizes
- [ ] Criterion benchmarks comparing alloc vs prealloc paths
- [ ] Output safety tests across parameter extremes
- [ ] Zero reported latency (persistent state, no buffering)
