# Full Rust Architecture Implementation Plan

## Overview

Port both lossy and reverb DSP engines to Rust, wrap with nih-plug (VST3/CLAP/standalone), add pyo3 Python bindings, and build an OSC bridge for live DAW<->Claude agentic tuning. All new code lives in `rust/` at the project root. **No existing files are modified.**

## Current State Analysis

| Component | Lines | Status |
|-----------|-------|--------|
| Lossy DSP (`lossy/engine/`) | ~1,038 | Complete, partially Numba |
| Reverb DSP (`reverb/engine/`) | ~780 | Complete, fully Numba |
| Reverb Primitives (`reverb/primitives/`) | ~303 | Complete, Numba |
| Shared (`shared/`) | ~1,586 | Complete, Python |
| Presets | 110+ JSON | Rich library |
| Rust code | 0 | Nothing exists |

### Key Discoveries:
- DSP engines have zero GUI/IO deps — clean port targets
- Parameter dict is the only interface between all layers
- Lossy STFT engine (`spectral.py:30-180`) is the most complex module — needs `realfft`
- Lossy uses `np.random.RandomState(seed)` but user says Rust RNG is fine
- All Numba `@njit` functions are sample-loop processing — maps directly to Rust
- Preset JSON format is flat dict + `_meta` object, backward-compatible with default merging
- Lossy GUI is 1,665 lines (`lossy/gui/gui.py`) — will be copied and rewired
- nih-plug is git-only (not on crates.io), requires Rust 1.80+
- `realfft` crate matches `numpy.fft.rfft` output format (N/2+1 complex bins)

## Desired End State

```
rust/                           <- NEW directory, all new code here
  Cargo.toml                    <- workspace root
  crates/
    lossy-dsp/                  <- lossy DSP engine in pure Rust
    reverb-dsp/                 <- reverb DSP engine in pure Rust
    shared-dsp/                 <- OSC, ring buffer, metrics, smoothing
    lossy-plugin/               <- nih-plug VST3/CLAP/standalone
    reverb-plugin/              <- nih-plug VST3/CLAP/standalone
    lossy-python/               <- pyo3 bindings for lossy
    reverb-python/              <- pyo3 bindings for reverb
  test-audio/                   <- WAV files for integration testing
  py/                           <- copied + modified Python GUI wired to Rust
```

Both pedals:
- Render identically (perceptually) to Python/Numba versions
- Load existing JSON presets unchanged
- Run as VST3/CLAP/standalone plugins with egui GUI
- Callable from Python via pyo3 (drop-in replacement for Numba)
- Communicate with Claude via OSC bridge for live agentic tuning

### Verification:
- Load same preset in Python and Rust, render same WAV input
- Listen to both outputs — should be indistinguishable
- All 50+ lossy presets and 60+ reverb presets load without error
- VST3 loads in a DAW, processes audio, responds to OSC parameter changes
- Python GUI works in both Local (pyo3) and Plugin (OSC) modes

## What We're NOT Doing

- Not modifying any existing Python code in `reverb/`, `lossy/`, or `shared/`
- Not achieving bit-exact numerical matching (perceptually identical is the target)
- Not matching numpy's RNG — Rust's own RNG is fine
- Not porting `shared/analysis.py` or `shared/audio_features.py` to Rust (stays Python)
- Not porting the LLM tuner to Rust (stays Python, Claude Agent SDK is Python-only)
- Not building a marketing-ready plugin GUI — functional egui controls are sufficient
- Not implementing MIDI mapping or automation recording in Phase 1

## Implementation Approach

**Lossy first** because the user requested it and because lossy's STFT engine is the hardest porting challenge — solving it first de-risks the project. Reverb's FDN is algorithmically simpler.

**Phase ordering:** DSP core -> pyo3 bindings -> shared-dsp -> nih-plug plugin -> reverb port -> OSC agentic loop. Each phase produces a testable artifact.

---

## Phase 1: Rust Workspace + Lossy DSP Core

### Overview
Set up the Cargo workspace and port all lossy DSP modules to Rust. Produce a CLI binary that reads a WAV, processes it through the Rust lossy engine, and writes the output WAV. This is the foundation everything else builds on.

### Changes Required:

#### 1. Workspace Root
**File**: `rust/Cargo.toml`

```toml
[workspace]
resolver = "2"
members = [
    "crates/lossy-dsp",
]

[workspace.dependencies]
realfft = "3.5"
num-complex = "0.4"
rand = "0.9"
rand_chacha = "0.9"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
hound = "3.5"
```

#### 2. Lossy DSP Crate
**File**: `rust/crates/lossy-dsp/Cargo.toml`

```toml
[package]
name = "lossy-dsp"
version = "0.1.0"
edition = "2021"

[dependencies]
realfft.workspace = true
num-complex.workspace = true
rand.workspace = true
rand_chacha.workspace = true
serde.workspace = true
serde_json.workspace = true

[[bin]]
name = "lossy-cli"
path = "src/bin/cli.rs"

[dev-dependencies]
hound.workspace = true
```

#### 3. Parameter Module
**File**: `rust/crates/lossy-dsp/src/params.rs`
**Port of**: `lossy/engine/params.py` (209 lines)

```rust
pub const SR: f64 = 44100.0;
pub const N_DELAY_NODES: usize = 8; // not used by lossy but shared convention

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LossyParams {
    // Spectral Loss
    pub inverse: i32,          // 0=standard, 1=residual
    pub jitter: f64,           // 0.0-1.0
    pub loss: f64,             // 0.0-1.0
    pub window_size: i32,      // 64-16384
    pub hop_divisor: i32,      // 1-8
    pub n_bands: i32,          // 2-64
    pub global_amount: f64,    // 0.0-1.0
    pub phase_loss: f64,       // 0.0-1.0
    pub quantizer: i32,        // 0=uniform, 1=compand
    pub pre_echo: f64,         // 0.0-1.0
    pub noise_shape: f64,      // 0.0-1.0
    pub weighting: f64,        // 0.0-1.0
    pub hf_threshold: f64,     // 0.0-1.0
    pub transient_ratio: f64,  // 1.5-20.0
    pub slushy_rate: f64,      // 0.001-0.5
    // Crush
    pub crush: f64,            // 0.0-1.0
    pub decimate: f64,         // 0.0-1.0
    // Packets
    pub packets: i32,          // 0=clean, 1=loss, 2=repeat
    pub packet_rate: f64,      // 0.0-1.0
    pub packet_size: f64,      // 5.0-200.0 ms
    // Filter
    pub filter_type: i32,      // 0=bypass, 1=bandpass, 2=notch
    pub filter_freq: f64,      // 20.0-20000.0
    pub filter_width: f64,     // 0.0-1.0
    pub filter_slope: i32,     // 0=6dB, 1=24dB, 2=96dB
    // Effects
    pub verb: f64,             // 0.0-1.0
    pub decay: f64,            // 0.0-1.0
    pub verb_position: i32,    // 0=pre, 1=post
    pub freeze: i32,           // 0=off, 1=on
    pub freeze_mode: i32,      // 0=slushy, 1=solid
    pub freezer: f64,          // 0.0-1.0
    pub gate: f64,             // 0.0-1.0
    // Hidden
    pub threshold: f64,        // 0.0-1.0
    pub auto_gain: f64,        // 0.0-1.0
    pub loss_gain: f64,        // 0.0-1.0
    // Bounce
    pub bounce: i32,           // 0=off, 1=on
    pub bounce_target: i32,    // index into BOUNCE_TARGETS
    pub bounce_rate: f64,      // 0.0-1.0
    pub bounce_lfo_min: f64,   // 0.01-50.0
    pub bounce_lfo_max: f64,   // 0.01-50.0
    // Output
    pub wet_dry: f64,          // 0.0-1.0
    // Internal
    pub seed: i32,
}

impl Default for LossyParams { /* match Python defaults exactly */ }

impl LossyParams {
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error>;
    pub fn from_json_with_defaults(json: &str) -> Self; // merge sparse preset with defaults
}
```

Key: Use `#[serde(default)]` on each field so sparse preset JSON loads correctly (missing keys get defaults).

#### 4. Bitcrush Module
**File**: `rust/crates/lossy-dsp/src/bitcrush.rs`
**Port of**: `lossy/engine/bitcrush.py` (65 lines)

Two functions:
- `crush(audio: &mut [f64], crush: f64)` — quantize to reduced bit depth
- `decimate(audio: &mut [f64], decimate: f64)` — zero-order hold sample rate reduction
- `crush_and_decimate(audio: &mut [f64], params: &LossyParams)` — entry point, scales by `global_amount`

Algorithm is straightforward: bit depth mapping, mid-tread quantization, phase accumulator for hold.

#### 5. Filters Module
**File**: `rust/crates/lossy-dsp/src/filters.rs`
**Port of**: `lossy/engine/filters.py` (188 lines)

Four components:
- `compute_biquad_coeffs(filter_type, freq, width, slope_idx) -> (BiquadCoeffs, usize)` — coefficient computation
- `apply_filter(audio: &mut [f64], params: &LossyParams)` — cascaded biquad application
- `lofi_reverb(audio: &[f64], params: &LossyParams) -> Vec<f64>` — 4-comb + 1-allpass Schroeder reverb
- `noise_gate(audio: &mut [f64], params: &LossyParams)` — RMS-based ducking gate
- `limiter(audio: &mut [f64], params: &LossyParams)` — peak limiter

Constants to preserve exactly:
- Comb delays: [1031, 1327, 1657, 1973] samples
- Comb damping: 0.45
- Allpass delay: 379 samples, gain: 0.6
- Gate window: 512 samples
- Q mapping: 0.3 to 20.0 log-scaled

#### 6. Packets Module
**File**: `rust/crates/lossy-dsp/src/packets.rs`
**Port of**: `lossy/engine/packets.py` (94 lines)

- `packet_process(audio: &[f64], params: &LossyParams) -> Vec<f64>` — Gilbert-Elliott packet loss/repeat
- Uses `rand_chacha::ChaCha8Rng` seeded from `params.seed + 1000`
- Crossfade with Hann window halves at state transitions
- Constants: `p_g2b = rate * 0.3`, `p_b2g = 0.4`, crossfade 3ms

#### 7. Spectral Module (most complex)
**File**: `rust/crates/lossy-dsp/src/spectral.rs`
**Port of**: `lossy/engine/spectral.py` (297 lines)

This is the core STFT codec emulation engine. Key design decisions:

- Use `realfft` crate — `RealFftPlanner::plan_fft_forward()` produces N/2+1 complex bins, matching `numpy.fft.rfft`
- Use `realfft::RealFftPlanner::plan_fft_inverse()` for IFFT
- Hann window: compute at init, reuse
- Overlap-add accumulation buffer
- Reflection padding at boundaries

Functions:
- `spectral_process(audio: &[f64], params: &LossyParams) -> Vec<f64>` — entry point
- `compute_ath_weights(band_edges, n_bands, n_bins, window_size) -> Vec<f64>` — psychoacoustic weighting
- `standard_degrade(magnitudes, loss, rng, band_edges, ath_weights, quantizer, noise_shape, weighting) -> Vec<f64>` — quantization + band gating
- `shape_delta(magnitudes, base_delta, amount) -> Vec<f64>` — envelope-following noise shaping

Key numerical constants to preserve:
- Compand exponents: 0.75 (compress), 4/3 (expand)
- HF rolloff: `cutoff = n_bins * (1.0 - 0.6 * loss)`
- Band gating probability: 0.6 base
- Noise shape max scale: 4x
- Log-spaced band edges mimicking Bark scale

Freeze state needs to persist across calls for streaming. For the initial non-streaming port, freeze operates within a single `spectral_process` call (matching Python behavior).

#### 8. Signal Chain
**File**: `rust/crates/lossy-dsp/src/chain.rs`
**Port of**: `lossy/engine/lossy.py` (184 lines)

- `render_chain(dry: &[f64], params: &LossyParams) -> Vec<f64>` — core signal chain
- `render_with_bounce(dry: &[f64], params: &LossyParams) -> Vec<f64>` — bounce modulation wrapper
- `render_lossy_mono(dry: &[f64], params: &LossyParams) -> Vec<f64>` — dispatch to chain or bounce

Signal chain order (matching Python exactly):
```
PRE mode:  Verb -> Spectral -> AutoGain -> LossGain -> Crush -> Packets -> Filter -> Gate -> Limiter
POST mode: Spectral -> AutoGain -> LossGain -> Crush -> Packets -> Filter -> Verb -> Gate -> Limiter
```

#### 9. Public API
**File**: `rust/crates/lossy-dsp/src/lib.rs`

```rust
pub mod params;
pub mod spectral;
pub mod bitcrush;
pub mod packets;
pub mod filters;
pub mod chain;

/// Process audio through the lossy engine.
/// Input: mono or interleaved stereo f64 samples.
/// Output: same format as input.
pub fn render_lossy(input: &[f64], params: &LossyParams) -> Vec<f64>;

/// Process stereo audio (2D: samples x channels).
pub fn render_lossy_stereo(left: &[f64], right: &[f64], params: &LossyParams) -> (Vec<f64>, Vec<f64>);
```

#### 10. CLI Test Binary
**File**: `rust/crates/lossy-dsp/src/bin/cli.rs`

```rust
// Usage: lossy-cli input.wav output.wav [preset.json]
// Reads WAV, applies lossy processing, writes WAV.
// If no preset given, uses default params.
```

Uses `hound` crate for WAV I/O.

### Success Criteria:

#### Automated Verification:
- [ ] `cargo build` compiles with no errors: `cd rust && cargo build`
- [ ] `cargo test` passes all unit tests: `cd rust && cargo test`
- [ ] `cargo clippy` passes with no warnings: `cd rust && cargo clippy`
- [ ] CLI binary renders a WAV file: `cd rust && cargo run --bin lossy-cli -- ../test.wav /tmp/out.wav`
- [ ] CLI binary loads a preset: `cd rust && cargo run --bin lossy-cli -- ../test.wav /tmp/out.wav ../lossy/gui/presets/bad_connection.json`
- [ ] All 50+ lossy presets load without error (unit test iterates preset directory)

#### Manual Verification:
- [ ] Load `bad_connection.json` preset, render same WAV in both Python and Rust CLI — listen to both, confirm they sound equivalent
- [ ] Load `codec_32kbps.json` preset — listen, confirm spectral degradation character matches
- [ ] Load a freeze preset (e.g., `frozen_solid.json`) — listen, confirm freeze behavior
- [ ] Load a bounce preset — listen, confirm LFO modulation character
- [ ] Try 3-4 diverse presets from different categories — confirm general sonic character

**Implementation Note**: After completing this phase and all automated verification passes, pause here for manual listening confirmation before proceeding to Phase 2.

---

## Phase 2: pyo3 Python Bindings (Lossy)

### Overview
Expose `render_lossy()` to Python via pyo3/maturin. Create a copied Python GUI that imports from the Rust module instead of the Numba engine. This lets us A/B test Rust vs Python DSP using the familiar GUI.

### Changes Required:

#### 1. Python Binding Crate
**File**: `rust/crates/lossy-python/Cargo.toml`

```toml
[package]
name = "lossy-python"
version = "0.1.0"
edition = "2021"

[lib]
name = "lossy_rust"
crate-type = ["cdylib"]

[dependencies]
lossy-dsp = { path = "../lossy-dsp" }
pyo3 = { version = "0.23", features = ["extension-module"] }
numpy = "0.23"
serde_json = "1"
```

**File**: `rust/crates/lossy-python/src/lib.rs`

```rust
use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1, IntoPyArray};

/// render_lossy(input_audio: np.ndarray, params: dict) -> np.ndarray
/// Drop-in replacement for lossy.engine.lossy.render_lossy
#[pyfunction]
fn render_lossy<'py>(
    py: Python<'py>,
    input_audio: PyReadonlyArray1<'py, f64>,
    params_json: &str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let params = lossy_dsp::params::LossyParams::from_json_with_defaults(params_json);
    let input = input_audio.as_slice()?;
    let output = lossy_dsp::render_lossy(input, &params);
    Ok(output.into_pyarray(py))
}
```

Note: params passed as JSON string from Python for simplicity. The Python wrapper converts the dict to JSON.

#### 2. Maturin Config
**File**: `rust/crates/lossy-python/pyproject.toml`

```toml
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[project]
name = "lossy-rust"
requires-python = ">=3.12"

[tool.maturin]
features = ["pyo3/extension-module"]
```

#### 3. Copied Python GUI
**File**: `rust/py/lossy_gui.py` (copied from `lossy/gui/gui.py` with import changes)

Changes from original:
- Line 37: `from lossy.engine.lossy import render_lossy` -> `from lossy_rust import render_lossy as render_lossy_native`
- Add a thin wrapper that converts params dict to JSON before calling Rust:
  ```python
  def render_lossy(audio, params, **kwargs):
      import json
      return render_lossy_native(audio, json.dumps(params))
  ```
- Preset path: point to `../lossy/gui/presets/` (read from original preset directory)
- All other code identical

#### 4. Python Wrapper Script
**File**: `rust/py/run_lossy.py`

```python
#!/usr/bin/env python
"""Launch lossy GUI backed by Rust DSP engine."""
import sys, os
# Add parent project to path for shared/ imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from lossy_gui import main
main()
```

### Success Criteria:

#### Automated Verification:
- [ ] Maturin builds successfully: `cd rust/crates/lossy-python && maturin develop`
- [ ] Python can import the module: `python -c "import lossy_rust; print('ok')"`
- [ ] Render produces output: `python -c "import lossy_rust; import numpy as np; out = lossy_rust.render_lossy(np.zeros(44100), '{}'); print(len(out))"`

#### Manual Verification:
- [ ] Launch GUI: `cd rust/py && python run_lossy.py`
- [ ] Load a WAV file, apply presets, listen — confirm it works
- [ ] A/B test: render same preset in original Python GUI and Rust GUI, listen to both
- [ ] Try 5+ presets spanning different effect categories

**Implementation Note**: After completing this phase, pause for A/B listening comparison before proceeding.

---

## Phase 3: shared-dsp Crate

### Overview
Build the shared infrastructure needed by both plugins: OSC protocol handler, audio ring buffer for snapshots, running metrics computation, and lock-free parameter smoothing.

### Changes Required:

#### 1. Shared DSP Crate
**File**: `rust/crates/shared-dsp/Cargo.toml`

```toml
[package]
name = "shared-dsp"
version = "0.1.0"
edition = "2021"

[dependencies]
rosc = "0.10"
hound = "3.5"
```

#### 2. Modules

**`src/osc.rs`** — OSC protocol handler
- `OscServer` struct: binds UDP socket, listens for param messages
- `OscClient` struct: sends metrics, snapshot-ready notifications
- Message routing: `/lossy/param/{name}`, `/lossy/preset/load`, `/lossy/request_snapshot`
- Background thread with channel-based message passing to audio thread

**`src/ring_buffer.rs`** — Audio ring buffer
- `RingBuffer<const N: usize>` struct: fixed-size circular buffer
- `write(&mut self, sample: f64)` — write one sample
- `dump_to_wav(&self, path: &str, sr: u32)` — write contents to WAV file
- Size: 3 seconds at 44100 Hz = 132,300 samples per channel

**`src/metrics.rs`** — Running audio metrics
- `RunningMetrics` struct: maintains windowed RMS, peak, spectral centroid
- `push_sample(&mut self, sample: f64)` — update running stats
- `snapshot(&self) -> MetricsSnapshot` — read current values
- Spectral centroid via small rolling FFT (1024-sample window)

**`src/smoothing.rs`** — Parameter smoothing
- `SmoothedParam` struct: current value, target value, ramp rate
- `set_target(&mut self, target: f64)` — set new target (from OSC)
- `next(&mut self) -> f64` — get next smoothed value (called per sample)
- Default ramp: ~10ms (441 samples at 44.1kHz)

### Success Criteria:

#### Automated Verification:
- [ ] `cargo build` compiles: `cd rust && cargo build`
- [ ] `cargo test` passes: `cd rust && cargo test -p shared-dsp`
- [ ] Unit tests: ring buffer write/read, smoothing ramp behavior, OSC encode/decode

#### Manual Verification:
- [ ] None needed for this phase — infrastructure only, tested via plugin in Phase 4

---

## Phase 4: nih-plug Lossy Plugin

### Overview
Wrap the lossy DSP in a nih-plug plugin providing VST3/CLAP/standalone output. Build an egui GUI with parameter controls. Integrate the OSC bridge from shared-dsp for Claude communication.

### Changes Required:

#### 1. Plugin Crate
**File**: `rust/crates/lossy-plugin/Cargo.toml`

```toml
[package]
name = "lossy-plugin"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[dependencies]
lossy-dsp = { path = "../lossy-dsp" }
shared-dsp = { path = "../shared-dsp" }
nih_plug = { git = "https://github.com/robbert-vdh/nih-plug.git", features = ["standalone"] }
nih_plug_egui = { git = "https://github.com/robbert-vdh/nih-plug.git" }
serde_json = "1"
```

#### 2. Plugin Implementation
**File**: `rust/crates/lossy-plugin/src/lib.rs`

- Implement `nih_plug::Plugin` trait for `LossyPlugin` struct
- Map all 42 lossy params to nih-plug `FloatParam`/`IntParam`/`EnumParam` with appropriate ranges
- `process()`: convert nih-plug buffer to f64 slices, call `lossy_dsp::render_lossy()`, write back
- For real-time: process in fixed blocks (e.g., 512 samples) to bound latency
- Important: STFT engine needs state that persists across process calls (overlap-add buffers)
  - This requires refactoring `spectral_process` to accept/return state struct
  - Or: collect input samples into window-sized blocks, process complete windows

**File**: `rust/crates/lossy-plugin/src/gui.rs`
- egui-based GUI with sliders matching the Python GUI layout
- Organized in sections: Spectral, Crush, Packets, Filter, Effects, Output
- Preset browser (reads JSON from embedded or external directory)

**File**: `rust/crates/lossy-plugin/src/presets.rs`
- Load JSON presets into nih-plug parameter state
- Save current state as JSON preset
- Embed a few essential presets in the binary

**File**: `rust/crates/lossy-plugin/src/osc_bridge.rs`
- Spawn background thread running `shared_dsp::osc::OscServer`
- Route incoming OSC param messages to nih-plug params
- Send metrics and snapshot WAVs via `shared_dsp::osc::OscClient`
- Ring buffers for input/output audio capture

#### 3. Build Configuration
**File**: `rust/crates/lossy-plugin/build.rs` (if needed for nih-plug bundling)

Build command: `cargo xtask bundle lossy-plugin --release`

### Important: Real-Time STFT Considerations

The current `spectral_process()` operates on a complete audio buffer. For real-time plugin use, we need a streaming STFT:

1. Accumulate input samples in a ring buffer
2. When a full hop's worth of samples arrives, run one STFT frame
3. Overlap-add the IFFT output to the output buffer
4. Report latency to the DAW: `window_size` samples

This is a significant refactor of the spectral module. The approach:
- Add `SpectralState` struct holding: window, FFT planner, overlap-add buffers, freeze state
- `spectral_process_block(state: &mut SpectralState, input_block: &[f64], output_block: &mut [f64])` — streaming interface
- Keep the original `spectral_process()` as a convenience wrapper for offline use

### Success Criteria:

#### Automated Verification:
- [ ] `cargo xtask bundle lossy-plugin --release` produces VST3 and CLAP bundles
- [ ] Standalone launches: `./target/bundled/lossy-plugin` (or platform equivalent)
- [ ] Unit tests for param mapping: all 42 params round-trip correctly

#### Manual Verification:
- [ ] Load VST3 in a DAW (Logic, Ableton, REAPER, etc.)
- [ ] Plugin processes audio without glitches at 256-sample buffer
- [ ] All preset categories sound correct
- [ ] egui GUI controls work and update audio in real time
- [ ] Standalone mode: load WAV, process, hear output
- [ ] OSC bridge: send `/lossy/param/loss 0.8` from Python, hear change in DAW

**Implementation Note**: This phase has the most unknowns (real-time STFT, nih-plug integration, egui GUI). Pause for thorough manual testing before proceeding.

---

## Phase 5: Reverb DSP + pyo3 + Plugin

### Overview
Port the reverb FDN engine to Rust following the same patterns established in Phases 1-4. This is simpler than lossy — no FFT, just delay lines, filters, and matrix operations.

### Changes Required:

#### 1. Reverb DSP Crate
**File**: `rust/crates/reverb-dsp/Cargo.toml`

Modules:
- `params.rs` — 32+ params struct (port of `reverb/engine/params.py`)
- `matrix.rs` — 7 matrix types + fast apply functions (port of `reverb/primitives/matrix.py`)
- `fdn.rs` — static FDN loop (port of `reverb/engine/numba_fdn.py`)
- `fdn_mod.rs` — modulated FDN loop (port of `reverb/engine/numba_fdn_mod.py`)
- `lfo.rs` — LFO waveforms (sine, triangle, sample-and-hold)
- `lib.rs` — public API: `render_fdn(input, params) -> stereo output`

Key algorithms to port:
- Pre-delay circular buffer
- Input diffusion allpass chain (Schroeder allpass)
- 8-node delay line read/write
- One-pole damping filters
- Matrix multiply (Householder O(N), Hadamard O(N log N), generic O(N^2))
- DC blocker (1-pole highpass at 5 Hz)
- Constant-power stereo panning
- Saturation: `(1-sat)*x + sat*tanh(x)`
- LFO with correlation-based phase distribution
- Fractional delay with linear interpolation
- Matrix blending (two matrices, per-element lerp)

#### 2. Reverb Python Bindings
**File**: `rust/crates/reverb-python/` — same pattern as lossy-python

#### 3. Reverb Plugin
**File**: `rust/crates/reverb-plugin/` — same pattern as lossy-plugin

The reverb is inherently real-time friendly (no STFT), so the plugin is simpler:
- Process per-sample in `process()` callback
- All state (delay buffers, filter states, LFO phases) in plugin struct
- Matrix selection done at param change, not per-sample

#### 4. Copied Reverb GUI
**File**: `rust/py/reverb_gui.py` (copied from `reverb/gui/gui.py` with import changes)

### Success Criteria:

#### Automated Verification:
- [ ] `cargo build` compiles all reverb crates
- [ ] `cargo test -p reverb-dsp` passes all tests
- [ ] CLI binary renders WAV with reverb presets
- [ ] All 60+ reverb presets load without error
- [ ] `maturin develop` for reverb-python succeeds
- [ ] `cargo xtask bundle reverb-plugin --release` produces VST3/CLAP

#### Manual Verification:
- [ ] A/B listen: same preset in Python and Rust — confirm perceptual match
- [ ] Test modulated presets (breathing_space, lush_chorus_room) — confirm modulation character
- [ ] Test edge cases (resonator, controlled_explosion) — confirm extreme param handling
- [ ] Load reverb VST in DAW, process audio, verify quality
- [ ] Standalone mode works

**Implementation Note**: Pause for A/B listening tests before proceeding to Phase 6.

---

## Phase 6: OSC Agentic Loop

### Overview
Wire the Python-side OSC client to the LLM tuner, add mode switching to the copied GUIs, and enable end-to-end Claude-in-the-loop tuning of running VST plugins.

### Changes Required:

#### 1. Python OSC Client
**File**: `rust/py/osc_client.py`

```python
from pythonosc import udp_client, dispatcher, osc_server
import threading

class PluginBridge:
    """Communicates with running VST plugin via OSC."""
    def __init__(self, plugin_host="127.0.0.1", plugin_port=9000, listen_port=9001):
        self.client = udp_client.SimpleUDPClient(plugin_host, plugin_port)
        self.metrics = {}
        self._setup_listener(listen_port)

    def set_param(self, name: str, value: float):
        self.client.send_message(f"/lossy/param/{name}", value)

    def load_preset(self, name: str):
        self.client.send_message("/lossy/preset/load", name)

    def request_snapshot(self):
        self.client.send_message("/lossy/request_snapshot", [])

    def get_metrics(self) -> dict:
        return self.metrics.copy()
```

#### 2. Mode Switch in GUI
**Files**: `rust/py/lossy_gui.py`, `rust/py/reverb_gui.py`

Add toggle at top of GUI:
```
[* Local (Rust/pyo3)]  [  Plugin (OSC) ]
```

- **Local mode**: render via pyo3 (default, same as Phase 2)
- **Plugin mode**: send params via OSC, receive metrics/snapshots

The render function checks the mode and routes accordingly:
```python
if self.mode == "local":
    output = render_lossy(audio, params)
elif self.mode == "plugin":
    self.bridge.set_params(params)
    # wait for metrics update
    metrics = self.bridge.get_metrics()
```

#### 3. LLM Tuner Integration
**File**: `rust/py/llm_bridge.py`

Copied from `shared/llm_tuner.py` with mode-aware transport:
- In local mode: renders via Rust pyo3, computes metrics locally
- In plugin mode: sends params via OSC, receives metrics from plugin, requests spectrograms from plugin snapshots
- The LLM tuner's interface (`send_prompt`, `get_metrics`, `get_spectrogram`) stays identical

### Success Criteria:

#### Automated Verification:
- [ ] OSC round-trip test: Python sends param, plugin echoes back, values match
- [ ] Python `PluginBridge` can connect and send messages without error

#### Manual Verification:
- [ ] Full agentic loop: launch VST in DAW, open Python GUI in plugin mode, type "make it darker" — hear change in DAW
- [ ] Request snapshot: get spectrogram from plugin audio, visible in GUI
- [ ] Mode switch: toggle between local and plugin mode, both work
- [ ] Multiple iterations: Claude makes 3-5 autonomous adjustments, each produces audible change

**Implementation Note**: This is the final integration phase. Test the full loop end-to-end.

---

## Testing Strategy

### Unit Tests (per crate):
- **lossy-dsp**: Each module tested independently
  - `params`: default construction, JSON loading, sparse preset merge, field clamping
  - `bitcrush`: crush at 0 = passthrough, crush at 1 = heavy quantization, decimate hold
  - `filters`: biquad at known frequency, lofi_reverb produces longer tail, gate silences quiet
  - `packets`: mode 0 = passthrough, mode 1 produces gaps, mode 2 produces repeats
  - `spectral`: loss=0 = passthrough, loss=1 = heavy degradation, window size affects resolution
  - `chain`: full chain with default params, verb position routing
- **shared-dsp**: ring buffer, smoothing, OSC encode/decode
- **reverb-dsp**: matrix types (verify unitarity), FDN energy decay, modulation sweep

### Integration Tests:
- Load every preset JSON file, render 1 second of noise, verify output is finite and non-silent
- Render same input with same params in Python and Rust, compute correlation coefficient (should be > 0.9)

### Manual Testing (pause points):
1. After Phase 1: Listen to Rust CLI output vs Python output
2. After Phase 2: A/B in GUI — Rust pyo3 vs Python Numba
3. After Phase 4: Plugin in DAW — real-time processing quality
4. After Phase 5: Reverb A/B listening
5. After Phase 6: Full agentic loop with Claude

## Performance Considerations

- **STFT real-time**: The lossy spectral engine at window_size=2048, hop_divisor=4 requires one FFT frame every 512 samples (~11.6ms at 44.1kHz). `realfft` should easily handle this.
- **Reverb real-time**: 8-node FDN is trivially real-time. Main cost is per-sample matrix multiply (64 multiply-adds for 8x8).
- **Memory**: Plugin needs delay buffers (reverb: ~26KB for 8 lines at max delay), STFT buffers (lossy: ~128KB for window=16384), ring buffers (shared: ~1MB for 3s stereo). Trivial.
- **f64 vs f32**: Internal processing in f64 for quality, convert to/from f32 at plugin boundary (DAW buffers are f32).

## Migration Notes

- **No migration needed** — all new code, existing code untouched
- Presets: read directly from `lossy/gui/presets/` and `reverb/gui/presets/`
- Python GUI copies in `rust/py/` are standalone — can run independently
- If user wants to switch to Rust DSP permanently, change one import line in original GUI (but we're not doing that per requirements)

## References

- Architecture research: `thoughts/shared/research/2026-02-07-architecture-analysis-and-vst-plan.md`
- nih-plug: `https://github.com/robbert-vdh/nih-plug` (git dependency, Rust 1.80+)
- pyo3: `https://pyo3.rs/` (v0.23+)
- realfft: `https://docs.rs/realfft/` (v3.5, matches numpy.fft.rfft format)
- rosc: `https://docs.rs/rosc/` (v0.10+, OSC 1.0 protocol)
- maturin: `https://www.maturin.rs/` (Python extension build tool)
