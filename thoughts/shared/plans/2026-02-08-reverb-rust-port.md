# Reverb Rust Port — Deferred Plan

## Overview

Port the reverb FDN engine to Rust following the same patterns established in the lossy port. This is simpler than lossy — no FFT, just delay lines, filters, and matrix operations. The reverb is inherently real-time friendly (per-sample processing, no STFT).

## Learnings from Lossy Port (apply these)

### Serde / JSON Preset Loading
- Use `#[serde(default, deny_unknown_fields)]` on the params struct — strict parsing, no silent fallbacks
- Add custom `as_i32` deserializer on all integer fields — GUI sliders emit floats like `4096.0`
- Strip GUI-only keys (`_meta`, `tail_length`) before passing JSON to Rust
- `from_json()` returns `Result`, `from_json_with_defaults()` panics on error (strict)
- In pyo3 bindings: use `from_json()` + `map_err(PyValueError)`, never `from_json_with_defaults`

### nih-plug Plugin Structure
- Collect-and-process strategy for STFT-based plugins (lossy uses 8192-sample blocks)
- **Reverb doesn't need this** — it's per-sample, so process directly in the `process()` callback
- Report latency via `context.set_latency_samples()` in `initialize()`, not as a trait method
- `AUDIO_IO_LAYOUTS`: include both stereo and mono-input->stereo-output layouts
- `crate-type = ["cdylib", "lib"]` — lib needed for standalone binary import
- Standalone binary: `nih_plug::nih_export_standalone::<Plugin>()`

### egui GUI
- `create_egui_editor` takes a state type — use it to hold loaded presets + selection
- Lazy-load presets on first frame (check `state.loaded` flag)
- Filesystem presets first, fall back to `load_embedded_presets()`
- ComboBox with category headers and separators for preset browser
- `ParamSlider::for_param()` for all param controls, organized in collapsible sections

### Preset System
- `find_preset_dir()`: walk up from exe, check cwd-relative paths, env var fallback
- `load_presets()`: read JSON, extract `_meta` for category/description, sort by category then name
- `apply_preset()`: use `begin_set_parameter` / `set_parameter` / `end_set_parameter` pattern
- `set_enum`: use `T::from_index(index as usize)` for enum params
- Embed ~8 representative presets via `include_str!()` as compile-time fallback

### OSC Bridge
- `/reverb/` namespace (not `/lossy/`)
- `OscServer::start(port)` in plugin init or editor init
- Poll `server.drain()` from editor thread, not audio thread
- `OscCommand::SetParam` → update nih-plug params via setter
- `OscCommand::LoadPreset` → load from preset manager
- `OscCommand::RequestSnapshot` → save current audio buffer to WAV
- `OscClient` sends metrics (RMS/peak) and snapshot-ready notifications

### Build & Bundling
- `cargo xtask bundle reverb-plugin --release` for VST3/CLAP
- xtask needs `anyhow` dependency
- `rosc` v0.11 — `decode_udp` returns `(&[u8], OscPacket)` (remaining bytes first)
- maturin: `maturin develop --release` for pyo3 bindings (debug mode is ~50x slower)

### pyo3 Bindings
- Accept `params_json: &str` (Python wrapper converts dict to JSON string)
- Strip `_GUI_ONLY_KEYS` in Python before JSON serialization
- Use `PyValueError` for error reporting, not Rust panics
- numpy crate: `PyReadonlyArray1` for input, `into_pyarray()` for output

## Reverb-Specific Architecture

### DSP Modules to Port

| Module | Python Source | Lines | Complexity |
|--------|-------------|-------|------------|
| params | `reverb/engine/params.py` | ~80 | Low — flat struct |
| matrix | `reverb/primitives/matrix.py` | ~120 | Medium — 7 matrix types |
| fdn | `reverb/engine/numba_fdn.py` | ~350 | Medium — core delay network |
| fdn_mod | `reverb/engine/numba_fdn_mod.py` | ~250 | Medium — modulated variant |
| lfo | `reverb/engine/lfo.py` | ~80 | Low — waveform generators |
| dsp primitives | `reverb/primitives/dsp.py` | ~300 | Low — standalone Numba funcs |

### Key Algorithms
- Pre-delay circular buffer
- Input diffusion allpass chain (Schroeder allpass)
- 8-node delay line read/write with fractional delay (linear interpolation)
- One-pole damping filters per node
- Matrix multiply: Householder O(N), Hadamard O(N log N), generic O(N^2)
- DC blocker (1-pole highpass at 5 Hz)
- Constant-power stereo panning
- Saturation: `(1-sat)*x + sat*tanh(x)`
- LFO with correlation-based phase distribution
- Matrix blending (two matrices, per-element lerp)
- RMS loudness limiter (target RMS 0.2)

### Real-Time Considerations
- Reverb is per-sample — no STFT, no block accumulation needed
- All state (delay buffers, filter states, LFO phases) lives in plugin struct
- Matrix selection done at param change, not per-sample
- Modulation rates are sub-audio — LFO update can be per-block (e.g., every 32 samples)

### Crate Structure
```
rust/crates/
  reverb-dsp/        Pure Rust reverb DSP
    src/
      lib.rs         Public API: render_fdn(input, params) -> stereo output
      params.rs      32+ params struct
      matrix.rs      7 matrix types + fast apply
      fdn.rs         Static FDN loop
      fdn_mod.rs     Modulated FDN loop
      lfo.rs         LFO waveforms
      primitives.rs  Delay line, filters, allpass, etc.
  reverb-python/     pyo3 bindings
  reverb-plugin/     nih-plug VST3/CLAP/standalone
    src/
      lib.rs         Plugin trait impl (per-sample processing)
      params.rs      nih-plug param declarations
      gui.rs         egui GUI
      presets.rs     Preset loading (filesystem + embedded)
      osc_bridge.rs  OSC communication
```

### Verification
- A/B with Python: same preset, same input WAV, correlation > 0.99
- All 60+ reverb presets load without error
- Test modulated presets (breathing_space, lush_chorus_room)
- Test edge cases (resonator, controlled_explosion)
- Plugin loads in DAW, processes audio, no glitches at 256-sample buffer

## Estimated Effort
- reverb-dsp: ~1200 lines Rust (vs ~1200 lines Python/Numba)
- reverb-python: ~80 lines (same pattern as lossy-python)
- reverb-plugin: ~800 lines (simpler than lossy — no block accumulation)
- Total: ~2100 lines new Rust code
