# Plugin Guidelines

Conventions for building new audio effect plugins. Refer to lossy and fractal as reference implementations.

## Three-Layer Architecture

Every plugin has three layers:

1. **Python package** (`<name>/`) — params, GUI, presets, CLI renderer, entry point via `<name>/main.py`
2. **Rust DSP crate** (`rust/crates/<name>-dsp/`) — pure algorithm, no plugin framework deps
3. **Rust plugin crate** (`rust/crates/<name>-plugin/`) — nih-plug VST3/CLAP wrapper + egui GUI

Plus a **PyO3 bindings crate** (`rust/crates/<name>-python/`) bridging Rust DSP to Python.

All three plugins (reverb, lossy, fractal) follow this pattern.

## Key Conventions

- **Stereo always.** All plugins declare stereo-in/stereo-out + mono-in/stereo-out layouts. Mono input duplicated at sample-collection time.
- **Sample rate:** 44100 Hz everywhere.
- **Block processing:** Plugins accumulate host samples into `input_buf`, process in `PROCESS_BLOCK`-sized chunks (8192), report that as latency.
- **Params as JSON:** Python serializes params dict to JSON; Rust deserializes via `#[serde(default)]` (sparse JSON OK, missing keys get defaults). Use `as_i32` deserializer for integer fields.
- **Signal chain order:** Effect-specific processing -> Output Gain -> Crush/Decimate -> Post-Filter -> Gate -> Limiter -> Wet/Dry Mix -> Output.
- **Presets:** JSON files in `<name>/gui/presets/`, embedded at compile time via `build.rs` + `include_str!()`. Plugin tries filesystem first, falls back to embedded.

## Rust Crate Patterns

- **DSP crate** exposes `render_<name>(input, params)` and `render_<name>_stereo(left, right, params)`. Params struct mirrors Python `params.py` field-for-field.
- **PyO3 crate** is a thin cdylib: numpy arrays in, numpy out, JSON params string. Module named `<name>_rust`. Built with maturin.
- **Plugin crate** uses `crate-type = ["cdylib", "lib"]` + `[[bin]]` for standalone. Implements `to_dsp_params()` to convert nih-plug params to DSP struct.
- **shared-dsp crate** provides `SmoothedParam`, `RingBuffer`, `RunningMetrics`, and OSC server/client.
- **Vendored deps:** nih-plug and baseview at `rust/patches/`, redirected via `[patch]` in workspace `Cargo.toml`.

## Python GUI Patterns

- **tkinter** with `ttk.Notebook` tabs: Parameters, Presets, Waveforms, Spectrograms, Spectrum, Guide
- **Top bar:** Load/Save WAV, Play/Dry/Stop, Randomize/Reset, Gen history `<`/`>`, output device
- **Section locks:** params and sections lockable to survive randomization
- **Safety pipeline:** `safety_check()` rejects NaN/inf/peak>1e6, `normalize_output()` applies RMS limiter
- **Generation history:** up to 50 rendered outputs with undo/redo
- **LLM tuner:** `shared/llm_tuner.py` bridges Claude Agent SDK with tkinter. Add `<NAME>_GUIDE` and `<NAME>_PARAM_DESCRIPTIONS` to `shared/llm_guide_text.py`.

## Plugin GUI (egui)

- Each plugin gets a distinct color theme (3 colors: `BG_DARK`, `PRIMARY`, `ACCENT`)
- Monospace font, zero corner radius, scanline overlay
- Two-column layout with collapsing sections
- Preset browser with ComboBox + prev/next

## Build Workflow

```bash
# PyO3 bindings (for Python GUI)
cd rust/crates/<name>-python && maturin develop --release

# Python GUI
uv run python -m <name>.main

# VST plugin build + install
cd rust && make <name> && make install

# Standalone (no DAW)
cargo run --bin <name>-standalone --release
```

## New Plugin Checklist

1. Rust DSP crate — params struct, signal chain, core algorithm
2. PyO3 bindings — mono + stereo render, maturin pyproject.toml
3. Python engine wrapper — `render_<name>()` calling Rust via JSON
4. Python params — `default_params()`, `PARAM_RANGES`, `PARAM_SECTIONS`
5. Python GUI — standard tabs, top bar, section locks
6. Presets — initial collection in `<name>/gui/presets/`
7. LLM guide text — add to `shared/llm_guide_text.py`
8. nih-plug plugin — block buffering, params, egui GUI, preset embedding
9. Standalone binary — `[[bin]]` target in plugin Cargo.toml
10. Registration — `pyproject.toml` (package + console script), `rust/Cargo.toml` (workspace members), `rust/Makefile` (build target)
