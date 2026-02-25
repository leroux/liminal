# Reverb Project

Three audio effect plugins — **Reverb**, **Lossy**, and **Fractal** — each available as:

- **VST3 / CLAP** plugins for any DAW
- **Standalone** desktop apps (no DAW required)
- **Python GUI** apps with full parameter control, waveform/spectrogram visualization, and AI-assisted sound design

## Plugins

### Reverb
8-node Feedback Delay Network algorithmic reverb with per-node delay times, damping, panning, diffusion allpasses, matrix feedback, and LFO modulation. 60+ presets ranging from natural rooms to infinite drones.

### Lossy
Spectral codec emulator inspired by the Chase Bliss / Goodhertz Lossy pedal. Decomposes audio via STFT and selectively degrades spectral content to recreate MP3/codec artifacts — phase loss, spectral jitter, packet loss, and freeze effects.

### Fractal
Audio fractalization effect that resamples audio at multiple scales with iterative feedback, creating layered textures from simple inputs. Per-scale gain control, spectral processing, and cross-block feedback.

---

## Installation

### VST3 / CLAP Plugins

**Download** pre-built binaries from the [Releases](../../releases) page for your platform:
- macOS (Apple Silicon, Intel, or Universal)
- Windows x86_64
- Linux x86_64

**Install** by copying to your system plugin directory:

| Platform | VST3 | CLAP |
|----------|------|------|
| macOS | `~/Library/Audio/Plug-Ins/VST3/` | `~/Library/Audio/Plug-Ins/CLAP/` |
| Windows | `C:\Program Files\Common Files\VST3\` | `C:\Program Files\Common Files\CLAP\` |
| Linux | `~/.vst3/` | `~/.clap/` |

Restart your DAW after installing.

### Build from Source (VST3 / CLAP / Standalone)

Requires [Rust](https://rustup.rs/) (stable toolchain).

```bash
cd rust

# Build all three plugins (VST3 + CLAP bundles)
make all-plugins

# Install to system plugin directories (macOS)
make install-all

# Or build/install individually
make reverb && make install-reverb
make release && make install           # lossy
make fractal && make install-fractal

# Run standalone (no DAW)
make run-reverb
make run            # lossy
make run-fractal
```

Output bundles are placed in `rust/target/bundled/`.

### Python GUI Apps

The Python GUIs use Rust DSP via pre-built wheels. Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

**Option A: Install pre-built wheels from Releases**

Download the `.whl` files for your platform from the [Releases](../../releases) page, then:

```bash
# Install project dependencies
uv sync

# Install pre-built Rust DSP wheels
uv pip install reverb_rust-*.whl lossy_rust-*.whl fractal_rust-*.whl

# Launch
uv run python -m reverb.main
uv run python -m lossy.main
uv run python -m fractal.main
```

**Option B: Build Rust DSP from source**

```bash
# Install project dependencies
uv sync

# Build PyO3 bindings (requires Rust + maturin)
cd rust
uv run maturin develop -m crates/reverb-python/Cargo.toml --release
uv run maturin develop -m crates/lossy-python/Cargo.toml --release
uv run maturin develop -m crates/fractal-python/Cargo.toml --release
cd ..

# Launch
uv run python -m reverb.main
uv run python -m lossy.main
uv run python -m fractal.main
```

---

## Python GUI Features

All three Python GUIs share:

- **Parameter controls** — sliders organized by section, with per-param and per-section locking during randomization
- **Preset browser** — load/save JSON presets, randomize, reset to defaults
- **Waveform display** — input/output waveform comparison
- **Spectrogram** — time-frequency visualization of output
- **Spectrum analyzer** — frequency domain comparison
- **Audio I/O** — load/save WAV files, real-time playback with device selection, dry/wet preview
- **Generation history** — undo/redo through up to 50 renders
- **AI sound designer** — Claude-powered parameter tuning via natural language ("make it sound like a cathedral", "more metallic")

---

## Presets

Each plugin ships with curated presets in `<name>/gui/presets/`. These are JSON files that map directly to DSP parameters. Presets are embedded into VST3/CLAP builds at compile time and also loaded from the filesystem at runtime when available.

Plugin preset counts: Reverb (60+), Lossy (30+), Fractal (20+).
