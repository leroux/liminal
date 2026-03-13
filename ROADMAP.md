# Reverb v1.0 Release Roadmap

Three mature audio effect plugins — **Reverb**, **Lossy**, **Fractal** — shipping as a clean, Rust-only project with Vizia-based VST3/CLAP plugins and standalone apps.

Key decisions:
- **License**: MIT
- **AI chat**: `claudewire` (Claude CLI protocol) + `sound-designer` (AI tuner)
- **GUI**: Vizia (unified for plugins + standalone apps)
- **Python**: Deleted entirely after port
- **Chordspace**: Moved to own branch, removed from main

---

## Phase 0: Cleanup & Housekeeping

- [x] Move `chordspace/` to its own branch, delete from main
- [x] Delete untracked root-level `claudewire/` (orphaned copy; real one is `rust/crates/claudewire/`)
- [x] Add MIT `LICENSE` file at repo root
- [x] Uncomment Vizia app crates in `rust/Cargo.toml` workspace members
- [ ] Delete duplicate/unnecessary files (research docs, planning files in subdirs)
- [ ] Commit or discard pending `shared/audio.py` change

## Phase 1: sound-designer — Reusable AI Sound Design Library

New `rust/crates/sound-designer/` crate, built on top of `claudewire` (Claude CLI protocol):

- [x] Audio metrics formatting for LLM context (`metrics.rs` — port of `shared/audio_features.py`)
- [x] Parameter schema, validation & clamping (`params.rs` — port of `shared/params.py`)
- [x] AI tuner with streaming text, JSON param extraction, and merge (`tuner.rs` — port of `shared/llm_tuner.py`)
- [x] Autonomous iteration loop (up to N silent renders with `_iterate` protocol)
- [x] Session management (reset, undo)
- [x] System prompt construction with guide text + rules
- [ ] Spectrogram/waveform image generation for multimodal context
- [ ] Integration tests with mock Claude responses

**Goal**: `sound-designer` is a self-contained crate any Rust audio app can depend on for AI-assisted parameter tuning. `claudewire` remains the low-level Claude CLI transport.

## Phase 2: Complete Vizia Plugin GUIs

Port all Python GUI features into the Vizia-based plugin and standalone app GUIs.

### shared-gui (common views)

- [ ] Finish `chat_panel.rs` — wire to claudewire backend (currently trait-only)
- [ ] Add playback cursor (40ms tick, synced across waveform + spectrogram views)
- [ ] Add click-to-seek on waveform/spectrogram
- [ ] Add per-param and per-section lock system for randomization
- [ ] Add preset save dialog (name, category, description)
- [ ] Add preset migration support for legacy formats
- [ ] Add generation history navigation (restore full state)
- [ ] Add autoplay debounce on param change
- [ ] Add tail-length rendering (append silence for decay capture)
- [ ] Add zoom system (font + layout scaling)

### reverb-app / reverb-plugin

- [ ] Verify params tab slider coverage is complete
- [ ] Add 8x8 interactive matrix heatmap editor (click, drag, snap unitary, randomize)
- [ ] Add XY pad (2D parameter controller)
- [ ] Add Signal Flow visualization tab

### lossy-app / lossy-plugin

- [ ] Port full ~38-param UI (spectral loss, crush, packets, filter, effects, freeze, bounce)
- [ ] Add spectrum extra tab (log-frequency dry/wet FFT overlay)
- [ ] Add color-coded guide tab

### fractal-app / fractal-plugin

- [ ] Port full ~35-param UI (fractal core, iteration, spectral, filter, layers, bounce)
- [ ] Add spectrum extra tab
- [ ] Add guide tab

## Phase 3: Finish Explore Project

- [ ] Fix ID naming inconsistency (standardize all to lowercase descriptive: `b001_schroeder_reverb`)
- [ ] Fix category string casing inconsistency
- [ ] Implement `--timeout` flag (currently declared but not enforced)
- [ ] Add JSON manifest output (which effects succeeded/failed, params used, output paths)
- [ ] Add summary HTML browser for exploring output WAVs
- [ ] Write `rust/crates/explore-dsp/README.md`
- [ ] Write `rust/crates/explore-cli/README.md`

## Phase 4: Delete Python & Restructure

- [ ] Delete `reverb/`, `lossy/`, `fractal/`, `shared/` Python packages
- [ ] Delete `pyproject.toml`, `uv.lock`, `.python-version`, `.venv` references
- [ ] Delete `tests/` Python test directory
- [ ] Delete `rust/crates/reverb-python/`, `lossy-python/`, `fractal-python/` PyO3 binding crates
- [ ] Delete `.github/workflows/build-wheels.yml`
- [ ] Remove Python-related entries from `.gitignore`
- [ ] Move presets from `<name>/gui/presets/` to `rust/crates/<name>-plugin/presets/` (or shared location)
- [ ] Update workspace `Cargo.toml` — remove python crate members

## Phase 5: Release Prep

- [ ] Set version to `1.0.0` across all Rust crates
- [ ] Update `README.md` — remove all Python references, document Rust-only install/build
- [ ] Write `CHANGELOG.md`
- [ ] Add `cargo test --workspace` to CI workflow
- [ ] Add standalone app builds to CI (Vizia apps)
- [ ] Add explore-cli build to CI (optional)
- [ ] Tag `v1.0.0`, verify CI builds and GitHub Release creation
- [ ] Publish release

---

## Verification Criteria

- `cargo build --workspace` compiles all crates including Vizia apps
- `cargo test --workspace` passes
- `cargo xtask bundle <plugin> --release` produces VST3/CLAP bundles
- `cargo run -p reverb-app --release` launches with full GUI features
- `cargo run -p explore-cli -- --help` shows usage
- CI workflow succeeds on push
- `v1.0.0` tag triggers release with all artifacts
