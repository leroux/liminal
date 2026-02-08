# Architecture Analysis & VST Migration Plan

**Date:** 2026-02-07
**Status:** Research complete, pending sequencing decision

---

## Codebase Analysis Summary

### Current State

| Layer | Files | Lines | Health |
|-------|-------|-------|--------|
| DSP Engine (`reverb/engine/`) | 5 | ~1,080 | Excellent - clean, no mixed concerns |
| Primitives (`reverb/primitives/`) | 2 | ~300 | Excellent - zero coupling |
| GUI (`reverb/gui/gui.py`) | 1 | ~1,900 | Monolithic but functional (waveform extracted) |
| Shared (`shared/`) | 6 | ~1,080 | Good - growing utility layer |
| Lossy (`lossy/`) | 9 | ~2,500 | Good - mirrors reverb structure |
| Presets | 50+ JSON | N/A | Good - backward-compatible |
| Audio Analysis | 1 | ~270 | New - comprehensive metrics |
| Audio Features | 1 | ~175 | New - LLM formatting + spectrogram |
| Shared Waveform | 1 | ~224 | New - extracted from both GUIs |

### What's Already Clean
- **DSP engine** has zero GUI/IO dependencies
- **Parameter dict** is the single interface between all layers
- **No circular dependencies** anywhere
- **Preset format** is backward-compatible with default merging
- **Numba JIT** god functions are justified by performance constraints
- **Lazy imports** in `fdn.py` avoid loading unused engine code
- **Matrix registry pattern** is clean and extensible
- **Audio analysis pipeline** now exists (`shared/analysis.py`) with comprehensive metrics
- **Spectrogram generation** implemented (`shared/audio_features.py`) for LLM vision
- **Waveform rendering** extracted to shared module (`shared/waveform.py`)
- **LLM tuner** now accepts metrics + spectrogram images (multimodal)

### Known Issues (not urgent)
- GUI still monolithic (~1,900 lines) but waveform code now extracted to shared
- No audio abstraction layer (audio I/O embedded in GUI) — stays in Python
- N=8 hardcoded in 3 files
- Code duplication between `numba_fdn.py` and `numba_fdn_mod.py` (accepted for Numba JIT)
- Single `pyproject.toml` bundles reverb + lossy + shared

### Recent Changes (2026-02-07, by another agent)

**New shared modules:**
- `shared/analysis.py` (270 lines) — comprehensive audio metrics: RT60 (T30 method),
  EDT, spectral centroid, echo density, C50/C80 clarity, crest factor, octave-band RT60,
  spectral flatness, bandwidth. Plus dry-vs-wet comparison: centroid shift, energy ratio,
  THD+N, bandwidth change. Uses Schroeder backward integration for decay analysis.
- `shared/audio_features.py` (175 lines) — `format_features()` formats metrics as readable
  text with A/B delta tracking (shows "brighter", "longer tail", etc.) for LLM consumption.
  `generate_spectrogram_png()` creates matplotlib spectrograms as PNG bytes for multimodal
  LLM input.
- `shared/waveform.py` (224 lines) — shared waveform canvas drawing extracted from both
  GUIs. Supports stereo, RMS overlay, amplitude/time grids, metrics display rows.

**LLM tuner updates (`shared/llm_tuner.py`):**
- `send_prompt()` now accepts `metrics`, `source_metrics`, `spectrogram_png`,
  `source_spectrogram_png` kwargs
- Builds multimodal content blocks (text + base64 PNG images) when spectrograms provided
- Tracks `_prev_metrics` for A/B delta between renders
- Tracks `_source_sent` to avoid re-sending input spectrogram every turn
- `reset_session()` clears metric/source tracking state

**Guide text updates (`shared/llm_guide_text.py`):**
- Added AUDIO METRICS reference section to both REVERB_GUIDE and LOSSY_GUIDE
- Documents what each metric means and typical ranges (so Claude can interpret them)

**GUI updates (both pedals):**
- Split single "Waveform" tab into "Input Waveform" and "Output Waveform" tabs
- Both use `shared/waveform.draw_waveform()` instead of inline drawing code
- `_analyze_source()` computes source metrics + spectrogram on WAV load
- Render pipeline now computes `rendered_metrics` and `rendered_spectrogram`
- All metrics/spectrograms passed to LLM tuner on each Claude request

**Dependency additions:**
- `matplotlib>=3.7` added to `pyproject.toml` (for spectrogram generation)

### Dependency Map

| Dependency | Used By | VST Compatible | Replacement |
|---|---|---|---|
| numpy | DSP core, analysis | No | Rust native arrays |
| numba | DSP core | No | Rust native (no JIT needed) |
| scipy | Analysis, WAV I/O | No | Rust DSP / hound crate / DAW provides |
| matplotlib | Spectrogram gen | No | Stays in Python (dev tooling) |
| tkinter | GUI | No | nih-plug egui (or stays in Python) |
| sounddevice | Audio I/O | No | DAW provides audio |
| claude_agent_sdk | LLM tuner | No | Stays in Python (pyo3 bridge) |

---

## Target Architecture

```
┌─────────────────────────┐
│   DSP Core (Rust)       │  ← single source of truth
│   render_fdn()          │
│   shared library        │
└──────┬──────┬───────────┘
       │      │
       ▼      ▼
  Python       nih-plug
  (pyo3)       (native Rust)
  ┌────────┐   ┌──────────────┐
  │ Claude │   │ Plugin GUI   │
  │ Code   │   │ Presets      │
  │ SDK    │   │ VST3 + CLAP  │
  │ Tuner  │   │ + Standalone │
  │ Agentic│   │ WAV loading  │
  │ Loop   │   │              │
  │ tkinter│   │              │
  └────────┘   └──────────────┘
  (dev/exploration)  (shipping/DAW use)
```

### Key Design Decisions

1. **DSP in Rust** — one implementation, called by both Python and plugin
2. **nih-plug** for VST3/CLAP/standalone — supports all three from same code
3. **Python stays alive** for Claude Code SDK agentic workflow
4. **pyo3** binds Rust DSP to Python — replaces Numba, same `render_fdn` interface
5. **JSON presets shared** between both sides — no format conversion
6. **Standalone mode** (nih-plug built-in) handles WAV loading + audio device selection
7. **OSC bridge** — live communication between running VST and Python/Claude for real-time agentic tuning

### Why This Architecture

- Claude Code SDK is Python/TypeScript only — can't run in Rust
- Need to hear results immediately during agentic tuning — Python calls Rust DSP via pyo3
- Want VST for DAW use — nih-plug provides VST3 + CLAP + standalone
- Don't want two DSP implementations — Rust is the single source
- Presets must flow both directions — JSON works for both sides
- Want Claude to tune reverb on real mix content, not just test signals

---

## Live OSC Bridge: Plugin <-> Claude

### Overview

The VST plugin and Python/Claude communicate in real-time via OSC (Open Sound Control).
This enables Claude to tune reverb parameters while the plugin is processing live audio
in the DAW — Claude hears (via metrics/spectrograms) how the reverb responds to the
actual mix content.

### Protocol

- **Transport:** UDP (standard OSC)
- **Python library:** `python-osc`
- **Plugin side:** background thread listening on UDP socket
- **Default ports:** Plugin listens on 9000, Python listens on 9001 (configurable)

### Parameter Control: Python → Plugin

Claude sets parameters via OSC messages. The plugin applies them with smoothing (~10ms
ramp) to avoid clicks.

```
Python sends:
  /reverb/param/feedback_gain  1.4
  /reverb/param/damping_coeffs 0.3 0.3 0.4 0.5 0.3 0.4 0.5 0.3
  /reverb/param/wet_dry        0.6
  /reverb/param/pre_delay      882
  /reverb/preset/load          "dark_cathedral"

Plugin receives → updates params with smoothing → you hear change immediately in DAW
```

### Audio Snapshots: Plugin → Python

The plugin maintains two ring buffers:
1. **Input ring buffer** — raw audio coming into the plugin (dry signal)
2. **Output ring buffer** — processed audio leaving the plugin (wet signal)

On request, the plugin dumps these to temp files for Python to analyze.

```
Python sends:
  /reverb/request_snapshot          (request both input + output)

Plugin responds:
  - Writes /tmp/reverb_input.wav    (last 2-3s of dry input)
  - Writes /tmp/reverb_output.wav   (last 2-3s of wet output)
  - Sends OSC: /reverb/snapshot_ready  "/tmp/reverb_input.wav" "/tmp/reverb_output.wav"

Python receives → generates spectrograms → sends to Claude as images
```

Claude can compare dry vs wet spectrograms to see exactly what the reverb is adding,
isolate the reverb tail, identify frequency buildup, etc.

### Live Metrics: Plugin → Python

The plugin computes running metrics on its audio thread and sends them periodically.

```
Plugin sends (every ~200ms):
  /reverb/metrics/rms              0.15
  /reverb/metrics/peak             0.6
  /reverb/metrics/spectral_centroid 1400.0
  /reverb/metrics/rt60_estimate    2.3
  /reverb/metrics/crest_factor     4.2
```

Claude reads these continuously to understand the effect of parameter changes
without needing a full spectrogram every time.

### Full Agentic Loop (Live in DAW)

```
Claude Code SDK (Python)
  │
  ├─ "make the reverb darker and longer"
  ├─ reads current metrics via OSC
  ├─ sends param changes via OSC
  │     /reverb/param/feedback_gain 1.6
  │     /reverb/param/damping_coeffs 0.3 0.3 ...
  │
  │   ── you hear the change in your DAW mix ──
  │
  ├─ waits ~500ms for params to settle
  ├─ reads updated metrics
  │     rms=0.18, centroid=1200Hz, rt60=3.1s
  ├─ requests snapshot
  │     gets input + output spectrograms
  ├─ analyzes: "rt60 is 3.1s good, but centroid
  │   dropped to 1200Hz — buildup around 300Hz,
  │   let me adjust damping on lower nodes"
  ├─ sends refined param changes via OSC
  │
  │   ── you hear the refinement ──
  │
  ├─ iterates until satisfied
  ├─ sends /reverb/preset/save "warm_cathedral_v2"
  └─ done
```

### Advantages Over Offline Rendering

- Claude tunes for how reverb sits **in context of the full mix**
- Real-time feedback — no render-then-play latency
- Can tune for specific content (vocals vs drums vs synths)
- Dry/wet comparison spectrograms show exactly what the reverb adds
- Metrics update continuously — Claude sees the effect of every change

### Implementation Notes

- **Parameter smoothing:** nih-plug has built-in `Smoother` types — use ~10ms ramp
  to avoid clicks when Claude changes params rapidly
- **Ring buffer size:** 3 seconds at 44100Hz = 132,300 samples per channel per buffer
  (~1MB for stereo pair). Trivial memory cost.
- **Thread safety:** OSC listener runs in background thread, writes to atomic param
  storage. Audio thread reads atomically. Standard lock-free plugin pattern.
- **Metrics computation:** RMS and peak are near-free. Spectral centroid needs a small
  FFT per metrics interval (~1024 samples). RT60 estimate is more complex — could use
  Schroeder backwards integration on the output buffer.

---

### Workflow: Preset Discovery (Updated)

Two modes of operation, both sharing presets:

**Mode 1: Offline (Python-only, no DAW needed)**
```
Claude Code SDK
  → set params
  → call Rust DSP via pyo3 (render test WAV)
  → compute metrics via shared/analysis.py (RT60, EDT, centroid, density, C50/C80, etc.)
  → generate spectrogram via shared/audio_features.py (matplotlib PNG)
  → Claude reads metrics text + sees spectrogram image (multimodal)
  → Claude adjusts params (with A/B delta from previous render)
  → iterate
  → save preset (JSON)
```
NOTE: The metrics + spectrogram + multimodal LLM pipeline is already implemented
in the current Python codebase. The offline agentic loop just needs the autonomous
iteration wrapper (currently human-in-the-loop via GUI text input).

**Mode 2: Live (plugin running in DAW)**
```
Claude Code SDK
  → send params via OSC to running plugin
  → you hear change in mix immediately
  → plugin sends back metrics + audio snapshots
  → Claude analyzes spectrograms in context
  → iterate
  → save preset via OSC
```

Both modes produce identical JSON presets. Discover offline, refine live, or vice versa.

### Python GUI Mode Switch

The Python GUI (for both reverb and lossy) has a mutually exclusive mode toggle:

```
[● Local (pyo3)]  [○ Plugin (OSC)]
```

**Local mode (pyo3):**
- Renders audio via Rust DSP through pyo3 (or Numba before port)
- Plays through sounddevice
- Computes metrics and spectrograms locally
- Works standalone — no DAW needed
- This is how it works today

**Plugin mode (OSC):**
- Sends params to running VST via OSC
- "Play" button becomes "Send to Plugin" (DAW handles audio)
- Receives metrics and audio snapshots from plugin
- Spectrograms generated from plugin's snapshot WAVs
- Render/playback controls disabled (DAW owns audio)

**Claude SDK doesn't know the difference.** The agentic tuning interface is:
- `set_params(params_dict)` — routes to pyo3 or OSC based on mode
- `get_metrics()` — returns dict from local computation or OSC
- `get_spectrogram()` — generates from local render or plugin snapshot

This abstraction means agentic tuning code works identically in both modes.
Switch modes to go from "exploring in isolation" to "tuning in context of a mix."

---

## What Needs to Be Ported to Rust

### Both Pedals

Both reverb and lossy follow the same architecture: Rust DSP core, nih-plug plugin wrapper,
pyo3 Python bindings, OSC bridge. They share the OSC protocol and preset format patterns
but have independent DSP implementations.

### Reverb DSP (~1,080 lines Python/Numba → Rust)

Files to port:
- `reverb/engine/params.py` → Rust struct with defaults
- `reverb/primitives/matrix.py` → Rust matrix types + fast apply
- `reverb/engine/numba_fdn.py` → Rust FDN processing loop
- `reverb/engine/numba_fdn_mod.py` → Rust modulated FDN loop
- `reverb/engine/fdn.py` → Rust dispatcher (static vs modulated)

### Lossy DSP (~1,200 lines Python/Numba → Rust)

Files to port:
- `lossy/engine/params.py` → Rust struct with defaults
- `lossy/engine/spectral.py` → Rust STFT codec emulation
- `lossy/engine/bitcrush.py` → Rust time-domain degradation
- `lossy/engine/packets.py` → Rust packet loss/repeat
- `lossy/engine/filters.py` → Rust biquad, reverb, gate, limiter
- `lossy/engine/lossy.py` → Rust signal chain dispatcher

### What Stays in Python
- `reverb/gui/gui.py` — tkinter dev GUI
- `lossy/gui/gui.py` — tkinter dev GUI
- `shared/llm_tuner.py` — Claude Code SDK bridge (multimodal: text + metrics + spectrograms)
- `shared/llm_guide_text.py` — LLM prompts + metric interpretation guides
- `shared/streaming.py` — dev audio playback
- `shared/analysis.py` — audio metrics (RT60, EDT, centroid, echo density, C50/C80, etc.)
- `shared/audio_features.py` — metrics formatting for LLM + spectrogram PNG generation
- `shared/waveform.py` — shared tkinter waveform visualization

Note: `shared/analysis.py` metrics may also be partially ported to Rust for the
`shared-dsp` crate's live metrics computation (RMS, centroid, crest factor).
The full Schroeder decay analysis (RT60, EDT) is less useful in real-time and
can stay Python-only for offline analysis.

### Rust Workspace Structure (proposed)

```
reverb-workspace/
  Cargo.toml              — workspace root

  crates/
    reverb-dsp/
      src/
        lib.rs            — public API: render_fdn()
        params.rs         — parameter struct + defaults
        matrix.rs         — matrix types + fast apply
        fdn.rs            — static FDN loop
        fdn_mod.rs        — modulated FDN loop
        lfo.rs            — LFO waveforms
      Cargo.toml

    lossy-dsp/
      src/
        lib.rs            — public API: render_lossy()
        params.rs         — parameter struct + defaults
        spectral.rs       — STFT codec emulation
        bitcrush.rs       — time-domain degradation
        packets.rs        — packet loss/repeat
        filters.rs        — biquad, reverb, gate, limiter
        chain.rs          — signal chain dispatcher
      Cargo.toml

    shared-dsp/
      src/
        lib.rs            — shared DSP utilities
        osc.rs            — OSC protocol handler (shared between plugins)
        ring_buffer.rs    — audio ring buffer for snapshots
        metrics.rs        — running audio metrics (RMS, centroid, etc.)
        smoothing.rs      — parameter smoothing
      Cargo.toml

    reverb-plugin/
      src/
        lib.rs            — nih-plug wrapper for reverb
        gui.rs            — egui GUI
        presets.rs        — JSON preset loader
      Cargo.toml

    lossy-plugin/
      src/
        lib.rs            — nih-plug wrapper for lossy
        gui.rs            — egui GUI
        presets.rs        — JSON preset loader
      Cargo.toml

    reverb-python/
      src/
        lib.rs            — pyo3 bindings for reverb DSP
      Cargo.toml

    lossy-python/
      src/
        lib.rs            — pyo3 bindings for lossy DSP
      Cargo.toml
```

### Shared Crate (`shared-dsp`)

Both plugins share:
- **OSC handler** — same protocol, same message format, different param schemas
- **Ring buffer** — input + output audio capture for snapshots
- **Metrics** — RMS, peak, spectral centroid, crest factor computation
- **Parameter smoothing** — lock-free atomic params with ramp
- **Snapshot writer** — dump ring buffer to WAV on OSC request

This mirrors how `shared/` works in the Python codebase — common infrastructure,
independent DSP implementations.

---

## Sequencing Options

### Path A — VST first, then agentic tuning
1. Port `render_fdn` to Rust
2. Wrap with nih-plug (VST3 + standalone)
3. Add pyo3 bindings
4. Build agentic tuning pipeline against Rust DSP

### Path B — Agentic tuning first, then VST
1. Build metrics/spectrogram/agentic loop with Numba DSP
2. Discover and refine presets with Claude
3. Port to Rust later, presets carry over

### Path C — Both in parallel
1. Start Rust DSP port
2. Build agentic tuning with Numba simultaneously
3. Connect them when Rust port is ready

### Scope Note
All paths apply to **both reverb and lossy** pedals. The Rust workspace structure
supports porting them independently — can do reverb first, lossy second, or both
in parallel. The shared-dsp crate (OSC, ring buffers, metrics) is built once and
used by both plugins.

### Decision: TBD
