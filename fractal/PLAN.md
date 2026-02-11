# Fractal Pedal — Implementation Plan

## Overview

A new effects pedal based on the R001 Audio Fractalization algorithm from `explore/effects/r_misc.py`. The core idea: replace each sample with scaled, tiled copies of the signal at multiple timescales, creating self-similar fractal-like texture. The GUI and project structure clone `lossy/` exactly.

**Launch:** `uv run python -m fractal.main`

---

## Package Structure

```
fractal/
├── __init__.py
├── main.py                    # tkinter entry point (clone lossy/main.py)
├── engine/
│   ├── __init__.py
│   ├── params.py              # parameter schema, ranges, sections
│   ├── fractal.py             # render_fractal() — main signal chain
│   ├── core.py                # fractalize() — the r001 algorithm (Numba)
│   └── filters.py             # biquad, gate, limiter (reuse from lossy)
├── audio/
│   ├── __init__.py
│   └── render.py              # CLI offline rendering
├── gui/
│   ├── __init__.py
│   ├── gui.py                 # FractalGUI (clone LossyGUI)
│   └── presets/               # JSON presets
└── PLAN.md                    # this file
```

---

## Parameter Design

### Core Fractal Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `num_scales` | 3 | 2–8 | Number of fractal scale layers (expanded from r001's max of 5) |
| `scale_ratio` | 0.5 | 0.1–0.9 | Compression ratio per scale (expanded range for more extreme effects) |
| `amplitude_decay` | 0.5 | 0.1–1.0 | Gain decay per scale level |
| `interp` | 0 | 0=nearest, 1=linear | Resampling interpolation (nearest = aliased/gritty, linear = smooth) |
| `reverse_scales` | 0 | 0=off, 1=on | Reverse the tiled chunks (backward fractal layers) |
| `scale_offset` | 0.0 | 0.0–1.0 | Phase offset for tile starting position (shifts where each scale's tiles begin) |

### Iteration / Feedback

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `iterations` | 1 | 1–4 | Feed output back through the fractalizer N times |
| `iter_decay` | 0.8 | 0.3–1.0 | Gain applied between iterations |
| `saturation` | 0.0 | 0.0–1.0 | tanh saturation in the iteration feedback loop |

### Spectral Fractal (apply fractalization in frequency domain)

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `spectral` | 0.0 | 0.0–1.0 | Blend: 0=time-domain only, 1=spectral-domain only |
| `window_size` | 2048 | 256–8192 | STFT window for spectral mode |

### Filter Section

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `filter_type` | 0 | 0=bypass, 1=lowpass, 2=highpass, 3=bandpass | Pre-fractal filter |
| `filter_freq` | 2000.0 | 20.0–20000.0 | Filter center/cutoff frequency |
| `filter_q` | 0.707 | 0.1–10.0 | Filter Q/resonance |
| `post_filter_type` | 0 | 0=bypass, 1=lowpass, 2=highpass | Post-fractal filter (tame aliasing) |
| `post_filter_freq` | 8000.0 | 20.0–20000.0 | Post-filter cutoff |

### Effects

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `gate` | 0.0 | 0.0–1.0 | Noise gate threshold |
| `crush` | 0.0 | 0.0–1.0 | Bitcrusher (post-fractal texture) |
| `decimate` | 0.0 | 0.0–1.0 | Sample rate reduction |

### Bounce (LFO Modulation)

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `bounce` | 0 | 0=off, 1=on | Enable LFO modulation |
| `bounce_target` | 0 | index into targets list | Which param to modulate |
| `bounce_rate` | 0.3 | 0.0–1.0 | LFO speed |
| `bounce_lfo_min` | 0.1 | 0.01–50.0 | LFO Hz minimum |
| `bounce_lfo_max` | 5.0 | 0.01–50.0 | LFO Hz maximum |

**Bounce targets:** `["scale_ratio", "amplitude_decay", "num_scales", "saturation", "filter_freq", "crush", "spectral"]`

### Output

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `wet_dry` | 1.0 | 0.0–1.0 | Dry/wet mix |
| `output_gain` | 0.5 | 0.0–1.0 | Output level (-36 to +36 dB) |
| `threshold` | 0.5 | 0.0–1.0 | Limiter threshold |

### Internal

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `seed` | 42 | — | Random seed for reproducibility |

### Parameter Sections (for lock groups)

```python
PARAM_SECTIONS = {
    "fractal": ["num_scales", "scale_ratio", "amplitude_decay", "interp",
                "reverse_scales", "scale_offset"],
    "iteration": ["iterations", "iter_decay", "saturation"],
    "spectral": ["spectral", "window_size"],
    "filter": ["filter_type", "filter_freq", "filter_q",
              "post_filter_type", "post_filter_freq"],
    "effects": ["gate", "crush", "decimate"],
    "bounce": ["bounce", "bounce_target", "bounce_rate",
              "bounce_lfo_min", "bounce_lfo_max"],
    "output": ["wet_dry", "output_gain", "threshold"],
}
```

---

## Signal Chain

```
Input
  → Pre-Filter (lowpass/highpass/bandpass)
    → Fractalize (core r001 algorithm, expanded)
      → [repeat if iterations > 1, with saturation + iter_decay]
    → Bitcrusher + Decimate
      → Post-Filter (tame aliasing artifacts)
        → Noise Gate
          → Limiter
            → Wet/Dry Mix
              → Output
```

When `spectral > 0`, the fractalize step blends between:
- Time-domain fractalization (tile compressed copies)
- Spectral fractalization (apply same compression/tiling logic to STFT magnitude frames)

---

## Engine Implementation

### `engine/core.py` — The Fractal Algorithm

```python
@numba.njit(cache=True)
def fractalize_time(samples, num_scales, scale_ratio, amplitude_decay,
                    interp, reverse_scales, scale_offset):
    """Core r001 algorithm with extensions. Numba JIT for speed."""
    # For each scale 1..num_scales:
    #   1. Compute compressed_len = n * (scale_ratio ** s)
    #   2. Downsample signal to compressed_len (nearest or linear interp)
    #   3. Optionally reverse the compressed chunk
    #   4. Apply scale_offset to shift tile start position
    #   5. Tile to fill original length
    #   6. out += (amplitude_decay ** s) * tiled
    # Normalize to input peak
```

```python
def fractalize_spectral(samples, sr, num_scales, scale_ratio,
                        amplitude_decay, window_size):
    """Apply fractalization to STFT magnitude."""
    # STFT → for each frame, tile/compress magnitude vector → ISTFT
```

```python
def render_fractal_core(samples, sr, params):
    """Blend time-domain and spectral fractalization based on 'spectral' param."""
    t = fractalize_time(...)
    if spectral > 0:
        s = fractalize_spectral(...)
        return (1 - spectral) * t + spectral * s
    return t
```

### `engine/fractal.py` — Signal Chain

```python
def render_fractal(input_audio, params, chunk_callback=None):
    """Main entry point. All callers use this."""
    # 1. Pre-filter
    # 2. Fractalize (with iterations loop)
    # 3. Crush + decimate
    # 4. Post-filter
    # 5. Gate
    # 6. Limiter
    # 7. Wet/dry mix
    # If bounce enabled, process in ~50ms blocks with LFO modulation
```

### `engine/filters.py`

Reuse/adapt lossy's Numba biquad, gate, and limiter implementations. Import from lossy or copy the 4 key functions:
- `biquad_filter()` — cascaded biquad
- `noise_gate()` — RMS-based
- `limiter()` — soft clip
- `crush_and_decimate()` — bitcrusher + zero-order hold

Decision: copy rather than import from lossy, to keep packages independent.

---

## GUI Implementation

### Clone from `lossy/gui/gui.py` → `fractal/gui/gui.py`

Class `FractalGUI` mirrors `LossyGUI` exactly in structure:

**6 Tabs:**
1. **Parameters** — Slider grid grouped by section (Fractal, Iteration, Spectral, Filter, Effects, Bounce, Output) + AI chat panel
2. **Presets** — Treeview with categories, favorites, search, save/load
3. **Waveforms** — Input/output waveform display with playback cursor
4. **Spectrograms** — Side-by-side input/output spectrograms
5. **Spectrum** — FFT frequency response overlay (dry vs wet)
6. **Guide** — Read-only help text explaining the fractal algorithm

**Reused patterns from lossy:**
- 400ms debounced auto-play on slider change
- Background render thread → `root.after()` callback
- Generation history (undo/redo, max 50 snapshots)
- Per-param and per-section lock toggles
- AI tuner integration via `shared.llm_tuner.LLMTuner`
- WAV file load/drag-and-drop
- Mousewheel fine-adjust on sliders
- Responsive layout (horizontal/vertical based on window width)

### Key Differences from Lossy GUI
- Window title: "Fractal"
- Parameter sections: Fractal, Iteration, Spectral, Filter, Effects, Bounce, Output (instead of lossy's Spectral Loss, Crush, Packets, etc.)
- Guide text: explains audio fractalization concept
- Preset categories: TBD after creating initial presets (e.g., Subtle, Textural, Glitch, Extreme, Modulated)
- Calls `render_fractal()` instead of `render_lossy()`

---

## Shared Integration

### `shared/llm_guide_text.py`

Add two new constants:
- `FRACTAL_GUIDE` — system prompt for the AI tuner (signal chain, parameter descriptions, aesthetic guidance)
- `FRACTAL_PARAM_DESCRIPTIONS` — per-param human-readable descriptions

### `pyproject.toml`

```toml
[tool.hatch.build.targets.wheel]
packages = ["reverb", "lossy", "fractal", "shared"]

[project.scripts]
reverb = "reverb.main:main"
lossy = "lossy.main:main"
fractal = "fractal.main:main"
```

---

## Implementation Order

### Phase 1: Engine
1. Create `fractal/engine/params.py` — full parameter schema with defaults, ranges, sections
2. Create `fractal/engine/core.py` — Numba-compiled fractalize_time + fractalize_spectral
3. Create `fractal/engine/filters.py` — copy biquad, gate, limiter, crush from lossy
4. Create `fractal/engine/fractal.py` — render_fractal() signal chain with bounce support
5. Test: `render_fractal(test_audio, default_params())` produces valid output

### Phase 2: GUI
6. Create `fractal/gui/gui.py` — clone LossyGUI, adapt to FractalGUI
7. Create `fractal/main.py` — entry point
8. Create `fractal/__init__.py`, `fractal/engine/__init__.py`, `fractal/gui/__init__.py`, `fractal/audio/__init__.py`
9. Update `pyproject.toml` — add fractal package + script entry point
10. Test: `uv run python -m fractal.main` launches GUI, loads WAV, plays audio

### Phase 3: Polish
11. Add `shared/llm_guide_text.py` entries (FRACTAL_GUIDE, FRACTAL_PARAM_DESCRIPTIONS)
12. Create `fractal/audio/render.py` — CLI offline rendering
13. Create 10-15 starter presets in `fractal/gui/presets/`
14. Test AI tuner integration

---

## Starter Presets

| Name | Category | Key Settings |
|------|----------|-------------|
| Clean Fractal | Subtle | 2 scales, ratio 0.5, decay 0.6, no iteration |
| Triple Layer | Subtle | 3 scales, ratio 0.5, decay 0.5 (the original r001 default) |
| Deep Fractal | Textural | 5 scales, ratio 0.3, decay 0.5, 2 iterations |
| Grainy | Textural | 4 scales, ratio 0.4, nearest interp, crush 0.3 |
| Aliased Mess | Glitch | 8 scales, ratio 0.2, nearest, 3 iterations, saturation 0.6 |
| Smooth Layers | Subtle | 3 scales, ratio 0.7, linear interp, post LP at 6kHz |
| Reverse Fractal | Textural | 4 scales, reverse on, ratio 0.5, decay 0.7 |
| Spectral Ghost | Textural | spectral 1.0, 4 scales, window 4096 |
| Hybrid | Textural | spectral 0.5, 3 scales, 2 iterations |
| Saturated Feedback | Extreme | 3 scales, 4 iterations, saturation 0.8, iter_decay 0.6 |
| Breathing Fractal | Modulated | bounce on, target scale_ratio, rate 0.4 |
| Pulsing Depth | Modulated | bounce on, target num_scales, rate 0.2 |
| Guitar Crunch | Subtle | 2 scales, ratio 0.6, saturation 0.3, pre HP 200Hz |
| Pad Wash | Textural | 6 scales, ratio 0.4, spectral 0.7, LP 4kHz |
