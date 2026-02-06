---
date: 2026-02-06T16:19:41+0000
researcher: user
git_commit: d66a8c76dbf3daa02421bffebc0dbd92450c0b22
branch: main
repository: reverb
topic: "Time-Varying Parameters and Modulation"
tags: [research, codebase, modulation, lfo, time-varying, fdn, phase-3]
status: complete
last_updated: 2026-02-06
last_updated_by: user
---

# Research: Time-Varying Parameters and Modulation

**Date**: 2026-02-06T16:19:41+0000
**Researcher**: user
**Git Commit**: d66a8c76dbf3daa02421bffebc0dbd92450c0b22
**Branch**: main
**Repository**: reverb

## Research Question
What is the current state of time-varying parameters and modulation in the reverb codebase? What exists, what is planned, and how is the architecture structured?

## Summary

**Phase 3 (Time-Varying FDN) is fully designed but not yet implemented.** The codebase currently operates as a static FDN reverb — all parameters are frozen at render time and remain constant throughout audio processing. No LFO, modulation, chorus, vibrato, or parameter automation code exists. The planning documents (`PLAN.md`, `TODO.md`) contain a detailed design for a three-timescale modulation system with structured parameter control.

---

## Detailed Findings

### 1. Current Engine: Entirely Static

#### Parameter Schema (`engine/params.py`)

All parameters in `default_params()` are static values:

| Parameter | Type | Default | Modulated? |
|-----------|------|---------|------------|
| `delay_times` | int[8] | `[1310, 1637, 1821, 2113, 2342, 2615, 2986, 3224]` samples | No |
| `damping_coeffs` | float[8] | `[0.3]*8` | No |
| `feedback_gain` | float | `0.85` | No |
| `input_gains` | float[8] | `[0.125]*8` | No |
| `output_gains` | float[8] | `[1.0]*8` | No |
| `pre_delay` | int | `441` samples | No |
| `diffusion` | float | `0.5` | No |
| `diffusion_delays` | int[4] | `[234, 349, 516, 710]` samples | No |
| `wet_dry` | float | `0.5` | No |
| `saturation` | float | `0.0` | No |
| `node_pans` | float[8] | `[-1.0 .. 1.0]` spread | No |
| `stereo_width` | float | `1.0` | No |
| `matrix_type` | str | `"householder"` | No |

`PARAM_RANGES` (lines 76-89) defines ML exploration bounds for continuous params. No modulation-related keys exist.

#### Core FDN Loop (`engine/numba_fdn.py`)

The `_process_block()` function (lines 13-102) is a `@njit(cache=True)` Numba JIT function. It receives 19 parameter groups as flat arrays/scalars.

**Data flow for all parameters:**
1. Loaded from `params` dict once in `render_fdn_fast()` (lines 105-172)
2. Converted to numpy arrays or scalars
3. Passed unchanged to `_process_block()`
4. Used as constants in the per-sample loop (lines 39-101)

Key static reads in the inner loop:
- **Delay read** (line 65): `rd = (wi - 1 - delay_times[i]) % delay_buf_len` — integer index, no interpolation
- **Damping** (line 73): `a = damping_coeffs[i]` — fixed coefficient
- **Matrix** (line 82): `s += matrix[i, j] * reads[j]` — static matrix
- **Feedback** (line 88): `val = feedback_gain * mixed[i] + input_gains[i] * diffused` — fixed gains
- **Saturation** (line 90): `(1.0 - saturation) * val + saturation * np.tanh(val)` — fixed amount
- **Mix** (line 99): `(1.0 - wet_dry) * x + wet_dry * wet_L` — fixed ratio

The only values that change per-sample are buffer indices and filter state variables (inherent to DSP, not modulation).

#### Wrapper (`engine/fdn.py`)

Thin pass-through (lines 22-34):
```python
def render_fdn(input_audio, params):
    from engine.numba_fdn import render_fdn_fast
    return render_fdn_fast(input_audio, params)
```

Signal flow documented in lines 1-14: Input -> Pre-delay -> Input Diffusion -> FDN Loop -> Wet/Dry Mix -> Output. No mention of modulation.

### 2. DSP Primitives: No Fractional Delay or LFO

#### `primitives/dsp.py`

Seven standalone Numba-JIT functions, all operating on full buffers with static parameters:

| Function | Lines | Time-Varying? |
|----------|-------|--------------|
| `delay(audio, delay_samples)` | 11-19 | No — integer sample delay |
| `delay_feedback(audio, delay_samples, feedback, wet)` | 23-35 | No — static delay, feedback, mix |
| `one_pole_lowpass(audio, coeff)` | 39-47 | No — static coefficient |
| `allpass(audio, delay_samples, gain)` | 51-63 | No — integer delay, static gain |
| `allpass_chain(audio, delay_times, gain)` | 67-82 | No — series of static allpasses |
| `comb_filter(audio, delay_samples, feedback, damping)` | 86-101 | No — static params |
| `saturate(audio, drive)` | 105-111 | No — static drive |

**Notable absence:** No fractional delay interpolation (linear or cubic) despite `TODO.md` marking it as complete. All delay reads use integer sample indices. Fractional delay is a prerequisite for smooth delay-time modulation.

#### `primitives/matrix.py`

Seven matrix constructors (householder, hadamard, diagonal, random_orthogonal, circulant, stautner_puckette, zero) plus fast apply functions and utilities. All matrices are static — no time-varying matrix support.

### 3. GUI: Static Controls Only

#### `gui/gui.py`

The tkinter GUI exposes all parameters from `engine/params.py` as sliders and controls. Organized in tabs: Parameters, Presets, Signal Flow, Waveform, Guide.

**Parameter update flow:**
1. User adjusts sliders — visual feedback updates immediately (XY pad sync, signal flow diagram, value labels)
2. User clicks "Play" — `_on_play()` (line 1437) calls `_read_params_from_ui()` (line 1050)
3. Parameters frozen as a dict snapshot
4. `_render()` (line 1384) launches background thread calling `render_fdn(audio, params)`
5. Result played back via sounddevice

**No real-time parameter modulation.** Parameters only take effect on re-render.

**No modulation controls exist** — no LFO sliders, no modulation depth/rate, no waveform selectors.

#### Presets (`gui/presets/*.json`)

All preset files store only static parameters. Example keys from `decent_room.json`:
```
delay_times, damping_coeffs, feedback_gain, input_gains, output_gains,
pre_delay, diffusion, diffusion_stages, diffusion_delays, wet_dry,
saturation, matrix_type, matrix_seed
```

No `mod_depth`, `mod_rate`, `mod_waveform`, or `mod_phase` keys in any preset.

### 4. Offline Rendering: Static Parameters

#### `audio/render.py`

The batch renderer loads a preset JSON, creates a single params dict, appends a silence tail, and calls `render_fdn()` once for the entire buffer (line 104). No parameter automation or modulation in the render pipeline.

### 5. Phase 3 Design (Planned, Not Implemented)

#### Three Timescales (`PLAN.md:195-202`)

| Timescale | Range | Effect |
|-----------|-------|--------|
| Slow | 0.01-0.5 Hz | Character evolves over seconds; delay times drift, filter cutoffs shift, matrix blends between configs |
| Medium/LFO | 0.5-20 Hz | Eliminates metallic ringing (~2 Hz, ±4-7 samples); filter breathing, vocal quality; "cheap vs expensive reverb" |
| Fast/Audio-rate | 20 Hz+ | FM-like sidebands, inharmonic spectral content, ring-mod effects; topology modulation "essentially unexplored" |

#### Per-Parameter Modulation Controls (`PLAN.md:204-209`)

Each modulatable parameter receives four controls:
- `mod_depth` — amplitude of modulation (0 = static)
- `mod_rate` — frequency (0.01 Hz to 1000+ Hz, spanning all timescales)
- `mod_waveform` — sine, triangle, sample-and-hold, envelope-follower
- `mod_phase` — relative to other modulators

#### Modulatable Parameters (`PLAN.md:210-214`)

- **Delay times** — "most impactful — even tiny modulation transforms character"
- **Damping filter coefficients** — tonal character evolves
- **Feedback matrix coefficients** — topology breathes, signal rerouting
- **Output tap gains** — spatial image shifts

#### Structured Modulation (`PLAN.md:216-222`)

To reduce parameter explosion (8 nodes x 4 params = 32 new params per modulatable parameter type):
- Global master rate (one clock)
- Per-node rate multiplier (integer ratios: 1x, 2x, 3x for rhythmic relationships)
- Global depth with per-node depth scaling
- Correlation parameter (sync vs. independent movement)

#### Pseudocode (`PLAN.md:224-233`)

```python
for each sample n:
    for each node i:
        phase = (mod_rate[i] * n / sample_rate + mod_phase[i]) % 1.0
        lfo_value = waveform_func(phase, mod_waveform[i])
        current_delay[i] = base_delay[i] + mod_depth_delay[i] * lfo_value
        current_damping[i] = base_damping[i] + mod_depth_damping[i] * lfo_value
    # ... process FDN with current_* values
```

#### Expanded Parameter Space (`PLAN.md:235-248`)

With modulation, the explorable space grows to include: conventional rooms/halls/plates, lush chorus-like reverbs, tape-warped lo-fi, metallic resonators, self-oscillating drones, FM-like alien textures, breathing soundscapes, rhythmic pulsing effects, and "everything in between."

#### Planned File (`PLAN.md:372`)

`engine/fdn_modulated.py` — the Phase 3 extension, separate from the Phase 1 static `engine/fdn.py`.

### 6. TODO.md Phase 3 Checklist (Lines 82-100, All Unchecked)

- [ ] LFO generator (sine, triangle, sample-and-hold, envelope-follower)
- [ ] Per-parameter modulation: depth, rate, waveform, phase
- [ ] Structured modulation (global master rate, per-node multipliers, correlation param)
- [ ] Delay time modulation
- [ ] Damping coefficient modulation
- [ ] Feedback matrix coefficient modulation
- [ ] Output tap gain modulation
- [ ] Slow timescale (0.01-0.5 Hz)
- [ ] Medium/LFO timescale (0.5-20 Hz)
- [ ] Fast/audio-rate timescale (20 Hz+)
- [ ] Create `engine/fdn_modulated.py`
- [ ] Extend params dict with modulation parameters
- [ ] Extend GUI with modulation controls
- [ ] Hand-discover interesting modulation configurations

### 7. ML Interaction with Modulation (`PLAN.md:259-290, 420-426`)

The ML exploration loop is designed to call `render_fdn(test_signal, params)` with modulation params included. Novel research areas identified:

- "ML-discovered multi-timescale modulation patterns" — rate, depth, waveform, phase optimized across all timescales
- "VAE latent space of dynamic FDN configurations" — including modulation envelopes, not just static params
- "Systematic audio-rate FDN modulation exploration via optimization" — Erbe did it manually; ML search is novel

### 8. Academic References

- **Schlecht & Habets (2015)** — Time-varying feedback matrices (topology modulation theory)
- **Tom Erbe, "Building the Erbe-Verb" (ICMC 2015)** — Audio-rate FDN modulation, the most creatively ambitious existing reverb
- **EMT 250 (1976)** — First commercial reverb with LFO-rate delay modulation

---

## Code References

- `engine/params.py:10-89` — Complete parameter schema and ML ranges (all static)
- `engine/numba_fdn.py:13-102` — Core FDN per-sample loop (static parameters throughout)
- `engine/numba_fdn.py:105-189` — Parameter initialization (loaded once, never updated)
- `engine/numba_fdn.py:65` — Delay line read: `rd = (wi - 1 - delay_times[i]) % delay_buf_len` (integer, static)
- `engine/fdn.py:22-34` — Thin wrapper, no modulation logic
- `primitives/dsp.py:11-111` — Seven DSP primitives, all static, no fractional delay
- `primitives/matrix.py:13-190` — Matrix types and utilities, all static
- `gui/gui.py:1050-1066` — `_read_params_from_ui()`, parameter snapshot for rendering
- `gui/gui.py:1384-1430` — `_render()`, background thread, static params for entire buffer
- `audio/render.py:104` — Offline render, single `render_fdn()` call with frozen params
- `PLAN.md:191-255` — Phase 3 modulation system design
- `TODO.md:82-100` — Phase 3 checklist (all unchecked)

## Architecture Documentation

### Current Architecture (Static)

```
GUI Sliders ─────────────────────────────────────────────────────┐
                                                                 │
Preset JSON ──→ params dict ──→ render_fdn() ──→ render_fdn_fast()
                 (static)        (engine/fdn.py)   (engine/numba_fdn.py)
                                                        │
ML Optimizer ───────────────────────────────────────────┘
                                                        │
                                                   _process_block()
                                                   [per-sample loop]
                                                   ALL PARAMS CONSTANT
```

### Planned Architecture (Phase 3, Not Implemented)

```
params dict ──→ render_fdn() ──→ fdn_modulated.py
  static base params                │
  + mod_depth_*                     ├── LFO generator (per node, per param)
  + mod_rate_*                      │     phase = (rate * n / SR + phase_offset) % 1.0
  + mod_waveform_*                  │     lfo = waveform_func(phase)
  + mod_phase_*                     │
  + master_rate                     ├── Per-sample parameter update
  + correlation                     │     current_delay[i] = base + depth * lfo
                                    │     current_damping[i] = base + depth * lfo
                                    │
                                    └── FDN loop with time-varying params
                                          (requires fractional delay for smooth delay modulation)
```

### Key Prerequisite Gap

Fractional delay interpolation (linear/cubic) is required for smooth delay-time modulation but does not exist in the codebase, despite `TODO.md` marking it as complete. Current delay reads are integer-sample only.

## Related Research

No other research documents exist in `thoughts/shared/research/` — this is the first.

## Open Questions

1. Where did the fractional delay interpolation code go? `TODO.md` marks it complete, but the deleted files (`primitives/delay_line.py`, `primitives/filters.py`) per git status may have contained it.
2. Will `engine/fdn_modulated.py` be a new Numba JIT function or wrap `_process_block()` with modulation applied externally?
3. How will the GUI handle modulation controls given the current render-on-play (non-realtime) architecture?
4. Test signals `05_vibrato_*.wav`, `06_sweep_*.wav`, `07_modulated_feedback.wav` exist — what generated them?
