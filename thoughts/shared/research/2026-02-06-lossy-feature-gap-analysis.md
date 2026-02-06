---
date: 2026-02-06T21:30:29Z
git_commit: 6c35439
branch: main
repository: reverb
topic: "Lossy plugin: implemented vs RESEARCH.md feature gap analysis"
tags: [research, lossy, spectral, dsp, codec-emulation]
status: complete
last_updated: 2026-02-06
---

# Lossy Plugin: Feature Gap Analysis

## Research Question

What does RESEARCH.md describe that isn't yet implemented? What's the plan to close the gaps?

---

## What Exists Today

### Spectral Engine (`engine/spectral.py`)
- STFT overlap-add: Hann window, 75% overlap, 5 discrete window sizes (256–4096)
- **Standard mode**: uniform mid-tread magnitude quantization (16→2 bits) + random band gating across ~21 log-spaced Bark-like bands
- **Inverse mode**: spectral residual (original − standard)
- **Jitter mode**: uniform random phase perturbation per bin
- Bandwidth limiting: soft HF rolloff when loss > 0.3
- Freeze: slushy (exponential average) and solid (static snapshot)

### Packets (`engine/packets.py`)
- Gilbert-Elliott two-state Markov model
- Packet Loss (silence insertion) and Packet Repeat (last-good stutter)
- Hard chunk boundaries — no crossfade

### Filters / Effects (`engine/filters.py`)
- Biquad bandpass and notch (Audio EQ Cookbook)
- Cascaded sections: 1/2/8 for 6/24/96 dB slopes
- Lo-fi reverb: 4 comb filters + 1 allpass (Numba)
- RMS noise gate, peak limiter

### Signal Chain (`engine/lossy.py`)
- Input → Spectral Loss → Packets → Filter → Verb → Gate → Limiter → Mix

---

## Gaps: RESEARCH.md Features Not Yet Implemented

### 1. Psychoacoustic masking model
**Research says:** MP3 computes masking thresholds using a spreading function in the Bark domain. The signal-to-mask ratio per scalefactor band drives which bands get zeroed. The gating isn't random — it's signal-dependent.

**Current state:** Band gating is purely random (coin flip per band per frame). This sounds good but doesn't track the input signal's spectral energy.

**Implementation:** Compute per-band energy each frame. Apply the ATH (absolute threshold of hearing) curve as a floor. Gate bands whose energy falls below a signal-dependent threshold (mean energy × loss factor), with random perturbation to keep the frame-to-frame variation. This makes quiet bands more likely to be gated while loud bands survive — matching real codec behavior.

### 2. Nonuniform power-law quantizer
**Research says:** MP3 uses `nint(|xr|^0.75 / step)` — a companding quantizer that gives finer resolution to quiet signals and coarser resolution to loud ones.

**Current state:** Uniform quantization: `Δ · round(mag/Δ)`.

**Implementation:** Add a `quantizer_type` param (0=uniform, 1=power-law). For power-law: `sign(x) · step · round(|x/step|^0.75)` then invert on reconstruction. Changes the distortion character — more "codec-like" vs the current "bitcrusher-like" quantization.

### 3. Crossfade at packet boundaries
**Research says:** Real network audio and granular synthesis use windowed grains to prevent clicks at boundaries.

**Current state:** Hard cuts at packet edges. Audible clicks when packets drop/repeat.

**Implementation:** Apply short (2–5 ms) fade-in/fade-out Hann half-windows at packet boundaries. Trivial to add in `packet_process()`.

### 4. Phase quantization in Standard mode
**Research says:** Phase can be left intact (more natural) or quantized (more destructive).

**Current state:** Standard mode preserves phase. Only Jitter perturbs phase.

**Implementation:** Add `phase_quantize` param (0.0–1.0). When > 0, quantize phase angles to N levels: `round(phase / step) * step` where N scales with loss. Subtle at low values, increasingly metallic/robotic at high values.

### 5. Bitcrusher + sample rate reducer
**Research says:** Complementary time-domain degradation. Bitcrushing: `floor(x × 2^(bits-1)) / 2^(bits-1)`. Sample rate reduction: zero-order hold with phase accumulator. Sounds fundamentally different from spectral processing — could be layered.

**Current state:** Not implemented. The current effect is purely spectral.

**Implementation:** New `engine/bitcrush.py`. Two params: `crush_bits` (4–16, 0=off) and `crush_rate` (fraction of SR, 0=off). Insert in chain after spectral, before packets. Optional — off by default so the plugin stays spectral-first.

### 6. Freezer blend parameter
**Research says:** The pedal has a hidden "Freezer" parameter controlling the balance between live input and frozen signal.

**Current state:** Freeze is binary — either the spectrum updates (slushy) or doesn't (solid).

**Implementation:** Add `freeze_blend` param (0.0–1.0). Output = `blend × frozen + (1-blend) × live_processed`. Allows partial freeze where the frozen spectrum acts as a pad underneath the live signal.

### 7. Pre-echo enhancement
**Research says:** Transients encoded with long windows spread quantization noise across the entire window, making noise audible before the attack.

**Current state:** The STFT already creates some temporal smearing, but it's symmetrical (spread equally before and after the transient).

**Implementation:** Detect transient frames (energy ratio between adjacent frames). For transient frames, apply extra quantization noise to the preceding frame's lower-energy bins. This makes the smear more audible *before* the transient — the signature pre-echo artifact.

### 8. Signal-envelope noise shaping
**Research says:** Noise shaping that follows the spectral envelope of the signal makes quantization noise less perceptible in regions where the signal is loud, and more audible in quiet regions — the opposite of dithering, which is the desired lo-fi effect.

**Current state:** Quantization is uniform across the spectrum.

**Implementation:** Compute spectral envelope (smoothed magnitude curve). Shape the quantization step size inversely: coarser quantization in spectral valleys, finer near peaks. This creates "noise filling" in quiet bands, mimicking how codec noise follows the signal shape.

---

## Implementation Priority

### Phase A — Quick wins (improve what exists)
1. **Packet crossfade** — 15 lines in packets.py. Eliminates clicks.
2. **Psychoacoustic gating** — Replace random gating with signal-dependent threshold + ATH curve. Same code structure, better sound.
3. **Phase quantization param** — Small addition to spectral.py Standard mode.

### Phase B — New capabilities
4. **Nonuniform quantizer** — Power-law option in `_standard()`.
5. **Freezer blend** — New param, 3 lines in spectral.py freeze section.
6. **Bitcrusher + sample rate reducer** — New module, new params, new GUI controls.

### Phase C — Polish
7. **Pre-echo enhancement** — Transient detection + preceding-frame noise injection.
8. **Noise shaping** — Envelope-following quantization step size.

---

## Code References

- `lossy/engine/spectral.py:128-156` — `_standard()` where quantization + gating happens
- `lossy/engine/spectral.py:94-98` — bandwidth limiting
- `lossy/engine/spectral.py:100-109` — freeze logic
- `lossy/engine/packets.py:48-62` — packet processing loop (needs crossfade)
- `lossy/engine/filters.py:71-83` — Numba biquad
- `lossy/engine/params.py:28-62` — full param schema
- `lossy/RESEARCH.md:27-33` — psychoacoustic model details
- `lossy/RESEARCH.md:43` — O'Brien frequency-domain quantization reference
- `lossy/RESEARCH.md:55-57` — bitcrusher/sample-rate-reducer formulas
- `lossy/RESEARCH.md:67-71` — granular/packet windowing details
