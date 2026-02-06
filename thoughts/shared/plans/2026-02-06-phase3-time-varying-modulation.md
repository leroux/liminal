# Phase 3: Time-Varying Parameters (Dynamic FDN) — Implementation Plan

## Overview

Transform the static 8-node FDN reverb into a modulated system where parameters change per-sample via LFO generators. This adds delay time modulation (chorus/vibrato effects, metallic ringing elimination), damping modulation (breathing tonal character), output gain modulation (spatial movement), and feedback matrix modulation (topology breathing). The modulation system spans three timescales: slow (0.01–0.5 Hz), medium/LFO (0.5–20 Hz), and fast/audio-rate (20 Hz+).

## Current State Analysis

The FDN engine is fully static:
- `engine/numba_fdn.py:65` — delay reads use integer indices: `rd = (wi - 1 - delay_times[i]) % delay_buf_len`
- `engine/numba_fdn.py:139` — delay buffer sized to `max(delay_times) + 1` with no headroom for modulation excursion
- `engine/params.py` — 14 parameter keys, all static, no modulation-related entries
- `engine/fdn.py:33-34` — thin wrapper delegates to `render_fdn_fast()`
- `gui/gui.py` — static slider controls only, no modulation UI
- All delay reads are integer-sample (no fractional delay interpolation exists)

### Key Discoveries:
- `engine/numba_fdn.py:13` — `_process_block()` is `@njit(cache=True)`, all params are flat arrays/scalars for Numba compatibility
- `engine/numba_fdn.py:88` — feedback path: `val = feedback_gain * mixed[i] + input_gains[i] * diffused` — saturation and DC blocker follow
- `primitives/dsp.py` — standalone Numba primitives exist but all use integer delays
- `primitives/matrix.py:141-164` — `build_matrix_apply()` returns a fast apply function; `get_matrix()` returns the matrix array
- `gui/gui.py:1050-1066` — `_read_params_from_ui()` builds the params dict from sliders
- `gui/gui.py:1068-1090` — `_write_params_to_ui()` sets sliders from a params dict (used by preset loading)
- Buffer clearing: after changing Numba function signatures, must clear `__pycache__` to invalidate JIT cache

## Desired End State

After implementation:
1. `render_fdn(audio, params)` works identically for static params (backward compatible)
2. When modulation params are present (any `mod_depth_*` > 0), delay times, damping coefficients, output gains, and feedback matrix coefficients vary per-sample via LFO
3. Fractional delay interpolation (linear) enables smooth delay-time modulation without zipper noise
4. GUI exposes modulation controls (depth, rate, waveform per target; structured per-node multipliers)
5. Presets exist that demonstrate slow, LFO-rate, and audio-rate modulation
6. All three timescales work: slow (evolving character), medium (chorus/ringing elimination), fast (FM sidebands)

### Verification:
- Static presets produce identical output (bit-exact or within float tolerance)
- Modulated presets produce audibly different, time-varying output
- GUI modulation controls work and presets save/load correctly
- No explosions, NaN, or instability with modulation enabled
- `uv run --prerelease=allow python audio/render.py` works with modulated presets

## What We're NOT Doing

- Envelope follower LFO waveform (deferred — requires audio input analysis, more complex)
- Real-time mic input with modulation (Phase 2 realtime.py doesn't exist yet)
- Per-node independent modulation with full 32-param-per-target expansion (using structured modulation instead)
- Biquad filter modulation (not in the Phase 3 spec)
- ML exploration of modulation space (Phase 4)

## Implementation Approach

Create a new `engine/numba_fdn_mod.py` containing the modulated FDN loop, keeping the static version intact. The modulated version:
- Adds fractional delay reads (linear interpolation) for smooth delay modulation
- Computes LFO values per-sample per-node using phase accumulators
- Modulates delay times, damping, output gains, and matrix blend per-sample
- Uses structured modulation: global master rate + per-node multipliers + correlation parameter

The `engine/fdn.py` wrapper routes to the modulated version when any `mod_depth_*` > 0, otherwise uses the static path.

---

## Phase 1: Fractional Delay + LFO Primitives

### Overview
Add the two foundational building blocks: fractional delay interpolation (needed for smooth delay modulation) and LFO waveform generators (sine, triangle, sample-and-hold).

### Changes Required:

#### 1. LFO waveform functions
**File**: `engine/numba_fdn_mod.py` (new file)

Add Numba-compatible LFO functions that take a phase (0.0–1.0) and return a value (-1.0 to +1.0):

```python
import numpy as np
from numba import njit

N = 8

# --- LFO Waveforms ---
# Waveform codes: 0=sine, 1=triangle, 2=sample-and-hold

@njit(cache=True)
def lfo_value(phase, waveform):
    """Compute LFO value for a given phase (0-1) and waveform type.
    Returns value in range [-1, +1].
    """
    if waveform == 0:  # sine
        return np.sin(2.0 * np.pi * phase)
    elif waveform == 1:  # triangle
        # 0->1 in first half, 1->-1 in second half
        if phase < 0.25:
            return phase * 4.0
        elif phase < 0.75:
            return 2.0 - phase * 4.0
        else:
            return phase * 4.0 - 4.0
    else:  # sample-and-hold (waveform == 2)
        # Use phase to seed a simple hash for pseudo-random per-cycle value
        # This produces a new random value each cycle
        n = int(phase * 1000.0)
        # Simple hash
        h = ((n * 1103515245 + 12345) >> 16) & 0x7FFF
        return (h / 16383.5) - 1.0
```

#### 2. Fractional delay read helper
In the same file, add a linear-interpolation delay read:

```python
@njit(cache=True)
def read_delay_frac(buf, write_idx, delay_frac, buf_len):
    """Read from a delay line with fractional delay using linear interpolation.

    Args:
        buf: delay buffer (1D array for one node)
        write_idx: current write position
        delay_frac: delay time in fractional samples (float)
        buf_len: length of buffer

    Returns:
        interpolated sample value
    """
    delay_int = int(delay_frac)
    frac = delay_frac - delay_int

    idx0 = (write_idx - 1 - delay_int) % buf_len
    idx1 = (write_idx - 2 - delay_int) % buf_len

    return buf[idx0] * (1.0 - frac) + buf[idx1] * frac
```

### Success Criteria:

#### Automated Verification:
- [x] File `engine/numba_fdn_mod.py` exists with `lfo_value()` and `read_delay_frac()` functions
- [x] Numba JIT compilation succeeds: `uv run --prerelease=allow python -c "from engine.numba_fdn_mod import lfo_value, read_delay_frac"`
- [x] LFO produces correct range: sine at phase 0.25 returns ~1.0, at 0.75 returns ~-1.0
- [x] Fractional delay at integer values matches integer delay read (within float precision)

#### Manual Verification:
- [ ] Not needed for this phase — building blocks only

---

## Phase 2: Modulated FDN Engine

### Overview
Create the modulated `_process_block_mod()` function that applies per-sample LFO modulation to delay times, damping coefficients, output gains, and feedback matrix coefficients.

### Changes Required:

#### 1. Modulated process block
**File**: `engine/numba_fdn_mod.py` (extend)

The modulated version of `_process_block` adds these parameters:
- `mod_rate_delay`, `mod_depth_delay` — per-node delay time modulation (float arrays, length N)
- `mod_rate_damping`, `mod_depth_damping` — per-node damping modulation
- `mod_rate_output`, `mod_depth_output` — per-node output gain modulation
- `mod_rate_matrix`, `mod_depth_matrix` — matrix blend modulation (scalar rate, scalar depth)
- `mod_waveform` — waveform type (int: 0=sine, 1=triangle, 2=S&H)
- `mod_phases` — initial phase offsets per node (float array, length N)
- `matrix2` — second feedback matrix to blend toward during matrix modulation
- `sample_rate` — for phase computation

```python
@njit(cache=True)
def _process_block_mod(
    input_audio, output,
    # Pre-delay state
    pre_delay_buf, pre_delay_len, pre_delay_samples, pre_delay_write_idx,
    # Diffusion allpass state
    diff_bufs, diff_lens, diff_gains, diff_idxs, n_diff_stages,
    # FDN delay line state
    delay_bufs, delay_buf_len, delay_times_base, delay_write_idxs,
    # Damping filter state
    damping_coeffs_base, damping_y1,
    # Feedback matrices (base + target for blending)
    matrix, matrix2,
    # Gains
    feedback_gain, input_gains, output_gains_base, wet_dry,
    # Saturation
    saturation,
    # DC blocker state
    dc_x1, dc_y1, dc_R,
    # Stereo panning
    pan_gain_L, pan_gain_R,
    # --- Modulation params ---
    mod_depth_delay, mod_rate_delay,
    mod_depth_damping, mod_rate_damping,
    mod_depth_output, mod_rate_output,
    mod_depth_matrix, mod_rate_matrix,
    mod_waveform, mod_phases,
    sample_rate,
):
    n_samples = len(input_audio)
    pd_wi = pre_delay_write_idx[0]

    # S&H state: hold a random value per node, update when phase wraps
    sh_values = np.zeros(N)
    sh_prev_phase = np.zeros(N)
    # Matrix S&H
    sh_matrix_val = 0.0
    sh_matrix_prev = 0.0

    for n in range(n_samples):
        x = input_audio[n]

        # --- Pre-delay (unchanged) ---
        pre_delay_buf[pd_wi] = x
        pd_wi = (pd_wi + 1) % pre_delay_len
        rd_idx = (pd_wi - 1 - pre_delay_samples) % pre_delay_len
        x_delayed = pre_delay_buf[rd_idx]

        # --- Input diffusion (unchanged) ---
        diffused = x_delayed
        for s in range(n_diff_stages):
            idx = diff_idxs[s]
            delayed = diff_bufs[s, idx]
            g = diff_gains[s]
            v = diffused + g * delayed
            diffused = -g * v + delayed
            diff_bufs[s, idx] = v
            diff_idxs[s] = (idx + 1) % diff_lens[s]

        # --- Compute per-node LFO values ---
        lfo_delay = np.empty(N)
        lfo_damping = np.empty(N)
        lfo_output = np.empty(N)

        for i in range(N):
            t = n / sample_rate

            # Delay modulation LFO
            if mod_depth_delay[i] > 0.0:
                phase = (mod_rate_delay[i] * t + mod_phases[i]) % 1.0
                lfo_delay[i] = lfo_value(phase, mod_waveform)
            else:
                lfo_delay[i] = 0.0

            # Damping modulation LFO
            if mod_depth_damping[i] > 0.0:
                phase = (mod_rate_damping[i] * t + mod_phases[i]) % 1.0
                lfo_damping[i] = lfo_value(phase, mod_waveform)
            else:
                lfo_damping[i] = 0.0

            # Output gain modulation LFO
            if mod_depth_output[i] > 0.0:
                phase = (mod_rate_output[i] * t + mod_phases[i]) % 1.0
                lfo_output[i] = lfo_value(phase, mod_waveform)
            else:
                lfo_output[i] = 0.0

        # Matrix modulation LFO (scalar)
        if mod_depth_matrix > 0.0:
            t = n / sample_rate
            mat_phase = (mod_rate_matrix * t) % 1.0
            lfo_mat = lfo_value(mat_phase, mod_waveform)
            mat_blend = 0.5 + 0.5 * lfo_mat * mod_depth_matrix  # 0-1 blend
        else:
            mat_blend = 0.0

        # --- Read from delay lines (with fractional delay) ---
        wet_L = 0.0
        wet_R = 0.0
        reads = np.empty(N)
        for i in range(N):
            wi = delay_write_idxs[i]

            # Modulated delay time
            current_delay = delay_times_base[i] + mod_depth_delay[i] * lfo_delay[i]
            current_delay = max(1.0, current_delay)  # minimum 1 sample

            # Fractional delay read
            reads[i] = read_delay_frac(delay_bufs[i], wi, current_delay, delay_buf_len)

            # Modulated output gain
            current_out_gain = output_gains_base[i] * (1.0 + mod_depth_output[i] * lfo_output[i])
            current_out_gain = max(0.0, current_out_gain)

            tap = reads[i] * current_out_gain
            wet_L += tap * pan_gain_L[i]
            wet_R += tap * pan_gain_R[i]

        # --- Damping with modulated coefficients ---
        for i in range(N):
            current_damp = damping_coeffs_base[i] + mod_depth_damping[i] * lfo_damping[i]
            current_damp = max(0.0, min(0.999, current_damp))
            damping_y1[i] = (1.0 - current_damp) * reads[i] + current_damp * damping_y1[i]
            reads[i] = damping_y1[i]

        # --- Feedback matrix multiply (with optional blending) ---
        mixed = np.empty(N)
        for i in range(N):
            s = 0.0
            for j in range(N):
                if mat_blend > 0.0:
                    m = matrix[i, j] * (1.0 - mat_blend) + matrix2[i, j] * mat_blend
                else:
                    m = matrix[i, j]
                s += m * reads[j]
            mixed[i] = s

        # --- Write back (same as static) ---
        for i in range(N):
            wi = delay_write_idxs[i]
            val = feedback_gain * mixed[i] + input_gains[i] * diffused
            if saturation > 0.0:
                val = (1.0 - saturation) * val + saturation * np.tanh(val)
            dc_y = val - dc_x1[i] + dc_R * dc_y1[i]
            dc_x1[i] = val
            dc_y1[i] = dc_y
            delay_bufs[i, wi] = dc_y
            delay_write_idxs[i] = (wi + 1) % delay_buf_len

        # --- Wet/dry mix ---
        output[n, 0] = (1.0 - wet_dry) * x + wet_dry * wet_L
        output[n, 1] = (1.0 - wet_dry) * x + wet_dry * wet_R

    pre_delay_write_idx[0] = pd_wi
```

#### 2. Setup function for modulated rendering
**File**: `engine/numba_fdn_mod.py` (extend)

```python
def render_fdn_mod(input_audio: np.ndarray, params: dict) -> np.ndarray:
    """Modulated FDN rendering — drop-in replacement when modulation is active."""
    from primitives.matrix import get_matrix, MATRIX_TYPES
    from engine.params import SR as sample_rate

    n_samples = len(input_audio)

    # --- Build feedback matrices ---
    matrix_type = params.get("matrix_type", "householder")
    if matrix_type == "custom" and "matrix_custom" in params:
        matrix = np.array(params["matrix_custom"], dtype=np.float64)
    elif matrix_type in MATRIX_TYPES:
        matrix = get_matrix(matrix_type, N, seed=params.get("matrix_seed", 42))
    else:
        matrix = get_matrix("householder", N)

    # Second matrix for blending (use a different type or random orthogonal)
    matrix2_type = params.get("mod_matrix2_type", "random_orthogonal")
    matrix2_seed = params.get("mod_matrix2_seed", 137)
    if matrix2_type in MATRIX_TYPES:
        matrix2 = get_matrix(matrix2_type, N, seed=matrix2_seed)
    else:
        matrix2 = get_matrix("random_orthogonal", N, seed=matrix2_seed)

    # --- Pre-delay (same as static) ---
    pre_delay_samples = max(1, int(params["pre_delay"]))
    pre_delay_len = pre_delay_samples + 1
    pre_delay_buf = np.zeros(pre_delay_len)
    pre_delay_write_idx = np.array([0], dtype=np.int64)

    # --- Diffusion (same as static) ---
    n_diff_stages = min(params.get("diffusion_stages", 4),
                        len(params.get("diffusion_delays", [])))
    diff_delays = params.get("diffusion_delays", [])
    max_diff_len = max(diff_delays[:n_diff_stages]) if n_diff_stages > 0 else 1
    diff_bufs = np.zeros((max(n_diff_stages, 1), max_diff_len))
    diff_lens = np.array([diff_delays[i] for i in range(n_diff_stages)] or [1], dtype=np.int64)
    diff_gain = params.get("diffusion", 0.5)
    diff_gains = np.full(max(n_diff_stages, 1), diff_gain)
    diff_idxs = np.zeros(max(n_diff_stages, 1), dtype=np.int64)

    # --- FDN delay lines (enlarged buffer for modulation excursion) ---
    delay_times_base = np.array(params["delay_times"], dtype=np.float64)
    mod_depth_delay = np.array(params.get("mod_depth_delay", [0.0] * N), dtype=np.float64)
    # Buffer must accommodate base delay + max modulation excursion
    max_delay = int(np.max(delay_times_base + np.abs(mod_depth_delay))) + 4
    delay_buf_len = max_delay + 1
    delay_bufs = np.zeros((N, delay_buf_len))
    delay_write_idxs = np.zeros(N, dtype=np.int64)

    # --- Damping ---
    damping_coeffs_base = np.array(params["damping_coeffs"], dtype=np.float64)
    damping_y1 = np.zeros(N)

    # --- Gains ---
    feedback_gain = float(params["feedback_gain"])
    input_gains = np.array(params["input_gains"], dtype=np.float64)
    output_gains_base = np.array(params["output_gains"], dtype=np.float64)
    wet_dry = float(params["wet_dry"])
    saturation = float(params.get("saturation", 0.0))

    # --- DC blocker ---
    dc_R = 1.0 - 2.0 * np.pi * 5.0 / sample_rate
    dc_x1 = np.zeros(N)
    dc_y1 = np.zeros(N)

    # --- Stereo panning ---
    node_pans = np.array(params.get("node_pans",
        [-1.0, -0.714, -0.429, -0.143, 0.143, 0.429, 0.714, 1.0]),
        dtype=np.float64)
    stereo_width = float(params.get("stereo_width", 1.0))
    pan_gain_L = np.empty(N, dtype=np.float64)
    pan_gain_R = np.empty(N, dtype=np.float64)
    for i in range(N):
        pan = node_pans[i] * stereo_width
        angle = (pan + 1.0) * (np.pi / 4.0)
        pan_gain_L[i] = np.cos(angle)
        pan_gain_R[i] = np.sin(angle)

    # --- Modulation parameters ---
    # Structured modulation: global master rate * per-node multiplier
    master_rate = float(params.get("mod_master_rate", 1.0))  # Hz
    node_rate_mult = np.array(params.get("mod_node_rate_mult", [1.0] * N), dtype=np.float64)
    correlation = float(params.get("mod_correlation", 1.0))  # 1=sync, 0=independent
    mod_waveform = int(params.get("mod_waveform", 0))  # 0=sine, 1=tri, 2=S&H

    # Per-target modulation rates (master_rate * node_multiplier * target_scale)
    delay_rate_scale = float(params.get("mod_rate_scale_delay", 1.0))
    damping_rate_scale = float(params.get("mod_rate_scale_damping", 1.0))
    output_rate_scale = float(params.get("mod_rate_scale_output", 1.0))

    mod_rate_delay = np.array([master_rate * node_rate_mult[i] * delay_rate_scale for i in range(N)], dtype=np.float64)
    mod_rate_damping = np.array([master_rate * node_rate_mult[i] * damping_rate_scale for i in range(N)], dtype=np.float64)
    mod_rate_output = np.array([master_rate * node_rate_mult[i] * output_rate_scale for i in range(N)], dtype=np.float64)

    mod_depth_damping = np.array(params.get("mod_depth_damping", [0.0] * N), dtype=np.float64)
    mod_depth_output = np.array(params.get("mod_depth_output", [0.0] * N), dtype=np.float64)

    mod_depth_matrix = float(params.get("mod_depth_matrix", 0.0))
    mod_rate_matrix = float(params.get("mod_rate_matrix", master_rate))

    # Phase offsets: correlated (all same) vs independent (spread evenly)
    base_phases = np.array([i / N for i in range(N)], dtype=np.float64)
    mod_phases = base_phases * (1.0 - correlation)  # correlation=1 -> all phase 0

    output = np.empty((n_samples, 2), dtype=np.float64)

    _process_block_mod(
        input_audio.astype(np.float64), output,
        pre_delay_buf, pre_delay_len, pre_delay_samples, pre_delay_write_idx,
        diff_bufs, diff_lens, diff_gains, diff_idxs, n_diff_stages,
        delay_bufs, delay_buf_len, delay_times_base, delay_write_idxs,
        damping_coeffs_base, damping_y1,
        matrix, matrix2,
        feedback_gain, input_gains, output_gains_base, wet_dry,
        saturation,
        dc_x1, dc_y1, dc_R,
        pan_gain_L, pan_gain_R,
        mod_depth_delay, mod_rate_delay,
        mod_depth_damping, mod_rate_damping,
        mod_depth_output, mod_rate_output,
        mod_depth_matrix, mod_rate_matrix,
        mod_waveform, mod_phases,
        float(sample_rate),
    )

    return output
```

### Success Criteria:

#### Automated Verification:
- [x] `engine/numba_fdn_mod.py` compiles: `uv run --prerelease=allow python -c "from engine.numba_fdn_mod import render_fdn_mod"`
- [x] Static params (all mod_depth = 0) through `render_fdn_mod()` produces output similar to `render_fdn_fast()` (same character, may differ slightly due to float path through fractional delay)
- [x] Modulated params produce non-static output: variance across chunks of the output differs from static version
- [x] No NaN, Inf, or explosion with moderate modulation settings (depth=5 samples, rate=2 Hz)

#### Manual Verification:
- [ ] Render a test signal with delay modulation depth=5, rate=2 Hz — should hear chorus-like smearing, no metallic ringing
- [ ] Render with audio-rate modulation (rate=100 Hz, depth=3) — should hear FM-like inharmonic content
- [ ] Render with slow modulation (rate=0.1 Hz, depth=20) — should hear evolving character over seconds

**Implementation Note**: After completing this phase and all automated verification passes, pause here for manual confirmation.

---

## Phase 3: Params + Routing

### Overview
Extend the parameter schema with modulation parameters and update the routing in `engine/fdn.py` to automatically use the modulated path when modulation is active.

### Changes Required:

#### 1. Extend parameter schema
**File**: `engine/params.py`

Add modulation defaults to `default_params()`:

```python
# --- Modulation (Phase 3) ---
# Global master rate in Hz (all modulators derive from this)
"mod_master_rate": 0.0,  # 0 = no modulation (backward compatible)

# Per-node rate multipliers (integer ratios for rhythmic relationships)
"mod_node_rate_mult": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],

# Correlation: 1.0 = all nodes in sync, 0.0 = evenly spread phases
"mod_correlation": 1.0,

# Waveform: 0=sine, 1=triangle, 2=sample-and-hold
"mod_waveform": 0,

# Per-target modulation depth (0 = off)
# Delay: depth in samples (±depth around base delay time)
"mod_depth_delay": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
# Damping: depth as coefficient offset (±depth around base damping)
"mod_depth_damping": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
# Output gains: depth as multiplier offset (±depth * base gain)
"mod_depth_output": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
# Matrix blend depth: 0-1 (how far to blend toward matrix2)
"mod_depth_matrix": 0.0,

# Per-target rate scale factors (multiplied with master_rate * node_mult)
"mod_rate_scale_delay": 1.0,
"mod_rate_scale_damping": 1.0,
"mod_rate_scale_output": 1.0,
"mod_rate_matrix": 0.0,  # Hz, independent of master for flexibility

# Second matrix for topology modulation
"mod_matrix2_type": "random_orthogonal",
"mod_matrix2_seed": 137,
```

Add modulation ranges to `PARAM_RANGES`:

```python
"mod_master_rate": (0.0, 1000.0),   # Hz — spans slow to audio-rate
"mod_depth_delay": (0.0, 100.0),     # samples
"mod_depth_damping": (0.0, 0.5),     # coefficient offset
"mod_depth_output": (0.0, 1.0),      # gain multiplier offset
"mod_depth_matrix": (0.0, 1.0),      # blend fraction
"mod_correlation": (0.0, 1.0),
"mod_rate_scale_delay": (0.01, 10.0),
"mod_rate_scale_damping": (0.01, 10.0),
"mod_rate_scale_output": (0.01, 10.0),
"mod_rate_matrix": (0.0, 1000.0),
```

#### 2. Update routing
**File**: `engine/fdn.py`

```python
def render_fdn(input_audio: np.ndarray, params: dict) -> np.ndarray:
    # Check if any modulation is active
    has_mod = (
        params.get("mod_master_rate", 0.0) > 0.0 and (
            any(d > 0 for d in params.get("mod_depth_delay", [0]*8)) or
            any(d > 0 for d in params.get("mod_depth_damping", [0]*8)) or
            any(d > 0 for d in params.get("mod_depth_output", [0]*8)) or
            params.get("mod_depth_matrix", 0.0) > 0.0
        )
    )

    if has_mod:
        from engine.numba_fdn_mod import render_fdn_mod
        return render_fdn_mod(input_audio, params)
    else:
        from engine.numba_fdn import render_fdn_fast
        return render_fdn_fast(input_audio, params)
```

### Success Criteria:

#### Automated Verification:
- [x] `default_params()` returns all new modulation keys with zero/default values
- [x] `render_fdn(audio, default_params())` still uses the static path (backward compatible)
- [x] `render_fdn(audio, {**default_params(), "mod_master_rate": 2.0, "mod_depth_delay": [5.0]*8})` uses the modulated path and produces valid output
- [x] Existing presets load without error (new params filled with defaults)

#### Manual Verification:
- [ ] Not needed for this phase — infrastructure only

---

## Phase 4: GUI Modulation Controls

### Overview
Add modulation controls to the GUI. Add a "Modulation" section in the left column of the Parameters page, below the XY pad, with sliders for master rate, depth per target, waveform selector, correlation, and per-node rate multipliers.

### Changes Required:

#### 1. Add modulation section to GUI
**File**: `gui/gui.py`

In `_build_params_page()`, after the XY pad section (after line ~230), add a modulation section to the left column:

```python
# --- Modulation ---
row = self._section(f, row, "Modulation")

row = self._add_slider(f, row, "mod_master_rate", "Master Rate (Hz)", 0.0, 100.0, 0.0, length=220)
row = self._add_slider(f, row, "mod_depth_delay_global", "Delay Mod Depth (smp)", 0.0, 100.0, 0.0, length=220)
row = self._add_slider(f, row, "mod_depth_damping_global", "Damp Mod Depth", 0.0, 0.5, 0.0, length=220)
row = self._add_slider(f, row, "mod_depth_output_global", "Out Gain Mod Depth", 0.0, 1.0, 0.0, length=220)
row = self._add_slider(f, row, "mod_depth_matrix", "Matrix Mod Depth", 0.0, 1.0, 0.0, length=220)
row = self._add_slider(f, row, "mod_correlation", "Correlation", 0.0, 1.0, 1.0, length=220)

# Waveform selector
ttk.Label(f, text="Mod Waveform:").grid(row=row, column=0, sticky="w", pady=2)
self.mod_waveform_var = tk.StringVar(value="sine")
wf_combo = ttk.Combobox(f, textvariable=self.mod_waveform_var,
    values=["sine", "triangle", "sample_hold"], state="readonly", width=15)
wf_combo.grid(row=row, column=1, sticky="w", pady=2)
row += 1
```

In the right column, add per-node rate multipliers after the Node Pans section:

```python
# --- Per-node rate multipliers ---
row = self._section(f, row, "Mod Rate Multipliers")
self.mod_rate_mult_sliders = []
for i in range(8):
    row = self._add_node_slider(f, row, f"mrm_{i}", f"Node {i}", 0.25, 4.0, 1.0,
                                self.mod_rate_mult_sliders, fmt=".2f", length=slider_len)
```

#### 2. Update `_read_params_from_ui()`
**File**: `gui/gui.py`

Add modulation parameter reading:

```python
# Modulation
wf_map = {"sine": 0, "triangle": 1, "sample_hold": 2}
p["mod_master_rate"] = self.sliders["mod_master_rate"].get()
p["mod_waveform"] = wf_map.get(self.mod_waveform_var.get(), 0)
p["mod_correlation"] = self.sliders["mod_correlation"].get()
dd = self.sliders["mod_depth_delay_global"].get()
p["mod_depth_delay"] = [dd] * 8  # global depth applied to all nodes
ddamp = self.sliders["mod_depth_damping_global"].get()
p["mod_depth_damping"] = [ddamp] * 8
dout = self.sliders["mod_depth_output_global"].get()
p["mod_depth_output"] = [dout] * 8
p["mod_depth_matrix"] = self.sliders["mod_depth_matrix"].get()
p["mod_node_rate_mult"] = [s.get() for s in self.mod_rate_mult_sliders]
```

#### 3. Update `_write_params_to_ui()`
**File**: `gui/gui.py`

Add modulation parameter writing for preset loading:

```python
# Modulation
self.sliders["mod_master_rate"].set(p.get("mod_master_rate", 0.0))
dd = p.get("mod_depth_delay", [0.0] * 8)
self.sliders["mod_depth_delay_global"].set(dd[0] if isinstance(dd, list) else dd)
ddamp = p.get("mod_depth_damping", [0.0] * 8)
self.sliders["mod_depth_damping_global"].set(ddamp[0] if isinstance(ddamp, list) else ddamp)
dout = p.get("mod_depth_output", [0.0] * 8)
self.sliders["mod_depth_output_global"].set(dout[0] if isinstance(dout, list) else dout)
self.sliders["mod_depth_matrix"].set(p.get("mod_depth_matrix", 0.0))
self.sliders["mod_correlation"].set(p.get("mod_correlation", 1.0))
wf_rmap = {0: "sine", 1: "triangle", 2: "sample_hold"}
self.mod_waveform_var.set(wf_rmap.get(p.get("mod_waveform", 0), "sine"))
mrm = p.get("mod_node_rate_mult", [1.0] * 8)
for i in range(8):
    self.mod_rate_mult_sliders[i].set(mrm[i])
```

#### 4. Update `_on_randomize_knobs()`
**File**: `gui/gui.py`

Add randomization of modulation params:

```python
# Modulation randomization
self.sliders["mod_master_rate"].set(rng.uniform(0.0, 20.0))
self.sliders["mod_depth_delay_global"].set(rng.uniform(0.0, 30.0))
self.sliders["mod_depth_damping_global"].set(rng.uniform(0.0, 0.3))
self.sliders["mod_depth_output_global"].set(rng.uniform(0.0, 0.5))
self.sliders["mod_depth_matrix"].set(rng.uniform(0.0, 0.5))
self.sliders["mod_correlation"].set(rng.uniform(0.0, 1.0))
self.mod_waveform_var.set(rng.choice(["sine", "triangle", "sample_hold"]))
for i in range(8):
    self.mod_rate_mult_sliders[i].set(rng.choice([0.5, 1.0, 1.0, 2.0, 3.0]))
```

#### 5. Add XY pad modulation params and schedule diagram updates
**File**: `gui/gui.py`

Add `mod_master_rate` and `mod_depth_delay_global` to the XY pad parameter list so they can be assigned to axes. Also schedule diagram redraws when modulation sliders change.

### Success Criteria:

#### Automated Verification:
- [ ] GUI launches without errors: `uv run --prerelease=allow python gui/gui.py` (visual check that it opens)
- [ ] Modulation section visible with all sliders
- [ ] `_read_params_from_ui()` returns modulation keys
- [ ] Preset save/load round-trips modulation parameters correctly

#### Manual Verification:
- [ ] Adjust modulation sliders, press Play — hear modulated reverb
- [ ] Randomize & Play includes modulation variation
- [ ] Loading a static preset resets modulation to zero
- [ ] GUI layout is clean with no overlap or clipping

**Implementation Note**: After completing this phase and all automated verification passes, pause here for manual confirmation.

---

## Phase 5: Presets + Verification

### Overview
Create modulation presets showcasing the three timescales, update the Guide tab with modulation documentation, and verify the complete system end-to-end.

### Changes Required:

#### 1. Create modulation presets
**Directory**: `gui/presets/`

Create 3 presets demonstrating modulation:

**`lush_chorus_room.json`** — Medium/LFO timescale:
```json
{
  "delay_times": [1310, 1637, 1821, 2113, 2342, 2615, 2986, 3224],
  "damping_coeffs": [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
  "feedback_gain": 0.85,
  "input_gains": [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
  "output_gains": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
  "pre_delay": 441,
  "diffusion": 0.5,
  "diffusion_stages": 4,
  "diffusion_delays": [234, 349, 516, 710],
  "wet_dry": 0.5,
  "saturation": 0.0,
  "matrix_type": "householder",
  "node_pans": [-1.0, -0.714, -0.429, -0.143, 0.143, 0.429, 0.714, 1.0],
  "stereo_width": 1.0,
  "mod_master_rate": 2.0,
  "mod_depth_delay": [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
  "mod_depth_damping": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "mod_depth_output": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "mod_depth_matrix": 0.0,
  "mod_waveform": 0,
  "mod_correlation": 0.5,
  "mod_node_rate_mult": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
}
```

**`breathing_space.json`** — Slow timescale:
```json
{
  "delay_times": [2205, 2756, 3087, 3528, 3969, 4410, 5071, 5512],
  "damping_coeffs": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
  "feedback_gain": 0.92,
  "input_gains": [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
  "output_gains": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
  "pre_delay": 882,
  "diffusion": 0.55,
  "diffusion_stages": 4,
  "diffusion_delays": [234, 349, 516, 710],
  "wet_dry": 0.6,
  "saturation": 0.0,
  "matrix_type": "householder",
  "node_pans": [-1.0, -0.714, -0.429, -0.143, 0.143, 0.429, 0.714, 1.0],
  "stereo_width": 1.0,
  "mod_master_rate": 0.15,
  "mod_depth_delay": [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
  "mod_depth_damping": [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15],
  "mod_depth_output": [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
  "mod_depth_matrix": 0.3,
  "mod_waveform": 0,
  "mod_correlation": 0.3,
  "mod_node_rate_mult": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 1.0, 1.0]
}
```

**`fm_alien_texture.json`** — Audio-rate timescale:
```json
{
  "delay_times": [441, 551, 661, 772, 882, 993, 1103, 1213],
  "damping_coeffs": [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
  "feedback_gain": 0.88,
  "input_gains": [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
  "output_gains": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
  "pre_delay": 220,
  "diffusion": 0.4,
  "diffusion_stages": 4,
  "diffusion_delays": [234, 349, 516, 710],
  "wet_dry": 0.7,
  "saturation": 0.15,
  "matrix_type": "hadamard",
  "node_pans": [-1.0, -0.714, -0.429, -0.143, 0.143, 0.429, 0.714, 1.0],
  "stereo_width": 1.0,
  "mod_master_rate": 80.0,
  "mod_depth_delay": [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
  "mod_depth_damping": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "mod_depth_output": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "mod_depth_matrix": 0.0,
  "mod_waveform": 0,
  "mod_correlation": 0.0,
  "mod_node_rate_mult": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 1.0, 2.0]
}
```

#### 2. Update TODO.md
**File**: `TODO.md`

Check off completed Phase 3 items as they are verified.

#### 3. Update Guide tab
**File**: `gui/gui.py`

Add a "Modulation" section to the Guide tab content (in `_build_guide_page()`) documenting:
- Master Rate and what the three timescales do
- Delay Mod Depth and its effect
- Damping/Output/Matrix mod depth
- Correlation parameter
- Waveform types
- Per-node rate multipliers and rhythmic relationships

### Success Criteria:

#### Automated Verification:
- [ ] All 3 preset files are valid JSON and load without error
- [ ] `uv run --prerelease=allow python audio/render.py audio/test_signals/dry_chords.wav /tmp/test_mod.wav --preset gui/presets/lush_chorus_room.json` succeeds
- [ ] Static presets still work: `uv run --prerelease=allow python audio/render.py audio/test_signals/dry_chords.wav /tmp/test_static.wav --preset gui/presets/decent_room.json` succeeds
- [ ] GUI loads modulation presets and displays correct slider values

#### Manual Verification:
- [ ] `lush_chorus_room` preset sounds lush/chorused — no metallic ringing
- [ ] `breathing_space` preset evolves slowly — spatial character changes over 5+ seconds
- [ ] `fm_alien_texture` preset has inharmonic, FM-like spectral content
- [ ] Randomize & Play produces interesting modulated configurations
- [ ] Guide tab documents all modulation parameters

**Implementation Note**: After completing this phase and all automated verification passes, pause here for final manual confirmation.

---

## Testing Strategy

### Automated Tests:
- LFO waveform functions produce values in [-1, 1] range
- Fractional delay at integer positions matches integer delay
- Static params through modulated path produce similar output to static path
- Modulated params produce time-varying output (non-constant RMS across windows)
- No NaN/Inf with any modulation configuration
- Preset save/load round-trip preserves all modulation parameters

### Manual Listening Tests:
1. **A/B test**: Same preset with mod_master_rate=0 vs mod_master_rate=2, mod_depth_delay=5 — should hear clear difference
2. **Chorus test**: Medium room + 2 Hz delay mod, depth 5 samples — eliminates metallic ringing
3. **Slow evolve**: Long hall + 0.1 Hz everything — character drifts over 10 seconds
4. **FM territory**: Short delays + 80 Hz delay mod — alien spectral content
5. **Matrix breathing**: Enable matrix mod depth 0.5, rate 0.5 Hz — topology changes audibly

## Performance Considerations

- The modulated inner loop computes LFO phases per-sample per-node — this adds O(N) work per sample
- `n/sample_rate` float division in the inner loop: acceptable for Numba JIT, but could be optimized to phase accumulator if needed
- Linear interpolation for fractional delay adds one extra buffer read per node per sample
- Matrix blending when active doubles the matrix multiply work — only computed when `mod_depth_matrix > 0`
- Expected performance: still well within real-time for offline rendering; may need optimization for live audio

## References

- Research: `thoughts/shared/research/2026-02-06-time-varying-parameters-modulation.md`
- Phase 3 spec: `PLAN.md:191-255`
- Phase 3 TODO: `TODO.md:82-100`
- Static FDN: `engine/numba_fdn.py`
- Params schema: `engine/params.py`
- GUI: `gui/gui.py`
