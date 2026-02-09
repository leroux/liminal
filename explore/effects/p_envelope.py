"""P-series: Envelope effects (P001-P005).

P001 -- Envelope Reshaping
P002 -- Envelope Inversion
P003 -- Noise Gate with Decay
P004 -- Rhythmic Gain Sequencer
"""
import numpy as np
import numba


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

@numba.njit
def _extract_envelope(samples, smoothing_samples):
    """Extract amplitude envelope using a one-pole follower.

    Returns an envelope array the same length as samples.
    """
    n = len(samples)
    env = np.zeros(n, dtype=np.float32)
    if smoothing_samples < 1:
        smoothing_samples = 1
    coeff = np.float32(1.0 - 1.0 / smoothing_samples)
    prev = np.float32(0.0)
    for i in range(n):
        rect = np.float32(abs(samples[i]))
        if rect > prev:
            prev = np.float32(0.6) * prev + np.float32(0.4) * rect  # fast attack
        else:
            prev = coeff * prev + (np.float32(1.0) - coeff) * rect
        env[i] = prev
    return env


# ---------------------------------------------------------------------------
# P001 -- Envelope Reshaping
# ---------------------------------------------------------------------------

@numba.njit
def _generate_shape(n, shape_id):
    """Generate a target envelope shape of length n.

    shape_id mapping:
        0 = ramp_up
        1 = ramp_down
        2 = triangle
        3 = pulse
        4 = adsr
    """
    out = np.zeros(n, dtype=np.float32)
    if shape_id == 0:
        # ramp_up
        for i in range(n):
            out[i] = np.float32(i) / np.float32(max(n - 1, 1))
    elif shape_id == 1:
        # ramp_down
        for i in range(n):
            out[i] = np.float32(1.0) - np.float32(i) / np.float32(max(n - 1, 1))
    elif shape_id == 2:
        # triangle
        mid = n // 2
        for i in range(n):
            if i <= mid:
                out[i] = np.float32(i) / np.float32(max(mid, 1))
            else:
                out[i] = np.float32(1.0) - np.float32(i - mid) / np.float32(max(n - mid - 1, 1))
    elif shape_id == 3:
        # pulse: 50% duty cycle
        half = n // 2
        for i in range(n):
            if i < half:
                out[i] = np.float32(1.0)
            else:
                out[i] = np.float32(0.0)
    elif shape_id == 4:
        # adsr: attack 10%, decay 15%, sustain level 0.6 for 50%, release 25%
        a_end = n // 10
        d_end = a_end + (n * 15) // 100
        r_start = n - n // 4
        sustain_level = np.float32(0.6)
        for i in range(n):
            if i < a_end:
                out[i] = np.float32(i) / np.float32(max(a_end, 1))
            elif i < d_end:
                frac = np.float32(i - a_end) / np.float32(max(d_end - a_end, 1))
                out[i] = np.float32(1.0) - frac * (np.float32(1.0) - sustain_level)
            elif i < r_start:
                out[i] = sustain_level
            else:
                frac = np.float32(i - r_start) / np.float32(max(n - r_start - 1, 1))
                out[i] = sustain_level * (np.float32(1.0) - frac)
    return out


@numba.njit
def _apply_envelope_reshape(samples, env, target_shape):
    """Normalize original envelope then replace with target shape."""
    n = len(samples)
    out = np.zeros(n, dtype=np.float32)
    # Find peak of original envelope for normalization
    peak = np.float32(0.0)
    for i in range(n):
        if env[i] > peak:
            peak = env[i]
    if peak < np.float32(1e-10):
        peak = np.float32(1e-10)

    for i in range(n):
        # Divide out old envelope, multiply by new shape
        norm_env = env[i] / peak
        if norm_env < np.float32(1e-10):
            norm_env = np.float32(1e-10)
        gain = target_shape[i] / norm_env
        # Clamp gain to avoid extreme amplification
        if gain > np.float32(50.0):
            gain = np.float32(50.0)
        out[i] = samples[i] * gain
    return out


_SHAPE_MAP = {
    'ramp_up': 0,
    'ramp_down': 1,
    'triangle': 2,
    'pulse': 3,
    'adsr': 4,
}


def effect_p001_envelope_reshaping(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Envelope reshaping: extract the original envelope, normalize it,
    then impose a new target shape.

    Parameters
    ----------
    new_shape : str
        Target envelope shape.  One of 'ramp_up', 'ramp_down', 'triangle',
        'pulse', 'adsr'.
    """
    new_shape = str(params.get('new_shape', 'triangle'))
    shape_id = _SHAPE_MAP.get(new_shape, 2)

    samples = samples.astype(np.float32)
    n = len(samples)

    smoothing_samples = max(1, int(0.01 * sr))  # 10 ms smoothing
    env = _extract_envelope(samples, smoothing_samples)
    target = _generate_shape(n, shape_id)

    return _apply_envelope_reshape(samples, env, target)


def variants_p001():
    return [
        {'new_shape': 'ramp_up'},
        {'new_shape': 'ramp_down'},
        {'new_shape': 'triangle'},
        {'new_shape': 'pulse'},
        {'new_shape': 'adsr'},
    ]


# ---------------------------------------------------------------------------
# P002 -- Envelope Inversion
# ---------------------------------------------------------------------------

@numba.njit
def _invert_envelope_kernel(samples, env, max_gain):
    """Loud becomes quiet and quiet becomes loud.

    gain = (1 - normalized_envelope) * max_gain, clamped.
    """
    n = len(samples)
    out = np.zeros(n, dtype=np.float32)
    # Find peak of envelope
    peak = np.float32(0.0)
    for i in range(n):
        if env[i] > peak:
            peak = env[i]
    if peak < np.float32(1e-10):
        peak = np.float32(1e-10)

    for i in range(n):
        norm = env[i] / peak
        # Invert: when norm is high, gain is low and vice versa
        gain = (np.float32(1.0) - norm) * max_gain
        if gain > max_gain:
            gain = max_gain
        if gain < np.float32(0.0):
            gain = np.float32(0.0)
        out[i] = samples[i] * gain
    return out


def effect_p002_envelope_inversion(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Envelope inversion: loud parts become quiet, quiet parts become loud.

    Parameters
    ----------
    smoothing_ms : float [5, 50]
        Envelope follower smoothing time in milliseconds.
    max_gain : float [10, 100]
        Maximum gain applied to quiet sections.
    """
    smoothing_ms = float(params.get('smoothing_ms', 20))
    max_gain = np.float32(params.get('max_gain', 30))

    samples = samples.astype(np.float32)
    smoothing_samples = max(1, int(smoothing_ms * sr / 1000.0))
    env = _extract_envelope(samples, smoothing_samples)

    return _invert_envelope_kernel(samples, env, max_gain)


def variants_p002():
    return [
        # Fast follower, moderate gain
        {'smoothing_ms': 5, 'max_gain': 20},
        # Default
        {'smoothing_ms': 20, 'max_gain': 30},
        # Slow follower, high gain -- dreamy swells
        {'smoothing_ms': 50, 'max_gain': 60},
        # Very fast, extreme gain -- noisy artifacts
        {'smoothing_ms': 5, 'max_gain': 100},
        # Slow, gentle inversion
        {'smoothing_ms': 40, 'max_gain': 10},
        # Medium speed, moderate boost
        {'smoothing_ms': 15, 'max_gain': 40},
    ]


# ---------------------------------------------------------------------------
# P003 -- Noise Gate with Decay
# ---------------------------------------------------------------------------

@numba.njit
def _noise_gate_kernel(samples, env, threshold_lin, threshold_open_lin,
                       decay_coeff):
    """Gate with hysteresis and exponential decay below threshold.

    When envelope drops below threshold_lin, apply exponential decay.
    Gate reopens when envelope exceeds threshold_open_lin (threshold + hysteresis).
    """
    n = len(samples)
    out = np.zeros(n, dtype=np.float32)
    gate_open = True
    decay_gain = np.float32(1.0)

    for i in range(n):
        level = env[i]
        if gate_open:
            if level < threshold_lin:
                gate_open = False
                decay_gain = np.float32(1.0)
        else:
            if level > threshold_open_lin:
                gate_open = True
                decay_gain = np.float32(1.0)

        if gate_open:
            out[i] = samples[i]
        else:
            out[i] = samples[i] * decay_gain
            decay_gain *= decay_coeff
            if decay_gain < np.float32(1e-8):
                decay_gain = np.float32(1e-8)
    return out


def effect_p003_noise_gate_decay(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Noise gate with exponential decay: below threshold the signal
    decays exponentially rather than cutting abruptly.

    Parameters
    ----------
    threshold_db : float [-50, -20]
        Gate threshold in dB.
    decay_ms : float [50, 500]
        Time constant for the exponential decay when gated.
    hysteresis_db : float [2, 6]
        Hysteresis in dB.  The gate reopens at threshold + hysteresis.
    """
    threshold_db = float(params.get('threshold_db', -30))
    decay_ms = float(params.get('decay_ms', 200))
    hysteresis_db = float(params.get('hysteresis_db', 3))

    samples = samples.astype(np.float32)

    # Envelope extraction (5 ms smoothing for fast response)
    smoothing_samples = max(1, int(0.005 * sr))
    env = _extract_envelope(samples, smoothing_samples)

    threshold_lin = np.float32(10.0 ** (threshold_db / 20.0))
    threshold_open_lin = np.float32(10.0 ** ((threshold_db + hysteresis_db) / 20.0))

    # Decay coefficient: per-sample multiplier so that amplitude halves in decay_ms
    # exp(-ln(2) * 1000 / (decay_ms * sr))
    if decay_ms > 0:
        decay_coeff = np.float32(np.exp(-np.log(2.0) / (decay_ms * 0.001 * sr)))
    else:
        decay_coeff = np.float32(0.0)

    return _noise_gate_kernel(samples, env, threshold_lin, threshold_open_lin,
                              decay_coeff)


def variants_p003():
    return [
        # Gentle gate, slow decay
        {'threshold_db': -40, 'decay_ms': 400, 'hysteresis_db': 4},
        # Standard gate
        {'threshold_db': -30, 'decay_ms': 200, 'hysteresis_db': 3},
        # Aggressive gate, fast cutoff
        {'threshold_db': -20, 'decay_ms': 50, 'hysteresis_db': 2},
        # Sensitive gate, very slow decay (tail-like)
        {'threshold_db': -45, 'decay_ms': 500, 'hysteresis_db': 5},
        # Tight percussive gate
        {'threshold_db': -25, 'decay_ms': 80, 'hysteresis_db': 6},
        # Wide hysteresis, medium decay
        {'threshold_db': -35, 'decay_ms': 300, 'hysteresis_db': 6},
    ]


# ---------------------------------------------------------------------------
# P004 -- Rhythmic Gain Sequencer
# ---------------------------------------------------------------------------

@numba.njit
def _rhythmic_gain_kernel(samples, step_samples, pattern, num_steps):
    """Apply a 16-step gain pattern at the given tempo.

    Each step lasts step_samples.  Pattern repeats.  Short crossfade
    (64 samples) between steps to avoid clicks.
    """
    n = len(samples)
    out = np.zeros(n, dtype=np.float32)
    fade_len = 64
    if fade_len > step_samples // 2:
        fade_len = max(1, step_samples // 2)

    for i in range(n):
        step_idx = (i // step_samples) % num_steps
        pos_in_step = i % step_samples

        current_gain = pattern[step_idx]
        next_step_idx = (step_idx + 1) % num_steps
        next_gain = pattern[next_step_idx]

        # Crossfade at the end of each step
        if pos_in_step >= step_samples - fade_len:
            fade_pos = pos_in_step - (step_samples - fade_len)
            frac = np.float32(fade_pos) / np.float32(fade_len)
            gain = current_gain * (np.float32(1.0) - frac) + next_gain * frac
        else:
            gain = current_gain

        out[i] = samples[i] * gain
    return out


def effect_p004_rhythmic_gain_sequencer(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Rhythmic gain sequencer: apply a 16-step gain pattern at tempo.

    Parameters
    ----------
    step_ms : float [50, 300]
        Duration of each step in milliseconds.
    pattern : list of 16 floats
        Gain values for each step (0.0 = silence, 1.0 = full).
    """
    step_ms = float(params.get('step_ms', 125))
    pattern_input = params.get('pattern', None)

    if pattern_input is None:
        # Default: alternating 1.0 and 0.0
        pattern = np.zeros(16, dtype=np.float32)
        for i in range(16):
            pattern[i] = np.float32(1.0) if i % 2 == 0 else np.float32(0.0)
    else:
        pattern = np.array(pattern_input, dtype=np.float32)
        # Pad or truncate to 16 steps
        if len(pattern) < 16:
            padded = np.ones(16, dtype=np.float32)
            for i in range(len(pattern)):
                padded[i] = pattern[i]
            pattern = padded
        elif len(pattern) > 16:
            pattern = pattern[:16].copy()

    samples = samples.astype(np.float32)
    step_samples = max(1, int(step_ms * sr / 1000.0))

    return _rhythmic_gain_kernel(samples, step_samples, pattern, 16)


def variants_p004():
    return [
        # Classic trance gate (alternating on/off)
        {'step_ms': 125, 'pattern': [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
                                     1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]},
        # Funky syncopation
        {'step_ms': 100, 'pattern': [1.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.8,
                                     1.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0, 0.5]},
        # Slow swell and duck
        {'step_ms': 200, 'pattern': [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.0,
                                     1.0, 0.9, 0.7, 0.5, 0.3, 0.1, 0.0, 0.0]},
        # Staccato bursts
        {'step_ms': 60, 'pattern': [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                    1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]},
        # Half-time pulse
        {'step_ms': 250, 'pattern': [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
                                     1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]},
        # Crescendo pattern
        {'step_ms': 125, 'pattern': [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                                     0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.0, 1.0]},
        # Random-feel accents
        {'step_ms': 150, 'pattern': [1.0, 0.2, 0.0, 0.8, 0.0, 0.5, 1.0, 0.0,
                                     0.3, 1.0, 0.0, 0.0, 0.7, 0.0, 1.0, 0.4]},
    ]


# ---------------------------------------------------------------------------
# P005 -- Live Buffer Freeze / Stutter Hold
# Captures a loop at a trigger point and crossfades into repeating it.
# Distinct from granular freeze (I002): this captures a clean loop with
# crossfade, producing a smooth sustain rather than granular texture.
# ---------------------------------------------------------------------------

@numba.njit
def _buffer_freeze_kernel(samples, sr, freeze_pos, loop_ms, fade_ms, hold_ms):
    """Capture a loop at freeze_pos, crossfade in, hold, crossfade out."""
    n = len(samples)
    out = np.zeros(n, dtype=np.float32)

    loop_samps = max(64, int(loop_ms * 0.001 * sr))
    fade_samps = max(1, min(int(fade_ms * 0.001 * sr), loop_samps // 2))
    hold_samps = max(1, int(hold_ms * 0.001 * sr))

    freeze_start = int(freeze_pos * n)
    if freeze_start + loop_samps > n:
        freeze_start = max(0, n - loop_samps)

    # Extract loop buffer with crossfade at boundaries for seamless looping
    loop_buf = np.zeros(loop_samps, dtype=np.float32)
    for j in range(loop_samps):
        idx = freeze_start + j
        if idx < n:
            loop_buf[j] = samples[idx]

    # Apply crossfade at loop boundary for seamless repeat
    for j in range(fade_samps):
        frac = np.float32(j) / np.float32(fade_samps)
        # Fade end of loop
        loop_buf[loop_samps - 1 - j] *= frac
        # This creates a smooth loop point

    # Build output: original -> crossfade into freeze -> hold -> crossfade out
    fade_in_start = freeze_start
    hold_end = fade_in_start + fade_samps + hold_samps
    fade_out_end = hold_end + fade_samps

    for i in range(n):
        if i < fade_in_start:
            # Before freeze: pass through original
            out[i] = samples[i]
        elif i < fade_in_start + fade_samps:
            # Crossfade from original to frozen loop
            frac = np.float32(i - fade_in_start) / np.float32(fade_samps)
            loop_idx = (i - fade_in_start) % loop_samps
            out[i] = (np.float32(1.0) - frac) * samples[i] + frac * loop_buf[loop_idx]
        elif i < hold_end:
            # Frozen loop region
            loop_idx = (i - fade_in_start) % loop_samps
            out[i] = loop_buf[loop_idx]
        elif i < fade_out_end:
            # Crossfade back to original
            frac = np.float32(i - hold_end) / np.float32(fade_samps)
            loop_idx = (i - fade_in_start) % loop_samps
            orig = samples[i] if i < n else np.float32(0.0)
            out[i] = (np.float32(1.0) - frac) * loop_buf[loop_idx] + frac * orig
        else:
            # After freeze: pass through original
            if i < n:
                out[i] = samples[i]

    return out


def effect_p005_buffer_freeze(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Buffer freeze: capture a clean loop at a trigger point, crossfade in,
    hold the loop for a duration, then crossfade back out."""
    freeze_pos = float(params.get('freeze_pos', 0.3))
    loop_ms = float(params.get('loop_ms', 100))
    fade_ms = float(params.get('fade_ms', 20))
    hold_ms = float(params.get('hold_ms', 500))

    return _buffer_freeze_kernel(
        samples.astype(np.float32), sr, freeze_pos, loop_ms, fade_ms, hold_ms
    )


def variants_p005():
    return [
        {'freeze_pos': 0.2, 'loop_ms': 50, 'fade_ms': 10, 'hold_ms': 300},
        {'freeze_pos': 0.3, 'loop_ms': 100, 'fade_ms': 20, 'hold_ms': 500},
        {'freeze_pos': 0.5, 'loop_ms': 200, 'fade_ms': 30, 'hold_ms': 800},
        {'freeze_pos': 0.4, 'loop_ms': 30, 'fade_ms': 5, 'hold_ms': 1000},
        {'freeze_pos': 0.7, 'loop_ms': 150, 'fade_ms': 50, 'hold_ms': 400},
        {'freeze_pos': 0.1, 'loop_ms': 80, 'fade_ms': 15, 'hold_ms': 2000},
    ]
