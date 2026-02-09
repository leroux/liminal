"""E-series effects: Filters (E001-E012).

Biquad parametric EQ, state variable filter, Moog ladder, comb filter,
formant filter, vowel morph, auto-wah, resonant sweep, multi-mode
crossfade, and cascaded detuned resonators.
"""
import numpy as np
import numba


# ---------------------------------------------------------------------------
# Shared numba kernels
# ---------------------------------------------------------------------------

@numba.njit
def _biquad_kernel(samples, b0, b1, b2, a1, a2):
    """Direct Form II Transposed biquad filter (sample-by-sample)."""
    n = len(samples)
    out = np.empty(n, dtype=np.float32)
    z1 = np.float32(0.0)
    z2 = np.float32(0.0)
    for i in range(n):
        x = samples[i]
        y = b0 * x + z1
        z1 = b1 * x - a1 * y + z2
        z2 = b2 * x - a2 * y
        out[i] = y
    return out


def _biquad_coeffs(filter_type, freq_hz, sr, Q, gain_db=0.0):
    """Audio EQ Cookbook biquad coefficient calculation.

    Returns (b0, b1, b2, a1, a2) normalised by a0.
    """
    freq_hz = float(np.clip(freq_hz, 20.0, sr * 0.499))
    Q = float(np.clip(Q, 0.1, 100.0))
    w0 = 2.0 * np.pi * freq_hz / sr
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    alpha = sin_w0 / (2.0 * Q)

    if filter_type == 'lpf':
        b0 = (1.0 - cos_w0) / 2.0
        b1 = 1.0 - cos_w0
        b2 = (1.0 - cos_w0) / 2.0
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha
    elif filter_type == 'hpf':
        b0 = (1.0 + cos_w0) / 2.0
        b1 = -(1.0 + cos_w0)
        b2 = (1.0 + cos_w0) / 2.0
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha
    elif filter_type == 'bpf':
        b0 = alpha
        b1 = 0.0
        b2 = -alpha
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha
    elif filter_type == 'notch':
        b0 = 1.0
        b1 = -2.0 * cos_w0
        b2 = 1.0
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha
    elif filter_type == 'peak':
        A = 10.0 ** (gain_db / 40.0)
        b0 = 1.0 + alpha * A
        b1 = -2.0 * cos_w0
        b2 = 1.0 - alpha * A
        a0 = 1.0 + alpha / A
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha / A
    else:
        raise ValueError(f"Unknown filter_type: {filter_type}")

    inv_a0 = 1.0 / a0
    return (np.float32(b0 * inv_a0), np.float32(b1 * inv_a0),
            np.float32(b2 * inv_a0), np.float32(a1 * inv_a0),
            np.float32(a2 * inv_a0))


# ---------------------------------------------------------------------------
# E001 -- Biquad Filter (Parametric EQ)
# ---------------------------------------------------------------------------

def effect_e001_biquad_filter(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Parametric EQ biquad filter (Direct Form II Transposed).

    Params:
        filter_type: 'lpf', 'hpf', 'bpf', 'notch', 'peak'  (default 'lpf')
        freq_hz:     centre / cutoff frequency  [20, 20000]   (default 1000)
        Q:           quality factor              [0.1, 20]     (default 1.0)
        gain_db:     gain for peak type          [-24, 24]     (default 0.0)
    """
    filter_type = params.get('filter_type', 'lpf')
    freq_hz = params.get('freq_hz', 1000.0)
    Q = params.get('Q', 1.0)
    gain_db = params.get('gain_db', 0.0)

    b0, b1, b2, a1, a2 = _biquad_coeffs(filter_type, freq_hz, sr, Q, gain_db)
    return _biquad_kernel(samples.astype(np.float32), b0, b1, b2, a1, a2)


def variants_e001():
    return [
        {'filter_type': 'lpf', 'freq_hz': 400.0, 'Q': 0.707},
        {'filter_type': 'lpf', 'freq_hz': 2000.0, 'Q': 5.0},
        {'filter_type': 'hpf', 'freq_hz': 800.0, 'Q': 1.0},
        {'filter_type': 'bpf', 'freq_hz': 1200.0, 'Q': 8.0},
        {'filter_type': 'notch', 'freq_hz': 3000.0, 'Q': 10.0},
        {'filter_type': 'peak', 'freq_hz': 2500.0, 'Q': 2.0, 'gain_db': 12.0},
        {'filter_type': 'peak', 'freq_hz': 500.0, 'Q': 4.0, 'gain_db': -18.0},
        {'filter_type': 'lpf', 'freq_hz': 150.0, 'Q': 15.0},
    ]


# ---------------------------------------------------------------------------
# E002 -- State Variable Filter
# ---------------------------------------------------------------------------

@numba.njit
def _svf_kernel(samples, f_coeff, q_inv):
    """State variable filter kernel.

    hp = x - lp - q_inv * bp
    bp += f * hp
    lp += f * bp
    Returns (lp, hp, bp, notch) interleaved in a (N, 4) array.
    """
    n = len(samples)
    lp_out = np.empty(n, dtype=np.float32)
    hp_out = np.empty(n, dtype=np.float32)
    bp_out = np.empty(n, dtype=np.float32)
    notch_out = np.empty(n, dtype=np.float32)

    lp = np.float32(0.0)
    bp = np.float32(0.0)

    for i in range(n):
        x = samples[i]
        hp = x - lp - q_inv * bp
        bp = bp + f_coeff * hp
        lp = lp + f_coeff * bp
        notch = x - q_inv * bp  # lp + hp simplified

        # Clamp to avoid instability
        if bp > np.float32(10.0):
            bp = np.float32(10.0)
        elif bp < np.float32(-10.0):
            bp = np.float32(-10.0)
        if lp > np.float32(10.0):
            lp = np.float32(10.0)
        elif lp < np.float32(-10.0):
            lp = np.float32(-10.0)

        lp_out[i] = lp
        hp_out[i] = hp
        bp_out[i] = bp
        notch_out[i] = notch

    return lp_out, hp_out, bp_out, notch_out


def effect_e002_state_variable_filter(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """State variable filter with selectable output mode.

    Params:
        cutoff_hz:   cutoff frequency   [20, 20000]  (default 1000)
        Q:           quality factor     [0.5, 30]     (default 2.0)
        output_type: 'lp', 'hp', 'bp', 'notch'       (default 'lp')
    """
    cutoff_hz = float(np.clip(params.get('cutoff_hz', 1000.0), 20.0, sr * 0.499))
    Q = float(np.clip(params.get('Q', 2.0), 0.5, 30.0))
    output_type = params.get('output_type', 'lp')

    f_coeff = np.float32(2.0 * np.sin(np.pi * cutoff_hz / sr))
    q_inv = np.float32(1.0 / Q)

    lp, hp, bp, notch = _svf_kernel(samples.astype(np.float32), f_coeff, q_inv)

    if output_type == 'hp':
        return hp
    elif output_type == 'bp':
        return bp
    elif output_type == 'notch':
        return notch
    else:
        return lp


def variants_e002():
    return [
        {'cutoff_hz': 500.0, 'Q': 2.0, 'output_type': 'lp'},
        {'cutoff_hz': 3000.0, 'Q': 8.0, 'output_type': 'lp'},
        {'cutoff_hz': 1000.0, 'Q': 1.0, 'output_type': 'hp'},
        {'cutoff_hz': 1500.0, 'Q': 12.0, 'output_type': 'bp'},
        {'cutoff_hz': 2000.0, 'Q': 15.0, 'output_type': 'notch'},
        {'cutoff_hz': 200.0, 'Q': 25.0, 'output_type': 'bp'},
    ]


# ---------------------------------------------------------------------------
# E003 -- Moog Ladder Filter
# ---------------------------------------------------------------------------

@numba.njit
def _moog_ladder_kernel(samples, cutoff_norm, resonance):
    """4-cascaded one-pole stages with tanh nonlinearity and feedback.

    cutoff_norm = 2 * sin(pi * cutoff_hz / sr), clamped.
    resonance in [0, 4].
    """
    n = len(samples)
    out = np.empty(n, dtype=np.float32)
    s0 = np.float32(0.0)
    s1 = np.float32(0.0)
    s2 = np.float32(0.0)
    s3 = np.float32(0.0)

    for i in range(n):
        x = samples[i]
        # Feedback: subtract resonance * stage-4 output
        x = x - resonance * s3
        # tanh saturation on input
        x = np.float32(np.tanh(x))

        # Stage 1
        s0 = s0 + cutoff_norm * (x - s0)
        # Stage 2
        s1 = s1 + cutoff_norm * (s0 - s1)
        # Stage 3
        s2 = s2 + cutoff_norm * (s1 - s2)
        # Stage 4
        s3 = s3 + cutoff_norm * (s2 - s3)

        out[i] = s3

    return out


def effect_e003_moog_ladder(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Moog ladder filter: 4 cascaded one-pole stages with tanh and feedback.

    Params:
        cutoff_hz:  cutoff frequency  [20, 20000]  (default 1000)
        resonance:  feedback amount   [0, 4]        (default 2.0)
    """
    cutoff_hz = float(np.clip(params.get('cutoff_hz', 1000.0), 20.0, sr * 0.499))
    resonance = float(np.clip(params.get('resonance', 2.0), 0.0, 4.0))

    cutoff_norm = np.float32(2.0 * np.sin(np.pi * cutoff_hz / sr))
    resonance_f = np.float32(resonance)

    return _moog_ladder_kernel(samples.astype(np.float32), cutoff_norm, resonance_f)


def variants_e003():
    return [
        {'cutoff_hz': 300.0, 'resonance': 0.0},
        {'cutoff_hz': 800.0, 'resonance': 2.0},
        {'cutoff_hz': 1500.0, 'resonance': 3.5},
        {'cutoff_hz': 5000.0, 'resonance': 1.0},
        {'cutoff_hz': 200.0, 'resonance': 3.8},
        {'cutoff_hz': 3000.0, 'resonance': 0.5},
    ]


# ---------------------------------------------------------------------------
# E004 -- Comb Filter
# ---------------------------------------------------------------------------

@numba.njit
def _comb_feedback_kernel(samples, delay_samples, g):
    """Feedback comb filter: y[n] = x[n] + g * y[n - delay]."""
    n = len(samples)
    out = np.empty(n, dtype=np.float32)
    buf_len = max(delay_samples, 1)
    buf = np.zeros(buf_len, dtype=np.float32)
    write_pos = 0

    for i in range(n):
        read_pos = (write_pos - delay_samples) % buf_len
        delayed = buf[read_pos]
        y = samples[i] + g * delayed
        # Clamp to prevent blowup
        if y > np.float32(10.0):
            y = np.float32(10.0)
        elif y < np.float32(-10.0):
            y = np.float32(-10.0)
        buf[write_pos] = y
        out[i] = y
        write_pos = (write_pos + 1) % buf_len

    return out


@numba.njit
def _comb_feedforward_kernel(samples, delay_samples, g):
    """Feedforward comb filter: y[n] = x[n] + g * x[n - delay]."""
    n = len(samples)
    out = np.empty(n, dtype=np.float32)
    buf_len = max(delay_samples, 1)
    buf = np.zeros(buf_len, dtype=np.float32)
    write_pos = 0

    for i in range(n):
        read_pos = (write_pos - delay_samples) % buf_len
        delayed = buf[read_pos]
        out[i] = samples[i] + g * delayed
        buf[write_pos] = samples[i]
        write_pos = (write_pos + 1) % buf_len

    return out


def effect_e004_comb_filter(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Comb filter (feedforward or feedback).

    Params:
        freq_hz: comb frequency (delay = sr / freq_hz)  [50, 2000] (default 200)
        g:       gain coefficient                         [-0.99, 0.99] (default 0.7)
        mode:    'feedback' or 'feedforward'              (default 'feedback')
    """
    freq_hz = float(np.clip(params.get('freq_hz', 200.0), 50.0, 2000.0))
    g = float(np.clip(params.get('g', 0.7), -0.99, 0.99))
    mode = params.get('mode', 'feedback')

    delay_samples = max(int(round(sr / freq_hz)), 1)
    g_f = np.float32(g)
    inp = samples.astype(np.float32)

    if mode == 'feedforward':
        return _comb_feedforward_kernel(inp, delay_samples, g_f)
    else:
        return _comb_feedback_kernel(inp, delay_samples, g_f)


def variants_e004():
    return [
        {'freq_hz': 200.0, 'g': 0.8, 'mode': 'feedback'},
        {'freq_hz': 500.0, 'g': 0.6, 'mode': 'feedback'},
        {'freq_hz': 100.0, 'g': 0.9, 'mode': 'feedback'},
        {'freq_hz': 300.0, 'g': 0.5, 'mode': 'feedforward'},
        {'freq_hz': 800.0, 'g': -0.7, 'mode': 'feedforward'},
        {'freq_hz': 150.0, 'g': -0.85, 'mode': 'feedback'},
    ]


# ---------------------------------------------------------------------------
# E005 -- Formant Filter
# ---------------------------------------------------------------------------

# Vowel formant frequencies (F1, F2, F3) in Hz -- standard male voice approximations
_FORMANT_TABLE = {
    'a': (800.0, 1200.0, 2800.0),
    'e': (400.0, 2250.0, 2800.0),
    'i': (280.0, 2600.0, 3500.0),
    'o': (500.0, 800.0, 2800.0),
    'u': (320.0, 800.0, 2500.0),
}

# Relative amplitudes for each formant (linear)
_FORMANT_AMPS = {
    'a': (1.0, 0.5, 0.25),
    'e': (1.0, 0.6, 0.3),
    'i': (1.0, 0.4, 0.2),
    'o': (1.0, 0.5, 0.2),
    'u': (1.0, 0.4, 0.15),
}


def effect_e005_formant_filter(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """3 parallel bandpass filters tuned to vowel formant frequencies.

    Params:
        vowel: 'a', 'e', 'i', 'o', 'u'  (default 'a')
        Q:     quality factor             [5, 20]  (default 10)
    """
    vowel = params.get('vowel', 'a')
    Q = float(np.clip(params.get('Q', 10.0), 5.0, 20.0))

    if vowel not in _FORMANT_TABLE:
        vowel = 'a'

    freqs = _FORMANT_TABLE[vowel]
    amps = _FORMANT_AMPS[vowel]
    inp = samples.astype(np.float32)
    n = len(inp)
    out = np.zeros(n, dtype=np.float32)

    for freq, amp in zip(freqs, amps):
        b0, b1, b2, a1, a2 = _biquad_coeffs('bpf', freq, sr, Q)
        filtered = _biquad_kernel(inp, b0, b1, b2, a1, a2)
        out = out + np.float32(amp) * filtered

    return out


def variants_e005():
    return [
        {'vowel': 'a', 'Q': 10.0},
        {'vowel': 'e', 'Q': 10.0},
        {'vowel': 'i', 'Q': 12.0},
        {'vowel': 'o', 'Q': 8.0},
        {'vowel': 'u', 'Q': 15.0},
        {'vowel': 'a', 'Q': 20.0},
        {'vowel': 'i', 'Q': 5.0},
    ]


# ---------------------------------------------------------------------------
# E006 -- Vowel Morph
# ---------------------------------------------------------------------------

@numba.njit
def _vowel_morph_kernel(samples, freqs_from, amps_from, freqs_to, amps_to,
                        morph_rate_hz, sr, Q):
    """Morph between two sets of 3 formant bandpass filters using an LFO."""
    n = len(samples)
    out = np.empty(n, dtype=np.float32)

    # Pre-compute biquad coefficients for all 6 filters (3 per vowel)
    # We'll recompute coefficients per-sample to interpolate cutoff,
    # but that's expensive. Instead, crossfade the outputs of fixed filters.

    # Process: run all 6 bandpasses, crossfade outputs with LFO
    bp_from_0 = np.zeros(n, dtype=np.float32)
    bp_from_1 = np.zeros(n, dtype=np.float32)
    bp_from_2 = np.zeros(n, dtype=np.float32)
    bp_to_0 = np.zeros(n, dtype=np.float32)
    bp_to_1 = np.zeros(n, dtype=np.float32)
    bp_to_2 = np.zeros(n, dtype=np.float32)

    # We'll compute these outside njit, so just do the crossfade here
    phase_inc = np.float32(morph_rate_hz / sr)
    phase = np.float32(0.0)

    for i in range(n):
        # LFO: 0..1 using raised cosine for smooth morph
        morph = np.float32(0.5) * (np.float32(1.0) - np.float32(np.cos(
            np.float32(2.0) * np.float32(np.pi) * phase)))

        val_from = (amps_from[0] * bp_from_0[i] +
                    amps_from[1] * bp_from_1[i] +
                    amps_from[2] * bp_from_2[i])
        val_to = (amps_to[0] * bp_to_0[i] +
                  amps_to[1] * bp_to_1[i] +
                  amps_to[2] * bp_to_2[i])

        out[i] = (np.float32(1.0) - morph) * val_from + morph * val_to
        phase = phase + phase_inc
        if phase >= np.float32(1.0):
            phase = phase - np.float32(1.0)

    return out


def effect_e006_vowel_morph(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Interpolate formants between two vowels using an LFO.

    Params:
        vowel_from:    source vowel          (default 'a')
        vowel_to:      target vowel          (default 'o')
        morph_rate_hz: LFO rate in Hz        [0.1, 5]  (default 0.5)
        Q:             quality factor         [5, 20]   (default 10)
    """
    vowel_from = params.get('vowel_from', 'a')
    vowel_to = params.get('vowel_to', 'o')
    morph_rate_hz = float(np.clip(params.get('morph_rate_hz', 0.5), 0.1, 5.0))
    Q = float(np.clip(params.get('Q', 10.0), 5.0, 20.0))

    if vowel_from not in _FORMANT_TABLE:
        vowel_from = 'a'
    if vowel_to not in _FORMANT_TABLE:
        vowel_to = 'o'

    freqs_from = _FORMANT_TABLE[vowel_from]
    amps_from = _FORMANT_AMPS[vowel_from]
    freqs_to = _FORMANT_TABLE[vowel_to]
    amps_to = _FORMANT_AMPS[vowel_to]

    inp = samples.astype(np.float32)
    n = len(inp)

    # Run 6 bandpass filters (3 per vowel)
    bp_from = []
    for freq in freqs_from:
        b0, b1, b2, a1, a2 = _biquad_coeffs('bpf', freq, sr, Q)
        bp_from.append(_biquad_kernel(inp, b0, b1, b2, a1, a2))

    bp_to = []
    for freq in freqs_to:
        b0, b1, b2, a1, a2 = _biquad_coeffs('bpf', freq, sr, Q)
        bp_to.append(_biquad_kernel(inp, b0, b1, b2, a1, a2))

    # Crossfade with LFO
    amps_from_arr = np.array(amps_from, dtype=np.float32)
    amps_to_arr = np.array(amps_to, dtype=np.float32)

    return _vowel_morph_crossfade(bp_from[0], bp_from[1], bp_from[2],
                                  bp_to[0], bp_to[1], bp_to[2],
                                  amps_from_arr, amps_to_arr,
                                  np.float32(morph_rate_hz), np.int32(sr))


@numba.njit
def _vowel_morph_crossfade(bf0, bf1, bf2, bt0, bt1, bt2,
                           amps_from, amps_to, morph_rate_hz, sr):
    """Crossfade between pre-filtered vowel formant banks with LFO."""
    n = len(bf0)
    out = np.empty(n, dtype=np.float32)
    phase_inc = morph_rate_hz / np.float32(sr)
    phase = np.float32(0.0)

    for i in range(n):
        morph = np.float32(0.5) * (np.float32(1.0) - np.float32(
            np.cos(np.float32(2.0) * np.float32(np.pi) * phase)))

        val_from = amps_from[0] * bf0[i] + amps_from[1] * bf1[i] + amps_from[2] * bf2[i]
        val_to = amps_to[0] * bt0[i] + amps_to[1] * bt1[i] + amps_to[2] * bt2[i]

        out[i] = (np.float32(1.0) - morph) * val_from + morph * val_to
        phase = phase + phase_inc
        if phase >= np.float32(1.0):
            phase = phase - np.float32(1.0)

    return out


def variants_e006():
    return [
        {'vowel_from': 'a', 'vowel_to': 'o', 'morph_rate_hz': 0.5, 'Q': 10.0},
        {'vowel_from': 'e', 'vowel_to': 'i', 'morph_rate_hz': 1.0, 'Q': 12.0},
        {'vowel_from': 'a', 'vowel_to': 'u', 'morph_rate_hz': 0.2, 'Q': 8.0},
        {'vowel_from': 'i', 'vowel_to': 'o', 'morph_rate_hz': 2.0, 'Q': 15.0},
        {'vowel_from': 'u', 'vowel_to': 'e', 'morph_rate_hz': 0.3, 'Q': 6.0},
        {'vowel_from': 'o', 'vowel_to': 'a', 'morph_rate_hz': 4.0, 'Q': 10.0},
    ]


# ---------------------------------------------------------------------------
# E007 -- Auto-Wah
# ---------------------------------------------------------------------------

@numba.njit
def _autowah_kernel(samples, min_freq, max_freq, sensitivity, Q, sr):
    """Envelope follower drives bandpass filter cutoff.

    Uses a simplified biquad-style bandpass recomputed per-sample
    (lightweight approach: SVF driven by envelope).
    """
    n = len(samples)
    out = np.empty(n, dtype=np.float32)

    env = np.float32(0.0)
    attack = np.float32(0.001)   # fast attack
    release = np.float32(0.9995) # slow release

    lp = np.float32(0.0)
    bp = np.float32(0.0)
    q_inv = np.float32(1.0) / np.float32(Q)

    pi_over_sr = np.float32(np.pi) / np.float32(sr)

    for i in range(n):
        x = samples[i]

        # Envelope follower
        abs_x = abs(x)
        if abs_x > env:
            env = env + (np.float32(1.0) - attack) * (abs_x - env)
        else:
            env = release * env

        # Map envelope to cutoff frequency
        cutoff = min_freq + sensitivity * env * (max_freq - min_freq) / np.float32(0.01)
        if cutoff > max_freq:
            cutoff = max_freq
        if cutoff < min_freq:
            cutoff = min_freq

        # SVF coefficients
        f_coeff = np.float32(2.0) * np.float32(np.sin(cutoff * pi_over_sr))
        if f_coeff > np.float32(1.8):
            f_coeff = np.float32(1.8)

        # SVF step
        hp = x - lp - q_inv * bp
        bp = bp + f_coeff * hp
        lp = lp + f_coeff * bp

        # Clamp
        if bp > np.float32(10.0):
            bp = np.float32(10.0)
        elif bp < np.float32(-10.0):
            bp = np.float32(-10.0)
        if lp > np.float32(10.0):
            lp = np.float32(10.0)
        elif lp < np.float32(-10.0):
            lp = np.float32(-10.0)

        out[i] = bp  # bandpass output for wah character

    return out


def effect_e007_auto_wah(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Auto-wah: envelope follower modulates filter cutoff.

    Params:
        min_freq:    minimum cutoff Hz      [100, 500]    (default 200)
        max_freq:    maximum cutoff Hz      [1000, 8000]  (default 3000)
        sensitivity: envelope sensitivity   [0.001, 0.05] (default 0.01)
        Q:           resonance              [2, 15]       (default 5.0)
    """
    min_freq = np.float32(np.clip(params.get('min_freq', 200.0), 100.0, 500.0))
    max_freq = np.float32(np.clip(params.get('max_freq', 3000.0), 1000.0, 8000.0))
    sensitivity = np.float32(np.clip(params.get('sensitivity', 0.01), 0.001, 0.05))
    Q = float(np.clip(params.get('Q', 5.0), 2.0, 15.0))

    return _autowah_kernel(samples.astype(np.float32),
                           min_freq, max_freq, sensitivity, np.float32(Q),
                           np.int32(sr))


def variants_e007():
    return [
        {'min_freq': 200.0, 'max_freq': 3000.0, 'sensitivity': 0.01, 'Q': 5.0},
        {'min_freq': 100.0, 'max_freq': 5000.0, 'sensitivity': 0.03, 'Q': 8.0},
        {'min_freq': 300.0, 'max_freq': 2000.0, 'sensitivity': 0.005, 'Q': 3.0},
        {'min_freq': 150.0, 'max_freq': 8000.0, 'sensitivity': 0.02, 'Q': 12.0},
        {'min_freq': 400.0, 'max_freq': 4000.0, 'sensitivity': 0.04, 'Q': 10.0},
    ]


# ---------------------------------------------------------------------------
# E008 -- Resonant Filter Sweep
# ---------------------------------------------------------------------------

@numba.njit
def _resonant_sweep_kernel(samples, start_freq, end_freq, Q, sr, use_bpf):
    """Sweep a resonant filter cutoff linearly (in log space) over the signal duration.

    Uses SVF for per-sample cutoff modulation.
    """
    n = len(samples)
    out = np.empty(n, dtype=np.float32)

    log_start = np.float32(np.log(start_freq))
    log_end = np.float32(np.log(end_freq))
    q_inv = np.float32(1.0) / np.float32(Q)
    pi_over_sr = np.float32(np.pi) / np.float32(sr)

    lp = np.float32(0.0)
    bp = np.float32(0.0)

    n_f = np.float32(n)

    for i in range(n):
        # Log-linear sweep
        t = np.float32(i) / n_f
        log_freq = log_start + t * (log_end - log_start)
        cutoff = np.float32(np.exp(log_freq))

        f_coeff = np.float32(2.0) * np.float32(np.sin(cutoff * pi_over_sr))
        if f_coeff > np.float32(1.8):
            f_coeff = np.float32(1.8)

        x = samples[i]
        hp = x - lp - q_inv * bp
        bp = bp + f_coeff * hp
        lp = lp + f_coeff * bp

        # Clamp
        if bp > np.float32(10.0):
            bp = np.float32(10.0)
        elif bp < np.float32(-10.0):
            bp = np.float32(-10.0)
        if lp > np.float32(10.0):
            lp = np.float32(10.0)
        elif lp < np.float32(-10.0):
            lp = np.float32(-10.0)

        if use_bpf:
            out[i] = bp
        else:
            out[i] = lp

    return out


def effect_e008_resonant_sweep(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Sweep a resonant filter from start_freq to end_freq over the signal duration.

    Params:
        start_freq:  starting cutoff Hz  [100, 1000]   (default 200)
        end_freq:    ending cutoff Hz    [2000, 12000]  (default 6000)
        Q:           resonance           [5, 30]        (default 10)
        filter_type: 'lpf' or 'bpf'                    (default 'lpf')
    """
    start_freq = float(np.clip(params.get('start_freq', 200.0), 100.0, 1000.0))
    end_freq = float(np.clip(params.get('end_freq', 6000.0), 2000.0, min(12000.0, sr * 0.499)))
    Q = float(np.clip(params.get('Q', 10.0), 5.0, 30.0))
    filter_type = params.get('filter_type', 'lpf')
    use_bpf = filter_type == 'bpf'

    return _resonant_sweep_kernel(samples.astype(np.float32),
                                  np.float32(start_freq), np.float32(end_freq),
                                  np.float32(Q), np.int32(sr), use_bpf)


def variants_e008():
    return [
        {'start_freq': 200.0, 'end_freq': 6000.0, 'Q': 10.0, 'filter_type': 'lpf'},
        {'start_freq': 100.0, 'end_freq': 10000.0, 'Q': 20.0, 'filter_type': 'lpf'},
        {'start_freq': 500.0, 'end_freq': 4000.0, 'Q': 5.0, 'filter_type': 'bpf'},
        {'start_freq': 300.0, 'end_freq': 8000.0, 'Q': 25.0, 'filter_type': 'bpf'},
        {'start_freq': 800.0, 'end_freq': 3000.0, 'Q': 15.0, 'filter_type': 'lpf'},
        {'start_freq': 150.0, 'end_freq': 12000.0, 'Q': 8.0, 'filter_type': 'lpf'},
    ]


# ---------------------------------------------------------------------------
# E009 -- Multi-Mode Filter Crossfade
# ---------------------------------------------------------------------------

@numba.njit
def _multimode_crossfade_kernel(samples, cutoff_hz, Q, morph_rate_hz, sr):
    """LFO morphs between LP, BP, HP outputs of an SVF.

    LFO phase 0..0.33: LP -> BP
    LFO phase 0.33..0.67: BP -> HP
    LFO phase 0.67..1.0: HP -> LP
    """
    n = len(samples)
    out = np.empty(n, dtype=np.float32)

    f_coeff = np.float32(2.0 * np.sin(np.pi * cutoff_hz / sr))
    if f_coeff > np.float32(1.8):
        f_coeff = np.float32(1.8)
    q_inv = np.float32(1.0) / np.float32(Q)

    lp = np.float32(0.0)
    bp = np.float32(0.0)

    phase_inc = np.float32(morph_rate_hz / sr)
    phase = np.float32(0.0)
    third = np.float32(1.0 / 3.0)

    for i in range(n):
        x = samples[i]

        # SVF step
        hp = x - lp - q_inv * bp
        bp = bp + f_coeff * hp
        lp = lp + f_coeff * bp

        # Clamp
        if bp > np.float32(10.0):
            bp = np.float32(10.0)
        elif bp < np.float32(-10.0):
            bp = np.float32(-10.0)
        if lp > np.float32(10.0):
            lp = np.float32(10.0)
        elif lp < np.float32(-10.0):
            lp = np.float32(-10.0)

        # Three-way crossfade based on LFO phase
        if phase < third:
            # LP -> BP
            t = phase / third
            out[i] = (np.float32(1.0) - t) * lp + t * bp
        elif phase < np.float32(2.0) * third:
            # BP -> HP
            t = (phase - third) / third
            out[i] = (np.float32(1.0) - t) * bp + t * hp
        else:
            # HP -> LP
            t = (phase - np.float32(2.0) * third) / third
            out[i] = (np.float32(1.0) - t) * hp + t * lp

        phase = phase + phase_inc
        if phase >= np.float32(1.0):
            phase = phase - np.float32(1.0)

    return out


def effect_e009_multimode_crossfade(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Multi-mode filter with LFO morphing between LP/BP/HP.

    Params:
        morph_rate_hz: LFO rate            [0.1, 5]    (default 0.5)
        Q:             resonance           [2, 15]     (default 5.0)
        cutoff_hz:     filter cutoff       [500, 5000] (default 1500)
    """
    morph_rate_hz = float(np.clip(params.get('morph_rate_hz', 0.5), 0.1, 5.0))
    Q = float(np.clip(params.get('Q', 5.0), 2.0, 15.0))
    cutoff_hz = float(np.clip(params.get('cutoff_hz', 1500.0), 500.0, min(5000.0, sr * 0.499)))

    return _multimode_crossfade_kernel(samples.astype(np.float32),
                                       np.float32(cutoff_hz),
                                       np.float32(Q),
                                       np.float32(morph_rate_hz),
                                       np.int32(sr))


def variants_e009():
    return [
        {'morph_rate_hz': 0.3, 'Q': 5.0, 'cutoff_hz': 1500.0},
        {'morph_rate_hz': 1.0, 'Q': 8.0, 'cutoff_hz': 2000.0},
        {'morph_rate_hz': 0.1, 'Q': 3.0, 'cutoff_hz': 800.0},
        {'morph_rate_hz': 2.5, 'Q': 12.0, 'cutoff_hz': 3000.0},
        {'morph_rate_hz': 5.0, 'Q': 10.0, 'cutoff_hz': 1000.0},
        {'morph_rate_hz': 0.5, 'Q': 15.0, 'cutoff_hz': 4000.0},
    ]


# ---------------------------------------------------------------------------
# E010 -- Cascade of Detuned Resonators
# ---------------------------------------------------------------------------

def effect_e010_detuned_resonators(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """N bandpass filters spread around a base frequency.

    Each resonator is detuned from the base by a factor spread evenly
    in the range [1 - detune, 1 + detune].

    Params:
        base_freq:       centre frequency Hz         [200, 2000]  (default 500)
        num_resonators:  number of bandpass filters   [3, 8]       (default 5)
        detune:          fractional spread            [0.01, 0.1]  (default 0.05)
        Q:               quality factor               [10, 50]     (default 20)
    """
    base_freq = float(np.clip(params.get('base_freq', 500.0), 200.0, 2000.0))
    num_resonators = int(np.clip(params.get('num_resonators', 5), 3, 8))
    detune = float(np.clip(params.get('detune', 0.05), 0.01, 0.1))
    Q = float(np.clip(params.get('Q', 20.0), 10.0, 50.0))

    inp = samples.astype(np.float32)
    n = len(inp)
    out = np.zeros(n, dtype=np.float32)

    # Spread resonators symmetrically around base_freq
    if num_resonators == 1:
        offsets = [0.0]
    else:
        offsets = np.linspace(-detune, detune, num_resonators)

    gain = np.float32(1.0 / num_resonators)

    for offset in offsets:
        freq = base_freq * (1.0 + offset)
        freq = float(np.clip(freq, 20.0, sr * 0.499))
        b0, b1, b2, a1, a2 = _biquad_coeffs('bpf', freq, sr, Q)
        filtered = _biquad_kernel(inp, b0, b1, b2, a1, a2)
        out = out + gain * filtered

    return out


def variants_e010():
    return [
        {'base_freq': 500.0, 'num_resonators': 5, 'detune': 0.05, 'Q': 20.0},
        {'base_freq': 300.0, 'num_resonators': 8, 'detune': 0.08, 'Q': 30.0},
        {'base_freq': 1000.0, 'num_resonators': 3, 'detune': 0.02, 'Q': 40.0},
        {'base_freq': 800.0, 'num_resonators': 6, 'detune': 0.1, 'Q': 15.0},
        {'base_freq': 200.0, 'num_resonators': 4, 'detune': 0.03, 'Q': 50.0},
        {'base_freq': 1500.0, 'num_resonators': 7, 'detune': 0.06, 'Q': 25.0},
    ]


# ---------------------------------------------------------------------------
# E011 -- Allpass Lattice Filter
# Cascade of first-order allpass sections with coupled coefficients.
# Produces deep, swept notches — distinct from phaser (C003) because
# the lattice coupling creates different harmonic relationships.
# ---------------------------------------------------------------------------

@numba.njit
def _allpass_lattice_kernel(samples, coeffs, num_stages):
    """Lattice allpass filter: each stage is a first-order allpass with
    coupled forward/backward paths."""
    n = len(samples)
    out = np.zeros(n, dtype=np.float32)

    # State per stage
    state = np.zeros(num_stages, dtype=np.float32)

    for i in range(n):
        x = samples[i]

        for s in range(num_stages):
            k = coeffs[s]
            # Lattice allpass: y = k*x + state; state_next = x - k*y
            y = k * x + state[s]
            state[s] = x - k * y
            x = y

        out[i] = x

    return out


def effect_e011_allpass_lattice(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Allpass lattice filter: cascade of first-order lattice allpass sections.
    Creates comb-like notch patterns with a distinct character from phasers."""
    num_stages = int(params.get('num_stages', 6))
    base_coeff = float(params.get('base_coeff', 0.5))
    spread = float(params.get('spread', 0.3))
    mix = float(params.get('mix', 0.7))

    num_stages = max(2, min(16, num_stages))
    coeffs = np.zeros(num_stages, dtype=np.float32)
    for s in range(num_stages):
        frac = float(s) / max(1, num_stages - 1)
        coeffs[s] = np.float32(np.clip(base_coeff + spread * (frac - 0.5), -0.99, 0.99))

    samples = samples.astype(np.float32)
    filtered = _allpass_lattice_kernel(samples, coeffs, num_stages)

    # Mix original with allpass output to create notches
    out = np.float32(1.0 - mix) * samples + np.float32(mix) * filtered
    return out


def variants_e011():
    return [
        {'num_stages': 4, 'base_coeff': 0.3, 'spread': 0.2, 'mix': 0.7},
        {'num_stages': 6, 'base_coeff': 0.5, 'spread': 0.3, 'mix': 0.7},
        {'num_stages': 8, 'base_coeff': 0.7, 'spread': 0.4, 'mix': 0.8},
        {'num_stages': 12, 'base_coeff': 0.4, 'spread': 0.5, 'mix': 0.6},
        {'num_stages': 16, 'base_coeff': 0.6, 'spread': 0.2, 'mix': 0.9},
        {'num_stages': 6, 'base_coeff': -0.5, 'spread': 0.6, 'mix': 0.7},
    ]


# ---------------------------------------------------------------------------
# E012 -- Pitch-Tracking Resonator
# Detects dominant pitch via autocorrelation, tunes a resonator bank to
# harmonics of that pitch. Creates sympathetic resonance that follows input.
# ---------------------------------------------------------------------------

@numba.njit
def _autocorr_pitch(frame, sr, min_freq, max_freq):
    """Simple autocorrelation pitch detection on a frame."""
    n = len(frame)
    min_lag = max(1, int(sr / max_freq))
    max_lag = min(n - 1, int(sr / min_freq))

    best_lag = min_lag
    best_corr = np.float32(-1.0)

    for lag in range(min_lag, max_lag + 1):
        corr = np.float32(0.0)
        energy = np.float32(0.0)
        for j in range(n - lag):
            corr += frame[j] * frame[j + lag]
            energy += frame[j] * frame[j]
        if energy > np.float32(1e-10):
            norm_corr = corr / energy
            if norm_corr > best_corr:
                best_corr = norm_corr
                best_lag = lag

    if best_corr < np.float32(0.2):
        return np.float32(0.0)  # no clear pitch
    return np.float32(sr) / np.float32(best_lag)


@numba.njit
def _resonator_bank_kernel(samples, sr, freqs, num_harmonics, Q, wet):
    """Apply a bank of biquad bandpass resonators at given frequencies."""
    n = len(samples)
    out = np.zeros(n, dtype=np.float32)

    for h in range(num_harmonics):
        freq = freqs[h]
        if freq < np.float32(20.0) or freq > np.float32(sr * 0.499):
            continue

        # Biquad BPF coefficients
        w0 = np.float64(2.0 * 3.141592653589793 * freq / sr)
        alpha = np.float64(np.sin(w0) / (2.0 * Q))
        a0 = np.float64(1.0 + alpha)
        b0 = np.float32(alpha / a0)
        b1 = np.float32(0.0)
        b2 = np.float32(-alpha / a0)
        a1 = np.float32(-2.0 * np.cos(w0) / a0)
        a2 = np.float32((1.0 - alpha) / a0)

        z1 = np.float32(0.0)
        z2 = np.float32(0.0)
        gain = np.float32(1.0 / num_harmonics)

        for i in range(n):
            x = samples[i]
            y = b0 * x + z1
            z1 = b1 * x - a1 * y + z2
            z2 = b2 * x - a2 * y
            out[i] += y * gain

    # Wet/dry mix
    result = np.zeros(n, dtype=np.float32)
    dry = np.float32(1.0) - wet
    for i in range(n):
        result[i] = dry * samples[i] + wet * out[i]
    return result


def effect_e012_pitch_tracking_resonator(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Pitch-tracking resonator: detects input pitch, tunes resonator bank
    to its harmonics for sympathetic resonance that follows the input."""
    num_harmonics = int(params.get('num_harmonics', 8))
    Q = float(params.get('Q', 15.0))
    wet = np.float32(params.get('wet', 0.6))

    samples = samples.astype(np.float32)
    n = len(samples)

    # Detect pitch from a chunk near the beginning
    analysis_len = min(n, int(0.05 * sr))  # 50ms
    analysis_start = min(n // 4, n - analysis_len)
    frame = samples[analysis_start:analysis_start + analysis_len]

    fundamental = _autocorr_pitch(frame, sr, 50.0, 2000.0)
    if fundamental < 20.0:
        fundamental = np.float32(220.0)  # fallback to A3

    # Build harmonic frequencies
    freqs = np.zeros(num_harmonics, dtype=np.float32)
    for h in range(num_harmonics):
        freqs[h] = fundamental * np.float32(h + 1)

    return _resonator_bank_kernel(samples, sr, freqs, num_harmonics,
                                  np.float64(Q), wet)


def variants_e012():
    return [
        {'num_harmonics': 4, 'Q': 10.0, 'wet': 0.4},
        {'num_harmonics': 8, 'Q': 15.0, 'wet': 0.5},
        {'num_harmonics': 12, 'Q': 20.0, 'wet': 0.6},
        {'num_harmonics': 6, 'Q': 30.0, 'wet': 0.7},
        {'num_harmonics': 16, 'Q': 10.0, 'wet': 0.5},
        {'num_harmonics': 8, 'Q': 50.0, 'wet': 0.8},
    ]
