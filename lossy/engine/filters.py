"""Time-domain processing: biquad filter, lo-fi reverb, noise gate, limiter.

All sample-level loops use Numba for speed.
"""

import numpy as np
from numba import njit
from lossy.engine.params import SR, SLOPE_SECTIONS, SLOPE_OPTIONS


# ---------------------------------------------------------------------------
# Biquad filter (bandpass / notch)
# ---------------------------------------------------------------------------

def compute_biquad_coeffs(filter_type, freq, width, slope_idx):
    """Return (b, a, n_sections) for the requested filter configuration.

    filter_type: 1 = bandpass, 2 = notch
    freq:        centre frequency Hz
    width:       0 (narrow / high Q) to 1 (wide / low Q)
    slope_idx:   index into SLOPE_OPTIONS
    """
    slope = SLOPE_OPTIONS[min(slope_idx, len(SLOPE_OPTIONS) - 1)]
    n_sections = SLOPE_SECTIONS[slope]

    # Width → Q  (log mapping: 0 → Q=20, 1 → Q=0.3)
    q = 0.3 * (20.0 / 0.3) ** (1.0 - width)
    # Steeper slopes → boost Q for resonance
    q *= {6: 1.0, 24: 1.5, 96: 3.0}[slope]

    w0 = 2.0 * np.pi * freq / SR
    alpha = np.sin(w0) / (2.0 * q)
    cos_w0 = np.cos(w0)

    if filter_type == 1:  # bandpass (0 dB peak)
        b = np.array([alpha, 0.0, -alpha])
        a = np.array([1.0 + alpha, -2.0 * cos_w0, 1.0 - alpha])
    else:                 # notch
        b = np.array([1.0, -2.0 * cos_w0, 1.0])
        a = np.array([1.0 + alpha, -2.0 * cos_w0, 1.0 - alpha])

    return b, a, n_sections


def apply_filter(audio, params):
    """Run biquad filter chain on audio.

    Returns processed copy; original is not modified.
    """
    ft = int(params.get("filter_type", 0))
    if ft == 0:  # bypass
        return audio.copy()

    freq = float(params.get("filter_freq", 1000.0))
    width = float(params.get("filter_width", 0.5))
    slope_idx = int(params.get("filter_slope", 1))

    freq = np.clip(freq, 20.0, SR / 2.0 - 1.0)

    b, a, n_sections = compute_biquad_coeffs(ft, freq, width, slope_idx)
    # Normalise
    b = b / a[0]
    a = a / a[0]

    out = audio.copy()
    for _ in range(n_sections):
        out = _biquad(out, b[0], b[1], b[2], a[1], a[2])
    return out


@njit(cache=True)
def _biquad(audio, b0, b1, b2, a1, a2):
    """Direct-form II biquad, one pass."""
    n = len(audio)
    out = np.empty(n, dtype=np.float64)
    w1 = 0.0
    w2 = 0.0
    for i in range(n):
        w0 = audio[i] - a1 * w1 - a2 * w2
        out[i] = b0 * w0 + b1 * w1 + b2 * w2
        w2 = w1
        w1 = w0
    return out


# ---------------------------------------------------------------------------
# Lo-fi reverb  (4 comb filters + allpass, deliberately cheap & metallic)
# ---------------------------------------------------------------------------

def lofi_reverb(audio, params):
    """Blend in a lo-fi Schroeder reverb."""
    g = float(params.get("global_amount", 1.0))
    mix = float(params.get("verb", 0.0)) * g
    if mix <= 0.0:
        return audio.copy()
    decay = float(params.get("decay", 0.5))
    fb = 0.4 + 0.55 * decay  # range 0.4 - 0.95
    return _comb_reverb(audio, mix, fb)


@njit(cache=True)
def _comb_reverb(audio, mix, fb):
    n = len(audio)
    wet = np.zeros(n, dtype=np.float64)

    # Short prime-number delays → metallic, lo-fi
    delays = np.array([1031, 1327, 1657, 1973])
    damp = 0.45

    for c in range(4):
        d = delays[c]
        buf = np.zeros(d, dtype=np.float64)
        y1 = 0.0
        for i in range(n):
            idx = i % d
            rd = buf[idx]
            y1 = damp * rd + (1.0 - damp) * y1
            wet[i] += rd * 0.25
            buf[idx] = audio[i] * 0.25 + y1 * fb

    # Single allpass diffuser
    ap_d = 379
    ap_buf = np.zeros(ap_d, dtype=np.float64)
    ap_g = 0.6
    for i in range(n):
        idx = i % ap_d
        delayed = ap_buf[idx]
        inp = wet[i]
        wet[i] = delayed - ap_g * inp
        ap_buf[idx] = inp + ap_g * wet[i]

    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = audio[i] * (1.0 - mix) + wet[i] * mix
    return out


# ---------------------------------------------------------------------------
# Noise gate
# ---------------------------------------------------------------------------

def noise_gate(audio, params):
    """Simple RMS-based noise gate."""
    g = float(params.get("global_amount", 1.0))
    threshold = float(params.get("gate", 0.0)) * g
    if threshold <= 0.0:
        return audio.copy()
    return _gate(audio, threshold)


@njit(cache=True)
def _gate(audio, threshold):
    n = len(audio)
    out = audio.copy()
    win = 512  # ~11 ms gate window
    for start in range(0, n, win):
        end = min(start + win, n)
        s = 0.0
        for i in range(start, end):
            s += audio[i] * audio[i]
        rms = np.sqrt(s / (end - start))
        if rms < threshold:
            gain = rms / threshold
            for i in range(start, end):
                out[i] *= gain
    return out


# ---------------------------------------------------------------------------
# Limiter
# ---------------------------------------------------------------------------

def limiter(audio, ceiling=0.95, params=None):
    """Simple peak limiter with soft clipping near ceiling.

    If params is provided, reads 'threshold' to set the ceiling:
    threshold 0 = heavy limiting (ceiling 0.1), threshold 1 = light (ceiling 0.95).
    """
    if params is not None:
        threshold = float(params.get("threshold", 0.5))
        ceiling = 0.1 + 0.85 * threshold
    peak = np.max(np.abs(audio))
    if peak <= 0.0:
        return audio.copy()
    if peak > ceiling:
        return audio * (ceiling / peak)
    return audio.copy()
