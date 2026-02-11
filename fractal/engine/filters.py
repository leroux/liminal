"""Filters, noise gate, limiter, bitcrusher for the Fractal effect.

All sample-level loops use Numba for speed.
"""

import numpy as np
from numba import njit
from fractal.engine.params import SR


# ---------------------------------------------------------------------------
# Biquad filter (lowpass / highpass / bandpass)
# ---------------------------------------------------------------------------

def compute_biquad_coeffs(filter_type, freq, q):
    """Return (b, a) for the requested filter type.

    filter_type: 1=lowpass, 2=highpass, 3=bandpass
    freq:        cutoff/center frequency Hz
    q:           quality factor
    """
    w0 = 2.0 * np.pi * freq / SR
    alpha = np.sin(w0) / (2.0 * q)
    cos_w0 = np.cos(w0)

    if filter_type == 1:  # lowpass
        b = np.array([(1.0 - cos_w0) / 2.0, 1.0 - cos_w0, (1.0 - cos_w0) / 2.0])
        a = np.array([1.0 + alpha, -2.0 * cos_w0, 1.0 - alpha])
    elif filter_type == 2:  # highpass
        b = np.array([(1.0 + cos_w0) / 2.0, -(1.0 + cos_w0), (1.0 + cos_w0) / 2.0])
        a = np.array([1.0 + alpha, -2.0 * cos_w0, 1.0 - alpha])
    else:  # bandpass (type 3)
        b = np.array([alpha, 0.0, -alpha])
        a = np.array([1.0 + alpha, -2.0 * cos_w0, 1.0 - alpha])

    return b, a


def apply_pre_filter(audio, params):
    """Apply pre-fractal filter."""
    ft = int(params.get("filter_type", 0))
    if ft == 0:
        return audio.copy()

    freq = float(params.get("filter_freq", 2000.0))
    q = float(params.get("filter_q", 0.707))
    freq = np.clip(freq, 20.0, SR / 2.0 - 1.0)
    q = max(0.1, q)

    b, a = compute_biquad_coeffs(ft, freq, q)
    b = b / a[0]
    a = a / a[0]

    return _biquad(audio, b[0], b[1], b[2], a[1], a[2])


def apply_post_filter(audio, params):
    """Apply post-fractal filter."""
    ft = int(params.get("post_filter_type", 0))
    if ft == 0:
        return audio.copy()

    freq = float(params.get("post_filter_freq", 8000.0))
    freq = np.clip(freq, 20.0, SR / 2.0 - 1.0)

    # Post-filter uses gentle Q for taming
    b, a = compute_biquad_coeffs(ft, freq, 0.707)
    b = b / a[0]
    a = a / a[0]

    return _biquad(audio, b[0], b[1], b[2], a[1], a[2])


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
# Bitcrusher + sample rate reducer
# ---------------------------------------------------------------------------

def crush_and_decimate(audio, params):
    """Apply bitcrusher and/or sample rate reducer."""
    crush = float(params.get("crush", 0.0))
    decimate = float(params.get("decimate", 0.0))

    if crush <= 0.0 and decimate <= 0.0:
        return audio.copy()

    return _crush_decimate(audio, crush, decimate)


@njit(cache=True)
def _crush_decimate(audio, crush, decimate):
    n = len(audio)
    out = np.empty(n, dtype=np.float64)

    if crush > 0:
        bits = 16.0 - 12.0 * crush
        quant = 2.0 ** (bits - 1.0)
        for i in range(n):
            out[i] = np.floor(audio[i] * quant + 0.5) / quant
    else:
        for i in range(n):
            out[i] = audio[i]

    if decimate > 0:
        rate_factor = 1.0 + 31.0 * decimate
        phase = 0.0
        held = 0.0
        for i in range(n):
            phase += 1.0
            if phase >= rate_factor:
                held = out[i]
                phase -= rate_factor
            out[i] = held

    return out


# ---------------------------------------------------------------------------
# Noise gate
# ---------------------------------------------------------------------------

def noise_gate(audio, params):
    """Simple RMS-based noise gate."""
    threshold = float(params.get("gate", 0.0))
    if threshold <= 0.0:
        return audio.copy()
    return _gate(audio, threshold)


@njit(cache=True)
def _gate(audio, threshold):
    n = len(audio)
    out = audio.copy()
    win = 512
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

def limiter(audio, params=None):
    """Simple peak limiter with soft clipping near ceiling."""
    ceiling = 0.95
    if params is not None:
        threshold = float(params.get("threshold", 0.5))
        ceiling = 0.1 + 0.85 * threshold
    peak = np.max(np.abs(audio))
    if peak <= 0.0:
        return audio.copy()
    if peak > ceiling:
        return audio * (ceiling / peak)
    return audio.copy()
