"""Numba-based standalone DSP primitives — for listening to each building block.

Each function processes a full audio buffer and returns the output.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def delay(audio, delay_samples):
    """Simple delay line — delays the signal by N samples."""
    n = len(audio)
    out = np.zeros(n)
    for i in range(n):
        rd = i - delay_samples
        if rd >= 0:
            out[i] = audio[rd]
    return out


@njit(cache=True)
def delay_feedback(audio, delay_samples, feedback, wet=0.5):
    """Delay with feedback — creates echoes."""
    n = len(audio)
    buf = np.zeros(delay_samples + 1)
    wi = 0
    out = np.zeros(n)
    for i in range(n):
        rd = (wi - delay_samples) % len(buf)
        delayed = buf[rd]
        buf[wi] = audio[i] + feedback * delayed
        wi = (wi + 1) % len(buf)
        out[i] = (1.0 - wet) * audio[i] + wet * delayed
    return out


@njit(cache=True)
def one_pole_lowpass(audio, coeff):
    """One-pole lowpass filter. coeff=0 is bypass, higher = darker."""
    n = len(audio)
    out = np.zeros(n)
    y1 = 0.0
    for i in range(n):
        y1 = (1.0 - coeff) * audio[i] + coeff * y1
        out[i] = y1
    return out


@njit(cache=True)
def allpass(audio, delay_samples, gain):
    """Schroeder allpass filter — changes phase without changing amplitude spectrum."""
    n = len(audio)
    buf = np.zeros(delay_samples)
    idx = 0
    out = np.zeros(n)
    for i in range(n):
        delayed = buf[idx]
        v = audio[i] + gain * delayed
        out[i] = -gain * v + delayed
        buf[idx] = v
        idx = (idx + 1) % delay_samples
    return out


@njit(cache=True)
def allpass_chain(audio, delay_times, gain):
    """Chain of allpass filters — builds diffusion."""
    x = audio.copy()
    for s in range(len(delay_times)):
        dt = delay_times[s]
        buf = np.zeros(dt)
        idx = 0
        out = np.zeros(len(x))
        for i in range(len(x)):
            delayed = buf[idx]
            v = x[i] + gain * delayed
            out[i] = -gain * v + delayed
            buf[idx] = v
            idx = (idx + 1) % dt
        x = out
    return x


@njit(cache=True)
def comb_filter(audio, delay_samples, feedback, damping=0.0):
    """Comb filter with optional damping — a single FDN delay line."""
    n = len(audio)
    buf = np.zeros(delay_samples + 1)
    wi = 0
    y1 = 0.0
    out = np.zeros(n)
    for i in range(n):
        rd = (wi - delay_samples) % len(buf)
        delayed = buf[rd]
        # Damping (one-pole lowpass in feedback path)
        y1 = (1.0 - damping) * delayed + damping * y1
        buf[wi] = audio[i] + feedback * y1
        wi = (wi + 1) % len(buf)
        out[i] = delayed
    return out


@njit(cache=True)
def saturate(audio, drive):
    """Tanh soft saturation. drive=1.0 is unity, higher drives harder."""
    n = len(audio)
    out = np.zeros(n)
    for i in range(n):
        out[i] = np.tanh(audio[i] * drive)
    return out
