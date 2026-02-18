"""Shared audio I/O utilities for all pedals.

Provides load_wav, save_wav, and make_impulse used by GUIs and CLI renderers.
"""

from math import gcd

import numpy as np
from scipy.io import wavfile


def load_wav(path, sr=44100):
    """Load a WAV file and resample to the target sample rate.

    Returns (audio_array, sample_rate).
    Audio is float64, mono or stereo (samples, 2).
    """
    file_sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        audio = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        audio = data.astype(np.float64) / 2147483648.0
    else:
        audio = data.astype(np.float64)
    if file_sr != sr:
        from scipy.signal import resample_poly
        g = gcd(sr, file_sr)
        audio = resample_poly(audio, sr // g, file_sr // g, axis=0)
    return audio, sr


def save_wav(path, audio, sr=44100):
    """Save audio to a 16-bit WAV file with peak normalization."""
    peak = np.max(np.abs(audio))
    if peak > 1.0:
        audio = audio / peak * 0.95
    elif 0 < peak < 0.1:
        audio = audio / peak * 0.9
    out = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    wavfile.write(path, sr, out)


def make_impulse(sr=44100, seconds=0.5):
    """Generate a unit impulse (click) for testing."""
    n = int(sr * seconds)
    impulse = np.zeros(n)
    impulse[0] = 1.0
    return impulse
