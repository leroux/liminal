"""Shared audio I/O utilities for all pedals.

Provides load_wav, save_wav, and make_impulse used by GUIs and CLI renderers.
"""

from math import gcd

import numpy as np
import soundfile as sf


def load_wav(path, sr=44100):
    """Load a WAV file and resample to the target sample rate.

    Returns (audio_array, sample_rate).
    Audio is float64, mono or stereo (samples, 2).
    """
    audio, file_sr = sf.read(path, dtype='float64')
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
    sf.write(path, np.clip(audio, -1.0, 1.0), sr, subtype='PCM_16')


def make_impulse(sr=44100, seconds=0.5):
    """Generate a unit impulse (click) for testing."""
    n = int(sr * seconds)
    impulse = np.zeros(n)
    impulse[0] = 1.0
    return impulse
