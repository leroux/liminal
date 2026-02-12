"""DSP utility functions."""

import numpy as np


def normalize(audio: np.ndarray, target_peak: float = 0.9) -> np.ndarray:
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio * (target_peak / peak)
    return audio.astype(np.float32)


def build_colormap_lut(name: str = 'magma') -> np.ndarray:
    """Build a 256x3 uint8 colormap lookup table.

    Implements magma, inferno, and viridis-like colormaps without matplotlib.
    """
    lut = np.zeros((256, 3), dtype=np.uint8)
    t = np.linspace(0, 1, 256)

    if name == 'magma':
        # Approximate magma: black -> purple -> orange -> white
        r = np.clip(np.where(t < 0.25, t * 4 * 0.07,
                    np.where(t < 0.5, 0.07 + (t - 0.25) * 4 * 0.65,
                    np.where(t < 0.75, 0.72 + (t - 0.5) * 4 * 0.26,
                             0.98 + (t - 0.75) * 4 * 0.02))), 0, 1)
        g = np.clip(np.where(t < 0.25, t * 4 * 0.01,
                    np.where(t < 0.5, 0.01 + (t - 0.25) * 4 * 0.09,
                    np.where(t < 0.75, 0.10 + (t - 0.5) * 4 * 0.55,
                             0.65 + (t - 0.75) * 4 * 0.35))), 0, 1)
        b = np.clip(np.where(t < 0.25, t * 4 * 0.30,
                    np.where(t < 0.5, 0.30 + (t - 0.25) * 4 * 0.15,
                    np.where(t < 0.75, 0.45 - (t - 0.5) * 4 * 0.35,
                             0.10 + (t - 0.75) * 4 * 0.20))), 0, 1)
    elif name == 'inferno':
        # Approximate inferno: black -> purple -> orange -> yellow
        r = np.clip(t ** 0.5 * np.where(t < 0.5, t * 2 * 0.9, 0.9 + (t - 0.5) * 0.2), 0, 1)
        g = np.clip(np.where(t < 0.4, t * 0.1,
                    np.where(t < 0.8, (t - 0.4) * 2.0, 0.8 + (t - 0.8) * 1.0)), 0, 1)
        b = np.clip(np.where(t < 0.3, t * 3.0 * 0.5,
                    np.where(t < 0.6, 0.5 - (t - 0.3) * 1.5, 0.05)), 0, 1)
    else:  # viridis-like
        r = np.clip(np.where(t < 0.5, 0.27 - t * 0.3, (t - 0.5) * 2 * 0.73 + 0.12), 0, 1)
        g = np.clip(0.004 + t * 0.87, 0, 1)
        b = np.clip(np.where(t < 0.5, 0.33 + t * 0.7, 0.68 - (t - 0.5) * 1.3), 0, 1)

    lut[:, 0] = (r * 255).astype(np.uint8)
    lut[:, 1] = (g * 255).astype(np.uint8)
    lut[:, 2] = (b * 255).astype(np.uint8)
    return lut


def generate_sawtooth(duration: float, freq: float, sr: int = 44100) -> np.ndarray:
    n = int(duration * sr)
    t = np.arange(n, dtype=np.float32) / sr
    phase = (t * freq) % 1.0
    return (2.0 * phase - 1.0).astype(np.float32)


def generate_white_noise(duration: float, sr: int = 44100) -> np.ndarray:
    n = int(duration * sr)
    return np.random.randn(n).astype(np.float32)


def generate_pink_noise(duration: float, sr: int = 44100) -> np.ndarray:
    n = int(duration * sr)
    white = np.random.randn(n)
    # Voss-McCartney approximation
    pink = np.zeros(n, dtype=np.float64)
    n_rows = 16
    rows = np.zeros(n_rows)
    running_sum = 0.0
    for i in range(n):
        # Determine which row to update
        for j in range(n_rows):
            if i % (1 << j) == 0:
                running_sum -= rows[j]
                rows[j] = np.random.randn()
                running_sum += rows[j]
                break
        pink[i] = running_sum + white[i]
    peak = np.max(np.abs(pink))
    if peak > 0:
        pink /= peak
    return pink.astype(np.float32)


def generate_pulse_train(duration: float, freq: float, sr: int = 44100) -> np.ndarray:
    n = int(duration * sr)
    period = max(1, int(sr / freq))
    out = np.zeros(n, dtype=np.float32)
    for i in range(0, n, period):
        out[i] = 1.0
    return out
