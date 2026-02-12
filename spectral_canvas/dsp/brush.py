"""Numba-accelerated brush kernels for spectrogram painting."""

import math
import numba
import numpy as np


@numba.njit(cache=True)
def apply_brush(magnitude: np.ndarray, cx: int, cy: int,
                radius: float, softness: float, intensity: float,
                is_erase: bool):
    """Apply a Gaussian brush stamp to the magnitude array.

    Args:
        magnitude: (H, W) float32 array, modified in-place
        cx: center x (time frame index)
        cy: center y (frequency bin index)
        radius: brush radius in bins
        softness: sigma = radius * softness
        intensity: brush strength 0-1
        is_erase: True to subtract, False to add
    """
    h, w = magnitude.shape
    r = int(radius * 3)  # 3-sigma extent
    sigma = radius * softness
    if sigma < 0.1:
        sigma = 0.1
    inv_2sigma2 = 1.0 / (2.0 * sigma * sigma)

    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            y = cy + dy
            x = cx + dx
            if 0 <= y < h and 0 <= x < w:
                k = math.exp(-(dx * dx + dy * dy) * inv_2sigma2)
                if is_erase:
                    val = magnitude[y, x] - k * intensity
                    magnitude[y, x] = max(0.0, val)
                else:
                    val = magnitude[y, x] + k * intensity
                    magnitude[y, x] = min(1.0, val)


@numba.njit(cache=True)
def apply_brush_line(magnitude: np.ndarray, x0: int, y0: int, x1: int, y1: int,
                     radius: float, softness: float, intensity: float,
                     is_erase: bool):
    """Draw a Bresenham line with brush stamps at each point."""
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    steps = max(dx, dy) + 1
    step_interval = max(1, int(radius * 0.5))

    cx, cy = x0, y0
    for i in range(steps + 1):
        if i % step_interval == 0 or (cx == x1 and cy == y1):
            apply_brush(magnitude, cx, cy, radius, softness, intensity, is_erase)
        if cx == x1 and cy == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            cx += sx
        if e2 < dx:
            err += dx
            cy += sy


@numba.njit(cache=True)
def _get_harmonic_ratios(mode: int, n_harmonics: int):
    """Return array of frequency ratios for given harmonic mode.

    Modes: 0=all, 1=odd, 2=even, 3=octaves, 4=fifths, 5=sub, 6=both
    """
    # Allocate enough space (both mode needs 2*n_harmonics - 1)
    max_size = n_harmonics * 2
    ratios = np.empty(max_size, dtype=np.float64)
    count = 0
    if mode == 0:  # all overtones
        for i in range(1, n_harmonics + 1):
            ratios[count] = float(i)
            count += 1
    elif mode == 1:  # odd only
        h = 1
        while count < n_harmonics:
            ratios[count] = float(h)
            count += 1
            h += 2
    elif mode == 2:  # even (+ fundamental)
        ratios[0] = 1.0
        count = 1
        h = 2
        while count < n_harmonics:
            ratios[count] = float(h)
            count += 1
            h += 2
    elif mode == 3:  # octaves
        h = 1.0
        while count < n_harmonics:
            ratios[count] = h
            count += 1
            h *= 2.0
    elif mode == 4:  # fifths (power chord stack)
        ratios[0] = 1.0
        count = 1
        h = 1.5
        while count < n_harmonics:
            ratios[count] = h
            count += 1
            h *= 1.5
    elif mode == 5:  # sub (undertones): f, f/2, f/3, f/4...
        for i in range(1, n_harmonics + 1):
            ratios[count] = 1.0 / float(i)
            count += 1
    elif mode == 6:  # both (overtones + undertones)
        # Fundamental
        ratios[0] = 1.0
        count = 1
        # Interleave: overtone 2, undertone 2, overtone 3, undertone 3...
        h = 2
        while count < n_harmonics * 2 - 1 and h <= n_harmonics:
            ratios[count] = float(h)
            count += 1
            ratios[count] = 1.0 / float(h)
            count += 1
            h += 1
    else:
        for i in range(1, n_harmonics + 1):
            ratios[count] = float(i)
            count += 1
    return ratios[:count]


@numba.njit(cache=True)
def apply_harmonic_brush(magnitude: np.ndarray, cx: int, base_bin: int,
                         radius: float, softness: float, intensity: float,
                         n_harmonics: int, rolloff_alpha: float,
                         sr: int, n_fft: int, is_erase: bool,
                         harmonic_mode: int = 0):
    """Draw at base frequency and its harmonics.

    harmonic_mode: 0=all, 1=odd, 2=even, 3=octaves, 4=fifths, 5=sub, 6=both
    """
    h, w = magnitude.shape
    base_freq = base_bin * sr / n_fft
    if base_freq < 1.0:
        return

    ratios = _get_harmonic_ratios(harmonic_mode, n_harmonics)
    for i in range(len(ratios)):
        harm_freq = base_freq * ratios[i]
        harm_bin = int(round(harm_freq * n_fft / sr))
        if harm_bin < 1 or harm_bin >= h:
            continue  # skip out-of-range (don't break, undertones may be below)
        # For rolloff, use the distance from fundamental (works for both over/undertones)
        dist = ratios[i] if ratios[i] >= 1.0 else 1.0 / ratios[i]
        harm_intensity = intensity / (dist ** rolloff_alpha)
        apply_brush(magnitude, cx, harm_bin, radius, softness, harm_intensity, is_erase)
