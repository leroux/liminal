"""Core fractal algorithm — Numba-accelerated.

Based on R001 Audio Fractalization: compress the signal into progressively
shorter copies, tile them to fill the original length, and sum with decaying
gains to create self-similar texture at multiple timescales.
"""

import numpy as np
from numba import njit


# ---------------------------------------------------------------------------
# Time-domain fractalization
# ---------------------------------------------------------------------------

@njit(cache=True)
def _fractalize_time_nearest(samples, num_scales, scale_ratio, amplitude_decay,
                             reverse_scales, scale_offset):
    """Nearest-neighbor resampling variant (aliased, gritty)."""
    n = len(samples)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = samples[i]

    for s in range(1, num_scales):
        compressed_len = max(1, int(n * (scale_ratio ** s)))
        gain = amplitude_decay ** s

        # Build compressed version via nearest-neighbor
        compressed = np.empty(compressed_len, dtype=np.float64)
        for i in range(compressed_len):
            idx = int(i * (n - 1) / max(1, compressed_len - 1))
            if idx >= n:
                idx = n - 1
            compressed[i] = samples[idx]

        # Optionally reverse
        if reverse_scales:
            for i in range(compressed_len // 2):
                j = compressed_len - 1 - i
                tmp = compressed[i]
                compressed[i] = compressed[j]
                compressed[j] = tmp

        # Tile with offset
        offset_samples = int(scale_offset * compressed_len)
        for i in range(n):
            src = (i + offset_samples) % compressed_len
            out[i] += gain * compressed[src]

    return out


@njit(cache=True)
def _fractalize_time_linear(samples, num_scales, scale_ratio, amplitude_decay,
                            reverse_scales, scale_offset):
    """Linear interpolation resampling variant (smoother)."""
    n = len(samples)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = samples[i]

    for s in range(1, num_scales):
        compressed_len = max(2, int(n * (scale_ratio ** s)))
        gain = amplitude_decay ** s

        # Build compressed version via linear interpolation
        compressed = np.empty(compressed_len, dtype=np.float64)
        for i in range(compressed_len):
            pos = i * (n - 1) / max(1, compressed_len - 1)
            idx0 = int(pos)
            idx1 = min(idx0 + 1, n - 1)
            frac = pos - idx0
            compressed[i] = samples[idx0] * (1.0 - frac) + samples[idx1] * frac

        # Optionally reverse
        if reverse_scales:
            for i in range(compressed_len // 2):
                j = compressed_len - 1 - i
                tmp = compressed[i]
                compressed[i] = compressed[j]
                compressed[j] = tmp

        # Tile with offset
        offset_samples = int(scale_offset * compressed_len)
        for i in range(n):
            src = (i + offset_samples) % compressed_len
            out[i] += gain * compressed[src]

    return out


def fractalize_time(samples, params):
    """Apply time-domain fractalization to mono audio.

    Args:
        samples: float64 mono array
        params: parameter dict

    Returns:
        fractalized float64 mono array, same length
    """
    num_scales = max(2, min(8, int(params.get("num_scales", 3))))
    scale_ratio = np.clip(float(params.get("scale_ratio", 0.5)), 0.1, 0.9)
    amplitude_decay = np.clip(float(params.get("amplitude_decay", 0.5)), 0.1, 1.0)
    interp = int(params.get("interp", 0))
    reverse_scales = int(params.get("reverse_scales", 0))
    scale_offset = np.clip(float(params.get("scale_offset", 0.0)), 0.0, 1.0)

    x = samples.astype(np.float64)

    if interp == 1:
        out = _fractalize_time_linear(x, num_scales, scale_ratio, amplitude_decay,
                                      reverse_scales, scale_offset)
    else:
        out = _fractalize_time_nearest(x, num_scales, scale_ratio, amplitude_decay,
                                       reverse_scales, scale_offset)

    # Normalize to input peak to prevent clipping
    peak_out = np.max(np.abs(out))
    if peak_out > 0:
        peak_in = np.max(np.abs(x))
        if peak_out > peak_in and peak_in > 0:
            out *= peak_in / peak_out

    return out


# ---------------------------------------------------------------------------
# Spectral-domain fractalization
# ---------------------------------------------------------------------------

def fractalize_spectral(samples, params):
    """Apply fractalization to STFT magnitude frames.

    Same compression/tiling logic, but applied to the magnitude spectrum
    of each STFT frame rather than to raw samples.
    """
    num_scales = max(2, min(8, int(params.get("num_scales", 3))))
    scale_ratio = np.clip(float(params.get("scale_ratio", 0.5)), 0.1, 0.9)
    amplitude_decay = np.clip(float(params.get("amplitude_decay", 0.5)), 0.1, 1.0)
    window_size = max(256, min(8192, int(params.get("window_size", 2048))))
    hop_size = window_size // 4

    x = samples.astype(np.float64)
    n = len(x)
    if n < window_size:
        x = np.concatenate([x, np.zeros(window_size - n)])
        n = len(x)

    window = np.hanning(window_size)
    num_frames = 1 + (n - window_size) // hop_size

    # STFT
    frames = np.zeros((num_frames, window_size), dtype=np.float64)
    for i in range(num_frames):
        start = i * hop_size
        frames[i] = x[start:start + window_size] * window

    X = np.fft.rfft(frames, axis=1)
    mag = np.abs(X)
    phase = np.angle(X)
    num_bins = mag.shape[1]

    # Fractalize each frame's magnitude
    for fi in range(num_frames):
        orig_mag = mag[fi].copy()
        for s in range(1, num_scales):
            compressed_len = max(1, int(num_bins * (scale_ratio ** s)))
            gain = amplitude_decay ** s

            # Downsample magnitude via linear interpolation
            indices = np.linspace(0, num_bins - 1, compressed_len)
            compressed = np.interp(indices, np.arange(num_bins), orig_mag)

            # Tile to fill
            tiles_needed = (num_bins + compressed_len - 1) // compressed_len
            tiled = np.tile(compressed, tiles_needed)[:num_bins]
            mag[fi] += gain * tiled

    # Reconstruct
    Y = mag * np.exp(1j * phase)
    synth_frames = np.fft.irfft(Y, n=window_size, axis=1) * window

    out_len = len(samples)
    output = np.zeros(max(out_len, window_size + (num_frames - 1) * hop_size), dtype=np.float64)
    for i in range(num_frames):
        start = i * hop_size
        end = start + window_size
        if end <= len(output):
            output[start:end] += synth_frames[i]

    output = output[:out_len]

    # Normalize
    peak_out = np.max(np.abs(output))
    peak_in = np.max(np.abs(samples))
    if peak_out > 0 and peak_in > 0 and peak_out > peak_in:
        output *= peak_in / peak_out

    return output


# ---------------------------------------------------------------------------
# Combined entry point
# ---------------------------------------------------------------------------

@njit(cache=True)
def _apply_saturation(audio, amount):
    """tanh saturation: blend between clean and saturated."""
    n = len(audio)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        clean = audio[i]
        sat = np.tanh(audio[i])
        out[i] = (1.0 - amount) * clean + amount * sat
    return out


def render_fractal_core(samples, params):
    """Apply fractalization with iteration and spectral blend.

    Args:
        samples: float64 mono array
        params: parameter dict

    Returns:
        processed float64 mono array
    """
    spectral_blend = np.clip(float(params.get("spectral", 0.0)), 0.0, 1.0)
    iterations = max(1, min(4, int(params.get("iterations", 1))))
    iter_decay = np.clip(float(params.get("iter_decay", 0.8)), 0.3, 1.0)
    saturation = np.clip(float(params.get("saturation", 0.0)), 0.0, 1.0)

    current = samples.astype(np.float64)

    for it in range(iterations):
        # Time-domain fractalization
        if spectral_blend < 1.0:
            time_out = fractalize_time(current, params)
        else:
            time_out = current.copy()

        # Spectral-domain fractalization
        if spectral_blend > 0.0:
            spec_out = fractalize_spectral(current, params)
        else:
            spec_out = current.copy()

        # Blend
        if spectral_blend <= 0.0:
            result = time_out
        elif spectral_blend >= 1.0:
            result = spec_out
        else:
            result = (1.0 - spectral_blend) * time_out + spectral_blend * spec_out

        # Apply saturation between iterations
        if saturation > 0.0:
            result = _apply_saturation(result, saturation)

        # Apply iteration decay for subsequent passes
        if it < iterations - 1:
            result *= iter_decay

        current = result

    return current
