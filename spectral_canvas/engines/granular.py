"""Engine 6: Granular Synthesis — pointillist, cloud-like, shimmering texture."""

import numba
import numpy as np

from . import Engine, register_engine
from ..dsp.utils import normalize


@numba.njit(cache=True)
def _render_grains(magnitude, sr, n_fft, hop_length, threshold, grain_dur_ms):
    n_bins, n_frames = magnitude.shape
    total_samples = (n_frames - 1) * hop_length + n_fft
    output = np.zeros(total_samples, dtype=np.float32)
    grain_samples = int(grain_dur_ms * sr / 1000.0)
    if grain_samples < 4:
        grain_samples = 4

    # Precompute Hann window
    window = np.zeros(grain_samples, dtype=np.float32)
    for i in range(grain_samples):
        window[i] = 0.5 * (1.0 - np.cos(2.0 * np.pi * i / (grain_samples - 1)))

    for frame_idx in range(n_frames):
        for bin_idx in range(1, n_bins):  # skip DC
            amp = magnitude[bin_idx, frame_idx]
            if amp < threshold:
                continue
            freq = bin_idx * sr / n_fft
            if freq < 20 or freq > 20000:
                continue

            # Time position with jitter
            base_sample = frame_idx * hop_length
            jitter = int((np.random.random() - 0.5) * sr * 0.004)  # ±2ms
            start = base_sample + jitter
            if start < 0:
                start = 0

            # Render grain: windowed sine
            phase = np.random.random() * 2.0 * np.pi
            phase_inc = 2.0 * np.pi * freq / sr
            for i in range(grain_samples):
                idx = start + i
                if idx >= total_samples:
                    break
                val = amp * window[i] * np.sin(phase + phase_inc * i)
                output[idx] += val

    return output


@register_engine
class GranularEngine(Engine):
    name = "Granular"
    description = "Pointillist, cloud-like, shimmering texture."

    def __init__(self):
        self.grain_duration_ms = 20.0
        self.threshold = 0.15

    def render(self, magnitude, phase, sr, n_fft, hop_length):
        audio = _render_grains(magnitude, sr, n_fft, hop_length,
                               self.threshold, self.grain_duration_ms)
        return normalize(audio)
