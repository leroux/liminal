"""Engine 4: Karplus-Strong Physical Modeling — plucked strings, organic."""

import numba
import numpy as np
from scipy.ndimage import label

from . import Engine, register_engine
from ..dsp.utils import normalize


@numba.njit(cache=True)
def _karplus_strong(frequency, duration_samples, amplitude, sr, damping=0.996):
    period = int(sr / frequency)
    if period < 2:
        return np.zeros(duration_samples, dtype=np.float32)
    delay_line = np.random.randn(period).astype(np.float32) * amplitude
    output = np.zeros(duration_samples, dtype=np.float32)
    for i in range(duration_samples):
        idx = i % period
        output[i] = delay_line[idx]
        if i >= period:
            avg = 0.5 * (output[i - period] + output[i - period + 1])
            output[i] = damping * avg
            delay_line[idx] = output[i]
    return output


@register_engine
class KarplusStrongEngine(Engine):
    name = "Karplus-Strong"
    description = "Plucked strings, harps, marimbas. Strikingly organic."

    def render(self, magnitude, phase, sr, n_fft, hop_length):
        n_bins, n_frames = magnitude.shape
        total_samples = (n_frames - 1) * hop_length + n_fft
        output = np.zeros(total_samples, dtype=np.float32)

        # Threshold and find connected components
        binary = (magnitude > 0.1).astype(np.int32)
        labeled, n_objects = label(binary)

        for obj_id in range(1, n_objects + 1):
            ys, xs = np.where(labeled == obj_id)
            if len(ys) == 0:
                continue

            # Bounding box
            y_min, y_max = ys.min(), ys.max()
            x_min, x_max = xs.min(), xs.max()

            # Center frequency from median Y position
            center_bin = int(np.median(ys))
            freq = center_bin * sr / n_fft
            if freq < 20 or freq > 20000:
                continue

            # Amplitude from mean magnitude
            amp = float(np.mean(magnitude[ys, xs]))

            # Timing
            start_sample = x_min * hop_length
            end_sample = min((x_max + 1) * hop_length + n_fft, total_samples)
            dur_samples = end_sample - start_sample

            if dur_samples < 1:
                continue

            note = _karplus_strong(freq, dur_samples, amp, sr)
            end_idx = min(start_sample + dur_samples, total_samples)
            output[start_sample:end_idx] += note[:end_idx - start_sample]

        return normalize(output)
