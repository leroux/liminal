"""Engine 2: Additive Synthesis via IFFT — clean, organ-like, precise.

Uses coherent phase that advances correctly for each frequency bin,
producing smooth, musical tones instead of harsh noise-like artifacts.
"""

import numpy as np
from scipy.signal.windows import hann

from . import Engine, register_engine
from ..dsp.utils import normalize


@register_engine
class AdditiveEngine(Engine):
    name = "Additive"
    description = "Clean, organ-like, precise. Coherent phase for smooth tones."

    def render(self, magnitude, phase_in, sr, n_fft, hop_length):
        n_bins, n_frames = magnitude.shape
        total_samples = (n_frames - 1) * hop_length + n_fft
        output = np.zeros(total_samples, dtype=np.float64)
        window = hann(n_fft, sym=False)

        # Bass loudness compensation: low freqs need more magnitude
        # for equal perceived loudness (Fletcher-Munson)
        freqs = np.arange(n_bins) * sr / n_fft
        bass_boost = np.where(
            freqs < 100, 3.0,
            np.where(freqs < 500, 3.0 - (freqs - 100) / 400 * 2.0, 1.0))

        # Coherent phase: each bin advances by its natural frequency per hop
        # freq of bin b = b * sr / n_fft
        # Phase advance per hop = 2*pi * freq * hop_length / sr
        #                       = 2*pi * b * hop_length / n_fft
        phase_inc = np.arange(n_bins, dtype=np.float64) * 2 * np.pi * hop_length / n_fft
        phase = np.zeros(n_bins, dtype=np.float64)

        for i in range(n_frames):
            col = magnitude[:, i] * bass_boost
            spectrum = col * np.exp(1j * phase)
            frame = np.fft.irfft(spectrum, n=n_fft) * window
            start = i * hop_length
            end = start + n_fft
            if end <= total_samples:
                output[start:end] += frame
            else:
                valid = total_samples - start
                output[start:] += frame[:valid]

            # Advance phase coherently
            phase += phase_inc
            # Keep in [0, 2pi] to avoid float precision drift
            phase %= (2 * np.pi)

        return normalize(output)
