"""Engine 10: Phase Vocoder — crystalline to watery time-stretching."""

import numpy as np
from scipy.signal import istft

from . import Engine, register_engine
from ..dsp.utils import normalize


@register_engine
class PhaseVocoderEngine(Engine):
    name = "Phase Vocoder"
    description = "Crystalline for moderate stretch, watery/phasey at extremes."

    def __init__(self):
        self.time_stretch = 1.0  # 0.25x to 4x

    def render(self, magnitude, phase_data, sr, n_fft, hop_length):
        n_bins, n_frames = magnitude.shape
        stretch = max(0.25, min(4.0, self.time_stretch))

        # Target number of output frames
        out_frames = int(n_frames * stretch)
        out_mag = np.zeros((n_bins, out_frames), dtype=np.float32)

        # Resample magnitude to new time scale
        for k in range(n_bins):
            x_in = np.linspace(0, n_frames - 1, n_frames)
            x_out = np.linspace(0, n_frames - 1, out_frames)
            out_mag[k] = np.interp(x_out, x_in, magnitude[k])

        # Phase propagation using instantaneous frequency estimation
        out_phase = np.zeros((n_bins, out_frames), dtype=np.float32)
        expected_phase_advance = 2.0 * np.pi * np.arange(n_bins) * hop_length / n_fft

        if phase_data is not None:
            # Use original phase for instantaneous frequency
            for m in range(1, out_frames):
                src_frame = min(int(m / stretch), n_frames - 1)
                src_prev = max(0, src_frame - 1)
                dphi = phase_data[:, src_frame] - phase_data[:, src_prev] - expected_phase_advance
                # Unwrap
                dphi = dphi - 2 * np.pi * np.round(dphi / (2 * np.pi))
                inst_freq = expected_phase_advance + dphi
                out_phase[:, m] = out_phase[:, m - 1] + inst_freq * stretch
        else:
            # No original phase — use random phase with coherent propagation
            out_phase[:, 0] = np.random.uniform(0, 2 * np.pi, n_bins).astype(np.float32)
            for m in range(1, out_frames):
                out_phase[:, m] = out_phase[:, m - 1] + expected_phase_advance

        # Reconstruct
        Zxx = out_mag * np.exp(1j * out_phase)
        _, audio = istft(Zxx, fs=sr, window='hann', nperseg=n_fft,
                         noverlap=n_fft - hop_length)
        return normalize(audio)
