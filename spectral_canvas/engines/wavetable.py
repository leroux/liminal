"""Engine 8: Wavetable Synthesis — morphing, evolving, modern synth."""

import numpy as np
from scipy.signal.windows import hann

from . import Engine, register_engine
from ..dsp.utils import normalize


@register_engine
class WavetableEngine(Engine):
    name = "Wavetable"
    description = "Morphing, evolving, modern synth. Like Serum."

    def __init__(self):
        self.base_freq = 220.0

    def render(self, magnitude, phase, sr, n_fft, hop_length):
        n_bins, n_frames = magnitude.shape
        total_samples = (n_frames - 1) * hop_length + n_fft

        # Build wavetable: IFFT each column into a single cycle
        wavetable = np.zeros((n_frames, n_fft), dtype=np.float32)
        for i in range(n_frames):
            col = magnitude[:, i]
            rand_phase = np.random.uniform(0, 2 * np.pi, len(col))
            spectrum = col * np.exp(1j * rand_phase)
            cycle = np.fft.irfft(spectrum, n=n_fft).astype(np.float32)
            peak = np.max(np.abs(cycle))
            if peak > 0:
                cycle /= peak
            wavetable[i] = cycle

        # Playback: scan through wavetable at base_freq
        output = np.zeros(total_samples, dtype=np.float32)
        phase_acc = 0.0
        samples_per_cycle = sr / self.base_freq

        for i in range(total_samples):
            # Which wavetable frame are we at?
            wt_pos = i / total_samples * (n_frames - 1)
            wt_idx = int(wt_pos)
            wt_frac = wt_pos - wt_idx
            wt_idx = min(wt_idx, n_frames - 2)

            # Position within cycle
            cycle_pos = phase_acc % n_fft
            idx_a = int(cycle_pos)
            idx_b = (idx_a + 1) % n_fft
            frac = cycle_pos - idx_a

            # Interpolate between wavetable frames and within cycle
            val_a = wavetable[wt_idx, idx_a] * (1 - frac) + wavetable[wt_idx, idx_b] * frac
            val_b = wavetable[wt_idx + 1, idx_a] * (1 - frac) + wavetable[wt_idx + 1, idx_b] * frac
            output[i] = val_a * (1 - wt_frac) + val_b * wt_frac

            phase_acc += n_fft / samples_per_cycle

        return normalize(output)
