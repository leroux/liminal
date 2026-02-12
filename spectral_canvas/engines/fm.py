"""Engine 7: FM Synthesis — metallic bells, glassy tones, DX7 electric piano."""

import numba
import numpy as np

from . import Engine, register_engine
from ..dsp.utils import normalize


@numba.njit(cache=True)
def _find_peaks(col, n_bins):
    """Find local maxima in a magnitude column."""
    peaks = np.zeros(n_bins, dtype=numba.int32)
    amps = np.zeros(n_bins, dtype=numba.float32)
    n_peaks = 0
    for i in range(2, n_bins - 2):
        if col[i] > 0.1 and col[i] > col[i - 1] and col[i] > col[i + 1]:
            peaks[n_peaks] = i
            amps[n_peaks] = col[i]
            n_peaks += 1
            if n_peaks >= n_bins:
                break
    return peaks[:n_peaks], amps[:n_peaks]


@numba.njit(cache=True)
def _render_fm(magnitude, sr, n_fft, hop_length, cm_ratio_num, cm_ratio_den):
    n_bins, n_frames = magnitude.shape
    total_samples = (n_frames - 1) * hop_length + n_fft
    output = np.zeros(total_samples, dtype=np.float32)
    frame_dur = hop_length / sr

    for frame_idx in range(n_frames):
        col = magnitude[:, frame_idx]
        peaks, amps = _find_peaks(col, n_bins)

        for p in range(len(peaks)):
            bin_idx = peaks[p]
            amp = amps[p]
            fc = bin_idx * sr / n_fft
            if fc < 20 or fc > 20000:
                continue

            # FM parameters
            fm_freq = fc * cm_ratio_den / cm_ratio_num
            beta = amp * 8.0  # modulation index

            start = frame_idx * hop_length
            end = min(start + hop_length + n_fft // 2, total_samples)

            # Crossfade window
            dur = end - start
            for i in range(dur):
                t = (start + i) / sr
                # Hann fade in/out
                env = 1.0
                fade = min(128, dur // 4)
                if i < fade:
                    env = i / fade
                elif i > dur - fade:
                    env = (dur - i) / fade
                val = amp * env * np.sin(2.0 * np.pi * fc * t + beta * np.sin(2.0 * np.pi * fm_freq * t))
                if start + i < total_samples:
                    output[start + i] += val

    return output


@register_engine
class FMEngine(Engine):
    name = "FM"
    description = "Metallic bells, glassy tones, DX7 electric piano."

    def __init__(self):
        self.carrier_mod_ratio = (1, 1)  # carrier:modulator

    def render(self, magnitude, phase, sr, n_fft, hop_length):
        c, m = self.carrier_mod_ratio
        audio = _render_fm(magnitude, sr, n_fft, hop_length, c, m)
        return normalize(audio)
