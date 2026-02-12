"""Engine 1: Random Phase ISTFT — dreamy ambient textures."""

import numpy as np
from scipy.signal import istft

from . import Engine, register_engine
from ..dsp.utils import normalize


@register_engine
class RandomPhaseEngine(Engine):
    name = "Random Phase"
    description = "Dreamy, smeared, shimmering pad. All transients dissolve."

    def render(self, magnitude, phase, sr, n_fft, hop_length):
        # Assign random phase
        random_phase = np.random.uniform(0, 2 * np.pi, magnitude.shape).astype(np.float32)
        Zxx = magnitude * np.exp(1j * random_phase)
        _, audio = istft(Zxx, fs=sr, window='hann', nperseg=n_fft,
                         noverlap=n_fft - hop_length)
        return normalize(audio)
