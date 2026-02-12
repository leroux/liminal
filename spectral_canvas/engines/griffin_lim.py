"""Engine 5: Griffin-Lim Reconstruction — metallic, robotic, ghostly."""

import numpy as np
import librosa

from . import Engine, register_engine
from ..dsp.utils import normalize


@register_engine
class GriffinLimEngine(Engine):
    name = "Griffin-Lim"
    description = "Metallic, robotic, ghostly. Artifacts are the aesthetic."

    def render(self, magnitude, phase, sr, n_fft, hop_length):
        # Scale up from 0-1 range to reasonable amplitude for Griffin-Lim
        mag_scaled = magnitude * 80.0
        audio = librosa.griffinlim(
            mag_scaled,
            n_iter=60,
            hop_length=hop_length,
            win_length=n_fft,
            window='hann',
            momentum=0.99,
        )
        return normalize(audio)
