"""Engine 3: Subtractive Synthesis — warm, thick, analog."""

import numpy as np
from scipy.signal import stft, istft
from scipy.ndimage import zoom

from . import Engine, register_engine
from ..dsp.utils import normalize, generate_sawtooth, generate_white_noise, generate_pink_noise


@register_engine
class SubtractiveEngine(Engine):
    name = "Subtractive"
    description = "Warm, thick, analog. The source waveform's DNA bleeds through."

    def __init__(self):
        self.source_type = 'sawtooth_55'  # default
        self.original_magnitude: np.ndarray | None = None  # set by player

    def render(self, magnitude, phase, sr, n_fft, hop_length):
        n_bins, n_frames = magnitude.shape
        duration = (n_frames - 1) * hop_length / sr + n_fft / sr

        if self.source_type == 'imported' and self.original_magnitude is not None and phase is not None:
            # Reconstruct imported audio as the source signal
            orig = self.original_magnitude
            if orig.shape != phase.shape:
                zf = (phase.shape[0] / orig.shape[0], phase.shape[1] / orig.shape[1])
                orig = zoom(orig, zf, order=1)
            Zxx_source = orig * np.exp(1j * phase)
        else:
            # Generate source signal
            if self.source_type == 'sawtooth_55':
                source = generate_sawtooth(duration, 55.0, sr)
            elif self.source_type == 'sawtooth_110':
                source = generate_sawtooth(duration, 110.0, sr)
            elif self.source_type == 'pink_noise':
                source = generate_pink_noise(duration, sr)
            else:
                source = generate_white_noise(duration, sr)

            _, _, Zxx_source = stft(source, fs=sr, window='hann',
                                    nperseg=n_fft, noverlap=n_fft - hop_length)

        # Resize magnitude to match source STFT shape
        if Zxx_source.shape != magnitude.shape:
            zoom_factors = (Zxx_source.shape[0] / magnitude.shape[0],
                           Zxx_source.shape[1] / magnitude.shape[1])
            mag_resized = zoom(magnitude, zoom_factors, order=1).astype(np.float32)
        else:
            mag_resized = magnitude

        Zxx_filtered = Zxx_source * mag_resized
        _, audio = istft(Zxx_filtered, fs=sr, window='hann',
                         nperseg=n_fft, noverlap=n_fft - hop_length)
        return normalize(audio)
