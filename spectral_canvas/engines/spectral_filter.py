"""Engine 11: Spectral Filter — faithful playback of imported audio with editable canvas."""

import numpy as np
from scipy.signal import istft
from scipy.ndimage import zoom

from . import Engine, register_engine
from ..dsp.utils import normalize


@register_engine
class SpectralFilterEngine(Engine):
    name = "Spectral Filter"
    description = "Plays imported audio faithfully. Edits act as gain changes on the original."

    def __init__(self):
        self.mode = 'reconstruct'  # reconstruct, multiply, freeze
        self.original_magnitude: np.ndarray | None = None  # linear mag from STFT
        self.import_display_magnitude: np.ndarray | None = None  # 0-1 display mag at import

    @staticmethod
    def _bass_gain_limit(n_bins, sr, n_fft):
        """Frequency-dependent gain clamp — more headroom for bass.

        Bass needs much more STFT magnitude for equal perceived loudness
        (Fletcher-Munson). Below 200 Hz allow up to 20x boost, tapering
        to 4x above 500 Hz.
        """
        freqs = np.arange(n_bins) * sr / n_fft
        max_gain = np.where(
            freqs < 200, 20.0,
            np.where(freqs < 500, 20.0 - (freqs - 200) / 300 * 16.0, 4.0))
        return max_gain[:, np.newaxis]  # broadcastable (n_bins, 1)

    @staticmethod
    def _bass_boost_curve(n_bins, sr, n_fft):
        """Perceptual bass compensation for synthetic content.

        When painting on silence, bass bins need extra magnitude to
        actually sound bassy. 4x below 100 Hz, tapering to 1x at 500 Hz.
        """
        freqs = np.arange(n_bins) * sr / n_fft
        boost = np.where(
            freqs < 100, 4.0,
            np.where(freqs < 500, 4.0 - (freqs - 100) / 400 * 3.0, 1.0))
        return boost[:, np.newaxis]

    def render(self, magnitude, phase, sr, n_fft, hop_length):
        n_bins = magnitude.shape[0]

        if phase is None:
            # No imported audio — fall back to random phase with bass compensation
            bass_boost = self._bass_boost_curve(n_bins, sr, n_fft)
            boosted = magnitude * bass_boost
            random_phase = np.random.uniform(0, 2 * np.pi, magnitude.shape).astype(np.float32)
            Zxx = boosted * np.exp(1j * random_phase)
            _, audio = istft(Zxx, fs=sr, window='hann', nperseg=n_fft,
                             noverlap=n_fft - hop_length)
            return normalize(audio)

        n_bins, n_frames = magnitude.shape

        if self.mode == 'freeze':
            frame_energy = np.sum(magnitude, axis=0)
            best_frame = np.argmax(frame_energy)
            frozen_mag = np.tile(magnitude[:, best_frame:best_frame + 1], (1, n_frames))
            random_phase = np.random.uniform(0, 2 * np.pi, frozen_mag.shape).astype(np.float32)
            Zxx = frozen_mag * np.exp(1j * random_phase)
            _, audio = istft(Zxx, fs=sr, window='hann', nperseg=n_fft,
                             noverlap=n_fft - hop_length)
            return normalize(audio)

        if self.mode == 'reconstruct' and self.original_magnitude is not None:
            orig_lin = self.original_magnitude
            orig_disp = self.import_display_magnitude

            # Resize if shapes don't match
            if orig_lin.shape != phase.shape:
                zf = (phase.shape[0] / orig_lin.shape[0],
                      phase.shape[1] / orig_lin.shape[1])
                orig_lin = zoom(orig_lin, zf, order=1)
            if orig_disp is not None and orig_disp.shape != magnitude.shape:
                zf = (magnitude.shape[0] / orig_disp.shape[0],
                      magnitude.shape[1] / orig_disp.shape[1])
                orig_disp = zoom(orig_disp, zf, order=1)

            if orig_disp is not None:
                # Compute gain: how the user changed the canvas relative to import
                has_energy = orig_disp > 0.02
                gain = np.ones_like(magnitude)
                gain[has_energy] = magnitude[has_energy] / (orig_disp[has_energy] + 1e-6)

                # Frequency-dependent gain clamp: more headroom for bass
                max_gain = self._bass_gain_limit(n_bins, sr, n_fft)
                gain = np.clip(gain, 0, max_gain)

                output_mag = orig_lin * gain

                # For bins the user painted where original was silent,
                # add synthetic magnitude with bass compensation
                painted_on_silence = (~has_energy) & (magnitude > 0.02)
                if np.any(painted_on_silence):
                    ref_level = np.percentile(orig_lin[orig_lin > 0], 50) if np.any(orig_lin > 0) else 1.0
                    bass_boost = self._bass_boost_curve(n_bins, sr, n_fft)
                    synth_mag = magnitude * ref_level * bass_boost
                    output_mag = np.where(painted_on_silence, synth_mag, output_mag)
            else:
                output_mag = orig_lin

            Zxx = output_mag * np.exp(1j * phase)
            _, audio = istft(Zxx, fs=sr, window='hann', nperseg=n_fft,
                             noverlap=n_fft - hop_length)
            return normalize(audio)

        # Fallback: multiply mode — canvas magnitude * original phase
        if phase.shape != magnitude.shape:
            zf = (phase.shape[0] / magnitude.shape[0],
                  phase.shape[1] / magnitude.shape[1])
            mag_resized = zoom(magnitude, zf, order=1)
        else:
            mag_resized = magnitude

        Zxx = mag_resized * np.exp(1j * phase)
        _, audio = istft(Zxx, fs=sr, window='hann', nperseg=n_fft,
                         noverlap=n_fft - hop_length)
        return normalize(audio)
