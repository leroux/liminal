"""Engine 9: Formant Synthesis — vocal, choral, eerie singing."""

import numpy as np
from scipy.signal import lfilter
from scipy.ndimage import uniform_filter1d

from . import Engine, register_engine
from ..dsp.utils import normalize, generate_pulse_train, generate_white_noise


@register_engine
class FormantEngine(Engine):
    name = "Formant"
    description = "Vocal, choral, eerie singing. Drawing vowel shapes = hearing vowels."

    def render(self, magnitude, phase, sr, n_fft, hop_length):
        n_bins, n_frames = magnitude.shape
        total_samples = (n_frames - 1) * hop_length + n_fft
        output = np.zeros(total_samples, dtype=np.float64)

        # Generate excitation: pulse train + noise mix
        excitation = generate_pulse_train(total_samples / sr, 120.0, sr) * 0.7
        excitation += generate_white_noise(total_samples / sr, sr)[:len(excitation)] * 0.3

        for frame_idx in range(n_frames):
            col = magnitude[:, frame_idx].astype(np.float64)
            if np.max(col) < 0.05:
                continue

            # Smooth to find spectral envelope
            smooth_width = max(3, int(300.0 * n_fft / sr))
            envelope = uniform_filter1d(col, smooth_width)

            # Find formant peaks (local maxima of envelope)
            formants = []
            for i in range(2, n_bins - 2):
                if envelope[i] > envelope[i - 1] and envelope[i] > envelope[i + 1] and envelope[i] > 0.05:
                    freq = i * sr / n_fft
                    if 100 < freq < 8000:
                        formants.append((freq, envelope[i]))
            formants.sort(key=lambda x: -x[1])
            formants = formants[:5]  # max 5 formants

            if not formants:
                continue

            # Apply resonant bandpass filters
            start = frame_idx * hop_length
            end = min(start + hop_length, total_samples)
            segment = excitation[start:end].copy()

            frame_out = np.zeros(end - start, dtype=np.float64)
            for f_center, f_amp in formants:
                # 2nd order resonant bandpass (biquad)
                bw = 150.0  # bandwidth Hz
                w0 = 2.0 * np.pi * f_center / sr
                alpha = np.sin(w0) * np.sinh(np.log(2) / 2 * bw / f_center * w0 / np.sin(w0)) if np.sin(w0) > 1e-6 else 0.01

                b = np.array([alpha, 0, -alpha])
                a = np.array([1 + alpha, -2 * np.cos(w0), 1 - alpha])
                filtered = lfilter(b, a, segment)
                frame_out += filtered * f_amp

            output[start:end] += frame_out

        return normalize(output.astype(np.float32))
