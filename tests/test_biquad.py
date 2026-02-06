"""Test biquad filter — generates WAV files.

Run: uv run python tests/test_biquad.py

Key test: sweep cutoff frequency on white noise — hear the filter shape.
"""

import numpy as np
from scipy.io import wavfile
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from primitives.filters import BiquadFilter

SR = 44100
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "audio", "test_signals")


def save_wav(filename, audio, sr=SR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.9
    wavfile.write(path, sr, (audio * 32767).astype(np.int16))
    print(f"  Saved {path} ({len(audio)/sr:.2f}s)")


# ---------------------------------------------------------------------------
# Test 1: Static filters on white noise — hear each filter type
# ---------------------------------------------------------------------------
def test_static_filters():
    print("Test 1: Static biquad filters on white noise")
    noise = np.random.randn(int(SR * 2.0)) * 0.3

    configs = [
        ("lowpass_1000Hz", BiquadFilter.lowpass(1000, 0.707, SR)),
        ("highpass_1000Hz", BiquadFilter.highpass(1000, 0.707, SR)),
        ("bandpass_1000Hz", BiquadFilter.bandpass(1000, 2.0, SR)),
        ("lowpass_200Hz", BiquadFilter.lowpass(200, 0.707, SR)),
        ("lowpass_5000Hz", BiquadFilter.lowpass(5000, 0.707, SR)),
        ("lowpass_1000Hz_highQ", BiquadFilter.lowpass(1000, 8.0, SR)),
    ]

    for name, filt in configs:
        output = np.zeros_like(noise)
        for i in range(len(noise)):
            output[i] = filt.process(noise[i])
        filt.reset()
        save_wav(f"10_biquad_{name}.wav", output)


# ---------------------------------------------------------------------------
# Test 2: Sweep cutoff frequency on white noise
# ---------------------------------------------------------------------------
def test_cutoff_sweep():
    print("Test 2: Lowpass cutoff sweep on white noise")
    duration = 4.0
    n_samples = int(SR * duration)
    noise = np.random.randn(n_samples) * 0.3

    filt = BiquadFilter.lowpass(200, 0.707, SR)
    output = np.zeros(n_samples)

    # Sweep from 200 Hz to 10000 Hz over 4 seconds (exponential)
    for i in range(n_samples):
        t = i / n_samples
        freq = 200.0 * (10000.0 / 200.0) ** t  # exponential sweep
        # Recompute coefficients (not efficient, but fine for a test)
        new_filt = BiquadFilter.lowpass(freq, 0.707, SR)
        filt.set_coeffs(new_filt.b0, new_filt.b1, new_filt.b2, new_filt.a1, new_filt.a2)
        output[i] = filt.process(noise[i])

    save_wav("11_biquad_cutoff_sweep.wav", output)

    # Same with high Q — hear the resonant peak
    print("Test 2b: Cutoff sweep with high Q (resonant)")
    filt = BiquadFilter.lowpass(200, 8.0, SR)
    output = np.zeros(n_samples)
    for i in range(n_samples):
        t = i / n_samples
        freq = 200.0 * (10000.0 / 200.0) ** t
        new_filt = BiquadFilter.lowpass(freq, 8.0, SR)
        filt.set_coeffs(new_filt.b0, new_filt.b1, new_filt.b2, new_filt.a1, new_filt.a2)
        output[i] = filt.process(noise[i])

    save_wav("11_biquad_cutoff_sweep_highQ.wav", output)


# ---------------------------------------------------------------------------
# Test 3: Shelving filters on white noise
# ---------------------------------------------------------------------------
def test_shelving():
    print("Test 3: Shelving filters on white noise")
    noise = np.random.randn(int(SR * 2.0)) * 0.3

    configs = [
        ("low_shelf_+12dB_500Hz", BiquadFilter.low_shelf(500, 12.0, SR)),
        ("low_shelf_-12dB_500Hz", BiquadFilter.low_shelf(500, -12.0, SR)),
        ("high_shelf_+12dB_2000Hz", BiquadFilter.high_shelf(2000, 12.0, SR)),
        ("high_shelf_-12dB_2000Hz", BiquadFilter.high_shelf(2000, -12.0, SR)),
    ]

    for name, filt in configs:
        output = np.zeros_like(noise)
        for i in range(len(noise)):
            output[i] = filt.process(noise[i])
        filt.reset()
        save_wav(f"12_biquad_{name}.wav", output)


if __name__ == "__main__":
    print(f"Sample rate: {SR} Hz")
    print(f"Output dir: {OUTPUT_DIR}\n")
    test_static_filters()
    test_cutoff_sweep()
    test_shelving()
    print("\nDone! The cutoff sweep is the most fun — hear the filter open up.")
