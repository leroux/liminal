"""Test fractional delay interpolation — generates WAV files.

Run: uv run python tests/test_fractional_delay.py

Key test: slowly modulate delay time and hear pitch shift (Doppler effect).
Also compares linear vs cubic interpolation quality.
"""

import numpy as np
from scipy.io import wavfile
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from primitives.delay_line import DelayLine

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


def make_tone(freq=440.0, seconds=4.0):
    """Sustained sine wave — good for hearing pitch modulation artifacts."""
    t = np.arange(int(SR * seconds)) / SR
    return 0.5 * np.sin(2 * np.pi * freq * t)


def make_noise_burst(duration_ms=15, total_seconds=2.0):
    signal = np.zeros(int(SR * total_seconds))
    burst_len = int(SR * duration_ms / 1000)
    burst = np.random.randn(burst_len)
    burst *= np.linspace(1.0, 0.0, burst_len)
    signal[:burst_len] = burst
    return signal


# ---------------------------------------------------------------------------
# Test 1: Modulated delay on a tone — hear pitch wobble (vibrato)
# ---------------------------------------------------------------------------
def test_modulated_delay():
    print("Test 1: Modulated delay time — vibrato effect")
    tone = make_tone(freq=440.0, seconds=4.0)
    n_samples = len(tone)

    base_delay = 500.0        # ~11ms base delay
    mod_depth = 40.0          # ±40 samples (~0.9ms)
    mod_rate = 3.0            # 3 Hz modulation

    for method_name, read_method in [("linear", "read_linear"), ("cubic", "read_cubic")]:
        dl = DelayLine(max_delay=SR)
        output = np.zeros(n_samples)
        for i in range(n_samples):
            dl.write(tone[i])
            # Sinusoidal delay modulation
            mod = mod_depth * np.sin(2 * np.pi * mod_rate * i / SR)
            current_delay = base_delay + mod
            output[i] = getattr(dl, read_method)(current_delay)
        save_wav(f"05_vibrato_{method_name}.wav", output)


# ---------------------------------------------------------------------------
# Test 2: Slow sweep — delay time ramps up then down (Doppler pitch shift)
# ---------------------------------------------------------------------------
def test_slow_sweep():
    print("Test 2: Slow delay sweep — hear Doppler pitch shift")
    tone = make_tone(freq=440.0, seconds=4.0)
    n_samples = len(tone)

    for method_name, read_method in [("linear", "read_linear"), ("cubic", "read_cubic")]:
        dl = DelayLine(max_delay=SR)
        output = np.zeros(n_samples)
        for i in range(n_samples):
            dl.write(tone[i])
            # Delay sweeps from 200 to 800 samples and back (triangle wave)
            phase = (i / SR) * 0.5  # 0.5 Hz triangle
            tri = 2.0 * abs(2.0 * (phase - int(phase + 0.5)))
            current_delay = 200.0 + 600.0 * tri
            output[i] = getattr(dl, read_method)(current_delay)
        save_wav(f"06_sweep_{method_name}.wav", output)


# ---------------------------------------------------------------------------
# Test 3: Feedback with modulated delay — chorus/flanger territory
# ---------------------------------------------------------------------------
def test_modulated_feedback():
    print("Test 3: Modulated feedback delay — chorus effect on noise burst")
    burst = make_noise_burst(total_seconds=3.0)
    n_samples = len(burst)

    base_delay = 600.0        # ~14ms
    mod_depth = 30.0          # ±30 samples
    mod_rate = 1.5            # 1.5 Hz
    feedback = 0.6

    dl = DelayLine(max_delay=SR)
    output = np.zeros(n_samples)
    for i in range(n_samples):
        mod = mod_depth * np.sin(2 * np.pi * mod_rate * i / SR)
        current_delay = base_delay + mod
        delayed = dl.read_cubic(current_delay)
        output[i] = burst[i] + delayed
        dl.write(burst[i] + feedback * delayed)
    save_wav("07_modulated_feedback.wav", output)


if __name__ == "__main__":
    print(f"Sample rate: {SR} Hz")
    print(f"Output dir: {OUTPUT_DIR}\n")
    test_modulated_delay()
    test_slow_sweep()
    test_modulated_feedback()
    print("\nDone! Compare linear vs cubic — cubic should sound cleaner on the sweeps.")
