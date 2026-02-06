"""Test one-pole filter — generates WAV files.

Run: uv run python tests/test_one_pole.py

Key test: feedback delay with damping — repeats darken over time.
"""

import numpy as np
from scipy.io import wavfile
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from primitives.delay_line import DelayLine
from primitives.filters import OnePoleFilter

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


def make_noise_burst(duration_ms=15, total_seconds=2.0):
    signal = np.zeros(int(SR * total_seconds))
    burst_len = int(SR * duration_ms / 1000)
    burst = np.random.randn(burst_len)
    burst *= np.linspace(1.0, 0.0, burst_len)
    signal[:burst_len] = burst
    return signal


# ---------------------------------------------------------------------------
# Test 1: Filter white noise — hear lowpass effect
# ---------------------------------------------------------------------------
def test_filter_noise():
    print("Test 1: One-pole on white noise — compare coefficients")
    noise = np.random.randn(int(SR * 2.0)) * 0.5

    for coeff in [0.0, 0.5, 0.9, 0.99]:
        filt = OnePoleFilter(coeff=coeff)
        output = np.zeros_like(noise)
        for i in range(len(noise)):
            output[i] = filt.process(noise[i])
        save_wav(f"08_onepole_noise_a{coeff}.wav", output)


# ---------------------------------------------------------------------------
# Test 2: Feedback delay with damping — repeats darken over time
# ---------------------------------------------------------------------------
def test_damped_feedback():
    print("Test 2: Feedback delay + damping — repeats darken")
    burst = make_noise_burst(total_seconds=4.0)
    n_samples = len(burst)
    delay_samples = int(0.25 * SR)  # 250ms
    feedback = 0.7

    # Without damping
    dl = DelayLine(max_delay=SR)
    output = np.zeros(n_samples)
    for i in range(n_samples):
        delayed = dl.read(delay_samples)
        output[i] = burst[i] + delayed
        dl.write(burst[i] + feedback * delayed)
    save_wav("09_feedback_no_damping.wav", output)

    # With damping
    dl = DelayLine(max_delay=SR)
    filt = OnePoleFilter(coeff=0.7)
    output = np.zeros(n_samples)
    for i in range(n_samples):
        delayed = dl.read(delay_samples)
        damped = filt.process(delayed)
        output[i] = burst[i] + damped
        dl.write(burst[i] + feedback * damped)
    save_wav("09_feedback_damped_0.7.wav", output)

    # Heavy damping
    dl = DelayLine(max_delay=SR)
    filt = OnePoleFilter(coeff=0.9)
    output = np.zeros(n_samples)
    for i in range(n_samples):
        delayed = dl.read(delay_samples)
        damped = filt.process(delayed)
        output[i] = burst[i] + damped
        dl.write(burst[i] + feedback * damped)
    save_wav("09_feedback_damped_0.9.wav", output)


if __name__ == "__main__":
    print(f"Sample rate: {SR} Hz")
    print(f"Output dir: {OUTPUT_DIR}\n")
    test_filter_noise()
    test_damped_feedback()
    print("\nDone! Compare 09_feedback_no_damping vs 09_feedback_damped — repeats should get duller.")
