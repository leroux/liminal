"""Test the delay line primitive — generates WAV files you can listen to.

Run: uv run python tests/test_delay_line.py

Outputs go to audio/test_signals/. Listen to each one to verify the delay line works.
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
    # Normalize to prevent clipping, then convert to 16-bit
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.9
    wavfile.write(path, sr, (audio * 32767).astype(np.int16))
    print(f"  Saved {path} ({len(audio)/sr:.2f}s)")


def make_impulse(length_seconds=1.0, total_seconds=None):
    """Single sample of 1.0 at the start, rest zeros."""
    dur = total_seconds if total_seconds is not None else length_seconds
    signal = np.zeros(int(SR * dur))
    signal[0] = 1.0
    return signal


def make_noise_burst(duration_ms=15, total_seconds=1.0):
    """Short noise burst — sounds like a clap/snap. Good for hearing echoes."""
    signal = np.zeros(int(SR * total_seconds))
    burst_len = int(SR * duration_ms / 1000)
    burst = np.random.randn(burst_len)
    # Apply quick fade-out envelope
    burst *= np.linspace(1.0, 0.0, burst_len)
    signal[:burst_len] = burst
    return signal


def make_pluck(freq=220.0, total_seconds=2.0):
    """Decaying sine wave — like a plucked string. Good for hearing pitch effects."""
    t = np.arange(int(SR * total_seconds)) / SR
    envelope = np.exp(-t * 4.0)
    return np.sin(2 * np.pi * freq * t) * envelope


# ---------------------------------------------------------------------------
# Test 1: Single echo — impulse through delay line
# ---------------------------------------------------------------------------
def test_single_echo():
    print("Test 1: Single echo (impulse -> 200ms delay)")
    dl = DelayLine(max_delay=SR)  # 1 second max
    impulse = make_impulse(length_seconds=1.0)
    delay_samples = int(0.2 * SR)  # 200ms

    output = np.zeros_like(impulse)
    for i, sample in enumerate(impulse):
        dl.write(sample)
        output[i] = dl.read(delay_samples)

    # Mix dry + delayed so you hear the original then the echo
    result = impulse + 0.7 * output
    save_wav("01_single_echo_impulse.wav", result)

    # Same test with noise burst (easier to hear)
    dl.reset()
    burst = make_noise_burst(total_seconds=1.0)
    output = np.zeros_like(burst)
    for i, sample in enumerate(burst):
        dl.write(sample)
        output[i] = dl.read(delay_samples)
    result = burst + 0.7 * output
    save_wav("01_single_echo_burst.wav", result)


# ---------------------------------------------------------------------------
# Test 2: Feedback loop — repeating echoes
# ---------------------------------------------------------------------------
def test_feedback_echo():
    print("Test 2: Feedback echo (burst -> delay with feedback)")
    dl = DelayLine(max_delay=SR)
    burst = make_noise_burst(total_seconds=3.0)
    delay_samples = int(0.25 * SR)  # 250ms
    feedback = 0.6

    output = np.zeros_like(burst)
    for i, sample in enumerate(burst):
        delayed = dl.read(delay_samples)
        output[i] = sample + delayed
        dl.write(sample + feedback * delayed)

    save_wav("02_feedback_echo.wav", output)


# ---------------------------------------------------------------------------
# Test 3: Short delay + high feedback = comb filter (pitched tone)
# ---------------------------------------------------------------------------
def test_comb_filter():
    print("Test 3: Comb filter — short delay + high feedback = pitched tone")

    for delay_ms, label in [(4.5, "4.5ms_~222Hz"), (2.3, "2.3ms_~435Hz"), (10.0, "10ms_~100Hz")]:
        dl = DelayLine(max_delay=SR)
        impulse = make_impulse(total_seconds=2.0)
        delay_samples = int(delay_ms / 1000 * SR)
        feedback = 0.98

        output = np.zeros_like(impulse)
        for i, sample in enumerate(impulse):
            delayed = dl.read(delay_samples)
            output[i] = sample + delayed
            dl.write(sample + feedback * delayed)

        save_wav(f"03_comb_{label}.wav", output)


# ---------------------------------------------------------------------------
# Test 4: Delay applied to a tonal signal (pluck)
# ---------------------------------------------------------------------------
def test_pluck_echo():
    print("Test 4: Pluck with feedback echo")
    dl = DelayLine(max_delay=SR)
    pluck = make_pluck(freq=220.0, total_seconds=3.0)
    delay_samples = int(0.3 * SR)  # 300ms
    feedback = 0.5

    output = np.zeros_like(pluck)
    for i, sample in enumerate(pluck):
        delayed = dl.read(delay_samples)
        output[i] = sample + delayed
        dl.write(sample + feedback * delayed)

    save_wav("04_pluck_echo.wav", output)


if __name__ == "__main__":
    print(f"Sample rate: {SR} Hz")
    print(f"Output dir: {OUTPUT_DIR}\n")
    test_single_echo()
    test_feedback_echo()
    test_comb_filter()
    test_pluck_echo()
    print("\nDone! Listen to the WAVs in audio/test_signals/")
