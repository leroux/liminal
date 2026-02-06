"""Test each DSP primitive in isolation — generates WAV files.

Run: uv run python tests/test_primitives.py
"""

import numpy as np
from scipy.io import wavfile
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.params import SR
from primitives.dsp import (
    delay, delay_feedback, one_pole_lowpass,
    allpass, allpass_chain, comb_filter, saturate,
)

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


def make_click_train(n_clicks=4, interval_s=0.5, total_s=4.0):
    """Short clicks spaced apart — good for hearing delay/echo effects."""
    audio = np.zeros(int(SR * total_s))
    burst_len = int(SR * 0.005)
    burst = np.random.randn(burst_len) * np.linspace(1, 0, burst_len)
    for i in range(n_clicks):
        offset = int(i * interval_s * SR)
        end = min(offset + burst_len, len(audio))
        audio[offset:end] += burst[:end - offset]
    return audio


def make_noise(seconds=3.0):
    return np.random.randn(int(SR * seconds)) * 0.5


def make_pluck(freq=220.0, seconds=3.0):
    t = np.arange(int(SR * seconds)) / SR
    return np.sin(2 * np.pi * freq * t) * np.exp(-t * 4.0)


# ---- Delay ----
def test_delay():
    print("\n--- Delay ---")
    clicks = make_click_train()

    # Simple delay
    out = delay(clicks, int(0.25 * SR))
    save_wav("01_delay_250ms.wav", clicks + out)

    # Feedback delay (echo)
    for fb in [0.3, 0.7, 0.95]:
        out = delay_feedback(clicks, int(0.2 * SR), fb, wet=0.5)
        save_wav(f"02_delay_feedback_{fb}.wav", out)


# ---- One-pole lowpass ----
def test_one_pole():
    print("\n--- One-pole lowpass ---")
    noise = make_noise()
    for coeff in [0.0, 0.5, 0.9, 0.99]:
        out = one_pole_lowpass(noise, coeff)
        save_wav(f"03_onepole_{coeff}.wav", out)


# ---- Allpass ----
def test_allpass():
    print("\n--- Allpass ---")
    clicks = make_click_train()

    # Single allpass — hear the smearing
    out = allpass(clicks, int(5.0 / 1000 * SR), 0.5)
    save_wav("04_allpass_single.wav", out)

    # Chain of 1, 4, 8 allpasses — hear diffusion build
    delays = np.array([int(d / 1000 * SR) for d in [5.3, 7.9, 11.7, 16.1, 19.3, 23.7, 29.1, 31.7]])
    for n_stages in [1, 4, 8]:
        out = allpass_chain(clicks, delays[:n_stages], 0.5)
        save_wav(f"05_allpass_chain_{n_stages}.wav", out)


# ---- Comb filter ----
def test_comb():
    print("\n--- Comb filter ---")
    clicks = make_click_train()

    # Basic comb — hear the pitched resonance
    for delay_ms in [2.0, 10.0, 50.0]:
        dt = max(1, int(delay_ms / 1000 * SR))
        out = comb_filter(clicks, dt, 0.9)
        save_wav(f"06_comb_{delay_ms}ms.wav", out)

    # Comb with damping
    dt = int(30.0 / 1000 * SR)
    for damp in [0.0, 0.5, 0.9]:
        out = comb_filter(clicks, dt, 0.9, damping=damp)
        save_wav(f"07_comb_damped_{damp}.wav", out)


# ---- Saturation ----
def test_saturation():
    print("\n--- Saturation ---")
    pluck = make_pluck()

    for drive in [1.0, 3.0, 10.0, 50.0]:
        out = saturate(pluck, drive)
        save_wav(f"08_saturate_drive_{drive}.wav", out)


if __name__ == "__main__":
    print(f"Sample rate: {SR} Hz")
    print(f"Output dir: {OUTPUT_DIR}")
    test_delay()
    test_one_pole()
    test_allpass()
    test_comb()
    test_saturation()
    print("\nDone!")
