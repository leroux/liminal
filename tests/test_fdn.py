"""Test the full FDN engine — generates WAV files.

Run: uv run python tests/test_fdn.py
"""

import numpy as np
from scipy.io import wavfile
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.fdn import render_fdn
from engine.params import default_params, SR

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


def make_impulse(seconds=2.0):
    signal = np.zeros(int(SR * seconds))
    signal[0] = 1.0
    return signal


def make_noise_burst(duration_ms=15, total_seconds=3.0):
    signal = np.zeros(int(SR * total_seconds))
    burst_len = int(SR * duration_ms / 1000)
    burst = np.random.randn(burst_len)
    burst *= np.linspace(1.0, 0.0, burst_len)
    signal[:burst_len] = burst
    return signal


def make_pluck(freq=220.0, total_seconds=3.0):
    t = np.arange(int(SR * total_seconds)) / SR
    envelope = np.exp(-t * 4.0)
    return np.sin(2 * np.pi * freq * t) * envelope


def timed_render(input_audio, params, label):
    t0 = time.time()
    output = render_fdn(input_audio, params)
    elapsed = time.time() - t0
    realtime = len(input_audio) / SR
    ratio = realtime / elapsed
    print(f"  [{label}] {realtime:.1f}s audio in {elapsed:.2f}s ({ratio:.1f}x realtime)")
    return output


# ---------------------------------------------------------------------------
# Test 1: Impulse response — the reverb's fingerprint
# ---------------------------------------------------------------------------
def test_impulse_response():
    print("Test 1: Impulse response (default params)")
    params = default_params()
    impulse = make_impulse(seconds=4.0)
    output = timed_render(impulse, params, "impulse")
    save_wav("20_fdn_impulse_response.wav", output)


# ---------------------------------------------------------------------------
# Test 2: Noise burst — easier to hear reverb character
# ---------------------------------------------------------------------------
def test_burst():
    print("Test 2: Noise burst through FDN")
    params = default_params()
    burst = make_noise_burst(total_seconds=4.0)
    output = timed_render(burst, params, "burst")
    save_wav("21_fdn_burst.wav", output)


# ---------------------------------------------------------------------------
# Test 3: Pluck — musical signal
# ---------------------------------------------------------------------------
def test_pluck():
    print("Test 3: Pluck through FDN")
    params = default_params()
    pluck = make_pluck(total_seconds=4.0)
    output = timed_render(pluck, params, "pluck")
    save_wav("22_fdn_pluck.wav", output)


# ---------------------------------------------------------------------------
# Test 4: Vary feedback — short room to long hall
# ---------------------------------------------------------------------------
def test_feedback_range():
    print("Test 4: Varying feedback gain")
    burst = make_noise_burst(total_seconds=4.0)
    for fb in [0.5, 0.85, 0.95]:
        params = default_params()
        params["feedback_gain"] = fb
        output = timed_render(burst, params, f"fb={fb}")
        save_wav(f"23_fdn_feedback_{fb}.wav", output)


# ---------------------------------------------------------------------------
# Test 5: Vary damping — bright to dark
# ---------------------------------------------------------------------------
def test_damping_range():
    print("Test 5: Varying damping")
    burst = make_noise_burst(total_seconds=4.0)
    for damp in [0.0, 0.3, 0.7, 0.95]:
        params = default_params()
        params["damping_coeffs"] = [damp] * 8
        output = timed_render(burst, params, f"damp={damp}")
        save_wav(f"24_fdn_damping_{damp}.wav", output)


# ---------------------------------------------------------------------------
# Test 6: Wet/dry mix
# ---------------------------------------------------------------------------
def test_wet_dry():
    print("Test 6: Wet/dry mix")
    pluck = make_pluck(total_seconds=4.0)
    for mix in [0.0, 0.3, 0.7, 1.0]:
        params = default_params()
        params["wet_dry"] = mix
        output = timed_render(pluck, params, f"mix={mix}")
        save_wav(f"25_fdn_wetdry_{mix}.wav", output)


# ---------------------------------------------------------------------------
# Test 7: No diffusion vs full diffusion
# ---------------------------------------------------------------------------
def test_diffusion():
    print("Test 7: Diffusion off vs on")
    burst = make_noise_burst(total_seconds=4.0)

    params = default_params()
    params["diffusion"] = 0.0
    output = timed_render(burst, params, "diff=0")
    save_wav("26_fdn_diffusion_off.wav", output)

    params = default_params()
    params["diffusion"] = 0.6
    output = timed_render(burst, params, "diff=0.6")
    save_wav("26_fdn_diffusion_on.wav", output)


if __name__ == "__main__":
    print(f"Sample rate: {SR} Hz")
    print(f"Output dir: {OUTPUT_DIR}\n")
    test_impulse_response()
    test_burst()
    test_pluck()
    test_feedback_range()
    test_damping_range()
    test_wet_dry()
    test_diffusion()
    print("\nDone! Start with 20_fdn_impulse_response.wav and 21_fdn_burst.wav")
