"""Test allpass filter — generates WAV files.

Run: uv run python tests/test_allpass.py

Key test: impulse through chain of allpasses — hear transient smear into a cloud.
"""

import numpy as np
from scipy.io import wavfile
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from primitives.filters import AllpassFilter

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


def make_impulse(seconds=1.0):
    signal = np.zeros(int(SR * seconds))
    signal[0] = 1.0
    return signal


def make_noise_burst(duration_ms=15, total_seconds=2.0):
    signal = np.zeros(int(SR * total_seconds))
    burst_len = int(SR * duration_ms / 1000)
    burst = np.random.randn(burst_len)
    burst *= np.linspace(1.0, 0.0, burst_len)
    signal[:burst_len] = burst
    return signal


# ---------------------------------------------------------------------------
# Test 1: Single allpass on impulse — hear the smearing
# ---------------------------------------------------------------------------
def test_single_allpass():
    print("Test 1: Single allpass on impulse")
    impulse = make_impulse(seconds=1.0)

    for delay_ms in [5, 20, 50]:
        delay_samples = int(delay_ms / 1000 * SR)
        ap = AllpassFilter(delay_samples, gain=0.5)
        output = np.zeros_like(impulse)
        for i in range(len(impulse)):
            output[i] = ap.process(impulse[i])
        save_wav(f"13_allpass_single_{delay_ms}ms.wav", output)


# ---------------------------------------------------------------------------
# Test 2: Chain of allpasses — hear transient become diffuse cloud
# ---------------------------------------------------------------------------
def test_allpass_chain():
    print("Test 2: Chain of allpasses on noise burst — diffusion")
    burst = make_noise_burst(total_seconds=2.0)

    # Mutually prime delay lengths for maximum diffusion
    delay_lengths_ms = [7.1, 11.3, 16.9, 23.7]

    for n_stages in [1, 2, 4]:
        chain = [AllpassFilter(int(d / 1000 * SR), gain=0.6)
                 for d in delay_lengths_ms[:n_stages]]
        output = np.zeros_like(burst)
        for i in range(len(burst)):
            s = burst[i]
            for ap in chain:
                s = ap.process(s)
            output[i] = s
        save_wav(f"14_allpass_chain_{n_stages}stage.wav", output)


# ---------------------------------------------------------------------------
# Test 3: Impulse through long chain — the classic diffusion demo
# ---------------------------------------------------------------------------
def test_impulse_diffusion():
    print("Test 3: Impulse through 8-stage allpass chain — cloud from click")
    impulse = make_impulse(seconds=2.0)

    # 8 allpasses with varied prime-ish delay lengths
    delays_ms = [5.3, 7.9, 11.7, 16.1, 22.3, 31.1, 41.9, 53.7]
    chain = [AllpassFilter(int(d / 1000 * SR), gain=0.5) for d in delays_ms]

    output = np.zeros_like(impulse)
    for i in range(len(impulse)):
        s = impulse[i]
        for ap in chain:
            s = ap.process(s)
        output[i] = s
    save_wav("15_impulse_8stage_diffusion.wav", output)


if __name__ == "__main__":
    print(f"Sample rate: {SR} Hz")
    print(f"Output dir: {OUTPUT_DIR}\n")
    test_single_allpass()
    test_allpass_chain()
    test_impulse_diffusion()
    print("\nDone! The 8-stage impulse diffusion is the key one — a click becomes a wash.")
