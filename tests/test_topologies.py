"""Compare feedback matrix topologies — generates WAV files.

Run: uv run python tests/test_topologies.py
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


def make_noise_burst(total_seconds=4.0):
    signal = np.zeros(int(SR * total_seconds))
    burst_len = int(SR * 15 / 1000)
    burst = np.random.randn(burst_len)
    burst *= np.linspace(1.0, 0.0, burst_len)
    signal[:burst_len] = burst
    return signal


def make_chords(total_seconds=6.0):
    def pluck(freq, duration):
        t = np.arange(int(SR * duration)) / SR
        sig = np.sin(2*np.pi*freq*t) + 0.5*np.sin(2*np.pi*freq*2*t)
        return sig * np.exp(-t * 3.0)

    total = int(SR * total_seconds)
    audio = np.zeros(total)
    chords = [(0, [261.6, 329.6, 392.0]), (1.5, [220.0, 261.6, 329.6]),
              (3.0, [174.6, 220.0, 261.6]), (4.5, [196.0, 246.9, 293.7])]
    for t_off, freqs in chords:
        offset = int(t_off * SR)
        for f in freqs:
            p = pluck(f, 1.8)
            end = min(offset + len(p), total)
            audio[offset:end] += p[:end-offset] * 0.25
    return audio


topologies = [
    ("householder",       "Uniform coupling — smooth, standard FDN"),
    ("hadamard",          "Structured +/- coupling — similar to Householder but different decay"),
    ("diagonal",          "No coupling — isolated comb filters, metallic"),
    ("random_orthogonal", "Random unitary — asymmetric coupling, unique character"),
    ("circulant",         "Ring topology — energy circulates in one direction"),
    ("stautner_puckette", "Paired rotations — classic 1982 stereo reverb topology"),
]


if __name__ == "__main__":
    print(f"Sample rate: {SR} Hz")
    print(f"Output dir: {OUTPUT_DIR}\n")

    burst = make_noise_burst(total_seconds=5.0)
    chords = make_chords(total_seconds=8.0)

    for name, desc in topologies:
        print(f"\n--- {name}: {desc} ---")

        params = default_params()
        params["matrix_type"] = name
        params["wet_dry"] = 1.0  # 100% wet for clearer comparison

        t0 = time.time()
        out_burst = render_fdn(burst, params)
        elapsed_b = time.time() - t0

        params["wet_dry"] = 0.4
        t0 = time.time()
        out_chords = render_fdn(chords, params)
        elapsed_c = time.time() - t0

        save_wav(f"50_topology_{name}_burst.wav", out_burst)
        save_wav(f"50_topology_{name}_chords.wav", out_chords)
        print(f"  Render time: burst {elapsed_b:.2f}s, chords {elapsed_c:.2f}s")

    print("\n\nDone! Compare the topologies — listen especially for:")
    print("  - diagonal vs householder (metallic vs smooth)")
    print("  - circulant (energy swirls around)")
    print("  - stautner_puckette (paired stereo character)")
    print("  - random_orthogonal (unique asymmetric decay)")
