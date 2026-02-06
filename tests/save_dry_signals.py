"""Save the dry (unprocessed) test signals as WAVs so you can hear what goes into the FDN."""

import numpy as np
from scipy.io import wavfile
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.params import SR

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


# 1. Impulse — single sample click
impulse = np.zeros(int(SR * 2.0))
impulse[0] = 1.0
save_wav("dry_impulse.wav", impulse)

# 2. Noise burst — short clap-like transient
burst = np.zeros(int(SR * 2.0))
burst_len = int(SR * 15 / 1000)
b = np.random.randn(burst_len)
b *= np.linspace(1.0, 0.0, burst_len)
burst[:burst_len] = b
save_wav("dry_noise_burst.wav", burst)

# 3. Pluck — decaying sine (like a guitar string)
t = np.arange(int(SR * 3.0)) / SR
pluck = np.sin(2 * np.pi * 220.0 * t) * np.exp(-t * 4.0)
save_wav("dry_pluck.wav", pluck)
