"""Offline WAV rendering for the Lossy plugin.

Usage:
    python audio/render.py input.wav output.wav [--preset preset.json] [--loss 0.7] [--window_size 2048]
"""

import argparse
import json
import os
import numpy as np
from scipy.io import wavfile

from engine.params import SR, default_params, migrate_legacy_params
from engine.lossy import render_lossy


def load_wav(path):
    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        audio = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        audio = data.astype(np.float64) / 2147483648.0
    else:
        audio = data.astype(np.float64)
    # Resample to engine SR if needed (engine assumes SR throughout)
    if sr != SR:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(SR, sr)
        up, down = SR // g, sr // g
        if audio.ndim == 2:
            channels = []
            for ch in range(audio.shape[1]):
                channels.append(resample_poly(audio[:, ch], up, down))
            audio = np.column_stack(channels)
        else:
            audio = resample_poly(audio, up, down)
        sr = SR
    # Keep stereo as (samples, 2) — don't downmix
    return audio, sr


def save_wav(path, audio, sr=SR):
    peak = np.max(np.abs(audio))
    if peak > 1.0:
        audio = audio / peak * 0.95
    elif 0 < peak < 0.1:
        audio = audio / peak * 0.9
    out = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    wavfile.write(path, sr, out)


def load_preset(path):
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Lossy offline renderer")
    parser.add_argument("input", help="Input WAV file")
    parser.add_argument("output", help="Output WAV file")
    parser.add_argument("--preset", help="Preset JSON file")
    parser.add_argument("--loss", type=float)
    parser.add_argument("--window_size", type=int)
    parser.add_argument("--mode", type=int, choices=[0, 1, 2])
    parser.add_argument("--verb", type=float)
    parser.add_argument("--wet", type=float)
    args = parser.parse_args()

    params = default_params()
    if args.preset:
        preset = load_preset(args.preset)
        migrate_legacy_params(preset)
        params.update(preset)

    if args.loss is not None:
        params["loss"] = args.loss
    if args.window_size is not None:
        params["window_size"] = args.window_size
    if args.mode is not None:
        params["mode"] = args.mode
    if args.verb is not None:
        params["verb"] = args.verb
    if args.wet is not None:
        params["wet_dry"] = args.wet

    audio, sr = load_wav(args.input)
    n = audio.shape[0] if audio.ndim == 2 else len(audio)
    ch = "stereo" if audio.ndim == 2 else "mono"
    print(f"Loaded {args.input}: {n} samples, {sr} Hz, {ch}")

    output = render_lossy(audio, params)
    save_wav(args.output, output, SR)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
