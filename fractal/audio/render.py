"""Offline WAV rendering for the Fractal effect.

Usage:
    python -m fractal.audio.render input.wav output.wav [--preset preset.json]
"""

import argparse
import json
import numpy as np
from scipy.io import wavfile

from fractal.engine.params import SR, default_params
from fractal.engine.fractal import render_fractal


def load_wav(path):
    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        audio = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        audio = data.astype(np.float64) / 2147483648.0
    else:
        audio = data.astype(np.float64)
    if sr != SR:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(SR, sr)
        audio = resample_poly(audio, SR // g, sr // g, axis=0)
    return audio, SR


def save_wav(path, audio, sr=SR):
    peak = np.max(np.abs(audio))
    if peak > 1.0:
        audio = audio / peak * 0.95
    elif 0 < peak < 0.1:
        audio = audio / peak * 0.9
    out = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    wavfile.write(path, sr, out)


def main():
    parser = argparse.ArgumentParser(description="Fractal offline renderer")
    parser.add_argument("input", help="Input WAV file")
    parser.add_argument("output", help="Output WAV file")
    parser.add_argument("--preset", help="Preset JSON file")
    parser.add_argument("--num_scales", type=int)
    parser.add_argument("--scale_ratio", type=float)
    parser.add_argument("--amplitude_decay", type=float)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--spectral", type=float)
    parser.add_argument("--wet", type=float)
    args = parser.parse_args()

    params = default_params()
    if args.preset:
        with open(args.preset) as f:
            preset = json.load(f)
        preset.pop("_meta", None)
        params.update(preset)

    if args.num_scales is not None:
        params["num_scales"] = args.num_scales
    if args.scale_ratio is not None:
        params["scale_ratio"] = args.scale_ratio
    if args.amplitude_decay is not None:
        params["amplitude_decay"] = args.amplitude_decay
    if args.iterations is not None:
        params["iterations"] = args.iterations
    if args.spectral is not None:
        params["spectral"] = args.spectral
    if args.wet is not None:
        params["wet_dry"] = args.wet

    audio, sr = load_wav(args.input)
    n = audio.shape[0] if audio.ndim == 2 else len(audio)
    ch = "stereo" if audio.ndim == 2 else "mono"
    print(f"Loaded {args.input}: {n} samples, {sr} Hz, {ch}")

    output = render_fractal(audio, params)
    save_wav(args.output, output, SR)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
