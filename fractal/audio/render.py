"""Offline WAV rendering for the Fractal effect.

Usage:
    python -m fractal.audio.render input.wav output.wav [--preset preset.json]
"""

import argparse
import json
import numpy as np

from fractal.engine.params import SR, default_params
from fractal.engine.fractal import render_fractal
from shared.audio import load_wav, save_wav


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

    audio, sr = load_wav(args.input, SR)
    n = audio.shape[0] if audio.ndim == 2 else len(audio)
    ch = "stereo" if audio.ndim == 2 else "mono"
    print(f"Loaded {args.input}: {n} samples, {sr} Hz, {ch}")

    output = render_fractal(audio, params)
    save_wav(args.output, output, SR)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
