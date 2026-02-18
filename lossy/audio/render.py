"""Offline WAV rendering for the Lossy plugin.

Usage:
    python -m lossy.audio.render input.wav output.wav [--preset preset.json] [--loss 0.7]
"""

import argparse
import json
import numpy as np

from lossy.engine.params import SR, default_params, migrate_legacy_params
from lossy.engine.lossy import render_lossy
from shared.audio import load_wav, save_wav


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
        with open(args.preset) as f:
            preset = json.load(f)
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

    audio, sr = load_wav(args.input, SR)
    n = audio.shape[0] if audio.ndim == 2 else len(audio)
    ch = "stereo" if audio.ndim == 2 else "mono"
    print(f"Loaded {args.input}: {n} samples, {sr} Hz, {ch}")

    output = render_lossy(audio, params)
    save_wav(args.output, output, SR)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
