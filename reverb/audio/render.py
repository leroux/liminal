"""Offline WAV rendering for the FDN Reverb.

Usage:
    python -m reverb.audio.render input.wav output.wav [--preset preset.json]
"""

import argparse
import json
import numpy as np

from reverb.engine.params import SR, default_params
from reverb.engine.fdn import render_fdn
from shared.audio import load_wav, save_wav


def main():
    parser = argparse.ArgumentParser(description="Reverb offline renderer")
    parser.add_argument("input", help="Input WAV file")
    parser.add_argument("output", help="Output WAV file")
    parser.add_argument("--preset", help="Preset JSON file")
    parser.add_argument("--feedback_gain", type=float)
    parser.add_argument("--wet_dry", type=float)
    parser.add_argument("--diffusion", type=float)
    parser.add_argument("--saturation", type=float)
    parser.add_argument("--pre_delay_ms", type=float)
    parser.add_argument("--tail", type=float, default=2.0,
                        help="Tail length in seconds (default 2.0)")
    args = parser.parse_args()

    params = default_params()
    if args.preset:
        with open(args.preset) as f:
            preset = json.load(f)
        preset.pop("_meta", None)
        params.update(preset)

    if args.feedback_gain is not None:
        params["feedback_gain"] = args.feedback_gain
    if args.wet_dry is not None:
        params["wet_dry"] = args.wet_dry
    if args.diffusion is not None:
        params["diffusion"] = args.diffusion
    if args.saturation is not None:
        params["saturation"] = args.saturation
    if args.pre_delay_ms is not None:
        params["pre_delay"] = int(args.pre_delay_ms / 1000 * SR)

    audio, sr = load_wav(args.input, SR)
    n = audio.shape[0] if audio.ndim == 2 else len(audio)
    ch = "stereo" if audio.ndim == 2 else "mono"
    print(f"Loaded {args.input}: {n} samples, {sr} Hz, {ch}")

    # Append silence for reverb tail
    tail_samples = int(args.tail * SR)
    if audio.ndim == 2:
        audio = np.vstack([audio, np.zeros((tail_samples, audio.shape[1]))])
    else:
        audio = np.concatenate([audio, np.zeros(tail_samples)])

    output = render_fdn(audio, params)
    save_wav(args.output, output, SR)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
