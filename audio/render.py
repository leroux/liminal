"""Offline WAV rendering — load audio, process through FDN, save output.

Usage:
    uv run python audio/render.py input.wav output.wav [--preset presets/room.json]

Without --preset, uses default params.
"""

import numpy as np
from scipy.io import wavfile
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.fdn import render_fdn
from engine.params import default_params, SR


def load_wav(path):
    """Load a WAV file, convert to mono float64, resample to SR if needed."""
    sr, data = wavfile.read(path)

    # Convert to float64
    if data.dtype == np.int16:
        audio = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        audio = data.astype(np.float64) / 2147483648.0
    elif data.dtype == np.float32 or data.dtype == np.float64:
        audio = data.astype(np.float64)
    else:
        audio = data.astype(np.float64)

    # Stereo to mono
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if sr != SR:
        print(f"  Warning: input sample rate is {sr} Hz, expected {SR}. No resampling applied.")

    return audio, sr


def save_wav(path, audio, sr=SR):
    """Save float64 audio as 16-bit WAV, with peak normalization.

    Handles both mono (n_samples,) and stereo (n_samples, 2) arrays.
    """
    peak = np.max(np.abs(audio))
    if peak > 1.0:
        print(f"  Peak {peak:.2f} exceeds 1.0, normalizing.")
        audio = audio / peak * 0.95
    elif peak > 0 and peak < 0.1:
        print(f"  Peak {peak:.4f} is very low, normalizing.")
        audio = audio / peak * 0.9
    wavfile.write(path, sr, (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16))


def load_preset(path):
    """Load a params dict from JSON."""
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Process audio through FDN reverb")
    parser.add_argument("input", help="Input WAV file")
    parser.add_argument("output", help="Output WAV file")
    parser.add_argument("--preset", help="JSON preset file (default params if omitted)")
    parser.add_argument("--wet", type=float, help="Override wet/dry mix (0.0-1.0)")
    parser.add_argument("--feedback", type=float, help="Override feedback gain")
    parser.add_argument("--tail", type=float, default=2.0,
                        help="Extra seconds of tail after input ends (default: 2.0)")
    args = parser.parse_args()

    # Load params
    if args.preset:
        params = load_preset(args.preset)
        print(f"Loaded preset: {args.preset}")
    else:
        params = default_params()
        print("Using default params")

    if args.wet is not None:
        params["wet_dry"] = args.wet
    if args.feedback is not None:
        params["feedback_gain"] = args.feedback

    # Load input
    print(f"Loading: {args.input}")
    audio, sr = load_wav(args.input)
    print(f"  {len(audio)} samples, {len(audio)/sr:.2f}s, {sr} Hz")

    # Add tail (silence after input so reverb can decay)
    tail_samples = int(args.tail * SR)
    audio_padded = np.concatenate([audio, np.zeros(tail_samples)])
    print(f"  Added {args.tail}s tail -> {len(audio_padded)/SR:.2f}s total")

    # Render
    print("Rendering...")
    t0 = time.time()
    output = render_fdn(audio_padded, params)
    elapsed = time.time() - t0
    realtime = len(audio_padded) / SR
    ch_str = "stereo" if output.ndim == 2 else "mono"
    print(f"  {realtime:.1f}s audio in {elapsed:.2f}s ({realtime/elapsed:.1f}x realtime) [{ch_str}]")

    # Save
    save_wav(args.output, output, SR)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
