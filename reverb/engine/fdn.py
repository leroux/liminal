"""8-node Feedback Delay Network — the core reverb engine.

Signal flow:
    Input -> [Pre-delay] -> [Input Diffusion] -> FDN Loop -> [Wet/Dry Mix] -> Output

FDN Loop (per sample):
    1. Read from 8 delay lines
    2. Apply per-node one-pole damping
    3. Multiply by feedback matrix
    4. Scale by feedback gain + saturate (tanh)
    5. Add input (distributed via input gains)
    6. Write back to delay lines
    7. Sum weighted outputs (output gains)
"""

import json
import logging
import time

import numpy as np

from reverb_rust import render_fdn as _rust_render
from reverb_rust import render_fdn_stereo as _rust_render_stereo

log = logging.getLogger(__name__)

N = 8  # number of delay lines


def _render_mono(input_audio, params, chunk_callback=None, chunk_size=4096):
    """Render a mono channel through the Rust FDN engine."""
    params_json = json.dumps(params)
    interleaved = _rust_render(input_audio.astype(np.float64), params_json)
    # Rust returns interleaved stereo [L0, R0, L1, R1, ...] -> reshape to (n, 2)
    result = interleaved.reshape(-1, 2)

    if chunk_callback is not None:
        for start in range(0, len(result), chunk_size):
            end = min(start + chunk_size, len(result))
            if not chunk_callback(result[start:end]):
                break

    return result


def render_fdn(input_audio: np.ndarray, params: dict,
               chunk_callback=None, chunk_size=4096) -> np.ndarray:
    """The single entry point. GUI sliders, ML optimizers, and batch
    rendering all call this same function.

    Args:
        input_audio: float64 array -- mono (samples,) or stereo (samples, 2)
        params: parameter dict (see engine/params.py)
        chunk_callback: if provided, called with each rendered chunk (np array).
            Return True to continue, False to stop early.
        chunk_size: samples per chunk when streaming (default 4096 ~ 93ms)

    Returns:
        stereo output (samples, 2)
    """
    from reverb.engine.params import SR
    t0 = time.perf_counter()
    n_samples = input_audio.shape[0]
    duration = n_samples / SR
    stereo_in = input_audio.ndim == 2

    if stereo_in:
        if chunk_callback is not None:
            mono = input_audio.mean(axis=1)
            result = _render_mono(mono, params, chunk_callback, chunk_size)
        else:
            params_json = json.dumps(dict(params, wet_dry=1.0))
            left = input_audio[:, 0].astype(np.float64)
            right = (input_audio[:, 1] if input_audio.shape[1] > 1
                     else input_audio[:, 0]).astype(np.float64)
            out_l, out_r = _rust_render_stereo(left, right, params_json)

            mix = float(params.get("wet_dry", 1.0))
            dry_l = input_audio[:, 0]
            dry_r = input_audio[:, 1] if input_audio.shape[1] > 1 else input_audio[:, 0]
            result = np.column_stack([
                dry_l * (1.0 - mix) + out_l * mix,
                dry_r * (1.0 - mix) + out_r * mix,
            ])
    else:
        result = _render_mono(input_audio, params, chunk_callback, chunk_size)

    elapsed = time.perf_counter() - t0
    rtf = duration / elapsed if elapsed > 0 else float('inf')
    log.info("render %.1fs audio in %.3fs (rust, %s, %.0fx RT)",
             duration, elapsed, "stereo" if stereo_in else "mono", rtf)
    return result
