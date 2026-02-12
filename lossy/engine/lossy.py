"""Main render entry point for the Lossy plugin.

Signal chain (verb_position=0, PRE — default):
    Input -> Verb -> Spectral Loss -> Auto Gain -> Loss Gain -> Crush/Decimate
          -> Packets -> Filter -> Gate -> Limiter -> Mix -> Output

Signal chain (verb_position=1, POST):
    Input -> Spectral Loss -> Auto Gain -> Loss Gain -> Crush/Decimate
          -> Packets -> Filter -> Verb -> Gate -> Limiter -> Mix -> Output

Bounce wraps the entire chain, modulating one parameter via LFO.

All callers -- GUI, CLI, presets -- use render_lossy().

DSP is implemented in Rust (lossy-dsp crate) and exposed via the
lossy_rust Python module (built with maturin from lossy-python crate).
"""

import json
import numpy as np
from lossy_rust import render_lossy as _rust_render
from lossy_rust import render_lossy_stereo as _rust_render_stereo


_GUI_ONLY_KEYS = {"_meta", "tail_length"}


def render_lossy(input_audio, params, chunk_callback=None, chunk_size=16384):
    """The single entry point.  GUI sliders, CLI, and batch rendering all
    call this same function.

    Args:
        input_audio: float64 array — mono (samples,) or stereo (samples, 2)
        params: parameter dict (see engine/params.py)
        chunk_callback: if provided, called with each rendered chunk.
            Return True to continue, False to stop early.
        chunk_size: samples per chunk when streaming (default 16384 ~ 370ms)

    Returns:
        processed float64 array, same shape as input
    """
    dry = input_audio.astype(np.float64)
    params_json = json.dumps({k: v for k, v in params.items() if k not in _GUI_ONLY_KEYS})

    if chunk_callback is None:
        return _render_rust(dry, params_json)

    # Streaming: process in chunks
    n = dry.shape[0]
    output = np.zeros_like(dry)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = dry[start:end]
        output[start:end] = _render_rust(chunk, params_json)
        if not chunk_callback(output[start:end]):
            break

    return output


def _render_rust(dry, params_json):
    """Dispatch to the Rust backend."""
    if dry.ndim == 2:
        left = np.ascontiguousarray(dry[:, 0])
        right = np.ascontiguousarray(dry[:, 1]) if dry.shape[1] > 1 else left.copy()
        out_l, out_r = _rust_render_stereo(left, right, params_json)
        return np.column_stack([out_l, out_r])
    return np.asarray(_rust_render(np.ascontiguousarray(dry), params_json))
