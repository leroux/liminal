"""8-node Feedback Delay Network — the core reverb engine.

Signal flow:
    Input → [Pre-delay] → [Input Diffusion] → FDN Loop → [Wet/Dry Mix] → Output

FDN Loop (per sample):
    1. Read from 8 delay lines
    2. Apply per-node one-pole damping
    3. Multiply by feedback matrix
    4. Scale by feedback gain + saturate (tanh)
    5. Add input (distributed via input gains)
    6. Write back to delay lines
    7. Sum weighted outputs (output gains)
"""

import numpy as np


N = 8  # number of delay lines


def _render_mono(input_audio, params, chunk_callback=None, chunk_size=4096):
    """Route a mono channel to the appropriate engine."""
    has_mod = (
        params.get("mod_master_rate", 0.0) > 0.0 and (
            any(d > 0 for d in params.get("mod_depth_delay", [0] * 8)) or
            any(d > 0 for d in params.get("mod_depth_damping", [0] * 8)) or
            any(d > 0 for d in params.get("mod_depth_output", [0] * 8)) or
            params.get("mod_depth_matrix", 0.0) > 0.0
        )
    )

    if has_mod:
        from reverb.engine.numba_fdn_mod import render_fdn_mod
        return render_fdn_mod(input_audio, params, chunk_callback, chunk_size)
    else:
        from reverb.engine.numba_fdn import render_fdn_fast
        return render_fdn_fast(input_audio, params, chunk_callback, chunk_size)


def render_fdn(input_audio: np.ndarray, params: dict,
               chunk_callback=None, chunk_size=4096) -> np.ndarray:
    """The single entry point. GUI sliders, ML optimizers, and batch
    rendering all call this same function.

    Routes to the modulated engine when any modulation is active,
    otherwise uses the static (faster) path.

    Args:
        input_audio: float64 array — mono (samples,) or stereo (samples, 2)
        params: parameter dict (see engine/params.py)
        chunk_callback: if provided, called with each rendered chunk (np array).
            Return True to continue, False to stop early.
        chunk_size: samples per chunk when streaming (default 4096 ~ 93ms)

    Returns:
        stereo output (samples, 2)
    """
    if input_audio.ndim == 2:
        if chunk_callback is not None:
            # For streaming: mix stereo to mono so we can stream a single pass
            mono = input_audio.mean(axis=1)
            return _render_mono(mono, params, chunk_callback, chunk_size)

        # Full stereo processing (no streaming)
        wet_params = dict(params)
        wet_params["wet_dry"] = 1.0
        n = input_audio.shape[0]
        wet = np.zeros((n, 2), dtype=np.float64)
        for ch in range(input_audio.shape[1]):
            wet += _render_mono(input_audio[:, ch], wet_params)

        mix = float(params.get("wet_dry", 1.0))
        dry = np.column_stack([input_audio[:, 0],
                               input_audio[:, 1] if input_audio.shape[1] > 1
                               else input_audio[:, 0]])
        return dry * (1.0 - mix) + wet * mix

    return _render_mono(input_audio, params, chunk_callback, chunk_size)
