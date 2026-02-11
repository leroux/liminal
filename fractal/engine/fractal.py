"""Main render entry point for the Fractal effect.

Signal chain:
    Input -> Pre-Filter -> Fractalize (with iterations) -> Crush/Decimate
          -> Post-Filter -> Gate -> Limiter -> Output Gain -> Mix -> Output

Bounce wraps the entire chain, modulating one parameter via LFO.

All callers -- GUI, CLI, presets -- use render_fractal().
"""

import numpy as np
from fractal.engine.core import render_fractal_core
from fractal.engine.filters import (
    apply_pre_filter, apply_post_filter,
    crush_and_decimate, noise_gate, limiter,
)
from fractal.engine.params import BOUNCE_TARGETS, PARAM_RANGES, SR


def _render_chain(dry, params):
    """Core signal chain without bounce modulation."""
    # 1) Pre-filter
    wet = apply_pre_filter(dry, params)

    # 2) Fractalize (core algorithm with iterations)
    wet = render_fractal_core(wet, params)

    # 3) Output gain (-36 to +36 dB)
    output_gain_param = float(params.get("output_gain", 0.5))
    if output_gain_param != 0.5:
        db = (output_gain_param - 0.5) * 72.0
        wet = wet * (10.0 ** (db / 20.0))

    # 4) Bitcrusher + sample rate reducer
    wet = crush_and_decimate(wet, params)

    # 5) Post-filter
    wet = apply_post_filter(wet, params)

    # 6) Noise gate
    wet = noise_gate(wet, params)

    # 7) Limiter
    wet = limiter(wet, params=params)

    return wet


def _render_mono(dry, params):
    """Render a single mono channel."""
    bounce_on = int(params.get("bounce", 0))
    if bounce_on:
        wet = _render_with_bounce(dry, params)
    else:
        wet = _render_chain(dry, params)
    mix = float(params.get("wet_dry", 1.0))
    return dry * (1.0 - mix) + wet * mix


def render_fractal(input_audio, params, chunk_callback=None, chunk_size=16384):
    """The single entry point.  GUI sliders, CLI, and batch rendering all
    call this same function.

    Args:
        input_audio: float64 array -- mono (samples,) or stereo (samples, 2)
        params: parameter dict (see engine/params.py)
        chunk_callback: if provided, called with each rendered chunk.
            Return True to continue, False to stop early.
        chunk_size: samples per chunk when streaming (default 16384 ~ 370ms)

    Returns:
        processed float64 array, same shape as input
    """
    dry = input_audio.astype(np.float64)

    if chunk_callback is None:
        if dry.ndim == 2:
            channels = []
            for ch in range(dry.shape[1]):
                channels.append(_render_mono(dry[:, ch], params))
            return np.column_stack(channels)
        return _render_mono(dry, params)

    # Streaming: process in chunks
    n = dry.shape[0]
    if dry.ndim == 2:
        output = np.zeros_like(dry)
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk_channels = []
            for ch in range(dry.shape[1]):
                chunk_channels.append(_render_mono(dry[start:end, ch], params))
            output[start:end] = np.column_stack(chunk_channels)
            if not chunk_callback(output[start:end]):
                break
    else:
        output = np.zeros(n, dtype=np.float64)
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            output[start:end] = _render_mono(dry[start:end], params)
            if not chunk_callback(output[start:end]):
                break

    return output


def _render_with_bounce(dry, params):
    """Block-based render with LFO modulation of a target parameter."""
    bounce_target_idx = int(params.get("bounce_target", 0))
    bounce_rate_param = float(params.get("bounce_rate", 0.3))

    lfo_min = float(params.get("bounce_lfo_min", 0.1))
    lfo_max = float(params.get("bounce_lfo_max", 5.0))
    lfo_hz = lfo_min + (lfo_max - lfo_min) * bounce_rate_param

    if bounce_target_idx >= len(BOUNCE_TARGETS):
        bounce_target_idx = 0
    target_key = BOUNCE_TARGETS[bounce_target_idx]
    base_value = float(params.get(target_key, 0.5))

    lo, hi = PARAM_RANGES.get(target_key, (0.0, 1.0))

    block_samples = int(SR * 0.05)
    n = len(dry)
    wet = np.zeros(n, dtype=np.float64)

    for start in range(0, n, block_samples):
        end = min(start + block_samples, n)
        block_mid = (start + end) / 2.0

        t = block_mid / SR
        lfo = 0.5 + 0.5 * np.sin(2.0 * np.pi * lfo_hz * t)

        mod_value = lo + lfo * (base_value - lo)
        mod_value = np.clip(mod_value, lo, hi)

        block_params = dict(params)
        block_params[target_key] = mod_value
        block_params["bounce"] = 0  # prevent recursion

        block_wet = _render_chain(dry[start:end], block_params)
        wet[start:end] = block_wet

    return wet
