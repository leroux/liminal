"""Main render entry point for the Lossy plugin.

Signal chain (verb_position=0, PRE — default):
    Input -> Verb -> Spectral Loss -> Auto Gain -> Loss Gain -> Crush/Decimate
          -> Packets -> Filter -> Gate -> Limiter -> Mix -> Output

Signal chain (verb_position=1, POST):
    Input -> Spectral Loss -> Auto Gain -> Loss Gain -> Crush/Decimate
          -> Packets -> Filter -> Verb -> Gate -> Limiter -> Mix -> Output

Bounce wraps the entire chain, modulating one parameter via LFO.

All callers -- GUI, CLI, presets -- use render_lossy().
"""

import numpy as np
from engine.spectral import spectral_process
from engine.bitcrush import crush_and_decimate
from engine.packets import packet_process
from engine.filters import apply_filter, lofi_reverb, noise_gate, limiter
from engine.params import BOUNCE_TARGETS, PARAM_RANGES


def _render_chain(dry, params):
    """Core signal chain without bounce modulation."""
    verb_pos = int(params.get("verb_position", 0))

    # PRE verb: reverb runs on dry signal before spectral processing
    if verb_pos == 0:
        wet = lofi_reverb(dry, params)
    else:
        wet = dry.copy()

    # Measure input RMS before spectral loss (for auto_gain)
    rms_before = np.sqrt(np.mean(wet ** 2)) if float(params.get("auto_gain", 0.0)) > 0 else 0.0

    # 1) Spectral loss (STFT-based)
    wet = spectral_process(wet, params)

    # 2) Auto gain compensation
    auto_gain = float(params.get("auto_gain", 0.0))
    if auto_gain > 0 and rms_before > 1e-8:
        rms_after = np.sqrt(np.mean(wet ** 2))
        if rms_after > 1e-8:
            ratio = rms_before / rms_after
            # Blend between no compensation (1.0) and full compensation (ratio)
            gain = 1.0 + auto_gain * (ratio - 1.0)
            # Cap to prevent extreme boosts
            gain = min(gain, 10.0)
            wet = wet * gain

    # 3) Loss gain (wet signal volume: 0→-36dB, 0.5→0dB, 1→+36dB)
    loss_gain_param = float(params.get("loss_gain", 0.5))
    if loss_gain_param != 0.5:
        db = (loss_gain_param - 0.5) * 72.0  # -36 to +36 dB
        wet = wet * (10.0 ** (db / 20.0))

    # 4) Bitcrusher + sample rate reducer (time-domain)
    wet = crush_and_decimate(wet, params)

    # 5) Packet loss / repeat
    wet = packet_process(wet, params)

    # 6) Biquad filter (bandpass / notch)
    wet = apply_filter(wet, params)

    # POST verb: reverb runs after filter
    if verb_pos == 1:
        wet = lofi_reverb(wet, params)

    # 7) Noise gate
    wet = noise_gate(wet, params)

    # 8) Limiter (threshold-aware)
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


def render_lossy(input_audio, params):
    """The single entry point.  GUI sliders, CLI, and batch rendering all
    call this same function.

    Args:
        input_audio: float64 array — mono (samples,) or stereo (samples, 2)
        params: parameter dict (see engine/params.py)

    Returns:
        processed float64 array, same shape as input
    """
    dry = input_audio.astype(np.float64)

    if dry.ndim == 2:
        channels = []
        for ch in range(dry.shape[1]):
            channels.append(_render_mono(dry[:, ch], params))
        return np.column_stack(channels)

    return _render_mono(dry, params)


def _render_with_bounce(dry, params):
    """Block-based render with LFO modulation of a target parameter."""
    bounce_target_idx = int(params.get("bounce_target", 0))
    bounce_rate_param = float(params.get("bounce_rate", 0.3))

    # Map bounce_rate 0-1 to Hz 0.1-5.0
    lfo_hz = 0.1 + 4.9 * bounce_rate_param

    if bounce_target_idx >= len(BOUNCE_TARGETS):
        bounce_target_idx = 0
    target_key = BOUNCE_TARGETS[bounce_target_idx]
    base_value = float(params.get(target_key, 0.5))

    # Get parameter range for the target
    lo, hi = PARAM_RANGES.get(target_key, (0.0, 1.0))

    # Block size ~50ms
    from engine.params import SR
    block_samples = int(SR * 0.05)
    n = len(dry)
    wet = np.zeros(n, dtype=np.float64)

    for start in range(0, n, block_samples):
        end = min(start + block_samples, n)
        block_mid = (start + end) / 2.0

        # Compute LFO value at block midpoint (sine, 0 to 1 range)
        t = block_mid / SR
        lfo = 0.5 + 0.5 * np.sin(2.0 * np.pi * lfo_hz * t)

        # Modulate: sweep between lo and base_value
        mod_value = lo + lfo * (base_value - lo)
        mod_value = np.clip(mod_value, lo, hi)

        # Create modified params for this block
        block_params = dict(params)
        block_params[target_key] = mod_value
        block_params["bounce"] = 0  # prevent recursion

        block_wet = _render_chain(dry[start:end], block_params)
        wet[start:end] = block_wet

    return wet
