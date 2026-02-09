"""Q-series: Meta-effects that chain other effects (Q001-Q005)."""
import numpy as np
from registry import discover_effects


def _get_effect_fn(effect_id):
    """Look up an effect function by its ID string."""
    registry = discover_effects()
    entry = registry.get(effect_id)
    if entry is None:
        raise ValueError(f"Unknown effect ID: {effect_id}")
    return entry[0]  # (effect_fn, variants_fn) -> effect_fn


# ---------------------------------------------------------------------------
# Q001 -- Serial Chain (2 effects)
# ---------------------------------------------------------------------------

def effect_q001_serial_chain_2(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Run two effects in series: output_a -> input_b."""
    effect_a = params.get('effect_a', 'd002_soft_clipping_tanh')
    params_a = params.get('params_a', {})
    effect_b = params.get('effect_b', 'a001_simple_delay')
    params_b = params.get('params_b', {})

    fn_a = _get_effect_fn(effect_a)
    fn_b = _get_effect_fn(effect_b)

    out = fn_a(samples.astype(np.float32), sr, **params_a)
    # Ensure mono float32 going into second stage
    if out.ndim > 1:
        out = out[:, 0] if out.shape[1] >= 1 else out.flatten()
    out = fn_b(out.astype(np.float32), sr, **params_b)
    return out


def variants_q001():
    return [
        # Distortion into delay -- classic guitar pedal order
        {'effect_a': 'd002_soft_clipping_tanh', 'params_a': {'drive': 5.0},
         'effect_b': 'a001_simple_delay', 'params_b': {'delay_ms': 300, 'feedback': 0.5}},
        # Delay into distortion -- each echo gets more distorted
        {'effect_a': 'a001_simple_delay', 'params_a': {'delay_ms': 200, 'feedback': 0.6},
         'effect_b': 'd002_soft_clipping_tanh', 'params_b': {'drive': 8.0}},
        # Bit crusher into spectral blur
        {'effect_a': 'd008_bit_crusher', 'params_a': {'bits': 6},
         'effect_b': 'h002_spectral_blur', 'params_b': {'blur_width': 20}},
        # Phase randomization into tube saturation
        {'effect_a': 'h005_phase_randomization', 'params_a': {'amount': 0.5},
         'effect_b': 'd003_tube_saturation', 'params_b': {'drive': 4.0, 'asymmetry': 0.2}},
        # Reverse chunks into allpass diffuser
        {'effect_a': 'a012_reverse_chunks', 'params_a': {'chunk_ms': 100, 'reverse_probability': 0.7},
         'effect_b': 'a007_allpass_diffuser', 'params_b': {'num_stages': 8, 'delay_range_ms': 20, 'g': 0.6}},
    ]


# ---------------------------------------------------------------------------
# Q002 -- Serial Chain (3 effects)
# ---------------------------------------------------------------------------

def effect_q002_serial_chain_3(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Run three effects in series: a -> b -> c."""
    effect_a = params.get('effect_a', 'd002_soft_clipping_tanh')
    params_a = params.get('params_a', {})
    effect_b = params.get('effect_b', 'a001_simple_delay')
    params_b = params.get('params_b', {})
    effect_c = params.get('effect_c', 'h005_phase_randomization')
    params_c = params.get('params_c', {})

    fn_a = _get_effect_fn(effect_a)
    fn_b = _get_effect_fn(effect_b)
    fn_c = _get_effect_fn(effect_c)

    out = fn_a(samples.astype(np.float32), sr, **params_a)
    if out.ndim > 1:
        out = out[:, 0] if out.shape[1] >= 1 else out.flatten()
    out = fn_b(out.astype(np.float32), sr, **params_b)
    if out.ndim > 1:
        out = out[:, 0] if out.shape[1] >= 1 else out.flatten()
    out = fn_c(out.astype(np.float32), sr, **params_c)
    return out


def variants_q002():
    return [
        # Distortion -> delay -> spectral blur  (ambient distortion wash)
        {'effect_a': 'd002_soft_clipping_tanh', 'params_a': {'drive': 6.0},
         'effect_b': 'a001_simple_delay', 'params_b': {'delay_ms': 400, 'feedback': 0.6},
         'effect_c': 'h002_spectral_blur', 'params_c': {'blur_width': 15}},
        # Bit crush -> foldback -> sample rate reduce (lo-fi chain)
        {'effect_a': 'd008_bit_crusher', 'params_a': {'bits': 8},
         'effect_b': 'd004_foldback_distortion', 'params_b': {'threshold': 0.5, 'pre_gain': 3.0},
         'effect_c': 'd009_sample_rate_reduction', 'params_c': {'target_sr': 8000}},
        # Stutter -> reverse chunks -> spectral freeze
        {'effect_a': 'a010_stutter', 'params_a': {'window_ms': 80, 'repeats': 4, 'decay': 0.9},
         'effect_b': 'a012_reverse_chunks', 'params_b': {'chunk_ms': 150, 'reverse_probability': 0.6},
         'effect_c': 'h001_spectral_freeze', 'params_c': {'freeze_position': 0.5}},
        # Allpass diffuser -> tape delay -> tube saturation (warm ambient)
        {'effect_a': 'a007_allpass_diffuser', 'params_a': {'num_stages': 6, 'delay_range_ms': 15, 'g': 0.55},
         'effect_b': 'a005_tape_delay', 'params_b': {'delay_ms': 300, 'feedback': 0.5, 'wow_rate_hz': 1.0},
         'effect_c': 'd003_tube_saturation', 'params_c': {'drive': 2.0, 'asymmetry': 0.1}},
        # Phase randomize -> spectral shift -> slew rate limit (alien textures)
        {'effect_a': 'h005_phase_randomization', 'params_a': {'amount': 0.7},
         'effect_b': 'h004_spectral_shift', 'params_b': {'shift_bins': 15},
         'effect_c': 'd010_slew_rate_limiter', 'params_c': {'max_slew': 0.02}},
    ]


# ---------------------------------------------------------------------------
# Q003 -- Parallel Mix (2 effects)
# ---------------------------------------------------------------------------

def effect_q003_parallel_mix_2(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Run two effects in parallel and mix their outputs."""
    effect_a = params.get('effect_a', 'd002_soft_clipping_tanh')
    params_a = params.get('params_a', {})
    effect_b = params.get('effect_b', 'a001_simple_delay')
    params_b = params.get('params_b', {})
    mix_a = np.float32(params.get('mix_a', 0.5))
    mix_b = np.float32(params.get('mix_b', 0.5))

    fn_a = _get_effect_fn(effect_a)
    fn_b = _get_effect_fn(effect_b)

    out_a = fn_a(samples.astype(np.float32), sr, **params_a)
    out_b = fn_b(samples.astype(np.float32), sr, **params_b)

    # Handle potentially different shapes (mono vs stereo)
    if out_a.ndim > 1:
        out_a = out_a[:, 0] if out_a.shape[1] >= 1 else out_a.flatten()
    if out_b.ndim > 1:
        out_b = out_b[:, 0] if out_b.shape[1] >= 1 else out_b.flatten()

    # Match lengths
    min_len = min(len(out_a), len(out_b))
    out_a = out_a[:min_len]
    out_b = out_b[:min_len]

    return (mix_a * out_a + mix_b * out_b).astype(np.float32)


def variants_q003():
    return [
        # Distortion + clean delay blended equally
        {'effect_a': 'd002_soft_clipping_tanh', 'params_a': {'drive': 5.0},
         'effect_b': 'a001_simple_delay', 'params_b': {'delay_ms': 250, 'feedback': 0.5},
         'mix_a': 0.5, 'mix_b': 0.5},
        # Heavy distortion blended with spectral blur
        {'effect_a': 'd004_foldback_distortion', 'params_a': {'threshold': 0.3, 'pre_gain': 10.0},
         'effect_b': 'h002_spectral_blur', 'params_b': {'blur_width': 30},
         'mix_a': 0.3, 'mix_b': 0.7},
        # Phase randomization + spectral freeze layered
        {'effect_a': 'h005_phase_randomization', 'params_a': {'amount': 0.8},
         'effect_b': 'h001_spectral_freeze', 'params_b': {'freeze_position': 0.4},
         'mix_a': 0.6, 'mix_b': 0.4},
        # Two delays at different times for rhythmic pattern
        {'effect_a': 'a001_simple_delay', 'params_a': {'delay_ms': 150, 'feedback': 0.4},
         'effect_b': 'a001_simple_delay', 'params_b': {'delay_ms': 375, 'feedback': 0.5},
         'mix_a': 0.5, 'mix_b': 0.5},
        # Bit crusher + tube saturation for textural layering
        {'effect_a': 'd008_bit_crusher', 'params_a': {'bits': 4},
         'effect_b': 'd003_tube_saturation', 'params_b': {'drive': 6.0, 'asymmetry': 0.3},
         'mix_a': 0.4, 'mix_b': 0.6},
    ]


# ---------------------------------------------------------------------------
# Q004 -- Wet/Dry Crossfade Over Time
# ---------------------------------------------------------------------------

def effect_q004_wet_dry_crossfade(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Sweep from dry to wet (or wet to dry) over the duration of the signal."""
    effect_id = params.get('effect_id', 'h005_phase_randomization')
    effect_params = params.get('effect_params', {})
    direction = params.get('direction', 'dry_to_wet')

    fn = _get_effect_fn(effect_id)
    wet = fn(samples.astype(np.float32), sr, **effect_params)

    # Handle potentially different shapes
    if wet.ndim > 1:
        wet = wet[:, 0] if wet.shape[1] >= 1 else wet.flatten()

    dry = samples.astype(np.float32)
    min_len = min(len(dry), len(wet))
    dry = dry[:min_len]
    wet = wet[:min_len]

    # Linear crossfade envelope
    t = np.linspace(0.0, 1.0, min_len, dtype=np.float32)
    if direction == 'dry_to_wet':
        wet_amount = t
    else:  # wet_to_dry
        wet_amount = 1.0 - t

    out = (1.0 - wet_amount) * dry + wet_amount * wet
    return out.astype(np.float32)


def variants_q004():
    return [
        # Fade into phase randomization
        {'effect_id': 'h005_phase_randomization', 'effect_params': {'amount': 1.0},
         'direction': 'dry_to_wet'},
        # Fade out of spectral freeze
        {'effect_id': 'h001_spectral_freeze', 'effect_params': {'freeze_position': 0.3},
         'direction': 'wet_to_dry'},
        # Gradually introduce distortion
        {'effect_id': 'd002_soft_clipping_tanh', 'effect_params': {'drive': 10.0},
         'direction': 'dry_to_wet'},
        # Fade out of bit crusher
        {'effect_id': 'd008_bit_crusher', 'effect_params': {'bits': 4},
         'direction': 'wet_to_dry'},
        # Fade into spectral blur
        {'effect_id': 'h002_spectral_blur', 'effect_params': {'blur_width': 40},
         'direction': 'dry_to_wet'},
    ]


# ---------------------------------------------------------------------------
# Q005 -- Feedback Through Effect
# ---------------------------------------------------------------------------

def effect_q005_feedback_through_effect(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """y = x + feedback * effect(y_prev_block). Process in blocks."""
    effect_id = params.get('effect_id', 'd002_soft_clipping_tanh')
    effect_params = params.get('effect_params', {})
    block_size_ms = np.float32(params.get('block_size_ms', 50))
    feedback = np.float32(params.get('feedback', 0.4))

    fn = _get_effect_fn(effect_id)

    block_size_ms = np.clip(block_size_ms, 10.0, 100.0)
    feedback = np.clip(feedback, 0.1, 0.8)

    block_size = max(1, int(block_size_ms * sr / 1000.0))
    n = len(samples)
    out = np.zeros(n, dtype=np.float32)
    prev_block = np.zeros(block_size, dtype=np.float32)

    num_blocks = (n + block_size - 1) // block_size

    for b in range(num_blocks):
        start = b * block_size
        end = min(start + block_size, n)
        current_len = end - start

        # Process previous block through the effect
        effect_out = fn(prev_block.astype(np.float32), sr, **effect_params)
        # Handle shape
        if effect_out.ndim > 1:
            effect_out = effect_out[:, 0] if effect_out.shape[1] >= 1 else effect_out.flatten()
        # Trim/pad to block_size
        if len(effect_out) < block_size:
            padded = np.zeros(block_size, dtype=np.float32)
            padded[:len(effect_out)] = effect_out
            effect_out = padded
        else:
            effect_out = effect_out[:block_size]

        # y = x + feedback * effect(y_prev_block)
        for i in range(current_len):
            out[start + i] = samples[start + i] + feedback * effect_out[i]

        # Store current output block as prev_block for next iteration
        prev_block = np.zeros(block_size, dtype=np.float32)
        for i in range(current_len):
            prev_block[i] = out[start + i]

    return out


def variants_q005():
    return [
        # Feedback through soft clip -- self-limiting feedback distortion
        {'effect_id': 'd002_soft_clipping_tanh', 'effect_params': {'drive': 3.0},
         'block_size_ms': 50, 'feedback': 0.4},
        # Feedback through bit crusher -- increasingly degraded echoes
        {'effect_id': 'd008_bit_crusher', 'effect_params': {'bits': 6},
         'block_size_ms': 80, 'feedback': 0.5},
        # Feedback through slew limiter -- smoothing recirculation
        {'effect_id': 'd010_slew_rate_limiter', 'effect_params': {'max_slew': 0.05},
         'block_size_ms': 30, 'feedback': 0.6},
        # Feedback through foldback -- chaotic harmonics buildup
        {'effect_id': 'd004_foldback_distortion', 'effect_params': {'threshold': 0.5, 'pre_gain': 3.0},
         'block_size_ms': 40, 'feedback': 0.3},
        # Higher feedback for longer sustain
        {'effect_id': 'd002_soft_clipping_tanh', 'effect_params': {'drive': 5.0},
         'block_size_ms': 100, 'feedback': 0.7},
    ]
