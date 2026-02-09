"""I-series: Granular effects (I001-I007)."""
import numpy as np
import numba


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _hann_window(size: int) -> np.ndarray:
    """Precompute a Hann window of given size (float32)."""
    w = np.empty(size, dtype=np.float32)
    for i in range(size):
        w[i] = np.float32(0.5 * (1.0 - np.cos(2.0 * np.pi * i / max(size - 1, 1))))
    return w


@numba.njit
def _resample_grain(samples, start, grain_len, ratio, window, out, out_start, out_len, amplitude):
    """Read a grain from *samples* at *start* with pitch *ratio*, apply
    *window*, and overlap-add into *out* at *out_start*.

    ratio > 1 means higher pitch (reads source faster).
    Linear interpolation is used for fractional positions.
    """
    n_src = len(samples)
    for i in range(grain_len):
        if out_start + i < 0 or out_start + i >= out_len:
            continue
        # Source read position (fractional)
        src_pos = start + i * ratio
        idx = int(np.floor(src_pos))
        frac = np.float32(src_pos - idx)
        # Wrap / clamp source index
        if idx < 0:
            idx = 0
            frac = np.float32(0.0)
        if idx >= n_src - 1:
            idx = n_src - 2
            frac = np.float32(0.0)
            if idx < 0:
                continue
        val = samples[idx] * (1.0 - frac) + samples[idx + 1] * frac
        out[out_start + i] += val * window[i] * amplitude


@numba.njit
def _overlap_add_grains(samples, out, grain_starts, grain_positions,
                        grain_lens, grain_ratios, grain_amplitudes,
                        window, out_len):
    """Overlap-add all grains into *out*.

    * grain_starts  : output sample index where each grain begins
    * grain_positions: source sample index where each grain reads from
    * grain_lens    : length of each grain in samples
    * grain_ratios  : playback rate for each grain (1.0 = original pitch)
    * grain_amplitudes: per-grain amplitude
    * window        : Hann window (length == max grain_lens)
    """
    n_grains = len(grain_starts)
    n_src = len(samples)
    for g in range(n_grains):
        g_start = grain_starts[g]
        g_pos = grain_positions[g]
        g_len = grain_lens[g]
        g_ratio = grain_ratios[g]
        g_amp = grain_amplitudes[g]
        for i in range(g_len):
            oi = g_start + i
            if oi < 0 or oi >= out_len:
                continue
            src_pos = g_pos + np.float64(i) * g_ratio
            idx = int(np.floor(src_pos))
            frac = np.float32(src_pos - idx)
            if idx < 0:
                idx = 0
                frac = np.float32(0.0)
            if idx >= n_src - 1:
                idx = n_src - 2
                frac = np.float32(0.0)
                if idx < 0:
                    continue
            # Window index: scale i into window length
            w_idx = int(np.float64(i) / np.float64(g_len) * np.float64(len(window) - 1))
            if w_idx >= len(window):
                w_idx = len(window) - 1
            val = samples[idx] * (np.float32(1.0) - frac) + samples[idx + 1] * frac
            out[oi] += val * window[w_idx] * g_amp
    return out


def _schedule_grains_poisson(n_samples: int, sr: int, density: float,
                             rng: np.random.Generator | None = None) -> np.ndarray:
    """Return an array of grain onset sample-indices using Poisson scheduling.
    *density* is grains per second.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    avg_interval = sr / max(density, 0.01)
    # Generate inter-onset intervals (exponential distribution for Poisson process)
    times = []
    t = 0.0
    while t < n_samples:
        times.append(int(t))
        t += rng.exponential(avg_interval)
    return np.array(times, dtype=np.int64)


# ---------------------------------------------------------------------------
# I001 — Granular Cloud
# ---------------------------------------------------------------------------

def effect_i001_granular_cloud(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    grain_size_ms = float(params.get('grain_size_ms', 50))
    density = float(params.get('density', 20))
    position_spread = float(params.get('position_spread', 0.5))
    pitch_spread_st = float(params.get('pitch_spread_semitones', 0))
    amplitude_spread = float(params.get('amplitude_spread', 0.1))

    samples = samples.astype(np.float32)
    n = len(samples)
    grain_len = max(1, int(grain_size_ms * sr / 1000.0))
    window = _hann_window(grain_len)

    rng = np.random.default_rng(42)
    grain_starts = _schedule_grains_poisson(n, sr, density, rng)
    n_grains = len(grain_starts)
    if n_grains == 0:
        return np.zeros(n, dtype=np.float32)

    # Random source positions: center at grain_start, spread by position_spread
    half_spread = int(position_spread * n * 0.5)
    grain_positions = np.empty(n_grains, dtype=np.int64)
    for i in range(n_grains):
        offset = rng.integers(-half_spread, max(half_spread, 1))
        pos = grain_starts[i] + offset
        grain_positions[i] = max(0, min(pos, n - grain_len))

    # Pitch ratios
    if pitch_spread_st > 0:
        semitones = rng.uniform(-pitch_spread_st, pitch_spread_st, n_grains).astype(np.float64)
        grain_ratios = np.power(2.0, semitones / 12.0).astype(np.float64)
    else:
        grain_ratios = np.ones(n_grains, dtype=np.float64)

    # Amplitudes
    grain_amplitudes = (1.0 - amplitude_spread + rng.uniform(0, amplitude_spread, n_grains)).astype(np.float32)

    grain_lens = np.full(n_grains, grain_len, dtype=np.int64)

    out = np.zeros(n, dtype=np.float32)
    _overlap_add_grains(samples, out, grain_starts.astype(np.int64),
                        grain_positions.astype(np.int64), grain_lens,
                        grain_ratios, grain_amplitudes, window, n)
    return out


def variants_i001():
    return [
        {'grain_size_ms': 15, 'density': 80, 'position_spread': 0.3, 'pitch_spread_semitones': 0, 'amplitude_spread': 0.05},    # tight, high-density shimmer
        {'grain_size_ms': 50, 'density': 20, 'position_spread': 0.5, 'pitch_spread_semitones': 0, 'amplitude_spread': 0.1},      # default cloud
        {'grain_size_ms': 100, 'density': 10, 'position_spread': 0.8, 'pitch_spread_semitones': 0, 'amplitude_spread': 0.2},     # sparse, wide scatter
        {'grain_size_ms': 30, 'density': 40, 'position_spread': 0.2, 'pitch_spread_semitones': 5, 'amplitude_spread': 0.1},      # dense pitch-spread cloud
        {'grain_size_ms': 200, 'density': 5, 'position_spread': 1.0, 'pitch_spread_semitones': 12, 'amplitude_spread': 0.4},     # huge grains, full octave spread
        {'grain_size_ms': 10, 'density': 100, 'position_spread': 0.05, 'pitch_spread_semitones': 0, 'amplitude_spread': 0.0},    # near-transparent dense micro-grains
    ]


# ---------------------------------------------------------------------------
# I002 — Granular Freeze
# ---------------------------------------------------------------------------

def effect_i002_granular_freeze(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    freeze_position = float(params.get('freeze_position', 0.5))
    position_jitter_ms = float(params.get('position_jitter_ms', 10))
    pitch_jitter = float(params.get('pitch_jitter', 0.5))
    density = float(params.get('density', 30))
    grain_size_ms = float(params.get('grain_size_ms', 40))

    samples = samples.astype(np.float32)
    n = len(samples)
    grain_len = max(1, int(grain_size_ms * sr / 1000.0))
    window = _hann_window(grain_len)

    rng = np.random.default_rng(42)
    grain_starts = _schedule_grains_poisson(n, sr, density, rng)
    n_grains = len(grain_starts)
    if n_grains == 0:
        return np.zeros(n, dtype=np.float32)

    # Fixed source position with jitter
    center_pos = int(freeze_position * (n - grain_len))
    jitter_samples = int(position_jitter_ms * sr / 1000.0)
    grain_positions = np.empty(n_grains, dtype=np.int64)
    for i in range(n_grains):
        jit = rng.integers(-jitter_samples, max(jitter_samples, 1)) if jitter_samples > 0 else 0
        pos = center_pos + jit
        grain_positions[i] = max(0, min(pos, n - grain_len))

    # Pitch with jitter around 1.0
    if pitch_jitter > 0:
        semitones = rng.uniform(-pitch_jitter, pitch_jitter, n_grains).astype(np.float64)
        grain_ratios = np.power(2.0, semitones / 12.0).astype(np.float64)
    else:
        grain_ratios = np.ones(n_grains, dtype=np.float64)

    grain_amplitudes = np.ones(n_grains, dtype=np.float32)
    grain_lens = np.full(n_grains, grain_len, dtype=np.int64)

    out = np.zeros(n, dtype=np.float32)
    _overlap_add_grains(samples, out, grain_starts.astype(np.int64),
                        grain_positions.astype(np.int64), grain_lens,
                        grain_ratios, grain_amplitudes, window, n)
    return out


def variants_i002():
    return [
        {'freeze_position': 0.5, 'position_jitter_ms': 0, 'pitch_jitter': 0, 'density': 30, 'grain_size_ms': 40},       # perfectly frozen, no variation
        {'freeze_position': 0.5, 'position_jitter_ms': 10, 'pitch_jitter': 0.5, 'density': 30, 'grain_size_ms': 40},     # default freeze with jitter
        {'freeze_position': 0.25, 'position_jitter_ms': 30, 'pitch_jitter': 1.0, 'density': 50, 'grain_size_ms': 20},    # dense freeze near start, wide jitter
        {'freeze_position': 0.75, 'position_jitter_ms': 50, 'pitch_jitter': 2.0, 'density': 20, 'grain_size_ms': 80},    # slow, large grains, drunken pitch
        {'freeze_position': 0.1, 'position_jitter_ms': 5, 'pitch_jitter': 0, 'density': 100, 'grain_size_ms': 30},       # ultra-dense freeze, attack region
        {'freeze_position': 0.5, 'position_jitter_ms': 40, 'pitch_jitter': 1.5, 'density': 10, 'grain_size_ms': 100},    # sparse, dreamy freeze
    ]


# ---------------------------------------------------------------------------
# I003 — Granular Time Stretch
# ---------------------------------------------------------------------------

def effect_i003_granular_time_stretch(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    stretch_factor = float(params.get('stretch_factor', 4))
    grain_size_ms = float(params.get('grain_size_ms', 40))
    density = float(params.get('density', 30))

    samples = samples.astype(np.float32)
    n_src = len(samples)
    n_out = int(n_src * stretch_factor)
    grain_len = max(1, int(grain_size_ms * sr / 1000.0))
    window = _hann_window(grain_len)

    rng = np.random.default_rng(42)
    grain_starts = _schedule_grains_poisson(n_out, sr, density, rng)
    n_grains = len(grain_starts)
    if n_grains == 0:
        return np.zeros(n_out, dtype=np.float32)

    # Source read position moves slower: map output time to source time
    grain_positions = np.empty(n_grains, dtype=np.int64)
    for i in range(n_grains):
        # Map output position back to source position
        src_pos = int(grain_starts[i] / stretch_factor)
        grain_positions[i] = max(0, min(src_pos, n_src - grain_len))

    grain_ratios = np.ones(n_grains, dtype=np.float64)
    grain_amplitudes = np.ones(n_grains, dtype=np.float32)
    grain_lens = np.full(n_grains, grain_len, dtype=np.int64)

    out = np.zeros(n_out, dtype=np.float32)
    _overlap_add_grains(samples, out, grain_starts.astype(np.int64),
                        grain_positions.astype(np.int64), grain_lens,
                        grain_ratios, grain_amplitudes, window, n_out)
    return out


def variants_i003():
    return [
        {'stretch_factor': 1.5, 'grain_size_ms': 30, 'density': 40},     # slight stretch, minimal artifacts
        {'stretch_factor': 4, 'grain_size_ms': 40, 'density': 30},       # default moderate stretch
        {'stretch_factor': 10, 'grain_size_ms': 60, 'density': 25},      # extreme stretch, floating texture
        {'stretch_factor': 2, 'grain_size_ms': 20, 'density': 60},       # double length, dense small grains
        {'stretch_factor': 20, 'grain_size_ms': 80, 'density': 20},      # massive stretch, glacial
        {'stretch_factor': 50, 'grain_size_ms': 100, 'density': 15},     # frozen-in-time extreme stretch
    ]


# ---------------------------------------------------------------------------
# I004 — Granular Reverse Scatter
# ---------------------------------------------------------------------------

@numba.njit
def _overlap_add_grains_reversible(samples, out, grain_starts, grain_positions,
                                   grain_lens, grain_ratios, grain_amplitudes,
                                   grain_reversed, window, out_len):
    """Like _overlap_add_grains but with per-grain reverse flag."""
    n_grains = len(grain_starts)
    n_src = len(samples)
    w_len = len(window)
    for g in range(n_grains):
        g_start = grain_starts[g]
        g_pos = grain_positions[g]
        g_len = grain_lens[g]
        g_ratio = grain_ratios[g]
        g_amp = grain_amplitudes[g]
        is_rev = grain_reversed[g]
        for i in range(g_len):
            oi = g_start + i
            if oi < 0 or oi >= out_len:
                continue
            # If reversed, read source grain backwards
            if is_rev:
                src_pos = g_pos + np.float64(g_len - 1 - i) * g_ratio
            else:
                src_pos = g_pos + np.float64(i) * g_ratio
            idx = int(np.floor(src_pos))
            frac = np.float32(src_pos - idx)
            if idx < 0:
                idx = 0
                frac = np.float32(0.0)
            if idx >= n_src - 1:
                idx = n_src - 2
                frac = np.float32(0.0)
                if idx < 0:
                    continue
            w_idx = int(np.float64(i) / np.float64(g_len) * np.float64(w_len - 1))
            if w_idx >= w_len:
                w_idx = w_len - 1
            val = samples[idx] * (np.float32(1.0) - frac) + samples[idx + 1] * frac
            out[oi] += val * window[w_idx] * g_amp
    return out


def effect_i004_granular_reverse_scatter(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    reverse_probability = float(params.get('reverse_probability', 0.5))
    grain_size_ms = float(params.get('grain_size_ms', 40))
    density = float(params.get('density', 25))

    samples = samples.astype(np.float32)
    n = len(samples)
    grain_len = max(1, int(grain_size_ms * sr / 1000.0))
    window = _hann_window(grain_len)

    rng = np.random.default_rng(42)
    grain_starts = _schedule_grains_poisson(n, sr, density, rng)
    n_grains = len(grain_starts)
    if n_grains == 0:
        return np.zeros(n, dtype=np.float32)

    # Source positions follow output positions (like a pass-through with reversed grains)
    grain_positions = np.empty(n_grains, dtype=np.int64)
    for i in range(n_grains):
        pos = grain_starts[i]
        grain_positions[i] = max(0, min(pos, n - grain_len))

    grain_ratios = np.ones(n_grains, dtype=np.float64)
    grain_amplitudes = np.ones(n_grains, dtype=np.float32)
    grain_lens_arr = np.full(n_grains, grain_len, dtype=np.int64)

    # Decide which grains are reversed
    grain_reversed = (rng.random(n_grains) < reverse_probability).astype(np.int32)

    out = np.zeros(n, dtype=np.float32)
    _overlap_add_grains_reversible(samples, out, grain_starts.astype(np.int64),
                                   grain_positions.astype(np.int64), grain_lens_arr,
                                   grain_ratios, grain_amplitudes,
                                   grain_reversed, window, n)
    return out


def variants_i004():
    return [
        {'reverse_probability': 0.1, 'grain_size_ms': 40, 'density': 25},    # mostly forward, occasional reverse surprise
        {'reverse_probability': 0.5, 'grain_size_ms': 40, 'density': 25},    # default half-and-half chaos
        {'reverse_probability': 1.0, 'grain_size_ms': 40, 'density': 25},    # fully reversed granular
        {'reverse_probability': 0.5, 'grain_size_ms': 80, 'density': 15},    # large reversed chunks, sparse
        {'reverse_probability': 0.5, 'grain_size_ms': 20, 'density': 60},    # rapid small reverse flickers
        {'reverse_probability': 0.3, 'grain_size_ms': 100, 'density': 10},   # long grains, occasional dramatic reversal
    ]


# ---------------------------------------------------------------------------
# I005 — Granular Pitch Cloud
# ---------------------------------------------------------------------------

def effect_i005_granular_pitch_cloud(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    center_semitones = float(params.get('center_semitones', 0))
    spread_semitones = float(params.get('spread_semitones', 7))
    grain_size_ms = float(params.get('grain_size_ms', 50))
    density = float(params.get('density', 20))

    samples = samples.astype(np.float32)
    n = len(samples)
    grain_len = max(1, int(grain_size_ms * sr / 1000.0))
    window = _hann_window(grain_len)

    rng = np.random.default_rng(42)
    grain_starts = _schedule_grains_poisson(n, sr, density, rng)
    n_grains = len(grain_starts)
    if n_grains == 0:
        return np.zeros(n, dtype=np.float32)

    # All grains read from same position as their output position
    grain_positions = np.empty(n_grains, dtype=np.int64)
    for i in range(n_grains):
        grain_positions[i] = max(0, min(grain_starts[i], n - grain_len))

    # Random pitches around center
    semitones = center_semitones + rng.uniform(-spread_semitones, spread_semitones, n_grains)
    grain_ratios = np.power(2.0, semitones / 12.0).astype(np.float64)

    grain_amplitudes = np.ones(n_grains, dtype=np.float32)
    grain_lens_arr = np.full(n_grains, grain_len, dtype=np.int64)

    out = np.zeros(n, dtype=np.float32)
    _overlap_add_grains(samples, out, grain_starts.astype(np.int64),
                        grain_positions.astype(np.int64), grain_lens_arr,
                        grain_ratios, grain_amplitudes, window, n)
    return out


def variants_i005():
    return [
        {'center_semitones': 0, 'spread_semitones': 2, 'grain_size_ms': 50, 'density': 20},      # subtle detuned shimmer
        {'center_semitones': 0, 'spread_semitones': 7, 'grain_size_ms': 50, 'density': 20},      # default pitch cloud
        {'center_semitones': 0, 'spread_semitones': 12, 'grain_size_ms': 40, 'density': 30},     # full octave spread, dense
        {'center_semitones': 7, 'spread_semitones': 3, 'grain_size_ms': 60, 'density': 15},      # shifted up a fifth, tight cluster
        {'center_semitones': -12, 'spread_semitones': 5, 'grain_size_ms': 80, 'density': 10},    # octave down cloud, slow
        {'center_semitones': 0, 'spread_semitones': 24, 'grain_size_ms': 30, 'density': 40},     # two-octave chaos, dense
        {'center_semitones': 12, 'spread_semitones': 1, 'grain_size_ms': 20, 'density': 60},     # octave up, nearly unison, sparkly
    ]


# ---------------------------------------------------------------------------
# I006 — Granular Density Ramp
# ---------------------------------------------------------------------------

def _schedule_grains_density_ramp(n_samples: int, sr: int,
                                  start_density: float, end_density: float,
                                  ramp_curve: str = 'exponential',
                                  rng: np.random.Generator | None = None) -> np.ndarray:
    """Schedule grains with density that ramps from start to end over the duration."""
    if rng is None:
        rng = np.random.default_rng(42)
    times = []
    t = 0.0
    total = float(n_samples)
    while t < n_samples:
        times.append(int(t))
        # Compute local density at current position
        progress = t / total  # 0..1
        if ramp_curve == 'exponential':
            # Exponential interpolation in density space
            local_density = start_density * ((end_density / max(start_density, 0.01)) ** progress)
        else:
            # Linear interpolation
            local_density = start_density + (end_density - start_density) * progress
        local_density = max(local_density, 0.01)
        avg_interval = sr / local_density
        t += rng.exponential(avg_interval)
    return np.array(times, dtype=np.int64)


def effect_i006_granular_density_ramp(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    start_density = float(params.get('start_density', 2))
    end_density = float(params.get('end_density', 100))
    grain_size_ms = float(params.get('grain_size_ms', 30))
    ramp_curve = str(params.get('ramp_curve', 'exponential'))

    samples = samples.astype(np.float32)
    n = len(samples)
    grain_len = max(1, int(grain_size_ms * sr / 1000.0))
    window = _hann_window(grain_len)

    rng = np.random.default_rng(42)
    grain_starts = _schedule_grains_density_ramp(n, sr, start_density, end_density, ramp_curve, rng)
    n_grains = len(grain_starts)
    if n_grains == 0:
        return np.zeros(n, dtype=np.float32)

    grain_positions = np.empty(n_grains, dtype=np.int64)
    for i in range(n_grains):
        grain_positions[i] = max(0, min(grain_starts[i], n - grain_len))

    grain_ratios = np.ones(n_grains, dtype=np.float64)
    grain_amplitudes = np.ones(n_grains, dtype=np.float32)
    grain_lens_arr = np.full(n_grains, grain_len, dtype=np.int64)

    out = np.zeros(n, dtype=np.float32)
    _overlap_add_grains(samples, out, grain_starts.astype(np.int64),
                        grain_positions.astype(np.int64), grain_lens_arr,
                        grain_ratios, grain_amplitudes, window, n)
    return out


def variants_i006():
    return [
        {'start_density': 2, 'end_density': 50, 'grain_size_ms': 30, 'ramp_curve': 'linear'},          # gentle linear buildup
        {'start_density': 2, 'end_density': 100, 'grain_size_ms': 30, 'ramp_curve': 'exponential'},     # default exponential ramp
        {'start_density': 1, 'end_density': 200, 'grain_size_ms': 20, 'ramp_curve': 'exponential'},     # extreme buildup, tiny grains
        {'start_density': 10, 'end_density': 80, 'grain_size_ms': 50, 'ramp_curve': 'linear'},          # moderate range, larger grains
        {'start_density': 1, 'end_density': 150, 'grain_size_ms': 80, 'ramp_curve': 'exponential'},     # long grains, massive density ramp
        {'start_density': 5, 'end_density': 50, 'grain_size_ms': 40, 'ramp_curve': 'linear'},           # subtle linear growth
    ]


# ---------------------------------------------------------------------------
# I007 — Microsound Particles
# ---------------------------------------------------------------------------

def effect_i007_microsound_particles(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    grain_size_ms = float(params.get('grain_size_ms', 3))
    density = float(params.get('density', 300))
    pitch_range = float(params.get('pitch_range', 2))

    samples = samples.astype(np.float32)
    n = len(samples)
    grain_len = max(1, int(grain_size_ms * sr / 1000.0))
    window = _hann_window(grain_len)

    rng = np.random.default_rng(42)
    grain_starts = _schedule_grains_poisson(n, sr, density, rng)
    n_grains = len(grain_starts)
    if n_grains == 0:
        return np.zeros(n, dtype=np.float32)

    grain_positions = np.empty(n_grains, dtype=np.int64)
    for i in range(n_grains):
        grain_positions[i] = max(0, min(grain_starts[i], n - grain_len))

    # Random pitch within range
    if pitch_range > 0:
        semitones = rng.uniform(-pitch_range, pitch_range, n_grains).astype(np.float64)
        grain_ratios = np.power(2.0, semitones / 12.0).astype(np.float64)
    else:
        grain_ratios = np.ones(n_grains, dtype=np.float64)

    grain_amplitudes = np.ones(n_grains, dtype=np.float32)
    grain_lens_arr = np.full(n_grains, grain_len, dtype=np.int64)

    out = np.zeros(n, dtype=np.float32)
    _overlap_add_grains(samples, out, grain_starts.astype(np.int64),
                        grain_positions.astype(np.int64), grain_lens_arr,
                        grain_ratios, grain_amplitudes, window, n)
    return out


def variants_i007():
    return [
        {'grain_size_ms': 3, 'density': 300, 'pitch_range': 0},     # pure micro-texture, no pitch variation
        {'grain_size_ms': 3, 'density': 300, 'pitch_range': 2},     # default microsound particles
        {'grain_size_ms': 1, 'density': 1000, 'pitch_range': 1},    # ultra-fine, maximum density, narrow pitch
        {'grain_size_ms': 10, 'density': 100, 'pitch_range': 5},    # borderline micro/normal, wider pitch
        {'grain_size_ms': 2, 'density': 500, 'pitch_range': 12},    # very dense, full octave scatter
        {'grain_size_ms': 5, 'density': 200, 'pitch_range': 0},     # medium micro-grains, clean texture
        {'grain_size_ms': 1, 'density': 800, 'pitch_range': 8},     # smallest grains, wide pitch, noise-like
    ]
