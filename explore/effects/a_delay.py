"""A-series effects: Delay-based algorithms (A001-A013)."""
import numpy as np
import numba


# ---------------------------------------------------------------------------
# A001 -- Simple Feedback Delay
# ---------------------------------------------------------------------------

@numba.njit
def _simple_delay_kernel(samples, delay_samples, feedback):
    n = len(samples)
    buf_len = max(delay_samples + 1, 1)
    buf = np.zeros(buf_len, dtype=np.float32)
    out = np.zeros(n, dtype=np.float32)
    write_pos = 0
    for i in range(n):
        read_pos = (write_pos - delay_samples) % buf_len
        y = samples[i] + feedback * buf[read_pos]
        buf[write_pos] = y
        out[i] = y
        write_pos = (write_pos + 1) % buf_len
    return out


def effect_a001_simple_delay(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Simple feedback delay with ring buffer."""
    delay_ms = np.float32(params.get('delay_ms', 300))
    feedback = np.float32(params.get('feedback', 0.5))
    delay_samples = max(1, int(delay_ms * sr / 1000.0))
    return _simple_delay_kernel(samples.astype(np.float32), delay_samples, feedback)


def variants_a001():
    return [
        {'delay_ms': 100, 'feedback': 0.3},
        {'delay_ms': 250, 'feedback': 0.5},
        {'delay_ms': 500, 'feedback': 0.7},
        {'delay_ms': 75, 'feedback': 0.85},
        {'delay_ms': 1000, 'feedback': 0.4},
        {'delay_ms': 150, 'feedback': 0.0},
    ]


# ---------------------------------------------------------------------------
# A002 -- Multi-Tap Delay
# ---------------------------------------------------------------------------

@numba.njit
def _multi_tap_kernel(samples, tap_delays, tap_gains):
    n = len(samples)
    max_delay = 0
    for d in tap_delays:
        if d > max_delay:
            max_delay = d
    buf_len = max(max_delay + 1, 1)
    buf = np.zeros(buf_len, dtype=np.float32)
    out = np.zeros(n, dtype=np.float32)
    write_pos = 0
    num_taps = len(tap_delays)
    for i in range(n):
        y = samples[i]
        for t in range(num_taps):
            read_pos = (write_pos - tap_delays[t]) % buf_len
            y += tap_gains[t] * buf[read_pos]
        buf[write_pos] = samples[i]
        out[i] = y
        write_pos = (write_pos + 1) % buf_len
    return out


def effect_a002_multi_tap_delay(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Multi-tap delay with golden-ratio spaced taps."""
    num_taps = int(params.get('num_taps', 4))
    base_ms = np.float32(params.get('base_ms', 100))
    decay = np.float32(params.get('decay', 0.7))

    phi = np.float32(1.6180339887)
    tap_delays = np.zeros(num_taps, dtype=np.int64)
    tap_gains = np.zeros(num_taps, dtype=np.float32)
    for k in range(num_taps):
        delay_ms = base_ms * (phi ** k)
        tap_delays[k] = max(1, int(delay_ms * sr / 1000.0))
        tap_gains[k] = np.float32(decay ** k)

    return _multi_tap_kernel(samples.astype(np.float32), tap_delays, tap_gains)


def variants_a002():
    return [
        {'num_taps': 3, 'base_ms': 80, 'decay': 0.7},
        {'num_taps': 5, 'base_ms': 50, 'decay': 0.8},
        {'num_taps': 8, 'base_ms': 120, 'decay': 0.6},
        {'num_taps': 2, 'base_ms': 400, 'decay': 0.9},
        {'num_taps': 6, 'base_ms': 200, 'decay': 0.5},
    ]


# ---------------------------------------------------------------------------
# A003 -- Ping-Pong Stereo Delay
# ---------------------------------------------------------------------------

@numba.njit
def _ping_pong_kernel(samples, delay_samples, feedback):
    n = len(samples)
    buf_len = max(delay_samples + 1, 1)
    buf_l = np.zeros(buf_len, dtype=np.float32)
    buf_r = np.zeros(buf_len, dtype=np.float32)
    out = np.zeros((n, 2), dtype=np.float32)
    write_pos = 0
    for i in range(n):
        read_pos = (write_pos - delay_samples) % buf_len
        l_val = samples[i] + feedback * buf_r[read_pos]
        r_val = feedback * buf_l[read_pos]
        buf_l[write_pos] = l_val
        buf_r[write_pos] = r_val
        out[i, 0] = l_val
        out[i, 1] = r_val
        write_pos = (write_pos + 1) % buf_len
    return out


def effect_a003_ping_pong_delay(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Ping-pong stereo delay bouncing between L and R."""
    delay_ms = np.float32(params.get('delay_ms', 300))
    feedback = np.float32(params.get('feedback', 0.5))
    delay_samples = max(1, int(delay_ms * sr / 1000.0))
    return _ping_pong_kernel(samples.astype(np.float32), delay_samples, feedback)


def variants_a003():
    return [
        {'delay_ms': 150, 'feedback': 0.4},
        {'delay_ms': 300, 'feedback': 0.6},
        {'delay_ms': 500, 'feedback': 0.75},
        {'delay_ms': 100, 'feedback': 0.85},
        {'delay_ms': 800, 'feedback': 0.3},
    ]


# ---------------------------------------------------------------------------
# A004 -- Reverse Delay
# ---------------------------------------------------------------------------

@numba.njit
def _reverse_delay_kernel(samples, chunk_size, feedback):
    n = len(samples)
    out = np.zeros(n, dtype=np.float32)
    buf = np.zeros(chunk_size, dtype=np.float32)
    rev = np.zeros(chunk_size, dtype=np.float32)
    fb_accum = np.zeros(n, dtype=np.float32)

    # First pass: copy input
    for i in range(n):
        fb_accum[i] = samples[i]

    num_passes = 4  # number of feedback iterations
    gain = np.float32(1.0)
    for p in range(num_passes):
        num_chunks = n // chunk_size
        for c in range(num_chunks):
            start = c * chunk_size
            # Fill buf from fb_accum
            for j in range(chunk_size):
                buf[j] = fb_accum[start + j]
            # Reverse
            for j in range(chunk_size):
                rev[j] = buf[chunk_size - 1 - j]
            # Add reversed chunk to output, delayed by one chunk
            out_start = start + chunk_size
            if out_start + chunk_size <= n:
                for j in range(chunk_size):
                    out[out_start + j] += gain * rev[j]
        gain *= feedback

    # Mix in dry
    for i in range(n):
        out[i] += samples[i]

    return out


def effect_a004_reverse_delay(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Reverse delay: reverses audio chunks and mixes back."""
    delay_ms = np.float32(params.get('delay_ms', 250))
    feedback = np.float32(params.get('feedback', 0.5))
    chunk_size = max(1, int(delay_ms * sr / 1000.0))
    return _reverse_delay_kernel(samples.astype(np.float32), chunk_size, feedback)


def variants_a004():
    return [
        {'delay_ms': 100, 'feedback': 0.4},
        {'delay_ms': 200, 'feedback': 0.6},
        {'delay_ms': 500, 'feedback': 0.3},
        {'delay_ms': 150, 'feedback': 0.8},
        {'delay_ms': 350, 'feedback': 0.5},
    ]


# ---------------------------------------------------------------------------
# A005 -- Tape Delay Emulation
# ---------------------------------------------------------------------------

@numba.njit
def _tape_delay_kernel(samples, delay_samples, feedback, wow_rate_hz, wow_depth,
                       filter_coeff, sr):
    n = len(samples)
    buf_len = delay_samples + int(wow_depth) + 4  # extra headroom for modulation
    buf = np.zeros(buf_len, dtype=np.float32)
    out = np.zeros(n, dtype=np.float32)
    write_pos = 0
    lp_state = np.float32(0.0)
    two_pi = np.float32(2.0 * 3.141592653589793)

    for i in range(n):
        # Modulated delay with wow
        phase = two_pi * wow_rate_hz * np.float32(i) / np.float32(sr)
        mod = wow_depth * np.float32(np.sin(phase))
        frac_delay = np.float32(delay_samples) + mod
        int_delay = int(frac_delay)
        frac = frac_delay - np.float32(int_delay)

        # Linear interpolation for fractional read
        read_pos_0 = (write_pos - int_delay) % buf_len
        read_pos_1 = (write_pos - int_delay - 1) % buf_len
        delayed = (np.float32(1.0) - frac) * buf[read_pos_0] + frac * buf[read_pos_1]

        # One-pole lowpass in feedback path
        lp_state = filter_coeff * lp_state + (np.float32(1.0) - filter_coeff) * delayed

        y = samples[i] + feedback * lp_state
        buf[write_pos] = y
        out[i] = y
        write_pos = (write_pos + 1) % buf_len

    return out


def effect_a005_tape_delay(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Tape delay emulation with wow modulation and lowpass feedback."""
    delay_ms = np.float32(params.get('delay_ms', 300))
    feedback = np.float32(params.get('feedback', 0.5))
    wow_rate_hz = np.float32(params.get('wow_rate_hz', 1.5))
    wow_depth = np.float32(params.get('wow_depth_samples', 3.0))
    filter_cutoff = np.float32(params.get('filter_cutoff', 3500.0))

    delay_samples = max(1, int(delay_ms * sr / 1000.0))
    # Compute one-pole coefficient from cutoff
    dt = np.float32(1.0) / np.float32(sr)
    rc = np.float32(1.0 / (2.0 * 3.141592653589793 * filter_cutoff))
    filter_coeff = np.float32(np.exp(-dt / rc))

    return _tape_delay_kernel(
        samples.astype(np.float32), delay_samples, feedback,
        wow_rate_hz, wow_depth, filter_coeff, sr
    )


def variants_a005():
    return [
        {'delay_ms': 200, 'feedback': 0.5, 'wow_rate_hz': 0.5, 'wow_depth_samples': 2, 'filter_cutoff': 4000},
        {'delay_ms': 400, 'feedback': 0.6, 'wow_rate_hz': 1.0, 'wow_depth_samples': 5, 'filter_cutoff': 3000},
        {'delay_ms': 150, 'feedback': 0.75, 'wow_rate_hz': 2.5, 'wow_depth_samples': 8, 'filter_cutoff': 2000},
        {'delay_ms': 600, 'feedback': 0.3, 'wow_rate_hz': 0.3, 'wow_depth_samples': 1, 'filter_cutoff': 5000},
        {'delay_ms': 100, 'feedback': 0.8, 'wow_rate_hz': 3.0, 'wow_depth_samples': 4, 'filter_cutoff': 2500},
        {'delay_ms': 500, 'feedback': 0.45, 'wow_rate_hz': 1.8, 'wow_depth_samples': 6, 'filter_cutoff': 3500},
    ]


# ---------------------------------------------------------------------------
# A006 -- Granular Delay
# ---------------------------------------------------------------------------

@numba.njit
def _hann_window(size):
    """Generate a Hann window."""
    w = np.zeros(size, dtype=np.float32)
    for i in range(size):
        w[i] = np.float32(0.5 * (1.0 - np.cos(2.0 * 3.141592653589793 * i / (size - 1))))
    return w


@numba.njit
def _granular_delay_kernel(samples, delay_samples, grain_size, scatter_samples,
                           density, feedback, sr):
    n = len(samples)
    buf_len = max(delay_samples + scatter_samples + grain_size + 1, grain_size + 1)
    buf = np.zeros(buf_len, dtype=np.float32)
    out = np.zeros(n, dtype=np.float32)
    write_pos = 0

    window = _hann_window(grain_size)

    # Grain scheduling: interval between grains
    grain_interval = max(1, int(np.float32(sr) / np.float32(density)))

    # Simple pseudo-random via LCG
    rng_state = np.int64(42)

    for i in range(n):
        # Write input + feedback into delay buffer
        buf[write_pos] = samples[i] + feedback * out[max(0, i - delay_samples)] if i >= delay_samples else samples[i]
        if i < delay_samples:
            buf[write_pos] = samples[i]

        # Spawn grain at scheduled intervals
        if i % grain_interval == 0 and i >= delay_samples:
            # Pseudo-random scatter offset
            rng_state = (rng_state * np.int64(1103515245) + np.int64(12345)) & np.int64(0x7FFFFFFF)
            scatter_offset = int(rng_state % max(1, np.int64(scatter_samples * 2 + 1))) - scatter_samples

            for j in range(grain_size):
                out_idx = i + j
                if out_idx >= n:
                    break
                read_pos = (write_pos - delay_samples + scatter_offset - j) % buf_len
                if read_pos < 0:
                    read_pos += buf_len
                out[out_idx] += window[j] * buf[read_pos]

        write_pos = (write_pos + 1) % buf_len

    # Mix in dry signal
    for i in range(n):
        out[i] = samples[i] + out[i]

    return out


def effect_a006_granular_delay(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Granular delay with overlap-add grain scheduling."""
    delay_ms = np.float32(params.get('delay_ms', 300))
    grain_size_ms = np.float32(params.get('grain_size_ms', 50))
    scatter_ms = np.float32(params.get('scatter_ms', 50))
    density = np.float32(params.get('density', 20))
    feedback = np.float32(params.get('feedback', 0.3))

    delay_samples = max(1, int(delay_ms * sr / 1000.0))
    grain_size = max(2, int(grain_size_ms * sr / 1000.0))
    scatter_samples = max(0, int(scatter_ms * sr / 1000.0))

    return _granular_delay_kernel(
        samples.astype(np.float32), delay_samples, grain_size,
        scatter_samples, density, feedback, sr
    )


def variants_a006():
    return [
        {'delay_ms': 200, 'grain_size_ms': 30, 'scatter_ms': 10, 'density': 15, 'feedback': 0.2},
        {'delay_ms': 500, 'grain_size_ms': 80, 'scatter_ms': 100, 'density': 30, 'feedback': 0.5},
        {'delay_ms': 1000, 'grain_size_ms': 100, 'scatter_ms': 200, 'density': 50, 'feedback': 0.0},
        {'delay_ms': 150, 'grain_size_ms': 10, 'scatter_ms': 0, 'density': 40, 'feedback': 0.6},
        {'delay_ms': 300, 'grain_size_ms': 50, 'scatter_ms': 50, 'density': 10, 'feedback': 0.7},
        {'delay_ms': 700, 'grain_size_ms': 60, 'scatter_ms': 150, 'density': 5, 'feedback': 0.4},
    ]


# ---------------------------------------------------------------------------
# A007 -- Allpass Delay Diffuser
# ---------------------------------------------------------------------------

@numba.njit
def _allpass_diffuser_kernel(samples, delays, g):
    """Chain of allpass filters: y[n] = -g*x[n] + x[n-d] + g*y[n-d]."""
    n = len(samples)
    num_stages = len(delays)
    # Process each stage sequentially
    current = samples.copy()

    for s in range(num_stages):
        d = delays[s]
        buf_len = max(d + 1, 1)
        x_buf = np.zeros(buf_len, dtype=np.float32)
        y_buf = np.zeros(buf_len, dtype=np.float32)
        stage_out = np.zeros(n, dtype=np.float32)
        write_pos = 0

        for i in range(n):
            read_pos = (write_pos - d) % buf_len
            x_delayed = x_buf[read_pos]
            y_delayed = y_buf[read_pos]
            y_val = -g * current[i] + x_delayed + g * y_delayed
            x_buf[write_pos] = current[i]
            y_buf[write_pos] = y_val
            stage_out[i] = y_val
            write_pos = (write_pos + 1) % buf_len

        current = stage_out

    return current


def effect_a007_allpass_diffuser(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Chain of allpass filters with different delay lengths for diffusion."""
    num_stages = int(params.get('num_stages', 6))
    delay_range_ms = np.float32(params.get('delay_range_ms', 20))
    g = np.float32(params.get('g', 0.6))

    # Distribute delays across range using prime-like spacing
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    delays = np.zeros(num_stages, dtype=np.int64)
    for i in range(num_stages):
        prime_idx = i % len(primes)
        frac = np.float32(primes[prime_idx]) / np.float32(primes[min(num_stages - 1, len(primes) - 1)])
        delay_ms = np.float32(1.0) + frac * (delay_range_ms - np.float32(1.0))
        delays[i] = max(1, int(delay_ms * sr / 1000.0))

    return _allpass_diffuser_kernel(samples.astype(np.float32), delays, g)


def variants_a007():
    return [
        {'num_stages': 4, 'delay_range_ms': 5, 'g': 0.5},
        {'num_stages': 8, 'delay_range_ms': 20, 'g': 0.6},
        {'num_stages': 12, 'delay_range_ms': 50, 'g': 0.7},
        {'num_stages': 6, 'delay_range_ms': 10, 'g': 0.55},
        {'num_stages': 10, 'delay_range_ms': 35, 'g': 0.65},
    ]


# ---------------------------------------------------------------------------
# A008 -- Fibonacci Delay Network
# ---------------------------------------------------------------------------

@numba.njit
def _fibonacci_delay_kernel(samples, tap_delays, tap_gains):
    n = len(samples)
    max_delay = np.int64(0)
    for d in tap_delays:
        if d > max_delay:
            max_delay = d
    buf_len = max(max_delay + 1, 1)
    buf = np.zeros(buf_len, dtype=np.float32)
    out = np.zeros(n, dtype=np.float32)
    write_pos = 0
    num_taps = len(tap_delays)

    for i in range(n):
        y = samples[i]
        for t in range(num_taps):
            read_pos = (write_pos - tap_delays[t]) % buf_len
            y += tap_gains[t] * buf[read_pos]
        buf[write_pos] = y
        out[i] = y
        write_pos = (write_pos + 1) % buf_len
    return out


def effect_a008_fibonacci_delay(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Delay taps at Fibonacci number multiples of base_ms."""
    base_ms = np.float32(params.get('base_ms', 20))
    num_fibs = int(params.get('num_fibs', 8))
    decay = np.float32(params.get('decay', 0.8))

    # Generate fibonacci sequence
    fibs = [1, 1]
    while len(fibs) < num_fibs:
        fibs.append(fibs[-1] + fibs[-2])

    tap_delays = np.zeros(num_fibs, dtype=np.int64)
    tap_gains = np.zeros(num_fibs, dtype=np.float32)
    for k in range(num_fibs):
        delay_ms = base_ms * np.float32(fibs[k])
        tap_delays[k] = max(1, int(delay_ms * sr / 1000.0))
        tap_gains[k] = np.float32(decay ** (k + 1))

    # Normalize gains so their sum stays below 1.0 for feedback stability
    total_gain = np.float32(0.0)
    for k in range(num_fibs):
        total_gain += tap_gains[k]
    if total_gain > np.float32(0.95):
        scale = np.float32(0.95) / total_gain
        for k in range(num_fibs):
            tap_gains[k] *= scale

    return _fibonacci_delay_kernel(samples.astype(np.float32), tap_delays, tap_gains)


def variants_a008():
    return [
        {'base_ms': 5, 'num_fibs': 8, 'decay': 0.8},
        {'base_ms': 15, 'num_fibs': 6, 'decay': 0.9},
        {'base_ms': 50, 'num_fibs': 5, 'decay': 0.7},
        {'base_ms': 10, 'num_fibs': 12, 'decay': 0.6},
        {'base_ms': 30, 'num_fibs': 10, 'decay': 0.95},
        {'base_ms': 8, 'num_fibs': 7, 'decay': 0.85},
    ]


# ---------------------------------------------------------------------------
# A009 -- Prime Number Delay
# ---------------------------------------------------------------------------

@numba.njit
def _prime_delay_kernel(samples, tap_delays, feedback):
    n = len(samples)
    max_delay = np.int64(0)
    num_taps = len(tap_delays)
    for t in range(num_taps):
        if tap_delays[t] > max_delay:
            max_delay = tap_delays[t]
    buf_len = max(max_delay + 1, 1)
    buf = np.zeros(buf_len, dtype=np.float32)
    out = np.zeros(n, dtype=np.float32)
    write_pos = 0

    # Equal gain per tap scaled by feedback
    tap_gain = feedback / np.float32(num_taps)

    for i in range(n):
        y = samples[i]
        for t in range(num_taps):
            read_pos = (write_pos - tap_delays[t]) % buf_len
            y += tap_gain * buf[read_pos]
        buf[write_pos] = y
        out[i] = y
        write_pos = (write_pos + 1) % buf_len
    return out


def _gen_primes(count):
    """Generate the first `count` prime numbers."""
    primes = []
    candidate = 2
    while len(primes) < count:
        is_prime = True
        for p in primes:
            if p * p > candidate:
                break
            if candidate % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(candidate)
        candidate += 1
    return primes


def effect_a009_prime_delay(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Delay taps at prime number multiples of base_ms."""
    base_ms = np.float32(params.get('base_ms', 5))
    num_primes = int(params.get('num_primes', 8))
    feedback = np.float32(params.get('feedback', 0.5))

    primes = _gen_primes(num_primes)
    tap_delays = np.zeros(num_primes, dtype=np.int64)
    for k in range(num_primes):
        delay_ms = base_ms * np.float32(primes[k])
        tap_delays[k] = max(1, int(delay_ms * sr / 1000.0))

    return _prime_delay_kernel(samples.astype(np.float32), tap_delays, feedback)


def variants_a009():
    return [
        {'base_ms': 2, 'num_primes': 8, 'feedback': 0.5},
        {'base_ms': 5, 'num_primes': 12, 'feedback': 0.6},
        {'base_ms': 10, 'num_primes': 5, 'feedback': 0.8},
        {'base_ms': 1, 'num_primes': 15, 'feedback': 0.3},
        {'base_ms': 20, 'num_primes': 6, 'feedback': 0.7},
        {'base_ms': 15, 'num_primes': 10, 'feedback': 0.4},
    ]


# ---------------------------------------------------------------------------
# A010 -- Stutter / Retrigger
# ---------------------------------------------------------------------------

@numba.njit
def _stutter_kernel(samples, window_size, repeats, decay, pitch_drift):
    n = len(samples)
    out = np.zeros(n, dtype=np.float32)

    # Process in blocks of window_size
    num_windows = n // window_size
    out_pos = 0

    for w in range(num_windows):
        src_start = w * window_size
        # Capture the window
        window = np.zeros(window_size, dtype=np.float32)
        for j in range(window_size):
            if src_start + j < n:
                window[j] = samples[src_start + j]

        # Place original
        for j in range(window_size):
            if out_pos + j < n:
                out[out_pos + j] = window[j]

        # Place repeats within the same window duration
        # Each repeat gets a portion of the window time
        total_repeat_space = window_size
        repeat_len = max(1, total_repeat_space // max(1, repeats))

        gain = np.float32(1.0)
        for r in range(1, repeats):
            gain *= decay
            drift_factor = np.float32(1.0) + pitch_drift * np.float32(r)
            for j in range(repeat_len):
                src_j = int(np.float32(j) * drift_factor)
                if src_j >= window_size:
                    break
                out_idx = out_pos + r * repeat_len + j
                if out_idx < n:
                    out[out_idx] += gain * window[src_j]

        out_pos += window_size

    # Copy any remaining samples
    remaining_start = num_windows * window_size
    for i in range(remaining_start, n):
        out[i] = samples[i]

    return out


def effect_a010_stutter(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Stutter/retrigger: capture window and repeat with decay."""
    window_ms = np.float32(params.get('window_ms', 80))
    repeats = int(params.get('repeats', 8))
    decay = np.float32(params.get('decay', 0.9))
    pitch_drift = np.float32(params.get('pitch_drift', 0.0))

    window_size = max(1, int(window_ms * sr / 1000.0))
    return _stutter_kernel(samples.astype(np.float32), window_size, repeats, decay, pitch_drift)


def variants_a010():
    return [
        {'window_ms': 50, 'repeats': 4, 'decay': 0.9, 'pitch_drift': 0.0},
        {'window_ms': 100, 'repeats': 8, 'decay': 0.85, 'pitch_drift': 0.02},
        {'window_ms': 200, 'repeats': 16, 'decay': 0.95, 'pitch_drift': -0.05},
        {'window_ms': 30, 'repeats': 32, 'decay': 0.8, 'pitch_drift': 0.0},
        {'window_ms': 150, 'repeats': 6, 'decay': 1.0, 'pitch_drift': 0.1},
        {'window_ms': 20, 'repeats': 12, 'decay': 0.92, 'pitch_drift': -0.1},
    ]


# ---------------------------------------------------------------------------
# A011 -- Buffer Shuffle
# ---------------------------------------------------------------------------

def effect_a011_buffer_shuffle(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Divide signal into chunks and apply random permutation with crossfade."""
    chunk_ms = np.float32(params.get('chunk_ms', 200))
    seed = int(params.get('seed', 42))

    samples = samples.astype(np.float32)
    n = len(samples)
    chunk_size = max(1, int(chunk_ms * sr / 1000.0))
    num_chunks = n // chunk_size

    if num_chunks < 2:
        return samples.copy()

    # Create chunks
    chunks = []
    for c in range(num_chunks):
        start = c * chunk_size
        chunks.append(samples[start:start + chunk_size].copy())

    # Permute chunk order
    rng = np.random.RandomState(seed)
    perm = rng.permutation(num_chunks)

    # Reassemble with crossfade
    fade_samples = min(int(0.005 * sr), chunk_size // 4)  # 5ms crossfade
    out = np.zeros(n, dtype=np.float32)

    for idx in range(num_chunks):
        src_chunk = chunks[perm[idx]]
        start = idx * chunk_size

        # Apply fade in/out to each chunk
        chunk_copy = src_chunk.copy()
        for j in range(fade_samples):
            fade = np.float32(j) / np.float32(fade_samples)
            chunk_copy[j] *= fade
            chunk_copy[chunk_size - 1 - j] *= fade

        out[start:start + chunk_size] = chunk_copy

    # Copy remainder
    remainder_start = num_chunks * chunk_size
    if remainder_start < n:
        out[remainder_start:n] = samples[remainder_start:n]

    return out


def variants_a011():
    return [
        {'chunk_ms': 100, 'seed': 1},
        {'chunk_ms': 250, 'seed': 17},
        {'chunk_ms': 500, 'seed': 42},
        {'chunk_ms': 50, 'seed': 99},
        {'chunk_ms': 150, 'seed': 7},
    ]


# ---------------------------------------------------------------------------
# A012 -- Reverse Chunks
# ---------------------------------------------------------------------------

def effect_a012_reverse_chunks(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Reverse every other chunk (or random subset) with crossfade."""
    chunk_ms = np.float32(params.get('chunk_ms', 150))
    reverse_probability = np.float32(params.get('reverse_probability', 0.5))

    samples = samples.astype(np.float32)
    n = len(samples)
    chunk_size = max(1, int(chunk_ms * sr / 1000.0))
    num_chunks = n // chunk_size
    fade_samples = min(int(0.005 * sr), chunk_size // 4)  # 5ms crossfade

    out = np.zeros(n, dtype=np.float32)

    rng = np.random.RandomState(123)

    for c in range(num_chunks):
        start = c * chunk_size
        chunk = samples[start:start + chunk_size].copy()

        # Decide whether to reverse
        should_reverse = rng.random() < reverse_probability

        if should_reverse:
            chunk = chunk[::-1].copy()

        # Apply crossfade
        for j in range(fade_samples):
            fade = np.float32(j) / np.float32(fade_samples)
            chunk[j] *= fade
            chunk[chunk_size - 1 - j] *= fade

        out[start:start + chunk_size] = chunk

    # Copy remainder
    remainder_start = num_chunks * chunk_size
    if remainder_start < n:
        out[remainder_start:n] = samples[remainder_start:n]

    return out


def variants_a012():
    return [
        {'chunk_ms': 50, 'reverse_probability': 0.5},
        {'chunk_ms': 100, 'reverse_probability': 0.7},
        {'chunk_ms': 200, 'reverse_probability': 1.0},
        {'chunk_ms': 300, 'reverse_probability': 0.3},
        {'chunk_ms': 80, 'reverse_probability': 0.9},
        {'chunk_ms': 150, 'reverse_probability': 0.6},
    ]


# ---------------------------------------------------------------------------
# A013 -- Bouncing Ball Delay
# ---------------------------------------------------------------------------

@numba.njit
def _bouncing_ball_kernel(samples, sr, initial_delay_samples, decay, num_bounces,
                          damping_coeff):
    n = len(samples)
    out = np.zeros(n, dtype=np.float32)

    # Copy dry signal
    for i in range(n):
        out[i] = samples[i]

    # Each bounce: delay halves (like gravity), amplitude decays
    cumulative_delay = 0
    gain = np.float32(1.0)
    current_delay = initial_delay_samples

    for b in range(num_bounces):
        cumulative_delay += current_delay
        gain *= decay

        # Add delayed copy
        for i in range(n):
            src = i - cumulative_delay
            if 0 <= src < n:
                out[i] += gain * samples[src]

        # Next bounce: shorter interval (simulating gravity)
        current_delay = max(1, int(np.float32(current_delay) * damping_coeff))
        if current_delay < 1:
            break

    return out


def effect_a013_bouncing_ball(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Bouncing ball delay: exponentially decreasing delay times simulating a
    bouncing object. Each bounce gets shorter and quieter."""
    initial_delay_ms = float(params.get('initial_delay_ms', 400))
    decay = np.float32(params.get('decay', 0.7))
    num_bounces = int(params.get('num_bounces', 15))
    damping = np.float32(params.get('damping', 0.65))

    initial_delay_samples = max(1, int(initial_delay_ms * sr / 1000.0))
    return _bouncing_ball_kernel(
        samples.astype(np.float32), sr, initial_delay_samples,
        decay, num_bounces, damping
    )


def variants_a013():
    return [
        {'initial_delay_ms': 500, 'decay': 0.7, 'num_bounces': 12, 'damping': 0.6},
        {'initial_delay_ms': 300, 'decay': 0.8, 'num_bounces': 20, 'damping': 0.7},
        {'initial_delay_ms': 200, 'decay': 0.6, 'num_bounces': 10, 'damping': 0.5},
        {'initial_delay_ms': 800, 'decay': 0.75, 'num_bounces': 8, 'damping': 0.65},
        {'initial_delay_ms': 150, 'decay': 0.85, 'num_bounces': 25, 'damping': 0.75},
        {'initial_delay_ms': 1000, 'decay': 0.5, 'num_bounces': 6, 'damping': 0.55},
    ]
