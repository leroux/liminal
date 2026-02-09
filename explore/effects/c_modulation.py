"""C-series: Modulation effects (C001-C012).

Chorus, flanger, phaser, vibrato, tremolo, ring modulation variants,
frequency shifting, barber pole flanger, stereo auto-pan, doppler.
"""
import numpy as np
import numba


# ---------------------------------------------------------------------------
# C001 -- Chorus: modulated delay with multiple voices
# ---------------------------------------------------------------------------

@numba.njit
def _chorus_kernel(samples, sr, base_delay_ms, depth_ms, rate_hz, voices):
    n = len(samples)
    base_delay_samp = base_delay_ms * np.float32(0.001) * np.float32(sr)
    depth_samp = depth_ms * np.float32(0.001) * np.float32(sr)
    max_delay = int(base_delay_samp + depth_samp + 2)
    buf = np.zeros(max_delay + 1, dtype=np.float32)
    out = np.zeros(n, dtype=np.float32)
    write_pos = 0
    inv_voices = np.float32(1.0) / np.float32(voices)
    two_pi = np.float32(2.0 * np.pi)

    for i in range(n):
        buf[write_pos] = samples[i]
        wet = np.float32(0.0)
        for v in range(voices):
            phase_offset = two_pi * np.float32(v) / np.float32(voices)
            mod = np.sin(two_pi * rate_hz * np.float32(i) / np.float32(sr) + phase_offset)
            delay = base_delay_samp + depth_samp * np.float32(mod)
            read_pos_f = np.float32(write_pos) - delay
            if read_pos_f < 0.0:
                read_pos_f += np.float32(max_delay + 1)
            idx = int(read_pos_f)
            frac = read_pos_f - np.float32(idx)
            idx0 = idx % (max_delay + 1)
            idx1 = (idx + 1) % (max_delay + 1)
            wet += buf[idx0] * (np.float32(1.0) - frac) + buf[idx1] * frac
        out[i] = samples[i] * np.float32(0.5) + wet * inv_voices * np.float32(0.5)
        write_pos = (write_pos + 1) % (max_delay + 1)
    return out


def effect_c001_chorus(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Chorus: modulated delay with multiple voices and phase offsets."""
    base_delay_ms = np.float32(params.get('base_delay_ms', 15.0))
    depth_ms = np.float32(params.get('depth_ms', 3.0))
    rate_hz = np.float32(params.get('rate_hz', 1.5))
    voices = int(params.get('voices', 2))
    return _chorus_kernel(samples.astype(np.float32), sr, base_delay_ms,
                          depth_ms, rate_hz, voices)


def variants_c001():
    return [
        {'base_delay_ms': 7.0, 'depth_ms': 1.5, 'rate_hz': 0.8, 'voices': 1},
        {'base_delay_ms': 12.0, 'depth_ms': 3.0, 'rate_hz': 1.2, 'voices': 2},
        {'base_delay_ms': 20.0, 'depth_ms': 5.0, 'rate_hz': 0.3, 'voices': 3},
        {'base_delay_ms': 25.0, 'depth_ms': 8.0, 'rate_hz': 4.5, 'voices': 4},
        {'base_delay_ms': 5.0, 'depth_ms': 2.0, 'rate_hz': 2.0, 'voices': 2},
        {'base_delay_ms': 30.0, 'depth_ms': 10.0, 'rate_hz': 0.1, 'voices': 4},
    ]


# ---------------------------------------------------------------------------
# C002 -- Flanger: short modulated delay with feedback
# ---------------------------------------------------------------------------

@numba.njit
def _flanger_kernel(samples, sr, base_delay_ms, depth_ms, rate_hz, feedback):
    n = len(samples)
    base_delay_samp = base_delay_ms * np.float32(0.001) * np.float32(sr)
    depth_samp = depth_ms * np.float32(0.001) * np.float32(sr)
    max_delay = int(base_delay_samp + depth_samp + 2)
    buf = np.zeros(max_delay + 1, dtype=np.float32)
    out = np.zeros(n, dtype=np.float32)
    write_pos = 0
    two_pi = np.float32(2.0 * np.pi)
    fb_sample = np.float32(0.0)

    for i in range(n):
        inp = samples[i] + feedback * fb_sample
        buf[write_pos] = inp
        mod = np.sin(two_pi * rate_hz * np.float32(i) / np.float32(sr))
        delay = base_delay_samp + depth_samp * np.float32(mod)
        read_pos_f = np.float32(write_pos) - delay
        if read_pos_f < 0.0:
            read_pos_f += np.float32(max_delay + 1)
        idx = int(read_pos_f)
        frac = read_pos_f - np.float32(idx)
        idx0 = idx % (max_delay + 1)
        idx1 = (idx + 1) % (max_delay + 1)
        fb_sample = buf[idx0] * (np.float32(1.0) - frac) + buf[idx1] * frac
        out[i] = samples[i] * np.float32(0.5) + fb_sample * np.float32(0.5)
        write_pos = (write_pos + 1) % (max_delay + 1)
    return out


def effect_c002_flanger(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Flanger: short modulated delay with feedback for comb filtering."""
    base_delay_ms = np.float32(params.get('base_delay_ms', 2.0))
    depth_ms = np.float32(params.get('depth_ms', 2.0))
    rate_hz = np.float32(params.get('rate_hz', 0.3))
    feedback = np.float32(params.get('feedback', 0.7))
    return _flanger_kernel(samples.astype(np.float32), sr, base_delay_ms,
                           depth_ms, rate_hz, feedback)


def variants_c002():
    return [
        {'base_delay_ms': 0.5, 'depth_ms': 0.5, 'rate_hz': 0.1, 'feedback': 0.3},
        {'base_delay_ms': 1.5, 'depth_ms': 1.5, 'rate_hz': 0.25, 'feedback': 0.7},
        {'base_delay_ms': 3.0, 'depth_ms': 3.0, 'rate_hz': 0.5, 'feedback': -0.8},
        {'base_delay_ms': 5.0, 'depth_ms': 5.0, 'rate_hz': 2.0, 'feedback': 0.95},
        {'base_delay_ms': 1.0, 'depth_ms': 1.0, 'rate_hz': 0.05, 'feedback': -0.95},
        {'base_delay_ms': 2.5, 'depth_ms': 4.0, 'rate_hz': 1.0, 'feedback': 0.0},
    ]


# ---------------------------------------------------------------------------
# C003 -- Phaser: chain of allpass filters with swept cutoff
# ---------------------------------------------------------------------------

@numba.njit
def _phaser_kernel(samples, sr, num_stages, f_min, f_max, rate_hz, feedback, depth):
    n = len(samples)
    out = np.zeros(n, dtype=np.float32)
    # allpass states: one state per stage
    ap_z = np.zeros(num_stages, dtype=np.float32)
    fb_sample = np.float32(0.0)
    two_pi = np.float32(2.0 * np.pi)
    pi_val = np.float32(np.pi)
    sr_f = np.float32(sr)
    log_f_min = np.float32(np.log(f_min))
    log_f_max = np.float32(np.log(f_max))

    for i in range(n):
        # LFO: 0..1
        lfo = np.float32(0.5) * (np.float32(1.0) + np.float32(
            np.sin(two_pi * rate_hz * np.float32(i) / sr_f)))
        # Swept frequency in log space
        log_f = log_f_min + (log_f_max - log_f_min) * lfo
        f = np.float32(np.exp(log_f))
        # Clamp f to avoid instability
        max_f = sr_f * np.float32(0.49)
        if f > max_f:
            f = max_f
        # allpass coefficient
        tan_val = np.float32(np.tan(pi_val * f / sr_f))
        a = (np.float32(1.0) - tan_val) / (np.float32(1.0) + tan_val)

        # Input with feedback
        x = samples[i] + feedback * fb_sample

        # Chain of first-order allpass filters
        for s in range(num_stages):
            inp = x
            x = a * inp + ap_z[s]
            ap_z[s] = inp - a * x

        fb_sample = x
        # Mix: dry + depth * wet
        out[i] = samples[i] + depth * x
    return out


def effect_c003_phaser(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Phaser: chain of allpass filters with LFO-swept cutoff frequency."""
    num_stages = int(params.get('num_stages', 6))
    f_min = np.float32(params.get('f_min', 200.0))
    f_max = np.float32(params.get('f_max', 4000.0))
    rate_hz = np.float32(params.get('rate_hz', 0.3))
    feedback = np.float32(params.get('feedback', 0.5))
    depth = np.float32(params.get('depth', 0.8))
    return _phaser_kernel(samples.astype(np.float32), sr, num_stages,
                          f_min, f_max, rate_hz, feedback, depth)


def variants_c003():
    return [
        {'num_stages': 4, 'f_min': 100, 'f_max': 2000, 'rate_hz': 0.2, 'feedback': 0.3, 'depth': 0.7},
        {'num_stages': 6, 'f_min': 200, 'f_max': 4000, 'rate_hz': 0.5, 'feedback': 0.6, 'depth': 0.9},
        {'num_stages': 8, 'f_min': 300, 'f_max': 6000, 'rate_hz': 1.0, 'feedback': 0.8, 'depth': 1.0},
        {'num_stages': 12, 'f_min': 400, 'f_max': 8000, 'rate_hz': 0.05, 'feedback': 0.9, 'depth': 0.5},
        {'num_stages': 4, 'f_min': 150, 'f_max': 1000, 'rate_hz': 2.0, 'feedback': 0.0, 'depth': 1.0},
        {'num_stages': 8, 'f_min': 100, 'f_max': 5000, 'rate_hz': 0.1, 'feedback': 0.7, 'depth': 0.6},
    ]


# ---------------------------------------------------------------------------
# C004 -- Vibrato: pure pitch modulation via modulated delay, no dry mix
# ---------------------------------------------------------------------------

@numba.njit
def _vibrato_kernel(samples, sr, rate_hz, depth_ms):
    n = len(samples)
    depth_samp = depth_ms * np.float32(0.001) * np.float32(sr)
    # Base delay is depth so modulation swings around it
    base_delay_samp = depth_samp + np.float32(1.0)
    max_delay = int(base_delay_samp + depth_samp + 2)
    buf = np.zeros(max_delay + 1, dtype=np.float32)
    out = np.zeros(n, dtype=np.float32)
    write_pos = 0
    two_pi = np.float32(2.0 * np.pi)

    for i in range(n):
        buf[write_pos] = samples[i]
        mod = np.sin(two_pi * rate_hz * np.float32(i) / np.float32(sr))
        delay = base_delay_samp + depth_samp * np.float32(mod)
        read_pos_f = np.float32(write_pos) - delay
        if read_pos_f < 0.0:
            read_pos_f += np.float32(max_delay + 1)
        idx = int(read_pos_f)
        frac = read_pos_f - np.float32(idx)
        idx0 = idx % (max_delay + 1)
        idx1 = (idx + 1) % (max_delay + 1)
        out[i] = buf[idx0] * (np.float32(1.0) - frac) + buf[idx1] * frac
        write_pos = (write_pos + 1) % (max_delay + 1)
    return out


def effect_c004_vibrato(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Vibrato: pure pitch modulation via modulated delay, no dry signal."""
    rate_hz = np.float32(params.get('rate_hz', 5.0))
    depth_ms = np.float32(params.get('depth_ms', 3.0))
    return _vibrato_kernel(samples.astype(np.float32), sr, rate_hz, depth_ms)


def variants_c004():
    return [
        {'rate_hz': 1.0, 'depth_ms': 1.0},
        {'rate_hz': 3.0, 'depth_ms': 2.0},
        {'rate_hz': 5.0, 'depth_ms': 4.0},
        {'rate_hz': 7.0, 'depth_ms': 7.0},
        {'rate_hz': 8.0, 'depth_ms': 10.0},
        {'rate_hz': 2.0, 'depth_ms': 5.0},
    ]


# ---------------------------------------------------------------------------
# C005 -- Tremolo: amplitude modulation with LFO shapes
# ---------------------------------------------------------------------------

@numba.njit
def _tremolo_kernel(samples, sr, rate_hz, depth, shape_id):
    """shape_id: 0=sin, 1=tri, 2=square, 3=sample-and-hold"""
    n = len(samples)
    out = np.zeros(n, dtype=np.float32)
    period = np.float32(sr) / rate_hz
    two_pi = np.float32(2.0 * np.pi)
    # For sample-and-hold: random seed state
    sh_val = np.float32(1.0)
    sh_counter = np.int32(0)
    sh_period = max(np.int32(period), np.int32(1))
    # Simple LCG for S&H noise in numba
    rng_state = np.uint32(12345)

    for i in range(n):
        phase = np.float32(i) / period
        frac_phase = phase - np.float32(int(phase))

        if shape_id == 0:
            # Sine
            lfo = np.float32(0.5) * (np.float32(1.0) + np.float32(
                np.sin(two_pi * rate_hz * np.float32(i) / np.float32(sr))))
        elif shape_id == 1:
            # Triangle
            if frac_phase < np.float32(0.5):
                lfo = np.float32(4.0) * frac_phase - np.float32(1.0)
            else:
                lfo = np.float32(3.0) - np.float32(4.0) * frac_phase
            lfo = np.float32(0.5) * (lfo + np.float32(1.0))
        elif shape_id == 2:
            # Square
            if frac_phase < np.float32(0.5):
                lfo = np.float32(1.0)
            else:
                lfo = np.float32(0.0)
        else:
            # Sample and hold
            if sh_counter <= 0:
                rng_state = rng_state * np.uint32(1103515245) + np.uint32(12345)
                sh_val = np.float32((rng_state >> np.uint32(16)) & np.uint32(0x7FFF)) / np.float32(32767.0)
                sh_counter = sh_period
            sh_counter -= np.int32(1)
            lfo = sh_val

        # Modulation: 1 - depth + depth * lfo  (ranges from 1-depth to 1)
        mod = np.float32(1.0) - depth * (np.float32(1.0) - lfo)
        out[i] = samples[i] * mod
    return out


def effect_c005_tremolo(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Tremolo: amplitude modulation with selectable LFO shape."""
    rate_hz = np.float32(params.get('rate_hz', 5.0))
    depth = np.float32(params.get('depth', 0.7))
    shape = params.get('shape', 'sin')
    shape_map = {'sin': 0, 'tri': 1, 'square': 2, 'sh': 3}
    shape_id = shape_map.get(shape, 0)
    return _tremolo_kernel(samples.astype(np.float32), sr, rate_hz, depth, shape_id)


def variants_c005():
    return [
        {'rate_hz': 3.0, 'depth': 0.5, 'shape': 'sin'},
        {'rate_hz': 8.0, 'depth': 0.9, 'shape': 'sin'},
        {'rate_hz': 5.0, 'depth': 0.7, 'shape': 'tri'},
        {'rate_hz': 4.0, 'depth': 1.0, 'shape': 'square'},
        {'rate_hz': 12.0, 'depth': 0.6, 'shape': 'sh'},
        {'rate_hz': 20.0, 'depth': 0.3, 'shape': 'sin'},
        {'rate_hz': 1.0, 'depth': 1.0, 'shape': 'tri'},
    ]


# ---------------------------------------------------------------------------
# C006 -- Ring Modulation: y = x * sin(2*pi*freq*n/sr)
# ---------------------------------------------------------------------------

@numba.njit
def _ring_mod_kernel(samples, sr, carrier_freq_hz):
    n = len(samples)
    out = np.zeros(n, dtype=np.float32)
    two_pi = np.float32(2.0 * np.pi)
    for i in range(n):
        carrier = np.float32(np.sin(two_pi * carrier_freq_hz * np.float32(i) / np.float32(sr)))
        out[i] = samples[i] * carrier
    return out


def effect_c006_ring_mod(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Ring modulation with sine carrier."""
    carrier_freq_hz = np.float32(params.get('carrier_freq_hz', 200.0))
    return _ring_mod_kernel(samples.astype(np.float32), sr, carrier_freq_hz)


def variants_c006():
    return [
        {'carrier_freq_hz': 20.0},
        {'carrier_freq_hz': 80.0},
        {'carrier_freq_hz': 200.0},
        {'carrier_freq_hz': 440.0},
        {'carrier_freq_hz': 1000.0},
        {'carrier_freq_hz': 2000.0},
    ]


# ---------------------------------------------------------------------------
# C007 -- Ring Mod with Noise Carrier: bandpass filtered noise
# ---------------------------------------------------------------------------

@numba.njit
def _ring_mod_noise_kernel(samples, noise_carrier):
    n = len(samples)
    out = np.zeros(n, dtype=np.float32)
    for i in range(n):
        out[i] = samples[i] * noise_carrier[i]
    return out


@numba.njit
def _biquad_bpf_inline(noise, b0, b1, b2, a1, a2):
    """Inline biquad bandpass filter."""
    n = len(noise)
    out = np.zeros(n, dtype=np.float32)
    z1 = np.float32(0.0)
    z2 = np.float32(0.0)
    for i in range(n):
        x = noise[i]
        y = b0 * x + z1
        z1 = b1 * x - a1 * y + z2
        z2 = b2 * x - a2 * y
        out[i] = y
    return out


def effect_c007_ring_mod_noise(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Ring modulation with bandpass-filtered noise carrier."""
    center_freq = params.get('center_freq', 1000.0)
    bandwidth_hz = params.get('bandwidth_hz', 500.0)
    n = len(samples)

    # Generate white noise
    rng = np.random.default_rng(42)
    noise = rng.standard_normal(n).astype(np.float32)

    # Compute bandpass biquad coefficients inline
    Q = center_freq / max(bandwidth_hz, 1.0)
    w0 = 2.0 * np.pi * center_freq / sr
    alpha = np.sin(w0) / (2.0 * Q)
    b0 = np.float32(alpha)
    b1 = np.float32(0.0)
    b2 = np.float32(-alpha)
    a0 = 1.0 + alpha
    a1_c = np.float32(-2.0 * np.cos(w0) / a0)
    a2_c = np.float32((1.0 - alpha) / a0)
    b0 = np.float32(b0 / a0)
    b2 = np.float32(b2 / a0)

    # Apply bandpass filter
    carrier = _biquad_bpf_inline(noise, b0, b1, b2, a1_c, a2_c)

    # Normalize carrier to roughly unit amplitude
    peak = np.max(np.abs(carrier))
    if peak > 1e-10:
        carrier = carrier / peak

    return _ring_mod_noise_kernel(samples.astype(np.float32), carrier)


def variants_c007():
    return [
        {'center_freq': 100.0, 'bandwidth_hz': 50.0},
        {'center_freq': 500.0, 'bandwidth_hz': 200.0},
        {'center_freq': 1000.0, 'bandwidth_hz': 500.0},
        {'center_freq': 2000.0, 'bandwidth_hz': 100.0},
        {'center_freq': 3000.0, 'bandwidth_hz': 1000.0},
        {'center_freq': 5000.0, 'bandwidth_hz': 2000.0},
    ]


# ---------------------------------------------------------------------------
# C008 -- Ring Mod with Chaos Carrier: logistic map
# ---------------------------------------------------------------------------

@numba.njit
def _ring_mod_chaos_kernel(samples, sr, r, chaos_speed):
    n = len(samples)
    out = np.zeros(n, dtype=np.float32)
    x = np.float32(0.5)  # initial state of logistic map
    carrier_val = np.float32(0.0)
    counter = 0

    for i in range(n):
        if counter <= 0:
            # Iterate logistic map
            x = r * x * (np.float32(1.0) - x)
            # Map from [0,1] to [-1,1]
            carrier_val = np.float32(2.0) * x - np.float32(1.0)
            counter = chaos_speed
        counter -= 1
        out[i] = samples[i] * carrier_val
    return out


def effect_c008_ring_mod_chaos(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Ring modulation with logistic map chaos carrier."""
    r = np.float32(params.get('r', 3.9))
    chaos_speed = int(params.get('chaos_speed', 1))
    return _ring_mod_chaos_kernel(samples.astype(np.float32), sr, r, chaos_speed)


def variants_c008():
    return [
        {'r': 3.5, 'chaos_speed': 1},
        {'r': 3.7, 'chaos_speed': 1},
        {'r': 3.85, 'chaos_speed': 2},
        {'r': 3.95, 'chaos_speed': 1},
        {'r': 4.0, 'chaos_speed': 4},
        {'r': 3.99, 'chaos_speed': 10},
        {'r': 3.6, 'chaos_speed': 50},
    ]


# ---------------------------------------------------------------------------
# C009 -- Frequency Shifting (Hilbert): analytic signal approach
# ---------------------------------------------------------------------------

def effect_c009_freq_shift(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Frequency shifting via Hilbert transform (FFT-based).

    Computes analytic signal, multiplies by complex exponential, takes real part.
    """
    shift_hz = params.get('shift_hz', 50.0)
    x = samples.astype(np.float32)
    n = len(x)

    # Hilbert transform via FFT
    X = np.fft.fft(x)
    # Build analytic signal spectrum: zero negative frequencies, double positive
    H = np.zeros(n, dtype=np.float64)
    if n % 2 == 0:
        H[0] = 1.0
        H[1:n // 2] = 2.0
        H[n // 2] = 1.0
    else:
        H[0] = 1.0
        H[1:(n + 1) // 2] = 2.0
    analytic = np.fft.ifft(X * H)

    # Frequency shift: multiply by exp(j * 2pi * shift * n / sr)
    t = np.arange(n, dtype=np.float64)
    shift_carrier = np.exp(1j * 2.0 * np.pi * shift_hz * t / sr)
    shifted = analytic * shift_carrier

    return np.real(shifted).astype(np.float32)


def variants_c009():
    return [
        {'shift_hz': -500.0},
        {'shift_hz': -100.0},
        {'shift_hz': -20.0},
        {'shift_hz': 5.0},
        {'shift_hz': 50.0},
        {'shift_hz': 200.0},
        {'shift_hz': 500.0},
    ]


# ---------------------------------------------------------------------------
# C010 -- Barber Pole Flanger: staggered flangers with fade in/out
# ---------------------------------------------------------------------------

@numba.njit
def _barber_pole_kernel(samples, sr, num_voices, rate_hz, depth_ms, feedback):
    n = len(samples)
    depth_samp = depth_ms * np.float32(0.001) * np.float32(sr)
    # Base delay chosen so we can modulate around it
    base_delay_samp = depth_samp + np.float32(2.0)
    max_delay = int(base_delay_samp + depth_samp + 2)
    buf_size = max_delay + 1
    buf = np.zeros(buf_size, dtype=np.float32)
    out = np.zeros(n, dtype=np.float32)
    write_pos = 0
    two_pi = np.float32(2.0 * np.pi)
    inv_voices = np.float32(1.0) / np.float32(num_voices)
    fb_accum = np.float32(0.0)

    for i in range(n):
        inp = samples[i] + feedback * fb_accum
        buf[write_pos] = inp
        wet = np.float32(0.0)

        for v in range(num_voices):
            # Each voice has a staggered phase
            phase_offset = two_pi * np.float32(v) / np.float32(num_voices)
            # Sawtooth-like ramp for barber pole (triangle mod for smoother)
            raw_phase = rate_hz * np.float32(i) / np.float32(sr) + np.float32(v) / np.float32(num_voices)
            # Use triangle wave for smooth ramp
            frac_phase = raw_phase - np.float32(int(raw_phase))
            if frac_phase < np.float32(0.5):
                mod = np.float32(4.0) * frac_phase - np.float32(1.0)
            else:
                mod = np.float32(3.0) - np.float32(4.0) * frac_phase

            # Fade envelope: fade near ramp edges for seamless overlap
            # Use sine-based crossfade on the fractional phase
            fade = np.float32(np.sin(np.pi * frac_phase))

            delay = base_delay_samp + depth_samp * mod
            read_pos_f = np.float32(write_pos) - delay
            if read_pos_f < 0.0:
                read_pos_f += np.float32(buf_size)
            idx = int(read_pos_f)
            frac = read_pos_f - np.float32(idx)
            idx0 = idx % buf_size
            idx1 = (idx + 1) % buf_size
            voice_out = buf[idx0] * (np.float32(1.0) - frac) + buf[idx1] * frac
            wet += voice_out * fade

        fb_accum = wet * inv_voices
        out[i] = samples[i] * np.float32(0.5) + fb_accum * np.float32(0.5)
        write_pos = (write_pos + 1) % buf_size
    return out


def effect_c010_barber_pole_flanger(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Barber pole flanger: multiple flangers with staggered phases for infinite sweep."""
    num_voices = int(params.get('num_voices', 4))
    rate_hz = np.float32(params.get('rate_hz', 0.15))
    depth_ms = np.float32(params.get('depth_ms', 3.0))
    feedback = np.float32(params.get('feedback', 0.5))
    return _barber_pole_kernel(samples.astype(np.float32), sr, num_voices,
                               rate_hz, depth_ms, feedback)


def variants_c010():
    return [
        {'num_voices': 3, 'rate_hz': 0.05, 'depth_ms': 1.0, 'feedback': 0.3},
        {'num_voices': 4, 'rate_hz': 0.15, 'depth_ms': 3.0, 'feedback': 0.5},
        {'num_voices': 5, 'rate_hz': 0.3, 'depth_ms': 5.0, 'feedback': 0.7},
        {'num_voices': 6, 'rate_hz': 0.5, 'depth_ms': 2.0, 'feedback': -0.5},
        {'num_voices': 3, 'rate_hz': 0.1, 'depth_ms': 4.0, 'feedback': -0.7},
        {'num_voices': 4, 'rate_hz': 0.08, 'depth_ms': 1.5, 'feedback': 0.0},
    ]


# ---------------------------------------------------------------------------
# C011 -- Stereo Auto-Pan: L=cos, R=sin panning with LFO. Returns (N,2)
# ---------------------------------------------------------------------------

@numba.njit
def _autopan_kernel(samples, sr, rate_hz, depth, shape_id):
    """shape_id: 0=sin, 1=tri, 2=square"""
    n = len(samples)
    out = np.zeros((n, 2), dtype=np.float32)
    two_pi = np.float32(2.0 * np.pi)
    period = np.float32(sr) / rate_hz

    for i in range(n):
        phase = two_pi * rate_hz * np.float32(i) / np.float32(sr)

        if shape_id == 0:
            # Sine LFO mapped to pan angle
            lfo = np.float32(np.sin(phase))
        elif shape_id == 1:
            # Triangle
            frac_phase = (np.float32(i) / period)
            frac_phase = frac_phase - np.float32(int(frac_phase))
            if frac_phase < np.float32(0.5):
                lfo = np.float32(4.0) * frac_phase - np.float32(1.0)
            else:
                lfo = np.float32(3.0) - np.float32(4.0) * frac_phase
        else:
            # Square
            frac_phase = (np.float32(i) / period)
            frac_phase = frac_phase - np.float32(int(frac_phase))
            if frac_phase < np.float32(0.5):
                lfo = np.float32(1.0)
            else:
                lfo = np.float32(-1.0)

        # Pan angle: center (0) + depth * lfo * pi/4
        # At center: L and R both ~0.707
        # Full depth: sweeps from hard left to hard right
        angle = np.float32(np.pi * 0.25) * (np.float32(1.0) + depth * lfo)
        gain_l = np.float32(np.cos(angle))
        gain_r = np.float32(np.sin(angle))

        out[i, 0] = samples[i] * gain_l
        out[i, 1] = samples[i] * gain_r
    return out


def effect_c011_stereo_autopan(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Stereo auto-pan with LFO. Returns (N, 2) stereo array."""
    rate_hz = np.float32(params.get('rate_hz', 2.0))
    depth = np.float32(params.get('depth', 0.8))
    lfo_shape = params.get('lfo_shape', 'sin')
    shape_map = {'sin': 0, 'tri': 1, 'square': 2}
    shape_id = shape_map.get(lfo_shape, 0)
    return _autopan_kernel(samples.astype(np.float32), sr, rate_hz, depth, shape_id)


def variants_c011():
    return [
        {'rate_hz': 0.5, 'depth': 0.5, 'lfo_shape': 'sin'},
        {'rate_hz': 2.0, 'depth': 1.0, 'lfo_shape': 'sin'},
        {'rate_hz': 5.0, 'depth': 0.8, 'lfo_shape': 'tri'},
        {'rate_hz': 10.0, 'depth': 0.3, 'lfo_shape': 'sin'},
        {'rate_hz': 1.0, 'depth': 1.0, 'lfo_shape': 'square'},
        {'rate_hz': 0.1, 'depth': 1.0, 'lfo_shape': 'tri'},
    ]


# ---------------------------------------------------------------------------
# C012 -- Doppler Effect: moving source with variable delay, amplitude, filtering
# ---------------------------------------------------------------------------

@numba.njit
def _doppler_kernel(samples, sr, speed_mps, closest_distance_m, path_id):
    """path_id: 0=flyby, 1=orbit, 2=approach"""
    n = len(samples)
    out = np.zeros(n, dtype=np.float32)
    speed_of_sound = np.float32(343.0)
    two_pi = np.float32(2.0 * np.pi)
    sr_f = np.float32(sr)
    duration = np.float32(n) / sr_f

    # Delay buffer for variable-delay read
    max_delay_samp = int(np.float32(100.0) / speed_of_sound * sr_f) + int(sr_f * 0.5) + 2
    buf_size = max_delay_samp + 1
    buf = np.zeros(buf_size, dtype=np.float32)
    write_pos = 0

    # One-pole lowpass state for distance-based filtering
    lp_state = np.float32(0.0)

    for i in range(n):
        t = np.float32(i) / sr_f
        t_norm = t / duration  # 0..1

        if path_id == 0:
            # Flyby: source moves along x-axis, listener at origin
            # Source at (speed * (t - duration/2), closest_distance, 0)
            x_pos = speed_mps * (t - duration * np.float32(0.5))
            distance = np.float32(np.sqrt(x_pos * x_pos + closest_distance_m * closest_distance_m))
        elif path_id == 1:
            # Orbit: circular path around listener
            orbit_radius = closest_distance_m
            angle = two_pi * speed_mps * t / (two_pi * orbit_radius)
            distance = orbit_radius
            # Add slight variation for realism
            distance = orbit_radius + orbit_radius * np.float32(0.1) * np.float32(np.sin(angle * np.float32(3.0)))
        else:
            # Approach: source moves toward listener from far to close
            start_dist = closest_distance_m + speed_mps * duration
            distance = start_dist - speed_mps * t
            if distance < closest_distance_m:
                distance = closest_distance_m

        # Delay based on distance (propagation time)
        delay_sec = distance / speed_of_sound
        delay_samp = delay_sec * sr_f

        # Clamp delay
        if delay_samp < np.float32(1.0):
            delay_samp = np.float32(1.0)
        if delay_samp > np.float32(buf_size - 2):
            delay_samp = np.float32(buf_size - 2)

        buf[write_pos] = samples[i]

        # Read with fractional delay (linear interpolation)
        read_pos_f = np.float32(write_pos) - delay_samp
        if read_pos_f < 0.0:
            read_pos_f += np.float32(buf_size)
        idx = int(read_pos_f)
        frac = read_pos_f - np.float32(idx)
        idx0 = idx % buf_size
        idx1 = (idx + 1) % buf_size
        delayed = buf[idx0] * (np.float32(1.0) - frac) + buf[idx1] * frac

        # Amplitude: inverse distance law (normalized to closest distance)
        amp = closest_distance_m / max(distance, np.float32(0.1))
        if amp > np.float32(2.0):
            amp = np.float32(2.0)

        sig = delayed * amp

        # Distance-based lowpass: further = duller
        # Higher coefficient = more filtering
        dist_ratio = distance / closest_distance_m
        lp_coeff = np.float32(1.0) - np.float32(1.0) / (np.float32(1.0) + dist_ratio * np.float32(0.3))
        lp_state = lp_coeff * lp_state + (np.float32(1.0) - lp_coeff) * sig

        out[i] = lp_state
        write_pos = (write_pos + 1) % buf_size
    return out


def effect_c012_doppler(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Doppler effect: moving source with variable delay, amplitude, and filtering."""
    speed_mps = np.float32(params.get('speed_mps', 30.0))
    closest_distance_m = np.float32(params.get('closest_distance_m', 5.0))
    path = params.get('path', 'flyby')
    path_map = {'flyby': 0, 'orbit': 1, 'approach': 2}
    path_id = path_map.get(path, 0)
    return _doppler_kernel(samples.astype(np.float32), sr, speed_mps,
                           closest_distance_m, path_id)


def variants_c012():
    return [
        {'speed_mps': 10.0, 'closest_distance_m': 3.0, 'path': 'flyby'},
        {'speed_mps': 50.0, 'closest_distance_m': 5.0, 'path': 'flyby'},
        {'speed_mps': 100.0, 'closest_distance_m': 10.0, 'path': 'flyby'},
        {'speed_mps': 20.0, 'closest_distance_m': 8.0, 'path': 'orbit'},
        {'speed_mps': 60.0, 'closest_distance_m': 3.0, 'path': 'orbit'},
        {'speed_mps': 30.0, 'closest_distance_m': 20.0, 'path': 'approach'},
        {'speed_mps': 80.0, 'closest_distance_m': 1.0, 'path': 'approach'},
        {'speed_mps': 5.0, 'closest_distance_m': 2.0, 'path': 'flyby'},
    ]
