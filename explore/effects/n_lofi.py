"""N-series: Lo-fi effects (N001-N007).

Vinyl crackle, tape hiss, tape wow/flutter, telephone, radio tuning,
underwater, and AM radio simulation.
"""
import numpy as np
import numba


# ---------------------------------------------------------------------------
# N001 -- Vinyl Crackle Overlay
# ---------------------------------------------------------------------------

@numba.njit
def _vinyl_crackle(samples, sr, density, amplitude, seed):
    """Add sparse random impulses with exponential decay to signal."""
    n = len(samples)
    out = np.empty(n, dtype=np.float32)
    # Average spacing between crackles in samples
    avg_spacing = sr / density
    # Exponential decay time constant: crackle lasts ~1-3ms
    decay_rate = np.float32(1.0 / (np.float32(0.002) * np.float32(sr)))
    # Simple LCG RNG state
    rng = np.uint64(seed)
    crackle_val = np.float32(0.0)
    next_crackle = np.int64(0)

    for i in range(n):
        # Check if time to spawn a new crackle
        if i >= next_crackle:
            # Generate random sign and slight amplitude variation
            rng = rng * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
            rand_val = np.float32(np.int64((rng >> np.uint64(33))) & np.int64(0x7FFFFFFF)) / np.float32(2147483647.0)
            sign = np.float32(1.0) if rand_val > np.float32(0.5) else np.float32(-1.0)
            # Amplitude variation: 0.5x to 1.5x
            rng = rng * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
            amp_var = np.float32(0.5) + np.float32(np.int64((rng >> np.uint64(33))) & np.int64(0x7FFFFFFF)) / np.float32(2147483647.0)
            crackle_val = sign * amplitude * amp_var
            # Next crackle at random interval (Poisson-like)
            rng = rng * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
            spacing_rand = np.float32(np.int64((rng >> np.uint64(33))) & np.int64(0x7FFFFFFF)) / np.float32(2147483647.0)
            # Exponential distribution approximation: -ln(u) * mean
            if spacing_rand < np.float32(0.001):
                spacing_rand = np.float32(0.001)
            next_crackle = i + np.int64(-np.log(spacing_rand) * avg_spacing)
            if next_crackle <= i:
                next_crackle = i + 1
        else:
            # Exponential decay of current crackle
            crackle_val = crackle_val * (np.float32(1.0) - decay_rate)

        out[i] = samples[i] + crackle_val
    return out


def effect_n001_vinyl_crackle(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Vinyl crackle overlay: sparse random impulses with exponential decay.

    Params:
        crackle_density:   crackles per second  [5, 100]     (default 30)
        crackle_amplitude: peak amplitude        [0.01, 0.1]  (default 0.03)
    """
    density = np.float32(params.get('crackle_density', 30.0))
    amplitude = np.float32(params.get('crackle_amplitude', 0.03))
    seed = np.uint64(params.get('seed', 42))
    return _vinyl_crackle(samples.astype(np.float32), sr, density, amplitude, seed)


def variants_n001():
    return [
        {'crackle_density': 5, 'crackle_amplitude': 0.02},     # rare, quiet pops
        {'crackle_density': 15, 'crackle_amplitude': 0.03},    # gentle vintage
        {'crackle_density': 30, 'crackle_amplitude': 0.03},    # standard vinyl character
        {'crackle_density': 60, 'crackle_amplitude': 0.05},    # well-worn record
        {'crackle_density': 100, 'crackle_amplitude': 0.08},   # heavily damaged vinyl
        {'crackle_density': 80, 'crackle_amplitude': 0.01},    # dense but subtle texture
    ]


# ---------------------------------------------------------------------------
# N002 -- Tape Hiss
# ---------------------------------------------------------------------------

@numba.njit
def _tape_hiss_mix(samples, hiss, level_linear):
    """Mix bandpass-filtered hiss noise with signal."""
    n = len(samples)
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        out[i] = samples[i] + hiss[i] * level_linear
    return out


@numba.njit
def _biquad_filter(samples, b0, b1, b2, a1, a2):
    """Direct Form II Transposed biquad filter."""
    n = len(samples)
    out = np.empty(n, dtype=np.float32)
    z1 = np.float32(0.0)
    z2 = np.float32(0.0)
    for i in range(n):
        x = samples[i]
        y = b0 * x + z1
        z1 = b1 * x - a1 * y + z2
        z2 = b2 * x - a2 * y
        out[i] = y
    return out


def _compute_biquad_bpf(freq_hz, sr, Q):
    """Compute bandpass biquad coefficients (Audio EQ Cookbook)."""
    w0 = 2.0 * np.pi * freq_hz / sr
    alpha = np.sin(w0) / (2.0 * Q)
    b0 = alpha
    b1 = 0.0
    b2 = -alpha
    a0 = 1.0 + alpha
    a1 = -2.0 * np.cos(w0)
    a2 = 1.0 - alpha
    inv_a0 = 1.0 / a0
    return (np.float32(b0 * inv_a0), np.float32(b1 * inv_a0),
            np.float32(b2 * inv_a0), np.float32(a1 * inv_a0),
            np.float32(a2 * inv_a0))


def _compute_biquad_lpf(freq_hz, sr, Q):
    """Compute lowpass biquad coefficients (Audio EQ Cookbook)."""
    freq_hz = float(np.clip(freq_hz, 20.0, sr * 0.499))
    w0 = 2.0 * np.pi * freq_hz / sr
    alpha = np.sin(w0) / (2.0 * Q)
    b0 = (1.0 - np.cos(w0)) / 2.0
    b1 = 1.0 - np.cos(w0)
    b2 = (1.0 - np.cos(w0)) / 2.0
    a0 = 1.0 + alpha
    a1 = -2.0 * np.cos(w0)
    a2 = 1.0 - alpha
    inv_a0 = 1.0 / a0
    return (np.float32(b0 * inv_a0), np.float32(b1 * inv_a0),
            np.float32(b2 * inv_a0), np.float32(a1 * inv_a0),
            np.float32(a2 * inv_a0))


def _compute_biquad_hpf(freq_hz, sr, Q):
    """Compute highpass biquad coefficients (Audio EQ Cookbook)."""
    freq_hz = float(np.clip(freq_hz, 20.0, sr * 0.499))
    w0 = 2.0 * np.pi * freq_hz / sr
    alpha = np.sin(w0) / (2.0 * Q)
    b0 = (1.0 + np.cos(w0)) / 2.0
    b1 = -(1.0 + np.cos(w0))
    b2 = (1.0 + np.cos(w0)) / 2.0
    a0 = 1.0 + alpha
    a1 = -2.0 * np.cos(w0)
    a2 = 1.0 - alpha
    inv_a0 = 1.0 / a0
    return (np.float32(b0 * inv_a0), np.float32(b1 * inv_a0),
            np.float32(b2 * inv_a0), np.float32(a1 * inv_a0),
            np.float32(a2 * inv_a0))


def effect_n002_tape_hiss(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Tape hiss: bandpass-filtered noise (1kHz-8kHz) with level modulation.

    Params:
        hiss_level_db: noise level in dB   [-40, -15]    (default -25)
        color:         'bright' or 'warm'                 (default 'warm')
    """
    hiss_level_db = float(np.clip(params.get('hiss_level_db', -25.0), -40.0, -15.0))
    color = params.get('color', 'warm')
    level_linear = np.float32(10.0 ** (hiss_level_db / 20.0))

    n = len(samples)
    rng = np.random.default_rng(params.get('seed', 42))
    noise = rng.standard_normal(n).astype(np.float32)

    if color == 'bright':
        # Bright: highpass at 2kHz, lowpass at 12kHz
        hp_b0, hp_b1, hp_b2, hp_a1, hp_a2 = _compute_biquad_hpf(2000.0, sr, 0.707)
        lp_b0, lp_b1, lp_b2, lp_a1, lp_a2 = _compute_biquad_lpf(12000.0, sr, 0.707)
    else:
        # Warm: highpass at 800Hz, lowpass at 6kHz
        hp_b0, hp_b1, hp_b2, hp_a1, hp_a2 = _compute_biquad_hpf(800.0, sr, 0.707)
        lp_b0, lp_b1, lp_b2, lp_a1, lp_a2 = _compute_biquad_lpf(6000.0, sr, 0.707)

    # Apply highpass then lowpass to shape noise
    filtered = _biquad_filter(noise, hp_b0, hp_b1, hp_b2, hp_a1, hp_a2)
    filtered = _biquad_filter(filtered, lp_b0, lp_b1, lp_b2, lp_a1, lp_a2)

    return _tape_hiss_mix(samples.astype(np.float32), filtered, level_linear)


def variants_n002():
    return [
        {'hiss_level_db': -35, 'color': 'warm'},     # barely perceptible warmth
        {'hiss_level_db': -25, 'color': 'warm'},      # classic warm tape hiss
        {'hiss_level_db': -20, 'color': 'warm'},      # noticeable warm hiss
        {'hiss_level_db': -30, 'color': 'bright'},    # subtle bright tape character
        {'hiss_level_db': -20, 'color': 'bright'},    # prominent bright hiss
        {'hiss_level_db': -15, 'color': 'warm'},      # heavy worn tape noise
    ]


# ---------------------------------------------------------------------------
# N003 -- Tape Wow and Flutter
# ---------------------------------------------------------------------------

@numba.njit
def _tape_wow_flutter(samples, sr, wow_rate, wow_depth, flutter_rate, flutter_depth):
    """Modulated playback speed via variable-rate delay line.

    Wow: slow pitch drift. Flutter: faster pitch irregularity.
    Combined modulation drives a fractional delay read offset.
    """
    n = len(samples)
    # Maximum additional delay in samples from modulation
    max_mod_samples = (wow_depth + flutter_depth) * np.float32(sr)
    # Base delay so we can modulate around it
    base_delay = max_mod_samples + np.float32(2.0)
    buf_size = int(base_delay + max_mod_samples + 4)
    buf = np.zeros(buf_size, dtype=np.float32)
    out = np.empty(n, dtype=np.float32)
    write_pos = 0
    two_pi = np.float32(2.0 * np.pi)

    for i in range(n):
        buf[write_pos] = samples[i]
        t = np.float32(i) / np.float32(sr)
        # Wow modulation (slow)
        wow_mod = wow_depth * np.float32(np.sin(two_pi * wow_rate * t))
        # Flutter modulation (faster)
        flutter_mod = flutter_depth * np.float32(np.sin(two_pi * flutter_rate * t))
        # Total modulation in samples
        mod_samples = (wow_mod + flutter_mod) * np.float32(sr)
        delay = base_delay + mod_samples

        # Clamp delay
        if delay < np.float32(1.0):
            delay = np.float32(1.0)
        if delay > np.float32(buf_size - 2):
            delay = np.float32(buf_size - 2)

        # Fractional delay read with linear interpolation
        read_pos_f = np.float32(write_pos) - delay
        if read_pos_f < np.float32(0.0):
            read_pos_f += np.float32(buf_size)
        idx = int(read_pos_f)
        frac = read_pos_f - np.float32(idx)
        idx0 = idx % buf_size
        idx1 = (idx + 1) % buf_size
        out[i] = buf[idx0] * (np.float32(1.0) - frac) + buf[idx1] * frac

        write_pos = (write_pos + 1) % buf_size
    return out


def effect_n003_tape_wow_flutter(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Tape wow and flutter: modulated playback speed via LFOs.

    Params:
        wow_rate:       wow LFO rate Hz       [0.5, 3.0]     (default 1.5)
        wow_depth:      wow depth in seconds  [0.001, 0.01]  (default 0.003)
        flutter_rate:   flutter LFO rate Hz   [5, 20]        (default 10)
        flutter_depth:  flutter depth seconds [0.0001, 0.002] (default 0.0005)
    """
    wow_rate = np.float32(params.get('wow_rate', 1.5))
    wow_depth = np.float32(params.get('wow_depth', 0.003))
    flutter_rate = np.float32(params.get('flutter_rate', 10.0))
    flutter_depth = np.float32(params.get('flutter_depth', 0.0005))
    return _tape_wow_flutter(samples.astype(np.float32), sr,
                             wow_rate, wow_depth, flutter_rate, flutter_depth)


def variants_n003():
    return [
        {'wow_rate': 0.5, 'wow_depth': 0.001, 'flutter_rate': 6, 'flutter_depth': 0.0001},   # subtle, well-maintained tape
        {'wow_rate': 1.0, 'wow_depth': 0.003, 'flutter_rate': 10, 'flutter_depth': 0.0003},   # gentle cassette wobble
        {'wow_rate': 1.5, 'wow_depth': 0.003, 'flutter_rate': 10, 'flutter_depth': 0.0005},   # standard tape character
        {'wow_rate': 2.5, 'wow_depth': 0.006, 'flutter_rate': 15, 'flutter_depth': 0.001},    # worn-out mechanism
        {'wow_rate': 3.0, 'wow_depth': 0.01, 'flutter_rate': 20, 'flutter_depth': 0.002},     # broken deck, extreme wobble
        {'wow_rate': 0.7, 'wow_depth': 0.008, 'flutter_rate': 5, 'flutter_depth': 0.0002},    # slow deep wow, minimal flutter
        {'wow_rate': 2.0, 'wow_depth': 0.002, 'flutter_rate': 18, 'flutter_depth': 0.0015},   # dominant flutter, rapid shimmer
    ]


# ---------------------------------------------------------------------------
# N004 -- Telephone Effect
# ---------------------------------------------------------------------------

@numba.njit
def _telephone_distortion(samples, distortion_amount, noise_level, seed):
    """Apply subtle distortion and noise to bandpass-filtered signal."""
    n = len(samples)
    out = np.empty(n, dtype=np.float32)
    rng = np.uint64(seed)
    for i in range(n):
        x = samples[i]
        # Subtle distortion via soft clipping
        if distortion_amount > np.float32(0.0):
            drive = np.float32(1.0) + distortion_amount * np.float32(10.0)
            x = np.float32(np.tanh(x * drive)) / np.float32(np.tanh(drive))
        # Add noise
        if noise_level > np.float32(0.0):
            rng = rng * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
            noise_val = np.float32(np.int64((rng >> np.uint64(33))) & np.int64(0x7FFFFFFF)) / np.float32(2147483647.0)
            noise_val = (noise_val - np.float32(0.5)) * np.float32(2.0) * noise_level
            x = x + noise_val
        out[i] = x
    return out


def effect_n004_telephone(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Telephone effect: bandpass 300-3400Hz + subtle distortion + noise.

    Params:
        low_cut:           highpass cutoff Hz     [200, 500]   (default 300)
        high_cut:          lowpass cutoff Hz      [2500, 4000] (default 3400)
        distortion_amount: soft clipping amount   [0.0, 0.5]   (default 0.2)
        noise_level:       noise amplitude        [0.0, 0.05]  (default 0.01)
    """
    low_cut = float(np.clip(params.get('low_cut', 300.0), 200.0, 500.0))
    high_cut = float(np.clip(params.get('high_cut', 3400.0), 2500.0, min(4000.0, sr * 0.499)))
    distortion_amount = np.float32(np.clip(params.get('distortion_amount', 0.2), 0.0, 0.5))
    noise_level = np.float32(np.clip(params.get('noise_level', 0.01), 0.0, 0.05))

    inp = samples.astype(np.float32)

    # Highpass filter (remove low frequencies)
    hp_b0, hp_b1, hp_b2, hp_a1, hp_a2 = _compute_biquad_hpf(low_cut, sr, 0.707)
    filtered = _biquad_filter(inp, hp_b0, hp_b1, hp_b2, hp_a1, hp_a2)

    # Lowpass filter (remove high frequencies)
    lp_b0, lp_b1, lp_b2, lp_a1, lp_a2 = _compute_biquad_lpf(high_cut, sr, 0.707)
    filtered = _biquad_filter(filtered, lp_b0, lp_b1, lp_b2, lp_a1, lp_a2)

    # Apply second pass of each for steeper rolloff
    filtered = _biquad_filter(filtered, hp_b0, hp_b1, hp_b2, hp_a1, hp_a2)
    filtered = _biquad_filter(filtered, lp_b0, lp_b1, lp_b2, lp_a1, lp_a2)

    # Add distortion and noise
    seed = np.uint64(params.get('seed', 42))
    return _telephone_distortion(filtered, distortion_amount, noise_level, seed)


def variants_n004():
    return [
        {'low_cut': 300, 'high_cut': 3400, 'distortion_amount': 0.1, 'noise_level': 0.005},   # clean landline
        {'low_cut': 300, 'high_cut': 3400, 'distortion_amount': 0.2, 'noise_level': 0.01},     # standard telephone
        {'low_cut': 400, 'high_cut': 2800, 'distortion_amount': 0.3, 'noise_level': 0.02},     # poor connection
        {'low_cut': 500, 'high_cut': 2500, 'distortion_amount': 0.4, 'noise_level': 0.03},     # very narrow, distorted
        {'low_cut': 200, 'high_cut': 4000, 'distortion_amount': 0.0, 'noise_level': 0.0},      # wideband ISDN, clean
        {'low_cut': 350, 'high_cut': 3000, 'distortion_amount': 0.5, 'noise_level': 0.05},     # bad mobile connection
    ]


# ---------------------------------------------------------------------------
# N005 -- Radio Tuning Effect
# ---------------------------------------------------------------------------

@numba.njit
def _radio_tuning_kernel(samples, sr, sweep_rate, noise_level, signal_clarity, seed):
    """Sweeping bandpass + AM modulation + noise bursts.

    Simulates turning a radio dial: the signal fades in and out
    through sweeping bandpass, with static noise between stations.
    """
    n = len(samples)
    out = np.empty(n, dtype=np.float32)
    two_pi = np.float32(2.0 * np.pi)
    rng = np.uint64(seed)
    pi_over_sr = np.float32(np.pi) / np.float32(sr)

    # SVF state for sweeping bandpass
    lp = np.float32(0.0)
    bp = np.float32(0.0)
    q_inv = np.float32(1.0 / 3.0)  # moderate Q for bandpass

    for i in range(n):
        t = np.float32(i) / np.float32(sr)

        # Sweep the bandpass center frequency (200Hz - 6000Hz in log space)
        sweep_phase = np.float32(np.sin(two_pi * sweep_rate * t))
        # Map sine [-1, 1] to log frequency space
        log_min = np.float32(np.log(200.0))
        log_max = np.float32(np.log(6000.0))
        log_f = np.float32(0.5) * (log_min + log_max) + np.float32(0.5) * (log_max - log_min) * sweep_phase
        cutoff = np.float32(np.exp(log_f))

        f_coeff = np.float32(2.0) * np.float32(np.sin(cutoff * pi_over_sr))
        if f_coeff > np.float32(1.8):
            f_coeff = np.float32(1.8)

        # SVF step
        x = samples[i]
        hp = x - lp - q_inv * bp
        bp = bp + f_coeff * hp
        lp = lp + f_coeff * bp

        # Clamp
        if bp > np.float32(10.0):
            bp = np.float32(10.0)
        elif bp < np.float32(-10.0):
            bp = np.float32(-10.0)
        if lp > np.float32(10.0):
            lp = np.float32(10.0)
        elif lp < np.float32(-10.0):
            lp = np.float32(-10.0)

        # AM modulation: signal clarity varies with sweep position
        # When sweep is near center (sine near 0), clarity is highest
        clarity_mod = signal_clarity * (np.float32(1.0) - np.float32(0.5) * abs(sweep_phase))

        # Generate noise
        rng = rng * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
        noise_val = np.float32(np.int64((rng >> np.uint64(33))) & np.int64(0x7FFFFFFF)) / np.float32(2147483647.0)
        noise_val = (noise_val - np.float32(0.5)) * np.float32(2.0)

        # Mix: filtered signal * clarity + noise * (1 - clarity)
        out[i] = bp * clarity_mod + noise_val * noise_level * (np.float32(1.0) - clarity_mod)

    return out


def effect_n005_radio_tuning(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Radio tuning: sweeping bandpass + AM modulation + noise bursts.

    Params:
        sweep_rate:      sweep speed Hz         [0.1, 2.0]  (default 0.5)
        noise_level:     static noise level     [0.01, 0.1] (default 0.05)
        signal_clarity:  signal clarity factor   [0.3, 1.0]  (default 0.7)
    """
    sweep_rate = np.float32(np.clip(params.get('sweep_rate', 0.5), 0.1, 2.0))
    noise_level = np.float32(np.clip(params.get('noise_level', 0.05), 0.01, 0.1))
    signal_clarity = np.float32(np.clip(params.get('signal_clarity', 0.7), 0.3, 1.0))
    seed = np.uint64(params.get('seed', 42))
    return _radio_tuning_kernel(samples.astype(np.float32), sr,
                                sweep_rate, noise_level, signal_clarity, seed)


def variants_n005():
    return [
        {'sweep_rate': 0.1, 'noise_level': 0.02, 'signal_clarity': 0.9},    # slow scan, mostly clear
        {'sweep_rate': 0.3, 'noise_level': 0.05, 'signal_clarity': 0.7},    # gentle dial turning
        {'sweep_rate': 0.5, 'noise_level': 0.05, 'signal_clarity': 0.7},    # standard radio tuning
        {'sweep_rate': 1.0, 'noise_level': 0.08, 'signal_clarity': 0.5},    # frantic channel surfing
        {'sweep_rate': 2.0, 'noise_level': 0.1, 'signal_clarity': 0.3},     # chaotic dial spinning
        {'sweep_rate': 0.2, 'noise_level': 0.03, 'signal_clarity': 1.0},    # almost locked on station
    ]


# ---------------------------------------------------------------------------
# N006 -- Underwater Effect
# ---------------------------------------------------------------------------

@numba.njit
def _underwater_kernel(samples, sr, depth, bubble_density, chorus_rate, seed):
    """Strong lowpass + subtle chorus + pitch wobble + random bubble pops.

    Uses one-pole lowpass for the main filtering, modulated delay for chorus,
    and random impulses for bubble sounds.
    """
    n = len(samples)
    out = np.empty(n, dtype=np.float32)
    two_pi = np.float32(2.0 * np.pi)

    # One-pole lowpass coefficient: deeper = lower cutoff
    # Map depth [0.1, 1.0] to cutoff [2000Hz, 200Hz]
    cutoff_hz = np.float32(2000.0) - depth * np.float32(1800.0)
    if cutoff_hz < np.float32(100.0):
        cutoff_hz = np.float32(100.0)
    # One-pole coefficient: a = exp(-2*pi*fc/sr)
    lp_coeff = np.float32(np.exp(-two_pi * cutoff_hz / np.float32(sr)))
    lp_state = np.float32(0.0)

    # Second one-pole for steeper rolloff
    lp_state2 = np.float32(0.0)

    # Chorus delay line for underwater warble
    chorus_depth_ms = np.float32(3.0) + depth * np.float32(5.0)
    chorus_depth_samp = chorus_depth_ms * np.float32(0.001) * np.float32(sr)
    base_delay_samp = chorus_depth_samp + np.float32(2.0)
    buf_size = int(base_delay_samp + chorus_depth_samp + 4)
    buf = np.zeros(buf_size, dtype=np.float32)
    write_pos = 0

    # Pitch wobble LFO: slow random pitch variation
    wobble_rate = np.float32(0.3) + depth * np.float32(0.5)

    # Bubble RNG
    rng = np.uint64(seed)
    bubble_avg_spacing = np.float32(sr) / max(np.float32(bubble_density), np.float32(0.01))
    next_bubble = np.int64(0)
    bubble_val = np.float32(0.0)
    bubble_decay = np.float32(1.0 / (np.float32(0.005) * np.float32(sr)))  # 5ms decay

    for i in range(n):
        x = samples[i]

        # Two-pole lowpass (cascaded one-pole)
        lp_state = lp_coeff * lp_state + (np.float32(1.0) - lp_coeff) * x
        lp_state2 = lp_coeff * lp_state2 + (np.float32(1.0) - lp_coeff) * lp_state
        filtered = lp_state2

        # Write to chorus buffer
        buf[write_pos] = filtered
        t = np.float32(i) / np.float32(sr)

        # Chorus modulation with wobble
        mod = np.float32(np.sin(two_pi * chorus_rate * t))
        wobble = np.float32(np.sin(two_pi * wobble_rate * t)) * np.float32(0.3)
        delay = base_delay_samp + chorus_depth_samp * (mod + wobble)

        if delay < np.float32(1.0):
            delay = np.float32(1.0)
        if delay > np.float32(buf_size - 2):
            delay = np.float32(buf_size - 2)

        read_pos_f = np.float32(write_pos) - delay
        if read_pos_f < np.float32(0.0):
            read_pos_f += np.float32(buf_size)
        idx = int(read_pos_f)
        frac = read_pos_f - np.float32(idx)
        idx0 = idx % buf_size
        idx1 = (idx + 1) % buf_size
        chorus_out = buf[idx0] * (np.float32(1.0) - frac) + buf[idx1] * frac

        # Mix dry lowpassed with chorus wet
        mixed = filtered * np.float32(0.6) + chorus_out * np.float32(0.4)

        # Bubble pops
        if bubble_density > np.float32(0.0):
            if i >= next_bubble:
                rng = rng * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
                rand_val = np.float32(np.int64((rng >> np.uint64(33))) & np.int64(0x7FFFFFFF)) / np.float32(2147483647.0)
                bubble_val = (rand_val - np.float32(0.5)) * np.float32(0.1) * depth
                rng = rng * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
                spacing = np.float32(np.int64((rng >> np.uint64(33))) & np.int64(0x7FFFFFFF)) / np.float32(2147483647.0)
                if spacing < np.float32(0.001):
                    spacing = np.float32(0.001)
                next_bubble = i + np.int64(-np.log(spacing) * bubble_avg_spacing)
                if next_bubble <= i:
                    next_bubble = i + 1
            else:
                bubble_val = bubble_val * (np.float32(1.0) - bubble_decay)
            mixed = mixed + bubble_val

        out[i] = mixed
        write_pos = (write_pos + 1) % buf_size

    return out


def effect_n006_underwater(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Underwater effect: strong lowpass + subtle chorus + pitch wobble + bubbles.

    Params:
        depth:          depth factor            [0.1, 1.0]   (default 0.5)
        bubble_density: bubbles per second      [0, 20]      (default 5)
        chorus_rate:    chorus LFO rate Hz      [0.3, 2.0]   (default 0.5)
    """
    depth = np.float32(np.clip(params.get('depth', 0.5), 0.1, 1.0))
    bubble_density = np.float32(np.clip(params.get('bubble_density', 5.0), 0.0, 20.0))
    chorus_rate = np.float32(np.clip(params.get('chorus_rate', 0.5), 0.3, 2.0))
    seed = np.uint64(params.get('seed', 42))
    return _underwater_kernel(samples.astype(np.float32), sr,
                              depth, bubble_density, chorus_rate, seed)


def variants_n006():
    return [
        {'depth': 0.2, 'bubble_density': 2, 'chorus_rate': 0.3},     # shallow pool, subtle muffling
        {'depth': 0.4, 'bubble_density': 5, 'chorus_rate': 0.5},     # swimming pool depth
        {'depth': 0.5, 'bubble_density': 5, 'chorus_rate': 0.5},     # standard underwater
        {'depth': 0.7, 'bubble_density': 10, 'chorus_rate': 0.8},    # deep dive, many bubbles
        {'depth': 1.0, 'bubble_density': 15, 'chorus_rate': 1.5},    # deep ocean, heavy warble
        {'depth': 0.3, 'bubble_density': 0, 'chorus_rate': 0.4},     # muffled, no bubbles (tank)
        {'depth': 0.8, 'bubble_density': 20, 'chorus_rate': 2.0},    # agitated water, dense bubbles
    ]


# ---------------------------------------------------------------------------
# N007 -- AM Radio Effect
# ---------------------------------------------------------------------------

@numba.njit
def _am_radio_kernel(samples, sr, mod_index, noise_level, hum_level, seed):
    """AM modulation + bandlimit 5kHz + crackle + 60Hz hum.

    Simulates vintage AM radio reception with characteristic artifacts.
    """
    n = len(samples)
    out = np.empty(n, dtype=np.float32)
    two_pi = np.float32(2.0 * np.pi)
    rng = np.uint64(seed)

    # One-pole lowpass for 5kHz bandwidth limiting (cascaded for steeper rolloff)
    lp_coeff = np.float32(np.exp(-two_pi * np.float32(5000.0) / np.float32(sr)))
    lp_state1 = np.float32(0.0)
    lp_state2 = np.float32(0.0)

    # AM carrier frequency (standard broadcast ~540-1700kHz, but we simulate
    # the audible artifact of imperfect demodulation at a lower rate)
    am_carrier_hz = np.float32(1000.0)  # audible artifact frequency

    # Crackle parameters
    crackle_spacing = np.float32(sr) / np.float32(15.0)  # ~15 crackles/sec
    next_crackle = np.int64(0)
    crackle_val = np.float32(0.0)
    crackle_decay = np.float32(1.0 / (np.float32(0.001) * np.float32(sr)))

    for i in range(n):
        x = samples[i]
        t = np.float32(i) / np.float32(sr)

        # Bandlimit: two-pole lowpass at 5kHz
        lp_state1 = lp_coeff * lp_state1 + (np.float32(1.0) - lp_coeff) * x
        lp_state2 = lp_coeff * lp_state2 + (np.float32(1.0) - lp_coeff) * lp_state1
        bandlimited = lp_state2

        # AM modulation artifact: slight amplitude modulation at carrier-related frequency
        # Imperfect demodulation leaves residual carrier modulation
        am_mod = np.float32(1.0) - mod_index * np.float32(0.1) * (np.float32(1.0) - np.float32(
            np.cos(two_pi * am_carrier_hz * t)))
        modulated = bandlimited * am_mod

        # 60Hz power line hum
        hum = hum_level * np.float32(np.sin(two_pi * np.float32(60.0) * t))
        # Add 2nd harmonic (120Hz) for realism
        hum += hum_level * np.float32(0.3) * np.float32(np.sin(two_pi * np.float32(120.0) * t))

        # Background noise (static)
        rng = rng * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
        noise_val = np.float32(np.int64((rng >> np.uint64(33))) & np.int64(0x7FFFFFFF)) / np.float32(2147483647.0)
        noise_val = (noise_val - np.float32(0.5)) * np.float32(2.0) * noise_level

        # Crackle pops
        if i >= next_crackle:
            rng = rng * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
            sign_val = np.float32(np.int64((rng >> np.uint64(33))) & np.int64(0x7FFFFFFF)) / np.float32(2147483647.0)
            sign = np.float32(1.0) if sign_val > np.float32(0.5) else np.float32(-1.0)
            crackle_val = sign * noise_level * np.float32(2.0)
            rng = rng * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
            spacing = np.float32(np.int64((rng >> np.uint64(33))) & np.int64(0x7FFFFFFF)) / np.float32(2147483647.0)
            if spacing < np.float32(0.001):
                spacing = np.float32(0.001)
            next_crackle = i + np.int64(-np.log(spacing) * crackle_spacing)
            if next_crackle <= i:
                next_crackle = i + 1
        else:
            crackle_val = crackle_val * (np.float32(1.0) - crackle_decay)

        out[i] = modulated + hum + noise_val + crackle_val

    return out


def effect_n007_am_radio(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """AM radio effect: AM modulation + bandlimit 5kHz + crackle + 60Hz hum.

    Params:
        modulation_index: AM modulation depth   [0.3, 1.0]   (default 0.7)
        noise_level:      static noise level    [0.01, 0.1]  (default 0.03)
        hum_level:        60Hz hum amplitude    [0.0, 0.05]  (default 0.02)
    """
    mod_index = np.float32(np.clip(params.get('modulation_index', 0.7), 0.3, 1.0))
    noise_level = np.float32(np.clip(params.get('noise_level', 0.03), 0.01, 0.1))
    hum_level = np.float32(np.clip(params.get('hum_level', 0.02), 0.0, 0.05))
    seed = np.uint64(params.get('seed', 42))
    return _am_radio_kernel(samples.astype(np.float32), sr,
                            mod_index, noise_level, hum_level, seed)


def variants_n007():
    return [
        {'modulation_index': 0.3, 'noise_level': 0.01, 'hum_level': 0.0},     # clean AM reception
        {'modulation_index': 0.5, 'noise_level': 0.02, 'hum_level': 0.01},    # decent AM station
        {'modulation_index': 0.7, 'noise_level': 0.03, 'hum_level': 0.02},    # standard AM radio
        {'modulation_index': 0.8, 'noise_level': 0.06, 'hum_level': 0.03},    # weak signal reception
        {'modulation_index': 1.0, 'noise_level': 0.1, 'hum_level': 0.05},     # barely tuned in, heavy artifacts
        {'modulation_index': 0.5, 'noise_level': 0.04, 'hum_level': 0.04},    # ground loop hum dominant
    ]
