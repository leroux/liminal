"""F-series: Dynamics effects (F001-F015).

F001 — Compressor
F002 — Expander/Gate
F003 — Transient Shaper
F004 — Multiband Dynamics (3-band)
F005 — Sidechain Ducker
F006 — RMS Compressor
F007 — Feedback Compressor
F008 — Opto Compressor
F009 — FET Compressor
F010 — Soft-Knee Compressor
F011 — Parallel (NY) Compressor
F012 — Upward Compressor
F013 — Program-Dependent Compressor
F014 — Lookahead Limiter
F015 — Spectral Compressor
"""
import numpy as np
import numba


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ms_to_coeff(ms, sr):
    """Convert milliseconds to one-pole filter coefficient."""
    if ms <= 0:
        return np.float32(0.0)
    return np.float32(np.exp(-1.0 / (ms * 0.001 * sr)))


@numba.njit
def _envelope_follower(samples, attack_coeff, release_coeff):
    """Peak envelope follower with separate attack/release coefficients."""
    n = len(samples)
    env = np.empty(n, dtype=np.float32)
    prev = np.float32(0.0)
    for i in range(n):
        inp = np.float32(abs(samples[i]))
        if inp > prev:
            prev = attack_coeff * prev + (np.float32(1.0) - attack_coeff) * inp
        else:
            prev = release_coeff * prev + (np.float32(1.0) - release_coeff) * inp
        env[i] = prev
    return env


@numba.njit
def _biquad_process(samples, b0, b1, b2, a1, a2):
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


def _butter2_lpf_coeffs(freq_hz, sr):
    """2nd-order Butterworth lowpass biquad coefficients."""
    w0 = 2.0 * np.pi * freq_hz / sr
    Q = 0.7071067811865476  # 1/sqrt(2)
    alpha = np.sin(w0) / (2.0 * Q)
    cos_w0 = np.cos(w0)
    b0 = (1.0 - cos_w0) / 2.0
    b1 = 1.0 - cos_w0
    b2 = (1.0 - cos_w0) / 2.0
    a0 = 1.0 + alpha
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha
    return (np.float32(b0 / a0), np.float32(b1 / a0), np.float32(b2 / a0),
            np.float32(a1 / a0), np.float32(a2 / a0))


def _butter2_hpf_coeffs(freq_hz, sr):
    """2nd-order Butterworth highpass biquad coefficients."""
    w0 = 2.0 * np.pi * freq_hz / sr
    Q = 0.7071067811865476
    alpha = np.sin(w0) / (2.0 * Q)
    cos_w0 = np.cos(w0)
    b0 = (1.0 + cos_w0) / 2.0
    b1 = -(1.0 + cos_w0)
    b2 = (1.0 + cos_w0) / 2.0
    a0 = 1.0 + alpha
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha
    return (np.float32(b0 / a0), np.float32(b1 / a0), np.float32(b2 / a0),
            np.float32(a1 / a0), np.float32(a2 / a0))


# ===================================================================
# F001 — Compressor
# ===================================================================

@numba.njit
def _compress_kernel(samples, env, threshold_lin, ratio, makeup_gain):
    """Apply gain reduction based on envelope and threshold."""
    n = len(samples)
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        level = env[i]
        if level > threshold_lin and level > np.float32(1e-12):
            over_db = np.float32(20.0) * np.float32(np.log10(level / threshold_lin))
            reduction_db = over_db * (np.float32(1.0) - np.float32(1.0) / ratio)
            gain = np.float32(10.0 ** (-reduction_db / np.float32(20.0)))
        else:
            gain = np.float32(1.0)
        out[i] = samples[i] * gain * makeup_gain
    return out


def effect_f001_compressor(samples: np.ndarray, sr: int, *,
                           threshold_db: float = -20.0,
                           ratio: float = 4.0,
                           attack_ms: float = 10.0,
                           release_ms: float = 100.0) -> np.ndarray:
    """Compressor: envelope-following gain reduction above threshold.

    Parameters
    ----------
    threshold_db : float [-40, 0]
        Level above which compression begins.
    ratio : float [2, 20]
        Compression ratio (e.g. 4:1).
    attack_ms : float [0.1, 100]
        How quickly the compressor reacts to rising levels.
    release_ms : float [10, 1000]
        How quickly the compressor releases after level drops.
    """
    samples = samples.astype(np.float32)
    attack_coeff = _ms_to_coeff(attack_ms, sr)
    release_coeff = _ms_to_coeff(release_ms, sr)
    env = _envelope_follower(samples, attack_coeff, release_coeff)
    threshold_lin = np.float32(10.0 ** (threshold_db / 20.0))
    ratio_f = np.float32(ratio)
    # Makeup gain: compensate for average gain reduction
    makeup_db = (abs(threshold_db) * (1.0 - 1.0 / ratio)) * 0.5
    makeup_gain = np.float32(10.0 ** (makeup_db / 20.0))
    return _compress_kernel(samples, env, threshold_lin, ratio_f, makeup_gain)


def variants_f001():
    """Return perceptually distinct compressor variants."""
    return [
        # Gentle bus glue
        dict(threshold_db=-10.0, ratio=2.0, attack_ms=30.0, release_ms=200.0),
        # Moderate vocal compression
        dict(threshold_db=-20.0, ratio=4.0, attack_ms=10.0, release_ms=100.0),
        # Heavy squash
        dict(threshold_db=-30.0, ratio=10.0, attack_ms=5.0, release_ms=150.0),
        # Brick-wall limiting
        dict(threshold_db=-15.0, ratio=20.0, attack_ms=0.1, release_ms=50.0),
        # Slow breathing compressor
        dict(threshold_db=-25.0, ratio=6.0, attack_ms=80.0, release_ms=800.0),
        # Fast transient tamer
        dict(threshold_db=-18.0, ratio=8.0, attack_ms=0.5, release_ms=60.0),
    ]


# ===================================================================
# F002 — Expander / Gate
# ===================================================================

@numba.njit
def _expand_kernel(samples, env, threshold_lin, ratio):
    """Attenuate signal below threshold based on ratio."""
    n = len(samples)
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        level = env[i]
        if level < threshold_lin and level > np.float32(1e-12):
            under_db = np.float32(20.0) * np.float32(np.log10(threshold_lin / level))
            attenuation_db = under_db * (ratio - np.float32(1.0))
            gain = np.float32(10.0 ** (-attenuation_db / np.float32(20.0)))
        elif level <= np.float32(1e-12):
            gain = np.float32(0.0)
        else:
            gain = np.float32(1.0)
        out[i] = samples[i] * gain
    return out


def effect_f002_expander_gate(samples: np.ndarray, sr: int, *,
                              threshold_db: float = -30.0,
                              ratio: float = 4.0,
                              attack_ms: float = 5.0,
                              release_ms: float = 50.0) -> np.ndarray:
    """Expander/Gate: attenuate signal below threshold.

    Parameters
    ----------
    threshold_db : float [-50, -10]
        Level below which expansion/gating begins.
    ratio : float [2, 20]
        Expansion ratio. Higher = more gate-like.
    attack_ms : float [0.1, 50]
        How quickly the gate opens.
    release_ms : float [10, 500]
        How quickly the gate closes.
    """
    samples = samples.astype(np.float32)
    attack_coeff = _ms_to_coeff(attack_ms, sr)
    release_coeff = _ms_to_coeff(release_ms, sr)
    env = _envelope_follower(samples, attack_coeff, release_coeff)
    threshold_lin = np.float32(10.0 ** (threshold_db / 20.0))
    ratio_f = np.float32(ratio)
    return _expand_kernel(samples, env, threshold_lin, ratio_f)


def variants_f002():
    """Return perceptually distinct expander/gate variants."""
    return [
        # Gentle noise reduction
        dict(threshold_db=-45.0, ratio=3.0, attack_ms=5.0, release_ms=100.0),
        # Medium gate
        dict(threshold_db=-30.0, ratio=8.0, attack_ms=2.0, release_ms=80.0),
        # Hard gate (tight)
        dict(threshold_db=-25.0, ratio=20.0, attack_ms=0.5, release_ms=30.0),
        # Slow expander
        dict(threshold_db=-35.0, ratio=4.0, attack_ms=20.0, release_ms=300.0),
        # Aggressive chop
        dict(threshold_db=-20.0, ratio=15.0, attack_ms=1.0, release_ms=20.0),
    ]


# ===================================================================
# F003 — Transient Shaper
# ===================================================================

@numba.njit
def _transient_shaper_kernel(samples, attack_gain, sustain_gain,
                             fast_coeff_a, fast_coeff_r,
                             slow_coeff_a, slow_coeff_r):
    """Shape transients using the difference between fast and slow envelopes."""
    n = len(samples)
    out = np.empty(n, dtype=np.float32)
    fast_env = np.float32(0.0)
    slow_env = np.float32(0.0)
    for i in range(n):
        x = samples[i]
        inp = np.float32(abs(x))

        # Fast envelope (tracks transients)
        if inp > fast_env:
            fast_env = fast_coeff_a * fast_env + (np.float32(1.0) - fast_coeff_a) * inp
        else:
            fast_env = fast_coeff_r * fast_env + (np.float32(1.0) - fast_coeff_r) * inp

        # Slow envelope (tracks sustain)
        if inp > slow_env:
            slow_env = slow_coeff_a * slow_env + (np.float32(1.0) - slow_coeff_a) * inp
        else:
            slow_env = slow_coeff_r * slow_env + (np.float32(1.0) - slow_coeff_r) * inp

        # Transient detection: fast - slow
        diff = fast_env - slow_env
        if diff > np.float32(0.0):
            # Transient region: apply attack gain
            gain = np.float32(1.0) + attack_gain * (diff / (fast_env + np.float32(1e-12)))
        else:
            # Sustain region: apply sustain gain
            gain = np.float32(1.0) + sustain_gain * (-diff / (slow_env + np.float32(1e-12)))

        # Clamp gain to avoid negative amplitudes
        if gain < np.float32(0.0):
            gain = np.float32(0.0)

        out[i] = x * gain
    return out


def effect_f003_transient_shaper(samples: np.ndarray, sr: int, *,
                                 attack_gain: float = 0.5,
                                 sustain_gain: float = 0.0) -> np.ndarray:
    """Transient Shaper: emphasize or de-emphasize transients and sustain.

    Uses the difference between a fast and slow envelope to detect transients.

    Parameters
    ----------
    attack_gain : float [-1, 2]
        Gain applied to transient portion. Positive = enhance, negative = soften.
    sustain_gain : float [-1, 2]
        Gain applied to sustain portion. Positive = enhance, negative = reduce.
    """
    samples = samples.astype(np.float32)
    # Fast envelope: ~0.3ms attack, ~5ms release
    fast_a = _ms_to_coeff(0.3, sr)
    fast_r = _ms_to_coeff(5.0, sr)
    # Slow envelope: ~20ms attack, ~200ms release
    slow_a = _ms_to_coeff(20.0, sr)
    slow_r = _ms_to_coeff(200.0, sr)
    return _transient_shaper_kernel(
        samples,
        np.float32(attack_gain), np.float32(sustain_gain),
        fast_a, fast_r, slow_a, slow_r
    )


def variants_f003():
    """Return perceptually distinct transient shaper variants."""
    return [
        # Punch up transients
        dict(attack_gain=1.0, sustain_gain=0.0),
        # Soften transients (smoother)
        dict(attack_gain=-0.8, sustain_gain=0.0),
        # Enhance sustain only
        dict(attack_gain=0.0, sustain_gain=1.5),
        # Reduce sustain (tighter)
        dict(attack_gain=0.0, sustain_gain=-0.8),
        # Max snap: big transients, less sustain
        dict(attack_gain=2.0, sustain_gain=-0.5),
        # Opposite: smoothed attack, blooming sustain
        dict(attack_gain=-0.6, sustain_gain=1.0),
        # Subtle presence
        dict(attack_gain=0.3, sustain_gain=0.3),
    ]


# ===================================================================
# F004 — Multiband Dynamics (3-band)
# ===================================================================

@numba.njit
def _compress_band(samples, attack_coeff, release_coeff, threshold_lin, ratio):
    """Compress a single band: envelope follow + gain reduction."""
    n = len(samples)
    out = np.empty(n, dtype=np.float32)
    env_prev = np.float32(0.0)
    for i in range(n):
        x = samples[i]
        inp = np.float32(abs(x))

        # Envelope follower
        if inp > env_prev:
            env_prev = attack_coeff * env_prev + (np.float32(1.0) - attack_coeff) * inp
        else:
            env_prev = release_coeff * env_prev + (np.float32(1.0) - release_coeff) * inp

        level = env_prev
        if level > threshold_lin and level > np.float32(1e-12):
            over_db = np.float32(20.0) * np.float32(np.log10(level / threshold_lin))
            reduction_db = over_db * (np.float32(1.0) - np.float32(1.0) / ratio)
            gain = np.float32(10.0 ** (-reduction_db / np.float32(20.0)))
        else:
            gain = np.float32(1.0)
        out[i] = x * gain
    return out


def effect_f004_multiband_dynamics(samples: np.ndarray, sr: int, *,
                                   low_xover: float = 250.0,
                                   high_xover: float = 4000.0,
                                   low_threshold_db: float = -15.0,
                                   mid_threshold_db: float = -15.0,
                                   high_threshold_db: float = -15.0,
                                   ratio: float = 4.0) -> np.ndarray:
    """Multiband Dynamics: 3-band split via Butterworth crossovers, compress each independently.

    Parameters
    ----------
    low_xover : float [100, 400]
        Crossover frequency between low and mid bands (Hz).
    high_xover : float [2000, 8000]
        Crossover frequency between mid and high bands (Hz).
    low_threshold_db : float [-30, -5]
        Compression threshold for low band.
    mid_threshold_db : float [-30, -5]
        Compression threshold for mid band.
    high_threshold_db : float [-30, -5]
        Compression threshold for high band.
    ratio : float [2, 10]
        Compression ratio applied to all bands.
    """
    samples = samples.astype(np.float32)

    # Compute crossover filter coefficients
    lp1_b0, lp1_b1, lp1_b2, lp1_a1, lp1_a2 = _butter2_lpf_coeffs(low_xover, sr)
    hp1_b0, hp1_b1, hp1_b2, hp1_a1, hp1_a2 = _butter2_hpf_coeffs(low_xover, sr)
    lp2_b0, lp2_b1, lp2_b2, lp2_a1, lp2_a2 = _butter2_lpf_coeffs(high_xover, sr)
    hp2_b0, hp2_b1, hp2_b2, hp2_a1, hp2_a2 = _butter2_hpf_coeffs(high_xover, sr)

    # Split into 3 bands using Linkwitz-Riley style (cascaded biquads)
    # Low band: lowpass at low_xover
    low_band = _biquad_process(samples, lp1_b0, lp1_b1, lp1_b2, lp1_a1, lp1_a2)

    # High-passed at low_xover (everything above low_xover)
    above_low = _biquad_process(samples, hp1_b0, hp1_b1, hp1_b2, hp1_a1, hp1_a2)

    # Mid band: the above_low signal, lowpassed at high_xover
    mid_band = _biquad_process(above_low, lp2_b0, lp2_b1, lp2_b2, lp2_a1, lp2_a2)

    # High band: the above_low signal, highpassed at high_xover
    high_band = _biquad_process(above_low, hp2_b0, hp2_b1, hp2_b2, hp2_a1, hp2_a2)

    # Compress each band independently
    # Use moderate attack/release times suitable for multiband
    attack_coeff = _ms_to_coeff(10.0, sr)
    release_coeff = _ms_to_coeff(100.0, sr)

    ratio_f = np.float32(ratio)

    low_thresh_lin = np.float32(10.0 ** (low_threshold_db / 20.0))
    mid_thresh_lin = np.float32(10.0 ** (mid_threshold_db / 20.0))
    high_thresh_lin = np.float32(10.0 ** (high_threshold_db / 20.0))

    low_comp = _compress_band(low_band, attack_coeff, release_coeff,
                              low_thresh_lin, ratio_f)
    mid_comp = _compress_band(mid_band, attack_coeff, release_coeff,
                              mid_thresh_lin, ratio_f)
    high_comp = _compress_band(high_band, attack_coeff, release_coeff,
                               high_thresh_lin, ratio_f)

    # Sum bands to reconstruct
    return low_comp + mid_comp + high_comp


def variants_f004():
    """Return perceptually distinct multiband dynamics variants."""
    return [
        # Balanced gentle multiband
        dict(low_xover=200.0, high_xover=4000.0,
             low_threshold_db=-15.0, mid_threshold_db=-15.0,
             high_threshold_db=-15.0, ratio=3.0),
        # Heavy bass control, light highs
        dict(low_xover=150.0, high_xover=3000.0,
             low_threshold_db=-25.0, mid_threshold_db=-12.0,
             high_threshold_db=-8.0, ratio=6.0),
        # Tame harsh highs
        dict(low_xover=300.0, high_xover=5000.0,
             low_threshold_db=-8.0, mid_threshold_db=-10.0,
             high_threshold_db=-25.0, ratio=5.0),
        # Aggressive full-range squash
        dict(low_xover=250.0, high_xover=4000.0,
             low_threshold_db=-28.0, mid_threshold_db=-28.0,
             high_threshold_db=-28.0, ratio=10.0),
        # Wide mid scoop compression
        dict(low_xover=400.0, high_xover=2000.0,
             low_threshold_db=-10.0, mid_threshold_db=-25.0,
             high_threshold_db=-10.0, ratio=4.0),
        # Broadcast style
        dict(low_xover=120.0, high_xover=6000.0,
             low_threshold_db=-20.0, mid_threshold_db=-18.0,
             high_threshold_db=-22.0, ratio=5.0),
    ]


# ===================================================================
# F005 — Sidechain Ducker
# ===================================================================

@numba.njit
def _generate_kick_pattern(n, sr, beat_ms, kick_freq, kick_dur_ms):
    """Generate synthetic kick sidechain signal: sine bursts at regular intervals."""
    sidechain = np.zeros(n, dtype=np.float32)
    beat_samples = int(beat_ms * np.float32(0.001) * np.float32(sr))
    kick_dur_samples = int(kick_dur_ms * np.float32(0.001) * np.float32(sr))
    if beat_samples < 1:
        beat_samples = 1
    if kick_dur_samples < 1:
        kick_dur_samples = 1

    pos = 0
    while pos < n:
        for j in range(kick_dur_samples):
            idx = pos + j
            if idx >= n:
                break
            # Sine burst with exponential decay
            t = np.float32(j) / np.float32(sr)
            decay = np.float32(np.exp(-j / (kick_dur_samples * 0.3)))
            sidechain[idx] = decay * np.float32(np.sin(
                np.float32(2.0) * np.float32(3.141592653589793) * kick_freq * t
            ))
        pos += beat_samples
    return sidechain


@numba.njit
def _ducker_kernel(samples, sc_env, duck_amount):
    """Apply ducking based on sidechain envelope."""
    n = len(samples)
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        # Duck amount scales how much gain reduction from the sidechain
        gain = np.float32(1.0) - duck_amount * sc_env[i]
        if gain < np.float32(0.0):
            gain = np.float32(0.0)
        out[i] = samples[i] * gain
    return out


def effect_f005_sidechain_ducker(samples: np.ndarray, sr: int, *,
                                 beat_ms: float = 500.0,
                                 duck_amount: float = 0.8,
                                 attack_ms: float = 2.0,
                                 release_ms: float = 100.0) -> np.ndarray:
    """Sidechain Ducker: generate synthetic kick pattern and duck the input.

    Creates a 50 Hz sine burst (~20ms) every beat_ms milliseconds as the sidechain
    trigger, then uses its envelope to attenuate the input signal.

    Parameters
    ----------
    beat_ms : float [200, 600]
        Interval between kick triggers in milliseconds.
    duck_amount : float [0.3, 1.0]
        How much the signal is ducked (1.0 = full ducking).
    attack_ms : float [1, 10]
        How quickly the ducker reacts to the kick.
    release_ms : float [50, 300]
        How quickly the signal recovers after ducking.
    """
    samples = samples.astype(np.float32)
    n = len(samples)

    # Generate synthetic kick sidechain: 50 Hz sine burst, ~20ms duration
    sidechain = _generate_kick_pattern(n, sr, beat_ms,
                                       np.float32(50.0), np.float32(20.0))

    # Envelope follow the sidechain signal
    attack_coeff = _ms_to_coeff(attack_ms, sr)
    release_coeff = _ms_to_coeff(release_ms, sr)
    sc_env = _envelope_follower(sidechain, attack_coeff, release_coeff)

    # Normalize sidechain envelope to [0, 1]
    peak = np.max(sc_env)
    if peak > 1e-12:
        sc_env = sc_env / peak

    return _ducker_kernel(samples, sc_env, np.float32(duck_amount))


def variants_f005():
    """Return perceptually distinct sidechain ducker variants."""
    return [
        # Classic four-on-the-floor pump (120 BPM)
        dict(beat_ms=500.0, duck_amount=0.9, attack_ms=2.0, release_ms=150.0),
        # Fast techno pump (140 BPM)
        dict(beat_ms=428.0, duck_amount=1.0, attack_ms=1.0, release_ms=100.0),
        # Slow half-time (80 BPM, beats on halves)
        dict(beat_ms=375.0, duck_amount=0.7, attack_ms=5.0, release_ms=250.0),
        # Subtle rhythmic ducking
        dict(beat_ms=500.0, duck_amount=0.4, attack_ms=3.0, release_ms=200.0),
        # Extreme choppy gate effect
        dict(beat_ms=250.0, duck_amount=1.0, attack_ms=1.0, release_ms=60.0),
    ]


# ===================================================================
# F006 — RMS Compressor
# ===================================================================

@numba.njit
def _rms_envelope(samples, window_samples):
    """Compute running RMS envelope."""
    n = len(samples)
    env = np.empty(n, dtype=np.float32)
    sum_sq = np.float32(0.0)
    w = int(window_samples)
    if w < 1:
        w = 1
    for i in range(n):
        sum_sq += samples[i] * samples[i]
        if i >= w:
            sum_sq -= samples[i - w] * samples[i - w]
            if sum_sq < np.float32(0.0):
                sum_sq = np.float32(0.0)
        count = min(i + 1, w)
        env[i] = np.float32(np.sqrt(sum_sq / np.float32(count)))
    return env


@numba.njit
def _rms_compress_kernel(samples, env, threshold_lin, ratio, makeup_gain,
                         attack_coeff, release_coeff):
    """Compress based on smoothed RMS envelope."""
    n = len(samples)
    out = np.empty(n, dtype=np.float32)
    smooth_env = np.float32(0.0)
    for i in range(n):
        level = env[i]
        if level > smooth_env:
            smooth_env = attack_coeff * smooth_env + (np.float32(1.0) - attack_coeff) * level
        else:
            smooth_env = release_coeff * smooth_env + (np.float32(1.0) - release_coeff) * level
        if smooth_env > threshold_lin and smooth_env > np.float32(1e-12):
            over_db = np.float32(20.0) * np.float32(np.log10(smooth_env / threshold_lin))
            reduction_db = over_db * (np.float32(1.0) - np.float32(1.0) / ratio)
            gain = np.float32(10.0 ** (-reduction_db / np.float32(20.0)))
        else:
            gain = np.float32(1.0)
        out[i] = samples[i] * gain * makeup_gain
    return out


def effect_f006_rms_compressor(samples: np.ndarray, sr: int, *,
                                threshold_db: float = -20.0,
                                ratio: float = 4.0,
                                attack_ms: float = 10.0,
                                release_ms: float = 100.0,
                                rms_window_ms: float = 50.0) -> np.ndarray:
    """RMS Compressor: envelope follows RMS level instead of peak.

    Smoother, more musical response — ignores brief transients and reacts to
    the perceived loudness of the signal. Great for vocals, pads, mix bus.

    Parameters
    ----------
    threshold_db : float [-40, 0]
        RMS level above which compression begins.
    ratio : float [2, 20]
        Compression ratio.
    attack_ms : float [0.1, 100]
        Smoothing attack time.
    release_ms : float [10, 1000]
        Smoothing release time.
    rms_window_ms : float [10, 200]
        RMS averaging window in milliseconds.
    """
    samples = samples.astype(np.float32)
    window_samples = int(rms_window_ms * 0.001 * sr)
    env = _rms_envelope(samples, window_samples)
    attack_coeff = _ms_to_coeff(attack_ms, sr)
    release_coeff = _ms_to_coeff(release_ms, sr)
    threshold_lin = np.float32(10.0 ** (threshold_db / 20.0))
    ratio_f = np.float32(ratio)
    makeup_db = (abs(threshold_db) * (1.0 - 1.0 / ratio)) * 0.5
    makeup_gain = np.float32(10.0 ** (makeup_db / 20.0))
    return _rms_compress_kernel(samples, env, threshold_lin, ratio_f, makeup_gain,
                                attack_coeff, release_coeff)


def variants_f006():
    return [
        # Gentle vocal leveler
        dict(threshold_db=-18.0, ratio=3.0, attack_ms=20.0, release_ms=200.0, rms_window_ms=80.0),
        # Mix bus glue
        dict(threshold_db=-14.0, ratio=2.0, attack_ms=30.0, release_ms=300.0, rms_window_ms=100.0),
        # Heavy sustain boost
        dict(threshold_db=-28.0, ratio=8.0, attack_ms=10.0, release_ms=150.0, rms_window_ms=50.0),
        # Pad leveler (very slow)
        dict(threshold_db=-22.0, ratio=4.0, attack_ms=60.0, release_ms=600.0, rms_window_ms=150.0),
        # Tight RMS squash
        dict(threshold_db=-24.0, ratio=12.0, attack_ms=5.0, release_ms=80.0, rms_window_ms=20.0),
        # Bass smoothing
        dict(threshold_db=-20.0, ratio=5.0, attack_ms=15.0, release_ms=120.0, rms_window_ms=60.0),
    ]


# ===================================================================
# F007 — Feedback Compressor
# ===================================================================

@numba.njit
def _feedback_compress_kernel(samples, threshold_lin, ratio, makeup_gain,
                               attack_coeff, release_coeff):
    """Feedback topology: envelope follows the output, not the input."""
    n = len(samples)
    out = np.empty(n, dtype=np.float32)
    env = np.float32(0.0)
    prev_gain = np.float32(1.0)
    for i in range(n):
        # Envelope follows the previous output sample
        out_level = np.float32(abs(samples[i] * prev_gain))
        if out_level > env:
            env = attack_coeff * env + (np.float32(1.0) - attack_coeff) * out_level
        else:
            env = release_coeff * env + (np.float32(1.0) - release_coeff) * out_level
        if env > threshold_lin and env > np.float32(1e-12):
            over_db = np.float32(20.0) * np.float32(np.log10(env / threshold_lin))
            reduction_db = over_db * (np.float32(1.0) - np.float32(1.0) / ratio)
            gain = np.float32(10.0 ** (-reduction_db / np.float32(20.0)))
        else:
            gain = np.float32(1.0)
        prev_gain = gain
        out[i] = samples[i] * gain * makeup_gain
    return out


def effect_f007_feedback_compressor(samples: np.ndarray, sr: int, *,
                                     threshold_db: float = -20.0,
                                     ratio: float = 4.0,
                                     attack_ms: float = 10.0,
                                     release_ms: float = 100.0) -> np.ndarray:
    """Feedback Compressor: envelope follows the output signal.

    Self-regulating — heavier compression reduces the signal, which in turn
    reduces the envelope, which eases the compression. More forgiving and
    vintage-sounding than feedforward designs (dbx 160 style).

    Parameters
    ----------
    threshold_db : float [-40, 0]
        Level above which compression begins.
    ratio : float [2, 20]
        Compression ratio.
    attack_ms : float [0.1, 100]
        How quickly the compressor reacts.
    release_ms : float [10, 1000]
        How quickly the compressor releases.
    """
    samples = samples.astype(np.float32)
    attack_coeff = _ms_to_coeff(attack_ms, sr)
    release_coeff = _ms_to_coeff(release_ms, sr)
    threshold_lin = np.float32(10.0 ** (threshold_db / 20.0))
    ratio_f = np.float32(ratio)
    makeup_db = (abs(threshold_db) * (1.0 - 1.0 / ratio)) * 0.4
    makeup_gain = np.float32(10.0 ** (makeup_db / 20.0))
    return _feedback_compress_kernel(samples, threshold_lin, ratio_f, makeup_gain,
                                     attack_coeff, release_coeff)


def variants_f007():
    return [
        # Subtle vintage leveling
        dict(threshold_db=-16.0, ratio=3.0, attack_ms=20.0, release_ms=200.0),
        # Aggressive pumping
        dict(threshold_db=-24.0, ratio=8.0, attack_ms=5.0, release_ms=80.0),
        # Bass fattener
        dict(threshold_db=-20.0, ratio=6.0, attack_ms=15.0, release_ms=150.0),
        # Gentle mix bus
        dict(threshold_db=-12.0, ratio=2.0, attack_ms=30.0, release_ms=300.0),
        # Vocal warmth
        dict(threshold_db=-22.0, ratio=4.0, attack_ms=8.0, release_ms=120.0),
        # Hard feedback slam
        dict(threshold_db=-30.0, ratio=15.0, attack_ms=2.0, release_ms=60.0),
    ]


# ===================================================================
# F008 — Opto Compressor
# ===================================================================

@numba.njit
def _opto_compress_kernel(samples, threshold_lin, ratio, makeup_gain,
                           attack_coeff, fast_rel_coeff, slow_rel_coeff,
                           slow_blend):
    """Opto-style: two-stage release models photocell behavior.

    The release has a fast initial component and a slow exponential tail.
    Louder signals drive the slow component harder (program-dependent).
    """
    n = len(samples)
    out = np.empty(n, dtype=np.float32)
    fast_env = np.float32(0.0)
    slow_env = np.float32(0.0)
    for i in range(n):
        inp = np.float32(abs(samples[i]))
        # Fast envelope
        if inp > fast_env:
            fast_env = attack_coeff * fast_env + (np.float32(1.0) - attack_coeff) * inp
        else:
            fast_env = fast_rel_coeff * fast_env
        # Slow envelope — driven harder by loud signals
        if inp > slow_env:
            slow_env = attack_coeff * slow_env + (np.float32(1.0) - attack_coeff) * inp
        else:
            slow_env = slow_rel_coeff * slow_env
        # Blend the two envelopes
        env = (np.float32(1.0) - slow_blend) * fast_env + slow_blend * slow_env
        if env > threshold_lin and env > np.float32(1e-12):
            over_db = np.float32(20.0) * np.float32(np.log10(env / threshold_lin))
            reduction_db = over_db * (np.float32(1.0) - np.float32(1.0) / ratio)
            gain = np.float32(10.0 ** (-reduction_db / np.float32(20.0)))
        else:
            gain = np.float32(1.0)
        out[i] = samples[i] * gain * makeup_gain
    return out


def effect_f008_opto_compressor(samples: np.ndarray, sr: int, *,
                                 threshold_db: float = -20.0,
                                 ratio: float = 4.0,
                                 attack_ms: float = 10.0,
                                 fast_release_ms: float = 60.0,
                                 slow_release_ms: float = 500.0,
                                 slow_blend: float = 0.5) -> np.ndarray:
    """Opto Compressor: models photocell with two-stage release.

    Fast initial release followed by a slow exponential tail. Program-dependent —
    loud sustained signals engage the slow release more. LA-2A character.

    Parameters
    ----------
    threshold_db : float [-40, 0]
        Level above which compression begins.
    ratio : float [2, 20]
        Compression ratio.
    attack_ms : float [1, 50]
        Attack time.
    fast_release_ms : float [20, 200]
        Fast release component.
    slow_release_ms : float [200, 2000]
        Slow release tail.
    slow_blend : float [0, 1]
        How much the slow release dominates (0=all fast, 1=all slow).
    """
    samples = samples.astype(np.float32)
    attack_coeff = _ms_to_coeff(attack_ms, sr)
    fast_rel_coeff = _ms_to_coeff(fast_release_ms, sr)
    slow_rel_coeff = _ms_to_coeff(slow_release_ms, sr)
    threshold_lin = np.float32(10.0 ** (threshold_db / 20.0))
    ratio_f = np.float32(ratio)
    makeup_db = (abs(threshold_db) * (1.0 - 1.0 / ratio)) * 0.45
    makeup_gain = np.float32(10.0 ** (makeup_db / 20.0))
    return _opto_compress_kernel(samples, threshold_lin, ratio_f, makeup_gain,
                                  attack_coeff, fast_rel_coeff, slow_rel_coeff,
                                  np.float32(slow_blend))


def variants_f008():
    return [
        # Classic LA-2A vocal
        dict(threshold_db=-20.0, ratio=4.0, attack_ms=10.0, fast_release_ms=60.0,
             slow_release_ms=500.0, slow_blend=0.6),
        # Fingerpicked guitar
        dict(threshold_db=-18.0, ratio=3.0, attack_ms=15.0, fast_release_ms=80.0,
             slow_release_ms=800.0, slow_blend=0.4),
        # Pad leveling (mostly slow)
        dict(threshold_db=-24.0, ratio=5.0, attack_ms=20.0, fast_release_ms=100.0,
             slow_release_ms=1500.0, slow_blend=0.8),
        # Snappy opto (mostly fast)
        dict(threshold_db=-16.0, ratio=6.0, attack_ms=5.0, fast_release_ms=40.0,
             slow_release_ms=300.0, slow_blend=0.2),
        # Heavy opto squash
        dict(threshold_db=-28.0, ratio=10.0, attack_ms=8.0, fast_release_ms=50.0,
             slow_release_ms=600.0, slow_blend=0.5),
        # Gentle acoustic smoothing
        dict(threshold_db=-15.0, ratio=2.5, attack_ms=12.0, fast_release_ms=70.0,
             slow_release_ms=1000.0, slow_blend=0.7),
    ]


# ===================================================================
# F009 — FET Compressor
# ===================================================================

@numba.njit
def _fet_compress_kernel(samples, threshold_lin, ratio, makeup_gain,
                          attack_coeff, release_coeff, drive):
    """FET-style: ultra-fast, can push into harmonic distortion."""
    n = len(samples)
    out = np.empty(n, dtype=np.float32)
    env = np.float32(0.0)
    for i in range(n):
        # Drive the input
        driven = samples[i] * drive
        inp = np.float32(abs(driven))
        if inp > env:
            env = attack_coeff * env + (np.float32(1.0) - attack_coeff) * inp
        else:
            env = release_coeff * env + (np.float32(1.0) - release_coeff) * inp
        if env > threshold_lin and env > np.float32(1e-12):
            over_db = np.float32(20.0) * np.float32(np.log10(env / threshold_lin))
            reduction_db = over_db * (np.float32(1.0) - np.float32(1.0) / ratio)
            gain = np.float32(10.0 ** (-reduction_db / np.float32(20.0)))
        else:
            gain = np.float32(1.0)
        # FET-style saturation on heavy gain reduction
        out_sample = driven * gain * makeup_gain
        # Soft clip at extremes
        if out_sample > np.float32(1.0):
            out_sample = np.float32(np.tanh(out_sample))
        elif out_sample < np.float32(-1.0):
            out_sample = np.float32(np.tanh(out_sample))
        out[i] = out_sample
    return out


def effect_f009_fet_compressor(samples: np.ndarray, sr: int, *,
                                threshold_db: float = -20.0,
                                ratio: float = 8.0,
                                attack_ms: float = 0.2,
                                release_ms: float = 50.0,
                                input_drive: float = 1.0) -> np.ndarray:
    """FET Compressor: ultra-fast attack, aggressive character.

    Models FET (field-effect transistor) topology like the 1176. Sub-millisecond
    attack interacts with low frequencies causing harmonic coloration. Input drive
    pushes the detector harder and adds soft saturation at extremes.

    Parameters
    ----------
    threshold_db : float [-40, 0]
        Level above which compression begins.
    ratio : float [4, 20]
        Compression ratio.
    attack_ms : float [0.02, 1.0]
        Ultra-fast attack time.
    release_ms : float [20, 200]
        Release time.
    input_drive : float [0.5, 4.0]
        Input gain driving the detector and saturation.
    """
    samples = samples.astype(np.float32)
    attack_coeff = _ms_to_coeff(attack_ms, sr)
    release_coeff = _ms_to_coeff(release_ms, sr)
    threshold_lin = np.float32(10.0 ** (threshold_db / 20.0))
    ratio_f = np.float32(ratio)
    makeup_db = (abs(threshold_db) * (1.0 - 1.0 / ratio)) * 0.4
    makeup_gain = np.float32(10.0 ** (makeup_db / 20.0))
    return _fet_compress_kernel(samples, threshold_lin, ratio_f, makeup_gain,
                                attack_coeff, release_coeff, np.float32(input_drive))


def variants_f009():
    return [
        # All-buttons-in (1176 trick: everything fast, max ratio)
        dict(threshold_db=-20.0, ratio=20.0, attack_ms=0.05, release_ms=30.0, input_drive=2.0),
        # Drum smash
        dict(threshold_db=-24.0, ratio=12.0, attack_ms=0.1, release_ms=50.0, input_drive=1.5),
        # Bass crunch
        dict(threshold_db=-18.0, ratio=8.0, attack_ms=0.5, release_ms=80.0, input_drive=1.8),
        # Vocal bite
        dict(threshold_db=-22.0, ratio=8.0, attack_ms=0.2, release_ms=60.0, input_drive=1.2),
        # Gentle FET color
        dict(threshold_db=-14.0, ratio=4.0, attack_ms=0.8, release_ms=100.0, input_drive=1.0),
        # Extreme destruction
        dict(threshold_db=-30.0, ratio=20.0, attack_ms=0.02, release_ms=20.0, input_drive=4.0),
    ]


# ===================================================================
# F010 — Soft-Knee Compressor
# ===================================================================

@numba.njit
def _softknee_compress_kernel(samples, env, threshold_db, ratio, knee_db, makeup_gain):
    """Soft-knee: gradual onset of compression around the threshold."""
    n = len(samples)
    out = np.empty(n, dtype=np.float32)
    half_knee = knee_db * np.float32(0.5)
    for i in range(n):
        level = env[i]
        if level < np.float32(1e-12):
            out[i] = samples[i] * makeup_gain
            continue
        level_db = np.float32(20.0) * np.float32(np.log10(level))
        if level_db < threshold_db - half_knee:
            # Below knee: no compression
            gain_db = np.float32(0.0)
        elif level_db > threshold_db + half_knee:
            # Above knee: full compression
            over_db = level_db - threshold_db
            gain_db = -over_db * (np.float32(1.0) - np.float32(1.0) / ratio)
        else:
            # In the knee: quadratic interpolation
            x = level_db - threshold_db + half_knee
            gain_db = -(np.float32(1.0) - np.float32(1.0) / ratio) * x * x / (np.float32(2.0) * knee_db)
        gain = np.float32(10.0 ** (gain_db / np.float32(20.0)))
        out[i] = samples[i] * gain * makeup_gain
    return out


def effect_f010_softknee_compressor(samples: np.ndarray, sr: int, *,
                                     threshold_db: float = -20.0,
                                     ratio: float = 4.0,
                                     knee_db: float = 10.0,
                                     attack_ms: float = 10.0,
                                     release_ms: float = 100.0) -> np.ndarray:
    """Soft-Knee Compressor: gradual onset of compression.

    Instead of an abrupt threshold, compression ramps in smoothly over a knee_db
    range. More transparent — good for mastering and mix bus.

    Parameters
    ----------
    threshold_db : float [-40, 0]
        Center of the knee region.
    ratio : float [2, 20]
        Compression ratio above the knee.
    knee_db : float [0, 20]
        Width of the soft knee (0 = hard knee).
    attack_ms : float [0.1, 100]
        Attack time.
    release_ms : float [10, 1000]
        Release time.
    """
    samples = samples.astype(np.float32)
    attack_coeff = _ms_to_coeff(attack_ms, sr)
    release_coeff = _ms_to_coeff(release_ms, sr)
    env = _envelope_follower(samples, attack_coeff, release_coeff)
    threshold_f = np.float32(threshold_db)
    ratio_f = np.float32(ratio)
    knee_f = np.float32(max(knee_db, 0.01))
    makeup_db = (abs(threshold_db) * (1.0 - 1.0 / ratio)) * 0.4
    makeup_gain = np.float32(10.0 ** (makeup_db / 20.0))
    return _softknee_compress_kernel(samples, env, threshold_f, ratio_f, knee_f, makeup_gain)


def variants_f010():
    return [
        # Wide knee transparent (mastering)
        dict(threshold_db=-16.0, ratio=2.5, knee_db=16.0, attack_ms=20.0, release_ms=200.0),
        # Narrow knee punchy
        dict(threshold_db=-22.0, ratio=6.0, knee_db=3.0, attack_ms=5.0, release_ms=80.0),
        # Mastering limiter
        dict(threshold_db=-8.0, ratio=15.0, knee_db=6.0, attack_ms=1.0, release_ms=50.0),
        # Gentle bus glue (wide knee)
        dict(threshold_db=-14.0, ratio=2.0, knee_db=20.0, attack_ms=30.0, release_ms=300.0),
        # Vocal polish
        dict(threshold_db=-20.0, ratio=4.0, knee_db=10.0, attack_ms=10.0, release_ms=120.0),
        # Medium knee drum tamer
        dict(threshold_db=-18.0, ratio=8.0, knee_db=8.0, attack_ms=2.0, release_ms=60.0),
    ]


# ===================================================================
# F011 — Parallel (NY) Compressor
# ===================================================================

def effect_f011_parallel_compressor(samples: np.ndarray, sr: int, *,
                                     threshold_db: float = -30.0,
                                     ratio: float = 10.0,
                                     attack_ms: float = 5.0,
                                     release_ms: float = 100.0,
                                     wet_mix: float = 0.5) -> np.ndarray:
    """Parallel (NY) Compressor: heavy compression blended with dry.

    Heavily compresses the signal then blends it back with the original.
    Effectively provides upward compression — quiet parts come up while
    loud parts stay roughly the same. Classic drum room and vocal trick.

    Parameters
    ----------
    threshold_db : float [-40, -10]
        Compression threshold (set low for heavy squash).
    ratio : float [4, 100]
        Compression ratio (high for heavy squash).
    attack_ms : float [0.1, 50]
        Attack time.
    release_ms : float [20, 500]
        Release time.
    wet_mix : float [0.1, 0.8]
        Blend of compressed signal (0=dry only, 1=compressed only).
    """
    samples = samples.astype(np.float32)
    attack_coeff = _ms_to_coeff(attack_ms, sr)
    release_coeff = _ms_to_coeff(release_ms, sr)
    env = _envelope_follower(samples, attack_coeff, release_coeff)
    threshold_lin = np.float32(10.0 ** (threshold_db / 20.0))
    ratio_f = np.float32(ratio)
    # Heavy makeup for the compressed signal
    makeup_db = (abs(threshold_db) * (1.0 - 1.0 / ratio)) * 0.7
    makeup_gain = np.float32(10.0 ** (makeup_db / 20.0))
    compressed = _compress_kernel(samples, env, threshold_lin, ratio_f, makeup_gain)
    wet = np.float32(wet_mix)
    dry = np.float32(1.0) - wet
    return dry * samples + wet * compressed


def variants_f011():
    return [
        # Drum punch (fast, 50% wet)
        dict(threshold_db=-28.0, ratio=12.0, attack_ms=2.0, release_ms=80.0, wet_mix=0.5),
        # Vocal thickness (slower, 30% wet)
        dict(threshold_db=-24.0, ratio=8.0, attack_ms=15.0, release_ms=150.0, wet_mix=0.3),
        # Full smash blend
        dict(threshold_db=-35.0, ratio=100.0, attack_ms=1.0, release_ms=50.0, wet_mix=0.4),
        # Subtle body
        dict(threshold_db=-20.0, ratio=6.0, attack_ms=10.0, release_ms=120.0, wet_mix=0.25),
        # Ambient room lift
        dict(threshold_db=-32.0, ratio=20.0, attack_ms=20.0, release_ms=300.0, wet_mix=0.6),
        # Bass parallel thickening
        dict(threshold_db=-26.0, ratio=10.0, attack_ms=8.0, release_ms=100.0, wet_mix=0.45),
    ]


# ===================================================================
# F012 — Upward Compressor
# ===================================================================

@numba.njit
def _upward_compress_kernel(samples, env, threshold_lin, ratio, max_boost_lin,
                             attack_coeff, release_coeff):
    """Boost signal below threshold — upward compression."""
    n = len(samples)
    out = np.empty(n, dtype=np.float32)
    smooth = np.float32(0.0)
    for i in range(n):
        level = env[i]
        if level > smooth:
            smooth = attack_coeff * smooth + (np.float32(1.0) - attack_coeff) * level
        else:
            smooth = release_coeff * smooth + (np.float32(1.0) - release_coeff) * level
        if smooth < threshold_lin and smooth > np.float32(1e-12):
            under_db = np.float32(20.0) * np.float32(np.log10(threshold_lin / smooth))
            boost_db = under_db * (np.float32(1.0) - np.float32(1.0) / ratio)
            gain = np.float32(10.0 ** (boost_db / np.float32(20.0)))
            if gain > max_boost_lin:
                gain = max_boost_lin
        else:
            gain = np.float32(1.0)
        out[i] = samples[i] * gain
    return out


def effect_f012_upward_compressor(samples: np.ndarray, sr: int, *,
                                   threshold_db: float = -30.0,
                                   ratio: float = 3.0,
                                   attack_ms: float = 10.0,
                                   release_ms: float = 150.0,
                                   max_boost_db: float = 20.0) -> np.ndarray:
    """Upward Compressor: boosts quiet signals below threshold.

    Unlike downward compression which reduces loud signals, this boosts quiet
    ones. Brings up room tone, reverb tails, ghost notes, subtle details.

    Parameters
    ----------
    threshold_db : float [-40, -10]
        Level below which boosting begins.
    ratio : float [1.5, 10]
        Upward compression ratio.
    attack_ms : float [0.1, 100]
        Attack time.
    release_ms : float [10, 1000]
        Release time.
    max_boost_db : float [6, 30]
        Maximum boost to prevent noise floor explosion.
    """
    samples = samples.astype(np.float32)
    attack_coeff = _ms_to_coeff(attack_ms, sr)
    release_coeff = _ms_to_coeff(release_ms, sr)
    env = _envelope_follower(samples, attack_coeff, release_coeff)
    threshold_lin = np.float32(10.0 ** (threshold_db / 20.0))
    ratio_f = np.float32(ratio)
    max_boost_lin = np.float32(10.0 ** (max_boost_db / 20.0))
    return _upward_compress_kernel(samples, env, threshold_lin, ratio_f, max_boost_lin,
                                    attack_coeff, release_coeff)


def variants_f012():
    return [
        # Room tone lifter
        dict(threshold_db=-35.0, ratio=3.0, attack_ms=15.0, release_ms=200.0, max_boost_db=18.0),
        # Reverb tail enhancer
        dict(threshold_db=-40.0, ratio=4.0, attack_ms=20.0, release_ms=400.0, max_boost_db=24.0),
        # Subtle detail boost
        dict(threshold_db=-28.0, ratio=2.0, attack_ms=10.0, release_ms=150.0, max_boost_db=12.0),
        # Aggressive uplift
        dict(threshold_db=-25.0, ratio=8.0, attack_ms=5.0, release_ms=100.0, max_boost_db=20.0),
        # Ghost note reveal
        dict(threshold_db=-32.0, ratio=5.0, attack_ms=3.0, release_ms=80.0, max_boost_db=15.0),
        # Ambient texture lift
        dict(threshold_db=-38.0, ratio=3.5, attack_ms=30.0, release_ms=500.0, max_boost_db=22.0),
    ]


# ===================================================================
# F013 — Program-Dependent Compressor
# ===================================================================

@numba.njit
def _program_dependent_kernel(samples, threshold_lin, ratio, makeup_gain,
                               attack_coeff, min_rel_coeff, max_rel_coeff):
    """Release adapts based on crest factor: transients get fast release,
    sustained signals get slow release."""
    n = len(samples)
    out = np.empty(n, dtype=np.float32)
    env = np.float32(0.0)
    rms_acc = np.float32(0.0)
    rms_coeff = np.float32(0.999)
    for i in range(n):
        inp = np.float32(abs(samples[i]))
        # Running RMS estimate
        rms_acc = rms_coeff * rms_acc + (np.float32(1.0) - rms_coeff) * (samples[i] * samples[i])
        rms = np.float32(np.sqrt(rms_acc))
        # Crest factor: peak / rms (high = transient, low = sustained)
        if rms > np.float32(1e-12):
            crest = inp / rms
        else:
            crest = np.float32(1.0)
        # Map crest to release coefficient: high crest -> fast release (min_rel_coeff)
        # Clamp crest to [1, 10] range for mapping
        crest_norm = (min(max(crest, np.float32(1.0)), np.float32(10.0)) - np.float32(1.0)) / np.float32(9.0)
        rel_coeff = max_rel_coeff + (min_rel_coeff - max_rel_coeff) * crest_norm
        if inp > env:
            env = attack_coeff * env + (np.float32(1.0) - attack_coeff) * inp
        else:
            env = rel_coeff * env + (np.float32(1.0) - rel_coeff) * inp
        if env > threshold_lin and env > np.float32(1e-12):
            over_db = np.float32(20.0) * np.float32(np.log10(env / threshold_lin))
            reduction_db = over_db * (np.float32(1.0) - np.float32(1.0) / ratio)
            gain = np.float32(10.0 ** (-reduction_db / np.float32(20.0)))
        else:
            gain = np.float32(1.0)
        out[i] = samples[i] * gain * makeup_gain
    return out


def effect_f013_program_dependent_compressor(samples: np.ndarray, sr: int, *,
                                              threshold_db: float = -20.0,
                                              ratio: float = 4.0,
                                              attack_ms: float = 5.0,
                                              min_release_ms: float = 20.0,
                                              max_release_ms: float = 500.0) -> np.ndarray:
    """Program-Dependent Compressor: release adapts to signal content.

    Uses crest factor (peak/RMS ratio) to modulate release time. Transient bursts
    get fast release (no pumping), sustained loud passages get slow release (smooth).
    Adapts automatically to any material.

    Parameters
    ----------
    threshold_db : float [-40, 0]
        Level above which compression begins.
    ratio : float [2, 20]
        Compression ratio.
    attack_ms : float [0.1, 50]
        Attack time.
    min_release_ms : float [5, 100]
        Fastest release (used for transient material).
    max_release_ms : float [100, 2000]
        Slowest release (used for sustained material).
    """
    samples = samples.astype(np.float32)
    attack_coeff = _ms_to_coeff(attack_ms, sr)
    min_rel_coeff = _ms_to_coeff(min_release_ms, sr)
    max_rel_coeff = _ms_to_coeff(max_release_ms, sr)
    threshold_lin = np.float32(10.0 ** (threshold_db / 20.0))
    ratio_f = np.float32(ratio)
    makeup_db = (abs(threshold_db) * (1.0 - 1.0 / ratio)) * 0.5
    makeup_gain = np.float32(10.0 ** (makeup_db / 20.0))
    return _program_dependent_kernel(samples, threshold_lin, ratio_f, makeup_gain,
                                      attack_coeff, min_rel_coeff, max_rel_coeff)


def variants_f013():
    return [
        # Adaptive drums
        dict(threshold_db=-22.0, ratio=6.0, attack_ms=2.0, min_release_ms=15.0, max_release_ms=300.0),
        # Adaptive full mix
        dict(threshold_db=-16.0, ratio=3.0, attack_ms=10.0, min_release_ms=30.0, max_release_ms=500.0),
        # Aggressive auto-release
        dict(threshold_db=-26.0, ratio=10.0, attack_ms=1.0, min_release_ms=10.0, max_release_ms=200.0),
        # Gentle vocal riding
        dict(threshold_db=-20.0, ratio=4.0, attack_ms=8.0, min_release_ms=40.0, max_release_ms=800.0),
        # Fast transient preserve
        dict(threshold_db=-18.0, ratio=5.0, attack_ms=5.0, min_release_ms=5.0, max_release_ms=400.0),
        # Wide auto range
        dict(threshold_db=-24.0, ratio=8.0, attack_ms=3.0, min_release_ms=10.0, max_release_ms=1500.0),
    ]


# ===================================================================
# F014 — Lookahead Limiter
# ===================================================================

@numba.njit
def _lookahead_limiter_kernel(samples, threshold_lin, release_coeff, lookahead):
    """Lookahead limiter: sliding window max + delayed audio."""
    n = len(samples)
    out = np.empty(n, dtype=np.float32)
    la = int(lookahead)
    if la < 1:
        la = 1

    # Sliding window max using deque-style ring buffer
    # Store (value, index) pairs; ring_val[head..tail] is monotone decreasing
    ring_idx = np.empty(la + 1, dtype=numba.int64)
    ring_val = np.empty(la + 1, dtype=np.float32)
    head = 0
    tail = 0  # tail is one past the last valid entry

    peak_env = np.empty(n, dtype=np.float32)
    for i in range(n):
        # The window covers samples[i .. i+la]
        right = i + la
        if right >= n:
            right = n - 1
        v = np.float32(abs(samples[right]))
        # Remove from back anything smaller than v
        while tail > head and ring_val[tail - 1] <= v:
            tail -= 1
        ring_idx[tail] = right
        ring_val[tail] = v
        tail += 1
        # Remove from front anything outside the window
        while head < tail and ring_idx[head] < i:
            head += 1
        peak_env[i] = ring_val[head]

    # Smooth the envelope (release only — attack is instant due to lookahead)
    smooth = peak_env[0]
    for i in range(n):
        if peak_env[i] > smooth:
            smooth = peak_env[i]
        else:
            smooth = release_coeff * smooth + (np.float32(1.0) - release_coeff) * peak_env[i]
        peak_env[i] = smooth

    # Apply gain reduction to delayed audio
    for i in range(n):
        delayed_idx = i - la
        if delayed_idx < 0:
            sample = np.float32(0.0)
        else:
            sample = samples[delayed_idx]
        level = peak_env[i]
        if level > threshold_lin:
            gain = threshold_lin / level
        else:
            gain = np.float32(1.0)
        out[i] = sample * gain
    return out


def effect_f014_lookahead_limiter(samples: np.ndarray, sr: int, *,
                                   threshold_db: float = -1.0,
                                   release_ms: float = 50.0,
                                   lookahead_ms: float = 5.0) -> np.ndarray:
    """Lookahead Limiter: zero-overshoot brickwall limiting.

    Delays the audio and reads the envelope ahead of time, catching every
    transient before it passes. True peak limiting with no overshoot.

    Parameters
    ----------
    threshold_db : float [-12, 0]
        Ceiling — output will never exceed this level.
    release_ms : float [10, 200]
        How quickly gain returns to unity after limiting.
    lookahead_ms : float [1, 10]
        Lookahead window in milliseconds.
    """
    samples = samples.astype(np.float32)
    threshold_lin = np.float32(10.0 ** (threshold_db / 20.0))
    release_coeff = _ms_to_coeff(release_ms, sr)
    lookahead_samples = int(lookahead_ms * 0.001 * sr)
    return _lookahead_limiter_kernel(samples, threshold_lin, release_coeff, lookahead_samples)


def variants_f014():
    return [
        # Transparent brickwall (mastering)
        dict(threshold_db=-1.0, release_ms=50.0, lookahead_ms=5.0),
        # Punchy short lookahead
        dict(threshold_db=-2.0, release_ms=30.0, lookahead_ms=1.0),
        # Aggressive 5ms lookahead
        dict(threshold_db=-3.0, release_ms=40.0, lookahead_ms=5.0),
        # Heavy ceiling
        dict(threshold_db=-6.0, release_ms=60.0, lookahead_ms=3.0),
        # Long lookahead, slow release
        dict(threshold_db=-1.5, release_ms=100.0, lookahead_ms=10.0),
        # Broadcast ceiling
        dict(threshold_db=-0.5, release_ms=80.0, lookahead_ms=5.0),
    ]


# ===================================================================
# F015 — Spectral Compressor
# ===================================================================

def effect_f015_spectral_compressor(samples: np.ndarray, sr: int, *,
                                     threshold_db: float = -20.0,
                                     ratio: float = 4.0,
                                     fft_size: int = 2048,
                                     attack_ms: float = 10.0,
                                     release_ms: float = 100.0) -> np.ndarray:
    """Spectral Compressor: compress each frequency bin independently.

    FFT-based compressor that applies gain reduction per-bin. Like a multiband
    compressor with hundreds of bands. Tames resonances and evens out the spectrum
    without affecting neighboring frequencies.

    Parameters
    ----------
    threshold_db : float [-40, 0]
        Per-bin threshold.
    ratio : float [2, 20]
        Compression ratio per bin.
    fft_size : int [512, 4096]
        FFT window size.
    attack_ms : float [1, 50]
        Smoothing attack for per-bin envelope.
    release_ms : float [10, 500]
        Smoothing release for per-bin envelope.
    """
    from scipy.signal import stft, istft
    samples = samples.astype(np.float64)
    n = len(samples)
    threshold_lin = 10.0 ** (threshold_db / 20.0)
    inv_ratio = 1.0 - 1.0 / ratio

    # Batched STFT — all frames at once
    freqs, times, Zxx = stft(samples, fs=sr, nperseg=fft_size,
                              noverlap=fft_size - fft_size // 4, window='hann')
    mag = np.abs(Zxx)  # shape: (n_bins, n_frames)
    n_bins, n_frames = mag.shape

    # Per-bin envelope smoothing across frames
    hop = fft_size // 4
    attack_c = np.exp(-1.0 / (attack_ms * 0.001 * sr / hop)) if attack_ms > 0 else 0.0
    release_c = np.exp(-1.0 / (release_ms * 0.001 * sr / hop)) if release_ms > 0 else 0.0

    bin_env = np.zeros(n_bins, dtype=np.float64)
    gains = np.ones_like(mag)

    for t in range(n_frames):
        frame_mag = mag[:, t]
        rising = frame_mag > bin_env
        bin_env = np.where(rising,
                           attack_c * bin_env + (1.0 - attack_c) * frame_mag,
                           release_c * bin_env + (1.0 - release_c) * frame_mag)
        active = (bin_env > threshold_lin) & (bin_env > 1e-12)
        if np.any(active):
            over_db = 20.0 * np.log10(bin_env[active] / threshold_lin)
            reduction_db = over_db * inv_ratio
            gains[active, t] = 10.0 ** (-reduction_db / 20.0)

    # Apply gains and reconstruct
    Zxx *= gains
    _, out = istft(Zxx, fs=sr, nperseg=fft_size,
                   noverlap=fft_size - fft_size // 4, window='hann')
    # Match original length
    if len(out) > n:
        out = out[:n]
    elif len(out) < n:
        out = np.pad(out, (0, n - len(out)))
    return out.astype(np.float32)


def variants_f015():
    return [
        # Resonance tamer
        dict(threshold_db=-18.0, ratio=6.0, fft_size=2048, attack_ms=5.0, release_ms=60.0),
        # Spectral leveler
        dict(threshold_db=-24.0, ratio=4.0, fft_size=2048, attack_ms=15.0, release_ms=150.0),
        # De-harshener (smaller FFT, faster)
        dict(threshold_db=-16.0, ratio=3.0, fft_size=1024, attack_ms=8.0, release_ms=80.0),
        # Fine spectral control
        dict(threshold_db=-22.0, ratio=8.0, fft_size=4096, attack_ms=10.0, release_ms=100.0),
        # Aggressive spectral squash
        dict(threshold_db=-30.0, ratio=12.0, fft_size=2048, attack_ms=3.0, release_ms=40.0),
        # Gentle spectral evening
        dict(threshold_db=-14.0, ratio=2.0, fft_size=2048, attack_ms=20.0, release_ms=200.0),
    ]
