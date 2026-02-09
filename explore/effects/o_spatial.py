"""O-series: Spatial effects (O001-O004).

O001 -- Haas Effect (Stereo Widener)
O002 -- Mid-Side Processing
O003 -- Binaural Panning
O004 -- Distance Simulation
"""
import numpy as np
import numba


# ---------------------------------------------------------------------------
# O001 -- Haas Effect (Stereo Widener)
# ---------------------------------------------------------------------------

@numba.njit
def _haas_kernel(samples, delay_samples, gain):
    """L = x[n], R = x[n - delay] * gain.  Returns (N, 2)."""
    n = len(samples)
    out = np.zeros((n, 2), dtype=np.float32)
    buf_len = max(delay_samples + 1, 1)
    buf = np.zeros(buf_len, dtype=np.float32)
    write_pos = 0
    for i in range(n):
        buf[write_pos] = samples[i]
        read_pos = (write_pos - delay_samples) % buf_len
        out[i, 0] = samples[i]
        out[i, 1] = buf[read_pos] * gain
        write_pos = (write_pos + 1) % buf_len
    return out


def effect_o001_haas_effect(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Haas effect stereo widener: delays one channel to create width.

    Parameters
    ----------
    haas_delay_ms : float [1, 30]
        Delay applied to the right channel in milliseconds.
    gain : float [0.5, 1.0]
        Amplitude of the delayed (right) channel.
    """
    haas_delay_ms = float(params.get('haas_delay_ms', 10))
    gain = np.float32(params.get('gain', 0.8))
    delay_samples = max(1, int(haas_delay_ms * sr / 1000.0))
    return _haas_kernel(samples.astype(np.float32), delay_samples, gain)


def variants_o001():
    return [
        # Subtle widening
        {'haas_delay_ms': 3, 'gain': 0.9},
        # Classic Haas
        {'haas_delay_ms': 10, 'gain': 0.8},
        # Wide stereo
        {'haas_delay_ms': 20, 'gain': 0.7},
        # Maximum width, near-echo territory
        {'haas_delay_ms': 30, 'gain': 1.0},
        # Tight, quiet side
        {'haas_delay_ms': 1, 'gain': 0.5},
        # Mid-range delay, full gain
        {'haas_delay_ms': 15, 'gain': 1.0},
    ]


# ---------------------------------------------------------------------------
# O002 -- Mid-Side Processing
# ---------------------------------------------------------------------------

@numba.njit
def _mid_side_kernel(samples_l, samples_r, mid_gain, side_gain):
    """Encode to M/S, adjust gains, decode back to L/R.  Returns (N, 2)."""
    n = len(samples_l)
    out = np.zeros((n, 2), dtype=np.float32)
    for i in range(n):
        mid = (samples_l[i] + samples_r[i]) * np.float32(0.5)
        side = (samples_l[i] - samples_r[i]) * np.float32(0.5)
        mid *= mid_gain
        side *= side_gain
        out[i, 0] = mid + side
        out[i, 1] = mid - side
    return out


def effect_o002_mid_side(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Mid-Side processing: adjust mid and side gains independently.

    Input is mono, so pseudo-stereo is created first by adding a slight
    delay (5 ms) to one channel before encoding to M/S.

    Parameters
    ----------
    mid_gain_db : float [-6, 6]
        Gain applied to the mid (centre) signal in dB.
    side_gain_db : float [-6, 12]
        Gain applied to the side (difference) signal in dB.
    """
    mid_gain_db = float(params.get('mid_gain_db', 0))
    side_gain_db = float(params.get('side_gain_db', 3))

    mid_gain = np.float32(10.0 ** (mid_gain_db / 20.0))
    side_gain = np.float32(10.0 ** (side_gain_db / 20.0))

    samples = samples.astype(np.float32)
    n = len(samples)

    # Create pseudo-stereo: L = original, R = delayed by 5 ms
    delay_samples = max(1, int(0.005 * sr))
    samples_l = samples.copy()
    samples_r = np.zeros(n, dtype=np.float32)
    for i in range(n):
        j = i - delay_samples
        if j >= 0:
            samples_r[i] = samples[j]

    return _mid_side_kernel(samples_l, samples_r, mid_gain, side_gain)


def variants_o002():
    return [
        # Wide: boost sides
        {'mid_gain_db': 0, 'side_gain_db': 6},
        # Narrow: cut sides
        {'mid_gain_db': 0, 'side_gain_db': -6},
        # Mid boost, subtle widening
        {'mid_gain_db': 4, 'side_gain_db': 3},
        # Extreme width
        {'mid_gain_db': -3, 'side_gain_db': 12},
        # Mono-ish: kill sides
        {'mid_gain_db': 6, 'side_gain_db': -6},
        # Balanced default
        {'mid_gain_db': 0, 'side_gain_db': 3},
    ]


# ---------------------------------------------------------------------------
# O003 -- Binaural Panning
# ---------------------------------------------------------------------------

@numba.njit
def _binaural_kernel(samples, itd_samples, ild_gain_near, ild_gain_far,
                     left_is_near):
    """Apply ITD and ILD to produce binaural stereo.  Returns (N, 2).

    The near ear gets the signal first (no delay) and louder.
    The far ear gets the signal delayed by itd_samples and attenuated.
    """
    n = len(samples)
    out = np.zeros((n, 2), dtype=np.float32)
    buf_len = max(itd_samples + 1, 1)
    buf = np.zeros(buf_len, dtype=np.float32)
    write_pos = 0

    for i in range(n):
        buf[write_pos] = samples[i]
        read_pos = (write_pos - itd_samples) % buf_len

        near_val = samples[i] * ild_gain_near
        far_val = buf[read_pos] * ild_gain_far

        if left_is_near:
            out[i, 0] = near_val
            out[i, 1] = far_val
        else:
            out[i, 0] = far_val
            out[i, 1] = near_val

        write_pos = (write_pos + 1) % buf_len
    return out


def effect_o003_binaural_panning(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Binaural panning using interaural time difference (ITD) and
    interaural level difference (ILD).

    ITD = (d / c) * sin(theta), where d = head diameter and c = speed of sound.
    ILD is approximated as a frequency-independent gain difference.

    Parameters
    ----------
    azimuth_degrees : float [-90, 90]
        Source azimuth. 0 = centre, +90 = full right, -90 = full left.
    head_size_cm : float [15, 20]
        Effective head diameter in centimetres.
    """
    azimuth_deg = float(params.get('azimuth_degrees', 45))
    head_size_cm = float(params.get('head_size_cm', 17))

    samples = samples.astype(np.float32)

    # Physics
    c = 343.0  # speed of sound in m/s
    d = head_size_cm / 100.0  # head diameter in metres
    theta_rad = np.abs(azimuth_deg) * np.pi / 180.0

    # ITD in seconds, then convert to samples
    itd_sec = (d / c) * np.sin(theta_rad)
    itd_samples = max(0, int(round(itd_sec * sr)))

    # ILD model: simple cosine-law panning plus a level boost/cut
    # Near ear is louder, far ear is quieter
    # At 90 degrees the difference is roughly 6-10 dB; we use ~8 dB max
    ild_db = 8.0 * np.sin(theta_rad)
    ild_gain_near = np.float32(10.0 ** (ild_db / 2.0 / 20.0))
    ild_gain_far = np.float32(10.0 ** (-ild_db / 2.0 / 20.0))

    # Determine which ear is near
    # Positive azimuth = right, so right ear is near
    left_is_near = azimuth_deg < 0

    return _binaural_kernel(samples, itd_samples, ild_gain_near, ild_gain_far,
                            left_is_near)


def variants_o003():
    return [
        # Centre (no panning)
        {'azimuth_degrees': 0, 'head_size_cm': 17},
        # Moderate right
        {'azimuth_degrees': 45, 'head_size_cm': 17},
        # Full right
        {'azimuth_degrees': 90, 'head_size_cm': 17},
        # Moderate left
        {'azimuth_degrees': -45, 'head_size_cm': 17},
        # Full left
        {'azimuth_degrees': -90, 'head_size_cm': 17},
        # Slight right, large head (more ITD)
        {'azimuth_degrees': 30, 'head_size_cm': 20},
        # Slight left, small head (less ITD)
        {'azimuth_degrees': -30, 'head_size_cm': 15},
    ]


# ---------------------------------------------------------------------------
# O004 -- Distance Simulation
# ---------------------------------------------------------------------------

@numba.njit
def _distance_lpf_kernel(samples, coeff):
    """One-pole lowpass filter for air absorption simulation."""
    n = len(samples)
    out = np.zeros(n, dtype=np.float32)
    y = np.float32(0.0)
    for i in range(n):
        y = coeff * y + (np.float32(1.0) - coeff) * samples[i]
        out[i] = y
    return out


@numba.njit
def _add_diffuse_reverb(samples, delay_times, gains, mix):
    """Simple multi-tap delay to approximate diffuse late reflections."""
    n = len(samples)
    max_delay = np.int64(0)
    num_taps = len(delay_times)
    for t in range(num_taps):
        if delay_times[t] > max_delay:
            max_delay = delay_times[t]
    buf_len = max(max_delay + 1, 1)
    buf = np.zeros(buf_len, dtype=np.float32)
    out = np.zeros(n, dtype=np.float32)
    write_pos = 0
    for i in range(n):
        buf[write_pos] = samples[i]
        wet = np.float32(0.0)
        for t in range(num_taps):
            read_pos = (write_pos - delay_times[t]) % buf_len
            wet += gains[t] * buf[read_pos]
        out[i] = (np.float32(1.0) - mix) * samples[i] + mix * wet
        write_pos = (write_pos + 1) % buf_len
    return out


def effect_o004_distance_simulation(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Distance simulation: amplitude falloff (1/distance), lowpass for air
    absorption, and diffuse reverb that increases with distance.

    Parameters
    ----------
    distance : float [0.5, 100]
        Simulated source distance in metres.  1.0 = unity gain reference.
    """
    distance = float(params.get('distance', 10))
    distance = max(0.5, distance)

    samples = samples.astype(np.float32)

    # 1. Amplitude: inverse-distance law, normalised so distance=1 => gain=1
    amplitude = np.float32(1.0 / distance)
    attenuated = samples * amplitude

    # 2. Air absorption lowpass: cutoff decreases with distance
    #    At 1m -> 20 kHz (essentially bypass), at 100m -> ~1 kHz
    cutoff_hz = max(200.0, 20000.0 / (1.0 + (distance - 1.0) * 0.2))
    cutoff_hz = min(cutoff_hz, sr * 0.499)
    dt = 1.0 / sr
    rc = 1.0 / (2.0 * np.pi * cutoff_hz)
    coeff = np.float32(np.exp(-dt / rc))
    filtered = _distance_lpf_kernel(attenuated, coeff)

    # 3. Reverb mix: increases with distance (more diffuse at distance)
    reverb_mix = np.float32(min(0.8, 0.05 + (distance - 1.0) * 0.008))
    if reverb_mix < np.float32(0.0):
        reverb_mix = np.float32(0.0)

    # Multi-tap delays using prime-ish spacing for diffusion
    tap_delays_ms = np.array([11.0, 23.0, 37.0, 53.0, 71.0, 97.0],
                             dtype=np.float64)
    tap_delays = np.zeros(len(tap_delays_ms), dtype=np.int64)
    tap_gains = np.zeros(len(tap_delays_ms), dtype=np.float32)
    for t in range(len(tap_delays_ms)):
        tap_delays[t] = max(1, int(tap_delays_ms[t] * sr / 1000.0))
        # Decaying gains
        tap_gains[t] = np.float32(0.7 ** (t + 1))

    return _add_diffuse_reverb(filtered, tap_delays, tap_gains, reverb_mix)


def variants_o004():
    return [
        # Intimate / close-up
        {'distance': 0.5},
        # Arm's length
        {'distance': 2},
        # Across the room
        {'distance': 10},
        # Down the hall
        {'distance': 30},
        # Distant outdoor
        {'distance': 60},
        # Far away
        {'distance': 100},
    ]
