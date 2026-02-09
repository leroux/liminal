"""M-series: Physical modeling effects (M001-M005).

Karplus-Strong, waveguide string, mass-spring chain, membrane/drum
resonator, tube resonator. All use physical simulation to color the input.
"""
import numpy as np
import numba


# ---------------------------------------------------------------------------
# M001 -- Karplus-Strong as Effect
# ---------------------------------------------------------------------------

@numba.njit
def _karplus_strong_kernel(samples, delay_len, decay_factor):
    n = len(samples)
    # Ring buffer sized to the KS period
    buf_len = max(delay_len + 2, 2)
    buf = np.zeros(buf_len, dtype=np.float32)
    out = np.zeros(n, dtype=np.float32)

    # Initialize buffer with input (first delay_len samples)
    init_len = min(delay_len, n)
    for i in range(init_len):
        buf[i] = samples[i]

    write_pos = 0
    for i in range(n):
        # Read from delay_len samples back
        read_pos_0 = (write_pos - delay_len) % buf_len
        read_pos_1 = (write_pos - delay_len - 1) % buf_len
        # KS averaging filter
        y = decay_factor * np.float32(0.5) * (buf[read_pos_0] + buf[read_pos_1])
        # Mix input with KS output
        y += samples[i]
        # Clamp to prevent explosion
        if y > np.float32(10.0):
            y = np.float32(10.0)
        elif y < np.float32(-10.0):
            y = np.float32(-10.0)
        buf[write_pos] = y
        out[i] = y
        write_pos = (write_pos + 1) % buf_len
    return out


def effect_m001_karplus_strong(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Karplus-Strong as effect: feed input into KS delay line with averaging filter."""
    freq_hz = np.float32(params.get('freq_hz', 220.0))
    decay_factor = np.float32(params.get('decay_factor', 0.99))
    delay_len = max(1, int(np.float32(sr) / freq_hz))
    return _karplus_strong_kernel(samples.astype(np.float32), delay_len, decay_factor)


def variants_m001():
    return [
        {'freq_hz': 80, 'decay_factor': 0.995},     # deep bass drone
        {'freq_hz': 150, 'decay_factor': 0.99},      # low pluck coloring
        {'freq_hz': 220, 'decay_factor': 0.98},       # mid-range metallic ring
        {'freq_hz': 440, 'decay_factor': 0.97},       # bright pluck resonance
        {'freq_hz': 880, 'decay_factor': 0.95},       # high-pitched ping
        {'freq_hz': 1500, 'decay_factor': 0.92},      # very bright, short decay
        {'freq_hz': 55, 'decay_factor': 0.999},       # sub-bass, long sustain
    ]


# ---------------------------------------------------------------------------
# M002 -- Waveguide String Resonator
# ---------------------------------------------------------------------------

@numba.njit
def _waveguide_string_kernel(samples, upper_len, lower_len, excite_pos_upper,
                              excite_pos_lower, decay, brightness):
    n = len(samples)
    # Two rail delay lines
    upper_buf = np.zeros(upper_len, dtype=np.float32)
    lower_buf = np.zeros(lower_len, dtype=np.float32)
    out = np.zeros(n, dtype=np.float32)

    # One-pole lowpass coefficient for brightness at nut reflection
    # brightness=1 -> no filtering, brightness=0 -> heavy filtering
    lp_coeff = brightness
    lp_state_upper = np.float32(0.0)
    lp_state_lower = np.float32(0.0)

    upper_write = 0
    lower_write = 0

    for i in range(n):
        # Read from end of upper rail (nut end)
        upper_read = (upper_write - upper_len + 1) % upper_len
        upper_end = upper_buf[upper_read]

        # Read from end of lower rail (bridge end)
        lower_read = (lower_write - lower_len + 1) % lower_len
        lower_end = lower_buf[lower_read]

        # Nut reflection: invert + lowpass (brightness control)
        lp_state_upper = lp_coeff * upper_end + (np.float32(1.0) - lp_coeff) * lp_state_upper
        nut_reflect = -decay * lp_state_upper

        # Bridge reflection: invert + slight lowpass
        lp_state_lower = lp_coeff * lower_end + (np.float32(1.0) - lp_coeff) * lp_state_lower
        bridge_reflect = -decay * lp_state_lower

        # Feed reflections back: nut reflection -> lower rail, bridge -> upper rail
        lower_buf[lower_write] = nut_reflect
        upper_buf[upper_write] = bridge_reflect

        # Inject input at excitation position on both rails
        excite_upper_write = (upper_write + excite_pos_upper) % upper_len
        excite_lower_write = (lower_write + excite_pos_lower) % lower_len
        upper_buf[excite_upper_write] += samples[i] * np.float32(0.5)
        lower_buf[excite_lower_write] += samples[i] * np.float32(0.5)

        # Output from bridge (sum of both rails at bridge)
        y = upper_end + lower_end
        # Clamp
        if y > np.float32(10.0):
            y = np.float32(10.0)
        elif y < np.float32(-10.0):
            y = np.float32(-10.0)
        out[i] = y

        upper_write = (upper_write + 1) % upper_len
        lower_write = (lower_write + 1) % lower_len

    return out


def effect_m002_waveguide_string(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Waveguide string resonator: two delay lines (upper/lower rail) with boundary reflections."""
    freq_hz = np.float32(params.get('freq_hz', 220.0))
    decay = np.float32(params.get('decay', 0.995))
    brightness = np.float32(params.get('brightness', 0.5))
    excitation_position = np.float32(params.get('excitation_position', 0.3))

    # Total delay = sr / freq_hz, split between upper and lower rails
    total_delay = max(4, int(np.float32(sr) / freq_hz))
    upper_len = max(2, total_delay // 2)
    lower_len = max(2, total_delay - upper_len)

    # Excitation position as fraction of each rail
    excite_pos_upper = max(1, int(excitation_position * upper_len))
    excite_pos_lower = max(1, int(excitation_position * lower_len))

    return _waveguide_string_kernel(
        samples.astype(np.float32), upper_len, lower_len,
        excite_pos_upper, excite_pos_lower, decay, brightness
    )


def variants_m002():
    return [
        {'freq_hz': 110, 'decay': 0.998, 'brightness': 0.3, 'excitation_position': 0.5},   # dark low string
        {'freq_hz': 220, 'decay': 0.995, 'brightness': 0.5, 'excitation_position': 0.3},   # warm mid string
        {'freq_hz': 330, 'decay': 0.99, 'brightness': 0.7, 'excitation_position': 0.15},    # bright string, near bridge
        {'freq_hz': 440, 'decay': 0.985, 'brightness': 0.8, 'excitation_position': 0.1},    # brilliant high string
        {'freq_hz': 82, 'decay': 0.999, 'brightness': 0.2, 'excitation_position': 0.5},     # deep bass string, center excite
        {'freq_hz': 660, 'decay': 0.97, 'brightness': 0.9, 'excitation_position': 0.25},    # high, very bright, short
        {'freq_hz': 150, 'decay': 0.997, 'brightness': 0.4, 'excitation_position': 0.7},    # low, near nut excitation
    ]


# ---------------------------------------------------------------------------
# M003 -- Mass-Spring Damper Chain
# ---------------------------------------------------------------------------

@numba.njit
def _mass_spring_kernel(samples, num_masses, stiffness, damping, mass, sr):
    n = len(samples)
    out = np.zeros(n, dtype=np.float32)

    # State: position and velocity for each mass
    pos = np.zeros(num_masses, dtype=np.float32)
    vel = np.zeros(num_masses, dtype=np.float32)
    force = np.zeros(num_masses, dtype=np.float32)

    # Scale stiffness to get audio-rate resonances.
    # Natural freq of chain mode k: f_k = (1/(2*pi)) * 2*sqrt(K/M) * sin(k*pi/(2*(N+1)))
    # Lowest mode ~ sqrt(K/M) * pi / (N+1) / (2*pi) = sqrt(K/M) / (2*(N+1))
    # To get ~200 Hz lowest mode with N=8: K/M ~ (200 * 2 * 9)^2 ~ 1.3e7
    # So we scale stiffness by sr^2 to bring it into audio range
    sr_f = np.float32(sr)
    k_scaled = stiffness * sr_f * sr_f * np.float32(0.001)
    d_scaled = damping * sr_f * np.float32(2.0)
    inv_mass = np.float32(1.0) / mass

    dt = np.float32(1.0) / sr_f

    # Substeps for stability: CFL condition dt < sqrt(m/k)
    # With our scaled k, we need substeps = ceil(dt * sqrt(k_scaled / mass) * 2)
    omega_max = np.float32(np.sqrt(np.float64(k_scaled * np.float32(4.0) * inv_mass)))
    num_substeps = max(1, int(omega_max * np.float64(dt) * np.float64(1.5)))
    if num_substeps > 32:
        num_substeps = 32
    sub_dt = dt / np.float32(num_substeps)

    for i in range(n):
        # Drive first mass with input
        drive_force = samples[i] * k_scaled

        for _ in range(num_substeps):
            # Compute forces on all masses
            for m in range(num_masses):
                if m == 0:
                    left_pos = np.float32(0.0)
                else:
                    left_pos = pos[m - 1]
                if m == num_masses - 1:
                    right_pos = np.float32(0.0)
                else:
                    right_pos = pos[m + 1]

                spring_f = k_scaled * (left_pos - pos[m]) + k_scaled * (right_pos - pos[m])
                damp_f = -d_scaled * vel[m]
                ext_f = np.float32(0.0)
                if m == 0:
                    ext_f = drive_force
                force[m] = spring_f + damp_f + ext_f

            # Update velocities then positions (symplectic Euler)
            for m in range(num_masses):
                vel[m] += force[m] * inv_mass * sub_dt
                if vel[m] > np.float32(10.0):
                    vel[m] = np.float32(10.0)
                elif vel[m] < np.float32(-10.0):
                    vel[m] = np.float32(-10.0)

            for m in range(num_masses):
                pos[m] += vel[m] * sub_dt
                if pos[m] > np.float32(10.0):
                    pos[m] = np.float32(10.0)
                elif pos[m] < np.float32(-10.0):
                    pos[m] = np.float32(-10.0)

        # Output from last mass
        out[i] = pos[num_masses - 1]

    return out


def effect_m003_mass_spring_chain(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Mass-spring damper chain: N masses connected by springs with Euler integration.
    Input drives first mass, output from last mass."""
    num_masses = int(params.get('num_masses', 8))
    stiffness = np.float32(params.get('stiffness', 1.0))
    damping = np.float32(params.get('damping', 0.1))
    mass = np.float32(params.get('mass', 0.5))
    out = _mass_spring_kernel(
        samples.astype(np.float32), num_masses, stiffness, damping, mass, sr
    )
    # Auto-gain: normalize output to match input peak level
    # The physical displacement scale varies wildly with parameters
    out_peak = np.max(np.abs(out))
    in_peak = np.max(np.abs(samples))
    if out_peak > np.float32(1e-10) and in_peak > np.float32(1e-10):
        out = out * (in_peak / out_peak)
    return out


def variants_m003():
    return [
        {'num_masses': 3, 'stiffness': 0.5, 'damping': 0.05, 'mass': 0.3},   # short chain, low stiffness, ringy
        {'num_masses': 8, 'stiffness': 1.0, 'damping': 0.1, 'mass': 0.5},     # medium chain, balanced
        {'num_masses': 15, 'stiffness': 2.0, 'damping': 0.05, 'mass': 0.2},   # long chain, high stiffness, metallic
        {'num_masses': 5, 'stiffness': 4.0, 'damping': 0.3, 'mass': 1.0},     # stiff, heavily damped, thud
        {'num_masses': 20, 'stiffness': 0.3, 'damping': 0.02, 'mass': 1.5},   # long, floppy, slow propagation
        {'num_masses': 10, 'stiffness': 3.0, 'damping': 0.5, 'mass': 0.1},    # light masses, fast, damped
        {'num_masses': 6, 'stiffness': 0.8, 'damping': 0.01, 'mass': 0.4},    # low damping, long sustain
    ]


# ---------------------------------------------------------------------------
# M004 -- Membrane / Drum Resonator (2D waveguide mesh)
# ---------------------------------------------------------------------------

@numba.njit
def _membrane_kernel(samples, grid_size, tension, damping):
    n = len(samples)
    out = np.zeros(n, dtype=np.float32)

    # Courant number: c = tension, must be <= 0.5 for 2D stability
    c = tension
    if c > np.float32(0.5):
        c = np.float32(0.5)
    c2 = c * c

    # Two time steps of the 2D grid (current and previous)
    grid_curr = np.zeros((grid_size, grid_size), dtype=np.float32)
    grid_prev = np.zeros((grid_size, grid_size), dtype=np.float32)
    grid_next = np.zeros((grid_size, grid_size), dtype=np.float32)

    # Input injection at center
    cx = grid_size // 2
    cy = grid_size // 2
    # Output read from edge
    ox = grid_size - 2
    oy = grid_size // 2

    damp = np.float32(1.0) - damping

    for i in range(n):
        # Inject input at center
        grid_curr[cx, cy] += samples[i] * np.float32(0.5)

        # Update interior grid points using finite difference wave equation
        for gx in range(1, grid_size - 1):
            for gy in range(1, grid_size - 1):
                laplacian = (grid_curr[gx + 1, gy] + grid_curr[gx - 1, gy]
                             + grid_curr[gx, gy + 1] + grid_curr[gx, gy - 1]
                             - np.float32(4.0) * grid_curr[gx, gy])
                grid_next[gx, gy] = (np.float32(2.0) * grid_curr[gx, gy]
                                     - grid_prev[gx, gy]
                                     + c2 * laplacian) * damp

        # Boundary: fixed edges (Dirichlet) - grid_next edges stay 0

        # Clamp values to prevent explosion
        for gx in range(grid_size):
            for gy in range(grid_size):
                if grid_next[gx, gy] > np.float32(10.0):
                    grid_next[gx, gy] = np.float32(10.0)
                elif grid_next[gx, gy] < np.float32(-10.0):
                    grid_next[gx, gy] = np.float32(-10.0)

        # Read output from edge point
        out[i] = grid_next[ox, oy]

        # Swap grids: prev = curr, curr = next
        for gx in range(grid_size):
            for gy in range(grid_size):
                grid_prev[gx, gy] = grid_curr[gx, gy]
                grid_curr[gx, gy] = grid_next[gx, gy]
                grid_next[gx, gy] = np.float32(0.0)

    return out


def effect_m004_membrane_resonator(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Membrane/drum resonator: 2D waveguide mesh with fixed boundary.
    Input at center, output at edge."""
    grid_size = int(params.get('grid_size', 8))
    # Clamp grid_size to [5, 20] for performance
    if grid_size < 5:
        grid_size = 5
    if grid_size > 20:
        grid_size = 20
    tension = np.float32(params.get('tension', 0.3))
    damping = np.float32(params.get('damping', 0.01))
    return _membrane_kernel(samples.astype(np.float32), grid_size, tension, damping)


def variants_m004():
    return [
        {'grid_size': 5, 'tension': 0.4, 'damping': 0.005},    # small tight membrane, high pitch
        {'grid_size': 8, 'tension': 0.3, 'damping': 0.01},      # medium drum, balanced
        {'grid_size': 12, 'tension': 0.2, 'damping': 0.008},    # large membrane, lower pitch
        {'grid_size': 6, 'tension': 0.45, 'damping': 0.002},    # small, very tight, long ring
        {'grid_size': 15, 'tension': 0.15, 'damping': 0.03},    # large, loose, short decay
        {'grid_size': 10, 'tension': 0.35, 'damping': 0.05},    # medium, heavily damped, thud
        {'grid_size': 7, 'tension': 0.1, 'damping': 0.005},     # small, low tension, flabby
    ]


# ---------------------------------------------------------------------------
# M005 -- Tube Resonator (1D waveguide with scattering junctions)
# ---------------------------------------------------------------------------

@numba.njit
def _tube_resonator_kernel(samples, segment_lengths, scattering_coeffs, num_segments,
                            total_delay, damping):
    n = len(samples)
    out = np.zeros(n, dtype=np.float32)

    # Forward and backward traveling waves for each segment
    # Each segment has a delay line for forward and backward propagation
    max_seg_len = 1
    for s in range(num_segments):
        if segment_lengths[s] > max_seg_len:
            max_seg_len = segment_lengths[s]

    # Allocate delay buffers for each segment (forward and backward)
    # Using flat arrays: segment s occupies indices [s*max_seg_len : (s+1)*max_seg_len]
    fwd_bufs = np.zeros(num_segments * max_seg_len, dtype=np.float32)
    bwd_bufs = np.zeros(num_segments * max_seg_len, dtype=np.float32)
    fwd_write = np.zeros(num_segments, dtype=np.int64)
    bwd_write = np.zeros(num_segments, dtype=np.int64)

    damp = np.float32(1.0) - damping

    for i in range(n):
        # Inject input into forward wave of first segment
        seg0_offset = 0
        seg0_len = segment_lengths[0]
        fwd_bufs[seg0_offset + fwd_write[0]] += samples[i]

        # Process each junction between segments
        for s in range(num_segments - 1):
            seg_offset = s * max_seg_len
            seg_len = segment_lengths[s]
            next_seg_offset = (s + 1) * max_seg_len
            next_seg_len = segment_lengths[s + 1]

            # Read from end of forward delay of segment s
            fwd_read = (fwd_write[s] - seg_len + 1) % seg_len
            if fwd_read < 0:
                fwd_read += seg_len
            p_plus = fwd_bufs[seg_offset + fwd_read]

            # Read from beginning of backward delay of segment s+1
            bwd_read = (bwd_write[s + 1] - next_seg_len + 1) % next_seg_len
            if bwd_read < 0:
                bwd_read += next_seg_len
            p_minus = bwd_bufs[next_seg_offset + bwd_read]

            # Scattering junction
            k = scattering_coeffs[s]
            # Transmitted and reflected waves
            # p_plus_out = (1+k)/2 * p_plus + (1-k)/2 * p_minus  (transmitted forward)
            # p_minus_out = (1-k)/2 * p_plus + (1+k)/2 * p_minus  (reflected backward)
            half_1pk = np.float32(0.5) * (np.float32(1.0) + k)
            half_1mk = np.float32(0.5) * (np.float32(1.0) - k)

            fwd_out = half_1pk * p_plus + half_1mk * p_minus
            bwd_out = half_1mk * p_plus + half_1pk * p_minus

            # Apply damping
            fwd_out *= damp
            bwd_out *= damp

            # Clamp
            if fwd_out > np.float32(10.0):
                fwd_out = np.float32(10.0)
            elif fwd_out < np.float32(-10.0):
                fwd_out = np.float32(-10.0)
            if bwd_out > np.float32(10.0):
                bwd_out = np.float32(10.0)
            elif bwd_out < np.float32(-10.0):
                bwd_out = np.float32(-10.0)

            # Write transmitted forward into next segment
            fwd_bufs[next_seg_offset + fwd_write[s + 1]] = fwd_out
            # Write reflected backward into current segment
            bwd_bufs[seg_offset + bwd_write[s]] = bwd_out

        # End reflections
        # Open end at output (last segment): partial reflection, inverted
        last_s = num_segments - 1
        last_offset = last_s * max_seg_len
        last_len = segment_lengths[last_s]
        last_fwd_read = (fwd_write[last_s] - last_len + 1) % last_len
        if last_fwd_read < 0:
            last_fwd_read += last_len
        end_out = fwd_bufs[last_offset + last_fwd_read]

        # Output: radiated wave from open end
        out[i] = end_out

        # Reflect back (open end: inversion, partial reflection)
        reflected = -end_out * np.float32(0.6) * damp
        bwd_bufs[last_offset + bwd_write[last_s]] = reflected

        # Closed end at input (first segment): reflect backward wave with no inversion
        first_offset = 0
        first_len = segment_lengths[0]
        first_bwd_read = (bwd_write[0] - first_len + 1) % first_len
        if first_bwd_read < 0:
            first_bwd_read += first_len
        closed_reflect = bwd_bufs[first_offset + first_bwd_read] * np.float32(0.8) * damp
        fwd_bufs[first_offset + fwd_write[0]] += closed_reflect

        # Advance write positions
        for s in range(num_segments):
            fwd_write[s] = (fwd_write[s] + 1) % segment_lengths[s]
            bwd_write[s] = (bwd_write[s] + 1) % segment_lengths[s]

    return out


def effect_m005_tube_resonator(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Tube resonator: 1D waveguide with varying cross-section and scattering junctions.
    Scattering coefficients depend on cross-section ratios between adjacent segments."""
    num_segments = int(params.get('num_segments', 5))
    tube_length_ms = np.float32(params.get('tube_length_ms', 20.0))

    # Clamp num_segments
    if num_segments < 3:
        num_segments = 3
    if num_segments > 10:
        num_segments = 10

    # Total delay in samples
    total_delay = max(num_segments * 2, int(tube_length_ms * np.float32(sr) / np.float32(1000.0)))
    base_seg_len = max(2, total_delay // num_segments)

    # Create varying cross-sections (conical-ish profile)
    # Cross-section varies from narrow at input to wider at bell
    cross_sections = np.zeros(num_segments, dtype=np.float32)
    for s in range(num_segments):
        # Exponential flare: narrow -> wide
        t = np.float32(s) / np.float32(num_segments - 1) if num_segments > 1 else np.float32(0.5)
        cross_sections[s] = np.float32(1.0) + np.float32(3.0) * t * t

    # Compute segment lengths (slight variation for richer modes)
    segment_lengths = np.zeros(num_segments, dtype=np.int64)
    for s in range(num_segments):
        # Vary length slightly based on cross-section
        segment_lengths[s] = max(2, int(base_seg_len * (np.float32(0.8) + np.float32(0.4) * cross_sections[s] / np.float32(4.0))))

    # Compute scattering coefficients from cross-section ratios
    # k = (S_{i+1} - S_i) / (S_{i+1} + S_i)
    scattering_coeffs = np.zeros(num_segments, dtype=np.float32)
    for s in range(num_segments - 1):
        s1 = cross_sections[s]
        s2 = cross_sections[s + 1]
        scattering_coeffs[s] = (s2 - s1) / (s2 + s1)

    # Damping: per-junction loss, scaled to tube length
    # Shorter tubes need more damping per sample to avoid resonance buildup
    damping = np.float32(0.005)

    out = _tube_resonator_kernel(
        samples.astype(np.float32), segment_lengths, scattering_coeffs,
        num_segments, total_delay, damping
    )
    # Normalize to prevent excessive amplitude from resonance buildup
    peak = np.max(np.abs(out))
    in_peak = np.max(np.abs(samples))
    if peak > np.float32(0.0) and in_peak > np.float32(0.0):
        target_peak = in_peak * np.float32(2.0)  # allow up to 2x gain
        if peak > target_peak:
            out = out * (target_peak / peak)
    return out


def variants_m005():
    return [
        {'num_segments': 3, 'tube_length_ms': 10},    # short tube, bright, nasal
        {'num_segments': 5, 'tube_length_ms': 20},     # medium tube, balanced resonance
        {'num_segments': 8, 'tube_length_ms': 35},     # long tube, rich harmonics
        {'num_segments': 4, 'tube_length_ms': 5},      # very short, high-pitched resonance
        {'num_segments': 10, 'tube_length_ms': 50},    # long complex tube, deep formants
        {'num_segments': 6, 'tube_length_ms': 15},     # medium-short, tight resonance
    ]
