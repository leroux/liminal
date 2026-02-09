"""B-series effects: Reverb algorithms (B001-B011)."""
import numpy as np
import numba


# ---------------------------------------------------------------------------
# B001 — Schroeder Reverb
# 4 parallel comb filters -> 2 series allpass filters
# ---------------------------------------------------------------------------

@numba.njit
def _schroeder_reverb_kernel(samples, sr, rt60, wet_mix):
    n = len(samples)
    # Comb filter delays in seconds -> samples
    comb_delays_ms = np.array([29.7, 37.1, 41.1, 43.7], dtype=np.float64)
    comb_delays = np.empty(4, dtype=np.int32)
    comb_g = np.empty(4, dtype=np.float32)
    for i in range(4):
        d = int(comb_delays_ms[i] * 0.001 * sr)
        comb_delays[i] = d
        # g = 10^(-3 * d / (sr * rt60))
        comb_g[i] = np.float32(10.0 ** (-3.0 * d / (sr * rt60)))

    # Allpass delays
    ap_delays_ms = np.array([5.0, 1.7], dtype=np.float64)
    ap_delays = np.empty(2, dtype=np.int32)
    ap_g = np.empty(2, dtype=np.float32)
    for i in range(2):
        d = int(ap_delays_ms[i] * 0.001 * sr)
        ap_delays[i] = d
        ap_g[i] = np.float32(0.7)

    # Allocate comb buffers
    max_comb = 0
    for i in range(4):
        if comb_delays[i] > max_comb:
            max_comb = comb_delays[i]
    comb_bufs = np.zeros((4, max_comb + 1), dtype=np.float32)
    comb_idx = np.zeros(4, dtype=np.int32)

    # Allocate allpass buffers
    max_ap = 0
    for i in range(2):
        if ap_delays[i] > max_ap:
            max_ap = ap_delays[i]
    ap_bufs = np.zeros((2, max_ap + 1), dtype=np.float32)
    ap_idx = np.zeros(2, dtype=np.int32)

    out = np.zeros(n, dtype=np.float32)

    for i in range(n):
        x = samples[i]

        # 4 parallel comb filters summed
        comb_sum = np.float32(0.0)
        for c in range(4):
            dl = comb_delays[c]
            read_pos = (comb_idx[c] - dl) % dl
            delayed = comb_bufs[c, read_pos]
            val = x + comb_g[c] * delayed
            comb_bufs[c, comb_idx[c] % dl] = val
            comb_idx[c] += 1
            comb_sum += delayed

        # Scale comb output
        y = comb_sum * np.float32(0.25)

        # 2 series allpass filters
        for a in range(2):
            dl = ap_delays[a]
            read_pos = (ap_idx[a] - dl) % dl
            delayed = ap_bufs[a, read_pos]
            g = ap_g[a]
            v = y - g * delayed
            ap_bufs[a, ap_idx[a] % dl] = v
            ap_idx[a] += 1
            y = delayed + g * v

        out[i] = (np.float32(1.0) - wet_mix) * x + wet_mix * y

    return out


def effect_b001_schroeder_reverb(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    rt60 = np.float32(params.get('rt60', 1.5))
    wet_mix = np.float32(params.get('wet_mix', 0.5))
    return _schroeder_reverb_kernel(
        samples.astype(np.float32), sr, rt60, wet_mix
    )


def variants_b001():
    return [
        {'rt60': 0.5, 'wet_mix': 0.3},
        {'rt60': 1.0, 'wet_mix': 0.5},
        {'rt60': 2.0, 'wet_mix': 0.5},
        {'rt60': 3.5, 'wet_mix': 0.6},
        {'rt60': 5.0, 'wet_mix': 0.8},
    ]


# ---------------------------------------------------------------------------
# B002 — Moorer Reverb
# 6 comb filters with one-pole lowpass in feedback -> allpass chain
# ---------------------------------------------------------------------------

@numba.njit
def _moorer_reverb_kernel(samples, sr, rt60, damping, wet_mix):
    n = len(samples)

    # 6 comb filter delays (mutually prime-ish sample counts)
    comb_delays_ms = np.array([50.0, 56.0, 61.0, 68.0, 72.0, 78.0], dtype=np.float64)
    num_combs = 6
    comb_delays = np.empty(num_combs, dtype=np.int32)
    comb_g = np.empty(num_combs, dtype=np.float32)
    for i in range(num_combs):
        d = int(comb_delays_ms[i] * 0.001 * sr)
        if d < 1:
            d = 1
        comb_delays[i] = d
        comb_g[i] = np.float32(10.0 ** (-3.0 * d / (sr * rt60)))

    max_comb = 0
    for i in range(num_combs):
        if comb_delays[i] > max_comb:
            max_comb = comb_delays[i]

    comb_bufs = np.zeros((num_combs, max_comb + 1), dtype=np.float32)
    comb_idx = np.zeros(num_combs, dtype=np.int32)
    comb_lp_state = np.zeros(num_combs, dtype=np.float32)

    # One-pole lowpass coefficient from damping
    lp_coeff = np.float32(damping)

    # 1 allpass for diffusion after combs
    ap_delay = int(0.005 * sr)
    if ap_delay < 1:
        ap_delay = 1
    ap_buf = np.zeros(ap_delay, dtype=np.float32)
    ap_idx = np.int32(0)
    ap_g = np.float32(0.7)

    out = np.zeros(n, dtype=np.float32)

    for i in range(n):
        x = samples[i]
        comb_sum = np.float32(0.0)

        for c in range(num_combs):
            dl = comb_delays[c]
            read_pos = (comb_idx[c] - dl + max_comb + 1) % (max_comb + 1)
            delayed = comb_bufs[c, read_pos]

            # One-pole lowpass in feedback
            comb_lp_state[c] = lp_coeff * comb_lp_state[c] + (np.float32(1.0) - lp_coeff) * delayed
            filtered = comb_lp_state[c]

            val = x + comb_g[c] * filtered
            write_pos = comb_idx[c] % (max_comb + 1)
            comb_bufs[c, write_pos] = val
            comb_idx[c] += 1
            comb_sum += delayed

        y = comb_sum / np.float32(num_combs)

        # Allpass
        read_pos = ap_idx % ap_delay
        delayed_ap = ap_buf[read_pos]
        v = y - ap_g * delayed_ap
        ap_buf[ap_idx % ap_delay] = v
        ap_idx += 1
        y = delayed_ap + ap_g * v

        out[i] = (np.float32(1.0) - wet_mix) * x + wet_mix * y

    return out


def effect_b002_moorer_reverb(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    rt60 = np.float32(params.get('rt60', 2.0))
    damping = np.float32(params.get('damping', 0.5))
    wet_mix = np.float32(params.get('wet_mix', 0.5))
    return _moorer_reverb_kernel(
        samples.astype(np.float32), sr, rt60, damping, wet_mix
    )


def variants_b002():
    return [
        {'rt60': 0.5, 'damping': 0.3, 'wet_mix': 0.3},
        {'rt60': 1.5, 'damping': 0.5, 'wet_mix': 0.5},
        {'rt60': 3.0, 'damping': 0.7, 'wet_mix': 0.5},
        {'rt60': 5.0, 'damping': 0.9, 'wet_mix': 0.6},
        {'rt60': 8.0, 'damping': 0.4, 'wet_mix': 0.8},
    ]


# ---------------------------------------------------------------------------
# B003 — FDN Reverb
# N delay lines with NxN Hadamard feedback matrix, lowpass in feedback
# ---------------------------------------------------------------------------

@numba.njit
def _hadamard4():
    """Return 4x4 Hadamard matrix scaled by 1/2."""
    h = np.array([
        [1.0, 1.0, 1.0, 1.0],
        [1.0, -1.0, 1.0, -1.0],
        [1.0, 1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0, 1.0],
    ], dtype=np.float32)
    return h * np.float32(0.5)


@numba.njit
def _hadamard8():
    """Return 8x8 Hadamard matrix scaled by 1/sqrt(8)."""
    h4 = np.array([
        [1.0, 1.0, 1.0, 1.0],
        [1.0, -1.0, 1.0, -1.0],
        [1.0, 1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0, 1.0],
    ], dtype=np.float32)
    h8 = np.zeros((8, 8), dtype=np.float32)
    for i in range(4):
        for j in range(4):
            h8[i, j] = h4[i, j]
            h8[i, j + 4] = h4[i, j]
            h8[i + 4, j] = h4[i, j]
            h8[i + 4, j + 4] = -h4[i, j]
    scale = np.float32(1.0 / np.sqrt(8.0))
    for i in range(8):
        for j in range(8):
            h8[i, j] *= scale
    return h8


@numba.njit
def _fdn_reverb_kernel(samples, sr, n_delays, rt60, damping, wet_mix):
    n = len(samples)

    # Mutually prime delay lengths in samples
    if n_delays == 8:
        delay_ms = np.array([29.7, 37.1, 41.1, 43.7, 53.0, 59.9, 67.3, 73.1], dtype=np.float64)
        mat = _hadamard8()
    else:
        delay_ms = np.array([29.7, 37.1, 41.1, 43.7], dtype=np.float64)
        mat = _hadamard4()
        n_delays = 4

    delays = np.empty(n_delays, dtype=np.int32)
    gains = np.empty(n_delays, dtype=np.float32)
    for i in range(n_delays):
        d = int(delay_ms[i] * 0.001 * sr)
        if d < 1:
            d = 1
        delays[i] = d
        gains[i] = np.float32(10.0 ** (-3.0 * d / (sr * rt60)))

    max_delay = 0
    for i in range(n_delays):
        if delays[i] > max_delay:
            max_delay = delays[i]

    bufs = np.zeros((n_delays, max_delay + 1), dtype=np.float32)
    write_idx = np.zeros(n_delays, dtype=np.int32)
    lp_state = np.zeros(n_delays, dtype=np.float32)
    lp_coeff = np.float32(damping)

    out = np.zeros(n, dtype=np.float32)

    for i in range(n):
        x = samples[i]

        # Read from delay lines
        delayed = np.empty(n_delays, dtype=np.float32)
        for d in range(n_delays):
            dl = delays[d]
            read_pos = (write_idx[d] - dl + max_delay + 1) % (max_delay + 1)
            delayed[d] = bufs[d, read_pos]

        # Apply feedback matrix
        feedback = np.zeros(n_delays, dtype=np.float32)
        for r in range(n_delays):
            s = np.float32(0.0)
            for c in range(n_delays):
                s += mat[r, c] * delayed[c]
            feedback[r] = s

        # Apply gain and lowpass, write back
        out_sum = np.float32(0.0)
        for d in range(n_delays):
            # Lowpass in feedback
            lp_state[d] = lp_coeff * lp_state[d] + (np.float32(1.0) - lp_coeff) * feedback[d]
            val = x + gains[d] * lp_state[d]
            write_pos = write_idx[d] % (max_delay + 1)
            bufs[d, write_pos] = val
            write_idx[d] += 1
            out_sum += delayed[d]

        y = out_sum / np.float32(n_delays)
        out[i] = (np.float32(1.0) - wet_mix) * x + wet_mix * y

    return out


def effect_b003_fdn_reverb(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    n_delays = int(params.get('n_delays', 4))
    rt60 = np.float32(params.get('rt60', 2.0))
    damping = np.float32(params.get('damping', 0.3))
    wet_mix = np.float32(params.get('wet_mix', 0.5))
    return _fdn_reverb_kernel(
        samples.astype(np.float32), sr, n_delays, rt60, damping, wet_mix
    )


def variants_b003():
    return [
        {'n_delays': 4, 'rt60': 1.0, 'damping': 0.2, 'wet_mix': 0.4},
        {'n_delays': 4, 'rt60': 3.0, 'damping': 0.5, 'wet_mix': 0.5},
        {'n_delays': 8, 'rt60': 2.0, 'damping': 0.3, 'wet_mix': 0.5},
        {'n_delays': 8, 'rt60': 5.0, 'damping': 0.6, 'wet_mix': 0.6},
        {'n_delays': 8, 'rt60': 10.0, 'damping': 0.8, 'wet_mix': 0.7},
        {'n_delays': 8, 'rt60': 20.0, 'damping': 0.9, 'wet_mix': 0.8},
    ]


# ---------------------------------------------------------------------------
# B004 — Plate Reverb
# Nested allpass + delay sections with modulation
# ---------------------------------------------------------------------------

@numba.njit
def _plate_reverb_kernel(samples, sr, decay, damping, mod_rate, mod_depth, pre_delay_ms):
    n = len(samples)

    # Pre-delay
    pre_delay_samps = int(pre_delay_ms * 0.001 * sr)
    if pre_delay_samps < 1:
        pre_delay_samps = 1
    pre_buf = np.zeros(pre_delay_samps, dtype=np.float32)
    pre_idx = np.int32(0)

    # Allpass delays for diffusion (4 allpass filters in series)
    ap_delays_ms = np.array([4.77, 3.60, 12.73, 9.31], dtype=np.float64)
    num_ap = 4
    ap_delays = np.empty(num_ap, dtype=np.int32)
    for i in range(num_ap):
        d = int(ap_delays_ms[i] * 0.001 * sr)
        if d < 1:
            d = 1
        ap_delays[i] = d

    max_ap = 0
    for i in range(num_ap):
        if ap_delays[i] > max_ap:
            max_ap = ap_delays[i]
    ap_bufs = np.zeros((num_ap, max_ap + 1), dtype=np.float32)
    ap_idx = np.zeros(num_ap, dtype=np.int32)
    ap_g = np.float32(0.6)

    # Two delay lines for the tank
    tank_delay_ms = np.array([30.51, 22.58], dtype=np.float64)
    num_tank = 2
    tank_delays = np.empty(num_tank, dtype=np.int32)
    for i in range(num_tank):
        d = int(tank_delay_ms[i] * 0.001 * sr) + int(mod_depth) + 2
        if d < 2:
            d = 2
        tank_delays[i] = d
    max_tank = 0
    for i in range(num_tank):
        if tank_delays[i] > max_tank:
            max_tank = tank_delays[i]
    tank_bufs = np.zeros((num_tank, max_tank + 1), dtype=np.float32)
    tank_idx = np.zeros(num_tank, dtype=np.int32)

    # Decay gain
    decay_g = np.float32(10.0 ** (-3.0 * 0.03 / decay))
    lp_coeff = np.float32(damping)
    lp_state = np.zeros(num_tank, dtype=np.float32)

    # Two tank allpasses
    tank_ap_delays_ms = np.array([8.93, 6.28], dtype=np.float64)
    tank_ap_delays = np.empty(num_tank, dtype=np.int32)
    for i in range(num_tank):
        d = int(tank_ap_delays_ms[i] * 0.001 * sr)
        if d < 1:
            d = 1
        tank_ap_delays[i] = d
    max_tap = 0
    for i in range(num_tank):
        if tank_ap_delays[i] > max_tap:
            max_tap = tank_ap_delays[i]
    tank_ap_bufs = np.zeros((num_tank, max_tap + 1), dtype=np.float32)
    tank_ap_idx = np.zeros(num_tank, dtype=np.int32)
    tank_ap_g = np.float32(0.5)

    out = np.zeros(n, dtype=np.float32)

    # Mod phase
    mod_phase = np.float32(0.0)
    mod_inc = np.float32(2.0 * np.pi * mod_rate / sr)

    # Tank feedback state
    tank_fb = np.zeros(num_tank, dtype=np.float32)

    for i in range(n):
        x = samples[i]

        # Pre-delay
        pre_out = pre_buf[pre_idx % pre_delay_samps]
        pre_buf[pre_idx % pre_delay_samps] = x
        pre_idx += 1

        # Input diffusion: 4 series allpass
        y = pre_out
        for a in range(num_ap):
            dl = ap_delays[a]
            rd = (ap_idx[a] - dl + max_ap + 1) % (max_ap + 1)
            delayed = ap_bufs[a, rd]
            v = y - ap_g * delayed
            ap_bufs[a, ap_idx[a] % (max_ap + 1)] = v
            ap_idx[a] += 1
            y = delayed + ap_g * v

        # Feed into tank with cross-feedback
        tank_in_0 = y + decay_g * tank_fb[1]
        tank_in_1 = y + decay_g * tank_fb[0]

        out_sum = np.float32(0.0)
        for t in range(num_tank):
            if t == 0:
                tank_in = tank_in_0
            else:
                tank_in = tank_in_1

            # Tank allpass
            dl_tap = tank_ap_delays[t]
            rd = (tank_ap_idx[t] - dl_tap + max_tap + 1) % (max_tap + 1)
            delayed = tank_ap_bufs[t, rd]
            v = tank_in - tank_ap_g * delayed
            tank_ap_bufs[t, tank_ap_idx[t] % (max_tap + 1)] = v
            tank_ap_idx[t] += 1
            ap_out = delayed + tank_ap_g * v

            # Modulated delay
            base_dl = int(tank_delay_ms[t] * 0.001 * sr)
            mod_offset = mod_depth * np.sin(mod_phase + np.float32(t) * np.float32(np.pi * 0.5))
            # Integer modulated delay
            mod_dl = base_dl + int(mod_offset)
            if mod_dl < 1:
                mod_dl = 1
            if mod_dl >= tank_delays[t]:
                mod_dl = tank_delays[t] - 1

            rd = (tank_idx[t] - mod_dl + max_tank + 1) % (max_tank + 1)
            delayed_tank = tank_bufs[t, rd]

            # Lowpass in feedback
            lp_state[t] = lp_coeff * lp_state[t] + (np.float32(1.0) - lp_coeff) * delayed_tank
            tank_fb[t] = lp_state[t]

            tank_bufs[t, tank_idx[t] % (max_tank + 1)] = ap_out
            tank_idx[t] += 1

            out_sum += delayed_tank

        mod_phase += mod_inc
        if mod_phase > np.float32(2.0 * np.pi):
            mod_phase -= np.float32(2.0 * np.pi)

        wet = out_sum * np.float32(0.5)
        out[i] = np.float32(0.5) * x + np.float32(0.5) * wet

    return out


def effect_b004_plate_reverb(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    decay = np.float32(params.get('decay', 2.0))
    damping = np.float32(params.get('damping', 0.4))
    mod_rate = np.float32(params.get('mod_rate', 1.0))
    mod_depth = np.float32(params.get('mod_depth', 4.0))
    pre_delay_ms = np.float32(params.get('pre_delay_ms', 20.0))
    return _plate_reverb_kernel(
        samples.astype(np.float32), sr, decay, damping, mod_rate, mod_depth, pre_delay_ms
    )


def variants_b004():
    return [
        {'decay': 0.5, 'damping': 0.3, 'mod_rate': 0.5, 'mod_depth': 2.0, 'pre_delay_ms': 0.0},
        {'decay': 1.5, 'damping': 0.4, 'mod_rate': 1.0, 'mod_depth': 4.0, 'pre_delay_ms': 10.0},
        {'decay': 3.0, 'damping': 0.5, 'mod_rate': 1.5, 'mod_depth': 6.0, 'pre_delay_ms': 30.0},
        {'decay': 5.0, 'damping': 0.6, 'mod_rate': 0.8, 'mod_depth': 3.0, 'pre_delay_ms': 50.0},
        {'decay': 8.0, 'damping': 0.8, 'mod_rate': 2.0, 'mod_depth': 8.0, 'pre_delay_ms': 80.0},
        {'decay': 10.0, 'damping': 0.2, 'mod_rate': 1.2, 'mod_depth': 1.0, 'pre_delay_ms': 100.0},
    ]


# ---------------------------------------------------------------------------
# B005 — Spring Reverb
# Allpass chain with frequency-dependent delay + nonlinearity
# ---------------------------------------------------------------------------

@numba.njit
def _spring_reverb_kernel(samples, sr, num_springs, tension, damping, chaos):
    n = len(samples)
    out = np.zeros(n, dtype=np.float32)

    # Each spring is modeled as a chain of allpass filters with
    # slightly different delays to simulate chirp dispersion
    ap_per_spring = 8
    total_ap = num_springs * ap_per_spring

    # Compute delays for each allpass based on spring index and position in chain
    # Higher tension = shorter delays = higher frequency chirp
    base_delay_ms = np.float32(3.0) * (np.float32(1.0) - tension) + np.float32(0.5)
    max_delay_samps = int(base_delay_ms * 2.0 * 0.001 * sr) + 4
    if max_delay_samps < 4:
        max_delay_samps = 4

    ap_delays = np.empty(total_ap, dtype=np.int32)
    ap_g_vals = np.empty(total_ap, dtype=np.float32)

    for s in range(num_springs):
        for a in range(ap_per_spring):
            idx = s * ap_per_spring + a
            # Dispersion: delay varies along the chain (chirp effect)
            frac = np.float32(a) / np.float32(ap_per_spring)
            delay_ms = base_delay_ms * (np.float32(1.0) + frac * np.float32(1.5))
            # Offset per spring for decorrelation
            delay_ms += np.float32(s) * np.float32(0.7)
            d = int(delay_ms * 0.001 * sr)
            if d < 1:
                d = 1
            if d >= max_delay_samps:
                d = max_delay_samps - 1
            ap_delays[idx] = d
            # Feedback gain, damped along chain
            ap_g_vals[idx] = np.float32(0.5) + np.float32(0.3) * (np.float32(1.0) - damping) * (np.float32(1.0) - frac * np.float32(0.5))

    ap_bufs = np.zeros((total_ap, max_delay_samps), dtype=np.float32)
    ap_idx = np.zeros(total_ap, dtype=np.int32)

    # Lowpass state per spring
    lp_state = np.zeros(num_springs, dtype=np.float32)
    lp_coeff = damping

    for i in range(n):
        x = samples[i]
        spring_sum = np.float32(0.0)

        for s in range(num_springs):
            y = x

            # Allpass chain for this spring
            for a in range(ap_per_spring):
                idx = s * ap_per_spring + a
                dl = ap_delays[idx]
                g = ap_g_vals[idx]
                rd = (ap_idx[idx] - dl + max_delay_samps) % max_delay_samps
                delayed = ap_bufs[idx, rd]
                v = y - g * delayed
                ap_bufs[idx, ap_idx[idx] % max_delay_samps] = v
                ap_idx[idx] += 1
                y = delayed + g * v

            # Nonlinearity (soft clip for spring bounce character)
            if chaos > 0.0:
                y = y + chaos * (np.tanh(y * np.float32(3.0)) - y)

            # Damping lowpass
            lp_state[s] = lp_coeff * lp_state[s] + (np.float32(1.0) - lp_coeff) * y
            spring_sum += lp_state[s]

        spring_out = spring_sum / np.float32(max(num_springs, 1))
        out[i] = np.float32(0.5) * x + np.float32(0.5) * spring_out

    return out


def effect_b005_spring_reverb(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    num_springs = int(params.get('num_springs', 2))
    tension = np.float32(params.get('tension', 0.6))
    damping = np.float32(params.get('damping', 0.5))
    chaos = np.float32(params.get('chaos', 0.1))
    return _spring_reverb_kernel(
        samples.astype(np.float32), sr, num_springs, tension, damping, chaos
    )


def variants_b005():
    return [
        {'num_springs': 1, 'tension': 0.3, 'damping': 0.5, 'chaos': 0.0},
        {'num_springs': 2, 'tension': 0.6, 'damping': 0.5, 'chaos': 0.1},
        {'num_springs': 3, 'tension': 0.9, 'damping': 0.3, 'chaos': 0.0},
        {'num_springs': 2, 'tension': 0.4, 'damping': 0.8, 'chaos': 0.3},
        {'num_springs': 3, 'tension': 0.7, 'damping': 0.6, 'chaos': 0.5},
        {'num_springs': 1, 'tension': 0.5, 'damping': 0.4, 'chaos': 0.2},
    ]


# ---------------------------------------------------------------------------
# B006 — Shimmer Reverb
# FDN reverb with pitch-shifted feedback (+12 semitones via simple resampling)
# ---------------------------------------------------------------------------

@numba.njit
def _pitch_shift_resample(buf, read_pos, length, ratio):
    """Read from circular buffer with resampling for pitch shift.

    ratio > 1.0 = pitch up (read faster through buffer).
    Uses linear interpolation.
    """
    out = np.zeros(length, dtype=np.float32)
    buf_len = len(buf)
    pos = np.float64(read_pos)
    for i in range(length):
        idx0 = int(pos) % buf_len
        idx1 = (idx0 + 1) % buf_len
        frac = np.float32(pos - int(pos))
        out[i] = (np.float32(1.0) - frac) * buf[idx0] + frac * buf[idx1]
        pos += ratio
    return out


@numba.njit
def _shimmer_reverb_kernel(samples, sr, rt60, pitch_shift_semitones, shimmer_amount, wet_mix):
    n = len(samples)

    # Pitch shift ratio
    ratio = np.float64(2.0 ** (pitch_shift_semitones / 12.0))

    # FDN with 4 delay lines
    n_delays = 4
    delay_ms = np.array([29.7, 37.1, 41.1, 43.7], dtype=np.float64)
    delays = np.empty(n_delays, dtype=np.int32)
    gains = np.empty(n_delays, dtype=np.float32)
    for i in range(n_delays):
        d = int(delay_ms[i] * 0.001 * sr)
        if d < 1:
            d = 1
        delays[i] = d
        gains[i] = np.float32(10.0 ** (-3.0 * d / (sr * rt60)))

    max_delay = 0
    for i in range(n_delays):
        if delays[i] > max_delay:
            max_delay = delays[i]

    bufs = np.zeros((n_delays, max_delay + 1), dtype=np.float32)
    write_idx = np.zeros(n_delays, dtype=np.int32)

    # Hadamard/2 matrix
    mat = np.array([
        [1.0, 1.0, 1.0, 1.0],
        [1.0, -1.0, 1.0, -1.0],
        [1.0, 1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0, 1.0],
    ], dtype=np.float32)
    for r in range(4):
        for c in range(4):
            mat[r, c] *= np.float32(0.5)

    # Shimmer pitch shift buffer (accumulates output for resampling)
    shimmer_buf_len = max_delay * 4
    if shimmer_buf_len < 1024:
        shimmer_buf_len = 1024
    shimmer_buf = np.zeros(shimmer_buf_len, dtype=np.float32)
    shimmer_write = np.int32(0)
    shimmer_read_pos = np.float64(0.0)

    lp_state = np.zeros(n_delays, dtype=np.float32)
    lp_coeff = np.float32(0.3)

    out = np.zeros(n, dtype=np.float32)

    for i in range(n):
        x = samples[i]

        # Read from delay lines
        delayed = np.empty(n_delays, dtype=np.float32)
        for d in range(n_delays):
            dl = delays[d]
            read_pos = (write_idx[d] - dl + max_delay + 1) % (max_delay + 1)
            delayed[d] = bufs[d, read_pos]

        # Sum delayed for output
        out_sum = np.float32(0.0)
        for d in range(n_delays):
            out_sum += delayed[d]

        # Write current output sum to shimmer buffer for pitch shifting
        wet_out = out_sum / np.float32(n_delays)
        shimmer_buf[shimmer_write % shimmer_buf_len] = wet_out
        shimmer_write += 1

        # Read pitch-shifted sample from shimmer buffer
        rd_idx0 = int(shimmer_read_pos) % shimmer_buf_len
        rd_idx1 = (rd_idx0 + 1) % shimmer_buf_len
        frac = np.float32(shimmer_read_pos - int(shimmer_read_pos))
        pitched = (np.float32(1.0) - frac) * shimmer_buf[rd_idx0] + frac * shimmer_buf[rd_idx1]
        shimmer_read_pos += ratio
        # Keep read pos from getting too far behind
        if shimmer_write - shimmer_read_pos > shimmer_buf_len * 0.5:
            shimmer_read_pos = np.float64(shimmer_write) - np.float64(shimmer_buf_len) * 0.25

        # Apply feedback matrix
        feedback = np.zeros(n_delays, dtype=np.float32)
        for r in range(n_delays):
            s = np.float32(0.0)
            for c in range(n_delays):
                s += mat[r, c] * delayed[c]
            feedback[r] = s

        # Mix shimmer into feedback
        for d in range(n_delays):
            fb = feedback[d]
            shimmer_contribution = shimmer_amount * pitched
            fb_mixed = (np.float32(1.0) - shimmer_amount) * fb + shimmer_contribution

            # Lowpass
            lp_state[d] = lp_coeff * lp_state[d] + (np.float32(1.0) - lp_coeff) * fb_mixed
            val = x + gains[d] * lp_state[d]
            write_pos = write_idx[d] % (max_delay + 1)
            bufs[d, write_pos] = val
            write_idx[d] += 1

        out[i] = (np.float32(1.0) - wet_mix) * x + wet_mix * wet_out

    return out


def effect_b006_shimmer_reverb(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    rt60 = np.float32(params.get('rt60', 3.0))
    pitch_shift_semitones = np.float32(params.get('pitch_shift_semitones', 12.0))
    shimmer_amount = np.float32(params.get('shimmer_amount', 0.3))
    wet_mix = np.float32(params.get('wet_mix', 0.5))
    return _shimmer_reverb_kernel(
        samples.astype(np.float32), sr, rt60, pitch_shift_semitones, shimmer_amount, wet_mix
    )


def variants_b006():
    return [
        {'rt60': 1.0, 'pitch_shift_semitones': 12.0, 'shimmer_amount': 0.2, 'wet_mix': 0.4},
        {'rt60': 3.0, 'pitch_shift_semitones': 12.0, 'shimmer_amount': 0.3, 'wet_mix': 0.5},
        {'rt60': 5.0, 'pitch_shift_semitones': 7.0, 'shimmer_amount': 0.4, 'wet_mix': 0.6},
        {'rt60': 7.0, 'pitch_shift_semitones': 5.0, 'shimmer_amount': 0.2, 'wet_mix': 0.5},
        {'rt60': 10.0, 'pitch_shift_semitones': 12.0, 'shimmer_amount': 0.6, 'wet_mix': 0.8},
        {'rt60': 4.0, 'pitch_shift_semitones': 9.0, 'shimmer_amount': 0.5, 'wet_mix': 0.7},
    ]


# ---------------------------------------------------------------------------
# B007 — Convolution Reverb with Synthetic IR
# Generate noise*exp decay IR + early reflection spikes, convolve via FFT
# ---------------------------------------------------------------------------

def _generate_synthetic_ir(sr, ir_length_ms, decay_rate, num_early_reflections, er_spacing_ms):
    """Generate a synthetic impulse response."""
    ir_length_samps = int(ir_length_ms * 0.001 * sr)
    if ir_length_samps < 1:
        ir_length_samps = 1

    # Time axis
    t = np.arange(ir_length_samps, dtype=np.float32) / np.float32(sr)

    # Exponential decay envelope
    envelope = np.exp(-decay_rate * t).astype(np.float32)

    # Noise tail
    rng = np.random.default_rng(42)
    noise = rng.standard_normal(ir_length_samps).astype(np.float32)
    ir = noise * envelope

    # Early reflections: discrete spikes
    er_spacing_samps = int(er_spacing_ms * 0.001 * sr)
    if er_spacing_samps < 1:
        er_spacing_samps = 1
    for r in range(num_early_reflections):
        pos = (r + 1) * er_spacing_samps
        if pos < ir_length_samps:
            # Decreasing amplitude for later reflections
            amp = np.float32(0.8) * np.float32(0.7 ** r)
            # Alternate polarity for more natural sound
            if r % 2 == 1:
                amp = -amp
            ir[pos] += amp

    # Normalize IR
    peak = np.max(np.abs(ir))
    if peak > 1e-10:
        ir = ir / peak

    return ir


def effect_b007_convolution_reverb(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    ir_length_ms = float(params.get('ir_length_ms', 1500.0))
    decay_rate = float(params.get('decay_rate', 2.0))
    num_early_reflections = int(params.get('num_early_reflections', 8))
    er_spacing_ms = float(params.get('er_spacing_ms', 15.0))

    samples = samples.astype(np.float32)
    ir = _generate_synthetic_ir(sr, ir_length_ms, decay_rate, num_early_reflections, er_spacing_ms)

    # FFT-based convolution
    n = len(samples)
    ir_len = len(ir)
    fft_size = 1
    while fft_size < n + ir_len - 1:
        fft_size *= 2

    S = np.fft.rfft(samples, n=fft_size)
    IR = np.fft.rfft(ir, n=fft_size)
    convolved = np.fft.irfft(S * IR, n=fft_size).astype(np.float32)

    # Trim to original + tail length, but cap at input + ir
    out_len = min(n + ir_len - 1, fft_size)
    out = convolved[:out_len]

    # Wet/dry mix (50/50 by default for convolution reverb)
    wet_mix = float(params.get('wet_mix', 0.5))
    dry_pad = np.zeros(out_len, dtype=np.float32)
    dry_pad[:n] = samples
    result = np.float32(1.0 - wet_mix) * dry_pad + np.float32(wet_mix) * out

    return result


def variants_b007():
    return [
        {'ir_length_ms': 200.0, 'decay_rate': 5.0, 'num_early_reflections': 3, 'er_spacing_ms': 5.0},
        {'ir_length_ms': 800.0, 'decay_rate': 3.0, 'num_early_reflections': 6, 'er_spacing_ms': 10.0},
        {'ir_length_ms': 1500.0, 'decay_rate': 2.0, 'num_early_reflections': 8, 'er_spacing_ms': 15.0},
        {'ir_length_ms': 3000.0, 'decay_rate': 1.0, 'num_early_reflections': 12, 'er_spacing_ms': 20.0},
        {'ir_length_ms': 5000.0, 'decay_rate': 0.5, 'num_early_reflections': 15, 'er_spacing_ms': 30.0},
        {'ir_length_ms': 1000.0, 'decay_rate': 4.0, 'num_early_reflections': 5, 'er_spacing_ms': 8.0},
    ]


# ---------------------------------------------------------------------------
# B008 — Metallic Resonator
# Bank of short comb filters (1-5ms) in parallel with high feedback
# ---------------------------------------------------------------------------

@numba.njit
def _metallic_resonator_kernel(samples, sr, num_resonators, base_freq_hz, freq_spread, feedback):
    n = len(samples)

    # Compute delay lengths from frequencies
    # Base frequency determines shortest delay, spread determines range
    delays = np.empty(num_resonators, dtype=np.int32)
    for r in range(num_resonators):
        # Distribute frequencies logarithmically
        freq_ratio = np.float32(1.0) + freq_spread * np.float32(r) / np.float32(max(num_resonators - 1, 1))
        freq = base_freq_hz * freq_ratio
        d = int(sr / freq)
        if d < 1:
            d = 1
        # Clamp to 1-5ms range
        min_d = int(0.001 * sr)
        max_d = int(0.005 * sr)
        if d < min_d:
            d = min_d
        if d > max_d:
            d = max_d
        delays[r] = d

    max_delay = 0
    for r in range(num_resonators):
        if delays[r] > max_delay:
            max_delay = delays[r]

    bufs = np.zeros((num_resonators, max_delay + 1), dtype=np.float32)
    write_idx = np.zeros(num_resonators, dtype=np.int32)

    fb = np.float32(feedback)
    out = np.zeros(n, dtype=np.float32)

    for i in range(n):
        x = samples[i]
        res_sum = np.float32(0.0)

        for r in range(num_resonators):
            dl = delays[r]
            read_pos = (write_idx[r] - dl + max_delay + 1) % (max_delay + 1)
            delayed = bufs[r, read_pos]

            # Comb filter: input + feedback * delayed
            val = x + fb * delayed
            write_pos = write_idx[r] % (max_delay + 1)
            bufs[r, write_pos] = val
            write_idx[r] += 1

            res_sum += delayed

        res_out = res_sum / np.float32(max(num_resonators, 1))

        # Soft clip to prevent blowup with high feedback
        if res_out > np.float32(1.0):
            res_out = np.float32(1.0) - np.float32(1.0) / (res_out + np.float32(1.0))
        elif res_out < np.float32(-1.0):
            res_out = np.float32(-1.0) + np.float32(1.0) / (-res_out + np.float32(1.0))

        out[i] = np.float32(0.4) * x + np.float32(0.6) * res_out

    return out


def effect_b008_metallic_resonator(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    num_resonators = int(params.get('num_resonators', 6))
    base_freq_hz = np.float32(params.get('base_freq_hz', 500.0))
    freq_spread = np.float32(params.get('freq_spread', 1.0))
    feedback = np.float32(params.get('feedback', 0.95))
    return _metallic_resonator_kernel(
        samples.astype(np.float32), sr, num_resonators, base_freq_hz, freq_spread, feedback
    )


def variants_b008():
    return [
        {'num_resonators': 4, 'base_freq_hz': 200.0, 'freq_spread': 0.5, 'feedback': 0.92},
        {'num_resonators': 6, 'base_freq_hz': 500.0, 'freq_spread': 1.0, 'feedback': 0.95},
        {'num_resonators': 8, 'base_freq_hz': 800.0, 'freq_spread': 1.5, 'feedback': 0.97},
        {'num_resonators': 10, 'base_freq_hz': 1200.0, 'freq_spread': 2.0, 'feedback': 0.99},
        {'num_resonators': 12, 'base_freq_hz': 2000.0, 'freq_spread': 0.8, 'feedback': 0.993},
        {'num_resonators': 5, 'base_freq_hz': 350.0, 'freq_spread': 1.2, 'feedback': 0.98},
        {'num_resonators': 8, 'base_freq_hz': 1000.0, 'freq_spread': 0.5, 'feedback': 0.995},
    ]


# ---------------------------------------------------------------------------
# B009 — Dattorro Plate Reverb
# Distinct from B004: uses Dattorro's specific topology with two nested
# allpass chains feeding a figure-eight tank with modulated delays.
# ---------------------------------------------------------------------------

@numba.njit
def _dattorro_plate_kernel(samples, sr, decay, damping, bandwidth, pre_delay_ms):
    n = len(samples)

    # Pre-delay
    pre_delay_samps = max(1, int(pre_delay_ms * 0.001 * sr))
    pre_buf = np.zeros(pre_delay_samps, dtype=np.float32)
    pre_idx = np.int32(0)

    # Input diffusion: 4 allpass filters in series
    in_ap_delays_ms = np.array([4.77, 3.60, 12.73, 9.31], dtype=np.float64)
    in_ap_g = np.array([0.75, 0.75, 0.625, 0.625], dtype=np.float32)
    num_in_ap = 4
    in_ap_delays = np.empty(num_in_ap, dtype=np.int32)
    for i in range(num_in_ap):
        d = max(1, int(in_ap_delays_ms[i] * 0.001 * sr))
        in_ap_delays[i] = d
    max_in_ap = 0
    for i in range(num_in_ap):
        if in_ap_delays[i] > max_in_ap:
            max_in_ap = in_ap_delays[i]
    in_ap_bufs = np.zeros((num_in_ap, max_in_ap + 1), dtype=np.float32)
    in_ap_idx = np.zeros(num_in_ap, dtype=np.int32)

    # Tank: two halves, each has allpass -> delay -> lowpass -> decay
    tank_ap_delays_ms = np.array([22.58, 30.51], dtype=np.float64)
    tank_delay_ms = np.array([149.63, 125.0], dtype=np.float64)
    tank_ap_g = np.float32(0.5)

    tank_ap_delays = np.empty(2, dtype=np.int32)
    tank_delays = np.empty(2, dtype=np.int32)
    for i in range(2):
        tank_ap_delays[i] = max(1, int(tank_ap_delays_ms[i] * 0.001 * sr))
        tank_delays[i] = max(1, int(tank_delay_ms[i] * 0.001 * sr))

    max_tank_ap = max(tank_ap_delays[0], tank_ap_delays[1])
    max_tank_dl = max(tank_delays[0], tank_delays[1])

    tank_ap_bufs = np.zeros((2, max_tank_ap + 1), dtype=np.float32)
    tank_ap_idx = np.zeros(2, dtype=np.int32)
    tank_dl_bufs = np.zeros((2, max_tank_dl + 1), dtype=np.float32)
    tank_dl_idx = np.zeros(2, dtype=np.int32)
    tank_lp = np.zeros(2, dtype=np.float32)
    tank_fb = np.zeros(2, dtype=np.float32)

    bw_coeff = np.float32(bandwidth)
    lp_coeff = np.float32(damping)
    decay_g = np.float32(decay)

    # Input bandwidth lowpass
    bw_state = np.float32(0.0)

    out = np.zeros(n, dtype=np.float32)

    for i in range(n):
        x = samples[i]

        # Input bandwidth control (one-pole lowpass)
        bw_state = bw_coeff * bw_state + (np.float32(1.0) - bw_coeff) * x
        y = bw_state

        # Pre-delay
        pre_out = pre_buf[pre_idx % pre_delay_samps]
        pre_buf[pre_idx % pre_delay_samps] = y
        pre_idx += 1
        y = pre_out

        # Input diffusion allpass chain
        for a in range(num_in_ap):
            dl = in_ap_delays[a]
            rd = (in_ap_idx[a] - dl + max_in_ap + 1) % (max_in_ap + 1)
            delayed = in_ap_bufs[a, rd]
            g = in_ap_g[a]
            v = y - g * delayed
            in_ap_bufs[a, in_ap_idx[a] % (max_in_ap + 1)] = v
            in_ap_idx[a] += 1
            y = delayed + g * v

        # Feed into tank with cross-feedback (figure-eight)
        tank_in_0 = y + decay_g * tank_fb[1]
        tank_in_1 = y + decay_g * tank_fb[0]

        out_sum = np.float32(0.0)
        for t in range(2):
            tank_in = tank_in_0 if t == 0 else tank_in_1

            # Tank allpass
            dl = tank_ap_delays[t]
            rd = (tank_ap_idx[t] - dl + max_tank_ap + 1) % (max_tank_ap + 1)
            delayed = tank_ap_bufs[t, rd]
            v = tank_in - tank_ap_g * delayed
            tank_ap_bufs[t, tank_ap_idx[t] % (max_tank_ap + 1)] = v
            tank_ap_idx[t] += 1
            ap_out = delayed + tank_ap_g * v

            # Tank delay line
            dl2 = tank_delays[t]
            rd2 = (tank_dl_idx[t] - dl2 + max_tank_dl + 1) % (max_tank_dl + 1)
            delayed2 = tank_dl_bufs[t, rd2]
            tank_dl_bufs[t, tank_dl_idx[t] % (max_tank_dl + 1)] = ap_out
            tank_dl_idx[t] += 1

            # Damping lowpass
            tank_lp[t] = lp_coeff * tank_lp[t] + (np.float32(1.0) - lp_coeff) * delayed2
            tank_fb[t] = tank_lp[t]

            out_sum += delayed2

        wet = out_sum * np.float32(0.5)
        out[i] = np.float32(0.5) * x + np.float32(0.5) * wet

    return out


def effect_b009_dattorro_plate(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Dattorro plate reverb with figure-eight tank topology."""
    decay = np.float32(params.get('decay', 0.7))
    damping = np.float32(params.get('damping', 0.4))
    bandwidth = np.float32(params.get('bandwidth', 0.7))
    pre_delay_ms = np.float32(params.get('pre_delay_ms', 10.0))
    return _dattorro_plate_kernel(
        samples.astype(np.float32), sr, decay, damping, bandwidth, pre_delay_ms
    )


def variants_b009():
    return [
        {'decay': 0.3, 'damping': 0.2, 'bandwidth': 0.9, 'pre_delay_ms': 0.0},
        {'decay': 0.5, 'damping': 0.4, 'bandwidth': 0.7, 'pre_delay_ms': 10.0},
        {'decay': 0.7, 'damping': 0.5, 'bandwidth': 0.7, 'pre_delay_ms': 20.0},
        {'decay': 0.85, 'damping': 0.6, 'bandwidth': 0.5, 'pre_delay_ms': 40.0},
        {'decay': 0.95, 'damping': 0.8, 'bandwidth': 0.3, 'pre_delay_ms': 60.0},
        {'decay': 0.99, 'damping': 0.3, 'bandwidth': 0.8, 'pre_delay_ms': 100.0},
    ]


# ---------------------------------------------------------------------------
# B010 — Freeverb (Jezar)
# 8 parallel comb filters -> 4 series allpass, distinct from Schroeder/Moorer
# ---------------------------------------------------------------------------

@numba.njit
def _freeverb_kernel(samples, sr, room_size, damping, wet_mix):
    n = len(samples)

    # Jezar's original comb filter delays (scaled by sr/44100)
    scale = np.float64(sr) / 44100.0
    comb_delays_base = np.array([1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617], dtype=np.int64)
    num_combs = 8
    comb_delays = np.empty(num_combs, dtype=np.int32)
    for i in range(num_combs):
        comb_delays[i] = max(1, int(comb_delays_base[i] * scale))

    max_comb = 0
    for i in range(num_combs):
        if comb_delays[i] > max_comb:
            max_comb = comb_delays[i]

    comb_bufs = np.zeros((num_combs, max_comb + 1), dtype=np.float32)
    comb_idx = np.zeros(num_combs, dtype=np.int32)
    comb_filter_state = np.zeros(num_combs, dtype=np.float32)

    fb = np.float32(room_size)
    damp1 = np.float32(damping)
    damp2 = np.float32(1.0) - damp1

    # 4 series allpass filters
    ap_delays_base = np.array([556, 441, 341, 225], dtype=np.int64)
    num_ap = 4
    ap_delays = np.empty(num_ap, dtype=np.int32)
    for i in range(num_ap):
        ap_delays[i] = max(1, int(ap_delays_base[i] * scale))

    max_ap = 0
    for i in range(num_ap):
        if ap_delays[i] > max_ap:
            max_ap = ap_delays[i]

    ap_bufs = np.zeros((num_ap, max_ap + 1), dtype=np.float32)
    ap_idx = np.zeros(num_ap, dtype=np.int32)
    ap_g = np.float32(0.5)

    out = np.zeros(n, dtype=np.float32)

    for i in range(n):
        x = samples[i]

        # 8 parallel Lowpass-Feedback-Comb filters summed
        comb_sum = np.float32(0.0)
        for c in range(num_combs):
            dl = comb_delays[c]
            rd = (comb_idx[c] - dl + max_comb + 1) % (max_comb + 1)
            delayed = comb_bufs[c, rd]

            # One-pole lowpass in feedback (Jezar's damping)
            comb_filter_state[c] = delayed * damp2 + comb_filter_state[c] * damp1

            val = x + fb * comb_filter_state[c]
            comb_bufs[c, comb_idx[c] % (max_comb + 1)] = val
            comb_idx[c] += 1
            comb_sum += delayed

        y = comb_sum * np.float32(0.125)  # scale by 1/8

        # 4 series allpass filters
        for a in range(num_ap):
            dl = ap_delays[a]
            rd = (ap_idx[a] - dl + max_ap + 1) % (max_ap + 1)
            delayed = ap_bufs[a, rd]
            v = y + ap_g * delayed
            ap_bufs[a, ap_idx[a] % (max_ap + 1)] = y
            ap_idx[a] += 1
            y = delayed - ap_g * v

        out[i] = (np.float32(1.0) - wet_mix) * x + wet_mix * y

    return out


def effect_b010_freeverb(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Freeverb (Jezar): 8 LBCF combs -> 4 series allpass."""
    room_size = np.float32(params.get('room_size', 0.7))
    damping = np.float32(params.get('damping', 0.5))
    wet_mix = np.float32(params.get('wet_mix', 0.5))
    return _freeverb_kernel(
        samples.astype(np.float32), sr, room_size, damping, wet_mix
    )


def variants_b010():
    return [
        {'room_size': 0.3, 'damping': 0.3, 'wet_mix': 0.3},
        {'room_size': 0.5, 'damping': 0.5, 'wet_mix': 0.4},
        {'room_size': 0.7, 'damping': 0.5, 'wet_mix': 0.5},
        {'room_size': 0.85, 'damping': 0.7, 'wet_mix': 0.6},
        {'room_size': 0.95, 'damping': 0.2, 'wet_mix': 0.7},
        {'room_size': 0.99, 'damping': 0.9, 'wet_mix': 0.8},
    ]


# ---------------------------------------------------------------------------
# B011 — Velvet Noise Reverb
# Sparse random +1/-1 impulse sequence as FIR, convolved via FFT.
# Very efficient, perceptually distinct smooth character.
# ---------------------------------------------------------------------------

def _generate_velvet_ir(sr, ir_length_ms, density, decay_rate, seed=42):
    """Generate a velvet noise impulse response.

    Velvet noise consists of sparse +1/-1 impulses at pseudo-random
    positions. Combined with an exponential decay envelope, this
    creates a smooth reverb tail without the metallic artifacts
    of regular noise-based IRs.
    """
    ir_length_samps = max(1, int(ir_length_ms * 0.001 * sr))
    ir = np.zeros(ir_length_samps, dtype=np.float32)

    rng = np.random.default_rng(seed)

    # Average spacing between impulses
    avg_spacing = max(1, int(sr / density))

    pos = 0
    while pos < ir_length_samps:
        # Place impulse with random +1/-1 polarity
        polarity = np.float32(1.0) if rng.random() > 0.5 else np.float32(-1.0)
        # Decay envelope
        t = np.float32(pos) / np.float32(sr)
        amp = np.float32(np.exp(-decay_rate * t))
        ir[pos] = polarity * amp

        # Next position: jittered spacing
        jitter = rng.integers(1, max(2, avg_spacing * 2))
        pos += jitter

    # Normalize
    peak = np.max(np.abs(ir))
    if peak > 1e-10:
        ir /= peak

    return ir


def effect_b011_velvet_noise_reverb(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Velvet noise reverb: sparse +1/-1 impulse response convolved via FFT."""
    ir_length_ms = float(params.get('ir_length_ms', 1500))
    density = float(params.get('density', 2000))
    decay_rate = float(params.get('decay_rate', 2.0))
    wet_mix = float(params.get('wet_mix', 0.5))
    seed = int(params.get('seed', 42))

    samples = samples.astype(np.float32)
    ir = _generate_velvet_ir(sr, ir_length_ms, density, decay_rate, seed)

    n = len(samples)
    ir_len = len(ir)
    fft_size = 1
    while fft_size < n + ir_len - 1:
        fft_size *= 2

    S = np.fft.rfft(samples, n=fft_size)
    IR = np.fft.rfft(ir, n=fft_size)
    convolved = np.fft.irfft(S * IR, n=fft_size).astype(np.float32)

    out_len = min(n + ir_len - 1, fft_size)
    out = convolved[:out_len]

    dry_pad = np.zeros(out_len, dtype=np.float32)
    dry_pad[:n] = samples
    result = np.float32(1.0 - wet_mix) * dry_pad + np.float32(wet_mix) * out
    return result


def variants_b011():
    return [
        {'ir_length_ms': 500, 'density': 1000, 'decay_rate': 4.0, 'wet_mix': 0.3, 'seed': 42},
        {'ir_length_ms': 1000, 'density': 2000, 'decay_rate': 2.5, 'wet_mix': 0.4, 'seed': 42},
        {'ir_length_ms': 1500, 'density': 2000, 'decay_rate': 2.0, 'wet_mix': 0.5, 'seed': 42},
        {'ir_length_ms': 2500, 'density': 3000, 'decay_rate': 1.5, 'wet_mix': 0.5, 'seed': 7},
        {'ir_length_ms': 4000, 'density': 4000, 'decay_rate': 1.0, 'wet_mix': 0.6, 'seed': 99},
        {'ir_length_ms': 1500, 'density': 500, 'decay_rate': 2.0, 'wet_mix': 0.5, 'seed': 17},
    ]
