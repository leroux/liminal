"""Numba-optimized modulated FDN inner loop.

Extends the static FDN (numba_fdn.py) with per-sample LFO modulation of
delay times, damping coefficients, output gains, and feedback matrix blend.
Uses fractional delay interpolation for smooth delay-time modulation.
"""

import numpy as np
from numba import njit

N = 8


# ---------------------------------------------------------------------------
# LFO Waveforms
# ---------------------------------------------------------------------------
# Waveform codes: 0=sine, 1=triangle, 2=sample-and-hold

@njit(cache=True)
def lfo_value(phase, waveform):
    """Compute LFO value for a given phase (0-1) and waveform type.
    Returns value in range [-1, +1].
    """
    if waveform == 0:  # sine
        return np.sin(2.0 * np.pi * phase)
    elif waveform == 1:  # triangle
        if phase < 0.25:
            return phase * 4.0
        elif phase < 0.75:
            return 2.0 - phase * 4.0
        else:
            return phase * 4.0 - 4.0
    else:  # sample-and-hold (waveform == 2)
        # Simple hash produces pseudo-random value per phase bucket
        n = int(phase * 1000.0)
        h = ((n * 1103515245 + 12345) >> 16) & 0x7FFF
        return (h / 16383.5) - 1.0


# ---------------------------------------------------------------------------
# Fractional Delay
# ---------------------------------------------------------------------------

@njit(cache=True)
def read_delay_frac(buf, write_idx, delay_frac, buf_len):
    """Read from a delay line with fractional delay using linear interpolation.

    Args:
        buf: delay buffer (1D array for one node)
        write_idx: current write position
        delay_frac: delay time in fractional samples (float)
        buf_len: length of buffer

    Returns:
        interpolated sample value
    """
    delay_int = int(delay_frac)
    frac = delay_frac - delay_int

    idx0 = (write_idx - 1 - delay_int) % buf_len
    idx1 = (write_idx - 2 - delay_int) % buf_len

    return buf[idx0] * (1.0 - frac) + buf[idx1] * frac


# ---------------------------------------------------------------------------
# Modulated FDN Process Block
# ---------------------------------------------------------------------------

@njit(cache=True)
def _process_block_mod(
    input_audio, output,
    # Pre-delay state
    pre_delay_buf, pre_delay_len, pre_delay_samples, pre_delay_write_idx,
    # Diffusion allpass state
    diff_bufs, diff_lens, diff_gains, diff_idxs, n_diff_stages,
    # FDN delay line state
    delay_bufs, delay_buf_len, delay_times_base, delay_write_idxs,
    # Damping filter state
    damping_coeffs_base, damping_y1,
    # Feedback matrices (base + target for blending)
    matrix, matrix2,
    # Gains
    feedback_gain, input_gains, output_gains_base, wet_dry,
    # Saturation
    saturation,
    # DC blocker state
    dc_x1, dc_y1, dc_R,
    # Stereo panning
    pan_gain_L, pan_gain_R,
    # --- Modulation params ---
    mod_depth_delay, mod_rate_delay,
    mod_depth_damping, mod_rate_damping,
    mod_depth_output, mod_rate_output,
    mod_depth_matrix, mod_rate_matrix,
    mod_waveform, mod_phases,
    sample_rate,
):
    n_samples = len(input_audio)
    pd_wi = pre_delay_write_idx[0]

    for n in range(n_samples):
        x = input_audio[n]

        # --- Pre-delay (unchanged) ---
        pre_delay_buf[pd_wi] = x
        pd_wi = (pd_wi + 1) % pre_delay_len
        rd_idx = (pd_wi - 1 - pre_delay_samples) % pre_delay_len
        x_delayed = pre_delay_buf[rd_idx]

        # --- Input diffusion (unchanged) ---
        diffused = x_delayed
        for s in range(n_diff_stages):
            idx = diff_idxs[s]
            delayed = diff_bufs[s, idx]
            g = diff_gains[s]
            v = diffused + g * delayed
            diffused = -g * v + delayed
            diff_bufs[s, idx] = v
            diff_idxs[s] = (idx + 1) % diff_lens[s]

        # --- Compute per-node LFO values ---
        lfo_delay = np.empty(N)
        lfo_damping = np.empty(N)
        lfo_output = np.empty(N)
        t = n / sample_rate

        for i in range(N):
            # Delay modulation LFO
            if mod_depth_delay[i] > 0.0:
                phase = (mod_rate_delay[i] * t + mod_phases[i]) % 1.0
                lfo_delay[i] = lfo_value(phase, mod_waveform)
            else:
                lfo_delay[i] = 0.0

            # Damping modulation LFO
            if mod_depth_damping[i] > 0.0:
                phase = (mod_rate_damping[i] * t + mod_phases[i]) % 1.0
                lfo_damping[i] = lfo_value(phase, mod_waveform)
            else:
                lfo_damping[i] = 0.0

            # Output gain modulation LFO
            if mod_depth_output[i] > 0.0:
                phase = (mod_rate_output[i] * t + mod_phases[i]) % 1.0
                lfo_output[i] = lfo_value(phase, mod_waveform)
            else:
                lfo_output[i] = 0.0

        # Matrix modulation LFO (scalar)
        if mod_depth_matrix > 0.0:
            mat_phase = (mod_rate_matrix * t) % 1.0
            lfo_mat = lfo_value(mat_phase, mod_waveform)
            mat_blend = 0.5 + 0.5 * lfo_mat * mod_depth_matrix  # 0-1 blend
        else:
            mat_blend = 0.0

        # --- Read from delay lines (with fractional delay) ---
        wet_L = 0.0
        wet_R = 0.0
        reads = np.empty(N)
        for i in range(N):
            wi = delay_write_idxs[i]

            # Modulated delay time
            current_delay = delay_times_base[i] + mod_depth_delay[i] * lfo_delay[i]
            current_delay = max(1.0, current_delay)  # minimum 1 sample

            # Fractional delay read
            reads[i] = read_delay_frac(delay_bufs[i], wi, current_delay, delay_buf_len)

            # Modulated output gain
            current_out_gain = output_gains_base[i] * (1.0 + mod_depth_output[i] * lfo_output[i])
            current_out_gain = max(0.0, current_out_gain)

            tap = reads[i] * current_out_gain
            wet_L += tap * pan_gain_L[i]
            wet_R += tap * pan_gain_R[i]

        # --- Damping with modulated coefficients ---
        for i in range(N):
            current_damp = damping_coeffs_base[i] + mod_depth_damping[i] * lfo_damping[i]
            current_damp = max(0.0, min(0.999, current_damp))
            damping_y1[i] = (1.0 - current_damp) * reads[i] + current_damp * damping_y1[i]
            reads[i] = damping_y1[i]

        # --- Feedback matrix multiply (with optional blending) ---
        mixed = np.empty(N)
        for i in range(N):
            s = 0.0
            for j in range(N):
                if mat_blend > 0.0:
                    m = matrix[i, j] * (1.0 - mat_blend) + matrix2[i, j] * mat_blend
                else:
                    m = matrix[i, j]
                s += m * reads[j]
            mixed[i] = s

        # --- Write back (with optional saturation + DC blocker) ---
        for i in range(N):
            wi = delay_write_idxs[i]
            val = feedback_gain * mixed[i] + input_gains[i] * diffused
            if saturation > 0.0:
                val = (1.0 - saturation) * val + saturation * np.tanh(val)
            dc_y = val - dc_x1[i] + dc_R * dc_y1[i]
            dc_x1[i] = val
            dc_y1[i] = dc_y
            delay_bufs[i, wi] = dc_y
            delay_write_idxs[i] = (wi + 1) % delay_buf_len

        # --- Wet/dry mix (stereo) ---
        output[n, 0] = (1.0 - wet_dry) * x + wet_dry * wet_L
        output[n, 1] = (1.0 - wet_dry) * x + wet_dry * wet_R

    pre_delay_write_idx[0] = pd_wi


# ---------------------------------------------------------------------------
# Setup + Entry Point
# ---------------------------------------------------------------------------

def render_fdn_mod(input_audio: np.ndarray, params: dict) -> np.ndarray:
    """Modulated FDN rendering — drop-in replacement when modulation is active."""
    from primitives.matrix import get_matrix, MATRIX_TYPES
    from engine.params import SR as sample_rate

    n_samples = len(input_audio)

    # --- Build feedback matrices ---
    matrix_type = params.get("matrix_type", "householder")
    if matrix_type == "custom" and "matrix_custom" in params:
        matrix = np.array(params["matrix_custom"], dtype=np.float64)
    elif matrix_type in MATRIX_TYPES:
        matrix = get_matrix(matrix_type, N, seed=params.get("matrix_seed", 42))
    else:
        matrix = get_matrix("householder", N)

    # Second matrix for blending
    matrix2_type = params.get("mod_matrix2_type", "random_orthogonal")
    matrix2_seed = params.get("mod_matrix2_seed", 137)
    if matrix2_type in MATRIX_TYPES:
        matrix2 = get_matrix(matrix2_type, N, seed=matrix2_seed)
    else:
        matrix2 = get_matrix("random_orthogonal", N, seed=matrix2_seed)

    # --- Pre-delay ---
    pre_delay_samples = max(1, int(params["pre_delay"]))
    pre_delay_len = pre_delay_samples + 1
    pre_delay_buf = np.zeros(pre_delay_len)
    pre_delay_write_idx = np.array([0], dtype=np.int64)

    # --- Diffusion allpasses ---
    n_diff_stages = min(params.get("diffusion_stages", 4),
                        len(params.get("diffusion_delays", [])))
    diff_delays = params.get("diffusion_delays", [])
    max_diff_len = max(diff_delays[:n_diff_stages]) if n_diff_stages > 0 else 1
    diff_bufs = np.zeros((max(n_diff_stages, 1), max_diff_len))
    diff_lens = np.array([diff_delays[i] for i in range(n_diff_stages)] or [1], dtype=np.int64)
    diff_gain = params.get("diffusion", 0.5)
    diff_gains = np.full(max(n_diff_stages, 1), diff_gain)
    diff_idxs = np.zeros(max(n_diff_stages, 1), dtype=np.int64)

    # --- FDN delay lines (enlarged buffer for modulation excursion) ---
    delay_times_base = np.array(params["delay_times"], dtype=np.float64)
    mod_depth_delay = np.array(params.get("mod_depth_delay", [0.0] * N), dtype=np.float64)
    max_delay = int(np.max(delay_times_base + np.abs(mod_depth_delay))) + 4
    delay_buf_len = max_delay + 1
    delay_bufs = np.zeros((N, delay_buf_len))
    delay_write_idxs = np.zeros(N, dtype=np.int64)

    # --- Damping ---
    damping_coeffs_base = np.array(params["damping_coeffs"], dtype=np.float64)
    damping_y1 = np.zeros(N)

    # --- Gains ---
    feedback_gain = float(params["feedback_gain"])
    input_gains = np.array(params["input_gains"], dtype=np.float64)
    output_gains_base = np.array(params["output_gains"], dtype=np.float64)
    wet_dry = float(params["wet_dry"])
    saturation = float(params.get("saturation", 0.0))

    # --- DC blocker ---
    dc_R = 1.0 - 2.0 * np.pi * 5.0 / sample_rate
    dc_x1 = np.zeros(N)
    dc_y1 = np.zeros(N)

    # --- Stereo panning ---
    node_pans = np.array(params.get("node_pans",
        [-1.0, -0.714, -0.429, -0.143, 0.143, 0.429, 0.714, 1.0]),
        dtype=np.float64)
    stereo_width = float(params.get("stereo_width", 1.0))
    pan_gain_L = np.empty(N, dtype=np.float64)
    pan_gain_R = np.empty(N, dtype=np.float64)
    for i in range(N):
        pan = node_pans[i] * stereo_width
        angle = (pan + 1.0) * (np.pi / 4.0)
        pan_gain_L[i] = np.cos(angle)
        pan_gain_R[i] = np.sin(angle)

    # --- Modulation parameters ---
    master_rate = float(params.get("mod_master_rate", 1.0))
    node_rate_mult = np.array(params.get("mod_node_rate_mult", [1.0] * N), dtype=np.float64)
    correlation = float(params.get("mod_correlation", 1.0))
    mod_waveform = int(params.get("mod_waveform", 0))

    delay_rate_scale = float(params.get("mod_rate_scale_delay", 1.0))
    damping_rate_scale = float(params.get("mod_rate_scale_damping", 1.0))
    output_rate_scale = float(params.get("mod_rate_scale_output", 1.0))

    mod_rate_delay = np.array([master_rate * node_rate_mult[i] * delay_rate_scale
                               for i in range(N)], dtype=np.float64)
    mod_rate_damping = np.array([master_rate * node_rate_mult[i] * damping_rate_scale
                                 for i in range(N)], dtype=np.float64)
    mod_rate_output = np.array([master_rate * node_rate_mult[i] * output_rate_scale
                                for i in range(N)], dtype=np.float64)

    mod_depth_damping = np.array(params.get("mod_depth_damping", [0.0] * N), dtype=np.float64)
    mod_depth_output = np.array(params.get("mod_depth_output", [0.0] * N), dtype=np.float64)

    mod_depth_matrix = float(params.get("mod_depth_matrix", 0.0))
    mod_rate_matrix = float(params.get("mod_rate_matrix", master_rate))

    # Phase offsets: correlated (all same) vs independent (spread evenly)
    base_phases = np.array([i / N for i in range(N)], dtype=np.float64)
    mod_phases = base_phases * (1.0 - correlation)

    output = np.empty((n_samples, 2), dtype=np.float64)

    _process_block_mod(
        input_audio.astype(np.float64), output,
        pre_delay_buf, pre_delay_len, pre_delay_samples, pre_delay_write_idx,
        diff_bufs, diff_lens, diff_gains, diff_idxs, n_diff_stages,
        delay_bufs, delay_buf_len, delay_times_base, delay_write_idxs,
        damping_coeffs_base, damping_y1,
        matrix, matrix2,
        feedback_gain, input_gains, output_gains_base, wet_dry,
        saturation,
        dc_x1, dc_y1, dc_R,
        pan_gain_L, pan_gain_R,
        mod_depth_delay, mod_rate_delay,
        mod_depth_damping, mod_rate_damping,
        mod_depth_output, mod_rate_output,
        mod_depth_matrix, mod_rate_matrix,
        mod_waveform, mod_phases,
        float(sample_rate),
    )

    return output
