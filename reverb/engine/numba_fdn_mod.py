"""Numba-optimized modulated FDN inner loop.

Extends the static FDN (numba_fdn.py) with per-sample LFO modulation of
delay times, damping coefficients, output gains, and feedback matrix blend.
Uses fractional delay interpolation for smooth delay-time modulation.
"""

import math
import numpy as np
from numba import njit

N = 8

# Matrix type constants
MATRIX_HOUSEHOLDER = 0
MATRIX_GENERIC = 1


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
        return math.sin(2.0 * math.pi * phase)
    elif waveform == 1:  # triangle
        if phase < 0.25:
            return phase * 4.0
        elif phase < 0.75:
            return 2.0 - phase * 4.0
        else:
            return phase * 4.0 - 4.0
    else:  # sample-and-hold (waveform == 2)
        n = int(phase * 1000.0)
        h = ((n * 1103515245 + 12345) >> 16) & 0x7FFF
        return (h / 16383.5) - 1.0


# ---------------------------------------------------------------------------
# Fractional Delay (inlined by Numba when called from njit)
# ---------------------------------------------------------------------------

@njit(cache=True)
def read_delay_frac(buf, write_idx, delay_frac, buf_len):
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
    # Matrix type flag (0=householder O(N), 1=generic O(N^2))
    matrix_type_flag,
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
    # Phase increments (pre-computed: rate / sample_rate)
    phase_inc_delay, phase_inc_damping, phase_inc_output, phase_inc_matrix,
    # Initial phases (pre-computed for sample_offset)
    init_phases_delay, init_phases_damping, init_phases_output, init_phase_matrix,
    # Active masks (pre-computed booleans)
    any_delay_mod, any_damping_mod, any_output_mod,
):
    n_samples = len(input_audio)
    pd_wi = pre_delay_write_idx[0]
    dry_gain = 1.0 - wet_dry

    # Pre-allocate scratch arrays once
    reads = np.empty(N)
    mixed = np.empty(N)

    # Running LFO phases (avoid per-sample division)
    cur_phase_delay = np.empty(N)
    cur_phase_damping = np.empty(N)
    cur_phase_output = np.empty(N)
    for i in range(N):
        cur_phase_delay[i] = init_phases_delay[i]
        cur_phase_damping[i] = init_phases_damping[i]
        cur_phase_output[i] = init_phases_output[i]
    cur_phase_matrix = init_phase_matrix

    for n in range(n_samples):
        x = input_audio[n]

        # --- Pre-delay ---
        pre_delay_buf[pd_wi] = x
        pd_wi = (pd_wi + 1) % pre_delay_len
        rd_idx = (pd_wi - 1 - pre_delay_samples) % pre_delay_len
        x_delayed = pre_delay_buf[rd_idx]

        # --- Input diffusion ---
        diffused = x_delayed
        for s in range(n_diff_stages):
            idx = diff_idxs[s]
            delayed = diff_bufs[s, idx]
            g = diff_gains[s]
            v = diffused + g * delayed
            diffused = -g * v + delayed
            diff_bufs[s, idx] = v
            diff_idxs[s] = (idx + 1) % diff_lens[s]

        # --- Compute per-node LFO values using phase accumulators ---
        # Matrix modulation LFO (scalar)
        if mod_depth_matrix > 0.0:
            lfo_mat = lfo_value(cur_phase_matrix, mod_waveform)
            mat_blend = 0.5 + 0.5 * lfo_mat * mod_depth_matrix
            cur_phase_matrix = (cur_phase_matrix + phase_inc_matrix) % 1.0
        else:
            mat_blend = 0.0

        # --- Read from delay lines (with fractional delay) ---
        wet_L = 0.0
        wet_R = 0.0
        for i in range(N):
            wi = delay_write_idxs[i]

            # Modulated delay time
            if any_delay_mod and mod_depth_delay[i] > 0.0:
                lfo_d = lfo_value(cur_phase_delay[i], mod_waveform)
                current_delay = delay_times_base[i] + mod_depth_delay[i] * lfo_d
                if current_delay < 1.0:
                    current_delay = 1.0
            else:
                current_delay = delay_times_base[i]

            # Fractional delay read
            reads[i] = read_delay_frac(delay_bufs[i], wi, current_delay, delay_buf_len)

            # Modulated output gain
            if any_output_mod and mod_depth_output[i] > 0.0:
                lfo_o = lfo_value(cur_phase_output[i], mod_waveform)
                current_out_gain = output_gains_base[i] * (1.0 + mod_depth_output[i] * lfo_o)
                if current_out_gain < 0.0:
                    current_out_gain = 0.0
            else:
                current_out_gain = output_gains_base[i]

            tap = reads[i] * current_out_gain
            wet_L += tap * pan_gain_L[i]
            wet_R += tap * pan_gain_R[i]

        # --- Damping with modulated coefficients ---
        for i in range(N):
            if any_damping_mod and mod_depth_damping[i] > 0.0:
                lfo_da = lfo_value(cur_phase_damping[i], mod_waveform)
                current_damp = damping_coeffs_base[i] + mod_depth_damping[i] * lfo_da
                if current_damp < 0.0:
                    current_damp = 0.0
                elif current_damp > 0.999:
                    current_damp = 0.999
            else:
                current_damp = damping_coeffs_base[i]
            damping_y1[i] = (1.0 - current_damp) * reads[i] + current_damp * damping_y1[i]
            reads[i] = damping_y1[i]

        # --- Advance LFO phases (cheap addition instead of per-sample division) ---
        for i in range(N):
            cur_phase_delay[i] = (cur_phase_delay[i] + phase_inc_delay[i]) % 1.0
            cur_phase_damping[i] = (cur_phase_damping[i] + phase_inc_damping[i]) % 1.0
            cur_phase_output[i] = (cur_phase_output[i] + phase_inc_output[i]) % 1.0

        # --- Feedback matrix multiply (with optional blending) ---
        if mat_blend > 0.0:
            inv_blend = 1.0 - mat_blend
            for i in range(N):
                s = 0.0
                for j in range(N):
                    m = matrix[i, j] * inv_blend + matrix2[i, j] * mat_blend
                    s += m * reads[j]
                mixed[i] = s
        elif matrix_type_flag == MATRIX_HOUSEHOLDER:
            # Householder O(N): mixed = reads - (2/N) * sum(reads)
            s = 0.0
            for i in range(N):
                s += reads[i]
            s *= (2.0 / N)
            for i in range(N):
                mixed[i] = reads[i] - s
        else:
            for i in range(N):
                s = 0.0
                for j in range(N):
                    s += matrix[i, j] * reads[j]
                mixed[i] = s

        # --- Write back (with optional saturation + DC blocker) ---
        for i in range(N):
            wi = delay_write_idxs[i]
            val = feedback_gain * mixed[i] + input_gains[i] * diffused
            if saturation > 0.0:
                val = (1.0 - saturation) * val + saturation * math.tanh(val)
            dc_y = val - dc_x1[i] + dc_R * dc_y1[i]
            dc_x1[i] = val
            dc_y1[i] = dc_y
            delay_bufs[i, wi] = dc_y
            delay_write_idxs[i] = (wi + 1) % delay_buf_len

        # --- Wet/dry mix (stereo) ---
        output[n, 0] = dry_gain * x + wet_dry * wet_L
        output[n, 1] = dry_gain * x + wet_dry * wet_R

    pre_delay_write_idx[0] = pd_wi


# ---------------------------------------------------------------------------
# Setup + Entry Point
# ---------------------------------------------------------------------------

def render_fdn_mod(input_audio: np.ndarray, params: dict,
                   chunk_callback=None, chunk_size=4096) -> np.ndarray:
    """Modulated FDN rendering — drop-in replacement when modulation is active."""
    from reverb.primitives.matrix import get_matrix, MATRIX_TYPES
    from reverb.engine.params import SR as sample_rate

    n_samples = len(input_audio)

    # --- Build feedback matrices ---
    matrix_type = params.get("matrix_type", "householder")
    if matrix_type == "custom" and "matrix_custom" in params:
        matrix = np.array(params["matrix_custom"], dtype=np.float64)
        matrix_type_flag = MATRIX_GENERIC
    elif matrix_type in MATRIX_TYPES:
        matrix = get_matrix(matrix_type, N, seed=params.get("matrix_seed", 42))
        matrix_type_flag = MATRIX_HOUSEHOLDER if matrix_type == "householder" else MATRIX_GENERIC
    else:
        matrix = get_matrix("householder", N)
        matrix_type_flag = MATRIX_HOUSEHOLDER

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

    # --- Stereo panning (vectorized) ---
    node_pans = np.array(params.get("node_pans",
        [-1.0, -0.714, -0.429, -0.143, 0.143, 0.429, 0.714, 1.0]),
        dtype=np.float64)
    stereo_width = float(params.get("stereo_width", 1.0))
    angles = (node_pans * stereo_width + 1.0) * (np.pi / 4.0)
    pan_gain_L = np.cos(angles)
    pan_gain_R = np.sin(angles)

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

    # Pre-compute phase increments (rate / sample_rate) — replaces per-sample division
    sr = float(sample_rate)
    phase_inc_delay = mod_rate_delay / sr
    phase_inc_damping = mod_rate_damping / sr
    phase_inc_output = mod_rate_output / sr
    phase_inc_matrix = mod_rate_matrix / sr

    # Pre-compute active masks
    any_delay_mod = bool(np.any(mod_depth_delay > 0.0))
    any_damping_mod = bool(np.any(mod_depth_damping > 0.0))
    any_output_mod = bool(np.any(mod_depth_output > 0.0))

    input_f64 = input_audio.astype(np.float64)
    output = np.empty((n_samples, 2), dtype=np.float64)

    def _call_block(inp, out, s_offset):
        # Compute initial phases for this block's offset
        t0 = s_offset / sr
        init_delay = np.empty(N, dtype=np.float64)
        init_damping = np.empty(N, dtype=np.float64)
        init_output = np.empty(N, dtype=np.float64)
        for i in range(N):
            init_delay[i] = (mod_rate_delay[i] * t0 + mod_phases[i]) % 1.0
            init_damping[i] = (mod_rate_damping[i] * t0 + mod_phases[i]) % 1.0
            init_output[i] = (mod_rate_output[i] * t0 + mod_phases[i]) % 1.0
        init_mat = (mod_rate_matrix * t0) % 1.0

        _process_block_mod(
            inp, out,
            pre_delay_buf, pre_delay_len, pre_delay_samples, pre_delay_write_idx,
            diff_bufs, diff_lens, diff_gains, diff_idxs, n_diff_stages,
            delay_bufs, delay_buf_len, delay_times_base, delay_write_idxs,
            damping_coeffs_base, damping_y1,
            matrix, matrix2, matrix_type_flag,
            feedback_gain, input_gains, output_gains_base, wet_dry,
            saturation,
            dc_x1, dc_y1, dc_R,
            pan_gain_L, pan_gain_R,
            mod_depth_delay, mod_rate_delay,
            mod_depth_damping, mod_rate_damping,
            mod_depth_output, mod_rate_output,
            mod_depth_matrix, mod_rate_matrix,
            mod_waveform, mod_phases,
            phase_inc_delay, phase_inc_damping, phase_inc_output, phase_inc_matrix,
            init_delay, init_damping, init_output, init_mat,
            any_delay_mod, any_damping_mod, any_output_mod,
        )

    if chunk_callback is None:
        _call_block(input_f64, output, 0)
    else:
        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            _call_block(input_f64[start:end], output[start:end], start)
            if not chunk_callback(output[start:end]):
                break

    return output
