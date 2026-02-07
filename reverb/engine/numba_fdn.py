"""Numba-optimized FDN inner loop.

Same algorithm as engine/fdn.py, but all state is flat numpy arrays
so Numba can JIT the entire per-sample loop.
"""

import numpy as np
from numba import njit

N = 8


@njit(cache=True)
def _process_block(
    input_audio,
    output,
    # Pre-delay state
    pre_delay_buf, pre_delay_len, pre_delay_samples, pre_delay_write_idx,
    # Diffusion allpass state (flattened)
    diff_bufs, diff_lens, diff_gains, diff_idxs, n_diff_stages,
    # FDN delay line state
    delay_bufs, delay_buf_len, delay_times, delay_write_idxs,
    # Damping filter state
    damping_coeffs, damping_y1,
    # Feedback matrix (8x8)
    matrix,
    # Gains
    feedback_gain, input_gains, output_gains, wet_dry,
    # Saturation
    saturation,
    # DC blocker state (per node)
    dc_x1, dc_y1, dc_R,
    # Stereo panning (pre-computed L/R gains per node)
    pan_gain_L, pan_gain_R,
):
    n_samples = len(input_audio)
    pd_wi = pre_delay_write_idx[0]

    for n in range(n_samples):
        x = input_audio[n]

        # --- Pre-delay ---
        pre_delay_buf[pd_wi] = x
        pd_wi = (pd_wi + 1) % pre_delay_len
        rd_idx = (pd_wi - 1 - pre_delay_samples) % pre_delay_len
        x_delayed = pre_delay_buf[rd_idx]

        # --- Input diffusion (allpass chain) ---
        diffused = x_delayed
        for s in range(n_diff_stages):
            idx = diff_idxs[s]
            delayed = diff_bufs[s, idx]
            g = diff_gains[s]
            v = diffused + g * delayed
            diffused = -g * v + delayed
            diff_bufs[s, idx] = v
            diff_idxs[s] = (idx + 1) % diff_lens[s]

        # --- Read from delay lines ---
        wet_L = 0.0
        wet_R = 0.0
        reads = np.empty(N)
        for i in range(N):
            wi = delay_write_idxs[i]
            rd = (wi - 1 - delay_times[i]) % delay_buf_len
            reads[i] = delay_bufs[i, rd]
            tap = reads[i] * output_gains[i]
            wet_L += tap * pan_gain_L[i]
            wet_R += tap * pan_gain_R[i]

        # --- Damping (one-pole lowpass) ---
        for i in range(N):
            a = damping_coeffs[i]
            damping_y1[i] = (1.0 - a) * reads[i] + a * damping_y1[i]
            reads[i] = damping_y1[i]

        # --- Feedback matrix multiply ---
        mixed = np.empty(N)
        for i in range(N):
            s = 0.0
            for j in range(N):
                s += matrix[i, j] * reads[j]
            mixed[i] = s

        # --- Write back to delay lines (with optional saturation + DC blocker) ---
        for i in range(N):
            wi = delay_write_idxs[i]
            val = feedback_gain * mixed[i] + input_gains[i] * diffused
            if saturation > 0.0:
                val = (1.0 - saturation) * val + saturation * np.tanh(val)
            # DC blocker: y[n] = x[n] - x[n-1] + R * y[n-1]
            dc_y = val - dc_x1[i] + dc_R * dc_y1[i]
            dc_x1[i] = val
            dc_y1[i] = dc_y
            delay_bufs[i, wi] = dc_y
            delay_write_idxs[i] = (wi + 1) % delay_buf_len

        # --- Wet/dry mix (stereo) ---
        output[n, 0] = (1.0 - wet_dry) * x + wet_dry * wet_L
        output[n, 1] = (1.0 - wet_dry) * x + wet_dry * wet_R

    pre_delay_write_idx[0] = pd_wi


def render_fdn_fast(input_audio: np.ndarray, params: dict,
                    chunk_callback=None, chunk_size=4096) -> np.ndarray:
    """Drop-in replacement for engine.fdn.render_fdn, but Numba-accelerated."""
    from reverb.primitives.matrix import build_matrix_apply, get_matrix, MATRIX_TYPES

    n_samples = len(input_audio)

    # --- Build feedback matrix as array ---
    matrix_type = params.get("matrix_type", "householder")
    if matrix_type == "custom" and "matrix_custom" in params:
        matrix = np.array(params["matrix_custom"], dtype=np.float64)
    elif matrix_type in MATRIX_TYPES:
        matrix = get_matrix(matrix_type, N, seed=params.get("matrix_seed", 42))
    else:
        matrix = get_matrix("householder", N)

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

    # --- FDN delay lines ---
    delay_times = np.array(params["delay_times"], dtype=np.int64)
    delay_buf_len = int(np.max(delay_times)) + 1
    delay_bufs = np.zeros((N, delay_buf_len))
    delay_write_idxs = np.zeros(N, dtype=np.int64)

    # --- Damping ---
    damping_coeffs = np.array(params["damping_coeffs"], dtype=np.float64)
    damping_y1 = np.zeros(N)

    # --- Gains ---
    feedback_gain = float(params["feedback_gain"])
    input_gains = np.array(params["input_gains"], dtype=np.float64)
    output_gains = np.array(params["output_gains"], dtype=np.float64)
    wet_dry = float(params["wet_dry"])
    saturation = float(params.get("saturation", 0.0))

    # --- DC blocker state ---
    # One-pole high-pass at ~5 Hz: R = 1 - 2*pi*fc/SR
    from reverb.engine.params import SR as sample_rate
    dc_R = 1.0 - 2.0 * np.pi * 5.0 / sample_rate
    dc_x1 = np.zeros(N)
    dc_y1 = np.zeros(N)

    # --- Stereo panning: pre-compute per-node L/R gains ---
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

    input_f64 = input_audio.astype(np.float64)
    output = np.empty((n_samples, 2), dtype=np.float64)

    def _call_block(inp, out):
        _process_block(
            inp, out,
            pre_delay_buf, pre_delay_len, pre_delay_samples, pre_delay_write_idx,
            diff_bufs, diff_lens, diff_gains, diff_idxs, n_diff_stages,
            delay_bufs, delay_buf_len, delay_times, delay_write_idxs,
            damping_coeffs, damping_y1,
            matrix,
            feedback_gain, input_gains, output_gains, wet_dry,
            saturation,
            dc_x1, dc_y1, dc_R,
            pan_gain_L, pan_gain_R,
        )

    if chunk_callback is None:
        _call_block(input_f64, output)
    else:
        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            _call_block(input_f64[start:end], output[start:end])
            if not chunk_callback(output[start:end]):
                break

    return output
