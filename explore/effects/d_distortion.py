"""D-series: Distortion effects (D001-D016)."""
import numpy as np
import numba


# ---------------------------------------------------------------------------
# D001 — Hard Clipping
# ---------------------------------------------------------------------------

@numba.njit
def _hard_clip(samples, threshold, pre_gain):
    n = len(samples)
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        x = samples[i] * pre_gain
        if x > threshold:
            out[i] = threshold
        elif x < -threshold:
            out[i] = -threshold
        else:
            out[i] = x
    return out


def effect_d001_hard_clipping(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    threshold = np.float32(params.get('threshold', 0.5))
    pre_gain = np.float32(params.get('pre_gain', 4.0))
    return _hard_clip(samples.astype(np.float32), threshold, pre_gain)


def variants_d001():
    return [
        {'threshold': 0.8, 'pre_gain': 1.5},    # subtle warmth, barely clips
        {'threshold': 0.5, 'pre_gain': 4.0},     # moderate crunch
        {'threshold': 0.3, 'pre_gain': 8.0},     # aggressive distortion
        {'threshold': 0.1, 'pre_gain': 15.0},    # extreme square-wave destruction
        {'threshold': 0.05, 'pre_gain': 20.0},   # nearly pure square wave
    ]


# ---------------------------------------------------------------------------
# D002 — Soft Clipping (Tanh)
# ---------------------------------------------------------------------------

@numba.njit
def _soft_clip_tanh(samples, drive):
    n = len(samples)
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        out[i] = np.float32(np.tanh(samples[i] * drive))
    return out


def effect_d002_soft_clipping_tanh(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    drive = np.float32(params.get('drive', 3.0))
    return _soft_clip_tanh(samples.astype(np.float32), drive)


def variants_d002():
    return [
        {'drive': 1.2},    # gentle saturation, barely audible
        {'drive': 3.0},    # warm overdrive
        {'drive': 7.0},    # crunchy distortion
        {'drive': 12.0},   # heavy saturation
        {'drive': 20.0},   # extreme, almost hard clip
    ]


# ---------------------------------------------------------------------------
# D003 — Tube Saturation
# ---------------------------------------------------------------------------

@numba.njit
def _tube_saturation(samples, drive, asymmetry):
    n = len(samples)
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        x = samples[i]
        d_pos = drive * (1.0 + asymmetry)
        d_neg = drive * (1.0 - asymmetry)
        if x >= 0.0:
            out[i] = np.float32(1.0 - np.exp(-d_pos * x))
        else:
            out[i] = np.float32(-(1.0 - np.exp(d_neg * x)))
    return out


def effect_d003_tube_saturation(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    drive = np.float32(params.get('drive', 3.0))
    asymmetry = np.float32(params.get('asymmetry', 0.1))
    return _tube_saturation(samples.astype(np.float32), drive, asymmetry)


def variants_d003():
    return [
        {'drive': 1.5, 'asymmetry': 0.0},    # clean tube warmth, symmetric
        {'drive': 3.0, 'asymmetry': 0.1},     # classic tube overdrive
        {'drive': 6.0, 'asymmetry': 0.25},    # pushed tube with even harmonics
        {'drive': 10.0, 'asymmetry': 0.0},    # high-gain symmetric saturation
        {'drive': 10.0, 'asymmetry': 0.5},    # max asymmetry, buzzy harmonics
        {'drive': 2.0, 'asymmetry': 0.4},     # mild drive but strong asymmetry
    ]


# ---------------------------------------------------------------------------
# D004 — Foldback Distortion
# ---------------------------------------------------------------------------

@numba.njit
def _foldback(samples, threshold, pre_gain):
    n = len(samples)
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        x = samples[i] * pre_gain
        # Iterative foldback: fold until within [-threshold, threshold]
        for _ in range(20):  # max iterations to prevent infinite loop
            if x > threshold:
                x = threshold - (x - threshold)
            elif x < -threshold:
                x = -threshold - (x + threshold)
            else:
                break
        out[i] = np.float32(x)
    return out


def effect_d004_foldback_distortion(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    threshold = np.float32(params.get('threshold', 0.5))
    pre_gain = np.float32(params.get('pre_gain', 5.0))
    return _foldback(samples.astype(np.float32), threshold, pre_gain)


def variants_d004():
    return [
        {'threshold': 0.8, 'pre_gain': 2.0},     # gentle fold, occasional crinkle
        {'threshold': 0.5, 'pre_gain': 5.0},      # moderate folding
        {'threshold': 0.3, 'pre_gain': 10.0},     # aggressive metallic texture
        {'threshold': 0.2, 'pre_gain': 20.0},     # extreme aliased harmonics
        {'threshold': 0.1, 'pre_gain': 30.0},     # chaotic waveform mangling
        {'threshold': 0.6, 'pre_gain': 8.0},      # mid-range with complex harmonics
    ]


# ---------------------------------------------------------------------------
# D005 — Chebyshev Polynomial Waveshaper
# ---------------------------------------------------------------------------

@numba.njit
def _chebyshev_waveshape(samples, order, coeffs):
    n = len(samples)
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        x = samples[i]
        # Clamp to [-1, 1] for Chebyshev polynomials
        if x > 1.0:
            x = 1.0
        elif x < -1.0:
            x = -1.0
        # Compute Chebyshev polynomials T_0 through T_order
        # T_0(x) = 1, T_1(x) = x, T_n(x) = 2*x*T_{n-1}(x) - T_{n-2}(x)
        y = np.float32(0.0)
        t_prev2 = np.float32(1.0)   # T_0
        t_prev1 = x                  # T_1
        # Add T_1 component (fundamental passthrough not weighted by coeffs here)
        # We start weighting from T_2 onwards using coeffs
        y += x  # Always pass through fundamental
        for k in range(2, order + 1):
            t_curr = np.float32(2.0 * x * t_prev1 - t_prev2)
            idx = k - 2
            if idx < len(coeffs):
                y += coeffs[idx] * t_curr
            t_prev2 = t_prev1
            t_prev1 = t_curr
        out[i] = y
    return out


def effect_d005_chebyshev_waveshaper(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    order = int(params.get('order', 5))
    coeffs = params.get('coefficients', [0.5, 0.3, 0.1, 0.0, 0.0, 0.0])
    coeffs_arr = np.array(coeffs[:order], dtype=np.float32)
    # Pad if needed
    if len(coeffs_arr) < order:
        coeffs_arr = np.concatenate([coeffs_arr, np.zeros(order - len(coeffs_arr), dtype=np.float32)])
    return _chebyshev_waveshape(samples.astype(np.float32), order, coeffs_arr)


def variants_d005():
    return [
        {'order': 2, 'coefficients': [0.3]},                          # pure 2nd harmonic, warm
        {'order': 3, 'coefficients': [0.0, 0.5]},                     # pure 3rd harmonic, hollow
        {'order': 5, 'coefficients': [0.5, 0.3, 0.2, 0.1]},          # rich harmonic blend
        {'order': 8, 'coefficients': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]},# dense upper harmonics
        {'order': 4, 'coefficients': [0.8, 0.0, 0.4]},               # even harmonics emphasis
        {'order': 6, 'coefficients': [0.0, 0.6, 0.0, 0.4, 0.0]},     # odd harmonics only
    ]


# ---------------------------------------------------------------------------
# D006 — Polynomial Waveshaper
# ---------------------------------------------------------------------------

@numba.njit
def _polynomial_waveshape(samples, a1, a2, a3, a4, a5):
    n = len(samples)
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        x = samples[i]
        x2 = x * x
        x3 = x2 * x
        x4 = x2 * x2
        x5 = x4 * x
        out[i] = np.float32(a1 * x + a2 * x2 + a3 * x3 + a4 * x4 + a5 * x5)
    return out


def effect_d006_polynomial_waveshaper(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    a1 = np.float32(params.get('a1', 1.0))
    a2 = np.float32(params.get('a2', 0.0))
    a3 = np.float32(params.get('a3', -0.3))
    a4 = np.float32(params.get('a4', 0.0))
    a5 = np.float32(params.get('a5', 0.1))
    return _polynomial_waveshape(samples.astype(np.float32), a1, a2, a3, a4, a5)


def variants_d006():
    return [
        {'a1': 1.0, 'a2': 0.0, 'a3': -0.3, 'a4': 0.0, 'a5': 0.0},   # gentle cubic softening
        {'a1': 1.0, 'a2': 0.5, 'a3': 0.0, 'a4': 0.0, 'a5': 0.0},    # asymmetric warmth (even harmonics)
        {'a1': 0.8, 'a2': 0.0, 'a3': -0.5, 'a4': 0.0, 'a5': 0.3},   # odd harmonic richness
        {'a1': 1.5, 'a2': -1.0, 'a3': 0.5, 'a4': 0.3, 'a5': -0.2},  # complex waveshape
        {'a1': 0.5, 'a2': 1.5, 'a3': -1.0, 'a4': 0.5, 'a5': 0.5},   # aggressive nonlinear
        {'a1': 2.0, 'a2': -2.0, 'a3': 2.0, 'a4': -2.0, 'a5': 2.0},  # extreme oscillating transfer
    ]


# ---------------------------------------------------------------------------
# D007 — Sigmoid Family
# ---------------------------------------------------------------------------

@numba.njit
def _sigmoid_atan(samples, drive):
    n = len(samples)
    out = np.empty(n, dtype=np.float32)
    scale = np.float32(2.0 / np.pi)
    for i in range(n):
        out[i] = np.float32(scale * np.arctan(samples[i] * drive))
    return out


@numba.njit
def _sigmoid_algebraic(samples, drive):
    n = len(samples)
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        x = samples[i] * drive
        out[i] = np.float32(x / np.sqrt(1.0 + x * x))
    return out


@numba.njit
def _sigmoid_erf(samples, drive):
    n = len(samples)
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        x = samples[i] * drive
        # Approximate erf using tanh approximation: erf(x) ~ tanh(sqrt(pi)*x * (1 + 0.044715*x^2) * ...)
        # Simple: erf(x) ~ tanh(1.1283791671 * x) for small x, gets close enough
        # Use a polynomial approx that numba can handle
        # Abramowitz & Stegun approx for erf
        ax = abs(x)
        t = 1.0 / (1.0 + 0.3275911 * ax)
        poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
        e = 1.0 - poly * np.exp(-ax * ax)
        if x < 0.0:
            e = -e
        out[i] = np.float32(e)
    return out


def effect_d007_sigmoid_family(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    drive = np.float32(params.get('drive', 5.0))
    sig_type = params.get('type', 'atan')
    s = samples.astype(np.float32)
    if sig_type == 'atan':
        return _sigmoid_atan(s, drive)
    elif sig_type == 'erf':
        return _sigmoid_erf(s, drive)
    elif sig_type == 'algebraic':
        return _sigmoid_algebraic(s, drive)
    else:
        return _sigmoid_atan(s, drive)


def variants_d007():
    return [
        {'drive': 2.0, 'type': 'atan'},        # warm subtle atan
        {'drive': 10.0, 'type': 'atan'},        # aggressive atan
        {'drive': 50.0, 'type': 'atan'},        # extreme atan, near hard clip
        {'drive': 5.0, 'type': 'erf'},          # smooth erf saturation
        {'drive': 20.0, 'type': 'erf'},         # heavy erf saturation
        {'drive': 3.0, 'type': 'algebraic'},    # gentle algebraic curve
        {'drive': 30.0, 'type': 'algebraic'},   # harsh algebraic limiting
    ]


# ---------------------------------------------------------------------------
# D008 — Bit Crusher
# ---------------------------------------------------------------------------

@numba.njit
def _bit_crush(samples, bits):
    n = len(samples)
    out = np.empty(n, dtype=np.float32)
    levels = np.float32(2.0 ** bits)
    for i in range(n):
        # Quantize to N bits in [-1, 1] range
        out[i] = np.float32(np.floor(samples[i] * levels + 0.5) / levels)
    return out


def effect_d008_bit_crusher(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    bits = int(params.get('bits', 8))
    return _bit_crush(samples.astype(np.float32), bits)


def variants_d008():
    return [
        {'bits': 12},   # subtle quantization noise
        {'bits': 8},    # classic 8-bit lo-fi
        {'bits': 6},    # noticeable staircase
        {'bits': 4},    # chunky retro
        {'bits': 2},    # extreme 4-level quantization
        {'bits': 1},    # binary: only two levels, pure square
    ]


# ---------------------------------------------------------------------------
# D009 — Sample Rate Reduction
# ---------------------------------------------------------------------------

@numba.njit
def _sample_rate_reduce(samples, factor):
    n = len(samples)
    out = np.empty(n, dtype=np.float32)
    held = np.float32(0.0)
    counter = 0
    for i in range(n):
        if counter <= 0:
            held = samples[i]
            counter = factor
        out[i] = held
        counter -= 1
    return out


def effect_d009_sample_rate_reduction(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    target_sr = int(params.get('target_sr', 8000))
    # Compute hold factor: how many original samples per output sample
    factor = max(1, sr // target_sr)
    return _sample_rate_reduce(samples.astype(np.float32), factor)


def variants_d009():
    return [
        {'target_sr': 16000},   # mild aliasing, like telephone
        {'target_sr': 8000},    # classic lo-fi, obvious stepping
        {'target_sr': 4000},    # heavily aliased, fuzzy
        {'target_sr': 2000},    # extreme aliasing, metallic
        {'target_sr': 1000},    # near-unusable, rhythmic artifacts
        {'target_sr': 500},     # pure pitched buzzing
    ]


# ---------------------------------------------------------------------------
# D010 — Slew Rate Limiter
# ---------------------------------------------------------------------------

@numba.njit
def _slew_rate_limit(samples, max_slew):
    n = len(samples)
    out = np.empty(n, dtype=np.float32)
    out[0] = samples[0]
    for i in range(1, n):
        diff = samples[i] - out[i - 1]
        if diff > max_slew:
            out[i] = out[i - 1] + max_slew
        elif diff < -max_slew:
            out[i] = out[i - 1] - max_slew
        else:
            out[i] = samples[i]
    return out


def effect_d010_slew_rate_limiter(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    max_slew = np.float32(params.get('max_slew', 0.05))
    return _slew_rate_limit(samples.astype(np.float32), max_slew)


def variants_d010():
    return [
        {'max_slew': 0.3},     # barely noticeable on most signals
        {'max_slew': 0.1},     # gentle high-frequency rolloff
        {'max_slew': 0.05},    # audible smoothing, rounded transients
        {'max_slew': 0.01},    # heavy slewing, triangular waves
        {'max_slew': 0.005},   # extreme low-pass effect via slewing
        {'max_slew': 0.001},   # near-DC, everything becomes slow ramps
    ]


# ---------------------------------------------------------------------------
# D011 — Diode Clipper
# ---------------------------------------------------------------------------

@numba.njit
def _diode_clipper(samples, forward_voltage, num_diodes):
    n = len(samples)
    out = np.empty(n, dtype=np.float32)
    # Simple cubic approximation of diode clipping
    # Each diode adds a forward voltage drop
    vt = forward_voltage * num_diodes
    for i in range(n):
        x = samples[i]
        ax = abs(x)
        if ax <= vt:
            # Below threshold: slight nonlinearity via cubic
            # y = x - (x^3) / (3 * vt^2) gives smooth onset
            out[i] = np.float32(x - (x * x * x) / (3.0 * vt * vt))
        else:
            # Above threshold: soft saturate
            s = np.float32(1.0) if x >= 0.0 else np.float32(-1.0)
            out[i] = np.float32(s * (vt - vt / (3.0 * vt * vt) * vt * vt * vt + 0.0))
            # Simplify: output = sign * (2/3 * vt) at hard limit
            out[i] = np.float32(s * (2.0 / 3.0 * vt))
    return out


def effect_d011_diode_clipper(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    forward_voltage = np.float32(params.get('forward_voltage', 0.5))
    num_diodes = int(params.get('num_diodes', 2))
    # Apply some pre-gain to push signal into clipping
    pre_gain = np.float32(params.get('pre_gain', 3.0))
    s = samples.astype(np.float32) * pre_gain
    return _diode_clipper(s, forward_voltage, num_diodes)


def variants_d011():
    return [
        {'forward_voltage': 0.7, 'num_diodes': 1, 'pre_gain': 2.0},   # single germanium diode, gentle
        {'forward_voltage': 0.5, 'num_diodes': 2, 'pre_gain': 3.0},   # classic dual-diode clip
        {'forward_voltage': 0.3, 'num_diodes': 2, 'pre_gain': 5.0},   # low Vf, easy clipping
        {'forward_voltage': 0.2, 'num_diodes': 4, 'pre_gain': 4.0},   # many diodes, high headroom
        {'forward_voltage': 0.7, 'num_diodes': 4, 'pre_gain': 8.0},   # high Vf + gain = aggressive
        {'forward_voltage': 0.3, 'num_diodes': 1, 'pre_gain': 10.0},  # single diode overdriven hard
    ]


# ---------------------------------------------------------------------------
# D012 — Rectification
# ---------------------------------------------------------------------------

@numba.njit
def _rectify_half(samples, bias):
    n = len(samples)
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        x = samples[i] + bias
        if x > 0.0:
            out[i] = np.float32(x)
        else:
            out[i] = np.float32(0.0)
    return out


@numba.njit
def _rectify_full(samples, bias):
    n = len(samples)
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        out[i] = np.float32(abs(samples[i] + bias))
    return out


@numba.njit
def _rectify_biased(samples, bias):
    n = len(samples)
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        x = samples[i] + bias
        # Biased rectification: pass signal above bias threshold
        if x > 0.0:
            out[i] = np.float32(x)
        else:
            out[i] = np.float32(x * 0.1)  # leak a bit of negative
    return out


def effect_d012_rectification(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    rect_type = params.get('type', 'full')
    bias = np.float32(params.get('bias', 0.0))
    s = samples.astype(np.float32)
    if rect_type == 'half':
        return _rectify_half(s, bias)
    elif rect_type == 'full':
        return _rectify_full(s, bias)
    elif rect_type == 'biased':
        return _rectify_biased(s, bias)
    else:
        return _rectify_full(s, bias)


def variants_d012():
    return [
        {'type': 'full', 'bias': 0.0},      # classic full-wave: octave up effect
        {'type': 'half', 'bias': 0.0},       # half-wave: buzzy octave up + fundamental
        {'type': 'full', 'bias': 0.3},       # biased full: asymmetric octave
        {'type': 'half', 'bias': -0.2},      # biased half: more signal passes
        {'type': 'biased', 'bias': 0.0},     # leaky rectifier, subtle grit
        {'type': 'biased', 'bias': 0.4},     # heavy bias offset, sputtery
        {'type': 'half', 'bias': 0.5},       # extreme bias, mostly silent with bursts
    ]


# ---------------------------------------------------------------------------
# D013 — Dynamic Waveshaping
# ---------------------------------------------------------------------------

@numba.njit
def _dynamic_waveshape(samples, base_drive, env_drive, env_speed):
    n = len(samples)
    out = np.empty(n, dtype=np.float32)
    env = np.float32(0.0)
    for i in range(n):
        # Envelope follower
        inp = abs(samples[i])
        if inp > env:
            env = env + env_speed * (inp - env)
        else:
            env = env - env_speed * 0.25 * (env - inp)  # slower release
        # Drive depends on envelope
        drive = base_drive + env_drive * env
        # Apply tanh waveshaping with dynamic drive
        out[i] = np.float32(np.tanh(samples[i] * drive))
    return out


def effect_d013_dynamic_waveshaping(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    base_drive = np.float32(params.get('base_drive', 2.0))
    env_drive = np.float32(params.get('env_drive', 8.0))
    env_speed = np.float32(params.get('env_speed', 0.01))
    return _dynamic_waveshape(samples.astype(np.float32), base_drive, env_drive, env_speed)


def variants_d013():
    return [
        {'base_drive': 1.0, 'env_drive': 5.0, 'env_speed': 0.01},    # quiet=clean, loud=crunchy
        {'base_drive': 1.0, 'env_drive': 15.0, 'env_speed': 0.005},  # dramatic dynamic range
        {'base_drive': 3.0, 'env_drive': 10.0, 'env_speed': 0.05},   # always dirty, louder=filthier
        {'base_drive': 5.0, 'env_drive': 0.0, 'env_speed': 0.01},    # static high drive (control)
        {'base_drive': 1.0, 'env_drive': 20.0, 'env_speed': 0.1},    # fast envelope, percussive grit
        {'base_drive': 2.0, 'env_drive': 10.0, 'env_speed': 0.001},  # very slow envelope, swelling distortion
    ]


# ---------------------------------------------------------------------------
# D014 — XOR / Bitwise Distortion
# ---------------------------------------------------------------------------

@numba.njit
def _bitwise_distortion(samples, bit_depth, operation, pattern):
    n = len(samples)
    out = np.empty(n, dtype=np.float32)
    max_val = (1 << bit_depth) - 1
    half_val = max_val // 2
    pat = np.int64(pattern) & np.int64(max_val)
    for i in range(n):
        # Quantize float to integer
        x = samples[i]
        if x > 1.0:
            x = 1.0
        elif x < -1.0:
            x = -1.0
        quantized = np.int64((x * 0.5 + 0.5) * max_val) & np.int64(max_val)
        # Apply bitwise operation
        if operation == 0:    # XOR
            result = quantized ^ pat
        elif operation == 1:  # AND
            result = quantized & pat
        else:                 # OR
            result = quantized | pat
        # Convert back to float [-1, 1]
        out[i] = np.float32((np.float64(result) / np.float64(max_val)) * 2.0 - 1.0)
    return out


def effect_d014_bitwise_distortion(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    bit_depth = int(params.get('bit_depth', 12))
    op_name = params.get('operation', 'xor')
    op_map = {'xor': 0, 'and': 1, 'or': 2}
    operation = op_map.get(op_name, 0)
    pattern = int(params.get('pattern', 0xAA))
    return _bitwise_distortion(samples.astype(np.float32), bit_depth, operation, pattern)


def variants_d014():
    return [
        {'bit_depth': 16, 'operation': 'xor', 'pattern': 0x00FF},   # XOR lower byte, subtle glitch
        {'bit_depth': 12, 'operation': 'xor', 'pattern': 0xAAA},    # alternating bit XOR
        {'bit_depth': 8, 'operation': 'xor', 'pattern': 0xFF},      # 8-bit full XOR invert
        {'bit_depth': 8, 'operation': 'and', 'pattern': 0xF0},      # AND mask, lose low bits
        {'bit_depth': 10, 'operation': 'or', 'pattern': 0x155},     # OR pattern, boost bits
        {'bit_depth': 8, 'operation': 'xor', 'pattern': 0x55},      # checkerboard XOR, metallic
        {'bit_depth': 12, 'operation': 'and', 'pattern': 0xFC0},    # aggressive bit masking
    ]


# ---------------------------------------------------------------------------
# D015 — Modular Arithmetic Distortion
# ---------------------------------------------------------------------------

@numba.njit
def _modular_arithmetic(samples, scale, modulus, offset):
    n = len(samples)
    out = np.empty(n, dtype=np.float32)
    half_mod = modulus * 0.5
    for i in range(n):
        x = samples[i] * scale + offset
        # Modular wrap: bring into [0, modulus) then center
        # Use fmod and adjust for negative values
        y = x - modulus * np.floor(x / modulus)  # equivalent to x % modulus for floats
        out[i] = np.float32(y - half_mod)
    return out


def effect_d015_modular_arithmetic(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    scale = np.float32(params.get('scale', 3.0))
    modulus = np.float32(params.get('modulus', 1.0))
    offset = np.float32(params.get('offset', 0.0))
    return _modular_arithmetic(samples.astype(np.float32), scale, modulus, offset)


def variants_d015():
    return [
        {'scale': 2.0, 'modulus': 1.5, 'offset': 0.0},     # gentle wrap, occasional fold
        {'scale': 4.0, 'modulus': 1.0, 'offset': 0.0},      # sawtooth-like wrapping
        {'scale': 8.0, 'modulus': 0.5, 'offset': 0.0},      # dense harmonic wrapping
        {'scale': 3.0, 'modulus': 0.8, 'offset': 0.5},      # offset adds asymmetry
        {'scale': 10.0, 'modulus': 0.3, 'offset': 0.0},     # extreme modular, buzzy texture
        {'scale': 1.5, 'modulus': 2.0, 'offset': 0.0},      # wide modulus, subtle effect
        {'scale': 6.0, 'modulus': 0.2, 'offset': 0.3},      # chaotic rapid wrapping
    ]


# ---------------------------------------------------------------------------
# D016 — Serge-Style Wavefolder
# Asymmetric wavefolder with sine and triangle fold curves.
# Unlike D004's symmetric foldback, this uses shaped transfer functions
# that produce distinctly different even/odd harmonic content.
# ---------------------------------------------------------------------------

@numba.njit
def _serge_wavefold(samples, pre_gain, fold_type, stages, asymmetry):
    """Wavefolder with selectable fold curve.

    fold_type: 0=sine, 1=triangle, 2=tanh-fold
    """
    n = len(samples)
    out = np.empty(n, dtype=np.float32)
    pi = np.float32(3.141592653589793)

    for i in range(n):
        x = samples[i] * pre_gain

        # Apply asymmetry bias
        x = x + asymmetry

        # Multi-stage folding
        for _ in range(stages):
            if fold_type == 0:
                # Sine fold: sin(pi * x) — creates smooth harmonic folds
                x = np.float32(np.sin(pi * x))
            elif fold_type == 1:
                # Triangle fold: reflected triangle wave
                # Normalize to [-1,1] via triangle wave function
                x = np.float32(2.0 * abs(x * 0.5 - np.floor(x * 0.5 + 0.5)) * 2.0 - 1.0)
            else:
                # Tanh-fold: soft fold using tanh wrapping
                x = np.float32(np.tanh(np.sin(x * pi * 0.5)))

        # Remove asymmetry DC offset
        out[i] = x

    return out


def effect_d016_serge_wavefolder(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Serge-style wavefolder with sine, triangle, or tanh fold curves.
    Multi-stage folding creates increasingly complex harmonic content."""
    pre_gain = np.float32(params.get('pre_gain', 3.0))
    fold_type_str = params.get('fold_type', 'sine')
    stages = int(params.get('stages', 1))
    asymmetry = np.float32(params.get('asymmetry', 0.0))

    fold_map = {'sine': 0, 'triangle': 1, 'tanh': 2}
    fold_type = fold_map.get(fold_type_str, 0)

    return _serge_wavefold(samples.astype(np.float32), pre_gain, fold_type,
                           stages, asymmetry)


def variants_d016():
    return [
        {'pre_gain': 2.0, 'fold_type': 'sine', 'stages': 1, 'asymmetry': 0.0},
        {'pre_gain': 4.0, 'fold_type': 'sine', 'stages': 2, 'asymmetry': 0.0},
        {'pre_gain': 8.0, 'fold_type': 'sine', 'stages': 3, 'asymmetry': 0.2},
        {'pre_gain': 3.0, 'fold_type': 'triangle', 'stages': 1, 'asymmetry': 0.0},
        {'pre_gain': 6.0, 'fold_type': 'triangle', 'stages': 2, 'asymmetry': 0.1},
        {'pre_gain': 10.0, 'fold_type': 'triangle', 'stages': 3, 'asymmetry': 0.3},
        {'pre_gain': 3.0, 'fold_type': 'tanh', 'stages': 1, 'asymmetry': 0.0},
        {'pre_gain': 7.0, 'fold_type': 'tanh', 'stages': 2, 'asymmetry': 0.15},
        {'pre_gain': 5.0, 'fold_type': 'sine', 'stages': 1, 'asymmetry': 0.4},
    ]
