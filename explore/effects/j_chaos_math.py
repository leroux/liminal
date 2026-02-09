"""J-series effects: Chaos and mathematical algorithms (J001-J020)."""
import numpy as np
import numba


# ---------------------------------------------------------------------------
# Shared STFT / ISTFT helpers (for J009, J010, J015, J016, J018)
# ---------------------------------------------------------------------------

def _stft(x, fft_size=2048, hop_size=512):
    window = np.hanning(fft_size).astype(np.float32)
    n = len(x)
    if n < fft_size:
        x = np.concatenate([x, np.zeros(fft_size - n, dtype=np.float32)])
        n = fft_size
    num_frames = 1 + (n - fft_size) // hop_size
    frames = np.zeros((num_frames, fft_size), dtype=np.float32)
    for i in range(num_frames):
        start = i * hop_size
        frames[i] = x[start:start + fft_size] * window
    return np.fft.rfft(frames, axis=1)


def _istft(X, fft_size=2048, hop_size=512, length=None):
    window = np.hanning(fft_size).astype(np.float32)
    frames = np.fft.irfft(X, n=fft_size, axis=1).astype(np.float32) * window
    num_frames = X.shape[0]
    if length is None:
        length = fft_size + (num_frames - 1) * hop_size
    output = np.zeros(length, dtype=np.float32)
    for i in range(num_frames):
        start = i * hop_size
        end = min(start + fft_size, length)
        output[start:end] += frames[i, :end - start]
    return output


# ---------------------------------------------------------------------------
# J001 -- Logistic Map Modulator
# ---------------------------------------------------------------------------

@numba.njit
def _logistic_map_modulator(samples, r, step_rate, mod_target, mod_depth):
    """Logistic map x_{n+1} = r * x * (1 - x) modulates amplitude or cutoff."""
    n = len(samples)
    out = np.zeros(n, dtype=np.float32)

    x = np.float32(0.5)  # initial state
    mod_val = np.float32(0.0)
    step_counter = 0

    # One-pole filter state for cutoff modulation
    lp_state = np.float32(0.0)

    for i in range(n):
        step_counter += 1
        if step_counter >= step_rate:
            step_counter = 0
            x = r * x * (np.float32(1.0) - x)
            # Clamp to avoid divergence
            if x < np.float32(0.0):
                x = np.float32(0.01)
            elif x > np.float32(1.0):
                x = np.float32(0.99)
            mod_val = x

        if mod_target == 0:
            # Amplitude modulation
            gain = np.float32(1.0) - mod_depth + mod_depth * mod_val
            out[i] = samples[i] * gain
        else:
            # Cutoff modulation (one-pole lowpass)
            coeff = np.float32(0.05) + np.float32(0.94) * mod_depth * mod_val
            lp_state = coeff * lp_state + (np.float32(1.0) - coeff) * samples[i]
            out[i] = lp_state

    return out


def effect_j001_logistic_map_modulator(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Logistic map modulates amplitude or filter cutoff chaotically."""
    r = np.float32(params.get('r', 3.8))
    step_rate = int(params.get('step_rate', 100))
    mod_target_str = params.get('mod_target', 'amplitude')
    mod_depth = np.float32(params.get('mod_depth', 0.5))

    mod_target = 0 if mod_target_str == 'amplitude' else 1
    step_rate = max(1, step_rate)

    return _logistic_map_modulator(
        samples.astype(np.float32), r, step_rate, mod_target, mod_depth
    )


def variants_j001():
    return [
        {'r': 3.57, 'step_rate': 200, 'mod_target': 'amplitude', 'mod_depth': 0.3},
        {'r': 3.8, 'step_rate': 100, 'mod_target': 'amplitude', 'mod_depth': 0.6},
        {'r': 3.99, 'step_rate': 50, 'mod_target': 'amplitude', 'mod_depth': 1.0},
        {'r': 3.7, 'step_rate': 500, 'mod_target': 'cutoff', 'mod_depth': 0.5},
        {'r': 3.95, 'step_rate': 30, 'mod_target': 'cutoff', 'mod_depth': 0.8},
        {'r': 3.6, 'step_rate': 150, 'mod_target': 'amplitude', 'mod_depth': 0.2},
    ]


# ---------------------------------------------------------------------------
# J002 -- Logistic Map Waveshaper
# ---------------------------------------------------------------------------

@numba.njit
def _logistic_waveshaper(samples, r, num_iterations):
    """Run logistic map from x0=|sample|, take value after N iterations."""
    n = len(samples)
    out = np.zeros(n, dtype=np.float32)

    for i in range(n):
        x = abs(samples[i])
        # Clamp to (0, 1) for logistic map stability
        if x >= np.float32(1.0):
            x = np.float32(0.999)
        if x <= np.float32(0.0):
            x = np.float32(0.001)

        for _ in range(num_iterations):
            x = r * x * (np.float32(1.0) - x)
            if x < np.float32(0.0):
                x = np.float32(0.01)
            elif x > np.float32(1.0):
                x = np.float32(0.99)

        # Map back to [-1, 1] preserving sign
        sign = np.float32(1.0) if samples[i] >= 0.0 else np.float32(-1.0)
        out[i] = sign * (np.float32(2.0) * x - np.float32(1.0))

    return out


def effect_j002_logistic_map_waveshaper(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Waveshaping via logistic map iterations seeded from input amplitude."""
    r = np.float32(params.get('r', 3.8))
    num_iterations = int(params.get('num_iterations', 5))
    num_iterations = max(1, min(num_iterations, 20))
    return _logistic_waveshaper(samples.astype(np.float32), r, num_iterations)


def variants_j002():
    return [
        {'r': 3.5, 'num_iterations': 2},
        {'r': 3.7, 'num_iterations': 5},
        {'r': 3.85, 'num_iterations': 8},
        {'r': 3.95, 'num_iterations': 3},
        {'r': 4.0, 'num_iterations': 12},
        {'r': 3.6, 'num_iterations': 20},
    ]


# ---------------------------------------------------------------------------
# J003 -- Lorenz Attractor Modulation
# ---------------------------------------------------------------------------

@numba.njit
def _lorenz_modulation(samples, sigma, rho, beta, integration_speed, mod_depth):
    """Lorenz attractor modulates amplitude via x-coordinate."""
    n = len(samples)
    out = np.zeros(n, dtype=np.float32)

    # Lorenz state, start near attractor
    x = np.float64(1.0)
    y = np.float64(1.0)
    z = np.float64(1.0)

    dt = np.float64(integration_speed)

    for i in range(n):
        # Euler integration
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        x += dx * dt
        y += dy * dt
        z += dz * dt

        # Normalize x to roughly [-1, 1] (Lorenz x typically in [-20, 20])
        mod_val = np.float32(x / 20.0)
        if mod_val > np.float32(1.0):
            mod_val = np.float32(1.0)
        elif mod_val < np.float32(-1.0):
            mod_val = np.float32(-1.0)

        # Modulate amplitude
        gain = np.float32(1.0) + mod_depth * mod_val
        if gain < np.float32(0.0):
            gain = np.float32(0.0)
        out[i] = samples[i] * gain

    return out


def effect_j003_lorenz_attractor_modulation(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Lorenz attractor state modulates audio amplitude chaotically."""
    sigma = np.float64(10.0)
    beta = np.float64(8.0 / 3.0)
    rho = np.float64(params.get('rho', 28.0))
    integration_speed = np.float64(params.get('integration_speed', 0.005))
    mod_depth = np.float32(params.get('mod_depth', 0.5))

    return _lorenz_modulation(
        samples.astype(np.float32), sigma, rho, beta,
        integration_speed, mod_depth
    )


def variants_j003():
    return [
        {'rho': 20.0, 'integration_speed': 0.001, 'mod_depth': 0.3},
        {'rho': 28.0, 'integration_speed': 0.005, 'mod_depth': 0.5},
        {'rho': 28.0, 'integration_speed': 0.01, 'mod_depth': 0.8},
        {'rho': 35.0, 'integration_speed': 0.003, 'mod_depth': 1.0},
        {'rho': 24.0, 'integration_speed': 0.008, 'mod_depth': 0.4},
        {'rho': 32.0, 'integration_speed': 0.002, 'mod_depth': 0.7},
    ]


# ---------------------------------------------------------------------------
# J004 -- Lorenz as Audio-Rate Signal
# ---------------------------------------------------------------------------

@numba.njit
def _lorenz_audio_rate(samples, sigma, rho, beta, mix_mode, mix_amount):
    """Integrate Lorenz at audio rate, use as signal for ring mod / add / AM."""
    n = len(samples)
    out = np.zeros(n, dtype=np.float32)

    x = np.float64(0.1)
    y = np.float64(0.0)
    z = np.float64(0.0)
    dt = np.float64(0.001)

    for i in range(n):
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        x += dx * dt
        y += dy * dt
        z += dz * dt

        # Normalize to [-1, 1]
        lorenz_val = np.float32(x / 25.0)
        if lorenz_val > np.float32(1.0):
            lorenz_val = np.float32(1.0)
        elif lorenz_val < np.float32(-1.0):
            lorenz_val = np.float32(-1.0)

        if mix_mode == 0:
            # Ring modulation
            out[i] = samples[i] * lorenz_val * mix_amount + samples[i] * (np.float32(1.0) - mix_amount)
        elif mix_mode == 1:
            # Additive
            out[i] = samples[i] + lorenz_val * mix_amount
        else:
            # AM: (1 + lorenz) * signal
            am = (np.float32(1.0) + lorenz_val * mix_amount) * np.float32(0.5)
            out[i] = samples[i] * am

    return out


def effect_j004_lorenz_audio_rate(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Lorenz attractor integrated at audio rate as ring mod / additive / AM source."""
    sigma = np.float64(10.0)
    beta = np.float64(8.0 / 3.0)
    rho = np.float64(params.get('rho', 28.0))
    mix_mode_str = params.get('mix_mode', 'ring_mod')
    mix_amount = np.float32(params.get('mix_amount', 0.5))

    mode_map = {'ring_mod': 0, 'add': 1, 'am': 2}
    mix_mode = mode_map.get(mix_mode_str, 0)

    return _lorenz_audio_rate(
        samples.astype(np.float32), sigma, rho, beta, mix_mode, mix_amount
    )


def variants_j004():
    return [
        {'rho': 28.0, 'mix_mode': 'ring_mod', 'mix_amount': 0.3},
        {'rho': 28.0, 'mix_mode': 'ring_mod', 'mix_amount': 0.7},
        {'rho': 35.0, 'mix_mode': 'add', 'mix_amount': 0.4},
        {'rho': 22.0, 'mix_mode': 'am', 'mix_amount': 0.5},
        {'rho': 28.0, 'mix_mode': 'am', 'mix_amount': 0.8},
        {'rho': 30.0, 'mix_mode': 'add', 'mix_amount': 0.2},
    ]


# ---------------------------------------------------------------------------
# J005 -- Henon Map Distortion
# ---------------------------------------------------------------------------

@numba.njit
def _henon_map_distortion(samples, a, b):
    """Henon map x_{n+1} = 1 - a*x^2 + y, y_{n+1} = b*x applied sample-by-sample."""
    n = len(samples)
    out = np.zeros(n, dtype=np.float32)

    hx = np.float32(0.0)
    hy = np.float32(0.0)

    for i in range(n):
        # Use input as perturbation to the map
        inp = samples[i]
        new_hx = np.float32(1.0) - a * hx * hx + hy + inp * np.float32(0.1)
        new_hy = b * hx

        # Clamp to prevent explosion
        if new_hx > np.float32(2.0):
            new_hx = np.float32(2.0)
        elif new_hx < np.float32(-2.0):
            new_hx = np.float32(-2.0)
        if new_hy > np.float32(2.0):
            new_hy = np.float32(2.0)
        elif new_hy < np.float32(-2.0):
            new_hy = np.float32(-2.0)

        hx = new_hx
        hy = new_hy

        # Output is mix of input and map state
        out[i] = np.float32(np.tanh(hx))

    return out


def effect_j005_henon_map_distortion(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Henon map driven by input signal produces chaotic waveshaping."""
    a = np.float32(params.get('a', 1.2))
    b = np.float32(params.get('b', 0.3))
    return _henon_map_distortion(samples.astype(np.float32), a, b)


def variants_j005():
    return [
        {'a': 1.0, 'b': 0.2},
        {'a': 1.2, 'b': 0.3},
        {'a': 1.3, 'b': 0.3},
        {'a': 1.4, 'b': 0.3},
        {'a': 1.1, 'b': 0.4},
        {'a': 1.35, 'b': 0.2},
    ]


# ---------------------------------------------------------------------------
# J006 -- Duffing Oscillator
# ---------------------------------------------------------------------------

@numba.njit
def _duffing_oscillator(samples, delta, alpha, beta, gamma, omega, sr):
    """Driven Duffing oscillator: x'' + delta*x' + alpha*x + beta*x^3 = gamma*cos(omega*t) + input."""
    n = len(samples)
    out = np.zeros(n, dtype=np.float32)

    x = np.float64(0.1)
    v = np.float64(0.0)
    dt = 1.0 / np.float64(sr)
    two_pi = 6.283185307179586

    for i in range(n):
        t = np.float64(i) * dt
        # Driving force: external sinusoid + audio input
        forcing = gamma * np.cos(two_pi * omega * t) + np.float64(samples[i])

        # Duffing equation: x'' = forcing - delta*x' - alpha*x - beta*x^3
        accel = forcing - delta * v - alpha * x - beta * x * x * x

        v += accel * dt
        x += v * dt

        # Clamp state to prevent blowup
        if x > 10.0:
            x = 10.0
        elif x < -10.0:
            x = -10.0
        if v > 100.0:
            v = 100.0
        elif v < -100.0:
            v = -100.0

        out[i] = np.float32(np.tanh(x * 0.2))

    return out


def effect_j006_duffing_oscillator(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Duffing oscillator: driven nonlinear resonator producing chaotic tones."""
    delta = np.float64(params.get('delta', 0.3))
    alpha = np.float64(params.get('alpha', -1.0))
    beta = np.float64(params.get('beta', 1.0))
    gamma = np.float64(params.get('gamma', 0.5))
    omega_hz = np.float64(params.get('omega_hz', 200.0))

    return _duffing_oscillator(
        samples.astype(np.float32), delta, alpha, beta, gamma, omega_hz, sr
    )


def variants_j006():
    return [
        {'delta': 0.1, 'alpha': -1.0, 'beta': 1.0, 'gamma': 0.3, 'omega_hz': 100},
        {'delta': 0.3, 'alpha': -1.0, 'beta': 1.0, 'gamma': 0.5, 'omega_hz': 200},
        {'delta': 0.5, 'alpha': 1.0, 'beta': 0.5, 'gamma': 1.0, 'omega_hz': 300},
        {'delta': 0.2, 'alpha': -0.5, 'beta': 2.0, 'gamma': 1.5, 'omega_hz': 80},
        {'delta': 0.15, 'alpha': 0.0, 'beta': 1.5, 'gamma': 0.8, 'omega_hz': 500},
        {'delta': 0.4, 'alpha': -1.0, 'beta': 1.0, 'gamma': 0.1, 'omega_hz': 50},
    ]


# ---------------------------------------------------------------------------
# J007 -- Double Pendulum Modulation
# ---------------------------------------------------------------------------

@numba.njit
def _double_pendulum_modulation(samples, l1, l2, m1, m2, g,
                                 theta1_0, theta2_0, mod_depth, sr):
    """Double pendulum with RK4 integration modulates audio amplitude."""
    n = len(samples)
    out = np.zeros(n, dtype=np.float32)

    theta1 = np.float64(theta1_0)
    theta2 = np.float64(theta2_0)
    omega1 = np.float64(0.0)
    omega2 = np.float64(0.0)

    # Time step: integrate at lower rate for performance, hold value
    steps_per_sample = 1
    dt = 1.0 / (np.float64(sr) * np.float64(steps_per_sample))

    for i in range(n):
        for _ in range(steps_per_sample):
            # Double pendulum equations of motion
            # Using Lagrangian mechanics
            d_theta = theta1 - theta2
            sin_d = np.sin(d_theta)
            cos_d = np.cos(d_theta)
            sin1 = np.sin(theta1)
            sin2 = np.sin(theta2)

            denom1 = (m1 + m2) * l1 - m2 * l1 * cos_d * cos_d
            denom2 = (l2 / l1) * denom1

            if abs(denom1) < 1e-10:
                denom1 = 1e-10
            if abs(denom2) < 1e-10:
                denom2 = 1e-10

            # Angular accelerations (simplified Lagrangian form)
            alpha1 = (-g * (m1 + m2) * sin1
                      - m2 * g * np.sin(theta1 - 2.0 * theta2) * 0.5  # correction term
                      - m2 * sin_d * (omega2 * omega2 * l2 + omega1 * omega1 * l1 * cos_d)
                      ) / denom1

            # Correction: swapped l1/l2 for second pendulum
            denom2_actual = l2 * (m1 + m2 - m2 * cos_d * cos_d)
            if abs(denom2_actual) < 1e-10:
                denom2_actual = 1e-10

            alpha2 = (sin_d * ((m1 + m2) * (omega1 * omega1 * l1 + g * np.cos(theta1))
                      + omega2 * omega2 * l2 * m2 * cos_d)
                      ) / denom2_actual

            # RK4 step
            # k1
            k1_t1 = omega1
            k1_t2 = omega2
            k1_o1 = alpha1
            k1_o2 = alpha2

            # Half step for k2
            t1_mid = theta1 + 0.5 * dt * k1_t1
            t2_mid = theta2 + 0.5 * dt * k1_t2
            o1_mid = omega1 + 0.5 * dt * k1_o1
            o2_mid = omega2 + 0.5 * dt * k1_o2

            d_mid = t1_mid - t2_mid
            sin_dm = np.sin(d_mid)
            cos_dm = np.cos(d_mid)
            den1m = (m1 + m2) * l1 - m2 * l1 * cos_dm * cos_dm
            if abs(den1m) < 1e-10:
                den1m = 1e-10
            den2m = l2 * (m1 + m2 - m2 * cos_dm * cos_dm)
            if abs(den2m) < 1e-10:
                den2m = 1e-10

            k2_t1 = o1_mid
            k2_t2 = o2_mid
            k2_o1 = (-g * (m1 + m2) * np.sin(t1_mid)
                      - m2 * g * np.sin(t1_mid - 2.0 * t2_mid) * 0.5
                      - m2 * sin_dm * (o2_mid * o2_mid * l2 + o1_mid * o1_mid * l1 * cos_dm)
                      ) / den1m
            k2_o2 = (sin_dm * ((m1 + m2) * (o1_mid * o1_mid * l1 + g * np.cos(t1_mid))
                      + o2_mid * o2_mid * l2 * m2 * cos_dm)
                      ) / den2m

            # k3
            t1_mid2 = theta1 + 0.5 * dt * k2_t1
            t2_mid2 = theta2 + 0.5 * dt * k2_t2
            o1_mid2 = omega1 + 0.5 * dt * k2_o1
            o2_mid2 = omega2 + 0.5 * dt * k2_o2

            d_mid2 = t1_mid2 - t2_mid2
            sin_dm2 = np.sin(d_mid2)
            cos_dm2 = np.cos(d_mid2)
            den1m2 = (m1 + m2) * l1 - m2 * l1 * cos_dm2 * cos_dm2
            if abs(den1m2) < 1e-10:
                den1m2 = 1e-10
            den2m2 = l2 * (m1 + m2 - m2 * cos_dm2 * cos_dm2)
            if abs(den2m2) < 1e-10:
                den2m2 = 1e-10

            k3_t1 = o1_mid2
            k3_t2 = o2_mid2
            k3_o1 = (-g * (m1 + m2) * np.sin(t1_mid2)
                      - m2 * g * np.sin(t1_mid2 - 2.0 * t2_mid2) * 0.5
                      - m2 * sin_dm2 * (o2_mid2 * o2_mid2 * l2 + o1_mid2 * o1_mid2 * l1 * cos_dm2)
                      ) / den1m2
            k3_o2 = (sin_dm2 * ((m1 + m2) * (o1_mid2 * o1_mid2 * l1 + g * np.cos(t1_mid2))
                      + o2_mid2 * o2_mid2 * l2 * m2 * cos_dm2)
                      ) / den2m2

            # k4
            t1_end = theta1 + dt * k3_t1
            t2_end = theta2 + dt * k3_t2
            o1_end = omega1 + dt * k3_o1
            o2_end = omega2 + dt * k3_o2

            d_end = t1_end - t2_end
            sin_de = np.sin(d_end)
            cos_de = np.cos(d_end)
            den1e = (m1 + m2) * l1 - m2 * l1 * cos_de * cos_de
            if abs(den1e) < 1e-10:
                den1e = 1e-10
            den2e = l2 * (m1 + m2 - m2 * cos_de * cos_de)
            if abs(den2e) < 1e-10:
                den2e = 1e-10

            k4_t1 = o1_end
            k4_t2 = o2_end
            k4_o1 = (-g * (m1 + m2) * np.sin(t1_end)
                      - m2 * g * np.sin(t1_end - 2.0 * t2_end) * 0.5
                      - m2 * sin_de * (o2_end * o2_end * l2 + o1_end * o1_end * l1 * cos_de)
                      ) / den1e
            k4_o2 = (sin_de * ((m1 + m2) * (o1_end * o1_end * l1 + g * np.cos(t1_end))
                      + o2_end * o2_end * l2 * m2 * cos_de)
                      ) / den2e

            # Update state
            theta1 += dt / 6.0 * (k1_t1 + 2.0 * k2_t1 + 2.0 * k3_t1 + k4_t1)
            theta2 += dt / 6.0 * (k1_t2 + 2.0 * k2_t2 + 2.0 * k3_t2 + k4_t2)
            omega1 += dt / 6.0 * (k1_o1 + 2.0 * k2_o1 + 2.0 * k3_o1 + k4_o1)
            omega2 += dt / 6.0 * (k1_o2 + 2.0 * k2_o2 + 2.0 * k3_o2 + k4_o2)

        # Use angular velocity of second pendulum as modulation source
        # Normalize: typical omega2 range is roughly [-10, 10]
        mod_val = np.float32(np.tanh(omega2 * 0.1))

        gain = np.float32(1.0) + mod_depth * mod_val
        if gain < np.float32(0.0):
            gain = np.float32(0.0)
        out[i] = samples[i] * gain

    return out


def effect_j007_double_pendulum_modulation(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Double pendulum chaotic trajectory modulates audio amplitude."""
    l1 = np.float64(params.get('l1', 1.5))
    l2 = np.float64(params.get('l2', 1.0))
    initial_angle1 = np.float64(params.get('initial_angle1', 2.0))
    initial_angle2 = np.float64(params.get('initial_angle2', 2.0))
    mod_depth = np.float32(params.get('mod_depth', 0.5))

    m1 = np.float64(1.0)
    m2 = np.float64(1.0)
    g = np.float64(9.81)

    return _double_pendulum_modulation(
        samples.astype(np.float32), l1, l2, m1, m2, g,
        initial_angle1, initial_angle2, mod_depth, sr
    )


def variants_j007():
    return [
        {'l1': 1.0, 'l2': 1.0, 'initial_angle1': 1.5, 'initial_angle2': 1.5, 'mod_depth': 0.3},
        {'l1': 1.5, 'l2': 1.0, 'initial_angle1': 2.0, 'initial_angle2': 2.0, 'mod_depth': 0.5},
        {'l1': 2.0, 'l2': 2.0, 'initial_angle1': 3.0, 'initial_angle2': 1.0, 'mod_depth': 0.8},
        {'l1': 1.0, 'l2': 2.0, 'initial_angle1': 2.5, 'initial_angle2': 3.0, 'mod_depth': 1.0},
        {'l1': 1.5, 'l2': 1.5, 'initial_angle1': 1.0, 'initial_angle2': 3.0, 'mod_depth': 0.6},
        {'l1': 2.0, 'l2': 1.0, 'initial_angle1': 2.8, 'initial_angle2': 1.5, 'mod_depth': 0.4},
    ]


# ---------------------------------------------------------------------------
# J008 -- Cellular Automaton Rhythm Gate
# ---------------------------------------------------------------------------

@numba.njit
def _ca_rhythm_gate(samples, rule, num_cells, cell_duration_samples):
    """1D elementary cellular automaton; each cell gates an audio chunk."""
    n = len(samples)
    out = np.zeros(n, dtype=np.float32)

    # Initialize CA state: single cell in center
    state = np.zeros(num_cells, dtype=np.int32)
    state[num_cells // 2] = 1

    # Total samples per CA row = num_cells * cell_duration_samples
    row_len = num_cells * cell_duration_samples
    num_rows = (n + row_len - 1) // row_len

    sample_idx = 0
    for row in range(num_rows):
        # Apply gate pattern from current state
        for cell in range(num_cells):
            for s in range(cell_duration_samples):
                if sample_idx >= n:
                    break
                if state[cell] == 1:
                    out[sample_idx] = samples[sample_idx]
                else:
                    out[sample_idx] = samples[sample_idx] * np.float32(0.05)  # near-silent
                sample_idx += 1

        # Evolve CA to next generation
        new_state = np.zeros(num_cells, dtype=np.int32)
        for c in range(num_cells):
            left = state[(c - 1) % num_cells]
            center = state[c]
            right = state[(c + 1) % num_cells]
            neighborhood = (left << 2) | (center << 1) | right
            if (rule >> neighborhood) & 1:
                new_state[c] = 1
            else:
                new_state[c] = 0
        for c in range(num_cells):
            state[c] = new_state[c]

    return out


def effect_j008_cellular_automaton_rhythm_gate(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """1D cellular automaton generates rhythm gate pattern for audio."""
    rule = int(params.get('rule', 30))
    num_cells = int(params.get('num_cells', 32))
    cell_duration_ms = float(params.get('cell_duration_ms', 50))

    rule = max(0, min(255, rule))
    num_cells = max(4, min(128, num_cells))
    cell_duration_samples = max(1, int(cell_duration_ms * sr / 1000.0))

    return _ca_rhythm_gate(
        samples.astype(np.float32), rule, num_cells, cell_duration_samples
    )


def variants_j008():
    return [
        {'rule': 30, 'num_cells': 32, 'cell_duration_ms': 50},
        {'rule': 110, 'num_cells': 64, 'cell_duration_ms': 30},
        {'rule': 90, 'num_cells': 16, 'cell_duration_ms': 100},
        {'rule': 150, 'num_cells': 48, 'cell_duration_ms': 20},
        {'rule': 60, 'num_cells': 128, 'cell_duration_ms': 10},
        {'rule': 184, 'num_cells': 24, 'cell_duration_ms': 80},
    ]


# ---------------------------------------------------------------------------
# J009 -- Cellular Automaton Spectral Mask
# ---------------------------------------------------------------------------

def effect_j009_cellular_automaton_spectral_mask(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """2D-like CA evolves across STFT frames to mask spectral bins."""
    rule = int(params.get('rule', 90))
    initial_density = float(params.get('initial_density', 0.5))
    rule = max(0, min(255, rule))

    samples = samples.astype(np.float32)
    fft_size = 2048
    hop_size = 512

    X = _stft(samples, fft_size, hop_size)
    num_frames, num_bins = X.shape

    # Initialize CA state from initial_density
    rng = np.random.RandomState(42)
    state = (rng.random(num_bins) < initial_density).astype(np.int32)

    for frame in range(num_frames):
        # Apply mask: bins where state==0 are attenuated
        for b in range(num_bins):
            if state[b] == 0:
                X[frame, b] *= 0.05

        # Evolve CA
        new_state = np.zeros(num_bins, dtype=np.int32)
        for c in range(num_bins):
            left = state[(c - 1) % num_bins]
            center = state[c]
            right = state[(c + 1) % num_bins]
            neighborhood = (left << 2) | (center << 1) | right
            if (rule >> neighborhood) & 1:
                new_state[c] = 1
        state = new_state

    return _istft(X, fft_size, hop_size, length=len(samples))


def variants_j009():
    return [
        {'rule': 30, 'initial_density': 0.5},
        {'rule': 90, 'initial_density': 0.4},
        {'rule': 110, 'initial_density': 0.6},
        {'rule': 150, 'initial_density': 0.3},
        {'rule': 60, 'initial_density': 0.7},
        {'rule': 184, 'initial_density': 0.5},
    ]


# ---------------------------------------------------------------------------
# J010 -- Reaction-Diffusion Spectrogram
# ---------------------------------------------------------------------------

def effect_j010_reaction_diffusion_spectrogram(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Gray-Scott reaction-diffusion on spectrogram magnitudes."""
    F = float(params.get('F', 0.04))
    k = float(params.get('k', 0.06))
    num_iterations = int(params.get('num_iterations', 50))

    samples = samples.astype(np.float32)
    fft_size = 2048
    hop_size = 512

    X = _stft(samples, fft_size, hop_size)
    mag = np.abs(X).astype(np.float64)
    phase = np.angle(X)

    num_frames, num_bins = mag.shape

    # Normalize magnitudes to [0, 1] for reaction-diffusion
    mag_max = mag.max()
    if mag_max < 1e-10:
        return samples.copy()
    mag_norm = mag / mag_max

    # Initialize U and V grids
    # U = "substrate", V = mag_norm (treat spectrogram as catalyst)
    U = np.ones((num_frames, num_bins), dtype=np.float64)
    V = mag_norm.copy()

    Du = 0.16
    Dv = 0.08
    dt_rd = 1.0

    for _ in range(num_iterations):
        # Laplacian via simple 2D convolution with [0,1,0; 1,-4,1; 0,1,0]
        lap_U = np.zeros_like(U)
        lap_V = np.zeros_like(V)

        # Interior points
        for r in range(1, num_frames - 1):
            for c in range(1, num_bins - 1):
                lap_U[r, c] = U[r-1, c] + U[r+1, c] + U[r, c-1] + U[r, c+1] - 4.0 * U[r, c]
                lap_V[r, c] = V[r-1, c] + V[r+1, c] + V[r, c-1] + V[r, c+1] - 4.0 * V[r, c]

        # Gray-Scott update
        UVV = U * V * V
        U += dt_rd * (Du * lap_U - UVV + F * (1.0 - U))
        V += dt_rd * (Dv * lap_V + UVV - (F + k) * V)

        # Clamp
        U = np.clip(U, 0.0, 1.0)
        V = np.clip(V, 0.0, 1.0)

    # Use V as new magnitude scaling
    new_mag = (V * mag_max).astype(np.float32)
    X_out = new_mag * np.exp(1j * phase).astype(np.complex64)

    return _istft(X_out, fft_size, hop_size, length=len(samples))


def variants_j010():
    return [
        {'F': 0.02, 'k': 0.05, 'num_iterations': 30},
        {'F': 0.04, 'k': 0.06, 'num_iterations': 50},
        {'F': 0.03, 'k': 0.055, 'num_iterations': 100},
        {'F': 0.06, 'k': 0.07, 'num_iterations': 20},
        {'F': 0.025, 'k': 0.045, 'num_iterations': 150},
        {'F': 0.05, 'k': 0.065, 'num_iterations': 80},
    ]


# ---------------------------------------------------------------------------
# J011 -- L-System Parameter Sequencer
# ---------------------------------------------------------------------------

@numba.njit
def _lsystem_filter_sequencer(samples, cutoff_sequence, step_duration_samples):
    """Apply one-pole lowpass with cutoff changing per step from L-system."""
    n = len(samples)
    out = np.zeros(n, dtype=np.float32)
    num_steps = len(cutoff_sequence)

    lp_state = np.float32(0.0)
    step_idx = 0
    step_counter = 0

    for i in range(n):
        # Advance step
        step_counter += 1
        if step_counter >= step_duration_samples and step_idx < num_steps - 1:
            step_counter = 0
            step_idx += 1

        # Get cutoff coefficient from sequence
        coeff = cutoff_sequence[step_idx]

        # One-pole lowpass
        lp_state = coeff * lp_state + (np.float32(1.0) - coeff) * samples[i]
        out[i] = lp_state

    return out


def _expand_lsystem(axiom, rules, iterations):
    """Expand L-system grammar for given iterations."""
    current = axiom
    for _ in range(iterations):
        next_str = []
        for ch in current:
            if ch in rules:
                next_str.append(rules[ch])
            else:
                next_str.append(ch)
        current = ''.join(next_str)
    return current


def effect_j011_lsystem_parameter_sequencer(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """L-system grammar generates symbol string mapped to filter cutoff changes."""
    axiom = str(params.get('axiom', 'F'))
    rules_str = str(params.get('rules_str', 'F->F+F-F'))
    iterations = int(params.get('iterations', 4))
    step_duration_ms = float(params.get('step_duration_ms', 50))

    iterations = max(1, min(iterations, 8))
    step_duration_samples = max(1, int(step_duration_ms * sr / 1000.0))

    # Parse rules
    rules = {}
    for part in rules_str.split(','):
        part = part.strip()
        if '->' in part:
            lhs, rhs = part.split('->', 1)
            rules[lhs.strip()] = rhs.strip()

    # Expand
    expanded = _expand_lsystem(axiom, rules, iterations)

    # Map symbols to cutoff coefficients
    # F = high cutoff (bright), + = increase, - = decrease, other = mid
    cutoff_val = 0.5
    cutoff_list = []
    for ch in expanded:
        if ch == 'F':
            cutoff_val = min(cutoff_val + 0.05, 0.98)
        elif ch == '+':
            cutoff_val = min(cutoff_val + 0.1, 0.98)
        elif ch == '-':
            cutoff_val = max(cutoff_val - 0.1, 0.05)
        else:
            cutoff_val = 0.5
        cutoff_list.append(cutoff_val)

    if len(cutoff_list) == 0:
        cutoff_list = [0.5]

    # Limit sequence length to prevent enormous arrays
    max_steps = 10000
    if len(cutoff_list) > max_steps:
        cutoff_list = cutoff_list[:max_steps]

    cutoff_arr = np.array(cutoff_list, dtype=np.float32)

    return _lsystem_filter_sequencer(
        samples.astype(np.float32), cutoff_arr, step_duration_samples
    )


def variants_j011():
    return [
        {'axiom': 'F', 'rules_str': 'F->F+F-F', 'iterations': 3, 'step_duration_ms': 50},
        {'axiom': 'F', 'rules_str': 'F->F+F-F', 'iterations': 5, 'step_duration_ms': 20},
        {'axiom': 'F', 'rules_str': 'F->FF+F-', 'iterations': 4, 'step_duration_ms': 40},
        {'axiom': 'F', 'rules_str': 'F->F-F+F+F-F', 'iterations': 3, 'step_duration_ms': 30},
        {'axiom': 'F', 'rules_str': 'F->F+F--F+F', 'iterations': 4, 'step_duration_ms': 80},
        {'axiom': 'F', 'rules_str': 'F->+F-F+', 'iterations': 6, 'step_duration_ms': 100},
    ]


# ---------------------------------------------------------------------------
# J012 -- IFS Audio
# ---------------------------------------------------------------------------

@numba.njit
def _ifs_audio(samples, num_transforms, contraction, iterations):
    """Iterated Function System on audio chunks: contractive affine transforms."""
    n = len(samples)
    out = np.zeros(n, dtype=np.float32)

    # Chunk size based on contraction and iterations
    chunk_size = max(64, n // 16)

    # Generate transform parameters deterministically
    # Each transform: scale and offset
    scales = np.zeros(num_transforms, dtype=np.float32)
    offsets = np.zeros(num_transforms, dtype=np.float32)
    for t in range(num_transforms):
        scales[t] = contraction
        offsets[t] = np.float32(t) / np.float32(num_transforms)

    num_chunks = max(1, n // chunk_size)

    for c in range(num_chunks):
        start = c * chunk_size
        end = min(start + chunk_size, n)
        actual_len = end - start

        # Copy chunk
        chunk = np.zeros(actual_len, dtype=np.float32)
        for j in range(actual_len):
            chunk[j] = samples[start + j]

        # Apply IFS iterations
        result = np.zeros(actual_len, dtype=np.float32)
        for j in range(actual_len):
            result[j] = chunk[j]

        for it in range(iterations):
            new_result = np.zeros(actual_len, dtype=np.float32)
            for t in range(num_transforms):
                for j in range(actual_len):
                    # Contractive affine: read from scaled position
                    src_pos = np.float32(j) * scales[t] + offsets[t] * np.float32(actual_len)
                    src_idx = int(src_pos) % actual_len
                    new_result[j] += result[src_idx] * scales[t]
            # Normalize
            max_val = np.float32(0.0)
            for j in range(actual_len):
                if abs(new_result[j]) > max_val:
                    max_val = abs(new_result[j])
            if max_val > np.float32(0.001):
                for j in range(actual_len):
                    new_result[j] /= max_val
            for j in range(actual_len):
                result[j] = new_result[j]

        for j in range(actual_len):
            out[start + j] = result[j]

    return out


def effect_j012_ifs_audio(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Iterated Function System applies contractive affine transforms to audio chunks."""
    num_transforms = int(params.get('num_transforms', 3))
    contraction = np.float32(params.get('contraction', 0.5))
    iterations = int(params.get('iterations', 5))

    num_transforms = max(2, min(num_transforms, 5))
    iterations = max(1, min(iterations, 10))

    return _ifs_audio(samples.astype(np.float32), num_transforms, contraction, iterations)


def variants_j012():
    return [
        {'num_transforms': 2, 'contraction': 0.5, 'iterations': 3},
        {'num_transforms': 3, 'contraction': 0.5, 'iterations': 5},
        {'num_transforms': 4, 'contraction': 0.4, 'iterations': 7},
        {'num_transforms': 5, 'contraction': 0.3, 'iterations': 10},
        {'num_transforms': 2, 'contraction': 0.7, 'iterations': 4},
        {'num_transforms': 3, 'contraction': 0.6, 'iterations': 6},
    ]


# ---------------------------------------------------------------------------
# J013 -- Fibonacci Rhythmic Gate
# ---------------------------------------------------------------------------

@numba.njit
def _fibonacci_rhythmic_gate(samples, gate_pattern, base_samples):
    """Gate audio using Fibonacci word pattern."""
    n = len(samples)
    out = np.zeros(n, dtype=np.float32)
    pat_len = len(gate_pattern)

    if pat_len == 0:
        for i in range(n):
            out[i] = samples[i]
        return out

    sample_idx = 0
    pat_idx = 0

    while sample_idx < n:
        gate_on = gate_pattern[pat_idx % pat_len]
        duration = base_samples

        for s in range(duration):
            if sample_idx >= n:
                break
            if gate_on == 1:
                out[sample_idx] = samples[sample_idx]
            else:
                out[sample_idx] = samples[sample_idx] * np.float32(0.02)
            sample_idx += 1

        pat_idx += 1

    return out


def _fibonacci_word(num_generations):
    """Generate Fibonacci word: start with '1', '0', concatenate previous two."""
    if num_generations <= 0:
        return [1]
    if num_generations == 1:
        return [0]

    prev2 = [1]
    prev1 = [0]
    for _ in range(2, num_generations + 1):
        current = prev1 + prev2
        prev2 = prev1
        prev1 = current
    return prev1


def effect_j013_fibonacci_rhythmic_gate(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Gate pattern derived from Fibonacci word creates organic rhythmic gating."""
    base_ms = float(params.get('base_ms', 80))
    num_generations = int(params.get('num_generations', 8))

    num_generations = max(2, min(num_generations, 15))
    base_samples = max(1, int(base_ms * sr / 1000.0))

    pattern = _fibonacci_word(num_generations)
    # Limit pattern length
    if len(pattern) > 5000:
        pattern = pattern[:5000]

    gate_arr = np.array(pattern, dtype=np.int32)

    return _fibonacci_rhythmic_gate(
        samples.astype(np.float32), gate_arr, base_samples
    )


def variants_j013():
    return [
        {'base_ms': 50, 'num_generations': 6},
        {'base_ms': 80, 'num_generations': 8},
        {'base_ms': 120, 'num_generations': 10},
        {'base_ms': 200, 'num_generations': 5},
        {'base_ms': 20, 'num_generations': 12},
        {'base_ms': 30, 'num_generations': 9},
    ]


# ---------------------------------------------------------------------------
# J014 -- Brownian Motion Walk
# ---------------------------------------------------------------------------

@numba.njit
def _brownian_walk_filter(samples, sigma, smoothing, min_coeff, max_coeff, sr):
    """Random walk controls one-pole filter cutoff."""
    n = len(samples)
    out = np.zeros(n, dtype=np.float32)

    lp_state = np.float32(0.0)
    walk_val = np.float32(0.5)  # normalized walk position [0, 1]

    # Simple LCG for deterministic random in njit
    rng = np.int64(12345)

    for i in range(n):
        # Generate pseudo-random step
        rng = (rng * np.int64(1103515245) + np.int64(12345)) & np.int64(0x7FFFFFFF)
        # Map to [-1, 1]
        rand_val = np.float32(rng) / np.float32(0x7FFFFFFF) * np.float32(2.0) - np.float32(1.0)

        # Update walk with smoothing
        step = sigma * rand_val
        walk_val = smoothing * walk_val + (np.float32(1.0) - smoothing) * (walk_val + step)

        # Clamp to [0, 1]
        if walk_val < np.float32(0.0):
            walk_val = np.float32(0.0)
        elif walk_val > np.float32(1.0):
            walk_val = np.float32(1.0)

        # Map walk to filter coefficient
        coeff = min_coeff + walk_val * (max_coeff - min_coeff)

        # One-pole lowpass
        lp_state = coeff * lp_state + (np.float32(1.0) - coeff) * samples[i]
        out[i] = lp_state

    return out


def effect_j014_brownian_motion_walk(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Brownian motion random walk controls filter cutoff for evolving timbre."""
    sigma = np.float32(params.get('sigma', 0.01))
    smoothing = np.float32(params.get('smoothing', 0.995))
    min_freq = float(params.get('min_freq', 400))
    max_freq = float(params.get('max_freq', 6000))

    # Convert frequencies to one-pole coefficients
    # coeff = exp(-2*pi*f/sr)
    two_pi = 6.283185307179586
    min_coeff = np.float32(np.exp(-two_pi * min_freq / sr))
    max_coeff = np.float32(np.exp(-two_pi * max_freq / sr))

    # Note: higher freq = lower coefficient, so swap
    # Actually for one-pole y = a*y + (1-a)*x, higher a = more filtering (lower cutoff)
    # So min_freq should correspond to max_coeff
    return _brownian_walk_filter(
        samples.astype(np.float32), sigma, smoothing, max_coeff, min_coeff, sr
    )


def variants_j014():
    return [
        {'sigma': 0.005, 'smoothing': 0.999, 'min_freq': 300, 'max_freq': 5000},
        {'sigma': 0.01, 'smoothing': 0.995, 'min_freq': 400, 'max_freq': 8000},
        {'sigma': 0.03, 'smoothing': 0.99, 'min_freq': 200, 'max_freq': 10000},
        {'sigma': 0.05, 'smoothing': 0.9, 'min_freq': 500, 'max_freq': 4000},
        {'sigma': 0.001, 'smoothing': 0.999, 'min_freq': 1000, 'max_freq': 3000},
        {'sigma': 0.02, 'smoothing': 0.98, 'min_freq': 600, 'max_freq': 7000},
    ]


# ---------------------------------------------------------------------------
# J015 -- Strange Attractor Spectral Curve
# ---------------------------------------------------------------------------

def effect_j015_strange_attractor_spectral_curve(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Lorenz trajectory mapped to spectral boost curve applied via STFT."""
    trajectory_width_bins = int(params.get('trajectory_width_bins', 5))
    boost_db = float(params.get('boost_db', 10.0))

    samples = samples.astype(np.float32)
    fft_size = 2048
    hop_size = 512

    X = _stft(samples, fft_size, hop_size)
    num_frames, num_bins = X.shape

    # Generate Lorenz trajectory
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    lx, ly, lz = 1.0, 1.0, 1.0
    dt_lorenz = 0.005

    boost_linear = 10.0 ** (boost_db / 20.0)

    for frame in range(num_frames):
        # Integrate Lorenz for several steps per frame
        for _ in range(10):
            dx = sigma * (ly - lx)
            dy = lx * (rho - lz) - ly
            dz = lx * ly - beta * lz
            lx += dx * dt_lorenz
            ly += dy * dt_lorenz
            lz += dz * dt_lorenz

        # Map x-coordinate to bin index (x typically [-20, 20])
        norm_x = (lx + 20.0) / 40.0  # -> [0, 1]
        norm_x = max(0.0, min(1.0, norm_x))
        center_bin = int(norm_x * (num_bins - 1))

        # Apply boost around center_bin
        for b in range(max(0, center_bin - trajectory_width_bins),
                       min(num_bins, center_bin + trajectory_width_bins + 1)):
            dist = abs(b - center_bin)
            # Gaussian-like falloff
            weight = np.exp(-0.5 * (dist / max(1, trajectory_width_bins * 0.5)) ** 2)
            gain = 1.0 + (boost_linear - 1.0) * weight
            X[frame, b] *= gain

    return _istft(X, fft_size, hop_size, length=len(samples))


def variants_j015():
    return [
        {'trajectory_width_bins': 3, 'boost_db': 6},
        {'trajectory_width_bins': 5, 'boost_db': 10},
        {'trajectory_width_bins': 8, 'boost_db': 15},
        {'trajectory_width_bins': 2, 'boost_db': 20},
        {'trajectory_width_bins': 10, 'boost_db': 5},
        {'trajectory_width_bins': 6, 'boost_db': 12},
    ]


# ---------------------------------------------------------------------------
# J016 -- Mobius Transform on Spectrum
# ---------------------------------------------------------------------------

def effect_j016_mobius_transform_spectrum(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Mobius transform w = (a*z + b) / (c*z + d) applied to complex spectrum."""
    a_real = float(params.get('a_real', 1.0))
    b_real = float(params.get('b_real', 0.0))
    c_real = float(params.get('c_real', 0.1))
    d_real = float(params.get('d_real', 1.0))

    samples = samples.astype(np.float32)
    fft_size = 2048
    hop_size = 512

    X = _stft(samples, fft_size, hop_size)
    num_frames, num_bins = X.shape

    a = complex(a_real, 0.0)
    b = complex(b_real, 0.0)
    c = complex(c_real, 0.0)
    d = complex(d_real, 0.0)

    for frame in range(num_frames):
        for bin_idx in range(num_bins):
            z = X[frame, bin_idx]
            denom = c * z + d
            # Avoid division by zero
            if abs(denom) < 1e-10:
                denom = complex(1e-10, 0.0)
            w = (a * z + b) / denom
            X[frame, bin_idx] = w

    return _istft(X, fft_size, hop_size, length=len(samples))


def variants_j016():
    return [
        {'a_real': 1.0, 'b_real': 0.0, 'c_real': 0.1, 'd_real': 1.0},
        {'a_real': 1.5, 'b_real': 0.5, 'c_real': 0.2, 'd_real': 1.5},
        {'a_real': 2.0, 'b_real': -1.0, 'c_real': 0.3, 'd_real': 0.5},
        {'a_real': 0.5, 'b_real': 1.0, 'c_real': 0.0, 'd_real': 2.0},
        {'a_real': 1.0, 'b_real': -0.5, 'c_real': 0.5, 'd_real': 1.0},
        {'a_real': 1.8, 'b_real': 0.3, 'c_real': 0.05, 'd_real': 1.2},
    ]


# ---------------------------------------------------------------------------
# J017 -- Fractal Delay Network
# ---------------------------------------------------------------------------

@numba.njit
def _fractal_delay_network(samples, delays, gains, feedback):
    """Multiple delay lines at self-similar ratios with feedback."""
    n = len(samples)
    num_levels = len(delays)

    max_delay = np.int64(0)
    for d in delays:
        if d > max_delay:
            max_delay = d

    buf_len = max(max_delay + 1, 1)
    buf = np.zeros(buf_len, dtype=np.float32)
    out = np.zeros(n, dtype=np.float32)
    write_pos = 0

    for i in range(n):
        # Sum delayed signals
        fb_sum = np.float32(0.0)
        for lev in range(num_levels):
            read_pos = (write_pos - delays[lev]) % buf_len
            fb_sum += gains[lev] * buf[read_pos]

        y = samples[i] + feedback * fb_sum
        # Soft clip to prevent runaway
        if y > np.float32(2.0):
            y = np.float32(2.0)
        elif y < np.float32(-2.0):
            y = np.float32(-2.0)

        buf[write_pos] = y
        out[i] = y
        write_pos = (write_pos + 1) % buf_len

    return out


def effect_j017_fractal_delay_network(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Delays at self-similar fractal ratios create recursive echo patterns."""
    base_delay_ms = float(params.get('base_delay_ms', 100))
    ratio = float(params.get('ratio', 2.0))
    num_levels = int(params.get('num_levels', 5))
    feedback = np.float32(params.get('feedback', 0.5))

    num_levels = max(2, min(num_levels, 7))

    delays = np.zeros(num_levels, dtype=np.int64)
    gains = np.zeros(num_levels, dtype=np.float32)

    for lev in range(num_levels):
        delay_ms = base_delay_ms * (ratio ** lev)
        delays[lev] = max(1, int(delay_ms * sr / 1000.0))
        # Decreasing gain for longer delays
        gains[lev] = np.float32(1.0 / (lev + 1))

    return _fractal_delay_network(
        samples.astype(np.float32), delays, gains, feedback
    )


def variants_j017():
    return [
        {'base_delay_ms': 50, 'ratio': 2.0, 'num_levels': 4, 'feedback': 0.4},
        {'base_delay_ms': 100, 'ratio': 2.0, 'num_levels': 5, 'feedback': 0.5},
        {'base_delay_ms': 200, 'ratio': 1.5, 'num_levels': 6, 'feedback': 0.6},
        {'base_delay_ms': 500, 'ratio': 3.0, 'num_levels': 3, 'feedback': 0.7},
        {'base_delay_ms': 80, 'ratio': 2.5, 'num_levels': 5, 'feedback': 0.3},
        {'base_delay_ms': 150, 'ratio': 1.618, 'num_levels': 7, 'feedback': 0.8},
    ]


# ---------------------------------------------------------------------------
# J018 -- Audio Boids
# ---------------------------------------------------------------------------

def effect_j018_audio_boids(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Spectral peaks treated as boids with flocking rules; positions shift bins."""
    num_boids = int(params.get('num_boids', 10))
    speed = float(params.get('speed', 2.0))

    samples = samples.astype(np.float32)
    fft_size = 2048
    hop_size = 512

    X = _stft(samples, fft_size, hop_size)
    num_frames, num_bins = X.shape

    # Initialize boid positions randomly across frequency bins
    rng = np.random.RandomState(42)
    positions = rng.uniform(0, num_bins - 1, num_boids).astype(np.float64)
    velocities = rng.uniform(-speed, speed, num_boids).astype(np.float64)

    X_out = X.copy()

    for frame in range(num_frames):
        mag = np.abs(X[frame])

        # Boid flocking rules
        for b in range(num_boids):
            # Cohesion: move toward center of flock
            center = np.mean(positions)
            cohesion_force = (center - positions[b]) * 0.01

            # Separation: avoid nearby boids
            sep_force = 0.0
            for other in range(num_boids):
                if other == b:
                    continue
                diff = positions[b] - positions[other]
                dist = abs(diff)
                if dist < 5.0 and dist > 0.01:
                    sep_force += diff / (dist * dist)
            sep_force *= 0.5

            # Alignment: match average velocity
            avg_vel = np.mean(velocities)
            align_force = (avg_vel - velocities[b]) * 0.05

            # Attraction to spectral energy
            bin_idx = int(positions[b]) % num_bins
            attract_force = 0.0
            # Look for nearby energy peaks
            for offset in range(-10, 11):
                check_bin = (bin_idx + offset) % num_bins
                if mag[check_bin] > mag[bin_idx]:
                    attract_force += float(offset) * float(mag[check_bin]) * 0.001

            velocities[b] += cohesion_force + sep_force + align_force + attract_force
            # Limit speed
            if velocities[b] > speed:
                velocities[b] = speed
            elif velocities[b] < -speed:
                velocities[b] = -speed

            positions[b] += velocities[b]
            # Wrap around
            positions[b] = positions[b] % num_bins

        # Apply boid positions: each boid boosts spectral energy at its position
        boost_mask = np.ones(num_bins, dtype=np.float64)
        for b in range(num_boids):
            center = int(positions[b]) % num_bins
            for offset in range(-3, 4):
                idx = (center + offset) % num_bins
                dist_factor = 1.0 - abs(offset) / 4.0
                boost_mask[idx] += dist_factor * 0.5

        for b_idx in range(num_bins):
            X_out[frame, b_idx] = X[frame, b_idx] * np.float32(boost_mask[b_idx])

    return _istft(X_out, fft_size, hop_size, length=len(samples))


def variants_j018():
    return [
        {'num_boids': 5, 'speed': 1.0},
        {'num_boids': 10, 'speed': 2.0},
        {'num_boids': 20, 'speed': 3.0},
        {'num_boids': 30, 'speed': 1.5},
        {'num_boids': 8, 'speed': 5.0},
        {'num_boids': 15, 'speed': 0.5},
    ]


# ---------------------------------------------------------------------------
# J019 -- Stochastic Resonance
# ---------------------------------------------------------------------------

@numba.njit
def _stochastic_resonance(samples, noise_amplitude, threshold):
    """Add calibrated noise to enhance sub-threshold signal features."""
    n = len(samples)
    out = np.zeros(n, dtype=np.float32)

    rng = np.int64(54321)

    for i in range(n):
        # Generate pseudo-random noise
        rng = (rng * np.int64(1103515245) + np.int64(12345)) & np.int64(0x7FFFFFFF)
        noise = (np.float32(rng) / np.float32(0x7FFFFFFF) * np.float32(2.0) - np.float32(1.0)) * noise_amplitude

        # Add noise to signal
        noisy = samples[i] + noise

        # Threshold detection: signal passes if above threshold, otherwise attenuated
        if abs(noisy) > threshold:
            # Pass through enhanced signal (subtract noise contribution partially)
            out[i] = noisy
        else:
            # Sub-threshold: the noise may have pushed it over
            out[i] = noisy * np.float32(0.3)

    return out


def effect_j019_stochastic_resonance(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Add calibrated noise to enhance sub-threshold features via stochastic resonance."""
    noise_amplitude = np.float32(params.get('noise_amplitude', 0.1))
    threshold = np.float32(params.get('threshold', 0.15))

    return _stochastic_resonance(samples.astype(np.float32), noise_amplitude, threshold)


def variants_j019():
    return [
        {'noise_amplitude': 0.02, 'threshold': 0.05},
        {'noise_amplitude': 0.05, 'threshold': 0.1},
        {'noise_amplitude': 0.1, 'threshold': 0.15},
        {'noise_amplitude': 0.2, 'threshold': 0.2},
        {'noise_amplitude': 0.3, 'threshold': 0.25},
        {'noise_amplitude': 0.5, 'threshold': 0.3},
    ]


# ---------------------------------------------------------------------------
# J020 -- Chua's Circuit
# ---------------------------------------------------------------------------

@numba.njit
def _chua_circuit(samples, alpha, beta, drive_amount, sr):
    """Chua's circuit: piecewise-linear chaotic oscillator mixed with audio."""
    n = len(samples)
    out = np.zeros(n, dtype=np.float32)

    # Chua's circuit state
    x = np.float64(0.1)
    y = np.float64(0.0)
    z = np.float64(0.0)

    # Chua diode parameters (piecewise-linear)
    m0 = np.float64(-1.143)
    m1 = np.float64(-0.714)
    bp = np.float64(1.0)  # breakpoint

    dt = 0.01

    for i in range(n):
        # Chua diode function h(x) = m1*x + 0.5*(m0-m1)*(|x+bp| - |x-bp|)
        h = m1 * x + 0.5 * (m0 - m1) * (abs(x + bp) - abs(x - bp))

        # Chua's equations
        dx = alpha * (y - x - h) + np.float64(samples[i]) * drive_amount * 0.5
        dy = x - y + z
        dz = -beta * y

        x += dx * dt
        y += dy * dt
        z += dz * dt

        # Clamp
        if x > 5.0:
            x = 5.0
        elif x < -5.0:
            x = -5.0
        if y > 5.0:
            y = 5.0
        elif y < -5.0:
            y = -5.0
        if z > 50.0:
            z = 50.0
        elif z < -50.0:
            z = -50.0

        # Output: mix of dry signal and Chua x-state
        chua_norm = np.float32(np.tanh(x * 0.3))
        out[i] = samples[i] * (np.float32(1.0) - drive_amount) + chua_norm * drive_amount

    return out


def effect_j020_chua_circuit(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Chua's circuit piecewise-linear chaotic oscillator driven by audio input."""
    alpha = np.float64(params.get('alpha', 10.0))
    beta = np.float64(params.get('beta', 14.87))
    drive_amount = np.float32(params.get('drive_amount', 0.5))

    return _chua_circuit(samples.astype(np.float32), alpha, beta, drive_amount, sr)


def variants_j020():
    return [
        {'alpha': 9.0, 'beta': 14.0, 'drive_amount': 0.3},
        {'alpha': 10.0, 'beta': 14.87, 'drive_amount': 0.5},
        {'alpha': 12.0, 'beta': 14.5, 'drive_amount': 0.6},
        {'alpha': 15.0, 'beta': 15.0, 'drive_amount': 0.4},
        {'alpha': 16.0, 'beta': 14.2, 'drive_amount': 0.8},
        {'alpha': 11.0, 'beta': 14.87, 'drive_amount': 0.2},
    ]
