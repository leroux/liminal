"""R-series: Miscellaneous / experimental effects (R001-R016)."""
import numpy as np
import numba


# ---------------------------------------------------------------------------
# STFT / ISTFT helpers
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
# R001 -- Audio Fractalization
# ---------------------------------------------------------------------------

def effect_r001_audio_fractalization(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Replace each sample with a scaled copy of the signal, creating
    self-similar fractal-like structure at multiple timescales."""
    num_scales = int(params.get('num_scales', 3))
    scale_ratio = np.float32(params.get('scale_ratio', 0.5))
    amplitude_per_scale = np.float32(params.get('amplitude_per_scale', 0.5))

    num_scales = max(2, min(5, num_scales))
    scale_ratio = np.clip(scale_ratio, 0.3, 0.7)
    amplitude_per_scale = np.clip(amplitude_per_scale, 0.3, 0.8)

    samples = samples.astype(np.float32)
    n = len(samples)
    out = samples.copy()

    for s in range(1, num_scales):
        # Each scale compresses the entire signal into a shorter version
        compressed_len = max(1, int(n * (scale_ratio ** s)))
        # Resample by nearest-neighbor
        indices = np.linspace(0, n - 1, compressed_len).astype(np.int64)
        compressed = samples[indices]

        # Tile the compressed signal to fill the original length
        gain = np.float32(amplitude_per_scale ** s)
        tiles_needed = (n + compressed_len - 1) // compressed_len
        tiled = np.tile(compressed, tiles_needed)[:n]
        out += gain * tiled

    # Normalize to prevent clipping
    peak = np.max(np.abs(out))
    if peak > 0:
        in_peak = np.max(np.abs(samples))
        out *= (in_peak / peak) if peak > in_peak else 1.0

    return out


def variants_r001():
    return [
        {'num_scales': 2, 'scale_ratio': 0.5, 'amplitude_per_scale': 0.6},
        {'num_scales': 3, 'scale_ratio': 0.5, 'amplitude_per_scale': 0.5},
        {'num_scales': 4, 'scale_ratio': 0.4, 'amplitude_per_scale': 0.4},
        {'num_scales': 5, 'scale_ratio': 0.3, 'amplitude_per_scale': 0.5},
        {'num_scales': 3, 'scale_ratio': 0.7, 'amplitude_per_scale': 0.7},
    ]


# ---------------------------------------------------------------------------
# R002 -- Spectral Peak Tracking + Resynthesis
# ---------------------------------------------------------------------------

def effect_r002_spectral_peak_resynthesis(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Find top N magnitude peaks per STFT frame and resynthesize with sine
    oscillators, optionally adding vibrato."""
    num_peaks = int(params.get('num_peaks', 20))
    vibrato_depth = np.float32(params.get('vibrato_depth', 0.0))
    vibrato_rate = np.float32(params.get('vibrato_rate', 0.0))

    num_peaks = max(5, min(50, num_peaks))
    fft_size = 2048
    hop_size = 512

    X = _stft(samples.astype(np.float32), fft_size, hop_size)
    num_frames = X.shape[0]
    num_bins = X.shape[1]
    bin_freq = np.float32(sr) / np.float32(fft_size)

    n = len(samples)
    out = np.zeros(n, dtype=np.float32)

    for frame_idx in range(num_frames):
        mag = np.abs(X[frame_idx])
        # Find top N peaks
        k = min(num_peaks, num_bins)
        peak_bins = np.argsort(mag)[-k:]

        frame_start = frame_idx * hop_size
        frame_end = min(frame_start + hop_size, n)
        frame_len = frame_end - frame_start

        for b in peak_bins:
            freq = np.float32(b) * bin_freq
            amp = mag[b] / np.float32(fft_size)
            if freq < 20.0 or amp < 1e-8:
                continue
            phase_offset = np.angle(X[frame_idx, b])

            for j in range(frame_len):
                t = np.float32(frame_start + j) / np.float32(sr)
                vib = vibrato_depth * np.sin(2.0 * np.pi * vibrato_rate * t)
                out[frame_start + j] += amp * np.sin(
                    2.0 * np.pi * freq * (1.0 + vib) * t + phase_offset
                )

    # Normalize
    peak_val = np.max(np.abs(out))
    if peak_val > 0:
        in_rms = np.sqrt(np.mean(samples ** 2)) + 1e-10
        out_rms = np.sqrt(np.mean(out ** 2)) + 1e-10
        out *= np.float32(in_rms / out_rms)

    return out.astype(np.float32)


def variants_r002():
    return [
        {'num_peaks': 5, 'vibrato_depth': 0, 'vibrato_rate': 0},
        {'num_peaks': 10, 'vibrato_depth': 0, 'vibrato_rate': 0},
        {'num_peaks': 20, 'vibrato_depth': 0, 'vibrato_rate': 0},
        {'num_peaks': 50, 'vibrato_depth': 0, 'vibrato_rate': 0},
        {'num_peaks': 15, 'vibrato_depth': 1.0, 'vibrato_rate': 3.0},
        {'num_peaks': 20, 'vibrato_depth': 2.0, 'vibrato_rate': 5.0},
    ]


# ---------------------------------------------------------------------------
# R003 -- Autoregressive Model Resynthesis (LPC)
# ---------------------------------------------------------------------------

def _levinson_durbin(r, order):
    """Levinson-Durbin recursion for LPC coefficients from autocorrelation."""
    a = np.zeros(order + 1, dtype=np.float64)
    e = r[0]
    a[0] = 1.0

    for i in range(1, order + 1):
        # Compute reflection coefficient
        acc = 0.0
        for j in range(1, i):
            acc += a[j] * r[i - j]
        k = -(r[i] + acc) / (e + 1e-30)

        # Update coefficients
        a_new = a.copy()
        a_new[i] = k
        for j in range(1, i):
            a_new[j] = a[j] + k * a[i - j]
        a = a_new

        e *= (1.0 - k * k)
        if e <= 0:
            e = 1e-10

    return a[1:], e  # return coefficients (without a[0]=1) and prediction error


def effect_r003_lpc_resynthesis(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """LPC analysis, modify coefficients, resynthesize via all-pole filter
    driven by noise excitation."""
    lpc_order = int(params.get('lpc_order', 20))
    modification = params.get('modification', 'scale')
    mod_amount = np.float64(params.get('mod_amount', 1.2))

    lpc_order = max(10, min(50, lpc_order))

    x = samples.astype(np.float64)
    n = len(x)

    # Frame-based LPC analysis
    frame_size = 1024
    hop = 512
    num_frames = max(1, (n - frame_size) // hop + 1)

    out = np.zeros(n, dtype=np.float64)
    window = np.hanning(frame_size)

    rng = np.random.RandomState(42)

    for f in range(num_frames):
        start = f * hop
        end = min(start + frame_size, n)
        frame_len = end - start
        frame = np.zeros(frame_size, dtype=np.float64)
        frame[:frame_len] = x[start:end] * window[:frame_len]

        # Autocorrelation
        r = np.correlate(frame, frame, mode='full')
        r = r[frame_size - 1:]  # positive lags only
        r = r[:lpc_order + 1]

        if r[0] < 1e-10:
            # Silent frame -- pass through
            out[start:end] += frame[:frame_len]
            continue

        # Levinson-Durbin
        coeffs, error = _levinson_durbin(r, lpc_order)

        # Modify coefficients
        if modification == 'scale':
            coeffs = coeffs * mod_amount
        elif modification == 'jitter':
            jitter = rng.randn(len(coeffs)) * (mod_amount - 1.0) * 0.1
            coeffs = coeffs + jitter

        # Stability check: scale coefficients down if filter is unstable
        # Check via polynomial root magnitudes -- if too expensive, just
        # scale down until energy stays bounded
        poly = np.concatenate([[1.0], coeffs])
        roots = np.roots(poly)
        max_root = np.max(np.abs(roots)) if len(roots) > 0 else 0.0
        if max_root >= 1.0:
            # Scale coefficients to bring roots inside unit circle
            scale_factor = 0.98 / max_root
            coeffs = coeffs * (scale_factor ** np.arange(1, len(coeffs) + 1))

        # Excitation: white noise scaled by prediction error
        excitation = rng.randn(frame_size).astype(np.float64) * np.sqrt(max(error, 1e-10))

        # All-pole synthesis: y[n] = excitation[n] - sum(a[k]*y[n-k])
        synth = np.zeros(frame_size, dtype=np.float64)
        for i in range(frame_size):
            val = excitation[i]
            for k in range(min(i, lpc_order)):
                val -= coeffs[k] * synth[i - 1 - k]
            # Clip to prevent runaway
            if val > 10.0:
                val = 10.0
            elif val < -10.0:
                val = -10.0
            synth[i] = val

        synth *= window
        out_end = min(start + frame_size, n)
        out[start:out_end] += synth[:out_end - start]

    # Replace any NaN/inf with zeros
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize
    in_rms = np.sqrt(np.mean(x ** 2)) + 1e-10
    out_rms = np.sqrt(np.mean(out ** 2)) + 1e-10
    out *= (in_rms / out_rms)

    return out.astype(np.float32)


def variants_r003():
    return [
        {'lpc_order': 10, 'modification': 'scale', 'mod_amount': 1.0},
        {'lpc_order': 20, 'modification': 'scale', 'mod_amount': 1.2},
        {'lpc_order': 30, 'modification': 'scale', 'mod_amount': 1.5},
        {'lpc_order': 50, 'modification': 'scale', 'mod_amount': 2.0},
        {'lpc_order': 20, 'modification': 'jitter', 'mod_amount': 1.2},
        {'lpc_order': 30, 'modification': 'jitter', 'mod_amount': 1.5},
    ]


# ---------------------------------------------------------------------------
# R004 -- Spectral Painting
# ---------------------------------------------------------------------------

def effect_r004_spectral_painting(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Multiply STFT magnitudes by a mathematical shape that evolves over time."""
    shape_type = params.get('shape_type', 'gaussian_peaks')
    evolution_rate = np.float32(params.get('evolution_rate', 0.3))

    fft_size = 2048
    hop_size = 512

    X = _stft(samples.astype(np.float32), fft_size, hop_size)
    mag = np.abs(X)
    phase = np.angle(X)
    num_frames, num_bins = X.shape

    bins = np.arange(num_bins, dtype=np.float32) / num_bins

    for i in range(num_frames):
        t = np.float32(i) / max(1, num_frames - 1)  # 0..1 over time
        phase_offset = t * evolution_rate * 2.0 * np.pi

        if shape_type == 'sine_wave':
            # Sinusoidal mask across frequency
            mask = 0.5 + 0.5 * np.sin(2.0 * np.pi * 3.0 * bins + phase_offset)
        elif shape_type == 'gaussian_peaks':
            # Multiple gaussian peaks that drift over time
            mask = np.zeros(num_bins, dtype=np.float32)
            num_gaussians = 5
            for g in range(num_gaussians):
                center = (np.float32(g) / num_gaussians + t * evolution_rate * 0.1) % 1.0
                sigma = 0.05
                mask += np.exp(-0.5 * ((bins - center) / sigma) ** 2)
            mask = mask / (np.max(mask) + 1e-10)
        elif shape_type == 'sawtooth':
            # Sawtooth ramp that shifts over time
            mask = ((bins * 5.0 + phase_offset / (2.0 * np.pi)) % 1.0).astype(np.float32)
        elif shape_type == 'random_curve':
            # Smooth random curve via low-frequency sinusoids
            rng = np.random.RandomState(int(i * 7 + 42))
            mask = np.zeros(num_bins, dtype=np.float32)
            for h in range(8):
                freq = rng.uniform(1.0, 10.0)
                amp = rng.uniform(0.1, 0.3)
                ph = rng.uniform(0, 2.0 * np.pi)
                mask += amp * np.sin(2.0 * np.pi * freq * bins + ph + phase_offset)
            mask = 0.5 + 0.5 * mask / (np.max(np.abs(mask)) + 1e-10)
        else:
            mask = np.ones(num_bins, dtype=np.float32)

        mag[i] *= mask.astype(np.float32)

    Y = mag * np.exp(1j * phase)
    return _istft(Y, fft_size, hop_size, length=len(samples))


def variants_r004():
    return [
        {'shape_type': 'sine_wave', 'evolution_rate': 0.0},
        {'shape_type': 'sine_wave', 'evolution_rate': 0.5},
        {'shape_type': 'gaussian_peaks', 'evolution_rate': 0.3},
        {'shape_type': 'gaussian_peaks', 'evolution_rate': 1.0},
        {'shape_type': 'sawtooth', 'evolution_rate': 0.2},
        {'shape_type': 'random_curve', 'evolution_rate': 0.5},
    ]


# ---------------------------------------------------------------------------
# R005 -- Phase Gradient Manipulation
# ---------------------------------------------------------------------------

def effect_r005_phase_gradient(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Modify the phase gradient across frequency bins, altering the
    time-domain alignment of spectral components."""
    gradient_scale = np.float32(params.get('gradient_scale', 2.0))

    fft_size = 2048
    hop_size = 512

    X = _stft(samples.astype(np.float32), fft_size, hop_size)
    mag = np.abs(X)
    phase = np.angle(X)
    num_frames, num_bins = X.shape

    for i in range(num_frames):
        # Compute phase differences between adjacent bins (gradient)
        phase_diff = np.diff(phase[i])
        # Scale the gradient
        scaled_diff = phase_diff * gradient_scale
        # Reconstruct phase from scaled gradient
        new_phase = np.zeros(num_bins, dtype=np.float32)
        new_phase[0] = phase[i, 0]
        for b in range(1, num_bins):
            new_phase[b] = new_phase[b - 1] + scaled_diff[b - 1]
        phase[i] = new_phase

    Y = mag * np.exp(1j * phase)
    return _istft(Y, fft_size, hop_size, length=len(samples))


def variants_r005():
    return [
        {'gradient_scale': 0.1},
        {'gradient_scale': 0.5},
        {'gradient_scale': 1.0},
        {'gradient_scale': 2.0},
        {'gradient_scale': 3.0},
        {'gradient_scale': 5.0},
    ]


# ---------------------------------------------------------------------------
# R006 -- Spectral Entropy Gate
# ---------------------------------------------------------------------------

def effect_r006_spectral_entropy_gate(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Gate STFT frames based on spectral entropy. Low entropy = tonal,
    high entropy = noisy."""
    entropy_threshold = np.float32(params.get('entropy_threshold', 0.7))
    mode = params.get('mode', 'keep_tonal')

    entropy_threshold = np.clip(entropy_threshold, 0.3, 0.9)
    fft_size = 2048
    hop_size = 512

    X = _stft(samples.astype(np.float32), fft_size, hop_size)
    mag = np.abs(X)
    num_frames, num_bins = X.shape

    # Maximum possible entropy for normalization
    max_entropy = np.log2(num_bins)

    for i in range(num_frames):
        # Compute spectral probability distribution
        total = np.sum(mag[i]) + 1e-10
        p = mag[i] / total
        # Shannon entropy
        entropy = 0.0
        for b in range(num_bins):
            if p[b] > 1e-10:
                entropy -= p[b] * np.log2(p[b])
        # Normalize to [0, 1]
        normalized_entropy = entropy / max_entropy

        if mode == 'keep_tonal':
            # Keep frames with low entropy (tonal)
            if normalized_entropy > entropy_threshold:
                X[i] *= 0.0
        else:  # keep_noisy
            # Keep frames with high entropy (noisy)
            if normalized_entropy < entropy_threshold:
                X[i] *= 0.0

    return _istft(X, fft_size, hop_size, length=len(samples))


def variants_r006():
    return [
        {'entropy_threshold': 0.5, 'mode': 'keep_tonal'},
        {'entropy_threshold': 0.7, 'mode': 'keep_tonal'},
        {'entropy_threshold': 0.9, 'mode': 'keep_tonal'},
        {'entropy_threshold': 0.5, 'mode': 'keep_noisy'},
        {'entropy_threshold': 0.7, 'mode': 'keep_noisy'},
        {'entropy_threshold': 0.9, 'mode': 'keep_noisy'},
    ]


# ---------------------------------------------------------------------------
# R007 -- Wavelet Decomposition (Haar)
# ---------------------------------------------------------------------------

def _haar_decompose(signal, num_levels):
    """Haar wavelet decomposition: returns list of (approx, detail) at each level."""
    coeffs = []
    current = signal.copy()
    for _ in range(num_levels):
        n = len(current)
        if n < 2:
            break
        half = n // 2
        approx = np.zeros(half, dtype=np.float32)
        detail = np.zeros(half, dtype=np.float32)
        for i in range(half):
            approx[i] = (current[2 * i] + current[2 * i + 1]) * 0.5
            detail[i] = (current[2 * i] - current[2 * i + 1]) * 0.5
        coeffs.append((approx, detail))
        current = approx
    return coeffs


def _haar_reconstruct(coeffs, original_length):
    """Reconstruct signal from Haar wavelet coefficients."""
    # Start from the deepest level approximation
    current = coeffs[-1][0].copy()

    for level in range(len(coeffs) - 1, -1, -1):
        approx_level, detail_level = coeffs[level]
        n = len(approx_level)
        reconstructed = np.zeros(2 * n, dtype=np.float32)
        for i in range(n):
            reconstructed[2 * i] = current[i] + detail_level[i]
            reconstructed[2 * i + 1] = current[i] - detail_level[i]
        current = reconstructed

    # Trim or pad to original length
    if len(current) < original_length:
        padded = np.zeros(original_length, dtype=np.float32)
        padded[:len(current)] = current
        current = padded
    else:
        current = current[:original_length]

    return current


def effect_r007_wavelet_decomposition(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Haar wavelet decomposition with detail coefficient modification."""
    num_levels = int(params.get('num_levels', 5))
    modification = params.get('modification', 'amplify')
    mod_amount = np.float32(params.get('mod_amount', 2.0))

    num_levels = max(3, min(8, num_levels))

    x = samples.astype(np.float32)
    n = len(x)

    # Pad to nearest power of 2 for clean decomposition
    pow2 = 1
    while pow2 < n:
        pow2 *= 2
    padded = np.zeros(pow2, dtype=np.float32)
    padded[:n] = x

    coeffs = _haar_decompose(padded, num_levels)

    # Modify detail coefficients
    for level in range(len(coeffs)):
        approx, detail = coeffs[level]
        if modification == 'zero':
            # Remove detail (smooth)
            coeffs[level] = (approx, np.zeros_like(detail))
        elif modification == 'amplify':
            # Boost detail (enhance texture)
            coeffs[level] = (approx, detail * mod_amount)
        elif modification == 'threshold':
            # Hard threshold: zero out small details
            thresh = np.max(np.abs(detail)) * (1.0 / mod_amount)
            thresholded = detail.copy()
            thresholded[np.abs(thresholded) < thresh] = 0.0
            coeffs[level] = (approx, thresholded)

    out = _haar_reconstruct(coeffs, pow2)[:n]

    # Normalize to match input level
    in_rms = np.sqrt(np.mean(x ** 2)) + 1e-10
    out_rms = np.sqrt(np.mean(out ** 2)) + 1e-10
    out *= np.float32(in_rms / out_rms)

    return out


def variants_r007():
    return [
        {'num_levels': 3, 'modification': 'amplify', 'mod_amount': 1.5},
        {'num_levels': 5, 'modification': 'amplify', 'mod_amount': 2.0},
        {'num_levels': 8, 'modification': 'amplify', 'mod_amount': 3.0},
        {'num_levels': 5, 'modification': 'zero', 'mod_amount': 1.0},
        {'num_levels': 5, 'modification': 'threshold', 'mod_amount': 1.5},
        {'num_levels': 5, 'modification': 'threshold', 'mod_amount': 3.0},
    ]


# ---------------------------------------------------------------------------
# R008 -- Hilbert Envelope + Fine Structure Swap
# ---------------------------------------------------------------------------

def effect_r008_hilbert_envelope_swap(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Separate envelope and fine structure via Hilbert transform, then replace
    the fine structure with a synthetic source."""
    fine_structure_source = params.get('fine_structure_source', 'noise')

    x = samples.astype(np.float64)
    n = len(x)

    # Hilbert transform via FFT
    X = np.fft.fft(x)
    h = np.zeros(n, dtype=np.float64)
    if n > 0:
        h[0] = 1
        if n % 2 == 0:
            h[n // 2] = 1
            h[1:n // 2] = 2
        else:
            h[1:(n + 1) // 2] = 2
    analytic = np.fft.ifft(X * h)
    envelope = np.abs(analytic).astype(np.float32)

    # Generate replacement fine structure
    if fine_structure_source == 'noise':
        rng = np.random.RandomState(42)
        fine = rng.randn(n).astype(np.float32)
        # Normalize noise to unit amplitude
        fine = fine / (np.max(np.abs(fine)) + 1e-10)
    elif fine_structure_source == 'sine':
        # Use a sine wave at a default frequency
        t = np.arange(n, dtype=np.float32) / sr
        fine = np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)
    else:  # 'original'
        # Use original fine structure
        fine = np.real(analytic / (envelope + 1e-10)).astype(np.float32)

    out = envelope * fine
    return out.astype(np.float32)


def variants_r008():
    return [
        {'fine_structure_source': 'original'},
        {'fine_structure_source': 'noise'},
        {'fine_structure_source': 'sine'},
    ]


# ---------------------------------------------------------------------------
# R009 -- Spectral Freeze with Drift
# ---------------------------------------------------------------------------

def effect_r009_spectral_freeze_drift(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Freeze the spectrum at a specific point, then slowly drift toward the
    current frame's spectrum over time."""
    freeze_point = np.float32(params.get('freeze_point', 0.3))
    drift_rate = np.float32(params.get('drift_rate', 0.01))

    freeze_point = np.clip(freeze_point, 0.1, 0.9)
    drift_rate = np.clip(drift_rate, 0.001, 0.1)

    fft_size = 2048
    hop_size = 512

    X = _stft(samples.astype(np.float32), fft_size, hop_size)
    num_frames = X.shape[0]
    if num_frames == 0:
        return samples.copy()

    freeze_frame = int(freeze_point * (num_frames - 1))
    frozen_mag = np.abs(X[freeze_frame]).copy()
    current_mag = frozen_mag.copy()

    mag = np.abs(X)
    phase = np.angle(X)

    Y = np.zeros_like(X)

    for i in range(num_frames):
        if i <= freeze_frame:
            # Before freeze point: pass through
            Y[i] = X[i]
        else:
            # After freeze: drift from frozen toward current
            target_mag = mag[i]
            current_mag = current_mag + drift_rate * (target_mag - current_mag)
            Y[i] = current_mag * np.exp(1j * phase[i])

    return _istft(Y, fft_size, hop_size, length=len(samples))


def variants_r009():
    return [
        {'freeze_point': 0.1, 'drift_rate': 0.01},
        {'freeze_point': 0.3, 'drift_rate': 0.01},
        {'freeze_point': 0.5, 'drift_rate': 0.005},
        {'freeze_point': 0.3, 'drift_rate': 0.05},
        {'freeze_point': 0.3, 'drift_rate': 0.1},
        {'freeze_point': 0.7, 'drift_rate': 0.001},
    ]


# ---------------------------------------------------------------------------
# R010 -- Sample-Level Markov Chain
# ---------------------------------------------------------------------------

@numba.njit
def _build_markov_table(samples, num_levels, order):
    """Build transition probability table from quantized audio samples."""
    n = len(samples)

    # Quantize samples to discrete levels
    quantized = np.empty(n, dtype=np.int64)
    for i in range(n):
        val = (samples[i] + 1.0) * 0.5  # map [-1,1] to [0,1]
        if val < 0.0:
            val = 0.0
        elif val > 1.0:
            val = 1.0
        quantized[i] = min(int(val * num_levels), num_levels - 1)

    # For order-1 Markov chain: transition[current_state, next_state]
    # We only implement order 1 in numba for simplicity
    transitions = np.zeros((num_levels, num_levels), dtype=np.float32)
    for i in range(n - 1):
        transitions[quantized[i], quantized[i + 1]] += 1.0

    # Normalize rows to probabilities
    for s in range(num_levels):
        row_sum = np.float32(0.0)
        for t in range(num_levels):
            row_sum += transitions[s, t]
        if row_sum > 0:
            for t in range(num_levels):
                transitions[s, t] /= row_sum
        else:
            # Uniform distribution if no data
            for t in range(num_levels):
                transitions[s, t] = np.float32(1.0) / np.float32(num_levels)

    return transitions, quantized


@numba.njit
def _generate_markov(transitions, num_levels, length, seed_state):
    """Generate audio from Markov transition table."""
    out = np.empty(length, dtype=np.float32)
    state = seed_state

    # Simple LCG random number generator for numba
    rng = np.int64(42)

    for i in range(length):
        # Convert state back to float
        out[i] = np.float32(state) / np.float32(num_levels) * 2.0 - 1.0

        # Choose next state based on transition probabilities
        rng = (rng * np.int64(1103515245) + np.int64(12345)) & np.int64(0x7FFFFFFF)
        rand_val = np.float32(rng & np.int64(0xFFFF)) / np.float32(0xFFFF)

        cumsum = np.float32(0.0)
        next_state = state
        for s in range(num_levels):
            cumsum += transitions[state, s]
            if rand_val <= cumsum:
                next_state = s
                break
        state = next_state

    return out


def effect_r010_markov_chain(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Build a Markov transition matrix from the audio and generate new audio
    by walking the chain."""
    num_levels = int(params.get('num_levels', 64))
    order = int(params.get('order', 1))

    num_levels = max(16, min(256, num_levels))
    order = max(1, min(3, order))

    x = samples.astype(np.float32)
    n = len(x)

    transitions, quantized = _build_markov_table(x, num_levels, order)

    # Seed with the middle sample's quantized value
    seed_state = quantized[n // 2]

    out = _generate_markov(transitions, num_levels, n, seed_state)

    # Light smoothing to reduce harsh quantization
    # Simple 3-sample moving average
    smoothed = np.empty(n, dtype=np.float32)
    smoothed[0] = out[0]
    smoothed[n - 1] = out[n - 1]
    for i in range(1, n - 1):
        smoothed[i] = (out[i - 1] + out[i] + out[i + 1]) / 3.0

    return smoothed


def variants_r010():
    return [
        {'num_levels': 16, 'order': 1},
        {'num_levels': 32, 'order': 1},
        {'num_levels': 64, 'order': 1},
        {'num_levels': 128, 'order': 1},
        {'num_levels': 256, 'order': 1},
        {'num_levels': 64, 'order': 2},
    ]


# ---------------------------------------------------------------------------
# R011 -- Frequency Domain Convolution
# ---------------------------------------------------------------------------

def effect_r011_freq_domain_convolution(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Pointwise multiply FFT magnitudes with a synthetic spectrum."""
    synthetic_type = params.get('synthetic_type', 'sawtooth')

    fft_size = 2048
    hop_size = 512

    X = _stft(samples.astype(np.float32), fft_size, hop_size)
    mag = np.abs(X)
    phase = np.angle(X)
    num_frames, num_bins = X.shape

    # Generate synthetic magnitude spectrum
    freqs = np.arange(num_bins, dtype=np.float32) * (np.float32(sr) / fft_size)
    synth_mag = np.zeros(num_bins, dtype=np.float32)

    if synthetic_type == 'sawtooth':
        # Sawtooth harmonic series: 1/k amplitude for harmonics of 100 Hz
        fundamental = 100.0
        for k in range(1, 50):
            harmonic_freq = fundamental * k
            bin_idx = int(harmonic_freq / (np.float32(sr) / fft_size))
            if 0 <= bin_idx < num_bins:
                synth_mag[bin_idx] = 1.0 / k
                # Spread to neighboring bins
                if bin_idx > 0:
                    synth_mag[bin_idx - 1] += 0.3 / k
                if bin_idx < num_bins - 1:
                    synth_mag[bin_idx + 1] += 0.3 / k
    elif synthetic_type == 'harmonic_series':
        # Pure harmonic series with equal amplitude
        fundamental = 200.0
        for k in range(1, 30):
            harmonic_freq = fundamental * k
            bin_idx = int(harmonic_freq / (np.float32(sr) / fft_size))
            if 0 <= bin_idx < num_bins:
                synth_mag[bin_idx] = 1.0
                if bin_idx > 0:
                    synth_mag[bin_idx - 1] += 0.5
                if bin_idx < num_bins - 1:
                    synth_mag[bin_idx + 1] += 0.5
    elif synthetic_type == 'noise_shaped':
        # Shaped noise: pink-ish spectrum (1/f)
        for b in range(num_bins):
            f = max(1.0, freqs[b])
            synth_mag[b] = 1.0 / np.sqrt(f)
    else:
        synth_mag[:] = 1.0

    # Normalize synthetic spectrum
    synth_max = np.max(synth_mag) + 1e-10
    synth_mag /= synth_max

    # Apply: pointwise multiply
    for i in range(num_frames):
        mag[i] *= synth_mag

    Y = mag * np.exp(1j * phase)
    return _istft(Y, fft_size, hop_size, length=len(samples))


def variants_r011():
    return [
        {'synthetic_type': 'sawtooth'},
        {'synthetic_type': 'harmonic_series'},
        {'synthetic_type': 'noise_shaped'},
    ]


# ---------------------------------------------------------------------------
# R012 -- Audio Quine
# ---------------------------------------------------------------------------

def effect_r012_audio_quine(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Convolve the input signal with a short chunk of itself."""
    chunk_start_ms = np.float32(params.get('chunk_start_ms', 100))
    chunk_length_ms = np.float32(params.get('chunk_length_ms', 100))

    x = samples.astype(np.float32)
    n = len(x)

    chunk_start = int(np.clip(chunk_start_ms, 0, 500) * sr / 1000.0)
    chunk_length = int(np.clip(chunk_length_ms, 50, 200) * sr / 1000.0)

    # Clamp to valid range
    chunk_start = min(chunk_start, max(0, n - chunk_length))
    chunk_end = min(chunk_start + chunk_length, n)
    chunk = x[chunk_start:chunk_end].copy()

    if len(chunk) == 0:
        return x.copy()

    # Normalize chunk to prevent explosion
    chunk_peak = np.max(np.abs(chunk)) + 1e-10
    chunk = chunk / chunk_peak

    # FFT convolution
    conv_len = n + len(chunk) - 1
    # Pad to next power of 2 for efficiency
    fft_len = 1
    while fft_len < conv_len:
        fft_len *= 2

    X = np.fft.rfft(x, fft_len)
    H = np.fft.rfft(chunk, fft_len)
    out = np.fft.irfft(X * H, fft_len)[:n].astype(np.float32)

    # Normalize output to match input level
    in_rms = np.sqrt(np.mean(x ** 2)) + 1e-10
    out_rms = np.sqrt(np.mean(out ** 2)) + 1e-10
    out *= np.float32(in_rms / out_rms)

    return out


def variants_r012():
    return [
        {'chunk_start_ms': 0, 'chunk_length_ms': 50},
        {'chunk_start_ms': 100, 'chunk_length_ms': 100},
        {'chunk_start_ms': 200, 'chunk_length_ms': 150},
        {'chunk_start_ms': 0, 'chunk_length_ms': 200},
        {'chunk_start_ms': 500, 'chunk_length_ms': 100},
    ]


# ---------------------------------------------------------------------------
# R013 -- Spectral Phase Vocoder Modified Hop
# ---------------------------------------------------------------------------

def effect_r013_modified_hop_vocoder(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Phase vocoder with non-standard analysis/synthesis hop ratio for
    time-stretching or compression without pitch change."""
    analysis_hop = int(params.get('analysis_hop', 512))
    synthesis_hop = int(params.get('synthesis_hop', 1024))

    analysis_hop = max(256, min(1024, analysis_hop))
    synthesis_hop = max(128, min(2048, synthesis_hop))

    fft_size = 2048
    x = samples.astype(np.float32)

    # Analysis STFT with analysis_hop
    X = _stft(x, fft_size, analysis_hop)
    num_frames, num_bins = X.shape

    if num_frames < 2:
        return x.copy()

    # Phase vocoder: adjust phases for synthesis_hop
    mag = np.abs(X)
    phase = np.angle(X)

    # Expected phase advance per analysis hop
    omega = 2.0 * np.pi * np.arange(num_bins, dtype=np.float64) * analysis_hop / fft_size
    hop_ratio = np.float64(synthesis_hop) / np.float64(analysis_hop)

    synth_phase = np.zeros((num_frames, num_bins), dtype=np.float64)
    synth_phase[0] = phase[0]

    for i in range(1, num_frames):
        # Phase difference
        dphi = phase[i] - phase[i - 1] - omega
        # Wrap to [-pi, pi]
        dphi = dphi - 2.0 * np.pi * np.round(dphi / (2.0 * np.pi))
        # Instantaneous frequency deviation
        freq_dev = dphi / analysis_hop
        # Accumulate synthesis phase
        synth_phase[i] = synth_phase[i - 1] + (omega + freq_dev * analysis_hop) * hop_ratio

    # Synthesize with synthesis_hop
    Y = mag * np.exp(1j * synth_phase)

    # Output length adjusted by hop ratio
    out_length = int(len(x) * hop_ratio)
    out = _istft(Y, fft_size, synthesis_hop, length=out_length)

    # Resample back to original length via linear interpolation
    if len(out) != len(x):
        indices = np.linspace(0, len(out) - 1, len(x))
        int_indices = indices.astype(np.int64)
        frac = (indices - int_indices).astype(np.float32)
        int_indices = np.clip(int_indices, 0, len(out) - 2)
        result = (1.0 - frac) * out[int_indices] + frac * out[int_indices + 1]
        return result.astype(np.float32)

    return out.astype(np.float32)


def variants_r013():
    return [
        {'analysis_hop': 512, 'synthesis_hop': 256},     # time compress 2x (higher pitch character)
        {'analysis_hop': 512, 'synthesis_hop': 512},      # identity (control)
        {'analysis_hop': 512, 'synthesis_hop': 1024},     # time stretch 2x
        {'analysis_hop': 512, 'synthesis_hop': 2048},     # time stretch 4x
        {'analysis_hop': 256, 'synthesis_hop': 1024},     # extreme stretch
        {'analysis_hop': 1024, 'synthesis_hop': 256},     # extreme compress
    ]


# ---------------------------------------------------------------------------
# R014 -- Karplus-Strong Cloud
# ---------------------------------------------------------------------------

@numba.njit
def _karplus_strong_cloud(samples, num_strings, delays, gains, decay):
    """Multiple Karplus-Strong string resonators at different frequencies."""
    n = len(samples)
    out = np.zeros(n, dtype=np.float32)

    for s in range(num_strings):
        delay = delays[s]
        gain = gains[s]
        buf_len = max(delay + 1, 2)
        buf = np.zeros(buf_len, dtype=np.float32)
        write_pos = 0

        # Initialize buffer with the input signal (noise burst excitation)
        for i in range(min(delay, n)):
            buf[i] = samples[i] * gain

        for i in range(n):
            # Read from delay line
            read_pos = (write_pos - delay) % buf_len
            read_pos_next = (read_pos + 1) % buf_len
            # KS averaging filter
            y = decay * 0.5 * (buf[read_pos] + buf[read_pos_next])

            # Add input excitation
            if i < delay:
                y += samples[i] * gain

            buf[write_pos] = y
            out[i] += y
            write_pos = (write_pos + 1) % buf_len

    return out


def effect_r014_karplus_strong_cloud(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Multiple Karplus-Strong resonators at random frequencies, creating a
    cloud of plucked-string-like resonances."""
    num_strings = int(params.get('num_strings', 15))
    min_freq = np.float32(params.get('min_freq', 80))
    max_freq = np.float32(params.get('max_freq', 1000))
    decay = np.float32(params.get('decay', 0.99))

    num_strings = max(10, min(30, num_strings))
    min_freq = np.clip(min_freq, 50, 200)
    max_freq = np.clip(max_freq, 500, 2000)
    decay = np.clip(decay, 0.95, 0.999)

    # Generate random frequencies (log-spaced)
    rng = np.random.RandomState(42)
    freqs = np.exp(rng.uniform(np.log(min_freq), np.log(max_freq), num_strings)).astype(np.float32)

    delays = np.zeros(num_strings, dtype=np.int64)
    gains = np.zeros(num_strings, dtype=np.float32)
    for i in range(num_strings):
        delays[i] = max(2, int(sr / freqs[i]))
        gains[i] = np.float32(1.0 / num_strings)

    out = _karplus_strong_cloud(samples.astype(np.float32), num_strings, delays, gains, decay)

    # Normalize
    in_rms = np.sqrt(np.mean(samples.astype(np.float32) ** 2)) + 1e-10
    out_rms = np.sqrt(np.mean(out ** 2)) + 1e-10
    out *= np.float32(in_rms / out_rms)

    return out


def variants_r014():
    return [
        {'num_strings': 10, 'min_freq': 80, 'max_freq': 500, 'decay': 0.99},
        {'num_strings': 15, 'min_freq': 80, 'max_freq': 1000, 'decay': 0.99},
        {'num_strings': 20, 'min_freq': 100, 'max_freq': 1500, 'decay': 0.995},
        {'num_strings': 30, 'min_freq': 50, 'max_freq': 2000, 'decay': 0.998},
        {'num_strings': 10, 'min_freq': 200, 'max_freq': 800, 'decay': 0.95},
        {'num_strings': 20, 'min_freq': 100, 'max_freq': 600, 'decay': 0.999},
    ]


# ---------------------------------------------------------------------------
# R015 -- Feedback FM Synthesis
# ---------------------------------------------------------------------------

@numba.njit
def _feedback_fm(samples, carrier_freq, mod_index, feedback_amount, sr):
    """y = x + fb * sin(carrier * n / sr + mod_index * y_prev)."""
    n = len(samples)
    out = np.empty(n, dtype=np.float32)
    two_pi = np.float32(2.0 * 3.141592653589793)
    y_prev = np.float32(0.0)

    for i in range(n):
        phase = two_pi * carrier_freq * np.float32(i) / np.float32(sr)
        fm_signal = np.float32(np.sin(phase + mod_index * y_prev))
        y = samples[i] + feedback_amount * fm_signal
        # Soft clip to prevent runaway
        if y > 1.0:
            y = np.float32(np.tanh(y))
        elif y < -1.0:
            y = np.float32(np.tanh(y))
        out[i] = y
        y_prev = y

    return out


def effect_r015_feedback_fm(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Feedback FM synthesis: y = x + fb * sin(carrier * n/sr + mod_index * y_prev).
    Creates metallic, bell-like, or chaotic timbres depending on parameters."""
    carrier_freq = np.float32(params.get('carrier_freq', 440))
    mod_index = np.float32(params.get('mod_index', 3.0))
    feedback = np.float32(params.get('feedback', 0.4))

    carrier_freq = np.clip(carrier_freq, 50, 2000)
    mod_index = np.clip(mod_index, 0.5, 10.0)
    feedback = np.clip(feedback, 0.1, 0.9)

    return _feedback_fm(samples.astype(np.float32), carrier_freq, mod_index, feedback, sr)


def variants_r015():
    return [
        {'carrier_freq': 200, 'mod_index': 1.0, 'feedback': 0.2},     # subtle FM coloring
        {'carrier_freq': 440, 'mod_index': 3.0, 'feedback': 0.4},      # metallic bell tone
        {'carrier_freq': 880, 'mod_index': 5.0, 'feedback': 0.3},      # bright FM shimmer
        {'carrier_freq': 100, 'mod_index': 7.0, 'feedback': 0.5},      # deep growling FM
        {'carrier_freq': 1500, 'mod_index': 2.0, 'feedback': 0.6},     # high-frequency buzz
        {'carrier_freq': 300, 'mod_index': 10.0, 'feedback': 0.9},     # chaotic extreme FM
        {'carrier_freq': 660, 'mod_index': 4.0, 'feedback': 0.15},     # gentle FM overtones
    ]


# ---------------------------------------------------------------------------
# R016 — Feedback AM Synthesis Effect
# y[n] = x[n] * sin(carrier + fb * y[n-1])
# Distinct from R015 (Feedback FM): AM multiplies the carrier with input,
# FM adds a modulated sine. AM creates sum/difference sidebands; feedback
# makes them evolve chaotically.
# ---------------------------------------------------------------------------

@numba.njit
def _feedback_am(samples, carrier_freq, feedback, depth, sr):
    n = len(samples)
    out = np.zeros(n, dtype=np.float32)
    two_pi = np.float32(2.0 * 3.141592653589793)
    y_prev = np.float32(0.0)

    for i in range(n):
        # Carrier phase modulated by feedback
        phase = two_pi * carrier_freq * np.float32(i) / np.float32(sr) + feedback * y_prev
        carrier = np.float32(np.sin(phase))

        # AM: multiply input by (1 + depth*carrier) to preserve some dry signal
        modulated = samples[i] * (np.float32(1.0) - depth + depth * carrier)

        # Soft clip
        if modulated > np.float32(1.0):
            modulated = np.float32(np.tanh(modulated))
        elif modulated < np.float32(-1.0):
            modulated = np.float32(np.tanh(modulated))

        out[i] = modulated
        y_prev = modulated

    return out


def effect_r016_feedback_am(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Feedback AM synthesis: y = x * sin(carrier + fb * y_prev).
    Creates sum/difference sidebands with chaotic feedback evolution."""
    carrier_freq = np.float32(params.get('carrier_freq', 200))
    feedback = np.float32(params.get('feedback', 0.3))
    depth = np.float32(params.get('depth', 0.8))

    carrier_freq = np.float32(np.clip(carrier_freq, 20, 2000))
    feedback = np.float32(np.clip(feedback, 0.0, 0.95))
    depth = np.float32(np.clip(depth, 0.1, 1.0))

    return _feedback_am(samples.astype(np.float32), carrier_freq, feedback, depth, sr)


def variants_r016():
    return [
        {'carrier_freq': 100, 'feedback': 0.0, 'depth': 0.5},     # pure AM, no feedback
        {'carrier_freq': 200, 'feedback': 0.2, 'depth': 0.7},      # mild feedback AM
        {'carrier_freq': 440, 'feedback': 0.4, 'depth': 0.8},      # musical feedback AM
        {'carrier_freq': 800, 'feedback': 0.3, 'depth': 1.0},      # bright sideband AM
        {'carrier_freq': 150, 'feedback': 0.7, 'depth': 0.9},      # chaotic low AM
        {'carrier_freq': 1200, 'feedback': 0.6, 'depth': 0.6},     # high chaotic AM
        {'carrier_freq': 300, 'feedback': 0.95, 'depth': 1.0},     # extreme feedback AM
    ]
