"""L-series effects: Convolution-based algorithms (L001-L006)."""
import numpy as np
import numba


def _next_pow2(n):
    """Return the smallest power of 2 >= n."""
    p = 1
    while p < n:
        p <<= 1
    return p


def _fft_convolve(x, h):
    """Convolve x with h using FFT overlap, padded to next power of 2."""
    n = len(x) + len(h) - 1
    fft_size = _next_pow2(n)
    X = np.fft.rfft(x, fft_size)
    H = np.fft.rfft(h, fft_size)
    out = np.fft.irfft(X * H, fft_size)
    return out[:len(x)].astype(np.float32)


# ---------------------------------------------------------------------------
# L001 -- Convolve with Mathematical IR
# ---------------------------------------------------------------------------

def _generate_ir(ir_type, ir_length_samples, decay_rate, freq_hz, sr):
    """Generate an impulse response from a mathematical function."""
    t = np.arange(ir_length_samples, dtype=np.float64) / sr

    if ir_type == 'exponential':
        ir = np.exp(-decay_rate * t)
    elif ir_type == 'sinc':
        # Sinc centered at midpoint with frequency scaling
        mid = ir_length_samples / 2.0
        t_centered = (np.arange(ir_length_samples, dtype=np.float64) - mid) / sr
        arg = freq_hz * t_centered
        ir = np.sinc(arg)  # np.sinc(x) = sin(pi*x) / (pi*x)
    elif ir_type == 'chirp':
        # Linear chirp from freq_hz/4 to freq_hz
        f0 = freq_hz * 0.25
        f1 = freq_hz
        phase = 2.0 * np.pi * (f0 * t + (f1 - f0) / (2.0 * t[-1] + 1e-12) * t * t)
        ir = np.sin(phase) * np.exp(-decay_rate * 0.5 * t)
    elif ir_type == 'gaussian':
        # Gaussian pulse centered at midpoint
        mid = ir_length_samples / 2.0
        sigma = ir_length_samples / (2.0 * decay_rate + 1e-12)
        idx = np.arange(ir_length_samples, dtype=np.float64)
        ir = np.exp(-0.5 * ((idx - mid) / sigma) ** 2)
        # Modulate with frequency
        ir *= np.cos(2.0 * np.pi * freq_hz * t)
    else:
        ir = np.exp(-decay_rate * t)

    # Normalize IR energy
    energy = np.sqrt(np.sum(ir * ir) + 1e-12)
    ir = ir / energy
    return ir.astype(np.float64)


def effect_l001_convolve_mathematical_ir(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Convolve signal with a mathematically generated impulse response."""
    ir_type = params.get('ir_type', 'exponential')
    ir_length_ms = float(params.get('ir_length_ms', 100))
    decay_rate = float(params.get('decay_rate', 5.0))
    freq_hz = float(params.get('freq_hz', 500))

    ir_length_samples = max(1, int(ir_length_ms * sr / 1000.0))
    ir = _generate_ir(ir_type, ir_length_samples, decay_rate, freq_hz, sr)
    return _fft_convolve(samples.astype(np.float64), ir)


def variants_l001():
    return [
        {'ir_type': 'exponential', 'ir_length_ms': 200, 'decay_rate': 3.0, 'freq_hz': 500},
        {'ir_type': 'exponential', 'ir_length_ms': 50, 'decay_rate': 15.0, 'freq_hz': 500},
        {'ir_type': 'sinc', 'ir_length_ms': 100, 'decay_rate': 5.0, 'freq_hz': 400},
        {'ir_type': 'sinc', 'ir_length_ms': 200, 'decay_rate': 5.0, 'freq_hz': 1500},
        {'ir_type': 'chirp', 'ir_length_ms': 150, 'decay_rate': 4.0, 'freq_hz': 800},
        {'ir_type': 'gaussian', 'ir_length_ms': 80, 'decay_rate': 6.0, 'freq_hz': 600},
        {'ir_type': 'gaussian', 'ir_length_ms': 300, 'decay_rate': 2.0, 'freq_hz': 1200},
    ]


# ---------------------------------------------------------------------------
# L002 -- Auto-Convolution
# ---------------------------------------------------------------------------

def effect_l002_auto_convolution(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Convolve signal with itself via FFT, iterated num_iterations times."""
    num_iterations = int(params.get('num_iterations', 1))
    num_iterations = max(1, min(4, num_iterations))

    x = samples.astype(np.float64)
    for _ in range(num_iterations):
        n = 2 * len(x) - 1
        fft_size = _next_pow2(n)
        X = np.fft.rfft(x, fft_size)
        x = np.fft.irfft(X * X, fft_size)
        # Trim back to original length
        x = x[:len(samples)]
        # Normalize to prevent explosion
        peak = np.max(np.abs(x)) + 1e-12
        x = x / peak

    return x.astype(np.float32)


def variants_l002():
    return [
        {'num_iterations': 1},
        {'num_iterations': 2},
        {'num_iterations': 3},
        {'num_iterations': 4},
    ]


# ---------------------------------------------------------------------------
# L003 -- Deconvolution
# ---------------------------------------------------------------------------

def effect_l003_deconvolution(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Deconvolution: y = ifft(fft(x) / (fft(ir) + eps))."""
    ir_type = params.get('ir_type', 'exponential')
    ir_length_ms = float(params.get('ir_length_ms', 50))
    epsilon = float(params.get('epsilon', 0.01))

    ir_length_samples = max(1, int(ir_length_ms * sr / 1000.0))

    # Generate IR for deconvolution
    t = np.arange(ir_length_samples, dtype=np.float64) / sr
    if ir_type == 'gaussian':
        mid = ir_length_samples / 2.0
        sigma = ir_length_samples / 6.0
        idx = np.arange(ir_length_samples, dtype=np.float64)
        ir = np.exp(-0.5 * ((idx - mid) / sigma) ** 2)
    else:  # exponential
        ir = np.exp(-5.0 * t)

    # Normalize IR
    energy = np.sqrt(np.sum(ir * ir) + 1e-12)
    ir = ir / energy

    x = samples.astype(np.float64)
    n = len(x) + len(ir) - 1
    fft_size = _next_pow2(n)

    X = np.fft.rfft(x, fft_size)
    H = np.fft.rfft(ir, fft_size)

    # Deconvolution with regularization
    Y = X / (H + epsilon)
    out = np.fft.irfft(Y, fft_size)[:len(x)]

    # Normalize output
    peak = np.max(np.abs(out)) + 1e-12
    out = out / peak

    return out.astype(np.float32)


def variants_l003():
    return [
        {'ir_type': 'exponential', 'ir_length_ms': 30, 'epsilon': 0.01},
        {'ir_type': 'exponential', 'ir_length_ms': 100, 'epsilon': 0.005},
        {'ir_type': 'exponential', 'ir_length_ms': 50, 'epsilon': 0.1},
        {'ir_type': 'gaussian', 'ir_length_ms': 50, 'epsilon': 0.01},
        {'ir_type': 'gaussian', 'ir_length_ms': 150, 'epsilon': 0.05},
        {'ir_type': 'exponential', 'ir_length_ms': 200, 'epsilon': 0.001},
    ]


# ---------------------------------------------------------------------------
# L004 -- Spectral Morphing
# ---------------------------------------------------------------------------

def _stft(x, fft_size, hop_size, window):
    """Short-time Fourier transform (analysis)."""
    n = len(x)
    num_frames = max(1, (n - fft_size) // hop_size + 1)
    num_bins = fft_size // 2 + 1
    stft_out = np.zeros((num_frames, num_bins), dtype=np.complex128)
    for f in range(num_frames):
        start = f * hop_size
        frame = x[start:start + fft_size] * window
        stft_out[f] = np.fft.rfft(frame)
    return stft_out


def _istft(stft_data, fft_size, hop_size, window, output_length):
    """Inverse STFT with overlap-add synthesis."""
    num_frames = stft_data.shape[0]
    out = np.zeros(output_length, dtype=np.float64)
    window_sum = np.zeros(output_length, dtype=np.float64)
    for f in range(num_frames):
        start = f * hop_size
        frame = np.fft.irfft(stft_data[f], fft_size) * window
        end = min(start + fft_size, output_length)
        length = end - start
        out[start:end] += frame[:length]
        window_sum[start:end] += window[:length] ** 2
    # Normalize by window sum to avoid modulation artifacts
    nonzero = window_sum > 1e-8
    out[nonzero] /= window_sum[nonzero]
    return out


def _generate_target_spectrum(target_type, num_bins, sr, fft_size):
    """Generate a target magnitude spectrum for morphing."""
    freqs = np.arange(num_bins, dtype=np.float64) * sr / fft_size

    if target_type == 'noise':
        # Flat white noise spectrum
        return np.ones(num_bins, dtype=np.float64)
    elif target_type == 'sawtooth':
        # Sawtooth: 1/k harmonic series based on 100 Hz fundamental
        mag = np.zeros(num_bins, dtype=np.float64)
        f0 = 100.0
        for k in range(1, num_bins):
            harmonic_freq = f0 * k
            bin_idx = int(harmonic_freq * fft_size / sr)
            if bin_idx < num_bins:
                mag[bin_idx] += 1.0 / k
        # Smooth slightly so it's not just spikes
        kernel = np.array([0.25, 0.5, 0.25], dtype=np.float64)
        padded = np.zeros(num_bins + 2, dtype=np.float64)
        padded[1:-1] = mag
        smoothed = np.zeros(num_bins, dtype=np.float64)
        for i in range(num_bins):
            smoothed[i] = padded[i] * kernel[0] + padded[i + 1] * kernel[1] + padded[i + 2] * kernel[2]
        return smoothed + 0.01  # small floor
    elif target_type == 'formant':
        # Vocal formant: peaks at ~700, 1200, 2500 Hz
        formant_freqs = [700.0, 1200.0, 2500.0]
        formant_bws = [130.0, 70.0, 160.0]
        formant_amps = [1.0, 0.6, 0.3]
        mag = np.zeros(num_bins, dtype=np.float64)
        for fi in range(len(formant_freqs)):
            fc = formant_freqs[fi]
            bw = formant_bws[fi]
            amp = formant_amps[fi]
            mag += amp * np.exp(-0.5 * ((freqs - fc) / bw) ** 2)
        return mag + 0.01
    else:
        return np.ones(num_bins, dtype=np.float64)


def effect_l004_spectral_morphing(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Morph magnitude spectrum toward a target while preserving phase."""
    target_type = params.get('target_type', 'noise')
    alpha = float(params.get('alpha', 0.5))
    alpha = max(0.0, min(1.0, alpha))

    x = samples.astype(np.float64)
    n = len(x)

    fft_size = 2048
    hop_size = fft_size // 4
    window = np.hanning(fft_size).astype(np.float64)

    # Analysis
    spec = _stft(x, fft_size, hop_size, window)
    num_bins = fft_size // 2 + 1

    # Generate target magnitude spectrum
    target_mag = _generate_target_spectrum(target_type, num_bins, sr, fft_size)
    # Normalize target to similar energy as average frame
    avg_mag = np.mean(np.abs(spec), axis=0) + 1e-12
    target_mag = target_mag * (np.mean(avg_mag) / (np.mean(target_mag) + 1e-12))

    # Morph each frame
    num_frames = spec.shape[0]
    morphed = np.zeros_like(spec)
    for f in range(num_frames):
        mag = np.abs(spec[f])
        phase = np.angle(spec[f])
        # Linear interpolation of magnitude
        new_mag = (1.0 - alpha) * mag + alpha * target_mag
        morphed[f] = new_mag * np.exp(1j * phase)

    # Synthesis
    out = _istft(morphed, fft_size, hop_size, window, n)

    return out.astype(np.float32)


def variants_l004():
    return [
        {'target_type': 'noise', 'alpha': 0.3},
        {'target_type': 'noise', 'alpha': 0.7},
        {'target_type': 'noise', 'alpha': 1.0},
        {'target_type': 'sawtooth', 'alpha': 0.5},
        {'target_type': 'sawtooth', 'alpha': 0.9},
        {'target_type': 'formant', 'alpha': 0.4},
        {'target_type': 'formant', 'alpha': 0.8},
    ]


# ---------------------------------------------------------------------------
# L005 -- Convolution with Chirp IR
# ---------------------------------------------------------------------------

def effect_l005_convolution_chirp_ir(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Convolve signal with a chirp impulse response sweeping from f_start to f_end."""
    f_start = float(params.get('f_start', 100))
    f_end = float(params.get('f_end', 8000))
    chirp_duration_ms = float(params.get('chirp_duration_ms', 100))

    chirp_samples = max(1, int(chirp_duration_ms * sr / 1000.0))
    t = np.arange(chirp_samples, dtype=np.float64) / sr
    duration = t[-1] + 1e-12

    # Linear chirp: instantaneous frequency sweeps linearly from f_start to f_end
    phase = 2.0 * np.pi * (f_start * t + (f_end - f_start) / (2.0 * duration) * t * t)
    ir = np.sin(phase)

    # Apply exponential decay envelope
    ir *= np.exp(-3.0 * t / duration)

    # Normalize IR
    energy = np.sqrt(np.sum(ir * ir) + 1e-12)
    ir = ir / energy

    return _fft_convolve(samples.astype(np.float64), ir)


def variants_l005():
    return [
        {'f_start': 100, 'f_end': 4000, 'chirp_duration_ms': 50},
        {'f_start': 50, 'f_end': 10000, 'chirp_duration_ms': 100},
        {'f_start': 200, 'f_end': 2000, 'chirp_duration_ms': 200},
        {'f_start': 500, 'f_end': 15000, 'chirp_duration_ms': 30},
        {'f_start': 20, 'f_end': 8000, 'chirp_duration_ms': 500},
        {'f_start': 1000, 'f_end': 1500, 'chirp_duration_ms': 150},
    ]


# ---------------------------------------------------------------------------
# L006 -- Morphological Audio Processing
# ---------------------------------------------------------------------------

@numba.njit
def _morpho_dilate(samples, kernel_size):
    """Dilation: sliding maximum."""
    n = len(samples)
    out = np.empty(n, dtype=np.float32)
    half = kernel_size // 2
    for i in range(n):
        max_val = np.float32(-1e30)
        for k in range(-half, half + 1):
            idx = i + k
            if 0 <= idx < n:
                if samples[idx] > max_val:
                    max_val = samples[idx]
        out[i] = max_val
    return out


@numba.njit
def _morpho_erode(samples, kernel_size):
    """Erosion: sliding minimum."""
    n = len(samples)
    out = np.empty(n, dtype=np.float32)
    half = kernel_size // 2
    for i in range(n):
        min_val = np.float32(1e30)
        for k in range(-half, half + 1):
            idx = i + k
            if 0 <= idx < n:
                if samples[idx] < min_val:
                    min_val = samples[idx]
        out[i] = min_val
    return out


def effect_l006_morphological_processing(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Morphological audio processing: dilation, erosion, opening, closing."""
    kernel_size = int(params.get('kernel_size', 11))
    operation = params.get('operation', 'dilate')

    # Force odd kernel size
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = max(3, min(51, kernel_size))

    s = samples.astype(np.float32)

    if operation == 'dilate':
        return _morpho_dilate(s, kernel_size)
    elif operation == 'erode':
        return _morpho_erode(s, kernel_size)
    elif operation == 'open':
        # Opening = erosion followed by dilation
        eroded = _morpho_erode(s, kernel_size)
        return _morpho_dilate(eroded, kernel_size)
    elif operation == 'close':
        # Closing = dilation followed by erosion
        dilated = _morpho_dilate(s, kernel_size)
        return _morpho_erode(dilated, kernel_size)
    else:
        return _morpho_dilate(s, kernel_size)


def variants_l006():
    return [
        {'kernel_size': 5, 'operation': 'dilate'},
        {'kernel_size': 21, 'operation': 'dilate'},
        {'kernel_size': 5, 'operation': 'erode'},
        {'kernel_size': 21, 'operation': 'erode'},
        {'kernel_size': 11, 'operation': 'open'},
        {'kernel_size': 11, 'operation': 'close'},
        {'kernel_size': 31, 'operation': 'open'},
        {'kernel_size': 31, 'operation': 'close'},
    ]
