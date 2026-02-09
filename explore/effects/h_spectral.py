"""H-series: FFT/spectral effects (H001-H019)."""
import numpy as np
import numba
from scipy.ndimage import uniform_filter1d, median_filter


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
# H001 -- Spectral Freeze
# ---------------------------------------------------------------------------

def effect_h001_spectral_freeze(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Freeze magnitudes at a chosen position, letting phases advance freely.

    Creates a sustained, shimmering texture from whatever spectral content
    exists at the freeze point.
    """
    freeze_position = params.get('freeze_position', 0.3)
    fft_size = params.get('fft_size', 2048)
    hop_size = fft_size // 4

    X = _stft(samples, fft_size, hop_size)
    num_frames = X.shape[0]
    if num_frames == 0:
        return samples.copy()

    freeze_frame = int(np.clip(freeze_position, 0.0, 1.0) * (num_frames - 1))
    frozen_mag = np.abs(X[freeze_frame])

    phases = np.angle(X)
    Y = frozen_mag[np.newaxis, :] * np.exp(1j * phases)

    return _istft(Y, fft_size, hop_size, length=len(samples))


def variants_h001():
    return [
        {'freeze_position': 0.0, 'fft_size': 2048},
        {'freeze_position': 0.25, 'fft_size': 2048},
        {'freeze_position': 0.5, 'fft_size': 2048},
        {'freeze_position': 0.75, 'fft_size': 2048},
        {'freeze_position': 1.0, 'fft_size': 2048},
        {'freeze_position': 0.5, 'fft_size': 4096},
        {'freeze_position': 0.5, 'fft_size': 1024},
    ]


# ---------------------------------------------------------------------------
# H002 -- Spectral Blur
# ---------------------------------------------------------------------------

def effect_h002_spectral_blur(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Gaussian-style blur across frequency bins.

    Smears spectral detail, creating dreamy, washed-out tonal textures.
    """
    blur_width = params.get('blur_width', 10)
    fft_size = params.get('fft_size', 2048)
    hop_size = fft_size // 4

    X = _stft(samples, fft_size, hop_size)
    mag = np.abs(X)
    phase = np.angle(X)

    blur_width = max(1, int(blur_width))
    # uniform_filter1d along the frequency axis (axis=1) approximates gaussian blur
    mag_blurred = uniform_filter1d(mag, size=blur_width, axis=1).astype(np.float32)

    Y = mag_blurred * np.exp(1j * phase)
    return _istft(Y, fft_size, hop_size, length=len(samples))


def variants_h002():
    return [
        {'blur_width': 1},
        {'blur_width': 5},
        {'blur_width': 10},
        {'blur_width': 20},
        {'blur_width': 35},
        {'blur_width': 50},
    ]


# ---------------------------------------------------------------------------
# H003 -- Spectral Gate
# ---------------------------------------------------------------------------

def effect_h003_spectral_gate(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Zero out frequency bins below a percentile threshold.

    Retains only the strongest spectral content, removing quiet components.
    """
    gate_percentile = params.get('gate_percentile', 75)
    fft_size = params.get('fft_size', 2048)
    hop_size = fft_size // 4

    X = _stft(samples, fft_size, hop_size)
    mag = np.abs(X)

    for i in range(X.shape[0]):
        threshold = np.percentile(mag[i], gate_percentile)
        mask = mag[i] >= threshold
        X[i] *= mask

    return _istft(X, fft_size, hop_size, length=len(samples))


def variants_h003():
    return [
        {'gate_percentile': 50},
        {'gate_percentile': 65},
        {'gate_percentile': 75},
        {'gate_percentile': 85},
        {'gate_percentile': 92},
        {'gate_percentile': 99},
    ]


# ---------------------------------------------------------------------------
# H004 -- Spectral Shift
# ---------------------------------------------------------------------------

def effect_h004_spectral_shift(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Shift all frequency bins up or down by a fixed number of bins.

    Produces inharmonic, bell-like or metallic timbres.
    """
    shift_bins = params.get('shift_bins', 10)
    fft_size = params.get('fft_size', 2048)
    hop_size = fft_size // 4

    X = _stft(samples, fft_size, hop_size)
    num_bins = X.shape[1]
    shift = int(shift_bins)

    Y = np.zeros_like(X)
    for i in range(num_frames := X.shape[0]):
        for b in range(num_bins):
            src = b - shift
            if 0 <= src < num_bins:
                Y[i, b] = X[i, src]

    return _istft(Y, fft_size, hop_size, length=len(samples))


def variants_h004():
    return [
        {'shift_bins': -100},
        {'shift_bins': -30},
        {'shift_bins': -10},
        {'shift_bins': 5},
        {'shift_bins': 10},
        {'shift_bins': 30},
        {'shift_bins': 100},
    ]


# ---------------------------------------------------------------------------
# H005 -- Phase Randomization
# ---------------------------------------------------------------------------

def effect_h005_phase_randomization(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Replace original phases with random phases, blended by amount.

    Full randomization creates a noise-like signal with the original spectrum.
    """
    amount = params.get('amount', 0.5)
    fft_size = params.get('fft_size', 2048)
    hop_size = fft_size // 4

    X = _stft(samples, fft_size, hop_size)
    mag = np.abs(X)
    orig_phase = np.angle(X)

    rng = np.random.default_rng(42)
    random_phase = rng.uniform(-np.pi, np.pi, size=X.shape).astype(np.float32)

    blended_phase = (1.0 - amount) * orig_phase + amount * random_phase
    Y = mag * np.exp(1j * blended_phase)

    return _istft(Y, fft_size, hop_size, length=len(samples))


def variants_h005():
    return [
        {'amount': 0.0},
        {'amount': 0.2},
        {'amount': 0.5},
        {'amount': 0.8},
        {'amount': 1.0},
    ]


# ---------------------------------------------------------------------------
# H006 -- Robotization
# ---------------------------------------------------------------------------

def effect_h006_robotization(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Set all phases to zero, creating a buzzy, robotic tonal quality.

    The perceived pitch becomes the hop rate rather than the original pitch.
    """
    fft_size = params.get('fft_size', 2048)
    hop_size = fft_size // 4

    X = _stft(samples, fft_size, hop_size)
    mag = np.abs(X)
    # All phases zero
    Y = mag.astype(np.complex128)

    return _istft(Y, fft_size, hop_size, length=len(samples))


def variants_h006():
    return [
        {'fft_size': 512},
        {'fft_size': 1024},
        {'fft_size': 2048},
        {'fft_size': 4096},
        {'fft_size': 8192},
    ]


# ---------------------------------------------------------------------------
# H007 -- Spectral Bin Sorting
# ---------------------------------------------------------------------------

def effect_h007_spectral_bin_sorting(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Sort magnitude bins in ascending or descending order.

    Creates unusual spectral envelopes -- ascending pushes energy to high
    frequencies, descending to low frequencies.
    """
    order = params.get('order', 'descending')
    partial_sort = params.get('partial_sort', 1.0)
    fft_size = params.get('fft_size', 2048)
    hop_size = fft_size // 4

    X = _stft(samples, fft_size, hop_size)
    mag = np.abs(X)
    phase = np.angle(X)

    ascending = (order == 'ascending')

    for i in range(X.shape[0]):
        num_bins = mag.shape[1]
        n_sort = max(1, int(num_bins * np.clip(partial_sort, 0.1, 1.0)))
        indices = np.argsort(mag[i, :n_sort])
        if not ascending:
            indices = indices[::-1]
        sorted_mag = np.copy(mag[i])
        sorted_mag[:n_sort] = mag[i, :n_sort][indices]
        mag[i] = sorted_mag

    Y = mag * np.exp(1j * phase)
    return _istft(Y, fft_size, hop_size, length=len(samples))


def variants_h007():
    return [
        {'order': 'ascending', 'partial_sort': 1.0},
        {'order': 'descending', 'partial_sort': 1.0},
        {'order': 'ascending', 'partial_sort': 0.3},
        {'order': 'descending', 'partial_sort': 0.3},
        {'order': 'ascending', 'partial_sort': 0.1},
        {'order': 'descending', 'partial_sort': 0.6},
    ]


# ---------------------------------------------------------------------------
# H008 -- Spectral Bin Permutation
# ---------------------------------------------------------------------------

def effect_h008_spectral_bin_permutation(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Randomly permute frequency bins.

    Scrambles the spectral content while preserving overall energy.
    Partial permutation blends original and permuted positions.
    """
    seed = params.get('seed', 42)
    permutation_amount = params.get('permutation_amount', 0.5)
    fft_size = params.get('fft_size', 2048)
    hop_size = fft_size // 4

    X = _stft(samples, fft_size, hop_size)
    num_bins = X.shape[1]

    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_bins)

    Y = np.zeros_like(X)
    amount = np.clip(permutation_amount, 0.0, 1.0)
    for i in range(X.shape[0]):
        permuted = X[i, perm]
        Y[i] = (1.0 - amount) * X[i] + amount * permuted

    return _istft(Y, fft_size, hop_size, length=len(samples))


def variants_h008():
    return [
        {'seed': 42, 'permutation_amount': 0.2},
        {'seed': 42, 'permutation_amount': 0.5},
        {'seed': 42, 'permutation_amount': 1.0},
        {'seed': 123, 'permutation_amount': 0.5},
        {'seed': 7, 'permutation_amount': 0.8},
        {'seed': 999, 'permutation_amount': 1.0},
    ]


# ---------------------------------------------------------------------------
# H009 -- Spectral Cross-Synthesis
# ---------------------------------------------------------------------------

def _make_carrier(source_type, length, sr):
    """Generate a carrier signal for cross-synthesis."""
    t = np.arange(length, dtype=np.float32) / sr
    if source_type == 'noise':
        rng = np.random.default_rng(42)
        return rng.standard_normal(length).astype(np.float32) * 0.5
    elif source_type == 'sine_sweep':
        # Sweep from 100 Hz to 8000 Hz
        duration = length / sr
        phase = 2.0 * np.pi * 100.0 * duration * (
            np.exp(t / duration * np.log(8000.0 / 100.0)) - 1.0
        ) / np.log(8000.0 / 100.0)
        return (0.5 * np.sin(phase)).astype(np.float32)
    elif source_type == 'sawtooth':
        freq = 100.0
        return (0.5 * (2.0 * (t * freq - np.floor(t * freq + 0.5)))).astype(np.float32)
    else:
        rng = np.random.default_rng(42)
        return rng.standard_normal(length).astype(np.float32) * 0.5


def effect_h009_spectral_cross_synthesis(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Cross-synthesize: use original magnitudes with carrier phases, blended.

    Imposes the spectral envelope of the input onto a synthetic carrier signal.
    """
    source_type = params.get('source_type', 'noise')
    blend = params.get('blend', 0.5)
    fft_size = params.get('fft_size', 2048)
    hop_size = fft_size // 4

    carrier = _make_carrier(source_type, len(samples), sr)

    X_input = _stft(samples, fft_size, hop_size)
    X_carrier = _stft(carrier, fft_size, hop_size)

    mag_input = np.abs(X_input)
    mag_carrier = np.abs(X_carrier)
    phase_carrier = np.angle(X_carrier)

    # Blend magnitudes: input magnitudes shaped by carrier phase
    blended_mag = (1.0 - blend) * mag_input + blend * mag_carrier
    # Use carrier phases entirely for the cross-synth effect
    Y = blended_mag * np.exp(1j * phase_carrier)

    # Actually for cross-synthesis the classic approach is input magnitudes + carrier phases
    # Rewrite: always use input magnitudes, blend controls how much carrier phase vs input phase
    mag = mag_input
    phase_input = np.angle(X_input)
    blended_phase = (1.0 - blend) * phase_input + blend * phase_carrier
    Y = mag * np.exp(1j * blended_phase)

    return _istft(Y, fft_size, hop_size, length=len(samples))


def variants_h009():
    return [
        {'source_type': 'noise', 'blend': 0.3},
        {'source_type': 'noise', 'blend': 0.7},
        {'source_type': 'noise', 'blend': 1.0},
        {'source_type': 'sine_sweep', 'blend': 0.5},
        {'source_type': 'sine_sweep', 'blend': 1.0},
        {'source_type': 'sawtooth', 'blend': 0.5},
        {'source_type': 'sawtooth', 'blend': 1.0},
    ]


# ---------------------------------------------------------------------------
# H010 -- Classic Channel Vocoder
# ---------------------------------------------------------------------------

@numba.njit
def _envelope_follow(signal, attack_coeff, release_coeff):
    """Envelope follower for vocoder band."""
    n = len(signal)
    env = np.zeros(n, dtype=np.float32)
    prev = np.float32(0.0)
    for i in range(n):
        inp = abs(signal[i])
        if inp > prev:
            prev = attack_coeff * prev + (np.float32(1.0) - attack_coeff) * inp
        else:
            prev = release_coeff * prev + (np.float32(1.0) - release_coeff) * inp
        env[i] = prev
    return env


@numba.njit
def _biquad_bandpass(samples, freq_hz, sr, Q):
    """Biquad bandpass filter (numba-compatible)."""
    w0 = 2.0 * np.pi * freq_hz / sr
    alpha = np.sin(w0) / (2.0 * Q)
    b0 = np.float32(alpha / (1.0 + alpha))
    b1 = np.float32(0.0)
    b2 = np.float32(-alpha / (1.0 + alpha))
    a1 = np.float32(-2.0 * np.cos(w0) / (1.0 + alpha))
    a2 = np.float32((1.0 - alpha) / (1.0 + alpha))

    n = len(samples)
    out = np.zeros(n, dtype=np.float32)
    z1 = np.float32(0.0)
    z2 = np.float32(0.0)
    for i in range(n):
        x = samples[i]
        y = b0 * x + z1
        z1 = b1 * x - a1 * y + z2
        z2 = b2 * x - a2 * y
        out[i] = y
    return out


@numba.njit
def _vocoder_process(analysis_signal, carrier_signal, band_freqs, sr,
                     attack_coeff, release_coeff, Q):
    """Per-band vocoder processing using numba."""
    n = len(analysis_signal)
    num_bands = len(band_freqs)
    output = np.zeros(n, dtype=np.float32)

    for b in range(num_bands):
        freq = band_freqs[b]
        # Filter analysis signal through bandpass
        analysis_band = _biquad_bandpass(analysis_signal, freq, sr, Q)
        # Extract envelope
        env = _envelope_follow(analysis_band, attack_coeff, release_coeff)
        # Filter carrier through same bandpass
        carrier_band = _biquad_bandpass(carrier_signal, freq, sr, Q)
        # Modulate carrier band by envelope
        for i in range(n):
            output[i] += carrier_band[i] * env[i]

    return output


def effect_h010_classic_channel_vocoder(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Classic channel vocoder using bandpass filter bank.

    Analyzes the input's spectral envelope per band and applies it to a carrier.
    """
    num_bands = params.get('num_bands', 16)
    carrier_type = params.get('carrier_type', 'noise')
    fft_size = params.get('fft_size', 2048)

    num_bands = max(8, min(64, int(num_bands)))
    n = len(samples)

    # Generate carrier
    if carrier_type == 'saw':
        t = np.arange(n, dtype=np.float32) / sr
        freq = 100.0
        carrier = (2.0 * (t * freq - np.floor(t * freq + 0.5))).astype(np.float32)
    elif carrier_type == 'input_self':
        carrier = samples.copy()
    else:  # noise
        rng = np.random.default_rng(42)
        carrier = rng.standard_normal(n).astype(np.float32)

    # Logarithmically spaced band center frequencies
    low_freq = 80.0
    high_freq = min(sr / 2.0 - 100.0, 12000.0)
    band_freqs = np.exp(np.linspace(np.log(low_freq), np.log(high_freq), num_bands)).astype(np.float64)

    # Envelope follower coefficients: ~5 ms attack, ~20 ms release
    attack_coeff = np.float32(np.exp(-1.0 / (0.005 * sr)))
    release_coeff = np.float32(np.exp(-1.0 / (0.020 * sr)))
    Q = np.float64(2.0)

    output = _vocoder_process(
        samples.astype(np.float32),
        carrier.astype(np.float32),
        band_freqs,
        np.float64(sr),
        attack_coeff,
        release_coeff,
        Q,
    )

    # Normalize to match input level
    in_rms = np.sqrt(np.mean(samples ** 2)) + 1e-10
    out_rms = np.sqrt(np.mean(output ** 2)) + 1e-10
    output *= (in_rms / out_rms)

    return output


def variants_h010():
    return [
        {'num_bands': 8, 'carrier_type': 'noise'},
        {'num_bands': 16, 'carrier_type': 'noise'},
        {'num_bands': 32, 'carrier_type': 'noise'},
        {'num_bands': 64, 'carrier_type': 'noise'},
        {'num_bands': 16, 'carrier_type': 'saw'},
        {'num_bands': 32, 'carrier_type': 'saw'},
        {'num_bands': 16, 'carrier_type': 'input_self'},
    ]


# ---------------------------------------------------------------------------
# H011 -- Harmonic/Percussive Separation
# ---------------------------------------------------------------------------

def effect_h011_harmonic_percussive_separation(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Separate harmonic and percussive components using median filtering.

    Harmonic content is stable across time (horizontal median), percussive
    content is stable across frequency (vertical median).
    """
    filter_length = params.get('filter_length', 17)
    output_mode = params.get('output', 'harmonic')
    fft_size = params.get('fft_size', 2048)
    hop_size = fft_size // 4

    # Ensure odd filter length
    filter_length = max(3, int(filter_length))
    if filter_length % 2 == 0:
        filter_length += 1

    X = _stft(samples, fft_size, hop_size)
    mag = np.abs(X)
    phase = np.angle(X)

    # Harmonic: median filter along time axis (axis=0) -- stable across time
    harmonic_mag = median_filter(mag, size=(filter_length, 1)).astype(np.float32)
    # Percussive: median filter along frequency axis (axis=1) -- stable across freq
    percussive_mag = median_filter(mag, size=(1, filter_length)).astype(np.float32)

    # Soft masks (Wiener-style)
    total = harmonic_mag + percussive_mag + 1e-10
    harmonic_mask = harmonic_mag / total
    percussive_mask = percussive_mag / total

    if output_mode == 'harmonic':
        Y = mag * harmonic_mask * np.exp(1j * phase)
    elif output_mode == 'percussive':
        Y = mag * percussive_mask * np.exp(1j * phase)
    else:  # remix -- boost harmonic, keep percussive
        remix_mag = mag * (0.7 * harmonic_mask + 1.3 * percussive_mask)
        Y = remix_mag * np.exp(1j * phase)

    return _istft(Y, fft_size, hop_size, length=len(samples))


def variants_h011():
    return [
        {'filter_length': 7, 'output': 'harmonic'},
        {'filter_length': 17, 'output': 'harmonic'},
        {'filter_length': 31, 'output': 'harmonic'},
        {'filter_length': 7, 'output': 'percussive'},
        {'filter_length': 17, 'output': 'percussive'},
        {'filter_length': 31, 'output': 'percussive'},
        {'filter_length': 17, 'output': 'remix'},
    ]


# ---------------------------------------------------------------------------
# H012 -- Spectral Mirror
# ---------------------------------------------------------------------------

def effect_h012_spectral_mirror(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Mirror magnitude spectrum around a center frequency.

    Bins below the center are reflected above and vice versa, creating
    unusual harmonic relationships.
    """
    mirror_center_hz = params.get('mirror_center_hz', 2000)
    fft_size = params.get('fft_size', 2048)
    hop_size = fft_size // 4

    X = _stft(samples, fft_size, hop_size)
    mag = np.abs(X)
    phase = np.angle(X)
    num_bins = X.shape[1]

    bin_hz = sr / fft_size
    center_bin = int(np.clip(mirror_center_hz / bin_hz, 1, num_bins - 2))

    mirrored_mag = np.zeros_like(mag)
    for b in range(num_bins):
        # Mirror around center_bin: new_bin = 2 * center_bin - b
        src = 2 * center_bin - b
        if 0 <= src < num_bins:
            mirrored_mag[:, b] = mag[:, src]
        # Bins outside range stay zero (silence)

    Y = mirrored_mag * np.exp(1j * phase)
    return _istft(Y, fft_size, hop_size, length=len(samples))


def variants_h012():
    return [
        {'mirror_center_hz': 500},
        {'mirror_center_hz': 1000},
        {'mirror_center_hz': 2000},
        {'mirror_center_hz': 3000},
        {'mirror_center_hz': 5000},
    ]


# ---------------------------------------------------------------------------
# H013 -- Spectral Stretch/Compress
# ---------------------------------------------------------------------------

def effect_h013_spectral_stretch_compress(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Stretch or compress the magnitude spectrum along the frequency axis.

    Stretch > 1 spreads harmonics apart (inharmonic); compress < 1 squeezes
    them together.
    """
    spectral_stretch = params.get('spectral_stretch', 1.5)
    fft_size = params.get('fft_size', 2048)
    hop_size = fft_size // 4

    X = _stft(samples, fft_size, hop_size)
    mag = np.abs(X)
    phase = np.angle(X)
    num_bins = X.shape[1]

    stretched_mag = np.zeros_like(mag)
    src_indices = np.arange(num_bins, dtype=np.float64) / spectral_stretch

    for b in range(num_bins):
        src = src_indices[b]
        src_int = int(src)
        frac = np.float32(src - src_int)
        if src_int < 0 or src_int >= num_bins:
            continue
        if src_int + 1 < num_bins:
            stretched_mag[:, b] = (1.0 - frac) * mag[:, src_int] + frac * mag[:, src_int + 1]
        else:
            stretched_mag[:, b] = mag[:, src_int]

    Y = stretched_mag * np.exp(1j * phase)
    return _istft(Y, fft_size, hop_size, length=len(samples))


def variants_h013():
    return [
        {'spectral_stretch': 0.5},
        {'spectral_stretch': 0.75},
        {'spectral_stretch': 1.0},
        {'spectral_stretch': 1.25},
        {'spectral_stretch': 1.5},
        {'spectral_stretch': 2.0},
    ]


# ---------------------------------------------------------------------------
# H014 -- Cepstral Processing
# ---------------------------------------------------------------------------

def effect_h014_cepstral_processing(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Lifter in the cepstral domain to smooth or modify spectral envelope.

    'smooth_envelope' low-passes the cepstrum (keeps vocal formants, removes
    pitch). 'remove_pitch' high-passes the cepstrum (removes harmonics, keeps
    noise-like residual).
    """
    lifter_cutoff = params.get('lifter_cutoff', 30)
    operation = params.get('operation', 'smooth_envelope')
    fft_size = params.get('fft_size', 2048)
    hop_size = fft_size // 4

    X = _stft(samples, fft_size, hop_size)
    num_frames, num_bins = X.shape

    Y = np.zeros_like(X)
    lifter_cutoff = max(1, int(lifter_cutoff))

    for i in range(num_frames):
        log_mag = np.log(np.abs(X[i]) + 1e-10)
        cepstrum = np.fft.irfft(log_mag)

        if operation == 'smooth_envelope':
            # Low-pass lifter: zero out high quefrency components
            liftered = np.zeros_like(cepstrum)
            n_cep = len(cepstrum)
            cutoff = min(lifter_cutoff, n_cep)
            liftered[:cutoff] = cepstrum[:cutoff]
        else:  # remove_pitch
            # High-pass lifter: zero out low quefrency (envelope)
            liftered = cepstrum.copy()
            cutoff = min(lifter_cutoff, len(liftered))
            liftered[:cutoff] = 0.0

        smoothed_log_mag = np.fft.rfft(liftered).real
        smoothed_mag = np.exp(smoothed_log_mag).astype(np.float32)
        Y[i] = smoothed_mag * np.exp(1j * np.angle(X[i]))

    return _istft(Y, fft_size, hop_size, length=len(samples))


def variants_h014():
    return [
        {'lifter_cutoff': 10, 'operation': 'smooth_envelope'},
        {'lifter_cutoff': 30, 'operation': 'smooth_envelope'},
        {'lifter_cutoff': 60, 'operation': 'smooth_envelope'},
        {'lifter_cutoff': 100, 'operation': 'smooth_envelope'},
        {'lifter_cutoff': 10, 'operation': 'remove_pitch'},
        {'lifter_cutoff': 30, 'operation': 'remove_pitch'},
        {'lifter_cutoff': 60, 'operation': 'remove_pitch'},
    ]


# ---------------------------------------------------------------------------
# H015 -- Spectral Delay
# ---------------------------------------------------------------------------

def effect_h015_spectral_delay(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Apply per-bin delay: higher bins are delayed more than lower bins.

    Creates a frequency-dependent smearing / dispersive effect.
    """
    base_delay_ms = params.get('base_delay_ms', 10)
    delay_slope_ms_per_bin = params.get('delay_slope_ms_per_bin', 0.1)
    fft_size = params.get('fft_size', 2048)
    hop_size = fft_size // 4

    X = _stft(samples, fft_size, hop_size)
    num_frames, num_bins = X.shape

    # Compute delay in frames for each bin
    hop_duration_ms = (hop_size / sr) * 1000.0
    Y = np.zeros_like(X)

    for b in range(num_bins):
        delay_ms = base_delay_ms + delay_slope_ms_per_bin * b
        delay_frames = int(delay_ms / hop_duration_ms)
        for t in range(num_frames):
            src_t = t - delay_frames
            if 0 <= src_t < num_frames:
                Y[t, b] = X[src_t, b]

    return _istft(Y, fft_size, hop_size, length=len(samples))


def variants_h015():
    return [
        {'base_delay_ms': 0, 'delay_slope_ms_per_bin': 0.1},
        {'base_delay_ms': 0, 'delay_slope_ms_per_bin': 0.5},
        {'base_delay_ms': 0, 'delay_slope_ms_per_bin': 1.0},
        {'base_delay_ms': 20, 'delay_slope_ms_per_bin': 0.0},
        {'base_delay_ms': 50, 'delay_slope_ms_per_bin': 0.2},
        {'base_delay_ms': 100, 'delay_slope_ms_per_bin': 0.5},
    ]


# ---------------------------------------------------------------------------
# H016 -- Spectral Compressor
# ---------------------------------------------------------------------------

def effect_h016_spectral_compressor(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Compress magnitude dynamics independently per frequency bin.

    Reduces the dynamic range of each spectral bin, making quiet frequencies
    louder and loud frequencies quieter.
    """
    threshold_db = params.get('threshold_db', -30)
    ratio = params.get('ratio', 4)
    fft_size = params.get('fft_size', 2048)
    hop_size = fft_size // 4

    X = _stft(samples, fft_size, hop_size)
    mag = np.abs(X)
    phase = np.angle(X)

    # Convert to dB
    mag_db = 20.0 * np.log10(mag + 1e-10)

    # Compress: for levels above threshold, reduce by ratio
    compressed_db = np.where(
        mag_db > threshold_db,
        threshold_db + (mag_db - threshold_db) / ratio,
        mag_db,
    )

    compressed_mag = (10.0 ** (compressed_db / 20.0)).astype(np.float32)

    # Make-up gain: compensate for average level reduction
    avg_reduction = np.mean(mag_db - compressed_db)
    makeup_linear = 10.0 ** (avg_reduction / 40.0)  # half the avg reduction as makeup
    compressed_mag *= makeup_linear

    Y = compressed_mag * np.exp(1j * phase)
    return _istft(Y, fft_size, hop_size, length=len(samples))


def variants_h016():
    return [
        {'threshold_db': -10, 'ratio': 2},
        {'threshold_db': -20, 'ratio': 2},
        {'threshold_db': -30, 'ratio': 4},
        {'threshold_db': -40, 'ratio': 4},
        {'threshold_db': -60, 'ratio': 10},
        {'threshold_db': -30, 'ratio': 20},
    ]


# ---------------------------------------------------------------------------
# H017 -- Spectral Reassignment
# ---------------------------------------------------------------------------

def effect_h017_spectral_reassignment(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Sharpen the spectrogram via spectral reassignment.

    Redistributes energy from each bin to its reassigned frequency, creating
    a tighter, more focused spectral representation. The sharpening amount
    controls interpolation between original and reassigned.
    """
    sharpening_amount = params.get('sharpening_amount', 0.5)
    fft_size = params.get('fft_size', 2048)
    hop_size = fft_size // 4

    x = samples.astype(np.float32)
    n = len(x)
    if n < fft_size:
        x = np.concatenate([x, np.zeros(fft_size - n, dtype=np.float32)])
        n = fft_size

    window = np.hanning(fft_size).astype(np.float64)
    # Time-weighted window for reassignment
    t_window = (np.arange(fft_size) - fft_size / 2.0) / sr
    dwindow = t_window * window  # time-ramped window

    num_frames = 1 + (n - fft_size) // hop_size
    num_bins = fft_size // 2 + 1

    # Standard STFT
    X = _stft(x, fft_size, hop_size)

    # Time-derivative STFT (using time-ramped window)
    frames_d = np.zeros((num_frames, fft_size), dtype=np.float64)
    for i in range(num_frames):
        start = i * hop_size
        end = start + fft_size
        if end <= n:
            frames_d[i] = x[start:end].astype(np.float64) * dwindow
    X_d = np.fft.rfft(frames_d, axis=1)

    mag = np.abs(X)
    phase = np.angle(X)

    # Reassigned frequency: shift based on time-derivative
    # The reassigned frequency for each bin
    eps = 1e-10
    reassign_shift = np.zeros((num_frames, num_bins), dtype=np.float64)
    for i in range(num_frames):
        for b in range(num_bins):
            if mag[i, b] > eps:
                # Frequency reassignment via time-ramped window
                reassign_shift[i, b] = -np.imag(X_d[i, b] / (X[i, b] + eps))

    # Build sharpened spectrogram by reassigning energy
    sharpened_mag = np.zeros_like(mag, dtype=np.float64)
    amount = np.clip(sharpening_amount, 0.0, 1.0)

    for i in range(num_frames):
        for b in range(num_bins):
            if mag[i, b] < eps:
                continue
            # Target bin after reassignment
            target_b = b + amount * reassign_shift[i, b] * fft_size / sr
            target_b_int = int(round(target_b))
            if 0 <= target_b_int < num_bins:
                sharpened_mag[i, target_b_int] += mag[i, b]

    # Blend with original phase
    Y = sharpened_mag.astype(np.float32) * np.exp(1j * phase)
    return _istft(Y, fft_size, hop_size, length=len(samples))


def variants_h017():
    return [
        {'sharpening_amount': 0.0},
        {'sharpening_amount': 0.2},
        {'sharpening_amount': 0.5},
        {'sharpening_amount': 0.8},
        {'sharpening_amount': 1.0},
    ]


# ---------------------------------------------------------------------------
# H018 -- Spectral Subtraction
# Estimate noise floor from quiet frames, subtract it. At extreme settings
# creates "musical noise" artifacts that are interesting as an effect.
# ---------------------------------------------------------------------------

def effect_h018_spectral_subtraction(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Spectral subtraction: estimate noise floor and subtract it.

    At moderate settings, acts as a denoiser. At extreme over-subtraction
    settings, creates "musical noise" artifacts — tonal, ghostly residuals.
    """
    subtraction_factor = params.get('subtraction_factor', 2.0)
    noise_percentile = params.get('noise_percentile', 10)
    fft_size = params.get('fft_size', 2048)
    hop_size = fft_size // 4

    X = _stft(samples, fft_size, hop_size)
    mag = np.abs(X)
    phase = np.angle(X)
    num_frames = X.shape[0]

    # Estimate noise floor: take magnitude percentile across frames per bin
    noise_floor = np.percentile(mag, noise_percentile, axis=0).astype(np.float32)

    # Subtract scaled noise floor
    sub = subtraction_factor
    cleaned_mag = np.maximum(mag - sub * noise_floor[np.newaxis, :], 0.0).astype(np.float32)

    Y = cleaned_mag * np.exp(1j * phase)
    return _istft(Y, fft_size, hop_size, length=len(samples))


def variants_h018():
    return [
        {'subtraction_factor': 1.0, 'noise_percentile': 10},    # gentle denoise
        {'subtraction_factor': 2.0, 'noise_percentile': 10},     # moderate subtraction
        {'subtraction_factor': 4.0, 'noise_percentile': 15},     # aggressive, musical noise appears
        {'subtraction_factor': 8.0, 'noise_percentile': 20},     # extreme: ghostly residuals
        {'subtraction_factor': 15.0, 'noise_percentile': 30},    # heavy: only strongest peaks survive
        {'subtraction_factor': 3.0, 'noise_percentile': 50},     # subtract median — unusual effect
    ]


# ---------------------------------------------------------------------------
# H019 -- Spectral Transfer / Timbre Stamp
# Extract spectral envelope of input via cepstral smoothing, apply to a
# synthetic carrier. Preserves phase coherence via LPC-style envelope
# rather than filter bank, giving a different character than vocoder (H010).
# ---------------------------------------------------------------------------

def effect_h019_spectral_transfer(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Spectral transfer: extract spectral envelope of input, impose it onto
    a synthetic carrier signal. Like a vocoder but preserves phase coherence."""
    carrier_type = params.get('carrier_type', 'noise')
    envelope_order = int(params.get('envelope_order', 30))
    blend = float(params.get('blend', 0.7))
    fft_size = params.get('fft_size', 2048)
    hop_size = fft_size // 4

    n = len(samples)

    # Generate carrier
    rng = np.random.default_rng(42)
    if carrier_type == 'noise':
        carrier = rng.standard_normal(n).astype(np.float32) * 0.5
    elif carrier_type == 'chirp':
        t = np.arange(n, dtype=np.float32) / sr
        duration = n / sr
        phase = 2.0 * np.pi * 80.0 * duration * (
            np.exp(t / duration * np.log(6000.0 / 80.0)) - 1.0
        ) / np.log(6000.0 / 80.0)
        carrier = (0.5 * np.sin(phase)).astype(np.float32)
    elif carrier_type == 'pulse':
        # Pulse train at 100 Hz
        carrier = np.zeros(n, dtype=np.float32)
        period = int(sr / 100.0)
        for i in range(0, n, period):
            carrier[i] = 1.0
    else:
        carrier = rng.standard_normal(n).astype(np.float32) * 0.5

    X_input = _stft(samples, fft_size, hop_size)
    X_carrier = _stft(carrier, fft_size, hop_size)

    num_frames, num_bins = X_input.shape
    Y = np.zeros_like(X_input)

    for i in range(num_frames):
        # Extract spectral envelope via cepstral smoothing
        log_mag = np.log(np.abs(X_input[i]) + 1e-10)
        cepstrum = np.fft.irfft(log_mag)
        liftered = np.zeros_like(cepstrum)
        cutoff = min(envelope_order, len(cepstrum))
        liftered[:cutoff] = cepstrum[:cutoff]
        envelope_log = np.fft.rfft(liftered).real
        envelope = np.exp(envelope_log).astype(np.float32)

        # Apply envelope to carrier magnitudes
        carrier_mag = np.abs(X_carrier[i])
        carrier_phase = np.angle(X_carrier[i])

        # Normalize carrier magnitude, then shape with input envelope
        carrier_peak = np.max(carrier_mag) + 1e-10
        shaped_mag = envelope * (carrier_mag / carrier_peak)

        # Blend with original
        orig_mag = np.abs(X_input[i])
        blended_mag = (1.0 - blend) * orig_mag + blend * shaped_mag

        # Use carrier phase for transferred part, input phase for original
        blended_phase = (1.0 - blend) * np.angle(X_input[i]) + blend * carrier_phase
        Y[i] = blended_mag.astype(np.float32) * np.exp(1j * blended_phase)

    return _istft(Y, fft_size, hop_size, length=n)


def variants_h019():
    return [
        {'carrier_type': 'noise', 'envelope_order': 20, 'blend': 0.5},
        {'carrier_type': 'noise', 'envelope_order': 40, 'blend': 0.8},
        {'carrier_type': 'noise', 'envelope_order': 10, 'blend': 1.0},
        {'carrier_type': 'chirp', 'envelope_order': 30, 'blend': 0.6},
        {'carrier_type': 'chirp', 'envelope_order': 30, 'blend': 1.0},
        {'carrier_type': 'pulse', 'envelope_order': 30, 'blend': 0.7},
        {'carrier_type': 'pulse', 'envelope_order': 15, 'blend': 1.0},
    ]
