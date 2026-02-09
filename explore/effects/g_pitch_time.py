"""G-series: Pitch & Time effects (G001-G007)."""
import numpy as np
import numba


# ---------------------------------------------------------------------------
# Shared STFT / ISTFT utilities
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
# G001 — Phase Vocoder Pitch Shift
# ---------------------------------------------------------------------------

def effect_g001_phase_vocoder_pitch_shift(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Phase vocoder pitch shift with proper phase accumulation.

    Shifts frequency bins by a semitone ratio, applying phase correction
    to maintain coherent synthesis.
    """
    semitones = float(params.get('semitones', 7))
    fft_size = int(params.get('fft_size', 2048))
    hop_size = int(params.get('hop_size', 512))

    ratio = 2.0 ** (semitones / 12.0)
    x = samples.astype(np.float32)
    n = len(x)

    # Analysis
    X = _stft(x, fft_size, hop_size)
    num_frames = X.shape[0]
    num_bins = X.shape[1]  # fft_size // 2 + 1

    magnitudes = np.abs(X)
    phases = np.angle(X)

    # Phase vocoder with proper phase accumulation
    synth_magnitudes = np.zeros_like(magnitudes)
    synth_phases = np.zeros_like(phases)

    # Previous analysis and synthesis phases
    prev_phase = np.zeros(num_bins, dtype=np.float64)
    prev_synth_phase = np.zeros(num_bins, dtype=np.float64)

    two_pi = 2.0 * np.pi

    for frame_idx in range(num_frames):
        for bin_idx in range(num_bins):
            # Expected phase advance for this bin
            expected_phase = prev_phase[bin_idx] + two_pi * hop_size * bin_idx / fft_size
            # Phase deviation (instantaneous frequency deviation)
            deviation = phases[frame_idx, bin_idx] - expected_phase
            # Wrap deviation to [-pi, pi]
            deviation = deviation - two_pi * np.round(deviation / two_pi)
            # True frequency for this bin
            true_freq = (two_pi * bin_idx / fft_size) + deviation / hop_size

            # Map this bin to the shifted output bin
            new_bin = int(round(bin_idx * ratio))
            if 0 <= new_bin < num_bins:
                synth_magnitudes[frame_idx, new_bin] += magnitudes[frame_idx, bin_idx]
                # Synthesis phase: accumulate using shifted frequency
                synth_phases[frame_idx, new_bin] = prev_synth_phase[new_bin] + hop_size * true_freq * ratio

            prev_phase[bin_idx] = phases[frame_idx, bin_idx]

        # Update previous synthesis phases for all bins
        for bin_idx in range(num_bins):
            prev_synth_phase[bin_idx] = synth_phases[frame_idx, bin_idx]

    # Reconstruct complex spectrogram
    Y = synth_magnitudes * np.exp(1j * synth_phases)

    # Synthesis
    output = _istft(Y, fft_size, hop_size, length=n)

    # Normalize to avoid clipping
    peak = np.max(np.abs(output))
    if peak > 0.0:
        input_peak = np.max(np.abs(x))
        if input_peak > 0.0:
            output *= input_peak / peak
    return output


def variants_g001():
    return [
        {'semitones': 7, 'fft_size': 2048, 'hop_size': 512},     # perfect fifth up
        {'semitones': 12, 'fft_size': 2048, 'hop_size': 512},    # octave up
        {'semitones': -12, 'fft_size': 2048, 'hop_size': 512},   # octave down
        {'semitones': -5, 'fft_size': 2048, 'hop_size': 512},    # fourth down
        {'semitones': 3, 'fft_size': 4096, 'hop_size': 1024},    # minor third up, larger window
        {'semitones': -24, 'fft_size': 2048, 'hop_size': 256},   # two octaves down, fine hop
        {'semitones': 1, 'fft_size': 2048, 'hop_size': 512},     # subtle half-step up (detuned)
    ]


# ---------------------------------------------------------------------------
# G002 — Granular Pitch Shift
# ---------------------------------------------------------------------------

@numba.njit
def _granular_pitch_shift_kernel(samples, ratio, grain_size, hop):
    """Granular pitch shift: read grains at resampled rate, overlap-add."""
    n = len(samples)
    out = np.zeros(n, dtype=np.float32)

    # Hanning window for grain (pre-computed)
    window = np.zeros(grain_size, dtype=np.float32)
    for i in range(grain_size):
        window[i] = np.float32(0.5 * (1.0 - np.cos(2.0 * np.pi * i / (grain_size - 1))))

    pos = 0
    while pos < n:
        for i in range(grain_size):
            # Read position in original signal, resampled by ratio
            read_pos = pos + i * ratio
            read_idx = int(read_pos)
            frac = np.float32(read_pos - read_idx)
            if read_idx < 0 or read_idx >= n - 1:
                val = np.float32(0.0)
            else:
                # Linear interpolation
                val = samples[read_idx] * (1.0 - frac) + samples[read_idx + 1] * frac

            out_idx = pos + i
            if 0 <= out_idx < n:
                out[out_idx] += val * window[i]

        pos += hop
    return out


def effect_g002_granular_pitch_shift(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Granular pitch shift: extracts grains and replays them at resampled speed."""
    semitones = float(params.get('semitones', -5))
    grain_size_ms = float(params.get('grain_size_ms', 50))

    ratio = 2.0 ** (semitones / 12.0)
    grain_size = max(4, int(grain_size_ms * sr / 1000.0))
    hop = grain_size // 2  # 50% overlap

    x = samples.astype(np.float32)
    output = _granular_pitch_shift_kernel(x, np.float32(ratio), grain_size, hop)

    # Normalize
    peak = np.max(np.abs(output))
    if peak > 0.0:
        input_peak = np.max(np.abs(x))
        if input_peak > 0.0:
            output *= input_peak / peak
    return output


def variants_g002():
    return [
        {'semitones': -5, 'grain_size_ms': 50},    # fourth down, medium grains
        {'semitones': 7, 'grain_size_ms': 30},      # fifth up, small grains (more artifacts)
        {'semitones': -12, 'grain_size_ms': 80},    # octave down, large grains (smoother)
        {'semitones': 12, 'grain_size_ms': 40},     # octave up
        {'semitones': -3, 'grain_size_ms': 100},    # minor third down, very smooth
        {'semitones': 5, 'grain_size_ms': 20},      # fourth up, tiny grains (glitchy)
    ]


# ---------------------------------------------------------------------------
# G003 — Harmonizer
# ---------------------------------------------------------------------------

def effect_g003_harmonizer(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Mix original signal with pitch-shifted copies at specified intervals."""
    intervals = params.get('intervals_semitones', [7, 12])
    wet_mix = float(params.get('wet_mix', 0.5))

    x = samples.astype(np.float32)
    n = len(x)
    fft_size = 2048
    hop_size = 512

    # Start with dry signal
    output = x * (1.0 - wet_mix)

    # Add each pitch-shifted voice
    voice_gain = wet_mix / max(1, len(intervals))
    for semitones in intervals:
        shifted = effect_g001_phase_vocoder_pitch_shift(
            x, sr, semitones=semitones, fft_size=fft_size, hop_size=hop_size
        )
        output += shifted * voice_gain

    # Normalize
    peak = np.max(np.abs(output))
    if peak > 1.0:
        output /= peak
    return output


def variants_g003():
    return [
        {'intervals_semitones': [7], 'wet_mix': 0.5},                  # parallel fifth
        {'intervals_semitones': [12], 'wet_mix': 0.4},                 # octave doubler
        {'intervals_semitones': [7, 12], 'wet_mix': 0.5},             # fifth + octave (power chord)
        {'intervals_semitones': [4, 7], 'wet_mix': 0.5},              # major triad
        {'intervals_semitones': [3, 7], 'wet_mix': 0.5},              # minor triad
        {'intervals_semitones': [-12, 12], 'wet_mix': 0.6},           # sub-octave + octave up
        {'intervals_semitones': [5, 7, 12], 'wet_mix': 0.6},         # fourth + fifth + octave
        {'intervals_semitones': [7], 'wet_mix': 0.8},                  # heavy wet fifth, shimmery
    ]


# ---------------------------------------------------------------------------
# G004 — Octave Up
# ---------------------------------------------------------------------------

@numba.njit
def _fullwave_rectify(samples):
    """Full-wave rectification doubles the frequency (octave up)."""
    n = len(samples)
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        out[i] = np.float32(abs(samples[i]))
    return out


@numba.njit
def _biquad_bandpass(samples, b0, b1, b2, a1, a2):
    """Second-order IIR bandpass filter."""
    n = len(samples)
    out = np.empty(n, dtype=np.float32)
    x1 = np.float32(0.0)
    x2 = np.float32(0.0)
    y1 = np.float32(0.0)
    y2 = np.float32(0.0)
    for i in range(n):
        x0 = samples[i]
        y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        out[i] = np.float32(y0)
        x2 = x1
        x1 = x0
        y2 = y1
        y1 = np.float32(y0)
    return out


def effect_g004_octave_up(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Octave up via full-wave rectification followed by bandpass filtering.

    The rectification creates an octave-up component. The bandpass filter
    centered at the fundamental's octave isolates the desired harmonic.
    """
    fundamental_hz = float(params.get('fundamental_hz', 220))
    filter_Q = float(params.get('filter_Q', 5.0))
    wet_mix = float(params.get('wet_mix', 0.6))

    x = samples.astype(np.float32)

    # Full-wave rectification
    rectified = _fullwave_rectify(x)

    # Remove DC offset from rectification
    rectified -= np.mean(rectified)

    # Bandpass filter centered at 2 * fundamental (the octave)
    target_freq = fundamental_hz * 2.0
    omega = 2.0 * np.pi * target_freq / sr
    sin_w = np.sin(omega)
    cos_w = np.cos(omega)
    alpha = sin_w / (2.0 * filter_Q)

    # Bandpass coefficients (constant-0dB-peak-gain)
    b0 = np.float32(alpha)
    b1 = np.float32(0.0)
    b2 = np.float32(-alpha)
    a0 = 1.0 + alpha
    a1 = np.float32(-2.0 * cos_w / a0)
    a2 = np.float32((1.0 - alpha) / a0)
    b0 /= np.float32(a0)
    b2 /= np.float32(a0)

    filtered = _biquad_bandpass(rectified, b0, b1, b2, a1, a2)

    # Normalize filtered to match input level
    peak_f = np.max(np.abs(filtered))
    peak_x = np.max(np.abs(x))
    if peak_f > 0.0 and peak_x > 0.0:
        filtered *= peak_x / peak_f

    # Mix
    output = x * (1.0 - wet_mix) + filtered * wet_mix
    return output


def variants_g004():
    return [
        {'fundamental_hz': 220, 'filter_Q': 5.0, 'wet_mix': 0.6},    # guitar A3 fundamental
        {'fundamental_hz': 110, 'filter_Q': 4.0, 'wet_mix': 0.5},    # bass A2 fundamental
        {'fundamental_hz': 330, 'filter_Q': 6.0, 'wet_mix': 0.7},    # higher voice, tight filter
        {'fundamental_hz': 440, 'filter_Q': 8.0, 'wet_mix': 0.5},    # A4, very narrow filter
        {'fundamental_hz': 220, 'filter_Q': 2.0, 'wet_mix': 0.8},    # wide filter, more harmonics bleed
        {'fundamental_hz': 150, 'filter_Q': 10.0, 'wet_mix': 0.4},   # narrow isolation, subtle
    ]


# ---------------------------------------------------------------------------
# G005 — Time Stretch (WSOLA)
# ---------------------------------------------------------------------------

@numba.njit
def _wsola_cross_corr_search(x, target_pos, prev_end, window_size, search_range, n):
    """Find the best alignment offset using cross-correlation."""
    best_offset = 0
    best_corr = np.float32(-1e30)

    # The segment that ended previous grain (for cross-correlation matching)
    ref_start = prev_end - window_size // 4
    ref_len = window_size // 4
    if ref_start < 0:
        ref_start = 0
    if ref_start + ref_len > n:
        ref_len = n - ref_start
    if ref_len <= 0:
        return 0

    for offset in range(-search_range, search_range + 1):
        cand_start = target_pos + offset
        if cand_start < 0 or cand_start + ref_len > n:
            continue
        # Cross-correlation
        corr = np.float32(0.0)
        for j in range(ref_len):
            corr += x[ref_start + j] * x[cand_start + j]
        if corr > best_corr:
            best_corr = corr
            best_offset = offset

    return best_offset


@numba.njit
def _wsola_kernel(x, stretch_factor, window_size, n):
    """WSOLA time stretching kernel."""
    hop_in = window_size // 2
    hop_out = int(hop_in * stretch_factor)
    out_len = int(n * stretch_factor)
    output = np.zeros(out_len, dtype=np.float32)

    # Hanning window
    window = np.zeros(window_size, dtype=np.float32)
    for i in range(window_size):
        window[i] = np.float32(0.5 * (1.0 - np.cos(2.0 * np.pi * i / (window_size - 1))))

    search_range = hop_in // 4
    read_pos = 0
    write_pos = 0
    prev_end = 0

    while write_pos + window_size <= out_len and read_pos + window_size <= n:
        # Find best alignment near expected read position
        if write_pos > 0:
            offset = _wsola_cross_corr_search(x, read_pos, prev_end, window_size, search_range, n)
            aligned_pos = read_pos + offset
        else:
            aligned_pos = read_pos

        if aligned_pos < 0:
            aligned_pos = 0
        if aligned_pos + window_size > n:
            aligned_pos = n - window_size
            if aligned_pos < 0:
                break

        # Overlap-add
        for i in range(window_size):
            if write_pos + i < out_len:
                output[write_pos + i] += x[aligned_pos + i] * window[i]

        prev_end = aligned_pos + window_size
        write_pos += hop_out
        read_pos += hop_in

    return output


def effect_g005_time_stretch_wsola(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """WSOLA (Waveform Similarity Overlap-Add) time stretching.

    Stretches or compresses the signal in time without changing pitch.
    Uses cross-correlation to find optimal overlap alignment.
    """
    stretch_factor = float(params.get('stretch_factor', 1.5))
    window_ms = float(params.get('window_ms', 40))

    x = samples.astype(np.float32)
    n = len(x)
    window_size = max(4, int(window_ms * sr / 1000.0))
    # Ensure window_size is even
    if window_size % 2 != 0:
        window_size += 1

    output = _wsola_kernel(x, np.float32(stretch_factor), window_size, n)

    # Normalize
    peak = np.max(np.abs(output))
    if peak > 0.0:
        input_peak = np.max(np.abs(x))
        if input_peak > 0.0:
            output *= input_peak / peak
    return output


def variants_g005():
    return [
        {'stretch_factor': 1.5, 'window_ms': 40},    # moderate stretch, standard window
        {'stretch_factor': 2.0, 'window_ms': 50},     # double length
        {'stretch_factor': 0.5, 'window_ms': 30},     # half speed (compress time)
        {'stretch_factor': 0.75, 'window_ms': 40},    # slight speedup
        {'stretch_factor': 3.0, 'window_ms': 60},     # extreme stretch
        {'stretch_factor': 1.25, 'window_ms': 80},    # subtle stretch, large window (smooth)
        {'stretch_factor': 4.0, 'window_ms': 20},     # extreme stretch, small window (grainy)
    ]


# ---------------------------------------------------------------------------
# G006 — Paulstretch
# ---------------------------------------------------------------------------

def effect_g006_paulstretch(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Paulstretch extreme time stretching.

    Randomizes phases of each STFT frame while preserving magnitudes,
    producing a smooth, ambient stretch. Output length = input_length * stretch_factor.
    """
    stretch_factor = float(params.get('stretch_factor', 8.0))
    window_size = int(params.get('window_size', 4096))

    x = samples.astype(np.float32)
    n = len(x)

    # Pad input if shorter than window
    if n < window_size:
        x = np.concatenate([x, np.zeros(window_size - n, dtype=np.float32)])
        n = len(x)

    out_length = int(n * stretch_factor)
    hop_in = window_size // 4
    hop_out = int(hop_in * stretch_factor)

    # Number of input frames
    num_frames = max(1, 1 + (n - window_size) // hop_in)

    # Build window
    window = np.hanning(window_size).astype(np.float32)

    # Output buffer
    output = np.zeros(out_length, dtype=np.float32)

    rng = np.random.RandomState(42)  # deterministic for reproducibility

    for frame_idx in range(num_frames):
        # Extract frame
        start = frame_idx * hop_in
        end = start + window_size
        if end > n:
            # Pad last frame
            frame = np.zeros(window_size, dtype=np.float32)
            valid = n - start
            frame[:valid] = x[start:start + valid]
        else:
            frame = x[start:end].copy()

        frame *= window

        # FFT
        spectrum = np.fft.rfft(frame)
        magnitudes = np.abs(spectrum)

        # Randomize phases
        random_phases = rng.uniform(-np.pi, np.pi, len(magnitudes)).astype(np.float64)
        new_spectrum = magnitudes * np.exp(1j * random_phases)

        # IFFT
        grain = np.fft.irfft(new_spectrum, n=window_size).astype(np.float32)
        grain *= window

        # Place grain in output
        out_start = int(frame_idx * hop_out)
        out_end = min(out_start + window_size, out_length)
        valid_len = out_end - out_start
        if valid_len > 0:
            output[out_start:out_end] += grain[:valid_len]

    # Normalize
    peak = np.max(np.abs(output))
    if peak > 0.0:
        input_peak = np.max(np.abs(x))
        if input_peak > 0.0:
            output *= input_peak / peak

    return output


def variants_g006():
    return [
        {'stretch_factor': 8.0, 'window_size': 4096},      # classic paulstretch, ambient wash
        {'stretch_factor': 2.0, 'window_size': 2048},       # mild stretch, still recognizable
        {'stretch_factor': 20.0, 'window_size': 8192},      # extreme stretch, glacial drone
        {'stretch_factor': 50.0, 'window_size': 16384},     # ultra stretch, pure texture
        {'stretch_factor': 100.0, 'window_size': 65536},    # maximal stretch, frozen sound
        {'stretch_factor': 4.0, 'window_size': 2048},       # moderate stretch, smaller window
        {'stretch_factor': 10.0, 'window_size': 32768},     # large window, very smooth
    ]


# ---------------------------------------------------------------------------
# G007 — Formant-Preserving Pitch Shift
# ---------------------------------------------------------------------------

def _cepstral_envelope(spectrum, lpc_order):
    """Extract spectral envelope via cepstral method.

    Takes log-magnitude spectrum, computes cepstrum, windows it to keep
    only the low-quefrency components (spectral envelope), transforms back.
    """
    log_mag = np.log(np.maximum(np.abs(spectrum), 1e-10))

    # Real cepstrum: IFFT of log magnitude
    cepstrum = np.fft.irfft(log_mag)

    # Lifter: keep only first lpc_order coefficients (spectral envelope)
    liftered = np.zeros_like(cepstrum)
    order = min(lpc_order, len(cepstrum) // 2)
    liftered[0] = cepstrum[0]
    liftered[1:order] = cepstrum[1:order] * 2.0  # double for one-sided
    # DC stays as-is, Nyquist if present
    if len(cepstrum) % 2 == 0 and order >= len(cepstrum) // 2:
        liftered[len(cepstrum) // 2] = cepstrum[len(cepstrum) // 2]

    # Transform back to get spectral envelope
    envelope = np.exp(np.fft.rfft(liftered, n=len(cepstrum)).real)

    return envelope


def effect_g007_formant_preserving_pitch_shift(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Formant-preserving pitch shift using cepstral envelope separation.

    1. Extract spectral envelope (formants) via cepstral analysis
    2. Pitch shift the fine spectral detail (excitation)
    3. Reapply original spectral envelope to preserve formant structure

    This avoids the "chipmunk" effect of naive pitch shifting.
    """
    semitones = float(params.get('semitones', 5))
    lpc_order = int(params.get('lpc_order', 30))
    fft_size = 2048
    hop_size = 512

    ratio = 2.0 ** (semitones / 12.0)
    x = samples.astype(np.float32)
    n = len(x)

    # Analysis STFT
    X = _stft(x, fft_size, hop_size)
    num_frames = X.shape[0]
    num_bins = X.shape[1]

    magnitudes = np.abs(X)
    phases = np.angle(X)

    # For each frame: separate envelope from fine structure, shift fine, reapply envelope
    Y_mag = np.zeros_like(magnitudes)
    Y_phase = np.zeros_like(phases)

    prev_phase = np.zeros(num_bins, dtype=np.float64)
    prev_synth_phase = np.zeros(num_bins, dtype=np.float64)
    two_pi = 2.0 * np.pi

    for frame_idx in range(num_frames):
        frame_spectrum = X[frame_idx]

        # Extract spectral envelope
        envelope = _cepstral_envelope(frame_spectrum, lpc_order)
        # Ensure envelope matches num_bins
        if len(envelope) > num_bins:
            envelope = envelope[:num_bins]
        elif len(envelope) < num_bins:
            envelope = np.concatenate([envelope, np.ones(num_bins - len(envelope))])

        # Fine structure = magnitude / envelope
        fine_structure = magnitudes[frame_idx] / np.maximum(envelope, 1e-10)

        # Pitch shift the fine structure (shift bins)
        shifted_fine = np.zeros(num_bins, dtype=np.float64)
        for bin_idx in range(num_bins):
            new_bin = int(round(bin_idx * ratio))
            if 0 <= new_bin < num_bins:
                shifted_fine[new_bin] += fine_structure[bin_idx]

        # Reapply original envelope to shifted fine structure
        Y_mag[frame_idx] = shifted_fine * envelope

        # Phase vocoder with proper accumulation for shifted bins
        for bin_idx in range(num_bins):
            expected_phase = prev_phase[bin_idx] + two_pi * hop_size * bin_idx / fft_size
            deviation = phases[frame_idx, bin_idx] - expected_phase
            deviation = deviation - two_pi * np.round(deviation / two_pi)
            true_freq = (two_pi * bin_idx / fft_size) + deviation / hop_size

            new_bin = int(round(bin_idx * ratio))
            if 0 <= new_bin < num_bins:
                Y_phase[frame_idx, new_bin] = prev_synth_phase[new_bin] + hop_size * true_freq * ratio

            prev_phase[bin_idx] = phases[frame_idx, bin_idx]

        for bin_idx in range(num_bins):
            prev_synth_phase[bin_idx] = Y_phase[frame_idx, bin_idx]

    # Reconstruct
    Y = Y_mag * np.exp(1j * Y_phase)
    output = _istft(Y, fft_size, hop_size, length=n)

    # Normalize
    peak = np.max(np.abs(output))
    if peak > 0.0:
        input_peak = np.max(np.abs(x))
        if input_peak > 0.0:
            output *= input_peak / peak
    return output


def variants_g007():
    return [
        {'semitones': 5, 'lpc_order': 30},     # fourth up, natural voice preservation
        {'semitones': -5, 'lpc_order': 30},    # fourth down, retains formants
        {'semitones': 12, 'lpc_order': 20},    # octave up without chipmunk effect
        {'semitones': -12, 'lpc_order': 20},   # octave down without muddiness
        {'semitones': 7, 'lpc_order': 40},     # fifth up, high-order envelope (precise formants)
        {'semitones': -3, 'lpc_order': 10},    # minor third down, coarse envelope (colored)
        {'semitones': 2, 'lpc_order': 30},     # whole step up, subtle pitch correction feel
    ]
