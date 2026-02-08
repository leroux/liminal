"""Audio quality metrics for both pedals.

Computes output-only metrics (RT60, EDT, spectral centroid, echo density,
C50/C80, crest factor, octave-band RT60, spectral flatness, bandwidth) and
optional dry-vs-wet comparison metrics (centroid shift, energy ratio, THD+N,
bandwidth change).

Dependencies: numpy, scipy (already installed).
"""

import numpy as np
from scipy.signal import butter, sosfilt
from scipy.stats import linregress


def analyze(audio, sr=44100, reference=None):
    """Analyze audio quality metrics.

    Args:
        audio: Output audio, mono (N,) or stereo (N,2). Mixed to mono internally.
        sr: Sample rate.
        reference: Optional dry/input audio for comparison metrics.

    Returns dict with keys:
        rt60, edt, spectral_centroid, echo_density, c50, c80, crest_factor,
        rt60_bands, spectral_flatness, bandwidth

        When reference is provided, also:
        spectral_centroid_shift, energy_ratio_db, thd_n_percent, bandwidth_change
    """
    mono = _to_mono(audio)
    if len(mono) < 256:
        return {}

    result = {}
    rms = float(np.sqrt(np.mean(mono ** 2)))
    peak = float(np.max(np.abs(mono)))
    result["rms_db"] = round(20.0 * np.log10(rms), 1) if rms > 1e-12 else None
    result["peak_db"] = round(20.0 * np.log10(peak), 1) if peak > 1e-12 else None
    result["rt60"] = _find_rt60(mono, sr)
    result["edt"] = _find_edt(mono, sr)
    result["spectral_centroid"] = _spectral_centroid(mono, sr)
    result["echo_density"] = _echo_density(mono, sr)
    c50, c80 = _clarity(mono, sr)
    result["c50"] = c50
    result["c80"] = c80
    result["crest_factor"] = _crest_factor(mono)
    result["rt60_bands"] = _octave_band_rt60(mono, sr)
    result["spectral_flatness"] = _spectral_flatness(mono)
    result["bandwidth"] = _bandwidth(mono, sr)

    if reference is not None:
        ref_mono = _to_mono(reference)
        if len(ref_mono) >= 256:
            ref_centroid = _spectral_centroid(ref_mono, sr)
            result["spectral_centroid_shift"] = (
                result["spectral_centroid"] - ref_centroid
                if result["spectral_centroid"] is not None and ref_centroid is not None
                else None
            )

            ref_rms = np.sqrt(np.mean(ref_mono ** 2))
            wet_rms = np.sqrt(np.mean(mono ** 2))
            if ref_rms > 1e-12:
                result["energy_ratio_db"] = round(20.0 * np.log10(wet_rms / ref_rms), 2)
            else:
                result["energy_ratio_db"] = None

            # THD+N: RMS of difference / RMS of reference
            min_len = min(len(mono), len(ref_mono))
            diff = mono[:min_len] - ref_mono[:min_len]
            diff_rms = np.sqrt(np.mean(diff ** 2))
            if ref_rms > 1e-12:
                result["thd_n_percent"] = round(100.0 * diff_rms / ref_rms, 2)
            else:
                result["thd_n_percent"] = None

            ref_bw = _bandwidth(ref_mono, sr)
            if result["bandwidth"] is not None and ref_bw is not None:
                result["bandwidth_change"] = round(result["bandwidth"] - ref_bw, 1)
            else:
                result["bandwidth_change"] = None

    return result


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _to_mono(audio):
    """Convert to 1-D mono float64."""
    a = np.asarray(audio, dtype=np.float64)
    if a.ndim == 2:
        a = a.mean(axis=1)
    return a


def _schroeder_decay(mono):
    """Backward-integrated energy decay curve (dB), normalised to 0 dB start."""
    energy = mono ** 2
    # Cumulative sum from end to start
    decay = np.cumsum(energy[::-1])[::-1]
    decay = np.maximum(decay, 1e-30)
    db = 10.0 * np.log10(decay / decay[0])
    return db


def _find_rt60(mono, sr):
    """T30 extrapolation: fit line from -5 to -35 dB, extrapolate to -60 dB."""
    db = _schroeder_decay(mono)
    # Find indices where decay crosses -5 dB and -35 dB
    i5 = np.searchsorted(-db, 5)    # first index where db <= -5
    i35 = np.searchsorted(-db, 35)  # first index where db <= -35

    if i5 >= i35 or i35 >= len(db):
        return None

    x = np.arange(i5, i35, dtype=np.float64) / sr
    y = db[i5:i35]
    if len(x) < 3:
        return None

    slope, intercept, _, _, _ = linregress(x, y)
    if slope >= 0:
        return None

    # Time for decay to drop 60 dB from 0
    rt60 = -60.0 / slope
    return round(max(0.0, rt60), 3)


def _find_edt(mono, sr):
    """Early Decay Time: fit from 0 to -10 dB, extrapolate to -60 dB."""
    db = _schroeder_decay(mono)
    i0 = 0
    i10 = np.searchsorted(-db, 10)  # first index where db <= -10

    if i10 <= i0 or i10 >= len(db):
        return None

    x = np.arange(i0, i10, dtype=np.float64) / sr
    y = db[i0:i10]
    if len(x) < 3:
        return None

    slope, intercept, _, _, _ = linregress(x, y)
    if slope >= 0:
        return None

    edt = -60.0 / slope
    return round(max(0.0, edt), 3)


def _spectral_centroid(mono, sr):
    """Brightness: weighted mean of FFT frequencies (Hz)."""
    n = len(mono)
    windowed = mono * np.hanning(n)
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    total = np.sum(spectrum)
    if total < 1e-12:
        return None
    centroid = float(np.sum(freqs * spectrum) / total)
    return round(centroid, 1)


def _echo_density(mono, sr):
    """Normalised echo density (0-1) via sliding-window std-dev."""
    # Use 1ms windows
    win_samples = max(1, int(sr * 0.001))
    n = len(mono)
    if n < win_samples * 10:
        return None

    # Compute in chunks for efficiency
    n_windows = n // win_samples
    trimmed = mono[:n_windows * win_samples].reshape(n_windows, win_samples)
    stds = np.std(trimmed, axis=1)

    # Normalise: ratio of windows with significant energy vs total
    threshold = np.max(stds) * 0.05
    if threshold < 1e-12:
        return 0.0
    density = float(np.sum(stds > threshold) / len(stds))
    return round(density, 3)


def _clarity(mono, sr):
    """C50 and C80: early-to-late energy ratio in dB."""
    energy = mono ** 2

    n50 = min(int(sr * 0.050), len(energy))
    n80 = min(int(sr * 0.080), len(energy))

    early50 = np.sum(energy[:n50])
    late50 = np.sum(energy[n50:])
    early80 = np.sum(energy[:n80])
    late80 = np.sum(energy[n80:])

    c50 = round(10.0 * np.log10(early50 / late50), 2) if late50 > 1e-30 else None
    c80 = round(10.0 * np.log10(early80 / late80), 2) if late80 > 1e-30 else None
    return c50, c80


def _crest_factor(mono):
    """Peak-to-RMS ratio in dB."""
    rms = np.sqrt(np.mean(mono ** 2))
    peak = np.max(np.abs(mono))
    if rms < 1e-12:
        return None
    return round(20.0 * np.log10(peak / rms), 2)


def _octave_band_rt60(mono, sr):
    """RT60 at standard octave bands: 125, 250, 500, 1k, 2k, 4k, 8k Hz."""
    bands = [125, 250, 500, 1000, 2000, 4000, 8000]
    result = {}
    nyquist = sr / 2.0

    for fc in bands:
        lo = fc / np.sqrt(2)
        hi = fc * np.sqrt(2)
        # Skip bands that exceed Nyquist
        if hi >= nyquist * 0.95:
            continue
        try:
            sos = butter(4, [lo / nyquist, hi / nyquist], btype="band", output="sos")
            filtered = sosfilt(sos, mono)
            rt = _find_rt60(filtered, sr)
            result[str(fc)] = rt
        except Exception:
            result[str(fc)] = None

    return result


def _spectral_flatness(mono):
    """Wiener entropy: geometric/arithmetic mean of power spectrum. 0=tonal, 1=noise."""
    spectrum = np.abs(np.fft.rfft(mono)) ** 2
    spectrum = spectrum[1:]  # skip DC
    spectrum = np.maximum(spectrum, 1e-30)

    log_mean = np.mean(np.log(spectrum))
    geo_mean = np.exp(log_mean)
    arith_mean = np.mean(spectrum)

    if arith_mean < 1e-30:
        return 0.0
    flatness = float(geo_mean / arith_mean)
    return round(min(1.0, max(0.0, flatness)), 4)


def _bandwidth(mono, sr):
    """Frequency at -3dB from spectral peak (Hz)."""
    n = len(mono)
    spectrum = np.abs(np.fft.rfft(mono))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)

    if len(spectrum) < 2:
        return None

    peak_mag = np.max(spectrum)
    if peak_mag < 1e-12:
        return None

    threshold = peak_mag * (10 ** (-3.0 / 20.0))  # -3 dB below peak

    # Find highest frequency above threshold
    above = np.where(spectrum >= threshold)[0]
    if len(above) == 0:
        return None

    bw = float(freqs[above[-1]])
    return round(bw, 1)
