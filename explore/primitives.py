"""Shared DSP primitives and post-processing utilities."""
import numpy as np
import numba
from scipy.signal import butter, sosfilt


def dc_remove(signal):
    """Remove DC offset by subtracting mean."""
    return signal - np.mean(signal)


def normalize(signal, peak_db=-1.0):
    """Peak normalize to specified dB level."""
    peak = np.max(np.abs(signal))
    if peak < 1e-10:
        return signal
    target = 10.0 ** (peak_db / 20.0)
    return signal * (target / peak)


@numba.njit
def _soft_limit_kernel(signal, threshold):
    out = np.empty_like(signal)
    for i in range(len(signal)):
        x = signal[i]
        ax = abs(x)
        if ax <= threshold:
            out[i] = x
        else:
            out[i] = np.sign(x) * (threshold + np.tanh(ax - threshold) * (1.0 - threshold))
    return out


def soft_limit(signal, threshold=0.9):
    """Soft limiter using tanh curve above threshold."""
    return _soft_limit_kernel(signal.astype(np.float32), np.float32(threshold))


def gentle_lowpass(signal, sr, cutoff=16000):
    """2nd order Butterworth lowpass."""
    nyq = sr / 2.0
    if cutoff >= nyq:
        return signal
    sos = butter(2, cutoff / nyq, btype='low', output='sos')
    return sosfilt(sos, signal).astype(np.float32)


@numba.njit
def _fade_kernel(signal, fade_in, fade_out):
    out = signal.copy()
    n = len(out)
    for i in range(min(fade_in, n)):
        out[i] *= np.float32(i) / np.float32(fade_in)
    for i in range(min(fade_out, n)):
        out[n - 1 - i] *= np.float32(i) / np.float32(fade_out)
    return out


def fade_in_out(signal, fade_samples=256):
    """Linear fade at start and end to avoid clicks."""
    return _fade_kernel(signal.astype(np.float32), fade_samples, fade_samples)


def post_process(signal, sr):
    """Full post-processing chain: dc_remove -> gentle_lowpass -> soft_limit -> normalize -> fade_in_out"""
    s = dc_remove(signal.astype(np.float32))
    s = gentle_lowpass(s, sr)
    s = soft_limit(s)
    s = normalize(s, peak_db=-1.0)
    s = fade_in_out(s)
    return s


def post_process_stereo(signal, sr):
    """Post-process stereo signal (N,2)."""
    left = post_process(signal[:, 0], sr)
    right = post_process(signal[:, 1], sr)
    return np.column_stack([left, right])


def mix_wet_dry(dry, wet, mix=0.5):
    """Crossfade between dry and wet signal."""
    n = min(len(dry), len(wet))
    return (1.0 - mix) * dry[:n] + mix * wet[:n]


def stft(x, fft_size=2048, hop_size=512):
    """Forward STFT. Returns complex array of shape (num_frames, fft_size//2+1)."""
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


def istft(X, fft_size=2048, hop_size=512, length=None):
    """Inverse STFT with overlap-add."""
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


@numba.njit
def biquad_filter(samples, b0, b1, b2, a1, a2):
    """Direct Form II Transposed biquad filter."""
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


def biquad_coeffs_lpf(freq_hz, sr, Q=0.707):
    """Compute biquad coefficients for lowpass filter."""
    w0 = 2.0 * np.pi * freq_hz / sr
    alpha = np.sin(w0) / (2.0 * Q)
    b0 = (1.0 - np.cos(w0)) / 2.0
    b1 = 1.0 - np.cos(w0)
    b2 = (1.0 - np.cos(w0)) / 2.0
    a0 = 1.0 + alpha
    a1 = -2.0 * np.cos(w0)
    a2 = 1.0 - alpha
    return np.float32(b0/a0), np.float32(b1/a0), np.float32(b2/a0), np.float32(a1/a0), np.float32(a2/a0)


def biquad_coeffs_hpf(freq_hz, sr, Q=0.707):
    """Compute biquad coefficients for highpass filter."""
    w0 = 2.0 * np.pi * freq_hz / sr
    alpha = np.sin(w0) / (2.0 * Q)
    b0 = (1.0 + np.cos(w0)) / 2.0
    b1 = -(1.0 + np.cos(w0))
    b2 = (1.0 + np.cos(w0)) / 2.0
    a0 = 1.0 + alpha
    a1 = -2.0 * np.cos(w0)
    a2 = 1.0 - alpha
    return np.float32(b0/a0), np.float32(b1/a0), np.float32(b2/a0), np.float32(a1/a0), np.float32(a2/a0)


def biquad_coeffs_bpf(freq_hz, sr, Q=1.0):
    """Compute biquad coefficients for bandpass filter."""
    w0 = 2.0 * np.pi * freq_hz / sr
    alpha = np.sin(w0) / (2.0 * Q)
    b0 = alpha
    b1 = 0.0
    b2 = -alpha
    a0 = 1.0 + alpha
    a1 = -2.0 * np.cos(w0)
    a2 = 1.0 - alpha
    return np.float32(b0/a0), np.float32(b1/a0), np.float32(b2/a0), np.float32(a1/a0), np.float32(a2/a0)


def biquad_coeffs_notch(freq_hz, sr, Q=1.0):
    """Compute biquad coefficients for notch filter."""
    w0 = 2.0 * np.pi * freq_hz / sr
    alpha = np.sin(w0) / (2.0 * Q)
    b0 = 1.0
    b1 = -2.0 * np.cos(w0)
    b2 = 1.0
    a0 = 1.0 + alpha
    a1 = -2.0 * np.cos(w0)
    a2 = 1.0 - alpha
    return np.float32(b0/a0), np.float32(b1/a0), np.float32(b2/a0), np.float32(a1/a0), np.float32(a2/a0)


def biquad_coeffs_peak(freq_hz, sr, Q=1.0, gain_db=0.0):
    """Compute biquad coefficients for peaking EQ filter."""
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * freq_hz / sr
    alpha = np.sin(w0) / (2.0 * Q)
    b0 = 1.0 + alpha * A
    b1 = -2.0 * np.cos(w0)
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * np.cos(w0)
    a2 = 1.0 - alpha / A
    return np.float32(b0/a0), np.float32(b1/a0), np.float32(b2/a0), np.float32(a1/a0), np.float32(a2/a0)


@numba.njit
def one_pole_lp(samples, coeff):
    """One-pole lowpass filter. coeff in [0, 1), higher = more smoothing."""
    n = len(samples)
    out = np.zeros(n, dtype=np.float32)
    prev = np.float32(0.0)
    for i in range(n):
        prev = coeff * prev + (np.float32(1.0) - coeff) * samples[i]
        out[i] = prev
    return out


@numba.njit
def envelope_follower(samples, attack_coeff, release_coeff):
    """Envelope follower with separate attack and release."""
    n = len(samples)
    env = np.zeros(n, dtype=np.float32)
    prev = np.float32(0.0)
    for i in range(n):
        inp = abs(samples[i])
        if inp > prev:
            prev = attack_coeff * prev + (np.float32(1.0) - attack_coeff) * inp
        else:
            prev = release_coeff * prev + (np.float32(1.0) - release_coeff) * inp
        env[i] = prev
    return env


def ms_to_coeff(ms, sr):
    """Convert milliseconds to one-pole filter coefficient."""
    if ms <= 0:
        return np.float32(0.0)
    return np.float32(np.exp(-1.0 / (ms * 0.001 * sr)))
