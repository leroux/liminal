"""Generate synthetic test input signals for the effects explorer."""
import numpy as np
import soundfile as sf
import os

SR = 44100


def click(sr=SR, duration=3.0):
    """Single sample impulse + silence. Shows raw impulse response."""
    n = int(sr * duration)
    signal = np.zeros(n, dtype=np.float32)
    signal[0] = 1.0
    return signal


def sparse_hits(sr=SR, duration=5.0):
    """4 percussive hits spaced ~1s apart with silence between.

    Mix of timbres: low thump, mid snap, high click, broad burst.
    Reveals tail character, transient handling, rhythmic interaction.
    """
    n = int(sr * duration)
    signal = np.zeros(n, dtype=np.float32)

    def add_hit(offset_s, freq, decay_ms, noise_mix=0.0, noise_decay_ms=5.0):
        start = int(offset_s * sr)
        length = int(0.3 * sr)
        t = np.arange(length, dtype=np.float32) / sr
        tone_env = np.exp(-t / (decay_ms * 0.001))
        tone = np.sin(2.0 * np.pi * freq * t) * tone_env
        if noise_mix > 0:
            noise_env = np.exp(-t / (noise_decay_ms * 0.001))
            noise = np.random.default_rng(42).standard_normal(length).astype(np.float32) * noise_env
            tone = (1.0 - noise_mix) * tone + noise_mix * noise
        end = min(start + length, n)
        signal[start:end] += tone[:end - start].astype(np.float32)

    # Low thump (kick-like)
    add_hit(0.5, 60, 80, noise_mix=0.1, noise_decay_ms=10)
    # Mid snap (snare-like)
    add_hit(1.5, 200, 30, noise_mix=0.6, noise_decay_ms=20)
    # High click (hat-like)
    add_hit(2.5, 800, 5, noise_mix=0.9, noise_decay_ms=8)
    # Broad burst (full-range transient)
    add_hit(3.5, 150, 50, noise_mix=0.4, noise_decay_ms=40)

    peak = np.max(np.abs(signal))
    if peak > 0:
        signal *= 0.9 / peak
    return signal


def sustained_chord(sr=SR, duration=5.0):
    """Rich sustained chord with harmonics. Simulates organ/pad.

    C major chord (C3, E3, G3) with 4 harmonics each, slight detune.
    Shows modulation, filtering, pitch effects on tonal material.
    """
    n = int(sr * duration)
    t = np.arange(n, dtype=np.float64) / sr
    signal = np.zeros(n, dtype=np.float64)

    # C3, E3, G3 with slight detune
    fundamentals = [130.81, 164.81, 196.00]
    detune_cents = [-3, 0, 2]

    for fund, cents in zip(fundamentals, detune_cents):
        freq = fund * (2.0 ** (cents / 1200.0))
        for harmonic in range(1, 5):
            amplitude = 1.0 / harmonic
            signal += amplitude * np.sin(2.0 * np.pi * freq * harmonic * t)

    # Gentle fade in/out
    fade_in = int(0.05 * sr)
    fade_out = int(0.3 * sr)
    signal[:fade_in] *= np.linspace(0, 1, fade_in)
    signal[-fade_out:] *= np.linspace(1, 0, fade_out)

    peak = np.max(np.abs(signal))
    if peak > 0:
        signal *= 0.9 / peak
    return signal.astype(np.float32)


def pink_noise_burst(sr=SR, duration=3.0, burst_duration=1.0):
    """1s of pink noise + silence. Broadband spectral diagnostic.

    Pink noise (1/f) via Voss-McCartney algorithm.
    Reveals frequency-dependent behavior without harmonic bias.
    """
    n = int(sr * duration)
    burst_n = int(sr * burst_duration)

    # Voss-McCartney pink noise
    rng = np.random.default_rng(123)
    num_rows = 16
    # Generate white noise rows updated at different rates
    rows = rng.standard_normal((burst_n, num_rows)).astype(np.float32)
    for i in range(1, num_rows):
        step = 2 ** i
        for j in range(burst_n):
            if j % step != 0:
                rows[j, i] = rows[j - 1, i]
    pink = np.sum(rows, axis=1)

    # Fade edges of burst to avoid clicks
    fade = int(0.005 * sr)
    pink[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
    pink[burst_n - fade:burst_n] *= np.linspace(1, 0, fade, dtype=np.float32)

    signal = np.zeros(n, dtype=np.float32)
    signal[:burst_n] = pink

    peak = np.max(np.abs(signal))
    if peak > 0:
        signal *= 0.9 / peak
    return signal


GENERATORS = {
    'click': click,
    'sparse_hits': sparse_hits,
    'sustained_chord': sustained_chord,
    'pink_noise_burst': pink_noise_burst,
}


def generate_all(output_dir='inputs'):
    os.makedirs(output_dir, exist_ok=True)
    for name, fn in GENERATORS.items():
        path = os.path.join(output_dir, f'{name}.wav')
        signal = fn()
        sf.write(path, signal, SR, format='WAV')
        dur = len(signal) / SR
        print(f"  {path} ({dur:.1f}s, {len(signal)} samples)")


if __name__ == '__main__':
    generate_all(os.path.join(os.path.dirname(__file__), 'inputs'))
