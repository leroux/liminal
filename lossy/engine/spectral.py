"""STFT-based spectral degradation engine.

Core algorithm: window -> FFT -> degrade spectral content -> IFFT -> overlap-add.

Three modes:
  Standard -- spectral quantization + psychoacoustic band gating
  Inverse  -- output the spectral residual (everything Standard discards)
  Jitter   -- phase perturbation (digital clock inaccuracy emulation)

The "underwater" warbling comes from zeroing *different* spectral bands each
frame, mimicking how perceptual codecs discard different sub-threshold
components frame-to-frame.  Band gating is weighted by psychoacoustic masking
(ATH curve + signal energy) so that quieter bands at less-sensitive frequencies
are more likely to be dropped -- matching real codec behaviour.

Additional features beyond basic STFT:
  Phase Loss   -- quantize phase angles for metallic/robotic character
  Compand      -- power-law (|x|^0.75) quantizer matching MP3's nonuniform step
  Pre-Echo     -- enhanced transient smearing by boosting loss before attacks
  Noise Shape  -- envelope-following quantization (coarser in spectral valleys)
  Freezer      -- blend between frozen spectrum and live signal
"""

import numpy as np
from lossy.engine.params import SR


def spectral_process(input_audio, params):
    """Run STFT spectral degradation on input audio.

    Args:
        input_audio: mono float64 array
        params: parameter dict (see engine/params.py)

    Returns:
        processed mono float64 array, same length as input
    """
    g = float(params.get("global_amount", 1.0))
    loss = float(params["loss"]) * g
    mode = int(params["mode"])
    seed = int(params.get("seed", 42))
    freeze = int(params.get("freeze", 0))
    freeze_mode = int(params.get("freeze_mode", 0))
    freezer_blend = float(params.get("freezer", 1.0))
    phase_loss = float(params.get("phase_loss", 0.0)) * g
    quantizer_type = int(params.get("quantizer", 0))
    pre_echo_amount = float(params.get("pre_echo", 0.0)) * g
    noise_shape = float(params.get("noise_shape", 0.0))
    weighting = float(params.get("weighting", 1.0))
    hf_threshold = float(params.get("hf_threshold", 0.3))
    transient_ratio = float(params.get("transient_ratio", 4.0))
    slushy_rate_param = float(params.get("slushy_rate", 0.03))

    if loss <= 0.0 and freeze == 0 and phase_loss <= 0.0:
        return input_audio.copy()

    # --- Window and hop from params ---
    window_size = max(2, int(params.get("window_size", 2048)))
    hop_divisor = max(1, int(params.get("hop_divisor", 4)))
    hop_size = max(1, window_size // hop_divisor)
    n_bins = window_size // 2 + 1
    n_bands = max(2, int(params.get("n_bands", 21)))

    window = np.hanning(window_size).astype(np.float64)

    # Pad input so edges are handled cleanly.
    # Use edge reflection instead of zeros so STFT frames near boundaries
    # see realistic signal content (avoids fade-in/fade-out artifacts).
    pad = window_size
    if len(input_audio) > pad:
        padded = np.pad(input_audio, pad, mode='reflect')
    else:
        padded = np.concatenate([np.zeros(pad), input_audio, np.zeros(pad)])
    n_samples = len(padded)

    output = np.zeros(n_samples, dtype=np.float64)
    win_sum = np.zeros(n_samples, dtype=np.float64)

    n_frames = (n_samples - window_size) // hop_size + 1
    rng = np.random.RandomState(seed)

    # ~N Bark-like bands (log-spaced edges), mimicking scalefactor bands
    band_edges = np.unique(
        np.clip(
            np.logspace(np.log10(1), np.log10(n_bins), n_bands + 1).astype(int),
            0,
            n_bins,
        )
    )
    n_bands = len(band_edges) - 1

    # Pre-compute ATH weighting per band
    ath_weights = _compute_ath_weights(band_edges, n_bands, n_bins, window_size)

    # --- Pre-echo detection pass ---
    transient_flags = None
    if pre_echo_amount > 0 and n_frames > 1:
        energies = np.zeros(n_frames)
        for fi in range(n_frames):
            start = fi * hop_size
            frame = padded[start : start + window_size] * window
            energies[fi] = np.sum(frame ** 2)
        transient_flags = np.zeros(n_frames, dtype=np.bool_)
        for fi in range(1, n_frames):
            if energies[fi - 1] > 1e-12:
                if energies[fi] / energies[fi - 1] > transient_ratio:
                    transient_flags[fi] = True

    frozen_spectrum = None

    for fi in range(n_frames):
        start = fi * hop_size
        frame = padded[start : start + window_size] * window

        spectrum = np.fft.rfft(frame)
        magnitudes = np.abs(spectrum)
        phases = np.angle(spectrum)

        # Pre-echo: boost loss for frames right before a transient
        frame_loss = loss
        if transient_flags is not None and fi < n_frames - 1:
            if transient_flags[fi + 1]:
                frame_loss = min(1.0, loss + pre_echo_amount * 0.5)

        # ---------- mode processing ----------
        if mode == 0:  # Standard
            proc_mag = _standard(
                magnitudes, frame_loss, rng, band_edges, n_bands,
                ath_weights, quantizer_type, noise_shape, weighting,
            )
            proc_phase = phases
            # Phase quantization
            if phase_loss > 0:
                n_levels = max(4, int(64 * (1.0 - phase_loss)))
                step = 2.0 * np.pi / n_levels
                proc_phase = step * np.round(proc_phase / step)

        elif mode == 1:  # Inverse
            std_mag = _standard(
                magnitudes, frame_loss, rng, band_edges, n_bands,
                ath_weights, quantizer_type, noise_shape, weighting,
            )
            proc_mag = np.maximum(magnitudes - std_mag, 0.0)
            proc_phase = phases

        else:  # Jitter
            proc_mag = magnitudes.copy()
            noise = rng.uniform(-np.pi, np.pi, len(phases)) * frame_loss
            proc_phase = phases + noise

        # ---------- bandwidth limiting (like low-bitrate MP3) ----------
        if frame_loss > hf_threshold:
            cutoff = int(n_bins * (1.0 - 0.6 * frame_loss))
            cutoff = max(cutoff, n_bins // 8)
            hf_range = 1.0 - hf_threshold if hf_threshold < 1.0 else 1.0
            proc_mag[cutoff:] *= max(0.0, 1.0 - (frame_loss - hf_threshold) / hf_range)

        # ---------- freeze ----------
        if freeze:
            if frozen_spectrum is None:
                frozen_spectrum = proc_mag.copy()
            if freeze_mode == 1:  # solid
                frozen_out = frozen_spectrum.copy()
            else:  # slushy
                frozen_spectrum = (1.0 - slushy_rate_param) * frozen_spectrum + slushy_rate_param * proc_mag
                frozen_out = frozen_spectrum.copy()
            # Freezer blend: 1.0 = fully frozen, 0.0 = bypass freeze
            proc_mag = freezer_blend * frozen_out + (1.0 - freezer_blend) * proc_mag

        # ---------- reconstruct ----------
        proc_spectrum = proc_mag * np.exp(1j * proc_phase)
        proc_frame = np.fft.irfft(proc_spectrum, n=window_size)

        output[start : start + window_size] += proc_frame * window
        win_sum[start : start + window_size] += window ** 2

    # Normalise overlap-add
    win_sum[win_sum < 1e-8] = 1.0
    output /= win_sum

    return output[pad : pad + len(input_audio)]


# ---------- internal helpers ----------


def _compute_ath_weights(band_edges, n_bands, n_bins, window_size):
    """Absolute Threshold of Hearing weighting per band.

    Uses Terhardt's approximation.  Higher weight = ear is less sensitive
    at that frequency = codec is more likely to gate it.
    """
    bin_freqs = np.arange(n_bins) * SR / window_size
    f_khz = np.clip(bin_freqs / 1000.0, 0.02, 20.0)
    ath_db = (3.64 * f_khz ** (-0.8)
              - 6.5 * np.exp(-0.6 * (f_khz - 3.3) ** 2)
              + 1e-3 * f_khz ** 4)
    # Normalise to 0-1 (higher = less sensitive)
    ath_norm = ath_db - np.min(ath_db)
    mx = np.max(ath_norm)
    if mx > 0:
        ath_norm /= mx

    weights = np.zeros(n_bands)
    for b in range(n_bands):
        band = ath_norm[band_edges[b] : band_edges[b + 1]]
        if len(band) > 0:
            weights[b] = np.mean(band)
    return weights


def _standard(magnitudes, loss, rng, band_edges, n_bands,
              ath_weights, quantizer_type, noise_shape, weighting=1.0):
    """Standard mode: quantization + psychoacoustic band gating.

    Quantization reduces magnitude precision.  Two quantizer types:
      Uniform  -- classic mid-tread: delta * round(x / delta)
      Compand  -- MP3-style power-law: compress via x^0.75, quantize, expand via x^(4/3)

    Noise shaping makes quantization coarser in spectral valleys (where the
    signal is quiet) and finer near peaks, following the spectral envelope.

    Band gating uses signal energy + ATH curve to decide which bands to zero.
    Quiet bands at less-sensitive frequencies are gated first -- like a real
    codec running out of bits.  Random perturbation preserves the essential
    frame-to-frame variation that creates the "underwater" sound.
    """
    if loss <= 0.0:
        return magnitudes.copy()

    proc = magnitudes.copy()

    # ---- Quantization ----
    max_mag = np.max(proc)
    if max_mag > 0:
        bits = 16.0 - 14.0 * loss  # loss 0 -> 16 bits, loss 1 -> 2 bits
        n_levels = 2.0 ** bits

        if quantizer_type == 1:  # Compand (power-law, MP3-style)
            compressed = proc ** 0.75
            max_c = np.max(compressed)
            if max_c > 0:
                delta = 2.0 * max_c / n_levels
                if noise_shape > 0:
                    delta = _shape_delta(compressed, delta, noise_shape)
                    compressed = delta * np.round(compressed / np.maximum(delta, 1e-20))
                else:
                    compressed = delta * np.round(compressed / delta)
            proc = np.maximum(compressed, 0.0) ** (4.0 / 3.0)

        else:  # Uniform (mid-tread)
            delta = 2.0 * max_mag / n_levels
            if noise_shape > 0:
                delta = _shape_delta(proc, delta, noise_shape)
                proc = delta * np.round(proc / np.maximum(delta, 1e-20))
            else:
                proc = delta * np.round(proc / delta)

    # ---- Psychoacoustic band gating ----
    band_energy = np.zeros(n_bands)
    for b in range(n_bands):
        band = proc[band_edges[b] : band_edges[b + 1]]
        if len(band) > 0:
            band_energy[b] = np.mean(band ** 2)

    mean_energy = np.mean(band_energy) + 1e-12

    for b in range(n_bands):
        # Signal-dependent: low-energy bands more likely to be gated
        relative = min(band_energy[b] / mean_energy, 2.0) / 2.0
        # ATH-weighted: frequencies where hearing is less sensitive
        # weighting=0 -> equal (0.75), weighting=1 -> full psychoacoustic ATH
        ath_factor = (1.0 - weighting) * 0.75 + weighting * (0.5 + 0.5 * ath_weights[b])
        # Combined gating probability
        gate_prob = loss * 0.6 * (1.0 - relative) * ath_factor
        # Random perturbation for frame-to-frame variation
        gate_prob += rng.random() * loss * 0.2
        if rng.random() < gate_prob:
            proc[band_edges[b] : band_edges[b + 1]] = 0.0

    return proc


def _shape_delta(magnitudes, base_delta, amount):
    """Envelope-following noise shaping.

    Returns a per-bin quantization step that is coarser in spectral valleys
    and finer near peaks.  This makes quantization noise follow the signal
    shape -- the opposite of dithering, which is exactly the lo-fi aesthetic.
    """
    envelope = np.convolve(magnitudes, np.ones(7) / 7.0, mode="same")
    envelope = np.maximum(envelope, 1e-12)
    inv_env = 1.0 / envelope
    inv_env /= np.max(inv_env)
    # amount=0 -> uniform delta, amount=1 -> 4x coarser in valleys
    shape_factor = 1.0 + amount * inv_env * 3.0
    return base_delta * shape_factor
