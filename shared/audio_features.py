"""Format audio metrics for LLM consumption + spectrogram generation.

Used by LLMTuner to send readable, interpretive audio analysis to Claude.
"""

import io


def format_features(metrics, prev_metrics=None, source_metrics=None):
    """Format analyze() dict as readable text with interpretive hints.

    Args:
        metrics: Current output audio metrics dict from analyze().
        prev_metrics: Previous output metrics for A/B delta (optional).
        source_metrics: Input/source audio metrics (optional).

    Returns:
        Formatted string for inclusion in LLM prompt.
    """
    if not metrics and not source_metrics:
        return ""

    parts = []

    if source_metrics:
        parts.append("INPUT AUDIO:")
        parts.append(_format_source(source_metrics))
        parts.append("")

    if metrics:
        parts.append("OUTPUT AUDIO (current render):")
        parts.append(_format_output(metrics))

        if prev_metrics:
            delta = _format_delta(prev_metrics, metrics)
            if delta:
                parts.append("")
                parts.append("CHANGES FROM PREVIOUS RENDER:")
                parts.append(delta)

    return "\n".join(parts)


def _format_source(m):
    """Format source/input audio metrics."""
    lines = []
    if m.get("rms_db") is not None:
        lines.append(f"  RMS level: {m['rms_db']:.1f} dB")
    if m.get("peak_db") is not None:
        lines.append(f"  Peak level: {m['peak_db']:.1f} dB")
    if m.get("rt60") is not None:
        lines.append(f"  RT60: {m['rt60']:.3f}s")
    if m.get("edt") is not None:
        lines.append(f"  EDT: {m['edt']:.3f}s")
    if m.get("spectral_centroid") is not None:
        lines.append(f"  Spectral centroid: {m['spectral_centroid']:.0f} Hz")
    if m.get("echo_density") is not None:
        lines.append(f"  Echo density: {m['echo_density']:.2f}")
    if m.get("crest_factor") is not None:
        lines.append(f"  Crest factor: {m['crest_factor']:.1f} dB")
    if m.get("c50") is not None:
        lines.append(f"  C50: {m['c50']:.1f} dB")
    if m.get("c80") is not None:
        lines.append(f"  C80: {m['c80']:.1f} dB")
    if m.get("spectral_flatness") is not None:
        lines.append(f"  Spectral flatness: {m['spectral_flatness']:.3f}")
    if m.get("bandwidth") is not None:
        lines.append(f"  Bandwidth: {m['bandwidth']:.0f} Hz")
    return "\n".join(lines)


def _format_output(m):
    """Format output audio metrics with units."""
    lines = []
    if m.get("rms_db") is not None:
        lines.append(f"  RMS level: {m['rms_db']:.1f} dB")
    if m.get("peak_db") is not None:
        lines.append(f"  Peak level: {m['peak_db']:.1f} dB")
    if m.get("rt60") is not None:
        lines.append(f"  RT60: {m['rt60']:.3f}s")
    if m.get("edt") is not None:
        lines.append(f"  EDT: {m['edt']:.3f}s")
    if m.get("spectral_centroid") is not None:
        lines.append(f"  Spectral centroid: {m['spectral_centroid']:.0f} Hz")
    if m.get("echo_density") is not None:
        lines.append(f"  Echo density: {m['echo_density']:.2f}")
    if m.get("crest_factor") is not None:
        lines.append(f"  Crest factor: {m['crest_factor']:.1f} dB")
    if m.get("c50") is not None:
        lines.append(f"  C50: {m['c50']:.1f} dB")
    if m.get("c80") is not None:
        lines.append(f"  C80: {m['c80']:.1f} dB")
    if m.get("spectral_flatness") is not None:
        lines.append(f"  Spectral flatness: {m['spectral_flatness']:.3f}")
    if m.get("bandwidth") is not None:
        lines.append(f"  Bandwidth: {m['bandwidth']:.0f} Hz")
    # Comparison metrics (lossy pedal)
    if m.get("energy_ratio_db") is not None:
        lines.append(f"  Energy ratio: {m['energy_ratio_db']:.1f} dB")
    if m.get("thd_n_percent") is not None:
        lines.append(f"  THD+N: {m['thd_n_percent']:.1f}%")
    if m.get("spectral_centroid_shift") is not None:
        lines.append(f"  Centroid shift: {m['spectral_centroid_shift']:.0f} Hz")
    if m.get("bandwidth_change") is not None:
        lines.append(f"  Bandwidth change: {m['bandwidth_change']:.0f} Hz")
    # Band RT60
    bands = m.get("rt60_bands", {})
    if bands:
        band_strs = []
        for freq, val in bands.items():
            if val is not None:
                band_strs.append(f"{freq}Hz={val:.2f}s")
        if band_strs:
            lines.append(f"  Band RT60: {', '.join(band_strs)}")
    return "\n".join(lines)


# Metric definitions for delta computation:
# key -> (unit, precision, directional hints for increase/decrease)
_DELTA_DEFS = {
    "rt60":               ("s",  3, "longer tail",    "shorter tail"),
    "edt":                ("s",  3, "longer perceived","shorter perceived"),
    "spectral_centroid":  ("Hz", 0, "brighter",       "darker"),
    "echo_density":       ("",   2, "denser",         "sparser"),
    "crest_factor":       ("dB", 1, "more dynamic",   "more compressed"),
    "c50":                ("dB", 1, "clearer",         "more diffuse"),
    "c80":                ("dB", 1, "clearer",         "more diffuse"),
    "spectral_flatness":  ("",   3, "noisier",         "more tonal"),
    "bandwidth":          ("Hz", 0, "wider",           "narrower"),
    "energy_ratio_db":    ("dB", 1, "louder",          "quieter"),
    "thd_n_percent":      ("%",  1, "more distorted",  "cleaner"),
}


def _format_delta(prev, curr):
    """Format significant changes between two metrics dicts."""
    lines = []
    for key, (unit, prec, hint_up, hint_down) in _DELTA_DEFS.items():
        old = prev.get(key)
        new = curr.get(key)
        if old is None or new is None:
            continue
        diff = new - old
        # Skip insignificant changes: >5% relative or meaningful absolute
        if abs(old) > 1e-6:
            rel = abs(diff / old)
            if rel < 0.05:
                continue
        elif abs(diff) < 0.01:
            continue
        hint = hint_up if diff > 0 else hint_down
        sign = "+" if diff > 0 else ""
        lines.append(
            f"  {key}: {old:.{prec}f} -> {new:.{prec}f}{unit} "
            f"({sign}{diff:.{prec}f}{unit}, {hint})"
        )
    return "\n".join(lines)


def generate_spectrogram_png(audio, sr):
    """Generate a small spectrogram PNG as bytes.

    Returns bytes or None if matplotlib is not available.
    Uses Agg backend to avoid tkinter conflicts.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    import numpy as np
    mono = np.asarray(audio, dtype=np.float64)
    if mono.ndim == 2:
        mono = mono.mean(axis=1)
    if len(mono) < 2048:
        return None

    fig, ax = plt.subplots(1, 1, figsize=(6, 3), dpi=100)
    ax.specgram(mono, Fs=sr, NFFT=2048, noverlap=1536, cmap="magma")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_ylim(0, min(sr / 2, 16000))
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()
