"""Shared waveform drawing for both pedal GUIs.

Reverb-style visualization: dark background, filled envelope, RMS overlay,
amplitude/time grids, channel labels, and metrics rows.
"""

import numpy as np


def draw_waveform(canvas, audio, sr, metrics=None, title_prefix="Output", warning=""):
    """Draw waveform with grid, filled envelope, RMS overlay, and metrics.

    Args:
        canvas: tk.Canvas to draw on.
        audio: numpy array — mono (N,) or stereo (N,2). None shows placeholder.
        sr: Sample rate.
        metrics: Optional dict from shared.analysis.analyze().
        title_prefix: "Output" or "Input" for title bar.
        warning: Optional warning string (e.g. loudness normalization note).
    """
    c = canvas
    c.delete("all")
    c.update_idletasks()
    W = c.winfo_width() or 900
    H = c.winfo_height() or 500

    TITLE_COL = "#e0e0e0"
    GRID_COL = "#222233"
    LABEL_COL = "#666688"
    WAVE_COL = "#4a9acc"
    RMS_COL = "#cc8844"
    F_TITLE = ("Helvetica", 12, "bold")
    F_LABEL = ("Helvetica", 9)

    pad_top = 35
    pad_bot = 88
    pad_left = 50
    pad_right = 20
    plot_w = W - pad_left - pad_right
    plot_h = H - pad_top - pad_bot

    if plot_w < 50 or plot_h < 30:
        return

    if audio is None:
        c.create_text(W // 2, H // 2,
                      text=f"No {title_prefix.lower()} audio yet",
                      fill=LABEL_COL, font=F_TITLE)
        return

    raw_audio = audio
    stereo = raw_audio.ndim == 2
    n_samples = raw_audio.shape[0] if stereo else len(raw_audio)
    duration = n_samples / sr

    if stereo:
        ch_list = [(raw_audio[:, 0], WAVE_COL, "#1a3550", "L"),
                   (raw_audio[:, 1], "#cc6644", "#3a2218", "R")]
    else:
        ch_list = [(raw_audio, WAVE_COL, "#1a3550", None)]

    n_channels = len(ch_list)
    ch_label = "Stereo" if stereo else "Mono"
    c.create_text(W // 2, 16,
                  text=f"{title_prefix} Waveform — {ch_label} — {duration:.1f}s — {n_samples} samples",
                  fill=TITLE_COL, font=F_TITLE)

    for ch_idx, (ch_audio, wave_col, fill_col, label) in enumerate(ch_list):
        ch_top = pad_top + ch_idx * (plot_h // n_channels)
        ch_h = plot_h // n_channels
        zero_y = ch_top + ch_h / 2

        # Plot area background
        c.create_rectangle(pad_left, ch_top, pad_left + plot_w, ch_top + ch_h,
                           fill="#0a0a14", outline="#333355")

        # Amplitude grid lines and labels
        for amp in [-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75]:
            y = zero_y - amp * ch_h / 2
            c.create_line(pad_left, y, pad_left + plot_w, y, fill=GRID_COL)
            if amp in (-0.5, 0.0, 0.5) and ch_idx == 0:
                c.create_text(pad_left - 5, y, text=f"{amp:.1f}",
                              fill=LABEL_COL, font=F_LABEL, anchor="e")

        # Zero line
        c.create_line(pad_left, zero_y, pad_left + plot_w, zero_y,
                      fill="#333355", width=1)

        # Channel label
        if label:
            c.create_text(pad_left + 6, ch_top + 4, text=label,
                          fill=wave_col, font=F_LABEL, anchor="nw")

        # Time grid (first channel only)
        if ch_idx == 0:
            time_step = max(0.5, round(duration / 8, 1))
            t = 0.0
            while t <= duration:
                x = pad_left + (t / duration) * plot_w
                c.create_line(x, pad_top, x, pad_top + plot_h, fill=GRID_COL)
                c.create_text(x, pad_top + plot_h + 12, text=f"{t:.1f}s",
                              fill=LABEL_COL, font=F_LABEL)
                t += time_step

        # Downsample: min/max per pixel column
        n_cols = min(plot_w, n_samples)
        wave_points_top = []
        wave_points_bot = []
        for col in range(n_cols):
            start = int(col * n_samples / n_cols)
            end = min(int((col + 1) * n_samples / n_cols), n_samples)
            if start >= end:
                continue
            chunk = ch_audio[start:end]
            mn = float(np.min(chunk))
            mx = float(np.max(chunk))
            x = pad_left + col
            y_top = zero_y - mx * (ch_h / 2)
            y_bot = zero_y - mn * (ch_h / 2)
            y_top = max(ch_top, min(ch_top + ch_h, y_top))
            y_bot = max(ch_top, min(ch_top + ch_h, y_bot))
            wave_points_top.append((x, y_top))
            wave_points_bot.append((x, y_bot))

        # Filled waveform envelope
        if wave_points_top and wave_points_bot:
            poly = []
            for x, y in wave_points_top:
                poly.extend([x, y])
            for x, y in reversed(wave_points_bot):
                poly.extend([x, y])
            if len(poly) >= 6:
                c.create_polygon(poly, fill=fill_col, outline="")
            if len(wave_points_top) >= 2:
                top_flat = []
                for x, y in wave_points_top:
                    top_flat.extend([x, y])
                c.create_line(*top_flat, fill=wave_col, width=1)
            if len(wave_points_bot) >= 2:
                bot_flat = []
                for x, y in wave_points_bot:
                    bot_flat.extend([x, y])
                c.create_line(*bot_flat, fill=wave_col, width=1)

        # RMS envelope
        rms_window = max(1, n_samples // 200)
        rms_points = []
        for col in range(min(200, n_cols)):
            start = int(col * n_samples / 200)
            end = min(start + rms_window * 2, n_samples)
            if start >= end:
                continue
            chunk = ch_audio[start:end]
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            x = pad_left + (col / 200) * plot_w
            y = zero_y - rms * (ch_h / 2)
            y = max(ch_top, min(ch_top + ch_h, y))
            rms_points.extend([x, y])
        if len(rms_points) >= 4:
            c.create_line(*rms_points, fill=RMS_COL, width=1.5, smooth=True)

    # Stats
    peak = float(np.max(np.abs(raw_audio)))
    rms_total = float(np.sqrt(np.mean(raw_audio ** 2)))
    stats_y = pad_top + plot_h + 28
    c.create_text(pad_left, stats_y, text=f"Peak: {peak:.3f}",
                  fill=WAVE_COL, font=F_LABEL, anchor="w")
    c.create_text(pad_left + 120, stats_y, text=f"RMS: {rms_total:.3f}",
                  fill=RMS_COL, font=F_LABEL, anchor="w")
    if warning:
        c.create_text(pad_left + 240, stats_y, text=warning.strip(),
                      fill="#dd6666", font=F_LABEL, anchor="w")

    # Metrics rows
    if metrics:
        _draw_metrics(c, metrics, pad_left, stats_y, F_LABEL)


def _draw_metrics(c, m, pad_left, stats_y, font):
    """Draw metrics text rows below the waveform."""
    def fmt(v, unit="", prec=2):
        if v is None:
            return "N/A"
        return f"{v:.{prec}f}{unit}"

    row1_y = stats_y + 14
    items1 = [
        f"RT60: {fmt(m.get('rt60'), 's', 3)}",
        f"EDT: {fmt(m.get('edt'), 's', 3)}",
        f"Crest: {fmt(m.get('crest_factor'), 'dB')}",
    ]
    c.create_text(pad_left, row1_y, text="   ".join(items1),
                  fill="#88cc88", font=font, anchor="w")

    row2_y = row1_y + 14
    items2 = [
        f"C50: {fmt(m.get('c50'), 'dB')}",
        f"C80: {fmt(m.get('c80'), 'dB')}",
        f"Centroid: {fmt(m.get('spectral_centroid'), 'Hz', 0)}",
        f"Density: {fmt(m.get('echo_density'), '', 2)}",
        f"Flatness: {fmt(m.get('spectral_flatness'), '', 3)}",
    ]
    c.create_text(pad_left, row2_y, text="   ".join(items2),
                  fill="#8888cc", font=font, anchor="w")

    row3_y = row2_y + 14
    bands = m.get("rt60_bands", {})
    if bands:
        band_str = "  ".join(f"{k}: {fmt(v, 's', 2)}" for k, v in bands.items())
        c.create_text(pad_left, row3_y, text=f"Band RT60  {band_str}",
                      fill="#aa88aa", font=font, anchor="w")
        row3_y += 14

    # Comparison metrics (lossy: energy ratio, THD+N, bandwidth, centroid shift)
    if "energy_ratio_db" in m:
        items3 = [
            f"Energy: {fmt(m.get('energy_ratio_db'), 'dB')}",
            f"THD+N: {fmt(m.get('thd_n_percent'), '%')}",
            f"BW: {fmt(m.get('bandwidth'), 'Hz', 0)}",
            f"\u0394Centroid: {fmt(m.get('spectral_centroid_shift'), 'Hz', 0)}",
        ]
        c.create_text(pad_left, row3_y, text="   ".join(items3),
                      fill="#cc88cc", font=font, anchor="w")
