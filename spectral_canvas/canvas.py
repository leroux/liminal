"""CanvasPanel — Tkinter Canvas with PIL backing for spectrogram painting."""

import tkinter as tk
import numpy as np
from PIL import Image, ImageTk

from .model import SpectrogramModel
from .dsp.pitch import (
    pixel_y_to_freq, freq_to_pixel_y, freq_to_bin, bin_to_freq,
    pixel_x_to_frame, frame_to_pixel_x, hz_to_note, get_scale_freqs
)
from .dsp.brush import apply_brush, apply_brush_line, apply_harmonic_brush
from .dsp.utils import build_colormap_lut


class CanvasPanel(tk.Frame):
    """The spectrogram canvas with drawing, rulers, and playback cursor."""

    CANVAS_W = 1200
    CANVAS_H = 600
    RULER_W = 40   # frequency ruler width
    RULER_H = 20   # time ruler height

    def __init__(self, parent, model: SpectrogramModel, **kwargs):
        super().__init__(parent, **kwargs)
        self.model = model
        self.colormap_lut = build_colormap_lut('magma')

        # Display mode
        self.freq_scale = 'linear'  # 'log' or 'linear'

        # Drawing state
        self.brush_mode = 'draw'  # draw, erase, line, harmonic, select
        self.brush_radius = 8.0
        self.brush_softness = 0.8
        self.brush_intensity = 0.5
        self.harmonic_rolloff = 0.5
        self.n_harmonics = 8
        self.harmonic_mode = 0  # 0=all, 1=odd, 2=even, 3=octaves, 4=fifths
        self.harmonic_sustain = 20  # frames to extend each harmonic stamp
        self.scale_snap = False
        self.scale_name = 'Chromatic'
        self.scale_root = 'C'

        self._last_draw_x = None
        self._last_draw_y = None
        self._line_start = None
        self._select_start = None
        self._select_rect = None
        self._cursor_line = None
        self._photo_image = None  # prevent GC
        self._log_bin_map = None  # cached pixel-row -> bin lookup

        # Zoom/pan state (fractions of full spectrogram, 0-1)
        self._zoom = 1.0
        self._view_x0 = 0.0   # visible time range start
        self._view_x1 = 1.0   # visible time range end
        self._view_y0 = 0.0   # visible freq range start (0=top=high freq)
        self._view_y1 = 1.0   # visible freq range end (1=bottom=low freq)
        self._pan_last = None  # for shift+drag pan

        # Status callback
        self.on_status_update = None  # callable(freq_hz, note, time_s)

        self._build_ui()
        self._bind_events()
        self.refresh_display()

    def _build_ui(self):
        total_w = self.RULER_W + self.CANVAS_W
        total_h = self.RULER_H + self.CANVAS_H

        self.canvas = tk.Canvas(self, width=total_w, height=total_h,
                                bg='black', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

    def _bind_events(self):
        self.canvas.bind('<Button-1>', self._on_press)
        self.canvas.bind('<B1-Motion>', self._on_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_release)
        self.canvas.bind('<Button-2>', self._on_right_press)
        self.canvas.bind('<Button-3>', self._on_right_press)
        self.canvas.bind('<B2-Motion>', self._on_right_drag)
        self.canvas.bind('<B3-Motion>', self._on_right_drag)
        self.canvas.bind('<ButtonRelease-2>', self._on_right_release)
        self.canvas.bind('<ButtonRelease-3>', self._on_right_release)
        self.canvas.bind('<MouseWheel>', self._on_scroll)
        self.canvas.bind('<Motion>', self._on_hover)
        self.canvas.bind('<Configure>', self._on_resize)

    # --- Zoom / pan helpers ---

    def _get_view_freq_range(self):
        """Get visible frequency range in Hz based on current zoom/pan."""
        if self.freq_scale == 'linear':
            max_freq = self.model.sr / 2
            freq_hi = max_freq * (1.0 - self._view_y0)
            freq_lo = max_freq * (1.0 - self._view_y1)
            return max(1.0, freq_lo), max(1.0, freq_hi)
        else:
            log_max = np.log2(20000.0)
            log_min = np.log2(20.0)
            log_range = log_max - log_min
            view_log_hi = log_max - self._view_y0 * log_range
            view_log_lo = log_max - self._view_y1 * log_range
            return max(1.0, 2**view_log_lo), 2**view_log_hi

    def _clamp_view(self):
        """Clamp view bounds to [0, 1] while keeping the view size."""
        vw = self._view_x1 - self._view_x0
        vh = self._view_y1 - self._view_y0
        # Clamp X
        if self._view_x0 < 0:
            self._view_x0 = 0.0
            self._view_x1 = min(1.0, vw)
        if self._view_x1 > 1.0:
            self._view_x1 = 1.0
            self._view_x0 = max(0.0, 1.0 - vw)
        # Clamp Y
        if self._view_y0 < 0:
            self._view_y0 = 0.0
            self._view_y1 = min(1.0, vh)
        if self._view_y1 > 1.0:
            self._view_y1 = 1.0
            self._view_y0 = max(0.0, 1.0 - vh)

    def zoom_at(self, canvas_x, canvas_y, factor):
        """Zoom centered on a canvas pixel position."""
        canvas_w = max(1, self.canvas.winfo_width() - self.RULER_W)
        canvas_h = max(1, self.canvas.winfo_height() - self.RULER_H)
        cx = max(0, canvas_x - self.RULER_W)
        cy = max(0, canvas_y - self.RULER_H)

        # Mouse position as fraction within current view
        mx = cx / canvas_w
        my = cy / canvas_h

        # Spec point under mouse (in 0-1 normalized coords)
        spec_x = self._view_x0 + mx * (self._view_x1 - self._view_x0)
        spec_y = self._view_y0 + my * (self._view_y1 - self._view_y0)

        # Apply zoom
        new_zoom = max(1.0, min(32.0, self._zoom * factor))
        if new_zoom == self._zoom:
            return
        self._zoom = new_zoom

        # New view size
        new_vw = 1.0 / self._zoom
        new_vh = 1.0 / self._zoom

        # Reposition so spec point stays under mouse
        self._view_x0 = spec_x - mx * new_vw
        self._view_x1 = self._view_x0 + new_vw
        self._view_y0 = spec_y - my * new_vh
        self._view_y1 = self._view_y0 + new_vh

        self._clamp_view()
        self._log_bin_map = None
        self.refresh_display()

    def zoom_in(self):
        """Zoom in centered on view."""
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        self.zoom_at(self.RULER_W + canvas_w // 2, self.RULER_H + canvas_h // 2, 1.3)

    def zoom_out(self):
        """Zoom out centered on view."""
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        self.zoom_at(self.RULER_W + canvas_w // 2, self.RULER_H + canvas_h // 2, 1 / 1.3)

    def reset_zoom(self):
        """Reset to full view."""
        self._zoom = 1.0
        self._view_x0 = 0.0
        self._view_x1 = 1.0
        self._view_y0 = 0.0
        self._view_y1 = 1.0
        self._log_bin_map = None
        self.refresh_display()

    def pan_by(self, dx_frac, dy_frac):
        """Pan by a fraction of the view size."""
        vw = self._view_x1 - self._view_x0
        vh = self._view_y1 - self._view_y0
        self._view_x0 += dx_frac * vw
        self._view_x1 += dx_frac * vw
        self._view_y0 += dy_frac * vh
        self._view_y1 += dy_frac * vh
        self._clamp_view()
        self._log_bin_map = None
        self.refresh_display()

    # --- Bin map and coordinate conversion ---

    def _build_log_bin_map(self, canvas_h):
        """Build a lookup: for each display pixel row, which spectrogram bin?

        Uses the visible frequency range (accounting for zoom/pan).
        """
        pixel_rows = np.arange(canvas_h, dtype=np.float64)
        freq_lo, freq_hi = self._get_view_freq_range()

        if self.freq_scale == 'linear':
            bin_hi = freq_to_bin(freq_hi, self.model.sr, self.model.n_fft)
            bin_lo = freq_to_bin(freq_lo, self.model.sr, self.model.n_fft)
            bins = np.round(bin_hi - pixel_rows / canvas_h * (bin_hi - bin_lo)).astype(int)
        else:
            log_hi = np.log2(max(1.0, freq_hi))
            log_lo = np.log2(max(1.0, freq_lo))
            freqs = 2.0 ** (log_hi - pixel_rows / canvas_h * (log_hi - log_lo))
            bins = np.round(freqs * self.model.n_fft / self.model.sr).astype(int)
        bins = np.clip(bins, 0, self.model.n_freq_bins - 1)
        self._log_bin_map = bins
        self._log_bin_map_h = canvas_h

    def _canvas_to_spec(self, event_x, event_y):
        """Convert canvas pixel coords to spectrogram (frame, bin)."""
        cx = event_x - self.RULER_W
        cy = event_y - self.RULER_H

        canvas_w = max(1, self.canvas.winfo_width() - self.RULER_W)
        canvas_h = max(1, self.canvas.winfo_height() - self.RULER_H)

        if cx < 0 or cy < 0 or cx >= canvas_w or cy >= canvas_h:
            return None, None

        # X: map pixel to frame within visible range
        x_frac = cx / canvas_w
        spec_x = self._view_x0 + x_frac * (self._view_x1 - self._view_x0)
        frame = int(round(spec_x * (self.model.n_frames - 1)))

        # Y: map pixel to freq within visible range
        freq_lo, freq_hi = self._get_view_freq_range()

        if self.freq_scale == 'linear':
            frac = cy / canvas_h  # 0=top, 1=bottom
            freq = freq_hi - frac * (freq_hi - freq_lo)
            if self.scale_snap and self.scale_name != 'Chromatic':
                from .dsp.pitch import snap_freq_to_scale
                freq = snap_freq_to_scale(freq, self.scale_name, self.scale_root)
            fbin = freq_to_bin(freq, self.model.sr, self.model.n_fft)
        else:
            freq = pixel_y_to_freq(cy, canvas_h, min_freq=freq_lo, max_freq=freq_hi)
            if self.scale_snap and self.scale_name != 'Chromatic':
                from .dsp.pitch import snap_freq_to_scale
                freq = snap_freq_to_scale(freq, self.scale_name, self.scale_root)
            fbin = freq_to_bin(freq, self.model.sr, self.model.n_fft)

        fbin = max(0, min(fbin, self.model.n_freq_bins - 1))
        frame = max(0, min(frame, self.model.n_frames - 1))
        return frame, fbin

    def _brush_radius_in_bins(self, fbin: int | None = None):
        """Convert pixel brush radius to spectrogram bins.

        Uses the log-frequency bin map to compute the correct radius: we look
        at how many unique bins fall within +/-brush_radius pixels of the current
        position. This means the brush looks the same size on screen at any
        frequency -- at low freq it covers fewer bins, at high freq more bins.
        """
        if fbin is not None and self._log_bin_map is not None:
            bin_map = self._log_bin_map
            # Find the pixel row closest to this bin
            diffs = np.abs(bin_map.astype(np.int32) - fbin)
            center_pixel = int(np.argmin(diffs))
            r_px = int(self.brush_radius)
            lo_px = max(0, center_pixel - r_px)
            hi_px = min(len(bin_map) - 1, center_pixel + r_px)
            # bin range covered by those pixels
            bin_lo = int(bin_map[hi_px])  # hi pixel = lower freq = lower bin
            bin_hi = int(bin_map[lo_px])  # lo pixel = higher freq = higher bin
            r_bins = max(1.0, (bin_hi - bin_lo) / 2.0)
            return r_bins
        # Fallback
        canvas_h = max(1, self.canvas.winfo_height() - self.RULER_H)
        bins_per_pixel = self.model.n_freq_bins / canvas_h
        return max(1.0, self.brush_radius * bins_per_pixel)

    def _apply_draw(self, frame, fbin, is_erase=False, dragging=False):
        """Apply brush at spectrogram coordinates.

        Args:
            dragging: True when called from drag handler. Suppresses sustain
                      spread since the drag itself provides temporal continuity.
        """
        r = self._brush_radius_in_bins(fbin)
        if self.brush_mode == 'harmonic' and not is_erase:
            # Use a fixed minimum radius for harmonics so they're always visible
            harm_r = max(r, 3.0)
            # Sustain only on click, not during drag (drag provides its own
            # temporal extent — applying sustain per drag event causes massive
            # overlap that floods the canvas)
            if dragging:
                apply_harmonic_brush(
                    self.model.magnitude, frame, fbin,
                    harm_r, self.brush_softness, self.brush_intensity,
                    self.n_harmonics, self.harmonic_rolloff,
                    self.model.sr, self.model.n_fft, False,
                    self.harmonic_mode
                )
            else:
                sustain = max(1, self.harmonic_sustain)
                half = sustain // 2
                for dt in range(-half, half + 1):
                    f = frame + dt
                    if 0 <= f < self.model.n_frames:
                        # Fade intensity at edges for smooth attack/release
                        edge_dist = half - abs(dt)
                        fade = min(1.0, (edge_dist + 1) / max(1, half * 0.2))
                        apply_harmonic_brush(
                            self.model.magnitude, f, fbin,
                            harm_r, self.brush_softness, self.brush_intensity * fade,
                            self.n_harmonics, self.harmonic_rolloff,
                            self.model.sr, self.model.n_fft, False,
                            self.harmonic_mode
                        )
        else:
            apply_brush(self.model.magnitude, frame, fbin,
                       r, self.brush_softness, self.brush_intensity, is_erase)
        r_int = int(r * 3)
        y_lo = max(0, fbin - r_int)
        y_hi = min(self.model.n_freq_bins, fbin + r_int + 1)
        x_spread = max(r_int, self.harmonic_sustain) if not dragging else r_int
        x_lo = max(0, frame - x_spread)
        x_hi = min(self.model.n_frames, frame + x_spread + 1)
        self.model.modified_mask[y_lo:y_hi, x_lo:x_hi] = True

    # --- Mouse events ---

    def _on_press(self, event):
        # Shift+click starts pan
        if event.state & 0x1 and self._zoom > 1.01:
            self._pan_last = (event.x, event.y)
            return

        frame, fbin = self._canvas_to_spec(event.x, event.y)
        if frame is None:
            return

        if self.brush_mode == 'line':
            if self._line_start is not None:
                f0, b0 = self._line_start
                self.model.push_undo()
                r = self._brush_radius_in_bins(fbin)
                apply_brush_line(self.model.magnitude, f0, b0, frame, fbin,
                                r, self.brush_softness, self.brush_intensity, False)
                self._line_start = None
                self.refresh_display()
            else:
                self._line_start = (frame, fbin)
            return

        if self.brush_mode == 'select':
            self._select_start = (event.x, event.y)
            return

        self.model.push_undo()
        self._apply_draw(frame, fbin, is_erase=(self.brush_mode == 'erase'))
        self._last_draw_x = frame
        self._last_draw_y = fbin
        self.refresh_display()

    def _on_drag(self, event):
        # Shift+drag pans
        if self._pan_last is not None:
            canvas_w = max(1, self.canvas.winfo_width() - self.RULER_W)
            canvas_h = max(1, self.canvas.winfo_height() - self.RULER_H)
            dx = -(event.x - self._pan_last[0]) / canvas_w
            dy = -(event.y - self._pan_last[1]) / canvas_h
            vw = self._view_x1 - self._view_x0
            vh = self._view_y1 - self._view_y0
            self._view_x0 += dx * vw
            self._view_x1 += dx * vw
            self._view_y0 += dy * vh
            self._view_y1 += dy * vh
            self._clamp_view()
            self._pan_last = (event.x, event.y)
            self._log_bin_map = None
            self.refresh_display()
            return

        if self.brush_mode == 'select' and self._select_start:
            self._update_selection(event.x, event.y)
            return
        if self.brush_mode == 'line':
            return

        frame, fbin = self._canvas_to_spec(event.x, event.y)
        if frame is None:
            return

        is_erase = (self.brush_mode == 'erase')

        # Harmonic mode and snap mode need per-stamp drawing (not brush_line)
        # because brush_line interpolates through non-scale/non-harmonic bins
        use_stamp = (self.brush_mode == 'harmonic') or self.scale_snap

        if self._last_draw_x is not None and not use_stamp:
            r = self._brush_radius_in_bins(fbin)
            apply_brush_line(self.model.magnitude,
                           self._last_draw_x, self._last_draw_y,
                           frame, fbin,
                           r, self.brush_softness, self.brush_intensity, is_erase)
        else:
            self._apply_draw(frame, fbin, is_erase, dragging=True)

        self._last_draw_x = frame
        self._last_draw_y = fbin
        self.refresh_display()

    def _on_release(self, event):
        self._pan_last = None
        self._last_draw_x = None
        self._last_draw_y = None
        if self.brush_mode == 'select' and self._select_start:
            self._finalize_selection(event.x, event.y)
            self._select_start = None

    def _on_right_press(self, event):
        frame, fbin = self._canvas_to_spec(event.x, event.y)
        if frame is None:
            return
        self.model.push_undo()
        self._apply_draw(frame, fbin, is_erase=True)
        self._last_draw_x = frame
        self._last_draw_y = fbin
        self.refresh_display()

    def _on_right_drag(self, event):
        frame, fbin = self._canvas_to_spec(event.x, event.y)
        if frame is None:
            return
        if self._last_draw_x is not None:
            r = self._brush_radius_in_bins(fbin)
            apply_brush_line(self.model.magnitude,
                           self._last_draw_x, self._last_draw_y,
                           frame, fbin,
                           r, self.brush_softness, self.brush_intensity, True)
        else:
            self._apply_draw(frame, fbin, is_erase=True)
        self._last_draw_x = frame
        self._last_draw_y = fbin
        self.refresh_display()

    def _on_right_release(self, event):
        self._last_draw_x = None
        self._last_draw_y = None

    def _on_scroll(self, event):
        # Shift + scroll = zoom centered on cursor
        if event.state & 0x1:
            factor = 1.3 if event.delta > 0 else 1 / 1.3
            self.zoom_at(event.x, event.y, factor)
        else:
            # Scroll = brush radius
            if event.delta > 0:
                self.brush_radius = min(50, self.brush_radius + 1)
            else:
                self.brush_radius = max(1, self.brush_radius - 1)

    def _on_hover(self, event):
        cx = event.x - self.RULER_W
        cy = event.y - self.RULER_H
        canvas_w = max(1, self.canvas.winfo_width() - self.RULER_W)
        canvas_h = max(1, self.canvas.winfo_height() - self.RULER_H)

        if cx < 0 or cy < 0 or cx >= canvas_w or cy >= canvas_h:
            return

        freq_lo, freq_hi = self._get_view_freq_range()

        if self.freq_scale == 'linear':
            frac = cy / canvas_h
            freq = freq_hi - frac * (freq_hi - freq_lo)
        else:
            freq = pixel_y_to_freq(cy, canvas_h, min_freq=freq_lo, max_freq=freq_hi)
        note = hz_to_note(freq)

        x_frac = cx / canvas_w
        time_s = (self._view_x0 + x_frac * (self._view_x1 - self._view_x0)) * self.model.duration

        if self.on_status_update:
            self.on_status_update(freq, note, time_s)

    def _on_resize(self, event):
        self._log_bin_map = None  # invalidate cache
        self.refresh_display()

    # --- Selection ---

    def _update_selection(self, x, y):
        if self._select_rect:
            self.canvas.delete(self._select_rect)
        sx, sy = self._select_start
        self._select_rect = self.canvas.create_rectangle(
            sx, sy, x, y, outline='red', width=1, dash=(4, 4))

    def _finalize_selection(self, x, y):
        pass

    def fill_selection(self, brightness: float):
        """Fill the selected region with a constant brightness."""
        if self._select_start is None or self._select_rect is None:
            return
        coords = self.canvas.coords(self._select_rect)
        if not coords or len(coords) < 4:
            return
        x0, y0, x1, y1 = coords
        f0, b0 = self._canvas_to_spec(int(x0), int(y0))
        f1, b1 = self._canvas_to_spec(int(x1), int(y1))
        if f0 is None or f1 is None:
            return
        self.model.push_undo()
        fmin_b, fmax_b = min(b0, b1), max(b0, b1)
        fmin_f, fmax_f = min(f0, f1), max(f0, f1)
        self.model.magnitude[fmin_b:fmax_b + 1, fmin_f:fmax_f + 1] = brightness
        self.canvas.delete(self._select_rect)
        self._select_rect = None
        self.refresh_display()

    # --- Display ---

    def refresh_display(self):
        """Redraw the entire canvas from the model's magnitude array."""
        canvas_w = max(1, self.canvas.winfo_width() - self.RULER_W)
        canvas_h = max(1, self.canvas.winfo_height() - self.RULER_H)

        if canvas_w < 10 or canvas_h < 10:
            return

        # Build/use cached log-frequency bin mapping
        if self._log_bin_map is None or self._log_bin_map_h != canvas_h:
            self._build_log_bin_map(canvas_h)

        mag = self.model.magnitude  # (n_freq_bins, n_frames)

        # Extract visible frame range
        frame_lo = max(0, int(self._view_x0 * self.model.n_frames))
        frame_hi = min(self.model.n_frames, max(frame_lo + 1,
                       int(np.ceil(self._view_x1 * self.model.n_frames))))
        visible_mag = mag[:, frame_lo:frame_hi]

        # Remap to log-frequency display using the (zoomed) bin map
        display_mag = visible_mag[self._log_bin_map, :]

        # Colormap
        indices = (np.clip(display_mag, 0, 1) * 255).astype(np.uint8)
        rgb = self.colormap_lut[indices]

        img = Image.fromarray(rgb, 'RGB')
        interp = Image.NEAREST if self._zoom <= 2.0 else Image.BILINEAR
        img = img.resize((canvas_w, canvas_h), interp)

        # Create full image with rulers
        total_w = self.RULER_W + canvas_w
        total_h = self.RULER_H + canvas_h
        full_img = Image.new('RGB', (total_w, total_h), 'black')
        full_img.paste(img, (self.RULER_W, self.RULER_H))

        self._photo_image = ImageTk.PhotoImage(full_img)
        self.canvas.delete('spectrogram')
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self._photo_image,
                                tags='spectrogram')

        # Draw rulers on top
        self._draw_freq_ruler(canvas_h)
        self._draw_time_ruler(canvas_w)
        self._draw_scale_lines(canvas_w, canvas_h)

        # Zoom indicator
        if self._zoom > 1.01:
            self.canvas.delete('zoom_ind')
            self.canvas.create_text(
                self.RULER_W + canvas_w - 5, self.RULER_H + 5,
                text=f'{self._zoom:.1f}x', anchor=tk.NE,
                fill='#aaaaaa', font=('Monaco', 10), tags='zoom_ind')
        else:
            self.canvas.delete('zoom_ind')

    def _draw_freq_ruler(self, canvas_h):
        """Draw frequency labels on the left ruler."""
        self.canvas.delete('freq_ruler')
        freq_lo, freq_hi = self._get_view_freq_range()

        if self.freq_scale == 'linear':
            all_ticks = [0, 100, 200, 500, 1000, 2000, 3000, 4000, 5000,
                         6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000]
            for freq in all_ticks:
                if freq < freq_lo or freq > freq_hi:
                    continue
                if freq_hi <= freq_lo:
                    continue
                frac = (freq_hi - freq) / (freq_hi - freq_lo)
                y = frac * canvas_h + self.RULER_H
                if self.RULER_H + 8 < y < canvas_h + self.RULER_H - 8:
                    label = f'{freq // 1000}k' if freq >= 1000 else str(freq)
                    self.canvas.create_text(
                        self.RULER_W - 5, y, text=label, anchor=tk.E,
                        fill='gray', font=('Monaco', 9), tags='freq_ruler')
                    self.canvas.create_line(
                        self.RULER_W - 2, y, self.RULER_W + 3, y,
                        fill='gray', tags='freq_ruler')
        else:
            hz_ticks = [20, 30, 40, 50, 75, 100, 150, 200, 300, 400, 500,
                        700, 1000, 1500, 2000, 3000, 5000, 7000, 10000,
                        15000, 20000]
            for freq in hz_ticks:
                if freq < freq_lo * 0.95 or freq > freq_hi * 1.05:
                    continue
                y = freq_to_pixel_y(freq, canvas_h,
                                    min_freq=freq_lo, max_freq=freq_hi) + self.RULER_H
                if self.RULER_H + 8 < y < canvas_h + self.RULER_H - 8:
                    label = f'{freq // 1000}k' if freq >= 1000 else str(freq)
                    self.canvas.create_text(
                        self.RULER_W - 5, y, text=label, anchor=tk.E,
                        fill='gray', font=('Monaco', 9), tags='freq_ruler')
                    self.canvas.create_line(
                        self.RULER_W - 2, y, self.RULER_W + 3, y,
                        fill='gray', tags='freq_ruler')

    def _draw_time_ruler(self, canvas_w):
        """Draw time labels on the top ruler."""
        self.canvas.delete('time_ruler')
        dur = self.model.duration
        t_lo = self._view_x0 * dur
        t_hi = self._view_x1 * dur
        visible_dur = t_hi - t_lo

        if visible_dur <= 0.5:
            interval = 0.05
        elif visible_dur <= 2:
            interval = 0.2
        elif visible_dur <= 5:
            interval = 0.5
        elif visible_dur <= 15:
            interval = 1.0
        else:
            interval = 2.0

        # Start at first tick at or after t_lo
        t = int(t_lo / interval) * interval
        if t < t_lo:
            t += interval
        while t <= t_hi:
            if visible_dur > 0:
                x_frac = (t - t_lo) / visible_dur
                x = int(x_frac * canvas_w) + self.RULER_W
                if self.RULER_W < x < canvas_w + self.RULER_W:
                    if interval < 0.5:
                        label = f'{t:.2f}s'
                    elif interval < 1:
                        label = f'{t:.1f}s'
                    else:
                        label = f'{t:.0f}s'
                    self.canvas.create_text(
                        x, self.RULER_H - 3, text=label, anchor=tk.S,
                        fill='gray', font=('Monaco', 9), tags='time_ruler')
                    self.canvas.create_line(
                        x, self.RULER_H - 2, x, self.RULER_H + 3,
                        fill='gray', tags='time_ruler')
            t += interval

    def _draw_scale_lines(self, canvas_w, canvas_h):
        """Draw faint guide lines for the active scale."""
        self.canvas.delete('scale_lines')
        if not self.scale_snap or self.scale_name == 'Chromatic':
            return
        freq_lo, freq_hi = self._get_view_freq_range()
        freqs = get_scale_freqs(self.scale_name, self.scale_root,
                                min_freq=freq_lo, max_freq=freq_hi)
        for freq in freqs:
            if self.freq_scale == 'linear':
                if freq_hi <= freq_lo:
                    continue
                frac = (freq_hi - freq) / (freq_hi - freq_lo)
                y = frac * canvas_h + self.RULER_H
            else:
                y = freq_to_pixel_y(freq, canvas_h,
                                    min_freq=freq_lo, max_freq=freq_hi) + self.RULER_H
            if self.RULER_H < y < canvas_h + self.RULER_H:
                self.canvas.create_line(
                    self.RULER_W, y, self.RULER_W + canvas_w, y,
                    fill='#555555', width=1, tags='scale_lines')

    def set_cursor_position(self, fraction: float):
        """Set the playback cursor position (0-1)."""
        self.canvas.delete('cursor')
        if fraction <= 0 or fraction >= 1:
            return
        # Map fraction to visible range
        if fraction < self._view_x0 or fraction > self._view_x1:
            return
        vw = self._view_x1 - self._view_x0
        if vw <= 0:
            return
        visible_frac = (fraction - self._view_x0) / vw
        canvas_w = max(1, self.canvas.winfo_width() - self.RULER_W)
        x = self.RULER_W + int(visible_frac * canvas_w)
        canvas_h = self.canvas.winfo_height()
        self.canvas.create_line(
            x, self.RULER_H, x, canvas_h,
            fill='red', width=1, tags='cursor')
