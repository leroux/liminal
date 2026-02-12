"""Lossy codec emulation GUI — tkinter.

Tabs: Parameters · Presets · Waveforms · Spectrograms · Spectrum
"""

import json
import math
import os
import random
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, simpledialog

import numpy as np
import sounddevice as sd
from scipy.io import wavfile

from lossy.engine.params import (
    SR,
    default_params,
    bypass_params,
    migrate_legacy_params,

    PARAM_RANGES,
    PARAM_SECTIONS,
    CHOICE_RANGES,
    QUANTIZER_NAMES,
    PACKET_NAMES,
    FILTER_NAMES,
    SLOPE_OPTIONS,
    FREEZE_NAMES,
    VERB_POSITION_NAMES,
    BOUNCE_TARGETS,
    BOUNCE_TARGET_NAMES,
)
from lossy.engine.lossy import render_lossy

from shared.llm_guide_text import LOSSY_GUIDE, LOSSY_PARAM_DESCRIPTIONS
from shared.llm_tuner import LLMTuner
from shared.streaming import safety_check, normalize_output
from shared.waveform import (draw_waveform as _shared_draw_waveform, draw_spectrogram,
                              WAVE_PAD_LEFT, WAVE_PAD_RIGHT,
                              SPEC_PAD_LEFT, SPEC_PAD_RIGHT)

PRESET_DIR = os.path.join(os.path.dirname(__file__), "presets")
DEFAULT_SOURCE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    os.pardir,
    "audio",
    "test_signals",
    "dry_noise_burst.wav",
)


class LossyGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Lossy — codec artifact emulator")
        self.root.geometry("900x680")
        # Window / dock icon
        _icon_path = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir,
                                   "icons", "lossy.png")
        if os.path.exists(_icon_path):
            self._app_icon = tk.PhotoImage(file=_icon_path)
            self.root.iconphoto(True, self._app_icon)
        self.params = default_params()
        self.source_audio = None
        self.rendered_audio = None
        self.rendered_params = None
        self.rendered_warning = ""
        self.rendered_metrics = None
        self.rendering = False
        self.sliders = {}
        self.locks = {}
        self.section_locks = {}
        self._scroll_widgets = {}
        self._autoplay_id = None
        self._cursor_timer = None
        self._cursor_canvases = []
        self._playback_start = None
        self._playback_length = 0.0
        self._playback_audio = None

        # Generation history for rewind/forward
        self._gen_history = []     # list of dicts: {params, audio, metrics, warning, spectrogram}
        self._gen_index = -1       # current position in history
        self._gen_max = 50         # max history entries
        self._output_device_idx = None  # None = system default
        self._output_devices = []       # [(sd_index, name), ...]
        self._load_wav(DEFAULT_SOURCE)
        self._build_ui()

        self.llm = LLMTuner(
            guide_text=LOSSY_GUIDE,
            param_descriptions=LOSSY_PARAM_DESCRIPTIONS,
            param_ranges=PARAM_RANGES,
            default_params_fn=default_params,
            root=self.root,
        )

        # Global mousewheel handler — routes to scale under cursor
        self.root.bind_all("<MouseWheel>", self._on_global_scroll)
        # Tk 9.0+ on macOS: trackpad generates TouchpadScroll instead of MouseWheel
        try:
            self.root.bind_all("<TouchpadScroll>", self._on_global_scroll)
        except tk.TclError:
            pass

    # ------------------------------------------------------------------
    # WAV helpers
    # ------------------------------------------------------------------

    def _load_wav(self, path):
        if not os.path.isfile(path):
            self.source_audio = self._make_impulse()
            self._analyze_source()
            return
        sr, data = wavfile.read(path)
        if data.dtype == np.int16:
            audio = data.astype(np.float64) / 32768.0
        elif data.dtype == np.int32:
            audio = data.astype(np.float64) / 2147483648.0
        else:
            audio = data.astype(np.float64)
        # Resample to engine SR if needed (engine assumes SR throughout)
        if sr != SR:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(SR, sr)
            audio = resample_poly(audio, SR // g, sr // g, axis=0)
        # Keep stereo as (samples, 2) — don't downmix
        self.source_audio = audio
        self._analyze_source()

    def _analyze_source(self):
        """Compute source audio metrics and spectrogram for LLM context."""
        from shared.analysis import analyze
        from shared.audio_features import generate_spectrogram_png
        self.source_metrics = analyze(self.source_audio, SR)
        self.source_spectrogram = generate_spectrogram_png(self.source_audio, SR)
        if hasattr(self, 'llm'):
            self.llm.reset_session()

    @staticmethod
    def _make_impulse(seconds=2.0):
        n = int(SR * seconds)
        audio = np.zeros(n)
        audio[0] = 1.0
        return audio

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        # Top bar
        top = ttk.Frame(self.root)
        top.pack(fill="x", padx=8, pady=4)
        ttk.Button(top, text="Load WAV", command=self._on_load).pack(side="left", padx=2)
        ttk.Button(top, text="Save WAV", command=self._on_save).pack(side="left", padx=2)
        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=6)
        ttk.Button(top, text="Play", command=self._on_play).pack(side="left", padx=2)
        ttk.Button(top, text="Dry", command=self._on_play_dry).pack(side="left", padx=2)
        ttk.Button(top, text="Stop", command=self._on_stop).pack(side="left", padx=2)
        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=6)
        ttk.Button(top, text="Randomize", command=self._randomize_params).pack(side="left", padx=2)
        ttk.Button(top, text="Reset", command=self._reset_params).pack(side="left", padx=2)
        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=6)
        self._gen_back_btn = ttk.Button(top, text="<", width=2, command=self._on_gen_back)
        self._gen_back_btn.pack(side="left", padx=1)
        self._gen_label_var = tk.StringVar(value="Gen 0")
        ttk.Label(top, textvariable=self._gen_label_var, width=8, anchor="center").pack(side="left")
        self._gen_fwd_btn = ttk.Button(top, text=">", width=2, command=self._on_gen_forward)
        self._gen_fwd_btn.pack(side="left", padx=1)
        self._update_gen_buttons()

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(top, textvariable=self.status_var).pack(side="right")
        ttk.Button(top, text="\u21bb", width=2, command=self._refresh_devices).pack(side="right", padx=1)
        self._device_combo = ttk.Combobox(top, state="readonly", width=28)
        self._device_combo.pack(side="right", padx=2)
        self._device_combo.bind("<<ComboboxSelected>>", self._on_device_changed)
        ttk.Label(top, text="Output:").pack(side="right", padx=(5, 2))
        self._refresh_devices()

        # Notebook
        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=8, pady=4)

        self.params_frame = ttk.Frame(nb)
        self.presets_frame = ttk.Frame(nb)
        self.waveforms_frame = ttk.Frame(nb, padding=5)
        self.spectrograms_frame = ttk.Frame(nb, padding=5)
        self.spec_frame = ttk.Frame(nb)
        self.guide_frame = ttk.Frame(nb)

        nb.add(self.params_frame, text="Parameters")
        nb.add(self.presets_frame, text="Presets")
        nb.add(self.waveforms_frame, text="Waveforms")
        nb.add(self.spectrograms_frame, text="Spectrograms")
        nb.add(self.spec_frame, text="Spectrum")
        nb.add(self.guide_frame, text="Guide")

        self._build_params_page()
        self._build_presets_page()
        self._build_waveforms_page()
        self._build_spectrograms_page()
        self._build_spec_page()
        self._build_guide_page()

    # ------------------------------------------------------------------
    # Parameters tab
    # ------------------------------------------------------------------

    def _build_params_page(self):
        # Responsive container for params + AI chat
        self._params_ai_container = ttk.Frame(self.params_frame)
        self._params_ai_container.pack(fill="both", expand=True)

        self._params_scroll_frame = ttk.Frame(self._params_ai_container)
        scroll_container = self._params_scroll_frame

        canvas = tk.Canvas(scroll_container, highlightthickness=0)
        vscrollbar = ttk.Scrollbar(scroll_container, orient="vertical", command=canvas.yview)
        hscrollbar = ttk.Scrollbar(scroll_container, orient="horizontal", command=canvas.xview)
        f = ttk.Frame(canvas)

        f.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=f, anchor="nw")
        canvas.configure(yscrollcommand=vscrollbar.set, xscrollcommand=hscrollbar.set)

        hscrollbar.pack(side="bottom", fill="x")
        canvas.pack(side="left", fill="both", expand=True)
        vscrollbar.pack(side="right", fill="y")

        # Mouse wheel / trackpad scrolling for the page canvas
        def _on_mousewheel(event):
            raw = event.delta if event.delta else (-1 if event.num == 5 else 1)
            y = raw & 0xFFFF
            if y >= 0x8000:
                y -= 0x10000
            if event.state & 1:  # Shift held → horizontal scroll
                canvas.xview_scroll(-y, "units")
            else:
                canvas.yview_scroll(-y, "units")
        canvas.bind("<MouseWheel>", _on_mousewheel)
        canvas.bind("<Button-4>", _on_mousewheel)
        canvas.bind("<Button-5>", _on_mousewheel)
        try:
            canvas.bind("<TouchpadScroll>", _on_mousewheel)
        except tk.TclError:
            pass
        self._params_scroll_canvas = canvas

        r = 0
        SL = 400  # slider length

        # ---- Spectral Loss ----
        r = self._add_section_header(f, r, "spectral", "SPECTRAL LOSS")

        self._inverse_var = tk.IntVar(value=self.params["inverse"])
        ttk.Checkbutton(f, text="Inverse (residual)", variable=self._inverse_var).grid(row=r, column=0, columnspan=2, sticky="w", padx=8)
        self._add_lock(f, r, "inverse")
        r += 1

        r = self._add_slider(f, r, "jitter", "Jitter", 0.0, 1.0, self.params["jitter"], length=SL)
        r = self._add_slider(f, r, "loss", "Loss", 0.0, 1.0, self.params["loss"], length=SL)
        r = self._add_slider(f, r, "window_size", "Window Size", 64, 16384, self.params["window_size"], length=SL, integer=True)
        r = self._add_slider(f, r, "hop_divisor", "Hop Divisor", 1, 8, self.params["hop_divisor"], length=SL, integer=True)
        r = self._add_slider(f, r, "n_bands", "Bands", 2, 64, self.params["n_bands"], length=SL, integer=True)
        r = self._add_slider(f, r, "global_amount", "Global", 0.0, 1.0, self.params["global_amount"], length=SL)
        r = self._add_slider(f, r, "phase_loss", "Phase", 0.0, 1.0, self.params["phase_loss"], length=SL)
        r = self._add_slider(f, r, "pre_echo", "Pre-Echo", 0.0, 1.0, self.params["pre_echo"], length=SL)
        r = self._add_slider(f, r, "transient_ratio", "Transient Thr", 1.5, 20.0, self.params["transient_ratio"], length=SL)
        r = self._add_slider(f, r, "noise_shape", "Noise Shape", 0.0, 1.0, self.params["noise_shape"], length=SL)
        r = self._add_slider(f, r, "hf_threshold", "HF Threshold", 0.0, 1.0, self.params["hf_threshold"], length=SL)
        r = self._add_slider(f, r, "slushy_rate", "Slushy Rate", 0.001, 0.5, self.params["slushy_rate"], length=SL)

        ttk.Label(f, text="Quantizer").grid(row=r, column=0, sticky="w", padx=8)
        self._quantizer_var = tk.IntVar(value=self.params["quantizer"])
        q_frame = ttk.Frame(f)
        q_frame.grid(row=r, column=1, sticky="w")
        for i, name in enumerate(QUANTIZER_NAMES):
            ttk.Radiobutton(q_frame, text=name, variable=self._quantizer_var, value=i).pack(side="left", padx=3)
        self._add_lock(f, r, "quantizer")
        r += 1

        # ---- Crush (time-domain) ----
        r = self._add_separator(f, r)
        r = self._add_section_header(f, r, "crush", "CRUSH")

        r = self._add_slider(f, r, "crush", "Crush", 0.0, 1.0, self.params["crush"], length=SL)
        r = self._add_slider(f, r, "decimate", "Decimate", 0.0, 1.0, self.params["decimate"], length=SL)

        # ---- Packets ----
        r = self._add_separator(f, r)
        r = self._add_section_header(f, r, "packets", "PACKETS")

        ttk.Label(f, text="Packets").grid(row=r, column=0, sticky="w", padx=8)
        self._packets_var = tk.IntVar(value=self.params["packets"])
        pkt_frame = ttk.Frame(f)
        pkt_frame.grid(row=r, column=1, sticky="w")
        for i, name in enumerate(PACKET_NAMES):
            ttk.Radiobutton(pkt_frame, text=name, variable=self._packets_var, value=i).pack(side="left", padx=3)
        self._add_lock(f, r, "packets")
        r += 1

        r = self._add_slider(f, r, "packet_rate", "Pkt Rate", 0.0, 1.0, self.params["packet_rate"], length=SL)
        r = self._add_slider(f, r, "packet_size", "Pkt Size (ms)", 5.0, 200.0, self.params["packet_size"], length=SL)

        # ---- Filter ----
        r = self._add_separator(f, r)
        r = self._add_section_header(f, r, "filter", "FILTER")

        ttk.Label(f, text="Filter").grid(row=r, column=0, sticky="w", padx=8)
        self._filter_var = tk.IntVar(value=self.params["filter_type"])
        fil_frame = ttk.Frame(f)
        fil_frame.grid(row=r, column=1, sticky="w")
        for i, name in enumerate(FILTER_NAMES):
            ttk.Radiobutton(fil_frame, text=name, variable=self._filter_var, value=i).pack(side="left", padx=3)
        self._add_lock(f, r, "filter_type")
        r += 1

        r = self._add_slider(f, r, "filter_freq", "Freq (Hz)", 20.0, 20000.0, self.params["filter_freq"], length=SL, log=True)
        r = self._add_slider(f, r, "filter_width", "Width", 0.0, 1.0, self.params["filter_width"], length=SL)

        ttk.Label(f, text="Slope").grid(row=r, column=0, sticky="w", padx=8)
        self._slope_var = tk.IntVar(value=self.params["filter_slope"])
        sl_frame = ttk.Frame(f)
        sl_frame.grid(row=r, column=1, sticky="w")
        for i, val in enumerate(SLOPE_OPTIONS):
            ttk.Radiobutton(sl_frame, text=f"{val} dB", variable=self._slope_var, value=i).pack(side="left", padx=3)
        self._add_lock(f, r, "filter_slope")
        r += 1

        # ---- Verb / Freeze / Gate ----
        r = self._add_separator(f, r)
        r = self._add_section_header(f, r, "effects", "EFFECTS")

        r = self._add_slider(f, r, "verb", "Verb", 0.0, 1.0, self.params["verb"], length=SL)
        r = self._add_slider(f, r, "decay", "Decay", 0.0, 1.0, self.params["decay"], length=SL)

        ttk.Label(f, text="Verb Pos").grid(row=r, column=0, sticky="w", padx=8)
        self._verb_pos_var = tk.IntVar(value=self.params["verb_position"])
        vp_frame = ttk.Frame(f)
        vp_frame.grid(row=r, column=1, sticky="w")
        for i, name in enumerate(VERB_POSITION_NAMES):
            ttk.Radiobutton(vp_frame, text=name, variable=self._verb_pos_var, value=i).pack(side="left", padx=3)
        self._add_lock(f, r, "verb_position")
        r += 1

        r = self._add_slider(f, r, "gate", "Gate", 0.0, 1.0, self.params["gate"], length=SL)

        self._freeze_var = tk.IntVar(value=self.params["freeze"])
        ttk.Checkbutton(f, text="Freeze", variable=self._freeze_var).grid(row=r, column=0, sticky="w", padx=8)
        self._freeze_mode_var = tk.IntVar(value=self.params["freeze_mode"])
        fm_frame = ttk.Frame(f)
        fm_frame.grid(row=r, column=1, sticky="w")
        for i, name in enumerate(FREEZE_NAMES):
            ttk.Radiobutton(fm_frame, text=name, variable=self._freeze_mode_var, value=i).pack(side="left", padx=3)
        freeze_lock = tk.BooleanVar(value=False)
        self.locks["freeze"] = freeze_lock
        self.locks["freeze_mode"] = freeze_lock
        ttk.Checkbutton(f, variable=freeze_lock).grid(row=r, column=3, sticky="w", padx=2)
        r += 1

        r = self._add_slider(f, r, "freezer", "Freezer", 0.0, 1.0, self.params["freezer"], length=SL)

        # ---- Hidden Options ----
        r = self._add_separator(f, r)
        r = self._add_section_header(f, r, "hidden", "HIDDEN OPTIONS")

        r = self._add_slider(f, r, "threshold", "Threshold", 0.0, 1.0, self.params["threshold"], length=SL)
        r = self._add_slider(f, r, "auto_gain", "Auto Gain", 0.0, 1.0, self.params["auto_gain"], length=SL)
        r = self._add_slider(f, r, "loss_gain", "Loss Gain", 0.0, 1.0, self.params["loss_gain"], length=SL)
        r = self._add_slider(f, r, "weighting", "Weighting", 0.0, 1.0, self.params["weighting"], length=SL)

        # ---- Bounce (parameter modulation) ----
        r = self._add_separator(f, r)
        r = self._add_section_header(f, r, "bounce", "BOUNCE")

        self._bounce_var = tk.IntVar(value=self.params["bounce"])
        ttk.Checkbutton(f, text="Bounce", variable=self._bounce_var).grid(row=r, column=0, sticky="w", padx=8)
        self._add_lock(f, r, "bounce")
        r += 1

        ttk.Label(f, text="Target").grid(row=r, column=0, sticky="w", padx=8)
        self._bounce_target_var = tk.IntVar(value=self.params["bounce_target"])
        bt_frame = ttk.Frame(f)
        bt_frame.grid(row=r, column=1, sticky="w")
        for i, name in enumerate(BOUNCE_TARGET_NAMES):
            ttk.Radiobutton(bt_frame, text=name, variable=self._bounce_target_var, value=i).pack(side="left", padx=2)
        self._add_lock(f, r, "bounce_target")
        r += 1

        r = self._add_slider(f, r, "bounce_rate", "Rate", 0.0, 1.0, self.params["bounce_rate"], length=SL)
        r = self._add_slider(f, r, "bounce_lfo_min", "LFO Min (Hz)", 0.01, 50.0, self.params["bounce_lfo_min"], length=SL)
        r = self._add_slider(f, r, "bounce_lfo_max", "LFO Max (Hz)", 0.01, 50.0, self.params["bounce_lfo_max"], length=SL)

        # ---- Output ----
        r = self._add_separator(f, r)
        r = self._add_section_header(f, r, "output", "OUTPUT")

        r = self._add_slider(f, r, "wet_dry", "Wet / Dry", 0.0, 1.0, self.params["wet_dry"], length=SL)
        r = self._add_slider(f, r, "tail_length", "Tail Length (s)", 0.0, 60.0, 2.0, length=SL)

        # Auto-play on discrete parameter change
        for var in [self._inverse_var, self._quantizer_var, self._packets_var,
                    self._filter_var, self._slope_var, self._freeze_var,
                    self._freeze_mode_var, self._verb_pos_var, self._bounce_var,
                    self._bounce_target_var]:
            var.trace_add("write", lambda *_: self._schedule_autoplay())

        # ============ AI TUNER (responsive right/bottom) ============
        self._ai_chat_frame = ttk.Frame(self._params_ai_container)
        self._build_ai_prompt(self._ai_chat_frame)
        self._ai_layout_mode = None
        self._params_ai_container.bind("<Configure>", self._on_ai_layout_configure)
        self._apply_ai_layout("vertical")

    # ------------------------------------------------------------------
    # Slider helper
    # ------------------------------------------------------------------

    def _add_slider(self, parent, row, key, label, lo, hi, value, length=280, log=False, integer=False):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=4)
        var = tk.DoubleVar(value=value)
        # Frame holding [min_label | scale | max_label]
        slider_frame = ttk.Frame(parent)
        slider_frame.grid(row=row, column=1, sticky="ew", padx=4)
        min_text = self._fmt_val(lo, log, integer)
        max_text = self._fmt_val(hi, log, integer)
        ttk.Label(slider_frame, text=min_text, width=len(min_text)+1,
                  foreground="#888888", font=("Helvetica", 9)).pack(side="left")
        scale = ttk.Scale(slider_frame, from_=lo, to=hi, variable=var,
                          orient="horizontal", length=length)
        scale.pack(side="left", fill="x", expand=True)
        ttk.Label(slider_frame, text=max_text, width=len(max_text)+1,
                  foreground="#888888", font=("Helvetica", 9)).pack(side="left")
        self._bind_scroll(scale, var, lo, hi)
        # Typeable entry field
        entry_var = tk.StringVar(value=self._fmt_val(value, log, integer))
        entry = ttk.Entry(parent, textvariable=entry_var, width=8, justify="right")
        entry.grid(row=row, column=2, sticky="w", padx=2)
        # Sync slider → entry
        def _on_slider_change(*_a, v=var, ev=entry_var, lg=log, intg=integer):
            ev.set(self._fmt_val(v.get(), lg, intg))
        var.trace_add("write", _on_slider_change)
        # Sync entry → slider on Return
        def _on_entry_return(event, v=var, ev=entry_var, _lo=lo, _hi=hi, intg=integer):
            try:
                val = int(ev.get()) if intg else float(ev.get())
                val = max(_lo, min(_hi, val))
                v.set(val)
                self._schedule_autoplay()
            except ValueError:
                ev.set(self._fmt_val(v.get(), False, intg))
        entry.bind("<Return>", _on_entry_return)
        entry.bind("<FocusOut>", _on_entry_return)
        self.sliders[key] = var
        if log:
            var._log = True
        if integer:
            var._integer = True
        lock_var = tk.BooleanVar(value=False)
        self.locks[key] = lock_var
        ttk.Checkbutton(parent, variable=lock_var).grid(row=row, column=3, sticky="w", padx=2)
        scale.bind("<ButtonRelease-1>", lambda e: self._schedule_autoplay())
        return row + 1

    @staticmethod
    def _fmt_val(val, log=False, integer=False):
        if integer:
            return f"{int(round(val))}"
        if log:
            return f"{val:.0f}"
        if abs(val) >= 100:
            return f"{val:.1f}"
        if abs(val) >= 10:
            return f"{val:.2f}"
        return f"{val:.3f}"

    def _bind_scroll(self, scale, var, lo, hi):
        self._scroll_widgets[str(scale)] = (var, lo, hi)

    def _on_global_scroll(self, event):
        """Route mousewheel/touchpad scroll to the Scale widget under the cursor."""
        w = self.root.winfo_containing(event.x_root, event.y_root)
        if w is None:
            return
        path = str(w)
        if path not in self._scroll_widgets:
            return
        var, lo, hi = self._scroll_widgets[path]
        raw = event.delta if event.delta else (-1 if event.num == 5 else 1)
        # Tk 9.0 TouchpadScroll packs X,Y into one int: Y in low 16 bits (signed)
        y = raw & 0xFFFF
        if y >= 0x8000:
            y -= 0x10000
        step = (hi - lo) * 0.002
        var.set(max(lo, min(hi, var.get() + y * step)))
        self._schedule_autoplay()

    # ------------------------------------------------------------------
    # Section header / lock / randomize helpers
    # ------------------------------------------------------------------

    def _add_section_header(self, parent, row, section_key, title):
        hdr = ttk.Frame(parent)
        hdr.grid(row=row, column=0, columnspan=4, sticky="w", padx=8, pady=(8, 2))
        ttk.Label(hdr, text=title, font=("Helvetica", 11, "bold")).pack(side="left")
        lock_var = tk.BooleanVar(value=False)
        self.section_locks[section_key] = lock_var
        ttk.Checkbutton(
            hdr, text="Lock All", variable=lock_var,
            command=lambda s=section_key: self._toggle_section_lock(s)
        ).pack(side="left", padx=(12, 0))
        return row + 1

    @staticmethod
    def _add_separator(parent, row):
        ttk.Separator(parent, orient="horizontal").grid(
            row=row, column=0, columnspan=4, sticky="ew", padx=8, pady=6)
        return row + 1

    def _add_lock(self, parent, row, key):
        lock_var = tk.BooleanVar(value=False)
        self.locks[key] = lock_var
        ttk.Checkbutton(parent, variable=lock_var).grid(row=row, column=3, sticky="w", padx=2)

    def _toggle_section_lock(self, section_key):
        locked = self.section_locks[section_key].get()
        for param_key in PARAM_SECTIONS[section_key]:
            if param_key in self.locks:
                self.locks[param_key].set(locked)

    def _schedule_autoplay(self):
        if self._autoplay_id is not None:
            self.root.after_cancel(self._autoplay_id)
        self._autoplay_id = self.root.after(400, self._on_play)

    # ------------------------------------------------------------------
    # AI Tuner
    # ------------------------------------------------------------------

    def _build_ai_prompt(self, parent):
        """Build the AI tuner widget group — output on top, input at bottom."""
        ttk.Label(parent, text="AI Tuner", font=("Helvetica", 11, "bold")).pack(anchor="w", padx=5, pady=(5, 2))

        # Response/output area (top, expands to fill available space)
        resp_frame = ttk.Frame(parent)
        resp_frame.pack(fill="both", expand=True, padx=5, pady=(2, 3))
        resp_scroll = ttk.Scrollbar(resp_frame, orient="vertical")
        self.ai_response = tk.Text(resp_frame, height=10, wrap="word", state="disabled",
                                   bg="#222233", fg="#dddddd", font=("Helvetica", 10),
                                   relief="solid", bd=1,
                                   yscrollcommand=resp_scroll.set)
        resp_scroll.configure(command=self.ai_response.yview)
        self.ai_response.pack(side="left", fill="both", expand=True)
        resp_scroll.pack(side="right", fill="y")
        self.ai_response.tag_configure("user_label", foreground="#66ccff", font=("Helvetica", 10, "bold"))
        self.ai_response.tag_configure("user_msg", foreground="#aaccdd")
        self.ai_response.tag_configure("asst_label", foreground="#bb88ff", font=("Helvetica", 10, "bold"))
        self.ai_response.tag_configure("system_notice", foreground="#888899", font=("Helvetica", 9, "italic"))

        # Buttons
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill="x", padx=5, pady=(0, 3))
        self.ai_ask_btn = ttk.Button(btn_frame, text="Ask Claude", command=self._on_ask_claude)
        self.ai_ask_btn.pack(side="left", padx=(0, 4))
        ttk.Button(btn_frame, text="Undo", command=self._on_ai_undo).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="New Session", command=self._on_ai_new_session).pack(side="left", padx=2)
        self.ai_stop_btn = ttk.Button(btn_frame, text="Stop Tuning",
                                       command=self._on_stop_tuning, state="disabled")
        self.ai_stop_btn.pack(side="left", padx=2)
        ttk.Label(btn_frame, text="Enter to send", foreground="#666688",
                  font=("Helvetica", 9)).pack(side="left", padx=8)
        self.ai_status_var = tk.StringVar(value="")
        ttk.Label(btn_frame, textvariable=self.ai_status_var).pack(side="left", padx=4)

        # Input area (bottom, 4 lines)
        input_frame = ttk.Frame(parent)
        input_frame.pack(fill="x", padx=5, pady=(0, 5))
        input_scroll = ttk.Scrollbar(input_frame, orient="vertical")
        self.ai_input = tk.Text(input_frame, height=4, wrap="word",
                                bg="#2a2a3e", fg="#eeeeee", insertbackground="#eeeeee",
                                font=("Helvetica", 11), relief="solid", bd=1,
                                yscrollcommand=input_scroll.set)
        input_scroll.configure(command=self.ai_input.yview)
        self.ai_input.pack(side="left", fill="both", expand=True)
        input_scroll.pack(side="right", fill="y")
        self.ai_input.bind("<Return>", self._on_ai_return)
        self.ai_input.bind("<Shift-Return>", self._on_ai_shift_return)

    def _on_ai_return(self, event):
        """Send on Return and prevent newline insertion."""
        self._on_ask_claude()
        return "break"

    def _on_ai_shift_return(self, event):
        """Insert newline on Shift+Return."""
        self.ai_input.insert("insert", "\n")
        return "break"

    def _on_ai_layout_configure(self, event):
        """Switch between horizontal and vertical AI layout based on width."""
        width = event.width
        if width > 1200 and self._ai_layout_mode != "horizontal":
            self._apply_ai_layout("horizontal")
        elif width <= 1200 and self._ai_layout_mode != "vertical":
            self._apply_ai_layout("vertical")

    def _apply_ai_layout(self, mode):
        """Apply the given layout mode for params + AI chat."""
        self._ai_layout_mode = mode
        self._params_scroll_frame.grid_forget()
        self._ai_chat_frame.grid_forget()
        if mode == "horizontal":
            self._params_ai_container.columnconfigure(0, weight=4, minsize=400)
            self._params_ai_container.columnconfigure(1, weight=1, minsize=280)
            self._params_ai_container.rowconfigure(0, weight=1)
            self._params_ai_container.rowconfigure(1, weight=0)
            self._params_scroll_frame.grid(row=0, column=0, sticky="nsew")
            self._ai_chat_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        else:
            self._params_ai_container.columnconfigure(0, weight=1)
            self._params_ai_container.columnconfigure(1, weight=0)
            self._params_ai_container.rowconfigure(0, weight=1)
            self._params_ai_container.rowconfigure(1, weight=0)
            self._params_scroll_frame.grid(row=0, column=0, sticky="nsew")
            self._ai_chat_frame.grid(row=1, column=0, sticky="ew")
            self.ai_response.configure(height=10)

    def _on_ask_claude(self):
        prompt = self.ai_input.get("1.0", "end-1c").strip()
        if not prompt:
            return
        self.ai_status_var.set("Thinking...")
        self.ai_ask_btn.configure(state="disabled")
        self.ai_stop_btn.configure(state="normal")
        # Show user message in response box
        self.ai_response.configure(state="normal")
        self.ai_response.insert("end", "\nYou: ", "user_label")
        self.ai_response.insert("end", f"{prompt}\n\n", "user_msg")
        self.ai_response.configure(state="disabled")
        self.ai_input.delete("1.0", "end")
        self._ai_needs_label = True
        current = self._read_params_from_ui()
        self.llm.send_prompt(
            prompt, current,
            on_text=self._on_claude_text,
            on_params=self._on_claude_params,
            on_done=self._on_claude_done,
            on_error=self._on_claude_error,
            on_iterate=self._on_claude_iterate,
            metrics=self.rendered_metrics,
            source_metrics=getattr(self, 'source_metrics', None),
            spectrogram_png=getattr(self, 'rendered_spectrogram', None),
            source_spectrogram_png=getattr(self, 'source_spectrogram', None),
        )

    def _on_claude_text(self, text):
        self.ai_response.configure(state="normal")
        if self._ai_needs_label:
            self.ai_response.insert("end", "Claude: ", "asst_label")
            self._ai_needs_label = False
        self.ai_response.insert("end", text)
        self.ai_response.see("end")
        self.ai_response.configure(state="disabled")
        self.ai_status_var.set("")

    def _on_claude_params(self, merged_params):
        self._write_params_to_ui(merged_params)
        self._pending_gen_notice = True
        self.ai_status_var.set("Applied params")
        self._on_play()

    def _on_claude_done(self):
        self.ai_response.configure(state="normal")
        self.ai_response.insert("end", "\n")
        self.ai_response.configure(state="disabled")
        self.ai_ask_btn.configure(state="normal")
        self.ai_stop_btn.configure(state="disabled")

    def _on_claude_error(self, error_msg):
        self.ai_status_var.set(f"Error: {error_msg[:80]}")
        self.ai_ask_btn.configure(state="normal")
        self.ai_stop_btn.configure(state="disabled")

    def _on_claude_iterate(self, merged_params):
        """Claude requested iteration — render and play."""
        self._write_params_to_ui(merged_params)
        self._pending_gen_notice = True
        self.ai_status_var.set(f"Tuning... iteration {self.llm._iterate_count}/{self.llm.MAX_ITERATE}")

        def on_render_done():
            self.llm.continue_iteration(
                current_params=self._read_params_from_ui(),
                metrics=self.rendered_metrics,
                spectrogram_png=getattr(self, 'rendered_spectrogram', None),
            )

        sd.stop()
        self._stop_cursor()
        self._render_and_play(merged_params, on_complete=on_render_done,
                              status_prefix="Tuning... iteration")

    def _on_stop_tuning(self):
        self.llm.stop_iterating()
        self.ai_status_var.set("Stopping after current render...")

    def _on_ai_undo(self):
        prev = self.llm.undo()
        if prev:
            self._write_params_to_ui(prev)
            self.ai_status_var.set("Reverted")
            self._on_play()

    def _on_ai_new_session(self):
        self.llm.reset_session()
        self.ai_response.configure(state="normal")
        self.ai_response.delete("1.0", "end")
        self.ai_response.configure(state="disabled")
        self.ai_status_var.set("New session")

    def _reset_params(self):
        self._write_params_to_ui(bypass_params())
        self._on_play()

    def _check_autoplay(self):
        """Re-render if params changed during a render."""
        params = self._read_params_from_ui()
        if self._params_changed(params):
            self._on_play()

    def _randomize_params(self):
        # Continuous params
        for key, (lo, hi) in PARAM_RANGES.items():
            if key in self.locks and self.locks[key].get():
                continue
            if key not in self.sliders:
                continue
            if hasattr(self.sliders[key], '_integer') and self.sliders[key]._integer:
                val = random.randint(int(lo), int(hi))
            elif hasattr(self.sliders[key], '_log') and self.sliders[key]._log:
                val = 10 ** random.uniform(math.log10(max(lo, 1e-10)),
                                           math.log10(max(hi, 1e-10)))
            else:
                val = random.uniform(lo, hi)
            self.sliders[key].set(val)
        # Integer/choice params
        var_map = {
            "inverse": self._inverse_var,
            "quantizer": self._quantizer_var,
            "packets": self._packets_var,
            "filter_type": self._filter_var,
            "filter_slope": self._slope_var,
            "freeze": self._freeze_var,
            "freeze_mode": self._freeze_mode_var,
            "verb_position": self._verb_pos_var,
            "bounce": self._bounce_var,
            "bounce_target": self._bounce_target_var,
        }
        for key, num_choices in CHOICE_RANGES.items():
            if key in self.locks and self.locks[key].get():
                continue
            if key in var_map:
                var_map[key].set(random.randint(0, num_choices - 1))
        self._on_play()

    # ------------------------------------------------------------------
    # Presets tab
    # ------------------------------------------------------------------

    def _build_presets_page(self):
        f = self.presets_frame

        # Search box
        search_frame = ttk.Frame(f)
        search_frame.pack(fill="x", padx=8, pady=(8, 0))
        ttk.Label(search_frame, text="Search:").pack(side="left", padx=(0, 4))
        self._preset_search_var = tk.StringVar()
        self._preset_search_var.trace_add("write", lambda *_: self._refresh_preset_list())
        ttk.Entry(search_frame, textvariable=self._preset_search_var).pack(side="left", fill="x", expand=True)

        # Left: Treeview grouped by category
        tree_frame = ttk.Frame(f)
        tree_frame.pack(side="left", fill="both", expand=True, padx=8, pady=8)

        self.preset_tree = ttk.Treeview(tree_frame, columns=("desc", "name"),
                                         show="tree headings",
                                         selectmode="browse", height=20)
        self.preset_tree.heading("#0", text="Preset", anchor="w")
        self.preset_tree.heading("desc", text="Description", anchor="w")
        self.preset_tree.column("#0", width=200, minwidth=120)
        self.preset_tree.column("desc", width=420, minwidth=200)
        self.preset_tree.column("name", width=0, stretch=False)  # hidden data column
        tree_scroll = ttk.Scrollbar(tree_frame, orient="vertical",
                                     command=self.preset_tree.yview)
        self.preset_tree.configure(yscrollcommand=tree_scroll.set)
        self.preset_tree.pack(side="left", fill="both", expand=True)
        tree_scroll.pack(side="right", fill="y")
        self.preset_tree.bind("<Double-1>", lambda e: self._on_load_preset(play=True))
        self.preset_tree.bind("<Return>", lambda e: self._on_load_preset(play=True))

        # Right: buttons + description
        right = ttk.Frame(f)
        right.pack(side="left", fill="y", padx=8, pady=8)

        btn = ttk.Frame(right)
        btn.pack(fill="x")
        ttk.Button(btn, text="Load", command=self._on_load_preset).pack(fill="x", pady=2)
        ttk.Button(btn, text="Save", command=self._on_save_preset).pack(fill="x", pady=2)
        ttk.Button(btn, text="Delete", command=self._on_delete_preset).pack(fill="x", pady=2)
        ttk.Button(btn, text="Refresh", command=self._refresh_preset_list).pack(fill="x", pady=2)
        ttk.Button(btn, text="★ Favorite", command=self._on_toggle_favorite).pack(fill="x", pady=2)

        self._preset_meta = {}  # name -> {category, description}
        self._refresh_preset_list()

    def _refresh_preset_list(self):
        self.preset_tree.delete(*self.preset_tree.get_children())
        self._preset_meta.clear()
        os.makedirs(PRESET_DIR, exist_ok=True)
        favorites = self._load_favorites()

        # Read all presets and group by category
        categories = {}
        for filename in sorted(os.listdir(PRESET_DIR)):
            if not filename.endswith(".json") or filename == "favorites.json":
                continue
            name = filename[:-5]
            path = os.path.join(PRESET_DIR, filename)
            try:
                with open(path) as fh:
                    data = json.load(fh)
            except (json.JSONDecodeError, OSError):
                continue
            meta = data.get("_meta", {})
            cat = meta.get("category", "Uncategorized")
            desc = meta.get("description", "")
            self._preset_meta[name] = {"category": cat, "description": desc}
            categories.setdefault(cat, []).append(name)

        # Filter by search query
        query = ""
        if hasattr(self, "_preset_search_var"):
            query = self._preset_search_var.get().strip().lower()

        def _matches(name):
            if not query:
                return True
            meta = self._preset_meta.get(name, {})
            return query in name.lower() or query in meta.get("description", "").lower()

        # Favorites group at top (only if any favorites exist among loaded presets)
        fav_names = sorted(n for n in favorites if n in self._preset_meta and _matches(n))
        if fav_names:
            fav_id = self.preset_tree.insert("", "end", text="★ Favorites", open=True,
                                              values=("", ""))
            for name in fav_names:
                desc = self._preset_meta[name].get("description", "")
                self.preset_tree.insert(fav_id, "end", text=f"★ {name}",
                                         values=(desc, name))

        # Category display order
        cat_order = ["Codec", "Communication", "Lo-fi", "Textural",
                     "Ghost / Residue", "Glitch", "Modulated",
                     "Sound Design", "Uncategorized"]
        seen = set()
        ordered_cats = []
        for c in cat_order:
            if c in categories:
                ordered_cats.append(c)
                seen.add(c)
        for c in sorted(categories.keys()):
            if c not in seen:
                ordered_cats.append(c)

        for cat in ordered_cats:
            filtered = [n for n in categories[cat] if _matches(n)]
            if not filtered:
                continue
            cat_id = self.preset_tree.insert("", "end", text=cat, open=True,
                                              values=("", ""))
            for name in filtered:
                desc = self._preset_meta[name].get("description", "")
                display = f"★ {name}" if name in favorites else name
                self.preset_tree.insert(cat_id, "end", text=display,
                                         values=(desc, name))

    def _load_favorites(self):
        path = os.path.join(PRESET_DIR, "favorites.json")
        try:
            with open(path) as fh:
                return set(json.load(fh))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return set()

    def _save_favorites(self, favorites):
        path = os.path.join(PRESET_DIR, "favorites.json")
        os.makedirs(PRESET_DIR, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(sorted(favorites), fh, indent=2)
            fh.write("\n")

    def _on_toggle_favorite(self):
        name = self._get_selected_preset_name()
        if not name:
            return
        favorites = self._load_favorites()
        if name in favorites:
            favorites.discard(name)
            self.status_var.set(f"Removed from favorites: {name}")
        else:
            favorites.add(name)
            self.status_var.set(f"Added to favorites: {name}")
        self._save_favorites(favorites)
        self._refresh_preset_list()

    def _get_selected_preset_name(self):
        sel = self.preset_tree.selection()
        if not sel:
            return None
        item = sel[0]
        vals = self.preset_tree.item(item, "values")
        # name is in the second (hidden) values column
        name = vals[1] if vals and len(vals) > 1 else ""
        if not name or name not in self._preset_meta:
            return None
        return name

    def _on_load_preset(self, play=False):
        name = self._get_selected_preset_name()
        if not name:
            return
        path = os.path.join(PRESET_DIR, name + ".json")
        with open(path) as fh:
            p = json.load(fh)
        p.pop("_meta", None)
        migrate_legacy_params(p)
        full = default_params()
        full.update(p)
        self._write_params_to_ui(full)
        self.status_var.set(f"Loaded preset: {name}")
        if play:
            self._on_play()

    def _on_save_preset(self):
        save_win = tk.Toplevel(self.root)
        save_win.title("Save Preset")
        save_win.geometry("360x200")
        save_win.transient(self.root)
        save_win.grab_set()

        ttk.Label(save_win, text="Name:").grid(row=0, column=0, sticky="w", padx=8, pady=4)
        name_var = tk.StringVar()
        ttk.Entry(save_win, textvariable=name_var, width=30).grid(row=0, column=1, padx=8, pady=4)

        ttk.Label(save_win, text="Category:").grid(row=1, column=0, sticky="w", padx=8, pady=4)
        cat_var = tk.StringVar(value="Uncategorized")
        cat_choices = ["Codec", "Communication", "Lo-fi", "Textural",
                       "Ghost / Residue", "Glitch", "Modulated", "Sound Design"]
        ttk.Combobox(save_win, textvariable=cat_var, values=cat_choices,
                     width=28).grid(row=1, column=1, padx=8, pady=4)

        ttk.Label(save_win, text="Description:").grid(row=2, column=0, sticky="nw", padx=8, pady=4)
        desc_text = tk.Text(save_win, width=30, height=3, wrap="word")
        desc_text.grid(row=2, column=1, padx=8, pady=4)

        def do_save():
            n = name_var.get().strip()
            if not n:
                return
            params = self._read_params_from_ui()
            params["_meta"] = {
                "category": cat_var.get(),
                "description": desc_text.get("1.0", "end").strip(),
            }
            os.makedirs(PRESET_DIR, exist_ok=True)
            path = os.path.join(PRESET_DIR, n + ".json")
            with open(path, "w") as fh:
                json.dump(params, fh, indent=2)
                fh.write("\n")
            self._refresh_preset_list()
            self.status_var.set(f"Saved preset: {n}")
            save_win.destroy()

        ttk.Button(save_win, text="Save", command=do_save).grid(
            row=3, column=1, sticky="e", padx=8, pady=8)

    def _on_delete_preset(self):
        name = self._get_selected_preset_name()
        if not name:
            return
        path = os.path.join(PRESET_DIR, name + ".json")
        if os.path.isfile(path):
            os.remove(path)
        self._refresh_preset_list()

    # ------------------------------------------------------------------
    # Waveform tab
    # ------------------------------------------------------------------

    def _build_waveforms_page(self):
        """Combined input + output waveforms, stacked vertically."""
        parent = self.waveforms_frame
        ttk.Label(parent, text="Input", font=("Helvetica", 9, "bold"),
                  foreground="#888").pack(anchor="w")
        self.input_wave_canvas = tk.Canvas(parent, bg="#111118", highlightthickness=0)
        self.input_wave_canvas.pack(fill="both", expand=True, pady=(0, 3))
        self.input_wave_canvas.bind("<Configure>", lambda e: self._draw_input_waveform())
        self.input_wave_canvas.bind("<Button-1>", self._on_input_seek)

        ttk.Label(parent, text="Output", font=("Helvetica", 9, "bold"),
                  foreground="#888").pack(anchor="w")
        self.wave_canvas = tk.Canvas(parent, bg="#111118", highlightthickness=0)
        self.wave_canvas.pack(fill="both", expand=True)
        self.wave_canvas.bind("<Configure>", lambda e: self._draw_waveform())
        self.wave_canvas.bind("<Button-1>", self._on_output_seek)

    def _build_spectrograms_page(self):
        """Side-by-side input + output spectrograms drawn from audio data."""
        parent = self.spectrograms_frame
        self._spec_images = {}  # keep PhotoImage references to prevent GC
        row = ttk.Frame(parent)
        row.pack(fill="both", expand=True)
        row.columnconfigure(0, weight=1)
        row.columnconfigure(1, weight=1)
        row.rowconfigure(0, weight=1)

        self.spec_input_canvas = tk.Canvas(row, bg="#111118", highlightthickness=0)
        self.spec_input_canvas.grid(row=0, column=0, sticky="nsew", padx=(0, 2))
        self.spec_output_canvas = tk.Canvas(row, bg="#111118", highlightthickness=0)
        self.spec_output_canvas.grid(row=0, column=1, sticky="nsew", padx=(2, 0))

        self.spec_input_canvas.bind("<Configure>", lambda e: self._draw_spectrograms())
        self.spec_output_canvas.bind("<Configure>", lambda e: self._draw_spectrograms())
        self.spec_input_canvas.bind("<Button-1>", self._on_input_seek)
        self.spec_output_canvas.bind("<Button-1>", self._on_output_seek)

    @staticmethod
    def _to_mono(audio):
        if audio is not None and audio.ndim == 2:
            return audio.mean(axis=1)
        return audio

    def _draw_input_waveform(self):
        _shared_draw_waveform(self.input_wave_canvas, self.source_audio, SR,
                              getattr(self, 'source_metrics', None), "Input")

    def _draw_waveform(self):
        _shared_draw_waveform(self.wave_canvas, self.rendered_audio, SR,
                              self.rendered_metrics, "Output", self.rendered_warning)

    def _draw_spectrograms(self):
        """Draw spectrograms computed from audio data."""
        draw_spectrogram(self.spec_input_canvas, self.source_audio, SR,
                         "Input", self._spec_images)
        draw_spectrogram(self.spec_output_canvas, self.rendered_audio, SR,
                         "Output", self._spec_images)

    # ------------------------------------------------------------------
    # Spectrum tab
    # ------------------------------------------------------------------

    def _build_spec_page(self):
        self.spec_canvas = tk.Canvas(self.spec_frame, bg="#1a1a2e", highlightthickness=0)
        self.spec_canvas.pack(fill="both", expand=True)
        self.spec_canvas.bind("<Configure>", lambda e: self._draw_spectrum())

    def _draw_spectrum(self):
        c = self.spec_canvas
        c.delete("all")
        w = c.winfo_width()
        h = c.winfo_height()
        if w < 10 or h < 10:
            return

        pad_bottom = 30
        pad_top = 20
        plot_h = h - pad_bottom - pad_top

        if self.rendered_audio is None and self.source_audio is None:
            c.create_text(w // 2, h // 2, text="No audio", fill="#666688", font=("Helvetica", 14))
            return

        fft_size = 4096

        def calc_spectrum(audio):
            if audio is None:
                return None, None
            mono = self._to_mono(audio)
            if len(mono) < fft_size:
                return None, None
            # Use a chunk from the middle of the audio
            mid = len(mono) // 2
            start = max(0, mid - fft_size // 2)
            chunk = mono[start : start + fft_size]
            windowed = chunk * np.hanning(len(chunk))
            spec = np.abs(np.fft.rfft(windowed))
            spec[spec < 1e-12] = 1e-12
            db = 20.0 * np.log10(spec)
            freqs = np.fft.rfftfreq(len(chunk), 1.0 / SR)
            return freqs, db

        freqs_dry, db_dry = calc_spectrum(self.source_audio)
        freqs_wet, db_wet = calc_spectrum(self.rendered_audio)

        # dB range
        db_min, db_max = -90.0, 0.0
        if db_wet is not None:
            db_max = max(db_max, np.max(db_wet) + 6)
        if db_dry is not None:
            db_max = max(db_max, np.max(db_dry) + 6)

        def freq_to_x(f):
            if f <= 20:
                return 0
            return int(np.log10(f / 20.0) / np.log10(SR / 2.0 / 20.0) * w)

        def db_to_y(d):
            frac = (d - db_min) / (db_max - db_min)
            return int(pad_top + plot_h * (1.0 - np.clip(frac, 0, 1)))

        # Grid lines
        for f in [100, 1000, 10000]:
            x = freq_to_x(f)
            c.create_line(x, pad_top, x, h - pad_bottom, fill="#222244")
            c.create_text(x, h - pad_bottom + 4, anchor="n", fill="#666688",
                           text=f"{f}" if f < 1000 else f"{f // 1000}k")

        for db in range(-80, 1, 20):
            y = db_to_y(db)
            c.create_line(0, y, w, y, fill="#222244")
            c.create_text(4, y - 2, anchor="sw", fill="#555577", text=f"{db} dB")

        def draw_curve(freqs, db, color, tag):
            if freqs is None:
                return
            points = []
            step = max(1, len(freqs) // w)
            for i in range(0, len(freqs), step):
                x = freq_to_x(freqs[i])
                y = db_to_y(db[i])
                points.append(x)
                points.append(y)
            if len(points) >= 4:
                c.create_line(points, fill=color, width=1, smooth=True, tags=tag)

        draw_curve(freqs_dry, db_dry, "#555577", "dry")
        draw_curve(freqs_wet, db_wet, "#ff6644", "wet")

        # Legend
        c.create_text(w - 8, pad_top + 4, anchor="ne", fill="#555577", text="Dry")
        c.create_text(w - 8, pad_top + 18, anchor="ne", fill="#ff6644", text="Wet")

    # ------------------------------------------------------------------
    # Guide tab
    # ------------------------------------------------------------------

    def _build_guide_page(self):
        text = tk.Text(self.guide_frame, wrap="word", bg="#1a1a2e", fg="#ccccdd",
                       font=("Helvetica", 12), padx=16, pady=12, relief="flat",
                       selectbackground="#334466", insertbackground="#ccccdd")
        text.pack(fill="both", expand=True)

        # Tag styles
        text.tag_configure("h1", font=("Helvetica", 16, "bold"), foreground="#ffffff",
                           spacing3=6)
        text.tag_configure("h2", font=("Helvetica", 13, "bold"), foreground="#ff9966",
                           spacing1=14, spacing3=4)
        text.tag_configure("pedal", font=("Helvetica", 12, "bold"), foreground="#66bbff")
        text.tag_configure("arrow", foreground="#666688")
        text.tag_configure("gui", font=("Helvetica", 12, "bold"), foreground="#88ff88")
        text.tag_configure("dim", foreground="#777799")
        text.tag_configure("body", foreground="#ccccdd", spacing1=2)

        def h1(s):
            text.insert("end", s + "\n", "h1")

        def h2(s):
            text.insert("end", s + "\n", "h2")

        def mapping(pedal, gui, note=""):
            text.insert("end", "  " + pedal, "pedal")
            text.insert("end", "  ->  ", "arrow")
            text.insert("end", gui, "gui")
            if note:
                text.insert("end", "   " + note, "dim")
            text.insert("end", "\n")

        def body(s):
            text.insert("end", s + "\n", "body")

        h1("Chase Bliss Lossy  ->  This GUI")
        body("")

        h2("Core Controls")
        mapping("Loss knob", "Loss",
                "Spectral quantization depth + psychoacoustic band gating intensity.")
        mapping("Speed knob", "Window Size",
                "FFT window size in samples. Large (4096) = dark smear, small (256) = garbled.")
        mapping("(new)", "Hop Divisor",
                "Overlap ratio = 1/hop_divisor. 4=75% overlap (default), 2=50%, 8=87.5%.")
        mapping("(new)", "Bands",
                "Number of Bark-like bands for gating. Fewer = coarser, more dramatic.")
        mapping("Global knob", "Global",
                "Master intensity — scales Loss, Phase, Crush, Packets, Verb, Gate together.")
        body("")

        h2("Inverse + Jitter (independent controls)")
        mapping("Standard/Inverse", "Inverse checkbox",
                "Off = hear processed signal. On = hear the residual (everything Standard discards).")
        mapping("Jitter", "Jitter slider",
                "Random phase perturbation per FFT bin (0=off, 1=max). Emulates bad digital clocking.")
        body("")
        body("Inverse and Jitter are independent — you can combine them (e.g. inverse+jitter).")
        body('The "underwater" warble comes from zeroing different bands each frame.')
        body("Band gating is weighted by signal energy and the ATH (Absolute Threshold")
        body("of Hearing) curve — quieter bands at less-sensitive frequencies get gated")
        body("first, matching how real codecs run out of bits.")
        body("")

        h2("Advanced Spectral Controls")
        mapping("(not on pedal)", "Phase",
                "Quantize phase angles to N levels. 0=off, high=metallic/robotic.")
        mapping("(codec internals)", "Quantizer: Uniform / Compand",
                "Uniform = classic. Compand = MP3-style power-law (|x|^0.75).")
        mapping("(codec artifact)", "Pre-Echo",
                "Boost loss before transients, spreading noise ahead of attacks.")
        mapping("(new)", "Transient Thr",
                "Energy ratio for pre-echo detection. Lower = more sensitive to transients.")
        mapping("(codec internals)", "Noise Shape",
                "Coarser quantization in spectral valleys, finer near peaks.")
        mapping("(new)", "HF Threshold",
                "Loss level where HF rolloff begins. 0=always roll off, 1=never. Default 0.3.")
        mapping("(new)", "Slushy Rate",
                "Freeze slushy drift speed. Low = slow morphing, high = tracks input closely.")
        body("")

        h2("Crush (time-domain degradation)")
        mapping("(not on pedal)", "Crush",
                "Bitcrusher — reduces amplitude quantization levels (16-bit down to ~4-bit).")
        mapping("(not on pedal)", "Decimate",
                "Sample rate reducer — zero-order hold. Aliasing creates metallic overtones.")
        body("")
        body("These are complementary to spectral loss. Crush creates amplitude staircase")
        body("distortion; Decimate creates inharmonic aliasing. Neither sounds like a codec —")
        body("they sound like early digital hardware (SP-1200, Fairlight, NES).")
        body("")

        h2("Packets Toggle (3-way)")
        mapping("Clean", "Packets: Clean",
                "No packet processing.")
        mapping("Packet Loss", "Packets: Packet Loss",
                "Gilbert-Elliott dropout model — bursty silence gaps with crossfade.")
        mapping("Packet Repeat", "Packets: Packet Repeat",
                "Fills dropout gaps with the last good packet (stutter/glitch).")
        body("")
        body("Pkt Rate = probability of entering the bad state.  Pkt Size = chunk length in ms.")
        body("Hann crossfades at packet boundaries prevent clicks.")
        body("")

        h2("Filter Section")
        mapping("Filter knob", "Width",
                "0=narrow (high Q, resonant peak) to 1=wide (gentle shape).")
        mapping("Freq knob", "Freq",
                "Center frequency, 20 Hz to 20 kHz.")
        mapping("Filter toggle", "Filter: Bypass / Bandpass / Notch")
        mapping("Slope (6/24/96)", "Slope radio buttons",
                "Cascaded biquads: 1/2/8 sections. Higher slope = more resonance.")
        body("")

        h2("Effects")
        mapping("Verb knob", "Verb",
                "Lo-fi Schroeder reverb — short combs + allpass. Deliberately cheap & metallic.")
        mapping("Decay (hidden)", "Decay",
                "Reverb size/length. 0=short metallic, 1=long wash.")
        mapping("Verb Pre/Post dip", "Verb Pos: Pre / Post",
                "Pre = verb before loss (default, PDF p.27). Post = verb after filter.")
        mapping("Freeze footswitch", "Freeze checkbox",
                "Captures a spectral snapshot and holds it.")
        mapping("Freeze: slushy", "Freeze Mode: Slushy",
                "Frozen spectrum slowly updates at Slushy Rate.")
        mapping("Freeze: solid", "Freeze Mode: Solid",
                "Spectrum is truly frozen — static drone.")
        mapping("Freezer (hidden)", "Freezer",
                "Blend between frozen spectrum and live signal. 1=frozen, 0=live.")
        mapping("Gate", "Gate",
                "RMS noise gate. Cleans up residual artifacts in quiet passages.")
        body("")

        h2("Hidden Options (PDF pp.14-16)")
        mapping("Threshold (hidden)", "Threshold",
                "Limiter threshold — 0=heavy limiting, 1=light. Lower = more compression.")
        mapping("Auto Gain (hidden)", "Auto Gain",
                "Loudness compensation for Loss modes. Keeps volume consistent as loss increases.")
        mapping("Loss Gain (hidden)", "Loss Gain",
                "Wet signal volume. 0=-36dB, 0.5=0dB (unity), 1=+36dB boost.")
        mapping("Weighting (hidden)", "Weighting",
                "0=equal freq weighting, 1=psychoacoustic ATH model. Favors some freqs over others.")
        body("")

        h2("Bounce (PDF pp.34-35)")
        mapping("Ramping", "Bounce checkbox",
                "Enables continuous LFO modulation of a chosen parameter.")
        mapping("Bounce target", "Target radio buttons",
                "Which parameter the LFO modulates: Loss, Window, Crush, etc.")
        mapping("Bounce rate", "Rate",
                "LFO speed. Maps 0-1 to LFO Min..LFO Max Hz range.")
        mapping("(new)", "LFO Min / LFO Max",
                "Hz range for bounce LFO. Default 0.1-5.0 Hz.")
        body("")

        h2("Output")
        mapping("(pedal is 100% wet)", "Wet / Dry",
                "0=original signal, 1=fully processed.")
        body("")

        h2("Signal Chain")
        body("PRE mode (default):  Input -> Verb -> Spectral Loss -> Auto Gain")
        body("  -> Loss Gain -> Crush/Decimate -> Packets -> Filter -> Gate -> Limiter -> Mix")
        body("")
        body("POST mode:  Input -> Spectral Loss -> Auto Gain -> Loss Gain")
        body("  -> Crush/Decimate -> Packets -> Filter -> Verb -> Gate -> Limiter -> Mix")
        body("")

        h2("What's Happening Inside")
        body("1. Audio is windowed (Hann) and transformed to frequency domain via FFT.")
        body("2. Magnitudes are quantized (Uniform or Compand). Noise Shape varies the")
        body("   step size per bin — coarser in spectral valleys, finer near peaks.")
        body("3. N Bark-like bands (Bands param, default 21) are gated using psychoacoustic")
        body("   masking: bands with less energy at ATH-insensitive frequencies are dropped first. Random")
        body("   perturbation ensures the frame-to-frame variation that creates the warble.")
        body("4. Phase is optionally quantized to N levels (Phase slider).")
        body("5. Pre-Echo boosts loss on frames preceding transients, spreading noise ahead.")
        body("6. HF bandwidth is rolled off when Loss > HF Threshold (like low-bitrate MP3).")
        body("7. IFFT + overlap-add (overlap set by Hop Divisor) reconstructs audio.")
        body("8. Crush/Decimate, Packets, Filter, Verb, Gate, Limiter in time domain.")
        body("")
        body("Window Size is now direct (in samples). Common values: 4096 (93ms) / 2048 (46ms)")
        body("  / 1024 (23ms) / 512 (12ms) / 256 (6ms). Any value 64-16384 is valid.")
        body("")

        h2("Presets to Try")
        body("  underwater         Heavy loss, slow window — classic codec-degraded sound")
        body("  low_bitrate_mp3    Moderate loss — sounds like a 64kbps MP3")
        body("  spectral_residue   Inverse mode — ghostly harmonics the codec discards")
        body("  bad_connection     Packet loss + bandpass — choppy VoIP call")
        body("  glitch_stutter     Packet repeat — rhythmic buffer-freeze glitches")
        body("  frozen_pad         Freeze + verb — evolving spectral drone")
        body("  resonant_telephone Bandpass 96dB slope + gate — lo-fi phone line")

        text.config(state="disabled")

    # ------------------------------------------------------------------
    # Param read / write
    # ------------------------------------------------------------------

    def _read_params_from_ui(self):
        p = default_params()
        p["inverse"] = self._inverse_var.get()
        p["quantizer"] = self._quantizer_var.get()
        p["packets"] = self._packets_var.get()
        p["filter_type"] = self._filter_var.get()
        p["filter_slope"] = self._slope_var.get()
        p["freeze"] = self._freeze_var.get()
        p["freeze_mode"] = self._freeze_mode_var.get()
        p["verb_position"] = self._verb_pos_var.get()
        p["bounce"] = self._bounce_var.get()
        p["bounce_target"] = self._bounce_target_var.get()
        for key, var in self.sliders.items():
            p[key] = var.get()
        return p

    def _write_params_to_ui(self, p):
        self._inverse_var.set(p.get("inverse", 0))
        self._quantizer_var.set(p.get("quantizer", 0))
        self._packets_var.set(p.get("packets", 0))
        self._filter_var.set(p.get("filter_type", 0))
        self._slope_var.set(p.get("filter_slope", 1))
        self._freeze_var.set(p.get("freeze", 0))
        self._freeze_mode_var.set(p.get("freeze_mode", 0))
        self._verb_pos_var.set(p.get("verb_position", 0))
        self._bounce_var.set(p.get("bounce", 0))
        self._bounce_target_var.set(p.get("bounce_target", 0))
        for key, var in self.sliders.items():
            if key in p:
                var.set(p[key])

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _on_load(self):
        path = filedialog.askopenfilename(filetypes=[("WAV", "*.wav")])
        if path:
            self._load_wav(path)
            self.rendered_audio = None
            self.status_var.set(f"Loaded: {os.path.basename(path)}")
            self._draw_input_waveform()

    def _on_save(self):
        if self.rendered_audio is None:
            self.status_var.set("Nothing to save — render first")
            return
        path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV", "*.wav")])
        if path:
            audio = self.rendered_audio.copy()
            peak = np.max(np.abs(audio))
            if peak > 1.0:
                audio = audio / peak * 0.95
            out = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
            wavfile.write(path, SR, out)
            self.status_var.set(f"Saved: {os.path.basename(path)}")

    def _params_changed(self, params):
        if self.rendered_params is None or self.rendered_audio is None:
            return True
        return params != self.rendered_params

    # ------------------------------------------------------------------
    # Playback cursor & seek
    # ------------------------------------------------------------------

    def _output_canvases(self):
        return [(self.wave_canvas, WAVE_PAD_LEFT, WAVE_PAD_RIGHT),
                (self.spec_output_canvas, SPEC_PAD_LEFT, SPEC_PAD_RIGHT)]

    def _input_canvases(self):
        return [(self.input_wave_canvas, WAVE_PAD_LEFT, WAVE_PAD_RIGHT),
                (self.spec_input_canvas, SPEC_PAD_LEFT, SPEC_PAD_RIGHT)]

    def _start_cursor(self, audio, canvases):
        self._stop_cursor()
        self._playback_audio = audio
        self._playback_start = time.time()
        self._playback_length = len(audio) / SR
        self._cursor_canvases = canvases
        self._tick_cursor()

    def _stop_cursor(self):
        if self._cursor_timer is not None:
            self.root.after_cancel(self._cursor_timer)
            self._cursor_timer = None
        for canvas, _, _ in self._cursor_canvases:
            canvas.delete("cursor")
        self._cursor_canvases = []
        self._playback_audio = None
        self._playback_start = None

    def _tick_cursor(self):
        if self._playback_start is None:
            return
        elapsed = time.time() - self._playback_start
        if elapsed >= self._playback_length or elapsed < 0:
            self._stop_cursor()
            return
        frac = elapsed / self._playback_length
        for canvas, pl, pr in self._cursor_canvases:
            canvas.delete("cursor")
            w = canvas.winfo_width()
            h = canvas.winfo_height()
            x = pl + frac * (w - pl - pr)
            canvas.create_line(x, 0, x, h, fill="#ffffff",
                               width=1, tags="cursor")
        self._cursor_timer = self.root.after(40, self._tick_cursor)

    def _on_output_seek(self, event):
        if self.rendered_audio is None:
            return
        pl, pr = self._canvas_padding(event.widget)
        self._seek_and_play(event, self.rendered_audio, pl, pr,
                            self._output_canvases())
        dur = len(self.rendered_audio) / SR
        self.status_var.set(f"Playing ({dur:.1f}s){self.rendered_warning}")

    def _on_input_seek(self, event):
        if self.source_audio is None:
            return
        pl, pr = self._canvas_padding(event.widget)
        self._seek_and_play(event, self.source_audio, pl, pr,
                            self._input_canvases())
        self.status_var.set("Playing dry...")

    def _canvas_padding(self, widget):
        if widget in (self.spec_input_canvas, self.spec_output_canvas):
            return SPEC_PAD_LEFT, SPEC_PAD_RIGHT
        return WAVE_PAD_LEFT, WAVE_PAD_RIGHT

    def _seek_and_play(self, event, audio, pad_left, pad_right, canvases):
        w = event.widget.winfo_width()
        plot_w = w - pad_left - pad_right
        if plot_w <= 0:
            return
        frac = max(0.0, min(1.0, (event.x - pad_left) / plot_w))
        sample = int(frac * len(audio))
        sd.stop()
        sd.default.reset()
        sd.play(audio[sample:], SR, device=self._output_device_idx)
        self._start_cursor(audio, canvases)
        self._playback_start = time.time() - frac * self._playback_length

    def _on_play(self):
        if self.source_audio is None:
            self.status_var.set("No source audio loaded")
            return
        if self.rendering:
            return

        params = self._read_params_from_ui()

        # Cache hit — just play
        if not self._params_changed(params):
            sd.stop()
            self._stop_cursor()
            sd.default.reset()
            sd.play(self.rendered_audio, SR, device=self._output_device_idx)
            self._start_cursor(self.rendered_audio, self._output_canvases())
            dur = len(self.rendered_audio) / SR
            self.status_var.set(f"Playing ({dur:.1f}s){self.rendered_warning}")
            return

        self._render_and_play(params)

    # ------------------------------------------------------------------
    # Generation history
    # ------------------------------------------------------------------

    def _save_generation(self):
        """Save current render as a generation snapshot."""
        if self.rendered_audio is None:
            return
        snap = {
            'params': self.rendered_params.copy() if isinstance(self.rendered_params, dict) else self.rendered_params,
            'audio': self.rendered_audio,
            'metrics': self.rendered_metrics,
            'warning': self.rendered_warning,
            'spectrogram': getattr(self, 'rendered_spectrogram', None),
        }
        # Truncate forward history if we're not at the end
        if self._gen_index < len(self._gen_history) - 1:
            self._gen_history = self._gen_history[:self._gen_index + 1]
        self._gen_history.append(snap)
        if len(self._gen_history) > self._gen_max:
            self._gen_history = self._gen_history[-self._gen_max:]
        self._gen_index = len(self._gen_history) - 1
        self._update_gen_buttons()
        # Show notice in AI chat if triggered by Claude
        if getattr(self, '_pending_gen_notice', False):
            self._pending_gen_notice = False
            gen = self._gen_index + 1
            self.ai_response.configure(state="normal")
            self.ai_response.insert("end", f"\n[Params applied → Gen {gen}]\n", "system_notice")
            self.ai_response.see("end")
            self.ai_response.configure(state="disabled")

    def _update_gen_buttons(self):
        """Update rewind/forward button state and generation label."""
        total = len(self._gen_history)
        if total == 0:
            self._gen_label_var.set("Gen 0")
            self._gen_back_btn.configure(state="disabled")
            self._gen_fwd_btn.configure(state="disabled")
        else:
            self._gen_label_var.set(f"Gen {self._gen_index + 1}/{total}")
            self._gen_back_btn.configure(state="normal" if self._gen_index > 0 else "disabled")
            self._gen_fwd_btn.configure(state="normal" if self._gen_index < total - 1 else "disabled")

    def _restore_generation(self, index):
        """Restore a generation snapshot by index."""
        snap = self._gen_history[index]
        self._gen_index = index
        self._write_params_to_ui(snap['params'])
        self.rendered_audio = snap['audio']
        self.rendered_params = snap['params']
        self.rendered_metrics = snap['metrics']
        self.rendered_warning = snap['warning']
        self.rendered_spectrogram = snap.get('spectrogram')
        self._play_rendered()
        dur = len(snap['audio']) / SR
        self.status_var.set(f"Gen {index + 1} — Playing ({dur:.1f}s){snap['warning']}")
        self._draw_waveform()
        self._draw_spectrograms()
        self._draw_spectrum()
        self._update_gen_buttons()

    def _on_gen_back(self):
        if self._gen_index > 0:
            self._restore_generation(self._gen_index - 1)

    def _on_gen_forward(self):
        if self._gen_index < len(self._gen_history) - 1:
            self._restore_generation(self._gen_index + 1)

    def _render_and_play(self, params, on_complete=None, status_prefix=None):
        """Render full buffer in background, then play with sd.play()."""
        if self._autoplay_id is not None:
            self.root.after_cancel(self._autoplay_id)
            self._autoplay_id = None
        self._stop_cursor()
        self.rendering = True
        if status_prefix:
            self.status_var.set(f"{status_prefix} {self.llm._iterate_count}/{self.llm.MAX_ITERATE}")
        else:
            self.status_var.set("Rendering...")

        def do_render():
            try:
                tail_len = int(self.sliders["tail_length"].get() * SR)
                if tail_len > 0:
                    if self.source_audio.ndim == 2:
                        tail = np.zeros((tail_len, self.source_audio.shape[1]))
                    else:
                        tail = np.zeros(tail_len)
                    audio_in = np.concatenate([self.source_audio, tail])
                else:
                    audio_in = self.source_audio
                output = render_lossy(audio_in, params)

                ok, err = safety_check(output)
                if not ok:
                    self.root.after(0, lambda: self.status_var.set(err))
                    return

                output, warning = normalize_output(output)

                from shared.analysis import analyze
                from shared.audio_features import generate_spectrogram_png
                self.rendered_metrics = analyze(output, SR, reference=self.source_audio)
                self.rendered_spectrogram = generate_spectrogram_png(output, SR)
                self.rendered_audio = output
                self.rendered_warning = warning
                self.rendered_params = params.copy()
                dur = len(output) / SR

                def on_gui():
                    self._save_generation()
                    self._play_rendered()
                    gen = self._gen_index + 1
                    if status_prefix:
                        self.status_var.set(
                            f"Gen {gen} — {status_prefix} {self.llm._iterate_count}/{self.llm.MAX_ITERATE} complete")
                    else:
                        self.status_var.set(f"Gen {gen} — Playing ({dur:.1f}s){warning}")
                    self._draw_waveform()
                    self._draw_spectrograms()
                    self._draw_spectrum()
                    if on_complete:
                        on_complete()

                self.root.after(0, on_gui)
                self.root.after(50, self._check_autoplay)
            except Exception as exc:
                self.root.after(0, lambda: self.status_var.set(f"ERROR: {exc}"))
            finally:
                self.rendering = False

        threading.Thread(target=do_render, daemon=True).start()

    def _play_rendered(self):
        if self.rendered_audio is not None:
            sd.default.reset()
            sd.play(self.rendered_audio, SR, device=self._output_device_idx)
            self._start_cursor(self.rendered_audio, self._output_canvases())
            self.status_var.set("Playing...")

    def _on_play_dry(self):
        if self.source_audio is not None:
            sd.default.reset()
            sd.play(self.source_audio, SR, device=self._output_device_idx)
            self._start_cursor(self.source_audio, self._input_canvases())
            self.status_var.set("Playing dry...")

    # ------------------------------------------------------------------
    # Output device selection
    # ------------------------------------------------------------------

    def _get_output_devices(self):
        devices = sd.query_devices()
        out = []
        for i, d in enumerate(devices):
            if d['max_output_channels'] > 0:
                out.append((i, d['name']))
        return out

    def _refresh_devices(self):
        devs = self._get_output_devices()
        self._output_devices = devs
        names = ["System Default"] + [name for _, name in devs]
        self._device_combo['values'] = names
        if self._output_device_idx is not None:
            for i, (idx, _) in enumerate(devs):
                if idx == self._output_device_idx:
                    self._device_combo.current(i + 1)
                    return
        self._output_device_idx = None
        self._device_combo.current(0)

    def _on_device_changed(self, event=None):
        sel = self._device_combo.current()
        if sel <= 0:
            self._output_device_idx = None
        else:
            self._output_device_idx = self._output_devices[sel - 1][0]

    def _on_stop(self):
        if self._autoplay_id is not None:
            self.root.after_cancel(self._autoplay_id)
            self._autoplay_id = None
        sd.stop()
        self._stop_cursor()
        self.status_var.set("Stopped")
