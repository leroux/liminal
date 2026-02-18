"""PedalGUIBase — shared tkinter GUI infrastructure for all pedals.

Subclasses implement:
    _build_params_page(parent_frame)  — pedal-specific slider layout
    _read_params_from_ui() -> dict    — read all params from UI controls
    _write_params_to_ui(params)       — set all UI controls from params dict
"""

from __future__ import annotations

import json
import math
import os
import random
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog

import numpy as np
import sounddevice as sd

from shared.audio import load_wav as _load_wav_file, save_wav as _save_wav_file, make_impulse
from shared.llm_tuner import LLMTuner
from shared.streaming import safety_check, normalize_output
from shared.waveform import (draw_waveform as _shared_draw_waveform,
                              draw_spectrogram,
                              WAVE_PAD_LEFT, WAVE_PAD_RIGHT,
                              SPEC_PAD_LEFT, SPEC_PAD_RIGHT)

from shared.gui.config import PedalConfig


class PedalGUIBase:
    """Base class containing ~1200 lines of shared GUI infrastructure."""

    # Subclass may override with a set of param keys to skip during randomize
    _RANDOMIZE_SKIP: set = set()
    # Subclass may override with {param_key: (lo, hi)} to clamp during randomize
    _RANDOMIZE_CLAMP: dict = {}

    _ZOOM_MIN = 0.5
    _ZOOM_MAX = 3.0
    _ZOOM_STEP = 0.1

    def __init__(self, root: tk.Tk, config: PedalConfig):
        self.root = root
        self.cfg = config

        # Zoom setup: read the OS-corrected tk scaling as base so we
        # multiply on top of it rather than overwrite it.
        self._base_scale = float(self.root.tk.call('tk', 'scaling'))
        self._zoom = 1.0

        self.root.title(config.window_title)
        if config.window_geometry:
            self.root.geometry(config.window_geometry)
            # Parse W×H from geometry string so zoom can resize the window.
            try:
                wh = config.window_geometry.split('+')[0]
                w, h = wh.split('x')
                self._base_window_wh = (int(w), int(h))
            except Exception:
                self._base_window_wh = None
        else:
            self._base_window_wh = None

        # Window / dock icon
        if config.icon_path and os.path.exists(config.icon_path):
            self._app_icon = tk.PhotoImage(file=config.icon_path)
            self.root.iconphoto(True, self._app_icon)

        self.params = config.default_params()
        self.source_audio = None
        self.rendered_audio = None
        self.rendered_params = None
        self.rendered_warning = ""
        self.rendered_metrics = None
        self.rendered_spectrogram = None
        self.rendering = False
        self.sliders: dict[str, tk.DoubleVar] = {}
        self.locks: dict[str, tk.BooleanVar] = {}
        self.section_locks: dict[str, tk.BooleanVar] = {}
        self.choice_vars: dict[str, tk.IntVar] = {}  # for randomization
        self._scroll_widgets: dict[str, tuple] = {}
        self._autoplay_id = None
        self._cursor_timer = None
        self._cursor_canvases: list[tuple] = []
        self._playback_start = None
        self._playback_length = 0.0
        self._playback_audio = None
        self._pending_gen_notice = False

        # Generation history
        self._gen_history: list[dict] = []
        self._gen_index = -1
        self._gen_max = 50
        self._output_device_idx = None
        self._output_devices: list[tuple[int, str]] = []

        # Load default source audio
        self._load_wav(config.default_source)
        self._build_ui()

        # Capture named font sizes after UI is built so zoom can scale them.
        import tkinter.font as _tkfont
        self._base_font_sizes: dict[str, int] = {}
        for _name in _tkfont.names(root=self.root):
            try:
                _sz = _tkfont.nametofont(_name, root=self.root).cget('size')
                if _sz != 0:
                    self._base_font_sizes[_name] = _sz
            except Exception:
                pass

        self.llm = LLMTuner(
            guide_text=config.guide_text,
            param_descriptions=config.param_descriptions,
            param_ranges=config.param_ranges,
            default_params_fn=config.default_params,
            root=self.root,
            schema=config.schema,
        )

        # Global mousewheel handler
        self.root.bind_all("<MouseWheel>", self._on_global_scroll)
        try:
            self.root.bind_all("<TouchpadScroll>", self._on_global_scroll)
        except tk.TclError:
            pass

        # Zoom keyboard shortcuts (Cmd on macOS, Ctrl on others)
        for mod in ("<Command-equal>", "<Control-equal>"):
            self.root.bind(mod, lambda e: self._zoom_in())
        for mod in ("<Command-plus>", "<Control-plus>"):
            self.root.bind(mod, lambda e: self._zoom_in())
        for mod in ("<Command-minus>", "<Control-minus>"):
            self.root.bind(mod, lambda e: self._zoom_out())
        for mod in ("<Command-0>", "<Control-0>"):
            self.root.bind(mod, lambda e: self._zoom_reset())

    # ==================================================================
    # WAV helpers
    # ==================================================================

    def _load_wav(self, path):
        if not path or not os.path.isfile(path):
            self.source_audio = self._make_impulse()
            self._analyze_source()
            return
        audio, _ = _load_wav_file(path, self.cfg.sample_rate)
        self.source_audio = audio
        self._analyze_source()

    def _analyze_source(self):
        from shared.analysis import analyze
        from shared.audio_features import generate_spectrogram_png
        SR = self.cfg.sample_rate
        self.source_metrics = analyze(self.source_audio, SR)
        self.source_spectrogram = generate_spectrogram_png(self.source_audio, SR)
        if hasattr(self, 'llm'):
            self.llm.reset_session()

    def _make_impulse(self, seconds=2.0):
        return make_impulse(self.cfg.sample_rate, seconds)

    # ==================================================================
    # UI construction
    # ==================================================================

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

        # Generation navigation
        self._gen_back_btn = ttk.Button(top, text="<", width=2, command=self._on_gen_back)
        self._gen_back_btn.pack(side="left", padx=1)
        self._gen_label_var = tk.StringVar(value="Gen 0")
        ttk.Label(top, textvariable=self._gen_label_var, width=8, anchor="center").pack(side="left")
        self._gen_fwd_btn = ttk.Button(top, text=">", width=2, command=self._on_gen_forward)
        self._gen_fwd_btn.pack(side="left", padx=1)
        self._update_gen_buttons()

        # Zoom controls
        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=6)
        ttk.Button(top, text="−", width=2, command=self._zoom_out).pack(side="left", padx=1)
        self._zoom_label_var = tk.StringVar(value="100%")
        ttk.Label(top, textvariable=self._zoom_label_var, width=5, anchor="center").pack(side="left")
        ttk.Button(top, text="+", width=2, command=self._zoom_in).pack(side="left", padx=1)

        # Right side: status + device
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
        self._notebook = nb

        self.params_frame = ttk.Frame(nb)
        self.presets_frame = ttk.Frame(nb)
        self.waveforms_frame = ttk.Frame(nb, padding=5)
        self.spectrograms_frame = ttk.Frame(nb, padding=5)
        self.guide_frame = ttk.Frame(nb)

        nb.add(self.params_frame, text="Parameters")
        nb.add(self.presets_frame, text="Presets")

        # Extra tabs (e.g. Signal Flow for reverb, Spectrum for lossy/fractal)
        self._extra_tab_frames = {}
        for tab_name, builder_fn in self.cfg.extra_tabs:
            frame = ttk.Frame(nb)
            self._extra_tab_frames[tab_name] = frame
            # Don't add yet — they go between core tabs

        nb.add(self.waveforms_frame, text="Waveforms")
        nb.add(self.spectrograms_frame, text="Spectrograms")
        for tab_name, _ in self.cfg.extra_tabs:
            nb.add(self._extra_tab_frames[tab_name], text=tab_name)
        nb.add(self.guide_frame, text="Guide")

        self._build_params_page(self.params_frame)
        self._build_presets_page()
        self._build_waveforms_page()
        self._build_spectrograms_page()
        for tab_name, builder_fn in self.cfg.extra_tabs:
            builder_fn(self._extra_tab_frames[tab_name])
        self._build_guide_page()

        nb.bind("<<NotebookTabChanged>>", self._on_notebook_changed)

    def _on_notebook_changed(self, event=None):
        """Redraw visualizations when their tabs become active."""
        current = self._notebook.select()
        if current == str(self.waveforms_frame):
            self._draw_input_waveform()
            self._draw_waveform()
        elif current == str(self.spectrograms_frame):
            self._draw_spectrograms()

    # ==================================================================
    # Zoom
    # ==================================================================

    def _apply_zoom(self):
        """Scale all named fonts and resize window proportionally.

        tk scaling alone is ignored by macOS Aqua theme fonts, so we resize
        the actual named font objects which guarantees visible scaling.
        """
        import tkinter.font as _tkfont
        for name, base_size in self._base_font_sizes.items():
            try:
                f = _tkfont.nametofont(name, root=self.root)
                new_size = max(6, round(abs(base_size) * self._zoom))
                f.configure(size=new_size if base_size > 0 else -new_size)
            except Exception:
                pass
        # Resize window proportionally so content fills it
        if self._base_window_wh:
            bw, bh = self._base_window_wh
            self.root.geometry(f"{int(bw * self._zoom)}x{int(bh * self._zoom)}")
        if hasattr(self, '_zoom_label_var'):
            self._zoom_label_var.set(f"{int(self._zoom * 100)}%")

    def _zoom_in(self):
        self._zoom = min(self._ZOOM_MAX, round(self._zoom + self._ZOOM_STEP, 1))
        self._apply_zoom()

    def _zoom_out(self):
        self._zoom = max(self._ZOOM_MIN, round(self._zoom - self._ZOOM_STEP, 1))
        self._apply_zoom()

    def _zoom_reset(self):
        self._zoom = 1.0
        self._apply_zoom()

    # ==================================================================
    # Abstract methods — subclass MUST implement
    # ==================================================================

    def _build_params_page(self, parent):
        raise NotImplementedError

    def _read_params_from_ui(self) -> dict:
        raise NotImplementedError

    def _write_params_to_ui(self, params: dict):
        raise NotImplementedError

    # ==================================================================
    # Slider helpers
    # ==================================================================

    def _add_slider(self, parent, row, key, label, lo, hi, value,
                    length=280, log=False, integer=False):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=4)
        var = tk.DoubleVar(value=value)
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
        # Typeable entry
        entry_var = tk.StringVar(value=self._fmt_val(value, log, integer))
        entry = ttk.Entry(parent, textvariable=entry_var, width=8, justify="right")
        entry.grid(row=row, column=2, sticky="w", padx=2)
        def _on_slider_change(*_a, v=var, ev=entry_var, lg=log, intg=integer):
            ev.set(self._fmt_val(v.get(), lg, intg))
        var.trace_add("write", _on_slider_change)
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
        w = self.root.winfo_containing(event.x_root, event.y_root)
        if w is None:
            return
        path = str(w)
        if path not in self._scroll_widgets:
            return
        var, lo, hi = self._scroll_widgets[path]
        raw = event.delta if event.delta else (-1 if event.num == 5 else 1)
        y = raw & 0xFFFF
        if y >= 0x8000:
            y -= 0x10000
        step = (hi - lo) * 0.002
        var.set(max(lo, min(hi, var.get() + y * step)))
        self._schedule_autoplay()

    def _add_choice(self, parent, row, key, label, names, value=0):
        """Add a row of radio buttons for a choice parameter."""
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=8)
        var = tk.IntVar(value=value)
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=1, sticky="w")
        for i, name in enumerate(names):
            ttk.Radiobutton(frame, text=name, variable=var, value=i).pack(side="left", padx=3)
        self.choice_vars[key] = var
        self._add_lock(parent, row, key)
        var.trace_add("write", lambda *_: self._schedule_autoplay())
        return var, row + 1

    # ==================================================================
    # Section header / lock helpers
    # ==================================================================

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
        for param_key in self.cfg.param_sections.get(section_key, []):
            if param_key in self.locks:
                self.locks[param_key].set(locked)

    def _schedule_autoplay(self):
        if self._autoplay_id is not None:
            self.root.after_cancel(self._autoplay_id)
        self._autoplay_id = self.root.after(400, self._on_play)

    def _check_autoplay(self):
        params = self._read_params_from_ui()
        if self._params_changed(params):
            self._on_play()

    # ==================================================================
    # Params scrollable container + AI layout (call from subclass _build_params_page)
    # ==================================================================

    def _build_params_container(self, parent):
        """Create scrollable params frame + AI chat. Returns the inner frame for sliders."""
        self._params_ai_container = ttk.Frame(parent)
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

        def _on_mousewheel(event):
            raw = event.delta if event.delta else (-1 if event.num == 5 else 1)
            y = raw & 0xFFFF
            if y >= 0x8000:
                y -= 0x10000
            if event.state & 1:
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

        # AI chat frame
        self._ai_chat_frame = ttk.Frame(self._params_ai_container)
        self._build_ai_prompt(self._ai_chat_frame)
        self._ai_layout_mode = None
        self._params_ai_container.bind("<Configure>", self._on_ai_layout_configure)
        self._apply_ai_layout("vertical")

        return f

    # ==================================================================
    # AI Tuner
    # ==================================================================

    def _build_ai_prompt(self, parent):
        ttk.Label(parent, text="AI Tuner", font=("Helvetica", 11, "bold")).pack(anchor="w", padx=5, pady=(5, 2))

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
        self._on_ask_claude()
        return "break"

    def _on_ai_shift_return(self, event):
        self.ai_input.insert("insert", "\n")
        return "break"

    def _on_ai_layout_configure(self, event):
        width = event.width
        if width > 1200 and self._ai_layout_mode != "horizontal":
            self._apply_ai_layout("horizontal")
        elif width <= 1200 and self._ai_layout_mode != "vertical":
            self._apply_ai_layout("vertical")

    def _apply_ai_layout(self, mode):
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

    # ==================================================================
    # Randomize / Reset
    # ==================================================================

    def _reset_params(self):
        self._write_params_to_ui(self.cfg.bypass_params())
        self._on_play()

    def _randomize_params(self):
        skip = self.cfg.randomize_skip | self._RANDOMIZE_SKIP
        clamp = {**self.cfg.randomize_clamp, **self._RANDOMIZE_CLAMP}

        for key, (lo, hi) in self.cfg.param_ranges.items():
            if key in skip:
                continue
            if key in self.locks and self.locks[key].get():
                continue
            if key not in self.sliders:
                continue
            r_lo, r_hi = clamp.get(key, (lo, hi))
            if hasattr(self.sliders[key], '_integer') and self.sliders[key]._integer:
                val = random.randint(int(r_lo), int(r_hi))
            elif hasattr(self.sliders[key], '_log') and self.sliders[key]._log:
                val = 10 ** random.uniform(math.log10(max(r_lo, 1e-10)),
                                           math.log10(max(r_hi, 1e-10)))
            else:
                val = random.uniform(r_lo, r_hi)
            self.sliders[key].set(val)

        for key, num_choices in self.cfg.choice_ranges.items():
            if key in self.locks and self.locks[key].get():
                continue
            if key in self.choice_vars:
                self.choice_vars[key].set(random.randint(0, num_choices - 1))

        self._on_play()

    # ==================================================================
    # Presets tab
    # ==================================================================

    def _build_presets_page(self):
        f = self.presets_frame
        search_frame = ttk.Frame(f)
        search_frame.pack(fill="x", padx=8, pady=(8, 0))
        ttk.Label(search_frame, text="Search:").pack(side="left", padx=(0, 4))
        self._preset_search_var = tk.StringVar()
        self._preset_search_var.trace_add("write", lambda *_: self._refresh_preset_list())
        ttk.Entry(search_frame, textvariable=self._preset_search_var).pack(side="left", fill="x", expand=True)

        tree_frame = ttk.Frame(f)
        tree_frame.pack(side="left", fill="both", expand=True, padx=8, pady=8)

        self.preset_tree = ttk.Treeview(tree_frame, columns=("desc", "name"),
                                         show="tree headings",
                                         selectmode="browse", height=20)
        self.preset_tree.heading("#0", text="Preset", anchor="w")
        self.preset_tree.heading("desc", text="Description", anchor="w")
        self.preset_tree.column("#0", width=200, minwidth=120)
        self.preset_tree.column("desc", width=420, minwidth=200)
        self.preset_tree.column("name", width=0, stretch=False)
        tree_scroll = ttk.Scrollbar(tree_frame, orient="vertical",
                                     command=self.preset_tree.yview)
        self.preset_tree.configure(yscrollcommand=tree_scroll.set)
        self.preset_tree.pack(side="left", fill="both", expand=True)
        tree_scroll.pack(side="right", fill="y")
        self.preset_tree.bind("<Double-1>", lambda e: self._on_load_preset(play=True))
        self.preset_tree.bind("<Return>", lambda e: self._on_load_preset(play=True))

        right = ttk.Frame(f)
        right.pack(side="left", fill="y", padx=8, pady=8)
        btn = ttk.Frame(right)
        btn.pack(fill="x")
        ttk.Button(btn, text="Load", command=self._on_load_preset).pack(fill="x", pady=2)
        ttk.Button(btn, text="Save", command=self._on_save_preset).pack(fill="x", pady=2)
        ttk.Button(btn, text="Delete", command=self._on_delete_preset).pack(fill="x", pady=2)
        ttk.Button(btn, text="Refresh", command=self._refresh_preset_list).pack(fill="x", pady=2)
        ttk.Button(btn, text="\u2605 Favorite", command=self._on_toggle_favorite).pack(fill="x", pady=2)

        self._preset_meta = {}
        self._refresh_preset_list()

    def _refresh_preset_list(self):
        preset_dir = self.cfg.preset_dir
        self.preset_tree.delete(*self.preset_tree.get_children())
        self._preset_meta.clear()
        os.makedirs(preset_dir, exist_ok=True)
        favorites = self._load_favorites()

        categories = {}
        for filename in sorted(os.listdir(preset_dir)):
            if not filename.endswith(".json") or filename == "favorites.json":
                continue
            name = filename[:-5]
            path = os.path.join(preset_dir, filename)
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

        query = ""
        if hasattr(self, "_preset_search_var"):
            query = self._preset_search_var.get().strip().lower()

        def _matches(name):
            if not query:
                return True
            meta = self._preset_meta.get(name, {})
            return query in name.lower() or query in meta.get("description", "").lower()

        fav_names = sorted(n for n in favorites if n in self._preset_meta and _matches(n))
        if fav_names:
            fav_id = self.preset_tree.insert("", "end", text="\u2605 Favorites", open=True,
                                              values=("", ""))
            for name in fav_names:
                desc = self._preset_meta[name].get("description", "")
                self.preset_tree.insert(fav_id, "end", text=f"\u2605 {name}",
                                         values=(desc, name))

        cat_order = self.cfg.preset_categories + ["Uncategorized"]
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
            filtered = [n for n in categories.get(cat, []) if _matches(n)]
            if not filtered:
                continue
            cat_id = self.preset_tree.insert("", "end", text=cat, open=True,
                                              values=("", ""))
            for name in filtered:
                desc = self._preset_meta[name].get("description", "")
                display = f"\u2605 {name}" if name in favorites else name
                self.preset_tree.insert(cat_id, "end", text=display,
                                         values=(desc, name))

    def _load_favorites(self):
        path = os.path.join(self.cfg.preset_dir, "favorites.json")
        try:
            with open(path) as fh:
                return set(json.load(fh))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return set()

    def _save_favorites(self, favorites):
        path = os.path.join(self.cfg.preset_dir, "favorites.json")
        os.makedirs(self.cfg.preset_dir, exist_ok=True)
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
        name = vals[1] if vals and len(vals) > 1 else ""
        if not name or name not in self._preset_meta:
            return None
        return name

    def _on_load_preset(self, play=False):
        name = self._get_selected_preset_name()
        if not name:
            return
        path = os.path.join(self.cfg.preset_dir, name + ".json")
        with open(path) as fh:
            p = json.load(fh)
        p.pop("_meta", None)
        if self.cfg.migrate_preset:
            self.cfg.migrate_preset(p)
        full = self.cfg.default_params()
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
        ttk.Combobox(save_win, textvariable=cat_var, values=self.cfg.preset_categories,
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
            os.makedirs(self.cfg.preset_dir, exist_ok=True)
            path = os.path.join(self.cfg.preset_dir, n + ".json")
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
        path = os.path.join(self.cfg.preset_dir, name + ".json")
        if os.path.isfile(path):
            os.remove(path)
        self._refresh_preset_list()

    # ==================================================================
    # Waveform / Spectrogram tabs
    # ==================================================================

    def _build_waveforms_page(self):
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
        parent = self.spectrograms_frame
        self._spec_images = {}
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
        SR = self.cfg.sample_rate
        _shared_draw_waveform(self.input_wave_canvas, self.source_audio, SR,
                              getattr(self, 'source_metrics', None), "Input")

    def _draw_waveform(self):
        SR = self.cfg.sample_rate
        _shared_draw_waveform(self.wave_canvas, self.rendered_audio, SR,
                              self.rendered_metrics, "Output", self.rendered_warning)

    def _draw_spectrograms(self):
        SR = self.cfg.sample_rate
        draw_spectrogram(self.spec_input_canvas, self.source_audio, SR,
                         "Input", self._spec_images)
        draw_spectrogram(self.spec_output_canvas, self.rendered_audio, SR,
                         "Output", self._spec_images)

    # ==================================================================
    # Guide tab
    # ==================================================================

    def _build_guide_page(self):
        """Build the guide page. Override in subclass for custom content."""
        text = tk.Text(self.guide_frame, wrap="word", bg="#1a1a2e", fg="#ccccdd",
                       font=("Helvetica", 12), padx=16, pady=12, relief="flat",
                       selectbackground="#334466", insertbackground="#ccccdd")
        text.pack(fill="both", expand=True)
        text.tag_configure("h1", font=("Helvetica", 16, "bold"), foreground="#ffffff", spacing3=6)
        text.tag_configure("h2", font=("Helvetica", 13, "bold"), foreground="#ff9966",
                           spacing1=14, spacing3=4)
        text.tag_configure("body", foreground="#ccccdd", spacing1=2)
        # Default: just insert guide text as body
        text.insert("end", self.cfg.guide_text, "body")
        text.config(state="disabled")

    # ==================================================================
    # File I/O
    # ==================================================================

    def _on_load(self):
        path = filedialog.askopenfilename(filetypes=[("WAV", "*.wav")])
        if path:
            self._load_wav(path)
            self.rendered_audio = None
            self.status_var.set(f"Loaded: {os.path.basename(path)}")
            self._draw_input_waveform()

    def _on_save(self):
        if self.rendered_audio is None:
            self.status_var.set("Nothing to save -- render first")
            return
        path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV", "*.wav")])
        if path:
            _save_wav_file(path, self.rendered_audio.copy(), self.cfg.sample_rate)
            self.status_var.set(f"Saved: {os.path.basename(path)}")

    def _params_changed(self, params):
        if self.rendered_params is None or self.rendered_audio is None:
            return True
        return params != self.rendered_params

    # ==================================================================
    # Playback cursor & seek
    # ==================================================================

    def _output_canvases(self):
        return [(self.wave_canvas, WAVE_PAD_LEFT, WAVE_PAD_RIGHT),
                (self.spec_output_canvas, SPEC_PAD_LEFT, SPEC_PAD_RIGHT)]

    def _input_canvases(self):
        return [(self.input_wave_canvas, WAVE_PAD_LEFT, WAVE_PAD_RIGHT),
                (self.spec_input_canvas, SPEC_PAD_LEFT, SPEC_PAD_RIGHT)]

    def _start_cursor(self, audio, canvases):
        SR = self.cfg.sample_rate
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
            canvas.create_line(x, 0, x, h, fill="#ffffff", width=1, tags="cursor")
        self._cursor_timer = self.root.after(40, self._tick_cursor)

    def _on_output_seek(self, event):
        if self.rendered_audio is None:
            return
        pl, pr = self._canvas_padding(event.widget)
        self._seek_and_play(event, self.rendered_audio, pl, pr,
                            self._output_canvases())
        SR = self.cfg.sample_rate
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
        SR = self.cfg.sample_rate
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

    # ==================================================================
    # Playback & rendering
    # ==================================================================

    def _on_play(self):
        if self.source_audio is None:
            self.status_var.set("No source audio loaded")
            return
        if self.rendering:
            return
        params = self._read_params_from_ui()
        SR = self.cfg.sample_rate
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

    def _render_and_play(self, params, on_complete=None, status_prefix=None):
        SR = self.cfg.sample_rate
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
                # Tail handling
                tail_key = self.cfg.tail_param
                tail_len = 0
                if tail_key and tail_key in self.sliders:
                    tail_len = int(self.sliders[tail_key].get() * SR)
                if tail_len > 0:
                    if self.source_audio.ndim == 2:
                        tail = np.zeros((tail_len, self.source_audio.shape[1]))
                    else:
                        tail = np.zeros(tail_len)
                    audio_in = np.concatenate([self.source_audio, tail])
                else:
                    audio_in = self.source_audio

                # Stereo dispatch
                if audio_in.ndim == 2 and self.cfg.render_stereo:
                    output_l, output_r = self.cfg.render_stereo(
                        audio_in[:, 0], audio_in[:, 1], params)
                    output = np.column_stack([output_l, output_r])
                else:
                    output = self.cfg.render(audio_in, params)

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
                self.rendered_params = params.copy() if isinstance(params, dict) else params
                dur = len(output) / SR

                def on_gui():
                    self._save_generation()
                    self._play_rendered()
                    gen = self._gen_index + 1
                    if status_prefix:
                        self.status_var.set(
                            f"Gen {gen} -- {status_prefix} {self.llm._iterate_count}/{self.llm.MAX_ITERATE} complete")
                    else:
                        self.status_var.set(f"Gen {gen} -- Playing ({dur:.1f}s){warning}")
                    self._draw_waveform()
                    self._draw_spectrograms()
                    self._on_render_complete()
                    if on_complete:
                        on_complete()

                self.root.after(0, on_gui)
                self.root.after(50, self._check_autoplay)
            except Exception as exc:
                self.root.after(0, lambda: self.status_var.set(f"ERROR: {exc}"))
            finally:
                self.rendering = False

        threading.Thread(target=do_render, daemon=True).start()

    def _on_render_complete(self):
        """Hook for subclasses to update extra visualizations after render."""
        pass

    def _play_rendered(self):
        SR = self.cfg.sample_rate
        if self.rendered_audio is not None:
            sd.default.reset()
            sd.play(self.rendered_audio, SR, device=self._output_device_idx)
            self._start_cursor(self.rendered_audio, self._output_canvases())
            self.status_var.set("Playing...")

    def _on_play_dry(self):
        SR = self.cfg.sample_rate
        if self.source_audio is not None:
            sd.default.reset()
            sd.play(self.source_audio, SR, device=self._output_device_idx)
            self._start_cursor(self.source_audio, self._input_canvases())
            self.status_var.set("Playing dry...")

    def _on_stop(self):
        if self._autoplay_id is not None:
            self.root.after_cancel(self._autoplay_id)
            self._autoplay_id = None
        sd.stop()
        self._stop_cursor()
        self.status_var.set("Stopped")

    # ==================================================================
    # Generation history
    # ==================================================================

    def _save_generation(self):
        if self.rendered_audio is None:
            return
        snap = {
            'params': self.rendered_params.copy() if isinstance(self.rendered_params, dict) else self.rendered_params,
            'audio': self.rendered_audio,
            'metrics': self.rendered_metrics,
            'warning': self.rendered_warning,
            'spectrogram': getattr(self, 'rendered_spectrogram', None),
        }
        if self._gen_index < len(self._gen_history) - 1:
            self._gen_history = self._gen_history[:self._gen_index + 1]
        self._gen_history.append(snap)
        if len(self._gen_history) > self._gen_max:
            self._gen_history = self._gen_history[-self._gen_max:]
        self._gen_index = len(self._gen_history) - 1
        self._update_gen_buttons()
        if self._pending_gen_notice:
            self._pending_gen_notice = False
            gen = self._gen_index + 1
            self.ai_response.configure(state="normal")
            self.ai_response.insert("end", f"\n[Params applied -> Gen {gen}]\n", "system_notice")
            self.ai_response.see("end")
            self.ai_response.configure(state="disabled")

    def _update_gen_buttons(self):
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
        SR = self.cfg.sample_rate
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
        self.status_var.set(f"Gen {index + 1} -- Playing ({dur:.1f}s){snap['warning']}")
        self._draw_waveform()
        self._draw_spectrograms()
        self._on_render_complete()
        self._update_gen_buttons()

    def _on_gen_back(self):
        if self._gen_index > 0:
            self._restore_generation(self._gen_index - 1)

    def _on_gen_forward(self):
        if self._gen_index < len(self._gen_history) - 1:
            self._restore_generation(self._gen_index + 1)

    # ==================================================================
    # Output device selection
    # ==================================================================

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
