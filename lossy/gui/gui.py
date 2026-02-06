"""Lossy codec emulation GUI — tkinter.

Tabs: Parameters · Presets · Waveform · Spectrum
"""

import json
import math
import os
import random
import threading
import tkinter as tk
from tkinter import ttk, filedialog, simpledialog

import numpy as np
import sounddevice as sd
from scipy.io import wavfile

from engine.params import (
    SR,
    default_params,
    bypass_params,
    PARAM_RANGES,
    PARAM_SECTIONS,
    CHOICE_RANGES,
    MODE_NAMES,
    QUANTIZER_NAMES,
    PACKET_NAMES,
    FILTER_NAMES,
    SLOPE_OPTIONS,
    FREEZE_NAMES,
    VERB_POSITION_NAMES,
    BOUNCE_TARGETS,
    BOUNCE_TARGET_NAMES,
)
from engine.lossy import render_lossy

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
        self.params = default_params()
        self.source_audio = None
        self.rendered_audio = None
        self.rendered_params = None
        self.rendering = False
        self.sliders = {}
        self.locks = {}
        self.section_locks = {}
        self._scroll_widgets = {}
        self._autoplay_id = None
        self._load_wav(DEFAULT_SOURCE)
        self._build_ui()

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
            return
        sr, data = wavfile.read(path)
        if data.dtype == np.int16:
            audio = data.astype(np.float64) / 32768.0
        elif data.dtype == np.int32:
            audio = data.astype(np.float64) / 2147483648.0
        else:
            audio = data.astype(np.float64)
        # Keep stereo as (samples, 2) — don't downmix
        self.source_audio = audio

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

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(top, textvariable=self.status_var).pack(side="right")

        # Notebook
        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=8, pady=4)

        self.params_frame = ttk.Frame(nb)
        self.presets_frame = ttk.Frame(nb)
        self.wave_frame = ttk.Frame(nb)
        self.spec_frame = ttk.Frame(nb)
        self.guide_frame = ttk.Frame(nb)

        nb.add(self.params_frame, text="Parameters")
        nb.add(self.presets_frame, text="Presets")
        nb.add(self.wave_frame, text="Waveform")
        nb.add(self.spec_frame, text="Spectrum")
        nb.add(self.guide_frame, text="Guide")

        self._build_params_page()
        self._build_presets_page()
        self._build_wave_page()
        self._build_spec_page()
        self._build_guide_page()

    # ------------------------------------------------------------------
    # Parameters tab
    # ------------------------------------------------------------------

    def _build_params_page(self):
        # Scrollable single-column layout
        canvas = tk.Canvas(self.params_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.params_frame, orient="vertical", command=canvas.yview)
        f = ttk.Frame(canvas)

        f.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=f, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Mouse wheel / trackpad scrolling for the page canvas
        def _on_mousewheel(event):
            raw = event.delta if event.delta else (-1 if event.num == 5 else 1)
            y = raw & 0xFFFF
            if y >= 0x8000:
                y -= 0x10000
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

        ttk.Label(f, text="Mode").grid(row=r, column=0, sticky="w", padx=8)
        self._mode_var = tk.IntVar(value=self.params["mode"])
        mode_frame = ttk.Frame(f)
        mode_frame.grid(row=r, column=1, sticky="w")
        for i, name in enumerate(MODE_NAMES):
            ttk.Radiobutton(mode_frame, text=name, variable=self._mode_var, value=i).pack(side="left", padx=3)
        self._add_lock(f, r, "mode")
        r += 1

        r = self._add_slider(f, r, "loss", "Loss", 0.0, 1.0, self.params["loss"], length=SL)
        r = self._add_slider(f, r, "speed", "Speed", 0.0, 1.0, self.params["speed"], length=SL)
        r = self._add_slider(f, r, "global_amount", "Global", 0.0, 1.0, self.params["global_amount"], length=SL)
        r = self._add_slider(f, r, "phase_loss", "Phase", 0.0, 1.0, self.params["phase_loss"], length=SL)
        r = self._add_slider(f, r, "pre_echo", "Pre-Echo", 0.0, 1.0, self.params["pre_echo"], length=SL)
        r = self._add_slider(f, r, "noise_shape", "Noise Shape", 0.0, 1.0, self.params["noise_shape"], length=SL)

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

        # ---- Output ----
        r = self._add_separator(f, r)
        r = self._add_section_header(f, r, "output", "OUTPUT")

        r = self._add_slider(f, r, "wet_dry", "Wet / Dry", 0.0, 1.0, self.params["wet_dry"], length=SL)

        # Auto-play on discrete parameter change
        for var in [self._mode_var, self._quantizer_var, self._packets_var,
                    self._filter_var, self._slope_var, self._freeze_var,
                    self._freeze_mode_var, self._verb_pos_var, self._bounce_var,
                    self._bounce_target_var]:
            var.trace_add("write", lambda *_: self._schedule_autoplay())

    # ------------------------------------------------------------------
    # Slider helper
    # ------------------------------------------------------------------

    def _add_slider(self, parent, row, key, label, lo, hi, value, length=280, log=False):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=4)
        var = tk.DoubleVar(value=value)
        scale = ttk.Scale(parent, from_=lo, to=hi, variable=var, orient="horizontal", length=length)
        scale.grid(row=row, column=1, sticky="ew", padx=4)
        self._bind_scroll(scale, var, lo, hi)
        val_lbl = ttk.Label(parent, text=self._fmt(value, log, lo, hi), width=10)
        val_lbl.grid(row=row, column=2, sticky="w")
        var.trace_add("write", lambda *_a, v=var, l=val_lbl, lg=log, _lo=lo, _hi=hi: l.config(text=self._fmt(v.get(), lg, _lo, _hi)))
        self.sliders[key] = var
        if log:
            var._log = True
        lock_var = tk.BooleanVar(value=False)
        self.locks[key] = lock_var
        ttk.Checkbutton(parent, variable=lock_var).grid(row=row, column=3, sticky="w", padx=2)
        scale.bind("<ButtonRelease-1>", lambda e: self._schedule_autoplay())
        return row + 1

    @staticmethod
    def _fmt(val, log, lo, hi):
        if log and lo > 0 and hi > 0:
            return f"{val:.0f}"
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
            if hasattr(self.sliders[key], '_log') and self.sliders[key]._log:
                val = 10 ** random.uniform(math.log10(max(lo, 1e-10)),
                                           math.log10(max(hi, 1e-10)))
            else:
                val = random.uniform(lo, hi)
            self.sliders[key].set(val)
        # Integer/choice params
        var_map = {
            "mode": self._mode_var,
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

        # Right: buttons + description
        right = ttk.Frame(f)
        right.pack(side="left", fill="y", padx=8, pady=8)

        btn = ttk.Frame(right)
        btn.pack(fill="x")
        ttk.Button(btn, text="Load", command=self._on_load_preset).pack(fill="x", pady=2)
        ttk.Button(btn, text="Save", command=self._on_save_preset).pack(fill="x", pady=2)
        ttk.Button(btn, text="Delete", command=self._on_delete_preset).pack(fill="x", pady=2)
        ttk.Button(btn, text="Refresh", command=self._refresh_preset_list).pack(fill="x", pady=2)

        self._preset_meta = {}  # name -> {category, description}
        self._refresh_preset_list()

    def _refresh_preset_list(self):
        self.preset_tree.delete(*self.preset_tree.get_children())
        self._preset_meta.clear()
        os.makedirs(PRESET_DIR, exist_ok=True)

        # Read all presets and group by category
        categories = {}
        for filename in sorted(os.listdir(PRESET_DIR)):
            if not filename.endswith(".json"):
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
            cat_id = self.preset_tree.insert("", "end", text=cat, open=True,
                                              values=("", ""))
            for name in categories[cat]:
                desc = self._preset_meta[name].get("description", "")
                self.preset_tree.insert(cat_id, "end", text=name,
                                         values=(desc, name))

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

    def _build_wave_page(self):
        self.wave_canvas = tk.Canvas(self.wave_frame, bg="#1a1a2e", highlightthickness=0)
        self.wave_canvas.pack(fill="both", expand=True)
        self.wave_canvas.bind("<Configure>", lambda e: self._draw_waveform())

    @staticmethod
    def _to_mono(audio):
        if audio is not None and audio.ndim == 2:
            return audio.mean(axis=1)
        return audio

    def _draw_waveform(self):
        c = self.wave_canvas
        c.delete("all")
        w = c.winfo_width()
        h = c.winfo_height()
        if w < 10 or h < 10:
            return

        audio = self.rendered_audio
        if audio is None:
            mid = h // 2
            c.create_line(0, mid, w, mid, fill="#333355")
            c.create_text(w // 2, mid, text="No render yet", fill="#666688", font=("Helvetica", 14))
            return

        stereo = audio.ndim == 2
        n = audio.shape[0] if stereo else len(audio)
        samples_per_px = max(1, n // w)

        if stereo:
            # Draw L and R in separate halves
            channels = [(audio[:, 0], "#4488ff", 0, h // 2),
                        (audio[:, 1], "#ff6644", h // 2, h)]
        else:
            channels = [(audio, "#4488ff", 0, h)]

        for ch_audio, color, y0, y1 in channels:
            ch_h = y1 - y0
            mid = y0 + ch_h // 2
            c.create_line(0, mid, w, mid, fill="#333355")
            for x in range(w):
                i0 = x * samples_per_px
                i1 = min(i0 + samples_per_px, n)
                if i0 >= n:
                    break
                chunk = ch_audio[i0:i1]
                lo_val = np.min(chunk)
                hi_val = np.max(chunk)
                y_lo = int(mid - lo_val * ch_h // 2 * 0.9)
                y_hi = int(mid - hi_val * ch_h // 2 * 0.9)
                c.create_line(x, y_lo, x, y_hi, fill=color)

        if stereo:
            c.create_text(8, 4, anchor="nw", fill="#4488ff", text="L")
            c.create_text(8, h // 2 + 4, anchor="nw", fill="#ff6644", text="R")

        peak = np.max(np.abs(audio))
        rms = np.sqrt(np.mean(audio ** 2))
        ch_label = "Stereo" if stereo else "Mono"
        c.create_text(w - 8, 12, anchor="ne", fill="#aaaacc",
                       text=f"{ch_label}  Peak: {peak:.3f}  RMS: {rms:.3f}  Samples: {n}")

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
        mapping("Speed knob", "Speed",
                "FFT window size. 0=slow (4096, darker, more smear) to 1=fast (256, garbled).")
        mapping("Global knob", "Global",
                "Master intensity — scales Loss, Phase, Crush, Packets, Verb, Gate together.")
        body("")

        h2("Mode Toggle (3-way)")
        mapping("Standard", "Mode: Standard",
                "Quantize FFT magnitudes + psychoacoustic band gating each frame.")
        mapping("Inverse", "Mode: Inverse",
                "Output the spectral residual — everything Standard throws away.")
        mapping("Jitter", "Mode: Jitter",
                "Random phase perturbation per FFT bin. Emulates bad digital clocking.")
        body("")
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
        mapping("(codec internals)", "Noise Shape",
                "Coarser quantization in spectral valleys, finer near peaks.")
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
                "Frozen spectrum slowly updates at a rate set by Speed.")
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
                "Which parameter the LFO modulates: Loss, Speed, Crush, etc.")
        mapping("Bounce rate", "Rate",
                "LFO speed. 0=0.1 Hz (slow sweep), 1=5 Hz (fast wobble).")
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
        body("3. ~21 Bark-like bands are gated using psychoacoustic masking: bands with")
        body("   less energy at ATH-insensitive frequencies are dropped first. Random")
        body("   perturbation ensures the frame-to-frame variation that creates the warble.")
        body("4. Phase is optionally quantized to N levels (Phase slider).")
        body("5. Pre-Echo boosts loss on frames preceding transients, spreading noise ahead.")
        body("6. HF bandwidth is rolled off proportional to Loss (like low-bitrate MP3).")
        body("7. IFFT + overlap-add (75%) reconstructs audio.")
        body("8. Crush/Decimate, Packets, Filter, Verb, Gate, Limiter in time domain.")
        body("")
        body("Window sizes: 4096 (93ms) / 2048 (46ms) / 1024 (23ms) / 512 (12ms) / 256 (6ms)")
        body('The "Slow" dip switch on the pedal = Speed at 0 here (4096 window).')
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
        p["mode"] = self._mode_var.get()
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
        self._mode_var.set(p.get("mode", 0))
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

    def _on_play(self):
        if self.source_audio is None:
            self.status_var.set("No source audio loaded")
            return
        if self.rendering:
            return

        params = self._read_params_from_ui()

        # Cache hit — just play
        if not self._params_changed(params):
            sd.default.reset()
            sd.play(self.rendered_audio, SR)
            self.status_var.set("Playing (cached)...")
            return

        # Render then play
        self.rendering = True
        self.status_var.set("Rendering...")

        def do_render():
            try:
                output = render_lossy(self.source_audio, params)
                # Safety checks
                if not np.all(np.isfinite(output)):
                    self.root.after(0, lambda: self.status_var.set("ERROR: non-finite output"))
                    return
                peak = np.max(np.abs(output))
                if peak > 1e6:
                    self.root.after(0, lambda: self.status_var.set(f"ERROR: output exploded (peak={peak:.0e})"))
                    return
                # RMS limiter (works for both mono and stereo)
                if peak > 0:
                    output = output / peak
                rms = np.sqrt(np.mean(output ** 2))
                target_rms = 0.2
                if rms > target_rms:
                    output *= target_rms / rms
                output *= 0.9  # headroom

                self.rendered_audio = output
                self.rendered_params = params.copy()
                self.root.after(0, self._draw_waveform)
                self.root.after(0, self._draw_spectrum)
                # Play immediately after render
                self.root.after(0, lambda: self._play_rendered())
            except Exception as exc:
                self.root.after(0, lambda: self.status_var.set(f"ERROR: {exc}"))
            finally:
                self.rendering = False
                self.root.after(0, self._check_autoplay)

        threading.Thread(target=do_render, daemon=True).start()

    def _play_rendered(self):
        if self.rendered_audio is not None:
            sd.default.reset()
            sd.play(self.rendered_audio, SR)
            self.status_var.set("Playing...")

    def _on_play_dry(self):
        if self.source_audio is not None:
            sd.default.reset()
            sd.play(self.source_audio, SR)
            self.status_var.set("Playing dry...")

    def _on_stop(self):
        sd.stop()
        self.status_var.set("Stopped")
