"""FDN Reverb GUI — parameter control with WAV file rendering and playback.

Tabs: Parameters · Presets · Waveforms · Spectrograms · Signal Flow · Guide
"""

import math
import os
import tkinter as tk
from tkinter import ttk

import numpy as np

from reverb.engine.fdn import render_fdn
from reverb.engine.params import (
    SR, SCHEMA, default_params, bypass_params, PARAM_RANGES, PARAM_SECTIONS, CHOICE_RANGES,
)
from reverb.engine.matrix import MATRIX_TYPES, get_matrix, nearest_unitary, is_unitary
from shared.llm_guide_text import REVERB_GUIDE, REVERB_PARAM_DESCRIPTIONS
from shared.gui import PedalGUIBase, PedalConfig

PRESET_DIR = os.path.join(os.path.dirname(__file__), "presets")
DEFAULT_SOURCE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "audio", "test_signals", "dry_chords.wav",
)

PRESET_CATEGORIES = [
    "Plate", "Room", "Hall", "Ambient", "Modulated",
    "Experimental", "Sound Design", "ML Generated",
]

SECTION_DESCRIPTIONS = {
    "Delay Times (ms)": "<15=small  50-80=room  long=hall  use prime-ish ratios",
    "Damping (per node)": "0=bright  0.3=warm  0.7+=dark/muffled",
    "Input Gains": "how much input feeds each node (default 0.125)",
    "Output Gains": "0=silent  1=normal  >1=amplified",
    "Node Pans (L/R)": "-1=left  0=center  1=right",
    "Mod Rate Multipliers": "per-node LFO rate = master x multiplier",
}


def _make_config():
    return PedalConfig(
        name="Reverb",
        preset_dir=PRESET_DIR,
        preset_categories=PRESET_CATEGORIES,
        window_title="FDN Reverb",
        window_geometry="1250x950",
        default_params=default_params,
        bypass_params=bypass_params,
        param_ranges=PARAM_RANGES,
        param_sections=PARAM_SECTIONS,
        choice_ranges=CHOICE_RANGES,
        render=render_fdn,
        render_stereo=None,
        guide_text=REVERB_GUIDE,
        param_descriptions=REVERB_PARAM_DESCRIPTIONS,
        sample_rate=SR,
        default_source=DEFAULT_SOURCE,
        icon_path=os.path.join(os.path.dirname(__file__), os.pardir, os.pardir,
                               "icons", "reverb.png"),
        extra_tabs=[],  # set in __init__ (needs self ref)
        tail_param="tail_length",
        schema=SCHEMA,
    )


class ReverbGUI(PedalGUIBase):

    def __init__(self, root):
        cfg = _make_config()
        cfg.extra_tabs = [("Signal Flow", self._build_signal_flow_page)]
        self.custom_matrix = None
        self._param_section = {}
        self._current_section = None
        self._diagram_redraw_id = None
        super().__init__(root, cfg)

    # ------------------------------------------------------------------
    # Slider helpers
    # ------------------------------------------------------------------

    def _add_slider(self, parent, row, key, label, lo, hi, value, **kw):
        """Wrap base _add_slider to track param->section mapping."""
        row = super()._add_slider(parent, row, key, label, lo, hi, value, **kw)
        self._param_section[key] = self._current_section
        return row

    def _section(self, parent, row, title, lockable=True):
        """Section separator with optional lock and description."""
        ttk.Separator(parent, orient="horizontal").grid(
            row=row, column=0, columnspan=4, sticky="ew", pady=(10, 2))
        header = ttk.Frame(parent)
        header.grid(row=row + 1, column=0, columnspan=4, sticky="w", pady=(0, 1))
        ttk.Label(header, text=title, font=("TkDefaultFont", 0, "bold")).pack(side="left")
        if lockable:
            lock_var = tk.BooleanVar(value=False)
            self.section_locks[title] = lock_var
            ttk.Checkbutton(header, text="\U0001F512", variable=lock_var).pack(
                side="left", padx=(6, 0))
        next_row = row + 2
        if title in SECTION_DESCRIPTIONS:
            ttk.Label(parent, text=SECTION_DESCRIPTIONS[title],
                      foreground="#888888", font=("Helvetica", 9)).grid(
                row=next_row, column=0, columnspan=4, sticky="w", pady=(0, 3))
            next_row += 1
        self._current_section = title
        return next_row

    def _add_node_slider(self, parent, row, key, label, lo, hi, value,
                         var_list, fmt=".2f", length=250):
        """Add a per-node slider (simpler than global, no description)."""
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=0)
        var = tk.DoubleVar(value=value)
        slider_frame = ttk.Frame(parent)
        slider_frame.grid(row=row, column=1, sticky="ew", padx=5, pady=0)
        min_text = f"{lo:{fmt}}"
        max_text = f"{hi:{fmt}}"
        ttk.Label(slider_frame, text=min_text, width=len(min_text) + 1,
                  foreground="#888888", font=("Helvetica", 9)).pack(side="left")
        scale = ttk.Scale(slider_frame, from_=lo, to=hi, variable=var,
                          orient="horizontal", length=length)
        scale.pack(side="left", fill="x", expand=True)
        ttk.Label(slider_frame, text=max_text, width=len(max_text) + 1,
                  foreground="#888888", font=("Helvetica", 9)).pack(side="left")
        self._bind_scroll(scale, var, lo, hi)
        entry_var = tk.StringVar(value=f"{value:{fmt}}")
        entry = ttk.Entry(parent, textvariable=entry_var, width=7, justify="right")
        entry.grid(row=row, column=2, sticky="w", pady=0)

        def _on_slider_change(*_a, v=var, ev=entry_var, f=fmt):
            ev.set(f"{v.get():{f}}")
        var.trace_add("write", _on_slider_change)

        def _on_entry_return(event, v=var, ev=entry_var, _lo=lo, _hi=hi, f=fmt):
            try:
                val = float(ev.get())
                val = max(_lo, min(_hi, val))
                v.set(val)
                self._schedule_autoplay()
            except ValueError:
                ev.set(f"{v.get():{f}}")
        entry.bind("<Return>", _on_entry_return)
        entry.bind("<FocusOut>", _on_entry_return)
        var_list.append(var)
        lock_var = tk.BooleanVar(value=False)
        self.locks[key] = lock_var
        self._param_section[key] = self._current_section
        ttk.Checkbutton(parent, variable=lock_var).grid(row=row, column=3, pady=0)
        scale.bind("<ButtonRelease-1>", lambda e: self._schedule_autoplay())
        return row + 1

    def _is_locked(self, key):
        """Check individual lock or section-level lock."""
        if key in self.locks and self.locks[key].get():
            return True
        section = self._param_section.get(key)
        if section and section in self.section_locks and self.section_locks[section].get():
            return True
        return False

    # ------------------------------------------------------------------
    # Parameters tab
    # ------------------------------------------------------------------

    def _build_params_page(self, parent):
        inner = self._build_params_container(parent)

        columns = ttk.Frame(inner)
        columns.pack(side="top", fill="both", expand=True)
        left = ttk.Frame(columns, padding=5)
        left.pack(side="left", fill="both", expand=True, anchor="n")
        right = ttk.Frame(columns, padding=5)
        right.pack(side="left", fill="both", expand=True, anchor="n")

        # ============ LEFT COLUMN ============
        f = left
        row = 0

        # --- Global ---
        row = self._section(f, row, "Global")
        ttk.Button(f, text="Randomize Knobs", command=self._randomize_knobs).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(0, 4))
        row += 1
        row = self._add_slider(f, row, "feedback_gain", "Feedback Gain", 0.0, 2.0,
                               self.params["feedback_gain"], length=220)
        row = self._add_slider(f, row, "wet_dry", "Wet/Dry Mix", 0.0, 1.0,
                               self.params["wet_dry"], length=220)
        self.locks["wet_dry"].set(True)  # locked by default
        row = self._add_slider(f, row, "diffusion", "Diffusion", 0.0, 0.7,
                               self.params["diffusion"], length=220)
        row = self._add_slider(f, row, "saturation", "Saturation", 0.0, 1.0,
                               self.params.get("saturation", 0.0), length=220)
        row = self._add_slider(f, row, "stereo_width", "Stereo Width", 0.0, 1.0,
                               self.params.get("stereo_width", 1.0), length=220)
        row = self._add_slider(f, row, "pre_delay_ms", "Pre-delay (ms)", 0.0, 250.0,
                               self.params["pre_delay"] / SR * 1000, length=220)
        row = self._add_slider(f, row, "tail_length", "Tail Length (s)", 0.0, 60.0, 2.0,
                               length=220)

        # Matrix topology combobox
        ttk.Label(f, text="Matrix Topology:").grid(row=row, column=0, sticky="w", padx=4)
        self.matrix_var = tk.StringVar(value=self.params.get("matrix_type", "householder"))
        topology_options = list(MATRIX_TYPES.keys()) + ["custom"]
        combo = ttk.Combobox(f, textvariable=self.matrix_var, values=topology_options,
                             state="readonly", width=20)
        combo.grid(row=row, column=1, sticky="w", padx=4)
        combo.bind("<<ComboboxSelected>>", self._on_topology_changed)
        self.locks["matrix_type"] = tk.BooleanVar(value=False)
        self._param_section["matrix_type"] = "Global"
        row += 1

        # --- Matrix Heatmap ---
        row = self._section(f, row, "Feedback Matrix")
        row = self._build_heatmap_ui(f, row)

        # --- XY Pad ---
        row = self._section(f, row, "XY Pad", lockable=False)
        row = self._build_xy_pad(f, row)

        # --- Modulation ---
        row = self._section(f, row, "Modulation")
        row = self._add_slider(f, row, "mod_master_rate", "Master Rate (Hz)",
                               0.0, 100.0, 0.0, length=220)
        row = self._add_slider(f, row, "mod_depth_delay_global", "Delay Depth (smp)",
                               0.0, 100.0, 0.0, length=220)
        row = self._add_slider(f, row, "mod_depth_damping_global", "Damp Depth",
                               0.0, 0.5, 0.0, length=220)
        row = self._add_slider(f, row, "mod_depth_output_global", "Out Gain Depth",
                               0.0, 1.0, 0.0, length=220)
        row = self._add_slider(f, row, "mod_depth_matrix", "Matrix Depth",
                               0.0, 1.0, 0.0, length=220)
        row = self._add_slider(f, row, "mod_correlation", "Correlation",
                               0.0, 1.0, 1.0, length=220)

        # Mod waveform combobox
        ttk.Label(f, text="Mod Waveform:").grid(row=row, column=0, sticky="w", padx=4)
        self.mod_waveform_var = tk.StringVar(value="sine")
        ttk.Combobox(f, textvariable=self.mod_waveform_var,
                     values=["sine", "triangle", "sample_hold"],
                     state="readonly", width=15).grid(row=row, column=1, sticky="w", padx=4)
        self.locks["mod_waveform"] = tk.BooleanVar(value=False)
        self._param_section["mod_waveform"] = self._current_section
        ttk.Checkbutton(f, variable=self.locks["mod_waveform"]).grid(row=row, column=3)
        row += 1

        f.columnconfigure(1, weight=1)

        # Sync XY crosshair when sliders change
        xy_params = list(self._xy_ranges.keys())
        for key in xy_params:
            if key in self.sliders:
                self.sliders[key].trace_add("write", lambda *a: self._xy_sync())
        self.xy_x_var.trace_add("write", lambda *a: self._xy_sync())
        self.xy_y_var.trace_add("write", lambda *a: self._xy_sync())

        # Schedule diagram redraw when any parameter changes
        for var in self.sliders.values():
            var.trace_add("write", self._schedule_diagram_redraw)
        self.matrix_var.trace_add("write", self._schedule_diagram_redraw)

        # ============ RIGHT COLUMN ============
        f = right
        row = 0
        sl = 250

        row = self._section(f, row, "Delay Times (ms)")
        self.delay_sliders = []
        for i in range(8):
            ms = self.params["delay_times"][i] / SR * 1000
            row = self._add_node_slider(f, row, f"delay_{i}", f"Node {i}",
                                        0.5, 300.0, ms, self.delay_sliders,
                                        fmt=".1f", length=sl)

        row = self._section(f, row, "Damping (per node)")
        self.damping_sliders = []
        for i in range(8):
            row = self._add_node_slider(f, row, f"damp_{i}", f"Node {i}",
                                        0.0, 0.99, self.params["damping_coeffs"][i],
                                        self.damping_sliders, fmt=".2f", length=sl)

        row = self._section(f, row, "Input Gains")
        self.input_gain_sliders = []
        for i in range(8):
            row = self._add_node_slider(f, row, f"ig_{i}", f"Node {i}",
                                        0.0, 0.5, self.params["input_gains"][i],
                                        self.input_gain_sliders, fmt=".3f", length=sl)

        row = self._section(f, row, "Output Gains")
        self.output_gain_sliders = []
        for i in range(8):
            row = self._add_node_slider(f, row, f"og_{i}", f"Node {i}",
                                        0.0, 2.0, self.params["output_gains"][i],
                                        self.output_gain_sliders, fmt=".2f", length=sl)

        row = self._section(f, row, "Node Pans (L/R)")
        self.node_pan_sliders = []
        default_pans = self.params.get("node_pans",
            [-1.0, -0.714, -0.429, -0.143, 0.143, 0.429, 0.714, 1.0])
        for i in range(8):
            row = self._add_node_slider(f, row, f"pan_{i}", f"Node {i}",
                                        -1.0, 1.0, default_pans[i],
                                        self.node_pan_sliders, fmt=".2f", length=sl)

        row = self._section(f, row, "Mod Rate Multipliers")
        self.mod_rate_mult_sliders = []
        for i in range(8):
            row = self._add_node_slider(f, row, f"mrm_{i}", f"Node {i}",
                                        0.25, 4.0, 1.0,
                                        self.mod_rate_mult_sliders, fmt=".2f", length=sl)

        # Diagram redraw on per-node changes
        for var_list in (self.delay_sliders, self.damping_sliders,
                         self.input_gain_sliders, self.output_gain_sliders,
                         self.node_pan_sliders, self.mod_rate_mult_sliders):
            for var in var_list:
                var.trace_add("write", self._schedule_diagram_redraw)

        f.columnconfigure(1, weight=1)

        # Autoplay on discrete param changes
        self.matrix_var.trace_add("write", lambda *_: self._schedule_autoplay())
        self.mod_waveform_var.trace_add("write", lambda *_: self._schedule_autoplay())

    # ------------------------------------------------------------------
    # Heatmap UI
    # ------------------------------------------------------------------

    def _build_heatmap_ui(self, parent, row):
        controls = ttk.Frame(parent)
        controls.grid(row=row, column=0, columnspan=4, sticky="w")
        row += 1

        ttk.Button(controls, text="Snap Unitary",
                   command=self._on_snap_unitary).pack(side="left", padx=2)
        ttk.Button(controls, text="Use Custom",
                   command=self._on_use_custom_matrix).pack(side="left", padx=2)
        ttk.Button(controls, text="Randomize",
                   command=self._on_randomize_matrix).pack(side="left", padx=2)
        self.random_unitary_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls, text="Unitary",
                        variable=self.random_unitary_var).pack(side="left", padx=2)
        self.unitary_label = ttk.Label(controls, text="")
        self.unitary_label.pack(side="left", padx=5)

        self.heatmap_size = 200
        self.heatmap_cell = self.heatmap_size // 8
        self.heatmap_canvas = tk.Canvas(parent, width=self.heatmap_size + 40,
                                         height=self.heatmap_size + 40,
                                         bg="gray20", highlightthickness=0)
        self.heatmap_canvas.grid(row=row, column=0, columnspan=4, pady=3)
        self.heatmap_canvas.bind("<Button-1>", self._on_heatmap_click)
        self.heatmap_canvas.bind("<B1-Motion>", self._on_heatmap_drag)
        self.heatmap_canvas.bind("<Button-3>", self._on_heatmap_right_click)
        row += 1

        edit_frame = ttk.Frame(parent)
        edit_frame.grid(row=row, column=0, columnspan=4, sticky="w")
        row += 1
        ttk.Label(edit_frame, text="Cell:").pack(side="left")
        self.cell_label = ttk.Label(edit_frame, text="(-, -)", width=6)
        self.cell_label.pack(side="left", padx=2)
        ttk.Label(edit_frame, text="Val:").pack(side="left", padx=(5, 0))
        self.cell_value_var = tk.StringVar(value="0.00")
        cell_entry = ttk.Entry(edit_frame, textvariable=self.cell_value_var, width=8)
        cell_entry.pack(side="left", padx=2)
        cell_entry.bind("<Return>", self._on_cell_value_enter)
        self.selected_cell = None

        self._load_matrix_from_topology()
        self._draw_heatmap()
        return row

    # ------------------------------------------------------------------
    # XY Pad
    # ------------------------------------------------------------------

    _xy_ranges = {
        "feedback_gain": (0.0, 2.0), "wet_dry": (0.0, 1.0),
        "diffusion": (0.0, 0.7), "pre_delay_ms": (0.0, 250.0),
        "saturation": (0.0, 1.0), "stereo_width": (0.0, 1.0),
        "mod_master_rate": (0.0, 100.0), "mod_depth_delay_global": (0.0, 100.0),
    }

    def _build_xy_pad(self, parent, row):
        xy_controls = ttk.Frame(parent)
        xy_controls.grid(row=row, column=0, columnspan=4, sticky="w")
        row += 1

        xy_params = list(self._xy_ranges.keys())
        ttk.Label(xy_controls, text="X:").pack(side="left")
        self.xy_x_var = tk.StringVar(value="feedback_gain")
        ttk.Combobox(xy_controls, textvariable=self.xy_x_var, values=xy_params,
                     state="readonly", width=15).pack(side="left", padx=(2, 10))
        ttk.Label(xy_controls, text="Y:").pack(side="left")
        self.xy_y_var = tk.StringVar(value="diffusion")
        ttk.Combobox(xy_controls, textvariable=self.xy_y_var, values=xy_params,
                     state="readonly", width=15).pack(side="left", padx=2)

        self.xy_size = 200
        self.xy_canvas = tk.Canvas(parent, width=self.xy_size, height=self.xy_size,
                                   bg="black", highlightthickness=1,
                                   highlightbackground="gray")
        self.xy_canvas.grid(row=row, column=0, columnspan=4, pady=3)
        self.xy_canvas.bind("<B1-Motion>", self._on_xy_drag)
        self.xy_canvas.bind("<Button-1>", self._on_xy_drag)
        self._xy_draw_crosshair(self.xy_size // 2, self.xy_size // 2)
        return row + 1

    def _xy_draw_crosshair(self, x, y):
        self.xy_canvas.delete("crosshair")
        s = self.xy_size
        self.xy_canvas.create_line(x, 0, x, s, fill="gray30", tags="crosshair")
        self.xy_canvas.create_line(0, y, s, y, fill="gray30", tags="crosshair")
        self.xy_canvas.create_oval(x - 6, y - 6, x + 6, y + 6,
                                   outline="cyan", width=2, tags="crosshair")

    def _xy_sync(self):
        """Update crosshair position from current slider values."""
        s = self.xy_size

        def _norm(key):
            if key in self.sliders and key in self._xy_ranges:
                lo, hi = self._xy_ranges[key]
                return (self.sliders[key].get() - lo) / (hi - lo)
            return 0.5

        px = int(max(0, min(s, _norm(self.xy_x_var.get()) * s)))
        py = int(max(0, min(s, (1.0 - _norm(self.xy_y_var.get())) * s)))
        self._xy_draw_crosshair(px, py)

    def _on_xy_drag(self, event):
        s = self.xy_size
        nx = max(0.0, min(1.0, event.x / s))
        ny = max(0.0, min(1.0, 1.0 - event.y / s))
        for key, val in [(self.xy_x_var.get(), nx), (self.xy_y_var.get(), ny)]:
            if key in self.sliders and key in self._xy_ranges:
                lo, hi = self._xy_ranges[key]
                self.sliders[key].set(lo + val * (hi - lo))
        self._xy_draw_crosshair(event.x, event.y)

    # ------------------------------------------------------------------
    # Param read/write
    # ------------------------------------------------------------------

    def _read_params_from_ui(self):
        p = dict(self.params)
        p["feedback_gain"] = self.sliders["feedback_gain"].get()
        p["wet_dry"] = self.sliders["wet_dry"].get()
        p["diffusion"] = self.sliders["diffusion"].get()
        p["saturation"] = self.sliders["saturation"].get()
        p["pre_delay"] = int(self.sliders["pre_delay_ms"].get() / 1000 * SR)
        p["matrix_type"] = self.matrix_var.get()
        p["delay_times"] = [max(1, int(s.get() / 1000 * SR)) for s in self.delay_sliders]
        p["damping_coeffs"] = [s.get() for s in self.damping_sliders]
        p["input_gains"] = [s.get() for s in self.input_gain_sliders]
        p["output_gains"] = [s.get() for s in self.output_gain_sliders]
        p["stereo_width"] = self.sliders["stereo_width"].get()
        p["node_pans"] = [s.get() for s in self.node_pan_sliders]
        # Modulation
        wf_map = {"sine": 0, "triangle": 1, "sample_hold": 2}
        p["mod_master_rate"] = self.sliders["mod_master_rate"].get()
        p["mod_waveform"] = wf_map.get(self.mod_waveform_var.get(), 0)
        p["mod_correlation"] = self.sliders["mod_correlation"].get()
        p["mod_depth_delay"] = [self.sliders["mod_depth_delay_global"].get()] * 8
        p["mod_depth_damping"] = [self.sliders["mod_depth_damping_global"].get()] * 8
        p["mod_depth_output"] = [self.sliders["mod_depth_output_global"].get()] * 8
        p["mod_depth_matrix"] = self.sliders["mod_depth_matrix"].get()
        p["mod_node_rate_mult"] = [s.get() for s in self.mod_rate_mult_sliders]
        if p["matrix_type"] == "custom" and self.custom_matrix is not None:
            p["matrix_custom"] = self.custom_matrix.tolist()
        return p

    def _write_params_to_ui(self, p):
        self.params = p
        self.sliders["feedback_gain"].set(p["feedback_gain"])
        self.sliders["wet_dry"].set(p["wet_dry"])
        self.sliders["diffusion"].set(p["diffusion"])
        self.sliders["saturation"].set(p.get("saturation", 0.0))
        self.sliders["pre_delay_ms"].set(p["pre_delay"] / SR * 1000)
        self.matrix_var.set(p.get("matrix_type", "householder"))
        self.sliders["stereo_width"].set(p.get("stereo_width", 1.0))
        node_pans = p.get("node_pans",
            [-1.0, -0.714, -0.429, -0.143, 0.143, 0.429, 0.714, 1.0])
        for i in range(8):
            self.delay_sliders[i].set(p["delay_times"][i] / SR * 1000)
            self.damping_sliders[i].set(p["damping_coeffs"][i])
            self.input_gain_sliders[i].set(p["input_gains"][i])
            self.output_gain_sliders[i].set(p["output_gains"][i])
            self.node_pan_sliders[i].set(node_pans[i])
        # Modulation
        self.sliders["mod_master_rate"].set(p.get("mod_master_rate", 0.0))
        dd = p.get("mod_depth_delay", [0.0] * 8)
        self.sliders["mod_depth_delay_global"].set(dd[0] if isinstance(dd, list) else dd)
        ddamp = p.get("mod_depth_damping", [0.0] * 8)
        self.sliders["mod_depth_damping_global"].set(ddamp[0] if isinstance(ddamp, list) else ddamp)
        dout = p.get("mod_depth_output", [0.0] * 8)
        self.sliders["mod_depth_output_global"].set(dout[0] if isinstance(dout, list) else dout)
        self.sliders["mod_depth_matrix"].set(p.get("mod_depth_matrix", 0.0))
        self.sliders["mod_correlation"].set(p.get("mod_correlation", 1.0))
        wf_rmap = {0: "sine", 1: "triangle", 2: "sample_hold"}
        self.mod_waveform_var.set(wf_rmap.get(p.get("mod_waveform", 0), "sine"))
        mrm = p.get("mod_node_rate_mult", [1.0] * 8)
        for i in range(8):
            self.mod_rate_mult_sliders[i].set(mrm[i])
        # Update heatmap
        if "matrix_custom" in p and p.get("matrix_type") == "custom":
            self.custom_matrix = np.array(p["matrix_custom"])
        else:
            self._load_matrix_from_topology()
        self._draw_heatmap()

    # ------------------------------------------------------------------
    # Randomize (custom: handles per-node arrays + matrix)
    # ------------------------------------------------------------------

    def _randomize_knobs(self):
        """Randomize all unlocked parameters without playing."""
        rng = np.random.default_rng()
        for key, lo, hi in [("feedback_gain", 0.0, 2.0), ("wet_dry", 0.0, 1.0),
                             ("diffusion", 0.0, 0.7), ("saturation", 0.0, 1.0),
                             ("pre_delay_ms", 0.0, 250.0), ("stereo_width", 0.0, 1.0)]:
            if not self._is_locked(key):
                self.sliders[key].set(rng.uniform(lo, hi))
        node_groups = [
            (self.delay_sliders, "delay", 0.5, 300.0),
            (self.damping_sliders, "damp", 0.0, 0.99),
            (self.input_gain_sliders, "ig", 0.0, 0.5),
            (self.output_gain_sliders, "og", 0.0, 2.0),
            (self.node_pan_sliders, "pan", -1.0, 1.0),
        ]
        for var_list, prefix, lo, hi in node_groups:
            for i in range(8):
                if not self._is_locked(f"{prefix}_{i}"):
                    var_list[i].set(rng.uniform(lo, hi))
        for key, lo, hi in [("mod_master_rate", 0.0, 20.0),
                             ("mod_depth_delay_global", 0.0, 30.0),
                             ("mod_depth_damping_global", 0.0, 0.3),
                             ("mod_depth_output_global", 0.0, 0.5),
                             ("mod_depth_matrix", 0.0, 0.5),
                             ("mod_correlation", 0.0, 1.0)]:
            if not self._is_locked(key):
                self.sliders[key].set(rng.uniform(lo, hi))
        if not self._is_locked("mod_waveform"):
            self.mod_waveform_var.set(rng.choice(["sine", "triangle", "sample_hold"]))
        for i in range(8):
            if not self._is_locked(f"mrm_{i}"):
                self.mod_rate_mult_sliders[i].set(rng.choice([0.5, 1.0, 1.0, 2.0, 3.0]))
        # Matrix — respect section lock
        matrix_locked = ("Feedback Matrix" in self.section_locks
                         and self.section_locks["Feedback Matrix"].get())
        if not matrix_locked:
            mat = rng.standard_normal((8, 8))
            if rng.random() < 0.5:
                mat = nearest_unitary(mat)
            self.custom_matrix = mat
            self.matrix_var.set("custom")
            self._draw_heatmap()
        self.status_var.set("Randomized all")

    def _randomize_params(self):
        """Override base: randomize + play."""
        self._randomize_knobs()
        self._on_play()

    # ------------------------------------------------------------------
    # Matrix heatmap
    # ------------------------------------------------------------------

    def _load_matrix_from_topology(self):
        name = self.matrix_var.get()
        if name in MATRIX_TYPES:
            self.custom_matrix = get_matrix(name, 8,
                                            seed=self.params.get("matrix_seed", 42))
        elif self.custom_matrix is None:
            self.custom_matrix = get_matrix("householder", 8)

    def _draw_heatmap(self):
        if self.custom_matrix is None:
            return
        c = self.heatmap_canvas
        c.delete("all")
        cell = self.heatmap_cell
        pad = 30
        vmax = max(np.max(np.abs(self.custom_matrix)), 0.01)

        for row in range(8):
            for col in range(8):
                val = self.custom_matrix[row, col]
                intensity = min(abs(val) / vmax, 1.0)
                if val >= 0:
                    r, g, b = int(40 + 215 * intensity), int(40 + 80 * intensity), 40
                else:
                    r, g, b = 40, int(40 + 80 * intensity), int(40 + 215 * intensity)
                color = f"#{r:02x}{g:02x}{b:02x}"
                x0, y0 = pad + col * cell, pad + row * cell
                c.create_rectangle(x0, y0, x0 + cell, y0 + cell,
                                   fill=color, outline="gray40")
                text_color = "white" if intensity > 0.3 else "gray70"
                fmt = f"{val:.1f}" if cell < 30 else f"{val:.2f}"
                c.create_text(x0 + cell // 2, y0 + cell // 2,
                              text=fmt, fill=text_color, font=("TkDefaultFont", 7))

        for i in range(8):
            c.create_text(pad + i * cell + cell // 2, pad - 12,
                          text=str(i), fill="gray70", font=("TkDefaultFont", 9))
            c.create_text(pad - 12, pad + i * cell + cell // 2,
                          text=str(i), fill="gray70", font=("TkDefaultFont", 9))
        c.create_text(pad + 4 * cell, pad - 25, text="from delay line",
                      fill="gray50", font=("TkDefaultFont", 8))
        c.create_text(pad - 25, pad + 4 * cell, text="to",
                      fill="gray50", font=("TkDefaultFont", 8), angle=90)

        if self.selected_cell:
            sr, sc = self.selected_cell
            x0, y0 = pad + sc * cell, pad + sr * cell
            c.create_rectangle(x0, y0, x0 + cell, y0 + cell, outline="cyan", width=2)

        unitary = is_unitary(self.custom_matrix)
        self.unitary_label.config(
            text="Unitary: Yes" if unitary else "Unitary: No",
            foreground="green" if unitary else "orange")

    def _heatmap_cell_at(self, x, y):
        pad, cell = 30, self.heatmap_cell
        col, row = (x - pad) // cell, (y - pad) // cell
        if 0 <= row < 8 and 0 <= col < 8:
            return int(row), int(col)
        return None

    def _on_heatmap_click(self, event):
        pos = self._heatmap_cell_at(event.x, event.y)
        if pos is None:
            return
        row, col = pos
        self.selected_cell = (row, col)
        self.cell_label.config(text=f"({row}, {col})")
        self.cell_value_var.set(f"{self.custom_matrix[row, col]:.3f}")
        self._draw_heatmap()

    def _on_heatmap_drag(self, event):
        if self.selected_cell is None or self.custom_matrix is None:
            return
        row, col = self.selected_cell
        pad, cell = 30, self.heatmap_cell
        cell_center_y = pad + row * cell + cell // 2
        delta = (cell_center_y - event.y) / (cell * 2)
        new_val = max(-1.5, min(1.5, delta * 2))
        self.custom_matrix[row, col] = new_val
        self.cell_value_var.set(f"{new_val:.3f}")
        self._draw_heatmap()

    def _on_heatmap_right_click(self, event):
        pos = self._heatmap_cell_at(event.x, event.y)
        if pos is None or self.custom_matrix is None:
            return
        row, col = pos
        self.custom_matrix[row, col] = 0.0
        self.selected_cell = (row, col)
        self.cell_label.config(text=f"({row}, {col})")
        self.cell_value_var.set("0.000")
        self._draw_heatmap()

    def _on_cell_value_enter(self, event=None):
        if self.selected_cell is None or self.custom_matrix is None:
            return
        try:
            val = float(self.cell_value_var.get())
        except ValueError:
            return
        row, col = self.selected_cell
        self.custom_matrix[row, col] = val
        self._draw_heatmap()

    def _on_topology_changed(self, event=None):
        name = self.matrix_var.get()
        if name != "custom":
            self._load_matrix_from_topology()
            self._draw_heatmap()
            self.status_var.set(f"Matrix: {name}")

    def _on_snap_unitary(self):
        if self.custom_matrix is None:
            return
        self.custom_matrix = nearest_unitary(self.custom_matrix)
        self._draw_heatmap()
        if self.selected_cell:
            r, c = self.selected_cell
            self.cell_value_var.set(f"{self.custom_matrix[r, c]:.3f}")
        self.status_var.set("Snapped to nearest unitary matrix")

    def _on_use_custom_matrix(self):
        self.matrix_var.set("custom")
        self.status_var.set("Using custom matrix \u2014 edit the heatmap, then Play")

    def _on_randomize_matrix(self):
        if ("Feedback Matrix" in self.section_locks
                and self.section_locks["Feedback Matrix"].get()):
            self.status_var.set("Matrix is locked")
            return
        rng = np.random.default_rng()
        mat = rng.standard_normal((8, 8))
        if self.random_unitary_var.get():
            mat = nearest_unitary(mat)
        self.custom_matrix = mat
        self.matrix_var.set("custom")
        self._draw_heatmap()
        label = "unitary" if self.random_unitary_var.get() else "non-unitary"
        self.status_var.set(f"Randomized matrix ({label})")

    # ------------------------------------------------------------------
    # Tab change & diagram scheduling
    # ------------------------------------------------------------------

    def _on_notebook_changed(self, event=None):
        """Extend base to also refresh signal flow diagram."""
        super()._on_notebook_changed(event)
        current = self._notebook.select()
        sf_frame = self._extra_tab_frames.get("Signal Flow")
        if sf_frame and current == str(sf_frame):
            self._draw_diagram()

    def _schedule_diagram_redraw(self, *args):
        if self._diagram_redraw_id is not None:
            self.root.after_cancel(self._diagram_redraw_id)
        self._diagram_redraw_id = self.root.after(200, self._diagram_redraw_if_visible)

    def _diagram_redraw_if_visible(self):
        self._diagram_redraw_id = None
        try:
            current = self._notebook.select()
            sf_frame = self._extra_tab_frames.get("Signal Flow")
            if sf_frame and current == str(sf_frame):
                self._draw_diagram()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Signal flow diagram
    # ------------------------------------------------------------------

    def _build_signal_flow_page(self, parent):
        self.diagram_canvas = tk.Canvas(parent, bg="#111118", highlightthickness=0)
        self.diagram_canvas.pack(fill="both", expand=True)
        self.diagram_canvas.bind("<Configure>", lambda e: self._draw_diagram())

    def _draw_diagram(self):
        p = self._read_params_from_ui()
        c = self.diagram_canvas
        c.delete("all")
        c.update_idletasks()
        W = c.winfo_width() or 900
        H = c.winfo_height() or 700

        # Colors & Fonts
        WIRE = "#555577"
        TITLE_COL = "#e0e0e0"
        LABEL_COL = "#88aacc"
        VAL_COL = "#ddd8a0"
        DESC_COL = "#666688"
        WARN_COL = "#dd4444"
        NODE_FILL = "#1e2e3e"
        NODE_STROKE = "#4a8aaa"
        POS_COL = "#cc4444"
        NEG_COL = "#4488dd"
        SELF_COL = "#88cc88"
        DAMP_COL = "#cc8844"

        F_TITLE = ("Helvetica", 12, "bold")
        F_LABEL = ("Helvetica", 10)
        F_VAL = ("Helvetica", 11, "bold")
        F_DESC = ("Helvetica", 9)
        F_NODE = ("Helvetica", 10, "bold")
        F_NODE_SM = ("Helvetica", 8)

        # Layout
        left_w = min(240, int(W * 0.28))
        ring_cx = left_w + (W - left_w) / 2
        ring_cy = H / 2
        ring_r = min((W - left_w) / 2 - 60, H / 2 - 60, 260)
        node_r = max(22, ring_r * 0.16)

        def info_row(y, label, val, desc="", warn=False):
            c.create_text(14, y, text=label, fill=LABEL_COL, font=F_LABEL, anchor="w")
            c.create_text(left_w - 10, y - (7 if desc else 0), text=val,
                          fill=WARN_COL if warn else VAL_COL, font=F_VAL, anchor="e")
            if desc:
                c.create_text(left_w - 10, y + 10, text=desc,
                              fill=DESC_COL, font=F_DESC, anchor="e")
            return y + (32 if desc else 26)

        def info_heading(y, text_str):
            c.create_text(14, y, text=text_str, fill=TITLE_COL, font=F_TITLE, anchor="w")
            c.create_line(14, y + 12, left_w - 10, y + 12, fill="#3a3a5a")
            return y + 24

        # ========== LEFT COLUMN ==========
        y = 20
        y = info_heading(y, "SIGNAL CHAIN")

        c.create_text(14, y, text="\u25b6 Input", fill="#6a8a6a", font=F_LABEL, anchor="w")
        y += 22
        c.create_text(left_w // 2, y, text="\u2193", fill=WIRE, font=F_TITLE)
        y += 18

        pre_ms = p["pre_delay"] / SR * 1000
        y = info_row(y, "Pre-delay", f"{pre_ms:.0f} ms",
                     "silence before reverb" if pre_ms > 5 else "")
        c.create_text(left_w // 2, y, text="\u2193", fill=WIRE, font=F_TITLE)
        y += 18

        diff = p.get("diffusion", 0)
        n_st = p.get("diffusion_stages", 0)
        if diff > 0.01 and n_st > 0:
            desc = "heavy smear" if diff > 0.5 else ("softens attack" if diff > 0.2 else "")
            y = info_row(y, "Diffusion", f"{n_st}x AP g={diff:.2f}", desc)
        else:
            y = info_row(y, "Diffusion", "OFF")
        c.create_text(left_w // 2, y, text="\u2193", fill=WIRE, font=F_TITLE)
        y += 18

        y = info_heading(y, "FDN (8 nodes)")

        delays_ms = [dt / SR * 1000 for dt in p["delay_times"]]
        avg_d = sum(delays_ms) / len(delays_ms)
        min_d, max_d = min(delays_ms), max(delays_ms)
        if avg_d < 5: size_word = "resonator"
        elif avg_d < 15: size_word = "small room"
        elif avg_d < 50: size_word = "medium room"
        elif avg_d < 120: size_word = "large hall"
        else: size_word = "vast space"
        y = info_row(y, "Delays", f"{min_d:.0f}\u2013{max_d:.0f} ms", size_word)

        damps = p["damping_coeffs"]
        avg_damp = sum(damps) / len(damps)
        if avg_damp < 0.1: damp_word = "bright"
        elif avg_damp < 0.4: damp_word = "warm"
        elif avg_damp < 0.7: damp_word = "dark"
        else: damp_word = "very dark"
        y = info_row(y, "Damping", f"avg {avg_damp:.2f}", damp_word)

        fb = p["feedback_gain"]
        if fb < 0.01: fb_word = "none"
        elif fb < 0.5: fb_word = "short decay"
        elif fb < 0.85: fb_word = "medium decay"
        elif fb < 0.98: fb_word = "long tail"
        elif fb <= 1.0: fb_word = "near-infinite"
        else: fb_word = "GROWING"
        y = info_row(y, "Feedback", f"{fb:.2f}", fb_word, warn=(fb > 1.0))

        sat = p.get("saturation", 0)
        if sat > 0.01:
            if sat < 0.3: sat_word = "warm"
            elif sat < 0.7: sat_word = "distorted"
            else: sat_word = "heavy"
            y = info_row(y, "Saturation", f"{sat:.2f}", sat_word)

        matrix_type = p.get("matrix_type", "householder")
        mat_labels = {
            "householder": "uniform coupling", "hadamard": "structured +/\u2212",
            "diagonal": "isolated nodes", "random_orthogonal": "random unitary",
            "circulant": "ring topology", "stautner_puckette": "paired nodes",
            "zero": "disconnected", "custom": "custom wiring",
        }
        y = info_row(y, "Matrix", matrix_type, mat_labels.get(matrix_type, ""))

        c.create_text(left_w // 2, y, text="\u2193", fill=WIRE, font=F_TITLE)
        y += 18
        mix = p["wet_dry"]
        if mix < 0.01: mix_word = "dry only"
        elif mix < 0.3: mix_word = "subtle reverb"
        elif mix < 0.7: mix_word = "balanced"
        elif mix < 0.99: mix_word = "mostly wet"
        else: mix_word = "100% reverb"
        y = info_row(y, "Wet / Dry", f"{mix:.2f}", mix_word)
        c.create_text(left_w // 2, y, text="\u2193", fill=WIRE, font=F_TITLE)
        y += 18
        c.create_text(14, y, text="\u25a0 Output", fill="#6a8a6a", font=F_LABEL, anchor="w")

        c.create_line(left_w, 10, left_w, H - 10, fill="#2a2a4a", width=1)

        # ========== RIGHT AREA: 8-node ring with matrix wiring ==========
        mat = None
        if self.custom_matrix is not None:
            mat = self.custom_matrix
        else:
            try:
                mat = get_matrix(matrix_type, 8, seed=p.get("matrix_seed", 42))
            except Exception:
                mat = np.eye(8)

        node_pos = []
        for i in range(8):
            angle = -math.pi / 2 + i * 2 * math.pi / 8
            nx = ring_cx + ring_r * math.cos(angle)
            ny = ring_cy + ring_r * math.sin(angle)
            node_pos.append((nx, ny))

        vmax = max(np.max(np.abs(mat)), 0.001)
        thresh = vmax * 0.05

        # Draw connections (behind nodes)
        for i in range(8):
            for j in range(8):
                if i == j:
                    continue
                w = mat[i, j]
                if abs(w) < thresh:
                    continue
                strength = abs(w) / vmax
                line_w = max(1, strength * 4.5)
                alpha = max(0.15, min(1.0, strength))
                bg_r, bg_g, bg_b = 0x11, 0x11, 0x18
                cr, cg, cb = (0xcc, 0x44, 0x44) if w > 0 else (0x44, 0x88, 0xdd)
                br = int(bg_r + (cr - bg_r) * alpha)
                bgc = int(bg_g + (cg - bg_g) * alpha)
                bb = int(bg_b + (cb - bg_b) * alpha)
                color = f"#{br:02x}{bgc:02x}{bb:02x}"

                x1, y1 = node_pos[j]
                x2, y2 = node_pos[i]
                dx, dy = x2 - x1, y2 - y1
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < 1:
                    continue
                ux, uy = dx / dist, dy / dist
                gap = node_r + 4
                sx, sy = x1 + ux * gap, y1 + uy * gap
                ex, ey = x2 - ux * gap, y2 - uy * gap
                perp_x, perp_y = -uy, ux
                off = 3.5
                mx = (sx + ex) / 2 + perp_x * off * 4
                my = (sy + ey) / 2 + perp_y * off * 4
                c.create_line(
                    sx + perp_x * off, sy + perp_y * off,
                    mx, my,
                    ex + perp_x * off, ey + perp_y * off,
                    fill=color, width=line_w, smooth=True,
                    arrow="last", arrowshape=(8, 10, 4))

        # Draw nodes
        for i in range(8):
            nx, ny = node_pos[i]

            # Self-loop
            self_w = mat[i, i]
            if abs(self_w) > thresh:
                angle = -math.pi / 2 + i * 2 * math.pi / 8
                out_x = nx + math.cos(angle) * (node_r + 14)
                out_y = ny + math.sin(angle) * (node_r + 14)
                loop_r = max(6, abs(self_w) / vmax * 12)
                c.create_oval(out_x - loop_r, out_y - loop_r,
                              out_x + loop_r, out_y + loop_r,
                              outline=SELF_COL, width=max(1, abs(self_w) / vmax * 2.5))

            # Node circle
            c.create_oval(nx - node_r, ny - node_r, nx + node_r, ny + node_r,
                          fill=NODE_FILL, outline=NODE_STROKE, width=2)
            c.create_text(nx, ny - 6, text=str(i), fill=TITLE_COL, font=F_NODE)
            c.create_text(nx, ny + 8, text=f"{delays_ms[i]:.0f}ms",
                          fill=VAL_COL, font=F_NODE_SM)

            # Damping bar
            if damps[i] > 0.01:
                bar_w = node_r * 1.4
                bar_y = ny + node_r + 4
                c.create_rectangle(nx - bar_w, bar_y, nx + bar_w, bar_y + 4,
                                   fill="#1a1a2e", outline="")
                c.create_rectangle(nx - bar_w, bar_y,
                                   nx - bar_w + bar_w * 2 * damps[i], bar_y + 4,
                                   fill=DAMP_COL, outline="")

        c.create_text(ring_cx, 16, text="FEEDBACK DELAY NETWORK \u2014 Matrix Wiring",
                      fill=TITLE_COL, font=F_TITLE)

        # Legend
        leg_y = H - 50
        leg_x = left_w + 20
        c.create_line(leg_x, leg_y, leg_x + 25, leg_y, fill=POS_COL, width=2,
                      arrow="last", arrowshape=(6, 8, 3))
        c.create_text(leg_x + 30, leg_y, text="positive", fill=DESC_COL,
                      font=F_DESC, anchor="w")
        c.create_line(leg_x + 90, leg_y, leg_x + 115, leg_y, fill=NEG_COL, width=2,
                      arrow="last", arrowshape=(6, 8, 3))
        c.create_text(leg_x + 120, leg_y, text="negative", fill=DESC_COL,
                      font=F_DESC, anchor="w")
        c.create_oval(leg_x + 185, leg_y - 5, leg_x + 195, leg_y + 5,
                      outline=SELF_COL, width=1.5)
        c.create_text(leg_x + 200, leg_y, text="self-loop", fill=DESC_COL,
                      font=F_DESC, anchor="w")
        c.create_rectangle(leg_x + 265, leg_y - 2, leg_x + 280, leg_y + 2,
                           fill=DAMP_COL, outline="")
        c.create_text(leg_x + 285, leg_y, text="damping", fill=DESC_COL,
                      font=F_DESC, anchor="w")
        c.create_text(leg_x, leg_y + 16, text="Line thickness = connection strength",
                      fill="#444466", font=F_DESC, anchor="w")

        if fb > 1.0 and sat < 0.1:
            c.create_rectangle(0, 0, W, 28, fill="#441111", outline="")
            c.create_text(W // 2, 14,
                          text="WARNING: Feedback > 1.0 with no saturation will explode!",
                          fill=WARN_COL, font=F_VAL)

    # ------------------------------------------------------------------
    # Guide page (custom formatted)
    # ------------------------------------------------------------------

    def _build_guide_page(self):
        parent = self.guide_frame
        text = tk.Text(parent, wrap="word", bg="#1a1a2e", fg="#cccccc",
                       font=("Helvetica", 11), padx=20, pady=15,
                       relief="flat", cursor="arrow", spacing1=2, spacing3=4)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        text.pack(fill="both", expand=True)

        text.tag_configure("h1", font=("Helvetica", 16, "bold"), foreground="#e0e0e0",
                           spacing1=14, spacing3=6)
        text.tag_configure("h2", font=("Helvetica", 13, "bold"), foreground="#88aacc",
                           spacing1=12, spacing3=4)
        text.tag_configure("h3", font=("Helvetica", 11, "bold"), foreground="#99bbdd",
                           spacing1=8, spacing3=2)
        text.tag_configure("param", font=("Helvetica", 11, "bold"), foreground="#ddd8a0")
        text.tag_configure("body", font=("Helvetica", 11), foreground="#cccccc")
        text.tag_configure("range", font=("Helvetica", 10), foreground="#888888")
        text.tag_configure("tip", font=("Helvetica", 10, "italic"), foreground="#7a9a7a")
        text.tag_configure("warn", font=("Helvetica", 10, "bold"), foreground="#dd6666")
        text.tag_configure("sep", font=("Helvetica", 4), foreground="#333355")

        def h1(s): text.insert("end", s + "\n", "h1")
        def h2(s): text.insert("end", "\n", "sep"); text.insert("end", s + "\n", "h2")
        def h3(s): text.insert("end", s + "\n", "h3")
        def param(name, rng, desc):
            text.insert("end", f"  {name}", "param")
            text.insert("end", f"  ({rng})\n", "range")
            text.insert("end", f"    {desc}\n\n", "body")
        def tip(s): text.insert("end", f"    {s}\n", "tip")
        def warn(s): text.insert("end", f"    {s}\n", "warn")
        def body(s): text.insert("end", s + "\n\n", "body")

        # ===== CONTENT =====
        h1("FDN Reverb \u2014 Guide")
        body("This is an 8-node Feedback Delay Network reverb built from scratch. "
             "Audio enters the network, passes through 8 parallel delay lines with "
             "damping filters, and is fed back through a mixing matrix. The result "
             "is a dense, natural-sounding reverb tail.")

        h2("Top Bar Controls")
        param("Load WAV", "button", "Load a source audio file (.wav) to process.")
        param("Play Dry", "button", "Play the original unprocessed audio.")
        param("Play", "button",
              "Render the reverb with current settings and play. If settings "
              "haven't changed since the last render, replays the cached result.")
        param("Stop", "button", "Stop audio playback.")
        param("Save WAV", "button",
              "Export the last rendered output as a 16-bit WAV file.")
        param("Save Preset", "button",
              "Save all current parameter values as a named JSON preset.")
        param("Randomize & Play", "button",
              "Randomize every parameter (including the matrix) and immediately "
              "render and play. Great for discovering unexpected sounds.")

        h2("Global Parameters")
        param("Feedback Gain", "0.0 \u2013 2.0",
              "Controls how much energy recirculates through the delay network. "
              "This is the single biggest factor in reverb length.")
        tip("0.0 = no reverb (single echo).  0.5 = short decay.  "
            "0.85 = medium room.  0.95+ = long tail.")
        warn("Values above 1.0 cause the signal to grow each loop \u2014 "
             "this WILL explode unless Saturation is turned up to tame it.")
        text.insert("end", "\n", "body")

        param("Wet/Dry Mix", "0.0 \u2013 1.0",
              "Blend between the original (dry) signal and the reverb (wet) signal.")
        tip("0.0 = dry only.  0.5 = equal blend.  1.0 = 100% reverb.")
        text.insert("end", "\n", "body")

        param("Diffusion", "0.0 \u2013 0.7",
              "Smears the input through a chain of allpass filters before it "
              "enters the FDN. Softens transients and thickens the early "
              "reflections. Uses 4 allpass stages internally.")
        tip("0.0 = bypass (sharp attacks).  0.3 = subtle softening.  "
            "0.5+ = heavy smearing.")
        text.insert("end", "\n", "body")

        param("Saturation", "0.0 \u2013 1.0",
              "Applies tanh soft-clipping inside the feedback loop. At low values "
              "it adds subtle warmth. At high values it creates distortion, drones, "
              "and metallic textures. Critically, it prevents explosion when "
              "Feedback Gain > 1.0.")
        tip("0.0 = clean/linear (classic FDN).  0.3 = warm.  "
            "0.7+ = aggressive distortion.")
        text.insert("end", "\n", "body")

        param("Pre-delay", "0 \u2013 250 ms",
              "Silence inserted before the reverb tail begins. Simulates the "
              "time gap between the direct sound and the first reflections "
              "in a large space.")
        tip("0\u201310 ms = intimate/close.  20\u201350 ms = medium room.  "
            "80+ ms = large hall or cathedral.")
        text.insert("end", "\n", "body")

        param("Tail Length", "0 \u2013 60 s",
              "How many seconds of silence to append after the input, giving "
              "the reverb tail time to decay. This only affects offline "
              "rendering length, not the reverb character itself.")
        text.insert("end", "\n", "body")

        h2("Feedback Matrix")
        body("The feedback matrix determines how energy flows between the 8 delay "
             "lines after each loop. It is the \"shape\" of the reverb's internal "
             "mixing. Unitary matrices preserve energy (no growth or loss from the "
             "matrix itself), which is important for stable, natural decay.")

        h3("Matrix Topologies")
        param("Householder", "unitary",
              "The standard FDN choice (Jot 1991). Every node feeds equally into "
              "every other node. Produces smooth, uniform decay. The default.")
        param("Hadamard", "unitary",
              "Structured +/\u2212 coupling. Some node pairs add, others subtract. "
              "Can sound slightly different in high-frequency decay.")
        param("Diagonal (Identity)", "unitary",
              "No coupling \u2014 each delay line is independent. Sounds metallic "
              "and comb-filter-like. Useful for special effects.")
        param("Random Orthogonal", "unitary",
              "A random unitary matrix (seeded for reproducibility). Non-uniform "
              "coupling \u2014 some paths are stronger than others.")
        param("Circulant", "unitary",
              "Ring topology: node 0 \u2192 1 \u2192 2 \u2192 ... \u2192 7 \u2192 0. "
              "Energy travels in a circle. Sparse coupling.")
        param("Stautner-Puckette", "unitary",
              "Classic paired cross-coupling (1982). Creates 4 pairs of nodes, "
              "each rotated by 45\u00b0. Historically used in early digital reverbs.")
        param("Zero", "NOT unitary",
              "No feedback at all. Each delay line produces one echo then silence. "
              "Useful for testing or building comb-filter effects.")
        param("Custom", "user-editable",
              "Use the heatmap editor to set arbitrary matrix values. "
              "Click cells to select, drag to adjust, right-click to zero.")

        h3("Heatmap Controls")
        param("Click", "heatmap", "Select a cell. Its row/col and value appear below.")
        param("Drag", "heatmap", "Drag up/down on a selected cell to adjust its value.")
        param("Right-click", "heatmap", "Zero out a cell.")
        param("Cell value entry", "text field",
              "Type an exact value and press Enter to set the selected cell.")
        param("Snap Unitary", "button",
              "Project the current matrix to the nearest unitary matrix via SVD. "
              "Useful after manual edits to restore energy preservation.")
        param("Use Custom", "button",
              "Switch to custom matrix mode so the heatmap values are used for rendering.")
        param("Randomize", "button",
              "Generate a random 8\u00d78 matrix. If the Unitary checkbox is on, "
              "projects it to the nearest unitary matrix.")

        h2("XY Pad")
        body("A 2D controller that maps mouse position to any two global parameters. "
             "Select which parameter the X and Y axes control using the dropdowns. "
             "Click or drag on the pad to adjust both parameters simultaneously. "
             "The crosshair syncs with the sliders in both directions.")

        h2("Modulation (Time-Varying FDN)")
        body("Modulation adds life and movement to the reverb by continuously varying "
             "delay times, damping, output gains, and the feedback matrix over time. "
             "An internal LFO (low-frequency oscillator) drives these changes. The result "
             "ranges from subtle chorus and spatial drift to alien FM-like textures.")

        h3("Three Timescales")
        body("Slow (0.01\u20130.5 Hz): The reverb character drifts over several seconds \u2014 "
             "spatial size, brightness, and density evolve gradually. "
             "Medium/LFO (0.5\u201320 Hz): Classic chorus and vibrato effects \u2014 "
             "eliminates metallic ringing from static delay lines. "
             "Fast/Audio-rate (20+ Hz): Creates FM-like inharmonic sidebands \u2014 "
             "alien, bell-like, or granular textures.")

        param("Master Rate", "0 \u2013 100 Hz",
              "The fundamental modulation speed. All per-node LFOs derive their "
              "rate from this value multiplied by their rate multiplier. "
              "Set to 0 for static (classic FDN) behavior.")
        tip("0 = off.  0.1 = slow evolve.  2 = chorus.  80+ = FM territory.")
        text.insert("end", "\n", "body")

        param("Delay Depth", "0 \u2013 100 samples",
              "How far the delay times swing around their base values. "
              "Small values (3\u20135 smp) create subtle chorus. "
              "Large values (20+ smp) create audible pitch wobble.")
        text.insert("end", "\n", "body")

        param("Damp Depth", "0.0 \u2013 0.5",
              "How much the damping coefficients swing. Creates "
              "time-varying brightness \u2014 the reverb breathes between "
              "bright and dark.")
        text.insert("end", "\n", "body")

        param("Out Gain Depth", "0.0 \u2013 1.0",
              "Modulates output gains per node. Creates tremolo-like "
              "amplitude variations in the reverb tail.")
        text.insert("end", "\n", "body")

        param("Matrix Depth", "0.0 \u2013 1.0",
              "Blends between the primary feedback matrix and a secondary "
              "matrix (random orthogonal). The energy routing between "
              "nodes changes over time \u2014 the reverb's internal structure breathes.")
        text.insert("end", "\n", "body")

        param("Correlation", "0.0 \u2013 1.0",
              "Controls phase relationships between the 8 per-node LFOs. "
              "1.0 = all nodes modulate in sync (uniform). "
              "0.0 = phases spread evenly (maximum decorrelation).")
        tip("Low correlation creates wider stereo movement and more complex textures.")
        text.insert("end", "\n", "body")

        param("Mod Waveform", "sine / triangle / sample_hold",
              "The LFO shape. Sine is smooth and natural. Triangle is "
              "slightly brighter. Sample-and-hold creates stepped, "
              "random-sounding modulation.")
        text.insert("end", "\n", "body")

        param("Rate Multipliers", "0.25 \u2013 4.0 per node",
              "Each node's LFO rate = master rate \u00d7 its multiplier. "
              "Use integer ratios (1, 2, 3) for rhythmic relationships "
              "or non-integer values for more chaotic modulation.")
        text.insert("end", "\n", "body")

        h2("Per-Node Parameters (8 nodes)")
        body("Each of the 8 delay lines in the FDN has its own settings. Using "
             "different values per node (rather than identical values) produces "
             "denser, more natural reverb because the echo patterns don't "
             "repeat at a single interval.")

        param("Delay Times", "0.5 \u2013 300 ms",
              "The delay length for each of the 8 delay lines. These are the "
              "\"room dimensions\" of the reverb. Short times (< 15 ms) create "
              "small resonant spaces; long times (50+ ms) create large halls. "
              "Prime-number-like ratios between the delay times produce the "
              "densest, least metallic reverb.")
        tip("Defaults are prime-ish values from 29\u201373 ms (medium room).")
        text.insert("end", "\n", "body")

        param("Damping", "0.0 \u2013 0.99",
              "One-pole lowpass filter coefficient per delay line. Higher values "
              "remove more high frequencies each time the signal passes through "
              "the loop, simulating absorption by soft surfaces (carpet, curtains).")
        tip("0.0 = no filtering (bright).  0.3 = warm.  0.7+ = dark/muffled.")
        text.insert("end", "\n", "body")

        param("Input Gains", "0.0 \u2013 0.5",
              "How much of the input signal is fed into each delay line. "
              "Equal gains (default 1/8 = 0.125) distribute the input evenly. "
              "Unequal gains emphasize certain delay lines over others.")
        text.insert("end", "\n", "body")

        param("Output Gains", "0.0 \u2013 2.0",
              "How much each delay line contributes to the final output mix. "
              "Zeroing a node's output silences it in the output but it still "
              "participates in feedback. Values > 1.0 amplify that node.")
        text.insert("end", "\n", "body")

        h2("Presets Tab")
        body("Save and load parameter snapshots as JSON files. Presets are stored "
             "in gui/presets/. Select a preset from the list and click Load to "
             "restore those settings. Use Save As... to create a new preset from "
             "the current knob positions.")

        h2("Signal Flow Tab")
        body("A live diagram showing the current signal chain from input to "
             "output. Updates automatically to reflect parameter values. Shows "
             "warnings (e.g. feedback > 1.0 with no saturation) and plain-English "
             "descriptions of the current sound character.")

        h2("Tips & Recipes")
        h3("Natural Room Reverb")
        body("Feedback 0.7\u20130.9, damping 0.2\u20130.4, diffusion 0.4\u20130.5, "
             "Householder matrix, delay times 20\u201380 ms with prime-ish ratios.")
        h3("Infinite Drone / Shimmer")
        body("Feedback 1.0\u20131.5, saturation 0.3\u20130.6 (to prevent explosion), "
             "low damping. The signal sustains indefinitely and distorts gently.")
        h3("Metallic / Comb Filter")
        body("Diagonal matrix (no coupling), short delay times (1\u20135 ms), "
             "high feedback. Each delay line resonates independently.")
        h3("Gated Reverb")
        body("Short tail length, high feedback for initial density, "
             "wet/dry around 0.7. The reverb cuts off abruptly.")
        h3("Dark Ambient Wash")
        body("High damping (0.7+), long delay times, feedback 0.9+, "
             "high diffusion. High frequencies die quickly, leaving a warm pad.")

        text.configure(state="disabled")


def main():
    root = tk.Tk()
    ReverbGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
