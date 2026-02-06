"""FDN Reverb GUI — parameter control with WAV file rendering and playback.

Run: uv run python gui/gui.py

Single page with all params. Play button auto-renders if params changed.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import numpy as np
import sounddevice as sd
from scipy.io import wavfile
import json
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.fdn import render_fdn
from engine.params import default_params, SR
from primitives.matrix import MATRIX_TYPES, get_matrix, nearest_unitary, is_unitary

PRESET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets")
TEST_SIGNALS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "audio", "test_signals")


class ReverbGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("FDN Reverb")
        self.params = default_params()
        self.source_audio = None
        self.source_path = None
        self.rendered_audio = None
        self.rendered_warning = ""
        self.rendered_params = None
        self.rendering = False
        self.sliders = {}
        self.custom_matrix = None  # 8x8 numpy array when using custom topology

        self._scroll_widgets = {}  # widget path -> (var, from_, to_)
        self._diagram_redraw_id = None  # for debounced diagram refresh

        default_source = os.path.join(TEST_SIGNALS_DIR, "dry_chords.wav")
        if os.path.exists(default_source):
            self._load_wav(default_source)

        self._build_ui()

    def _build_ui(self):
        self.root.configure(padx=10, pady=10)

        # Top bar
        top = ttk.Frame(self.root)
        top.pack(fill="x", pady=(0, 8))

        ttk.Button(top, text="Load WAV", command=self._on_load_wav).pack(side="left", padx=2)
        self.source_label = ttk.Label(top, text=self._source_text(), width=30, anchor="w")
        self.source_label.pack(side="left", padx=5)
        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=5)
        ttk.Button(top, text="Play Dry", command=self._on_play_dry).pack(side="left", padx=2)
        ttk.Button(top, text="Play", command=self._on_play).pack(side="left", padx=2)
        ttk.Button(top, text="Stop", command=self._on_stop).pack(side="left", padx=2)
        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=5)
        ttk.Button(top, text="Save WAV", command=self._on_save_wav).pack(side="left", padx=2)
        ttk.Button(top, text="Save Preset", command=self._on_save_preset).pack(side="left", padx=2)
        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=5)
        ttk.Button(top, text="Randomize & Play", command=self._on_randomize_and_play).pack(side="left", padx=2)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(top, textvariable=self.status_var, width=25, anchor="e").pack(side="right")

        # Two pages: Params and Presets
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True)

        params_page = ttk.Frame(notebook)
        notebook.add(params_page, text="Parameters")
        self._build_params_page(params_page)

        presets_page = ttk.Frame(notebook, padding=10)
        notebook.add(presets_page, text="Presets")
        self._build_presets_page(presets_page)

        desc_page = ttk.Frame(notebook, padding=10)
        notebook.add(desc_page, text="Signal Flow")
        self._build_description_page(desc_page)

        wave_page = ttk.Frame(notebook, padding=10)
        notebook.add(wave_page, text="Waveform")
        self._build_waveform_page(wave_page)

        guide_page = ttk.Frame(notebook, padding=10)
        notebook.add(guide_page, text="Guide")
        self._build_guide_page(guide_page)

        # Update description when switching to that tab
        notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        self._notebook = notebook

        # Global mousewheel handler — routes to scale under cursor
        self.root.bind_all("<MouseWheel>", self._on_global_scroll)
        # Tk 9.0+ on macOS: trackpad generates TouchpadScroll instead of MouseWheel
        try:
            self.root.bind_all("<TouchpadScroll>", self._on_global_scroll)
        except tk.TclError:
            pass

    def _build_params_page(self, parent):
        # Two-column layout: left = global/matrix/XY, right = per-node sliders
        left = ttk.Frame(parent, padding=5)
        left.pack(side="left", fill="both", expand=True, anchor="n")
        right = ttk.Frame(parent, padding=5)
        right.pack(side="left", fill="both", expand=True, anchor="n")

        # ============ LEFT COLUMN ============
        f = left
        row = 0

        # --- Global ---
        row = self._section(f, row, "Global")
        ttk.Button(f, text="Randomize Knobs", command=self._on_randomize_knobs).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(0, 4))
        row += 1
        row = self._add_slider(f, row, "feedback_gain", "Feedback Gain", 0.0, 2.0,
                               self.params["feedback_gain"], length=220)
        row = self._add_slider(f, row, "wet_dry", "Wet/Dry Mix", 0.0, 1.0,
                               self.params["wet_dry"], length=220)
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

        # Matrix type
        ttk.Label(f, text="Matrix Topology:").grid(row=row, column=0, sticky="w", pady=2)
        self.matrix_var = tk.StringVar(value=self.params.get("matrix_type", "householder"))
        topology_options = list(MATRIX_TYPES.keys()) + ["custom"]
        combo = ttk.Combobox(f, textvariable=self.matrix_var, values=topology_options,
                     state="readonly", width=20)
        combo.grid(row=row, column=1, sticky="w", pady=2)
        combo.bind("<<ComboboxSelected>>", self._on_topology_changed)
        row += 1

        # --- Matrix Heatmap ---
        row = self._section(f, row, "Feedback Matrix")

        heatmap_controls = ttk.Frame(f)
        heatmap_controls.grid(row=row, column=0, columnspan=3, sticky="w")
        row += 1

        ttk.Button(heatmap_controls, text="Snap Unitary",
                   command=self._on_snap_unitary).pack(side="left", padx=2)
        ttk.Button(heatmap_controls, text="Use Custom",
                   command=self._on_use_custom_matrix).pack(side="left", padx=2)
        ttk.Button(heatmap_controls, text="Randomize",
                   command=self._on_randomize_matrix).pack(side="left", padx=2)
        self.random_unitary_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(heatmap_controls, text="Unitary",
                        variable=self.random_unitary_var).pack(side="left", padx=2)
        self.unitary_label = ttk.Label(heatmap_controls, text="")
        self.unitary_label.pack(side="left", padx=5)

        # 8x8 heatmap canvas
        self.heatmap_size = 200
        self.heatmap_cell = self.heatmap_size // 8
        self.heatmap_canvas = tk.Canvas(f, width=self.heatmap_size + 40,
                                         height=self.heatmap_size + 40,
                                         bg="gray20", highlightthickness=0)
        self.heatmap_canvas.grid(row=row, column=0, columnspan=3, pady=3)
        self.heatmap_canvas.bind("<Button-1>", self._on_heatmap_click)
        self.heatmap_canvas.bind("<B1-Motion>", self._on_heatmap_drag)
        self.heatmap_canvas.bind("<Button-3>", self._on_heatmap_right_click)
        row += 1

        # Value editor for selected cell
        edit_frame = ttk.Frame(f)
        edit_frame.grid(row=row, column=0, columnspan=3, sticky="w")
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

        # Initialize heatmap with current topology
        self._load_matrix_from_topology()
        self._draw_heatmap()

        # --- XY Pad ---
        row = self._section(f, row, "XY Pad")
        xy_controls = ttk.Frame(f)
        xy_controls.grid(row=row, column=0, columnspan=3, sticky="w")
        row += 1

        ttk.Label(xy_controls, text="X:").pack(side="left")
        self.xy_x_var = tk.StringVar(value="feedback_gain")
        xy_params = ["feedback_gain", "wet_dry", "diffusion", "saturation", "pre_delay_ms", "stereo_width"]
        ttk.Combobox(xy_controls, textvariable=self.xy_x_var, values=xy_params,
                     state="readonly", width=15).pack(side="left", padx=(2, 10))
        ttk.Label(xy_controls, text="Y:").pack(side="left")
        self.xy_y_var = tk.StringVar(value="diffusion")
        ttk.Combobox(xy_controls, textvariable=self.xy_y_var, values=xy_params,
                     state="readonly", width=15).pack(side="left", padx=2)

        self.xy_size = 200
        self.xy_canvas = tk.Canvas(f, width=self.xy_size, height=self.xy_size, bg="black",
                                   highlightthickness=1, highlightbackground="gray")
        self.xy_canvas.grid(row=row, column=0, columnspan=3, pady=3)
        self.xy_canvas.bind("<B1-Motion>", self._on_xy_drag)
        self.xy_canvas.bind("<Button-1>", self._on_xy_drag)
        self._xy_draw_crosshair(self.xy_size // 2, self.xy_size // 2)

        f.columnconfigure(1, weight=1)

        # Sync XY crosshair when sliders change
        for key in xy_params:
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
        slider_len = 200

        # --- Per-node delay times ---
        row = self._section(f, row, "Delay Times (ms)")
        self.delay_sliders = []
        for i in range(8):
            ms = self.params["delay_times"][i] / SR * 1000
            row = self._add_node_slider(f, row, f"delay_{i}", f"Node {i}", 0.5, 300.0, ms,
                                        self.delay_sliders, fmt=".1f", length=slider_len)

        # --- Per-node damping ---
        row = self._section(f, row, "Damping (per node)")
        self.damping_sliders = []
        for i in range(8):
            row = self._add_node_slider(f, row, f"damp_{i}", f"Node {i}", 0.0, 0.99,
                                        self.params["damping_coeffs"][i],
                                        self.damping_sliders, fmt=".2f", length=slider_len)

        # --- Input gains ---
        row = self._section(f, row, "Input Gains")
        self.input_gain_sliders = []
        for i in range(8):
            row = self._add_node_slider(f, row, f"ig_{i}", f"Node {i}", 0.0, 0.5,
                                        self.params["input_gains"][i],
                                        self.input_gain_sliders, fmt=".3f", length=slider_len)

        # --- Output gains ---
        row = self._section(f, row, "Output Gains")
        self.output_gain_sliders = []
        for i in range(8):
            row = self._add_node_slider(f, row, f"og_{i}", f"Node {i}", 0.0, 2.0,
                                        self.params["output_gains"][i],
                                        self.output_gain_sliders, fmt=".2f", length=slider_len)

        # --- Node Pans ---
        row = self._section(f, row, "Node Pans (L/R)")
        self.node_pan_sliders = []
        default_pans = self.params.get("node_pans",
            [-1.0, -0.714, -0.429, -0.143, 0.143, 0.429, 0.714, 1.0])
        for i in range(8):
            row = self._add_node_slider(f, row, f"pan_{i}", f"Node {i}", -1.0, 1.0,
                                        default_pans[i],
                                        self.node_pan_sliders, fmt=".2f", length=slider_len)

        # Schedule diagram redraw when per-node sliders change
        for var_list in (self.delay_sliders, self.damping_sliders,
                         self.input_gain_sliders, self.output_gain_sliders,
                         self.node_pan_sliders):
            for var in var_list:
                var.trace_add("write", self._schedule_diagram_redraw)

        f.columnconfigure(1, weight=1)

    def _build_presets_page(self, parent):
        ttk.Label(parent, text="Presets:").pack(anchor="w")
        list_frame = ttk.Frame(parent)
        list_frame.pack(fill="both", expand=True, pady=5)

        self.preset_listbox = tk.Listbox(list_frame, height=15)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.preset_listbox.yview)
        self.preset_listbox.configure(yscrollcommand=scrollbar.set)
        self.preset_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self._refresh_preset_list()

        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill="x", pady=5)
        ttk.Button(btn_frame, text="Load", command=self._on_load_preset).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="Save As...", command=self._on_save_preset).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="Refresh", command=self._refresh_preset_list).pack(side="left", padx=2)

    def _build_description_page(self, parent):
        self.diagram_canvas = tk.Canvas(parent, bg="#111118", highlightthickness=0)
        self.diagram_canvas.pack(fill="both", expand=True)
        self.diagram_canvas.bind("<Configure>", lambda e: self._draw_diagram())

    def _build_guide_page(self, parent):
        text = tk.Text(parent, wrap="word", bg="#1a1a2e", fg="#cccccc",
                       font=("Helvetica", 11), padx=20, pady=15,
                       relief="flat", cursor="arrow", spacing1=2, spacing3=4)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        text.pack(fill="both", expand=True)

        # Tag styles
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

        def h1(s):
            text.insert("end", s + "\n", "h1")

        def h2(s):
            text.insert("end", "\n", "sep")
            text.insert("end", s + "\n", "h2")

        def h3(s):
            text.insert("end", s + "\n", "h3")

        def param(name, rng, desc):
            text.insert("end", f"  {name}", "param")
            text.insert("end", f"  ({rng})\n", "range")
            text.insert("end", f"    {desc}\n\n", "body")

        def tip(s):
            text.insert("end", f"    {s}\n", "tip")

        def warn(s):
            text.insert("end", f"    {s}\n", "warn")

        def body(s):
            text.insert("end", s + "\n\n", "body")

        # ===== CONTENT =====

        h1("FDN Reverb — Guide")
        body("This is an 8-node Feedback Delay Network reverb built from scratch. "
             "Audio enters the network, passes through 8 parallel delay lines with "
             "damping filters, and is fed back through a mixing matrix. The result "
             "is a dense, natural-sounding reverb tail.")

        # --- Top Bar ---
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

        # --- Global ---
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

        # --- Matrix ---
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

        # --- XY Pad ---
        h2("XY Pad")
        body("A 2D controller that maps mouse position to any two global parameters. "
             "Select which parameter the X and Y axes control using the dropdowns. "
             "Click or drag on the pad to adjust both parameters simultaneously. "
             "The crosshair syncs with the sliders in both directions.")

        # --- Per-Node ---
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
        tip("0.0 = no filtering (bright).  0.3 = warm.  "
            "0.7+ = dark/muffled.")
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

        # --- Presets ---
        h2("Presets Tab")
        body("Save and load parameter snapshots as JSON files. Presets are stored "
             "in gui/presets/. Select a preset from the list and click Load to "
             "restore those settings. Use Save As... to create a new preset from "
             "the current knob positions.")

        # --- Signal Flow ---
        h2("Signal Flow Tab")
        body("A live diagram showing the current signal chain from input to "
             "output. Updates automatically to reflect parameter values. Shows "
             "warnings (e.g. feedback > 1.0 with no saturation) and plain-English "
             "descriptions of the current sound character.")

        # --- Tips ---
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

    def _build_waveform_page(self, parent):
        self.waveform_canvas = tk.Canvas(parent, bg="#111118", highlightthickness=0)
        self.waveform_canvas.pack(fill="both", expand=True)
        self.waveform_canvas.bind("<Configure>", lambda e: self._draw_waveform())

    def _draw_waveform(self):
        c = self.waveform_canvas
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
        pad_bot = 40
        pad_left = 50
        pad_right = 20
        plot_w = W - pad_left - pad_right
        plot_h = H - pad_top - pad_bot

        if plot_w < 50 or plot_h < 30:
            return

        audio = self.rendered_audio
        if audio is None:
            c.create_text(W // 2, H // 2, text="No rendered audio yet — press Play first",
                          fill=LABEL_COL, font=F_TITLE)
            return

        # Sum stereo to mono for waveform display
        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        n_samples = len(audio)
        duration = n_samples / SR

        c.create_text(W // 2, 16, text=f"Rendered Waveform — {duration:.1f}s — {n_samples} samples",
                      fill=TITLE_COL, font=F_TITLE)

        # Plot area background
        c.create_rectangle(pad_left, pad_top, pad_left + plot_w, pad_top + plot_h,
                           fill="#0a0a14", outline="#333355")

        # Grid lines and labels (amplitude)
        for amp in [-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75]:
            y = pad_top + plot_h / 2 - amp * plot_h / 2
            c.create_line(pad_left, y, pad_left + plot_w, y, fill=GRID_COL)
            if amp in (-0.5, 0.0, 0.5):
                c.create_text(pad_left - 5, y, text=f"{amp:.1f}",
                              fill=LABEL_COL, font=F_LABEL, anchor="e")

        # Zero line
        zero_y = pad_top + plot_h / 2
        c.create_line(pad_left, zero_y, pad_left + plot_w, zero_y, fill="#333355", width=1)

        # Time grid lines
        time_step = max(0.5, round(duration / 8, 1))
        t = 0.0
        while t <= duration:
            x = pad_left + (t / duration) * plot_w
            c.create_line(x, pad_top, x, pad_top + plot_h, fill=GRID_COL)
            c.create_text(x, pad_top + plot_h + 12, text=f"{t:.1f}s",
                          fill=LABEL_COL, font=F_LABEL)
            t += time_step

        # Downsample for display: compute min/max per pixel column
        samples_per_pixel = max(1, n_samples // plot_w)
        n_cols = min(plot_w, n_samples)

        # Build min/max envelope
        wave_points_top = []
        wave_points_bot = []
        for col in range(n_cols):
            start = int(col * n_samples / n_cols)
            end = min(int((col + 1) * n_samples / n_cols), n_samples)
            if start >= end:
                continue
            chunk = audio[start:end]
            mn = float(np.min(chunk))
            mx = float(np.max(chunk))
            x = pad_left + col
            y_top = zero_y - mx * (plot_h / 2)
            y_bot = zero_y - mn * (plot_h / 2)
            # Clamp to plot area
            y_top = max(pad_top, min(pad_top + plot_h, y_top))
            y_bot = max(pad_top, min(pad_top + plot_h, y_bot))
            wave_points_top.append((x, y_top))
            wave_points_bot.append((x, y_bot))

        # Draw filled waveform envelope
        if wave_points_top and wave_points_bot:
            # Build polygon: top left-to-right, then bottom right-to-left
            poly = []
            for x, y in wave_points_top:
                poly.extend([x, y])
            for x, y in reversed(wave_points_bot):
                poly.extend([x, y])
            if len(poly) >= 6:
                c.create_polygon(poly, fill="#1a3550", outline="")
            # Draw top and bottom edges
            if len(wave_points_top) >= 2:
                top_flat = []
                for x, y in wave_points_top:
                    top_flat.extend([x, y])
                c.create_line(*top_flat, fill=WAVE_COL, width=1)
            if len(wave_points_bot) >= 2:
                bot_flat = []
                for x, y in wave_points_bot:
                    bot_flat.extend([x, y])
                c.create_line(*bot_flat, fill=WAVE_COL, width=1)

        # RMS envelope (windowed)
        rms_window = max(1, n_samples // 200)
        rms_points = []
        for col in range(min(200, n_cols)):
            start = int(col * n_samples / 200)
            end = min(start + rms_window * 2, n_samples)
            if start >= end:
                continue
            chunk = audio[start:end]
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            x = pad_left + (col / 200) * plot_w
            y = zero_y - rms * (plot_h / 2)
            y = max(pad_top, min(pad_top + plot_h, y))
            rms_points.extend([x, y])
        if len(rms_points) >= 4:
            c.create_line(*rms_points, fill=RMS_COL, width=1.5, smooth=True)

        # Stats
        peak = float(np.max(np.abs(audio)))
        rms_total = float(np.sqrt(np.mean(audio ** 2)))
        stats_y = pad_top + plot_h + 28
        c.create_text(pad_left, stats_y, text=f"Peak: {peak:.3f}",
                      fill=WAVE_COL, font=F_LABEL, anchor="w")
        c.create_text(pad_left + 120, stats_y, text=f"RMS: {rms_total:.3f}",
                      fill=RMS_COL, font=F_LABEL, anchor="w")
        if hasattr(self, 'rendered_warning') and self.rendered_warning:
            c.create_text(pad_left + 240, stats_y, text=self.rendered_warning.strip(),
                          fill="#dd6666", font=F_LABEL, anchor="w")

    def _on_tab_changed(self, event=None):
        idx = self._notebook.index("current")
        if idx == 2:  # Signal Flow tab
            self._draw_diagram()
        elif idx == 3:  # Waveform tab
            self._draw_waveform()

    def _schedule_diagram_redraw(self, *args):
        """Debounced diagram refresh — redraws 200ms after the last param change."""
        if self._diagram_redraw_id is not None:
            self.root.after_cancel(self._diagram_redraw_id)
        self._diagram_redraw_id = self.root.after(200, self._diagram_redraw_if_visible)

    def _diagram_redraw_if_visible(self):
        """Redraw the signal flow diagram only if its tab is currently selected."""
        self._diagram_redraw_id = None
        try:
            if self._notebook.index("current") == 2:
                self._draw_diagram()
        except Exception:
            pass

    def _draw_diagram(self):
        import math
        p = self._read_params_from_ui()
        c = self.diagram_canvas
        c.delete("all")
        c.update_idletasks()
        W = c.winfo_width() or 900
        H = c.winfo_height() or 700

        # --- Colors & Fonts ---
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

        # --- Layout ---
        left_w = min(240, int(W * 0.28))
        ring_cx = left_w + (W - left_w) / 2
        ring_cy = H / 2
        ring_r = min((W - left_w) / 2 - 60, H / 2 - 60, 260)
        node_r = max(22, ring_r * 0.16)

        # --- Helpers ---
        def info_row(y, label, val, desc="", warn=False):
            c.create_text(14, y, text=label, fill=LABEL_COL, font=F_LABEL, anchor="w")
            c.create_text(left_w - 10, y - (7 if desc else 0), text=val,
                          fill=WARN_COL if warn else VAL_COL, font=F_VAL, anchor="e")
            if desc:
                c.create_text(left_w - 10, y + 10, text=desc,
                              fill=DESC_COL, font=F_DESC, anchor="e")
            return y + (32 if desc else 26)

        def info_heading(y, text):
            c.create_text(14, y, text=text, fill=TITLE_COL, font=F_TITLE, anchor="w")
            c.create_line(14, y + 12, left_w - 10, y + 12, fill="#3a3a5a")
            return y + 24

        # ========== LEFT COLUMN ==========
        y = 20
        y = info_heading(y, "SIGNAL CHAIN")

        c.create_text(14, y, text="▶ Input", fill="#6a8a6a", font=F_LABEL, anchor="w")
        y += 22
        c.create_text(left_w // 2, y, text="↓", fill=WIRE, font=F_TITLE)
        y += 18

        pre_ms = p["pre_delay"] / SR * 1000
        y = info_row(y, "Pre-delay", f"{pre_ms:.0f} ms",
                     "silence before reverb" if pre_ms > 5 else "")
        c.create_text(left_w // 2, y, text="↓", fill=WIRE, font=F_TITLE)
        y += 18

        diff = p.get("diffusion", 0)
        n_st = p.get("diffusion_stages", 0)
        if diff > 0.01 and n_st > 0:
            desc = "heavy smear" if diff > 0.5 else ("softens attack" if diff > 0.2 else "")
            y = info_row(y, "Diffusion", f"{n_st}x AP g={diff:.2f}", desc)
        else:
            y = info_row(y, "Diffusion", "OFF")
        c.create_text(left_w // 2, y, text="↓", fill=WIRE, font=F_TITLE)
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
        y = info_row(y, "Delays", f"{min_d:.0f}–{max_d:.0f} ms", size_word)

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
            "householder": "uniform coupling", "hadamard": "structured +/−",
            "diagonal": "isolated nodes", "random_orthogonal": "random unitary",
            "circulant": "ring topology", "stautner_puckette": "paired nodes",
            "zero": "disconnected", "custom": "custom wiring",
        }
        y = info_row(y, "Matrix", matrix_type, mat_labels.get(matrix_type, ""))

        c.create_text(left_w // 2, y, text="↓", fill=WIRE, font=F_TITLE)
        y += 18
        mix = p["wet_dry"]
        if mix < 0.01: mix_word = "dry only"
        elif mix < 0.3: mix_word = "subtle reverb"
        elif mix < 0.7: mix_word = "balanced"
        elif mix < 0.99: mix_word = "mostly wet"
        else: mix_word = "100% reverb"
        y = info_row(y, "Wet / Dry", f"{mix:.2f}", mix_word)
        c.create_text(left_w // 2, y, text="↓", fill=WIRE, font=F_TITLE)
        y += 18
        c.create_text(14, y, text="■ Output", fill="#6a8a6a", font=F_LABEL, anchor="w")

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

        c.create_text(ring_cx, 16, text="FEEDBACK DELAY NETWORK — Matrix Wiring",
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
    # Slider helpers
    # ------------------------------------------------------------------

    def _section(self, parent, row, title):
        sep = ttk.Separator(parent, orient="horizontal")
        sep.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(10, 2))
        ttk.Label(parent, text=title, font=("TkDefaultFont", 0, "bold")).grid(
            row=row+1, column=0, columnspan=3, sticky="w", pady=(0, 4))
        return row + 2

    def _on_global_scroll(self, event):
        """Route mousewheel/touchpad scroll to the Scale widget under the cursor."""
        w = self.root.winfo_containing(event.x_root, event.y_root)
        if w is None:
            return
        path = str(w)
        if path not in self._scroll_widgets:
            return
        var, from_, to_ = self._scroll_widgets[path]
        raw = event.delta if event.delta else (-1 if event.num == 5 else 1)
        # Tk 9.0 TouchpadScroll packs X,Y into one int: Y in low 16 bits (signed)
        y = raw & 0xFFFF
        if y >= 0x8000:
            y -= 0x10000
        step = (to_ - from_) * 0.002
        var.set(max(from_, min(to_, var.get() + y * step)))

    def _bind_scroll(self, widget, var, from_, to_):
        self._scroll_widgets[str(widget)] = (var, from_, to_)

    def _add_slider(self, parent, row, key, label, from_, to_, value, length=350):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=1)
        var = tk.DoubleVar(value=value)
        scale = ttk.Scale(parent, from_=from_, to=to_, variable=var,
                          orient="horizontal", length=length)
        scale.grid(row=row, column=1, sticky="ew", padx=5, pady=1)
        self._bind_scroll(scale, var, from_, to_)
        val_label = ttk.Label(parent, text=f"{value:.2f}", width=7)
        val_label.grid(row=row, column=2, sticky="w", pady=1)
        var.trace_add("write", lambda *a, v=var, l=val_label: l.config(text=f"{v.get():.2f}"))
        self.sliders[key] = var
        return row + 1

    def _add_node_slider(self, parent, row, key, label, from_, to_, value, var_list, fmt=".2f", length=350):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=0)
        var = tk.DoubleVar(value=value)
        scale = ttk.Scale(parent, from_=from_, to=to_, variable=var,
                          orient="horizontal", length=length)
        scale.grid(row=row, column=1, sticky="ew", padx=5, pady=0)
        self._bind_scroll(scale, var, from_, to_)
        val_label = ttk.Label(parent, text=f"{value:{fmt}}", width=7)
        val_label.grid(row=row, column=2, sticky="w", pady=0)
        var.trace_add("write", lambda *a, v=var, l=val_label, f=fmt: l.config(text=f"{v.get():{f}}"))
        var_list.append(var)
        return row + 1

    # ------------------------------------------------------------------
    # Param sync
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
        # Update heatmap
        if "matrix_custom" in p and p.get("matrix_type") == "custom":
            self.custom_matrix = np.array(p["matrix_custom"])
        else:
            self._load_matrix_from_topology()
        self._draw_heatmap()

    # ------------------------------------------------------------------
    # Matrix Heatmap
    # ------------------------------------------------------------------

    def _load_matrix_from_topology(self):
        """Load the current named topology into the custom matrix."""
        name = self.matrix_var.get()
        if name in MATRIX_TYPES:
            self.custom_matrix = get_matrix(name, 8,
                                            seed=self.params.get("matrix_seed", 42))
        elif self.custom_matrix is None:
            self.custom_matrix = get_matrix("householder", 8)

    def _draw_heatmap(self):
        """Draw the 8x8 matrix as a color heatmap."""
        if self.custom_matrix is None:
            return
        c = self.heatmap_canvas
        c.delete("all")
        cell = self.heatmap_cell
        pad = 30  # space for labels

        # Find max absolute value for color scaling
        vmax = max(np.max(np.abs(self.custom_matrix)), 0.01)

        for row in range(8):
            for col in range(8):
                val = self.custom_matrix[row, col]
                # Color: blue for negative, red for positive, black for zero
                intensity = min(abs(val) / vmax, 1.0)
                if val >= 0:
                    r = int(40 + 215 * intensity)
                    g = int(40 + 80 * intensity)
                    b = 40
                else:
                    r = 40
                    g = int(40 + 80 * intensity)
                    b = int(40 + 215 * intensity)
                color = f"#{r:02x}{g:02x}{b:02x}"

                x0 = pad + col * cell
                y0 = pad + row * cell
                x1 = x0 + cell
                y1 = y0 + cell

                c.create_rectangle(x0, y0, x1, y1, fill=color, outline="gray40")

                # Show value text for larger cells
                text_color = "white" if intensity > 0.3 else "gray70"
                fmt = f"{val:.1f}" if cell < 30 else f"{val:.2f}"
                c.create_text(x0 + cell//2, y0 + cell//2,
                              text=fmt, fill=text_color, font=("TkDefaultFont", 7))

        # Row/column labels
        for i in range(8):
            c.create_text(pad + i * cell + cell//2, pad - 12,
                          text=str(i), fill="gray70", font=("TkDefaultFont", 9))
            c.create_text(pad - 12, pad + i * cell + cell//2,
                          text=str(i), fill="gray70", font=("TkDefaultFont", 9))

        # Axis labels
        c.create_text(pad + 4 * cell, pad - 25, text="from delay line",
                      fill="gray50", font=("TkDefaultFont", 8))
        c.create_text(pad - 25, pad + 4 * cell, text="to",
                      fill="gray50", font=("TkDefaultFont", 8), angle=90)

        # Selected cell highlight
        if self.selected_cell:
            sr, sc = self.selected_cell
            x0 = pad + sc * cell
            y0 = pad + sr * cell
            c.create_rectangle(x0, y0, x0 + cell, y0 + cell,
                               outline="cyan", width=2)

        # Unitary indicator
        unitary = is_unitary(self.custom_matrix)
        if unitary:
            self.unitary_label.config(text="Unitary: Yes", foreground="green")
        else:
            self.unitary_label.config(text="Unitary: No", foreground="orange")

    def _heatmap_cell_at(self, x, y):
        """Return (row, col) for canvas coordinates, or None."""
        pad = 30
        cell = self.heatmap_cell
        col = (x - pad) // cell
        row = (y - pad) // cell
        if 0 <= row < 8 and 0 <= col < 8:
            return int(row), int(col)
        return None

    def _on_heatmap_click(self, event):
        pos = self._heatmap_cell_at(event.x, event.y)
        if pos is None:
            return
        row, col = pos
        self.selected_cell = (row, col)
        val = self.custom_matrix[row, col]
        self.cell_label.config(text=f"({row}, {col})")
        self.cell_value_var.set(f"{val:.3f}")
        self._draw_heatmap()

    def _on_heatmap_drag(self, event):
        """Drag up/down on a cell to adjust its value."""
        pos = self._heatmap_cell_at(event.x, event.y)
        if pos is None or self.custom_matrix is None:
            return
        if self.selected_cell is None:
            return
        row, col = self.selected_cell
        # Use y position relative to cell center to set value
        pad = 30
        cell = self.heatmap_cell
        cell_center_y = pad + row * cell + cell // 2
        delta = (cell_center_y - event.y) / (cell * 2)  # ±0.5 range per cell height
        new_val = max(-1.5, min(1.5, delta * 2))
        self.custom_matrix[row, col] = new_val
        self.cell_value_var.set(f"{new_val:.3f}")
        self._draw_heatmap()

    def _on_heatmap_right_click(self, event):
        """Right-click to zero a cell."""
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
        """Set cell value from the entry field."""
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
        """Switch to custom matrix mode — uses the heatmap matrix for rendering."""
        self.matrix_var.set("custom")
        self.status_var.set("Using custom matrix — edit the heatmap, then Play")

    def _on_randomize_knobs(self):
        """Randomize all parameters across the full range, including extreme values."""
        rng = np.random.default_rng()
        self.sliders["feedback_gain"].set(rng.uniform(0.0, 2.0))
        self.sliders["wet_dry"].set(rng.uniform(0.0, 1.0))
        self.sliders["diffusion"].set(rng.uniform(0.0, 0.7))
        self.sliders["saturation"].set(rng.uniform(0.0, 1.0))
        self.sliders["pre_delay_ms"].set(rng.uniform(0.0, 250.0))
        self.sliders["stereo_width"].set(rng.uniform(0.0, 1.0))
        for i in range(8):
            self.delay_sliders[i].set(rng.uniform(0.5, 300.0))
            self.damping_sliders[i].set(rng.uniform(0.0, 0.99))
            self.input_gain_sliders[i].set(rng.uniform(0.0, 0.5))
            self.output_gain_sliders[i].set(rng.uniform(0.0, 2.0))
            self.node_pan_sliders[i].set(rng.uniform(-1.0, 1.0))
        # Also randomize the matrix (50/50 unitary vs non-unitary)
        mat = rng.standard_normal((8, 8))
        if rng.random() < 0.5:
            mat = nearest_unitary(mat)
        self.custom_matrix = mat
        self.matrix_var.set("custom")
        self._draw_heatmap()
        self.status_var.set("Randomized all")

    def _on_randomize_and_play(self):
        """Randomize everything then immediately render and play."""
        self._on_randomize_knobs()
        self._on_play()

    def _on_randomize_matrix(self):
        """Generate a random 8x8 matrix, optionally projected to unitary."""
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
    # XY Pad
    # ------------------------------------------------------------------

    _xy_ranges = {"feedback_gain": (0.0, 2.0), "wet_dry": (0.0, 1.0),
                   "diffusion": (0.0, 0.7), "pre_delay_ms": (0.0, 250.0),
                   "saturation": (0.0, 1.0), "stereo_width": (0.0, 1.0)}

    def _xy_draw_crosshair(self, x, y):
        self.xy_canvas.delete("crosshair")
        s = self.xy_size
        self.xy_canvas.create_line(x, 0, x, s, fill="gray30", tags="crosshair")
        self.xy_canvas.create_line(0, y, s, y, fill="gray30", tags="crosshair")
        self.xy_canvas.create_oval(x-6, y-6, x+6, y+6, outline="cyan", width=2, tags="crosshair")

    def _xy_sync(self):
        """Update crosshair position from current slider values."""
        s = self.xy_size
        x_key = self.xy_x_var.get()
        y_key = self.xy_y_var.get()
        # Compute normalized positions
        def _norm(key):
            if key in self.sliders and key in self._xy_ranges:
                lo, hi = self._xy_ranges[key]
                return (self.sliders[key].get() - lo) / (hi - lo)
            return 0.5
        px = int(max(0, min(s, _norm(x_key) * s)))
        py = int(max(0, min(s, (1.0 - _norm(y_key)) * s)))
        self._xy_draw_crosshair(px, py)

    def _on_xy_drag(self, event):
        s = self.xy_size
        nx = max(0.0, min(1.0, event.x / s))
        ny = max(0.0, min(1.0, 1.0 - event.y / s))
        x_key = self.xy_x_var.get()
        y_key = self.xy_y_var.get()
        for key, val in [(x_key, nx), (y_key, ny)]:
            if key in self.sliders and key in self._xy_ranges:
                lo, hi = self._xy_ranges[key]
                self.sliders[key].set(lo + val * (hi - lo))
        self._xy_draw_crosshair(event.x, event.y)

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def _load_wav(self, path):
        sr, data = wavfile.read(path)
        if data.dtype == np.int16:
            audio = data.astype(np.float64) / 32768.0
        elif data.dtype == np.int32:
            audio = data.astype(np.float64) / 2147483648.0
        else:
            audio = data.astype(np.float64)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        self.source_audio = audio
        self.source_path = path
        self.rendered_audio = None
        self.rendered_warning = ""
        self.rendered_params = None

    def _source_text(self):
        if self.source_path:
            name = os.path.basename(self.source_path)
            dur = len(self.source_audio) / SR if self.source_audio is not None else 0
            return f"{name} ({dur:.1f}s)"
        return "(no file loaded)"

    def _on_load_wav(self):
        path = filedialog.askopenfilename(
            title="Select source WAV", initialdir=TEST_SIGNALS_DIR,
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")])
        if path:
            self._load_wav(path)
            self.source_label.config(text=self._source_text())
            self.status_var.set("Loaded: " + os.path.basename(path))

    # ------------------------------------------------------------------
    # Render & Playback
    # ------------------------------------------------------------------

    def _params_snapshot(self, params):
        return json.dumps(params, sort_keys=True)

    def _render(self, params, callback):
        self.rendering = True
        self.status_var.set("Rendering...")
        self.root.update()

        def do_render():
            tail = np.zeros(int(self.sliders["tail_length"].get() * SR))
            audio_in = np.concatenate([self.source_audio, tail])
            output = render_fdn(audio_in, params)

            # --- Safety: reject unstable output ---
            if not np.all(np.isfinite(output)):
                self.rendering = False
                self.root.after(0, lambda: (
                    sd.stop(),
                    self.status_var.set(
                        "ERROR: output diverged (non-finite values). Adjust params.")))
                return
            peak = np.max(np.abs(output))
            if peak > 1e6:
                self.rendering = False
                self.root.after(0, lambda p=peak: (
                    sd.stop(),
                    self.status_var.set(
                        f"ERROR: output exploded (peak={p:.0e}). Reduce feedback/gain.")))
                return

            # --- Loudness limiter ---
            if peak > 0:
                output = output / peak
            rms = np.sqrt(np.mean(output ** 2))  # works for both mono and stereo
            target_rms = 0.2
            if rms > target_rms:
                gain = target_rms / rms
                output *= gain
                loud_warning = f" (loud — reduced {1/gain:.0f}x)"
            else:
                loud_warning = ""
            output *= 0.9  # headroom

            self.rendered_audio = output
            self.rendered_warning = loud_warning
            self.rendered_params = self._params_snapshot(params)
            self.rendering = False
            self.root.after(0, lambda: callback(output, loud_warning))

        threading.Thread(target=do_render, daemon=True).start()

    def _play_audio(self, audio):
        """Play audio through the current default output device."""
        sd.default.reset()
        sd.play(audio, SR)

    def _on_play(self):
        if self.source_audio is None:
            messagebox.showwarning("No source", "Load a WAV file first.")
            return
        if self.rendering:
            return

        # Always stop previous playback immediately
        sd.stop()

        params = self._read_params_from_ui()
        snap = self._params_snapshot(params)

        if self.rendered_audio is not None and self.rendered_params == snap:
            self._play_audio(np.clip(self.rendered_audio, -1.0, 1.0).astype(np.float32))
            dur = len(self.rendered_audio) / SR
            self.status_var.set(f"Playing ({dur:.1f}s){self.rendered_warning}")
            return

        def on_done(output, loud_warning=""):
            self._play_audio(np.clip(output, -1.0, 1.0).astype(np.float32))
            self.status_var.set(f"Playing ({len(output)/SR:.1f}s){loud_warning}")
            self._draw_waveform()

        self._render(params, on_done)

    def _on_play_dry(self):
        if self.source_audio is None:
            messagebox.showwarning("No source", "Load a WAV file first.")
            return
        sd.stop()
        self._play_audio(self.source_audio.astype(np.float32))
        self.status_var.set("Playing dry...")

    def _on_stop(self):
        sd.stop()
        self.status_var.set("Stopped")

    def _on_save_wav(self):
        if self.rendered_audio is None:
            messagebox.showinfo("Nothing to save", "Press Play first to render.")
            return
        path = filedialog.asksaveasfilename(
            title="Save rendered WAV", initialdir=TEST_SIGNALS_DIR,
            defaultextension=".wav", filetypes=[("WAV files", "*.wav")])
        if path:
            audio = self.rendered_audio.copy()
            peak = np.max(np.abs(audio))
            if peak > 0:
                audio = audio / peak * 0.9
            wavfile.write(path, SR, (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16))
            self.status_var.set(f"Saved: {os.path.basename(path)}")

    # ------------------------------------------------------------------
    # Presets
    # ------------------------------------------------------------------

    def _refresh_preset_list(self):
        self.preset_listbox.delete(0, tk.END)
        if os.path.isdir(PRESET_DIR):
            for f in sorted(os.listdir(PRESET_DIR)):
                if f.endswith(".json"):
                    self.preset_listbox.insert(tk.END, f[:-5])

    def _on_load_preset(self):
        sel = self.preset_listbox.curselection()
        if not sel:
            return
        name = self.preset_listbox.get(sel[0])
        path = os.path.join(PRESET_DIR, name + ".json")
        with open(path) as f:
            p = json.load(f)
        full = default_params()
        full.update(p)
        self._write_params_to_ui(full)
        self.status_var.set(f"Loaded preset: {name}")

    def _on_save_preset(self):
        name = simpledialog.askstring("Save Preset", "Preset name:")
        if not name:
            return
        params = self._read_params_from_ui()
        os.makedirs(PRESET_DIR, exist_ok=True)
        path = os.path.join(PRESET_DIR, name + ".json")
        with open(path, "w") as f:
            json.dump(params, f, indent=2)
        self._refresh_preset_list()
        self.status_var.set(f"Saved preset: {name}")


def main():
    root = tk.Tk()
    root.geometry("1050x850")
    ReverbGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
