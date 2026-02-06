"""FDN Reverb GUI — parameter control with WAV file rendering and playback.

Run: uv run python gui/gui.py
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
from primitives.matrix import MATRIX_TYPES

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
        self.rendering = False
        self.playing = False

        # Load a default source
        default_source = os.path.join(TEST_SIGNALS_DIR, "dry_chords.wav")
        if os.path.exists(default_source):
            self._load_wav(default_source)

        self._build_ui()

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        self.root.configure(padx=10, pady=10)

        # Top bar: file controls
        top = ttk.Frame(self.root)
        top.pack(fill="x", pady=(0, 10))

        ttk.Button(top, text="Load WAV", command=self._on_load_wav).pack(side="left", padx=2)
        self.source_label = ttk.Label(top, text=self._source_text(), width=40, anchor="w")
        self.source_label.pack(side="left", padx=5)

        ttk.Button(top, text="Play Dry", command=self._on_play_dry).pack(side="left", padx=2)
        ttk.Button(top, text="Render", command=self._on_render).pack(side="left", padx=2)
        ttk.Button(top, text="Play Wet", command=self._on_play_wet).pack(side="left", padx=2)
        ttk.Button(top, text="Stop", command=self._on_stop).pack(side="left", padx=2)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(top, textvariable=self.status_var, width=30, anchor="e").pack(side="right")

        # Main area: notebook with tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True)

        # Tab 1: Main parameters
        main_frame = ttk.Frame(notebook, padding=10)
        notebook.add(main_frame, text="Main")
        self._build_main_tab(main_frame)

        # Tab 2: Delay times
        delay_frame = ttk.Frame(notebook, padding=10)
        notebook.add(delay_frame, text="Delays")
        self._build_delay_tab(delay_frame)

        # Tab 3: Gains
        gains_frame = ttk.Frame(notebook, padding=10)
        notebook.add(gains_frame, text="Gains")
        self._build_gains_tab(gains_frame)

        # Tab 4: Presets
        preset_frame = ttk.Frame(notebook, padding=10)
        notebook.add(preset_frame, text="Presets")
        self._build_preset_tab(preset_frame)

        # Tab 5: XY Pad
        xy_frame = ttk.Frame(notebook, padding=10)
        notebook.add(xy_frame, text="XY Pad")
        self._build_xy_tab(xy_frame)

    def _build_main_tab(self, parent):
        self.sliders = {}

        row = 0
        # Feedback gain
        row = self._add_slider(parent, row, "feedback_gain", "Feedback Gain",
                               0.0, 1.1, self.params["feedback_gain"], resolution=0.01)

        # Wet/dry
        row = self._add_slider(parent, row, "wet_dry", "Wet/Dry Mix",
                               0.0, 1.0, self.params["wet_dry"], resolution=0.01)

        # Diffusion
        row = self._add_slider(parent, row, "diffusion", "Diffusion",
                               0.0, 0.7, self.params["diffusion"], resolution=0.01)

        # Pre-delay (in ms, converted to samples)
        pre_delay_ms = self.params["pre_delay"] / SR * 1000
        row = self._add_slider(parent, row, "pre_delay_ms", "Pre-delay (ms)",
                               0.0, 250.0, pre_delay_ms, resolution=1.0)

        # Damping (uniform for all 8 nodes)
        row = self._add_slider(parent, row, "damping_uniform", "Damping (all nodes)",
                               0.0, 0.99, self.params["damping_coeffs"][0], resolution=0.01)

        # Matrix type
        ttk.Label(parent, text="Matrix Topology:").grid(row=row, column=0, sticky="w", pady=5)
        self.matrix_var = tk.StringVar(value=self.params.get("matrix_type", "householder"))
        matrix_combo = ttk.Combobox(parent, textvariable=self.matrix_var,
                                     values=list(MATRIX_TYPES.keys()), state="readonly", width=20)
        matrix_combo.grid(row=row, column=1, sticky="w", pady=5)

    def _build_delay_tab(self, parent):
        ttk.Label(parent, text="Per-node delay times (ms):").grid(row=0, column=0,
                                                                    columnspan=3, sticky="w")
        self.delay_sliders = []
        for i in range(8):
            delay_ms = self.params["delay_times"][i] / SR * 1000
            ttk.Label(parent, text=f"Node {i}:").grid(row=i+1, column=0, sticky="w")
            var = tk.DoubleVar(value=delay_ms)
            scale = ttk.Scale(parent, from_=0.5, to=300.0, variable=var,
                              orient="horizontal", length=400)
            scale.grid(row=i+1, column=1, sticky="ew", padx=5)
            val_label = ttk.Label(parent, text=f"{delay_ms:.1f}")
            val_label.grid(row=i+1, column=2, sticky="w")
            var.trace_add("write", lambda *a, v=var, l=val_label: l.config(text=f"{v.get():.1f}"))
            self.delay_sliders.append(var)
        parent.columnconfigure(1, weight=1)

        # Per-node damping
        ttk.Label(parent, text="\nPer-node damping:").grid(row=10, column=0,
                                                            columnspan=3, sticky="w")
        self.damping_sliders = []
        for i in range(8):
            coeff = self.params["damping_coeffs"][i]
            ttk.Label(parent, text=f"Node {i}:").grid(row=i+11, column=0, sticky="w")
            var = tk.DoubleVar(value=coeff)
            scale = ttk.Scale(parent, from_=0.0, to=0.99, variable=var,
                              orient="horizontal", length=400)
            scale.grid(row=i+11, column=1, sticky="ew", padx=5)
            val_label = ttk.Label(parent, text=f"{coeff:.2f}")
            val_label.grid(row=i+11, column=2, sticky="w")
            var.trace_add("write", lambda *a, v=var, l=val_label: l.config(text=f"{v.get():.2f}"))
            self.damping_sliders.append(var)

    def _build_gains_tab(self, parent):
        ttk.Label(parent, text="Input gains:").grid(row=0, column=0, columnspan=3, sticky="w")
        self.input_gain_sliders = []
        for i in range(8):
            g = self.params["input_gains"][i]
            ttk.Label(parent, text=f"Node {i}:").grid(row=i+1, column=0, sticky="w")
            var = tk.DoubleVar(value=g)
            scale = ttk.Scale(parent, from_=0.0, to=0.5, variable=var,
                              orient="horizontal", length=400)
            scale.grid(row=i+1, column=1, sticky="ew", padx=5)
            val_label = ttk.Label(parent, text=f"{g:.3f}")
            val_label.grid(row=i+1, column=2, sticky="w")
            var.trace_add("write", lambda *a, v=var, l=val_label: l.config(text=f"{v.get():.3f}"))
            self.input_gain_sliders.append(var)
        parent.columnconfigure(1, weight=1)

        ttk.Label(parent, text="\nOutput gains:").grid(row=10, column=0, columnspan=3, sticky="w")
        self.output_gain_sliders = []
        for i in range(8):
            g = self.params["output_gains"][i]
            ttk.Label(parent, text=f"Node {i}:").grid(row=i+11, column=0, sticky="w")
            var = tk.DoubleVar(value=g)
            scale = ttk.Scale(parent, from_=0.0, to=2.0, variable=var,
                              orient="horizontal", length=400)
            scale.grid(row=i+11, column=1, sticky="ew", padx=5)
            val_label = ttk.Label(parent, text=f"{g:.2f}")
            val_label.grid(row=i+11, column=2, sticky="w")
            var.trace_add("write", lambda *a, v=var, l=val_label: l.config(text=f"{v.get():.2f}"))
            self.output_gain_sliders.append(var)

    def _build_preset_tab(self, parent):
        # Preset list
        ttk.Label(parent, text="Presets:").pack(anchor="w")
        list_frame = ttk.Frame(parent)
        list_frame.pack(fill="both", expand=True, pady=5)

        self.preset_listbox = tk.Listbox(list_frame, height=12)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.preset_listbox.yview)
        self.preset_listbox.configure(yscrollcommand=scrollbar.set)
        self.preset_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self._refresh_preset_list()

        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill="x", pady=5)
        ttk.Button(btn_frame, text="Load Selected", command=self._on_load_preset).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="Save As...", command=self._on_save_preset).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="Refresh", command=self._refresh_preset_list).pack(side="left", padx=2)

    def _build_xy_tab(self, parent):
        controls = ttk.Frame(parent)
        controls.pack(fill="x", pady=(0, 5))

        ttk.Label(controls, text="X axis:").pack(side="left")
        self.xy_x_var = tk.StringVar(value="feedback_gain")
        xy_params = ["feedback_gain", "wet_dry", "diffusion", "pre_delay_ms", "damping_uniform"]
        ttk.Combobox(controls, textvariable=self.xy_x_var, values=xy_params,
                     state="readonly", width=15).pack(side="left", padx=5)
        ttk.Label(controls, text="Y axis:").pack(side="left", padx=(10, 0))
        self.xy_y_var = tk.StringVar(value="damping_uniform")
        ttk.Combobox(controls, textvariable=self.xy_y_var, values=xy_params,
                     state="readonly", width=15).pack(side="left", padx=5)

        self.xy_canvas = tk.Canvas(parent, width=400, height=400, bg="black",
                                    highlightthickness=1, highlightbackground="gray")
        self.xy_canvas.pack(expand=True)
        self.xy_canvas.bind("<B1-Motion>", self._on_xy_drag)
        self.xy_canvas.bind("<Button-1>", self._on_xy_drag)

        # Draw crosshair
        self._xy_draw_crosshair(200, 200)

    # ------------------------------------------------------------------
    # Slider helper
    # ------------------------------------------------------------------

    def _add_slider(self, parent, row, key, label, from_, to_, value, resolution=0.01):
        ttk.Label(parent, text=label + ":").grid(row=row, column=0, sticky="w", pady=3)
        var = tk.DoubleVar(value=value)
        scale = ttk.Scale(parent, from_=from_, to=to_, variable=var,
                          orient="horizontal", length=400)
        scale.grid(row=row, column=1, sticky="ew", padx=5, pady=3)
        val_label = ttk.Label(parent, text=f"{value:.2f}", width=8)
        val_label.grid(row=row, column=2, sticky="w", pady=3)
        var.trace_add("write", lambda *a, v=var, l=val_label: l.config(text=f"{v.get():.2f}"))
        parent.columnconfigure(1, weight=1)
        self.sliders[key] = var
        return row + 1

    # ------------------------------------------------------------------
    # Param sync (UI -> dict)
    # ------------------------------------------------------------------

    def _read_params_from_ui(self):
        p = dict(self.params)  # start from current (preserves fields not in UI)

        # Main tab
        p["feedback_gain"] = self.sliders["feedback_gain"].get()
        p["wet_dry"] = self.sliders["wet_dry"].get()
        p["diffusion"] = self.sliders["diffusion"].get()
        p["pre_delay"] = int(self.sliders["pre_delay_ms"].get() / 1000 * SR)
        p["matrix_type"] = self.matrix_var.get()

        # Uniform damping from main tab
        uniform_damp = self.sliders["damping_uniform"].get()

        # Per-node delay times
        if hasattr(self, 'delay_sliders'):
            p["delay_times"] = [int(s.get() / 1000 * SR) for s in self.delay_sliders]
            p["damping_coeffs"] = [s.get() for s in self.damping_sliders]
        else:
            p["damping_coeffs"] = [uniform_damp] * 8

        # Per-node gains
        if hasattr(self, 'input_gain_sliders'):
            p["input_gains"] = [s.get() for s in self.input_gain_sliders]
            p["output_gains"] = [s.get() for s in self.output_gain_sliders]

        return p

    def _write_params_to_ui(self, p):
        self.params = p

        self.sliders["feedback_gain"].set(p["feedback_gain"])
        self.sliders["wet_dry"].set(p["wet_dry"])
        self.sliders["diffusion"].set(p["diffusion"])
        self.sliders["pre_delay_ms"].set(p["pre_delay"] / SR * 1000)
        self.sliders["damping_uniform"].set(p["damping_coeffs"][0])
        self.matrix_var.set(p.get("matrix_type", "householder"))

        if hasattr(self, 'delay_sliders'):
            for i in range(8):
                self.delay_sliders[i].set(p["delay_times"][i] / SR * 1000)
                self.damping_sliders[i].set(p["damping_coeffs"][i])

        if hasattr(self, 'input_gain_sliders'):
            for i in range(8):
                self.input_gain_sliders[i].set(p["input_gains"][i])
                self.output_gain_sliders[i].set(p["output_gains"][i])

    # ------------------------------------------------------------------
    # XY Pad
    # ------------------------------------------------------------------

    def _xy_draw_crosshair(self, x, y):
        self.xy_canvas.delete("crosshair")
        w = self.xy_canvas.winfo_width() or 400
        h = self.xy_canvas.winfo_height() or 400
        self.xy_canvas.create_line(x, 0, x, h, fill="gray30", tags="crosshair")
        self.xy_canvas.create_line(0, y, w, y, fill="gray30", tags="crosshair")
        self.xy_canvas.create_oval(x-6, y-6, x+6, y+6, outline="cyan", width=2, tags="crosshair")

    def _on_xy_drag(self, event):
        w = self.xy_canvas.winfo_width()
        h = self.xy_canvas.winfo_height()
        nx = max(0.0, min(1.0, event.x / w))
        ny = max(0.0, min(1.0, 1.0 - event.y / h))  # y inverted: bottom=0, top=1

        x_param = self.xy_x_var.get()
        y_param = self.xy_y_var.get()

        # Map 0-1 to slider range
        for param, val in [(x_param, nx), (y_param, ny)]:
            if param in self.sliders:
                scale = self.sliders[param]
                low, high = self._slider_range(param)
                scale.set(low + val * (high - low))

        self._xy_draw_crosshair(event.x, event.y)

    def _slider_range(self, key):
        ranges = {
            "feedback_gain": (0.0, 1.1),
            "wet_dry": (0.0, 1.0),
            "diffusion": (0.0, 0.7),
            "pre_delay_ms": (0.0, 250.0),
            "damping_uniform": (0.0, 0.99),
        }
        return ranges.get(key, (0.0, 1.0))

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

    def _source_text(self):
        if self.source_path:
            name = os.path.basename(self.source_path)
            dur = len(self.source_audio) / SR if self.source_audio is not None else 0
            return f"{name} ({dur:.1f}s)"
        return "(no file loaded)"

    def _on_load_wav(self):
        path = filedialog.askopenfilename(
            title="Select source WAV",
            initialdir=TEST_SIGNALS_DIR,
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")])
        if path:
            self._load_wav(path)
            self.source_label.config(text=self._source_text())
            self.status_var.set("Loaded: " + os.path.basename(path))

    # ------------------------------------------------------------------
    # Render & Playback
    # ------------------------------------------------------------------

    def _on_render(self):
        if self.source_audio is None:
            messagebox.showwarning("No source", "Load a WAV file first.")
            return
        if self.rendering:
            return

        params = self._read_params_from_ui()
        self.rendering = True
        self.status_var.set("Rendering...")
        self.root.update()

        def do_render():
            # Add 2s tail
            tail = np.zeros(int(2.0 * SR))
            audio_in = np.concatenate([self.source_audio, tail])
            output = render_fdn(audio_in, params)
            # Normalize
            peak = np.max(np.abs(output))
            if peak > 1.0:
                output = output / peak * 0.95
            self.rendered_audio = output
            self.rendering = False
            self.root.after(0, lambda: self.status_var.set(
                f"Rendered ({len(output)/SR:.1f}s). Press Play Wet."))

        threading.Thread(target=do_render, daemon=True).start()

    def _on_play_dry(self):
        if self.source_audio is None:
            messagebox.showwarning("No source", "Load a WAV file first.")
            return
        sd.stop()
        sd.play(self.source_audio.astype(np.float32), SR)
        self.status_var.set("Playing dry...")

    def _on_play_wet(self):
        if self.rendered_audio is None:
            messagebox.showinfo("Not rendered", "Click Render first.")
            return
        sd.stop()
        sd.play(self.rendered_audio.astype(np.float32), SR)
        self.status_var.set("Playing wet...")

    def _on_stop(self):
        sd.stop()
        self.status_var.set("Stopped")

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
        # Merge with defaults for any missing keys
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
    root.geometry("700x550")
    ReverbGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
