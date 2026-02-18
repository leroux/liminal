"""Fractal audio fractalization GUI — tkinter.

Tabs: Parameters · Presets · Waveforms · Spectrograms · Spectrum · Guide
"""

import os
import tkinter as tk
from tkinter import ttk

import numpy as np

from fractal.engine.params import (
    SR,
    SCHEMA,
    default_params,
    bypass_params,
    PARAM_RANGES,
    PARAM_SECTIONS,
    CHOICE_RANGES,
    INTERP_NAMES,
    FILTER_NAMES,
    POST_FILTER_NAMES,
    BOUNCE_TARGET_NAMES,
)
from fractal.engine.fractal import render_fractal
from shared.llm_guide_text import FRACTAL_GUIDE, FRACTAL_PARAM_DESCRIPTIONS
from shared.gui import PedalGUIBase, PedalConfig

PRESET_DIR = os.path.join(os.path.dirname(__file__), "presets")
DEFAULT_SOURCE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    os.pardir, "audio", "test_signals", "dry_noise_burst.wav",
)

PRESET_CATEGORIES = [
    "Subtle Enhancement", "Rhythmic & Percussive",
    "Ambient & Pad", "Aggressive & Industrial",
    "Glitch & Experimental", "Sound Design",
    "Filter & Crush Combos", "Modulated",
]


def _make_config():
    return PedalConfig(
        name="Fractal",
        preset_dir=PRESET_DIR,
        preset_categories=PRESET_CATEGORIES,
        window_title="Fractal \u2014 audio fractalization",
        window_geometry="900x680",
        default_params=default_params,
        bypass_params=bypass_params,
        param_ranges=PARAM_RANGES,
        param_sections=PARAM_SECTIONS,
        choice_ranges=CHOICE_RANGES,
        render=render_fractal,
        render_stereo=None,
        guide_text=FRACTAL_GUIDE,
        param_descriptions=FRACTAL_PARAM_DESCRIPTIONS,
        sample_rate=SR,
        default_source=DEFAULT_SOURCE,
        icon_path=os.path.join(os.path.dirname(__file__), os.pardir, os.pardir,
                               "icons", "fractal.png"),
        extra_tabs=[],   # set in __init__
        randomize_skip={"wet_dry", "output_gain", "threshold", "tail_length"},
        randomize_clamp={
            "gate": (0.0, 0.3),
            "crush": (0.0, 0.7),
            "decimate": (0.0, 0.5),
            "filter_freq": (80.0, 8000.0),
            "post_filter_freq": (200.0, 16000.0),
        },
        tail_param="tail_length",
        schema=SCHEMA,
    )


class FractalGUI(PedalGUIBase):

    def __init__(self, root):
        cfg = _make_config()
        cfg.extra_tabs = [("Spectrum", self._build_spec_page)]
        super().__init__(root, cfg)

    # ------------------------------------------------------------------
    # Parameters tab
    # ------------------------------------------------------------------

    def _build_params_page(self, parent):
        f = self._build_params_container(parent)
        r = 0
        SL = 400

        # ---- Core Fractal ----
        r = self._add_section_header(f, r, "fractal", "FRACTAL")

        r = self._add_slider(f, r, "num_scales", "Scales", 2, 8, self.params["num_scales"], length=SL, integer=True)
        r = self._add_slider(f, r, "scale_ratio", "Ratio", 0.1, 0.9, self.params["scale_ratio"], length=SL)
        r = self._add_slider(f, r, "amplitude_decay", "Amp Decay", 0.1, 1.0, self.params["amplitude_decay"], length=SL)
        r = self._add_slider(f, r, "scale_offset", "Offset", 0.0, 1.0, self.params["scale_offset"], length=SL)

        self._interp_var, r = self._add_choice(f, r, "interp", "Interp", INTERP_NAMES, self.params["interp"])

        self._reverse_var = tk.IntVar(value=self.params["reverse_scales"])
        ttk.Checkbutton(f, text="Reverse Scales", variable=self._reverse_var).grid(
            row=r, column=0, columnspan=2, sticky="w", padx=8)
        self._add_lock(f, r, "reverse_scales")
        self.choice_vars["reverse_scales"] = self._reverse_var
        r += 1

        # ---- Layers ----
        r = self._add_separator(f, r)
        r = self._add_section_header(f, r, "layers", "LAYERS")

        for lg in range(1, 8):
            key = f"layer_gain_{lg}"
            r = self._add_slider(f, r, key, f"Layer {lg} Gain", 0.0, 2.0, self.params[key], length=SL)

        self._only_wet_var = tk.IntVar(value=self.params["fractal_only_wet"])
        ttk.Checkbutton(f, text="Fractal Only Wet", variable=self._only_wet_var).grid(
            row=r, column=0, columnspan=2, sticky="w", padx=8)
        self._add_lock(f, r, "fractal_only_wet")
        self.choice_vars["fractal_only_wet"] = self._only_wet_var
        r += 1

        r = self._add_slider(f, r, "layer_spread", "Spread", 0.0, 1.0, self.params["layer_spread"], length=SL)
        r = self._add_slider(f, r, "layer_detune", "Detune", 0.0, 1.0, self.params["layer_detune"], length=SL)
        r = self._add_slider(f, r, "layer_delay", "Layer Delay", 0.0, 1.0, self.params["layer_delay"], length=SL)
        r = self._add_slider(f, r, "layer_tilt", "Tilt", -1.0, 1.0, self.params["layer_tilt"], length=SL)

        # ---- Iteration / Feedback ----
        r = self._add_separator(f, r)
        r = self._add_section_header(f, r, "iteration", "ITERATION")

        r = self._add_slider(f, r, "iterations", "Iterations", 1, 4, self.params["iterations"], length=SL, integer=True)
        r = self._add_slider(f, r, "iter_decay", "Iter Decay", 0.3, 1.0, self.params["iter_decay"], length=SL)
        r = self._add_slider(f, r, "saturation", "Saturation", 0.0, 1.0, self.params["saturation"], length=SL)
        r = self._add_slider(f, r, "feedback", "Feedback", 0.0, 0.95, self.params["feedback"], length=SL)

        # ---- Spectral Fractal ----
        r = self._add_separator(f, r)
        r = self._add_section_header(f, r, "spectral", "SPECTRAL")

        r = self._add_slider(f, r, "spectral", "Spectral", 0.0, 1.0, self.params["spectral"], length=SL)
        r = self._add_slider(f, r, "window_size", "Window Size", 256, 8192, self.params["window_size"], length=SL, integer=True)

        # ---- Filter ----
        r = self._add_separator(f, r)
        r = self._add_section_header(f, r, "filter", "FILTER")

        self._filter_var, r = self._add_choice(f, r, "filter_type", "Pre-Filter", FILTER_NAMES, self.params["filter_type"])
        r = self._add_slider(f, r, "filter_freq", "Freq (Hz)", 20.0, 20000.0, self.params["filter_freq"], length=SL, log=True)
        r = self._add_slider(f, r, "filter_q", "Q", 0.1, 10.0, self.params["filter_q"], length=SL)
        self._post_filter_var, r = self._add_choice(f, r, "post_filter_type", "Post-Filter", POST_FILTER_NAMES, self.params["post_filter_type"])
        r = self._add_slider(f, r, "post_filter_freq", "Post Freq (Hz)", 20.0, 20000.0, self.params["post_filter_freq"], length=SL, log=True)

        # ---- Effects ----
        r = self._add_separator(f, r)
        r = self._add_section_header(f, r, "effects", "EFFECTS")

        r = self._add_slider(f, r, "gate", "Gate", 0.0, 1.0, self.params["gate"], length=SL)
        r = self._add_slider(f, r, "crush", "Crush", 0.0, 1.0, self.params["crush"], length=SL)
        r = self._add_slider(f, r, "decimate", "Decimate", 0.0, 1.0, self.params["decimate"], length=SL)

        # ---- Bounce ----
        r = self._add_separator(f, r)
        r = self._add_section_header(f, r, "bounce", "BOUNCE")

        self._bounce_var = tk.IntVar(value=self.params["bounce"])
        ttk.Checkbutton(f, text="Bounce", variable=self._bounce_var).grid(
            row=r, column=0, sticky="w", padx=8)
        self._add_lock(f, r, "bounce")
        self.choice_vars["bounce"] = self._bounce_var
        r += 1

        self._bounce_target_var, r = self._add_choice(f, r, "bounce_target", "Target", BOUNCE_TARGET_NAMES, self.params["bounce_target"])
        r = self._add_slider(f, r, "bounce_rate", "Rate", 0.0, 1.0, self.params["bounce_rate"], length=SL)
        r = self._add_slider(f, r, "bounce_lfo_min", "LFO Min (Hz)", 0.01, 50.0, self.params["bounce_lfo_min"], length=SL)
        r = self._add_slider(f, r, "bounce_lfo_max", "LFO Max (Hz)", 0.01, 50.0, self.params["bounce_lfo_max"], length=SL)

        # ---- Output ----
        r = self._add_separator(f, r)
        r = self._add_section_header(f, r, "output", "OUTPUT")

        r = self._add_slider(f, r, "wet_dry", "Wet / Dry", 0.0, 1.0, self.params["wet_dry"], length=SL)
        r = self._add_slider(f, r, "output_gain", "Output Gain", 0.0, 1.0, self.params["output_gain"], length=SL)
        r = self._add_slider(f, r, "threshold", "Threshold", 0.0, 1.0, self.params["threshold"], length=SL)
        r = self._add_slider(f, r, "tail_length", "Tail Length (s)", 0.0, 60.0, 2.0, length=SL)

        # Auto-play on manually-created discrete vars
        for var in [self._reverse_var, self._bounce_var, self._only_wet_var]:
            var.trace_add("write", lambda *_: self._schedule_autoplay())

    # ------------------------------------------------------------------
    # Param read / write
    # ------------------------------------------------------------------

    def _read_params_from_ui(self):
        p = self.cfg.default_params()
        p["interp"] = self._interp_var.get()
        p["reverse_scales"] = self._reverse_var.get()
        p["filter_type"] = self._filter_var.get()
        p["post_filter_type"] = self._post_filter_var.get()
        p["bounce"] = self._bounce_var.get()
        p["bounce_target"] = self._bounce_target_var.get()
        p["fractal_only_wet"] = self._only_wet_var.get()
        for key, var in self.sliders.items():
            p[key] = var.get()
        return p

    def _write_params_to_ui(self, p):
        self._interp_var.set(p.get("interp", 0))
        self._reverse_var.set(p.get("reverse_scales", 0))
        self._filter_var.set(p.get("filter_type", 0))
        self._post_filter_var.set(p.get("post_filter_type", 0))
        self._bounce_var.set(p.get("bounce", 0))
        self._bounce_target_var.set(p.get("bounce_target", 0))
        self._only_wet_var.set(p.get("fractal_only_wet", 0))
        for key, var in self.sliders.items():
            if key in p:
                var.set(p[key])

    # ------------------------------------------------------------------
    # Spectrum tab (extra tab)
    # ------------------------------------------------------------------

    def _build_spec_page(self, parent):
        self.spec_canvas = tk.Canvas(parent, bg="#1a1a2e", highlightthickness=0)
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
            mid = len(mono) // 2
            start = max(0, mid - fft_size // 2)
            chunk = mono[start : start + fft_size]
            windowed = chunk * np.hanning(len(chunk))
            spec = np.abs(np.fft.rfft(windowed))
            spec[spec < 1e-12] = 1e-12
            db = 20.0 * np.log10(spec)
            freqs = np.fft.rfftfreq(len(chunk), 1.0 / self.cfg.sample_rate)
            return freqs, db

        freqs_dry, db_dry = calc_spectrum(self.source_audio)
        freqs_wet, db_wet = calc_spectrum(self.rendered_audio)

        db_min, db_max = -90.0, 0.0
        if db_wet is not None:
            db_max = max(db_max, np.max(db_wet) + 6)
        if db_dry is not None:
            db_max = max(db_max, np.max(db_dry) + 6)

        half_sr = self.cfg.sample_rate / 2.0

        def freq_to_x(f):
            if f <= 20:
                return 0
            return int(np.log10(f / 20.0) / np.log10(half_sr / 20.0) * w)

        def db_to_y(d):
            frac = (d - db_min) / (db_max - db_min)
            return int(pad_top + plot_h * (1.0 - np.clip(frac, 0, 1)))

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

        c.create_text(w - 8, pad_top + 4, anchor="ne", fill="#555577", text="Dry")
        c.create_text(w - 8, pad_top + 18, anchor="ne", fill="#ff6644", text="Wet")

    def _on_render_complete(self):
        self._draw_spectrum()

    # ------------------------------------------------------------------
    # Guide tab
    # ------------------------------------------------------------------

    def _build_guide_page(self):
        text = tk.Text(self.guide_frame, wrap="word", bg="#1a1a2e", fg="#ccccdd",
                       font=("Helvetica", 12), padx=16, pady=12, relief="flat",
                       selectbackground="#334466", insertbackground="#ccccdd")
        text.pack(fill="both", expand=True)

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

        h1("Fractal \u2014 Audio Fractalization Effect")
        body("")

        h2("What It Does")
        body("Creates self-similar fractal-like structures at multiple timescales by")
        body("compressing the signal into progressively shorter copies, tiling them")
        body("to fill the original length, and summing with decaying gains.")
        body("")
        body("Each scale layer adds a time-compressed, repeated copy of the signal.")
        body("The result is a texture where the signal's character appears at")
        body("multiple zoom levels simultaneously \u2014 like a fractal.")
        body("")

        h2("Core Fractal Controls")
        mapping("Scales", "num_scales",
                "Number of fractal layers (2-8). More = more complex self-similarity.")
        mapping("Ratio", "scale_ratio",
                "Compression per layer. 0.5 = halve length each level. Low = extreme.")
        mapping("Amp Decay", "amplitude_decay",
                "Volume decay per layer. 0.5 = each layer half as loud.")
        mapping("Offset", "scale_offset",
                "Shifts where tiled chunks start. Creates phase variations.")
        mapping("Interp", "Nearest / Linear",
                "Nearest = aliased/gritty, Linear = smoother resampling.")
        mapping("Reverse", "Reverse Scales",
                "Reversed compressed chunks create backward fractal layers.")
        body("")

        h2("Iteration (Feedback)")
        mapping("Iterations", "iterations",
                "Re-feed output through fractalizer 1-4 times. More = deeper texture.")
        mapping("Iter Decay", "iter_decay",
                "Volume reduction between iterations. Prevents runaway.")
        mapping("Saturation", "saturation",
                "tanh soft-clipping between iterations. Adds warmth/distortion.")
        body("")

        h2("Spectral Mode")
        mapping("Spectral", "spectral",
                "0=time-domain only, 1=spectral-domain only. Blend between.")
        mapping("Window Size", "window_size",
                "STFT window for spectral mode. Larger = smoother, smaller = glitchy.")
        body("")
        body("Spectral mode applies the same compress-and-tile logic to STFT")
        body("magnitude frames instead of raw samples. Creates a different character \u2014")
        body("more diffuse and washy compared to time-domain's rhythmic tiling.")
        body("")

        h2("Pre/Post Filters")
        mapping("Pre-Filter", "Bypass / LP / HP / BP",
                "Filter BEFORE fractalization. Shapes what gets fractalized.")
        mapping("Post-Filter", "Bypass / LP / HP",
                "Filter AFTER fractalization. Tames aliasing artifacts.")
        mapping("Freq / Q", "filter_freq / filter_q",
                "Cutoff frequency and resonance.")
        body("")

        h2("Effects")
        mapping("Crush", "crush",
                "Bitcrusher (16-bit down to 4-bit). Post-fractal texture.")
        mapping("Decimate", "decimate",
                "Sample rate reduction. Metallic aliasing artifacts.")
        mapping("Gate", "gate",
                "Noise gate. Cleans up quiet fractal residue.")
        body("")

        h2("Bounce (LFO Modulation)")
        mapping("Bounce", "on/off",
                "Enables sine LFO modulation of one parameter.")
        mapping("Target", "Ratio / Decay / Scales / etc.",
                "Which parameter the LFO sweeps.")
        mapping("Rate", "bounce_rate",
                "LFO speed mapped to Min-Max Hz range.")
        body("")

        h2("Output")
        mapping("Wet/Dry", "wet_dry",
                "0=original signal, 1=fully fractalized.")
        mapping("Output Gain", "output_gain",
                "Volume. 0.5=unity, 0=-36dB, 1=+36dB.")
        mapping("Threshold", "threshold",
                "Limiter ceiling. Lower = more limiting.")
        body("")

        h2("Signal Chain")
        body("Input -> Pre-Filter -> Fractalize (x iterations, with saturation)")
        body("  -> Output Gain -> Crush/Decimate -> Post-Filter -> Gate -> Limiter -> Mix")
        body("")

        h2("Recipes")
        body("  Subtle texture: 2-3 scales, ratio 0.5, decay 0.5, linear interp")
        body("  Deep fractal:   5 scales, ratio 0.3, 2-3 iterations, some saturation")
        body("  Glitch:         8 scales, ratio 0.2, nearest, crush, decimate")
        body("  Spectral wash:  spectral 0.8, 4 scales, large window (4096)")
        body("  Guitar crunch:  2 scales, saturation 0.4, HP pre-filter at 200Hz")
        body("  Breathing:      bounce on, target ratio, slow rate")

        text.config(state="disabled")
