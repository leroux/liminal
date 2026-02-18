"""Lossy codec emulation GUI — tkinter.

Tabs: Parameters · Presets · Waveforms · Spectrograms · Spectrum · Guide
"""

import os
import tkinter as tk
from tkinter import ttk

import numpy as np

from lossy.engine.params import (
    SR,
    SCHEMA,
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
    BOUNCE_TARGET_NAMES,
)
from lossy.engine.lossy import render_lossy
from shared.llm_guide_text import LOSSY_GUIDE, LOSSY_PARAM_DESCRIPTIONS
from shared.gui import PedalGUIBase, PedalConfig

PRESET_DIR = os.path.join(os.path.dirname(__file__), "presets")
DEFAULT_SOURCE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    os.pardir, "audio", "test_signals", "dry_noise_burst.wav",
)

PRESET_CATEGORIES = [
    "Codec", "Communication", "Lo-fi", "Textural",
    "Ghost / Residue", "Glitch", "Modulated", "Sound Design",
]


def _make_config():
    return PedalConfig(
        name="Lossy",
        preset_dir=PRESET_DIR,
        preset_categories=PRESET_CATEGORIES,
        window_title="Lossy \u2014 codec artifact emulator",
        window_geometry="900x680",
        default_params=default_params,
        bypass_params=bypass_params,
        param_ranges=PARAM_RANGES,
        param_sections=PARAM_SECTIONS,
        choice_ranges=CHOICE_RANGES,
        render=render_lossy,
        render_stereo=None,
        guide_text=LOSSY_GUIDE,
        param_descriptions=LOSSY_PARAM_DESCRIPTIONS,
        sample_rate=SR,
        default_source=DEFAULT_SOURCE,
        icon_path=os.path.join(os.path.dirname(__file__), os.pardir, os.pardir,
                               "icons", "lossy.png"),
        extra_tabs=[],   # set in __init__ (needs self ref)
        migrate_preset=migrate_legacy_params,
        tail_param="tail_length",
        schema=SCHEMA,
    )


class LossyGUI(PedalGUIBase):

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

        # ---- Spectral Loss ----
        r = self._add_section_header(f, r, "spectral", "SPECTRAL LOSS")

        self._inverse_var = tk.IntVar(value=self.params["inverse"])
        ttk.Checkbutton(f, text="Inverse (residual)", variable=self._inverse_var).grid(
            row=r, column=0, columnspan=2, sticky="w", padx=8)
        self._add_lock(f, r, "inverse")
        self.choice_vars["inverse"] = self._inverse_var
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

        self._quantizer_var, r = self._add_choice(f, r, "quantizer", "Quantizer", QUANTIZER_NAMES, self.params["quantizer"])

        # ---- Crush ----
        r = self._add_separator(f, r)
        r = self._add_section_header(f, r, "crush", "CRUSH")

        r = self._add_slider(f, r, "crush", "Crush", 0.0, 1.0, self.params["crush"], length=SL)
        r = self._add_slider(f, r, "decimate", "Decimate", 0.0, 1.0, self.params["decimate"], length=SL)

        # ---- Packets ----
        r = self._add_separator(f, r)
        r = self._add_section_header(f, r, "packets", "PACKETS")

        self._packets_var, r = self._add_choice(f, r, "packets", "Packets", PACKET_NAMES, self.params["packets"])
        r = self._add_slider(f, r, "packet_rate", "Pkt Rate", 0.0, 1.0, self.params["packet_rate"], length=SL)
        r = self._add_slider(f, r, "packet_size", "Pkt Size (ms)", 5.0, 200.0, self.params["packet_size"], length=SL)

        # ---- Filter ----
        r = self._add_separator(f, r)
        r = self._add_section_header(f, r, "filter", "FILTER")

        self._filter_var, r = self._add_choice(f, r, "filter_type", "Filter", FILTER_NAMES, self.params["filter_type"])
        r = self._add_slider(f, r, "filter_freq", "Freq (Hz)", 20.0, 20000.0, self.params["filter_freq"], length=SL, log=True)
        r = self._add_slider(f, r, "filter_width", "Width", 0.0, 1.0, self.params["filter_width"], length=SL)

        slope_names = [f"{v} dB" for v in SLOPE_OPTIONS]
        self._slope_var, r = self._add_choice(f, r, "filter_slope", "Slope", slope_names, self.params["filter_slope"])

        # ---- Effects ----
        r = self._add_separator(f, r)
        r = self._add_section_header(f, r, "effects", "EFFECTS")

        r = self._add_slider(f, r, "verb", "Verb", 0.0, 1.0, self.params["verb"], length=SL)
        r = self._add_slider(f, r, "decay", "Decay", 0.0, 1.0, self.params["decay"], length=SL)
        self._verb_pos_var, r = self._add_choice(f, r, "verb_position", "Verb Pos", VERB_POSITION_NAMES, self.params["verb_position"])
        r = self._add_slider(f, r, "gate", "Gate", 0.0, 1.0, self.params["gate"], length=SL)

        # Freeze checkbox + mode (share one lock)
        self._freeze_var = tk.IntVar(value=self.params["freeze"])
        ttk.Checkbutton(f, text="Freeze", variable=self._freeze_var).grid(
            row=r, column=0, sticky="w", padx=8)
        self._freeze_mode_var = tk.IntVar(value=self.params["freeze_mode"])
        fm_frame = ttk.Frame(f)
        fm_frame.grid(row=r, column=1, sticky="w")
        for i, name in enumerate(FREEZE_NAMES):
            ttk.Radiobutton(fm_frame, text=name, variable=self._freeze_mode_var, value=i).pack(side="left", padx=3)
        freeze_lock = tk.BooleanVar(value=False)
        self.locks["freeze"] = freeze_lock
        self.locks["freeze_mode"] = freeze_lock
        ttk.Checkbutton(f, variable=freeze_lock).grid(row=r, column=3, sticky="w", padx=2)
        self.choice_vars["freeze"] = self._freeze_var
        self.choice_vars["freeze_mode"] = self._freeze_mode_var
        r += 1

        r = self._add_slider(f, r, "freezer", "Freezer", 0.0, 1.0, self.params["freezer"], length=SL)

        # ---- Hidden Options ----
        r = self._add_separator(f, r)
        r = self._add_section_header(f, r, "hidden", "HIDDEN OPTIONS")

        r = self._add_slider(f, r, "threshold", "Threshold", 0.0, 1.0, self.params["threshold"], length=SL)
        r = self._add_slider(f, r, "auto_gain", "Auto Gain", 0.0, 1.0, self.params["auto_gain"], length=SL)
        r = self._add_slider(f, r, "loss_gain", "Loss Gain", 0.0, 1.0, self.params["loss_gain"], length=SL)
        r = self._add_slider(f, r, "weighting", "Weighting", 0.0, 1.0, self.params["weighting"], length=SL)

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
        r = self._add_slider(f, r, "tail_length", "Tail Length (s)", 0.0, 60.0, 2.0, length=SL)

        # Auto-play on manually-created discrete vars
        # (vars from _add_choice already have autoplay traces)
        for var in [self._inverse_var, self._freeze_var, self._freeze_mode_var,
                    self._bounce_var]:
            var.trace_add("write", lambda *_: self._schedule_autoplay())

    # ------------------------------------------------------------------
    # Param read / write
    # ------------------------------------------------------------------

    def _read_params_from_ui(self):
        p = self.cfg.default_params()
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
                "Master intensity \u2014 scales Loss, Phase, Crush, Packets, Verb, Gate together.")
        body("")

        h2("Inverse + Jitter (independent controls)")
        mapping("Standard/Inverse", "Inverse checkbox",
                "Off = hear processed signal. On = hear the residual (everything Standard discards).")
        mapping("Jitter", "Jitter slider",
                "Random phase perturbation per FFT bin (0=off, 1=max). Emulates bad digital clocking.")
        body("")
        body("Inverse and Jitter are independent \u2014 you can combine them (e.g. inverse+jitter).")
        body('The "underwater" warble comes from zeroing different bands each frame.')
        body("Band gating is weighted by signal energy and the ATH (Absolute Threshold")
        body("of Hearing) curve \u2014 quieter bands at less-sensitive frequencies get gated")
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
                "Bitcrusher \u2014 reduces amplitude quantization levels (16-bit down to ~4-bit).")
        mapping("(not on pedal)", "Decimate",
                "Sample rate reducer \u2014 zero-order hold. Aliasing creates metallic overtones.")
        body("")
        body("These are complementary to spectral loss. Crush creates amplitude staircase")
        body("distortion; Decimate creates inharmonic aliasing. Neither sounds like a codec \u2014")
        body("they sound like early digital hardware (SP-1200, Fairlight, NES).")
        body("")

        h2("Packets Toggle (3-way)")
        mapping("Clean", "Packets: Clean",
                "No packet processing.")
        mapping("Packet Loss", "Packets: Packet Loss",
                "Gilbert-Elliott dropout model \u2014 bursty silence gaps with crossfade.")
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
                "Lo-fi Schroeder reverb \u2014 short combs + allpass. Deliberately cheap & metallic.")
        mapping("Decay (hidden)", "Decay",
                "Reverb size/length. 0=short metallic, 1=long wash.")
        mapping("Verb Pre/Post dip", "Verb Pos: Pre / Post",
                "Pre = verb before loss (default, PDF p.27). Post = verb after filter.")
        mapping("Freeze footswitch", "Freeze checkbox",
                "Captures a spectral snapshot and holds it.")
        mapping("Freeze: slushy", "Freeze Mode: Slushy",
                "Frozen spectrum slowly updates at Slushy Rate.")
        mapping("Freeze: solid", "Freeze Mode: Solid",
                "Spectrum is truly frozen \u2014 static drone.")
        mapping("Freezer (hidden)", "Freezer",
                "Blend between frozen spectrum and live signal. 1=frozen, 0=live.")
        mapping("Gate", "Gate",
                "RMS noise gate. Cleans up residual artifacts in quiet passages.")
        body("")

        h2("Hidden Options (PDF pp.14-16)")
        mapping("Threshold (hidden)", "Threshold",
                "Limiter threshold \u2014 0=heavy limiting, 1=light. Lower = more compression.")
        mapping("Auto Gain (hidden)", "Auto Gain",
                "Loudness compensation for Loss modes. Keeps volume consistent as loss increases.")
        mapping("Loss Gain (hidden)", "Loss Gain",
                "Wet signal volume. 0=-36dB, 0.5=0dB (unity), 1=+36dB boost.")
        mapping("Weighting (hidden)", "Weighting",
                "0=equal freq weighting, 1=psychoacoustic ATH model. Favours some freqs over others.")
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
        body("   step size per bin \u2014 coarser in spectral valleys, finer near peaks.")
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
        body("  underwater         Heavy loss, slow window \u2014 classic codec-degraded sound")
        body("  low_bitrate_mp3    Moderate loss \u2014 sounds like a 64kbps MP3")
        body("  spectral_residue   Inverse mode \u2014 ghostly harmonics the codec discards")
        body("  bad_connection     Packet loss + bandpass \u2014 choppy VoIP call")
        body("  glitch_stutter     Packet repeat \u2014 rhythmic buffer-freeze glitches")
        body("  frozen_pad         Freeze + verb \u2014 evolving spectral drone")
        body("  resonant_telephone Bandpass 96dB slope + gate \u2014 lo-fi phone line")

        text.config(state="disabled")
