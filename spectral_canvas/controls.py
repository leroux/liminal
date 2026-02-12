"""ControlPanel — brush controls, engine selection, scale/tuning."""

import tkinter as tk
from tkinter import ttk

from .dsp.pitch import SCALES, NOTE_NAMES


class ControlPanel(tk.Frame):
    """Top toolbar with brush mode buttons, sliders, and dropdowns."""

    def __init__(self, parent, canvas_panel, **kwargs):
        super().__init__(parent, **kwargs)
        self.canvas_panel = canvas_panel
        self._build_ui()

    def _build_ui(self):
        # --- Row 1: Brush modes and basic controls ---
        row1 = tk.Frame(self)
        row1.pack(fill=tk.X, padx=5, pady=(5, 2))

        self._mode_var = tk.StringVar(value='draw')
        modes = [('Draw', 'draw'), ('Erase', 'erase'), ('Line', 'line'),
                 ('Harmonic', 'harmonic'), ('Select', 'select')]

        for label, mode in modes:
            rb = tk.Radiobutton(row1, text=label, variable=self._mode_var,
                               value=mode, command=self._on_mode_change,
                               indicatoron=0, padx=8, pady=2,
                               font=('Helvetica', 11))
            rb.pack(side=tk.LEFT, padx=2)

        tk.Frame(row1, width=20).pack(side=tk.LEFT)

        tk.Label(row1, text='Radius:', font=('Helvetica', 10)).pack(side=tk.LEFT)
        self._radius_var = tk.DoubleVar(value=8.0)
        self._radius_slider = tk.Scale(row1, from_=1, to=50, orient=tk.HORIZONTAL,
                                       variable=self._radius_var,
                                       command=self._on_radius_change,
                                       length=100, showvalue=1)
        self._radius_slider.pack(side=tk.LEFT, padx=(2, 8))

        tk.Label(row1, text='Soft:', font=('Helvetica', 10)).pack(side=tk.LEFT)
        self._soft_var = tk.DoubleVar(value=0.8)
        tk.Scale(row1, from_=0.1, to=2.0, resolution=0.1, orient=tk.HORIZONTAL,
                variable=self._soft_var, command=self._on_soft_change,
                length=100, showvalue=1
        ).pack(side=tk.LEFT, padx=(2, 8))

        tk.Label(row1, text='Intensity:', font=('Helvetica', 10)).pack(side=tk.LEFT)
        self._intensity_var = tk.DoubleVar(value=0.5)
        tk.Scale(row1, from_=0.0, to=1.0, resolution=0.05, orient=tk.HORIZONTAL,
                variable=self._intensity_var, command=self._on_intensity_change,
                length=100, showvalue=1
        ).pack(side=tk.LEFT, padx=(2, 8))

        # --- Row 2: Scale, harmonics ---
        row2 = tk.Frame(self)
        row2.pack(fill=tk.X, padx=5, pady=(2, 5))

        tk.Label(row2, text='Scale:', font=('Helvetica', 10)).pack(side=tk.LEFT)
        self._scale_var = tk.StringVar(value='Chromatic')
        scale_menu = ttk.Combobox(row2, textvariable=self._scale_var,
                                  values=list(SCALES.keys()), state='readonly',
                                  width=14)
        scale_menu.pack(side=tk.LEFT, padx=(2, 8))
        scale_menu.bind('<<ComboboxSelected>>', self._on_scale_change)

        tk.Label(row2, text='Root:', font=('Helvetica', 10)).pack(side=tk.LEFT)
        self._root_var = tk.StringVar(value='C')
        root_menu = ttk.Combobox(row2, textvariable=self._root_var,
                                 values=NOTE_NAMES, state='readonly', width=4)
        root_menu.pack(side=tk.LEFT, padx=(2, 8))
        root_menu.bind('<<ComboboxSelected>>', self._on_root_change)

        self._snap_var = tk.BooleanVar(value=False)
        tk.Checkbutton(row2, text='Snap', variable=self._snap_var,
                       command=self._on_snap_toggle,
                       font=('Helvetica', 10)).pack(side=tk.LEFT, padx=(0, 12))

        # Freq scale toggle
        tk.Label(row2, text='Y:', font=('Helvetica', 10)).pack(side=tk.LEFT)
        self._freq_scale_var = tk.StringVar(value='linear')
        ttk.Combobox(row2, textvariable=self._freq_scale_var,
                     values=['log', 'linear'], state='readonly', width=6
        ).pack(side=tk.LEFT, padx=(2, 12))
        self._freq_scale_var.trace_add('write', self._on_freq_scale_change)

        # Harmonic controls (shown when harmonic mode active)
        self._harm_frame = tk.Frame(row2)

        tk.Label(self._harm_frame, text='Rolloff:',
                font=('Helvetica', 10)).pack(side=tk.LEFT)
        self._rolloff_var = tk.DoubleVar(value=0.5)
        tk.Scale(self._harm_frame, from_=0.0, to=3.0, resolution=0.1,
                orient=tk.HORIZONTAL, variable=self._rolloff_var,
                command=self._on_rolloff_change,
                length=100, showvalue=1
        ).pack(side=tk.LEFT, padx=(2, 8))

        tk.Label(self._harm_frame, text='Harmonics:',
                font=('Helvetica', 10)).pack(side=tk.LEFT)
        self._nharm_var = tk.IntVar(value=8)
        tk.Scale(self._harm_frame, from_=1, to=16, orient=tk.HORIZONTAL,
                variable=self._nharm_var, command=self._on_nharm_change,
                length=100, showvalue=1
        ).pack(side=tk.LEFT, padx=(2, 8))

        tk.Label(self._harm_frame, text='Sustain:',
                font=('Helvetica', 10)).pack(side=tk.LEFT)
        self._sustain_var = tk.IntVar(value=20)
        tk.Scale(self._harm_frame, from_=1, to=80, orient=tk.HORIZONTAL,
                variable=self._sustain_var, command=self._on_sustain_change,
                length=100, showvalue=1
        ).pack(side=tk.LEFT, padx=(2, 8))

        tk.Label(self._harm_frame, text='Type:',
                font=('Helvetica', 10)).pack(side=tk.LEFT)
        self._harm_mode_var = tk.StringVar(value='All')
        harm_mode_menu = ttk.Combobox(
            self._harm_frame, textvariable=self._harm_mode_var,
            values=['All', 'Odd', 'Even', 'Octaves', 'Fifths', 'Sub', 'Both'],
            state='readonly', width=8)
        harm_mode_menu.pack(side=tk.LEFT, padx=(2, 8))
        harm_mode_menu.bind('<<ComboboxSelected>>', self._on_harm_mode_change)

        tk.Label(self._harm_frame, text='Timbre:',
                font=('Helvetica', 10)).pack(side=tk.LEFT)
        self._timbre_var = tk.StringVar(value='Custom')
        timbre_menu = ttk.Combobox(
            self._harm_frame, textvariable=self._timbre_var,
            values=['Custom', 'Organ', 'Flute', 'Strings', 'Bell',
                    'Pad', 'Brass', 'Sub Bass', 'Choir'],
            state='readonly', width=9)
        timbre_menu.pack(side=tk.LEFT, padx=(2, 8))
        timbre_menu.bind('<<ComboboxSelected>>', self._on_timbre_change)

        # --- Help button (right side of row1) ---
        self._help_btn = tk.Button(row1, text='?', command=self._toggle_help,
                                   font=('Helvetica', 11, 'bold'), width=2)
        self._help_btn.pack(side=tk.RIGHT, padx=5)

        # --- Help panel (hidden by default) ---
        self._help_frame = tk.Frame(self)
        self._help_visible = False
        help_text = (
            "MODES: [B] Draw  [E] Erase  [L] Line (click start, click end)  "
            "[H] Harmonic (paints overtone stack)  Select (drag rectangle)\n"
            "BRUSH: Scroll=radius  Soft=edge blur  Intensity=brightness per stroke  "
            "Right-click=erase anywhere\n"
            "SCALE: Pick a scale + root, check Snap to lock drawing to scale notes. "
            "Guide lines appear when Snap is on.\n"
            "HARMONIC: Rolloff=how fast upper harmonics fade  Harmonics=count  "
            "Sustain=note length in frames  Type=overtone structure  Timbre=presets\n"
            "ZOOM: Shift+scroll=zoom at cursor  Shift+drag=pan  [+][-] zoom  [0] reset  Arrows=pan\n"
            "PLAYBACK: [Space] play/stop  [1-9] select engine  "
            "[Cmd+O] import WAV  [Cmd+S] export WAV  [Cmd+Z] undo\n"
            "TIPS: Import a WAV and use Spectral Filter engine to edit it. "
            "Use Additive engine for cleanest painted sounds. "
            "Try Harmonic mode + Snap + Pentatonic scale for easy melodies."
        )
        tk.Label(self._help_frame, text=help_text, justify=tk.LEFT,
                font=('Monaco', 9), wraplength=1200, anchor=tk.W,
                padx=8, pady=5, relief=tk.GROOVE).pack(fill=tk.X)

    # --- Callbacks ---

    def _on_mode_change(self):
        mode = self._mode_var.get()
        self.canvas_panel.brush_mode = mode
        if mode == 'harmonic':
            self._harm_frame.pack(side=tk.LEFT)
        else:
            self._harm_frame.pack_forget()

    def _on_radius_change(self, val):
        self.canvas_panel.brush_radius = float(val)

    def _on_soft_change(self, val):
        self.canvas_panel.brush_softness = float(val)

    def _on_intensity_change(self, val):
        self.canvas_panel.brush_intensity = float(val)

    def _on_scale_change(self, event=None):
        self.canvas_panel.scale_name = self._scale_var.get()
        self.canvas_panel.refresh_display()

    def _on_root_change(self, event=None):
        self.canvas_panel.scale_root = self._root_var.get()
        self.canvas_panel.refresh_display()

    def _on_snap_toggle(self):
        self.canvas_panel.scale_snap = self._snap_var.get()
        self.canvas_panel.refresh_display()

    def _on_freq_scale_change(self, *args):
        self.canvas_panel.freq_scale = self._freq_scale_var.get()
        self.canvas_panel._log_bin_map = None  # invalidate cache
        self.canvas_panel.refresh_display()

    def _on_rolloff_change(self, val):
        self.canvas_panel.harmonic_rolloff = float(val)

    def _on_nharm_change(self, val):
        self.canvas_panel.n_harmonics = int(float(val))

    def _on_sustain_change(self, val):
        self.canvas_panel.harmonic_sustain = int(float(val))

    def _on_harm_mode_change(self, event=None):
        mode_map = {'All': 0, 'Odd': 1, 'Even': 2, 'Octaves': 3, 'Fifths': 4, 'Sub': 5, 'Both': 6}
        self.canvas_panel.harmonic_mode = mode_map.get(self._harm_mode_var.get(), 0)
        self._timbre_var.set('Custom')

    def _on_timbre_change(self, event=None):
        # Timbre presets: (n_harmonics, rolloff, mode_name, sustain, softness, intensity)
        timbres = {
            'Organ':    (8,  0.3, 'Odd',     30, 0.6, 0.6),
            'Flute':    (3,  2.5, 'All',     25, 1.2, 0.7),
            'Strings':  (12, 1.0, 'All',     40, 0.8, 0.5),
            'Bell':     (16, 0.3, 'All',     15, 0.5, 0.4),
            'Pad':      (16, 0.2, 'All',     60, 1.5, 0.4),
            'Brass':    (10, 0.6, 'Odd',     20, 0.5, 0.7),
            'Sub Bass': (4,  0.8, 'Sub',     40, 1.0, 0.8),
            'Choir':    (8,  0.5, 'Odd',     35, 1.2, 0.5),
        }
        name = self._timbre_var.get()
        if name == 'Custom' or name not in timbres:
            return
        n_harm, rolloff, mode, sustain, soft, intensity = timbres[name]
        # Apply all parameters
        self._nharm_var.set(n_harm)
        self._rolloff_var.set(rolloff)
        self._harm_mode_var.set(mode)
        self._sustain_var.set(sustain)
        self._soft_var.set(soft)
        self._intensity_var.set(intensity)
        # Push to canvas
        self.canvas_panel.n_harmonics = n_harm
        self.canvas_panel.harmonic_rolloff = rolloff
        mode_map = {'All': 0, 'Odd': 1, 'Even': 2, 'Octaves': 3, 'Fifths': 4, 'Sub': 5, 'Both': 6}
        self.canvas_panel.harmonic_mode = mode_map.get(mode, 0)
        self.canvas_panel.harmonic_sustain = sustain
        self.canvas_panel.brush_softness = soft
        self.canvas_panel.brush_intensity = intensity

    def _toggle_help(self):
        if self._help_visible:
            self._help_frame.pack_forget()
            self._help_visible = False
        else:
            self._help_frame.pack(fill=tk.X, padx=5, pady=(0, 5))
            self._help_visible = True

    def set_mode(self, mode: str):
        self._mode_var.set(mode)
        self._on_mode_change()
