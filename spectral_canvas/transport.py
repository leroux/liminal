"""TransportPanel — play, stop, export, import, engine selection, duration."""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import librosa

from .model import SpectrogramModel
from .player import AudioPlayer
from .engines import get_engine_names, create_engine, load_all_engines


class TransportPanel(tk.Frame):
    """Bottom bar with playback controls, engine selection, import/export."""

    def __init__(self, parent, model: SpectrogramModel, player: AudioPlayer,
                 canvas_panel, status_var=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.model = model
        self.player = player
        self.canvas_panel = canvas_panel
        self._status_var = status_var  # shared StringVar for status bar

        load_all_engines()
        self._engine_names = get_engine_names()
        self._current_engine = create_engine(self._engine_names[0])
        self._rendering = False

        self._build_ui()
        self._cursor_update_id = None

    def _set_status(self, text):
        if self._status_var:
            self._status_var.set(text)

    def _build_ui(self):
        row = tk.Frame(self)
        row.pack(fill=tk.X, padx=5, pady=5)

        self._play_btn = tk.Button(
            row, text='Play', command=self._on_play,
            font=('Helvetica', 11, 'bold'), padx=10, pady=3)
        self._play_btn.pack(side=tk.LEFT, padx=3)

        tk.Button(
            row, text='Stop', command=self._on_stop,
            font=('Helvetica', 11, 'bold'), padx=10, pady=3
        ).pack(side=tk.LEFT, padx=3)

        tk.Button(
            row, text='Export WAV', command=self._on_export,
            font=('Helvetica', 11), padx=8, pady=3
        ).pack(side=tk.LEFT, padx=3)

        tk.Button(
            row, text='Import WAV', command=self._on_import,
            font=('Helvetica', 11), padx=8, pady=3
        ).pack(side=tk.LEFT, padx=3)

        tk.Button(
            row, text='Clear', command=self._on_clear,
            font=('Helvetica', 11), padx=8, pady=3
        ).pack(side=tk.LEFT, padx=3)

        tk.Frame(row, width=20).pack(side=tk.LEFT)

        tk.Label(row, text='Engine:', font=('Helvetica', 10)).pack(side=tk.LEFT)
        self._engine_var = tk.StringVar(value=self._engine_names[0])
        engine_menu = ttk.Combobox(row, textvariable=self._engine_var,
                                   values=self._engine_names, state='readonly',
                                   width=16)
        engine_menu.pack(side=tk.LEFT, padx=(2, 12))
        engine_menu.bind('<<ComboboxSelected>>', self._on_engine_change)

        tk.Label(row, text='Duration:', font=('Helvetica', 10)).pack(side=tk.LEFT)
        self._dur_var = tk.StringVar(value='5.0')
        dur_entry = tk.Entry(row, textvariable=self._dur_var, width=5,
                            font=('Helvetica', 10))
        dur_entry.pack(side=tk.LEFT, padx=(2, 4))
        dur_entry.bind('<Return>', self._on_duration_change)

        tk.Label(row, text='s', font=('Helvetica', 10)).pack(side=tk.LEFT, padx=(0, 12))

        self._source_frame = tk.Frame(row)
        tk.Label(self._source_frame, text='Source:', font=('Helvetica', 10)).pack(side=tk.LEFT)
        self._source_var = tk.StringVar(value='sawtooth_55')
        source_menu = ttk.Combobox(self._source_frame, textvariable=self._source_var,
                                   values=['sawtooth_55', 'sawtooth_110',
                                          'white_noise', 'pink_noise', 'imported'],
                                   state='readonly', width=12)
        source_menu.pack(side=tk.LEFT, padx=(2, 12))
        source_menu.bind('<<ComboboxSelected>>', self._on_source_change)
        # Only show for engines that use a source
        self._update_source_visibility()

    def _on_engine_change(self, event=None):
        name = self._engine_var.get()
        self._current_engine = create_engine(name)
        self._apply_source_to_engine()
        self._update_source_visibility()

    def _update_source_visibility(self):
        if hasattr(self._current_engine, 'source_type'):
            self._source_frame.pack(side=tk.LEFT)
        else:
            self._source_frame.pack_forget()

    def _on_source_change(self, event=None):
        self._apply_source_to_engine()

    def _apply_source_to_engine(self):
        if hasattr(self._current_engine, 'source_type'):
            self._current_engine.source_type = self._source_var.get()

    def _on_duration_change(self, event=None):
        try:
            dur = float(self._dur_var.get())
            dur = max(0.5, min(30.0, dur))
            self.model.set_duration(dur)
            self._dur_var.set(f'{dur:.1f}')
            self.canvas_panel.refresh_display()
        except ValueError:
            pass

    def _on_play(self):
        if self._rendering:
            return
        self._rendering = True
        self._set_status('Rendering...')
        self._play_btn.config(state=tk.DISABLED)

        self.player.render_async(self._current_engine)
        self._check_render()

    def _check_render(self):
        result = self.player.check_render_result()
        if result is None:
            self.after(50, self._check_render)
            return

        self._rendering = False
        self._play_btn.config(state=tk.NORMAL)

        status, data = result
        if status == 'done':
            dur = len(data) / self.model.sr
            self._set_status(f'Playing ({dur:.1f}s)')
            self.player.play(data)
            self._start_cursor_update()
        else:
            self._set_status(f'Error: {data}')
            messagebox.showerror('Render Error', str(data))

    def _on_stop(self):
        self.player.stop()
        self._stop_cursor_update()
        self.canvas_panel.set_cursor_position(0)
        self._set_status('Ready')

    def _start_cursor_update(self):
        self._stop_cursor_update()
        self._update_cursor()

    def _update_cursor(self):
        if self.player.is_playing:
            frac = self.player.playback_fraction
            self.canvas_panel.set_cursor_position(frac)
            self._cursor_update_id = self.after(30, self._update_cursor)
        else:
            self.canvas_panel.set_cursor_position(0)
            self._set_status('Ready')

    def _stop_cursor_update(self):
        if self._cursor_update_id:
            self.after_cancel(self._cursor_update_id)
            self._cursor_update_id = None

    def _on_export(self):
        if self.player._audio is None:
            messagebox.showinfo('Export', 'Nothing to export. Click Play first.')
            return
        path = filedialog.asksaveasfilename(
            defaultextension='.wav',
            filetypes=[('WAV files', '*.wav'), ('All files', '*.*')],
            title='Export Audio')
        if path:
            self.player.export_wav(path)
            self._set_status(f'Exported to {path}')

    def _on_import(self):
        path = filedialog.askopenfilename(
            filetypes=[('Audio files', '*.wav *.flac *.ogg *.mp3'),
                      ('All files', '*.*')],
            title='Import Audio')
        if not path:
            return

        try:
            self._set_status('Importing...')
            self.update_idletasks()

            y, sr = librosa.load(path, sr=self.model.sr, mono=True)
            D = librosa.stft(y, n_fft=self.model.n_fft,
                            hop_length=self.model.hop_length, window='hann')
            magnitude = np.abs(D)
            phase = np.angle(D)

            mag_db = librosa.amplitude_to_db(magnitude, ref=np.max)
            mag_min = mag_db.min()
            mag_max = mag_db.max()
            mag_normalized = (mag_db - mag_min) / (mag_max - mag_min + 1e-8)

            self.model.load_from_stft(mag_normalized.astype(np.float32), phase,
                                      original_magnitude=magnitude)
            self._dur_var.set(f'{self.model.duration:.1f}')
            self.canvas_panel.refresh_display()
            self._set_status('Imported!')

        except Exception as e:
            messagebox.showerror('Import Error', str(e))
            self._set_status('Ready')

    def _on_clear(self):
        self.model.clear()
        self.canvas_panel.refresh_display()
        self._set_status('Canvas cleared')

    def select_engine_by_number(self, num: int):
        """Select engine by 1-based index."""
        if 1 <= num <= len(self._engine_names):
            name = self._engine_names[num - 1]
            self._engine_var.set(name)
            self._on_engine_change()

    def toggle_play(self):
        if self.player.is_playing:
            self._on_stop()
        else:
            self._on_play()
