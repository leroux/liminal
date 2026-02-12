"""Spectral Painter — main entry point."""

import tkinter as tk
from tkinter import ttk

from .model import SpectrogramModel
from .canvas import CanvasPanel
from .controls import ControlPanel
from .transport import TransportPanel
from .player import AudioPlayer


class SpectralPainterApp:
    """Main application window."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title('Spectral Painter')
        self.root.geometry('1300x780')
        self.root.minsize(800, 500)

        # Model
        self.model = SpectrogramModel(duration=5.0)

        # Player
        self.player = AudioPlayer(self.model)

        # Shared status text
        self._status_var = tk.StringVar(value='Ready')

        # Canvas panel (center)
        self.canvas_panel = CanvasPanel(self.root, self.model)

        # Control panel (top)
        self.control_panel = ControlPanel(self.root, self.canvas_panel)

        # Transport panel (bottom)
        self.transport_panel = TransportPanel(
            self.root, self.model, self.player, self.canvas_panel,
            status_var=self._status_var)

        # Status bar
        self.status_bar = tk.Label(
            self.root, textvariable=self._status_var,
            anchor=tk.W, font=('Monaco', 10), padx=10, pady=3,
            relief=tk.SUNKEN)

        # Layout
        self.control_panel.pack(fill=tk.X)
        self.canvas_panel.pack(fill=tk.BOTH, expand=True)
        self.transport_panel.pack(fill=tk.X)
        self.status_bar.pack(fill=tk.X)

        # Wire hover status updates
        self.canvas_panel.on_status_update = self._update_status

        # Keyboard shortcuts
        self._bind_shortcuts()

    def _update_status(self, freq_hz, note, time_s):
        engine_name = self.transport_panel._engine_var.get()
        self._status_var.set(
            f'freq={freq_hz:.1f} Hz ({note}) | time={time_s:.2f}s | engine={engine_name}')

    def _bind_shortcuts(self):
        self.root.bind('<space>', lambda e: self.transport_panel.toggle_play())
        self.root.bind('<Escape>', lambda e: self.transport_panel._on_stop())

        for i in range(1, 10):
            self.root.bind(str(i), lambda e, n=i: self.transport_panel.select_engine_by_number(n))

        self.root.bind('b', lambda e: self.control_panel.set_mode('draw'))
        self.root.bind('e', lambda e: self.control_panel.set_mode('erase'))
        self.root.bind('l', lambda e: self.control_panel.set_mode('line'))
        self.root.bind('h', lambda e: self.control_panel.set_mode('harmonic'))

        self.root.bind('bracketleft', lambda e: self._adjust_radius(-2))
        self.root.bind('bracketright', lambda e: self._adjust_radius(2))

        self.root.bind('<Command-z>', self._on_undo)
        self.root.bind('<Control-z>', self._on_undo)
        self.root.bind('<Command-Shift-z>', self._on_redo)
        self.root.bind('<Control-Shift-z>', self._on_redo)
        self.root.bind('<Command-Shift-Z>', self._on_redo)
        self.root.bind('<Control-Shift-Z>', self._on_redo)

        self.root.bind('<Command-s>', lambda e: self.transport_panel._on_export())
        self.root.bind('<Control-s>', lambda e: self.transport_panel._on_export())
        self.root.bind('<Command-o>', lambda e: self.transport_panel._on_import())
        self.root.bind('<Control-o>', lambda e: self.transport_panel._on_import())

        self.root.bind('<Delete>', lambda e: self._clear_canvas())
        self.root.bind('<BackSpace>', lambda e: self._clear_canvas())

        # Zoom: +/- keys, 0 to reset, arrow keys to pan
        self.root.bind('<equal>', lambda e: self.canvas_panel.zoom_in())
        self.root.bind('<plus>', lambda e: self.canvas_panel.zoom_in())
        self.root.bind('<minus>', lambda e: self.canvas_panel.zoom_out())
        self.root.bind('0', lambda e: self.canvas_panel.reset_zoom())
        self.root.bind('<Left>', lambda e: self.canvas_panel.pan_by(-0.2, 0))
        self.root.bind('<Right>', lambda e: self.canvas_panel.pan_by(0.2, 0))
        self.root.bind('<Up>', lambda e: self.canvas_panel.pan_by(0, -0.2))
        self.root.bind('<Down>', lambda e: self.canvas_panel.pan_by(0, 0.2))

    def _adjust_radius(self, delta):
        r = self.canvas_panel.brush_radius + delta
        r = max(1, min(50, r))
        self.canvas_panel.brush_radius = r
        self.control_panel._radius_var.set(r)

    def _on_undo(self, event=None):
        if self.model.undo():
            self.canvas_panel.refresh_display()

    def _on_redo(self, event=None):
        if self.model.redo():
            self.canvas_panel.refresh_display()

    def _clear_canvas(self):
        self.model.clear()
        self.canvas_panel.refresh_display()

    def run(self):
        self.root.mainloop()


def main():
    app = SpectralPainterApp()
    app.run()


if __name__ == '__main__':
    main()
