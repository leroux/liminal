"""SpectrogramModel — the data layer for the spectral painter."""

import numpy as np
from scipy.ndimage import zoom

SR = 44100
N_FFT = 2048
HOP_LENGTH = 512
N_FREQ_BINS = N_FFT // 2 + 1  # 1025
WINDOW = 'hann'


class SpectrogramModel:
    """Holds the magnitude/phase spectrogram arrays and metadata."""

    def __init__(self, duration: float = 5.0):
        self.sr = SR
        self.n_fft = N_FFT
        self.hop_length = HOP_LENGTH
        self.n_freq_bins = N_FREQ_BINS
        self._duration = duration
        self.n_frames = self._duration_to_frames(duration)

        # Core data arrays
        self.magnitude = np.zeros((self.n_freq_bins, self.n_frames), dtype=np.float32)
        self.phase: np.ndarray | None = None  # set when audio is imported
        self.original_magnitude: np.ndarray | None = None  # linear magnitude from STFT
        self.import_display_magnitude: np.ndarray | None = None  # 0-1 display magnitude at import time
        self.modified_mask = np.zeros((self.n_freq_bins, self.n_frames), dtype=bool)

        # Undo/redo
        self._undo_stack: list[np.ndarray] = []
        self._redo_stack: list[np.ndarray] = []
        self._max_undo = 20

    @property
    def duration(self) -> float:
        return self._duration

    def _duration_to_frames(self, dur: float) -> int:
        total_samples = int(dur * self.sr)
        return max(1, total_samples // self.hop_length + 1)

    def set_duration(self, new_duration: float):
        new_frames = self._duration_to_frames(new_duration)
        if new_frames == self.n_frames:
            self._duration = new_duration
            return
        # Resample magnitude
        zoom_factor = (1.0, new_frames / self.n_frames)
        self.magnitude = zoom(self.magnitude, zoom_factor, order=1).astype(np.float32)
        np.clip(self.magnitude, 0, 1, out=self.magnitude)
        if self.phase is not None:
            self.phase = zoom(self.phase, zoom_factor, order=1).astype(np.float32)
        self.modified_mask = zoom(self.modified_mask.astype(np.float32), zoom_factor, order=0).astype(bool)
        self.n_frames = self.magnitude.shape[1]
        self._duration = new_duration

    def clear(self):
        self.push_undo()
        self.magnitude[:] = 0
        self.phase = None
        self.original_magnitude = None
        self.import_display_magnitude = None
        self.modified_mask[:] = False

    def push_undo(self):
        self._undo_stack.append(self.magnitude.copy())
        if len(self._undo_stack) > self._max_undo:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def undo(self) -> bool:
        if not self._undo_stack:
            return False
        self._redo_stack.append(self.magnitude.copy())
        self.magnitude = self._undo_stack.pop()
        return True

    def redo(self) -> bool:
        if not self._redo_stack:
            return False
        self._undo_stack.append(self.magnitude.copy())
        self.magnitude = self._redo_stack.pop()
        return True

    def load_from_stft(self, magnitude: np.ndarray, phase: np.ndarray | None,
                       original_magnitude: np.ndarray | None = None):
        """Load an imported audio spectrogram.

        Args:
            magnitude: display magnitude (0-1 normalized)
            phase: original phase from STFT
            original_magnitude: linear magnitude before dB normalization (for faithful reconstruction)
        """
        self.push_undo()
        self.magnitude = magnitude.astype(np.float32)
        self.phase = phase.astype(np.float32) if phase is not None else None
        self.original_magnitude = original_magnitude.astype(np.float32) if original_magnitude is not None else None
        self.import_display_magnitude = magnitude.copy().astype(np.float32)
        self.n_frames = self.magnitude.shape[1]
        self._duration = (self.n_frames - 1) * self.hop_length / self.sr
        self.modified_mask = np.zeros_like(self.magnitude, dtype=bool)
