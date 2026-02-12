"""AudioPlayer — sounddevice integration with playback cursor."""

import threading
import queue
import numpy as np
import sounddevice as sd
import soundfile as sf

from .model import SpectrogramModel
from .engines import Engine
from .dsp.utils import normalize


class AudioPlayer:
    """Manages audio rendering, playback, and export."""

    def __init__(self, model: SpectrogramModel):
        self.model = model
        self._audio: np.ndarray | None = None
        self._playing = False
        self._playback_pos = 0  # current sample position
        self._render_queue = queue.Queue()
        self._stream: sd.OutputStream | None = None

    @property
    def is_playing(self) -> bool:
        return self._playing

    @property
    def playback_fraction(self) -> float:
        """Current playback position as fraction 0-1."""
        if self._audio is None:
            return 0.0
        total = len(self._audio)
        if total == 0:
            return 0.0
        return min(1.0, self._playback_pos / total)

    def render_async(self, engine: Engine, callback=None):
        """Render in background thread. Calls callback(audio) on completion."""
        mag = self.model.magnitude.copy()
        phase = self.model.phase.copy() if self.model.phase is not None else None
        orig_mag = self.model.original_magnitude.copy() if self.model.original_magnitude is not None else None
        sr = self.model.sr
        n_fft = self.model.n_fft
        hop = self.model.hop_length

        import_display_mag = self.model.import_display_magnitude.copy() if self.model.import_display_magnitude is not None else None

        def _do_render():
            try:
                if hasattr(engine, 'original_magnitude'):
                    engine.original_magnitude = orig_mag
                if hasattr(engine, 'import_display_magnitude'):
                    engine.import_display_magnitude = import_display_mag
                audio = engine.render(mag, phase, sr, n_fft, hop)
                audio = normalize(audio)
                self._render_queue.put(('done', audio))
                if callback:
                    callback(audio)
            except Exception as e:
                self._render_queue.put(('error', str(e)))

        t = threading.Thread(target=_do_render, daemon=True)
        t.start()

    def check_render_result(self):
        """Non-blocking check for render completion. Returns (status, data) or None."""
        try:
            return self._render_queue.get_nowait()
        except queue.Empty:
            return None

    def play(self, audio: np.ndarray | None = None):
        """Play audio via sounddevice."""
        if audio is not None:
            self._audio = audio
        if self._audio is None:
            return

        self.stop()
        self._playback_pos = 0
        self._playing = True

        sr = self.model.sr
        audio_data = self._audio.copy()

        def _callback(outdata, frames, time_info, status):
            start = self._playback_pos
            end = start + frames
            if end > len(audio_data):
                # Pad with zeros at end
                valid = len(audio_data) - start
                if valid > 0:
                    outdata[:valid, 0] = audio_data[start:start + valid]
                outdata[valid:, 0] = 0
                self._playing = False
                raise sd.CallbackStop()
            else:
                outdata[:, 0] = audio_data[start:end]
            self._playback_pos = end

        self._stream = sd.OutputStream(
            samplerate=sr,
            channels=1,
            dtype='float32',
            callback=_callback,
            finished_callback=self._on_finished,
        )
        self._stream.start()

    def _on_finished(self):
        self._playing = False

    def stop(self):
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        self._playing = False
        self._playback_pos = 0

    def export_wav(self, filepath: str):
        """Export current audio to WAV file."""
        if self._audio is None:
            return
        audio = normalize(self._audio)
        sf.write(filepath, audio, self.model.sr, subtype='PCM_16')
