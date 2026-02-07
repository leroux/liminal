"""Shared audio streaming utilities for both pedals.

StreamPlayer manages an sd.OutputStream for chunk-based playback.
safety_check / normalize_output deduplicate the post-render pipeline.
"""

import sys
import time

import numpy as np
import sounddevice as sd


class StreamPlayer:
    """Manages streaming audio playback via sd.OutputStream."""

    def __init__(self, sr=44100, channels=2):
        self.sr = sr
        self.channels = channels
        self._stream = None
        self._stop_flag = False

    def start(self):
        """Open and start the output stream."""
        self._stop_flag = False
        self._stream = sd.OutputStream(
            samplerate=self.sr, channels=self.channels, dtype='float32',
        )
        self._stream.start()

    def write_chunk(self, chunk):
        """Write a chunk to the stream.

        Usable directly as ``chunk_callback`` for render_fdn.
        Returns True to continue, False to stop.
        """
        if self._stop_flag:
            return False
        clipped = np.clip(chunk, -1.0, 1.0).astype(np.float32)
        # Ensure stereo for the output device
        if clipped.ndim == 1 and self.channels == 2:
            clipped = np.column_stack([clipped, clipped])
        try:
            self._stream.write(clipped)
            return True
        except Exception as exc:
            print(f"StreamPlayer.write_chunk error: {exc}", file=sys.stderr)
            return False

    def stream_buffer(self, audio, chunk_size=4096):
        """Stream a pre-rendered buffer in chunks.

        Returns True if completed, False if cancelled.
        """
        n = audio.shape[0]
        for start in range(0, n, chunk_size):
            if not self.write_chunk(audio[start:start + chunk_size]):
                return False
        return True

    def stop(self):
        """Signal stop and abort the stream immediately."""
        self._stop_flag = True
        if self._stream is not None:
            try:
                self._stream.abort()
            except Exception:
                pass

    def close(self, cancelled=False):
        """Close the stream gracefully (drain) or abruptly."""
        if self._stream is not None:
            try:
                if not cancelled:
                    time.sleep(0.05)  # let buffer drain
                    self._stream.stop()
                else:
                    self._stream.abort()
            except Exception:
                pass
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    @property
    def stopped(self):
        return self._stop_flag


def safety_check(output):
    """Reject non-finite or exploded output.

    Returns (ok, error_message).
    """
    if not np.all(np.isfinite(output)):
        return False, "ERROR: output diverged (non-finite values)"
    peak = np.max(np.abs(output))
    if peak > 1e6:
        return False, f"ERROR: output exploded (peak={peak:.0e})"
    return True, ""


def normalize_output(output, target_rms=0.2, headroom=0.9):
    """RMS limiter + headroom.

    Returns (normalized_output, warning_string).
    """
    peak = np.max(np.abs(output))
    if peak > 0:
        output = output / peak
    rms = np.sqrt(np.mean(output ** 2))
    if rms > target_rms:
        gain = target_rms / rms
        output = output * gain
        warning = f" (loud — reduced {1/gain:.0f}x)"
    else:
        warning = ""
    output = output * headroom
    return output, warning
