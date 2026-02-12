"""Frequency / pixel / bin conversions and scale snapping."""

import numpy as np

MIN_FREQ = 20.0
MAX_FREQ = 20000.0

# Scale intervals (semitones from root)
SCALES = {
    'Chromatic':       list(range(12)),
    'Major':           [0, 2, 4, 5, 7, 9, 11],
    'Minor':           [0, 2, 3, 5, 7, 8, 10],
    'Pentatonic':      [0, 2, 4, 7, 9],
    'Blues':            [0, 3, 5, 6, 7, 10],
    'Dorian':          [0, 2, 3, 5, 7, 9, 10],
    'Mixolydian':      [0, 2, 4, 5, 7, 9, 10],
    'Whole Tone':      [0, 2, 4, 6, 8, 10],
    'Harmonic Minor':  [0, 2, 3, 5, 7, 8, 11],
}

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def pixel_y_to_freq(y_pixel: float, canvas_height: int,
                    min_freq: float = MIN_FREQ, max_freq: float = MAX_FREQ) -> float:
    ratio = y_pixel / canvas_height
    log_max = np.log2(max_freq)
    log_min = np.log2(min_freq)
    return float(2 ** (log_max - ratio * (log_max - log_min)))


def freq_to_pixel_y(freq: float, canvas_height: int,
                    min_freq: float = MIN_FREQ, max_freq: float = MAX_FREQ) -> float:
    log_max = np.log2(max_freq)
    log_min = np.log2(min_freq)
    return canvas_height * (log_max - np.log2(max(freq, 1e-6))) / (log_max - log_min)


def freq_to_bin(freq: float, sr: int = 44100, n_fft: int = 2048) -> int:
    return int(round(freq * n_fft / sr))


def bin_to_freq(b: int, sr: int = 44100, n_fft: int = 2048) -> float:
    return b * sr / n_fft


def pixel_y_to_bin(y_pixel: float, canvas_height: int,
                   sr: int = 44100, n_fft: int = 2048) -> int:
    freq = pixel_y_to_freq(y_pixel, canvas_height)
    return freq_to_bin(freq, sr, n_fft)


def pixel_x_to_frame(x_pixel: float, canvas_width: int, n_frames: int) -> int:
    return int(round(x_pixel / canvas_width * (n_frames - 1)))


def frame_to_pixel_x(frame: int, canvas_width: int, n_frames: int) -> float:
    return frame / max(1, n_frames - 1) * canvas_width


def freq_to_midi(freq: float) -> float:
    if freq <= 0:
        return 0.0
    return 69.0 + 12.0 * np.log2(freq / 440.0)


def midi_to_freq(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


def hz_to_note(freq: float) -> str:
    if freq <= 0:
        return '---'
    midi = freq_to_midi(freq)
    note_num = int(round(midi))
    octave = note_num // 12 - 1
    name = NOTE_NAMES[note_num % 12]
    cents = int(round((midi - note_num) * 100))
    if abs(cents) < 5:
        return f'{name}{octave}'
    return f'{name}{octave}{cents:+d}c'


def get_scale_freqs(scale_name: str, root: str = 'C',
                    min_freq: float = MIN_FREQ, max_freq: float = MAX_FREQ) -> list[float]:
    """Return all frequencies in the given scale within the freq range."""
    intervals = SCALES.get(scale_name, SCALES['Chromatic'])
    root_idx = NOTE_NAMES.index(root) if root in NOTE_NAMES else 0
    freqs = []
    for octave in range(-1, 11):
        for interval in intervals:
            midi = (octave + 1) * 12 + root_idx + interval
            f = midi_to_freq(midi)
            if f < min_freq:
                continue
            if f > max_freq:
                return freqs
            freqs.append(f)
    return freqs


def snap_freq_to_scale(freq: float, scale_name: str, root: str = 'C') -> float:
    """Snap a frequency to the nearest note in the scale."""
    if scale_name == 'Chromatic':
        return freq
    intervals = SCALES.get(scale_name, SCALES['Chromatic'])
    root_idx = NOTE_NAMES.index(root) if root in NOTE_NAMES else 0
    midi = freq_to_midi(freq)
    note_in_octave = (round(midi) - root_idx) % 12
    # Find nearest scale degree
    best_dist = 999
    best_interval = 0
    for iv in intervals:
        dist = min(abs(note_in_octave - iv), 12 - abs(note_in_octave - iv))
        if dist < best_dist:
            best_dist = dist
            best_interval = iv
    # Snap
    base_midi = round(midi)
    current_in_oct = (base_midi - root_idx) % 12
    delta = best_interval - current_in_oct
    if delta > 6:
        delta -= 12
    elif delta < -6:
        delta += 12
    return midi_to_freq(base_midi + delta)
